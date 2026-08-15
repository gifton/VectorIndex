import XCTest
@testable import VectorIndex

// First dedicated coverage of the L2 early-exit path (`rangeScanL2_earlyExit`,
// reached via `rangeScanBlock(..., config: RangeScanConfig(earlyExit: .on))`)
// (P6c, Task 11). Prior to this task the branch-light early-exit kernel had
// no test forcing it to run at all -- everything else in the suite either
// left `earlyExit` at `.auto`/`.off` or exercised the generic path directly.
//
// Two independent checks:
//  1. The `.off` (generic, non-early-exit) result is validated against a
//     from-scratch scalar L2 reference computed in this test -- an oracle
//     independent of any VectorIndex kernel.
//  2. The `.on` (early-exit) result is compared against the `.off` result:
//     identical id sets and identical per-id scores (within tolerance).
//     Order can legitimately differ (early-exit processes in blocks of 8;
//     generic processes in tiles), so comparison is done as id->score maps,
//     not positional arrays.
//
// If check 1 or 2 fails, that is a pre-existing correctness bug in the
// early-exit path uncovered by this task's first-ever direct test of it --
// per the task brief, that is a STOP-and-report condition, not something to
// paper over while doing the perf hoist.
final class RangeQueryEarlyExitTests: XCTestCase {

    private let n = 500
    private let d = 16

    private func seededVectors(count: Int, seed: UInt64) -> [Float] {
        var s = seed
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            s = 2862933555777941757 &* s &+ 3037000493
            let u = Float(s >> 40) / Float(1 << 24) // [0, 1)
            out[i] = u * 2 - 1                      // [-1, 1)
        }
        return out
    }

    func testEarlyExitMatchesGenericPathAndScalarOracle() {
        let query = seededVectors(count: d, seed: 0x9E3779B97F4A7C15)
        let database = seededVectors(count: n * d, seed: 0xD1B54A32D192ED03)
        let ids: [Int64] = (0..<n).map { Int64(10_000 + $0) }

        // Scalar reference L2 distances, independent of any kernel.
        var refDistances = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sumSq: Float = 0
            for j in 0..<d {
                let diff = query[j] - database[i * d + j]
                sumSq += diff * diff
            }
            refDistances[i] = sumSq.squareRoot()
        }

        // Threshold chosen from the data itself so ~10% of rows pass. Picking
        // the raw 50th-smallest value as the threshold is fragile: the
        // generic path's L2 (via the SIMD/dot-trick L2Sqr kernel) and this
        // scalar reference sum in different orders, so their results can
        // differ by a handful of ULPs. If two order statistics happen to sit
        // within that noise band (as they do for a couple of indices in this
        // seeded set -- gaps as small as ~7e-5 next to the raw 50th value),
        // an exact-boundary threshold nondeterministically in/excludes the
        // boundary row depending on which summation order the kernel used.
        // Instead, search outward from the 10% mark for a split index whose
        // neighboring order statistics are separated by a safe margin
        // (>= 1e-2, well above observed kernel/scalar float discrepancy),
        // and set the threshold at the midpoint of that gap so the boundary
        // is unambiguous for any reasonable summation order.
        let sorted = refDistances.sorted()
        let target = n / 10
        let minGap: Float = 1e-2
        var splitIndex: Int? = nil
        searchLoop: for offset in 0..<target {
            for candidate in [target - offset, target + offset] where candidate > 0 && candidate < n {
                if sorted[candidate] - sorted[candidate - 1] >= minGap {
                    splitIndex = candidate
                    break searchLoop
                }
            }
        }
        guard let split = splitIndex else {
            XCTFail("could not find a numerically safe ~10% split point in the seeded data")
            return
        }
        let threshold = (sorted[split - 1] + sorted[split]) / 2
        let expectedIDs = Set((0..<n).filter { refDistances[$0] <= threshold }.map { ids[$0] })
        XCTAssertEqual(expectedIDs.count, split)
        XCTAssertTrue((n / 20...(n / 5)).contains(expectedIDs.count), "selectivity should be roughly ~10% (got \(expectedIDs.count)/\(n))")

        func run(earlyExit: EarlyExitStrategy) -> (kept: Int, ids: [Int64], scores: [Float]) {
            var idsOut = [Int64](repeating: -1, count: n)
            var scoresOut = [Float](repeating: 0, count: n)
            let config = RangeScanConfig(earlyExit: earlyExit, outputScores: true)
            let kept = query.withUnsafeBufferPointer { qp in
                database.withUnsafeBufferPointer { dbp in
                    ids.withUnsafeBufferPointer { idp in
                        idsOut.withUnsafeMutableBufferPointer { iop in
                            scoresOut.withUnsafeMutableBufferPointer { sop in
                                rangeScanBlock(
                                    query: qp.baseAddress!, database: dbp.baseAddress!,
                                    ids: idp.baseAddress, vectorCount: n, dimension: d,
                                    metric: .l2, threshold: threshold,
                                    idsOut: iop.baseAddress!, scoresOut: sop.baseAddress!,
                                    maxOut: n, config: config)
                            }
                        }
                    }
                }
            }
            return (kept, Array(idsOut.prefix(kept)), Array(scoresOut.prefix(kept)))
        }

        // Check 1: generic (.off) path against the scalar oracle.
        let generic = run(earlyExit: .off)
        XCTAssertEqual(generic.kept, expectedIDs.count)
        XCTAssertEqual(Set(generic.ids), expectedIDs)
        var genericByID: [Int64: Float] = [:]
        for (id, score) in zip(generic.ids, generic.scores) { genericByID[id] = score }
        for id in expectedIDs {
            let refIdx = Int(id) - 10_000
            XCTAssertEqual(genericByID[id]!, refDistances[refIdx], accuracy: 1e-4, "id \(id)")
        }

        // Check 2: early-exit (.on) path against the generic path (STOP
        // condition per brief if this diverges -- would indicate a
        // pre-existing early-exit bug, not something to fix here).
        let earlyExit = run(earlyExit: .on)
        XCTAssertEqual(earlyExit.kept, generic.kept)
        XCTAssertEqual(Set(earlyExit.ids), Set(generic.ids), "early-exit and generic paths must return identical id sets")
        var earlyExitByID: [Int64: Float] = [:]
        for (id, score) in zip(earlyExit.ids, earlyExit.scores) { earlyExitByID[id] = score }
        for id in expectedIDs {
            XCTAssertEqual(earlyExitByID[id]!, genericByID[id]!, accuracy: 1e-4, "id \(id) score mismatch between early-exit and generic paths")
        }
    }
}
