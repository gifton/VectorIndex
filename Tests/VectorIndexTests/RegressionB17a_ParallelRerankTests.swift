import XCTest
@testable import VectorIndex

/// Regression guard for B17a (`ExactRerank.swift` `scoreBlock`'s parallel branch).
/// Per the Phase-2 research brief: no existing test drives `C` above
/// `opts.parallelThreshold` (default 8192, with default `gatherTile` 128 so
/// `tile*2 == 256`), so the `DispatchQueue.concurrentPerform` branch -- including
/// the pointer-laundering code being replaced in this task -- was never exercised
/// by any test. Written *before* the fix to characterize current (correct)
/// behavior; must stay green after the `UnsafeSendablePtr`/`UnsafeSendableMutPtr`
/// rewrite.
///
/// Drives C = 8300 (> default parallelThreshold 8192) through `rerank_exact_topk`
/// twice -- once with `enableParallel: true` (parallel path) and once with
/// `enableParallel: false` (sequential path, already covered elsewhere) -- and
/// asserts identical top-K output, proving the parallel branch's scatter-back
/// writes to the same slots as the trusted sequential path.
final class RegressionB17a_ParallelRerankTests: XCTestCase {
    func testParallelPathAboveThresholdMatchesSequential() {
        let d = 4
        let C = 8300
        let K = 16

        // Candidate id i maps to vector [Float(i), 0, 0, 0]; query is the origin,
        // so euclidean distance increases monotonically with id -- smallest ids win.
        let reader = IndexOps.Rerank.CallbackReader(dim: d) { ids, count, dst, present in
            for i in 0..<count {
                let v = Float(ids[i])
                dst[i*d + 0] = v; dst[i*d + 1] = 0; dst[i*d + 2] = 0; dst[i*d + 3] = 0
                present[i] = 1
            }
            return count
        }

        let q: [Float] = [0, 0, 0, 0]
        let candIDs: [Int64] = (0..<C).map { Int64($0) }

        func run(enableParallel: Bool) -> (scores: [Float], ids: [Int64]) {
            var scores = [Float](repeating: 0, count: K)
            var outIDs = [Int64](repeating: -1, count: K)
            q.withUnsafeBufferPointer { qb in
                candIDs.withUnsafeBufferPointer { cb in
                    scores.withUnsafeMutableBufferPointer { sb in
                        outIDs.withUnsafeMutableBufferPointer { ib in
                            let opts = IndexOps.Rerank.RerankOpts(
                                backend: .callback,
                                enableParallel: enableParallel,
                                parallelThreshold: 8192
                            )
                            IndexOps.Rerank.rerank_exact_topk(
                                q: qb.baseAddress!, d: d, metric: .euclidean,
                                candIDs: cb.baseAddress!, C: C, K: K,
                                reader: reader, opts: opts,
                                topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
                        }
                    }
                }
            }
            return (scores, outIDs)
        }

        let parallel = run(enableParallel: true)
        let sequential = run(enableParallel: false)

        XCTAssertEqual(parallel.ids, sequential.ids,
                        "parallel scoreBlock branch must scatter results to the same slots as the sequential branch")
        XCTAssertEqual(parallel.scores, sequential.scores)
        // Smallest-id candidates (closest to the origin) must win.
        XCTAssertEqual(parallel.ids, Array((0..<Int64(K))))
    }
}
