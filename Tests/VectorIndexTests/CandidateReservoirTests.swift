import XCTest
@testable import VectorIndex

final class CandidateReservoirTests: XCTestCase {
    // Characterization test pinning extractTopK's current sort-based behavior BEFORE the
    // B15 quickselect-based rewrite, so the refactor can be verified for exact parity
    // (including tie-break-by-smaller-ID ordering).
    //
    // Fixture: 4 candidates pushed in .block mode with headroom 5 (capacity 4,
    // reserveExtra 0.10 -> bufferCapacity = 4 + ceil(4*0.10) = 5), so no auto-prune fires
    // during pushBatch (size only reaches 4, never >= bufferCapacity) and size/order are
    // exactly as pushed: id 10 score 5.0, id 20 score 3.0 (tie w/ id 30), id 30 score 3.0,
    // id 40 score 1.0. L2 metric (smaller is better). Expected best-first: 40, 20, 30, 10
    // (id 20 before 30 on the score tie, since 20 < 30).
    func testExtractTopKOrdersByScoreThenIDForL2() {
        let reservoir = CandidateReservoir(
            capacity: 4, metric: .l2,
            options: ReservoirOptions(mode: .block, reserveExtra: 0.10)
        )
        let ids: [Int64] = [10, 20, 30, 40]
        let scores: [Float] = [5.0, 3.0, 3.0, 1.0]
        ids.withUnsafeBufferPointer { idBuf in
            scores.withUnsafeBufferPointer { scoreBuf in
                _ = reservoir.pushBatch(ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!, count: 4)
            }
        }
        XCTAssertEqual(reservoir.count, 4, "sanity: no auto-prune should have fired yet")

        var outScores = [Float](repeating: .nan, count: 4)
        var outIDs = [Int64](repeating: -1, count: 4)
        outScores.withUnsafeMutableBufferPointer { sp in
            outIDs.withUnsafeMutableBufferPointer { ip in
                reservoir.extractTopK(k: 4, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
            }
        }
        XCTAssertEqual(outIDs, [40, 20, 30, 10])
        XCTAssertEqual(outScores, [1.0, 3.0, 3.0, 5.0])

        // Partial top-k must be a prefix of the full ordering.
        var outScores2 = [Float](repeating: .nan, count: 2)
        var outIDs2 = [Int64](repeating: -1, count: 2)
        outScores2.withUnsafeMutableBufferPointer { sp in
            outIDs2.withUnsafeMutableBufferPointer { ip in
                reservoir.extractTopK(k: 2, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
            }
        }
        XCTAssertEqual(outIDs2, [40, 20])
        XCTAssertEqual(outScores2, [1.0, 3.0])
    }

    // extractTopK is documented as read-only; a second extraction must return the
    // identical result and must not change reservoir.count.
    func testExtractTopKIsReadOnly() {
        let reservoir = CandidateReservoir(capacity: 4, metric: .l2, options: ReservoirOptions(mode: .block, reserveExtra: 0.10))
        let ids: [Int64] = [10, 20, 30, 40]
        let scores: [Float] = [5.0, 3.0, 3.0, 1.0]
        ids.withUnsafeBufferPointer { idBuf in
            scores.withUnsafeBufferPointer { scoreBuf in
                _ = reservoir.pushBatch(ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!, count: 4)
            }
        }
        func extract() -> ([Int64], [Float]) {
            var outScores = [Float](repeating: .nan, count: 4)
            var outIDs = [Int64](repeating: -1, count: 4)
            outScores.withUnsafeMutableBufferPointer { sp in
                outIDs.withUnsafeMutableBufferPointer { ip in
                    reservoir.extractTopK(k: 4, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
                }
            }
            return (outIDs, outScores)
        }
        let first = extract()
        let second = extract()
        XCTAssertEqual(first.0, second.0)
        XCTAssertEqual(first.1, second.1)
        XCTAssertEqual(reservoir.count, 4, "extractTopK must not change reservoir size")
    }

    // Regression test for the missing `.adaptive` overflow-prune guard (B15): with a small
    // enough capacity, the periodic (every-64-pushes) occupancy check can miss the
    // adaptiveThreshold crossing entirely before `size` reaches `bufferCapacity`, so
    // appendUnsorted's defensive fallback silently *grows* the buffer instead of pruning —
    // defeating the "no hot-path allocations" fixed-capacity design intent.
    //
    // capacity=100, reserveExtra=0.10 -> bufferCapacity = 100 + ceil(100*0.10) = 110.
    // adaptiveThreshold=0.75 -> switch-trigger occupancy = 75. The periodic check only
    // runs at size % 64 == 0, i.e. size=64 (occ 0.64, below threshold) then size=128
    // (already past bufferCapacity=110). Pushing 111 single-item batches crosses
    // bufferCapacity on push #111 without the periodic check ever tripping first.
    //
    // THIS TEST IS EXPECTED TO BE RED against the current, unmodified code (bufferCapacity
    // grows past 110 via appendUnsorted's defensive-grow branch) and MUST turn GREEN once
    // the `.adaptive` case gains the same `if size >= bufferCapacity { pruneToTopC() }`
    // guard `.block` already has. This is a standard red-green bugfix test, not a
    // characterization of current (buggy) behavior — do not "fix" the test to match the
    // bug.
    func testAdaptiveModePrunesBeforeBufferOverflow() {
        let capacity = 100
        let reservoir = CandidateReservoir(
            capacity: capacity, metric: .l2,
            options: ReservoirOptions(mode: .adaptive, reserveExtra: 0.10, adaptiveThreshold: 0.75, adaptiveInitialMode: .block)
        )
        let expectedBufferCapacity = capacity + Int(ceil(Double(capacity) * 0.10)) // 110

        let n = 111
        var oneID: Int64 = 0
        var oneScore: Float = 0
        for i in 0..<n {
            oneID = Int64(i)
            oneScore = Float(i)
            withUnsafePointer(to: &oneID) { idp in
                withUnsafePointer(to: &oneScore) { sp in
                    _ = reservoir.pushBatch(ids: idp, scores: sp, count: 1)
                }
            }
        }

        XCTAssertEqual(reservoir.bufferCapacity, expectedBufferCapacity,
                       "adaptive mode must prune at bufferCapacity instead of silently growing the buffer")
    }
}
