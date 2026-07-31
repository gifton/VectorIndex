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

    // Regression guard for the .adaptive overflow-prune + guaranteed switch (user
    // decision 2026-07-30 making .adaptive a working mode). capacity=100,
    // reserveExtra=0.10 -> bufferCapacity=110; adaptiveThreshold=0.75 -> the sampled
    // check (size % 64 == 0) can only fire at size=64 (occupancy 0.64, below
    // threshold), so the buffer-full guard -- not the sample -- must prune at 110,
    // heapify, and flip to .heap. Before the fix pair (currentMode collapsing to
    // .block at init; no guard in the .adaptive case) this config either never
    // reached the .adaptive code at all or, once reachable, grew the buffer without
    // bound via appendUnsorted's defensive-grow branch.
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
        XCTAssertEqual(reservoir.telemetry.modeSwitches, 1,
                       "filling the block-phase buffer must complete the adaptive switch")
        XCTAssertEqual(reservoir.count, capacity)
        // L2: smaller scores are better; ids 0..99 carry scores 0..99 -> exact top-100.
        var outScores = [Float](repeating: 0, count: capacity)
        var outIDs = [Int64](repeating: -1, count: capacity)
        outScores.withUnsafeMutableBufferPointer { sb in
            outIDs.withUnsafeMutableBufferPointer { ib in
                reservoir.extractTopK(k: capacity, topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
            }
        }
        XCTAssertEqual(outIDs, Array(0..<Int64(capacity)),
                       "post-switch reservoir must hold exactly the best C candidates, best-first")
    }

    /// Sampled-path switch: C=150, α=0.10 -> bufferCapacity=165; threshold 0.75 ->
    /// trigger occupancy > 112.5. The periodic check samples at size % 64 == 0 and
    /// size=128 lands inside (112.5, 165], so the SAMPLED check -- not the
    /// buffer-full guard -- performs the switch here (complement of the test above).
    /// Also proves end-to-end parity: adaptive must produce results identical to a
    /// pure .heap reservoir fed the same stream (same top-C, best-first, ties by id).
    func testAdaptiveSampledSwitchMatchesPureHeapResults() {
        let capacity = 150
        let adaptive = CandidateReservoir(
            capacity: capacity, metric: .l2,
            options: ReservoirOptions(mode: .adaptive, reserveExtra: 0.10, adaptiveThreshold: 0.75, adaptiveInitialMode: .block)
        )
        let pureHeap = CandidateReservoir(
            capacity: capacity, metric: .l2,
            options: ReservoirOptions(mode: .heap)
        )

        // 400 candidates with DESCENDING scores (L2: later is better), so the heap
        // phase after the switch keeps accepting -- exercises replaceRoot, not just
        // tau-rejection. One tie pair inside the winning range pins stable id order.
        let n = 400
        var ids = [Int64](repeating: 0, count: n)
        var scores = [Float](repeating: 0, count: n)
        for i in 0..<n { ids[i] = Int64(i); scores[i] = Float(n - i) }
        scores[301] = scores[300] // tie pair (ids 300, 301), both inside the top-150

        ids.withUnsafeBufferPointer { ip in
            scores.withUnsafeBufferPointer { sp in
                _ = adaptive.pushBatch(ids: ip.baseAddress!, scores: sp.baseAddress!, count: n)
                _ = pureHeap.pushBatch(ids: ip.baseAddress!, scores: sp.baseAddress!, count: n)
            }
        }

        XCTAssertEqual(adaptive.telemetry.modeSwitches, 1, "sampled occupancy check must switch exactly once")
        XCTAssertEqual(adaptive.count, capacity)
        XCTAssertEqual(pureHeap.count, capacity)

        func topK(_ r: CandidateReservoir) -> (scores: [Float], ids: [Int64]) {
            var s = [Float](repeating: 0, count: capacity)
            var i = [Int64](repeating: -1, count: capacity)
            s.withUnsafeMutableBufferPointer { sb in
                i.withUnsafeMutableBufferPointer { ib in
                    r.extractTopK(k: capacity, topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
                }
            }
            return (s, i)
        }
        let a = topK(adaptive)
        let h = topK(pureHeap)
        XCTAssertEqual(a.ids, h.ids, "adaptive must produce results identical to pure heap")
        XCTAssertEqual(a.scores, h.scores)
    }
}
