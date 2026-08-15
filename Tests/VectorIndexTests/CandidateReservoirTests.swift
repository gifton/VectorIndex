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

    // Regression test for Task 15: `telemetry.accepted` and `telemetry.prunes` are declared
    // on `ReservoirTelemetry` but (pre-fix) never incremented anywhere in `pushBatch`/
    // `pruneToTopC` -- dead fields that always read 0. This test forces a real Block-mode
    // prune cycle plus one dedup rejection and one invalid (NaN) rejection in the same
    // batch, then checks:
    //   1) at least one prune fired (buffer-capacity overflow forces `pruneToTopC()`), and
    //   2) `accepted` reconciles exactly against the batch's outcome tally. Every pushed
    //      element takes exactly one of four mutually exclusive paths -- accepted, rejected
    //      by tau, rejected by dedup, rejected as invalid -- so `pushed == accepted +
    //      rejectedTau + rejectedDedup + rejectedInvalid` holds by construction, regardless
    //      of how many *later* prunes evict an already-accepted element from the buffer:
    //      `accepted` counts admission into the reservoir, not survival to the end of the
    //      query (mirrors the pre-existing `acceptedInBatch` return-value semantics of
    //      `pushBatch`).
    func testTelemetryTracksAcceptedAndPrunes() {
        let capacity = 8
        let reservoir = CandidateReservoir(
            capacity: capacity, metric: .l2,
            options: ReservoirOptions(mode: .block, reserveExtra: 0.25, telemetry: true)
        )
        // bufferCapacity = 8 + ceil(8*0.25) = 10 -> appending 30 unique valid items cycles
        // through "append until size==10, prune to 8" repeatedly, guaranteeing >=1 prune.
        let uniqueCount = 30
        var ids: [Int64] = (0..<uniqueCount).map { Int64($0) }
        var scores: [Float] = (0..<uniqueCount).map { Float($0) }
        // Trailing dedup rejection: id 5 repeated (already marked visited earlier in this
        // same batch).
        ids.append(5)
        scores.append(5.0)
        // Trailing invalid rejection: NaN score (checked before dedup, so it never touches
        // the visited set).
        ids.append(999)
        scores.append(Float.nan)

        let visited = DefaultVisitedSet(idCapacity: 1_000)
        ids.withUnsafeBufferPointer { idBuf in
            scores.withUnsafeBufferPointer { scoreBuf in
                _ = reservoir.pushBatch(
                    ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!,
                    count: ids.count, visited: visited
                )
            }
        }

        let t = reservoir.telemetry
        XCTAssertEqual(t.pushed, Int64(ids.count))
        XCTAssertEqual(t.rejectedInvalid, 1, "the NaN-score entry must be rejected as invalid")
        XCTAssertEqual(t.rejectedDedup, 1, "the repeated id-5 entry must be rejected by dedup")
        XCTAssertGreaterThanOrEqual(t.prunes, 1, "buffer-capacity overflow must trigger >=1 prune in Block mode")
        XCTAssertEqual(
            t.accepted, t.pushed - t.rejectedTau - t.rejectedDedup - t.rejectedInvalid,
            "accepted must reconcile exactly against the batch's outcome tally"
        )
    }

    // Task 16c: Pin telemetry.accepted on heap mode's two accept paths: heapInsert (size < C)
    // and replaceRoot (size >= C). Small capacity with DESCENDING scores (worst first, then
    // strictly improving), so early items fill the heap via heapInsert with bad candidates,
    // and all subsequent items are strictly better and must route through replaceRoot.
    // With unique stream of size N and capacity C: accepted == N (all accepted, C fills +
    // (N-C) replacements). This genuinely discriminates: if replaceRoot didn't increment,
    // accepted would equal C and both assertions fail.
    func testTelemetryTracksAcceptedHeapMode() {
        let capacity = 6
        let reservoir = CandidateReservoir(
            capacity: capacity, metric: .l2,
            options: ReservoirOptions(mode: .heap, telemetry: true)
        )
        // DESCENDING scores (L2: smaller is better, so worst first, then improving):
        // ids 0-19, scores 19.0, 18.0, ..., 0.0. Heap fills ids 0-5 with scores 19.0-14.0
        // via heapInsert, establishing root = 19.0 (the worst/largest score). Ids 6-19 have
        // scores 13.0-0.0, all strictly better than current root, so all must route through
        // replaceRoot, which repeatedly replaces the worst with a better candidate. Expected:
        // all 20 unique items accepted. Then one dedup (id 5) and one NaN.
        let uniqueCount = 20
        var ids: [Int64] = (0..<uniqueCount).map { Int64($0) }
        var scores: [Float] = (0..<uniqueCount).reversed().map { Float($0) }  // 19.0 down to 0.0
        ids.append(5)  // repeated id
        scores.append(15.0)
        ids.append(999) // NaN score
        scores.append(Float.nan)

        let visited = DefaultVisitedSet(idCapacity: 1_000)
        ids.withUnsafeBufferPointer { idBuf in
            scores.withUnsafeBufferPointer { scoreBuf in
                _ = reservoir.pushBatch(
                    ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!,
                    count: ids.count, visited: visited
                )
            }
        }

        let t = reservoir.telemetry
        XCTAssertEqual(t.pushed, Int64(ids.count))
        XCTAssertEqual(t.rejectedInvalid, 1, "the NaN-score entry must be rejected as invalid")
        XCTAssertEqual(t.rejectedDedup, 1, "the repeated id-5 entry must be rejected by dedup")
        // In heap mode: ids 0-5 accepted (heapInsert fills), ids 6-19 accepted (replaceRoot),
        // id 5 dedup-rejected, 999 invalid-rejected. No tau rejections (all 20 unique are better or fill).
        XCTAssertEqual(t.rejectedTau, 0, "strictly-improving stream must have no tau rejections")
        XCTAssertEqual(
            t.accepted, t.pushed - t.rejectedTau - t.rejectedDedup - t.rejectedInvalid,
            "accepted must reconcile exactly against the batch's outcome tally"
        )
        XCTAssertEqual(t.accepted, 20, "all 20 unique items must be accepted (C fills + (N-C) replaceRoot)")
    }

    // Task 16c: Pin telemetry.accepted on adaptive mode's block-phase accept path.
    // Adaptive starts in block mode and appends unconditionally until a threshold
    // (occupancy or buffer-full) triggers the switch to heap. We keep the stream size
    // small to stay in block phase (or just barely trigger the switch), and validate
    // that the accepted identity holds regardless.
    func testTelemetryTracksAcceptedAdaptiveMode() {
        let capacity = 8
        let reservoir = CandidateReservoir(
            capacity: capacity, metric: .l2,
            options: ReservoirOptions(mode: .adaptive, reserveExtra: 0.10, adaptiveThreshold: 0.75,
                                       adaptiveInitialMode: .block, telemetry: true)
        )
        // bufferCapacity = 8 + ceil(8*0.10) = 9. Adaptive starts in block, appends until
        // occupancy threshold or buffer-full. We stream 15 items (enough to fill and possibly
        // trigger switch), plus one dedup and one NaN, to exercise the block-phase append path
        // and validate telemetry.accepted tracks it correctly.
        let uniqueCount = 15
        var ids: [Int64] = (0..<uniqueCount).map { Int64($0) }
        var scores: [Float] = (0..<uniqueCount).map { Float($0) }
        ids.append(7)  // repeated id
        scores.append(7.0)
        ids.append(999) // NaN score
        scores.append(Float.nan)

        let visited = DefaultVisitedSet(idCapacity: 1_000)
        ids.withUnsafeBufferPointer { idBuf in
            scores.withUnsafeBufferPointer { scoreBuf in
                _ = reservoir.pushBatch(
                    ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!,
                    count: ids.count, visited: visited
                )
            }
        }

        let t = reservoir.telemetry
        XCTAssertEqual(t.pushed, Int64(ids.count))
        XCTAssertEqual(t.rejectedInvalid, 1, "the NaN-score entry must be rejected as invalid")
        XCTAssertEqual(t.rejectedDedup, 1, "the repeated id-7 entry must be rejected by dedup")
        XCTAssertEqual(
            t.accepted, t.pushed - t.rejectedTau - t.rejectedDedup - t.rejectedInvalid,
            "accepted must reconcile exactly against the batch's outcome tally in adaptive mode"
        )
        XCTAssertGreaterThanOrEqual(t.accepted, 1, "adaptive must accept at least one item")
    }
}
