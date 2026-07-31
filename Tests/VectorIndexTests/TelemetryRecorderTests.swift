//
//  TelemetryRecorderTests.swift
//  VectorIndexTests
//
//  Accuracy tests for the project's actual telemetry surface (Phase 2 cleanup,
//  Task 3): the six pre-existing `nonisolated(unsafe)` push-callback recorders,
//  the newly-wired RangeScanTelemetryRecorder, and CandidateDedup's pull-style
//  getTelemetry() API. Each test hand-computes the expected counters/fields for
//  a small, fully deterministic input and asserts them exactly.
//
//  Concurrency note: every recorder here is a `nonisolated(unsafe) static var`
//  callback (or, for CandidateDedup, an instance method) with no built-in
//  synchronization. All assertions below pin the deterministic single-threaded
//  path -- one call in, one recorded value out, on the calling thread -- and
//  never run kernels concurrently against a shared recorder. Do not add
//  parallel/XCTestCase-concurrent variants of these tests without adding your
//  own synchronization around `sink`/`record`, since nothing here does.
//

import XCTest
@testable import VectorIndex

final class TelemetryRecorderTests: XCTestCase {

    // MARK: - HNSWTelemetryRecorder
    //
    // NOTE: HNSWTraversal.greedyDescent/efSearch's calls to
    // `HNSWTelemetryRecorder.record?(t)` are wrapped in `#if ENABLE_TELEMETRY`
    // (HNSWTraversal.swift:284-299), a *second*, independent dead compile flag
    // from VINDEX_TELEM -- `ENABLE_TELEMETRY` is never defined anywhere in
    // Package.swift, and turning it on does not compile (verified live:
    // 7 Swift 6 strict-concurrency errors on bare `private var ..._accum`/
    // `...Count` globals in HNSWTraversal.swift). That make-it-compile-and-flip-
    // it-on fix is out of scope for this telemetry-consolidation task (see the
    // task's premise-correction note) and is routed to a follow-up. This test
    // therefore exercises the recorder's sink mechanism directly -- the part of
    // the system that does work today -- rather than through the (currently
    // unreachable) production call site.
    func testHNSWTelemetryRecorderSink() throws {
        var captured: HNSWTraversalTelemetry?
        HNSWTelemetryRecorder.record = { captured = $0 }
        defer { HNSWTelemetryRecorder.record = nil }

        var t = HNSWTraversalTelemetry()
        t.edgesVisited = 12
        t.neighborBatches = 3
        t.candidatesPushed = 7
        t.earlyExits = 1
        t.greedy_ns = 100
        t.efsearch_ns = 200
        t.scoring_ns = 50
        t.total_ns = 350

        HNSWTelemetryRecorder.record?(t)

        XCTAssertEqual(captured?.edgesVisited, 12)
        XCTAssertEqual(captured?.neighborBatches, 3)
        XCTAssertEqual(captured?.candidatesPushed, 7)
        XCTAssertEqual(captured?.earlyExits, 1)
        XCTAssertEqual(captured?.greedy_ns, 100)
        XCTAssertEqual(captured?.efsearch_ns, 200)
        XCTAssertEqual(captured?.scoring_ns, 50)
        XCTAssertEqual(captured?.total_ns, 350)
    }

    // MARK: - GlobalTelemetryRecorder (LayoutTransforms)
    //
    // Same caveat as HNSW above: all 6 of LayoutTransforms.swift's
    // `GlobalTelemetryRecorder.record?(telemetry)` call sites are individually
    // wrapped in `#if ENABLE_TELEMETRY` and are dead for the same reason
    // (verified live: turning the flag on additionally fails with 6 "internal
    // initializer referenced from an '@inlinable' function" errors, since
    // `LayoutTransformTelemetry`'s memberwise init is not `public`/
    // `@usableFromInline` but the callers are `@inlinable`). Tested directly.
    func testGlobalTelemetryRecorderSink() throws {
        var captured: LayoutTransformTelemetry?
        GlobalTelemetryRecorder.record = { captured = $0 }
        defer { GlobalTelemetryRecorder.record = nil }

        let t = LayoutTransformTelemetry(
            transformType: "vec_interleave",
            vectorCount: 100,
            dimension: 128,
            subquantizers: 0,
            rowBlockSize: 8,
            groupSize: 0,
            bytesTransformed: 51200,
            executionTimeNanos: 5000
        )

        GlobalTelemetryRecorder.record?(t)

        XCTAssertEqual(captured?.transformType, "vec_interleave")
        XCTAssertEqual(captured?.vectorCount, 100)
        XCTAssertEqual(captured?.dimension, 128)
        XCTAssertEqual(captured?.rowBlockSize, 8)
        XCTAssertEqual(captured?.bytesTransformed, 51200)
        XCTAssertEqual(captured?.executionTimeNanos, 5000)
    }

    // MARK: - IndexOps.Scoring.Cosine.TelemetryRecorder
    //
    // Real call: this recorder's call site is gated only by a runtime
    // `options.enableTelemetry` bool (default false), not by any `#if` -- it
    // already works today. n=1, d=4, q=xb=[1,0,0,0] (unit vectors, cosine=1),
    // explicit dbInvNorms/queryInvNorm=1.0 avoid relying on internally-computed
    // norms. d=4 is not one of the specialized dims (512/768/1024/1536), so the
    // fused-generic path runs (kernelVariant "generic_fused").
    func testCosineTelemetryRecorder() throws {
        var captured: IndexOps.Scoring.Cosine.Telemetry?
        IndexOps.Scoring.Cosine.TelemetryRecorder.sink = { captured = $0 }
        defer { IndexOps.Scoring.Cosine.TelemetryRecorder.sink = nil }

        let q: [Float] = [1, 0, 0, 0]
        let xb: [Float] = [1, 0, 0, 0]
        let invNorms: [Float] = [1.0]
        var out: [Float] = [0]
        var opts = IndexOps.Scoring.Cosine.Options()
        opts.enableTelemetry = true

        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                invNorms.withUnsafeBufferPointer { np in
                    out.withUnsafeMutableBufferPointer { op in
                        IndexOps.Scoring.Cosine.run(
                            q: qp.baseAddress!, xb: xbp.baseAddress!, n: 1, d: 4,
                            out: op.baseAddress!, dbInvNorms: np.baseAddress!,
                            queryInvNorm: 1.0, options: opts)
                    }
                }
            }
        }

        XCTAssertEqual(out[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(captured?.kernelVariant, "generic_fused")
        XCTAssertEqual(captured?.rowsProcessed, 1)
        // bytesRead = n*d*4 (xb) + d*4 (q) + n*4 (dbInvNorms) = 16 + 16 + 4
        XCTAssertEqual(captured?.bytesRead, 36)
        XCTAssertEqual(captured?.usedFusedPath, true)
        XCTAssertEqual(captured?.usedF16Norms, false)
        XCTAssertEqual(captured?.zeroNormCount, 0)
        XCTAssertEqual(captured?.clampedCount, 0)
    }

    // MARK: - IndexOps.Scoring.InnerProduct.TelemetryRecorder
    //
    // Real call: same runtime-bool gating pattern as Cosine. n=1, d=4,
    // q=[1,2,3,4], xb=[1,1,1,1] -> dot product 1+2+3+4=10. d=4 is not
    // specialized, so the generic (non-fast) path runs.
    func testInnerProductTelemetryRecorder() throws {
        var captured: IndexOps.Scoring.InnerProduct.Telemetry?
        IndexOps.Scoring.InnerProduct.TelemetryRecorder.sink = { captured = $0 }
        defer { IndexOps.Scoring.InnerProduct.TelemetryRecorder.sink = nil }

        let q: [Float] = [1, 2, 3, 4]
        let xb: [Float] = [1, 1, 1, 1]
        var out: [Float] = [0]
        var opts = IndexOps.Scoring.InnerProduct.Options()
        opts.enableTelemetry = true

        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                out.withUnsafeMutableBufferPointer { op in
                    IndexOps.Scoring.InnerProduct.run(
                        q: qp.baseAddress!, xb: xbp.baseAddress!, n: 1, d: 4,
                        out: op.baseAddress!, options: opts)
                }
            }
        }

        XCTAssertEqual(out[0], 10.0, accuracy: 1e-6)
        XCTAssertEqual(captured?.kernelVariant, "generic")
        XCTAssertEqual(captured?.rowsProcessed, 1)
        // bytesRead = (n*d + d) * 4 = (4 + 4) * 4
        XCTAssertEqual(captured?.bytesRead, 32)
        XCTAssertEqual(captured?.fastPathHit, false)
        XCTAssertEqual(captured?.vectorWidth, 4)
    }

    // MARK: - L2SqrTelemetryRecorder
    //
    // Real call: `l2sqr_f32_block` records unconditionally (no runtime gate at
    // all, unlike the other five -- verified by reading the source: the
    // `L2SqrTelemetryRecorder.record(...)` call has no enclosing `if`). q=zero
    // vector, xb=[1,1,1,1] -> squared L2 distance = 4. `opts.algo = .direct`
    // forces the direct path so `usedDotTrick` is deterministically false.
    func testL2SqrTelemetryRecorder() throws {
        var captured: L2SqrTelemetry?
        L2SqrTelemetryRecorder.sink = { captured = $0 }
        defer { L2SqrTelemetryRecorder.sink = nil }

        let q: [Float] = [0, 0, 0, 0]
        let xb: [Float] = [1, 1, 1, 1]
        var out: [Float] = [0]
        var opts = L2SqrOpts.default
        opts.algo = .direct

        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                out.withUnsafeMutableBufferPointer { op in
                    withUnsafePointer(to: opts) { optsP in
                        l2sqr_f32_block(qp.baseAddress!, xbp.baseAddress!, 1, 4, op.baseAddress!, nil, .nan, optsP)
                    }
                }
            }
        }

        XCTAssertEqual(out[0], 4.0, accuracy: 1e-6)
        XCTAssertEqual(captured?.rows, 1)
        XCTAssertEqual(captured?.dim, 4)
        XCTAssertEqual(captured?.usedDotTrick, false)
        XCTAssertNil(captured?.specializedDim)
        // bytesRead = n*d*4 (xb) + d*4 (q) + 0 (no norms) = 16 + 16
        XCTAssertEqual(captured?.bytesRead, 32)
    }

    // MARK: - IndexOps.Selection.TopKTelemetryRecorder
    //
    // Real call: 5 candidates, k=3, max-ordering, `forceAlgorithm: .streaming`
    // for a deterministic algorithm choice (n=5 is far under the default
    // hybridThreshold of 16,384 anyway, so `.streaming` would also be chosen
    // automatically). Expected top-3 by score: 9(id 14), 8(id 12), 5(id 10).
    func testTopKTelemetryRecorder() throws {
        var captured: IndexOps.Selection.TopKTelemetry?
        IndexOps.Selection.TopKTelemetryRecorder.sink = { captured = $0 }
        defer { IndexOps.Selection.TopKTelemetryRecorder.sink = nil }

        let scores: [Float] = [5, 3, 8, 1, 9]
        let ids: [Int32] = [10, 11, 12, 13, 14]
        let config = IndexOps.Selection.TopKConfig(enableTelemetry: true, forceAlgorithm: .streaming, hybridThreshold: 16_384)

        var pairs: [(score: Float, id: Int32)] = []
        scores.withUnsafeBufferPointer { sp in
            ids.withUnsafeBufferPointer { ip in
                let heap = IndexOps.Selection.selectTopK(scores: sp.baseAddress!, ids: ip.baseAddress, count: 5, k: 3, ordering: .max, config: config)
                pairs = heap.extractSorted().map { ($0.score, $0.id) }
                heap.deallocate()
            }
        }

        XCTAssertEqual(pairs.map(\.score), [9, 8, 5])
        XCTAssertEqual(pairs.map(\.id), [14, 12, 10])

        XCTAssertEqual(captured?.candidatesProcessed, 5)
        XCTAssertEqual(captured?.k, 3)
        // Verified against the real streaming-heap implementation on this input.
        XCTAssertEqual(captured?.comparisons, 2)
        XCTAssertEqual(captured?.heapPushes, 4)
        XCTAssertEqual(captured?.siftOperations, 2)
    }

    // MARK: - RangeScanTelemetryRecorder (newly wired in this task)
    //
    // Real call: 3 flat vectors, d=2, query=[0,0], database rows at distance²
    // 0, 1, and 25 from the query. threshold=2.0 (compared against the raw
    // L2Sqr score, i.e. squared distance, not sqrt'd -- verified: scoresOut
    // for the kept rows come back as 0.0 and 1.0, not 0.0 and 1.0's square
    // roots). `config.earlyExit = .off` forces the flat non-early-exit path
    // deterministically.
    func testRangeScanTelemetryRecorder() throws {
        var captured: RangeScanTelemetry?
        RangeScanTelemetryRecorder.sink = { captured = $0 }
        defer { RangeScanTelemetryRecorder.sink = nil }

        let query: [Float] = [0, 0]
        let database: [Float] = [0, 0, /* row 0 */ 1, 0, /* row 1 */ 5, 0 /* row 2 */]
        let ids: [Int64] = [100, 101, 102]
        var idsOut = [Int64](repeating: 0, count: 3)
        var scoresOut = [Float](repeating: 0, count: 3)
        let config = RangeScanConfig(earlyExit: .off, outputScores: true, idFilter: nil,
                                     visitedSet: nil, reservoir: nil, outputMode: .compacted,
                                     tileSize: 1024, enableTelemetry: true)

        let kept = query.withUnsafeBufferPointer { qp in
            database.withUnsafeBufferPointer { dbp in
                ids.withUnsafeBufferPointer { idp in
                    idsOut.withUnsafeMutableBufferPointer { iop in
                        scoresOut.withUnsafeMutableBufferPointer { sop in
                            rangeScanBlock(query: qp.baseAddress!, database: dbp.baseAddress!,
                                           ids: idp.baseAddress, vectorCount: 3, dimension: 2,
                                           metric: .l2, threshold: 2.0, idsOut: iop.baseAddress!,
                                           scoresOut: sop.baseAddress!, maxOut: 3, config: config)
                        }
                    }
                }
            }
        }

        XCTAssertEqual(kept, 2)
        XCTAssertEqual(Array(idsOut.prefix(2)), [100, 101])
        XCTAssertEqual(Array(scoresOut.prefix(2)), [0.0, 1.0])

        XCTAssertNotNil(captured, "RangeScanTelemetryRecorder.sink must fire when config.enableTelemetry is true")
        XCTAssertEqual(captured?.vectorsScanned, 3)
        XCTAssertEqual(captured?.vectorsKept, 2)
        XCTAssertEqual(captured?.usedEarlyExit, false)
        XCTAssertEqual(captured?.earlyExitHits, 0)
        XCTAssertEqual(captured?.usedADCPath, false)
        // bytesScored = n*d*4 = 3*2*4
        XCTAssertEqual(captured?.bytesScored, 24)
        XCTAssertEqual(captured?.bytesCodes, 0)
    }

    // MARK: - CandidateDedup pull API: DefaultVisitedSet.getTelemetry()
    //
    // Not a push callback -- exercised directly. Sequence [5,7,5,9,7,5] against
    // a fresh denseEpoch set: 5(new),7(new),5(dup),9(new),7(dup),5(dup) ->
    // totalChecks=6, uniqueCount=3, duplicateCount=3.
    func testCandidateDedupGetTelemetry() throws {
        let vs = DefaultVisitedSet(idCapacity: 100, opts: VisitedOpts(mode: .denseEpoch))
        vs.resetForNewQuery()

        let sequence: [Int64] = [5, 7, 5, 9, 7, 5]
        let results = sequence.map { vs.testAndSet(id: $0) }

        XCTAssertEqual(results, [true, true, false, true, false, false])

        let telemetry = vs.getTelemetry(elapsedNanos: 1234)
        XCTAssertEqual(telemetry.mode, .denseEpoch)
        XCTAssertEqual(telemetry.totalChecks, 6)
        XCTAssertEqual(telemetry.uniqueCount, 3)
        XCTAssertEqual(telemetry.duplicateCount, 3)
        XCTAssertEqual(telemetry.checkTimeNanos, 1234)
        XCTAssertEqual(telemetry.deduplicationRate, 0.5, accuracy: 1e-9)
    }
}
