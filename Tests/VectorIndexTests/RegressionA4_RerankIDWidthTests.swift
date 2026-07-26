import XCTest
@testable import VectorIndex

final class RegressionA4_RerankIDWidthTests: XCTestCase {
    func testRerankPreservesIDsAboveInt32Max() {
        let d = 2
        // Two candidates with large 64-bit IDs; id A is the exact match for the query.
        let idA: Int64 = (1 << 31) + 10          // > Int32.max
        let idB: Int64 = (1 << 32) + 7           // > UInt32.max
        let vecA: [Float] = [1, 0]
        let vecB: [Float] = [0, 1]

        let reader = IndexOps.Rerank.CallbackReader(dim: d) { ids, count, dst, present in
            var found = 0
            for i in 0..<count {
                let id = ids[i]
                let row: [Float]? = (id == idA) ? vecA : (id == idB ? vecB : nil)
                if let r = row {
                    dst[i*d + 0] = r[0]; dst[i*d + 1] = r[1]
                    present[i] = 1; found += 1
                } else { present[i] = 0 }
            }
            return found
        }

        let q: [Float] = [1, 0]                  // closest to vecA => idA
        let candIDs: [Int64] = [idA, idB]
        var scores = [Float](repeating: 0, count: 1)
        var outIDs = [Int64](repeating: -1, count: 1)

        q.withUnsafeBufferPointer { qb in
            candIDs.withUnsafeBufferPointer { cb in
                scores.withUnsafeMutableBufferPointer { sb in
                    outIDs.withUnsafeMutableBufferPointer { ib in
                        let opts = IndexOps.Rerank.RerankOpts(backend: .callback)
                        IndexOps.Rerank.rerank_exact_topk(
                            q: qb.baseAddress!, d: d, metric: .euclidean,
                            candIDs: cb.baseAddress!, C: candIDs.count, K: 1,
                            reader: reader, opts: opts,
                            topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
                    }
                }
            }
        }
        // Pre-fix: the returned id is Int32(truncatingIfNeeded: idA) widened back => corrupted.
        XCTAssertEqual(outIDs[0], idA, "top-1 id must be the exact 64-bit candidate id")
    }

    /// Runs a two-candidate equal-score tie: both candidates map to the identical vector, so
    /// their scores are bit-identical by construction (no float tolerance issues). Returns the
    /// single top-1 id chosen by `rerank_exact_topk`.
    private func runEqualScoreTie(candIDs: [Int64], K: Int) -> [Int64] {
        let d = 2
        let vec: [Float] = [1, 0]

        let reader = IndexOps.Rerank.CallbackReader(dim: d) { ids, count, dst, present in
            var found = 0
            for i in 0..<count {
                dst[i*d + 0] = vec[0]; dst[i*d + 1] = vec[1]
                present[i] = 1; found += 1
            }
            return found
        }

        let q: [Float] = [1, 0]
        var scores = [Float](repeating: 0, count: K)
        var outIDs = [Int64](repeating: -1, count: K)

        q.withUnsafeBufferPointer { qb in
            candIDs.withUnsafeBufferPointer { cb in
                scores.withUnsafeMutableBufferPointer { sb in
                    outIDs.withUnsafeMutableBufferPointer { ib in
                        let opts = IndexOps.Rerank.RerankOpts(backend: .callback)
                        IndexOps.Rerank.rerank_exact_topk(
                            q: qb.baseAddress!, d: d, metric: .euclidean,
                            candIDs: cb.baseAddress!, C: candIDs.count, K: K,
                            reader: reader, opts: opts,
                            topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
                    }
                }
            }
        }
        return outIDs
    }

    /// Case 1: two candidates tied on score, ids in REVERSE of position order.
    /// The positional tie-break policy (pre-9.5) picks position 0 => id 500 (wrong).
    /// The by-id policy must pick the smaller id, 3.
    func testTieBreakByFullWidthID_ReversePositionOrder() {
        let outIDs = runEqualScoreTie(candIDs: [500, 3], K: 1)
        XCTAssertEqual(outIDs[0], 3, "tie must break by smallest candidate id, not position")
    }

    /// Case 2 (full-width): ids straddle both Int32.max and UInt32.max, in REVERSE of position
    /// order. Proves the comparison happens on the full Int64 value, not any truncated/narrowed
    /// form (a narrowed comparison could flip the ordering of these two specific ids).
    func testTieBreakByFullWidthID_AboveInt32Max() {
        let idA: Int64 = (1 << 32) + 7   // position 0
        let idB: Int64 = (1 << 31) + 10  // position 1, smaller than idA, > Int32.max
        let outIDs = runEqualScoreTie(candIDs: [idA, idB], K: 1)
        XCTAssertEqual(outIDs[0], idB, "tie must break by smallest full-width id, above Int32.max")
    }

    /// Case 3 (K-selection boundary): three candidates tied on score. The positional policy
    /// would keep the first K=2 positions {900, 800} and evict 7 during selection -- a
    /// post-extract sort of the K survivors could never recover id 7 because it was already
    /// evicted from the heap before extraction. The by-id policy must select {7, 800}.
    func testTieBreakByFullWidthID_KBoundary() {
        let outIDs = runEqualScoreTie(candIDs: [900, 800, 7], K: 2)
        XCTAssertEqual(Set(outIDs), Set<Int64>([7, 800]), "K-boundary tie must keep the two smallest ids")
    }
}
