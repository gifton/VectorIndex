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
}
