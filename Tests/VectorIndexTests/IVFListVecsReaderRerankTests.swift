import XCTest
@testable import VectorIndex

final class IVFListVecsReaderRerankTests: XCTestCase {
    func testTopKWithIVFListReader_L2() {
        // Two lists, d=2. Internal IDs 0..3 mapped to (list,offset).
        // Vectors: id0 -> (0,0), id1 -> (1,0); id2 -> (0,1); id3 -> (1,1)
        let d = 2
        let list0: [Float] = [0, 0, 1, 0]   // 2 rows
        let list1: [Float] = [0, 1, 1, 1]   // 2 rows
        let N = 4
        // id2List and id2Offset
        var id2List = [Int32](repeating: 0, count: N)
        var id2Offset = [Int32](repeating: 0, count: N)
        id2List[0] = 0; id2Offset[0] = 0
        id2List[1] = 0; id2Offset[1] = 1
        id2List[2] = 1; id2Offset[2] = 0
        id2List[3] = 1; id2Offset[3] = 1
        // Build lists array
        var lists: [IndexOps.Rerank.IVFListVecsReader.List] = []
        list0.withUnsafeBufferPointer { p0 in
            list1.withUnsafeBufferPointer { p1 in
                lists.append(.init(base: p0.baseAddress!, len: 2))
                lists.append(.init(base: p1.baseAddress!, len: 2))
            }
        }
        // Query near (1,0); choose components so a+b>1 to avoid a tie between [0,0] and [1,1]
        // For L2, tie occurs when a+b == 1 (equal distance to [0,0] and [1,1]).
        let q: [Float] = [0.95, 0.06]
        let candIDs: [Int64] = [0, 1, 2, 3]
        var scores = [Float](repeating: 0, count: 2)
        var ids = [Int64](repeating: -1, count: 2)
        q.withUnsafeBufferPointer { qb in
            id2List.withUnsafeBufferPointer { lb in
                id2Offset.withUnsafeBufferPointer { ob in
                    let reader = IndexOps.Rerank.IVFListVecsReader(
                        lists: lists,
                        id2List: lb.baseAddress!,
                        id2Offset: ob.baseAddress!,
                        N: N,
                        dim: d,
                        invNorms: nil,
                        sqNorms: nil
                    )
                    candIDs.withUnsafeBufferPointer { cb in
                        scores.withUnsafeMutableBufferPointer { sb in
                            ids.withUnsafeMutableBufferPointer { ib in
                                let opts = IndexOps.Rerank.RerankOpts(backend: .ivfListVecs, reorderBySegment: true)
                                IndexOps.Rerank.rerank_exact_topk(
                                    q: qb.baseAddress!, d: d, metric: .euclidean,
                                    candIDs: cb.baseAddress!, C: candIDs.count, K: 2,
                                    reader: reader, opts: opts,
                                    topScores: sb.baseAddress!, topIDs: ib.baseAddress!
                                )
                            }
                        }
                    }
                }
            }
        }
        // Expect nearest ids to (1,0) are id1 (1,0) then id3 (1,1)
        XCTAssertEqual(ids[0], 1)
        XCTAssertEqual(ids[1], 3)
    }

    func testTieBreakingWithIVFListReader_L2() {
        // Construct a tie for the second-closest candidate under L2.
        // Two lists, d=2; mapping same as previous test.
        let d = 2
        let list0: [Float] = [0, 0, 1, 0]
        let list1: [Float] = [0, 1, 1, 1]
        let N = 4
        var id2List = [Int32](repeating: 0, count: N)
        var id2Offset = [Int32](repeating: 0, count: N)
        id2List[0] = 0; id2Offset[0] = 0
        id2List[1] = 0; id2Offset[1] = 1
        id2List[2] = 1; id2Offset[2] = 0
        id2List[3] = 1; id2Offset[3] = 1
        var lists: [IndexOps.Rerank.IVFListVecsReader.List] = []
        list0.withUnsafeBufferPointer { p0 in
            list1.withUnsafeBufferPointer { p1 in
                lists.append(.init(base: p0.baseAddress!, len: 2))
                lists.append(.init(base: p1.baseAddress!, len: 2))
            }
        }

        // Query q = [a,b] with a+b == 1 creates equal distances to [0,0] and [1,1].
        let q: [Float] = [0.95, 0.05]
        let candIDs: [Int64] = [0, 1, 2, 3]
        var scores = [Float](repeating: 0, count: 2)
        var ids = [Int64](repeating: -1, count: 2)

        q.withUnsafeBufferPointer { qb in
            id2List.withUnsafeBufferPointer { lb in
                id2Offset.withUnsafeBufferPointer { ob in
                    let reader = IndexOps.Rerank.IVFListVecsReader(
                        lists: lists,
                        id2List: lb.baseAddress!,
                        id2Offset: ob.baseAddress!,
                        N: N,
                        dim: d,
                        invNorms: nil,
                        sqNorms: nil
                    )
                    candIDs.withUnsafeBufferPointer { cb in
                        scores.withUnsafeMutableBufferPointer { sb in
                            ids.withUnsafeMutableBufferPointer { ib in
                                let opts = IndexOps.Rerank.RerankOpts(backend: .ivfListVecs, reorderBySegment: true)
                                IndexOps.Rerank.rerank_exact_topk(
                                    q: qb.baseAddress!, d: d, metric: .euclidean,
                                    candIDs: cb.baseAddress!, C: candIDs.count, K: 2,
                                    reader: reader, opts: opts,
                                    topScores: sb.baseAddress!, topIDs: ib.baseAddress!
                                )
                            }
                        }
                    }
                }
            }
        }

        // Expect first is id1 ([1,0]); for the tie between id0 and id3, policy prefers smaller ID.
        XCTAssertEqual(ids[0], 1)
        XCTAssertEqual(ids[1], 0)
    }
}
