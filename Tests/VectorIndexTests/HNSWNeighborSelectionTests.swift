import XCTest
@testable import VectorIndex

final class HNSWNeighborSelectionTests: XCTestCase {
    func testSelectNeighborsDiversityAndFill_L2() {
        // x_new at origin; candidates at 1, 2, and near 1 with slight offset
        // Diversity rule: after accepting id=1, id=2 and id=3 are rejected in diversity phase
        // (both closer to id=1 than to x_new), then fill picks the nearest remaining.
        let d = 2
        let x_new: [Float] = [0, 0]
        // ids: 1 -> (1,0), 2 -> (2,0), 3 -> (1,0.1)
        let xb: [Float] = [
            0, 0,   // id=0 (unused)
            1, 0,   // id=1
            2, 0,   // id=2
            1, 0.1  // id=3
        ]
        var candidates: [Int32] = [1, 2, 3]
        var selected = [Int32](repeating: -1, count: 2)
        let written = x_new.withUnsafeBufferPointer { q in
            xb.withUnsafeBufferPointer { base in
                candidates.withUnsafeBufferPointer { cb in
                    selected.withUnsafeMutableBufferPointer { sb in
                        hnsw_select_neighbors_f32_swift(
                            x_new: q.baseAddress!, d: d,
                            candidates: cb.baseAddress!, candCount: candidates.count,
                            xb: base.baseAddress!, N: 4,
                            M: 2, layer: 0,
                            metric: .L2,
                            optionalInvNorms: nil,
                            selectedOut: sb.baseAddress!
                        )
                    }
                }
            }
        }
        XCTAssertEqual(written, 2)
        // Expect the nearest first (1), then fill from remaining by ascending distance → (3)
        XCTAssertEqual(Array(selected.prefix(written)), [1, 3])
    }

    func testPruneNeighborsKeepsTopM_L2() {
        // Anchor u=0 at (0,0), neighbors: 1 (1,0), 2 (2,0), 3 (3,0). Keep M=2.
        let d = 2
        let xb: [Float] = [
            0, 0,   // id=0
            1, 0,   // id=1
            2, 0,   // id=2
            3, 0    // id=3
        ]
        let N = 4
        // CSR for layer 0: only node 0 has neighbors [1,2,3]
        var offsets: [Int32] = [0, 3, 0, 0, 0]
        var neighbors: [Int32] = [1, 2, 3]
        var out = [Int32](repeating: -1, count: 2)
        let kept = xb.withUnsafeBufferPointer { base in
            offsets.withUnsafeBufferPointer { ob in
                neighbors.withUnsafeBufferPointer { nb in
                    out.withUnsafeMutableBufferPointer { pb in
                        hnsw_prune_neighbors_f32_swift(
                            u: 0,
                            xb: base.baseAddress!, d: d,
                            offsetsL: ob.baseAddress!, neighborsL: nb.baseAddress!,
                            M: 2, metric: .L2,
                            optionalInvNorms: nil,
                            N: N,
                            prunedOut: pb.baseAddress!
                        )
                    }
                }
            }
        }
        XCTAssertEqual(kept, 2)
        XCTAssertEqual(Array(out.prefix(kept)), [1, 2])
    }
}

