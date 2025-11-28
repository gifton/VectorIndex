import XCTest
@testable import VectorIndex

final class HNSWTraversalKernelTests: XCTestCase {
    func testEfSearchTieBreak_L2() {
        // N=3, d=2. Node 0 neighbors {1,2}. Nodes 1 and 2 coincide at (1,0).
        // Query (1,0) => distances: 1->0, 2->0. Tie broken by smaller id.
        let d = 2
        let N = 3
        let xb: [Float] = [
            0, 0,   // id=0
            1, 0,   // id=1
            1, 0    // id=2
        ]
        // CSR layer 0: node 0 → [1,2], others empty
        let offsetsL0: [Int32] = [0, 2, 2, 2]
        let neighborsL0: [Int32] = [1, 2]
        let q: [Float] = [1, 0]
        var idsOut = [Int32](repeating: -1, count: 2)
        var distsOut = [Float](repeating: .infinity, count: 2)

        let written = q.withUnsafeBufferPointer { qb in
            xb.withUnsafeBufferPointer { base in
                offsetsL0.withUnsafeBufferPointer { ob in
                    neighborsL0.withUnsafeBufferPointer { nb in
                        HNSWTraversal.efSearch(
                            q: qb.baseAddress!, d: d,
                            enterL0: 0,
                            offsetsL0: ob.baseAddress!, neighborsL0: nb.baseAddress!,
                            xb: base.baseAddress!, N: N,
                            ef: 2, metric: .L2,
                            allowBits: nil, allowN: 0, invNorms: nil,
                            idsOut: &idsOut, distsOut: &distsOut
                        )
                    }
                }
            }
        }
        XCTAssertEqual(written, 2)
        XCTAssertEqual(idsOut[0], 1)
        XCTAssertEqual(idsOut[1], 2)
        XCTAssertEqual(distsOut[0], 0, accuracy: 1e-6)
        XCTAssertEqual(distsOut[1], 0, accuracy: 1e-6)
    }
}
