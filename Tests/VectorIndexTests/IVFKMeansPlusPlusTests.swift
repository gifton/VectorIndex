import XCTest
@testable import VectorIndex

final class IVFKMeansPlusPlusTests: XCTestCase {
    func testOptimizeAssignsAll() async throws {
        // Three clusters around axes
        let ivf = IVFIndex(dimension: 3, metric: .euclidean, config: .init(nlist: 3, nprobe: 1))
        try await ivf.batchInsert([
            ("a1", [1,0,0], nil), ("a2", [0.9,0,0], nil),
            ("b1", [0,1,0], nil), ("b2", [0,0.95,0], nil),
            ("c1", [0,0,1], nil), ("c2", [0,0,0.9], nil)
        ])
        try await ivf.optimize()
        let stats = await ivf.statistics()
        XCTAssertEqual(stats.indexType, "IVF")
        XCTAssertEqual(stats.vectorCount, 6)
        XCTAssertEqual(Int(stats.details["nlist"] ?? "0"), 3)
        XCTAssertEqual(Int(stats.details["assigned"] ?? "0"), 6)
    }

    func testSearchAfterOptimizeFindsCluster() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([
            ("x1", [0.0, 0.0], nil), ("x2", [0.05, -0.02], nil),
            ("y1", [1.0, 0.0], nil), ("y2", [0.95, 0.04], nil)
        ])
        try await ivf.optimize()
        // Query near [1,0]
        let res = try await ivf.search(query: [0.98, 0.0], k: 2, filter: nil)
        let got = Set(res.map{ $0.id })
        // Expect picks from y cluster
        XCTAssertFalse(got.isDisjoint(with: ["y1","y2"]))
    }
}
