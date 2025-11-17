import XCTest
@testable import VectorIndex

final class IVFListMaintenanceTests: XCTestCase {
    func testReplaceMovesBetweenLists() async throws {
        let cfg = IVFIndex.Configuration(nlist: 2, nprobe: 1)
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: cfg)
        try await ivf.batchInsert([("p", [0, 0], nil), ("q", [1, 0], nil)])
        try await ivf.optimize()
        // Query near [0,0] should return p
        var res = try await ivf.search(query: [0.01, 0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "p")
        // Replace p far from previous centroid
        try await ivf.insert(id: "p", vector: [1.0, 0], metadata: nil)
        // After replacement, search near [0,0] should not return p
        res = try await ivf.search(query: [0.01, 0], k: 1, filter: nil)
        XCTAssertNotEqual(res.first?.id, "p")
    }

    func testRemoveUpdatesLists() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 2))
        try await ivf.batchInsert([("p", [0, 0], nil), ("q", [1, 0], nil)])
        try await ivf.optimize()
        try await ivf.remove(id: "q")
        let res = try await ivf.search(query: [0.9, 0], k: 1, filter: nil)
        // Should not be q since it was removed
        XCTAssertNotEqual(res.first?.id, "q")
    }
}
