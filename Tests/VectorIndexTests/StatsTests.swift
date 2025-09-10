import XCTest
@testable import VectorIndex

final class StatsTests: XCTestCase {
    func testFlatStats() async throws {
        let idx = FlatIndex(dimension: 3)
        try await idx.insert(id: "a", vector: [0,0,1], metadata: nil)
        let stats = await idx.statistics()
        XCTAssertEqual(stats.indexType, "Flat")
        XCTAssertEqual(stats.vectorCount, 1)
        XCTAssertEqual(stats.dimension, 3)
    }

    func testIVFStats() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 4, nprobe: 2))
        try await ivf.batchInsert([("x", [0,0], nil), ("y", [1,0], nil)])
        try await ivf.optimize()
        let stats = await ivf.statistics()
        XCTAssertEqual(stats.indexType, "IVF")
        XCTAssertEqual(stats.vectorCount, 2)
        XCTAssertEqual(stats.details["nlist"], String(4))
        XCTAssertEqual(stats.details["nprobe"], String(2))
    }

    func testHNSWStats() async throws {
        let hnsw = HNSWIndex(dimension: 2)
        try await hnsw.batchInsert([("a", [0,0], nil), ("b", [1,0], nil)])
        let stats = await hnsw.statistics()
        XCTAssertEqual(stats.indexType, "HNSW")
        XCTAssertEqual(stats.vectorCount, 2)
        XCTAssertEqual(stats.dimension, 2)
        XCTAssertNotNil(stats.details["maxLevel"])
    }
}

