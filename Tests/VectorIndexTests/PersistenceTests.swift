import XCTest
@testable import VectorIndex

final class PersistenceTests: XCTestCase {
    func tempURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("VIndex_\(name)_\(UUID().uuidString).json")
    }

    func testFlatRoundTrip() async throws {
        let url = tempURL("flat")
        defer { try? FileManager.default.removeItem(at: url) }

        let idx = FlatIndex(dimension: 3)
        try await idx.batchInsert([
            ("a", [0,0,1], ["m":"x"]), ("b", [0,1,0], nil), ("c", [1,0,0], nil)
        ])
        try await idx.save(to: url)

        let loaded = try await FlatIndex.load(from: url)
        let res = try await loaded.search(query: [1,0,0], k: 2, filter: nil)
        XCTAssertEqual(res.first?.id, "c")
        let stats = await loaded.statistics()
        XCTAssertEqual(stats.indexType, "Flat")
        XCTAssertEqual(stats.vectorCount, 3)
    }

    func testHNSWRoundTrip() async throws {
        let url = tempURL("hnsw")
        defer { try? FileManager.default.removeItem(at: url) }

        let idx = HNSWIndex(dimension: 2)
        try await idx.batchInsert([("a", [0,0], nil), ("b", [1,0], nil), ("c", [0,1], nil)])
        try await idx.save(to: url)

        let loaded = try await HNSWIndex.load(from: url)
        let res = try await loaded.search(query: [0.9,0.1], k: 1, filter: nil)
        XCTAssertFalse(res.isEmpty)
        let stats = await loaded.statistics()
        XCTAssertEqual(stats.indexType, "HNSW")
        XCTAssertEqual(stats.dimension, 2)
    }

    func testIVFRoundTrip() async throws {
        let url = tempURL("ivf")
        defer { try? FileManager.default.removeItem(at: url) }

        let ivf = IVFIndex(dimension: 3, metric: .euclidean, config: .init(nlist: 4, nprobe: 2))
        try await ivf.batchInsert([("a", [0,0,1], nil), ("b", [0,1,0], nil), ("c", [1,0,0], nil)])
        try await ivf.optimize()
        try await ivf.save(to: url)

        let loaded = try await IVFIndex.load(from: url)
        let res = try await loaded.search(query: [1,0,0], k: 1, filter: nil)
        XCTAssertFalse(res.isEmpty)
        let stats = await loaded.statistics()
        XCTAssertEqual(stats.indexType, "IVF")
        XCTAssertGreaterThan(Int(stats.details["nlist"] ?? "0") ?? 0, 0)
    }
}
