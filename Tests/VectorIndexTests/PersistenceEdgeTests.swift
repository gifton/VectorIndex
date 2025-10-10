import XCTest
@testable import VectorIndex

final class PersistenceEdgeTests: XCTestCase {
    func tempURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("VIndex_Edge_\(name)_\(UUID().uuidString).json")
    }

    func testFlatEmptyRoundTrip() async throws {
        let url = tempURL("flat_empty")
        defer { try? FileManager.default.removeItem(at: url) }

        let idx = FlatIndex(dimension: 5)
        try await idx.save(to: url)
        let loaded = try await FlatIndex.load(from: url)
        let stats = await loaded.statistics()
        XCTAssertEqual(stats.vectorCount, 0)
        let res = try await loaded.search(query: Array(repeating: 0, count: 5), k: 1, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }

    func testIVFSaveWithoutOptimizeThenLoad() async throws {
        let url = tempURL("ivf_no_opt")
        defer { try? FileManager.default.removeItem(at: url) }

        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 4, nprobe: 2))
        try await ivf.batchInsert([("a", [0,0], nil), ("b", [1,0], nil), ("c", [0,1], nil)])
        // Save without calling optimize
        try await ivf.save(to: url)
        let loaded = try await IVFIndex.load(from: url)
        // load() runs optimize() internally; search should work
        let res = try await loaded.search(query: [0.9, 0.0], k: 1, filter: nil)
        XCTAssertFalse(res.isEmpty)
    }

    func testLoadWrongTypeThrows() async throws {
        // Save Flat, attempt to load as IVF and HNSW
        let url = tempURL("wrong_type")
        defer { try? FileManager.default.removeItem(at: url) }

        let flat = FlatIndex(dimension: 3)
        try await flat.batchInsert([("x", [0,0,1], nil)])
        try await flat.save(to: url)

        do {
            _ = try await IVFIndex.load(from: url)
            XCTFail("Expected IVFIndex.load to throw for Flat payload")
        } catch { /* expected */ }

        do {
            _ = try await HNSWIndex.load(from: url)
            XCTFail("Expected HNSWIndex.load to throw for Flat payload")
        } catch { /* expected */ }
    }

    func testLoadCorruptedJSONThrows() async throws {
        let url = tempURL("corrupt")
        defer { try? FileManager.default.removeItem(at: url) }
        let bad = Data("not json".utf8)
        try bad.write(to: url)
        await XCTAssertThrowsErrorAsync(try await FlatIndex.load(from: url))
        await XCTAssertThrowsErrorAsync(try await IVFIndex.load(from: url))
        await XCTAssertThrowsErrorAsync(try await HNSWIndex.load(from: url))
    }

    func testHNSWCompactReducesDeleted() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.batchInsert([("a", [0,0], nil), ("b", [1,0], nil), ("c", [0,1], nil)])
        try await idx.remove(id: "b")
        try await idx.compact()
        let stats = await idx.statistics()
        XCTAssertEqual(stats.vectorCount, 2)
        // Search should not return deleted id
        let res = try await idx.search(query: [0.9, 0], k: 3, filter: nil)
        XCTAssertFalse(res.map{ $0.id }.contains("b"))
    }
}
