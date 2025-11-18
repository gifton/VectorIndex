import XCTest
@testable import VectorIndex

final class HNSWMoreTests: XCTestCase {
    func testCountAndClear() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0, 0], metadata: nil)
        try await idx.insert(id: "b", vector: [1, 0], metadata: nil)
        // Count should be 2 (actor property)
        // We cannot access count synchronously here, but search non-empty indicates content
        var res = try await idx.search(query: [0.9, 0], k: 2, filter: nil)
        XCTAssertEqual(Set(res.map { $0.id }), Set(["a", "b"]))
        await idx.clear()
        res = try await idx.search(query: [0.9, 0], k: 2, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }

    func testFilterApplied() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.batchInsert([
            ("x", [0, 0], ["keep": "yes"]),
            ("y", [1, 0], ["keep": "no"])])
        let filter: @Sendable ([String: String]?) -> Bool = { meta in meta?["keep"] == "yes" }
        let res = try await idx.search(query: [0.1, 0], k: 2, filter: filter)
        XCTAssertEqual(res.map { $0.id }, ["x"])
    }

    func testDimensionMismatchThrows() async throws {
        let idx = HNSWIndex(dimension: 3)
        await XCTAssertThrowsErrorAsync(try await idx.insert(id: "bad", vector: [1, 2], metadata: nil))
    }
}
