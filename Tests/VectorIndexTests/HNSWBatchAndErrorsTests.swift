import XCTest
@testable import VectorIndex

final class HNSWBatchAndErrorsTests: XCTestCase {
    func testBatchSearchAndKClamp() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.batchInsert([("a", [0,0], nil), ("b", [1,0], nil)])
        let res = try await idx.batchSearch(queries: [[0,0], [1,0]], k: 10, filter: nil)
        XCTAssertEqual(res.count, 2)
        XCTAssertEqual(res[0].count, 2)
        XCTAssertEqual(res[1].count, 2)
    }

    func testSearchDimensionMismatchThrows() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: nil)
        await XCTAssertThrowsErrorAsync(try await idx.search(query: [0,0,1], k: 1, filter: nil))
    }

    func testRemoveNonexistentDoesNotThrow() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: nil)
        // Should not throw
        try await idx.remove(id: "zzz")
        let res = try await idx.search(query: [0,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
    }
}
