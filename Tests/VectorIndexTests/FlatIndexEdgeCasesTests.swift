import XCTest
@testable import VectorIndex

final class FlatIndexEdgeCasesTests: XCTestCase {
    func testKZeroReturnsEmpty() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.batchInsert([("a", [0,0], nil), ("b", [1,0], nil)])
        let res = try await idx.search(query: [0,0], k: 0, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }

    func testKGreaterThanCountClamps() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.batchInsert([("a", [0,0], nil), ("b", [1,0], nil)])
        let res = try await idx.search(query: [0,0], k: 10, filter: nil)
        XCTAssertEqual(res.count, 2)
    }

    func testSearchDimensionMismatchThrows() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: nil)
        await XCTAssertThrowsErrorAsync(try await idx.search(query: [0,0,1], k: 1, filter: nil))
    }

    func testBatchSearchDimensionMismatchThrows() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: nil)
        await XCTAssertThrowsErrorAsync(try await idx.batchSearch(queries: [[0,0], [0,0,1]], k: 1, filter: nil))
    }
}
