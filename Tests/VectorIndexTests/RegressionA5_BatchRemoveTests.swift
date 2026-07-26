import XCTest
@testable import VectorIndex

final class RegressionA5_BatchRemoveTests: XCTestCase {
    func testBatchRemoveSubsetKeepsIndexSearchable() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0, 0], metadata: nil)
        try await idx.insert(id: "b", vector: [1, 0], metadata: nil)
        try await idx.insert(id: "c", vector: [0, 1], metadata: nil)
        try await idx.insert(id: "d", vector: [1, 1], metadata: nil)

        // Remove a subset only.
        try await idx.batchRemove(["b", "d"])

        // Survivors must still be findable (pre-fix: entryPoint=nil => empty results).
        let res = try await idx.search(query: [0, 0], k: 2, filter: nil)
        let ids = Set(res.map { $0.id })
        let countVal = await idx.count
        XCTAssertTrue(ids.contains("a"), "surviving point 'a' should be found")
        XCTAssertFalse(ids.contains("b"), "removed point 'b' should not be found")
        XCTAssertEqual(countVal, 2, "count should reflect 2 survivors")
    }
}
