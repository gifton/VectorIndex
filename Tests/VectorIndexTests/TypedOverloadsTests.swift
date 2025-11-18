import XCTest
import VectorCore
@testable import VectorIndex

final class TypedOverloadsTests: XCTestCase {
    func testFlatTypedInsertSearch() async throws {
        let idx = FlatIndex(dimension: 2)
        let v: DynamicVector = DynamicVector([1, 0])
        try await idx.insert(id: "a", vector: v, metadata: nil)
        let q: DynamicVector = DynamicVector([0.9, 0])
        let res = try await idx.search(query: q, k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
    }

    func testHNSWTypedBatch() async throws {
        let idx = HNSWIndex(dimension: 2)
        let items: [(String, DynamicVector, [String: String]?)] = [
            ("x", DynamicVector([0, 0]), nil), ("y", DynamicVector([1, 0]), nil)
        ]
        try await idx.batchInsert(items)
        let queries = [DynamicVector([1, 0])]
        let res = try await idx.batchSearch(queries: queries, k: 1, filter: nil)
        XCTAssertEqual(res.first?.first?.id, "y")
    }

    func testIVFTyped() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([("p", DynamicVector([0, 0]), nil), ("q", DynamicVector([1, 0]), nil)])
        try await ivf.optimize()
        let res = try await ivf.search(query: DynamicVector([0.99, 0]), k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "q")
    }
}
