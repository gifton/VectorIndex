import XCTest
@testable import VectorIndex

final class VectorIndexTests: XCTestCase {
    func testFlatIndexBasic() async throws {
        let index = FlatIndex(dimension: 3)
        try await index.insert(id: "a", vector: [0, 0, 1], metadata: ["cat": "x"])
        try await index.insert(id: "b", vector: [0, 1, 0], metadata: ["cat": "y"])
        try await index.insert(id: "c", vector: [1, 0, 0], metadata: ["cat": "x"])

        let results = try await index.search(query: [1, 0, 0], k: 2, filter: nil)
        XCTAssertEqual(results.first?.id, "c")
        XCTAssertEqual(results.count, 2)
    }

    func testHNSWSkeletonCompiles() async throws {
        let hnsw = HNSWIndex(dimension: 3)
        try await hnsw.batchInsert([
            ("a", [0,0,1], nil),
            ("b", [0,1,0], nil),
            ("c", [1,0,0], nil)
        ])
        let results = try await hnsw.search(query: [1,0,0], k: 1, filter: nil)
        XCTAssertEqual(results.count, 1)
    }

    func testIVFSkeletonCompiles() async throws {
        let ivf = IVFIndex(dimension: 3, config: .init(nlist: 2, nprobe: 1))
        try await ivf.insert(id: "x", vector: [0.1, 0.2, 0.3], metadata: nil)
        let results = try await ivf.search(query: [0.0, 0.2, 0.3], k: 1, filter: nil)
        XCTAssertEqual(results.count, 1)
    }
}
