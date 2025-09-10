import XCTest
@testable import VectorIndex

final class APIPolishTests: XCTestCase {
    func testFlatContainsUpdateBatchRemove() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: ["m":"1"])
        XCTAssertTrue(await idx.contains(id: "a"))
        // Update vector and metadata
        let updated = try await idx.update(id: "a", vector: [1,0], metadata: ["m":"2"])
        XCTAssertTrue(updated)
        var res = try await idx.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
        // Batch remove
        try await idx.batchRemove(["a"]) 
        XCTAssertFalse(await idx.contains(id: "a"))
        res = try await idx.search(query: [0,0], k: 1, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }

    func testHNSWContainsUpdateBatchRemove() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "x", vector: [0,0], metadata: ["m":"1"])
        XCTAssertTrue(await idx.contains(id: "x"))
        // Move x near [1,0]
        let ok = try await idx.update(id: "x", vector: [1,0], metadata: ["m":"2"])
        XCTAssertTrue(ok)
        let res = try await idx.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "x")
        // Remove
        try await idx.batchRemove(["x"]) 
        XCTAssertFalse(await idx.contains(id: "x"))
    }

    func testIVFContainsUpdateBatchRemove() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([("p", [0,0], ["m":"1"]), ("q", [1,0], nil)])
        try await ivf.optimize()
        XCTAssertTrue(await ivf.contains(id: "p"))
        // Update p far away so it reassigns
        let ok = try await ivf.update(id: "p", vector: [1,0], metadata: ["m":"2"])
        XCTAssertTrue(ok)
        var res = try await ivf.search(query: [0.0, 0], k: 1, filter: nil)
        XCTAssertNotEqual(res.first?.id, "p")
        // Batch remove
        try await ivf.batchRemove(["p","q"]) 
        XCTAssertFalse(await ivf.contains(id: "p"))
        XCTAssertFalse(await ivf.contains(id: "q"))
        res = try await ivf.search(query: [0,0], k: 1, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }
}
