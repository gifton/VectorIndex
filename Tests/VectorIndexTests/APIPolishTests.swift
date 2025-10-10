import XCTest
@testable import VectorIndex

final class APIPolishTests: XCTestCase {
    func testFlatContainsUpdateBatchRemove() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: ["m":"1"])
        do {
            let c = await idx.contains(id: "a")
            XCTAssertTrue(c)
        }
        // Update vector and metadata
        let updated = try await idx.update(id: "a", vector: [1,0], metadata: ["m":"2"])
        XCTAssertTrue(updated)
        var res = try await idx.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
        // Batch remove
        try await idx.batchRemove(["a"]) 
        do {
            let c = await idx.contains(id: "a")
            XCTAssertFalse(c)
        }
        res = try await idx.search(query: [0,0], k: 1, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }

    func testHNSWContainsUpdateBatchRemove() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "x", vector: [0,0], metadata: ["m":"1"])
        do {
            let c = await idx.contains(id: "x")
            XCTAssertTrue(c)
        }
        // Move x near [1,0]
        let ok = try await idx.update(id: "x", vector: [1,0], metadata: ["m":"2"])
        XCTAssertTrue(ok)
        let res = try await idx.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "x")
        // Remove
        try await idx.batchRemove(["x"]) 
        do {
            let c = await idx.contains(id: "x")
            XCTAssertFalse(c)
        }
    }

    func testIVFContainsUpdateBatchRemove() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([("p", [0,0], ["m":"1"]), ("q", [1,0], nil)])
        try await ivf.optimize()
        do {
            let c = await ivf.contains(id: "p")
            XCTAssertTrue(c)
        }
        // Update p far away so it reassigns
        let ok = try await ivf.update(id: "p", vector: [1,0], metadata: ["m":"2"])
        XCTAssertTrue(ok)
        var res = try await ivf.search(query: [0.0, 0], k: 1, filter: nil)
        XCTAssertNotEqual(res.first?.id, "p")
        // Batch remove
        try await ivf.batchRemove(["p","q"]) 
        do {
            let c1 = await ivf.contains(id: "p")
            let c2 = await ivf.contains(id: "q")
            XCTAssertFalse(c1)
            XCTAssertFalse(c2)
        }
        res = try await ivf.search(query: [0,0], k: 1, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }
}
