import XCTest
@testable import VectorIndex

final class FlatIndexTests: XCTestCase {
    func testInsertAndSearchEuclidean() async throws {
        let idx = FlatIndex(dimension: 3, metric: .euclidean)
        try await idx.insert(id: "a", vector: [0,0,1], metadata: nil)
        try await idx.insert(id: "b", vector: [0,1,0], metadata: nil)
        try await idx.insert(id: "c", vector: [1,0,0], metadata: nil)
        let res = try await idx.search(query: [1,0,0], k: 2, filter: nil)
        XCTAssertEqual(res.first?.id, "c")
        XCTAssertEqual(res.count, 2)
    }

    func testFilterIsApplied() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.batchInsert([
            ("x", [0,0], ["keep": "yes"]),
            ("y", [1,0], ["keep": "no"])])
        let filter: @Sendable ([String:String]?) -> Bool = { meta in meta?["keep"] == "yes" }
        let res = try await idx.search(query: [0.1,0], k: 2, filter: filter)
        XCTAssertEqual(res.map{ $0.id }, ["x"])
    }

    func testBatchSearch() async throws {
        let idx = FlatIndex(dimension: 3)
        try await idx.batchInsert([
            ("a", [0,0,1], nil),
            ("b", [0,1,0], nil),
            ("c", [1,0,0], nil)
        ])
        let queries: [[Float]] = [[1,0,0], [0,1,0]]
        let res = try await idx.batchSearch(queries: queries, k: 1, filter: nil)
        XCTAssertEqual(res.count, 2)
        XCTAssertEqual(res[0].first?.id, "c")
        XCTAssertEqual(res[1].first?.id, "b")
    }

    func testRemoveAndClear() async throws {
        let idx = FlatIndex(dimension: 2)
        try await idx.batchInsert([
            ("a", [0,0], nil), ("b", [1,0], nil)
        ])
        var res = try await idx.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "b")
        try await idx.remove(id: "b")
        res = try await idx.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
        await idx.clear()
        let empty = try await idx.search(query: [0,0], k: 1, filter: nil)
        XCTAssertTrue(empty.isEmpty)
    }

    func testDimensionMismatchInsertThrows() async throws {
        let idx = FlatIndex(dimension: 3)
        await XCTAssertThrowsErrorAsync(try await idx.insert(id: "bad", vector: [1,2], metadata: nil))
    }

    func testVariousMetrics() async throws {
        // Points in 2D to make metric behavior obvious
        let a = [1.0, 0.0]
        let b = [0.0, 1.0]
        let q = [0.9, 0.1]

        // Euclidean
        do {
            let idx = FlatIndex(dimension: 2, metric: .euclidean)
            try await idx.batchInsert([("a", a, nil), ("b", b, nil)])
            let res = try await idx.search(query: q, k: 1, filter: nil)
            XCTAssertEqual(res.first?.id, "a")
        }
        // Manhattan
        do {
            let idx = FlatIndex(dimension: 2, metric: .manhattan)
            try await idx.batchInsert([("a", a, nil), ("b", b, nil)])
            let res = try await idx.search(query: q, k: 1, filter: nil)
            XCTAssertEqual(res.first?.id, "a")
        }
        // Chebyshev
        do {
            let idx = FlatIndex(dimension: 2, metric: .chebyshev)
            try await idx.batchInsert([("a", a, nil), ("b", b, nil)])
            let res = try await idx.search(query: q, k: 1, filter: nil)
            XCTAssertEqual(res.first?.id, "a")
        }
        // Cosine: q closer to a than b
        do {
            let idx = FlatIndex(dimension: 2, metric: .cosine)
            try await idx.batchInsert([("a", a, nil), ("b", b, nil)])
            let res = try await idx.search(query: q, k: 1, filter: nil)
            XCTAssertEqual(res.first?.id, "a")
        }
        // Dot product distance (negative dot): maximize dot product → minimize distance
        do {
            let idx = FlatIndex(dimension: 2, metric: .dotProduct)
            try await idx.batchInsert([("a", a, nil), ("b", b, nil)])
            let res = try await idx.search(query: q, k: 1, filter: nil)
            XCTAssertEqual(res.first?.id, "a")
        }
    }
}

// MARK: - Async throws expectation helper
extension XCTestCase {
    func XCTAssertThrowsErrorAsync(_ expression: @autoclosure () async throws -> Void, _ message: @autoclosure () -> String = "", file: StaticString = #filePath, line: UInt = #line) async {
        do {
            try await expression()
            XCTFail(message(), file: file, line: line)
        } catch { /* expected */ }
    }
}

