import XCTest
@testable import VectorIndex

final class IVFTests: XCTestCase {
    func testIVFBasicSearch() async throws {
        let ivf = IVFIndex(dimension: 3, metric: .euclidean, config: .init(nlist: 2, nprobe: 2))
        try await ivf.batchInsert([
            ("a", [0, 0, 1], ["cat": "x"]),
            ("b", [0, 1, 0], ["cat": "y"]),
            ("c", [1, 0, 0], ["cat": "x"])
        ])
        try await ivf.optimize()
        let res = try await ivf.search(query: [1, 0, 0], k: 2, filter: nil)
        XCTAssertEqual(res.first?.id, "c")
        XCTAssertEqual(res.count, 2)
    }

    func testIVFFilter() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 4, nprobe: 1))
        try await ivf.batchInsert([
            ("x", [0, 0], ["keep": "yes"]),
            ("y", [1, 0], ["keep": "no"])
        ])
        try await ivf.optimize()
        let res = try await ivf.search(query: [0.1, 0], k: 2, filter: { meta in meta?["keep"] == "yes" })
        XCTAssertEqual(res.map { $0.id }, ["x"])
    }

    func testIVFOptimizeDoesNotCrashSmall() async throws {
        let ivf = IVFIndex(dimension: 4, metric: .euclidean, config: .init(nlist: 8, nprobe: 2))
        // fewer points than nlist
        try await ivf.batchInsert([
            ("a", [1, 0, 0, 0], nil), ("b", [0, 1, 0, 0], nil), ("c", [0, 0, 1, 0], nil)
        ])
        try await ivf.optimize()
        let res = try await ivf.search(query: [0.9, 0.1, 0, 0], k: 1, filter: nil)
        XCTAssertFalse(res.isEmpty)
    }
}
