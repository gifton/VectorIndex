import XCTest
@testable import VectorIndex

final class IVFMoreTests: XCTestCase {
    func testLinearScanBeforeOptimize() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 4, nprobe: 2))
        try await ivf.batchInsert([
            ("a", [0,0], nil), ("b", [1,0], nil), ("c", [0,1], nil)
        ])
        // Without optimize, should still return valid results
        let res = try await ivf.search(query: [0.9,0], k: 2, filter: nil)
        XCTAssertEqual(res.first?.id, "b")
        XCTAssertEqual(res.count, 2)
    }

    func testNprobeClampingAndSearch() async throws {
        // nprobe > nlist should clamp to nlist without crash
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 10))
        try await ivf.batchInsert([
            ("x", [0,0], nil), ("y", [1,0], nil), ("z", [0,1], nil)
        ])
        try await ivf.optimize()
        let res = try await ivf.search(query: [0.9,0], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "y")
    }

    func testClearResets() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([
            ("x", [0,0], nil), ("y", [1,0], nil)
        ])
        try await ivf.optimize()
        await ivf.clear()
        let res = try await ivf.search(query: [0,0], k: 1, filter: nil)
        XCTAssertTrue(res.isEmpty)
    }

    func testDotProductMetric() async throws {
        // With dot product distance (negative dot), the closer in angle should win
        let ivf = IVFIndex(dimension: 2, metric: .dotProduct, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([
            ("a", [1,0], nil), ("b", [0,1], nil)
        ])
        try await ivf.optimize()
        let res = try await ivf.search(query: [0.9, 0.1], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
    }
}

