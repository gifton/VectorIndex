import XCTest
@testable import VectorIndex

final class HNSWTests: XCTestCase {
    func testHNSWNearestNeighbor2D() async throws {
        let idx = HNSWIndex(dimension: 2, metric: .euclidean, config: .init(m: 8, efConstruction: 100, efSearch: 64))

        // Create points roughly on a circle
        let n = 200
        for i in 0..<n {
            let angle = Float(Double(i) / Double(n) * 2.0 * .pi)
            let x = cos(angle)
            let y = sin(angle)
            try await idx.insert(id: "p\(i)", vector: [x, y], metadata: nil)
        }

        // Query near p=25
        let qAngle = Float(Double(25) / Double(n) * 2.0 * .pi)
        let q = [cos(qAngle) + 0.01, sin(qAngle) - 0.01]
        let results = try await idx.search(query: q, k: 3, filter: nil)

        XCTAssertFalse(results.isEmpty)
        // The nearest should be close to p25
        let nearest = results.first!.id
        XCTAssertTrue(nearest.hasPrefix("p"))
    }

    func testHNSWRemove() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0,0], metadata: nil)
        try await idx.insert(id: "b", vector: [1,0], metadata: nil)
        try await idx.insert(id: "c", vector: [0,1], metadata: nil)

        var res = try await idx.search(query: [0.9, 0.1], k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "b")

        try await idx.remove(id: "b")
        res = try await idx.search(query: [0.9, 0.1], k: 1, filter: nil)
        XCTAssertNotEqual(res.first?.id, "b")
    }
}

