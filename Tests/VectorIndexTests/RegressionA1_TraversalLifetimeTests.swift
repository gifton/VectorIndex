import XCTest
@testable import VectorIndex

final class RegressionA1_TraversalLifetimeTests: XCTestCase {
    func testRepeatedSearchAndKNNGraphStable() async throws {
        let dim = 16, n = 400
        let idx = HNSWIndex(dimension: dim)
        var rng: UInt64 = 0xDEADBEEF
        func rnd() -> Float { rng = 2862933555777941757 &* rng &+ 3037000493; return Float(rng >> 40) / Float(1 << 24) }
        var vecs: [[Float]] = []
        for i in 0..<n {
            let v = (0..<dim).map { _ in rnd() * 2 - 1 }
            vecs.append(v)
            try await idx.insert(id: "id\(i)", vector: v, metadata: nil)
        }
        // Many searches: results must be stable and non-empty across repeats.
        let q = vecs[0]
        let baseline = try await idx.search(query: q, k: 10, filter: nil).map { $0.id }
        XCTAssertFalse(baseline.isEmpty)
        for _ in 0..<200 {
            let r = try await idx.search(query: q, k: 10, filter: nil).map { $0.id }
            XCTAssertEqual(r, baseline, "search results must be deterministic/stable")
        }
        // buildKNNGraph exercises the third escaping site.
        let (g1, _) = try await idx.buildKNNGraph(k: 10)
        let (g2, _) = try await idx.buildKNNGraph(k: 10)
        XCTAssertEqual(g1.neighborIndices, g2.neighborIndices, "kNN graph must be deterministic")
    }
}
