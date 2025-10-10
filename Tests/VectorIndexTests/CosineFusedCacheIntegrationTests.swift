import XCTest
@testable import VectorIndex

final class CosineFusedCacheIntegrationTests: XCTestCase {
    func testFusedCosineCacheMatchesTwoPass() async throws {
        let d = 64
        let n = 200
        let k = 10

        let idx = FlatIndexOptimized(dimension: d, metric: .cosine)

        // Insert contiguous AoS data (compact layout)
        for i in 0..<n {
            let v = (0..<d).map { _ in Float.random(in: -1...1) }
            try await idx.insert(id: "id_\(i)", vector: v, metadata: nil)
        }

        let q = (0..<d).map { _ in Float.random(in: -1...1) }

        // Baseline: two-pass cosine (no fused cache)
        let baseline = try await idx.search(query: q, k: k, filter: nil)

        // Enable fused cosine cache (f16) and compare
        try await idx.enableCosineFusedNormCache(dtype: .float16)
        let fused = try await idx.search(query: q, k: k, filter: nil)

        XCTAssertEqual(baseline.map { $0.id }, fused.map { $0.id }, "Top-k IDs should match between two-pass and fused paths")
        zip(baseline, fused).forEach { (a, b) in
            XCTAssertLessThan(abs(a.score - b.score), 1e-4, "Scores should match within tolerance")
        }

        // Cleanup: disable cache to avoid affecting other tests
        await idx.disableCosineFusedNormCache()
    }
}

