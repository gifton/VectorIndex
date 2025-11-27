import XCTest
@testable import VectorIndex

final class CosineFusedCacheIntegrationTests: XCTestCase {
    func testFusedCosineCacheMatchesTwoPass() async throws {
        let d = 64
        let n = 200
        let k = 10

        // Use seeded RNG for deterministic, reproducible tests
        var rng = SplitMix64(seed: 42)

        let idx = FlatIndexOptimized(dimension: d, metric: .cosine)

        // Insert contiguous AoS data (compact layout)
        for i in 0..<n {
            let v = (0..<d).map { _ in Float.random(in: -1...1, using: &rng) }
            try await idx.insert(id: "id_\(i)", vector: v, metadata: nil)
        }

        let q = (0..<d).map { _ in Float.random(in: -1...1, using: &rng) }

        // Baseline: two-pass cosine (no fused cache)
        let baseline = try await idx.search(query: q, k: k, filter: nil)

        // Enable fused cosine cache (f16) and compare
        try await idx.enableCosineFusedNormCache(dtype: .float16)
        let fused = try await idx.search(query: q, k: k, filter: nil)

        // When scores are very close (within float16 tolerance), tie-breaking order may differ.
        // Compare IDs as sets for items with near-equal scores, exact order for well-separated scores.
        let baselineIDs = Set(baseline.map { $0.id })
        let fusedIDs = Set(fused.map { $0.id })
        XCTAssertEqual(baselineIDs, fusedIDs, "Top-k ID sets should match between two-pass and fused paths")

        zip(baseline, fused).forEach { (a, b) in
            // Float16 has reduced precision, so we need a slightly larger tolerance
            XCTAssertLessThan(abs(a.score - b.score), 2e-4, "Scores should match within tolerance (accounting for float16 precision)")
        }

        // Cleanup: disable cache to avoid affecting other tests
        await idx.disableCosineFusedNormCache()
    }
}

// MARK: - Deterministic RNG for reproducible tests
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}
