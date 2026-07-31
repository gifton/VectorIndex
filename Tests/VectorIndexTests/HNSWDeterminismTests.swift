import XCTest
import VectorCore
@testable import VectorIndex

/// Construction-time determinism gate: independent builds from the same seed and
/// insertion order must produce byte-identical graphs. `HNSWKNNGraphTests.testDeterminism`
/// only proves *query-time* determinism (calling `buildKNNGraph` twice on one already-built
/// index); nothing previously asserted that two separately constructed indices converge to
/// the same structure. Later Phase-3 gates (Task 5) depend on this test being meaningful.
final class HNSWDeterminismTests: XCTestCase {
    // Same LCG data-gen pattern as HNSWKNNGraphTests.generateDataset.
    struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = 2862933555777941757 &* state &+ 3037000493
            return state
        }
        mutating func nextFloat() -> Float {
            Float(next() >> 11) / Float(1 << 53)
        }
        mutating func nextInRange(_ range: ClosedRange<Float>) -> Float {
            range.lowerBound + (range.upperBound - range.lowerBound) * nextFloat()
        }
    }

    func generateDataset(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = LCG(state: seed)
        var data: [[Float]] = []
        data.reserveCapacity(count)
        for _ in 0..<count {
            var v = (0..<dim).map { _ in rng.nextInRange(-1...1) }
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            if norm > 0 { v = v.map { $0 / norm } }
            data.append(v)
        }
        return data
    }

    func testIndependentBuildsProduceIdenticalGraphs() async throws {
        let dim = 32
        let data = generateDataset(count: 600, dim: dim, seed: 987)
        func build() async throws -> HNSWIndex {
            let idx = HNSWIndex(dimension: dim, metric: .euclidean,
                                config: .init(m: 8, efConstruction: 64, efSearch: 32, rngSeed: 42))
            for (i, v) in data.enumerated() {
                try await idx.insert(id: "id\(i)", vector: v, metadata: nil)
            }
            return idx
        }
        let a = try await build()
        let b = try await build()
        let sa = await a._testGraphSnapshot()
        let sb = await b._testGraphSnapshot()
        XCTAssertEqual(sa.entryPoint, sb.entryPoint)
        XCTAssertEqual(sa.maxLevel, sb.maxLevel)
        XCTAssertEqual(sa.levels, sb.levels)
        XCTAssertEqual(sa.adjacency, sb.adjacency,
            "same seed + same insertion order must produce a byte-identical graph")
    }
}
