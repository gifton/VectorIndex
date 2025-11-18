import XCTest
@testable import VectorIndex

final class HNSWRecallTests: XCTestCase {
    // Simple deterministic PRNG for repeatable tests
    struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = 2862933555777941757 &* state &+ 3037000493
            return state
        }
        mutating func nextFloat() -> Float {
            let x = next() >> 11 // 53 bits
            return Float(x) / Float(1 << 53)
        }
        mutating func nextInRange(_ range: ClosedRange<Float>) -> Float {
            let r = nextFloat()
            return range.lowerBound + (range.upperBound - range.lowerBound) * r
        }
    }

    func generateDataset(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = LCG(state: seed)
        var data: [[Float]] = []
        data.reserveCapacity(count)
        for _ in 0..<count {
            var v = (0..<dim).map { _ in rng.nextInRange(-1...1) }
            // normalize to unit length to keep distances well-conditioned
            let norm = sqrt(v.reduce(0) { $0 + $1*$1 })
            if norm > 0 { v = v.map { $0 / norm } }
            data.append(v)
        }
        return data
    }

    func topKFlat(query: [Float], data: [[Float]], ids: [String], k: Int) -> [String] {
        var scored: [(String, Float)] = []
        scored.reserveCapacity(data.count)
        for (i, v) in data.enumerated() {
            // Euclidean distance
            var sum: Float = 0
            for j in 0..<query.count { let d = query[j] - v[j]; sum += d*d }
            scored.append((ids[i], sqrt(sum)))
        }
        scored.sort { $0.1 < $1.1 }
        return Array(scored.prefix(k).map { $0.0 })
    }

    func recall(atK k: Int, truth: [String], approx: [String]) -> Float {
        let truthSet = Set(truth.prefix(k))
        let approxSet = Set(approx.prefix(k))
        let inter = truthSet.intersection(approxSet)
        return Float(inter.count) / Float(k)
    }

    func testHNSWRecallVsFlat() async throws {
        let dim = 32
        let n = 300
        let q = 20
        let k = 5
        let data = generateDataset(count: n, dim: dim, seed: 42)
        let queries = generateDataset(count: q, dim: dim, seed: 4242)
        let ids = (0..<n).map { "id\($0)" }

        // Build HNSW index
        let hnsw = HNSWIndex(dimension: dim, metric: .euclidean, config: .init(m: 12, efConstruction: 100, efSearch: 64))
        for i in 0..<n {
            try await hnsw.insert(id: ids[i], vector: data[i], metadata: nil)
        }

        // Measure recall over queries
        var recalls: [Float] = []
        recalls.reserveCapacity(q)
        for qi in 0..<q {
            let truth = topKFlat(query: queries[qi], data: data, ids: ids, k: k)
            let approx = try await hnsw.search(query: queries[qi], k: k, filter: nil).map { $0.id }
            recalls.append(recall(atK: k, truth: truth, approx: approx))
        }
        let avgRecall = recalls.reduce(0, +) / Float(recalls.count)

        // For these modest params, target a reasonable recall threshold
        XCTAssertGreaterThanOrEqual(avgRecall, 0.6, "Avg recall too low: \(avgRecall)")
    }
}
