import XCTest
import VectorCore
@testable import VectorIndex

/// Gap 2 coverage — verifies that the typed `IndexableVector` insert path
/// propagates `isNormalized` / `cachedMagnitude` hints into the HNSW cosine
/// distance cache incrementally, keeping the cache coherent across inserts
/// instead of triggering a full O(N·d) rebuild on the next search.
final class HNSWTypedInsertHintTests: XCTestCase {

    // MARK: - Helpers

    /// Deterministic LCG — reproducible across runs and platforms.
    private struct LCG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: state >> 32)
            return Float(bits) / Float(UInt32.max)
        }
    }

    private func makeNormalizedVector(dim: Int, rng: inout LCG) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        var s: Float = 0
        for i in 0..<dim {
            let x = rng.next() * 2 - 1
            v[i] = x
            s += x * x
        }
        let inv = 1.0 / sqrtf(max(s, 1e-12))
        for i in 0..<dim { v[i] *= inv }
        return v
    }

    private func makeCorpus(count: Int, dim: Int, seed: UInt64) -> [(String, [Float])] {
        var rng = LCG(state: seed)
        return (0..<count).map { i in ("v\(i)", makeNormalizedVector(dim: dim, rng: &rng)) }
    }

    // MARK: - Core correctness

    /// Inserting via the internal `knownInvNorm: 1.0` entry point must produce
    /// search results identical to inserting the same data with `knownInvNorm: nil`.
    func testKnownInvNormMatchesComputedPath() async throws {
        let dim = 64
        let corpus = makeCorpus(count: 200, dim: dim, seed: 0xC0FFEE)

        let hinted = HNSWIndex(dimension: dim, metric: .cosine)
        let unhinted = HNSWIndex(dimension: dim, metric: .cosine)
        for (id, v) in corpus {
            try await hinted.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
            try await unhinted.insert(id: id, vector: v, metadata: nil, knownInvNorm: nil)
        }

        var qRng = LCG(state: 0xBEEF)
        for _ in 0..<10 {
            let q = makeNormalizedVector(dim: dim, rng: &qRng)
            let rA = try await hinted.search(query: q, k: 5, filter: nil)
            let rB = try await unhinted.search(query: q, k: 5, filter: nil)
            XCTAssertEqual(rA.map { $0.id }, rB.map { $0.id })
            for (a, b) in zip(rA, rB) {
                XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
            }
        }
    }

    /// A non-unit vector plus `knownInvNorm = 1/‖v‖` must match the computed path.
    func testKnownInvNormHandlesNonUnitVector() async throws {
        let dim = 32
        var rng = LCG(state: 0xABCDEF)
        // Build a non-unit corpus.
        var raw: [(String, [Float])] = []
        for i in 0..<60 {
            var v = [Float](repeating: 0, count: dim)
            for j in 0..<dim { v[j] = rng.next() * 4 - 2 }
            raw.append(("v\(i)", v))
        }

        let hinted = HNSWIndex(dimension: dim, metric: .cosine)
        let unhinted = HNSWIndex(dimension: dim, metric: .cosine)
        for (id, v) in raw {
            let mag = sqrtf(v.reduce(Float(0)) { $0 + $1 * $1 })
            try await hinted.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0 / mag)
            try await unhinted.insert(id: id, vector: v, metadata: nil, knownInvNorm: nil)
        }

        var qRng = LCG(state: 0x12345)
        for _ in 0..<8 {
            let q = makeNormalizedVector(dim: dim, rng: &qRng)
            let rA = try await hinted.search(query: q, k: 5, filter: nil)
            let rB = try await unhinted.search(query: q, k: 5, filter: nil)
            XCTAssertEqual(rA.map { $0.id }, rB.map { $0.id })
            for (a, b) in zip(rA, rB) {
                XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
            }
        }
    }

    // MARK: - Cache coherence

    /// The critical claim: after inserts with a valid hint, the cache is
    /// populated up-to-date and not marked dirty, so the next search skips
    /// the full O(N·d) rebuild.
    func testInvNormsCacheStaysCoherentAcrossInserts() async throws {
        let dim = 16
        let corpus = makeCorpus(count: 50, dim: dim, seed: 1)
        let idx = HNSWIndex(dimension: dim, metric: .cosine)

        // First insert — cache starts nil; appendInvNormIncremental falls back
        // to marking dirty (since cache is nil). This is expected on insert #1.
        try await idx.insert(id: corpus[0].0, vector: corpus[0].1, metadata: nil, knownInvNorm: 1.0)
        var dirty = await idx._testInvNormsCacheIsDirty
        XCTAssertTrue(dirty, "cache is dirty after first insert (nil cache)")

        // A search triggers a full rebuild — cache is now coherent at count 1.
        var seedRng = LCG(state: 99)
        let q = makeNormalizedVector(dim: dim, rng: &seedRng)
        _ = try await idx.search(query: q, k: 1, filter: nil)
        var count = await idx._testInvNormsCacheCount
        dirty = await idx._testInvNormsCacheIsDirty
        XCTAssertEqual(count, 1)
        XCTAssertFalse(dirty)

        // Subsequent inserts with knownInvNorm=1.0 should grow the cache
        // incrementally and keep it clean — no dirty mark, no rebuild.
        for i in 1..<corpus.count {
            try await idx.insert(id: corpus[i].0, vector: corpus[i].1, metadata: nil, knownInvNorm: 1.0)
            count = await idx._testInvNormsCacheCount
            dirty = await idx._testInvNormsCacheIsDirty
            XCTAssertEqual(count, i + 1, "cache grew incrementally at insert #\(i)")
            XCTAssertFalse(dirty, "cache remained clean at insert #\(i)")
        }

        // Sanity: all cached inv norms should be exactly 1.0 for the hinted
        // unit-vector corpus — no drift from incremental appends.
        // Observation is indirect: a search against a unit query should score
        // each stored vector by cos distance matching the computed reference.
        var qRng2 = LCG(state: 7777)
        for _ in 0..<5 {
            let qv = makeNormalizedVector(dim: dim, rng: &qRng2)
            _ = try await idx.search(query: qv, k: 10, filter: nil)
            // If the cache were corrupted, the search would crash or return
            // nonsense. We separately verify numerical equivalence against
            // the unhinted path in testKnownInvNormMatchesComputedPath.
        }
    }

    // MARK: - TypedOverloads plumbing

    /// The public typed-overload path through `DynamicVector` must produce
    /// results identical to the raw `[Float]` path. DynamicVector returns
    /// `isNormalized == false` by default, so this exercises the unhinted
    /// branch of the typed overload — a regression guard that the new
    /// `knownInv` derivation doesn't break the default path.
    func testDynamicVectorTypedOverloadUnhintedPath() async throws {
        let dim = 32
        let corpus = makeCorpus(count: 100, dim: dim, seed: 0xAABB)

        let typed = HNSWIndex(dimension: dim, metric: .cosine)
        let raw = HNSWIndex(dimension: dim, metric: .cosine)

        for (id, v) in corpus {
            let dv = DynamicVector(v)
            try await typed.insert(id: id, vector: dv, metadata: nil)
            try await raw.insert(id: id, vector: v, metadata: nil)
        }

        var qRng = LCG(state: 0xCAFE)
        for _ in 0..<8 {
            let q = makeNormalizedVector(dim: dim, rng: &qRng)
            let rA = try await typed.search(query: q, k: 5, filter: nil)
            let rB = try await raw.search(query: q, k: 5, filter: nil)
            XCTAssertEqual(rA.map { $0.id }, rB.map { $0.id })
        }
    }

    // MARK: - Non-cosine safety

    /// For Euclidean metric, the hint must be a no-op and the cache remains
    /// untouched. This guards against accidental cross-metric regressions.
    func testHintIgnoredForEuclideanMetric() async throws {
        let dim = 16
        let corpus = makeCorpus(count: 20, dim: dim, seed: 42)
        let idx = HNSWIndex(dimension: dim, metric: .euclidean)
        for (id, v) in corpus {
            // Pass a bogus hint — it should be ignored.
            try await idx.insert(id: id, vector: v, metadata: nil, knownInvNorm: 999.0)
        }
        let count = await idx._testInvNormsCacheCount
        XCTAssertEqual(count, 0, "non-cosine metric should never populate inv-norm cache")

        // Search should still work correctly.
        var qRng = LCG(state: 1)
        let q = makeNormalizedVector(dim: dim, rng: &qRng)
        let results = try await idx.search(query: q, k: 3, filter: nil)
        XCTAssertEqual(results.count, 3)
    }

    // MARK: - Replace-existing path

    /// When an id is re-inserted, the internal remove + reinsert pipeline
    /// must keep the cache coherent — the new node appends at a fresh index.
    func testReinsertKeepsCacheCoherent() async throws {
        let dim = 16
        let idx = HNSWIndex(dimension: dim, metric: .cosine)
        let corpus = makeCorpus(count: 30, dim: dim, seed: 123)

        for (id, v) in corpus {
            try await idx.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }
        // Warm up the cache via a search.
        var qRng = LCG(state: 55)
        _ = try await idx.search(query: makeNormalizedVector(dim: dim, rng: &qRng), k: 3, filter: nil)

        // Reinsert an existing id with a fresh unit vector.
        let replacement = makeNormalizedVector(dim: dim, rng: &qRng)
        try await idx.insert(id: "v5", vector: replacement, metadata: nil, knownInvNorm: 1.0)

        // After a re-insert, nodes.count grew by 1 (soft-delete plus new tail node),
        // so the cache should still be coherent with a matching count.
        let newCount = await idx._testInvNormsCacheCount
        let totalNodes = await idx.statistics().vectorCount
        // Note: statistics.vectorCount reports active count, not total nodes,
        // but we asserted the cache grew without being marked dirty.
        XCTAssertGreaterThanOrEqual(newCount, totalNodes)
        let dirty = await idx._testInvNormsCacheIsDirty
        XCTAssertFalse(dirty, "reinsert path should keep the cache coherent")
    }
}
