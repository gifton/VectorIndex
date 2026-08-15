//
//  IVFBatchGEMMParityTests.swift
//  VectorIndexTests
//
//  Task 7 (P3b): characterization test written BEFORE the GEMM cross-term
//  swap. Must pass against the pre-existing per-query batch path (Step 1)
//  and continue to pass once `batchSearch` is rewired onto
//  `CentroidBatchScore` (Step 3+). `batchSearch` and `search()` must agree
//  almost everywhere on a fully-seeded, deterministic fixture; any
//  divergence must be bounded (>=4-of-5 overlap) rather than open-ended.
//

import XCTest
import VectorCore
@testable import VectorIndex

final class IVFBatchGEMMParityTests: XCTestCase {
    // Deterministic RNG, same pattern as IVFRecallTests/IVFProbeMonotonicTests.
    struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = 2862933555777941757 &* state &+ 3037000493
            return state
        }
        mutating func nextFloat() -> Float {
            let x = next() >> 11
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
            // normalize
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            if norm > 0 { v = v.map { $0 / norm } }
            data.append(v)
        }
        return data
    }

    func makeOptimizedIVF(n: Int, dim: Int, nlist: Int, seed: UInt64, metric: SupportedDistanceMetric = .euclidean) async throws -> IVFIndex {
        let data = generateDataset(count: n, dim: dim, seed: seed)
        let ivf = IVFIndex(dimension: dim, metric: metric, config: .init(nlist: nlist, nprobe: 8))
        for i in 0..<n {
            try await ivf.insert(id: "id\(i)", vector: data[i], metadata: nil)
        }
        try await ivf.optimize()
        return ivf
    }

    func testBatchMatchesSingleQuerySearch() async throws {
        let idx = try await makeOptimizedIVF(n: 800, dim: 24, nlist: 16, seed: 4242)
        let queries = generateDataset(count: 60, dim: 24, seed: 777)
        let batch = try await idx.batchSearch(queries: queries, k: 5, filter: nil)
        var exact = 0
        for (qi, q) in queries.enumerated() {
            let single = try await idx.search(query: q, k: 5, filter: nil)
            let bIDs = batch[qi].map(\.id), sIDs = single.map(\.id)
            if bIDs == sIDs { exact += 1 }
            else {
                XCTAssertGreaterThanOrEqual(Set(bIDs).intersection(sIDs).count, 4,
                    "query \(qi): batch/single may differ only at FP-margin probe ties")
            }
        }
        XCTAssertGreaterThanOrEqual(exact, 57, "≥95% of queries must match exactly")
    }

    /// Fix round 1, Finding 1: the fixture above (d=24) never crosses
    /// `L2Sqr`'s d>=256 branch threshold, where the scalar per-query path
    /// (`ScoreBlock` -> `L2Sqr`) uses its own on-the-fly dot-trick while the
    /// GEMM path always computes `||c||^2 - 2<q,c>` -- two different
    /// algorithms with different rounding, so near-boundary probe-rank
    /// flips at d>=256 were a residual risk no prior test covered.
    /// Production's benchmark config uses dim=384, so this fixture exercises
    /// that exact regime (kept small: n=600, kc=16, q=40, to run in seconds).
    func testBatchMatchesSingleQuerySearchHighDimensional() async throws {
        let idx = try await makeOptimizedIVF(n: 600, dim: 384, nlist: 16, seed: 5151)
        let queries = generateDataset(count: 40, dim: 384, seed: 606)
        let batch = try await idx.batchSearch(queries: queries, k: 5, filter: nil)
        var exact = 0
        for (qi, q) in queries.enumerated() {
            let single = try await idx.search(query: q, k: 5, filter: nil)
            let bIDs = batch[qi].map(\.id), sIDs = single.map(\.id)
            if bIDs == sIDs { exact += 1 }
            else {
                XCTAssertGreaterThanOrEqual(Set(bIDs).intersection(sIDs).count, 4,
                    "query \(qi): batch/single may differ only at FP-margin probe ties (k=5, so >=k-1=4 overlap)")
            }
        }
        XCTAssertGreaterThanOrEqual(exact, 38, "≥95% of queries must match exactly (40*0.95=38)")
    }

    // MARK: - Direct CentroidBatchScore correctness (scalar reference)

    /// Scalar reference matching CentroidBatchScore's documented ordering
    /// contract: euclidean -> ||c||^2 - 2<q,c>, dotProduct -> -<q,c>,
    /// cosine -> 1 - <q,c>*qInv*cInv with the exact same near-zero-norm
    /// guard as the scalar `distance()`/`centroidScores`.
    private func scalarReferenceScores(
        query: [Float], centroids: [[Float]], metric: SupportedDistanceMetric, queryIsNormalized: Bool
    ) -> [Float] {
        let qNormSq: Float = queryIsNormalized ? 1.0 : query.reduce(0) { $0 + $1 * $1 }
        return centroids.map { c in
            switch metric {
            case .euclidean:
                let cNormSq = c.reduce(0) { $0 + $1 * $1 }
                var dot: Float = 0
                for i in 0..<query.count { dot += query[i] * c[i] }
                return cNormSq - 2 * dot
            case .dotProduct:
                var dot: Float = 0
                for i in 0..<query.count { dot += query[i] * c[i] }
                return -dot
            case .cosine:
                let cNormSq = c.reduce(0) { $0 + $1 * $1 }
                var dot: Float = 0
                for i in 0..<query.count { dot += query[i] * c[i] }
                let denom = (qNormSq * cNormSq).squareRoot()
                guard denom > .ulpOfOne else { return 1 }
                let qInv = queryIsNormalized ? 1.0 : 1.0 / (qNormSq.squareRoot() + 1e-12)
                let cInv = 1.0 / (cNormSq.squareRoot() + 1e-12)
                return 1 - dot * qInv * cInv
            default:
                fatalError("unsupported metric in reference")
            }
        }
    }

    private func runGEMM(
        queries: [[Float]], centroids: [[Float]], metric: SupportedDistanceMetric, queriesAreNormalized: Bool
    ) -> [Float]? {
        let q = queries.count, kc = centroids.count, d = centroids.first?.count ?? 0
        let flatQ = queries.flatMap { $0 }
        let flatC = centroids.flatMap { $0 }
        let centroidNormsSq = centroids.map { c in c.reduce(Float(0)) { $0 + $1 * $1 } }
        let centroidInvNorms = centroidNormsSq.map { 1.0 / ($0.squareRoot() + 1e-12) }
        var out = [Float](repeating: 0, count: q * kc)
        let ok = flatQ.withUnsafeBufferPointer { qb in
            flatC.withUnsafeBufferPointer { cb -> Bool in
                CentroidBatchScore.run(
                    queries: qb.baseAddress!, q: q,
                    centroids: cb.baseAddress!, kc: kc, d: d,
                    metric: metric,
                    centroidNormsSq: centroidNormsSq, centroidInvNorms: centroidInvNorms,
                    queriesAreNormalized: queriesAreNormalized,
                    out: &out)
            }
        }
        return ok ? out : nil
    }

    func testCentroidBatchScoreMatchesScalarReferenceEuclidean() throws {
        var rng = LCG(state: 99)
        let d = 12, kc = 7, nq = 5
        let queries = (0..<nq).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        let centroids = (0..<kc).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        guard let out = runGEMM(queries: queries, centroids: centroids, metric: .euclidean, queriesAreNormalized: false) else {
            return XCTFail("expected GEMM path to support euclidean")
        }
        for qi in 0..<nq {
            let ref = scalarReferenceScores(query: queries[qi], centroids: centroids, metric: .euclidean, queryIsNormalized: false)
            for ci in 0..<kc {
                XCTAssertEqual(out[qi * kc + ci], ref[ci], accuracy: 1e-3, "euclidean qi=\(qi) ci=\(ci)")
            }
        }
    }

    func testCentroidBatchScoreMatchesScalarReferenceDotProduct() throws {
        var rng = LCG(state: 123)
        let d = 12, kc = 7, nq = 5
        let queries = (0..<nq).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        let centroids = (0..<kc).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        guard let out = runGEMM(queries: queries, centroids: centroids, metric: .dotProduct, queriesAreNormalized: false) else {
            return XCTFail("expected GEMM path to support dotProduct")
        }
        for qi in 0..<nq {
            let ref = scalarReferenceScores(query: queries[qi], centroids: centroids, metric: .dotProduct, queryIsNormalized: false)
            for ci in 0..<kc {
                XCTAssertEqual(out[qi * kc + ci], ref[ci], accuracy: 1e-3, "dotProduct qi=\(qi) ci=\(ci)")
            }
        }
    }

    func testCentroidBatchScoreMatchesScalarReferenceCosineWithDegenerateCentroid() throws {
        var rng = LCG(state: 456)
        let d = 12, kc = 6, nq = 5
        let queries = (0..<nq).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        var centroids = (0..<kc).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        // Force one centroid to a degenerate near-zero norm so the guard is exercised.
        centroids[2] = [Float](repeating: 0, count: d)
        guard let out = runGEMM(queries: queries, centroids: centroids, metric: .cosine, queriesAreNormalized: false) else {
            return XCTFail("expected GEMM path to support cosine")
        }
        for qi in 0..<nq {
            let ref = scalarReferenceScores(query: queries[qi], centroids: centroids, metric: .cosine, queryIsNormalized: false)
            for ci in 0..<kc {
                XCTAssertEqual(out[qi * kc + ci], ref[ci], accuracy: 1e-3, "cosine qi=\(qi) ci=\(ci)")
            }
            // Degenerate centroid must be forced to exactly 1 (max distance), matching centroidScores.
            XCTAssertEqual(out[qi * kc + 2], 1.0, accuracy: 1e-6, "degenerate centroid must be forced to distance 1")
        }
    }

    /// Fix round 1, Finding 2: `queriesAreNormalized: true` in the cosine
    /// branch (`CentroidBatchScore.swift`, the `qInv`/`qNormSq` fast path)
    /// had zero direct coverage. It IS a reachable production path --
    /// `IVFIndex.batchSearch<V: IndexableVector>` (TypedOverloads.swift)
    /// passes `queryIsNormalized: true` for cosine whenever the input
    /// vector reports `isNormalized`. This test uses genuinely unit-norm
    /// queries (via `generateDataset`, which L2-normalizes), so the
    /// `queriesAreNormalized: true` fast path (qNormSq forced to 1.0,
    /// skipping recomputation) must agree both with a scalar reference
    /// computed the same way, AND with the `queriesAreNormalized: false`
    /// path recomputing the norm from the same (already-normalized) data --
    /// which is the whole correctness contract the hint relies on. A
    /// degenerate-norm centroid row pins the near-zero-norm guard's
    /// collapse specifically under the qNormSq=1.0-forced code path (not
    /// just the recomputed-qNormSq path already covered above).
    func testCentroidBatchScoreCosineQueriesAreNormalizedTrue() throws {
        var rng = LCG(state: 789)
        let d = 12, kc = 6, nq = 5
        let queries = generateDataset(count: nq, dim: d, seed: 789) // unit-norm by construction
        var centroids = (0..<kc).map { _ in (0..<d).map { _ in rng.nextInRange(-1...1) } }
        // Force one centroid to a degenerate near-zero norm so the guard's
        // qNormSq=1.0-forced collapse is exercised.
        centroids[3] = [Float](repeating: 0, count: d)

        guard let outNormTrue = runGEMM(queries: queries, centroids: centroids, metric: .cosine, queriesAreNormalized: true) else {
            return XCTFail("expected GEMM path to support cosine")
        }
        guard let outNormFalse = runGEMM(queries: queries, centroids: centroids, metric: .cosine, queriesAreNormalized: false) else {
            return XCTFail("expected GEMM path to support cosine")
        }

        for qi in 0..<nq {
            let refTrue = scalarReferenceScores(query: queries[qi], centroids: centroids, metric: .cosine, queryIsNormalized: true)
            for ci in 0..<kc {
                XCTAssertEqual(outNormTrue[qi * kc + ci], refTrue[ci], accuracy: 1e-3,
                    "cosine queriesAreNormalized=true qi=\(qi) ci=\(ci) vs scalar reference")
                XCTAssertEqual(outNormTrue[qi * kc + ci], outNormFalse[qi * kc + ci], accuracy: 1e-3,
                    "cosine qi=\(qi) ci=\(ci): normalized-hint fast path must match recomputed-norm path on already-normalized data")
            }
            // Degenerate centroid must still be forced to exactly 1 (max
            // distance) even under the queriesAreNormalized=true guard,
            // where qNormSq is forced to 1.0 rather than recomputed.
            XCTAssertEqual(outNormTrue[qi * kc + 3], 1.0, accuracy: 1e-6,
                "degenerate centroid must be forced to distance 1 even with queriesAreNormalized=true")
        }
    }

    func testCentroidBatchScoreReturnsFalseForUnsupportedMetric() throws {
        let d = 4, kc = 2, nq = 1
        let queries: [Float] = [1, 0, 0, 0]
        let centroids: [Float] = [1, 0, 0, 0, 0, 1, 0, 0]
        let centroidNormsSq: [Float] = [1, 1]
        let centroidInvNorms: [Float] = [1, 1]
        var out = [Float](repeating: 0, count: nq * kc)
        let ok = queries.withUnsafeBufferPointer { qb in
            centroids.withUnsafeBufferPointer { cb -> Bool in
                CentroidBatchScore.run(
                    queries: qb.baseAddress!, q: nq,
                    centroids: cb.baseAddress!, kc: kc, d: d,
                    metric: .manhattan,
                    centroidNormsSq: centroidNormsSq, centroidInvNorms: centroidInvNorms,
                    queriesAreNormalized: false,
                    out: &out)
            }
        }
        XCTAssertFalse(ok, "manhattan has no GEMM form; run() must return false")
    }
}
