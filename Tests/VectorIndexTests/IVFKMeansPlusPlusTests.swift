import XCTest
@testable import VectorIndex

final class IVFKMeansPlusPlusTests: XCTestCase {
    func testOptimizeAssignsAll() async throws {
        // Three clusters around axes
        let ivf = IVFIndex(dimension: 3, metric: .euclidean, config: .init(nlist: 3, nprobe: 1))
        try await ivf.batchInsert([
            ("a1", [1, 0, 0], nil), ("a2", [0.9, 0, 0], nil),
            ("b1", [0, 1, 0], nil), ("b2", [0, 0.95, 0], nil),
            ("c1", [0, 0, 1], nil), ("c2", [0, 0, 0.9], nil)
        ])
        try await ivf.optimize()
        let stats = await ivf.statistics()
        XCTAssertEqual(stats.indexType, "IVF")
        XCTAssertEqual(stats.vectorCount, 6)
        XCTAssertEqual(Int(stats.details["nlist"] ?? "0"), 3)
        XCTAssertEqual(Int(stats.details["assigned"] ?? "0"), 6)
    }

    /// Guards Task 16 (B18): before this task, optimizeKMeans(maxIterations:) ran
    /// its own standalone body that never populated idToListIndex. Same
    /// three-cluster fixture as testOptimizeAssignsAll, driven through
    /// optimizeKMeans() instead of optimize() -- "assigned" (backed by
    /// idToListIndex.count, see IVFIndex.statistics()) must now read 6, proving
    /// the delegation to optimize(maxIterations:) actually populates the map.
    func testOptimizeKMeansPopulatesIdToListIndex() async throws {
        let ivf = IVFIndex(dimension: 3, metric: .euclidean, config: .init(nlist: 3, nprobe: 1))
        try await ivf.batchInsert([
            ("a1", [1, 0, 0], nil), ("a2", [0.9, 0, 0], nil),
            ("b1", [0, 1, 0], nil), ("b2", [0, 0.95, 0], nil),
            ("c1", [0, 0, 1], nil), ("c2", [0, 0, 0.9], nil)
        ])
        try await ivf.optimizeKMeans()
        let stats = await ivf.statistics()
        XCTAssertEqual(stats.vectorCount, 6)
        XCTAssertEqual(Int(stats.details["assigned"] ?? "0"), 6,
                       "optimizeKMeans must populate idToListIndex for every assigned vector")
    }

    func testSearchAfterOptimizeFindsCluster() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([
            ("x1", [0.0, 0.0], nil), ("x2", [0.05, -0.02], nil),
            ("y1", [1.0, 0.0], nil), ("y2", [0.95, 0.04], nil)
        ])
        try await ivf.optimize()
        // Query near [1,0]
        let res = try await ivf.search(query: [0.98, 0.0], k: 2, filter: nil)
        let got = Set(res.map { $0.id })
        // Expect picks from y cluster
        XCTAssertFalse(got.isDisjoint(with: ["y1", "y2"]))
    }

    // MARK: - P3c (Task 8)

    /// Deterministic pseudo-random fixture: `n` points in `dim` dimensions,
    /// generated from a simple LCG seeded by `seed` (not cryptographic --
    /// just needs to be reproducible across calls/processes, unlike Swift's
    /// SystemRandomNumberGenerator). IDs are zero-padded so lexicographic
    /// (String) sort order and numeric insertion order coincide, keeping
    /// this fixture's own bookkeeping simple; the determinism guarantee
    /// under test (see testOptimizeIsDeterministicAcrossInsertionOrder)
    /// does NOT depend on that coincidence -- it inserts in two genuinely
    /// different orders into two separate instances.
    private func makeOptimizedIVF(n: Int, dim: Int, nlist: Int, seed: UInt64) async throws -> IVFIndex {
        let ivf = IVFIndex(dimension: dim, metric: .euclidean, config: .init(nlist: nlist, nprobe: 1))
        var rng = seed
        func nextFloat() -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: rng >> 32)
            return Float(bits) / Float(UInt32.max) * 2 - 1
        }
        var items: [(id: String, vector: [Float], metadata: [String: String]?)] = []
        items.reserveCapacity(n)
        for i in 0..<n {
            let vec = (0..<dim).map { _ in nextFloat() }
            items.append((id: String(format: "v%06d", i), vector: vec, metadata: nil))
        }
        try await ivf.batchInsert(items)
        try await ivf.optimize()
        return ivf
    }

    /// Pins the Step-1 semantics decision (kmeans_minibatch_f32's
    /// `assignOut` is a final-centroid pass, KMeansMiniBatchKernel.swift:
    /// 687-706) as an executable fact: every id `optimize()` places in a
    /// list must sit in the list of its nearest final centroid.
    func testOptimizeAssignmentsMatchNearestCentroid() async throws {
        let idx = try await makeOptimizedIVF(n: 300, dim: 8, nlist: 8, seed: 99)
        let check = await idx._testAssignmentConsistency()
        XCTAssertEqual(check.mismatches, 0,
            "post-optimize lists must reflect nearest-final-centroid assignment; \(check.detail)")
    }

    /// Determinism (controller addition): Swift Dictionary iteration order
    /// is randomized per-process, not derived from content, so the old
    /// `for (id, ...) in store` materialization in `optimize()` fed
    /// k-means a different point ordering on every process invocation even
    /// for byte-identical store content -- and since k-means' mini-batch
    /// updates and empty-cluster repair are order-sensitive, that produced
    /// different centroids from run to run (the recall swings 0.72-1.0
    /// measured/adjudicated in Tasks 6-7). `optimize()` now sorts
    /// `store.keys` before flattening, which is content-derived and
    /// process-independent.
    ///
    /// To prove the *sort* is what's doing the work (not just "same
    /// process, same Dictionary" happenstance), this builds two IVFIndex
    /// instances from the SAME content inserted in DIFFERENT orders --
    /// forward numeric vs. reverse -- which, absent the sort, would very
    /// likely walk their two backing Dictionaries in different bucket
    /// orders. The resulting centroids must be identical.
    func testOptimizeIsDeterministicAcrossInsertionOrder() async throws {
        let dim = 6, nlist = 5, n = 120
        var rng: UInt64 = 1234567
        func nextFloat() -> Float {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: rng >> 32)
            return Float(bits) / Float(UInt32.max) * 2 - 1
        }
        var items: [(id: String, vector: [Float])] = []
        items.reserveCapacity(n)
        for i in 0..<n {
            let vec = (0..<dim).map { _ in nextFloat() }
            items.append((id: String(format: "v%06d", i), vector: vec))
        }

        let forward = IVFIndex(dimension: dim, metric: .euclidean, config: .init(nlist: nlist, nprobe: 1))
        try await forward.batchInsert(items.map { ($0.id, $0.vector, nil) })
        try await forward.optimize()

        let reversed = IVFIndex(dimension: dim, metric: .euclidean, config: .init(nlist: nlist, nprobe: 1))
        try await reversed.batchInsert(items.reversed().map { ($0.id, $0.vector, nil) })
        try await reversed.optimize()

        let forwardCentroids = await forward._testCentroids()
        let reversedCentroids = await reversed._testCentroids()

        XCTAssertEqual(forwardCentroids.count, reversedCentroids.count)
        XCTAssertEqual(forwardCentroids, reversedCentroids,
            "optimize() must produce identical centroids regardless of store insertion order (sorted-key materialization)")
    }

    // MARK: - Fix Round 1 (CRITICAL): optimize() list-build must be metric-aware

    /// Deliberately constructed so an L2-nearest-centroid argmin and a
    /// cosine/dotProduct-nearest-centroid argmin genuinely disagree for at
    /// least one point:
    ///  - Cluster "a*": direction (1,0), LARGE magnitude (~1000).
    ///  - Cluster "b*": direction (0,1), SMALL magnitude (~1).
    ///  - "outlier": direction (1,0) (aligned with cluster A), but SMALL
    ///    magnitude (~0.5, close to cluster B's scale).
    ///
    /// After (L2-trained, unchanged by this fix) k-means converges to two
    /// centroids approximating the two clusters' means (C_A large-magnitude
    /// near (1000,0), C_B small-magnitude near (0,1)):
    ///  - Raw L2 distance from "outlier" to C_A is huge (~999.5, dominated
    ///    by the magnitude gap); to C_B is small (~1.1). L2 argmin = B.
    ///  - Cosine similarity from "outlier" to C_A is ~1 (angle ~0); to C_B
    ///    is small (near-orthogonal). Cosine argmin = A.
    ///  - Raw dot product from "outlier" to C_A is ~500 (huge, because
    ///    C_A's magnitude is huge even though the alignment factor is
    ///    small); to C_B is ~0.03. DotProduct argmin (max dot == min
    ///    -dot) = A.
    ///
    /// So for both cosine and dotProduct metrics, "outlier" must end up in
    /// A's list; kmeans_minibatch_f32's own (unconditionally L2)
    /// `assignOut` would put it in B's list instead -- exactly the CRITICAL
    /// metric-blindness bug this fixture is built to catch.
    private static func makeMetricDivergenceFixture() -> [(id: String, vector: [Float], metadata: [String: String]?)] {
        var items: [(id: String, vector: [Float], metadata: [String: String]?)] = []
        let aBase: [(Float, Float)] = [(1000, 0), (995, 5), (1005, -5), (998, 3), (1002, -3), (1000, 1)]
        for (i, xy) in aBase.enumerated() {
            items.append((id: "a\(i)", vector: [xy.0, xy.1], metadata: nil))
        }
        let bBase: [(Float, Float)] = [(0, 1), (0.05, 0.95), (-0.05, 1.05), (0.02, 0.98), (-0.02, 1.02), (0, 1.0)]
        for (i, xy) in bBase.enumerated() {
            items.append((id: "b\(i)", vector: [xy.0, xy.1], metadata: nil))
        }
        items.append((id: "outlier", vector: [0.5, 0.001], metadata: nil))
        return items
    }

    /// Pre-fix (commit a987575), this test fails: optimize()'s list-build
    /// consumed kmeans_minibatch_f32's own `assignOut`, which is
    /// unconditionally L2-squared regardless of `self.metric`, so "outlier"
    /// (see makeMetricDivergenceFixture) lands in the small-magnitude
    /// cluster's list even though it is angularly aligned with the
    /// large-magnitude cluster. `_testAssignmentConsistency` is
    /// metric-aware (it scores via `centroidDistances`, which dispatches on
    /// `self.metric`), so it catches the disagreement. Verified RED against
    /// a987575 (git-stashed IVFIndex.swift) before applying the fix; see
    /// task-8-report.md "Fix round 1" for the captured failure output.
    func testCosineOptimizeAssignmentsRespectCosineMetric() async throws {
        let idx = IVFIndex(dimension: 2, metric: .cosine, config: .init(nlist: 2, nprobe: 1))
        try await idx.batchInsert(Self.makeMetricDivergenceFixture())
        try await idx.optimize()
        let check = await idx._testAssignmentConsistency()
        XCTAssertEqual(check.mismatches, 0,
            "cosine-metric optimize() list-build must be cosine-aware, not the kernel's unconditional L2 assignOut; \(check.detail)")
    }

    /// Same fixture and rationale as testCosineOptimizeAssignmentsRespectCosineMetric,
    /// for metric == .dotProduct (cheap to add -- same fixture, same divergence:
    /// "outlier"'s raw dot product with the large-magnitude centroid dwarfs its
    /// dot product with the small-magnitude one, so dotProduct argmin also
    /// disagrees with L2 argmin here).
    func testDotProductOptimizeAssignmentsRespectDotProductMetric() async throws {
        let idx = IVFIndex(dimension: 2, metric: .dotProduct, config: .init(nlist: 2, nprobe: 1))
        try await idx.batchInsert(Self.makeMetricDivergenceFixture())
        try await idx.optimize()
        let check = await idx._testAssignmentConsistency()
        XCTAssertEqual(check.mismatches, 0,
            "dotProduct-metric optimize() list-build must be dotProduct-aware, not the kernel's unconditional L2 assignOut; \(check.detail)")
    }
}
