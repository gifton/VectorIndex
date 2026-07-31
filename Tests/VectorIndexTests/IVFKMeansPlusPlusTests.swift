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
}
