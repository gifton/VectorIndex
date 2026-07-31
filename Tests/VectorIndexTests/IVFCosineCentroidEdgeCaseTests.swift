import XCTest
@testable import VectorIndex

/// Fix round 1 (Task 6, P3a review) coverage.
///
/// Finding 1: the old scalar `distance()` (DistanceUtils.swift, cosine case)
/// short-circuits to distance 1 whenever `sqrt(amag2 * bmag2) <= .ulpOfOne`
/// -- guarding a degenerate (near-zero-norm) vector on either side, without
/// dividing. `IVFIndex.centroidDistances`/`centroidScores` (the P3a batched
/// coarse-scoring path) previously had no equivalent guard: a collapsed
/// centroid (plausible after k-means cluster collapse) would silently
/// produce a "real" similarity via the epsilon-inflated per-factor inverse
/// norms instead of the old "never win the argmin / probed last" behavior.
/// No pre-existing test exercised `centroidDistances`'s cosine branch at
/// all, so this file is the first, and is built to directly prove both the
/// exact-value semantics (`_testCentroidDistances`) and the resulting
/// probe-selection behavior via the public `search()` API.
///
/// Finding 2: `nearestCentroidIndex`'s all-NaN edge case (old semantics:
/// return nil when every centroid score is NaN) is also covered here via
/// `_testNearestCentroidIndex`, since the fix touches the same file/area
/// and the hooks needed for Finding 1 make it essentially free to add.
final class IVFCosineCentroidEdgeCaseTests: XCTestCase {

    // MARK: - Finding 1: cosine near-zero-norm short-circuit

    func testDegenerateCentroidScoresExactlyMaxDistance() async throws {
        let dim = 8
        let idx = IVFIndex(dimension: dim, metric: .cosine, config: .init(nlist: 3, nprobe: 2))

        // c0: same direction as the query (sim = 1, distance = 0).
        // c1: positively-similar but not identical (distance in (0, 1)).
        // c2: degenerate -- a real but tiny norm (‖c2‖ = 1e-8), the kind of
        // near-zero-norm vector a collapsed k-means cluster could plausibly
        // produce. bmag2 = 1e-16, so even against a query with norm up to
        // several units, sqrt(qNormSq * bmag2) is many orders of magnitude
        // below Float.ulpOfOne (~1.19e-7) -- this is not the trivial
        // all-zero case, it exercises the actual inequality.
        let c0: [Float] = [1, 1, 0, 0, 0, 0, 0, 0]
        let c1: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]
        let c2: [Float] = [1e-8, 0, 0, 0, 0, 0, 0, 0]
        await idx._testSetCentroids([c0, c1, c2])

        let q: [Float] = [1, 1, 0, 0, 0, 0, 0, 0]
        let dists = await idx._testCentroidDistances(for: q)

        XCTAssertEqual(dists.count, 3)
        // Exact match to the old scalar path's literal `return 1` -- not an
        // approximation, the guard forces this bit-for-bit.
        XCTAssertEqual(dists[2], 1.0)
        // Cross-check directly against the old per-pair scalar `distance()`
        // function (DistanceUtils.swift), still present and unmodified.
        XCTAssertEqual(dists[2], distance(q, c2, metric: .cosine))

        // The two well-formed, positively-similar centroids must score
        // strictly better (smaller) than the forced max, so the degenerate
        // centroid sorts last under `search()`'s ascending-distance probe
        // selection.
        XCTAssertLessThan(dists[0], dists[2])
        XCTAssertLessThan(dists[1], dists[2])
        XCTAssertEqual(dists[0], 0.0, accuracy: 1e-6) // identical direction to query
    }

    func testDegenerateCentroidIsExcludedWhenNprobeLessThanKc() async throws {
        let dim = 8
        // nlist=3, nprobe=2: only the 2 best-scoring centroids get probed.
        let idx = IVFIndex(dimension: dim, metric: .cosine, config: .init(nlist: 3, nprobe: 2))
        let c0: [Float] = [1, 1, 0, 0, 0, 0, 0, 0]
        let c1: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]
        let c2: [Float] = [1e-8, 0, 0, 0, 0, 0, 0, 0] // degenerate, list index 2
        await idx._testSetCentroids([c0, c1, c2])

        // Populate list 0 (a normal, findable candidate) and list 2 (a
        // candidate that only a probe of the degenerate centroid's list
        // could ever surface). `_testInjectListEntry` bypasses
        // `nearestCentroidIndex`-driven assignment on purpose: the
        // degenerate centroid can never win that argmin post-fix (its
        // forced distance is 1, worse than any positively-similar
        // centroid), so this is the only way to get something into list 2
        // to test probe *selection* behavior against.
        await idx._testInjectListEntry(listIndex: 0, id: "inList0", vector: [1, 1, 0, 0, 0, 0, 0, 0])
        await idx._testInjectListEntry(listIndex: 2, id: "onlyInDegenerateList", vector: [1, 1, 0, 0, 0, 0, 0, 0])

        let q: [Float] = [1, 1, 0, 0, 0, 0, 0, 0]
        let results = try await idx.search(query: q, k: 5, filter: nil)
        let ids = Set(results.map { $0.id })

        XCTAssertTrue(ids.contains("inList0"))
        XCTAssertFalse(
            ids.contains("onlyInDegenerateList"),
            "degenerate (near-zero-norm) centroid must score the old max distance and therefore not be among the nprobe=2 of 3 lists probed"
        )
    }

    func testDegenerateCentroidIsReachableWhenAllListsProbed() async throws {
        let dim = 8
        // nlist=3, nprobe=3: every list is probed, including the
        // degenerate centroid's -- confirms the previous test's negative
        // result is due to probe *selection* (nprobe < kc), not the entry
        // being unreachable/lost.
        let idx = IVFIndex(dimension: dim, metric: .cosine, config: .init(nlist: 3, nprobe: 3))
        let c0: [Float] = [1, 1, 0, 0, 0, 0, 0, 0]
        let c1: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]
        let c2: [Float] = [1e-8, 0, 0, 0, 0, 0, 0, 0]
        await idx._testSetCentroids([c0, c1, c2])
        await idx._testInjectListEntry(listIndex: 2, id: "onlyInDegenerateList", vector: [1, 1, 0, 0, 0, 0, 0, 0])

        let q: [Float] = [1, 1, 0, 0, 0, 0, 0, 0]
        let results = try await idx.search(query: q, k: 5, filter: nil)
        let ids = Set(results.map { $0.id })

        XCTAssertTrue(ids.contains("onlyInDegenerateList"))
    }

    // MARK: - Finding 2: nearestCentroidIndex all-NaN nil semantics

    func testNearestCentroidIndexAllNaNReturnsNil() async throws {
        // Every centroid carries a NaN component, so every euclidean L2^2
        // score is NaN regardless of the query. Old semantics: best = -1,
        // bestD = .infinity, strict `<` (NaN comparisons are always false)
        // -- every iteration is skipped, best stays -1, returns nil.
        let idx = IVFIndex(dimension: 4, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        await idx._testSetCentroids([
            [Float.nan, 0, 0, 0],
            [Float.nan, 1, 1, 1]
        ])
        let result = await idx._testNearestCentroidIndex(for: [0, 0, 0, 0])
        XCTAssertNil(result, "all-NaN centroid scores must return nil, matching the old best=-1/bestD=.infinity loop shape")
    }
}
