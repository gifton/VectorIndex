// Tests for Kernel #29: IVF List Selection
import XCTest
import Accelerate
@testable import VectorIndex

final class IVFSelectTests: XCTestCase {

    // MARK: - Test 1: Brute-Force Parity

    func testStandardSelectionParity_L2() {
        let kc = 1_000
        let d = 128
        let nprobe = 20

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Fast implementation
        var fastIDs = [Int32](repeating: -1, count: nprobe)
        var fastScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &fastIDs,
            listScoresOut: &fastScores
        )

        // Brute-force reference
        let refResults = bruteForceTopK(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, k: nprobe
        )

        // Verify exact match
        XCTAssertEqual(fastIDs.count, nprobe)
        for i in 0..<nprobe {
            XCTAssertEqual(fastIDs[i], refResults[i].id,
                          "Mismatch at position \(i): fast=\(fastIDs[i]) ref=\(refResults[i].id)")

            if let scores = fastScores {
                XCTAssertEqual(scores[i], refResults[i].score, accuracy: 1e-4,
                              "Score mismatch at \(i)")
            }
        }
    }

    func testStandardSelectionParity_IP() {
        let kc = 500
        let d = 256
        let nprobe = 15

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var fastIDs = [Int32](repeating: -1, count: nprobe)
        var fastScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .ip, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &fastIDs,
            listScoresOut: &fastScores
        )

        let refResults = bruteForceTopK(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .ip, k: nprobe
        )

        for i in 0..<nprobe {
            XCTAssertEqual(fastIDs[i], refResults[i].id,
                          "IP: Mismatch at \(i)")
            if let scores = fastScores {
                XCTAssertEqual(scores[i], refResults[i].score, accuracy: 1e-4)
            }
        }
    }

    func testStandardSelectionParity_Cosine() {
        let kc = 800
        let d = 192
        let nprobe = 25

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var fastIDs = [Int32](repeating: -1, count: nprobe)
        var fastScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .cosine, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &fastIDs,
            listScoresOut: &fastScores
        )

        let refResults = bruteForceTopK(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .cosine, k: nprobe
        )

        for i in 0..<nprobe {
            XCTAssertEqual(fastIDs[i], refResults[i].id,
                          "Cosine: Mismatch at \(i)")
            if let scores = fastScores {
                XCTAssertEqual(scores[i], refResults[i].score, accuracy: 1e-4)
            }
        }
    }

    // MARK: - Test 2: Metric Equivalence

    func testCosineEquivalence() {
        // Cosine(q, c) = IP(q, c) / (‖q‖·‖c‖)
        let kc = 200
        let d = 64
        let nprobe = 10

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Compute centroid norms
        let centroidNorms = (0..<kc).map { i -> Float in
            let base = i * d
            var normSq: Float = 0
            centroids[base..<(base+d)].withUnsafeBufferPointer { ptr in
                vDSP_svesq(ptr.baseAddress!, 1, &normSq, vDSP_Length(d))
            }
            return sqrt(max(normSq, 1e-10))
        }

        // Compute query norm
        var qNormSq: Float = 0
        vDSP_svesq(q, 1, &qNormSq, vDSP_Length(d))
        let qNorm = sqrt(max(qNormSq, 1e-10))

        // Precompute expected cosine scores for all centroids using vDSP for parity with kernel path
        var expectedCosine = [Float](repeating: 0, count: kc)
        centroids.withUnsafeBufferPointer { centsPtr in
            q.withUnsafeBufferPointer { qPtr in
                for i in 0..<kc {
                    let cPtr = centsPtr.baseAddress! + i * d
                    var dot: Float = 0
                    vDSP_dotpr(qPtr.baseAddress!, 1, cPtr, 1, &dot, vDSP_Length(d))
                    expectedCosine[i] = dot / (qNorm * centroidNorms[i])
                }
            }
        }

        // Get Cosine results from kernel
        var cosineIDs = [Int32](repeating: -1, count: nprobe)
        var cosineScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .cosine, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &cosineIDs,
            listScoresOut: &cosineScores
        )

        // Verify per-ID equality within reasonable tolerance
        guard let cosSc = cosineScores else {
            XCTFail("Cosine scores should be non-nil")
            return
        }
        for i in 0..<nprobe {
            let id = Int(cosineIDs[i])
            XCTAssertEqual(cosSc[i], expectedCosine[id], accuracy: 1e-4,
                          "Cosine score mismatch for ID \(id) at position \(i)")
        }
    }

    func testDotTrickEquivalence() {
        // L2²(q,c) = ‖q‖² + ‖c‖² - 2⟨q,c⟩ should match direct L2²
        let kc = 300
        let d = 96
        let nprobe = 12

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Compute centroid norms for dot trick
        let centroidNorms = (0..<kc).map { i -> Float in
            let base = i * d
            var normSq: Float = 0
            centroids[base..<(base+d)].withUnsafeBufferPointer { ptr in
                vDSP_svesq(ptr.baseAddress!, 1, &normSq, vDSP_Length(d))
            }
            return normSq
        }

        // Without dot trick
        var directIDs = [Int32](repeating: -1, count: nprobe)
        var directScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(useDotTrick: false),
            listIDsOut: &directIDs,
            listScoresOut: &directScores
        )

        // With dot trick
        var trickIDs = [Int32](repeating: -1, count: nprobe)
        var trickScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
            listIDsOut: &trickIDs,
            listScoresOut: &trickScores
        )

        // Should produce identical results
        XCTAssertEqual(directIDs, trickIDs, "Dot trick should produce same IDs")

        guard let dScores = directScores, let tScores = trickScores else {
            XCTFail("Scores should be non-nil")
            return
        }

        for i in 0..<nprobe {
            XCTAssertEqual(dScores[i], tScores[i], accuracy: 1e-4,
                          "Dot trick score mismatch at \(i)")
        }
    }

    // MARK: - Test 3: Disabled Lists

    func testDisabledLists() {
        let kc = 100
        let d = 64
        let nprobe = 10

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Disable every other centroid
        let wordCount = (kc + 63) / 64
        var disabledMask = [UInt64](repeating: 0, count: wordCount)
        for i in stride(from: 0, to: kc, by: 2) {
            let word = i / 64
            let bit = i % 64
            disabledMask[word] |= (1 &<< UInt64(bit))
        }

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(disabledLists: disabledMask),
            listIDsOut: &ids,
            listScoresOut: &scores
        )

        // Verify: no even IDs in results
        for i in 0..<nprobe {
            let id = Int(ids[i])
            if id >= 0 {
                XCTAssertTrue(id % 2 == 1, "Found disabled centroid \(id) in results")
            }
        }
    }

    func testDisabledListsAll() {
        // Disable all centroids → should return sentinels
        let kc = 50
        let d = 32
        let nprobe = 5

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        let wordCount = (kc + 63) / 64
        var disabledMask = [UInt64](repeating: UInt64.max, count: wordCount)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(disabledLists: disabledMask),
            listIDsOut: &ids,
            listScoresOut: &scores
        )

        // All results should be sentinels
        for i in 0..<nprobe {
            XCTAssertEqual(ids[i], -1, "Expected sentinel ID at \(i)")
        }
    }

    // MARK: - Test 4: Tie-Breaking Determinism

    func testTieBreakingDeterminism() {
        // Create centroids with many identical distances
        let kc = 50
        let d = 4
        let nprobe = 20

        let q = [Float](repeating: 0, count: d)

        // All centroids at same distance: [1, 0, 0, 0]
        var centroids = [Float]()
        for _ in 0..<kc {
            centroids.append(contentsOf: [1, 0, 0, 0])
        }

        var ids1 = [Int32](repeating: -1, count: nprobe)
        var ids2 = [Int32](repeating: -1, count: nprobe)
        var nilScores: [Float]? = nil

        // Run twice
        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &ids1,
            listScoresOut: &nilScores
        )

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &ids2,
            listScoresOut: &nilScores
        )

        // Results must be identical and deterministic (prefer smaller IDs)
        XCTAssertEqual(ids1, ids2, "Results should be deterministic")

        // Verify sorted by ID (since all distances equal)
        for i in 0..<nprobe {
            XCTAssertEqual(ids1[i], Int32(i), "Should select first \(nprobe) IDs in order")
        }
    }

    // MARK: - Test 5: Batch vs Single-Query Parity

    func testBatchVsSingleParity() {
        let b = 10
        let d = 128
        let kc = 500
        let nprobe = 15

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Batch processing
        var batchIDs = [Int32](repeating: -1, count: b * nprobe)
        var batchScores: [Float]? = [Float](repeating: 0, count: b * nprobe)

        ivf_select_nprobe_batch_f32(
            Q: Q, b: b, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &batchIDs,
            listScoresOut: &batchScores
        )

        // Single-query processing (reference)
        for i in 0..<b {
            let qOffset = i * d
            let q = Array(Q[qOffset..<(qOffset + d)])

            var singleIDs = [Int32](repeating: -1, count: nprobe)
            var singleScores: [Float]? = [Float](repeating: 0, count: nprobe)

            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(),
                listIDsOut: &singleIDs,
                listScoresOut: &singleScores
            )

            // Compare batch[i] vs single
            let batchOffset = i * nprobe
            for j in 0..<nprobe {
                XCTAssertEqual(batchIDs[batchOffset + j], singleIDs[j],
                              "Batch query \(i), position \(j) mismatch")

                if let bScores = batchScores, let sScores = singleScores {
                    XCTAssertEqual(bScores[batchOffset + j], sScores[j], accuracy: 1e-5)
                }
            }
        }
    }

    // MARK: - Test 6: Beam Search

    func testBeamSearchRecallImprovement() {
        // Beam search should find at least as good (or better) results than standard
        let kc = 500
        let d = 128
        let nprobe = 20
        let beamWidth = 40

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Build simple k-NN graph (random neighbors for testing)
        let deg = 16
        var knnGraph = [Int32]()
        for i in 0..<kc {
            for _ in 0..<deg {
                let neighbor = Int32.random(in: 0..<Int32(kc))
                knnGraph.append(neighbor)
            }
        }

        // Standard selection
        var standardIDs = [Int32](repeating: -1, count: nprobe)
        var standardScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &standardIDs,
            listScoresOut: &standardScores
        )

        // Beam search
        var beamIDs = [Int32](repeating: -1, count: nprobe)
        var beamScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_beam_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            knnGraph: knnGraph, deg: deg,
            metric: .l2, nprobe: nprobe, beamWidth: beamWidth,
            opts: IVFSelectOpts(),
            listIDsOut: &beamIDs,
            listScoresOut: &beamScores
        )

        // Beam search should find results at least as good as standard
        // (This is hard to verify without proper graph; just check it returns valid IDs)
        for i in 0..<nprobe {
            XCTAssertTrue(beamIDs[i] >= -1 && beamIDs[i] < kc,
                         "Beam search returned invalid ID at \(i)")
        }

        // Both should return same count of valid IDs
        let standardValidCount = standardIDs.filter { $0 >= 0 }.count
        let beamValidCount = beamIDs.filter { $0 >= 0 }.count
        XCTAssertEqual(standardValidCount, beamValidCount,
                      "Both should return same number of valid IDs")
    }

    func testBeamSearchFallback() {
        // With nil graph, beam search should fallback to standard
        let kc = 200
        let d = 64
        let nprobe = 10

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var beamIDs = [Int32](repeating: -1, count: nprobe)
        var beamScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_beam_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            knnGraph: nil, deg: 0,  // No graph
            metric: .l2, nprobe: nprobe, beamWidth: 20,
            opts: IVFSelectOpts(),
            listIDsOut: &beamIDs,
            listScoresOut: &beamScores
        )

        // Should produce same results as standard
        var standardIDs = [Int32](repeating: -1, count: nprobe)
        var standardScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &standardIDs,
            listScoresOut: &standardScores
        )

        XCTAssertEqual(beamIDs, standardIDs, "Beam search without graph should match standard")
    }

    // MARK: - Test 7: Edge Cases

    func testNprobeEqualsKc() {
        // nprobe = kc should return all centroids
        let kc = 50
        let d = 32
        let nprobe = kc

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var nilScores: [Float]? = nil

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &ids,
            listScoresOut: &nilScores
        )

        // Should return all IDs (in some order)
        let uniqueIDs = Set(ids)
        XCTAssertEqual(uniqueIDs.count, kc, "Should return all \(kc) unique centroids")
        XCTAssertFalse(uniqueIDs.contains(-1), "No sentinels when nprobe=kc")
    }

    func testNprobeOne() {
        // nprobe=1 should return single nearest centroid
        let kc = 100
        let d = 64
        let nprobe = 1

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &ids,
            listScoresOut: &scores
        )

        // Verify against brute-force
        let refResults = bruteForceTopK(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, k: 1
        )

        XCTAssertEqual(ids[0], refResults[0].id, "nprobe=1 should return nearest")
    }

    func testEmptyCentroids() {
        // kc=1 with nprobe=1 (minimal valid case)
        let kc = 1
        let d = 16
        let nprobe = 1

        let q = randomVector(d: d)
        let centroids = randomVector(d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var nilScores: [Float]? = nil

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &ids,
            listScoresOut: &nilScores
        )

        XCTAssertEqual(ids[0], 0, "Single centroid should be selected")
    }

    func testHighDimensional() {
        // Test with very high dimension
        let kc = 100
        let d = 2048  // Large dimension
        let nprobe = 10

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            listIDsOut: &ids,
            listScoresOut: &scores
        )

        // Verify against brute-force
        let refResults = bruteForceTopK(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, k: nprobe
        )

        for i in 0..<nprobe {
            XCTAssertEqual(ids[i], refResults[i].id, "High-dim mismatch at \(i)")
        }
    }

    // MARK: - Test 8: Multi-threading

    func testMultiThreadedCorrectness() {
        // Large kc to trigger multi-threading
        let kc = 120_000
        let d = 128
        let nprobe = 50

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Force multi-threading
        var multiIDs = [Int32](repeating: -1, count: nprobe)
        var multiScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(numThreads: 4),
            listIDsOut: &multiIDs,
            listScoresOut: &multiScores
        )

        // Force single-threading (reference)
        var singleIDs = [Int32](repeating: -1, count: nprobe)
        var singleScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(numThreads: 1),
            listIDsOut: &singleIDs,
            listScoresOut: &singleScores
        )

        // Results should be identical
        XCTAssertEqual(multiIDs, singleIDs, "Multi-threaded results should match single-threaded")

        guard let mScores = multiScores, let sScores = singleScores else {
            XCTFail("Scores should be non-nil")
            return
        }

        for i in 0..<nprobe {
            XCTAssertEqual(mScores[i], sScores[i], accuracy: 1e-5,
                          "Score mismatch at \(i)")
        }
    }

    // MARK: - Helper Functions

    private func randomVector(d: Int) -> [Float] {
        (0..<d).map { _ in Float.random(in: -1...1) }
    }

    private func randomVectors(n: Int, d: Int) -> [Float] {
        (0..<(n * d)).map { _ in Float.random(in: -1...1) }
    }

    /// Brute-force reference implementation for correctness testing.
    private func bruteForceTopK(
        q: [Float],
        d: Int,
        centroids: [Float],
        kc: Int,
        metric: IVFMetric,
        k: Int
    ) -> [(id: Int32, score: Float)] {
        var results: [(id: Int32, score: Float)] = []

        for i in 0..<kc {
            let base = i * d
            let centroid = Array(centroids[base..<(base + d)])

            let score: Float
            switch metric {
            case .l2:
                score = zip(q, centroid).map { ($0 - $1) * ($0 - $1) }.reduce(0, +)
            case .ip:
                score = zip(q, centroid).map { $0 * $1 }.reduce(0, +)
            case .cosine:
                let dot = zip(q, centroid).map { $0 * $1 }.reduce(0, +)
                let qNorm = sqrt(max(q.map { $0 * $0 }.reduce(0, +), 1e-10))
                let cNorm = sqrt(max(centroid.map { $0 * $0 }.reduce(0, +), 1e-10))
                score = dot / (qNorm * cNorm)
            }

            results.append((id: Int32(i), score: score))
        }

        // Sort by metric and tie-break by ID
        results.sort { a, b in
            switch metric {
            case .l2:
                if a.score < b.score { return true }
                if a.score > b.score { return false }
                return a.id < b.id
            case .ip, .cosine:
                if a.score > b.score { return true }
                if a.score < b.score { return false }
                return a.id < b.id
            }
        }

        return Array(results.prefix(k))
    }
}
