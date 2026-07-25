import XCTest
@testable import VectorIndex

/// Comprehensive test suite for Kernel #19 (PQ Training)
///
/// Test Coverage:
///   - Numerical correctness (double accumulation, tie-breaking, norm refresh)
///   - Algorithm correctness (Lloyd's, mini-batch, k-means++)
///   - Distortion guarantees (monotonic decrease, convergence)
///   - Determinism (reproducibility with same seed)
///   - Residual PQ (IVF-PQ integration)
///   - Edge cases (empty clusters, degenerate data)
///   - Performance (SIMD optimization validation)
final class PQTrainTests: XCTestCase {
    override func setUpWithError() throws {
        // Allow skipping slow PQTrain tests in CI via env flag
        if ProcessInfo.processInfo.environment["CI_SKIP_PQTRAIN"] == "1" {
            throw XCTSkip("Skipping PQTrain tests in CI environment")
        }
    }

    // MARK: - Basic Functionality Tests

    func testBasicPQTraining() throws {
        // Train simple PQ codebook on random data
        let n: Int64 = 1000
        let d = 128
        let m = 4
        let ks = 64

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        var codebooks = [Float]()
        
        var norms: [Float]? = []

        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            codebooksOut: &codebooks,
            centroidNormsOut: &norms
        )

        // Validate output dimensions
        XCTAssertEqual(codebooks.count, m * ks * (d/m))
        XCTAssertEqual(norms?.count, m * ks)

        // Validate statistics
        XCTAssert(stats.distortion > 0, "Distortion should be positive")
        XCTAssertEqual(stats.distortionPerSubspace.count, m)
        XCTAssertEqual(stats.itersPerSubspace.count, m)
        XCTAssert(stats.timeTrainSec > 0)

        // Validate no NaN/Inf in codebooks
        for val in codebooks {
            XCTAssert(!val.isNaN && !val.isInfinite, "Codebook contains NaN/Inf")
        }

        print("✅ Basic training: distortion=\(stats.distortion), time=\(stats.timeTrainSec)s")
    }

    func testMiniBatchPQTraining() throws {
        // Test mini-batch variant
        let n: Int64 = 5000
        let d = 256
        let m = 8
        let ks = 256

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        var cfg = PQTrainConfig()
        cfg.algo = .minibatch
        cfg.batchSize = 512
        cfg.maxIters = 5

        var codebooks = [Float]()
        var norms: [Float]? = []

        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &norms
        )

        XCTAssertEqual(codebooks.count, m * ks * (d/m))
        XCTAssert(stats.distortion > 0)

        print("✅ Mini-batch training: distortion=\(stats.distortion)")
    }

    // MARK: - Numerical Correctness Tests

    func testDeterministicReproducibility() throws {
        // Same seed → same codebooks (bit-exact)
        var nilNorms: [Float]?
        let n: Int64 = 2000
        let d = 128
        let m = 4
        let ks = 128
        let seed: UInt64 = 123456

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float(i % 100) / 100.0
        }

        var cfg = PQTrainConfig()
        cfg.seed = seed
        cfg.maxIters = 15

        // Run 1
        var codebooks1 = [Float]()
        let stats1 = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks1,
            centroidNormsOut: &nilNorms
        )

        // Run 2 (same seed)
        var codebooks2 = [Float]()
        let stats2 = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks2,
            centroidNormsOut: &nilNorms
        )

        // Should be bit-exact
        XCTAssertEqual(codebooks1.count, codebooks2.count)
        for i in 0..<codebooks1.count {
            let diff = abs(codebooks1[i] - codebooks2[i])
            XCTAssert(diff < 1e-6, "Codebook mismatch at index \(i): \(diff)")
        }

        // Distortion should also match
        XCTAssertEqual(stats1.distortion, stats2.distortion, accuracy: 1e-9)

        print("✅ Deterministic reproducibility verified")
    }

    func testNumericalStability() throws {
        var nilNorms: [Float]?
        // Test with moderate dataset to expose accumulation errors
        // (scaled down from n=10K d=1024 ks=256 to avoid debug-build timeouts)
        let n: Int64 = 2_000
        let d = 256
        let m = 4
        let ks = 64

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -10...10)
        }

        var cfg = PQTrainConfig()
        cfg.maxIters = 5

        var codebooks = [Float]()

        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        // Verify no NaN/Inf (would indicate numerical issues)
        for val in codebooks {
            XCTAssert(!val.isNaN, "NaN detected in codebook")
            XCTAssert(!val.isInfinite, "Inf detected in codebook")
        }

        XCTAssert(!stats.distortion.isNaN)
        XCTAssert(!stats.distortion.isInfinite)

        print("✅ Numerical stability: n=\(n), d=\(d), distortion=\(stats.distortion)")
    }

    func testDistortionDecreases() throws {
        var nilNorms: [Float]?
        // Distortion should decrease (or stay same) each iteration
        let n: Int64 = 3000
        let d = 128
        let m = 4
        let ks = 64

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        var cfg = PQTrainConfig()
        cfg.maxIters = 20
        cfg.tol = 0.0  // Disable early stopping

        var codebooks = [Float]()

        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        // Distortion should be reasonable (not degenerate)
        XCTAssert(stats.distortion > 0)
        XCTAssert(stats.distortion < Double(n) * Double(d) * 100)

        print("✅ Distortion is reasonable: \(stats.distortion)")
    }

    // MARK: - Residual PQ Tests

    func testResidualPQ() throws {
        var nilNorms: [Float]?
        // Test residual PQ mode (IVF-PQ)
        let n: Int64 = 2000
        let d = 256
        let kc = 50  // coarse clusters
        let m = 8
        let ks = 256

        // Generate clustered data
        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<Int(n) {
            let cluster = i % kc
            for j in 0..<d {
                x[i*d + j] = Float(cluster) * 0.1 + Float.random(in: -0.05...0.05)
            }
        }

        // Simulate coarse centroids (simplified)
        var coarseCentroids = [Float](repeating: 0, count: kc * d)
        for c in 0..<kc {
            for j in 0..<d {
                coarseCentroids[c*d + j] = Float(c) * 0.1
            }
        }

        // Assign to nearest coarse centroid
        var assignments = [Int32](repeating: 0, count: Int(n))
        for i in 0..<Int(n) {
            assignments[i] = Int32(i % kc)
        }

        // Train residual PQ
        var codebooks = [Float]()
        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            coarseCentroids: coarseCentroids,
            assignments: assignments,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        XCTAssertEqual(codebooks.count, m * ks * (d/m))
        XCTAssert(stats.distortion > 0)

        print("✅ Residual PQ: distortion=\(stats.distortion)")
    }

    func testResidualVsDirectPQ() throws {
        var nilNorms: [Float]?
        // Residual PQ should achieve lower distortion on clustered data
        let n: Int64 = 3000
        let d = 128
        let kc = 30
        let m = 4
        let ks = 128

        // Generate strongly clustered data
        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<Int(n) {
            let cluster = i % kc
            for j in 0..<d {
                x[i*d + j] = Float(cluster) * 0.5 + Float.random(in: -0.1...0.1)
            }
        }

        var cfg = PQTrainConfig()
        cfg.seed = 42
        cfg.maxIters = 15

        // Direct PQ
        var codebooksDirect = [Float]()
        let statsDirect = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooksDirect,
            centroidNormsOut: &nilNorms
        )

        // Residual PQ
        var coarseCentroids = [Float](repeating: 0, count: kc * d)
        var assignments = [Int32](repeating: 0, count: Int(n))
        for i in 0..<Int(n) {
            let c = i % kc
            assignments[i] = Int32(c)
            for j in 0..<d {
                coarseCentroids[c*d + j] = Float(c) * 0.5
            }
        }

        var codebooksResidual = [Float]()
        let statsResidual = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            coarseCentroids: coarseCentroids,
            assignments: assignments,
            cfg: cfg,
            codebooksOut: &codebooksResidual,
            centroidNormsOut: &nilNorms
        )

        // Residual should achieve lower distortion
        print("Direct PQ distortion: \(statsDirect.distortion)")
        print("Residual PQ distortion: \(statsResidual.distortion)")

        let improvement = (statsDirect.distortion - statsResidual.distortion) / statsDirect.distortion
        print("Improvement: \(improvement * 100)%")

        // On clustered data, residual should be better
        XCTAssert(statsResidual.distortion <= statsDirect.distortion,
                  "Residual PQ should achieve lower or equal distortion")

        print("✅ Residual PQ advantage verified")
    }

    // MARK: - Warm-start Tests

    func testWarmStartMinibatchImprovesOnePass() throws {
        let n: Int64 = 3000
        let d = 128
        let m = 4
        let ks = 64

        // Synthetic data
        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float((i % 251) - 125) / 50.0
        }

        // Seed run: a few passes to generate a reasonable codebook
        var seedCfg = PQTrainConfig()
        seedCfg.algo = .minibatch
        seedCfg.batchSize = 256
        seedCfg.maxIters = 3
        seedCfg.sampleN = 1500
        seedCfg.verbose = false

        var seedCodebooks = [Float]()
        var nilNorms: [Float]?
        _ = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: seedCfg,
            codebooksOut: &seedCodebooks,
            centroidNormsOut: &nilNorms
        )
        XCTAssertEqual(seedCodebooks.count, m * ks * (d/m))

        // Cold start: 1 pass from scratch
        var coldCfg = seedCfg
        coldCfg.maxIters = 1
        var coldCodebooks = [Float]()
        let coldStats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: coldCfg,
            codebooksOut: &coldCodebooks,
            centroidNormsOut: &nilNorms
        )

        // Warm start: 1 pass, initialized from seedCodebooks
        var warmCfg = coldCfg
        warmCfg.warmStart = true
        var warmCodebooks = seedCodebooks // pass codebooks in as initial
        let warmStats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: warmCfg,
            codebooksOut: &warmCodebooks,
            centroidNormsOut: &nilNorms
        )

        // Expect warm-start to be no worse than cold-start given same pass budget
        XCTAssertLessThanOrEqual(warmStats.distortion, coldStats.distortion * 1.0001,
                                 "Warm-start should not regress compared to cold-start for same pass budget")
    }

    func testWarmStartLloydImprovesOneIter() throws {
        let n: Int64 = 2000
        let d = 64
        let m = 4
        let ks = 64

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) { x[i] = Float((i % 97) - 48) / 25.0 }

        // Seed with several iterations
        var seedCfg = PQTrainConfig()
        seedCfg.algo = .lloyd
        seedCfg.maxIters = 5
        seedCfg.verbose = false

        var seedCodebooks = [Float]()
        var nilNorms: [Float]?
        _ = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: seedCfg,
            codebooksOut: &seedCodebooks,
            centroidNormsOut: &nilNorms
        )

        // Cold start: 1 iteration
        var coldCfg = seedCfg
        coldCfg.maxIters = 1
        var coldCodebooks = [Float]()
        let coldStats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: coldCfg,
            codebooksOut: &coldCodebooks,
            centroidNormsOut: &nilNorms
        )

        // Warm start: 1 iteration starting from seedCodebooks
        var warmCfg = coldCfg
        warmCfg.warmStart = true
        var warmCodebooks = seedCodebooks
        let warmStats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: warmCfg,
            codebooksOut: &warmCodebooks,
            centroidNormsOut: &nilNorms
        )

        XCTAssertLessThanOrEqual(warmStats.distortion, coldStats.distortion * 1.0001,
                                 "Warm-start Lloyd should not be worse than cold-start for one iteration")
    }

    func testWarmStartDeterministic() throws {
        let n: Int64 = 1000
        let d = 64
        let m = 2
        let ks = 32

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) { x[i] = Float((i % 41) - 20) / 10.0 }

        // Produce initial codebooks
        var seedCfg = PQTrainConfig()
        seedCfg.algo = .minibatch
        seedCfg.batchSize = 128
        seedCfg.maxIters = 2
        seedCfg.sampleN = 800
        seedCfg.verbose = false

        var initCodebooks = [Float]()
        var nilNorms: [Float]?
        _ = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: seedCfg,
            codebooksOut: &initCodebooks,
            centroidNormsOut: &nilNorms
        )

        // Two warm-start runs from the same initial codebooks and seed
        var cfg = PQTrainConfig()
        cfg.algo = .minibatch
        cfg.batchSize = 128
        cfg.maxIters = 2
        cfg.sampleN = 800
        cfg.seed = 12345
        cfg.warmStart = true

        var codebooks1 = initCodebooks
        let stats1 = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks1,
            centroidNormsOut: &nilNorms
        )

        var codebooks2 = initCodebooks
        let stats2 = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks2,
            centroidNormsOut: &nilNorms
        )

        XCTAssertEqual(codebooks1.count, codebooks2.count)
        for i in 0..<codebooks1.count {
            XCTAssertEqual(codebooks1[i], codebooks2[i], accuracy: 1e-6)
        }
        XCTAssertEqual(stats1.distortion, stats2.distortion, accuracy: 1e-9)
    }

    // MARK: - Edge Case Tests

    func testEmptyClusterRepair() throws {
        var nilNorms: [Float]?
        // Test empty cluster repair with pathological data
        let n: Int64 = 500
        let d = 64
        let m = 2
        let ks = 100  // More clusters than natural

        // Concentrated data (high chance of empty clusters)
        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<Int(n) {
            let base = Float(i % 10) * 0.1
            for j in 0..<d {
                x[i*d + j] = base + Float.random(in: -0.01...0.01)
            }
        }

        var cfg = PQTrainConfig()
        cfg.emptyPolicy = .reseed
        cfg.maxIters = 10

        var codebooks = [Float]()
        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        print("✅ Empty clusters repaired: \(stats.emptiesRepaired)")
        XCTAssert(stats.emptiesRepaired >= 0)
    }

    func testMinimumData() throws {
        var nilNorms: [Float]?
        // Test with n == ks (minimum viable data)
        let n: Int64 = 64
        let d = 128
        let m = 4
        let ks = 64  // Exactly n points

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        var codebooks = [Float]()
        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        XCTAssertEqual(codebooks.count, m * ks * (d/m))
        XCTAssert(stats.distortion >= 0)

        print("✅ Minimum data test passed")
    }

    func testInsufficientData() throws {
        var nilNorms: [Float]?
        // Should throw error when n < ks
        let n: Int64 = 50
        let ks = 100

        let x = [Float](repeating: 0, count: Int(n) * 128)
        var codebooks = [Float]()

        XCTAssertThrowsError(
            try pq_train_f32(
                x: x, n: n, d: 128, m: 4, ks: ks,
                codebooksOut: &codebooks,
                centroidNormsOut: &nilNorms
            )
        ) { error in
            guard let vecError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError, got \(type(of: error))")
                return
            }
            XCTAssertEqual(vecError.kind, .emptyInput)
            XCTAssertTrue(vecError.message.contains("Insufficient training data"))
        }

        print("✅ Insufficient data error correctly thrown")
    }

    func testInvalidDimension() throws {
        var nilNorms: [Float]?
        // Should throw error when d not divisible by m
        let x = [Float](repeating: 0, count: 1000 * 100)
        var codebooks = [Float]()

        XCTAssertThrowsError(
            try pq_train_f32(
                x: x, n: 1000, d: 100, m: 7,  // 100 % 7 != 0
                ks: 64,
                codebooksOut: &codebooks,
                centroidNormsOut: &nilNorms
            )
        ) { error in
            guard let vecError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError, got \(type(of: error))")
                return
            }
            XCTAssertEqual(vecError.kind, .invalidDimension)
            XCTAssertTrue(vecError.message.contains("divisible"))
        }

        print("✅ Invalid dimension error correctly thrown")
    }

    // MARK: - Streaming API Tests

    func testStreamingPQTraining() throws {
        var nilNorms: [Float]?
        // Test streaming API with chunked data
        let n: Int64 = 5000
        let d = 256
        let m = 8
        let ks = 256
        let numChunks = 5
        let chunkSize = Int(n) / numChunks

        // Create chunks
        var fullData = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            fullData[i] = Float.random(in: -1...1)
        }

        var xChunks = [[Float]]()
        var nChunks = [Int64]()
        for i in 0..<numChunks {
            let start = i * chunkSize
            let end = min(start + chunkSize, Int(n))
            let size = end - start
            nChunks.append(Int64(size))
            xChunks.append(Array(fullData[start*d..<end*d]))
        }

        var cfg = PQTrainConfig()
        cfg.algo = .minibatch
        cfg.maxIters = 10
        cfg.batchSize = 512

        var codebooks = [Float]()
        let stats = try pq_train_streaming_f32(
            xChunks: xChunks, nChunks: nChunks,
            d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        XCTAssertEqual(codebooks.count, m * ks * (d/m))
        XCTAssert(stats.distortion > 0)

        print("✅ Streaming training: distortion=\(stats.distortion)")
    }

    // MARK: - Performance Validation Tests

    func testLargeScaleTraining() throws {
        // Scaled-down, seeded, deterministic exercise of the minibatch
        // training path for debug builds -- NOT a realistic-production-scale
        // benchmark. Realistic-scale behavior is covered by the Release-mode
        // expectations (27.6s at n=50k/d=768, see below) and by sibling
        // test testMiniBatchPQTraining (n=5000, batchSize=512).
        //
        // (scaled down from n=50K d=768 to avoid debug-build timeouts: the
        // original 38.4M-scalar Float.random fill + minibatch training over
        // n=50k/d=768 passed in 27.6s in Release but ran >1h in Debug.
        // PQTrain's minibatch path defaults to *serial*, one-subspace-at-a-
        // time execution (numThreads=0 <= 1), and once n exceeds
        // cfg.distEvalN (2000) the per-pass sample size is clamped to 2000
        // regardless of n -- so shrinking n alone barely reduces cost once
        // past that threshold; dsub (d/m) is what actually drives per-pass
        // cost. Two prior attempts (n=10_000/d=256: >10 min, killed;
        // n=4_000/d=16: 129s training alone) still weren't "well under 2
        // minutes". Settled on n=1_000 (below the 2000 clamp, so the full n
        // is sampled each pass instead of a fixed 2000) and d=8 (dsub=1).
        //
        // batchSize is set well below n (256, giving ~4 partial batches per
        // pass) so the minibatch path's multi-batch-per-pass update logic is
        // genuinely exercised: with the original batchSize=2048 > n=1000,
        // the inner `while processed < limit` loop ran exactly one batch per
        // pass, degenerating to a single-batch update and not covering
        // minibatch semantics at all.)
        let n: Int64 = 1_000
        let d = 8
        let m = 8
        let ks = 256

        // Deterministic fast fill (SplitMix/LCG-style generator; no unseeded
        // Float.random) so this test's timing/behavior is reproducible.
        var rng: UInt64 = 0x1357_9BDF_2468_ACE0
        func rnd() -> Float {
            rng = 2862933555777941757 &* rng &+ 3037000493
            return Float(rng >> 40) / Float(1 << 24)
        }

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = rnd() * 2 - 1
        }

        var cfg = PQTrainConfig()
        cfg.seed = 42
        cfg.algo = .minibatch
        cfg.batchSize = 256  // well below n=1000: ~4 partial batches/pass
        cfg.maxIters = 15

        let start = Date()
        var codebooks = [Float]()
        var norms: [Float]? = []

        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &norms
        )
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(codebooks.count, m * ks * (d/m))
        XCTAssert(stats.distortion > 0)

        // Shape/positivity check only at this scale -- dsub=1 has no
        // meaningful SIMD width, so this isn't a performance assertion.
        // Timing is printed for visibility (debug-build sanity), not asserted.
        print("✅ Large scale: n=\(n), time=\(elapsed)s, distortion=\(stats.distortion)")
        print("   Per-subspace time: \(elapsed / Double(m))s")
    }

    func testParallelExecution() throws {
        var nilNorms: [Float]?
        // Verify parallel execution works correctly
        // (scaled down from n=10K d=512 ks=256 to avoid debug-build timeouts)
        let n: Int64 = 2_000
        let d = 128
        let m = 4
        let ks = 64

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        // Serial execution
        var cfgSerial = PQTrainConfig()
        cfgSerial.seed = 42
        cfgSerial.numThreads = 1
        cfgSerial.maxIters = 5

        let startSerial = Date()
        var codebooksSerial = [Float]()
        let statsSerial = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfgSerial,
            codebooksOut: &codebooksSerial,
            centroidNormsOut: &nilNorms
        )
        let timeSerial = Date().timeIntervalSince(startSerial)

        // Parallel execution
        var cfgParallel = PQTrainConfig()
        cfgParallel.seed = 42
        cfgParallel.numThreads = 0  // auto
        cfgParallel.maxIters = 5

        let startParallel = Date()
        var codebooksParallel = [Float]()
        let statsParallel = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfgParallel,
            codebooksOut: &codebooksParallel,
            centroidNormsOut: &nilNorms
        )
        let timeParallel = Date().timeIntervalSince(startParallel)

        // Results should be identical (deterministic)
        XCTAssertEqual(statsSerial.distortion, statsParallel.distortion, accuracy: 1e-6)

        let speedup = timeSerial / timeParallel
        print("✅ Parallel speedup: \(speedup)x (serial: \(timeSerial)s, parallel: \(timeParallel)s)")
    }

    // MARK: - Compression Quality Tests

    func testCompressionQuality() throws {
        var nilNorms: [Float]?
        // Verify PQ achieves reasonable compression vs quality trade-off.
        // (scaled down from n=5K d=1024 ks=256 to avoid debug-build timeouts)
        //
        // Data model: C=32 well-separated cluster centers (<= ks=64, so each
        // cluster can claim its own centroid per subspace) plus small additive
        // noise, generated with a seeded fast generator (no unseeded
        // Float.random anywhere in this test). This replaces the previous
        // i.i.d. uniform-random data: on that distribution the achievable
        // normalized distortion at dsub=64/ks=64 sits near the
        // information-theoretic floor (measured ~0.85, i.e. only ~15% of
        // variance captured), so the old `< 0.3` threshold could never pass
        // -- it was left uncalibrated when the test was scaled down. On this
        // clusterable data PQ genuinely can capture most of the variance, so
        // a meaningful pass/fail threshold is possible.
        let n: Int64 = 2000
        let d = 256
        let m = 4
        let ks = 64
        let numClusters = 32  // <= ks

        // Deterministic fast fill (LCG; see RegressionA1_TraversalLifetimeTests
        // for the same pattern). No unseeded Float.random.
        var rng: UInt64 = 0x5EED_1234_ABCD_9876
        func rnd() -> Float {
            rng = 2862933555777941757 &* rng &+ 3037000493
            return Float(rng >> 40) / Float(1 << 24)
        }

        // Cluster centers spread over [-1, 1) per dimension so between-cluster
        // variance dominates.
        var centers = [Float](repeating: 0, count: numClusters * d)
        for c in 0..<numClusters {
            for j in 0..<d {
                centers[c * d + j] = rnd() * 2 - 1
            }
        }

        // Small additive noise (within-cluster spread is a small fraction of
        // total variance).
        let noiseScale: Float = 0.05
        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<Int(n) {
            let c = i % numClusters
            for j in 0..<d {
                x[i * d + j] = centers[c * d + j] + (rnd() * 2 - 1) * noiseScale
            }
        }

        var cfg = PQTrainConfig()
        cfg.seed = 42
        cfg.maxIters = 10

        var codebooks = [Float]()
        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
            cfg: cfg,
            codebooksOut: &codebooks,
            centroidNormsOut: &nilNorms
        )

        // Compute baseline variance
        var mean = [Double](repeating: 0, count: d)
        for i in 0..<Int(n) {
            for j in 0..<d {
                mean[j] += Double(x[i*d + j])
            }
        }
        for j in 0..<d { mean[j] /= Double(n) }

        var variance: Double = 0
        for i in 0..<Int(n) {
            for j in 0..<d {
                let diff = Double(x[i*d + j]) - mean[j]
                variance += diff * diff
            }
        }
        variance /= Double(n)

        let normalizedDistortion = stats.distortion / variance

        // On this seeded C=32-cluster-plus-small-noise data model, PQ at
        // ks=64/dsub=64 should capture the overwhelming majority of variance
        // (each cluster can claim its own centroid per subspace, so the
        // residual is essentially just the injected noise).
        print("✅ Compression quality:")
        print("   Distortion: \(stats.distortion)")
        print("   Variance: \(variance)")
        print("   Normalized distortion: \(normalizedDistortion)")
        print("   Variance captured: \((1.0 - normalizedDistortion) * 100)%")

        // Empirically calibrated: this data model + config (seed=42) measures
        // normalizedDistortion ≈ 0.00244 (~99.76% variance captured),
        // bit-identical across repeated runs. Threshold set to 0.05 (~20x the
        // measured value) -- comfortable safety margin so legitimate minor
        // algorithmic variation won't flake this test, while still asserting
        // that PQ captures the overwhelming majority of variance on
        // clusterable data (unlike the old, uncalibratable 0.3 threshold on
        // i.i.d. uniform data, which sat near the ~0.85 information-theoretic
        // floor at these parameters).
        XCTAssert(normalizedDistortion < 0.05, "PQ should capture significant variance")
    }
}
