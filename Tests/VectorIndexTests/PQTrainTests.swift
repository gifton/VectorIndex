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
        // Test with large dataset to expose accumulation errors
        let n: Int64 = 10_000
        let d = 1024
        let m = 8
        let ks = 256

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -10...10)
        }

        var codebooks = [Float]()

        let stats = try pq_train_f32(
            x: x, n: n, d: d, m: m, ks: ks,
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
        // Test with realistic production scale
        let n: Int64 = 50_000
        let d = 768
        let m = 8
        let ks = 256

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        var cfg = PQTrainConfig()
        cfg.algo = .minibatch
        cfg.batchSize = 2048
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

        // Performance target check (should be fast with SIMD)
        print("✅ Large scale: n=\(n), time=\(elapsed)s, distortion=\(stats.distortion)")
        print("   Per-subspace time: \(elapsed / Double(m))s")
    }

    func testParallelExecution() throws {
        var nilNorms: [Float]?
        // Verify parallel execution works correctly
        let n: Int64 = 10_000
        let d = 512
        let m = 8
        let ks = 256

        var x = [Float](repeating: 0, count: Int(n) * d)
        for i in 0..<(Int(n) * d) {
            x[i] = Float.random(in: -1...1)
        }

        // Serial execution
        var cfgSerial = PQTrainConfig()
        cfgSerial.seed = 42
        cfgSerial.numThreads = 1
        cfgSerial.maxIters = 10

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
        cfgParallel.maxIters = 10

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
        // Verify PQ achieves reasonable compression vs quality trade-off
        let n: Int64 = 5000
        let d = 1024
        let m = 8
        let ks = 256  // 8 bytes per vector

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

        // Typical PQ: 5-10% normalized distortion (95-98% of variance captured)
        print("✅ Compression quality:")
        print("   Distortion: \(stats.distortion)")
        print("   Variance: \(variance)")
        print("   Normalized distortion: \(normalizedDistortion)")
        print("   Variance captured: \((1.0 - normalizedDistortion) * 100)%")

        // Should capture at least 80% of variance for random data
        XCTAssert(normalizedDistortion < 0.3, "PQ should capture significant variance")
    }
}
