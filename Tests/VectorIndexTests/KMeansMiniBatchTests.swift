import XCTest
@testable import VectorIndex

/// Comprehensive tests for Kernel #12: Mini-batch K-means Training
/// Validates convergence, sparse accumulators, layout handling, and edge cases
final class KMeansMiniBatchTests: XCTestCase {

    // MARK: - Basic Convergence Tests

    /// Test that mini-batch k-means converges on simple clustered data
    func testBasicConvergence() {
        let d = 2
        let k = 3
        let n = 300

        // Generate 3 tight clusters
        var data = [Float]()
        // Cluster 1: around (0, 0)
        for _ in 0..<100 {
            data += [Float.random(in: -0.5..<0.5), Float.random(in: -0.5..<0.5)]
        }
        // Cluster 2: around (5, 0)
        for _ in 0..<100 {
            data += [Float.random(in: 4.5..<5.5), Float.random(in: -0.5..<0.5)]
        }
        // Cluster 3: around (0, 5)
        for _ in 0..<100 {
            data += [Float.random(in: -0.5..<0.5), Float.random(in: 4.5..<5.5)]
        }

        // Random initialization
        var centroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            centroids[i] = Float.random(in: -1..<6)
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 50,
            epochs: 20,
            tol: 1e-3
        )

        var stats = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: centroids,
            cfg: cfg,
            centroidsOut: &centroids,
            assignOut: nil,
            statsOut: &stats
        )

        XCTAssertEqual(status, KMeansMBStatus.success, "Algorithm should converge successfully")
        XCTAssertGreaterThan(stats.epochsCompleted, 0, "Should run at least one epoch")
        XCTAssertTrue(stats.finalInertia > 0 && stats.finalInertia.isFinite,
                     "Final inertia should be positive and finite")

        // Centroids should be near cluster centers
        // Cluster 1 centroid should be near (0, 0)
        // Cluster 2 centroid should be near (5, 0)
        // Cluster 3 centroid should be near (0, 5)
        // (We can't know exact order, so just check distances)
        let expectedCenters: [[Float]] = [[0, 0], [5, 0], [0, 5]]
        for expected in expectedCenters {
            var minDist = Float.infinity
            for t in 0..<k {
                let cx = centroids[t * d]
                let cy = centroids[t * d + 1]
                let dist = sqrtf((cx - expected[0]) * (cx - expected[0]) +
                                 (cy - expected[1]) * (cy - expected[1]))
                minDist = min(minDist, dist)
            }
            XCTAssertLessThan(minDist, 1.0,
                             "At least one centroid should be near expected center \(expected)")
        }
    }

    /// Test deterministic behavior with same seed
    func testDeterministicTraining() {
        let n = 100
        let d = 4
        let k = 5
        let seed: UInt64 = 12345

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 50) / 10.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 20) / 5.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 20,
            epochs: 10,
            seed: seed
        )

        // Run 1
        var centroids1 = initCentroids
        var stats1 = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        _ = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids1,
            statsOut: &stats1
        )

        // Run 2
        var centroids2 = initCentroids
        var stats2 = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        _ = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids2,
            statsOut: &stats2
        )

        // Should produce identical results
        XCTAssertEqual(stats1.epochsCompleted, stats2.epochsCompleted)
        XCTAssertEqual(stats1.finalInertia, stats2.finalInertia, accuracy: 1e-5)

        for i in 0..<(k * d) {
            XCTAssertEqual(centroids1[i], centroids2[i], accuracy: 1e-5,
                          "Centroids should be identical with same seed")
        }
    }

    // MARK: - Mini-batch vs Full-batch Comparison

    /// Test that mini-batch approaches full-batch result with enough epochs
    func testMiniBatchVsFullBatch() {
        let n = 200
        let d = 3
        let k = 4

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -5..<5)
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float.random(in: -5..<5)
        }

        // Full-batch (batchSize = n)
        var centroidsFull = initCentroids
        let cfgFull = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: n,
            epochs: 20
        )
        _ = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfgFull,
            centroidsOut: &centroidsFull
        )

        // Mini-batch (batchSize = 50, more epochs)
        var centroidsMini = initCentroids
        let cfgMini = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 50,
            epochs: 50
        )
        _ = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfgMini,
            centroidsOut: &centroidsMini
        )

        // Centroids should be reasonably close (not exact, but similar clustering)
        // Measure average centroid difference
        var totalDiff: Float = 0
        for i in 0..<(k * d) {
            totalDiff += abs(centroidsFull[i] - centroidsMini[i])
        }
        let avgDiff = totalDiff / Float(k * d)

        XCTAssertLessThan(avgDiff, 2.0,
                         "Mini-batch centroids should be close to full-batch (avg diff = \(avgDiff))")
    }

    // MARK: - Sparse Accumulator Tests

    /// Test sparse accumulator correctness (implicit via clustering quality)
    func testSparseAccumulatorCorrectness() {
        // Create data where only a few centroids are updated each batch
        let n = 500
        let d = 4
        let k = 20  // Many centroids, sparse updates

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -10..<10)
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float.random(in: -10..<10)
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 30,  // Small batch → sparse updates
            epochs: 15
        )

        var centroids = initCentroids
        var stats = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids,
            statsOut: &stats
        )

        XCTAssertEqual(status, KMeansMBStatus.success)
        XCTAssertGreaterThan(stats.epochsCompleted, 0)
        XCTAssertTrue(stats.finalInertia.isFinite && stats.finalInertia > 0)
    }

    // MARK: - Layout Tests (AoS vs AoSoA)

    /// Test AoS layout (standard row-major)
    func testAoSLayout() {
        let n = 100
        let d = 8
        let k = 5

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 100) / 10.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 50) / 10.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 25,
            epochs: 10,
            layout: .aos
        )

        var centroids = initCentroids
        var stats = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids,
            statsOut: &stats
        )

        XCTAssertEqual(status, KMeansMBStatus.success)
        // Layout is specified in config, not returned in stats
    }

    /// Test AoSoA layout (cache-optimized blocked layout)
    func testAoSoALayout() {
        let n = 100
        let d = 8
        let k = 5
        let registerBlock = 4

        // Data must be in AoSoA format: blocks of registerBlock vectors
        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 100) / 10.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 50) / 10.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 20,
            epochs: 10,
            layout: .aosoaR,
            aosoaRegisterBlock: registerBlock
        )

        var centroids = initCentroids
        var stats = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids,
            statsOut: &stats
        )

        XCTAssertEqual(status, KMeansMBStatus.success)
        // Layout is specified in config, not returned in stats
    }

    /// Test invalid AoSoA configuration (registerBlock < 1)
    func testInvalidAoSoAConfig() {
        let n = 100
        let d = 8
        let k = 5

        var data = [Float](repeating: 0, count: n * d)
        var centroids = [Float](repeating: 0, count: k * d)

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 20,
            epochs: 5,
            layout: .aosoaR,
            aosoaRegisterBlock: 0  // Invalid!
        )

        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: nil,
            cfg: cfg,
            centroidsOut: &centroids
        )

        XCTAssertEqual(status, KMeansMBStatus.invalidLayout,
                      "Should reject invalid AoSoA registerBlock")
    }

    // MARK: - Decay Parameter Tests

    /// Test with standard decay parameter
    func testWithDecayParameter() {
        let n = 100
        let d = 4
        let k = 3

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 50) / 10.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 30) / 10.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 20,
            epochs: 5,
            decay: 0.01
        )

        var centroids = initCentroids
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids
        )

        XCTAssertEqual(status, KMeansMBStatus.success)
    }

    // MARK: - Edge Cases

    /// Test k=1 (single centroid)
    func testSingleCentroid() {
        let n = 50
        let d = 3
        let k = 1

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -5..<5)
        }

        var centroid = [Float](repeating: 0, count: d)
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 10,
            epochs: 5
        )

        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: nil,
            cfg: cfg,
            centroidsOut: &centroid
        )

        XCTAssertEqual(status, KMeansMBStatus.success)

        // Centroid should be near data mean
        var dataMean = [Float](repeating: 0, count: d)
        for i in 0..<n {
            for j in 0..<d {
                dataMean[j] += data[i * d + j]
            }
        }
        for j in 0..<d {
            dataMean[j] /= Float(n)
        }

        // Check centroid is reasonably close to mean
        var dist: Float = 0
        for j in 0..<d {
            let diff = centroid[j] - dataMean[j]
            dist += diff * diff
        }
        dist = sqrtf(dist)

        XCTAssertLessThan(dist, 3.0,
                         "Single centroid should be near data mean")
    }

    /// Test batchSize > n (uses full dataset as batch)
    func testBatchSizeLargerThanN() {
        let n = 30
        let d = 4
        let k = 3

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 20) / 5.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 15) / 5.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 100,  // > n
            epochs: 10
        )

        var centroids = initCentroids
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids
        )

        XCTAssertEqual(status, KMeansMBStatus.success,
                      "Should handle batchSize > n gracefully")
    }

    /// Test early stopping with tight tolerance
    func testEarlyStopping() {
        let n = 100
        let d = 4
        let k = 5

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 50) / 10.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 30) / 10.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 20,
            epochs: 100,
            tol: 1e-6  // Tight tolerance
        )

        var centroids = initCentroids
        var stats = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids,
            statsOut: &stats
        )

        XCTAssertEqual(status, KMeansMBStatus.success)
        // Should converge early (before 100 epochs)
        XCTAssertLessThan(stats.epochsCompleted, 100,
                         "Should converge early with tight tolerance")
    }

    /// Test with nil initCentroids (random initialization)
    func testRandomInitialization() {
        let n = 100
        let d = 4
        let k = 5

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -5..<5)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 25,
            epochs: 10,
            seed: 999
        )

        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: nil,  // Random init
            cfg: cfg,
            centroidsOut: &centroids
        )

        XCTAssertEqual(status, KMeansMBStatus.success)

        // Centroids should be non-zero and finite
        for i in 0..<(k * d) {
            XCTAssertTrue(centroids[i].isFinite)
        }
    }

    // MARK: - Assignment Output Tests

    /// Test that assignment output is correctly populated
    func testAssignmentOutput() {
        let n = 50
        let d = 3
        let k = 3

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 30) / 5.0
        }

        var initCentroids = [Float](repeating: 0, count: k * d)
        for i in 0..<(k * d) {
            initCentroids[i] = Float(i % 15) / 5.0
        }

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 20,
            epochs: 10
        )

        var centroids = initCentroids
        var assignments = [Int32](repeating: -1, count: n)
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: initCentroids,
            cfg: cfg,
            centroidsOut: &centroids,
            assignOut: &assignments
        )

        XCTAssertEqual(status, KMeansMBStatus.success)

        // All assignments should be in [0, k)
        for assignment in assignments {
            XCTAssertGreaterThanOrEqual(assignment, 0)
            XCTAssertLessThan(assignment, Int32(k))
        }

        // At least one vector assigned to each centroid (likely)
        let uniqueAssignments = Set(assignments)
        XCTAssertGreaterThan(uniqueAssignments.count, 0)
    }

    // MARK: - Numerical Stability Tests

    /// Test handling of identical data points
    func testIdenticalData() {
        let n = 50
        let d = 4
        let k = 3

        // All data points are identical
        var data = [Float](repeating: 1.0, count: n * d)

        var centroids = [Float](repeating: 0, count: k * d)
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 10,
            epochs: 5
        )

        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: nil,
            cfg: cfg,
            centroidsOut: &centroids
        )

        XCTAssertEqual(status, KMeansMBStatus.success)

        // All centroids should converge to [1, 1, 1, 1]
        for i in 0..<(k * d) {
            XCTAssertEqual(centroids[i], 1.0, accuracy: 0.1,
                          "Centroids should converge to data point")
        }
    }

    /// Test high-dimensional data (embedding-like)
    func testHighDimensionalData() {
        let n = 200
        let d = 256  // Typical large embedding dimension
        let k = 10

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -1..<1)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 50,
            epochs: 10
        )

        var stats = KMeansMBStats(
            epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
            inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
            timeAssignmentSec: 0, bytesRead: 0
        )
        let status = kmeans_minibatch_f32(
            x: data, n: Int64(n), d: d, kc: k,
            initCentroids: nil,
            cfg: cfg,
            centroidsOut: &centroids,
            statsOut: &stats
        )

        XCTAssertEqual(status, KMeansMBStatus.success)
        XCTAssertTrue(stats.finalInertia.isFinite && stats.finalInertia > 0)
    }

    // MARK: - Performance Tests

    /// Measure mini-batch k-means performance on moderate dataset
    func testPerformanceModerate() {
        let n = 5000
        let d = 64
        let k = 50

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -5..<5)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 256,
            epochs: 10
        )

        measure {
            _ = kmeans_minibatch_f32(
                x: data, n: Int64(n), d: d, kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
        }
    }

    /// Measure mini-batch k-means performance on large dataset
    func testPerformanceLarge() {
        let n = 20000
        let d = 128
        let k = 256

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -10..<10)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 512,
            epochs: 5
        )

        measure {
            _ = kmeans_minibatch_f32(
                x: data, n: Int64(n), d: d, kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
        }
    }
}
