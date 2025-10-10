import XCTest
@testable import VectorIndex

/// Comprehensive tests for Kernel #11: K-means++ Seeding
/// Validates D² sampling correctness, determinism, and edge cases
final class KMeansPPSeedingTests: XCTestCase {

    // MARK: - Basic Correctness Tests

    /// Test that k centroids are selected from the data
    func testBasicSeeding() {
        let n = 100
        let d = 4
        let k = 10

        // Generate random data
        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: 0..<1)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        var chosenIndices = [Int](repeating: -1, count: k)

        let stats = kmeansPlusPlusSeed(
            data: data,
            count: n,
            dimension: d,
            k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: &chosenIndices
        )

        // Verify stats
        XCTAssertEqual(stats.k, k)
        XCTAssertEqual(stats.n, n)
        XCTAssertEqual(stats.dimension, d)
        XCTAssertEqual(stats.chosenIndices.count, k)
        XCTAssertEqual(stats.passesOverData, k)

        // Verify all chosen indices are valid and unique
        XCTAssertEqual(Set(chosenIndices).count, k, "Chosen indices must be unique")
        for idx in chosenIndices {
            XCTAssertGreaterThanOrEqual(idx, 0)
            XCTAssertLessThan(idx, n)
        }

        // Verify centroids match data at chosen indices
        for t in 0..<k {
            let chosenIdx = chosenIndices[t]
            let dataVec = Array(data[(chosenIdx * d)..<((chosenIdx + 1) * d)])
            let centroidVec = Array(centroids[(t * d)..<((t + 1) * d)])
            XCTAssertEqual(dataVec, centroidVec,
                          "Centroid \(t) must match data[\(chosenIdx)]")
        }
    }

    /// Test determinism: same seed produces same centroids
    func testDeterministicSeeding() {
        let n = 50
        let d = 3
        let k = 5
        let seed: UInt64 = 12345

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 100) / 10.0
        }

        // Run seeding twice with same seed
        var centroids1 = [Float](repeating: 0, count: k * d)
        var indices1 = [Int](repeating: -1, count: k)
        let stats1 = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: seed),
            centroidsOut: &centroids1,
            chosenIndicesOut: &indices1
        )

        var centroids2 = [Float](repeating: 0, count: k * d)
        var indices2 = [Int](repeating: -1, count: k)
        let stats2 = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: seed),
            centroidsOut: &centroids2,
            chosenIndicesOut: &indices2
        )

        // Verify identical results
        XCTAssertEqual(indices1, indices2, "Same seed must produce same indices")
        XCTAssertEqual(centroids1, centroids2, "Same seed must produce same centroids")
        XCTAssertEqual(stats1.totalCost, stats2.totalCost, accuracy: 1e-5,
                      "Same seed must produce same total cost")
    }

    /// Test different seeds produce different results
    func testDifferentSeedsDiverge() {
        let n = 50
        let d = 3
        let k = 5

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 100) / 10.0
        }

        // Run with different seeds
        var indices1 = [Int](repeating: -1, count: k)
        var centroids1 = [Float](repeating: 0, count: k * d)
        _ = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: 111),
            centroidsOut: &centroids1,
            chosenIndicesOut: &indices1
        )

        var indices2 = [Int](repeating: -1, count: k)
        var centroids2 = [Float](repeating: 0, count: k * d)
        _ = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: 222),
            centroidsOut: &centroids2,
            chosenIndicesOut: &indices2
        )

        // Should get different selections
        XCTAssertNotEqual(indices1, indices2,
                         "Different seeds should produce different centroids")
    }

    // MARK: - D² Sampling Correctness

    /// Test that D² sampling favors distant points
    func testDSquaredSamplingBias() {
        // Create data with 3 tight clusters and 1 outlier
        // K-means++ should prefer outlier as 2nd centroid
        let d = 2

        // Cluster at origin: 10 points
        var data = [Float]()
        for _ in 0..<10 {
            data += [Float.random(in: -0.1..<0.1), Float.random(in: -0.1..<0.1)]
        }

        // Outlier far away
        data += [10.0, 10.0]

        let n = data.count / d
        let k = 2

        // Run seeding multiple times
        var outlierSelected = 0
        let trials = 100

        for seed in 0..<UInt64(trials) {
            var centroids = [Float](repeating: 0, count: k * d)
            var indices = [Int](repeating: -1, count: k)

            _ = kmeansPlusPlusSeed(
                data: data, count: n, dimension: d, k: k,
                config: KMeansSeedConfig(k: k, rngSeed: seed),
                centroidsOut: &centroids,
                chosenIndicesOut: &indices
            )

            // Check if outlier (index 10) was selected
            if indices.contains(10) {
                outlierSelected += 1
            }
        }

        // D² weighting should select outlier more often than 2/11 random chance
        let randomExpected = Double(trials) * 2.0 / Double(n)
        XCTAssertGreaterThan(Double(outlierSelected), randomExpected * 1.5,
                            "D² sampling should favor distant points (selected \(outlierSelected)/\(trials), expected >\(randomExpected * 1.5))")
    }

    // MARK: - Edge Cases

    /// Test k=1 (single centroid selection)
    func testKEqualsOne() {
        let n = 20
        let d = 3
        let k = 1

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i)
        }

        var centroid = [Float](repeating: 0, count: d)
        var chosenIndex = [Int](repeating: -1, count: 1)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: 999),
            centroidsOut: &centroid,
            chosenIndicesOut: &chosenIndex
        )

        XCTAssertEqual(stats.chosenIndices.count, 1)
        XCTAssertGreaterThanOrEqual(chosenIndex[0], 0)
        XCTAssertLessThan(chosenIndex[0], n)
        XCTAssertEqual(stats.passesOverData, 1)
    }

    /// Test k=n (select all points as centroids)
    func testKEqualsN() {
        let n = 10
        let d = 2
        let k = n

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        var chosenIndices = [Int](repeating: -1, count: k)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: &chosenIndices
        )

        // All points should be selected
        XCTAssertEqual(Set(chosenIndices).count, n, "All points must be selected when k=n")
        XCTAssertEqual(stats.totalCost, 0.0, accuracy: 1e-6,
                      "Total cost should be 0 when k=n (all points are centroids)")
    }

    /// Test with high-dimensional data
    func testHighDimension() {
        let n = 50
        let d = 128  // Typical embedding dimension
        let k = 10

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -1..<1)
        }

        var centroids = [Float](repeating: 0, count: k * d)
        var chosenIndices = [Int](repeating: -1, count: k)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: &chosenIndices
        )

        XCTAssertEqual(stats.dimension, d)
        XCTAssertEqual(stats.chosenIndices.count, k)
        XCTAssertEqual(Set(chosenIndices).count, k)

        // Verify total cost is reasonable (not NaN/Inf)
        XCTAssertTrue(stats.totalCost.isFinite, "Total cost must be finite")
        XCTAssertGreaterThan(stats.totalCost, 0, "Total cost must be positive")
    }

    /// Test with duplicate data points
    func testDuplicateData() {
        let d = 3
        let k = 3

        // Create data with duplicates: [0,0,0] repeated 5 times, then [1,1,1] repeated 5 times
        var data = [Float]()
        for _ in 0..<5 {
            data += [0, 0, 0]
        }
        for _ in 0..<5 {
            data += [1, 1, 1]
        }
        let n = 10

        var centroids = [Float](repeating: 0, count: k * d)
        var chosenIndices = [Int](repeating: -1, count: k)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: 555),
            centroidsOut: &centroids,
            chosenIndicesOut: &chosenIndices
        )

        // Should still select k unique indices (even if data points are duplicates)
        XCTAssertEqual(Set(chosenIndices).count, k)
        XCTAssertEqual(stats.chosenIndices.count, k)
    }

    // MARK: - Stream ID Tests

    /// Test that different stream IDs produce different results
    func testStreamIDIndependence() {
        let n = 30
        let d = 4
        let k = 5
        let seed: UInt64 = 42

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 50) / 5.0
        }

        // Run with different stream IDs
        var indices0 = [Int](repeating: -1, count: k)
        var centroids0 = [Float](repeating: 0, count: k * d)
        _ = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: seed, rngStreamID: 0),
            centroidsOut: &centroids0,
            chosenIndicesOut: &indices0
        )

        var indices1 = [Int](repeating: -1, count: k)
        var centroids1 = [Float](repeating: 0, count: k * d)
        _ = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: KMeansSeedConfig(k: k, rngSeed: seed, rngStreamID: 1),
            centroidsOut: &centroids1,
            chosenIndicesOut: &indices1
        )

        // Different streams should produce different selections
        XCTAssertNotEqual(indices0, indices1,
                         "Different stream IDs should produce different centroids")
    }

    // MARK: - Statistics Validation

    /// Test that average distance squared is computed correctly
    func testAverageDistanceSquared() {
        let n = 20
        let d = 2
        let k = 3

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i % 10)
        }

        var centroids = [Float](repeating: 0, count: k * d)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )

        // Verify average = totalCost / n
        let expectedAvg = stats.totalCost / Double(n)
        XCTAssertEqual(stats.averageDistanceSquared, expectedAvg, accuracy: 1e-6)

        // Average should be non-negative and finite
        XCTAssertGreaterThanOrEqual(stats.averageDistanceSquared, 0)
        XCTAssertTrue(stats.averageDistanceSquared.isFinite)
    }

    /// Test execution time is tracked
    func testExecutionTimeTracking() {
        let n = 100
        let d = 10
        let k = 10

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: 0..<1)
        }

        var centroids = [Float](repeating: 0, count: k * d)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )

        // Should have some execution time (non-zero)
        XCTAssertGreaterThan(stats.executionTimeNanos, 0)
    }

    // MARK: - Numerical Stability Tests

    /// Test handling of NaN/Inf in data
    func testNaNHandling() {
        let n = 10
        let d = 3
        let k = 3

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float(i)
        }

        // Inject NaN
        data[5] = Float.nan

        var centroids = [Float](repeating: 0, count: k * d)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )

        // Algorithm should guard against NaN (treat as 0 distance)
        // Total cost should still be finite
        XCTAssertTrue(stats.totalCost.isFinite || stats.totalCost == 0,
                     "NaN handling should produce finite cost")
    }

    /// Test with very small distances (numerical precision)
    func testSmallDistances() {
        let n = 20
        let d = 4
        let k = 5

        // Create data with very small variations
        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<n {
            for j in 0..<d {
                data[i * d + j] = Float(i) * 1e-6
            }
        }

        var centroids = [Float](repeating: 0, count: k * d)
        var chosenIndices = [Int](repeating: -1, count: k)

        let stats = kmeansPlusPlusSeed(
            data: data, count: n, dimension: d, k: k,
            config: .default,
            centroidsOut: &centroids,
            chosenIndicesOut: &chosenIndices
        )

        // Should still select k unique centroids
        XCTAssertEqual(Set(chosenIndices).count, k)
        XCTAssertTrue(stats.totalCost.isFinite)
    }

    // MARK: - Performance Tests

    /// Measure seeding performance on moderate dataset
    func testPerformanceModerate() {
        let n = 1000
        let d = 128
        let k = 50

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -1..<1)
        }

        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: data, count: n, dimension: d, k: k,
                config: .default,
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }
    }

    /// Measure seeding performance on large dataset
    func testPerformanceLarge() {
        let n = 10000
        let d = 64
        let k = 256

        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -1..<1)
        }

        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: data, count: n, dimension: d, k: k,
                config: .default,
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }
    }
}
