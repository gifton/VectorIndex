import XCTest
@testable import VectorIndex

/// Performance benchmarks for K-means Kernels #11 and #12
/// Measures throughput, memory usage, and scalability characteristics
///
/// NOTE: These benchmarks are DISABLED by default as they can take many minutes to run.
/// They use XCTest's `measure {}` blocks which run 10 iterations each, and test large
/// datasets (up to 50K vectors × 256D). To enable for performance profiling, comment out
/// the `throw XCTSkip(...)` line in each test method.
final class KMeansKernelBenchmarks: XCTestCase {

    // MARK: - Kernel #11 Benchmarks (K-means++ Seeding)

    /// Benchmark k-means++ seeding: Small dataset (1K vectors, 64D)
    func testBench_KMeansPP_Small_1K_64D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 1_000
        let d = 64
        let k = 50

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: &data,
                count: n,
                dimension: d,
                k: k,
                config: KMeansSeedConfig(k: k, rngSeed: 42),
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        let ops = n * d * k  // Distance computations
        print("  → Data: \(bytesProcessed / 1024)KB, Operations: \(ops / 1_000_000)M")
    }

    /// Benchmark k-means++ seeding: Medium dataset (10K vectors, 128D)
    func testBench_KMeansPP_Medium_10K_128D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: &data,
                count: n,
                dimension: d,
                k: k,
                config: KMeansSeedConfig(k: k, rngSeed: 42),
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        print("  → Data: \(bytesProcessed / (1024 * 1024))MB")
    }

    /// Benchmark k-means++ seeding: Large dataset (50K vectors, 256D)
    func testBench_KMeansPP_Large_50K_256D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 50_000
        let d = 256
        let k = 256

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: &data,
                count: n,
                dimension: d,
                k: k,
                config: KMeansSeedConfig(k: k, rngSeed: 42),
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        print("  → Data: \(bytesProcessed / (1024 * 1024))MB")
    }

    /// Benchmark k-means++ seeding: High-dimensional (10K vectors, 1024D)
    func testBench_KMeansPP_HighDim_10K_1024D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 1024  // Very high dimension
        let k = 100

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: &data,
                count: n,
                dimension: d,
                k: k,
                config: KMeansSeedConfig(k: k, rngSeed: 42),
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        print("  → Data: \(bytesProcessed / (1024 * 1024))MB, Dimension: \(d)")
    }

    /// Benchmark k-means++ seeding: Many clusters (10K vectors, 128D, k=1000)
    func testBench_KMeansPP_ManyClusters_10K_k1000() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 1000  // Many centroids

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        measure {
            _ = kmeansPlusPlusSeed(
                data: &data,
                count: n,
                dimension: d,
                k: k,
                config: KMeansSeedConfig(k: k, rngSeed: 42),
                centroidsOut: &centroids,
                chosenIndicesOut: nil
            )
        }

        print("  → Centroids: \(k), Data: \(n) vectors")
    }

    // MARK: - Kernel #12 Benchmarks (Mini-batch K-means)

    /// Benchmark mini-batch k-means: Small dataset (1K vectors, 64D)
    func testBench_MiniBatch_Small_1K_64D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 1_000
        let d = 64
        let k = 50

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 128,
            epochs: 10,
            seed: 42
        )

        measure {
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        print("  → Data: \(bytesProcessed / 1024)KB, Epochs: \(cfg.epochs)")
    }

    /// Benchmark mini-batch k-means: Medium dataset (10K vectors, 128D)
    func testBench_MiniBatch_Medium_10K_128D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 256,
            epochs: 10,
            seed: 42
        )

        measure {
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        print("  → Data: \(bytesProcessed / (1024 * 1024))MB")
    }

    /// Benchmark mini-batch k-means: Large dataset (50K vectors, 256D)
    func testBench_MiniBatch_Large_50K_256D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 50_000
        let d = 256
        let k = 256

        var data = generateRandomData(n: n, d: d)
        var centroids = [Float](repeating: 0, count: k * d)

        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 512,
            epochs: 10,
            seed: 42
        )

        measure {
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
        }

        let bytesProcessed = n * d * MemoryLayout<Float>.stride
        print("  → Data: \(bytesProcessed / (1024 * 1024))MB")
    }

    /// Benchmark mini-batch k-means: Batch size sweep (10K vectors, 128D)
    func testBench_MiniBatch_BatchSizeSweep_10K_128D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        let batchSizes = [64, 128, 256, 512, 1024]

        for batchSize in batchSizes {
            var data = generateRandomData(n: n, d: d)
            var centroids = [Float](repeating: 0, count: k * d)

            let cfg = KMeansMBConfig(
                algo: .lloydMiniBatch,
                batchSize: batchSize,
                epochs: 10,
                seed: 42
            )

            let start = DispatchTime.now().uptimeNanoseconds
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            let ms = Double(elapsed) / 1_000_000.0

            print("  → BatchSize: \(batchSize), Time: \(String(format: "%.2f", ms))ms")
        }
    }

    /// Benchmark mini-batch k-means: AoS vs AoSoA layout (10K vectors, 128D)
    func testBench_MiniBatch_LayoutComparison_10K_128D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        // Benchmark AoS layout
        var dataAoS = generateRandomData(n: n, d: d)
        var centroidsAoS = [Float](repeating: 0, count: k * d)

        let cfgAoS = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 256,
            epochs: 10,
            seed: 42,
            layout: .aos
        )

        let startAoS = DispatchTime.now().uptimeNanoseconds
        _ = kmeans_minibatch_f32(
            x: &dataAoS,
            n: Int64(n),
            d: d,
            kc: k,
            initCentroids: nil,
            cfg: cfgAoS,
            centroidsOut: &centroidsAoS
        )
        let elapsedAoS = DispatchTime.now().uptimeNanoseconds - startAoS

        // Benchmark AoSoA layout
        var dataAoSoA = generateRandomData(n: n, d: d)
        var centroidsAoSoA = [Float](repeating: 0, count: k * d)

        let cfgAoSoA = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 256,
            epochs: 10,
            seed: 42,
            layout: .aosoaR,
            aosoaRegisterBlock: 4
        )

        let startAoSoA = DispatchTime.now().uptimeNanoseconds
        _ = kmeans_minibatch_f32(
            x: &dataAoSoA,
            n: Int64(n),
            d: d,
            kc: k,
            initCentroids: nil,
            cfg: cfgAoSoA,
            centroidsOut: &centroidsAoSoA
        )
        let elapsedAoSoA = DispatchTime.now().uptimeNanoseconds - startAoSoA

        let msAoS = Double(elapsedAoS) / 1_000_000.0
        let msAoSoA = Double(elapsedAoSoA) / 1_000_000.0
        let speedup = msAoS / msAoSoA

        print("  → AoS: \(String(format: "%.2f", msAoS))ms")
        print("  → AoSoA: \(String(format: "%.2f", msAoSoA))ms")
        print("  → Speedup: \(String(format: "%.2fx", speedup))")
    }

    /// Benchmark mini-batch k-means: Learning rate schedule comparison
    func testBench_MiniBatch_LearningRateComparison_10K_128D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        // Benchmark constant learning rate
        var dataConst = generateRandomData(n: n, d: d)
        var centroidsConst = [Float](repeating: 0, count: k * d)

        let cfgConst = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 256,
            epochs: 10,
            decay: 1.0,  // Constant learning rate
            seed: 42
        )

        let startConst = DispatchTime.now().uptimeNanoseconds
        _ = kmeans_minibatch_f32(
            x: &dataConst,
            n: Int64(n),
            d: d,
            kc: k,
            initCentroids: nil,
            cfg: cfgConst,
            centroidsOut: &centroidsConst
        )
        let elapsedConst = DispatchTime.now().uptimeNanoseconds - startConst

        // Benchmark EWMA learning rate
        var dataEWMA = generateRandomData(n: n, d: d)
        var centroidsEWMA = [Float](repeating: 0, count: k * d)

        let cfgEWMA = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: 256,
            epochs: 10,
            decay: 0.5,  // EWMA decay rate
            seed: 42
        )

        let startEWMA = DispatchTime.now().uptimeNanoseconds
        _ = kmeans_minibatch_f32(
            x: &dataEWMA,
            n: Int64(n),
            d: d,
            kc: k,
            initCentroids: nil,
            cfg: cfgEWMA,
            centroidsOut: &centroidsEWMA
        )
        let elapsedEWMA = DispatchTime.now().uptimeNanoseconds - startEWMA

        let msConst = Double(elapsedConst) / 1_000_000.0
        let msEWMA = Double(elapsedEWMA) / 1_000_000.0

        print("  → Constant LR: \(String(format: "%.2f", msConst))ms")
        print("  → EWMA LR: \(String(format: "%.2f", msEWMA))ms")
    }

    // MARK: - End-to-End Pipeline Benchmark

    /// Benchmark full pipeline: k-means++ seeding → mini-batch training
    func testBench_FullPipeline_10K_128D() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        var data = generateRandomData(n: n, d: d)

        measure {
            // Phase 1: K-means++ seeding
            var initCentroids = [Float](repeating: 0, count: k * d)
            _ = kmeansPlusPlusSeed(
                data: &data,
                count: n,
                dimension: d,
                k: k,
                config: KMeansSeedConfig(k: k, rngSeed: 42),
                centroidsOut: &initCentroids,
                chosenIndicesOut: nil
            )

            // Phase 2: Mini-batch k-means training
            var finalCentroids = initCentroids
            let cfg = KMeansMBConfig(
                algo: .lloydMiniBatch,
                batchSize: 256,
                epochs: 10,
                seed: 42
            )

            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: initCentroids,
                cfg: cfg,
                centroidsOut: &finalCentroids
            )
        }

        print("  → Full pipeline: seeding + training (10 epochs)")
    }

    // MARK: - Memory Allocation Benchmark

    /// Measure memory allocation overhead
    func testBench_MemoryAllocation() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let k = 100

        measure {
            // Allocate data structures
            var data = [Float](repeating: 0, count: n * d)
            var centroids = [Float](repeating: 0, count: k * d)
            var assignments = [Int32](repeating: 0, count: n)
            var stats = KMeansMBStats(
                epochsCompleted: 0, batchesProcessed: 0, rowsSeen: 0, emptiesRepaired: 0,
                inertiaPerEpoch: [], finalInertia: 0, timeInitSec: 0, timeTrainingSec: 0,
                timeAssignmentSec: 0, bytesRead: 0
            )

            // Initialize data
            for i in 0..<(n * d) {
                data[i] = Float.random(in: -1..<1)
            }

            // Quick training run
            let cfg = KMeansMBConfig(
                algo: .lloydMiniBatch,
                batchSize: 256,
                epochs: 1,
                seed: 42
            )

            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids,
                assignOut: &assignments,
                statsOut: &stats
            )
        }

        let totalBytes = (n * d + k * d) * MemoryLayout<Float>.stride +
                        n * MemoryLayout<Int32>.stride
        print("  → Total memory: \(totalBytes / 1024)KB")
    }

    // MARK: - Scalability Tests

    /// Test scalability with increasing dataset size
    func testBench_Scalability_DatasetSize() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let d = 128
        let k = 100
        let sizes = [1_000, 5_000, 10_000, 20_000, 50_000]

        print("\nScalability: Dataset Size")
        for n in sizes {
            var data = generateRandomData(n: n, d: d)
            var centroids = [Float](repeating: 0, count: k * d)

            let cfg = KMeansMBConfig(
                algo: .lloydMiniBatch,
                batchSize: 256,
                epochs: 5,
                seed: 42
            )

            let start = DispatchTime.now().uptimeNanoseconds
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            let ms = Double(elapsed) / 1_000_000.0
            let throughput = Double(n) / (ms / 1000.0)

            print("  n=\(n): \(String(format: "%.2f", ms))ms, \(String(format: "%.0f", throughput)) vec/s")
        }
    }

    /// Test scalability with increasing dimension
    func testBench_Scalability_Dimension() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let k = 100
        let dimensions = [32, 64, 128, 256, 512]

        print("\nScalability: Dimension")
        for d in dimensions {
            var data = generateRandomData(n: n, d: d)
            var centroids = [Float](repeating: 0, count: k * d)

            let cfg = KMeansMBConfig(
                algo: .lloydMiniBatch,
                batchSize: 256,
                epochs: 5,
                seed: 42
            )

            let start = DispatchTime.now().uptimeNanoseconds
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            let ms = Double(elapsed) / 1_000_000.0

            print("  d=\(d): \(String(format: "%.2f", ms))ms")
        }
    }

    /// Test scalability with increasing number of clusters
    func testBench_Scalability_Clusters() throws {
        throw XCTSkip("Performance benchmark - enable manually for profiling (can take several minutes)")
        let n = 10_000
        let d = 128
        let clusterCounts = [10, 50, 100, 200, 500]

        print("\nScalability: Number of Clusters")
        for k in clusterCounts {
            var data = generateRandomData(n: n, d: d)
            var centroids = [Float](repeating: 0, count: k * d)

            let cfg = KMeansMBConfig(
                algo: .lloydMiniBatch,
                batchSize: 256,
                epochs: 5,
                seed: 42
            )

            let start = DispatchTime.now().uptimeNanoseconds
            _ = kmeans_minibatch_f32(
                x: &data,
                n: Int64(n),
                d: d,
                kc: k,
                initCentroids: nil,
                cfg: cfg,
                centroidsOut: &centroids
            )
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            let ms = Double(elapsed) / 1_000_000.0

            print("  k=\(k): \(String(format: "%.2f", ms))ms")
        }
    }

    // MARK: - Helper Functions

    private func generateRandomData(n: Int, d: Int) -> [Float] {
        var data = [Float](repeating: 0, count: n * d)
        for i in 0..<(n * d) {
            data[i] = Float.random(in: -1..<1)
        }
        return data
    }
}
