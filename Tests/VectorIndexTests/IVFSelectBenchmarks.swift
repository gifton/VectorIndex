// Performance Benchmarks for Kernel #29: IVF List Selection
// Target: Apple M2 Max, 1 P-core
import XCTest
import Accelerate
@testable import VectorIndex

final class IVFSelectBenchmarks: XCTestCase {
    private let enableBenchmarks: Bool = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"

    override func setUpWithError() throws {
        if !enableBenchmarks {
            throw XCTSkip("Benchmarks disabled by default. Set RUN_BENCHMARKS=1 to enable.")
        }
    }

    // MARK: - Performance Targets from Spec

    /// Test 1: Small kc (1K centroids, d=1024, nprobe=10)
    /// Target: 20 μs @ M2 Max 1 P-core
    func testPerformance_SmallKc() throws {
        let kc = 1_000
        let d = 1024
        let nprobe = 10
        let iterations = 1_000

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Warmup
        for _ in 0..<10 {
            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(),
                listIDsOut: &ids,
                listScoresOut: &scores
            )
        }

        // Measure
        measure {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Benchmark] Small kc
        Config: kc=\(kc), d=\(d), nprobe=\(nprobe)
        Target: 20 μs per query
        Note: Divide reported time by \(iterations) to get per-query latency
        """)
    }

    /// Test 2: Medium kc (10K centroids, d=1024, nprobe=50)
    /// Target: 50 μs @ M2 Max 1 P-core
    func testPerformance_MediumKc() throws {
        let kc = 10_000
        let d = 1024
        let nprobe = 50
        let iterations = 500

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Precompute centroid norms for dot trick
        let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Warmup
        for _ in 0..<5 {
            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                listIDsOut: &ids,
                listScoresOut: &scores
            )
        }

        // Measure
        measure {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Benchmark] Medium kc
        Config: kc=\(kc), d=\(d), nprobe=\(nprobe)
        Target: 50 μs per query
        Note: Divide reported time by \(iterations) to get per-query latency
        """)
    }

    /// Test 3: Large kc (100K centroids, d=1024, nprobe=100)
    /// Target: 500 μs @ M2 Max 1 P-core
    func testPerformance_LargeKc() throws {
        let kc = 100_000
        let d = 1024
        let nprobe = 100
        let iterations = 100

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Warmup
        for _ in 0..<3 {
            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                listIDsOut: &ids,
                listScoresOut: &scores
            )
        }

        // Measure
        measure {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Benchmark] Large kc
        Config: kc=\(kc), d=\(d), nprobe=\(nprobe)
        Target: 500 μs per query
        Note: Divide reported time by \(iterations) to get per-query latency
        """)
    }

    /// Test 4: Beam search overhead (10K centroids, beam=100)
    /// Target: 150 μs (3× standard selection)
    func testPerformance_BeamSearch() throws {
        let kc = 10_000
        let d = 1024
        let nprobe = 50
        let beamWidth = 100
        let iterations = 200

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        // Build k-NN graph (random for benchmark)
        let deg = 32
        let knnGraph = buildRandomKnnGraph(kc: kc, deg: deg)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Warmup
        for _ in 0..<5 {
            ivf_select_beam_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                knnGraph: knnGraph, deg: deg,
                metric: .l2, nprobe: nprobe, beamWidth: beamWidth,
                opts: IVFSelectOpts(),
                listIDsOut: &ids,
                listScoresOut: &scores
            )
        }

        // Measure
        measure {
            for _ in 0..<iterations {
                ivf_select_beam_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    knnGraph: knnGraph, deg: deg,
                    metric: .l2, nprobe: nprobe, beamWidth: beamWidth,
                    opts: IVFSelectOpts(),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Benchmark] Beam search
        Config: kc=\(kc), d=\(d), nprobe=\(nprobe), beam=\(beamWidth)
        Target: 150 μs per query (3× overhead vs standard)
        Note: Divide reported time by \(iterations) to get per-query latency
        """)
    }

    // MARK: - Batch Throughput

    /// Test 5: Batch throughput (b=100, kc=10K)
    /// Target: 20K queries/sec (batch parallelism)
    func testPerformance_BatchThroughput() throws {
        let b = 100
        let d = 1024
        let kc = 10_000
        let nprobe = 50
        let iterations = 10

        let Q = randomVectors(n: b, d: d)
        let centroids = randomVectors(n: kc, d: d)

        let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

        var ids = [Int32](repeating: -1, count: b * nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: b * nprobe)

        // Warmup
        for _ in 0..<3 {
            ivf_select_nprobe_batch_f32(
                Q: Q, b: b, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                listIDsOut: &ids,
                listScoresOut: &scores
            )
        }

        // Measure
        measure {
            for _ in 0..<iterations {
                ivf_select_nprobe_batch_f32(
                    Q: Q, b: b, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Benchmark] Batch throughput
        Config: b=\(b), kc=\(kc), d=\(d), nprobe=\(nprobe)
        Target: 20K queries/sec (50 μs per query amortized)
        Note: Divide reported time by (\(iterations) × \(b)) for per-query latency
        """)
    }

    // MARK: - Metric Comparisons

    /// Test 6: L2 vs IP vs Cosine performance
    func testPerformance_MetricComparison() throws {
        let kc = 10_000
        let d = 1024
        let nprobe = 50
        let iterations = 500

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)

        let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)
        let centroidInvNorms = centroidNorms.map { 1.0 / sqrt($0) }

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Test L2
        measure(metrics: [XCTClockMetric()]) {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("[Metric Comparison] L2 with dot-trick completed")

        // Test IP
        measure(metrics: [XCTClockMetric()]) {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .ip, nprobe: nprobe,
                    opts: IVFSelectOpts(),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("[Metric Comparison] IP completed")

        // Test Cosine
        measure(metrics: [XCTClockMetric()]) {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .cosine, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidInvNorms: centroidInvNorms),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("[Metric Comparison] Cosine with precomputed inv norms completed")
    }

    // MARK: - Scaling Tests

    /// Test 7: Scaling with kc
    func testPerformance_ScalingKc() throws {
        let d = 1024
        let nprobe = 50
        let kcValues = [1_000, 5_000, 10_000, 20_000, 50_000]

        for kc in kcValues {
            let q = randomVector(d: d)
            let centroids = randomVectors(n: kc, d: d)
            let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

            var ids = [Int32](repeating: -1, count: nprobe)
            var scores: [Float]? = [Float](repeating: 0, count: nprobe)

            let iterations = max(10, 10_000 / kc)

            // Warmup
            for _ in 0..<3 {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }

            let start = Date()
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
            let elapsed = Date().timeIntervalSince(start)
            let perQuery = (elapsed / Double(iterations)) * 1_000_000  // Convert to μs

            print("kc=\(kc): \(String(format: "%.1f", perQuery)) μs per query")
        }
    }

    /// Test 8: Scaling with dimension
    func testPerformance_ScalingDimension() throws {
        let kc = 10_000
        let nprobe = 50
        let dValues = [128, 256, 512, 1024, 2048]

        for d in dValues {
            let q = randomVector(d: d)
            let centroids = randomVectors(n: kc, d: d)
            let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

            var ids = [Int32](repeating: -1, count: nprobe)
            var scores: [Float]? = [Float](repeating: 0, count: nprobe)

            let iterations = max(10, 500_000 / (kc * d))

            // Warmup
            for _ in 0..<3 {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }

            let start = Date()
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
            let elapsed = Date().timeIntervalSince(start)
            let perQuery = (elapsed / Double(iterations)) * 1_000_000

            print("d=\(d): \(String(format: "%.1f", perQuery)) μs per query")
        }
    }

    // MARK: - Memory Efficiency

    /// Test 9: Memory pooling effectiveness
    func testPerformance_MemoryPooling() throws {
        let kc = 10_000
        let d = 1024
        let nprobe = 50
        let iterations = 1_000

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)
        let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Measure with pooling (default)
        measure(metrics: [XCTMemoryMetric(), XCTClockMetric()]) {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(centroidNorms: centroidNorms, useDotTrick: true),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Memory Pooling] Completed \(iterations) queries
        Expected: Low allocation count due to buffer pooling
        """)
    }

    // MARK: - Disabled Lists Performance

    /// Test 10: Performance impact of disabled lists
    func testPerformance_DisabledLists() throws {
        let kc = 10_000
        let d = 1024
        let nprobe = 50
        let iterations = 500

        let q = randomVector(d: d)
        let centroids = randomVectors(n: kc, d: d)
        let centroidNorms = computeCentroidNorms(centroids: centroids, kc: kc, d: d)

        // Disable 50% of centroids
        let wordCount = (kc + 63) / 64
        var disabledMask = [UInt64](repeating: 0, count: wordCount)
        for i in stride(from: 0, to: kc, by: 2) {
            let word = i / 64
            let bit = i % 64
            disabledMask[word] |= (1 &<< UInt64(bit))
        }

        var ids = [Int32](repeating: -1, count: nprobe)
        var scores: [Float]? = [Float](repeating: 0, count: nprobe)

        // Warmup
        for _ in 0..<5 {
            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(
                    disabledLists: disabledMask,
                    centroidNorms: centroidNorms,
                    useDotTrick: true
                ),
                listIDsOut: &ids,
                listScoresOut: &scores
            )
        }

        // Measure
        measure {
            for _ in 0..<iterations {
                ivf_select_nprobe_f32(
                    q: q, d: d, centroids: centroids, kc: kc,
                    metric: .l2, nprobe: nprobe,
                    opts: IVFSelectOpts(
                        disabledLists: disabledMask,
                        centroidNorms: centroidNorms,
                        useDotTrick: true
                    ),
                    listIDsOut: &ids,
                    listScoresOut: &scores
                )
            }
        }

        print("""
        [Disabled Lists] Completed with 50% masked
        Expected overhead: <10% vs unmasked
        """)
    }

    // MARK: - Helper Functions

    private func randomVector(d: Int) -> [Float] {
        (0..<d).map { _ in Float.random(in: -1...1) }
    }

    private func randomVectors(n: Int, d: Int) -> [Float] {
        (0..<(n * d)).map { _ in Float.random(in: -1...1) }
    }

    private func computeCentroidNorms(centroids: [Float], kc: Int, d: Int) -> [Float] {
        (0..<kc).map { i in
            let base = i * d
            var normSq: Float = 0
            centroids[base..<(base+d)].withUnsafeBufferPointer { ptr in
                vDSP_svesq(ptr.baseAddress!, 1, &normSq, vDSP_Length(d))
            }
            return normSq
        }
    }

    private func buildRandomKnnGraph(kc: Int, deg: Int) -> [Int32] {
        var graph = [Int32]()
        graph.reserveCapacity(kc * deg)

        for _ in 0..<kc {
            for _ in 0..<deg {
                graph.append(Int32.random(in: 0..<Int32(kc)))
            }
        }

        return graph
    }
}
