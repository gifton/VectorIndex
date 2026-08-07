// Performance Benchmarks for Kernel #39: Candidate Reservoir mode comparison
// Path: Tests/VectorIndexTests/ReservoirModeBenchmarks.swift
//
// Carried from the `.adaptive` enablement (Phase 2, Task 19): `.adaptive` was dead code
// (currentMode collapsed to its initial strategy at init, so pushBatch's `.adaptive` branch
// was unreachable). Now that it switches Block->Heap for real, this benchmark gives
// developers/researchers measured guidance for choosing `.heap` / `.block` / `.adaptive`
// (see the doc comment on `ReservoirOptions.mode`, which is populated from this table).
//
// Gate shape copied from IVFSelectBenchmarks: skipped unless RUN_BENCHMARKS=1, run in
// release for representative numbers:
//   RUN_BENCHMARKS=1 swift test -c release --filter ReservoirModeBenchmarks
//
// Informational only -- no assertions. Each cell pushes 100k (id, score) pairs in 1k-sized
// batches into a fresh reservoir and reports wall time (ns/op = elapsed / pushed count)
// plus the telemetry counters that explain *why* a mode is fast or slow in that cell.
import XCTest
@testable import VectorIndex

final class ReservoirModeBenchmarks: XCTestCase {
    private let enableBenchmarks: Bool = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"

    override func setUpWithError() throws {
        if !enableBenchmarks {
            throw XCTSkip("Benchmarks disabled by default. Set RUN_BENCHMARKS=1 to enable.")
        }
    }

    // MARK: - Stream shapes

    /// Naming matches the existing convention in CandidateReservoirTests
    /// (testAdaptiveSampledSwitchMatchesPureHeapResults): "descending" means the numeric
    /// score *value* decreases as the stream progresses, which for L2 (smaller-is-better)
    /// means later candidates are progressively BETTER.
    private enum StreamShape: String, CaseIterable {
        /// Scores decrease monotonically (later item is better for L2). Forces Heap to keep
        /// replacing its root (replaceRoot + sift-down) as better candidates keep arriving.
        case descending
        /// Scores increase monotonically (later item is worse for L2). Heap's tau stabilizes
        /// almost immediately (first C pushes), so nearly every later push is a cheap O(1)
        /// tau-reject.
        case ascending
        /// Seeded uniform-random scores; no ordering bias, representative of an unsorted scan.
        case random
    }

    /// Dependency-free, reproducible PRNG (SplitMix64) so the "random" shape is identical
    /// across runs without pulling in a third-party dependency.
    private struct SplitMix64 {
        private var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func nextFloat(in range: Range<Float>) -> Float {
            let unit = Float(next() >> 40) / Float(1 << 24) // [0, 1)
            return range.lowerBound + unit * (range.upperBound - range.lowerBound)
        }
    }

    private func makeStream(shape: StreamShape, n: Int, seed: UInt64) -> (ids: [Int64], scores: [Float]) {
        var ids = [Int64](repeating: 0, count: n)
        var scores = [Float](repeating: 0, count: n)
        switch shape {
        case .descending:
            for i in 0..<n {
                ids[i] = Int64(i)
                scores[i] = Float(n - i)
            }
        case .ascending:
            for i in 0..<n {
                ids[i] = Int64(i)
                scores[i] = Float(i)
            }
        case .random:
            var rng = SplitMix64(seed: seed)
            for i in 0..<n {
                ids[i] = Int64(i)
                scores[i] = rng.nextFloat(in: 0..<1_000_000)
            }
        }
        return (ids, scores)
    }

    /// Push `ids`/`scores` into `reservoir` in `batchSize`-sized chunks.
    private func runStream(
        _ reservoir: CandidateReservoir, ids: [Int64], scores: [Float], batchSize: Int
    ) {
        let n = ids.count
        ids.withUnsafeBufferPointer { idBuf in
            scores.withUnsafeBufferPointer { scoreBuf in
                var offset = 0
                while offset < n {
                    let count = min(batchSize, n - offset)
                    _ = reservoir.pushBatch(
                        ids: idBuf.baseAddress! + offset,
                        scores: scoreBuf.baseAddress! + offset,
                        count: count
                    )
                    offset += count
                }
            }
        }
    }

    // MARK: - Table row

    private struct Row {
        let capacity: Int
        let shape: String
        let mode: String
        let nsPerOp: Double
        let modeSwitches: Int64
        let prunes: Int64
        let rejectedTau: Int64
    }

    private func padRight(_ s: String, _ w: Int) -> String {
        s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
    }
    private func padLeft(_ s: String, _ w: Int) -> String {
        s.count >= w ? s : String(repeating: " ", count: w - s.count) + s
    }

    // MARK: - The benchmark

    /// C in {64, 1024} x shape in {descending, ascending, random} x mode in {.heap, .block,
    /// .adaptive}: push 100k scored ids (unique, 1k-sized batches) into a fresh reservoir per
    /// cell and report ns/op plus mode-relevant telemetry.
    func testModeComparisonTable() throws {
        let capacities = [64, 1024]
        let shapeOrder: [StreamShape] = [.descending, .ascending, .random]
        let modeOrder: [(name: String, mode: ReservoirMode)] = [
            ("heap", .heap), ("block", .block), ("adaptive", .adaptive)
        ]
        let n = 100_000
        let batchSize = 1_000
        let seed: UInt64 = 0xC0FFEE_1234_5678
        let metric: ReservoirMetric = .l2

        var rows: [Row] = []

        // Generate each shape's stream once (independent of capacity) so every capacity/mode
        // cell for a given shape sees byte-identical input -- an apples-to-apples comparison.
        var streams: [StreamShape: (ids: [Int64], scores: [Float])] = [:]
        for shape in shapeOrder {
            streams[shape] = makeStream(shape: shape, n: n, seed: seed)
        }

        // Single wall-clock samples on a shared dev machine are noisy enough to flip which
        // mode "wins" a cell (observed run-to-run swings >5x on some cells during development
        // of this benchmark, almost certainly scheduler/thermal noise rather than a real
        // algorithmic effect). Take the median of several trials per cell so the one captured
        // table (Step 4 still runs this suite exactly once) reports a stable number instead of
        // a coin flip. Telemetry (modeSwitches/prunes/rejectedTau) is deterministic given fixed
        // input + algorithm, so it is only recorded once (from the last trial).
        let trialsPerCell = 7

        for capacity in capacities {
            for shape in shapeOrder {
                let stream = streams[shape]!
                for (modeName, mode) in modeOrder {
                    let options = ReservoirOptions(
                        mode: mode,
                        reserveExtra: 0.10,
                        adaptiveThreshold: 0.75,
                        stableTies: true,
                        adaptiveInitialMode: .block,
                        telemetry: true
                    )
                    let reservoir = CandidateReservoir(capacity: capacity, metric: metric, options: options)

                    // Warm-up: one full pass, discarded, to pre-fault buffers before timing.
                    runStream(reservoir, ids: stream.ids, scores: stream.scores, batchSize: batchSize)
                    reservoir.reset() // clears size/tau/mode AND telemetry (telemetry: true)

                    var samplesNs: [Double] = []
                    samplesNs.reserveCapacity(trialsPerCell)
                    var lastTelemetry = reservoir.telemetry
                    for _ in 0..<trialsPerCell {
                        let start = DispatchTime.now().uptimeNanoseconds
                        runStream(reservoir, ids: stream.ids, scores: stream.scores, batchSize: batchSize)
                        let elapsedNs = DispatchTime.now().uptimeNanoseconds - start
                        samplesNs.append(Double(elapsedNs) / Double(n))
                        lastTelemetry = reservoir.telemetry
                        reservoir.reset()
                    }

                    let median = samplesNs.sorted()[trialsPerCell / 2]

                    rows.append(Row(
                        capacity: capacity, shape: shape.rawValue, mode: modeName,
                        nsPerOp: median,
                        modeSwitches: lastTelemetry.modeSwitches,
                        prunes: lastTelemetry.prunes,
                        rejectedTau: lastTelemetry.rejectedTau
                    ))
                }
            }
        }

        print("")
        print("[ReservoirModeBenchmarks] mode comparison: n=\(n) pushes, batch=\(batchSize), metric=l2, seed=0x\(String(seed, radix: 16)), ns/op=median of \(trialsPerCell) trials")
        print(
            padRight("C", 6) + padRight("shape", 11) + padRight("mode", 9)
            + padLeft("ns/op", 10) + padLeft("modeSwitches", 14) + padLeft("prunes", 9) + padLeft("rejectedTau", 13)
        )
        for row in rows {
            print(
                padRight(String(row.capacity), 6)
                + padRight(row.shape, 11)
                + padRight(row.mode, 9)
                + padLeft(String(format: "%.1f", row.nsPerOp), 10)
                + padLeft(String(row.modeSwitches), 14)
                + padLeft(String(row.prunes), 9)
                + padLeft(String(row.rejectedTau), 13)
            )
        }
        print("")
    }
}
