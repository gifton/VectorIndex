import XCTest
import Dispatch
@testable import VectorIndex

/// P4/P5 gate instrument: mmap append throughput vs commit count.
/// Enabled with RUN_BENCHMARKS=1; writes JSON to $MMAP_BENCH_OUT if set.
final class MmapAppendBenchmark: XCTestCase {
    override func setUpWithError() throws {
        if ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != "1" {
            throw XCTSkip("Benchmarks disabled by default. Set RUN_BENCHMARKS=1 to enable.")
        }
    }

    /// Builds and opens a minimal durable PQ8 container, mirroring the fixture setup in
    /// Kernel30AppendTests.testDurablePQ8AppendWithRemap (same builder call, same option
    /// values apart from capacities — see below).
    ///
    /// Deviation from the fixture's literal `idCap: 32, payloadCap: 4`: those work in the
    /// fixture only because its one-shot test appends just 10 records, never approaching
    /// either capacity's *physical* byte ceiling. At this benchmark's scale that combination
    /// hits two real, pre-existing limitations of the durable-append growth path in
    /// VIndexMmap.swift (not touched here — Sources/VectorIndex/ is out of scope for this task):
    ///   1. `ensureFileCapacity` refuses to grow any section that is not the last one by file
    ///      offset; this builder always places Codes/Vecs after IDs, so the IDs section can
    ///      never grow past its initial capacity at all.
    ///   2. A section's on-disk TOC `size` field is written once at container-creation time
    ///      (`writeTOCEntry` in VIndexContainerBuilder.swift) and is never rewritten by the
    ///      growth path — `ensureFileCapacity` only extends the physical file and remaps, so
    ///      once total bytes exceed the *original* page-rounded section size, every subsequent
    ///      `mmap_append_commit` fails its bounds check regardless of how much larger the file
    ///      has grown. In practice this caps Codes growth at one page (1024 records at m=16
    ///      on a 16 KB-page host) — far below the sweep sizes this benchmark needs.
    /// Both `idCap` and `payloadCap` are therefore sized to the run's full `commits * batch`
    /// record count up front, so neither section is ever asked to grow. This does not weaken
    /// the measurement: `mmap_append_commit` calls `updateSectionCRC` over the *entire current
    /// section* on every single commit (not just the delta), so cost-per-commit already scales
    /// with total section size and the O(commits²) signal this instrument exists to capture is
    /// present whether or not a growth event ever fires.
    private func makeDurableContainer(path: String, kc: Int, m: Int, idCap: Int, payloadCap: Int) throws -> IndexMmap {
        try VIndexContainerBuilder.createMinimalContainer(
            path: path, format: .pq8, k_c: kc, m: m, d: 0,
            idBits: 64, group: 4, idCap: idCap, payloadCap: payloadCap
        )
    }

    func testAppendThroughputSweep() throws {
        let m = 16              // PQ subspaces → 16 code bytes/record
        let batch = 32          // records per commit
        let sweeps = [1_000, 2_000, 4_000, 8_000]   // commits per run
        var points: [[String: Any]] = []
        for commits in sweeps {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("mmap-bench-\(commits)-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let path = dir.appendingPathComponent("bench.vindex").path
            let mmap = try makeDurableContainer(path: path, kc: 1, m: m,
                                                idCap: commits * batch, payloadCap: commits * batch)
            var ids = [UInt64](repeating: 0, count: batch)
            var codes = [UInt8](repeating: 0, count: batch * m)
            let t0 = DispatchTime.now()
            for c in 0..<commits {
                for r in 0..<batch { ids[r] = UInt64(c * batch + r) }
                for r in 0..<(batch * m) { codes[r] = UInt8((c + r) & 0xFF) }
                let res = try mmap.mmap_append_begin(listID: 0, addLen: batch)
                try ids.withUnsafeBytes { ib in
                    try codes.withUnsafeBytes { cb in
                        try mmap.mmap_append_commit(res, idsSrc: ib.baseAddress,
                                                    codesSrc: cb.baseAddress, vecsSrc: nil)
                    }
                }
            }
            let sec = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
            try mmap.close()
            points.append(["commits": commits, "seconds": sec,
                           "commitsPerSec": Double(commits) / sec])
            print("mmap-append commits=\(commits) sec=\(sec) rate=\(Double(commits)/sec)/s")
        }
        if let out = ProcessInfo.processInfo.environment["MMAP_BENCH_OUT"] {
            let payload: [String: Any] = ["benchmark": "mmap_append", "batch": batch, "m": m,
                                          "points": points]
            let data = try JSONSerialization.data(withJSONObject: payload,
                                                  options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: out))
        }
    }
}
