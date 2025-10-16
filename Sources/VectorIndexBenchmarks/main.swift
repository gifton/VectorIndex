import Foundation
import Darwin
import Dispatch
import VectorIndex
import VectorCore


struct VectorIndexBenchmarks {
    static func main() async {
        let args = CommandLine.arguments.dropFirst()
        var config = CLIConfig()
        parseArgs(args: Array(args), into: &config)

        if config.showHelp {
            printUsage()
            return
        }

        let bench = Benchmark(config: config)
        do {
            let result = try await bench.run()
            // If VectorBench format was requested, the runner already wrote output
            if config.outputFormat != "vb" {
                try output(result: result, to: config.output)
            }
        } catch {
            fputs("Benchmark failed: \(error)", stderr)
            exit(1)
        }
    }

    struct CLIConfig {
        var index: String = "all" // flat|hnsw|ivf|all
        var n: Int = 1000
        var q: Int = 100
        var dim: Int = 128
        var k: Int = 10
        var metric: SupportedDistanceMetric = .euclidean
        // HNSW
        var m: Int = 16
        var efc: Int = 200
        var efs: Int = 64
        // IVF
        var nlist: Int = 64
        var nprobe: Int = 4
        var seed: UInt64 = 42
        // IO
        var output: String? = nil // path to JSON; nil prints stdout
        var outputFormat: String = "simple" // simple | vb
        var progressFormat: ProgressFormat = .none // none | json
        var abOnly: Bool = false
        var progressInterval: Int = 1 // emit every N items (>=1)
        var showHelp: Bool = false
    }

    static func parseArgs(args: [String], into c: inout CLIConfig) {
        var i = 0
        func next() -> String? { guard i+1 < args.count else { return nil }; i += 1; return args[i] }
        while i < args.count {
            let a = args[i]
            switch a {
            case "--index": if let v = next() { c.index = v }
            case "--n": if let v = next(), let iv = Int(v) { c.n = iv }
            case "--q": if let v = next(), let iv = Int(v) { c.q = iv }
            case "--dim": if let v = next(), let iv = Int(v) { c.dim = iv }
            case "--k": if let v = next(), let iv = Int(v) { c.k = iv }
            case "--metric": if let v = next() { c.metric = SupportedDistanceMetric(rawValue: v) ?? .euclidean }
            case "--m": if let v = next(), let iv = Int(v) { c.m = iv }
            case "--efc": if let v = next(), let iv = Int(v) { c.efc = iv }
            case "--efs": if let v = next(), let iv = Int(v) { c.efs = iv }
            case "--nlist": if let v = next(), let iv = Int(v) { c.nlist = iv }
            case "--nprobe": if let v = next(), let iv = Int(v) { c.nprobe = iv }
            case "--seed": if let v = next(), let uv = UInt64(v) { c.seed = uv }
            case "--out": if let v = next() { c.output = v }
            case "--output-format": if let v = next() { c.outputFormat = v }
            case "--progress-format": if let v = next() { c.progressFormat = ProgressFormat(rawValue: v) ?? .none }
            case "--ab-only": c.abOnly = true
            case "--progress-interval": if let v = next(), let iv = Int(v), iv > 0 { c.progressInterval = iv }
            case "--help", "-h", "-?": c.showHelp = true
            default: break
            }
            i += 1
        }
    }

    static func printUsage() {
        let help = """
        Usage: VectorIndexBenchmarks [options]

          --index <flat|hnsw|ivf|all>     Which index to run (default: all)
          --n <int>                       Number of vectors (default: 1000)
          --q <int>                       Number of queries (default: 100)
          --dim <int>                     Vector dimension (default: 128)
          --k <int>                       Top-k (default: 10)
          --metric <euclidean|dotProduct|cosine>  Distance metric (default: euclidean)
          --m <int>                       HNSW M (default: 16)
          --efc <int>                     HNSW efConstruction (default: 200)
          --efs <int>                     HNSW efSearch (default: 64)
          --nlist <int>                   IVF nlist (default: 64)
          --nprobe <int>                  IVF nprobe (default: 4)
          --seed <uint64>                 Seed (default: 42)
          --out <path>                    Output file path (creates parent dirs)
          --output-format <simple|vb>     Output format (default: simple)
          --progress-format <none|json>   Progress streaming format (default: none)
          --progress-interval <int>       Emit progress every N items (default: 1)
          --ab-only                       Mark run as A/B only
          --help, -h                      Show this help and exit
        """
        print(help)
    }

    static func outputData(_ data: Data, to path: String?) throws {
        if let path = path {
            // Auto-create parent directories
            let url = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        } else {
            if let s = String(data: data, encoding: .utf8) { print(s) }
        }
    }

    static func output(result: BenchSuiteResult, to path: String?) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        try outputData(data, to: path)
    }
}

// MARK: - Benchmark Harness

struct Benchmark {
    let config: VectorIndexBenchmarks.CLIConfig

    func run() async throws -> BenchSuiteResult {
        let data = DataGen.generate(count: config.n, dim: config.dim, seed: 123)
        let queries = DataGen.generate(count: config.q, dim: config.dim, seed: 321)
        let ids = (0..<config.n).map { "id\($0)" }

        var results: [BenchResult] = []

        // Flat baseline always built for recall measurements
        let flatRes = try await benchFlat(data: data, ids: ids, queries: queries)
        results.append(flatRes)

        if config.index == "all" || config.index == "hnsw" {
            let hres = try await benchHNSW(data: data, ids: ids, queries: queries, baseline: flatRes)
            results.append(hres)
        }
        if config.index == "all" || config.index == "ivf" {
            let ivfRes = try await benchIVF(data: data, ids: ids, queries: queries, baseline: flatRes)
            results.append(ivfRes)
        }

        // If VectorBench format requested, emit BenchRun JSON now
        if config.outputFormat == "vb" {
            let run = makeBenchRun(results: results)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(run)
            try VectorIndexBenchmarks.outputData(data, to: config.output)
        }

        return BenchSuiteResult(config: encodeConfig(), results: results)
    }

    func encodeConfig() -> [String:String] {
        [
            "index": config.index,
            "n": String(config.n),
            "q": String(config.q),
            "dim": String(config.dim),
            "k": String(config.k),
            "metric": config.metric.rawValue,
            "m": String(config.m),
            "efc": String(config.efc),
            "efs": String(config.efs),
            "nlist": String(config.nlist),
            "nprobe": String(config.nprobe)
        ]
    }

    func benchFlat(data: [[Float]], ids: [String], queries: [[Float]]) async throws -> BenchResult {
        let clock = ContinuousClock()
        let idx = FlatIndex(dimension: config.dim, metric: config.metric)
        let t0 = clock.now
        // Insert with streaming progress (optional)
        for (i, pair) in zip(ids, data).enumerated() {
            try await idx.insert(id: pair.0, vector: pair.1, metadata: nil)
            if shouldEmitProgress(i + 1, total: config.n, interval: config.progressInterval) {
                emitProgress(config.progressFormat, event: ProgressEvent(phase: "insert", suite: "Flat", completed: i+1, total: config.n))
            }
        }
        let build = t0.duration(to: clock.now)

        // Search timings and throughput
        var times: [Double] = []
        times.reserveCapacity(queries.count)
        var dummy: Int = 0
        for (qi, q) in queries.enumerated() {
            let s0 = clock.now
            let res = try await idx.search(query: q, k: config.k, filter: nil)
            dummy += res.count
            let dt = s0.duration(to: clock.now)
            times.append(dt.seconds * 1000)
            if shouldEmitProgress(qi + 1, total: config.q, interval: config.progressInterval) {
                emitProgress(config.progressFormat, event: ProgressEvent(phase: "search", suite: "Flat", completed: qi+1, total: config.q))
            }
        }
        return BenchResult(
            indexType: "Flat",
            metric: config.metric.rawValue,
            n: config.n, q: config.q, dim: config.dim, k: config.k,
            params: [:],
            buildSeconds: build.seconds,
            optimizeSeconds: 0,
            searchAvgMs: avg(times),
            searchP95Ms: p(times, 0.95),
            recallAvg: 1.0,
            throughputQps: Double(config.q) / times.reduce(0,+) * 1000.0
        )
    }

    func benchHNSW(data: [[Float]], ids: [String], queries: [[Float]], baseline: BenchResult) async throws -> BenchResult {
        let clock = ContinuousClock()
        let idx = HNSWIndex(dimension: config.dim, metric: config.metric, config: .init(m: config.m, efConstruction: config.efc, efSearch: config.efs))
        let t0 = clock.now
        for (i, pair) in zip(ids, data).enumerated() {
            try await idx.insert(id: pair.0, vector: pair.1, metadata: nil)
            if shouldEmitProgress(i + 1, total: config.n, interval: config.progressInterval) {
                emitProgress(config.progressFormat, event: ProgressEvent(phase: "insert", suite: "HNSW", completed: i+1, total: config.n))
            }
        }
        let build = t0.duration(to: clock.now)
        var times: [Double] = []
        var recs: [Double] = []
        times.reserveCapacity(queries.count)
        for (qi, q) in queries.enumerated() {
            let truth = try await truthTopK(q: q, data: data, ids: ids)
            let s0 = clock.now
            let res = try await idx.search(query: q, k: config.k, filter: nil).map{ $0.id }
            let dt = s0.duration(to: clock.now)
            times.append(dt.seconds * 1000)
            recs.append(recall(k: config.k, truth: truth, approx: res))
            if shouldEmitProgress(qi + 1, total: config.q, interval: config.progressInterval) {
                emitProgress(config.progressFormat, event: ProgressEvent(phase: "search", suite: "HNSW", completed: qi+1, total: config.q))
            }
        }
        return BenchResult(
            indexType: "HNSW",
            metric: config.metric.rawValue,
            n: config.n, q: config.q, dim: config.dim, k: config.k,
            params: ["m": String(config.m), "efc": String(config.efc), "efs": String(config.efs)],
            buildSeconds: build.seconds,
            optimizeSeconds: 0,
            searchAvgMs: avg(times),
            searchP95Ms: p(times, 0.95),
            recallAvg: avg(recs),
            throughputQps: Double(config.q) / times.reduce(0,+) * 1000.0
        )
    }

    func benchIVF(data: [[Float]], ids: [String], queries: [[Float]], baseline: BenchResult) async throws -> BenchResult {
        let clock = ContinuousClock()
        let idx = IVFIndex(dimension: config.dim, metric: config.metric, config: .init(nlist: config.nlist, nprobe: config.nprobe))
        let t0 = clock.now
        for (i, pair) in zip(ids, data).enumerated() {
            try await idx.insert(id: pair.0, vector: pair.1, metadata: nil)
            if shouldEmitProgress(i + 1, total: config.n, interval: config.progressInterval) {
                emitProgress(config.progressFormat, event: ProgressEvent(phase: "insert", suite: "IVF", completed: i+1, total: config.n))
            }
        }
        let build = t0.duration(to: clock.now)
        let o0 = clock.now
        try await idx.optimize()
        emitProgress(config.progressFormat, event: ProgressEvent(phase: "optimize", suite: "IVF", completed: 1, total: 1))
        let optimize = o0.duration(to: clock.now)

        var times: [Double] = []
        var recs: [Double] = []
        for (qi, q) in queries.enumerated() {
            let truth = try await truthTopK(q: q, data: data, ids: ids)
            let s0 = clock.now
            let res = try await idx.search(query: q, k: config.k, filter: nil).map{ $0.id }
            let dt = s0.duration(to: clock.now)
            times.append(dt.seconds * 1000)
            recs.append(recall(k: config.k, truth: truth, approx: res))
            if shouldEmitProgress(qi + 1, total: config.q, interval: config.progressInterval) {
                emitProgress(config.progressFormat, event: ProgressEvent(phase: "search", suite: "IVF", completed: qi+1, total: config.q))
            }
        }
        return BenchResult(
            indexType: "IVF",
            metric: config.metric.rawValue,
            n: config.n, q: config.q, dim: config.dim, k: config.k,
            params: ["nlist": String(config.nlist), "nprobe": String(config.nprobe)],
            buildSeconds: build.seconds,
            optimizeSeconds: optimize.seconds,
            searchAvgMs: avg(times),
            searchP95Ms: p(times, 0.95),
            recallAvg: avg(recs),
            throughputQps: Double(config.q) / times.reduce(0,+) * 1000.0
        )
    }

    func truthTopK(q: [Float], data: [[Float]], ids: [String]) async throws -> [String] {
        var scored: [(String, Float)] = []
        scored.reserveCapacity(data.count)
        for (i, v) in data.enumerated() {
            var sum: Float = 0
            for j in 0..<q.count { let d = q[j] - v[j]; sum += d*d }
            scored.append((ids[i], sqrt(sum)))
        }
        scored.sort { $0.1 < $1.1 }
        return Array(scored.prefix(config.k).map { $0.0 })
    }

    // Build a VectorBench-compatible BenchRun envelope
    func makeBenchRun(results: [BenchResult]) -> BenchRun {
        let flags = RunFlags(
            index: config.index,
            n: config.n,
            q: config.q,
            dim: config.dim,
            k: config.k,
            metric: config.metric.rawValue,
            m: config.m,
            efc: config.efc,
            efs: config.efs,
            nlist: config.nlist,
            nprobe: config.nprobe,
            seed: config.seed,
            abOnly: config.abOnly,
            output: config.output,
            progressFormat: config.progressFormat,
            repeats: nil
        )

        let meta = BenchMetadata(
            device: HostInfo.machineModel,
            os: HostInfo.osVersion,
            cpu: HostInfo.cpuBrand,
            memoryGB: HostInfo.memoryGB
        )

        let cases: [BenchCase] = results.map { r in
            BenchCase(
                name: r.indexType,
                params: r.params,
                metric: r.metric,
                n: r.n,
                q: r.q,
                dim: r.dim,
                k: r.k,
                buildSeconds: r.buildSeconds,
                optimizeSeconds: r.optimizeSeconds,
                searchAvgMs: r.searchAvgMs,
                searchP95Ms: r.searchP95Ms,
                throughputQps: r.throughputQps,
                recallAvg: r.recallAvg
            )
        }
        return BenchRun(flags: flags, meta: meta, cases: cases)
    }
}

// MARK: - Result Types
struct BenchSuiteResult: Codable { let config: [String:String]; let results: [BenchResult] }
struct BenchResult: Codable {
    let indexType: String
    let metric: String
    let n: Int
    let q: Int
    let dim: Int
    let k: Int
    let params: [String:String]
    let buildSeconds: Double
    let optimizeSeconds: Double
    let searchAvgMs: Double
    let searchP95Ms: Double
    let recallAvg: Double
    let throughputQps: Double
}

// MARK: - Utilities
struct DataGen {
    struct LCG { var s: UInt64; mutating func next()->UInt64{ s = 2862933555777941757 &* s &+ 3037000493; return s }; mutating func f()->Float{ Float(next()>>11)/Float(1<<53) } }
    static func generate(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = LCG(s: seed)
        var arr: [[Float]] = []
        arr.reserveCapacity(count)
        for _ in 0..<count {
            var v = (0..<dim).map { _ in rng.f()*2-1 }
            let norm = sqrt(v.reduce(0){$0+$1*$1})
            if norm > 0 { v = v.map{ $0/norm } }
            arr.append(v)
        }
        return arr
    }
}

extension Duration { var seconds: Double { Double(self.components.seconds) + Double(self.components.attoseconds) / 1e18 } }
func avg(_ xs: [Double]) -> Double { xs.isEmpty ? 0 : xs.reduce(0,+)/Double(xs.count) }
func p(_ xs: [Double], _ q: Double) -> Double { let s = xs.sorted(); if s.isEmpty { return 0 }; let idx = Int(Double(s.count-1)*q); return s[idx] }
func recall(k: Int, truth: [String], approx: [String]) -> Double { let t = Set(truth.prefix(k)); let a = Set(approx.prefix(k)); return Double(t.intersection(a).count)/Double(k) }
@inline(__always) func shouldEmitProgress(_ completed: Int, total: Int, interval: Int) -> Bool {
    if interval <= 1 { return true }
    if completed == total { return true }
    return (completed % interval) == 0
}

// MARK: - Host Info for metadata
enum HostInfo {
    static var machineModel: String {
        #if os(macOS)
        var size: size_t = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(1, Int(size)))
        sysctlbyname("hw.model", &buf, &size, nil, 0)
        let end = (buf.firstIndex(of: 0) ?? buf.count)
        return buf.withUnsafeBytes { raw in
            let slice = raw[..<end]
            return String(data: Data(slice), encoding: .utf8) ?? "unknown"
        }
        #else
        return "unknown"
        #endif
    }
    static var osVersion: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
    static var cpuBrand: String {
        #if os(macOS)
        var size: size_t = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        if size == 0 { return "unknown" }
        var buf = [CChar](repeating: 0, count: Int(size))
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        let end = (buf.firstIndex(of: 0) ?? buf.count)
        return buf.withUnsafeBytes { raw in
            let slice = raw[..<end]
            return String(data: Data(slice), encoding: .utf8) ?? "unknown"
        }
        #else
        return "unknown"
        #endif
    }
    static var memoryGB: Int {
        let bytes = ProcessInfo.processInfo.physicalMemory
        return Int((bytes + (1<<30)-1) / (1<<30))
    }
}

// Kick off async main
private func _start() {
    Task {
        await VectorIndexBenchmarks.main()
        exit(0)
    }
    dispatchMain()
}

_start()
