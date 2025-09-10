import Foundation
import VectorIndex
import VectorCore


struct VectorIndexBenchmarks {
    static func main() async {
        let args = CommandLine.arguments.dropFirst()
        var config = CLIConfig()
        parseArgs(args: Array(args), into: &config)

        let bench = Benchmark(config: config)
        do {
            let result = try await bench.run()
            try output(result: result, to: config.output)
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
            default: break
            }
            i += 1
        }
    }

    static func output(result: BenchSuiteResult, to path: String?) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        if let path = path {
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        } else {
            if let s = String(data: data, encoding: .utf8) { print(s) }
        }
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
        try await idx.batchInsert(Array(zip(ids, data).map { ($0.0, $0.1, nil as [String:String]?) }))
        let build = t0.duration(to: clock.now)

        // Search timings and throughput
        var times: [Double] = []
        times.reserveCapacity(queries.count)
        var dummy: Int = 0
        for q in queries {
            let s0 = clock.now
            let res = try await idx.search(query: q, k: config.k, filter: nil)
            dummy += res.count
            let dt = s0.duration(to: clock.now)
            times.append(dt.seconds * 1000)
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
        try await idx.batchInsert(Array(zip(ids, data).map { ($0.0, $0.1, nil as [String:String]?) }))
        let build = t0.duration(to: clock.now)
        var times: [Double] = []
        var recs: [Double] = []
        times.reserveCapacity(queries.count)
        for q in queries {
            let truth = try await truthTopK(q: q, data: data, ids: ids)
            let s0 = clock.now
            let res = try await idx.search(query: q, k: config.k, filter: nil).map{ $0.id }
            let dt = s0.duration(to: clock.now)
            times.append(dt.seconds * 1000)
            recs.append(recall(k: config.k, truth: truth, approx: res))
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
        try await idx.batchInsert(Array(zip(ids, data).map { ($0.0, $0.1, nil as [String:String]?) }))
        let build = t0.duration(to: clock.now)
        let o0 = clock.now
        try await idx.optimize()
        let optimize = o0.duration(to: clock.now)

        var times: [Double] = []
        var recs: [Double] = []
        for q in queries {
            let truth = try await truthTopK(q: q, data: data, ids: ids)
            let s0 = clock.now
            let res = try await idx.search(query: q, k: config.k, filter: nil).map{ $0.id }
            let dt = s0.duration(to: clock.now)
            times.append(dt.seconds * 1000)
            recs.append(recall(k: config.k, truth: truth, approx: res))
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
