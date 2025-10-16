import Foundation

// Public models to interop with external VectorBench runner.
// Mirrors the schema VectorBench expects and remains stable for CLI output.

public enum ProgressFormat: String, Codable, Sendable { case none, json }

public struct RunFlags: Codable, Sendable {
    // Core
    public var index: String            // "flat" | "hnsw" | "ivf" | "all"
    public var n: Int
    public var q: Int
    public var dim: Int
    public var k: Int
    public var metric: String           // euclidean|dotProduct|cosine

    // HNSW
    public var m: Int?
    public var efc: Int?
    public var efs: Int?

    // IVF
    public var nlist: Int?
    public var nprobe: Int?

    // Misc
    public var seed: UInt64?
    public var abOnly: Bool?            // A/B only runs (VectorBench expects this)
    public var output: String?
    public var progressFormat: ProgressFormat?
    public var repeats: Int?

    public init(
        index: String = "all",
        n: Int = 1000,
        q: Int = 100,
        dim: Int = 128,
        k: Int = 10,
        metric: String = "euclidean",
        m: Int? = nil,
        efc: Int? = nil,
        efs: Int? = nil,
        nlist: Int? = nil,
        nprobe: Int? = nil,
        seed: UInt64? = nil,
        abOnly: Bool? = nil,
        output: String? = nil,
        progressFormat: ProgressFormat? = nil,
        repeats: Int? = nil
    ) {
        self.index = index
        self.n = n
        self.q = q
        self.dim = dim
        self.k = k
        self.metric = metric
        self.m = m
        self.efc = efc
        self.efs = efs
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.abOnly = abOnly
        self.output = output
        self.progressFormat = progressFormat
        self.repeats = repeats
    }
}

public struct BenchMetadata: Codable, Sendable {
    public var device: String
    public var os: String
    public var cpu: String?
    public var memoryGB: Int?
    public var timestamp: String

    public init(device: String, os: String, cpu: String? = nil, memoryGB: Int? = nil, timestamp: String = ISO8601DateFormatter().string(from: Date())) {
        self.device = device
        self.os = os
        self.cpu = cpu
        self.memoryGB = memoryGB
        self.timestamp = timestamp
    }
}

public struct BenchCase: Codable, Sendable {
    public var name: String               // e.g. Flat/HNSW/IVF
    public var params: [String:String]
    public var metric: String             // distance metric
    public var n: Int
    public var q: Int
    public var dim: Int
    public var k: Int

    // Timings
    public var buildSeconds: Double
    public var optimizeSeconds: Double
    public var searchAvgMs: Double
    public var searchP95Ms: Double
    public var throughputQps: Double

    // Quality
    public var recallAvg: Double

    public init(
        name: String,
        params: [String : String],
        metric: String,
        n: Int,
        q: Int,
        dim: Int,
        k: Int,
        buildSeconds: Double,
        optimizeSeconds: Double,
        searchAvgMs: Double,
        searchP95Ms: Double,
        throughputQps: Double,
        recallAvg: Double
    ) {
        self.name = name
        self.params = params
        self.metric = metric
        self.n = n
        self.q = q
        self.dim = dim
        self.k = k
        self.buildSeconds = buildSeconds
        self.optimizeSeconds = optimizeSeconds
        self.searchAvgMs = searchAvgMs
        self.searchP95Ms = searchP95Ms
        self.throughputQps = throughputQps
        self.recallAvg = recallAvg
    }

    // Minimal chart datapoint mapping that VectorBench can consume.
    public var chartDataPoints: [ChartDataPoint] {
        return [
            ChartDataPoint(label: "avg_ms", x: 0, y: searchAvgMs),
            ChartDataPoint(label: "p95_ms", x: 0, y: searchP95Ms),
            ChartDataPoint(label: "qps", x: 0, y: throughputQps),
            ChartDataPoint(label: "recall", x: 0, y: recallAvg)
        ]
    }
}

public struct BenchRun: Codable, Sendable {
    public var flags: RunFlags
    public var meta: BenchMetadata
    public var cases: [BenchCase]

    public init(flags: RunFlags, meta: BenchMetadata, cases: [BenchCase]) {
        self.flags = flags
        self.meta = meta
        self.cases = cases
    }
}

public struct ChartDataPoint: Codable, Sendable {
    public var label: String
    public var x: Double
    public var y: Double
    public init(label: String, x: Double, y: Double) {
        self.label = label
        self.x = x
        self.y = y
    }
}

// MARK: - Progress Events

public struct ProgressEvent: Codable, Sendable {
    public var event: String = "progress"
    public var phase: String     // insert/optimize/search
    public var suite: String     // Flat/HNSW/IVF
    public var completed: Int
    public var total: Int
    public var percent: Double { total == 0 ? 0 : (Double(completed) / Double(total)) * 100.0 }

    public init(phase: String, suite: String, completed: Int, total: Int) {
        self.phase = phase
        self.suite = suite
        self.completed = completed
        self.total = total
    }
}

@inline(__always)
public func emitProgress(_ format: ProgressFormat?, event: ProgressEvent) {
    guard format == .json else { return }
    if let data = try? JSONEncoder().encode(event), let line = String(data: data, encoding: .utf8) {
        // Print as a single JSON line
        fputs(line + "\n", stdout)
        fflush(stdout)
    }
}

