// Sources/VectorIndex/Operations/RangeQuery/RangeQuery.swift
//
// Kernel #07: Range/Threshold Query
// CPU-bound implementation with early-exit for L2, generic path for IP/Cosine,
// mask-only mode, and ADC (u8/u4) variants.
//
// Dependencies:
// - Kernel #01 (L2 Sqr) for L2 distance scoring
// - Kernel #02 (Inner Product) for IP scoring
// - Kernel #03 (Cosine) for cosine similarity scoring
// - Kernel #08 (ID Filter) for allow/deny filtering
// - Kernel #32 (Visited Set) for deduplication
// - Kernel #39 (Candidate Reservoir) for reservoir sampling
//
// Thread-safety: Query-local state; safe for concurrent queries

import Foundation

// MARK: - Public Metric & Threshold Semantics

public enum DistanceMetric {
    case l2
    case innerProduct
    case cosine

    @inlinable public var higherIsBetter: Bool {
        switch self {
        case .l2: return false
        case .innerProduct, .cosine: return true
        }
    }
}

/// Threshold comparison mode based on metric
public enum ThresholdMode {
    case l2LessOrEqual          // keep if ‖q-x‖ ≤ τ
    case ipGreaterOrEqual       // keep if ⟨q,x⟩ ≥ τ
    case cosineGreaterOrEqual   // keep if cos(q,x) ≥ τ

    @inlinable
    public init(metric: DistanceMetric) {
        switch metric {
        case .l2:           self = .l2LessOrEqual
        case .innerProduct: self = .ipGreaterOrEqual
        case .cosine:       self = .cosineGreaterOrEqual
        }
    }

    @inlinable
    public func passes(score: Float, threshold: Float) -> Bool {
        switch self {
        case .l2LessOrEqual:         return score <= threshold
        case .ipGreaterOrEqual:      return score >= threshold
        case .cosineGreaterOrEqual:  return score >= threshold
        }
    }
}

// MARK: - Early Exit Strategy

public enum EarlyExitStrategy: Sendable { case auto, on, off }

// MARK: - Output Mode

public enum RangeOutputMode: Sendable { case compacted, mask, reservoir }

// MARK: - Reservoir Adapter

/// Adapts CandidateReservoir's batch API to single-insert interface
public final class ReservoirAdapter {
    @usableFromInline internal let reservoir: CandidateReservoir
    @usableFromInline internal var batchIDs: [Int64] = []
    @usableFromInline internal var batchScores: [Float] = []
    @usableFromInline internal let batchSize: Int

    public init(reservoir: CandidateReservoir, batchSize: Int = 64) {
        self.reservoir = reservoir
        self.batchSize = batchSize
        self.batchIDs.reserveCapacity(batchSize)
        self.batchScores.reserveCapacity(batchSize)
    }

    @inlinable
    public func insert(id: Int64, score: Float) {
        batchIDs.append(id)
        batchScores.append(score)

        if batchIDs.count >= batchSize {
            flush()
        }
    }

    @inlinable
    public func flush() {
        guard !batchIDs.isEmpty else { return }

        batchIDs.withUnsafeBufferPointer { ids in
            batchScores.withUnsafeBufferPointer { scores in
                reservoir.pushBatch(
                    ids: ids.baseAddress!,
                    scores: scores.baseAddress!,
                    count: ids.count,
                    visited: nil  // visited set handled externally
                )
            }
        }

        batchIDs.removeAll(keepingCapacity: true)
        batchScores.removeAll(keepingCapacity: true)
    }

    @inlinable
    public var count: Int {
        flush()
        return reservoir.count
    }

    deinit {
        flush()
    }
}

// MARK: - Configuration

public struct RangeScanConfig: @unchecked Sendable {
    public let earlyExit: EarlyExitStrategy
    public let outputScores: Bool
    public let idFilter: IDFilterOverlay?
    public let visitedSet: (any VisitedSet)?
    public let reservoir: ReservoirAdapter?
    public let outputMode: RangeOutputMode
    public let tileSize: Int
    public let enableTelemetry: Bool

    public init(
        earlyExit: EarlyExitStrategy = .auto,
        outputScores: Bool = true,
        idFilter: IDFilterOverlay? = nil,
        visitedSet: (any VisitedSet)? = nil,
        reservoir: ReservoirAdapter? = nil,
        outputMode: RangeOutputMode = .compacted,
        tileSize: Int = 1024,
        enableTelemetry: Bool = false
    ) {
        self.earlyExit = earlyExit
        self.outputScores = outputScores
        self.idFilter = idFilter
        self.visitedSet = visitedSet
        self.reservoir = reservoir
        self.outputMode = outputMode
        self.tileSize = max(1, tileSize)
        self.enableTelemetry = enableTelemetry
    }

    public static let `default` = RangeScanConfig()
}

// MARK: - Telemetry

public struct RangeScanTelemetry {
    public let metric: DistanceMetric
    public let threshold: Float
    public let vectorsScanned: Int
    public let vectorsKept: Int
    public let usedEarlyExit: Bool
    public let earlyExitHits: Int       // rows pruned early
    public let usedADCPath: Bool
    public let bytesScored: Int         // flat-vector bytes
    public let bytesCodes: Int          // ADC code bytes
    public let executionTimeNanos: UInt64

    public init(
        metric: DistanceMetric,
        threshold: Float,
        vectorsScanned: Int,
        vectorsKept: Int,
        usedEarlyExit: Bool,
        earlyExitHits: Int,
        usedADCPath: Bool,
        bytesScored: Int,
        bytesCodes: Int,
        executionTimeNanos: UInt64
    ) {
        self.metric = metric
        self.threshold = threshold
        self.vectorsScanned = vectorsScanned
        self.vectorsKept = vectorsKept
        self.usedEarlyExit = usedEarlyExit
        self.earlyExitHits = earlyExitHits
        self.usedADCPath = usedADCPath
        self.bytesScored = bytesScored
        self.bytesCodes = bytesCodes
        self.executionTimeNanos = executionTimeNanos
    }

    public var selectivityPercent: Double {
        guard vectorsScanned > 0 else { return 0 }
        return (Double(vectorsKept) / Double(vectorsScanned)) * 100.0
    }

    public var earlyExitEfficiency: Double {
        guard vectorsScanned > 0 else { return 0 }
        return Double(earlyExitHits) / Double(vectorsScanned)
    }
}

// Lightweight recorder hook — replace with your Telemetry (#46) integration.
@usableFromInline
@inline(__always) internal func recordTelemetry(_ t: RangeScanTelemetry) {
    _ = t  // TODO: Wire to global telemetry
}

// MARK: - Public API

/// Core range-query over flat vectors. Early-exit supported for L2.
/// - Returns: number of results written (≤ maxOut). For `reservoir` mode, returns number inserted.
@inlinable
public func rangeScanBlock(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig = .default
) -> Int {
    precondition(n >= 0 && d > 0 && maxOut >= 0)

    // Decide whether to use early-exit (only for L2)
    let useEarlyExit: Bool = {
        switch (metric, config.earlyExit) {
        case (.l2, .on):   return true
        case (.l2, .off):  return false
        case (.l2, .auto):
            let typical = estimateTypicalL2Distance(dimension: d)
            return threshold < 0.2 * typical
        default:
            return false
        }
    }()

    // Early-out when no room to store results and no reservoir configured
    let usingReservoir = (config.reservoir != nil)
    if maxOut == 0 && !usingReservoir {
        return 0
    }

    let t0 = DispatchTime.now().uptimeNanoseconds

    let kept: Int
    var earlyPruned = 0

    if useEarlyExit {
        var eeHits = 0
        kept = rangeScanL2_earlyExit(
            query: query,
            database: database,
            ids: ids,
            vectorCount: n,
            dimension: d,
            l2Threshold: threshold,
            idsOut: idsOut,
            scoresOut: scoresOut,
            maxOut: maxOut,
            config: config,
            earlyExitHitsOut: &eeHits
        )
        earlyPruned = eeHits
    } else {
        kept = rangeScanGeneric(
            query: query,
            database: database,
            ids: ids,
            vectorCount: n,
            dimension: d,
            metric: metric,
            threshold: threshold,
            idsOut: idsOut,
            scoresOut: scoresOut,
            maxOut: maxOut,
            config: config
        )
    }

    // Telemetry
    if config.enableTelemetry {
        let t1 = DispatchTime.now().uptimeNanoseconds
        let bytesScored = n * d * MemoryLayout<Float>.stride
        let telem = RangeScanTelemetry(
            metric: metric,
            threshold: threshold,
            vectorsScanned: n,
            vectorsKept: kept,
            usedEarlyExit: useEarlyExit,
            earlyExitHits: earlyPruned,
            usedADCPath: false,
            bytesScored: bytesScored,
            bytesCodes: 0,
            executionTimeNanos: t1 - t0
        )
        recordTelemetry(telem)
    }

    return kept
}

// MARK: - Mask-Only Variant

/// Generate a bit-mask (1=keep, 0=drop) using the same threshold semantics.
/// - Returns: number of set bits (kept vectors).
@inlinable
public func rangeMaskBlock(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    threshold: Float,
    mask: UnsafeMutablePointer<UInt8>,
    config: RangeScanConfig = .default
) -> Int {
    precondition(n >= 0 && d > 0)

    let mode = ThresholdMode(metric: metric)
    var kept = 0
    let tileSize = max(1, config.tileSize)

    var tileScores = [Float](repeating: 0, count: tileSize)

    for tileStart in stride(from: 0, to: n, by: tileSize) {
        let tileEnd = min(tileStart + tileSize, n)
        let m = tileEnd - tileStart

        // Score tile
        tileScores.withUnsafeMutableBufferPointer { outPtr in
            scoreBlock(
                query: query,
                database: database + tileStart * d,
                vectorCount: m,
                dimension: d,
                metric: metric,
                scores: outPtr.baseAddress!
            )
        }

        // Threshold + filters
        for i in 0..<m {
            let global = tileStart + i

            // ID filter
            if let f = config.idFilter {
                let id = ids?[global] ?? Int64(global)
                if !f.test(id: id) { mask[global] = 0; continue }
            }

            // threshold check
            let pass = mode.passes(score: tileScores[i], threshold: threshold)

            // visited set
            if pass, let vs = config.visitedSet {
                let id = ids?[global] ?? Int64(global)
                if !vs.testAndSet(id: id) {
                    mask[global] = 0
                    continue
                }
            }

            if pass {
                mask[global] = 1
                kept += 1
            } else {
                mask[global] = 0
            }
        }
    }

    return kept
}

// MARK: - ADC/PQ Range Scan

/// Range query over quantized codes (u8) using LUT (ADC). Keeps rows with score ≤ τ (L2-style).
@inlinable
public func rangeScanADC_u8(
    codes: UnsafePointer<UInt8>,
    vectorCount n: Int,
    subvectorCount m: Int,
    codebookSize ks: Int,
    lut: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig = .default
) -> Int {
    precondition(ks > 0 && m >= 1 && n >= 0)
    let usingReservoir = (config.reservoir != nil)
    if maxOut == 0 && !usingReservoir { return 0 }

    let t0 = DispatchTime.now().uptimeNanoseconds

    var outCount = 0
    var kept = 0

    for row in 0..<n {
        // ID filter
        if let f = config.idFilter {
            let id = ids?[row] ?? Int64(row)
            if !f.test(id: id) { continue }
        }

        // ADC accumulation
        var s: Float = 0
        for j in 0..<m {
            let code = Int(codes[row * m + j])
            s += lut[j * ks + code]
        }

        // Keep if score ≤ τ
        if s <= threshold {
            // visited
            if let vs = config.visitedSet {
                let id = ids?[row] ?? Int64(row)
                if !vs.testAndSet(id: id) { continue }
            }

            let id = ids?[row] ?? Int64(row)
            if let r = config.reservoir {
                r.insert(id: id, score: s)
                kept += 1
            } else {
                guard outCount < maxOut else { break }
                idsOut[outCount] = id
                if let so = scoresOut { so[outCount] = s }
                outCount += 1
                kept += 1
            }
        }
    }

    if config.enableTelemetry {
        let t1 = DispatchTime.now().uptimeNanoseconds
        let telem = RangeScanTelemetry(
            metric: .l2,
            threshold: threshold,
            vectorsScanned: n,
            vectorsKept: kept,
            usedEarlyExit: false,
            earlyExitHits: 0,
            usedADCPath: true,
            bytesScored: 0,
            bytesCodes: n * m * MemoryLayout<UInt8>.stride,
            executionTimeNanos: t1 - t0
        )
        recordTelemetry(telem)
    }

    return usingReservoir ? kept : outCount
}

/// Range query over 4-bit quantized codes (2 codes per byte).
@inlinable
public func rangeScanADC_u4(
    codes: UnsafePointer<UInt8>,
    vectorCount n: Int,
    subvectorCount m: Int,
    codebookSize ks: Int,
    lut: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig = .default
) -> Int {
    precondition(ks > 0 && m >= 1 && n >= 0)
    let usingReservoir = (config.reservoir != nil)
    if maxOut == 0 && !usingReservoir { return 0 }

    let t0 = DispatchTime.now().uptimeNanoseconds
    var outCount = 0
    var kept = 0

    let bytesPerRow = (m + 1) >> 1

    for row in 0..<n {
        if let f = config.idFilter {
            let id = ids?[row] ?? Int64(row)
            if !f.test(id: id) { continue }
        }

        let rowBase = row * bytesPerRow
        var s: Float = 0

        // Decode nibble-pairs
        var j = 0
        while j + 1 < m {
            let b = codes[rowBase + (j >> 1)]
            let c0 = Int(b & 0x0F)
            let c1 = Int((b >> 4) & 0x0F)
            s += lut[j * ks + c0]
            s += lut[(j + 1) * ks + c1]
            j += 2
        }
        if j < m {
            let b = codes[rowBase + (j >> 1)]
            let c0 = Int(b & 0x0F)
            s += lut[j * ks + c0]
        }

        if s <= threshold {
            if let vs = config.visitedSet {
                let id = ids?[row] ?? Int64(row)
                if !vs.testAndSet(id: id) { continue }
            }

            let id = ids?[row] ?? Int64(row)
            if let r = config.reservoir {
                r.insert(id: id, score: s)
                kept += 1
            } else {
                guard outCount < maxOut else { break }
                idsOut[outCount] = id
                if let so = scoresOut { so[outCount] = s }
                outCount += 1
                kept += 1
            }
        }
    }

    if config.enableTelemetry {
        let t1 = DispatchTime.now().uptimeNanoseconds
        let telem = RangeScanTelemetry(
            metric: .l2,
            threshold: threshold,
            vectorsScanned: n,
            vectorsKept: kept,
            usedEarlyExit: false,
            earlyExitHits: 0,
            usedADCPath: true,
            bytesScored: 0,
            bytesCodes: n * bytesPerRow * MemoryLayout<UInt8>.stride,
            executionTimeNanos: t1 - t0
        )
        recordTelemetry(telem)
    }

    return usingReservoir ? kept : outCount
}

// MARK: - Convenience High-level API

public enum RangeQueryKernel {
    /// High-level convenience: returns compacted (id, score) pairs.
    @inlinable
    public static func scan(
        query: [Float],
        database: [[Float]],
        metric: DistanceMetric,
        threshold: Float
    ) -> [(id: Int64, score: Float)] {
        guard let d = database.first?.count, d > 0 else { return [] }
        let n = database.count
        var flat = [Float](repeating: 0, count: n * d)
        for i in 0..<n {
            precondition(database[i].count == d, "ragged input")
            let base = i * d
            database[i].withUnsafeBufferPointer { src in
                flat.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.advanced(by: base).update(from: src.baseAddress!, count: d)
                }
            }
        }

        var ids = [Int64](repeating: 0, count: n)
        var scores = [Float](repeating: 0, count: n)
        let kept = rangeScanBlock(
            query: query,
            database: flat,
            ids: nil,
            vectorCount: n,
            dimension: d,
            metric: metric,
            threshold: threshold,
            idsOut: &ids,
            scoresOut: &scores,
            maxOut: n,
            config: .init(earlyExit: .auto)
        )
        return (0..<kept).map { (id: ids[$0], score: scores[$0]) }
    }
}

// =======================================================================
// MARK: - Internal Implementations
// =======================================================================

@inlinable
func rangeScanGeneric(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig
) -> Int {
    let mode = ThresholdMode(metric: metric)
    let tileSize = max(1, config.tileSize)
    var outCount = 0
    let usingReservoir = (config.reservoir != nil)

    var tileScores = [Float](repeating: 0, count: tileSize)

    for tileStart in stride(from: 0, to: n, by: tileSize) {
        let tileEnd = min(tileStart + tileSize, n)
        let m = tileEnd - tileStart

        // Score block
        tileScores.withUnsafeMutableBufferPointer { scoresPtr in
            scoreBlock(
                query: query,
                database: database + tileStart * d,
                vectorCount: m,
                dimension: d,
                metric: metric,
                scores: scoresPtr.baseAddress!
            )
        }

        // Filter & emit
        for i in 0..<m {
            let global = tileStart + i

            // Pre-filter by ID
            if let f = config.idFilter {
                let id = ids?[global] ?? Int64(global)
                if !f.test(id: id) { continue }
            }

            let score = tileScores[i]
            if !mode.passes(score: score, threshold: threshold) { continue }

            // visited set check
            if let vs = config.visitedSet {
                let id = ids?[global] ?? Int64(global)
                if !vs.testAndSet(id: id) { continue }
            }

            let id = ids?[global] ?? Int64(global)
            if let r = config.reservoir {
                r.insert(id: id, score: score)
                if usingReservoir { continue }
            } else {
                guard outCount < maxOut else { return outCount }
                idsOut[outCount] = id
                if let so = scoresOut, config.outputScores { so[outCount] = score }
                outCount += 1
            }
        }

        if !usingReservoir, outCount >= maxOut { break }
    }

    return usingReservoir ? (config.reservoir?.count ?? 0) : outCount
}

// Branch-light L2 early-exit with chunk checks.
@inlinable
func rangeScanL2_earlyExit(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    vectorCount n: Int,
    dimension d: Int,
    l2Threshold τ: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig,
    earlyExitHitsOut: inout Int
) -> Int {
    let R = 8
    let chunkSize = 16
    let numChunks = (d + chunkSize - 1) / chunkSize
    let τ2 = τ * τ
    var outCount = 0
    let usingReservoir = (config.reservoir != nil)

    var blockStart = 0
    while blockStart < n {
        let blockEnd = min(blockStart + R, n)
        let blockR = blockEnd - blockStart

        let part = UnsafeMutableBufferPointer<Float>.allocate(capacity: blockR)
        defer { part.deallocate() }
        for r in 0..<blockR { part[r] = 0 }

        var aliveMask: UInt16 = (blockR == 16) ? 0xFFFF : ((1 << blockR) - 1)

        chunkLoop: for ci in 0..<numChunks {
            let cStart = ci * chunkSize
            let cEnd = min(cStart + chunkSize, d)

            for r in 0..<blockR {
                if (aliveMask & (1 << r)) == 0 { continue }
                let row = database + (blockStart + r) * d
                var sum: Float = 0
                var j = cStart
                while j < cEnd {
                    let diff = query[j] - row[j]
                    sum += diff * diff
                    j += 1
                }
                part[r] += sum

                if part[r] > τ2 {
                    aliveMask &= ~(1 << r)
                }
            }
            if aliveMask == 0 { break chunkLoop }
        }

        // Collect survivors
        for r in 0..<blockR {
            let g = blockStart + r

            if let f = config.idFilter {
                let id = ids?[g] ?? Int64(g)
                if !f.test(id: id) { continue }
            }

            let finalL2sq = part[r]
            if finalL2sq > τ2 {
                earlyExitHitsOut += 1
                continue
            }

            if let vs = config.visitedSet {
                let id = ids?[g] ?? Int64(g)
                if !vs.testAndSet(id: id) { continue }
            }

            let id = ids?[g] ?? Int64(g)
            let l2 = sqrt(finalL2sq)
            if let rsv = config.reservoir {
                rsv.insert(id: id, score: l2)
            } else {
                guard outCount < maxOut else { return outCount }
                idsOut[outCount] = id
                if let so = scoresOut, config.outputScores { so[outCount] = l2 }
                outCount += 1
            }
        }

        blockStart = blockEnd
    }

    return usingReservoir ? (config.reservoir?.count ?? 0) : outCount
}

@usableFromInline
@inline(__always)
internal func estimateTypicalL2Distance(dimension d: Int) -> Float {
    return sqrt(Float(max(1, d)))
}

// =======================================================================
// MARK: - Scoring Kernel Integration
// =======================================================================

/// Score a block using real kernels from #01, #02, #03
@inlinable
func scoreBlock(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    scores: UnsafeMutablePointer<Float>
) {
    switch metric {
    case .l2:
        // Use kernel #01 (L2 Sqr) then sqrt for L2 distance
        l2sqr_f32_block(query, database, n, d, scores, nil, .nan, nil)
        // Convert L2^2 → L2
        for i in 0..<n {
            scores[i] = sqrt(scores[i])
        }

    case .innerProduct:
        // Use kernel #02 (Inner Product)
        IndexOps.Scoring.InnerProduct.run(
            q: query,
            xb: database,
            n: n,
            d: d,
            out: scores
        )

    case .cosine:
        // Use kernel #03 (Cosine Similarity)
        IndexOps.Scoring.Cosine.run(
            q: query,
            xb: database,
            n: n,
            d: d,
            out: scores
        )
    }
}
