Title: ✅ DONE — Range/Threshold Query Kernel — Early-Exit Similarity Search for Vector Retrieval

Summary
- Implement a high-performance range query kernel that returns all vectors whose similarity/distance to a query passes a user-defined threshold, without requiring top-k selection.
- Features branch-light early-exit optimization for L2 distance that prunes computation mid-accumulation when partial distance exceeds threshold.
- Supports multiple output modes: compacted ID/score arrays, bitmasks for in-place filtering, or direct insertion into reservoir structures.
- Integrates with ID filters (#08), visited sets (#32), and reservoir sampling (#39) for flexible query composition.
- Critical for applications requiring threshold-based retrieval: clustering, deduplication, radius search.

Project Context
- VectorIndex provides both top-k and threshold-based search capabilities
- Range queries are essential for specific use cases:
  - **Clustering**: Find all vectors within distance r of a centroid
  - **Deduplication**: Find all near-duplicate vectors (high similarity threshold)
  - **Radius Search**: Return all neighbors within a fixed radius
  - **Filtering**: Pre-filter candidates before expensive re-ranking
  - **Graph Construction**: Find all edges meeting connectivity criteria
- Industry context: Less common than top-k but critical for specific domains
  - Traditional databases: Range queries are fundamental (e.g., SQL WHERE clause)
  - Vector databases: ~15-20% of queries use thresholds instead of k
- Challenge: Efficiently handle variable output size (can be 0 to n results)
- Opportunity: Early-exit optimization can provide 2-10× speedup for tight thresholds
- VectorCore provides scoring; VectorIndex needs threshold-aware filtering
- Typical usage patterns:
  - τ (threshold) chosen based on application requirements
  - L2: τ ∈ [0, 10] for normalized vectors
  - Cosine: τ ∈ [0.7, 0.99] for high-similarity search
  - Output size varies from 0.1% to 10% of database

Goals
- Achieve 2-10× speedup via early-exit for tight thresholds (L2 distance)
- Zero overhead when early-exit disabled (matches plain scoring kernel)
- Support all three metrics: L2, inner product, cosine similarity
- Efficient handling of variable output sizes (dynamic arrays or masks)
- Integration with filters and visited sets for composed queries
- Branch-light implementation for predictable performance
- Support both flat vectors and quantized codes (ADC/PQ)
- Thread-safe for concurrent range queries

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/RangeQueryKernel.swift`
- Core implementations:
  - Flat vector API: `rangeScanBlock` with early-exit for L2
  - ADC/PQ API: `rangeScanADC_u8`, `rangeScanADC_u4` for quantized codes
  - Mask-only API: `rangeMaskBlock` for in-place compaction
  - Early-exit engine: Branch-light partial sum comparison
- Output modes:
  - Compacted arrays: (ids_out, scores_out)
  - Bitmask: uint8 mask[n] (1=keep, 0=drop)
  - Reservoir: Direct insertion into reservoir structure
- Integration points:
  - Uses ScoreBlockKernel (#04) for baseline scoring
  - Uses ADCScanKernel (#22) for quantized codes
  - Uses IDFilterOverlay (#08) for pre-filtering
  - Uses VisitedSet (#32) for deduplication
  - Uses Reservoir (#39) for sampling
  - Reports to Telemetry (#46)

API & Signatures

```swift
// MARK: - Threshold Semantics

/// Threshold comparison mode based on metric
public enum ThresholdMode {
    case l2LessOrEqual      // L2: keep if ‖q-x‖² ≤ τ (lower is better)
    case ipGreaterOrEqual   // IP: keep if ⟨q,x⟩ ≥ τ (higher is better)
    case cosineGreaterOrEqual  // Cosine: keep if cos(q,x) ≥ τ (higher is better)

    /// Initialize from distance metric
    init(metric: DistanceMetric) {
        switch metric {
        case .l2: self = .l2LessOrEqual
        case .innerProduct: self = .ipGreaterOrEqual
        case .cosine: self = .cosineGreaterOrEqual
        }
    }

    /// Whether score passes threshold
    func passes(score: Float, threshold: Float) -> Bool {
        switch self {
        case .l2LessOrEqual: return score <= threshold
        case .ipGreaterOrEqual: return score >= threshold
        case .cosineGreaterOrEqual: return score >= threshold
        }
    }
}

// MARK: - Early Exit Strategy

/// Early-exit configuration for L2 distance optimization
public enum EarlyExitStrategy {
    case auto       // Automatically decide based on threshold and data
    case on         // Always use early-exit
    case off        // Never use early-exit (baseline performance)
}

// MARK: - Output Mode

/// How to return results from range query
public enum RangeOutputMode {
    case compacted      // Write to ids_out/scores_out arrays
    case mask          // Write bitmask (1=keep, 0=drop)
    case reservoir     // Push directly to reservoir structure
}

// MARK: - Configuration

/// Configuration for range query operation
public struct RangeScanConfig {
    /// Early-exit strategy for L2 distance (default: auto)
    let earlyExit: EarlyExitStrategy

    /// Whether to compute and write scores (default: true)
    let outputScores: Bool

    /// Optional ID filter to pre-drop rows
    let idFilter: IDFilterOverlay?

    /// Optional visited set to drop duplicates
    let visitedSet: VisitedSet?

    /// Optional reservoir for direct insertion
    let reservoir: Reservoir?

    /// Output mode (default: compacted arrays)
    let outputMode: RangeOutputMode

    /// Tile size for processing (default: 1024)
    let tileSize: Int

    /// Enable telemetry (default: false)
    let enableTelemetry: Bool

    public static let `default` = RangeScanConfig(
        earlyExit: .auto,
        outputScores: true,
        idFilter: nil,
        visitedSet: nil,
        reservoir: nil,
        outputMode: .compacted,
        tileSize: 1024,
        enableTelemetry: false
    )
}

// MARK: - Core Range Query API (Flat Vectors)

/// Scan block of vectors and return all passing threshold
/// Supports early-exit optimization for L2 distance
///
/// - Complexity: O(n*d) worst case, O(n*d_partial) with early-exit
/// - Performance: 2-10× speedup for tight thresholds with early-exit
/// - Thread Safety: Reentrant; safe for concurrent calls
///
/// - Parameters:
///   - query: Query vector [d], 64-byte aligned
///   - database: Database vectors [n][d], row-major, 64-byte aligned
///   - ids: Vector IDs [n], if nil uses 0..<n
///   - vectorCount: Number of database vectors
///   - dimension: Vector dimension
///   - metric: Distance/similarity metric
///   - threshold: Threshold value τ
///   - idsOut: Output buffer for passing IDs [max_out]
///   - scoresOut: Output buffer for scores [max_out], optional
///   - maxOut: Maximum number of results to return
///   - config: Optional configuration
/// - Returns: Number of vectors passing threshold (≤ maxOut)
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
) -> Int

// MARK: - Mask-Only Variant

/// Generate bitmask of vectors passing threshold
/// Useful for in-place filtering without compaction
///
/// - Parameters:
///   - mask: Output bitmask [n], 1=keep, 0=drop
/// - Returns: Number of vectors passing threshold
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
) -> Int

// MARK: - ADC/PQ Range Query

/// Range query over quantized codes using ADC
/// Delegates to ADCScanKernel (#22) then applies threshold
///
/// - Parameters:
///   - codes: Quantized codes [n][m] (u8 or u4)
///   - lut: Lookup table [m][ks] from query
///   - threshold: Threshold value τ
@inlinable
public func rangeScanADC_u8(
    codes: UnsafePointer<UInt8>,
    vectorCount n: Int,
    subvectorCount m: Int,
    codebookSize ks: Int,  // 256 for u8
    lut: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig = .default
) -> Int

/// Range query over 4-bit quantized codes
@inlinable
public func rangeScanADC_u4(
    codes: UnsafePointer<UInt8>,
    vectorCount n: Int,
    subvectorCount m: Int,
    codebookSize ks: Int,  // 16 for u4
    lut: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig = .default
) -> Int

// MARK: - Telemetry

/// Per-query execution statistics
public struct RangeScanTelemetry {
    public let metric: DistanceMetric
    public let threshold: Float
    public let vectorsScanned: Int
    public let vectorsKept: Int
    public let usedEarlyExit: Bool
    public let earlyExitHits: Int       // Rows pruned early
    public let usedADCPath: Bool
    public let bytesScored: Int         // Flat vector bytes
    public let bytesCodes: Int          // ADC code bytes
    public let executionTimeNanos: UInt64

    public var selectivityPercent: Double {
        return (Double(vectorsKept) / Double(vectorsScanned)) * 100.0
    }

    public var earlyExitEfficiency: Double {
        guard earlyExitHits > 0 else { return 0 }
        return Double(earlyExitHits) / Double(vectorsScanned)
    }
}

// MARK: - Convenience API

extension RangeQueryKernel {
    /// High-level range query with automatic configuration
    public static func scan(
        query: [Float],
        database: [[Float]],
        metric: DistanceMetric,
        threshold: Float
    ) -> [(id: Int64, score: Float)]
}
```

Algorithm Details

**Main Dispatch Logic**:

```swift
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
    let mode = ThresholdMode(metric: metric)

    // Decide whether to use early-exit
    let useEarlyExit: Bool
    switch (metric, config.earlyExit) {
    case (.l2, .on):
        useEarlyExit = true
    case (.l2, .auto):
        // Use early-exit if threshold suggests tight filtering
        // Heuristic: tight if τ < 0.1 * typical_distance
        useEarlyExit = threshold < estimateTypicalDistance(dimension: d) * 0.1
    case (.l2, .off):
        useEarlyExit = false
    default:
        // Early-exit only applicable to L2
        useEarlyExit = false
    }

    if useEarlyExit {
        return rangeScanL2_earlyExit(
            query: query,
            database: database,
            ids: ids,
            vectorCount: n,
            dimension: d,
            threshold: threshold,
            idsOut: idsOut,
            scoresOut: scoresOut,
            maxOut: maxOut,
            config: config
        )
    } else {
        return rangeScanGeneric(
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
}
```

**Generic Range Scan** (no early-exit):

```swift
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
    var outputCount = 0
    let tileSize = config.tileSize

    // Process in tiles to maintain cache efficiency
    for tileStart in stride(from: 0, to: n, by: tileSize) {
        let tileEnd = min(tileStart + tileSize, n)
        let tileN = tileEnd - tileStart

        // Score tile using kernel #04
        var tileScores = [Float](repeating: 0, count: tileN)
        scoreBlock(
            query: query,
            database: database + tileStart * d,
            vectorCount: tileN,
            dimension: d,
            metric: metric,
            scores: &tileScores
        )

        // Filter and compact
        for i in 0..<tileN {
            let globalIdx = tileStart + i

            // Apply ID filter if provided
            if let filter = config.idFilter {
                let id = ids?[globalIdx] ?? Int64(globalIdx)
                if !filter.test(id: id) { continue }
            }

            // Check threshold
            if !mode.passes(score: tileScores[i], threshold: threshold) {
                continue
            }

            // Check visited set if provided
            if let visited = config.visitedSet {
                let id = ids?[globalIdx] ?? Int64(globalIdx)
                if !visited.testAndSet(id: id) { continue }  // Already visited
            }

            // Output handling
            if let reservoir = config.reservoir {
                // Direct insertion to reservoir
                let id = ids?[globalIdx] ?? Int64(globalIdx)
                reservoir.insert(id: id, score: tileScores[i])
            } else {
                // Compacted output
                guard outputCount < maxOut else { break }

                let id = ids?[globalIdx] ?? Int64(globalIdx)
                idsOut[outputCount] = id

                if let scores = scoresOut {
                    scores[outputCount] = tileScores[i]
                }

                outputCount += 1
            }
        }

        // Early termination if output buffer full
        if outputCount >= maxOut { break }
    }

    return outputCount
}
```

**Early-Exit L2 Range Scan** (branch-light):

```swift
@inlinable
func rangeScanL2_earlyExit(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    vectorCount n: Int,
    dimension d: Int,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    config: RangeScanConfig
) -> Int {
    // Early-exit strategy: Process R rows in parallel, check partial sums
    // against threshold after each chunk, prune exceeded rows

    let R = 8  // Process 8 rows simultaneously
    let chunkSize = 16  // Check threshold every 16 floats
    let numChunks = (d + chunkSize - 1) / chunkSize

    var outputCount = 0

    // Process in blocks of R rows
    for blockStart in stride(from: 0, to: n, by: R) {
        let blockEnd = min(blockStart + R, n)
        let blockR = blockEnd - blockStart

        // Partial sums for R rows
        var partialSums = SIMD8<Float>.zero
        var alive = UInt8((1 << blockR) - 1)  // Bitmask: 1=still computing

        // Process dimension in chunks
        for chunkIdx in 0..<numChunks {
            let chunkStart = chunkIdx * chunkSize
            let chunkEnd = min(chunkStart + chunkSize, d)

            // Accumulate for alive rows only
            for r in 0..<blockR {
                guard (alive & (1 << r)) != 0 else { continue }

                let rowPtr = database + (blockStart + r) * d

                // Compute chunk contribution
                var chunkSum: Float = 0
                for j in chunkStart..<chunkEnd {
                    let diff = query[j] - rowPtr[j]
                    chunkSum += diff * diff
                }

                partialSums[r] += chunkSum

                // Check if exceeded threshold (early-exit)
                if partialSums[r] > threshold {
                    alive &= ~(1 << r)  // Mark as dead
                }
            }

            // All rows dead? Skip remaining chunks
            if alive == 0 { break }
        }

        // Collect results from surviving rows
        for r in 0..<blockR {
            let globalIdx = blockStart + r

            // Apply ID filter if provided
            if let filter = config.idFilter {
                let id = ids?[globalIdx] ?? Int64(globalIdx)
                if !filter.test(id: id) { continue }
            }

            // Check final score against threshold
            let finalScore = partialSums[r]
            if finalScore > threshold { continue }

            // Check visited set if provided
            if let visited = config.visitedSet {
                let id = ids?[globalIdx] ?? Int64(globalIdx)
                if !visited.testAndSet(id: id) { continue }
            }

            // Output
            guard outputCount < maxOut else { return outputCount }

            let id = ids?[globalIdx] ?? Int64(globalIdx)
            idsOut[outputCount] = id

            if let scores = scoresOut {
                scores[outputCount] = sqrt(finalScore)  // Convert to L2 distance
            }

            outputCount += 1
        }
    }

    return outputCount
}
```

**Mask-Only Implementation**:

```swift
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
    let mode = ThresholdMode(metric: metric)
    var keepCount = 0

    // Score all vectors
    var scores = [Float](repeating: 0, count: n)
    scoreBlock(
        query: query,
        database: database,
        vectorCount: n,
        dimension: d,
        metric: metric,
        scores: &scores
    )

    // Generate mask
    for i in 0..<n {
        var keep = mode.passes(score: scores[i], threshold: threshold)

        // Apply ID filter
        if keep, let filter = config.idFilter {
            let id = ids?[i] ?? Int64(i)
            keep = filter.test(id: id)
        }

        mask[i] = keep ? 1 : 0
        if keep { keepCount += 1 }
    }

    return keepCount
}
```

**ADC Range Scan**:

```swift
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
    // Use ADC scan kernel (#22) to compute approximate distances
    var scores = [Float](repeating: 0, count: n)

    adcScan_u8(
        codes: codes,
        vectorCount: n,
        subvectorCount: m,
        codebookSize: ks,
        lut: lut,
        output: &scores
    )

    // Apply threshold and compact
    var outputCount = 0
    for i in 0..<n {
        // L2 distance semantics for ADC (residual space)
        if scores[i] > threshold { continue }

        // Apply filters
        if let filter = config.idFilter {
            let id = ids?[i] ?? Int64(i)
            if !filter.test(id: id) { continue }
        }

        // Output
        guard outputCount < maxOut else { break }

        let id = ids?[i] ?? Int64(i)
        idsOut[outputCount] = id

        if let scoresPtr = scoresOut {
            scoresPtr[outputCount] = scores[i]
        }

        outputCount += 1
    }

    return outputCount
}
```

Early-Exit Optimization Details

**Branch-Light Design**:

```swift
// Key insight: Minimize branches in inner loop for predictable performance

// Bad: Branchy early-exit (unpredictable)
for r in 0..<R {
    var sum: Float = 0
    for j in 0..<d {
        let diff = query[j] - database[r*d + j]
        sum += diff * diff
        if sum > threshold { break }  // Branch inside tight loop!
    }
}

// Good: Branch-light with periodic checks
for r in 0..<R {
    var partialSum: Float = 0
    var alive = true

    for chunkIdx in 0..<numChunks {
        if !alive { break }  // Branch outside chunk loop

        // No branches in tight accumulation loop
        for j in chunkStart..<chunkEnd {
            let diff = query[j] - database[r*d + j]
            partialSum += diff * diff
        }

        // Check once per chunk (amortized branch cost)
        if partialSum > threshold {
            alive = false
        }
    }
}
```

**SIMD-Friendly Row Processing**:

```swift
// Process R rows simultaneously with SIMD partial sums

let R = 8
var partialSums = SIMD8<Float>.zero

for chunkIdx in 0..<numChunks {
    // Load chunk from all R rows, compute differences
    for j in chunkStart..<chunkEnd {
        let qVal = query[j]

        // Gather from R rows (could use SIMD gather if available)
        var rowVals = SIMD8<Float>.zero
        for r in 0..<R {
            rowVals[r] = database[(blockStart + r) * d + j]
        }

        let diffs = SIMD8(repeating: qVal) - rowVals
        partialSums += diffs * diffs
    }

    // Check which rows exceeded threshold (vectorized comparison)
    let exceeded = partialSums .> SIMD8(repeating: threshold)
    // Update alive bitmask based on exceeded...
}
```

**Chunk Size Selection**:

```
Chunk size trade-offs:
- Small chunks (4-8 floats): More frequent checks, better early-exit
- Large chunks (32-64 floats): Fewer checks, less overhead, worse early-exit

Optimal for tight thresholds (high prune rate): 8-16 floats
Optimal for loose thresholds (low prune rate): 32-64 floats

Auto-tuning strategy:
- Measure prune rate in first tile
- Adjust chunk size for subsequent tiles
```

Performance Characteristics

**Early-Exit Speedup**:

```
Threshold Tightness vs Speedup (d=768, n=10000):

Selectivity | Prune Rate | Chunks Until Exit | Speedup
------------|------------|-------------------|----------
0.1%        | 99.9%      | 1-2 chunks        | 8-10×
1%          | 99%        | 2-4 chunks        | 5-7×
10%         | 90%        | 4-8 chunks        | 2-3×
50%         | 50%        | 16-32 chunks      | 1.2-1.5×
90%         | 10%        | 40+ chunks        | 0.9-1.0× (overhead)
```

**When to Use Early-Exit**:

```swift
func shouldUseEarlyExit(threshold: Float, dimension: Int, estimated selectivity: Float) -> Bool {
    // Early-exit beneficial when:
    // 1. Tight threshold (low selectivity)
    // 2. High dimension (more work to save)
    // 3. Prune rate > 50%

    let typicalDistance = estimateTypicalDistance(dimension: dimension)
    let relativeTightness = threshold / typicalDistance

    return relativeTightness < 0.2 && estimatedSelectivity < 0.3
}
```

**Latency Comparison** (Apple M1, d=768, n=10000):

```
Method                  | Selectivity 1% | Selectivity 10% | Selectivity 50%
------------------------|----------------|-----------------|------------------
Full scoring + filter   | 1.2 ms         | 1.2 ms          | 1.2 ms
Early-exit (auto)       | 0.2 ms         | 0.5 ms          | 1.1 ms
Speedup                 | 6×             | 2.4×            | 1.1×
```

Integration Patterns

**IVF with Range Query**:

```swift
struct IVFIndex {
    func rangeSearch(
        query: Vector,
        nProbe: Int,
        threshold: Float
    ) -> [SearchResult] {
        // Select cells to probe
        let probedCells = selectTopCells(query: query, nProbe: nProbe)

        var allResults: [SearchResult] = []

        // Range query each cell
        for cellID in probedCells {
            let cellVectors = cells[cellID]
            let cellIDs = getCellVectorIds(cellID)

            var cellResults = [Int64](repeating: 0, count: cellVectors.count)
            var cellScores = [Float](repeating: 0, count: cellVectors.count)

            let count = rangeScanBlock(
                query: query.data,
                database: cellVectors.flatMap { $0.data },
                ids: cellIDs,
                vectorCount: cellVectors.count,
                dimension: query.dimension,
                metric: metric,
                threshold: threshold,
                idsOut: &cellResults,
                scoresOut: &cellScores,
                maxOut: cellVectors.count
            )

            for i in 0..<count {
                allResults.append(SearchResult(id: cellResults[i], score: cellScores[i]))
            }
        }

        return allResults
    }
}
```

**Clustering with Range Queries**:

```swift
struct KMeansClustering {
    func assignToClusters(
        vectors: [Vector],
        centroids: [Vector],
        radius: Float
    ) -> [[Int]] {
        var assignments: [[Int]] = Array(repeating: [], count: centroids.count)

        for (centroidID, centroid) in centroids.enumerated() {
            var members = [Int64](repeating: 0, count: vectors.count)
            var scores = [Float](repeating: 0, count: vectors.count)

            let count = rangeScanBlock(
                query: centroid.data,
                database: vectors.flatMap { $0.data },
                ids: nil,
                vectorCount: vectors.count,
                dimension: centroid.dimension,
                metric: .l2,
                threshold: radius,
                idsOut: &members,
                scoresOut: &scores,
                maxOut: vectors.count
            )

            assignments[centroidID] = (0..<count).map { Int(members[$0]) }
        }

        return assignments
    }
}
```

**Deduplication**:

```swift
struct DeduplicationEngine {
    func findNearDuplicates(
        vectors: [Vector],
        similarityThreshold: Float = 0.95
    ) -> [(Int, Int)] {
        var duplicatePairs: [(Int, Int)] = []
        var visited = VisitedSet()

        for (i, query) in vectors.enumerated() {
            var matches = [Int64](repeating: 0, count: vectors.count)

            let count = rangeScanBlock(
                query: query.data,
                database: vectors.flatMap { $0.data },
                ids: nil,
                vectorCount: vectors.count,
                dimension: query.dimension,
                metric: .cosine,
                threshold: similarityThreshold,
                idsOut: &matches,
                scoresOut: nil,
                maxOut: vectors.count,
                config: RangeScanConfig(visitedSet: visited)
            )

            for j in 0..<count {
                let matchID = Int(matches[j])
                if matchID > i {  // Avoid duplicate pairs
                    duplicatePairs.append((i, matchID))
                }
            }
        }

        return duplicatePairs
    }
}
```

**Graph Construction**:

```swift
struct HNSWGraphBuilder {
    func buildLayer(
        vectors: [Vector],
        M: Int,  // Max connections per node
        searchRadius: Float
    ) -> [[Edge]] {
        var graph: [[Edge]] = Array(repeating: [], count: vectors.count)

        for (nodeID, query) in vectors.enumerated() {
            var neighbors = [Int64](repeating: 0, count: vectors.count)
            var distances = [Float](repeating: 0, count: vectors.count)

            let count = rangeScanBlock(
                query: query.data,
                database: vectors.flatMap { $0.data },
                ids: nil,
                vectorCount: vectors.count,
                dimension: query.dimension,
                metric: .l2,
                threshold: searchRadius,
                idsOut: &neighbors,
                scoresOut: &distances,
                maxOut: vectors.count
            )

            // Select M best neighbors (excluding self)
            let edges = (0..<count)
                .filter { neighbors[$0] != nodeID }
                .sorted { distances[$0] < distances[$1] }
                .prefix(M)
                .map { Edge(target: Int(neighbors[$0]), distance: distances[$0]) }

            graph[nodeID] = Array(edges)
        }

        return graph
    }
}
```

Correctness & Testing

**Golden Reference**:

```swift
func referenceRangeScan(
    query: [Float],
    database: [[Float]],
    metric: DistanceMetric,
    threshold: Float
) -> [(id: Int, score: Float)] {
    var results: [(Int, Float)] = []

    for (id, vector) in database.enumerated() {
        let score: Float
        switch metric {
        case .l2:
            score = l2Distance(query, vector)
        case .innerProduct:
            score = dotProduct(query, vector)
        case .cosine:
            score = cosineSimilarity(query, vector)
        }

        let mode = ThresholdMode(metric: metric)
        if mode.passes(score: score, threshold: threshold) {
            results.append((id, score))
        }
    }

    return results
}
```

**Test Cases**:

1. **Correctness**:
   - Compare early-exit vs full scoring (bit-exact)
   - Compare ADC vs flat path
   - Verify mask-only produces same results

2. **Threshold Edge Cases**:
   - τ = 0 (L2): Keep only exact matches
   - τ = ∞ (L2): Keep all vectors
   - τ = -1 (Cosine): Keep all vectors
   - τ = 1 (Cosine): Keep only identical vectors
   - τ = 0 (IP): Various behaviors

3. **Variable Output Size**:
   - Empty results (tight threshold)
   - Full results (loose threshold)
   - Partial results (moderate threshold)
   - Output buffer overflow (maxOut)

4. **Integration**:
   - With ID filter: Verify pre-filtering
   - With visited set: Verify deduplication
   - With reservoir: Verify direct insertion

5. **Performance**:
   - Early-exit speedup vs prune rate
   - No slowdown when early-exit=off
   - Tile size sensitivity

**Example Tests**:

```swift
func testRangeScan_EarlyExitCorrectness() {
    let d = 768
    let n = 1000
    let query = (0..<d).map { _ in Float.random(in: -1...1) }
    let database = (0..<n*d).map { _ in Float.random(in: -1...1) }
    let threshold: Float = 0.5

    // Full scoring path
    var idsNoExit = [Int64](repeating: 0, count: n)
    var scoresNoExit = [Float](repeating: 0, count: n)

    let configNoExit = RangeScanConfig(earlyExit: .off)
    let countNoExit = rangeScanBlock(
        query: query,
        database: database,
        ids: nil,
        vectorCount: n,
        dimension: d,
        metric: .l2,
        threshold: threshold,
        idsOut: &idsNoExit,
        scoresOut: &scoresNoExit,
        maxOut: n,
        config: configNoExit
    )

    // Early-exit path
    var idsExit = [Int64](repeating: 0, count: n)
    var scoresExit = [Float](repeating: 0, count: n)

    let configExit = RangeScanConfig(earlyExit: .on)
    let countExit = rangeScanBlock(
        query: query,
        database: database,
        ids: nil,
        vectorCount: n,
        dimension: d,
        metric: .l2,
        threshold: threshold,
        idsOut: &idsExit,
        scoresOut: &scoresExit,
        maxOut: n,
        config: configExit
    )

    // Must produce identical results
    XCTAssertEqual(countNoExit, countExit)
    for i in 0..<countNoExit {
        XCTAssertEqual(idsNoExit[i], idsExit[i])
        XCTAssertEqual(scoresNoExit[i], scoresExit[i], accuracy: 1e-5)
    }
}

func testRangeScan_ThresholdSemantics() {
    let d = 128
    let query = [Float](repeating: 0, count: d)
    let database = [
        [Float](repeating: 0.1, count: d),  // L2 dist ≈ 1.28
        [Float](repeating: 0.5, count: d),  // L2 dist ≈ 5.66
        [Float](repeating: 1.0, count: d),  // L2 dist ≈ 11.31
    ]

    var ids = [Int64](repeating: 0, count: 3)
    var scores = [Float](repeating: 0, count: 3)

    // Threshold = 6.0 should keep first two
    let count = rangeScanBlock(
        query: query,
        database: database.flatMap { $0 },
        ids: nil,
        vectorCount: 3,
        dimension: d,
        metric: .l2,
        threshold: 6.0,
        idsOut: &ids,
        scoresOut: &scores,
        maxOut: 3
    )

    XCTAssertEqual(count, 2)
    XCTAssertEqual(ids[0], 0)
    XCTAssertEqual(ids[1], 1)
}
```

Coding Guidelines

**Performance Best Practices**:
- Use early-exit for tight thresholds (selectivity < 20%)
- Disable early-exit for loose thresholds (avoid overhead)
- Tune chunk size based on observed prune rate
- Process in tiles to maintain cache efficiency
- Preallocate output buffers to avoid dynamic growth

**Memory Management**:
- Caller allocates output buffers (idsOut, scoresOut)
- Temporary score buffers allocated per tile
- Mask-only variant avoids score allocation
- No hidden allocations in hot path

**API Usage**:

```swift
// Good: Use early-exit for tight threshold
let config = RangeScanConfig(earlyExit: .on)
rangeScanBlock(query, database, ..., threshold: 0.1, config: config)

// Good: Disable early-exit for loose threshold
let config = RangeScanConfig(earlyExit: .off)
rangeScanBlock(query, database, ..., threshold: 10.0, config: config)

// Good: Use mask for in-place filtering
var mask = [UInt8](repeating: 0, count: n)
rangeMaskBlock(query, database, ..., threshold: 0.5, mask: &mask)

// Bad: Dynamic output array growth
var results: [(Int64, Float)] = []
for each tile {
    results.append(...)  // Expensive reallocation!
}
```

Non-Goals

- GPU/Metal acceleration (CPU-focused)
- Multi-query batching (process independently)
- Approximate range queries (exact threshold only)
- Adaptive thresholds (fixed threshold per query)
- Range aggregation (count/sum/avg of passing vectors)

Example Usage

```swift
import VectorIndex

// Example 1: Basic range query with early-exit
let query = Vector(data: [Float](repeating: 0.5, count: 768))
let database = loadDatabase()  // [Float] of n*768 elements
let n = database.count / 768

var ids = [Int64](repeating: 0, count: n)
var scores = [Float](repeating: 0, count: n)

let count = rangeScanBlock(
    query: query.data,
    database: database,
    ids: nil,
    vectorCount: n,
    dimension: 768,
    metric: .l2,
    threshold: 1.0,  // Tight threshold
    idsOut: &ids,
    scoresOut: &scores,
    maxOut: n,
    config: RangeScanConfig(earlyExit: .on)
)

print("Found \(count) vectors within distance 1.0")

// Example 2: Mask-only for in-place filtering
var mask = [UInt8](repeating: 0, count: n)

let keepCount = rangeMaskBlock(
    query: query.data,
    database: database,
    ids: nil,
    vectorCount: n,
    dimension: 768,
    metric: .cosine,
    threshold: 0.9,  // High similarity
    mask: &mask
)

// Use mask to filter other arrays
let filtered = database.enumerated().filter { mask[$0.offset] == 1 }

// Example 3: Range query with ID filter
let filter = IDFilterOverlay(allowedIDs: Set([10, 20, 30, 40]))
let config = RangeScanConfig(idFilter: filter)

var filteredIDs = [Int64](repeating: 0, count: n)
let filteredCount = rangeScanBlock(
    query: query.data,
    database: database,
    ids: nil,
    vectorCount: n,
    dimension: 768,
    metric: .l2,
    threshold: 2.0,
    idsOut: &filteredIDs,
    scoresOut: nil,
    maxOut: n,
    config: config
)

// Example 4: Deduplication with visited set
var visited = VisitedSet()
var uniqueMatches = [Int64](repeating: 0, count: n)

let uniqueCount = rangeScanBlock(
    query: query.data,
    database: database,
    ids: nil,
    vectorCount: n,
    dimension: 768,
    metric: .cosine,
    threshold: 0.95,
    idsOut: &uniqueMatches,
    scoresOut: nil,
    maxOut: n,
    config: RangeScanConfig(visitedSet: visited)
)

// Example 5: ADC range query for PQ-compressed index
let codes = loadPQCodes()  // [UInt8] of n*m elements
let lut = computeLUT(query, codebooks)  // [Float] of m*256

var adcIDs = [Int64](repeating: 0, count: n)
let adcCount = rangeScanADC_u8(
    codes: codes,
    vectorCount: n,
    subvectorCount: 32,  // m=32
    codebookSize: 256,
    lut: lut,
    ids: nil,
    threshold: 1.5,
    idsOut: &adcIDs,
    scoresOut: nil,
    maxOut: n
)
```

Mathematical Foundation

**Threshold Semantics**:

1. **L2 Distance**:
   ```
   Keep vector x if: ‖q - x‖₂ ≤ τ
   Equivalently: sqrt(Σ(qᵢ - xᵢ)²) ≤ τ
   Or: Σ(qᵢ - xᵢ)² ≤ τ²
   ```

2. **Inner Product**:
   ```
   Keep vector x if: ⟨q, x⟩ ≥ τ
   Equivalently: Σ qᵢ·xᵢ ≥ τ
   ```

3. **Cosine Similarity**:
   ```
   Keep vector x if: cos(q, x) ≥ τ
   Equivalently: ⟨q, x⟩/(‖q‖₂·‖x‖₂) ≥ τ
   Range: τ ∈ [-1, 1]
   ```

**Early-Exit Condition** (L2):
```
At chunk c, partial sum sₖ = Σⱼ₌₀ᶜ (qⱼ - xⱼ)²

If sₖ > τ², then final sum s = Σⱼ₌₀ᵈ (qⱼ - xⱼ)² ≥ sₖ > τ²

Therefore: Can safely prune without computing remaining chunks
```

**Expected Work Reduction**:
```
Let p = prune rate (fraction of vectors exceeding τ)
Let c_avg = average chunks computed before pruning

Expected chunks per vector: (1-p)·C + p·c_avg

Where C = total chunks = ⌈d / chunk_size⌉

Speedup ≈ C / ((1-p)·C + p·c_avg)

Example: p=0.9, c_avg=5, C=48
Speedup = 48 / (0.1·48 + 0.9·5) = 48 / 9.3 ≈ 5.2×
```

Dependencies

**Internal**:
- ScoreBlockKernel (#04): Baseline scoring without early-exit
- ADCScanKernel (#22): Quantized code scoring
- CosineSimilarityKernel (#03): Cosine metric
- NormKernel (#09): Norm computation for cosine
- IDFilterOverlay (#08): ID-based filtering
- VisitedSet (#32): Deduplication
- Reservoir (#39): Sampling integration
- Telemetry (#46): Performance tracking

**External**:
- Swift Standard Library: SIMD, UnsafePointer
- Foundation: None (pure compute)
- Dispatch: For parallel tile processing (optional)

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Performance**:
- Early-exit achieves 2-10× speedup for tight thresholds
- Zero overhead when early-exit disabled (≤2% difference vs baseline)
- Tile processing maintains cache efficiency
- Variable output size handled efficiently

✅ **Correctness**:
- Early-exit produces bit-exact results vs full scoring
- All threshold semantics correct (L2, IP, cosine)
- Edge cases handled (empty results, full results, overflow)
- Integration with filters/visited sets correct

✅ **Flexibility**:
- Supports all three metrics
- Multiple output modes (compacted, mask, reservoir)
- Configurable early-exit strategy
- Works with both flat and quantized vectors

✅ **Integration**:
- Successfully used in IVF range search
- Compatible with clustering algorithms
- Works with deduplication pipelines
- Supports graph construction workflows

✅ **Robustness**:
- Handles variable selectivity (0.1% to 90%)
- Automatic early-exit decision logic
- Graceful degradation for loose thresholds
- Thread-safe for concurrent queries
<!-- moved to docs/kernel-specs/ -->
