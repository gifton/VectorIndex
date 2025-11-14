Title: ✅ DONE — Batched Score Block Kernel — Unified Multi-Metric Scoring for Vector Search

Summary
- Implement a high-level batched scoring kernel that computes similarity/distance scores between a single query vector and a block of database vectors under configurable metrics (L2 distance, inner product, cosine similarity).
- Provides unified API that dispatches to specialized microkernels (#01 L2, #02 IP, #03 Cosine) based on metric selection.
- Serves as the primary scoring primitive for all search operations in VectorIndex.
- Optimizes for common search patterns: score large blocks, select top-k, support pre-computed auxiliary data (norms).
- Thread-safe and parallelizable across query batches or database partitions.

Project Context
- VectorIndex provides high-performance vector search with support for multiple distance metrics
- Score block is the fundamental operation called by all search algorithms:
  - **IVF Search**: Score all vectors in selected cells
  - **HNSW Graph Navigation**: Score candidate neighbors during beam search
  - **Flat Brute-Force**: Score entire database (n=1M+)
  - **Re-ranking**: Score refined candidate sets after initial filtering
- Industry context: Most vector databases support multiple metrics but require separate code paths
- Challenge: Provide unified API without performance regression vs. specialized implementations
- Typical usage pattern: 1 query × 100-10,000 candidates per search operation
- Performance critical: This kernel accounts for 80-90% of total search time
- VectorCore provides microkernels; VectorIndex needs high-level orchestration
- Metric distribution in production:
  - Cosine: ~60-70% (semantic search, recommendations)
  - L2: ~20-30% (traditional ML, clustering)
  - IP: ~10% (maximum inner product search, collaborative filtering)

Goals
- Zero-overhead abstraction: match specialized kernel performance within 2%
- Support all three major metrics with single unified API
- Leverage pre-computed auxiliary data (norms, squared norms) when available
- Thread-safe for concurrent scoring across partitions
- Enable efficient parallelization across both queries and database blocks
- Provide both high-level convenience API and low-level pointer API
- Seamless integration with search algorithms (IVF, HNSW, flat)
- Telemetry integration for profiling metric usage and performance

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/ScoreBlockKernel.swift`
- Core implementations:
  - Main kernel: `scoreBlock` with runtime metric selection
  - Metric-specific variants: `scoreBlockL2`, `scoreBlockIP`, `scoreBlockCosine`
  - Batch scoring: `scoreBatch` for multiple queries
  - Top-k integration: `scoreAndSelectTopK` (optional, may be separate kernel)
- Integration points:
  - Dispatches to L2DistanceKernel (#01)
  - Dispatches to InnerProductKernel (#02)
  - Dispatches to CosineSimilarityKernel (#03)
  - Uses NormKernel (#09) for on-the-fly norm computation
  - Exports scores to IVF (#05), HNSW (#29), re-ranking (#40)
- Supporting utilities:
  - Metric enum with dispatch logic
  - Auxiliary data handling (norm/inv-norm management)
  - Parallel scoring helpers
  - Telemetry integration (#46)

API & Signatures

```swift
// MARK: - Metric Definition

/// Distance/similarity metric for vector comparison
public enum DistanceMetric: String, Codable {
    case l2          // Euclidean distance: ‖q - x‖₂
    case innerProduct // Dot product: ⟨q, x⟩ (higher is better)
    case cosine      // Cosine similarity: ⟨q,x⟩/(‖q‖·‖x‖) (higher is better)

    /// Whether higher scores indicate better matches
    public var higherIsBetter: Bool {
        switch self {
        case .l2: return false          // Distance: lower is better
        case .innerProduct: return true // Similarity: higher is better
        case .cosine: return true       // Similarity: higher is better
        }
    }

    /// Worst possible score for this metric (used for initialization)
    public var worstScore: Float {
        switch self {
        case .l2: return .infinity
        case .innerProduct: return -.infinity
        case .cosine: return -1.0
        }
    }
}

// MARK: - Auxiliary Data

/// Pre-computed auxiliary data for scoring optimization
public struct AuxiliaryData {
    /// For L2: squared norms ‖xb[i]‖² (optional, saves recomputation)
    /// For Cosine: inverse norms 1/‖xb[i]‖ (required)
    let databaseAux: UnsafePointer<Float>?

    /// For L2: ‖q‖² (optional)
    /// For Cosine: 1/‖q‖ (required if using cosine)
    let queryAux: Float

    /// Whether auxiliary data is provided
    var hasAux: Bool { databaseAux != nil }

    public init(databaseAux: UnsafePointer<Float>? = nil, queryAux: Float = 0) {
        self.databaseAux = databaseAux
        self.queryAux = queryAux
    }

    public static let none = AuxiliaryData(databaseAux: nil, queryAux: 0)
}

// MARK: - Core Scoring API

/// Compute scores between a query and a block of database vectors
/// This is the primary scoring primitive used by all search operations
///
/// - Complexity: O(n * d) for metric computation
/// - Performance: Matches specialized kernels within 2% overhead
/// - Thread Safety: Reentrant; safe for concurrent calls with disjoint outputs
///
/// - Parameters:
///   - query: Query vector [d], 64-byte aligned
///   - database: Database vectors [n][d], 64-byte aligned, row-major
///   - vectorCount: Number of database vectors (n)
///   - dimension: Vector dimension (d)
///   - metric: Distance/similarity metric to use
///   - scores: Output scores [n], 64-byte aligned
///   - aux: Optional pre-computed auxiliary data (norms/inv-norms)
///   - config: Optional configuration (telemetry, verification, etc.)
@inlinable
public func scoreBlock(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    scores: UnsafeMutablePointer<Float>,
    aux: AuxiliaryData = .none,
    config: ScoreBlockConfig = .default
)

// MARK: - Metric-Specific Variants

/// Score block using L2 distance (Euclidean)
/// Optimized fast path when metric is statically known
@inlinable
public func scoreBlockL2(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dimension: Int,
    scores: UnsafeMutablePointer<Float>,
    dbSquaredNorms: UnsafePointer<Float>? = nil,
    querySquaredNorm: Float = 0
)

/// Score block using inner product
/// Optimized fast path when metric is statically known
@inlinable
public func scoreBlockIP(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dimension: Int,
    scores: UnsafeMutablePointer<Float>
)

/// Score block using cosine similarity
/// Requires pre-computed inverse norms for optimal performance
@inlinable
public func scoreBlockCosine(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dimension: Int,
    scores: UnsafeMutablePointer<Float>,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float
)

// MARK: - Batch Scoring

/// Score multiple queries against the same database block
/// Efficiently amortizes database loading across queries
///
/// - Parameters:
///   - queries: Multiple query vectors [numQueries][d]
///   - database: Database vectors [n][d]
///   - numQueries: Number of queries
///   - vectorCount: Number of database vectors
///   - dimension: Vector dimension
///   - metric: Distance/similarity metric
///   - scores: Output scores [numQueries][n]
///   - aux: Auxiliary data (shared across queries for database)
@inlinable
public func scoreBatch(
    queries: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    numQueries: Int,
    vectorCount: Int,
    dimension: Int,
    metric: DistanceMetric,
    scores: UnsafeMutablePointer<Float>,
    aux: AuxiliaryData = .none,
    config: ScoreBlockConfig = .default
)

// MARK: - Configuration

/// Configuration for score block kernel
public struct ScoreBlockConfig {
    /// Enable telemetry recording (default: false)
    let enableTelemetry: Bool

    /// Verify alignment in debug builds (default: false in release)
    let verifyAlignment: Bool

    /// Enable parallel scoring for large blocks (default: true for n > 10000)
    let enableParallelScoring: Bool

    /// Threshold for parallel scoring (number of vectors)
    let parallelThreshold: Int

    /// Prefetch distance for cache optimization (default: 2)
    let prefetchDistance: Int

    public static let `default` = ScoreBlockConfig(
        enableTelemetry: false,
        verifyAlignment: false,
        enableParallelScoring: true,
        parallelThreshold: 10000,
        prefetchDistance: 2
    )
}

// MARK: - Telemetry

/// Per-kernel execution statistics
public struct ScoreBlockTelemetry {
    public let metric: DistanceMetric
    public let vectorsScored: Int
    public let dimension: Int
    public let usedAuxData: Bool          // Whether auxiliary data was provided
    public let usedParallel: Bool         // Whether parallel scoring was used
    public let fastPathHit: Bool          // Whether specialized kernel was used
    public let executionTimeNanos: UInt64

    public var throughputVecsPerSec: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return Double(vectorsScored) / seconds
    }
}

// MARK: - Convenience API

extension ScoreBlockKernel {
    /// High-level API with automatic memory management
    /// Returns array of scores for convenience
    public static func score(
        query: [Float],
        database: [[Float]],
        metric: DistanceMetric,
        precomputedNorms: [Float]? = nil
    ) -> [Float]

    /// Score and select top-k in one operation
    /// More efficient than scoring all then sorting
    public static func scoreTopK(
        query: [Float],
        database: [[Float]],
        k: Int,
        metric: DistanceMetric,
        precomputedNorms: [Float]? = nil
    ) -> [(index: Int, score: Float)]
}
```

Algorithm Details

**Unified Dispatch Strategy**:

```swift
@inlinable
public func scoreBlock(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    scores: UnsafeMutablePointer<Float>,
    aux: AuxiliaryData = .none,
    config: ScoreBlockConfig = .default
) {
    #if DEBUG
    if config.verifyAlignment {
        verifyAlignment(query, 64, "query")
        verifyAlignment(database, 64, "database")
        verifyAlignment(scores, 64, "scores")
    }
    #endif

    #if ENABLE_TELEMETRY
    let startTime = config.enableTelemetry ? mach_absolute_time() : 0
    #endif

    // Dispatch to metric-specific kernel
    switch metric {
    case .l2:
        scoreBlockL2_impl(
            query: query,
            database: database,
            vectorCount: n,
            dimension: d,
            scores: scores,
            dbSquaredNorms: aux.databaseAux,
            querySquaredNorm: aux.queryAux
        )

    case .innerProduct:
        // Delegate directly to IP kernel (#02)
        innerProductBlock_f32(
            query: query,
            database: database,
            vectorCount: n,
            dimension: d,
            output: scores
        )

    case .cosine:
        guard let dbInvNorms = aux.databaseAux else {
            fatalError("Cosine metric requires pre-computed inverse norms in aux.databaseAux")
        }

        // Compute query inverse norm if not provided
        let queryInvNorm = aux.queryAux > 0
            ? aux.queryAux
            : computeQueryInvNorm_f32(query: query, dimension: d)

        // Delegate to cosine kernel (#03)
        cosineBlock_f32(
            query: query,
            database: database,
            vectorCount: n,
            dimension: d,
            dbInvNorms: dbInvNorms,
            queryInvNorm: queryInvNorm,
            output: scores
        )
    }

    #if ENABLE_TELEMETRY
    if config.enableTelemetry {
        let telemetry = ScoreBlockTelemetry(
            metric: metric,
            vectorsScored: n,
            dimension: d,
            usedAuxData: aux.hasAux,
            usedParallel: false,  // Update if parallel path taken
            fastPathHit: true,
            executionTimeNanos: mach_absolute_time() - startTime
        )
        GlobalTelemetryRecorder.record(telemetry)
    }
    #endif
}
```

**L2 Distance Implementation** (with norm optimization):

```swift
@inlinable
func scoreBlockL2_impl(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    scores: UnsafeMutablePointer<Float>,
    dbSquaredNorms: UnsafePointer<Float>?,
    querySquaredNorm: Float
) {
    if let dbNorms = dbSquaredNorms, querySquaredNorm > 0 {
        // Optimized path: use ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩
        var dotProducts = [Float](repeating: 0, count: n)

        // Compute dot products using IP kernel
        innerProductBlock_f32(
            query: query,
            database: database,
            vectorCount: n,
            dimension: d,
            output: &dotProducts
        )

        // Compute L2 distances from norms and dot products
        for i in 0..<n {
            let l2Squared = querySquaredNorm + dbNorms[i] - 2.0 * dotProducts[i]
            scores[i] = sqrt(max(0, l2Squared))  // Clamp negative values from fp errors
        }
    } else {
        // Direct L2 computation (delegate to L2 kernel #01)
        l2DistanceBlock_f32(
            query: query,
            database: database,
            vectorCount: n,
            dimension: d,
            output: scores
        )
    }
}
```

**Batch Scoring Implementation** (multiple queries):

```swift
@inlinable
public func scoreBatch(
    queries: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    numQueries: Int,
    vectorCount: Int,
    dimension: Int,
    metric: DistanceMetric,
    scores: UnsafeMutablePointer<Float>,
    aux: AuxiliaryData = .none,
    config: ScoreBlockConfig = .default
) {
    // Process each query independently
    // Database stays hot in cache across queries
    for queryIdx in 0..<numQueries {
        let queryPtr = queries + queryIdx * dimension
        let scoresPtr = scores + queryIdx * vectorCount

        // Compute auxiliary data for this query if needed
        var queryAux = aux
        if metric == .cosine && queryAux.queryAux == 0 {
            queryAux.queryAux = computeQueryInvNorm_f32(
                query: queryPtr,
                dimension: dimension
            )
        } else if metric == .l2 && queryAux.queryAux == 0 {
            queryAux.queryAux = l2NormSquared_f32(
                vector: queryPtr,
                dimension: dimension
            )
        }

        // Score this query
        scoreBlock(
            query: queryPtr,
            database: database,
            vectorCount: vectorCount,
            dimension: dimension,
            metric: metric,
            scores: scoresPtr,
            aux: queryAux,
            config: config
        )
    }
}
```

**Parallel Scoring** (for large blocks):

```swift
@inlinable
func scoreBlockParallel(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    scores: UnsafeMutablePointer<Float>,
    aux: AuxiliaryData,
    config: ScoreBlockConfig
) {
    let threadCount = ProcessInfo.processInfo.activeProcessorCount
    let vectorsPerThread = (n + threadCount - 1) / threadCount

    DispatchQueue.concurrentPerform(iterations: threadCount) { threadID in
        let start = threadID * vectorsPerThread
        let end = min(start + vectorsPerThread, n)
        let count = end - start

        guard count > 0 else { return }

        // Each thread scores a partition of vectors
        let dbPtr = database + start * d
        let scoresPtr = scores + start

        // Auxiliary data needs to be offset for this partition
        var threadAux = aux
        if let dbAux = aux.databaseAux {
            threadAux.databaseAux = dbAux + start
        }

        scoreBlock(
            query: query,
            database: dbPtr,
            vectorCount: count,
            dimension: d,
            metric: metric,
            scores: scoresPtr,
            aux: threadAux,
            config: config
        )
    }
}
```

Parallelization Strategies

**Parallelism Dimensions**:

1. **Across Queries** (for batch scoring):
   - Each query processed independently
   - Database stays hot in shared cache
   - Good for: Multiple user requests, batch inference

2. **Across Vectors** (for large blocks):
   - Partition n vectors across threads
   - Each thread has independent output buffer
   - Good for: Large database blocks (n > 10k)

3. **Across Partitions** (for IVF/HNSW):
   - Different cells/graph regions scored in parallel
   - Application-level parallelism
   - Good for: Multi-probe IVF, beam search HNSW

**Thread Safety**:
- Reentrant: Multiple threads can call scoreBlock concurrently
- Disjoint outputs: Each thread writes to separate score buffer
- No shared state: Kernels are pure functions

**Performance Considerations**:
```
Parallelization overhead vs benefit:

Small blocks (n < 1000):
  - Overhead: Thread spawning ~10μs
  - Benefit: ~2× speedup on 8 cores
  - Decision: Serial execution (overhead dominates)

Medium blocks (1000 < n < 10000):
  - Overhead: ~1% of total time
  - Benefit: ~4-6× speedup on 8 cores
  - Decision: Parallel if >5000 vectors

Large blocks (n > 10000):
  - Overhead: <0.1% of total time
  - Benefit: ~7-8× speedup on 8 cores
  - Decision: Always parallel
```

Metric-Specific Optimizations

**L2 Distance**:

Option 1: Direct computation
```swift
// Compute ‖q - x‖₂ directly using L2 kernel (#01)
// Advantage: Single kernel call, no auxiliary data needed
// Cost: O(n*d) with subtraction + square + sqrt
l2DistanceBlock_f32(query, database, n, d, scores)
```

Option 2: Norm-based computation
```swift
// Use identity: ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩
// Advantage: Reuses pre-computed norms, faster IP kernel
// Cost: O(n*d) for IP + O(n) for final computation
// Requires: Pre-computed ‖q‖² and ‖x‖²

// Step 1: Compute dot products
innerProductBlock_f32(query, database, n, d, dots)

// Step 2: Compute L2 from formula
for i in 0..<n:
    l2² = ‖q‖² + ‖x[i]‖² - 2*dots[i]
    scores[i] = sqrt(max(0, l2²))  // Clamp negatives
```

**Trade-off Analysis**:
- Direct: 1 kernel call, no pre-computation, ~10% slower
- Norm-based: 2 operations, requires pre-computation, ~10% faster if norms available
- Recommendation: Use norm-based if norms already computed (e.g., for cosine), else direct

**Inner Product**:
```swift
// Simplest case: direct delegation to IP kernel (#02)
innerProductBlock_f32(query, database, n, d, scores)
// No auxiliary data needed
// Optimal performance: 85-95% memory bandwidth
```

**Cosine Similarity**:
```swift
// Requires pre-computed inverse norms
// Database norms: computed once during index build
// Query norm: computed once per query

let queryInvNorm = computeQueryInvNorm_f32(query, d)
cosineBlock_f32(query, database, n, d, scores, dbInvNorms, queryInvNorm)

// Performance: 80-85% memory bandwidth
// Memory overhead: 4n bytes (f32) or 2n bytes (f16) for dbInvNorms
```

Memory Layout & Caching

**Cache Optimization Strategy**:

```
Cache Hierarchy Usage:
- L1d (128 KB): Query vector (hot, <6 KB for d=1536)
- L2 (12 MB): Recent database vectors + auxiliary data
- L3/SLC: Full database block (if n*d*4 < L3 size)

Example (d=768, n=10000):
- Query: 768 * 4 = 3 KB (fits in L1)
- Database: 10000 * 768 * 4 = 30 MB (exceeds L2)
- Norms: 10000 * 4 = 40 KB (fits in L2)
- Scores: 10000 * 4 = 40 KB (fits in L2)

Access Pattern:
1. Query loaded once into L1 (via IP/L2/Cosine kernels)
2. Database streamed sequentially (one pass)
3. Norms/auxiliary data streamed with database
4. Scores written sequentially
```

**Prefetching** (integration with #49):
```swift
// Prefetch next cache line while computing current
for i in stride(from: 0, to: n, by: prefetchStride) {
    if i + prefetchDistance < n {
        prefetch(database + (i + prefetchDistance) * d)
        if let aux = aux.databaseAux {
            prefetch(aux + i + prefetchDistance)
        }
    }

    // Score current block
    scoreBlock(query, database + i*d, min(prefetchStride, n-i), d, metric, scores + i, aux)
}
```

Performance Targets (Apple M1/M2/M3, Release Build)

**Throughput** (vectors/second, d=768):

```
Metric        | Throughput (M1) | % of IP Kernel
--------------|-----------------|---------------
Inner Product | 850,000 vec/s   | 100% (baseline)
L2 (direct)   | 600,000 vec/s   | 71%  (sqrt overhead)
L2 (norms)    | 750,000 vec/s   | 88%  (optimized)
Cosine (f32)  | 750,000 vec/s   | 88%  (scaling overhead)
Cosine (f16)  | 800,000 vec/s   | 94%  (less memory)
```

**Dispatch Overhead**:
- Switch statement: <0.1% of total time (negligible)
- Indirect call overhead: None (inlined in release builds)
- Auxiliary data preparation: <1% for cosine, 0% for IP

**Parallel Scaling** (n=100,000, d=768, M1 8-core):
```
Threads | Speedup | Efficiency
--------|---------|------------
1       | 1.0×    | 100%
2       | 1.95×   | 98%
4       | 3.85×   | 96%
8       | 7.20×   | 90%
```

**Memory Bandwidth Utilization**:
- IP: 85-95% (memory-bound, optimal)
- L2 direct: 60-70% (compute-bound due to sqrt)
- L2 norms: 80-85% (memory-bound, near-optimal)
- Cosine: 80-85% (memory-bound, scaling overhead)

Correctness & Testing

**Golden Reference**:
- L2: NumPy `np.linalg.norm(q - xb[i])`
- IP: NumPy `np.dot(q, xb[i])`
- Cosine: SciPy `1 - scipy.spatial.distance.cosine(q, xb[i])`

**Test Cases**:

1. **Metric Coverage**:
   - All three metrics: L2, IP, Cosine
   - With and without auxiliary data
   - With and without parallel execution

2. **Dimension Coverage**:
   - Specialized: d ∈ {512, 768, 1024, 1536}
   - Generic: d ∈ {64, 128, 513, 2000}

3. **Block Sizes**:
   - Small: n ∈ {1, 10, 100}
   - Medium: n ∈ {1000, 5000}
   - Large: n ∈ {10000, 100000}

4. **Edge Cases**:
   - Zero vectors
   - Identical vectors
   - Orthogonal vectors
   - All negative/positive values

5. **Consistency**:
   - scoreBlock(metric=IP) ≡ scoreBlockIP()
   - scoreBlock(metric=L2) ≡ scoreBlockL2()
   - scoreBlock(metric=Cosine) ≡ scoreBlockCosine()
   - Parallel ≡ Serial results (bit-exact)

6. **Auxiliary Data**:
   - L2 with/without norms produces same results
   - Cosine requires auxiliary data (test fatal error)
   - Cosine with computed vs provided query norm matches

**Example Tests**:

```swift
func testScoreBlock_L2_vsReference() {
    let d = 768
    let n = 1000
    let query = (0..<d).map { _ in Float.random(in: -1...1) }
    let database = (0..<n*d).map { _ in Float.random(in: -1...1) }

    var scores = [Float](repeating: 0, count: n)

    scoreBlock(
        query: query,
        database: database,
        vectorCount: n,
        dimension: d,
        metric: .l2,
        scores: &scores
    )

    // Verify against NumPy-like reference
    for i in 0..<n {
        var expected: Float = 0
        for j in 0..<d {
            let diff = query[j] - database[i*d + j]
            expected += diff * diff
        }
        expected = sqrt(expected)

        XCTAssertEqual(scores[i], expected, accuracy: 1e-5)
    }
}

func testScoreBlock_MetricDispatch() {
    let d = 512
    let n = 100
    let query = [Float](repeating: 1.0, count: d)
    let database = [Float](repeating: 0.5, count: n * d)

    // Test IP via unified API
    var scoresIP = [Float](repeating: 0, count: n)
    scoreBlock(query, database, n, d, .innerProduct, &scoresIP)

    // Test IP via specialized API
    var scoresIPDirect = [Float](repeating: 0, count: n)
    scoreBlockIP(query, database, n, d, &scoresIPDirect)

    // Should be identical
    for i in 0..<n {
        XCTAssertEqual(scoresIP[i], scoresIPDirect[i])
    }
}

func testScoreBlock_ParallelConsistency() {
    let d = 768
    let n = 50000  // Large enough to trigger parallel
    let query = (0..<d).map { _ in Float.random(in: -1...1) }
    let database = (0..<n*d).map { _ in Float.random(in: -1...1) }

    // Serial execution
    var configSerial = ScoreBlockConfig.default
    configSerial.enableParallelScoring = false
    var scoresSerial = [Float](repeating: 0, count: n)
    scoreBlock(query, database, n, d, .l2, &scoresSerial, config: configSerial)

    // Parallel execution
    var configParallel = ScoreBlockConfig.default
    configParallel.enableParallelScoring = true
    configParallel.parallelThreshold = 10000
    var scoresParallel = [Float](repeating: 0, count: n)
    scoreBlock(query, database, n, d, .l2, &scoresParallel, config: configParallel)

    // Should be bit-exact
    for i in 0..<n {
        XCTAssertEqual(scoresSerial[i], scoresParallel[i])
    }
}

func testScoreBatch_MultipleQueries() {
    let d = 768
    let n = 1000
    let numQueries = 10

    let queries = (0..<numQueries*d).map { _ in Float.random(in: -1...1) }
    let database = (0..<n*d).map { _ in Float.random(in: -1...1) }

    var batchScores = [Float](repeating: 0, count: numQueries * n)

    scoreBatch(
        queries: queries,
        database: database,
        numQueries: numQueries,
        vectorCount: n,
        dimension: d,
        metric: .cosine,
        scores: &batchScores
    )

    // Verify each query individually
    for queryIdx in 0..<numQueries {
        var singleScores = [Float](repeating: 0, count: n)
        let queryPtr = queries + queryIdx * d

        scoreBlock(
            query: queryPtr,
            database: database,
            vectorCount: n,
            dimension: d,
            metric: .cosine,
            scores: &singleScores
        )

        for i in 0..<n {
            let batchScore = batchScores[queryIdx * n + i]
            let singleScore = singleScores[i]
            XCTAssertEqual(batchScore, singleScore, accuracy: 1e-5)
        }
    }
}
```

Integration with Search Algorithms

**IVF Search Integration**:

```swift
struct IVFIndex {
    let cells: [[Vector]]
    let cellNorms: [[Float]]  // Pre-computed norms per cell
    let metric: DistanceMetric

    func search(query: Vector, nProbe: Int, k: Int) -> [SearchResult] {
        // Step 1: Select top-nProbe cells (using coarse quantizer)
        let probedCells = selectCells(query: query, nProbe: nProbe)

        // Step 2: Score all vectors in probed cells
        var allScores: [(cellID: Int, vectorID: Int, score: Float)] = []

        // Prepare query auxiliary data once
        let queryAux = prepareQueryAux(query: query, metric: metric)

        for (cellID, cell) in probedCells {
            let cellVectors = cells[cellID]
            var scores = [Float](repeating: 0, count: cellVectors.count)

            let aux = AuxiliaryData(
                databaseAux: cellNorms[cellID],
                queryAux: queryAux
            )

            scoreBlock(
                query: query.data,
                database: cellVectors.flatMap { $0.data },
                vectorCount: cellVectors.count,
                dimension: query.dimension,
                metric: metric,
                scores: &scores,
                aux: aux
            )

            for (vectorID, score) in scores.enumerated() {
                allScores.append((cellID, vectorID, score))
            }
        }

        // Step 3: Select top-k
        return selectTopK(allScores, k: k, metric: metric)
    }
}
```

**HNSW Graph Navigation**:

```swift
struct HNSWIndex {
    let graph: [[GraphNode]]
    let metric: DistanceMetric

    func searchLayer(
        query: Vector,
        entryPoint: GraphNode,
        ef: Int,
        layer: Int
    ) -> [GraphNode] {
        var candidates = MinHeap<GraphNode>(capacity: ef)
        var visited = Set<Int>()

        candidates.insert(entryPoint)
        visited.insert(entryPoint.id)

        while !candidates.isEmpty {
            let current = candidates.extractMin()
            let neighbors = graph[layer][current.id].neighbors

            // Score all neighbors at once
            let neighborVectors = neighbors.map { $0.vector }
            var scores = [Float](repeating: 0, count: neighbors.count)

            scoreBlock(
                query: query.data,
                database: neighborVectors.flatMap { $0.data },
                vectorCount: neighbors.count,
                dimension: query.dimension,
                metric: metric,
                scores: &scores
            )

            // Add unvisited neighbors to candidates
            for (neighbor, score) in zip(neighbors, scores) {
                if !visited.contains(neighbor.id) {
                    candidates.insert(neighbor, score: score)
                    visited.insert(neighbor.id)
                }
            }
        }

        return candidates.topK(ef)
    }
}
```

**Flat Brute-Force Search**:

```swift
struct FlatIndex {
    let vectors: [Vector]
    let norms: [Float]
    let metric: DistanceMetric

    func search(query: Vector, k: Int) -> [SearchResult] {
        let n = vectors.count
        var scores = [Float](repeating: 0, count: n)

        let queryAux = prepareQueryAux(query: query, metric: metric)
        let aux = AuxiliaryData(databaseAux: norms, queryAux: queryAux)

        // Optionally use parallel scoring for large indices
        let config = ScoreBlockConfig(
            enableParallelScoring: n > 10000,
            parallelThreshold: 10000
        )

        scoreBlock(
            query: query.data,
            database: vectors.flatMap { $0.data },
            vectorCount: n,
            dimension: query.dimension,
            metric: metric,
            scores: &scores,
            aux: aux,
            config: config
        )

        // Select top-k
        return selectTopK(scores, k: k, metric: metric)
    }
}
```

Coding Guidelines

**Performance Best Practices**:
- Always use specialized variants (scoreBlockIP, etc.) when metric is statically known
- Pre-compute auxiliary data during index construction, not per-query
- Enable parallel scoring for n > 10,000 vectors
- Reuse score buffers across queries when possible
- Use batch scoring API for multiple queries against same database

**API Usage Patterns**:
```swift
// Good: Metric known at compile time
scoreBlockIP(query, database, n, d, scores)

// Acceptable: Metric from configuration
scoreBlock(query, database, n, d, metric, scores, aux)

// Bad: Computing auxiliary data per query (for cosine)
for query in queries {
    let norms = precomputeInvNorms(database, n, d)  // Don't do this!
    scoreBlock(query, database, n, d, .cosine, scores, AuxiliaryData(norms, 0))
}

// Good: Pre-compute once, reuse
let dbNorms = precomputeInvNorms(database, n, d)
for query in queries {
    let queryNorm = computeQueryInvNorm(query, d)
    scoreBlock(query, database, n, d, .cosine, scores, AuxiliaryData(dbNorms, queryNorm))
}
```

**Error Handling**:
- Cosine without auxiliary data: Fatal error (programming error, not runtime)
- Dimension mismatch: Assert in debug, undefined in release
- Invalid metric enum: Compiler enforced (exhaustive switch)

**Memory Management**:
- Caller allocates score buffer (no hidden allocations)
- Temporary buffers (for L2 norm optimization) allocated on stack
- No reference counting or ARC overhead

Non-Goals

- Top-k selection (separate kernel or application logic)
- Multi-vector queries (process independently or use batch API)
- Approximate scoring (quantization handled by separate kernels)
- GPU/Metal acceleration (separate Metal kernels)
- Distance matrix computation (use batch API + application logic)
- Weighted/transformed metrics (application-level)

Example Usage

```swift
import VectorIndex

// Example 1: IVF cell scoring with cosine metric
let query = Vector(data: [Float](repeating: 0.5, count: 768))
let cellVectors = loadCell(cellID: 42)  // [[Float]] of length n
let cellInvNorms = loadCellNorms(cellID: 42)  // [Float] of length n

let queryInvNorm = computeQueryInvNorm_f32(query: query.data, dimension: 768)
var scores = [Float](repeating: 0, count: cellVectors.count)

scoreBlock(
    query: query.data,
    database: cellVectors.flatMap { $0 },
    vectorCount: cellVectors.count,
    dimension: 768,
    metric: .cosine,
    scores: &scores,
    aux: AuxiliaryData(databaseAux: cellInvNorms, queryAux: queryInvNorm)
)

// Example 2: Specialized IP scoring (metric known at compile time)
var ipScores = [Float](repeating: 0, count: n)
scoreBlockIP(
    query: query.data,
    database: database,
    vectorCount: n,
    dimension: d,
    scores: &ipScores
)

// Example 3: L2 with norm optimization
let dbSquaredNorms = precomputeSquaredNorms(database, n, d)
let querySquaredNorm = l2NormSquared_f32(query.data, d)

scoreBlockL2(
    query: query.data,
    database: database,
    vectorCount: n,
    dimension: d,
    scores: &scores,
    dbSquaredNorms: dbSquaredNorms,
    querySquaredNorm: querySquaredNorm
)

// Example 4: Batch scoring for multiple queries
let queries = [[Float]](repeating: [Float](repeating: 0.5, count: 768), count: 10)
let database = loadDatabase()  // [Float] of length n*d

var batchScores = [Float](repeating: 0, count: queries.count * n)

scoreBatch(
    queries: queries.flatMap { $0 },
    database: database,
    numQueries: queries.count,
    vectorCount: n,
    dimension: 768,
    metric: .l2,
    scores: &batchScores
)

// Access scores for query i, vector j:
let score = batchScores[i * n + j]

// Example 5: Parallel scoring for large database
let config = ScoreBlockConfig(
    enableTelemetry: true,
    enableParallelScoring: true,
    parallelThreshold: 5000
)

var parallelScores = [Float](repeating: 0, count: n)

scoreBlock(
    query: query.data,
    database: largeDatabase,
    vectorCount: n,  // n = 1,000,000
    dimension: 768,
    metric: .cosine,
    scores: &parallelScores,
    aux: aux,
    config: config
)

// Example 6: High-level convenience API
let scores = ScoreBlockKernel.score(
    query: [0.5, 0.3, ...],
    database: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    metric: .cosine
)
// Automatically handles memory layout and auxiliary data

// Example 7: Score and select top-k in one operation
let topK = ScoreBlockKernel.scoreTopK(
    query: [0.5, 0.3, ...],
    database: database,
    k: 10,
    metric: .l2
)
// Returns [(index: Int, score: Float)] sorted by score
```

Mathematical Foundation

**Metric Definitions**:

1. **L2 Distance** (Euclidean):
   ```
   d_L2(q, x) = ‖q - x‖₂ = √(Σᵢ (qᵢ - xᵢ)²)
   ```
   - Range: [0, ∞)
   - Lower is better (distance)
   - Identity: ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩

2. **Inner Product**:
   ```
   s_IP(q, x) = ⟨q, x⟩ = Σᵢ qᵢ·xᵢ
   ```
   - Range: (-∞, ∞)
   - Higher is better (similarity)
   - Used in MIPS (Maximum Inner Product Search)

3. **Cosine Similarity**:
   ```
   s_cos(q, x) = ⟨q, x⟩ / (‖q‖₂ · ‖x‖₂)
   ```
   - Range: [-1, 1]
   - Higher is better (similarity)
   - Magnitude-invariant

**Computational Complexity**:
```
Metric         | Time        | Space (aux)
---------------|-------------|-------------
Inner Product  | O(n·d)      | 0
L2 (direct)    | O(n·d)      | 0
L2 (optimized) | O(n·d + n)  | 4n bytes
Cosine         | O(n·d + n)  | 4n or 2n bytes
```

Dependencies

**Internal**:
- L2 Distance Kernel (#01): `l2DistanceBlock_f32`
- Inner Product Kernel (#02): `innerProductBlock_f32`
- Cosine Similarity Kernel (#03): `cosineBlock_f32`
- Norm Kernel (#09): `computeQueryInvNorm_f32`, `l2NormSquared_f32`
- Prefetch Helpers (#49): Cache optimization utilities
- Telemetry (#46): Performance instrumentation

**External**:
- Swift Standard Library: `SIMD4<Float>`, `UnsafePointer`
- Foundation: `ProcessInfo` for thread count
- Dispatch: `DispatchQueue.concurrentPerform` for parallelism

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Performance**:
- Zero overhead abstraction: within 2% of specialized kernels
- Parallel scaling: >90% efficiency on 8 cores for n>10k
- Dispatch overhead: <0.1% of total time

✅ **Correctness**:
- Bit-exact match with specialized kernels
- Matches NumPy/SciPy reference within 1e-5
- Parallel and serial produce identical results

✅ **Coverage**:
- All three metrics (L2, IP, Cosine) tested
- All dimension specializations (512, 768, 1024, 1536)
- All block sizes (1 to 1M+ vectors)
- All edge cases (zeros, orthogonal, identical)

✅ **Integration**:
- Successfully used by IVF search (#05)
- Compatible with HNSW navigation (#29)
- Works with flat brute-force search
- Supports batch and parallel workflows

✅ **Usability**:
- Unified API reduces code duplication
- Clear separation of pre-computation and query-time
- High-level convenience API for simple cases
- Low-level pointer API for performance-critical paths
- Comprehensive documentation with integration examples

✅ **Flexibility**:
- Supports optional auxiliary data optimization
- Configurable parallelization thresholds
- Telemetry integration for profiling
- Compatible with future metric extensions
<!-- moved to docs/kernel-specs/ -->
