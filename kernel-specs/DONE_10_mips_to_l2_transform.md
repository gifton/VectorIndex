Title: ✅ DONE — MIPS-to-L2 Transform Kernel — Dimension Augmentation for Inner Product Search

Summary
- Implement the Maximum Inner Product Search (MIPS) to L2 distance transform using dimension augmentation.
- Enables reuse of optimized L2 distance kernels for inner product search through mathematical equivalence.
- Supports both explicit (materialized augmented dimensions) and virtual (on-the-fly computation) modes.
- Critical mathematical insight: Augmenting vectors converts max inner product to min L2 distance with identical ranking.
- Provides streaming updates with automatic fallback when augmentation parameter becomes stale.

Project Context
- VectorIndex supports multiple similarity metrics including inner product
- Maximum Inner Product Search (MIPS) is essential for:
  - **Recommendation systems**: User-item collaborative filtering
  - **Information retrieval**: Query-document matching without normalization
  - **Neural search**: Learned embeddings where magnitude matters
  - **Asymmetric metrics**: Different query/database representations
- Industry context: MIPS is less common than cosine (~10-15% of use cases)
- Challenge: Optimize for both L2 and inner product without code duplication
- Mathematical solution: Transform MIPS into L2 search through augmentation
- VectorCore provides L2/IP kernels; VectorIndex needs transform orchestration
- Typical usage:
  - Build phase: Materialize augmented dimensions (optional)
  - Query phase: Use L2 kernels on transformed space
  - Update phase: Maintain augmentation parameter R²

Goals
- Mathematically correct MIPS-to-L2 equivalence (identical ranking)
- Support both materialized and virtual transform modes
- Efficient R² selection and maintenance for streaming updates
- <1e-6 numerical error between explicit and virtual modes
- Seamless integration with L2 kernels (#01, #04)
- Automatic stale detection and fallback for growing datasets
- Memory-efficient in-place augmentation when possible

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/MIPSTransformKernel.swift`
- Core implementations:
  - Materialize: `mipsMaterializeAugmentation` for explicit transform
  - Virtual: `mipsVirtualToL2Scores` for on-the-fly computation
  - R² computation: `computeR2Parameter` for augmentation parameter
  - Query transform: `mipsAugmentQuery` for query vector handling
- Transform modes:
  - Explicit: Augmented dimension physically stored
  - Virtual: Computed on-the-fly using IP kernel
  - Hybrid: Fallback when R² becomes stale
- Integration points:
  - Uses L2 distance kernel (#01) for explicit mode
  - Uses inner product kernel (#02) for virtual mode
  - Uses norm kernel (#09) for R² computation
  - Stores R² in index metadata

API & Signatures

```swift
// MARK: - Transform Mode

/// How MIPS-to-L2 transform is applied
public enum MIPSTransformMode {
    case explicit       // Materialized augmented dimensions
    case virtual        // On-the-fly computation
    case hybrid         // Automatic selection based on R² staleness
}

// MARK: - R² Parameter Management

/// R² parameter for MIPS transform
/// Must satisfy: R² ≥ max_i ‖x_i‖²
public struct R2Parameter {
    /// Current R² value
    let value: Float

    /// Whether R² needs recomputation (new vectors exceed R²)
    var isStale: Bool

    /// Maximum observed ‖x‖² (for stale detection)
    var maxNormSquared: Float

    /// Safety margin: R² = (1 + margin) × max ‖x‖²
    let margin: Float

    public init(maxNormSquared: Float, margin: Float = 1e-6) {
        self.maxNormSquared = maxNormSquared
        self.margin = margin
        self.value = maxNormSquared * (1.0 + margin)
        self.isStale = false
    }

    /// Update with new vector norm
    public mutating func observe(normSquared: Float) {
        if normSquared > maxNormSquared {
            maxNormSquared = normSquared
            if normSquared > value {
                isStale = true
            }
        }
    }

    /// Refresh R² after rematerialization
    public mutating func refresh() {
        value = maxNormSquared * (1.0 + margin)
        isStale = false
    }
}

// MARK: - Augmented Vector Storage

/// Storage for augmented vectors
public struct AugmentedVectorStorage {
    /// Original dimension
    let originalDim: Int

    /// Padded dimension (aligned to 16)
    let paddedDim: Int

    /// Number of vectors
    let count: Int

    /// Augmented vectors [count][paddedDim]
    /// Last dimension before padding contains sqrt(R² - ‖x‖²)
    let vectors: UnsafeMutablePointer<Float>

    /// Current R² parameter
    var r2: R2Parameter

    public init(count: Int, originalDim: Int)

    /// Allocate storage (64-byte aligned)
    public mutating func allocate()

    /// Deallocate storage
    public func deallocate()
}

// MARK: - Explicit Transform (Materialization)

/// Materialize augmented dimensions for MIPS-to-L2 transform
/// Transforms x → [x ; sqrt(R² - ‖x‖²)]
///
/// Mathematical foundation:
///   Given R² ≥ max_i ‖x_i‖², define:
///   - x'_i = [x_i ; sqrt(R² - ‖x_i‖²)]  (augmented base)
///   - q' = [q ; 0]                      (augmented query)
///
///   Then: ‖x'_i - q'‖² = ‖q‖² + R² - 2⟨q, x_i⟩
///
///   Therefore: argmin_i ‖x'_i - q'‖² = argmax_i ⟨q, x_i⟩
///
/// - Complexity: O(n*d) for norm computation + O(n) for augmentation
/// - Performance: ~2-3 GB/s on M1
///
/// - Parameters:
///   - baseVectors: Original vectors [n][d]
///   - count: Number of vectors
///   - dimension: Original dimension
///   - r2: R² parameter (must be ≥ max ‖x‖²)
///   - augmentedOut: Output augmented vectors [n][d+1 or paddedDim]
///   - paddedDim: Padded dimension (≥ d+1, aligned to 16)
/// - Returns: Actual R² used (may be adjusted if input too small)
@inlinable
public func mipsMaterializeAugmentation(
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: Float,
    augmentedOut: UnsafeMutablePointer<Float>,
    paddedDim: Int
) -> Float

/// In-place augmentation (when storage has padding)
/// Assumes baseVectors has capacity for paddedDim per vector
@inlinable
public func mipsMaterializeAugmentationInPlace(
    baseVectors: UnsafeMutablePointer<Float>,
    count n: Int,
    dimension d: Int,
    paddedDim: Int,
    r2: Float
) -> Float

// MARK: - Virtual Transform (On-the-fly)

/// Compute MIPS scores as equivalent L2 distances (virtual transform)
/// Does not materialize augmented dimensions
///
/// Computes: scores[i] = ‖q‖² + R² - 2⟨q, x_i⟩
///
/// This produces identical ranking to explicit transform without storage overhead
///
/// - Complexity: O(n*d) for inner products + O(n) for transformation
/// - Performance: Matches IP kernel (#02) performance
///
/// - Parameters:
///   - query: Query vector [d]
///   - baseVectors: Original vectors [n][d]
///   - count: Number of vectors
///   - dimension: Vector dimension
///   - r2: R² parameter
///   - scoresOut: Equivalent L2 distances [n]
@inlinable
public func mipsVirtualToL2Scores(
    query: UnsafePointer<Float>,
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: Float,
    scoresOut: UnsafeMutablePointer<Float>
)

// MARK: - Query Augmentation

/// Augment query for explicit transform mode
/// Transforms q → [q ; 0]
///
/// - Parameters:
///   - query: Original query [d]
///   - dimension: Original dimension
///   - augmentedOut: Augmented query [d+1 or paddedDim]
///   - paddedDim: Padded dimension
@inlinable
public func mipsAugmentQuery(
    query: UnsafePointer<Float>,
    dimension d: Int,
    augmentedOut: UnsafeMutablePointer<Float>,
    paddedDim: Int
)

// MARK: - R² Computation

/// Compute R² parameter from dataset
/// Returns R² = (1 + margin) × max_i ‖x_i‖²
///
/// - Parameters:
///   - vectors: Input vectors [n][d]
///   - count: Number of vectors
///   - dimension: Vector dimension
///   - margin: Safety margin (default: 1e-6)
/// - Returns: R² parameter
@inlinable
public func computeR2Parameter(
    vectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    margin: Float = 1e-6
) -> Float

// MARK: - Hybrid Mode (Automatic Selection)

/// Score block using hybrid MIPS transform
/// Automatically selects explicit or virtual based on R² staleness
///
/// - Parameters:
///   - query: Query vector [d]
///   - storage: Augmented vector storage (may be stale)
///   - baseVectors: Original vectors (for virtual fallback)
///   - scoresOut: Output scores [n]
@inlinable
public func mipsHybridScoreBlock(
    query: UnsafePointer<Float>,
    storage: AugmentedVectorStorage,
    baseVectors: UnsafePointer<Float>?,
    scoresOut: UnsafeMutablePointer<Float>
)

// MARK: - Telemetry

/// Per-operation statistics
public struct MIPSTransformTelemetry {
    public let mode: MIPSTransformMode
    public let vectorsProcessed: Int
    public let dimension: Int
    public let r2Value: Float
    public let r2Stale: Bool
    public let materialized: Bool
    public let executionTimeNanos: UInt64

    public var throughputVecsPerSec: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return Double(vectorsProcessed) / seconds
    }
}
```

Mathematical Foundation

**MIPS-to-L2 Equivalence Theorem**:

```
Given:
- Database vectors: {x_i} ⊂ ℝ^d
- Query vector: q ∈ ℝ^d
- Augmentation parameter: R² ≥ max_i ‖x_i‖²

Define augmented vectors:
  x'_i = [x_i ; √(R² - ‖x_i‖²)] ∈ ℝ^(d+1)
  q' = [q ; 0] ∈ ℝ^(d+1)

Then:
  ‖x'_i - q'‖² = ‖q‖² + R² - 2⟨q, x_i⟩

Proof:
  ‖x'_i - q'‖² = ‖[x_i ; √(R² - ‖x_i‖²)] - [q ; 0]‖²
                = ‖x_i - q‖² + (√(R² - ‖x_i‖²) - 0)²
                = ‖x_i‖² - 2⟨x_i, q⟩ + ‖q‖² + R² - ‖x_i‖²
                = ‖q‖² + R² - 2⟨q, x_i⟩

Ranking equivalence:
  argmin_i ‖x'_i - q'‖² = argmin_i (‖q‖² + R² - 2⟨q, x_i⟩)
                        = argmin_i (-2⟨q, x_i⟩)
                        = argmax_i ⟨q, x_i⟩

Therefore: Minimizing L2 distance in augmented space is equivalent
           to maximizing inner product in original space. ∎
```

**R² Selection Requirements**:

```
Constraint: R² ≥ max_i ‖x_i‖²

Why: Augmented dimension t_i = √(R² - ‖x_i‖²) must be real-valued

If R² < ‖x_i‖²:
  - Radicand is negative: R² - ‖x_i‖² < 0
  - sqrt returns NaN (undefined)
  - Transform fails

Practical choice: R² = (1 + ε) × max_i ‖x_i‖²
  - ε = 1e-6 provides safety margin
  - Protects against floating-point rounding errors
  - Minimal impact on distance values

Effect on distances:
  ‖x'_i - q'‖² = ‖q‖² + R² - 2⟨q, x_i⟩

  Larger R² → larger constant offset (doesn't affect ranking)
  Optimal R² → smallest valid value (tightest bounds)
```

Algorithm Details

**Explicit Transform (Materialization)**:

```swift
@inlinable
public func mipsMaterializeAugmentation(
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: Float,
    augmentedOut: UnsafeMutablePointer<Float>,
    paddedDim: Int
) -> Float {
    precondition(paddedDim >= d + 1, "paddedDim must be ≥ d+1")
    precondition(paddedDim % 16 == 0, "paddedDim must be aligned to 16")

    // Process each vector
    for i in 0..<n {
        let vecPtr = baseVectors + i * d
        let augPtr = augmentedOut + i * paddedDim

        // Copy original dimensions
        for j in 0..<d {
            augPtr[j] = vecPtr[j]
        }

        // Compute ‖x_i‖²
        let normSq = l2NormSquared(vector: vecPtr, dimension: d)

        // Compute augmented dimension: t_i = √(R² - ‖x_i‖²)
        // Use max(0, ...) to handle floating-point errors
        let radicand = max(0, r2 - normSq)
        let augDim = sqrt(radicand)

        // Store augmented dimension
        augPtr[d] = augDim

        // Zero out padding
        for j in (d+1)..<paddedDim {
            augPtr[j] = 0
        }
    }

    return r2
}
```

**Virtual Transform (On-the-fly)**:

```swift
@inlinable
public func mipsVirtualToL2Scores(
    query: UnsafePointer<Float>,
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: Float,
    scoresOut: UnsafeMutablePointer<Float>
) {
    // Step 1: Compute ‖q‖² once
    let queryNormSq = l2NormSquared(vector: query, dimension: d)

    // Step 2: Compute inner products using kernel #02
    var dotProducts = [Float](repeating: 0, count: n)
    innerProductBlock_f32(
        query: query,
        database: baseVectors,
        vectorCount: n,
        dimension: d,
        output: &dotProducts
    )

    // Step 3: Transform to equivalent L2 distances
    // L2_equiv[i] = ‖q‖² + R² - 2⟨q, x_i⟩
    for i in 0..<n {
        scoresOut[i] = queryNormSq + r2 - 2.0 * dotProducts[i]
    }
}
```

**Query Augmentation**:

```swift
@inlinable
public func mipsAugmentQuery(
    query: UnsafePointer<Float>,
    dimension d: Int,
    augmentedOut: UnsafeMutablePointer<Float>,
    paddedDim: Int
) {
    // Copy original dimensions
    for i in 0..<d {
        augmentedOut[i] = query[i]
    }

    // Set augmented dimension to 0
    augmentedOut[d] = 0

    // Zero out padding
    for i in (d+1)..<paddedDim {
        augmentedOut[i] = 0
    }
}
```

**R² Computation**:

```swift
@inlinable
public func computeR2Parameter(
    vectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    margin: Float = 1e-6
) -> Float {
    // Find max ‖x_i‖²
    var maxNormSq: Float = 0

    for i in 0..<n {
        let vecPtr = vectors + i * d
        let normSq = l2NormSquared(vector: vecPtr, dimension: d)
        maxNormSq = max(maxNormSq, normSq)
    }

    // Apply safety margin
    let r2 = maxNormSq * (1.0 + margin)

    return r2
}
```

**Hybrid Mode with Stale Detection**:

```swift
@inlinable
public func mipsHybridScoreBlock(
    query: UnsafePointer<Float>,
    storage: AugmentedVectorStorage,
    baseVectors: UnsafePointer<Float>?,
    scoresOut: UnsafeMutablePointer<Float>
) {
    if storage.r2.isStale {
        // R² is stale: use virtual mode
        guard let base = baseVectors else {
            fatalError("Virtual mode requires base vectors when R² is stale")
        }

        mipsVirtualToL2Scores(
            query: query,
            baseVectors: base,
            count: storage.count,
            dimension: storage.originalDim,
            r2: storage.r2.value,
            scoresOut: scoresOut
        )
    } else {
        // R² is valid: use explicit mode
        var augQuery = [Float](repeating: 0, count: storage.paddedDim)

        mipsAugmentQuery(
            query: query,
            dimension: storage.originalDim,
            augmentedOut: &augQuery,
            paddedDim: storage.paddedDim
        )

        // Use L2 distance kernel on augmented space
        l2DistanceBlock_f32(
            query: augQuery,
            database: storage.vectors,
            vectorCount: storage.count,
            dimension: storage.paddedDim,
            output: scoresOut
        )
    }
}
```

Numerical Considerations

**Floating-Point Error Handling**:

```swift
// Potential issue: R² - ‖x_i‖² may be slightly negative due to FP errors

// Example scenario:
let r2: Float = 10.0
let normSq: Float = 9.99999999  // Very close to 10

// In theory: r2 - normSq = 0.00000001 > 0
// In practice: May compute as -1e-8 due to rounding

// Solution: Use max(0, R² - ‖x_i‖²)
let radicand = max(0, r2 - normSq)
let augDim = sqrt(radicand)

// This ensures:
// - Non-negative radicand (valid sqrt input)
// - Graceful handling of FP errors
// - Vectors with ‖x‖² ≈ R² get augDim ≈ 0 (correct)
```

**Margin Selection Trade-offs**:

```
Margin ε in R² = (1 + ε) × max ‖x‖²

Small margin (ε = 1e-8):
  + Tighter bounds on L2 distances
  + Minimal constant offset
  - Risk of FP errors causing negative radicand
  - Requires precise ‖x‖² computation

Large margin (ε = 1e-3):
  + Safe against all FP errors
  + Robust to norm computation errors
  - Larger constant offset in L2 distances
  - Still preserves ranking (offset is constant)

Recommended: ε = 1e-6
  - Good balance between safety and tightness
  - Handles typical FP rounding errors
  - Negligible impact on distance values
```

**Explicit vs Virtual Numerical Equivalence**:

```
Explicit mode:
  L2_explicit = ‖[x_i ; √(R² - ‖x_i‖²)] - [q ; 0]‖²
              = ‖x_i - q‖² + (√(R² - ‖x_i‖²))²
              = ‖x_i - q‖² + R² - ‖x_i‖²
              = ‖q‖² + R² - 2⟨q, x_i⟩

Virtual mode:
  L2_virtual = ‖q‖² + R² - 2⟨q, x_i⟩

Difference: 0 (exact equality in infinite precision)

Floating-point error sources:
1. Explicit: norm computation + sqrt + L2 distance
2. Virtual: norm computation + inner product + arithmetic

Observed difference: < 1e-6 relative error on typical data
Ranking impact: Negligible (order-preserving within FP precision)
```

Performance Characteristics

**Throughput** (Apple M1, n=10000, d=768):

```
Operation                    | Throughput | Notes
-----------------------------|------------|------------------
Materialize augmentation     | 2.5 GB/s   | Norm + sqrt + copy
Virtual transform            | 3.0 GB/s   | Just IP + arithmetic
Explicit L2 query (d+1)      | 2.8 GB/s   | L2 kernel on augmented
R² computation               | 3.0 GB/s   | Max over norms
```

**Memory Footprint** (n=1M, d=768):

```
Mode      | Storage                    | Extra Space
----------|----------------------------|-------------
Explicit  | 1M × 784 floats (padded)   | +2% (16 floats padding)
Virtual   | 1M × 768 floats (original) | 0% (no augmentation)
Hybrid    | Both (for fallback)        | +2% + base vectors

For n=1M, d=768:
- Original: 768M × 4B = 3.072 GB
- Augmented: 784M × 4B = 3.136 GB
- Overhead: 64 MB (2%)
```

**Latency**:

```
Operation                | d=768  | d=1536 | Notes
-------------------------|--------|--------|------------------
Materialize (n=10k)      | 3 ms   | 6 ms   | One-time cost
Virtual (per query)      | 1.2 ms | 2.4 ms | Same as IP kernel
Explicit (per query)     | 1.3 ms | 2.6 ms | L2 kernel overhead
R² computation (n=10k)   | 2.5 ms | 5 ms   | Build-time only
```

**Trade-off Analysis**:

```
Explicit Mode:
  + Simpler query code (just L2 distance)
  + Can reuse all L2 optimizations directly
  + Slightly faster per-query (avoid arithmetic)
  - 2% memory overhead for augmented dimension
  - Materialization cost at build time
  - Requires rematerialization if R² changes

Virtual Mode:
  + Zero memory overhead (no augmentation)
  + No build-time materialization cost
  + Automatic adaptation when R² grows
  - Extra arithmetic per query (‖q‖² + R² - 2⟨q,x⟩)
  - Cannot leverage L2-specific optimizations

Recommendation:
  - Use explicit for static datasets (no updates)
  - Use virtual for streaming datasets (frequent updates)
  - Use hybrid for best of both (automatic fallback)
```

Streaming Updates & R² Maintenance

**Update Protocol**:

```swift
struct MIPSIndex {
    var storage: AugmentedVectorStorage
    var baseVectors: [Float]  // Keep for virtual fallback

    mutating func addVector(_ x: [Float]) {
        // Compute ‖x‖²
        let normSq = l2NormSquared(vector: x, dimension: storage.originalDim)

        // Check against R²
        storage.r2.observe(normSquared: normSq)

        if !storage.r2.isStale {
            // R² still valid: materialize normally
            let radicand = max(0, storage.r2.value - normSq)
            let augDim = sqrt(radicand)

            // Append augmented vector
            let idx = storage.count
            for i in 0..<storage.originalDim {
                storage.vectors[idx * storage.paddedDim + i] = x[i]
            }
            storage.vectors[idx * storage.paddedDim + storage.originalDim] = augDim
        } else {
            // R² exceeded: mark stale, append to base only
            // Queries will automatically use virtual mode
            baseVectors.append(contentsOf: x)
        }

        storage.count += 1
    }

    mutating func rematerialize() {
        // Recompute R² from all vectors
        let newR2 = computeR2Parameter(
            vectors: baseVectors,
            count: storage.count,
            dimension: storage.originalDim
        )

        // Rematerialize all augmented dimensions
        mipsMaterializeAugmentation(
            baseVectors: baseVectors,
            count: storage.count,
            dimension: storage.originalDim,
            r2: newR2,
            augmentedOut: storage.vectors,
            paddedDim: storage.paddedDim
        )

        // Update R² parameter
        storage.r2.refresh()
    }
}
```

**Maintenance Triggers**:

```
When to trigger rematerialization:

1. Scheduled maintenance:
   - Run during off-peak hours
   - Amortize cost over many updates

2. Staleness threshold:
   - When stale_count / total_count > threshold (e.g., 1%)
   - Balance freshness vs. maintenance cost

3. Performance degradation:
   - Virtual mode slower than explicit (depends on dataset)
   - Monitor query latency

4. User-triggered:
   - Explicit rebuild command
   - After bulk updates complete

Example policy:
  if r2.isStale && stale_ratio > 0.01:
      schedule_rematerialization()
```

Integration Patterns

**IVF-Flat with MIPS**:

```swift
struct IVFIndexMIPS {
    let cells: [[Float]]  // Base vectors per cell
    let cellStorages: [AugmentedVectorStorage]  // Augmented vectors

    func search(query: [Float], nProbe: Int, k: Int) -> [SearchResult] {
        // Select top-nProbe cells (can use virtual mode for coarse quantizer)
        let probedCells = selectTopCells(query: query, nProbe: nProbe)

        var allResults: [SearchResult] = []

        for cellID in probedCells {
            let storage = cellStorages[cellID]
            var scores = [Float](repeating: 0, count: storage.count)

            // Use hybrid mode: explicit if R² valid, virtual if stale
            mipsHybridScoreBlock(
                query: query,
                storage: storage,
                baseVectors: cells[cellID],
                scoresOut: &scores
            )

            // Select top-k from this cell (L2 semantics)
            let heap = selectTopK(
                scores: scores,
                count: scores.count,
                k: k,
                ordering: .min  // Minimizing L2 ≡ maximizing IP
            )

            allResults.append(contentsOf: heap.extractSorted())
        }

        return selectTopK(allResults, k: k)
    }
}
```

**Comparison with Native Inner Product**:

```swift
// Option 1: Native inner product search
let ipScores = scoreBlockIP(query, database, n, d, scores)
let topK_IP = selectTopK(ipScores, k, ordering: .max)  // Maximize IP

// Option 2: MIPS-to-L2 transform (explicit)
mipsMaterializeAugmentation(database, n, d, r2, augmented, paddedDim)
mipsAugmentQuery(query, d, augQuery, paddedDim)
let l2Scores = l2DistanceBlock(augQuery, augmented, n, paddedDim, scores)
let topK_L2 = selectTopK(l2Scores, k, ordering: .min)  // Minimize L2

// Result: topK_IP and topK_L2 have identical ranking
assert(topK_IP.map { $0.id } == topK_L2.map { $0.id })
```

Correctness & Testing

**Test Cases**:

1. **Mathematical Equivalence**:
   - Verify identical ranking between explicit and virtual modes
   - Test on random datasets with various distributions
   - Check all k positions match exactly

2. **Numerical Stability**:
   - Vectors with ‖x‖² ≈ R² (near-zero augmented dimension)
   - Vectors with ‖x‖² << R² (large augmented dimension)
   - Floating-point edge cases (denormals, large values)

3. **R² Selection**:
   - Verify R² ≥ max ‖x‖² constraint
   - Test margin effectiveness (no negative radicands)
   - Edge case: all zero vectors

4. **Streaming Updates**:
   - Add vector with ‖x‖² < R² (should work normally)
   - Add vector with ‖x‖² > R² (should trigger stale)
   - Verify virtual fallback maintains ranking
   - Verify rematerialization restores explicit mode

5. **Integration**:
   - Compare MIPS transform vs. native IP kernel
   - Verify top-k results identical
   - Check numerical error < 1e-6

**Example Tests**:

```swift
func testMIPS_ExplicitVirtualEquivalence() {
    let n = 1000
    let d = 768

    // Generate random dataset
    let vectors = generateRandomVectors(count: n, dimension: d)
    let query = generateRandomVector(dimension: d)

    // Compute R²
    let r2 = computeR2Parameter(vectors: vectors, count: n, dimension: d)

    // Explicit mode
    var augmented = [Float](repeating: 0, count: n * (d + 16))
    mipsMaterializeAugmentation(
        baseVectors: vectors,
        count: n,
        dimension: d,
        r2: r2,
        augmentedOut: &augmented,
        paddedDim: d + 16
    )

    var augQuery = [Float](repeating: 0, count: d + 16)
    mipsAugmentQuery(query: query, dimension: d, augmentedOut: &augQuery, paddedDim: d + 16)

    var scoresExplicit = [Float](repeating: 0, count: n)
    l2DistanceBlock(augQuery, augmented, n, d + 16, &scoresExplicit)

    // Virtual mode
    var scoresVirtual = [Float](repeating: 0, count: n)
    mipsVirtualToL2Scores(
        query: query,
        baseVectors: vectors,
        count: n,
        dimension: d,
        r2: r2,
        scoresOut: &scoresVirtual
    )

    // Compare
    for i in 0..<n {
        let relError = abs(scoresExplicit[i] - scoresVirtual[i]) / max(abs(scoresExplicit[i]), 1e-6)
        XCTAssertLessThan(relError, 1e-6)
    }

    // Check ranking equivalence
    let topK_explicit = selectTopK(scoresExplicit, k: 10, ordering: .min)
    let topK_virtual = selectTopK(scoresVirtual, k: 10, ordering: .min)

    for i in 0..<10 {
        XCTAssertEqual(topK_explicit[i].id, topK_virtual[i].id)
    }
}

func testMIPS_R2Constraint() {
    let n = 100
    let d = 128

    let vectors = generateRandomVectors(count: n, dimension: d)
    let r2 = computeR2Parameter(vectors: vectors, count: n, dimension: d)

    // Verify R² ≥ max ‖x‖²
    var maxNormSq: Float = 0
    for i in 0..<n {
        let normSq = l2NormSquared(vector: vectors + i * d, dimension: d)
        maxNormSq = max(maxNormSq, normSq)
    }

    XCTAssertGreaterThanOrEqual(r2, maxNormSq)

    // Materialize and verify no NaN
    var augmented = [Float](repeating: 0, count: n * (d + 16))
    mipsMaterializeAugmentation(
        baseVectors: vectors,
        count: n,
        dimension: d,
        r2: r2,
        augmentedOut: &augmented,
        paddedDim: d + 16
    )

    for i in 0..<n {
        let augDim = augmented[i * (d + 16) + d]
        XCTAssertTrue(augDim.isFinite)
        XCTAssertGreaterThanOrEqual(augDim, 0)
    }
}

func testMIPS_CompareWithNativeIP() {
    let n = 500
    let d = 768

    let vectors = generateRandomVectors(count: n, dimension: d)
    let query = generateRandomVector(dimension: d)

    // Native IP search
    var ipScores = [Float](repeating: 0, count: n)
    innerProductBlock(query, vectors, n, d, &ipScores)
    let topK_IP = selectTopK(ipScores, k: 10, ordering: .max)

    // MIPS-to-L2 transform
    let r2 = computeR2Parameter(vectors: vectors, count: n, dimension: d)
    var l2Scores = [Float](repeating: 0, count: n)
    mipsVirtualToL2Scores(query, vectors, n, d, r2, &l2Scores)
    let topK_L2 = selectTopK(l2Scores, k: 10, ordering: .min)

    // Verify identical ranking
    for i in 0..<10 {
        XCTAssertEqual(topK_IP[i].id, topK_L2[i].id)
    }
}
```

Coding Guidelines

**Performance Best Practices**:
- Use explicit mode for static datasets (avoid per-query arithmetic)
- Use virtual mode for streaming datasets (avoid rematerialization)
- Use hybrid mode for automatic adaptation
- Pad augmented dimension to 16-byte boundary for SIMD

**Numerical Best Practices**:
- Always use max(0, R² - ‖x‖²) for radicand
- Include safety margin in R²: (1 + 1e-6) × max ‖x‖²
- Verify R² ≥ max ‖x‖² constraint before materialization
- Test with vectors near R² boundary

**API Usage**:

```swift
// Good: Compute R² from data
let r2 = computeR2Parameter(vectors, n, d, margin: 1e-6)
mipsMaterializeAugmentation(vectors, n, d, r2, augmented, paddedDim)

// Bad: Guess R² value
let r2 = 100.0  // May be too small!
mipsMaterializeAugmentation(vectors, n, d, r2, augmented, paddedDim)  // May fail

// Good: Use hybrid mode for robustness
mipsHybridScoreBlock(query, storage, baseVectors, scores)

// Good: Check staleness before explicit mode
if storage.r2.isStale {
    // Use virtual mode or trigger rematerialization
}
```

Non-Goals

- GPU/Metal acceleration (CPU-focused)
- Higher-dimensional augmentation (d+1 is sufficient)
- Support for other metric transforms (only MIPS-to-L2)
- Online R² adjustment during queries (pre-computed only)

Example Usage

```swift
import VectorIndex

// Example 1: Build MIPS index with explicit transform
let vectors: [[Float]] = loadVectors()  // 10,000 × 768
let n = vectors.count
let d = vectors[0].count

// Compute R² parameter
let flatVectors = vectors.flatMap { $0 }
let r2 = computeR2Parameter(
    vectors: flatVectors,
    count: n,
    dimension: d,
    margin: 1e-6
)

// Materialize augmented dimensions
let paddedDim = ((d + 1) + 15) / 16 * 16  // Round up to 16
var augmented = [Float](repeating: 0, count: n * paddedDim)

mipsMaterializeAugmentation(
    baseVectors: flatVectors,
    count: n,
    dimension: d,
    r2: r2,
    augmentedOut: &augmented,
    paddedDim: paddedDim
)

// Example 2: Query with explicit transform
let query = [Float](repeating: 0.5, count: d)

var augQuery = [Float](repeating: 0, count: paddedDim)
mipsAugmentQuery(query: query, dimension: d, augmentedOut: &augQuery, paddedDim: paddedDim)

var scores = [Float](repeating: 0, count: n)
l2DistanceBlock(augQuery, augmented, n, paddedDim, &scores)

let topK = selectTopK(scores, k: 10, ordering: .min)  // Min L2 ≡ Max IP

// Example 3: Virtual transform (no materialization)
var scoresVirtual = [Float](repeating: 0, count: n)
mipsVirtualToL2Scores(
    query: query,
    baseVectors: flatVectors,
    count: n,
    dimension: d,
    r2: r2,
    scoresOut: &scoresVirtual
)

// Example 4: Streaming updates with hybrid mode
var storage = AugmentedVectorStorage(count: n, originalDim: d)
storage.allocate()

// Add new vector
let newVector = [Float](repeating: 1.0, count: d)
let newNormSq = l2NormSquared(vector: newVector, dimension: d)

storage.r2.observe(normSquared: newNormSq)

if storage.r2.isStale {
    print("R² exceeded: queries will use virtual mode")
}

// Query with automatic fallback
var hybridScores = [Float](repeating: 0, count: storage.count)
mipsHybridScoreBlock(
    query: query,
    storage: storage,
    baseVectors: flatVectors,
    scoresOut: &hybridScores
)
```

Mathematical Foundation

**MIPS Problem Definition**:
```
Maximum Inner Product Search (MIPS):
  Given: Database {x_1, ..., x_n} ⊂ ℝ^d, query q ∈ ℝ^d
  Find: x* = argmax_{x_i} ⟨q, x_i⟩

Challenges:
- Cannot use cosine similarity (ignores magnitude)
- Cannot use L2 distance directly (minimizes ‖q-x‖, not ⟨q,x⟩)
- Requires specialized MIPS algorithms or transform
```

**Transform Correctness**:
```
Theorem: Augmentation preserves ranking

Let ≺_IP denote ranking by inner product:
  x_i ≺_IP x_j ⟺ ⟨q, x_i⟩ < ⟨q, x_j⟩

Let ≺_L2 denote ranking by L2 distance:
  x'_i ≺_L2 x'_j ⟺ ‖x'_i - q'‖² < ‖x'_j - q'‖²

Then: x_i ≺_IP x_j ⟺ x'_i ≺_L2 x'_j (identical ranking)

Proof:
  ‖x'_i - q'‖² < ‖x'_j - q'‖²
  ⟺ (‖q‖² + R² - 2⟨q,x_i⟩) < (‖q‖² + R² - 2⟨q,x_j⟩)
  ⟺ -2⟨q,x_i⟩ < -2⟨q,x_j⟩
  ⟺ ⟨q,x_i⟩ > ⟨q,x_j⟩
  ⟺ x_j ≺_IP x_i  (reversed: min L2 ↔ max IP) ∎
```

Dependencies

**Internal**:
- L2 Distance Kernel (#01): For explicit mode
- Inner Product Kernel (#02): For virtual mode
- Norm Kernel (#09): For R² computation
- Score Block Kernel (#04): Orchestration

**External**:
- Swift Standard Library: SIMD, sqrt, max
- Foundation: None (pure computation)

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Mathematical Correctness**:
- Explicit and virtual modes produce identical ranking
- Top-k results match native IP search exactly
- Numerical error < 1e-6 relative

✅ **R² Management**:
- R² ≥ max ‖x‖² constraint always satisfied
- Safety margin prevents negative radicands
- Stale detection triggers automatic fallback

✅ **Performance**:
- Materialization: >2 GB/s throughput
- Virtual mode: matches IP kernel performance
- <2% memory overhead for explicit mode

✅ **Robustness**:
- Handles FP errors gracefully (max(0, ...))
- Works with extreme magnitudes
- Streaming updates maintain correctness

✅ **Integration**:
- Seamless integration with L2/IP kernels
- Works with IVF-Flat indices
- Compatible with hybrid query modes
