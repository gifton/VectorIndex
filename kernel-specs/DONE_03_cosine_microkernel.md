Title: ✅ DONE — Cosine Similarity Microkernel — Normalized Angular Distance for Vector Search

Summary
- Implement a high-performance cosine similarity kernel that computes normalized angular distance between query and database vectors. Builds on the inner-product microkernel (#02) and norm computation (#09) to provide the most common similarity metric in embedding-based search.
- Supports pre-computed inverse norms for database vectors (f32 or f16 storage) to minimize runtime overhead.
- Handles numerical edge cases (zero norms, near-parallel vectors) with configurable epsilon.
- Achieves >80% of memory bandwidth peak through kernel fusion and SIMD optimization.

Project Context
- VectorIndex provides search operations where cosine similarity is the dominant metric
- Cosine similarity is the standard metric for:
  - **Semantic search**: BERT, Sentence-BERT, OpenAI embeddings (normalized by default)
  - **Recommendation systems**: User/item embeddings with varying magnitudes
  - **Document clustering**: TF-IDF vectors with different document lengths
  - **Face recognition**: L2-normalized face embeddings
  - **Cross-modal retrieval**: Image-text matching (CLIP, ALIGN)
- Advantage over L2 distance: Magnitude-invariant, better for high-dimensional sparse data
- Industry usage: ~60-70% of vector databases use cosine as primary metric
- Performance critical path:
  - Used by IVF scoring (#04) when metric=cosine
  - Used by HNSW graph search (#29) for neighbor selection
  - Used by re-ranking (#40) for final top-k refinement
- Challenge: Naive implementation requires 3 passes (2 norms + 1 dot) → inefficient
- Solution: Pre-compute and store database inverse norms, compute query norm once
- Memory trade-off: Extra n×4 bytes (f32) or n×2 bytes (f16) for inverse norms
- VectorCore provides primitive operations; VectorIndex needs search-optimized composition

Goals
- Achieve >80% of memory bandwidth peak (higher overhead than raw IP due to scaling)
- Support pre-computed f32 and f16 inverse norms with transparent widening
- Single-pass computation: one kernel call computes all similarities
- Deterministic results for reproducible ranking across runs
- Handle edge cases gracefully: zero norms, denormals, inf/nan
- Fusion opportunity: combine IP + scaling in single pass for cache efficiency
- Thread-safe for concurrent query processing
- Zero allocations beyond output buffer

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/CosineSimilarityKernel.swift`
- Core implementations:
  - Main kernel: `cosineBlock_f32` with runtime dimension
  - Specialized variants: `cosine_f32_d{512,768,1024,1536}` for common dimensions
  - Norm utilities: `computeQueryInvNorm_f32`, `precomputeInvNorms_f32`
  - Half-precision support: f16 inv-norm loading and widening
- Integration points:
  - Uses `innerProductBlock_f32` from (#02) for dot products
  - Uses `l2Norm_f32` from (#09) for query norm computation
  - Exports cosine scores for search (#04), routing (#29), re-ranking (#40)
- Supporting utilities:
  - Epsilon handling for zero-norm protection
  - f16 ↔ f32 conversion helpers
  - Batch norm pre-computation
  - Telemetry integration (#46)

API & Signatures

```swift
// MARK: - Core Cosine Similarity API

/// Compute cosine similarities between query and database vectors
/// Computes: out[i] = ⟨q, xb[i]⟩ / (‖q‖₂ · ‖xb[i]‖₂)
///
/// - Complexity: O(n * d) for dot product + O(n) for scaling
/// - Performance: ~80-85% of peak bandwidth (vs 85-95% for raw IP)
/// - Thread Safety: Reentrant; safe for concurrent calls with disjoint outputs
///
/// - Parameters:
///   - query: Query vector [d], 64-byte aligned
///   - database: Database vectors [n][d], 64-byte aligned, row-major
///   - vectorCount: Number of database vectors (n)
///   - dimension: Vector dimension (d)
///   - dbInvNorms: Pre-computed 1/‖xb[i]‖₂ for each row [n], 64-byte aligned
///   - queryInvNorm: Pre-computed 1/‖q‖₂ (computed once per query)
///   - output: Cosine similarity scores [n], 64-byte aligned
///   - config: Optional configuration (epsilon, telemetry, etc.)
@inlinable
public func cosineBlock_f32(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>,
    config: CosineSimilarityConfig = .default
)

/// Cosine similarity with half-precision inverse norms (memory-efficient)
/// Same semantics as cosineBlock_f32 but loads f16 inv-norms and widens to f32
@inlinable
public func cosineBlock_f32_f16norms(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    dbInvNorms: UnsafePointer<Float16>,  // Half-precision norms
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>,
    config: CosineSimilarityConfig = .default
)

// MARK: - Specialized Fast Paths

/// Cosine similarity for 512-dimensional vectors
@inline(__always)
public func cosine_f32_d512(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>
)

/// Cosine similarity for 768-dimensional vectors
@inline(__always)
public func cosine_f32_d768(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>
)

/// Cosine similarity for 1536-dimensional vectors
@inline(__always)
public func cosine_f32_d1536(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>
)

// MARK: - Query Norm Utilities

/// Compute inverse L2 norm for query vector: 1/(‖q‖₂ + ε)
/// - Returns: 1/sqrt(Σq²) with epsilon protection against zero-norm
@inlinable
public func computeQueryInvNorm_f32(
    query: UnsafePointer<Float>,
    dimension: Int,
    epsilon: Float = 1e-12
) -> Float

/// Batch pre-compute inverse norms for database vectors
/// This should be called once during index construction, not per-query
@inlinable
public func precomputeInvNorms_f32(
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dimension: Int,
    output: UnsafeMutablePointer<Float>,
    epsilon: Float = 1e-12
)

/// Batch pre-compute inverse norms as f16 (50% memory savings)
@inlinable
public func precomputeInvNorms_f16(
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dimension: Int,
    output: UnsafeMutablePointer<Float16>,
    epsilon: Float = 1e-12
)

// MARK: - Configuration

/// Configuration for cosine similarity computation
public struct CosineSimilarityConfig {
    /// Epsilon for zero-norm protection (default: 1e-12)
    let epsilon: Float

    /// Enable telemetry recording (default: false)
    let enableTelemetry: Bool

    /// Verify alignment in debug builds (default: false in release)
    let verifyAlignment: Bool

    /// Clamp output to [-1, 1] to handle floating-point errors (default: true)
    let clampOutput: Bool

    /// Use fused kernel (IP + scaling in one pass) vs two-pass (default: true)
    let useFusedKernel: Bool

    public static let `default` = CosineSimilarityConfig(
        epsilon: 1e-12,
        enableTelemetry: false,
        verifyAlignment: false,
        clampOutput: true,
        useFusedKernel: true
    )
}

// MARK: - Telemetry

/// Per-kernel execution statistics
public struct CosineSimilarityTelemetry {
    public let kernelVariant: String         // "d768", "generic", "fused"
    public let rowsProcessed: Int
    public let bytesRead: Int                // Database + query + norms
    public let usedFusedPath: Bool           // True if IP+scale fused
    public let usedF16Norms: Bool            // True if f16 norms loaded
    public let zeroNormCount: Int            // Number of zero-norm vectors encountered
    public let clampedCount: Int             // Number of outputs clamped to [-1,1]
    public let executionTimeNanos: UInt64

    public var bandwidthGBps: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return (Double(bytesRead) / 1e9) / seconds
    }
}

// MARK: - Convenience API

extension CosineSimilarityKernel {
    /// High-level API: compute cosine similarity with automatic norm handling
    /// Computes query inv-norm on-the-fly if not provided
    public static func compute(
        query: [Float],
        database: [[Float]],
        dbInvNorms: [Float]? = nil,
        epsilon: Float = 1e-12
    ) -> [Float]
}
```

Algorithm Details

**Two-Phase Approach**:

1. **Pre-computation Phase** (once per database):
   ```swift
   // For each database vector xb[i]:
   norm = sqrt(Σⱼ xb[i][j]²)  // L2 norm from kernel #09
   inv_norm[i] = 1 / (norm + ε)
   // Store inv_norm[i] as f32 or f16
   ```

2. **Query Phase** (per query):
   ```swift
   // Step 1: Compute query inverse norm (once per query)
   q_norm = sqrt(Σⱼ q[j]²)
   q_inv_norm = 1 / (q_norm + ε)

   // Step 2: Compute dot products (reuse kernel #02)
   dots = innerProductBlock_f32(q, xb, n, d)

   // Step 3: Scale by product of inverse norms
   for i in 0..<n:
       cosine[i] = dots[i] * q_inv_norm * inv_norm[i]
       // Optional: clamp to [-1, 1] to handle fp errors
       cosine[i] = max(-1.0, min(1.0, cosine[i]))
   ```

**Fused Implementation** (single-pass, cache-efficient):

```swift
@inlinable
public func cosineBlock_f32_fused(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>
) {
    // Fuse IP computation + scaling in single kernel
    // Benefits: Single memory pass over database, better cache utilization

    let vecWidth = 4  // NEON: 4× Float32 per vector
    let dBlocked = (d / vecWidth) * vecWidth

    // Broadcast query inverse norm to SIMD vector
    let qInvNormVec = SIMD4<Float>(repeating: queryInvNorm)

    for i in 0..<n {
        var dotAcc = SIMD4<Float>.zero
        let rowPtr = database + i * d

        // Compute dot product (same as IP kernel)
        for j in stride(from: 0, to: dBlocked, by: vecWidth) {
            let q_vec = SIMD4<Float>(query + j)
            let xb_vec = SIMD4<Float>(rowPtr + j)
            dotAcc += q_vec * xb_vec
        }

        var dot = dotAcc.sum()

        // Scalar tail
        for j in dBlocked..<d {
            dot += query[j] * rowPtr[j]
        }

        // Fused epilogue: scale by both norms
        let dbInvNorm = dbInvNorms[i]
        let cosine = dot * queryInvNorm * dbInvNorm

        // Optional: clamp to [-1, 1]
        output[i] = max(-1.0, min(1.0, cosine))
    }
}
```

**Specialized Implementation** (d=768, vectorized scaling):

```swift
@inline(__always)
public func cosine_f32_d768(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>
) {
    // Step 1: Compute all dot products
    var dots = [Float](repeating: 0, count: n)
    innerProduct_f32_d768_r8(query, database, vectorCount: n, output: &dots)

    // Step 2: Vectorized scaling
    let qInvNormVec = SIMD4<Float>(repeating: queryInvNorm)
    let nBlocked = (n / 4) * 4

    // Process 4 similarities per iteration
    for i in stride(from: 0, to: nBlocked, by: 4) {
        let dotVec = SIMD4<Float>(dots + i)
        let invNormVec = SIMD4<Float>(dbInvNorms + i)

        var cosineVec = dotVec * qInvNormVec * invNormVec

        // Clamp to [-1, 1]
        cosineVec = max(SIMD4(repeating: -1.0), min(SIMD4(repeating: 1.0), cosineVec))

        // Store result
        cosineVec.store(to: output + i)
    }

    // Scalar tail
    for i in nBlocked..<n {
        let cosine = dots[i] * queryInvNorm * dbInvNorms[i]
        output[i] = max(-1.0, min(1.0, cosine))
    }
}
```

**Half-Precision Inverse Norms**:

```swift
@inlinable
public func cosineBlock_f32_f16norms(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    dbInvNorms: UnsafePointer<Float16>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>,
    config: CosineSimilarityConfig = .default
) {
    // Compute dot products
    var dots = [Float](repeating: 0, count: n)
    innerProductBlock_f32(query, database, n, d, &dots)

    // Scale with f16→f32 widening
    let qInvNormVec = SIMD4<Float>(repeating: queryInvNorm)
    let nBlocked = (n / 4) * 4

    for i in stride(from: 0, to: nBlocked, by: 4) {
        let dotVec = SIMD4<Float>(dots + i)

        // Load f16 norms and widen to f32
        let f16_0 = dbInvNorms[i + 0]
        let f16_1 = dbInvNorms[i + 1]
        let f16_2 = dbInvNorms[i + 2]
        let f16_3 = dbInvNorms[i + 3]
        let invNormVec = SIMD4<Float>(
            Float(f16_0), Float(f16_1), Float(f16_2), Float(f16_3)
        )

        var cosineVec = dotVec * qInvNormVec * invNormVec
        cosineVec = max(SIMD4(repeating: -1.0), min(SIMD4(repeating: 1.0), cosineVec))

        cosineVec.store(to: output + i)
    }

    // Scalar tail
    for i in nBlocked..<n {
        let invNorm = Float(dbInvNorms[i])
        let cosine = dots[i] * queryInvNorm * invNorm
        output[i] = max(-1.0, min(1.0, cosine))
    }
}
```

Vectorization Details

**SIMD Optimization Opportunities**:

1. **Dot Product Phase**: Fully vectorized via IP kernel (#02)
   - Already achieves 85-95% memory bandwidth
   - Uses NEON FMA for accumulation

2. **Scaling Phase**: Vectorize the post-processing
   - Load 4 dot products: `SIMD4<Float>(dots + i)`
   - Load 4 inverse norms: `SIMD4<Float>(dbInvNorms + i)`
   - Broadcast query inv-norm: `SIMD4<Float>(repeating: queryInvNorm)`
   - Multiply: `dots * qInvNorm * dbInvNorms` (2 FMUL ops)
   - Clamp: `max(-1, min(1, result))` (2 FCMP + 2 FSEL ops)
   - Store: 4 results at once

3. **Half-Precision Widening**:
   - Load 4× f16 values (64 bits total)
   - Widen to 4× f32 using `Float(_:Float16)` conversion
   - NEON has native `FCVTL` instruction for f16→f32
   - Memory savings: 50% (2 bytes vs 4 bytes per norm)
   - Precision loss: Minimal (~3-4 decimal digits vs 6-7 for f32)

**NEON Intrinsics** (conceptual):
```swift
// Scaling loop with NEON
for i in stride(from: 0, to: n, by: 4) {
    let dots = vld1q_f32(dotPtr + i)          // Load 4 dots
    let norms = vld1q_f32(normPtr + i)        // Load 4 inv-norms
    let qnorm = vdupq_n_f32(queryInvNorm)     // Broadcast query norm

    var result = vmulq_f32(dots, qnorm)       // dots * q_inv_norm
    result = vmulq_f32(result, norms)         // * db_inv_norms[i]

    // Clamp to [-1, 1]
    let minVec = vdupq_n_f32(-1.0)
    let maxVec = vdupq_n_f32(1.0)
    result = vmaxq_f32(result, minVec)
    result = vminq_f32(result, maxVec)

    vst1q_f32(outputPtr + i, result)          // Store 4 results
}
```

**Performance Analysis**:
- Dot product: Memory-bound (dominates runtime)
- Scaling: Compute-bound but negligible (6 FLOPs per element)
- Breakdown: ~95% dot product, ~5% scaling + clamping
- Total bandwidth: ~80-85% of peak (vs 85-95% for raw IP)

Memory Layout & Caching

**Data Layout**:
```
Query:        [d] floats                       // 1× read
Database:     [n][d] floats                    // 1× read (via IP kernel)
DB Inv Norms: [n] floats or [n] f16s          // 1× read (sequential)
Dots:         [n] floats (temp buffer)         // 1× write, 1× read
Output:       [n] floats                       // 1× write

Total reads:  n*d*4 + d*4 + n*4 (or n*2 for f16) bytes
Total writes: n*4 bytes
```

**Cache Optimization**:
1. **Query**: Reused across all n vectors (hot in L1 cache via IP kernel)
2. **Database**: Streamed once via IP kernel (L2/L3 cache)
3. **Inverse Norms**: Sequential access, prefetch-friendly
   - 64-byte alignment → 16 norms per cache line (f32) or 32 norms (f16)
4. **Temporal Locality**: Fused kernel keeps dots in registers, avoiding temp buffer

**Memory Bandwidth Calculation** (d=768, n=1000, f32 norms):
```
Reads:  1000 * 768 * 4 + 768 * 4 + 1000 * 4 = 3,083,072 bytes (~3 MB)
Writes: 1000 * 4 = 4,000 bytes
Total:  3,087,072 bytes

M1 @ 60 GB/s: 3.087 MB / 60 GB/s = 51 μs (theoretical minimum)
Target (85%): 51 / 0.85 = 60 μs
```

Numerical Stability

**Challenge 1: Zero-Norm Vectors**
- Problem: Division by zero if ‖q‖=0 or ‖xb[i]‖=0
- Solution: Add epsilon to denominator
  ```swift
  inv_norm = 1 / (sqrt(sum_squares) + epsilon)
  ```
- Default epsilon: 1e-12 (small enough to not affect normal vectors)
- Result: Zero-norm vectors get inv_norm ≈ 1/ε ≈ 1e12 → cosine → 0 after clamping

**Challenge 2: Floating-Point Errors**
- Problem: Dot product can slightly exceed ‖q‖·‖xb‖ due to rounding
- Example: For parallel vectors, theoretically cosine=1.0, but may get 1.0000001
- Solution: Clamp output to [-1, 1]
  ```swift
  cosine = max(-1.0, min(1.0, dotProduct * qInvNorm * dbInvNorm))
  ```
- Enabled by default via `clampOutput: true` in config

**Challenge 3: Denormals**
- Problem: Very small norms (< 1e-38) become denormal → slow
- Solution: Epsilon prevents denormals in inverse norms
- Flush-to-zero (FTZ) mode can be enabled for further protection

**Challenge 4: Half-Precision Accuracy**
- f16 range: [6e-8, 65504]
- f16 precision: ~3 decimal digits
- Inverse norms typically ∈ [0.5, 2.0] for normalized embeddings → well within f16 range
- Error analysis: f16 introduces ~0.1% relative error in cosine scores
- Acceptable for ranking (top-k order rarely changes)

**Epsilon Configuration**:
```swift
// For normalized embeddings (‖v‖ ≈ 1.0):
let epsilon: Float = 1e-12  // Minimal impact

// For unnormalized embeddings with possible zeros:
let epsilon: Float = 1e-8   // More aggressive protection

// For maximum stability (rare):
let epsilon: Float = 1e-6   // May bias small-norm vectors
```

Telemetry Integration

**Instrumentation**:
```swift
#if ENABLE_TELEMETRY
var telemetry = CosineSimilarityTelemetry(
    kernelVariant: "d768",
    rowsProcessed: n,
    bytesRead: n * d * 4 + d * 4 + n * 4,  // db + query + norms
    usedFusedPath: config.useFusedKernel,
    usedF16Norms: false,
    zeroNormCount: 0,
    clampedCount: 0,
    executionTimeNanos: 0
)

let start = mach_absolute_time()
#endif

cosineBlock_f32(query, database, n, d, dbInvNorms, queryInvNorm, output, config)

#if ENABLE_TELEMETRY
telemetry.executionTimeNanos = mach_absolute_time() - start

// Count clamped values
for i in 0..<n {
    if output[i] <= -1.0 || output[i] >= 1.0 {
        telemetry.clampedCount += 1
    }
}

GlobalTelemetryRecorder.record(telemetry)
#endif
```

Performance Targets (Apple M1/M2/M3, Release Build)

**Throughput** (vectors/second, d=768):
- Fused kernel: > 750,000 vec/s (M1)
- Two-pass kernel: > 700,000 vec/s (extra temp buffer overhead)
- f16 norms: > 800,000 vec/s (less memory traffic)

**Latency** (per-vector, single-threaded):
- d=512: < 0.6 μs
- d=768: < 0.8 μs
- d=1536: < 1.7 μs

**Bandwidth Utilization**:
- Target: ≥80% of peak (vs ≥85% for raw IP)
- Overhead: Scaling phase adds ~5-10% runtime
- f16 norms: Can reach 85% due to reduced memory traffic

**Memory Footprint**:
- f32 inv-norms: 4n bytes extra
- f16 inv-norms: 2n bytes extra (50% savings)
- Example: 1M vectors → 4MB (f32) or 2MB (f16)

**Comparison Benchmarks**:
```
Operation                  | Throughput (M1, d=768, n=10000)
---------------------------|--------------------------------
Raw IP kernel (#02)        | 850,000 vec/s
Cosine (fused, f32 norms)  | 750,000 vec/s  (88% of IP)
Cosine (two-pass, f32)     | 700,000 vec/s  (82% of IP)
Cosine (fused, f16 norms)  | 800,000 vec/s  (94% of IP)
```

Correctness & Testing

**Golden Reference**:
- NumPy: `cosine = np.dot(q, xb[i]) / (np.linalg.norm(q) * np.linalg.norm(xb[i]))`
- SciPy: `scipy.spatial.distance.cosine` (returns 1-cosine, convert)
- Tolerance: ≤1e-5 for f32 norms, ≤1e-4 for f16 norms

**Test Cases**:

1. **Dimension Coverage**:
   - Specialized: d ∈ {512, 768, 1024, 1536}
   - Generic: d ∈ {64, 100, 513, 2000, 4096}

2. **Vector Relationships**:
   - **Identical**: q = xb[i] → cosine = 1.0
   - **Opposite**: q = -xb[i] → cosine = -1.0
   - **Orthogonal**: q ⊥ xb[i] → cosine ≈ 0.0
   - **Random**: Normal distribution → cosine ∈ [-1, 1]

3. **Edge Cases**:
   - **Zero query**: ‖q‖ = 0 → all cosines = 0
   - **Zero database vector**: ‖xb[i]‖ = 0 → cosine[i] = 0
   - **Both zero**: Special case, should return 0
   - **Denormals**: Very small norms (1e-38)
   - **Large norms**: ‖v‖ > 1000 (unnormalized embeddings)

4. **Floating-Point Precision**:
   - **Clamping**: Verify outputs ∈ [-1, 1]
   - **Determinism**: Same input → same bit-exact output
   - **f16 accuracy**: Within 0.1% of f32 results

5. **Scale Testing**:
   - n ∈ {1, 10, 100, 1000, 10000, 100000}
   - Verify linear scaling with n

**Example Test**:
```swift
func testCosineSimilarity_identical() {
    let d = 768
    let n = 100

    // Create identical vectors
    var query = [Float](repeating: 0, count: d)
    for i in 0..<d { query[i] = Float.random(in: -1...1) }

    var database = [Float](repeating: 0, count: n * d)
    for i in 0..<n {
        for j in 0..<d {
            database[i * d + j] = query[j]
        }
    }

    // Pre-compute norms
    let qInvNorm = computeQueryInvNorm_f32(query, d)
    var dbInvNorms = [Float](repeating: 0, count: n)
    precomputeInvNorms_f32(database, n, d, &dbInvNorms)

    // Compute cosine
    var output = [Float](repeating: 0, count: n)
    cosineBlock_f32(query, database, n, d, dbInvNorms, qInvNorm, &output)

    // Verify all cosines ≈ 1.0
    for i in 0..<n {
        XCTAssertEqual(output[i], 1.0, accuracy: 1e-5)
    }
}

func testCosineSimilarity_orthogonal() {
    let d = 768
    let query = [Float](repeating: 1, count: d)
    var candidate = [Float](repeating: 0, count: d)
    candidate[0] = 1  // Orthogonal to query

    let qInvNorm = computeQueryInvNorm_f32(query, d)
    var dbInvNorm: Float = 0
    precomputeInvNorms_f32(candidate, 1, d, &dbInvNorm)

    var output: Float = 0
    cosineBlock_f32(query, candidate, 1, d, &dbInvNorm, qInvNorm, &output)

    // Should be near zero (not exactly due to fp errors)
    XCTAssertEqual(output, 0.0, accuracy: 1e-5)
}

func testCosineSimilarity_f16Accuracy() {
    let d = 768
    let n = 1000

    // Random vectors
    let query = (0..<d).map { _ in Float.random(in: -1...1) }
    let database = (0..<n*d).map { _ in Float.random(in: -1...1) }

    // Compute with f32 norms (ground truth)
    var normsF32 = [Float](repeating: 0, count: n)
    precomputeInvNorms_f32(database, n, d, &normsF32)
    var outputF32 = [Float](repeating: 0, count: n)
    cosineBlock_f32(query, database, n, d, normsF32, queryInvNorm, &outputF32)

    // Compute with f16 norms
    var normsF16 = [Float16](repeating: 0, count: n)
    precomputeInvNorms_f16(database, n, d, &normsF16)
    var outputF16 = [Float](repeating: 0, count: n)
    cosineBlock_f32_f16norms(query, database, n, d, normsF16, queryInvNorm, &outputF16)

    // Verify within 0.1% relative error
    for i in 0..<n {
        let relativeError = abs(outputF32[i] - outputF16[i]) / max(abs(outputF32[i]), 1e-6)
        XCTAssertLessThan(relativeError, 0.001)  // 0.1%
    }
}
```

Integration with VectorCore & VectorIndex

**Dependencies**:
```swift
// Uses IP kernel from #02
import VectorIndex.InnerProductKernel

// Uses norm computation from #09
import VectorIndex.NormKernel

// Example integration
func cosineBlock_f32(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    dbInvNorms: UnsafePointer<Float>,
    queryInvNorm: Float,
    output: UnsafeMutablePointer<Float>,
    config: CosineSimilarityConfig = .default
) {
    if config.useFusedKernel {
        // Fused implementation (single-pass)
        cosineBlock_f32_fused(query, database, n, d, dbInvNorms, queryInvNorm, output)
    } else {
        // Two-pass implementation
        var dots = [Float](repeating: 0, count: n)

        // Step 1: Compute dot products
        innerProductBlock_f32(
            query: query,
            database: database,
            vectorCount: n,
            dimension: d,
            output: &dots
        )

        // Step 2: Scale by norms
        for i in 0..<n {
            let cosine = dots[i] * queryInvNorm * dbInvNorms[i]
            output[i] = config.clampOutput ? max(-1.0, min(1.0, cosine)) : cosine
        }
    }
}
```

**Usage in IVF Scoring** (#04):
```swift
// IVFIndex uses cosine for cell scoring
func scoreCell(
    query: Vector,
    cell: [Vector],
    metric: DistanceMetric
) -> [Float] {
    guard metric == .cosine else {
        // Fall back to other metrics
        return scoreWithL2(query, cell)
    }

    var scores = [Float](repeating: 0, count: cell.count)

    // Pre-compute query norm once
    let qInvNorm = computeQueryInvNorm_f32(query.data, query.dimension)

    // Assume cell vectors have pre-computed norms
    let dbInvNorms = cell.map { $0.invNorm }

    cell.withUnsafeBufferPointer { cellPtr in
        dbInvNorms.withUnsafeBufferPointer { normsPtr in
            cosineBlock_f32(
                query: query.data,
                database: cellPtr.baseAddress!,
                vectorCount: cell.count,
                dimension: query.dimension,
                dbInvNorms: normsPtr.baseAddress!,
                queryInvNorm: qInvNorm,
                output: &scores
            )
        }
    }

    return scores
}
```

**Usage in HNSW Routing** (#29):
```swift
// HNSW uses cosine to find nearest neighbors in graph
func findNearestNeighbors(
    query: Vector,
    candidates: [GraphNode],
    k: Int
) -> [GraphNode] {
    let qInvNorm = computeQueryInvNorm_f32(query.data, query.dimension)

    var scores = [Float](repeating: 0, count: candidates.count)
    let candidateVectors = candidates.map { $0.vector }
    let candidateNorms = candidates.map { $0.invNorm }

    candidateVectors.withUnsafeBufferPointer { vecPtr in
        candidateNorms.withUnsafeBufferPointer { normPtr in
            cosineBlock_f32(
                query: query.data,
                database: vecPtr.baseAddress!,
                vectorCount: candidates.count,
                dimension: query.dimension,
                dbInvNorms: normPtr.baseAddress!,
                queryInvNorm: qInvNorm,
                output: &scores
            )
        }
    }

    // Return top-k by score
    return zip(candidates, scores)
        .sorted { $0.1 > $1.1 }
        .prefix(k)
        .map { $0.0 }
}
```

Coding Guidelines

**Performance Best Practices**:
- Always pre-compute database inverse norms during index construction
- Compute query inverse norm once per query, reuse across all cells/partitions
- Use fused kernel when possible (eliminates temp buffer allocation)
- Use f16 norms for memory-constrained systems (2× memory savings, <0.1% accuracy loss)
- Enable clamping to ensure outputs ∈ [-1, 1]

**Memory Management**:
- No allocations in hot path (caller provides output buffer)
- Temp buffer for dots only needed in two-pass mode
- Pre-allocate inverse norms during index build, not per-query

**Numerical Stability**:
- Always use epsilon in norm computation (default 1e-12)
- Enable clamping by default to handle fp rounding errors
- Test with zero-norm vectors to ensure graceful handling

**API Design**:
- Separate pre-computation (once) from query-time (many times)
- Provide both high-level convenience API and low-level pointer API
- Support both f32 and f16 inverse norms transparently

**Documentation**:
- Explain pre-computation requirement clearly
- Document epsilon behavior and recommended values
- Include memory footprint analysis (f32 vs f16)
- Provide integration examples with search kernels

Non-Goals

- Online norm computation (always pre-compute for performance)
- Other similarity metrics (L2 distance, angular distance) — separate kernels
- Batch query processing (process queries independently)
- GPU/Metal acceleration (separate Metal kernel)
- Unnormalized "cosine" (just use IP kernel if norms not needed)
- Approximate cosine (use exact computation, quantization handled elsewhere)

Example Usage

```swift
import VectorIndex

// Example 1: Basic cosine similarity computation
let d = 768
let n = 1000

let query = [Float](repeating: 0.5, count: d)
let database = (0..<n*d).map { _ in Float.random(in: -1...1) }

// Pre-compute database inverse norms (once during index build)
var dbInvNorms = [Float](repeating: 0, count: n)
precomputeInvNorms_f32(
    database: database,
    vectorCount: n,
    dimension: d,
    output: &dbInvNorms
)

// Compute query inverse norm (once per query)
let queryInvNorm = computeQueryInvNorm_f32(
    query: query,
    dimension: d
)

// Compute cosine similarities
var similarities = [Float](repeating: 0, count: n)
cosineBlock_f32(
    query: query,
    database: database,
    vectorCount: n,
    dimension: d,
    dbInvNorms: dbInvNorms,
    queryInvNorm: queryInvNorm,
    output: &similarities
)

print("Top similarity: \(similarities.max()!)")

// Example 2: Using f16 norms for memory efficiency
var dbInvNormsF16 = [Float16](repeating: 0, count: n)
precomputeInvNorms_f16(
    database: database,
    vectorCount: n,
    dimension: d,
    output: &dbInvNormsF16
)

var similaritiesF16 = [Float](repeating: 0, count: n)
cosineBlock_f32_f16norms(
    query: query,
    database: database,
    vectorCount: n,
    dimension: d,
    dbInvNorms: dbInvNormsF16,
    queryInvNorm: queryInvNorm,
    output: &similaritiesF16
)

// Memory saved: (n * 4) - (n * 2) = 2n bytes = 2KB for n=1000

// Example 3: High-level convenience API
let queryVec = [Float](repeating: 1.0, count: 768)
let databaseVecs = [[Float]](repeating: [Float](repeating: 0.5, count: 768), count: 100)

let scores = CosineSimilarityKernel.compute(
    query: queryVec,
    database: databaseVecs
)
// Automatically computes norms and handles memory layout

// Example 4: Integration with search index
struct VectorIndex {
    let vectors: [[Float]]
    let invNorms: [Float]
    let dimension: Int

    init(vectors: [[Float]]) {
        self.vectors = vectors
        self.dimension = vectors[0].count
        self.invNorms = [Float](repeating: 0, count: vectors.count)

        // Pre-compute all inverse norms
        let flatVectors = vectors.flatMap { $0 }
        precomputeInvNorms_f32(
            database: flatVectors,
            vectorCount: vectors.count,
            dimension: dimension,
            output: &invNorms
        )
    }

    func search(query: [Float], k: Int) -> [(index: Int, score: Float)] {
        let qInvNorm = computeQueryInvNorm_f32(query, dimension)
        var scores = [Float](repeating: 0, count: vectors.count)

        let flatVectors = vectors.flatMap { $0 }
        cosineBlock_f32(
            query: query,
            database: flatVectors,
            vectorCount: vectors.count,
            dimension: dimension,
            dbInvNorms: invNorms,
            queryInvNorm: qInvNorm,
            output: &scores
        )

        return scores.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(k)
            .map { ($0.offset, $0.element) }
    }
}

// Example 5: Parallel processing across index partitions
func searchParallel(
    query: [Float],
    partitions: [[Vector]],
    k: Int
) -> [SearchResult] {
    let qInvNorm = computeQueryInvNorm_f32(query, query.count)

    let partitionResults = DispatchQueue.concurrentPerform(iterations: partitions.count) { partitionID in
        let partition = partitions[partitionID]
        var scores = [Float](repeating: 0, count: partition.count)

        partition.withUnsafeBufferPointer { vecPtr in
            let norms = partition.map { $0.invNorm }
            norms.withUnsafeBufferPointer { normPtr in
                cosineBlock_f32(
                    query: query,
                    database: vecPtr.baseAddress!,
                    vectorCount: partition.count,
                    dimension: query.count,
                    dbInvNorms: normPtr.baseAddress!,
                    queryInvNorm: qInvNorm,
                    output: &scores
                )
            }
        }

        return zip(partition, scores)
            .sorted { $0.1 > $1.1 }
            .prefix(k)
            .map { SearchResult(vector: $0.0, score: $0.1) }
    }

    // Merge results from all partitions
    return partitionResults
        .flatMap { $0 }
        .sorted { $0.score > $1.score }
        .prefix(k)
        .map { $0 }
}
```

Mathematical Foundation

**Definition**:
Cosine similarity measures the cosine of the angle between two vectors:

cos(θ) = ⟨**q**, **x**⟩ / (‖**q**‖₂ · ‖**x**‖₂)

Where:
- ⟨**q**, **x**⟩ = Σⱼ qⱼ · xⱼ (inner product)
- ‖**v**‖₂ = √(Σⱼ vⱼ²) (Euclidean norm)

**Properties**:
1. **Range**: cos(θ) ∈ [-1, 1]
   - +1: Vectors are parallel (θ = 0°)
   - 0: Vectors are orthogonal (θ = 90°)
   - -1: Vectors are anti-parallel (θ = 180°)

2. **Magnitude Invariance**: cos(α**q**, β**x**) = cos(**q**, **x**) for α, β > 0
   - Scaling vectors doesn't change cosine similarity
   - Important for documents with different lengths

3. **Relation to Euclidean Distance** (for unit vectors):
   - ‖**q** - **x**‖₂² = 2(1 - cos(θ))
   - Minimizing distance ⟺ maximizing cosine

**Computational Complexity**:
- Naive: O(3nd) — 2 norms + 1 dot product
- Optimized: O(nd + n) — pre-compute db norms, compute query norm once
- Per-query amortized: O(nd) — same as inner product

**Numerical Considerations**:
1. **Epsilon Protection**:
   - inv_norm = 1/(‖v‖ + ε) prevents division by zero
   - Recommended: ε = 1e-12 for f32, 1e-6 for f16

2. **Clamping**:
   - Theoretically cos(θ) ∈ [-1, 1]
   - Floating-point errors can produce values slightly outside
   - Clamp to ensure valid range for downstream operations (e.g., arccos)

3. **Half-Precision Trade-offs**:
   - f16 range: [6e-8, 65504]
   - Typical inverse norms ∈ [0.5, 2.0] → well-represented
   - Relative error: ~0.1% (acceptable for ranking)

**Relation to Other Metrics**:
- **Angular Distance**: d_angular = arccos(cos(θ)) / π ∈ [0, 1]
- **Cosine Distance**: d_cosine = 1 - cos(θ) ∈ [0, 2]
- **L2 Distance** (unit vectors): d_L2² = 2(1 - cos(θ))

Dependencies

**Internal**:
- Inner Product Kernel (#02): `innerProductBlock_f32`, `innerProduct_f32_d*`
- Norm Kernel (#09): `l2Norm_f32`, `l2NormSquared_f32`
- Telemetry (#46): Performance instrumentation

**External**:
- Swift Standard Library: `SIMD4<Float>`, `Float16`
- Accelerate (optional): `vDSP` for reference implementations
- Foundation: `ProcessInfo` for thread count

**Build Requirements**:
- Swift 5.9+ (for `Float16` support)
- macOS 13+ / iOS 16+ (for half-precision)
- Optimization: `-O` (Release builds only)

Acceptance Criteria

✅ **Performance**:
- Achieves ≥80% of memory bandwidth peak on M1/M2/M3
- Fused kernel within 10% of raw IP performance
- f16 norms achieve 85%+ bandwidth (reduced memory traffic)

✅ **Correctness**:
- Matches NumPy/SciPy cosine within 1e-5 (f32) or 1e-4 (f16)
- All outputs ∈ [-1, 1] with clamping enabled
- Bit-exact determinism for repeated calls
- Graceful handling of zero-norm vectors

✅ **Coverage**:
- Specialized kernels for d ∈ {512, 768, 1024, 1536}
- Generic fallback for arbitrary dimensions
- Both f32 and f16 inverse norm support
- Tests cover all edge cases (zero norms, orthogonal, identical)

✅ **Integration**:
- Successfully used by IVF scoring (#04)
- Compatible with HNSW routing (#29)
- Works with re-ranking (#40)
- Composes cleanly with IP (#02) and norm (#09) kernels

✅ **Usability**:
- Clear distinction between pre-computation and query-time APIs
- High-level convenience API for simple use cases
- Low-level pointer API for performance-critical paths
- Comprehensive documentation with examples

✅ **Memory Efficiency**:
- f16 norms provide 50% memory savings with <0.1% accuracy loss
- No unnecessary allocations in hot path
- Fused kernel eliminates temporary dot product buffer
