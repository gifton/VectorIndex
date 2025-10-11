Title: ✅ DONE — Inner-Product (MIPS) Microkernel — High-Performance Dot Product Computation for Vector Search

Summary
- Implement a highly optimized inner-product (dot product) microkernel for computing similarity scores between query vectors and database vectors. This is the foundational computation kernel for maximum inner product search (MIPS), serving as a critical building block for cosine similarity, scoring blocks in IVF/HNSW indices, attention mechanisms, and re-ranking operations.
- Must achieve near-peak memory bandwidth utilization on Apple Silicon (>85% of theoretical roofline).
- Supports specialized implementations for common embedding dimensions (512, 768, 1024, 1536) with generic fallback.

Project Context
- VectorIndex provides high-performance vector search operations complementing VectorCore's computation primitives
- Inner product is the most frequently executed operation in vector search, accounting for 70-90% of total compute time
- Critical performance path for:
  - **IVF Scoring** (#04): Computing scores for candidate vectors in inverted file cells
  - **Cosine Similarity** (#03): Normalizing and computing angular similarity
  - **Graph Construction** (#29): Routing in HNSW/NSG graph navigation
  - **Training** (#19): Quantization training and centroid updates
  - **Re-ranking** (#40): Final scoring of top-k candidates
- Existing gap: No NEON-optimized inner-product kernel specialized for Apple Silicon
- Industry context: Similar to BLAS GEMV/DOT but with:
  - Fixed alignment guarantees (64-byte)
  - Known dimension specializations
  - Batch-oriented API for cache efficiency
  - Integration with quantized formats
- VectorCore provides baseline operations; VectorIndex needs search-optimized variants

Goals
- Achieve >85% of memory-bound theoretical peak on M1/M2/M3 processors
- Specialized fast paths for d ∈ {512, 768, 1024, 1536} at row counts r ∈ {1, 4, 8}
- Generic implementation with masked tail handling for arbitrary dimensions
- Deterministic results for reproducible ranking
- Thread-safe and data-race-free for parallel query processing
- Zero allocations in hot path
- Support both Array-of-Structures (AoS) and Structure-of-Arrays (SoA) layouts

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/InnerProductKernel.swift`
- Core implementations:
  - Specialized kernels: `ip_f32_d512_r{1,4,8}`, `ip_f32_d768_r{1,4,8}`, etc.
  - Generic kernel: `ip_f32_generic` with runtime dimension and count
  - Dispatch logic for optimal kernel selection
- Integration points:
  - Used by `ScoreBlockKernel` for batch scoring
  - Foundation for `CosineSimilarityKernel` after normalization
  - Compatible with both flat and quantized vector formats
- Supporting utilities:
  - Alignment verification helpers
  - Telemetry integration for profiling (#46)
  - SIMD intrinsic wrappers for clarity

API & Signatures

```swift
// MARK: - Core Inner Product API

/// Compute inner products between a query vector and a block of database vectors
/// Computes: out[i] = ⟨q, xb[i]⟩ for i ∈ [0, n)
///
/// - Complexity: O(n * d) FMAs
/// - Performance: Memory-bound; achieves 85-95% of peak bandwidth on M-series
/// - Thread Safety: Reentrant; safe for concurrent calls with disjoint outputs
@inlinable
public func innerProductBlock_f32(
    query q: UnsafePointer<Float>,      // [d] - 64-byte aligned
    database xb: UnsafePointer<Float>,  // [n][d] - 64-byte aligned, row-major
    vectorCount n: Int,
    dimension d: Int,
    output: UnsafeMutablePointer<Float>, // [n] - 64-byte aligned
    stride ldb: Int? = nil               // Optional: row stride if != d
)

// MARK: - Specialized Fast Paths

/// Inner product for 512-dimensional vectors (single row)
/// Optimized for: BERT-base embeddings, Ada-002 reduced, custom models
@inline(__always)
public func innerProduct_f32_d512_r1(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    output: UnsafeMutablePointer<Float>
)

/// Inner product for 512-dimensional vectors (4 rows)
/// Optimized for: Small batch scoring, L2 cache residency
@inline(__always)
public func innerProduct_f32_d512_r4(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    output: UnsafeMutablePointer<Float>
)

/// Inner product for 768-dimensional vectors (8 rows)
/// Optimized for: BERT-large, sentence-transformers, balanced batch size
@inline(__always)
public func innerProduct_f32_d768_r8(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    output: UnsafeMutablePointer<Float>
)

/// Inner product for 1536-dimensional vectors (4 rows)
/// Optimized for: OpenAI text-embedding-3-small/large, Cohere embeddings
@inline(__always)
public func innerProduct_f32_d1536_r4(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    output: UnsafeMutablePointer<Float>
)

// MARK: - Generic Fallback

/// Generic inner product with runtime dimension and count
/// Handles arbitrary dimensions with masked tail logic
@inlinable
public func innerProduct_f32_generic(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount: Int,
    dimension: Int,
    output: UnsafeMutablePointer<Float>,
    stride: Int
)

// MARK: - Dispatch & Configuration

/// Kernel dispatcher that selects optimal implementation based on (d, n)
public struct InnerProductDispatcher {

    /// Select and execute optimal kernel
    @inlinable
    public static func dispatch(
        query: UnsafePointer<Float>,
        database: UnsafePointer<Float>,
        vectorCount: Int,
        dimension: Int,
        output: UnsafeMutablePointer<Float>,
        config: Config = .default
    )

    /// Configuration for dispatch behavior
    public struct Config {
        let forceGeneric: Bool          // Skip specialization (for testing)
        let enableTelemetry: Bool        // Record kernel selection and timing
        let verifyAlignment: Bool        // Debug: check alignment (disabled in Release)
        let prefetchDistance: Int        // Rows ahead to prefetch (default: 2)

        public static let `default` = Config(
            forceGeneric: false,
            enableTelemetry: false,
            verifyAlignment: false,
            prefetchDistance: 2
        )
    }
}

// MARK: - Telemetry Integration

/// Per-kernel execution statistics for profiling
public struct InnerProductTelemetry {
    public let kernelVariant: String        // e.g., "d768_r8", "generic"
    public let rowsProcessed: Int
    public let bytesRead: Int               // Total memory read
    public let fastPathHit: Bool            // Specialized vs generic
    public let vectorWidth: Int             // SIMD width used (4 for NEON)
    public let executionTimeNanos: UInt64

    /// Compute achieved memory bandwidth
    public var bandwidthGBps: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return (Double(bytesRead) / 1e9) / seconds
    }
}

// MARK: - Helper Types

/// Alignment verification (debug builds only)
@inline(__always)
internal func verifyAlignment(_ ptr: UnsafeRawPointer, _ alignment: Int, _ label: String) {
    assert(Int(bitPattern: ptr) % alignment == 0,
           "\(label) must be \(alignment)-byte aligned, got: \(Int(bitPattern: ptr))")
}

/// Prefetch hint for upcoming cache line
@inline(__always)
internal func prefetch(_ ptr: UnsafeRawPointer) {
    // On Apple Silicon, prefetch is advisory; runtime may ignore
    // Uses PRFM PLDL1KEEP (prefetch for load, L1 cache, keep)
    #if arch(arm64)
    // Note: Swift doesn't expose PRFM directly; this is conceptual
    // In practice, compiler auto-prefetches or we use inline assembly
    #endif
}
```

Algorithm Details

**Core Algorithm**: Fused Multiply-Add (FMA) Accumulation

For a single row:
```
s = 0.0
for j in 0..<d:
    s += q[j] * xb[j]
output[0] = s
```

**Multi-Row Blocking** (e.g., r=4):
```
Pseudocode (conceptual):
s0 = s1 = s2 = s3 = 0.0

for j in 0..<d:
    q_val = q[j]
    s0 += q_val * xb[0][j]
    s1 += q_val * xb[1][j]
    s2 += q_val * xb[2][j]
    s3 += q_val * xb[3][j]

output[0] = s0
output[1] = s1
output[2] = s2
output[3] = s3
```

**Key Insight**: Load `q[j]` once, multiply with multiple `xb` rows → reduces query memory traffic by `r×`.

**Detailed Implementation Strategy**:

1. **Dimension Tiling**:
   - Tile dimension `d` into chunks of 64-128 elements
   - Each tile fits query chunk in NEON registers
   - Example for d=768: Use 3 tiles of 256 floats each

2. **NEON Vectorization** (ARM64):
   - 128-bit NEON vectors hold 4× Float32
   - Use `vld1q_f32` for aligned loads (64-byte → 16 floats per cache line)
   - Accumulate with `vfmaq_f32` (fused multiply-add)
   - Horizontal sum with `vaddvq_f32` at end

3. **Register Allocation**:
   - For r=4, d=512:
     - 4 accumulators (s0-s3): v0-v3 (SIMD4<Float>)
     - 1 query chunk register: v4
     - 4 database row registers: v5-v8
     - Total: 9 vector registers (well within 32 available)

4. **Loop Unrolling**:
   - Unroll inner loop by 4-8 vectors (16-32 floats)
   - Reduces loop overhead and branch mispredicts
   - Exposes instruction-level parallelism (ILP)

5. **Tail Handling**:
   - For dimensions not divisible by vector width:
     - Use masked loads/stores for final partial vector
     - Or: scalar cleanup loop for remaining elements
   - Example: d=770 → 192 full vectors + 2 scalar elements

**Specialized Implementation Example** (d=512, r=4):

```swift
@inline(__always)
public func innerProduct_f32_d512_r4(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    output: UnsafeMutablePointer<Float>
) {
    // d=512: 128 NEON vectors of 4 floats each
    // Process 4 rows simultaneously

    var s0 = SIMD4<Float>.zero
    var s1 = SIMD4<Float>.zero
    var s2 = SIMD4<Float>.zero
    var s3 = SIMD4<Float>.zero

    // Unroll by 4 vectors (16 floats) per iteration
    // Total: 512/16 = 32 iterations
    for i in stride(from: 0, to: 512, by: 16) {
        // Load query chunk (broadcast across iterations)
        let q0 = SIMD4<Float>(query + i + 0)
        let q1 = SIMD4<Float>(query + i + 4)
        let q2 = SIMD4<Float>(query + i + 8)
        let q3 = SIMD4<Float>(query + i + 12)

        // Row 0
        let xb0_0 = SIMD4<Float>(database + 0*512 + i + 0)
        let xb0_1 = SIMD4<Float>(database + 0*512 + i + 4)
        let xb0_2 = SIMD4<Float>(database + 0*512 + i + 8)
        let xb0_3 = SIMD4<Float>(database + 0*512 + i + 12)
        s0 += q0 * xb0_0 + q1 * xb0_1 + q2 * xb0_2 + q3 * xb0_3

        // Row 1
        let xb1_0 = SIMD4<Float>(database + 1*512 + i + 0)
        let xb1_1 = SIMD4<Float>(database + 1*512 + i + 4)
        let xb1_2 = SIMD4<Float>(database + 1*512 + i + 8)
        let xb1_3 = SIMD4<Float>(database + 1*512 + i + 12)
        s1 += q0 * xb1_0 + q1 * xb1_1 + q2 * xb1_2 + q3 * xb1_3

        // Row 2
        let xb2_0 = SIMD4<Float>(database + 2*512 + i + 0)
        let xb2_1 = SIMD4<Float>(database + 2*512 + i + 4)
        let xb2_2 = SIMD4<Float>(database + 2*512 + i + 8)
        let xb2_3 = SIMD4<Float>(database + 2*512 + i + 12)
        s2 += q0 * xb2_0 + q1 * xb2_1 + q2 * xb2_2 + q3 * xb2_3

        // Row 3
        let xb3_0 = SIMD4<Float>(database + 3*512 + i + 0)
        let xb3_1 = SIMD4<Float>(database + 3*512 + i + 4)
        let xb3_2 = SIMD4<Float>(database + 3*512 + i + 8)
        let xb3_3 = SIMD4<Float>(database + 3*512 + i + 12)
        s3 += q0 * xb3_0 + q1 * xb3_1 + q2 * xb3_2 + q3 * xb3_3
    }

    // Horizontal reduction: sum 4 lanes
    output[0] = s0.sum()
    output[1] = s1.sum()
    output[2] = s2.sum()
    output[3] = s3.sum()
}

extension SIMD4 where Scalar == Float {
    @inline(__always)
    func sum() -> Float {
        return self[0] + self[1] + self[2] + self[3]
    }
}
```

**Generic Implementation** (arbitrary d, n):

```swift
@inlinable
public func innerProduct_f32_generic(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    output: UnsafeMutablePointer<Float>,
    stride ldb: Int
) {
    let vecWidth = 4  // NEON: 4× Float32 per vector
    let dBlocked = (d / vecWidth) * vecWidth
    let dRemainder = d % vecWidth

    for i in 0..<n {
        var acc = SIMD4<Float>.zero
        let rowPtr = database + i * ldb

        // Vectorized loop
        for j in stride(from: 0, to: dBlocked, by: vecWidth) {
            let q_vec = SIMD4<Float>(query + j)
            let xb_vec = SIMD4<Float>(rowPtr + j)
            acc += q_vec * xb_vec
        }

        var sum = acc.sum()

        // Scalar tail for remainder
        for j in dBlocked..<d {
            sum += query[j] * rowPtr[j]
        }

        output[i] = sum
    }
}
```

Vectorization Details

**NEON SIMD (ARM64)**:
- **Register Set**: 32 × 128-bit vector registers (v0-v31)
- **Data Types**: `float32x4_t` or Swift's `SIMD4<Float>`
- **Key Intrinsics**:
  - `vld1q_f32(ptr)`: Load 4 aligned floats → vector
  - `vfmaq_f32(acc, a, b)`: acc += a * b (fused multiply-add)
  - `vaddvq_f32(vec)`: Horizontal sum of 4 lanes
  - `vst1q_f32(ptr, vec)`: Store vector → 4 floats

**Optimization Patterns**:
1. **Keep Query Hot**: Load `q[j]` into register, reuse across rows
2. **Multiple Accumulators**: 4-8 accumulators per row to hide FMA latency (3-4 cycles)
3. **Unroll by 4-8**: Expose ILP; M1 can dispatch 8 μops/cycle
4. **Aligned Access**: 64-byte alignment → single cache line, no split loads
5. **Prefetching**: Hint next row while computing current (2 rows ahead)

**Example Unrolled Inner Loop** (assembly-like pseudocode):
```asm
// Conceptual NEON assembly for one iteration
// Assume: v0=acc, v4=q_chunk, v8-v11=xb rows
vfmla.f32 v0, v4, v8   // acc[0] += q * xb[0]
vfmla.f32 v1, v4, v9   // acc[1] += q * xb[1]
vfmla.f32 v2, v4, v10  // acc[2] += q * xb[2]
vfmla.f32 v3, v4, v11  // acc[3] += q * xb[3]
```

Tiling & Cache Optimization

**Cache Hierarchy (Apple M1 example)**:
- L1d: 128 KB per core, ~3 cycle latency
- L2: 12 MB shared (P-cores), ~15 cycle latency
- Memory: ~100 GB/s bandwidth, ~100ns latency

**Tiling Strategy**:
1. **Dimension Tiling** (d-axis):
   - Tile size: 64-128 floats (256-512 bytes)
   - Goal: Keep query tile in L1 cache across multiple rows
   - Example: d=1536 → 12 tiles of 128 floats

2. **Row Blocking** (n-axis):
   - Block size: r ∈ {1, 4, 8, 16} rows
   - Goal: Amortize query loads across multiple outputs
   - Trade-off: Larger r → more register pressure, better query reuse

3. **Prefetching**:
   - Prefetch `xb[i+2]` while computing `xb[i]`
   - L1 prefetch distance: ~2-4 cache lines (128-256 bytes)
   - Helps hide memory latency (50-100ns)

**Layout Considerations**:
- **AoS (Array-of-Structures)**: `[n][d]` row-major
  - Natural for sequential row access
  - Used in most vector databases
- **SoA-Blocked**: `[n/block][d][block]`
  - Better for blocked algorithms
  - Requires explicit blocking parameter
  - Used in some quantized formats

Parallelism & Thread Safety

**Thread Model**:
- **Reentrant**: Multiple threads can call kernel concurrently
- **Disjoint Outputs**: Each thread writes to separate output buffer
- **No Shared State**: Kernels are pure functions (no globals)

**Parallelization Strategy** (across n):
```swift
// Shard rows across threads
let rowsPerThread = (n + threadCount - 1) / threadCount

DispatchQueue.concurrentPerform(iterations: threadCount) { threadID in
    let startRow = threadID * rowsPerThread
    let endRow = min(startRow + rowsPerThread, n)
    let rowCount = endRow - startRow

    if rowCount > 0 {
        innerProductBlock_f32(
            query: query,
            database: database + startRow * d,
            vectorCount: rowCount,
            dimension: d,
            output: output + startRow
        )
    }
}
```

**Synchronization**:
- No locks needed (embarrassingly parallel across rows)
- Memory barriers handled by dispatch queue
- Each thread has independent accumulator state

Numeric Stability

**Floating-Point Considerations**:
1. **Accumulation Order**:
   - Deterministic within a row (fixed loop order)
   - Associativity matters: `(a+b)+c ≠ a+(b+c)` in float
   - Guarantee: Same (d, alignment) → same bit-exact result

2. **Catastrophic Cancellation**:
   - Risk: Large positive/negative terms → precision loss
   - Mitigation: Accept as inherent to float32; use compensated summation if critical
   - Note: Inner product rarely suffers (terms usually same sign for normalized vectors)

3. **Overflow/Underflow**:
   - Range: float32 ∈ [±1.2e-38, ±3.4e38]
   - Typical embeddings: ∈ [-1, 1] or [-10, 10] → no overflow risk
   - Products: ~d terms of magnitude ~1 → result ~d (safe)

4. **Denormals**:
   - Values near zero (< 1.2e-38) become denormal → 10-100× slower
   - Mitigation: Flush-to-zero (FTZ) mode if performance-critical
   - Trade-off: Lose precision near zero

**Kahan Summation** (optional high-precision variant):
```swift
// Not default due to 2-3× slowdown, but available if needed
func kahanSum(_ values: [Float]) -> Float {
    var sum: Float = 0.0
    var c: Float = 0.0  // Compensation term

    for value in values {
        let y = value - c
        let t = sum + y
        c = (t - sum) - y
        sum = t
    }
    return sum
}
```

Telemetry Integration

**Instrumentation Points**:
1. **Kernel Selection**: Which specialized variant was chosen?
2. **Memory Traffic**: Total bytes read (query + database)
3. **Fast Path Hit**: Specialized vs generic fallback
4. **Execution Time**: Nanosecond-precision timing

**Usage**:
```swift
#if ENABLE_TELEMETRY
var telemetry = InnerProductTelemetry()
let startTime = mach_absolute_time()
#endif

innerProduct_f32_d512_r4(query, database, output)

#if ENABLE_TELEMETRY
let endTime = mach_absolute_time()
telemetry.executionTimeNanos = endTime - startTime
telemetry.kernelVariant = "d512_r4"
telemetry.rowsProcessed = 4
telemetry.bytesRead = 4 * 512 * 4 + 512 * 4  // xb + q
GlobalTelemetryRecorder.record(telemetry)
#endif
```

Performance Targets (Apple M1/M2/M3, Release Build)

**Memory Bandwidth Expectations**:
- M1 Max: ~200 GB/s peak (unified memory)
- M1 Pro: ~100 GB/s
- M1: ~60 GB/s

**Target Throughput** (vectors/second):
- d=512, r=4: > 1,000,000 vec/s (M1)
- d=768, r=8: > 800,000 vec/s
- d=1024, r=8: > 600,000 vec/s
- d=1536, r=4: > 400,000 vec/s

**Achieved Bandwidth**:
- Target: ≥85% of theoretical peak
- Example: M1 (60 GB/s) → ≥51 GB/s achieved

**Latency** (per-row, single-threaded):
- d=512: < 0.5 μs
- d=768: < 0.7 μs
- d=1536: < 1.5 μs

**Calculation Example** (d=512, r=4):
```
Bytes read: 4 rows × 512 dim × 4 bytes + 512 × 4 bytes (query) = 10,240 bytes
Operations: 4 rows × 512 FMAs = 2,048 FMAs = 4,096 FLOPs
Time budget (M1, 60 GB/s): 10,240 / 60e9 = 170 ns
Target: 170 / 0.85 = 200 ns
Throughput: 4 rows / 200 ns = 20M rows/s
```

Correctness & Testing

**Golden Reference**:
- Compare against BLAS `cblas_sgemv` (with TRANS='N', ALPHA=1, BETA=0)
- Tolerance: ≤ 1e-5 absolute error for normalized vectors

**Test Cases**:
1. **Dimension Coverage**:
   - Specialized: d ∈ {512, 768, 1024, 1536}
   - Generic: d ∈ {1, 64, 100, 513, 777, 2000, 4096}
   - Edge: d=1, d=3 (sub-vector width)

2. **Row Counts**:
   - Single: n=1
   - Blocked: n ∈ {4, 8, 16, 32}
   - Large: n ∈ {1000, 10000, 100000}
   - Odd: n ∈ {3, 7, 15, 1001}

3. **Adversarial Inputs**:
   - **Zeros**: All-zero query or database vectors
   - **Denormals**: Values near float min (1e-38)
   - **Large values**: Near float max (1e38)
   - **Mixed signs**: Positive and negative terms
   - **Uniform**: All elements identical (tests associativity)

4. **Alignment**:
   - Verify 64-byte alignment assertions
   - Test misaligned access (should fail in debug)

5. **Numerical Stability**:
   - Long vectors (d=10000) with compensated sum check
   - Accumulation order consistency test

**Performance Validation**:
```swift
func testPerformanceRoofline() {
    let d = 512
    let n = 10000
    let query = allocateAligned(count: d)
    let database = allocateAligned(count: n * d)
    let output = allocateAligned(count: n)

    // Warm-up
    for _ in 0..<10 {
        innerProductBlock_f32(query, database, n, d, output)
    }

    // Measure
    let iterations = 100
    let start = mach_absolute_time()
    for _ in 0..<iterations {
        innerProductBlock_f32(query, database, n, d, output)
    }
    let end = mach_absolute_time()

    let timeSeconds = Double(end - start) / 1e9 / Double(iterations)
    let bytesRead = (n * d + d) * 4  // xb + q
    let bandwidthGBps = Double(bytesRead) / 1e9 / timeSeconds

    // M1: 60 GB/s peak, expect ≥51 GB/s (85%)
    XCTAssertGreaterThan(bandwidthGBps, 51.0)
}
```

Integration with VectorCore & VectorIndex

**VectorCore Reuse**:
```swift
// VectorCore provides baseline operations
import VectorCore

// For specialized dimensions, route to VectorCore's optimized types
extension InnerProductDispatcher {
    static func dispatch_vcore_512(
        query: UnsafePointer<Float>,
        database: UnsafePointer<Float>,
        vectorCount: Int,
        output: UnsafeMutablePointer<Float>
    ) {
        // Conceptual: VectorCore's BK:IP512 if available
        // Otherwise, use local implementation
        #if VECTORCORE_AVAILABLE
        BK.IP512(query, database, vectorCount, output)
        #else
        innerProduct_f32_d512_r4(query, database, output)
        #endif
    }
}
```

**VectorIndex Usage**:
```swift
// Used by ScoreBlockKernel (#04)
func scoreBlock(
    query: Vector,
    candidates: [Vector],
    scores: inout [Float]
) {
    candidates.withUnsafeBufferPointer { candidatesPtr in
        scores.withUnsafeMutableBufferPointer { scoresPtr in
            innerProductBlock_f32(
                query: query.data,
                database: candidatesPtr.baseAddress!,
                vectorCount: candidates.count,
                dimension: query.dimension,
                output: scoresPtr.baseAddress!
            )
        }
    }
}

// Used by CosineSimilarityKernel (#03)
func cosineSimilarity(
    query: Vector,
    candidate: Vector
) -> Float {
    var dotProduct: Float = 0
    innerProduct_f32_d768_r1(
        query: query.data,
        database: candidate.data,
        output: &dotProduct
    )

    let normProduct = query.norm * candidate.norm
    return dotProduct / normProduct
}
```

Coding Guidelines

**Performance-Critical**:
- Mark all functions `@inlinable` or `@inline(__always)` for specialized variants
- Use `@_specialize` for generic functions with known types
- Avoid allocations in hot path (preallocate outputs)
- Prefer value types over reference types (avoid heap allocation)

**SIMD Usage**:
- Use Swift's `SIMD4<Float>` for clarity (compiler maps to NEON)
- For maximum control, use Accelerate framework's vDSP functions
- Document NEON register usage in comments

**Alignment**:
- All pointers must be 64-byte aligned (cache line boundary)
- Use `posix_memalign` or custom allocators
- Assert alignment in debug builds only (zero overhead in release)

**Error Handling**:
- Preconditions on alignment, dimension, count (debug only)
- No runtime errors in hot path
- Invalid inputs → undefined behavior (document in API)

**Documentation**:
- Include mathematical notation: ⟨q, xb[i]⟩ = Σⱼ q[j] × xb[i][j]
- Complexity analysis: O(n * d) FMAs
- Performance characteristics: Memory-bound vs compute-bound

Non-Goals

- GPU/Metal acceleration (handled by separate Metal kernels)
- fp16 or bfloat16 support (future work)
- Quantized formats (int8, uint8) — separate quantized IP kernel
- Multi-query batching (batch queries separately, then GEMM)
- Strided query vector (assume contiguous)
- Runtime dispatch overhead optimization (acceptable for flexibility)

Example Usage

```swift
import VectorIndex

// Allocate aligned memory
func allocateAligned<T>(count: Int, alignment: Int = 64) -> UnsafeMutablePointer<T> {
    var ptr: UnsafeMutableRawPointer?
    posix_memalign(&ptr, alignment, count * MemoryLayout<T>.stride)
    return ptr!.assumingMemoryBound(to: T.self)
}

// Example 1: Score a batch of 100 candidate vectors (d=768)
let query = allocateAligned<Float>(count: 768)
let candidates = allocateAligned<Float>(count: 100 * 768)
let scores = allocateAligned<Float>(count: 100)

// ... populate query and candidates ...

innerProductBlock_f32(
    query: query,
    database: candidates,
    vectorCount: 100,
    dimension: 768,
    output: scores
)

print("Top score: \(scores.max()!)")

// Example 2: Use specialized kernel directly
let query512 = allocateAligned<Float>(count: 512)
let db512 = allocateAligned<Float>(count: 4 * 512)  // 4 rows
let scores4 = allocateAligned<Float>(count: 4)

innerProduct_f32_d512_r4(
    query: query512,
    database: db512,
    output: scores4
)

// Example 3: Dispatcher with telemetry
let config = InnerProductDispatcher.Config(
    forceGeneric: false,
    enableTelemetry: true,
    verifyAlignment: true,  // Debug build only
    prefetchDistance: 2
)

InnerProductDispatcher.dispatch(
    query: query,
    database: candidates,
    vectorCount: 100,
    dimension: 768,
    output: scores,
    config: config
)

// Example 4: Parallel scoring across threads
func parallelScore(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    n: Int,
    d: Int,
    output: UnsafeMutablePointer<Float>
) {
    let threadCount = ProcessInfo.processInfo.activeProcessorCount
    let rowsPerThread = (n + threadCount - 1) / threadCount

    DispatchQueue.concurrentPerform(iterations: threadCount) { threadID in
        let start = threadID * rowsPerThread
        let end = min(start + rowsPerThread, n)
        let count = end - start

        if count > 0 {
            innerProductBlock_f32(
                query: query,
                database: database + start * d,
                vectorCount: count,
                dimension: d,
                output: output + start
            )
        }
    }
}

// Example 5: Integration with higher-level search
struct IVFIndex {
    let cells: [[Vector]]

    func search(query: Vector, k: Int) -> [SearchResult] {
        var allScores: [(cellID: Int, vectorID: Int, score: Float)] = []

        for (cellID, cell) in cells.enumerated() {
            var scores = [Float](repeating: 0, count: cell.count)

            cell.withUnsafeBufferPointer { cellPtr in
                innerProductBlock_f32(
                    query: query.data,
                    database: cellPtr.baseAddress!,
                    vectorCount: cell.count,
                    dimension: query.dimension,
                    output: &scores
                )
            }

            for (vectorID, score) in scores.enumerated() {
                allScores.append((cellID, vectorID, score))
            }
        }

        return allScores
            .sorted { $0.score > $1.score }
            .prefix(k)
            .map { SearchResult(cellID: $0.cellID, vectorID: $0.vectorID, score: $0.score) }
    }
}
```

Mathematical Foundation

**Definition**:
The inner product (dot product) of two vectors **q**, **x** ∈ ℝ^d is:

⟨**q**, **x**⟩ = Σⱼ₌₀^(d-1) qⱼ · xⱼ

**Properties**:
1. **Symmetry**: ⟨**q**, **x**⟩ = ⟨**x**, **q**⟩
2. **Linearity**: ⟨α**q**, **x**⟩ = α⟨**q**, **x**⟩
3. **Bilinearity**: ⟨**q**₁ + **q**₂, **x**⟩ = ⟨**q**₁, **x**⟩ + ⟨**q**₂, **x**⟩

**Relation to Cosine Similarity**:
cos(θ) = ⟨**q**, **x**⟩ / (‖**q**‖₂ · ‖**x**‖₂)

Where ‖·‖₂ is the Euclidean norm.

**Maximum Inner Product Search (MIPS)**:
Find: x* = argmax_(**x** ∈ 𝒟) ⟨**q**, **x**⟩

**Complexity**:
- Time: O(n · d) for n vectors
- Space: O(d) for query + O(n · d) for database (read-only)
- Arithmetic intensity: 2d FLOPs / (d · 4 bytes read) = 0.5 FLOP/byte → memory-bound

**Roofline Analysis**:
For M1 @ 60 GB/s, 3.2 TFLOP/s:
- Memory-bound threshold: 3.2 TFLOP/s / 60 GB/s = 53 FLOPs/byte
- Inner product intensity: 0.5 FLOP/byte << 53 → memory-bound
- Peak performance: 60 GB/s × 0.5 = 30 GFLOP/s (not 3.2 TFLOP/s)

Dependencies

**External**:
- Swift Standard Library: `SIMD4<Float>`, `UnsafePointer`
- Accelerate framework (optional): `vDSP_dotpr`, `cblas_sgemv` for reference

**Internal**:
- VectorCore (optional): Reuse existing `BK:IP*` kernels if available
- Telemetry module (#46): Performance instrumentation
- Alignment utilities: Custom allocators

**Build Requirements**:
- Swift 5.9+ (for `SIMD` improvements)
- macOS 13+ / iOS 16+ (for Accelerate availability)
- Optimization level: `-O` (Release) for performance testing

Acceptance Criteria

✅ **Performance**:
- Achieves ≥85% of memory bandwidth peak on M1/M2/M3
- Specialized kernels within 5% of hand-tuned assembly

✅ **Correctness**:
- All test cases pass with ≤1e-5 error vs BLAS reference
- Bit-exact determinism for repeated calls with same inputs
- No crashes or undefined behavior on adversarial inputs

✅ **Coverage**:
- Specialized kernels for d ∈ {512, 768, 1024, 1536}, r ∈ {1, 4, 8}
- Generic fallback handles arbitrary (d, n)
- Tests cover 95%+ code paths

✅ **Integration**:
- Successfully used by ScoreBlockKernel (#04)
- Compatible with CosineSimilarityKernel (#03)
- Works with both VectorCore and standalone

✅ **Documentation**:
- API docs with mathematical notation
- Performance characteristics documented
- Example usage for common scenarios

✅ **Telemetry**:
- Kernel selection tracked
- Bandwidth measurements accurate within 5%
- Integration with global profiling system (#46)
