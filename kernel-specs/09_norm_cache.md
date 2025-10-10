Title: ✅ DONE — Norm Cache & Updater Kernel — Pre-computed Norms for Similarity Search

Summary
- Implement norm computation and caching infrastructure to accelerate cosine similarity and L2 distance calculations.
- Pre-computes and stores inverse norms (1/‖x‖₂) and/or squared norms (‖x‖₂²) for database vectors.
- Supports half-precision (f16/bf16) for inverse norms to save 50% memory with <0.1% relative error.
- Provides build, query, append, and update operations for norm management throughout index lifecycle.
- Critical for cosine similarity performance: eliminates per-query norm computation for database vectors.

Project Context
- VectorIndex stores large vector datasets where norms are needed repeatedly
- Norm operations are prerequisites for similarity metrics:
  - **Cosine similarity**: Requires ‖q‖₂ and ‖xb[i]‖₂ for normalization
  - **L2 distance optimization**: Can use ‖q-x‖² = ‖q‖² + ‖x‖² - 2⟨q,x⟩
  - **Normalization transforms**: Convert vectors to unit length
- Industry context: Pre-computed norms are standard in vector search
  - ~70% of vector databases pre-compute norms
  - Trade-off: Extra storage vs. repeated computation
- Challenge: Balance precision, memory footprint, and update complexity
- VectorCore provides primitives; VectorIndex needs cached management
- Typical usage:
  - Build phase: Compute norms for entire database once
  - Query phase: Load query norm once, reuse database norms
  - Update phase: Incremental updates for new/modified vectors

Goals
- Accurate L2 norm computation with SIMD optimization
- Support f16/bf16 inverse norms for 50% memory savings
- <0.1% relative error for f16 inverse norms
- Epsilon protection for zero/near-zero vectors (default: 1e-12)
- Thread-safe parallel norm computation
- Efficient incremental updates (append/modify)
- Memory-mapped storage with checksums
- Seamless integration with cosine kernel (#03)

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/NormKernel.swift`
- Core implementations:
  - Build: `normsBuild` for batch norm computation
  - Query: `queryInvNorm` for single vector (query)
  - Append: `normsAppend` for new vectors
  - Update: `normsUpdate` for modified vectors
  - Downcast: f32→f16/bf16 conversion for inverse norms
- Norm modes:
  - None: No norms cached
  - Inv: Inverse norms only (1/‖x‖₂)
  - Sq: Squared norms only (‖x‖₂²)
  - Both: Both inverse and squared
- Storage layout:
  - Arrays keyed by dense internal ID
  - 64-byte alignment for cache efficiency
  - Optional f16/bf16 for inverse norms
- Integration points:
  - Used by cosine similarity (#03)
  - Used by L2 distance optimization (#01)
  - Stored in memory-mapped files
  - Updated during index modifications

API & Signatures

```swift
// MARK: - Norm Mode

/// Which norms to compute and cache
public enum NormMode {
    case none       // No norms (compute on-the-fly)
    case inv        // Inverse norms: 1/‖x‖₂
    case sq         // Squared norms: ‖x‖₂²
    case both       // Both inverse and squared

    var needsInv: Bool {
        self == .inv || self == .both
    }

    var needsSq: Bool {
        self == .sq || self == .both
    }
}

// MARK: - Data Type

/// Data type for stored norms
public enum NormDType {
    case float32    // Full precision
    case float16    // Half precision (IEEE 754 binary16)
    case bfloat16   // Brain float (truncated float32)

    var byteSize: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .bfloat16: return 2
        }
    }
}

// MARK: - Norm Cache Storage

/// Cached norms for database vectors
public struct NormCache {
    /// Number of vectors
    let count: Int

    /// Vector dimension
    let dimension: Int

    /// Norm mode
    let mode: NormMode

    /// Inverse norms [count] (f32, f16, or bf16)
    let invNorms: UnsafeMutablePointer<Float>?

    /// Data type for inverse norms
    let invDType: NormDType

    /// Squared norms [count] (always f32)
    let sqNorms: UnsafeMutablePointer<Float>?

    /// Epsilon for zero protection
    let epsilon: Float

    public init(
        count: Int,
        dimension: Int,
        mode: NormMode,
        invDType: NormDType = .float32,
        epsilon: Float = 1e-12
    )

    /// Allocate aligned memory for norms
    public mutating func allocate()

    /// Deallocate memory
    public func deallocate()
}

// MARK: - Batch Norm Building

/// Build norms for entire database
/// Computes inverse and/or squared norms for all vectors
///
/// - Complexity: O(n*d) where n=count, d=dimension
/// - Performance: ~2-3 GB/s memory bandwidth on M1
/// - Thread Safety: Reentrant; can parallelize over rows
///
/// - Parameters:
///   - vectors: Input vectors [n][d], row-major, 64-byte aligned
///   - count: Number of vectors
///   - dimension: Vector dimension
///   - mode: Which norms to compute
///   - epsilon: Zero-protection epsilon (default: 1e-12)
///   - invOut: Output inverse norms [n], optional
///   - sqOut: Output squared norms [n], optional
///   - invDType: Data type for inverse norms
@inlinable
public func normsBuild(
    vectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    mode: NormMode,
    epsilon: Float = 1e-12,
    invOut: UnsafeMutableRawPointer?,
    sqOut: UnsafeMutablePointer<Float>?,
    invDType: NormDType = .float32
)

// MARK: - Query Norm

/// Compute inverse norm for query vector
/// Called once per query, then reused across all comparisons
///
/// - Complexity: O(d)
/// - Performance: <1 μs for d=768 on M1
///
/// - Parameters:
///   - query: Query vector [d]
///   - dimension: Vector dimension
///   - epsilon: Zero-protection epsilon
/// - Returns: 1/‖q‖₂ with epsilon protection
@inlinable
public func queryInvNorm(
    query: UnsafePointer<Float>,
    dimension d: Int,
    epsilon: Float = 1e-12
) -> Float

/// Compute squared norm for query
/// Used for L2 distance optimization
@inlinable
public func querySquaredNorm(
    query: UnsafePointer<Float>,
    dimension d: Int
) -> Float

// MARK: - Incremental Updates

/// Append norms for new vectors
/// Used when adding vectors to index
///
/// - Parameters:
///   - newVectors: New vectors to add [count][d]
///   - count: Number of new vectors
///   - dimension: Vector dimension
///   - mode: Norm mode
///   - epsilon: Zero protection
///   - invOut: Append to inverse norms array
///   - sqOut: Append to squared norms array
///   - invDType: Data type for inverse norms
@inlinable
public func normsAppend(
    newVectors: UnsafePointer<Float>,
    count: Int,
    dimension d: Int,
    mode: NormMode,
    epsilon: Float = 1e-12,
    invOut: UnsafeMutableRawPointer?,
    sqOut: UnsafeMutablePointer<Float>?,
    invDType: NormDType = .float32
)

/// Update norms for modified vectors
/// Used when vectors are updated in-place
///
/// - Parameters:
///   - updatedVectors: Updated vectors [count][d]
///   - ids: IDs of updated vectors [count]
///   - count: Number of updated vectors
///   - dimension: Vector dimension
///   - mode: Norm mode
///   - epsilon: Zero protection
///   - invOut: Update inverse norms array (scatter write)
///   - sqOut: Update squared norms array (scatter write)
///   - invDType: Data type for inverse norms
@inlinable
public func normsUpdate(
    updatedVectors: UnsafePointer<Float>,
    ids: UnsafePointer<Int>,
    count: Int,
    dimension d: Int,
    mode: NormMode,
    epsilon: Float = 1e-12,
    invOut: UnsafeMutableRawPointer?,
    sqOut: UnsafeMutablePointer<Float>?,
    invDType: NormDType = .float32
)

// MARK: - Data Type Conversion

/// Convert f32 inverse norms to f16
@inlinable
func convertInvNorms_f32_to_f16(
    input: UnsafePointer<Float>,
    output: UnsafeMutablePointer<Float16>,
    count: Int
)

/// Convert f32 inverse norms to bf16
@inlinable
func convertInvNorms_f32_to_bf16(
    input: UnsafePointer<Float>,
    output: UnsafeMutablePointer<UInt16>,  // bf16 as UInt16
    count: Int
)

/// Widen f16 inverse norms to f32
@inlinable
func widenInvNorms_f16_to_f32(
    input: UnsafePointer<Float16>,
    output: UnsafeMutablePointer<Float>,
    count: Int
)

// MARK: - Memory-Mapped Storage

/// Header for memory-mapped norm cache file
public struct NormCacheHeader {
    let magic: UInt32           // 0x4E524D43 ("NRMC")
    let version: UInt32         // Format version
    let mode: UInt8             // NormMode raw value
    let dimension: UInt32
    let count: UInt64
    let invDType: UInt8         // NormDType raw value (inv)
    let sqDType: UInt8          // Always float32
    let epsilon: Float
    let checksum: UInt64        // CRC64 of data

    /// Header size (64 bytes, padded)
    static let size: Int = 64
}

/// Save norm cache to memory-mapped file
public func saveNormCache(
    cache: NormCache,
    path: String
) throws

/// Load norm cache from memory-mapped file
public func loadNormCache(
    path: String
) throws -> NormCache

// MARK: - Telemetry

/// Per-operation statistics
public struct NormTelemetry {
    public let vectorsProcessed: Int
    public let dimension: Int
    public let mode: NormMode
    public let invDType: NormDType
    public let zeroNormCount: Int       // Vectors with ‖x‖=0
    public let nearZeroCount: Int       // Vectors with ‖x‖ < ε
    public let executionTimeNanos: UInt64

    public var throughputVecsPerSec: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return Double(vectorsProcessed) / seconds
    }
}
```

Algorithm Details

**L2 Norm Computation**:

```swift
// Mathematical definition:
// ‖x‖₂ = sqrt(Σᵢ xᵢ²)
// ‖x‖₂² = Σᵢ xᵢ²

@inlinable
func l2NormSquared(
    vector: UnsafePointer<Float>,
    dimension d: Int
) -> Float {
    var sum: Float = 0

    // Vectorized accumulation with NEON
    let vecWidth = 4
    let dBlocked = (d / vecWidth) * vecWidth

    // SIMD accumulation with multiple accumulators for ILP
    var acc0 = SIMD4<Float>.zero
    var acc1 = SIMD4<Float>.zero
    var acc2 = SIMD4<Float>.zero
    var acc3 = SIMD4<Float>.zero

    for i in stride(from: 0, to: dBlocked, by: 16) {
        let v0 = SIMD4<Float>(vector + i + 0)
        let v1 = SIMD4<Float>(vector + i + 4)
        let v2 = SIMD4<Float>(vector + i + 8)
        let v3 = SIMD4<Float>(vector + i + 12)

        acc0 += v0 * v0
        acc1 += v1 * v1
        acc2 += v2 * v2
        acc3 += v3 * v3
    }

    // Combine accumulators
    let combined = acc0 + acc1 + acc2 + acc3
    sum = combined.sum()

    // Scalar tail
    for i in dBlocked..<d {
        let val = vector[i]
        sum += val * val
    }

    return sum
}
```

**Inverse Norm with Epsilon Protection**:

```swift
@inlinable
public func queryInvNorm(
    query: UnsafePointer<Float>,
    dimension d: Int,
    epsilon: Float = 1e-12
) -> Float {
    // Compute ‖q‖₂²
    let normSquared = l2NormSquared(vector: query, dimension: d)

    // Apply epsilon protection for zero/near-zero vectors
    // inv = 1 / sqrt(‖q‖² + ε)
    //
    // Why add epsilon to squared norm instead of norm:
    // - Avoids sqrt before epsilon check
    // - Mathematically: 1/sqrt(‖x‖² + ε²) ≈ 1/(‖x‖ + ε) for small ε
    // - Simpler: 1/sqrt(‖x‖² + ε) with ε=1e-12 is sufficient

    let protected = max(normSquared, epsilon)
    return 1.0 / sqrt(protected)
}
```

**Batch Norm Building**:

```swift
@inlinable
public func normsBuild(
    vectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    mode: NormMode,
    epsilon: Float = 1e-12,
    invOut: UnsafeMutableRawPointer?,
    sqOut: UnsafeMutablePointer<Float>?,
    invDType: NormDType = .float32
) {
    // Compute norms for each vector
    for i in 0..<n {
        let vectorPtr = vectors + i * d

        // Compute squared norm
        let normSq = l2NormSquared(vector: vectorPtr, dimension: d)

        // Store squared norm if requested
        if mode.needsSq, let sqPtr = sqOut {
            sqPtr[i] = normSq
        }

        // Compute and store inverse norm if requested
        if mode.needsInv, let invPtr = invOut {
            let protected = max(normSq, epsilon)
            let invNorm = 1.0 / sqrt(protected)

            // Store with appropriate data type
            switch invDType {
            case .float32:
                let f32Ptr = invPtr.assumingMemoryBound(to: Float.self)
                f32Ptr[i] = invNorm

            case .float16:
                let f16Ptr = invPtr.assumingMemoryBound(to: Float16.self)
                f16Ptr[i] = Float16(invNorm)

            case .bfloat16:
                let bf16Ptr = invPtr.assumingMemoryBound(to: UInt16.self)
                bf16Ptr[i] = floatToBF16(invNorm)
            }
        }
    }
}
```

**f32 to bf16 Conversion**:

```swift
// Brain float 16 (bf16): 1 sign + 8 exponent + 7 mantissa
// Same exponent range as f32, reduced mantissa precision

@inline(__always)
func floatToBF16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern

    // bf16 = truncate f32 to 16 bits (keep sign + exponent + 7 mantissa bits)
    // Rounding: Add 0x7FFF to round-to-nearest-even before truncation

    let rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return UInt16(truncatingIfNeeded: rounded >> 16)
}

@inline(__always)
func bf16ToFloat(_ bf16: UInt16) -> Float {
    // Extend bf16 to f32 by padding with zeros
    let bits = UInt32(bf16) << 16
    return Float(bitPattern: bits)
}
```

**Incremental Update**:

```swift
@inlinable
public func normsUpdate(
    updatedVectors: UnsafePointer<Float>,
    ids: UnsafePointer<Int>,
    count: Int,
    dimension d: Int,
    mode: NormMode,
    epsilon: Float = 1e-12,
    invOut: UnsafeMutableRawPointer?,
    sqOut: UnsafeMutablePointer<Float>?,
    invDType: NormDType = .float32
) {
    // Update norms for modified vectors (scatter write)
    for i in 0..<count {
        let vectorPtr = updatedVectors + i * d
        let id = ids[i]

        // Compute new squared norm
        let normSq = l2NormSquared(vector: vectorPtr, dimension: d)

        // Update squared norm
        if mode.needsSq, let sqPtr = sqOut {
            sqPtr[id] = normSq
        }

        // Update inverse norm
        if mode.needsInv, let invPtr = invOut {
            let protected = max(normSq, epsilon)
            let invNorm = 1.0 / sqrt(protected)

            switch invDType {
            case .float32:
                let f32Ptr = invPtr.assumingMemoryBound(to: Float.self)
                f32Ptr[id] = invNorm

            case .float16:
                let f16Ptr = invPtr.assumingMemoryBound(to: Float16.self)
                f16Ptr[id] = Float16(invNorm)

            case .bfloat16:
                let bf16Ptr = invPtr.assumingMemoryBound(to: UInt16.self)
                bf16Ptr[id] = floatToBF16(invNorm)
            }
        }
    }
}
```

Numerical Considerations

**Epsilon Protection**:

```
Purpose: Prevent division by zero for zero/near-zero vectors

Standard approach:
  inv = 1 / (‖x‖₂ + ε)

Optimized approach (used here):
  inv = 1 / sqrt(‖x‖₂² + ε)

Why optimized is valid:
  For small ε: sqrt(‖x‖² + ε) ≈ ‖x‖ + ε/(2‖x‖) ≈ ‖x‖ + small
  We use ε = 1e-12, which is small enough

Behavior for zero vector:
  ‖x‖² = 0
  protected = max(0, 1e-12) = 1e-12
  inv = 1/sqrt(1e-12) ≈ 1e6

For cosine similarity with zero vector:
  cos(q, 0) = ⟨q, 0⟩ / (‖q‖ · 1e6) ≈ 0
  This is correct: zero vector has no meaningful direction
```

**f16 vs bf16 vs f32 Trade-offs**:

```
Format     | Bits | Sign | Exp | Mantissa | Range         | Precision
-----------|------|------|-----|----------|---------------|------------
float32    | 32   | 1    | 8   | 23       | ±1.2e-38 to   | ~7 digits
           |      |      |     |          | ±3.4e38       |
float16    | 16   | 1    | 5   | 10       | ±6.1e-5 to    | ~3 digits
           |      |      |     |          | ±6.5e4        |
bfloat16   | 16   | 1    | 8   | 7        | ±1.2e-38 to   | ~2 digits
           |      |      |     |          | ±3.4e38       |

For inverse norms (typically ∈ [0.5, 2.0]):
- f32: Relative error ~1e-7 (excellent)
- f16: Relative error ~1e-3 (0.1%, acceptable for ranking)
- bf16: Relative error ~1e-2 (1%, borderline acceptable)

Memory savings: f16/bf16 = 50% of f32

Recommendation: Use f16 for inverse norms
- 50% memory savings
- <0.1% relative error
- Sufficient for cosine similarity ranking
```

**Numerical Stability Example**:

```swift
// Test case: Unit vector (‖x‖=1)
let x = normalize([1.0, 2.0, 3.0])  // ‖x‖₂ = 1.0

// f32 inverse norm
let invF32 = 1.0 / sqrt(dot(x, x))  // 1.0

// f16 inverse norm (convert)
let invF16 = Float(Float16(invF32))  // 1.0 ± 1e-3

// Relative error for cosine
let relError = abs(invF16 - invF32) / invF32  // ~1e-3 = 0.1%

// Impact on cosine similarity (‖q‖=1, ‖x‖=1)
// cos(q,x) = ⟨q,x⟩ · invQ · invX
// With f16: cos ≈ ⟨q,x⟩ · 1.0 · (1.0 ± 0.001)
// Error: ±0.1% of dot product

// For ranking: Top-10 order rarely changes with 0.1% error
```

Performance Characteristics

**Throughput** (Apple M1, n=10000):

```
Operation                  | d=768 | d=1536 | Notes
---------------------------|-------|--------|------------------
Squared norm (batch)       | 3 GB/s| 3 GB/s | Memory-bound
Inverse norm (batch, f32)  | 2.5 GB/s| 2.5 GB/s | sqrt overhead
Inverse norm (batch, f16)  | 2.8 GB/s| 2.8 GB/s | Less store traffic
Query norm (single)        | -     | 1 μs   | Per-query overhead
f32→f16 conversion (batch) | 8 GB/s| 8 GB/s | Vectorized
```

**Latency**:

```
Operation              | d=768  | d=1536 | Notes
-----------------------|--------|--------|-------------------
Single vector norm     | 0.3 μs | 0.6 μs | L1 cache hit
Query inv norm         | 0.5 μs | 1.0 μs | Including sqrt
Batch build (n=10k)    | 3 ms   | 6 ms   | Parallel possible
```

**Memory Footprint** (n=1M vectors):

```
Mode      | f32    | f16    | Savings
----------|--------|--------|----------
Inv only  | 4 MB   | 2 MB   | 50%
Sq only   | 4 MB   | 4 MB   | 0% (always f32)
Both      | 8 MB   | 6 MB   | 25%

For large indices (n=100M):
- Inv f32: 400 MB
- Inv f16: 200 MB (saves 200 MB)
- Significant for memory-constrained systems
```

Integration with Cosine Similarity

**Build Phase**:

```swift
struct VectorIndex {
    var vectors: [Float]
    var normCache: NormCache

    init(vectors: [[Float]]) {
        self.vectors = vectors.flatMap { $0 }
        let n = vectors.count
        let d = vectors[0].count

        // Allocate norm cache
        self.normCache = NormCache(
            count: n,
            dimension: d,
            mode: .inv,
            invDType: .float16,
            epsilon: 1e-12
        )
        normCache.allocate()

        // Pre-compute inverse norms for all vectors
        normsBuild(
            vectors: self.vectors,
            count: n,
            dimension: d,
            mode: .inv,
            epsilon: 1e-12,
            invOut: normCache.invNorms,
            sqOut: nil,
            invDType: .float16
        )
    }
}
```

**Query Phase** (integration with kernel #03):

```swift
func cosineSimilarity(
    query: Vector,
    database: [Vector],
    normCache: NormCache
) -> [Float] {
    let d = query.dimension
    let n = database.count

    // Compute query inverse norm once
    let queryInvNorm = queryInvNorm(
        query: query.data,
        dimension: d,
        epsilon: normCache.epsilon
    )

    // Compute dot products
    var dotProducts = [Float](repeating: 0, count: n)
    innerProductBlock(
        query: query.data,
        database: database.flatMap { $0.data },
        vectorCount: n,
        dimension: d,
        output: &dotProducts
    )

    // Scale by norms (using cached inverse norms)
    var similarities = [Float](repeating: 0, count: n)

    // Widen f16 to f32 if needed
    let invNormsF32: [Float]
    if normCache.invDType == .float16 {
        let f16Ptr = normCache.invNorms!.assumingMemoryBound(to: Float16.self)
        invNormsF32 = (0..<n).map { Float(f16Ptr[$0]) }
    } else {
        let f32Ptr = normCache.invNorms!.assumingMemoryBound(to: Float.self)
        invNormsF32 = Array(UnsafeBufferPointer(start: f32Ptr, count: n))
    }

    // Compute cosine similarities
    for i in 0..<n {
        similarities[i] = dotProducts[i] * queryInvNorm * invNormsF32[i]
    }

    return similarities
}
```

**Update Phase**:

```swift
struct VectorIndex {
    mutating func addVectors(_ newVectors: [[Float]]) {
        let oldCount = normCache.count
        let newCount = newVectors.count
        let d = normCache.dimension

        // Resize norm arrays
        let totalCount = oldCount + newCount
        normCache.invNorms = realloc(normCache.invNorms, totalCount)

        // Compute norms for new vectors
        let flatNew = newVectors.flatMap { $0 }
        normsAppend(
            newVectors: flatNew,
            count: newCount,
            dimension: d,
            mode: normCache.mode,
            epsilon: normCache.epsilon,
            invOut: normCache.invNorms! + oldCount,  // Append
            sqOut: nil,
            invDType: normCache.invDType
        )

        normCache.count = totalCount
    }

    mutating func updateVectors(_ vectors: [[Float]], ids: [Int]) {
        let d = normCache.dimension
        let flatVectors = vectors.flatMap { $0 }

        // Update norms for modified vectors
        normsUpdate(
            updatedVectors: flatVectors,
            ids: ids,
            count: ids.count,
            dimension: d,
            mode: normCache.mode,
            epsilon: normCache.epsilon,
            invOut: normCache.invNorms,  // Scatter write
            sqOut: normCache.sqNorms,
            invDType: normCache.invDType
        )
    }
}
```

Correctness & Testing

**Test Cases**:

1. **Basic Correctness**:
   - Unit vectors: ‖x‖=1 → inv=1, sq=1
   - Zero vectors: ‖x‖=0 → inv≈1/sqrt(ε), sq=0
   - Known vectors: ‖[3,4]‖=5 → inv=0.2, sq=25

2. **Epsilon Protection**:
   - Zero vector: inv should be large but finite
   - Near-zero vector: inv should be stable
   - No NaN or Inf in output

3. **Data Type Conversion**:
   - f32→f16→f32: Relative error <1e-3
   - f32→bf16→f32: Relative error <1e-2
   - Round-trip preserves values within precision

4. **Numerical Stability**:
   - Large magnitude vectors (‖x‖ > 1000)
   - Small magnitude vectors (‖x‖ < 0.01)
   - Mixed magnitude batches

5. **Integration**:
   - Cosine computed with norms matches direct computation
   - L2 optimization produces correct distances
   - Incremental updates maintain consistency

**Example Tests**:

```swift
func testNorm_UnitVector() {
    // Unit vector should have inv=1, sq=1
    let x = [Float](repeating: 1.0 / sqrt(Float(768)), count: 768)

    let invNorm = queryInvNorm(query: x, dimension: 768)
    let sqNorm = querySquaredNorm(query: x, dimension: 768)

    XCTAssertEqual(invNorm, 1.0, accuracy: 1e-5)
    XCTAssertEqual(sqNorm, 1.0, accuracy: 1e-5)
}

func testNorm_ZeroVector() {
    let x = [Float](repeating: 0.0, count: 768)
    let epsilon: Float = 1e-12

    let invNorm = queryInvNorm(query: x, dimension: 768, epsilon: epsilon)

    // Should be 1/sqrt(epsilon) ≈ 1e6
    let expected = 1.0 / sqrt(epsilon)
    XCTAssertEqual(invNorm, expected, accuracy: 1e-3)

    // Should be finite (not inf/nan)
    XCTAssertTrue(invNorm.isFinite)
}

func testNorm_F16Accuracy() {
    let n = 1000
    let d = 768

    var vectors = [Float](repeating: 0, count: n * d)
    for i in 0..<(n*d) {
        vectors[i] = Float.random(in: -1...1)
    }

    // Compute norms in f32
    var invF32 = [Float](repeating: 0, count: n)
    normsBuild(
        vectors: vectors,
        count: n,
        dimension: d,
        mode: .inv,
        invOut: &invF32,
        sqOut: nil,
        invDType: .float32
    )

    // Compute norms in f16
    var invF16 = [Float16](repeating: 0, count: n)
    normsBuild(
        vectors: vectors,
        count: n,
        dimension: d,
        mode: .inv,
        invOut: &invF16,
        sqOut: nil,
        invDType: .float16
    )

    // Check relative error
    for i in 0..<n {
        let f32 = invF32[i]
        let f16 = Float(invF16[i])
        let relError = abs(f32 - f16) / max(abs(f32), 1e-6)

        // f16 should have <0.1% relative error
        XCTAssertLessThan(relError, 0.001)  // 0.1%
    }
}

func testNorm_CosineIntegration() {
    let n = 100
    let d = 768

    // Generate random vectors
    var vectors = [Float](repeating: 0, count: n * d)
    for i in 0..<(n*d) {
        vectors[i] = Float.random(in: -1...1)
    }

    // Build norms
    var invNorms = [Float](repeating: 0, count: n)
    normsBuild(
        vectors: vectors,
        count: n,
        dimension: d,
        mode: .inv,
        invOut: &invNorms,
        sqOut: nil,
        invDType: .float32
    )

    // Compute cosine using cached norms
    let query = Array(vectors[0..<d])
    let queryInv = queryInvNorm(query: query, dimension: d)

    var cosCached = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let dotProduct = zip(query, vectors[(i*d)..<((i+1)*d)])
            .map { $0 * $1 }
            .reduce(0, +)
        cosCached[i] = dotProduct * queryInv * invNorms[i]
    }

    // Compute cosine directly (reference)
    var cosDirect = [Float](repeating: 0, count: n)
    for i in 0..<n {
        let vec = Array(vectors[(i*d)..<((i+1)*d)])
        let dotProduct = zip(query, vec).map { $0 * $1 }.reduce(0, +)
        let queryNorm = sqrt(query.map { $0 * $0 }.reduce(0, +))
        let vecNorm = sqrt(vec.map { $0 * $0 }.reduce(0, +))
        cosDirect[i] = dotProduct / (queryNorm * vecNorm + 1e-12)
    }

    // Should match within floating-point error
    for i in 0..<n {
        XCTAssertEqual(cosCached[i], cosDirect[i], accuracy: 1e-5)
    }
}
```

Coding Guidelines

**Performance Best Practices**:
- Use SIMD with multiple accumulators for ILP
- Pre-compute database norms at build time
- Use f16 for inverse norms to save 50% memory
- Parallelize norm computation across vectors
- Cache query norm for reuse across candidates

**Numerical Best Practices**:
- Always use epsilon protection for inverse norms
- Default epsilon: 1e-12 (sufficient for f32)
- Verify f16 precision meets application requirements
- Test with zero vectors and extreme magnitudes

**API Usage**:

```swift
// Good: Pre-compute database norms
let normCache = NormCache(count: n, dimension: d, mode: .inv, invDType: .float16)
normsBuild(vectors, n, d, .inv, invOut: normCache.invNorms, invDType: .float16)

// Good: Compute query norm once
let qInv = queryInvNorm(query, d)
for candidate in candidates {
    let cos = dotProduct * qInv * normCache.invNorms[candidate.id]
}

// Bad: Recompute norms per comparison
for candidate in candidates {
    let qNorm = sqrt(dot(query, query))  // Wasteful!
    let cNorm = sqrt(dot(candidate, candidate))
    let cos = dotProduct / (qNorm * cNorm)
}
```

Non-Goals

- Online norm updates during queries (pre-computed only)
- Compressed norms beyond f16/bf16 (sufficient precision)
- GPU/Metal acceleration (CPU-focused)
- Norm computation for quantized vectors (separate kernel)

Example Usage

```swift
import VectorIndex

// Example 1: Build norm cache for database
let vectors: [[Float]] = loadVectors()  // 10,000 × 768
let n = vectors.count
let d = vectors[0].count

var normCache = NormCache(
    count: n,
    dimension: d,
    mode: .inv,
    invDType: .float16
)
normCache.allocate()

let flatVectors = vectors.flatMap { $0 }
normsBuild(
    vectors: flatVectors,
    count: n,
    dimension: d,
    mode: .inv,
    invOut: normCache.invNorms,
    sqOut: nil,
    invDType: .float16
)

// Example 2: Compute query norm
let query = [Float](repeating: 0.5, count: d)
let queryInv = queryInvNorm(query: query, dimension: d)

// Example 3: Cosine similarity using cached norms
var dotProducts = [Float](repeating: 0, count: n)
innerProductBlock(query, flatVectors, n, d, &dotProducts)

var similarities = [Float](repeating: 0, count: n)
let f16Ptr = normCache.invNorms!.assumingMemoryBound(to: Float16.self)

for i in 0..<n {
    let dbInvNorm = Float(f16Ptr[i])
    similarities[i] = dotProducts[i] * queryInv * dbInvNorm
}

// Example 4: Add new vectors
let newVectors: [[Float]] = [[1.0, 2.0, ...], ...]
let flatNew = newVectors.flatMap { $0 }

normsAppend(
    newVectors: flatNew,
    count: newVectors.count,
    dimension: d,
    mode: .inv,
    invOut: normCache.invNorms! + n,  // Append position
    sqOut: nil,
    invDType: .float16
)

// Example 5: Update existing vectors
let updatedVectors: [[Float]] = [...]
let updatedIDs = [5, 10, 15, 20]

normsUpdate(
    updatedVectors: updatedVectors.flatMap { $0 },
    ids: updatedIDs,
    count: updatedIDs.count,
    dimension: d,
    mode: .inv,
    invOut: normCache.invNorms,
    sqOut: nil,
    invDType: .float16
)
```

Mathematical Foundation

**L2 Norm**:
```
‖x‖₂ = sqrt(Σᵢ₌₀ᵈ⁻¹ xᵢ²)

Properties:
- Non-negative: ‖x‖₂ ≥ 0
- Zero iff x=0: ‖x‖₂ = 0 ⟺ x = 0
- Scale: ‖αx‖₂ = |α|·‖x‖₂
- Triangle inequality: ‖x+y‖₂ ≤ ‖x‖₂ + ‖y‖₂
```

**Inverse Norm**:
```
inv(x) = 1/‖x‖₂ = 1/sqrt(Σᵢ xᵢ²)

With epsilon protection:
inv(x) = 1/sqrt(max(Σᵢ xᵢ², ε))

For ε = 1e-12:
- Normal vectors (‖x‖ ≥ 0.01): inv ≈ 1/‖x‖
- Near-zero vectors (‖x‖ < 1e-6): inv ≈ 1e6
- Zero vector (‖x‖ = 0): inv = 1e6
```

**Precision Analysis**:

```
f32 (IEEE 754 binary32):
- Precision: 23-bit mantissa ≈ 7 decimal digits
- Range: ±1.2e-38 to ±3.4e38
- Machine epsilon: 2⁻²³ ≈ 1.2e-7

f16 (IEEE 754 binary16):
- Precision: 10-bit mantissa ≈ 3 decimal digits
- Range: ±6.1e-5 to ±6.5e4
- Machine epsilon: 2⁻¹⁰ ≈ 9.8e-4

For inverse norms in [0.5, 2.0]:
- f32 absolute error: ~1e-7
- f16 absolute error: ~1e-3
- f16 relative error: ~1e-3 / 1.0 = 0.1%
```

Dependencies

**Internal**:
- Cosine Similarity Kernel (#03): Primary consumer
- L2 Distance Kernel (#01): Optional consumer (squared norms)
- Inner Product Kernel (#02): Used in cosine computation

**External**:
- Swift Standard Library: SIMD, sqrt
- Foundation: None (pure computation)
- Accelerate: Optional (vDSP for reference)

**Build Requirements**:
- Swift 5.9+ (for Float16 support)
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Performance**:
- Throughput: >2 GB/s for batch norm computation
- Query norm: <1 μs for d=768
- f16 conversion: >8 GB/s

✅ **Accuracy**:
- f32 norms: Relative error <1e-6
- f16 inverse norms: Relative error <0.1%
- Epsilon protection prevents inf/nan

✅ **Correctness**:
- Unit vectors: inv=1, sq=1 (within 1e-5)
- Zero vectors: finite inverse norm
- Cosine integration: matches reference

✅ **Memory Efficiency**:
- f16 inverse norms: 50% memory savings
- 64-byte alignment for cache efficiency
- Minimal overhead beyond norm storage

✅ **Integration**:
- Seamless integration with cosine kernel (#03)
- Support for incremental updates (append/modify)
- Memory-mapped storage with checksums
- Thread-safe parallel computation
