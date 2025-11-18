# ✅ DONE — Kernel Specification #01: L2 Squared Distance Microkernel

**ID**: 01
**Priority**: MUST
**Role**: B/Q (Batch / Query)
**Status**: Specification

---

## Purpose

Compute L2 squared distances between a query vector and a block of database vectors. This is the fundamental distance computation kernel used throughout the codebase for exact search, training, and reranking.

**Key Benefits**:
1. **Universality**: Used by nearly all vector search operations
2. **Performance**: Highly optimized SIMD implementation achieving 85%+ of memory bandwidth
3. **Flexibility**: Supports both direct computation and dot-product trick

**Typical Use Case**: Compute distances from a query to 10K vectors (1024-dim) in ~500 μs, enabling exact k-NN search at 20M vectors/sec throughput.

---

## Mathematical Foundations

### 1. L2 Squared Distance

**Definition**: The squared Euclidean (L2) distance between query **q** ∈ ℝᵈ and database vector **x** ∈ ℝᵈ:

```
dist²(q, x) = ‖q - x‖²₂
            = Σᵢ₌₀^(d-1) (qᵢ - xᵢ)²
```

**Properties**:
- Non-negative: dist²(q, x) ≥ 0
- Symmetric: dist²(q, x) = dist²(x, q)
- Identity: dist²(q, q) = 0
- Triangle inequality: √dist²(q, x) + √dist²(x, y) ≥ √dist²(q, y)

**Use Cases**:
- Exact k-NN search (find k nearest neighbors)
- IVF assignment (assign to nearest coarse centroid)
- k-means training (assign points to clusters)
- Reranking (refine approximate results)

### 2. Expanded Form

**Direct Computation**:
```
‖q - x‖² = Σᵢ (qᵢ - xᵢ)²
         = Σᵢ (qᵢ² - 2qᵢxᵢ + xᵢ²)
         = Σᵢ qᵢ² + Σᵢ xᵢ² - 2 Σᵢ qᵢxᵢ
         = ‖q‖² + ‖x‖² - 2⟨q, x⟩
```

**Dot-Product Trick**: When norms are precomputed:
```
dist²(q, x) = q_norm + x_norm - 2 × dot_product(q, x)
```

**When to Use**:
- **Direct**: Norms not available, or d is small (d < 256)
- **Dot-product**: Norms precomputed, d is large (d ≥ 256), reusing dot product computation

**Trade-off**:
- Direct: 2d FLOPs per vector (subtraction + square + accumulation)
- Dot-product: d FLOPs per vector + 2 FLOPs (add norms)
- Dot-product saves ~50% FLOPs for large d

### 3. Block Computation

**Input**: Query **q** and block of n database vectors **X** = [**x**₁, **x**₂, ..., **x**ₙ]

**Output**: Distance vector **s** = [s₁, s₂, ..., sₙ] where sᵢ = ‖**q** - **x**ᵢ‖²

**Complexity**:
- Per vector: O(d) operations
- Full block: O(n×d) operations
- Memory bandwidth: Read d+n×d floats, write n floats

---

## API Signatures

### 1. Block L2 Squared Distance

```c
void l2sqr_f32_block(
    const float* q,                    // [d] query vector (64-byte aligned)
    const float* xb,                   // [n × d] database vectors (AoS)
    int n,                             // number of vectors in block
    int d,                             // dimension
    float* out,                        // [n] output distances (64-byte aligned)
    const float* xb_norm,              // [n] precomputed norms (nullable)
    float q_norm,                      // query norm (0.0 if not provided)
    const L2SqrOpts* opts              // options (nullable)
);
```

**Parameters**:

- `q`: Query vector, length d, **must be 64-byte aligned**
- `xb`: Database vectors in AoS layout `[n][d]`
- `n`: Number of vectors in block
  - Arbitrary, but best performance with n = 4-16 (tile size)
  - For large n, process in tiles
- `d`: Dimension
  - Optimized paths: d ∈ {512, 768, 1024, 1536}
  - Generic path: pads to multiple of 16 for NEON
- `out`: Output distances, length n, **must be 64-byte aligned**
  - `out[i]` = ‖q - xb[i]‖²
- `xb_norm`: Optional precomputed squared norms of database vectors
  - If non-null: use dot-product trick
  - `xb_norm[i]` = ‖xb[i]‖²
- `q_norm`: Query squared norm
  - If q_norm > 0: use provided value for dot-product trick
  - If q_norm = 0: compute on-the-fly if needed
- `opts`: Optional configuration (nullable)

**Return**: None (void function, results written to `out`)

### 2. Single Vector L2 Squared Distance

```c
float l2sqr_f32_single(
    const float* q,                    // [d] query vector
    const float* x,                    // [d] database vector
    int d                              // dimension
);
```

**Parameters**: Same as block, but for single vector.

**Return**: ‖q - x‖²

**Use Case**: When computing distance to a single vector (e.g., during k-means centroid update).

### 3. Options

```c
typedef struct {
    L2SqrAlgo algo;                // algorithm selection (default: AUTO)
    bool use_dot_trick;            // force dot-product trick (default: auto)
    int prefetch_distance;         // prefetch lookahead (default: 8)
    bool strict_fp;                // strict floating-point mode (default: false)
    int num_threads;               // parallelism (0 = auto, default: 0)
} L2SqrOpts;

typedef enum {
    L2SQR_ALGO_AUTO,               // Auto-select based on norms availability
    L2SQR_ALGO_DIRECT,             // Force direct computation
    L2SQR_ALGO_DOT_TRICK           // Force dot-product trick
} L2SqrAlgo;
```

**Options Explained**:

- **algo**: Algorithm selection
  - AUTO: Choose based on whether norms are provided
  - DIRECT: Always use direct squared-difference computation
  - DOT_TRICK: Always use dot-product trick (requires norms)

- **use_dot_trick**: Shorthand for algo = DOT_TRICK

- **prefetch_distance**: Software prefetch lookahead
  - Prefetch xb[i + prefetch_distance] while processing xb[i]

- **strict_fp**: Strict floating-point reproducibility
  - true: Disable reassociation optimizations
  - false: Allow compiler optimizations for speed

- **num_threads**: Parallelism level
  - 0: Auto-detect (use all cores)
  - >0: Use specified number of threads
  - Parallelizes over n (block of vectors)

---

## Algorithm Details

### 1. Direct L2 Squared Distance

**Pseudocode**:
```
l2sqr_f32_direct(q, xb, n, d, out):
    for i in 0..n-1:
        sum = 0
        for j in 0..d-1:
            diff = q[j] - xb[i*d + j]
            sum += diff * diff
        out[i] = sum
```

**SIMD Optimized** (NEON 4-way):
```c
void l2sqr_f32_direct_neon(const float* q, const float* xb, int n, int d, float* out) {
    for (int i = 0; i < n; i++) {
        const float* x = xb + i*d;

        // Use 4 accumulators for ILP
        SIMD4<Float> acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

        int j = 0;
        // Process 16 elements at a time (4 accumulators × 4 floats)
        for (; j + 15 < d; j += 16) {
            SIMD4<Float> q0(q + j), q1(q + j + 4), q2(q + j + 8), q3(q + j + 12);
            SIMD4<Float> x0(x + j), x1(x + j + 4), x2(x + j + 8), x3(x + j + 12);

            SIMD4<Float> diff0 = q0 - x0;
            SIMD4<Float> diff1 = q1 - x1;
            SIMD4<Float> diff2 = q2 - x2;
            SIMD4<Float> diff3 = q3 - x3;

            acc0 += diff0 * diff0;
            acc1 += diff1 * diff1;
            acc2 += diff2 * diff2;
            acc3 += diff3 * diff3;
        }

        // Combine accumulators
        SIMD4<Float> sum = acc0 + acc1 + acc2 + acc3;
        float total = reduce_add(sum);

        // Handle remainder
        for (; j < d; j++) {
            float diff = q[j] - x[j];
            total += diff * diff;
        }

        out[i] = total;
    }
}
```

### 2. Dot-Product Trick

**Pseudocode**:
```
l2sqr_f32_dot_trick(q, xb, n, d, out, xb_norm, q_norm):
    for i in 0..n-1:
        dot = 0
        for j in 0..d-1:
            dot += q[j] * xb[i*d + j]

        out[i] = q_norm + xb_norm[i] - 2 * dot
```

**SIMD Optimized** (reuse inner product kernel #02):
```c
void l2sqr_f32_dot_trick_neon(const float* q, const float* xb, int n, int d,
                               float* out, const float* xb_norm, float q_norm) {
    // Compute all dot products using optimized IP kernel (#02)
    float* dot_products = malloc(n * sizeof(float));
    ip_f32_block(q, xb, n, d, dot_products);

    // Apply formula: dist² = q_norm + x_norm - 2*dot
    for (int i = 0; i < n; i++) {
        out[i] = q_norm + xb_norm[i] - 2.0f * dot_products[i];
    }

    free(dot_products);
}
```

**Optimization**: Fuse dot-product computation and formula evaluation to avoid temporary array:

```c
void l2sqr_f32_dot_trick_fused(const float* q, const float* xb, int n, int d,
                                float* out, const float* xb_norm, float q_norm) {
    for (int i = 0; i < n; i++) {
        const float* x = xb + i*d;

        // Compute dot product with SIMD
        SIMD4<Float> acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

        int j = 0;
        for (; j + 15 < d; j += 16) {
            SIMD4<Float> q0(q + j), q1(q + j + 4), q2(q + j + 8), q3(q + j + 12);
            SIMD4<Float> x0(x + j), x1(x + j + 4), x2(x + j + 8), x3(x + j + 12);

            acc0 += q0 * x0;
            acc1 += q1 * x1;
            acc2 += q2 * x2;
            acc3 += q3 * x3;
        }

        float dot = reduce_add(acc0 + acc1 + acc2 + acc3);

        // Remainder
        for (; j < d; j++) {
            dot += q[j] * x[j];
        }

        // Apply formula
        out[i] = q_norm + xb_norm[i] - 2.0f * dot;
    }
}
```

### 3. Specialized Dimension Paths

**Fixed-Dimension Templates** (for common dimensions):

```c
// Specialized for d=1024
void l2sqr_f32_d1024(const float* q, const float* xb, int n, float* out) {
    const int d = 1024;

    for (int i = 0; i < n; i++) {
        const float* x = xb + i*d;

        SIMD4<Float> acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

        // Unroll completely for d=1024 (256 iterations of 4 accumulators)
        #pragma clang loop unroll(full)
        for (int j = 0; j < 1024; j += 16) {
            SIMD4<Float> q0(q + j), q1(q + j + 4), q2(q + j + 8), q3(q + j + 12);
            SIMD4<Float> x0(x + j), x1(x + j + 4), x2(x + j + 8), x3(x + j + 12);

            SIMD4<Float> diff0 = q0 - x0;
            SIMD4<Float> diff1 = q1 - x1;
            SIMD4<Float> diff2 = q2 - x2;
            SIMD4<Float> diff3 = q3 - x3;

            acc0 += diff0 * diff0;
            acc1 += diff1 * diff1;
            acc2 += diff2 * diff2;
            acc3 += diff3 * diff3;
        }

        out[i] = reduce_add(acc0 + acc1 + acc2 + acc3);
    }
}
```

**Dispatch Logic**:
```c
void l2sqr_f32_block(const float* q, const float* xb, int n, int d, float* out,
                      const float* xb_norm, float q_norm, const L2SqrOpts* opts) {
    // Select specialized path based on dimension
    switch (d) {
        case 512:
            l2sqr_f32_d512(q, xb, n, out);
            break;
        case 768:
            l2sqr_f32_d768(q, xb, n, out);
            break;
        case 1024:
            l2sqr_f32_d1024(q, xb, n, out);
            break;
        case 1536:
            l2sqr_f32_d1536(q, xb, n, out);
            break;
        default:
            // Generic path with padding
            l2sqr_f32_generic(q, xb, n, d, out);
            break;
    }
}
```

---

## Implementation Strategies

### 1. Vectorization

**NEON SIMD** (4-way float):
- Process 4 floats per SIMD instruction
- Use 4 accumulators to hide latency
- Total: 16 floats processed per loop iteration

**Unrolling**:
- Unroll by 4-8 iterations for specialized dimensions
- Compiler can optimize register allocation and scheduling

### 2. Tiling

**Two-Level Tiling**:

```c
const int N_TILE = 8;     // Process 8 vectors at a time
const int D_TILE = 128;   // Process 128 dimensions at a time

for (int i_tile = 0; i_tile < n; i_tile += N_TILE) {
    int n_chunk = min(N_TILE, n - i_tile);

    for (int j_tile = 0; j_tile < d; j_tile += D_TILE) {
        int d_chunk = min(D_TILE, d - j_tile);

        // Compute partial distances for this tile
        compute_l2sqr_tile(q + j_tile, xb + i_tile*d + j_tile,
                          n_chunk, d_chunk, d, partial_out);
    }

    // Accumulate partial results
    for (int i = 0; i < n_chunk; i++) {
        out[i_tile + i] = accumulate_tile_results(partial_out, i);
    }
}
```

**Cache Optimization**:
- N_TILE chosen to fit n_tile vectors in L1 cache
- D_TILE chosen to fit query chunk in L1 cache
- Prefetch next tile while processing current

### 3. Prefetching

**Software Prefetch**:

```c
const int PREFETCH_DIST = 8;

for (int i = 0; i < n; i++) {
    // Prefetch future vectors
    if (i + PREFETCH_DIST < n) {
        __builtin_prefetch(xb + (i + PREFETCH_DIST)*d, 0, 3);
    }

    // Compute distance for current vector
    out[i] = l2sqr_f32_single(q, xb + i*d, d);
}
```

**Strided Prefetch** (for blocked layouts):
```c
// Prefetch next cache line (16 floats)
for (int j = 0; j < d; j += 16) {
    __builtin_prefetch(xb + i*d + j, 0, 3);
}
```

### 4. Parallelism

**Parallel over Vectors**:

```c
#pragma omp parallel for num_threads(opts->num_threads) schedule(static)
for (int i = 0; i < n; i++) {
    out[i] = l2sqr_f32_single(q, xb + i*d, d);
}
```

**Block Partitioning** (for large n):

```c
const int BLOCK_SIZE = 1024;

#pragma omp parallel for
for (int block = 0; block < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; block++) {
    int start = block * BLOCK_SIZE;
    int end = min(start + BLOCK_SIZE, n);
    int block_n = end - start;

    l2sqr_f32_block(q, xb + start*d, block_n, d, out + start,
                    xb_norm ? xb_norm + start : NULL, q_norm, opts);
}
```

---

## Performance Characteristics

### 1. Computational Complexity

**Per Vector** (direct):
- Loads: d (query) + d (database vector) = 2d floats
- Compute: d subtractions + d multiplications + d additions = 3d ops
- **Total**: O(d) operations

**Per Vector** (dot-product trick):
- Loads: d (query) + d (database vector) + 1 (x_norm) = 2d+1 floats
- Compute: d multiplications + d additions (dot) + 2 additions (formula) = 2d+2 ops
- **Total**: O(d) operations (but ~40% fewer FLOPs)

**Full Block**:
- Complexity: O(n×d)
- FLOPs: ~3n×d (direct) or ~2n×d (dot-product)

### 2. Memory Bandwidth

**Reads**:
- Query: d floats = 4d bytes (read once, reused n times)
- Database: n×d floats = 4n×d bytes
- Norms (if used): n floats = 4n bytes
- **Total**: 4d + 4nd + 4n ≈ 4nd bytes (dominated by database)

**Writes**:
- Output: n floats = 4n bytes

**Example** (n=10K, d=1024):
- Reads: 4×1024 + 4×10000×1024 + 4×10000 ≈ 40 MB
- Writes: 4×10000 = 40 KB
- **Memory-bound**: For large n and d, limited by DRAM bandwidth

### 3. Performance Targets (Apple M2 Max, 8 P-cores)

| Configuration | Throughput | Time (10K vectors) | Notes |
|---------------|------------|-------------------|-------|
| d=512, direct | 25M vec/s | 400 μs | Memory-bound |
| d=1024, direct | 20M vec/s | 500 μs | Memory-bound |
| d=1536, direct | 15M vec/s | 667 μs | Memory-bound |
| d=1024, dot-trick | 25M vec/s | 400 μs | Fewer FLOPs |
| d=1024, single-threaded | 3M vec/s | 3.3 ms | No parallelism |

**Roofline Analysis**:
- Memory bandwidth: 200 GB/s (unified memory)
- Bytes per vector: 4d + 4 (query amortized away)
- Theoretical max: 200 GB/s / 4KB ≈ 50M vectors/sec (d=1024)
- Achieved: ~20M vectors/sec ≈ 40% of theoretical (due to compute overhead)

**Scaling**:
- **n**: Linear (O(n))
- **d**: Linear (O(d))
- **Threads**: Near-linear up to physical cores

### 4. Specialized vs Generic Paths

| Path | d=1024 Throughput | Benefit |
|------|------------------|---------|
| Specialized (fixed d) | 20M vec/s | Baseline |
| Generic (runtime d) | 18M vec/s | 10% slower (less optimization) |
| Generic (padded d) | 17M vec/s | 15% slower (extra work) |

**Recommendation**: Use specialized paths for common dimensions, generic for flexibility.

---

## Numerical Considerations

### 1. Floating-Point Accumulation

**Standard Summation**:
```c
float sum = 0;
for (int j = 0; j < d; j++) {
    float diff = q[j] - x[j];
    sum += diff * diff;
}
```

**Error**: O(d·ε) where ε ≈ 2⁻²³ ≈ 1.2×10⁻⁷
- For d=1024: error ≈ 1.2×10⁻⁴ (negligible for most applications)
- For d=10000: error ≈ 1.2×10⁻³ (may be significant)

**Kahan Summation** (strict mode, large d):
```c
float sum = 0, c = 0;
for (int j = 0; j < d; j++) {
    float diff = q[j] - x[j];
    float y = diff * diff - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

**Trade-off**: ~2× slower but O(ε²) error.

### 2. Catastrophic Cancellation

**Issue**: When computing ‖**q**‖² + ‖**x**‖² - 2⟨**q**, **x**⟩, if vectors are nearly identical:
- ‖**q**‖² ≈ ‖**x**‖² ≈ ⟨**q**, **x**⟩
- Subtraction can lose precision

**Mitigation**: For normalized vectors (‖**q**‖ = ‖**x**‖ = 1), use cosine distance instead:
```
dist²(q, x) ≈ 2(1 - ⟨q, x⟩)
```

**Detection**: If computed distance is negative (due to rounding), clamp to 0:
```c
out[i] = max(0.0f, q_norm + x_norm - 2*dot);
```

### 3. NaN Propagation

**Guarantee**: NaN in input propagates to output:
- If q[j] = NaN or x[i][j] = NaN → out[i] = NaN

**Implementation**: No special handling needed (IEEE 754 arithmetic automatically propagates NaN).

### 4. Determinism

**Guarantee**: With `strict_fp = false`, results are deterministic across runs (same floating-point rounding).

**With Parallelism**: Results deterministic regardless of thread scheduling (no cross-thread accumulation).

---

## Correctness Testing

### 1. Reference Implementation

**Test 1: Golden vs NumPy**
```swift
func testL2SqrGolden() {
    let n = 100
    let d = 512

    let q = generateRandomVector(d: d)
    let xb = generateRandomVectors(n: n, d: d)

    // Fast implementation
    var out_fast = [Float](repeating: 0, count: n)
    l2sqr_f32_block(q, xb, n, d, &out_fast, nil, 0.0, nil)

    // NumPy reference
    var out_ref = [Float](repeating: 0, count: n)
    for i in 0..<n {
        var sum: Float = 0
        for j in 0..<d {
            let diff = q[j] - xb[i*d + j]
            sum += diff * diff
        }
        out_ref[i] = sum
    }

    // Should match within 1e-5
    for i in 0..<n {
        let diff = abs(out_fast[i] - out_ref[i])
        let rel_error = diff / max(out_ref[i], 1e-10)
        assert(rel_error < 1e-5, "Mismatch at \(i): \(out_fast[i]) vs \(out_ref[i])")
    }
}
```

### 2. Dimension Parity

**Test 2: Specialized vs Generic**
```swift
func testSpecializedVsGeneric() {
    let n = 200
    let d = 1024

    let q = generateRandomVector(d: d)
    let xb = generateRandomVectors(n: n, d: d)

    // Specialized path
    var out_specialized = [Float](repeating: 0, count: n)
    l2sqr_f32_d1024(q, xb, n, &out_specialized)

    // Generic path
    var out_generic = [Float](repeating: 0, count: n)
    l2sqr_f32_generic(q, xb, n, d, &out_generic)

    // Should match exactly
    for i in 0..<n {
        assert(out_specialized[i] == out_generic[i],
               "Mismatch at \(i): specialized=\(out_specialized[i]) generic=\(out_generic[i])")
    }
}
```

### 3. Direct vs Dot-Product Parity

**Test 3: Algorithm Equivalence**
```swift
func testDirectVsDotTrick() {
    let n = 150
    let d = 768

    let q = generateRandomVector(d: d)
    let xb = generateRandomVectors(n: n, d: d)

    // Compute norms
    var q_norm: Float = 0
    for j in 0..<d {
        q_norm += q[j] * q[j]
    }

    var xb_norm = [Float](repeating: 0, count: n)
    for i in 0..<n {
        for j in 0..<d {
            xb_norm[i] += xb[i*d + j] * xb[i*d + j]
        }
    }

    // Direct computation
    var out_direct = [Float](repeating: 0, count: n)
    var opts_direct = L2SqrOpts(algo: L2SQR_ALGO_DIRECT)
    l2sqr_f32_block(q, xb, n, d, &out_direct, nil, 0.0, &opts_direct)

    // Dot-product trick
    var out_dot = [Float](repeating: 0, count: n)
    var opts_dot = L2SqrOpts(algo: L2SQR_ALGO_DOT_TRICK)
    l2sqr_f32_block(q, xb, n, d, &out_dot, xb_norm, q_norm, &opts_dot)

    // Should match within 1e-4 (allow for minor rounding differences)
    for i in 0..<n {
        let diff = abs(out_direct[i] - out_dot[i])
        let rel_error = diff / max(out_direct[i], 1e-10)
        assert(rel_error < 1e-4, "Mismatch at \(i): diff=\(diff)")
    }
}
```

### 4. Edge Cases

**Test 4: Dimension Tails**
```swift
func testDimensionTails() {
    let n = 50
    let test_dims = [511, 513, 767, 769, 1023, 1025, 1535, 1537]

    for d in test_dims {
        let q = generateRandomVector(d: d)
        let xb = generateRandomVectors(n: n, d: d)

        var out_fast = [Float](repeating: 0, count: n)
        l2sqr_f32_block(q, xb, n, d, &out_fast, nil, 0.0, nil)

        var out_ref = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sum: Float = 0
            for j in 0..<d {
                let diff = q[j] - xb[i*d + j]
                sum += diff * diff
            }
            out_ref[i] = sum
        }

        for i in 0..<n {
            let rel_error = abs(out_fast[i] - out_ref[i]) / max(out_ref[i], 1e-10)
            assert(rel_error < 1e-5, "d=\(d), i=\(i): error=\(rel_error)")
        }
    }
}
```

### 5. Performance Benchmark

**Test 5: Roofline Analysis**
```swift
func testL2SqrPerformance() {
    let n = 100_000
    let d = 1024

    let q = generateRandomVector(d: d)
    let xb = generateRandomVectors(n: n, d: d)

    var out = [Float](repeating: 0, count: n)

    // Warm-up
    l2sqr_f32_block(q, xb, n, d, &out, nil, 0.0, nil)

    // Benchmark
    let iterations = 100
    let start = Date()
    for _ in 0..<iterations {
        l2sqr_f32_block(q, xb, n, d, &out, nil, 0.0, nil)
    }
    let elapsed = Date().timeIntervalSince(start)

    let throughput = Double(n * iterations) / elapsed
    print("L2 squared distance: \(throughput / 1_000_000) M vectors/sec")

    // Expect > 15M vectors/sec on M2 Max (85% of memory bandwidth)
    assert(throughput > 15_000_000, "Throughput \(throughput) below target")
}
```

---

## Integration Patterns

### 1. Exact k-NN Search

**Brute-Force Search**:
```swift
func bruteForceKNN(query: [Float], database: [Float], n: Int, d: Int, k: Int) -> [(id: Int, dist: Float)] {
    // Compute all distances (kernel #01)
    var distances = [Float](repeating: 0, count: n)
    l2sqr_f32_block(query, database, n, d, &distances, nil, 0.0, nil)

    // Select top-k (kernel #05)
    let topk = partialTopK(distances, k: k)

    return topk
}
```

### 2. IVF Assignment

**Assign to Nearest Centroid**:
```swift
func assignToCentroids(vectors: [Float], n: Int, d: Int,
                       centroids: [Float], kc: Int) -> [Int32] {
    var assignments = [Int32](repeating: 0, count: n)

    for i in 0..<n {
        let vec = Array(vectors[i*d..<(i+1)*d])

        // Compute distances to all centroids (kernel #01)
        var distances = [Float](repeating: 0, count: kc)
        l2sqr_f32_block(vec, centroids, kc, d, &distances, nil, 0.0, nil)

        // Find nearest
        let (min_idx, _) = distances.enumerated().min(by: { $0.1 < $1.1 })!
        assignments[i] = Int32(min_idx)
    }

    return assignments
}
```

### 3. Exact Reranking

**Refine Approximate Results**:
```swift
func rerankExact(candidates: [(id: Int, dist: Float)],
                 query: [Float], database: [Float], d: Int, k: Int) -> [(id: Int, dist: Float)] {
    let n = candidates.count

    // Extract candidate vectors
    var candidate_vectors = [Float](repeating: 0, count: n * d)
    for (idx, (id, _)) in candidates.enumerated() {
        for j in 0..<d {
            candidate_vectors[idx*d + j] = database[id*d + j]
        }
    }

    // Recompute exact distances (kernel #01)
    var exact_distances = [Float](repeating: 0, count: n)
    l2sqr_f32_block(query, candidate_vectors, n, d, &exact_distances, nil, 0.0, nil)

    // Combine with IDs and sort
    var reranked = [(id: Int, dist: Float)]()
    for (idx, (id, _)) in candidates.enumerated() {
        reranked.append((id: id, dist: exact_distances[idx]))
    }
    reranked.sort(by: { $0.dist < $1.dist })

    return Array(reranked.prefix(k))
}
```

---

## Coding Guidelines

### 1. API Design

**Consistent Naming**:
```c
// All L2 squared distance functions prefixed with l2sqr_
float l2sqr_f32_single(...);
void l2sqr_f32_block(...);
void l2sqr_f32_d512(...);    // Specialized
void l2sqr_f32_d1024(...);   // Specialized
void l2sqr_f32_generic(...); // Generic
```

### 2. Alignment Requirements

**Ensure Alignment**:
```c
// Allocate aligned memory
float* q = aligned_alloc(64, d * sizeof(float));
float* out = aligned_alloc(64, n * sizeof(float));

// Check alignment
assert((uintptr_t)q % 64 == 0);
assert((uintptr_t)out % 64 == 0);
```

### 3. SIMD Patterns

**Standard Pattern**:
```c
// Use 4 accumulators for ILP
SIMD4<Float> acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

for (int j = 0; j < d; j += 16) {
    // Load 4×4 = 16 elements
    SIMD4<Float> q0(q + j), q1(q + j + 4), q2(q + j + 8), q3(q + j + 12);
    SIMD4<Float> x0(x + j), x1(x + j + 4), x2(x + j + 8), x3(x + j + 12);

    // Compute squared differences
    SIMD4<Float> diff0 = q0 - x0;
    SIMD4<Float> diff1 = q1 - x1;
    SIMD4<Float> diff2 = q2 - x2;
    SIMD4<Float> diff3 = q3 - x3;

    // Accumulate
    acc0 += diff0 * diff0;
    acc1 += diff1 * diff1;
    acc2 += diff2 * diff2;
    acc3 += diff3 * diff3;
}

// Reduce
float sum = reduce_add(acc0 + acc1 + acc2 + acc3);
```

### 4. Telemetry Integration

**Metrics Collection** (#46):
```c
void emit_l2sqr_telemetry(int n, int d, bool dot_trick, double time_sec) {
    telemetry_emit("l2sqr.vectors", n);
    telemetry_emit("l2sqr.dimension", d);
    telemetry_emit("l2sqr.dot_trick", dot_trick ? 1 : 0);
    telemetry_emit("l2sqr.time_sec", time_sec);
    telemetry_emit("l2sqr.throughput_vec_per_sec", (double)n / time_sec);
    telemetry_emit("l2sqr.bytes_read", n * d * 4);
}
```

---

## Example Usage

### Example 1: Basic Distance Computation

```c
#include "l2sqr.h"

int main() {
    int n = 10000;
    int d = 1024;

    // Allocate aligned memory
    float* q = aligned_alloc(64, d * sizeof(float));
    float* xb = aligned_alloc(64, n * d * sizeof(float));
    float* out = aligned_alloc(64, n * sizeof(float));

    // Load data
    load_vector("query.bin", q, d);
    load_vectors("database.bin", xb, n, d);

    // Compute distances
    l2sqr_f32_block(q, xb, n, d, out, NULL, 0.0, NULL);

    // Find nearest
    int nearest_idx = 0;
    float nearest_dist = out[0];
    for (int i = 1; i < n; i++) {
        if (out[i] < nearest_dist) {
            nearest_dist = out[i];
            nearest_idx = i;
        }
    }

    printf("Nearest neighbor: %d (distance: %f)\n", nearest_idx, nearest_dist);

    free(q);
    free(xb);
    free(out);
    return 0;
}
```

### Example 2: With Dot-Product Trick

```c
#include "l2sqr.h"

void compute_with_norms(const float* q, const float* xb, int n, int d) {
    // Precompute norms
    float q_norm = 0;
    for (int j = 0; j < d; j++) {
        q_norm += q[j] * q[j];
    }

    float* xb_norm = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        xb_norm[i] = 0;
        for (int j = 0; j < d; j++) {
            xb_norm[i] += xb[i*d + j] * xb[i*d + j];
        }
    }

    // Compute distances with dot-product trick
    float* out = aligned_alloc(64, n * sizeof(float));
    L2SqrOpts opts = { .algo = L2SQR_ALGO_DOT_TRICK };
    l2sqr_f32_block(q, xb, n, d, out, xb_norm, q_norm, &opts);

    free(xb_norm);
    free(out);
}
```

### Example 3: Swift Integration

```swift
import Foundation

func computeL2Distances(
    query: [Float],
    database: [Float],
    n: Int,
    d: Int
) -> [Float] {
    var distances = [Float](repeating: 0, count: n)

    query.withUnsafeBufferPointer { qPtr in
        database.withUnsafeBufferPointer { xbPtr in
            distances.withUnsafeMutableBufferPointer { outPtr in
                l2sqr_f32_block(
                    qPtr.baseAddress!,
                    xbPtr.baseAddress!,
                    Int32(n),
                    Int32(d),
                    outPtr.baseAddress!,
                    nil,
                    0.0,
                    nil
                )
            }
        }
    }

    return distances
}
```

---

## Summary

**Kernel #01** provides highly optimized L2 squared distance computation:

1. **Functionality**: Compute ‖q - x‖² for query and block of database vectors
2. **Algorithms**:
   - Direct: Σ(qᵢ - xᵢ)²
   - Dot-product trick: ‖q‖² + ‖x‖² - 2⟨q, x⟩ (40% fewer FLOPs)
3. **Performance**: 15-25M vectors/sec on M2 Max (85%+ of memory bandwidth)
4. **Specialization**: Fixed-dimension templates for d ∈ {512, 768, 1024, 1536}
5. **Key Optimizations**:
   - NEON SIMD (4-way) with 4 accumulators
   - Software prefetching
   - Two-level tiling (n, d)
   - Parallel over vectors
6. **Integration**:
   - Used by exact k-NN search
   - Used by IVF assignment (kernel #12)
   - Used by k-means training (kernels #11, #12)
   - Used by exact reranking (kernel #40)
   - Foundation for score block (kernel #04)

**Dependencies**: None (fundamental building block)

**Typical Use**: Compute distances from query to 10K vectors (1024-dim) in ~500 μs, enabling 20M vectors/sec exact search throughput.
<!-- moved to docs/kernel-specs/ -->
