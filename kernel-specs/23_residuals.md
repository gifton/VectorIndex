# Kernel Specification #23: Residual Computation (x - centroid)

**ID**: 23
**Priority**: MUST
**Role**: B/Q (Batch / Query)
**Status**: Specification

---

## Purpose

Compute residual vectors **r** = **x** - **c** for IVF-PQ training and query pipelines. Provides both materialized (explicit) and fused (on-the-fly) computation paths to optimize memory usage and performance.

**Key Benefits**:
1. **Memory efficiency**: Fused paths avoid materializing n×d residual vectors
2. **Cache efficiency**: Compute residuals in registers during PQ operations
3. **Accuracy**: Residual PQ achieves 5-10% better recall than direct PQ

**Typical Use Case**: For IVF-PQ with 10M vectors, avoid materializing 40 GB of residuals by computing on-the-fly during encoding.

---

## Mathematical Foundations

### 1. IVF Residuals

**Inverted File (IVF) Structure**: Partition vectors into kc coarse clusters using k-means centroids **C** = {**c**₁, ..., **c**ₖc}.

**Coarse Assignment**: Each vector **x**ᵢ is assigned to nearest coarse centroid:
```
a(i) = argmin_j ‖xᵢ - cⱼ‖²
```

**Residual Vector**:
```
rᵢ = xᵢ - c_{a(i)}
```

The residual represents the difference between the original vector and its coarse cluster center.

**Properties**:
- **Lower variance**: ‖**r**ᵢ‖ < ‖**x**ᵢ‖ typically (residuals are closer to origin)
- **Better quantization**: PQ on residuals achieves lower distortion than PQ on original vectors
- **Typical improvement**: 5-10% better recall at same compression ratio

### 2. Why Residuals?

**Problem with Direct PQ**: Quantizing high-variance vectors with limited codebook size leads to high distortion.

**Residual PQ Solution**:
1. Coarse quantization reduces variance: **x** → **c**ₐ₍ᵢ₎
2. Fine quantization (PQ) on low-variance residuals: **r** → PQ(**r**)
3. Reconstruction: **x̂** = **c**ₐ₍ᵢ₎ + PQ(**r**)

**Distance Computation**:
```
‖q - x‖² ≈ ‖q - (c_{a(i)} + PQ(r))‖²
         = ‖(q - c_{a(i)}) - PQ(r)‖²
         = ‖r_q - PQ(r)‖²
```

where **r**_q = **q** - **c**ₐ₍ᵢ₎ is the query residual.

### 3. Fused vs Materialized Computation

**Materialized (Explicit)**:
```
1. Compute all residuals: R = [r₁, r₂, ..., rₙ]
2. Store R (n×d floats = 4n×d bytes)
3. Perform PQ training/encoding on R
```

**Memory**: O(n×d) floats

**Fused (On-the-fly)**:
```
1. For each vector xᵢ:
   2. Load centroid c_{a(i)}
   3. Compute residual r = x - c in registers
   4. Immediately use r for PQ operation (training/encoding/LUT)
   5. Discard r (never materialized)
```

**Memory**: O(d) floats (single vector residual in registers)

**Savings**: For n=10M, d=1024, save 40 GB of memory with fused computation.

### 4. Centroid Locality

**Challenge**: Computing residuals requires loading coarse centroids. If assignments are random, poor cache locality.

**Solution**: Group vectors by coarse assignment before computing residuals:
```
1. Sort indices by assignment: group all vectors assigned to centroid c₁, then c₂, etc.
2. Process each group sequentially
3. Load centroid once, reuse for all vectors in group
```

**Benefit**: Instead of kc random centroid loads, load each centroid once.

---

## API Signatures

### 1. Standard Residual Computation

```c
void residuals_f32(
    const float* x,                    // [n × d] input vectors (AoS)
    const int32_t* coarse_ids,         // [n] coarse assignments
    const float* coarse_centroids,     // [kc × d] coarse centroids
    int64_t n,                         // number of vectors
    int d,                             // dimension
    float* r_out,                      // [n × d] output residuals
    const ResidualOpts* opts           // options (nullable)
);
```

**Parameters**:

- `x`: Input vectors, layout `[n][d]`
- `coarse_ids`: Coarse centroid assignments
  - `coarse_ids[i]` = index of coarse centroid for vector i
  - Range: [0, kc-1]
- `coarse_centroids`: Coarse centroids from IVF training, layout `[kc][d]`
- `n`: Number of vectors
- `d`: Dimension (typical: 512, 768, 1024, 1536)
- `r_out`: Output residuals, **must be preallocated** to n×d floats
  - `r_out[i*d : (i+1)*d]` = residual for vector i
- `opts`: Optional configuration (nullable)

**Output**:
```
r_out[i] = x[i] - coarse_centroids[coarse_ids[i]]
```

### 2. In-Place Residual Computation

```c
void residuals_f32_inplace(
    float* x_io,                       // [n × d] input/output vectors (AoS)
    const int32_t* coarse_ids,         // [n] coarse assignments
    const float* coarse_centroids,     // [kc × d] coarse centroids
    int64_t n,                         // number of vectors
    int d,                             // dimension
    const ResidualOpts* opts           // options (nullable)
);
```

**Parameters**: Same as standard, except:
- `x_io`: Input vectors, overwritten with residuals in-place

**Output**: `x_io[i]` contains residual after call.

**Use Case**: When original vectors are no longer needed, save memory by computing residuals in-place.

### 3. Fused Residual PQ Encoding

Compute residuals on-the-fly during PQ encoding (kernel #20).

```c
void residual_pq_encode_u8_f32(
    const float* x,                    // [n × d] input vectors
    const int32_t* coarse_ids,         // [n] coarse assignments
    const float* coarse_centroids,     // [kc × d] coarse centroids
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int m,                             // number of subspaces
    int ks,                            // codebook size (256)
    const float* codebooks,            // [m × ks × dsub] PQ codebooks
    uint8_t* codes,                    // [n × m] output codes
    const PQEncodeOpts* opts           // encoding options (nullable)
);
```

**Workflow**:
```
For each vector i:
  1. Load x[i]
  2. Load centroid c[coarse_ids[i]]
  3. Compute residual r = x[i] - c in registers
  4. Encode r using PQ codebooks (kernel #20)
  5. Write codes[i]
```

**Benefit**: Never materialize full residual array, save O(n×d) memory.

### 4. Fused Residual LUT Construction

Compute query residual on-the-fly during LUT construction (kernel #21).

```c
void residual_pq_lut_f32(
    const float* q,                    // [d] query vector
    const float* coarse_centroid,      // [d] coarse centroid (single)
    int d,                             // dimension
    int m,                             // number of subspaces
    int ks,                            // codebook size
    const float* codebooks,            // [m × ks × dsub] PQ codebooks
    float* lut,                        // [m × ks] output LUT
    const float* centroid_norms,       // [m × ks] norms (nullable)
    const PQLutOpts* opts              // LUT options (nullable)
);
```

**Workflow**:
```
1. Compute query residual r_q = q - coarse_centroid
2. Build LUT for r_q (kernel #21)
```

**Benefit**: Avoids separate residual computation step, more cache-friendly.

### 5. Options

```c
typedef struct {
    bool group_by_centroid;        // group vectors by assignment (default: false)
    int prefetch_distance;         // prefetch lookahead (default: 8)
    bool check_bounds;             // validate coarse_ids range (default: false)
    int num_threads;               // parallelism (0 = auto, default: 0)
} ResidualOpts;
```

**Options Explained**:

- **group_by_centroid**: Whether to process vectors grouped by their coarse assignment
  - true: Sort/group by centroid for better cache locality
  - false: Process in original order
  - Trade-off: Grouping overhead vs cache benefits (beneficial for large kc and random assignments)

- **prefetch_distance**: Software prefetch lookahead
  - Prefetch x[i + prefetch_distance] and coarse_centroids[coarse_ids[i + prefetch_distance]]

- **check_bounds**: Validate coarse_ids are in valid range [0, kc-1]
  - Enable for debugging, disable for production (small overhead)

- **num_threads**: Parallelism level
  - 0: Auto-detect
  - >0: Use specified number of threads

---

## Algorithm Details

### 1. Standard Residual Computation

**Pseudocode**:
```
residuals_f32(x, coarse_ids, coarse_centroids, n, d, r_out, opts):
    for i in 0..n-1:
        a = coarse_ids[i]
        centroid = coarse_centroids[a*d : (a+1)*d]

        for j in 0..d-1:
            r_out[i*d + j] = x[i*d + j] - centroid[j]
```

**SIMD Optimized**:
```c
void residuals_f32(const float* x, const int32_t* coarse_ids,
                    const float* coarse_centroids, int64_t n, int d,
                    float* r_out, const ResidualOpts* opts) {
    for (int64_t i = 0; i < n; i++) {
        int32_t a = coarse_ids[i];
        const float* vec = x + i*d;
        const float* centroid = coarse_centroids + a*d;
        float* residual = r_out + i*d;

        // SIMD subtraction (process 4 floats at a time)
        int j = 0;
        for (; j + 3 < d; j += 4) {
            SIMD4<Float> v(vec + j);
            SIMD4<Float> c(centroid + j);
            SIMD4<Float> r = v - c;
            r.store(residual + j);
        }

        // Remainder
        for (; j < d; j++) {
            residual[j] = vec[j] - centroid[j];
        }
    }
}
```

### 2. In-Place Residual Computation

**Pseudocode**:
```
residuals_f32_inplace(x_io, coarse_ids, coarse_centroids, n, d, opts):
    for i in 0..n-1:
        a = coarse_ids[i]
        centroid = coarse_centroids[a*d : (a+1)*d]

        for j in 0..d-1:
            x_io[i*d + j] -= centroid[j]
```

**SIMD Optimized**:
```c
void residuals_f32_inplace(float* x_io, const int32_t* coarse_ids,
                            const float* coarse_centroids, int64_t n, int d,
                            const ResidualOpts* opts) {
    for (int64_t i = 0; i < n; i++) {
        int32_t a = coarse_ids[i];
        float* vec = x_io + i*d;
        const float* centroid = coarse_centroids + a*d;

        int j = 0;
        for (; j + 3 < d; j += 4) {
            SIMD4<Float> v(vec + j);
            SIMD4<Float> c(centroid + j);
            SIMD4<Float> r = v - c;
            r.store(vec + j);  // Overwrite in-place
        }

        for (; j < d; j++) {
            vec[j] -= centroid[j];
        }
    }
}
```

### 3. Grouped Residual Computation

**Motivation**: Improve cache locality by processing all vectors assigned to same centroid together.

**Pseudocode**:
```
residuals_f32_grouped(x, coarse_ids, coarse_centroids, n, d, r_out, opts):
    // Build inverse index: centroid_id -> [vector indices]
    groups = build_groups(coarse_ids, n)

    // Process each group
    for (c, indices) in groups:
        centroid = coarse_centroids[c*d : (c+1)*d]

        // Load centroid once, reuse for all vectors in group
        for i in indices:
            for j in 0..d-1:
                r_out[i*d + j] = x[i*d + j] - centroid[j]
```

**Implementation**:
```c
void residuals_f32_grouped(const float* x, const int32_t* coarse_ids,
                            const float* coarse_centroids, int64_t n, int d,
                            float* r_out, int kc) {
    // Count vectors per centroid
    int* counts = calloc(kc, sizeof(int));
    for (int64_t i = 0; i < n; i++) {
        counts[coarse_ids[i]]++;
    }

    // Build offset array (prefix sum)
    int* offsets = malloc((kc + 1) * sizeof(int));
    offsets[0] = 0;
    for (int c = 0; c < kc; c++) {
        offsets[c+1] = offsets[c] + counts[c];
    }

    // Build grouped indices
    int* grouped_indices = malloc(n * sizeof(int));
    int* current_pos = calloc(kc, sizeof(int));
    for (int64_t i = 0; i < n; i++) {
        int c = coarse_ids[i];
        grouped_indices[offsets[c] + current_pos[c]++] = i;
    }

    // Process by group
    for (int c = 0; c < kc; c++) {
        const float* centroid = coarse_centroids + c*d;
        int group_start = offsets[c];
        int group_end = offsets[c+1];

        for (int idx = group_start; idx < group_end; idx++) {
            int i = grouped_indices[idx];
            const float* vec = x + i*d;
            float* residual = r_out + i*d;

            // Compute residual (SIMD as before)
            for (int j = 0; j < d; j += 4) {
                SIMD4<Float> v(vec + j);
                SIMD4<Float> c_vec(centroid + j);
                SIMD4<Float> r = v - c_vec;
                r.store(residual + j);
            }
        }
    }

    free(counts);
    free(offsets);
    free(grouped_indices);
    free(current_pos);
}
```

**Complexity**: O(n + kc) setup + O(n×d) computation.

**When Beneficial**: Large kc (kc > 1000) with random assignments.

### 4. Fused Residual PQ Encoding

**Integration with Kernel #20**:

```c
void residual_pq_encode_u8_f32(const float* x, const int32_t* coarse_ids,
                                const float* coarse_centroids, int64_t n,
                                int d, int m, int ks, const float* codebooks,
                                uint8_t* codes, const PQEncodeOpts* opts) {
    int dsub = d / m;

    for (int64_t i = 0; i < n; i++) {
        int32_t a = coarse_ids[i];
        const float* vec = x + i*d;
        const float* centroid = coarse_centroids + a*d;

        // Encode each subspace
        for (int j = 0; j < m; j++) {
            const float* vec_sub = vec + j*dsub;
            const float* centroid_sub = centroid + j*dsub;
            const float* codebook_j = codebooks + j*ks*dsub;

            // Find nearest codeword for residual subspace
            int best_k = 0;
            float best_dist = INFINITY;

            for (int k = 0; k < ks; k++) {
                const float* codeword = codebook_j + k*dsub;

                // Compute distance on residual (fused)
                float dist = 0;
                for (int idx = 0; idx < dsub; idx += 4) {
                    SIMD4<Float> v(vec_sub + idx);
                    SIMD4<Float> c(centroid_sub + idx);
                    SIMD4<Float> cw(codeword + idx);
                    SIMD4<Float> residual = v - c;
                    SIMD4<Float> diff = residual - cw;
                    dist += reduce_add(diff * diff);
                }

                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }

            codes[i*m + j] = best_k;
        }
    }
}
```

**Key**: Residual computed in registers (`v - c`), never stored to memory.

### 5. Fused Residual LUT Construction

**Integration with Kernel #21**:

```c
void residual_pq_lut_f32(const float* q, const float* coarse_centroid,
                          int d, int m, int ks, const float* codebooks,
                          float* lut, const float* centroid_norms,
                          const PQLutOpts* opts) {
    int dsub = d / m;

    for (int j = 0; j < m; j++) {
        const float* q_sub = q + j*dsub;
        const float* c_sub = coarse_centroid + j*dsub;
        const float* codebook_j = codebooks + j*ks*dsub;

        for (int k = 0; k < ks; k++) {
            const float* codeword = codebook_j + k*dsub;

            // Compute distance for query residual (fused)
            float dist_sq = 0;
            for (int idx = 0; idx < dsub; idx += 4) {
                SIMD4<Float> q_vec(q_sub + idx);
                SIMD4<Float> c_vec(c_sub + idx);
                SIMD4<Float> cw_vec(codeword + idx);
                SIMD4<Float> residual_q = q_vec - c_vec;
                SIMD4<Float> diff = residual_q - cw_vec;
                dist_sq += reduce_add(diff * diff);
            }

            lut[j*ks + k] = dist_sq;
        }
    }
}
```

---

## Implementation Strategies

### 1. Vectorization

**SIMD Subtraction** (4-way or 8-way):

```c
// 4-way SIMD
for (int j = 0; j < d; j += 4) {
    SIMD4<Float> v(vec + j);
    SIMD4<Float> c(centroid + j);
    SIMD4<Float> r = v - c;
    r.store(residual + j);
}

// 8-way SIMD (dual accumulators)
for (int j = 0; j < d; j += 8) {
    SIMD4<Float> v0(vec + j), v1(vec + j + 4);
    SIMD4<Float> c0(centroid + j), c1(centroid + j + 4);
    SIMD4<Float> r0 = v0 - c0;
    SIMD4<Float> r1 = v1 - c1;
    r0.store(residual + j);
    r1.store(residual + j + 4);
}
```

**Unrolling** (process 4-8 vectors per iteration):

```c
int i = 0;
for (; i + 3 < n; i += 4) {
    // Process 4 vectors in parallel
    compute_residual(x, coarse_ids, coarse_centroids, i+0, d, r_out);
    compute_residual(x, coarse_ids, coarse_ids, i+1, d, r_out);
    compute_residual(x, coarse_ids, coarse_centroids, i+2, d, r_out);
    compute_residual(x, coarse_ids, coarse_centroids, i+3, d, r_out);
}

// Remainder
for (; i < n; i++) {
    compute_residual(x, coarse_ids, coarse_centroids, i, d, r_out);
}
```

### 2. Prefetching

**Prefetch Next Vector and Centroid**:

```c
const int PREFETCH_DIST = 8;

for (int64_t i = 0; i < n; i++) {
    // Prefetch future data
    if (i + PREFETCH_DIST < n) {
        __builtin_prefetch(x + (i + PREFETCH_DIST)*d, 0, 3);
        int32_t future_a = coarse_ids[i + PREFETCH_DIST];
        __builtin_prefetch(coarse_centroids + future_a*d, 0, 3);
    }

    // Compute current residual
    int32_t a = coarse_ids[i];
    compute_residual(x + i*d, coarse_centroids + a*d, r_out + i*d, d);
}
```

### 3. Parallelism

**Parallel over Vectors**:

```c
#pragma omp parallel for num_threads(opts->num_threads) schedule(static)
for (int64_t i = 0; i < n; i++) {
    int32_t a = coarse_ids[i];
    compute_residual(x + i*d, coarse_centroids + a*d, r_out + i*d, d);
}
```

**Scaling**: Near-linear (no synchronization needed, embarrassingly parallel).

---

## Performance Characteristics

### 1. Computational Complexity

**Per Vector**:
- Load: d floats (x) + d floats (centroid)
- Compute: d subtractions
- Store: d floats (residual)
- **Total**: O(d) operations

**Full Dataset**:
- Complexity: O(n×d)
- FLOPs: n×d subtractions

**Example** (n=10M, d=1024):
- Operations: 10⁷ × 1024 ≈ 10¹⁰

### 2. Memory Bandwidth

**Standard Computation**:
- Reads: n×d×4 (x) + n×d×4 (centroids, worst case)
- Writes: n×d×4 (residuals)
- Total: 3×n×d×4 bytes

**With Grouped Processing**:
- Reads: n×d×4 (x) + kc×d×4 (centroids, each loaded once)
- Writes: n×d×4 (residuals)
- Total: 2×n×d×4 + kc×d×4 bytes

**Example** (n=10M, d=1024, kc=10K):
- Standard: 3 × 10⁷ × 1024 × 4 = 120 GB
- Grouped: 2 × 10⁷ × 1024 × 4 + 10⁴ × 1024 × 4 = 80 GB + 40 MB ≈ 80 GB

**Fused Computation** (encoding/LUT):
- No explicit residual writes, save n×d×4 bytes
- Example: Save 40 GB for n=10M, d=1024

### 3. Performance Targets (Apple M2 Max, 8 P-cores)

| Configuration | Throughput | Time (10M vectors) | Notes |
|---------------|------------|-------------------|-------|
| d=512, standard | 50M vec/s | 200 ms | Memory-bound |
| d=1024, standard | 40M vec/s | 250 ms | Memory-bound |
| d=1536, standard | 30M vec/s | 333 ms | Memory-bound |
| d=1024, grouped (kc=10K) | 45M vec/s | 222 ms | Better cache locality |
| d=1024, in-place | 50M vec/s | 200 ms | Fewer memory ops |

**Scaling**:
- **n**: Linear (O(n))
- **d**: Linear (O(d))
- **Threads**: Near-linear up to physical cores

### 4. Fused vs Non-fused

**Non-fused Pipeline** (PQ encoding):
1. Compute residuals: 250 ms (n=10M, d=1024)
2. Encode residuals: 10 sec
3. **Total**: ~10.25 sec

**Fused Pipeline**:
1. Encode with residuals computed on-the-fly: 10 sec
2. **Total**: 10 sec

**Savings**: ~250 ms latency + 40 GB memory

---

## Numerical Considerations

### 1. Floating-Point Accuracy

**Subtraction**: **r** = **x** - **c**

**Error**: Subtraction is numerically stable for similar-magnitude operands.

**Potential Issue**: If ‖**x**‖ >> ‖**c**‖ or vice versa, catastrophic cancellation can occur.

**Mitigation**: In practice, **x** and **c** have similar magnitudes (both are embeddings in same space), so cancellation is rare.

### 2. Determinism

**Guarantee**: Residual computation is deterministic:
```
r[i] = x[i] - c[coarse_ids[i]]
```

Identical inputs produce bitwise identical outputs (same floating-point rounding).

**With Parallelism**: Results are deterministic regardless of thread scheduling (no accumulation or reduction involved).

### 3. Bounds Checking

**Validation** (optional, for debugging):
```c
if (opts && opts->check_bounds) {
    for (int64_t i = 0; i < n; i++) {
        if (coarse_ids[i] < 0 || coarse_ids[i] >= kc) {
            fprintf(stderr, "Invalid coarse_id %d at index %ld\n", coarse_ids[i], i);
            return ERROR_INVALID_ASSIGNMENT;
        }
    }
}
```

**Production**: Disable bounds checking for maximum performance.

---

## Correctness Testing

### 1. Scalar Reference

**Test 1: Exact Match**
```swift
func testResidualsCorrectness() {
    let n = 1_000
    let d = 512
    let kc = 100

    let x = generateRandomVectors(n: n, d: d)
    let centroids = generateRandomVectors(n: kc, d: d)
    let assignments = generateRandomAssignments(n: n, kc: kc)

    // Optimized computation
    var residuals_fast = [Float](repeating: 0, count: n * d)
    residuals_f32(x, assignments, centroids, n, d, &residuals_fast, nil)

    // Scalar reference
    var residuals_ref = [Float](repeating: 0, count: n * d)
    for i in 0..<n {
        let a = Int(assignments[i])
        for j in 0..<d {
            residuals_ref[i*d + j] = x[i*d + j] - centroids[a*d + j]
        }
    }

    // Should match exactly
    for i in 0..<(n*d) {
        assert(residuals_fast[i] == residuals_ref[i], "Mismatch at \(i)")
    }
}
```

### 2. In-Place Correctness

**Test 2: In-Place vs Standard**
```swift
func testInPlaceCorrectness() {
    let n = 500
    let d = 768
    let kc = 50

    let x = generateRandomVectors(n: n, d: d)
    let centroids = generateRandomVectors(n: kc, d: d)
    let assignments = generateRandomAssignments(n: n, kc: kc)

    // Standard (out-of-place)
    var residuals_standard = [Float](repeating: 0, count: n * d)
    residuals_f32(x, assignments, centroids, n, d, &residuals_standard, nil)

    // In-place
    var x_inplace = x  // Copy
    residuals_f32_inplace(&x_inplace, assignments, centroids, n, d, nil)

    // Should match
    for i in 0..<(n*d) {
        assert(residuals_standard[i] == x_inplace[i], "In-place mismatch at \(i)")
    }
}
```

### 3. Fused Encoding Parity

**Test 3: Fused vs Non-Fused Encoding**
```swift
func testFusedEncodingParity() {
    let n = 2_000
    let d = 1024
    let m = 8
    let ks = 256
    let kc = 100

    let x = generateRandomVectors(n: n, d: d)
    let centroids = generateRandomVectors(n: kc, d: d)
    let assignments = generateRandomAssignments(n: n, kc: kc)
    let pq_codebooks = trainPQCodebooks(..., m: m, ks: ks)

    // Non-fused: compute residuals, then encode
    var residuals = [Float](repeating: 0, count: n * d)
    residuals_f32(x, assignments, centroids, n, d, &residuals, nil)

    var codes_nonfused = [UInt8](repeating: 0, count: n * m)
    pq_encode_u8_f32(residuals, n, d, m, ks, pq_codebooks, &codes_nonfused, nil)

    // Fused: encode with residuals computed on-the-fly
    var codes_fused = [UInt8](repeating: 0, count: n * m)
    residual_pq_encode_u8_f32(x, assignments, centroids, n, d, m, ks,
                               pq_codebooks, &codes_fused, nil)

    // Should produce identical codes
    for i in 0..<(n*m) {
        assert(codes_nonfused[i] == codes_fused[i], "Fused encoding mismatch at \(i)")
    }
}
```

### 4. Fused LUT Parity

**Test 4: Fused vs Non-Fused LUT**
```swift
func testFusedLUTParity() {
    let d = 512
    let m = 8
    let ks = 256

    let query = generateRandomVector(d: d)
    let coarse_centroid = generateRandomVector(d: d)
    let pq_codebooks = trainPQCodebooks(..., m: m, ks: ks)

    // Non-fused: compute query residual, then build LUT
    var query_residual = [Float](repeating: 0, count: d)
    for i in 0..<d {
        query_residual[i] = query[i] - coarse_centroid[i]
    }

    var lut_nonfused = [Float](repeating: 0, count: m * ks)
    pq_lut_l2_f32(query_residual, d, m, ks, pq_codebooks, &lut_nonfused, nil, nil, nil)

    // Fused: LUT with query residual computed on-the-fly
    var lut_fused = [Float](repeating: 0, count: m * ks)
    residual_pq_lut_f32(query, coarse_centroid, d, m, ks, pq_codebooks,
                        &lut_fused, nil, nil)

    // Should match within floating-point precision
    for i in 0..<(m*ks) {
        let diff = abs(lut_nonfused[i] - lut_fused[i])
        assert(diff < 1e-5, "Fused LUT mismatch at \(i): diff=\(diff)")
    }
}
```

### 5. Grouped vs Ungrouped

**Test 5: Grouped Processing Parity**
```swift
func testGroupedParity() {
    let n = 5_000
    let d = 1024
    let kc = 500

    let x = generateRandomVectors(n: n, d: d)
    let centroids = generateRandomVectors(n: kc, d: d)
    let assignments = generateRandomAssignments(n: n, kc: kc)

    // Standard (ungrouped)
    var residuals_standard = [Float](repeating: 0, count: n * d)
    residuals_f32(x, assignments, centroids, n, d, &residuals_standard, nil)

    // Grouped
    var residuals_grouped = [Float](repeating: 0, count: n * d)
    var opts = ResidualOpts(group_by_centroid: true)
    residuals_f32(x, assignments, centroids, n, d, &residuals_grouped, &opts)

    // Should match (order may differ, but values should be same)
    for i in 0..<n {
        var found = false
        for ii in 0..<n {
            if arraysEqual(residuals_standard[i*d..<(i+1)*d],
                          residuals_grouped[ii*d..<(ii+1)*d]) {
                found = true
                break
            }
        }
        assert(found, "Residual for vector \(i) not found in grouped output")
    }
}
```

### 6. Performance Benchmark

**Test 6: Throughput**
```swift
func testResidualThroughput() {
    let n = 10_000_000
    let d = 1024
    let kc = 10_000

    let x = generateRandomVectors(n: n, d: d)
    let centroids = generateRandomVectors(n: kc, d: d)
    let assignments = generateRandomAssignments(n: n, kc: kc)

    var residuals = [Float](repeating: 0, count: n * d)

    let start = Date()
    residuals_f32(x, assignments, centroids, n, d, &residuals, nil)
    let elapsed = Date().timeIntervalSince(start)

    let throughput = Double(n) / elapsed
    print("Residual computation: \(throughput / 1_000_000) M vectors/sec")

    // Expect > 30M vectors/sec on M2 Max
    assert(throughput > 30_000_000, "Throughput \(throughput) below target")
}
```

---

## Integration Patterns

### 1. IVF-PQ Training with Residuals

**Complete Training Pipeline**:
```swift
func trainIVFPQ(data: [Float], n: Int, d: Int, kc: Int, m: Int, ks: Int) -> IVFPQIndex {
    // 1. Train IVF coarse quantizer (kernel #12)
    let coarse_centroids = trainIVFCentroids(data, n: n, d: d, k: kc)

    // 2. Assign vectors to coarse centroids
    var assignments = [Int32](repeating: 0, count: n)
    assignVectorsToCentroids(data, coarse_centroids, &assignments, n: n, d: d, kc: kc)

    // 3. Compute residuals (kernel #23)
    var residuals = [Float](repeating: 0, count: n * d)
    residuals_f32(data, assignments, coarse_centroids, n, d, &residuals, nil)

    // 4. Train PQ on residuals (kernel #19)
    var pq_codebooks = [Float](repeating: 0, count: m * ks * (d/m))
    pq_train_f32(residuals, n, d, m, ks, nil, nil, &cfg, &pq_codebooks, nil, nil)

    // 5. Encode residuals (kernel #20)
    var codes = [UInt8](repeating: 0, count: n * m)
    pq_encode_u8_f32(residuals, n, d, m, ks, pq_codebooks, &codes, nil)

    // 6. Build index
    return IVFPQIndex(coarse_centroids: coarse_centroids,
                      pq_codebooks: pq_codebooks,
                      codes: codes,
                      dimension: d, m: m, ks: ks)
}
```

### 2. IVF-PQ Training with Fused Residuals

**Memory-Efficient Pipeline**:
```swift
func trainIVFPQFused(data: [Float], n: Int, d: Int, kc: Int, m: Int, ks: Int) -> IVFPQIndex {
    // 1-2. Train IVF and assign (same as above)
    let coarse_centroids = trainIVFCentroids(data, n: n, d: d, k: kc)
    var assignments = [Int32](repeating: 0, count: n)
    assignVectorsToCentroids(data, coarse_centroids, &assignments, n: n, d: d, kc: kc)

    // 3. Train PQ on residuals (kernel #19 with fused residuals)
    var pq_codebooks = [Float](repeating: 0, count: m * ks * (d/m))
    pq_train_f32(data, n, d, m, ks, coarse_centroids, assignments, &cfg, &pq_codebooks, nil, nil)

    // 4. Encode with fused residuals (kernel #23 + #20)
    var codes = [UInt8](repeating: 0, count: n * m)
    residual_pq_encode_u8_f32(data, assignments, coarse_centroids, n, d, m, ks,
                               pq_codebooks, &codes, nil)

    return IVFPQIndex(coarse_centroids: coarse_centroids,
                      pq_codebooks: pq_codebooks,
                      codes: codes,
                      dimension: d, m: m, ks: ks)
}
```

**Memory Savings**: 40 GB (10M × 1024 × 4 bytes) by avoiding materialized residuals.

### 3. IVF-PQ Query with Fused Residuals

**Query Pipeline**:
```swift
func queryIVFPQFused(query: [Float], index: IVFPQIndex, k: Int, nprobe: Int) -> [(id: Int, dist: Float)] {
    // 1. Select nprobe IVF lists
    let probe_lists = selectIVFLists(query, index.coarse_centroids, nprobe)

    var candidates: [(id: Int, dist: Float)] = []

    // 2. For each probed list
    for list_id in probe_lists {
        let coarse_centroid = index.coarse_centroids[list_id]

        // 3. Build LUT with fused query residual (kernel #23 + #21)
        var lut = [Float](repeating: 0, count: index.m * index.ks)
        residual_pq_lut_f32(query, coarse_centroid, index.dimension,
                            index.m, index.ks, index.pq_codebooks,
                            &lut, index.centroid_norms, nil)

        // 4. ADC scan (kernel #22)
        let n_list = index.ivf_lists[list_id].count / index.m
        var distances = [Float](repeating: 0, count: n_list)
        adc_scan_u8(index.ivf_lists[list_id], n_list, index.m, index.ks,
                    lut, &distances, nil)

        // 5. Collect candidates
        let list_topk = partialTopK(distances, k: k)
        for (local_id, dist) in list_topk {
            let global_id = index.id_map[list_id][local_id]
            candidates.append((id: global_id, dist: dist))
        }
    }

    // 6. Merge and return
    return mergeTopK(candidates, k: k)
}
```

---

## Coding Guidelines

### 1. API Design

**Consistent Naming**:
```c
// All residual functions prefixed with residuals_ or residual_
void residuals_f32(...);
void residuals_f32_inplace(...);
void residual_pq_encode_u8_f32(...);
void residual_pq_lut_f32(...);
```

### 2. SIMD Patterns

**Standard SIMD Subtraction**:
```c
// Always process 4 or 8 floats at a time
for (int j = 0; j < d; j += 4) {
    SIMD4<Float> v(vec + j);
    SIMD4<Float> c(centroid + j);
    SIMD4<Float> r = v - c;
    r.store(residual + j);
}
```

### 3. Telemetry Integration

**Metrics Collection** (#46):
```c
void emit_residual_telemetry(int64_t n, int d, bool fused, double time_sec) {
    telemetry_emit("residuals.vectors", n);
    telemetry_emit("residuals.dimension", d);
    telemetry_emit("residuals.fused", fused ? 1 : 0);
    telemetry_emit("residuals.time_sec", time_sec);
    telemetry_emit("residuals.throughput_vec_per_sec", (double)n / time_sec);
    telemetry_emit("residuals.bytes_written", fused ? 0 : n * d * 4);
}
```

---

## Example Usage

### Example 1: Basic Residual Computation

```c
#include "residuals.h"

int main() {
    int64_t n = 1000000;
    int d = 1024;
    int kc = 10000;

    // Load data
    float* x = load_vectors("vectors.bin", n, d);
    float* centroids = load_centroids("centroids.bin", kc, d);
    int32_t* assignments = load_assignments("assignments.bin", n);

    // Compute residuals
    float* residuals = malloc(n * d * sizeof(float));
    residuals_f32(x, assignments, centroids, n, d, residuals, NULL);

    // Use residuals for PQ training
    train_pq(residuals, n, d, ...);

    free(x);
    free(centroids);
    free(assignments);
    free(residuals);
    return 0;
}
```

### Example 2: Fused Residual Encoding

```c
#include "residuals.h"

void encode_ivf_pq_fused(const float* x, int64_t n, int d) {
    int kc = 10000, m = 8, ks = 256;

    // Load IVF data
    float* centroids = load_centroids("centroids.bin", kc, d);
    int32_t* assignments = load_assignments("assignments.bin", n);
    float* pq_codebooks = load_pq_codebooks("pq_codebooks.bin", m, ks, d/m);

    // Encode with fused residuals (no materialization)
    uint8_t* codes = malloc(n * m);
    residual_pq_encode_u8_f32(x, assignments, centroids, n, d, m, ks,
                               pq_codebooks, codes, NULL);

    save_codes("codes.bin", codes, n, m);

    free(centroids);
    free(assignments);
    free(pq_codebooks);
    free(codes);
}
```

### Example 3: Swift Integration

```swift
import Foundation

func computeResiduals(
    vectors: [Float],
    n: Int,
    d: Int,
    assignments: [Int32],
    centroids: [Float]
) -> [Float] {
    var residuals = [Float](repeating: 0, count: n * d)

    vectors.withUnsafeBufferPointer { vecPtr in
        assignments.withUnsafeBufferPointer { assignPtr in
            centroids.withUnsafeBufferPointer { centroidPtr in
                residuals.withUnsafeMutableBufferPointer { resPtr in
                    residuals_f32(
                        vecPtr.baseAddress!,
                        assignPtr.baseAddress!,
                        centroidPtr.baseAddress!,
                        Int64(n),
                        Int32(d),
                        resPtr.baseAddress!,
                        nil
                    )
                }
            }
        }
    }

    return residuals
}
```

---

## Summary

**Kernel #23** provides efficient residual computation for IVF-PQ:

1. **Functionality**: Compute **r** = **x** - **c** for IVF residuals
2. **Modes**:
   - Standard: Materialize full residual array
   - In-place: Overwrite input vectors with residuals
   - Fused: Compute residuals on-the-fly during PQ operations
3. **Performance**: 30-50M vectors/sec on M2 Max
4. **Memory Savings**: Fused mode saves O(n×d) memory (40 GB for 10M × 1024-dim vectors)
5. **Key Optimizations**:
   - SIMD subtraction (4-8 way)
   - Grouped processing for better cache locality
   - Prefetching for centroid data
6. **Integration**:
   - Feeds PQ training (kernel #19)
   - Integrated with PQ encoding (kernel #20)
   - Integrated with LUT construction (kernel #21)

**Dependencies**:
- Kernel #12 (IVF training, coarse centroids)
- Kernel #19 (PQ training on residuals)
- Kernel #20 (PQ encoding with fused residuals)
- Kernel #21 (LUT construction with fused residuals)

**Typical Use**: For IVF-PQ index with 10M vectors, use fused residual computation to save 40 GB memory and 250 ms latency while maintaining identical accuracy.
