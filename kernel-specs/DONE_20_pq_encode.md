# ✅ DONE — Kernel Specification #20: Product Quantization (PQ) Encoding

**ID**: 20
**Priority**: MUST
**Role**: B/Q (Batch / Query)
**Status**: Specification

---

## Purpose

Encode high-dimensional vectors into compact PQ codes by quantizing each subspace to its nearest learned centroid. This is the core compression step in Product Quantization that enables:

1. **Massive compression**: 1024-dim float32 vectors (4096 bytes) → 8 bytes (512× compression)
2. **Fast distance computation**: Enable Asymmetric Distance Computation (ADC) via lookup tables
3. **Residual encoding**: Encode IVF residuals for accurate approximate search

**Typical Use Case**: Encode 10M vectors (1024-dim) into 8-byte PQ codes in ~10 seconds, enabling efficient storage and fast ADC-based search.

---

## Mathematical Foundations

### 1. PQ Encoding

**Input**: Vector **x** ∈ ℝᵈ and trained codebooks **C**₁, ..., **C**ₘ where **C**ⱼ = {**c**ⱼ,₁, ..., **c**ⱼ,ₖₛ} ⊂ ℝ^(dsub).

**Subspace Decomposition**:
```
x = [x₁ ; x₂ ; ... ; xₘ] where xⱼ ∈ ℝ^(dsub), dsub = d/m
```

**Per-Subspace Quantization**:

For each subspace j ∈ {1, ..., m}, find nearest centroid:
```
qⱼ = argmin_{k ∈ {1,...,ks}} ‖xⱼ - cⱼ,ₖ‖²₂
```

**PQ Code**: The encoding of **x** is the m-tuple of centroid indices:
```
PQ(x) = (q₁, q₂, ..., qₘ) ∈ {0,...,ks-1}ᵐ
```

**Storage**:
- For ks=256: each qⱼ fits in 1 byte (uint8_t) → m bytes total
- For ks=16: each qⱼ fits in 4 bits (nibble) → m/2 bytes total (packed)

**Reconstruction** (for validation):
```
x̂ = [c₁,q₁ ; c₂,q₂ ; ... ; cₘ,qₘ]
```

**Quantization Error**:
```
‖x - x̂‖² = Σⱼ₌₁ᵐ ‖xⱼ - cⱼ,qⱼ‖²
```

### 2. Distance Computation

**Naive Approach** (compute all distances):
```
For subspace j:
  For k = 0 to ks-1:
    dist²[k] = Σᵢ₌₀^(dsub-1) (xⱼ[i] - cⱼ,ₖ[i])²
  qⱼ = argmin_k dist²[k]
```

**Complexity**: O(ks × dsub) FLOPs per subspace = O(ks × d) total.

**Dot-Product Trick** (faster when ks is large):

Expand L2 distance:
```
‖xⱼ - cⱼ,ₖ‖² = ‖xⱼ‖² + ‖cⱼ,ₖ‖² - 2⟨xⱼ, cⱼ,ₖ⟩
```

If we precompute:
- ‖**x**ⱼ‖² (once per vector per subspace)
- ‖**c**ⱼ,ₖ‖² (once during training, stored with codebooks)

Then distance computation reduces to:
```
dist²[k] = query_norm + centroid_norm[k] - 2 × dot_product(xⱼ, cⱼ,ₖ)
```

**Complexity**: O(ks × dsub) FLOPs (same asymptotic, but ~2× fewer operations).

**When to Use Dot-Trick**:
- ks ≥ 64: Always beneficial (amortized norm precomputation)
- ks = 16: Marginal benefit, direct distance may be faster
- ks = 256: Significant benefit (~1.5-2× speedup)

### 3. Residual Encoding (IVF-PQ)

In IVF-PQ indexes, encode **residuals** rather than original vectors.

**IVF Assignment**: Vector **x** is assigned to coarse centroid **c**ₐ.

**Residual**:
```
r = x - cₐ
```

**Residual PQ Encoding**:
```
PQ(r) = (q₁, q₂, ..., qₘ) where qⱼ = argmin_k ‖rⱼ - cⱼ,ₖ‖²
```

**Fused Computation** (compute residual on-the-fly):
```
For subspace j:
  For k = 0 to ks-1:
    dist²[k] = Σᵢ₌₀^(dsub-1) ((xⱼ[i] - cₐ,ⱼ[i]) - cⱼ,ₖ[i])²
  qⱼ = argmin_k dist²[k]
```

**Benefit**: No need to materialize residuals, compute on-the-fly in registers.

### 4. 4-bit Packing (ks=16)

For ks=16, each code qⱼ ∈ {0,...,15} fits in 4 bits.

**Packing**: Store two codes per byte.
```
byte = (q₁ & 0xF) | ((q₂ & 0xF) << 4)
```

Low nibble: q₁, High nibble: q₂.

**Unpacking**:
```
q₁ = byte & 0xF
q₂ = (byte >> 4) & 0xF
```

**Storage Layout** (for m=8 subspaces, ks=16):
```
codes[i] = [q₀|q₁][q₂|q₃][q₄|q₅][q₆|q₇]  // 4 bytes per vector
```

**Compression Factor**: 1024-dim vector → 4 bytes (1024× compression).

### 5. Deterministic Tie-Breaking

When multiple centroids have identical distances, break ties deterministically:

```
qⱼ = argmin_{k ∈ {0,...,ks-1}} (‖xⱼ - cⱼ,ₖ‖², k)
```

Lexicographic ordering: prefer smaller distance, then smaller index.

**Implementation**:
```c
int best_k = 0;
float best_dist = compute_distance(x_j, c_j_0);

for (int k = 1; k < ks; k++) {
    float dist = compute_distance(x_j, c_j_k);
    if (dist < best_dist || (dist == best_dist && k < best_k)) {
        best_dist = dist;
        best_k = k;
    }
}
```

**Importance**: Ensures encoding is deterministic across runs and platforms.

---

## API Signatures

### 1. 8-bit Encoding (ks=256)

```c
void pq_encode_u8_f32(
    const float* x,                    // [n × d] input vectors (AoS)
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int m,                             // number of subspaces
    int ks,                            // codebook size (must be 256)
    const float* codebooks,            // [m × ks × dsub] codebooks
    uint8_t* codes,                    // [n × m] output codes
    const PQEncodeOpts* opts           // encoding options (nullable)
);
```

**Parameters**:
- `x`: Input vectors in AoS layout `[n][d]`
- `n`: Number of vectors to encode
- `d`: Dimension (must be divisible by m)
- `m`: Number of subspaces
- `ks`: Codebook size per subspace (must be 256 for u8)
- `codebooks`: Trained codebooks from kernel #19, layout `[m][ks][dsub]`
- `codes`: Output buffer for PQ codes, **must be preallocated** to n×m bytes
  - Layout: `codes[i*m + j]` = code for vector i, subspace j
- `opts`: Optional encoding options (nullable, use defaults if null)

**Output Layout**:
```
codes[0]: [q₀₀, q₀₁, ..., q₀,m-1]  // vector 0, all m subspaces
codes[1]: [q₁₀, q₁₁, ..., q₁,m-1]  // vector 1, all m subspaces
...
codes[n-1]: [qn-1,0, qn-1,1, ..., qn-1,m-1]
```

### 2. 4-bit Encoding (ks=16)

```c
void pq_encode_u4_f32(
    const float* x,                    // [n × d] input vectors (AoS)
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int m,                             // number of subspaces (must be even)
    int ks,                            // codebook size (must be 16)
    const float* codebooks,            // [m × ks × dsub] codebooks
    uint8_t* codes,                    // [n × m/2] output codes (packed)
    const PQEncodeOpts* opts           // encoding options (nullable)
);
```

**Parameters**: Same as u8 encoding, except:
- `m`: Must be even (for byte-aligned packing)
- `ks`: Must be 16 for u4
- `codes`: Output buffer, **must be preallocated** to n×(m/2) bytes
  - Packed layout: two codes per byte

**Output Layout** (packed nibbles):
```
codes[i*m/2 + j/2] = (q[i,j] & 0xF) | ((q[i,j+1] & 0xF) << 4)
```

Example (m=8):
```
codes[0]: [q₀|q₁][q₂|q₃][q₄|q₅][q₆|q₇]  // vector 0, 4 bytes
codes[1]: [q₀|q₁][q₂|q₃][q₄|q₅][q₆|q₇]  // vector 1, 4 bytes
```

### 3. Residual Encoding

For IVF-PQ, encode residuals on-the-fly.

```c
void pq_encode_residual_u8_f32(
    const float* x,                    // [n × d] input vectors
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int m,                             // number of subspaces
    int ks,                            // codebook size (256)
    const float* codebooks,            // [m × ks × dsub] residual codebooks
    const float* coarse_centroids,     // [kc × d] IVF coarse centroids
    const int32_t* assignments,        // [n] IVF assignments
    uint8_t* codes,                    // [n × m] output codes
    const PQEncodeOpts* opts           // encoding options (nullable)
);

void pq_encode_residual_u4_f32(
    const float* x,                    // [n × d] input vectors
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int m,                             // number of subspaces (must be even)
    int ks,                            // codebook size (16)
    const float* codebooks,            // [m × ks × dsub] residual codebooks
    const float* coarse_centroids,     // [kc × d] IVF coarse centroids
    const int32_t* assignments,        // [n] IVF assignments
    uint8_t* codes,                    // [n × m/2] output codes
    const PQEncodeOpts* opts           // encoding options (nullable)
);
```

**Additional Parameters**:
- `coarse_centroids`: IVF coarse centroids for residual computation
- `assignments`: IVF assignment for each vector (a[i] = coarse centroid index for x[i])

### 4. Packing/Unpacking Helpers

```c
// Pack two 4-bit codes into one byte
uint8_t pq_pack_u4_pair(uint8_t code0, uint8_t code1);

// Unpack byte into two 4-bit codes
void pq_unpack_u4_pair(uint8_t byte, uint8_t* code0, uint8_t* code1);

// Bulk pack m codes (m must be even) into m/2 bytes
void pq_pack_u4_bulk(const uint8_t* codes, int m, uint8_t* packed);

// Bulk unpack m/2 bytes into m codes
void pq_unpack_u4_bulk(const uint8_t* packed, int m, uint8_t* codes);
```

### 5. Encoding Options

```c
typedef struct {
    PQLayout layout;               // code layout (default: PQ_LAYOUT_AOS)
    bool use_dot_trick;            // use dot-product distance (default: auto)
    bool precompute_x_norm2;       // precompute query norms (default: true if dot_trick)
    int prefetch_distance;         // prefetch lookahead (default: 8)
    int num_threads;               // parallelism (0 = auto, default: 0)
} PQEncodeOpts;

typedef enum {
    PQ_LAYOUT_AOS,                 // [n][m] row-major codes
    PQ_LAYOUT_SOA_BLOCKED,         // [m][n/B][B] blocked by subspace
    PQ_LAYOUT_INTERLEAVED_BLOCK    // [n/g][m][g] interleaved groups
} PQLayout;
```

**Layout Options**:

**AoS** (Array of Structures):
```
codes[i*m + j] = code for vector i, subspace j
```
Best for: Sequential vector encoding, standard usage.

**SoA Blocked**:
```
codes[j*(n/B)*B + (i/B)*B + (i%B)] = code for vector i, subspace j
```
Best for: ADC with subspace-wise computation.

**Interleaved Block** (group size g):
```
codes[(i/g)*m*g + j*g + (i%g)] = code for vector i, subspace j
```
Best for: SIMD ADC with small g=4 or g=8.

---

## Algorithm Details

### 1. Direct Encoding (8-bit)

**Pseudocode**:
```
pq_encode_u8_f32(x, n, d, m, ks, codebooks, codes, opts):
    dsub = d / m

    parallel for i in 0..n-1:
        for j in 0..m-1:
            // Extract subspace
            x_sub = x[i*d + j*dsub : i*d + (j+1)*dsub]
            codebook_j = codebooks[j*ks*dsub : (j+1)*ks*dsub]

            // Find nearest centroid
            best_k = 0
            best_dist = l2_squared(x_sub, codebook_j[0], dsub)

            for k in 1..ks-1:
                dist = l2_squared(x_sub, codebook_j[k*dsub], dsub)
                if dist < best_dist or (dist == best_dist and k < best_k):
                    best_dist = dist
                    best_k = k

            // Store code
            codes[i*m + j] = best_k
```

**Optimization**: Vectorize distance computation using SIMD.

### 2. Dot-Product Encoding

**Precomputation** (per vector):
```
For each subspace j:
  query_norm[j] = Σᵢ₌₀^(dsub-1) xⱼ[i]²
```

**Encoding**:
```
parallel for i in 0..n-1:
    for j in 0..m-1:
        x_sub = x[i*d + j*dsub : i*d + (j+1)*dsub]
        codebook_j = codebooks[j*ks*dsub : (j+1)*ks*dsub]
        centroid_norms = precomputed_norms[j*ks : (j+1)*ks]
        q_norm = query_norm[i*m + j]

        best_k = 0
        best_dist = q_norm + centroid_norms[0] - 2*dot_product(x_sub, codebook_j[0], dsub)

        for k in 1..ks-1:
            dist = q_norm + centroid_norms[k] - 2*dot_product(x_sub, codebook_j[k*dsub], dsub)
            if dist < best_dist:
                best_dist = dist
                best_k = k

        codes[i*m + j] = best_k
```

### 3. Residual Encoding

**Fused Residual Computation**:
```
pq_encode_residual_u8_f32(x, n, d, m, ks, codebooks, coarse_centroids, assignments, codes, opts):
    dsub = d / m

    parallel for i in 0..n-1:
        a = assignments[i]  // coarse centroid index
        coarse = coarse_centroids[a*d : (a+1)*d]

        for j in 0..m-1:
            x_sub = x[i*d + j*dsub : i*d + (j+1)*dsub]
            coarse_sub = coarse[j*dsub : (j+1)*dsub]
            codebook_j = codebooks[j*ks*dsub : (j+1)*ks*dsub]

            best_k = 0
            best_dist = INFINITY

            for k in 0..ks-1:
                // Compute residual distance on-the-fly
                dist = 0
                for idx in 0..dsub-1:
                    residual = x_sub[idx] - coarse_sub[idx]
                    diff = residual - codebook_j[k*dsub + idx]
                    dist += diff * diff

                if dist < best_dist or (dist == best_dist and k < best_k):
                    best_dist = dist
                    best_k = k

            codes[i*m + j] = best_k
```

**SIMD Optimization**:
```c
// Vectorized residual distance computation
SIMD4<Float> acc = 0;
for (int idx = 0; idx < dsub; idx += 4) {
    SIMD4<Float> x_vec(x_sub + idx);
    SIMD4<Float> coarse_vec(coarse_sub + idx);
    SIMD4<Float> c_vec(codebook_j + k*dsub + idx);
    SIMD4<Float> residual = x_vec - coarse_vec;
    SIMD4<Float> diff = residual - c_vec;
    acc += diff * diff;
}
float dist = reduce_add(acc);
```

### 4. 4-bit Encoding & Packing

**Two-Phase Approach**:

Phase 1: Encode to temporary uint8_t buffer.
```
uint8_t temp_codes[m];
for (int j = 0; j < m; j++) {
    temp_codes[j] = find_nearest_centroid(x_sub[j], codebook[j], ks);
}
```

Phase 2: Pack pairs into bytes.
```
for (int j = 0; j < m; j += 2) {
    uint8_t low = temp_codes[j] & 0xF;
    uint8_t high = temp_codes[j+1] & 0xF;
    codes[i*m/2 + j/2] = low | (high << 4);
}
```

**Direct Packing** (single-phase):
```
for (int j = 0; j < m; j += 2) {
    uint8_t code0 = find_nearest_centroid(x_sub[j], codebook[j], ks);
    uint8_t code1 = find_nearest_centroid(x_sub[j+1], codebook[j+1], ks);
    codes[i*m/2 + j/2] = (code0 & 0xF) | ((code1 & 0xF) << 4);
}
```

**Batch Packing** (8 codes at a time):
```c
// Encode 8 codes
uint8_t temp[8];
for (int j = 0; j < 8; j++) {
    temp[j] = find_nearest_centroid(x_sub[j], codebook[j], 16);
}

// Pack into 4 bytes
uint32_t packed = 0;
packed |= (temp[0] & 0xF) << 0;
packed |= (temp[1] & 0xF) << 4;
packed |= (temp[2] & 0xF) << 8;
packed |= (temp[3] & 0xF) << 12;
packed |= (temp[4] & 0xF) << 16;
packed |= (temp[5] & 0xF) << 20;
packed |= (temp[6] & 0xF) << 24;
packed |= (temp[7] & 0xF) << 28;

*(uint32_t*)(codes + i*4) = packed;
```

---

## Implementation Strategies

### 1. Vectorization

**SIMD L2 Distance** (per subspace):
```c
float l2_squared_simd(const float* a, const float* b, int dsub) {
    SIMD4<Float> acc0 = 0, acc1 = 0;

    int d_vec = dsub & ~7;
    for (int i = 0; i < d_vec; i += 8) {
        SIMD4<Float> a0(a + i), a1(a + i + 4);
        SIMD4<Float> b0(b + i), b1(b + i + 4);
        SIMD4<Float> diff0 = a0 - b0;
        SIMD4<Float> diff1 = a1 - b1;
        acc0 += diff0 * diff0;
        acc1 += diff1 * diff1;
    }

    float sum = reduce_add(acc0) + reduce_add(acc1);

    // Remainder
    for (int i = d_vec; i < dsub; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}
```

**SIMD Dot Product**:
```c
float dot_product_simd(const float* a, const float* b, int dsub) {
    SIMD4<Float> acc0 = 0, acc1 = 0;

    int d_vec = dsub & ~7;
    for (int i = 0; i < d_vec; i += 8) {
        SIMD4<Float> a0(a + i), a1(a + i + 4);
        SIMD4<Float> b0(b + i), b1(b + i + 4);
        acc0 += a0 * b0;
        acc1 += a1 * a1;
    }

    float sum = reduce_add(acc0) + reduce_add(acc1);

    // Remainder
    for (int i = d_vec; i < dsub; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}
```

### 2. Centroid Tiling

**Motivation**: For ks=256, all centroids may not fit in L1 cache (256×128 floats = 128 KB).

**Approach**: Process centroids in tiles that fit in L1.

```c
const int TILE_SIZE = 32;  // 32 centroids × 128 floats = 16 KB

for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        const float* x_sub = x + i*d + j*dsub;
        const float* codebook_j = codebooks + j*ks*dsub;

        int best_k = 0;
        float best_dist = INFINITY;

        // Process centroids in tiles
        for (int tile = 0; tile < ks; tile += TILE_SIZE) {
            int tile_end = min(tile + TILE_SIZE, ks);

            // Prefetch next tile
            if (tile_end < ks) {
                __builtin_prefetch(codebook_j + tile_end*dsub, 0, 3);
            }

            // Evaluate current tile
            for (int k = tile; k < tile_end; k++) {
                float dist = l2_squared_simd(x_sub, codebook_j + k*dsub, dsub);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
        }

        codes[i*m + j] = best_k;
    }
}
```

### 3. Parallelism

**Parallel over Vectors** (coarse-grained):
```c
#pragma omp parallel for num_threads(opts->num_threads) schedule(dynamic, 1024)
for (int64_t i = 0; i < n; i++) {
    encode_vector(x + i*d, codebooks, codes + i*m, d, m, ks, dsub);
}
```

**Benefit**: Perfect scaling for large n, no synchronization needed.

**Parallel over Subspaces** (fine-grained, within each vector):
```c
// Usually not beneficial unless d is very large (d > 4096)
for (int64_t i = 0; i < n; i++) {
    #pragma omp parallel for num_threads(m)
    for (int j = 0; j < m; j++) {
        codes[i*m + j] = encode_subspace(x + i*d + j*dsub, codebooks + j*ks*dsub, ks, dsub);
    }
}
```

**Recommendation**: Parallelize over vectors (outer loop) for best performance.

### 4. Memory Prefetching

**Prefetch Input Vectors**:
```c
const int PREFETCH_DIST = 8;

for (int64_t i = 0; i < n; i++) {
    // Prefetch future vector
    if (i + PREFETCH_DIST < n) {
        __builtin_prefetch(x + (i + PREFETCH_DIST)*d, 0, 3);
    }

    // Encode current vector
    encode_vector(x + i*d, codebooks, codes + i*m, d, m, ks, dsub);
}
```

**Prefetch Codebooks** (when tiling):
```c
// Inside centroid tile loop
if (tile_end < ks) {
    for (int k = tile_end; k < min(tile_end + TILE_SIZE, ks); k++) {
        __builtin_prefetch(codebook_j + k*dsub, 0, 3);
    }
}
```

---

## Performance Characteristics

### 1. Computational Complexity

**Per Vector**:
- **Direct Distance**: O(m × ks × dsub) = O(ks × d) FLOPs
- **Dot-Product**: O(m × ks × dsub + d) ≈ O(ks × d) FLOPs (similar asymptotic, but ~2× fewer ops)

**Example** (d=1024, m=8, ks=256):
- Direct: 8 × 256 × 128 × 2 = 524,288 FLOPs
- Dot-product: 8 × 256 × 128 + 1024 ≈ 263,000 FLOPs
- Speedup: ~2×

**Full Dataset** (n vectors):
- Total FLOPs: O(n × ks × d)
- Example (n=10M, d=1024, ks=256): 10⁷ × 256 × 1024 × 2 ≈ 5×10¹² FLOPs

### 2. Memory Bandwidth

**Reads per Vector**:
- Input vector: d×4 bytes
- Codebooks: m×ks×dsub×4 bytes = ks×d×4 bytes
- Total: (1+ks)×d×4 bytes

**Writes per Vector**:
- u8: m bytes
- u4: m/2 bytes

**Example** (d=1024, ks=256):
- Reads: 257×1024×4 ≈ 1 MB per vector
- Writes: 8 bytes (u8) or 4 bytes (u4)
- **Memory-bound** for large ks

**With Centroid Tiling** (L1 cache hit):
- Reads: d×4 bytes (input only, codebooks in cache)
- Example: 1024×4 = 4 KB per vector
- Throughput: 200 GB/s / 4 KB ≈ 50M vectors/sec

### 3. Performance Targets (Apple M2 Max, 8 P-cores)

| Configuration | Throughput | Time (10M vectors) | Notes |
|---------------|------------|-------------------|-------|
| ks=256, d=1024, m=8 (u8, direct) | 2M vec/s | 5 sec | Memory-bound |
| ks=256, d=1024, m=8 (u8, dot) | 3M vec/s | 3.3 sec | Dot-trick speedup |
| ks=256, d=1024, m=8 (u8, tiled) | 10M vec/s | 1 sec | L1 cache hit |
| ks=16, d=1024, m=8 (u4) | 15M vec/s | 0.67 sec | Fewer centroids |
| ks=256, d=512, m=8 (u8) | 20M vec/s | 0.5 sec | Lower dimension |

**Scaling**:
- **n**: Linear (O(n))
- **ks**: Linear (O(ks)), but mitigated by tiling
- **d**: Linear (O(d))
- **m**: Linear (O(m)), but typically small (m ≤ 32)

### 4. Compression Ratios

| Configuration | Input Size | Code Size | Compression Ratio |
|---------------|------------|-----------|-------------------|
| d=1024, m=8, ks=256 (u8) | 4096 bytes | 8 bytes | 512× |
| d=1024, m=8, ks=16 (u4) | 4096 bytes | 4 bytes | 1024× |
| d=768, m=8, ks=256 (u8) | 3072 bytes | 8 bytes | 384× |
| d=512, m=8, ks=256 (u8) | 2048 bytes | 8 bytes | 256× |

**Trade-off**: Higher compression (smaller ks) → lower search quality.

---

## Numerical Considerations

### 1. Deterministic Tie-Breaking

**Distance Equality**:
```c
// Use exact equality for reproducibility
if (dist < best_dist || (dist == best_dist && k < best_k)) {
    best_dist = dist;
    best_k = k;
}
```

**Floating-Point Consistency**: Distances computed identically across runs will have identical bit patterns, ensuring determinism.

### 2. Norm Precomputation Accuracy

**Query Norms** (computed once per vector):
```c
float query_norms[m];
for (int j = 0; j < m; j++) {
    float norm_sq = 0;
    for (int idx = 0; idx < dsub; idx++) {
        float val = x[j*dsub + idx];
        norm_sq += val * val;
    }
    query_norms[j] = norm_sq;
}
```

**Centroid Norms** (precomputed during training, stored with codebooks):
```c
// During training (kernel #19)
for (int j = 0; j < m; j++) {
    for (int k = 0; k < ks; k++) {
        float norm_sq = 0;
        for (int idx = 0; idx < dsub; idx++) {
            float val = codebook[j*ks*dsub + k*dsub + idx];
            norm_sq += val * val;
        }
        centroid_norms[j*ks + k] = norm_sq;
    }
}
```

### 3. 4-bit Range Validation

**Constraint**: All codes must fit in 4 bits (0-15).

```c
assert(ks == 16);
assert(best_k >= 0 && best_k < 16);
codes_u4[i] = (uint8_t)best_k;  // Safe cast
```

**Packing Validation**:
```c
// Verify packing preserves codes
uint8_t code0 = 5, code1 = 12;
uint8_t packed = (code0 & 0xF) | ((code1 & 0xF) << 4);
uint8_t unpacked0 = packed & 0xF;
uint8_t unpacked1 = (packed >> 4) & 0xF;
assert(unpacked0 == code0);
assert(unpacked1 == code1);
```

---

## Correctness Testing

### 1. Encoding Correctness

**Test 1: Nearest Centroid Verification**
```swift
func testEncodingCorrectness() {
    let n = 1_000
    let d = 256
    let m = 4
    let ks = 64
    let dsub = d / m

    let x = generateRandomVectors(n: n, d: d)
    let codebooks = trainPQCodebooks(x, n: n, d: d, m: m, ks: ks)

    var codes = [UInt8](repeating: 0, count: n * m)
    pq_encode_u8_f32(x, n, d, m, ks, codebooks, &codes, nil)

    // Verify each code
    for i in 0..<n {
        for j in 0..<m {
            let x_sub = Array(x[i*d + j*dsub..<i*d + (j+1)*dsub])
            let code = Int(codes[i*m + j])

            // Verify code is nearest centroid
            var min_dist = Float.infinity
            var min_k = 0

            for k in 0..<ks {
                let c_sub = Array(codebooks[j*ks*dsub + k*dsub..<j*ks*dsub + (k+1)*dsub])
                let dist = l2SquaredDistance(x_sub, c_sub)
                if dist < min_dist {
                    min_dist = dist
                    min_k = k
                }
            }

            assert(code == min_k, "Code \(code) != nearest centroid \(min_k) for vector \(i), subspace \(j)")
        }
    }
}
```

### 2. Determinism Tests

**Test 2: Reproducibility**
```swift
func testEncodingDeterminism() {
    let x = generateRandomVectors(n: 10_000, d: 512)
    let codebooks = trainPQCodebooks(x, n: 10_000, d: 512, m: 8, ks: 256)

    var codes1 = [UInt8](repeating: 0, count: 10_000 * 8)
    var codes2 = [UInt8](repeating: 0, count: 10_000 * 8)

    pq_encode_u8_f32(x, 10_000, 512, 8, 256, codebooks, &codes1, nil)
    pq_encode_u8_f32(x, 10_000, 512, 8, 256, codebooks, &codes2, nil)

    // Should be bitwise identical
    for i in 0..<codes1.count {
        assert(codes1[i] == codes2[i], "Code mismatch at index \(i)")
    }
}
```

### 3. Packing Tests

**Test 3: u4 Pack/Unpack Idempotence**
```swift
func testU4PackUnpack() {
    for code0 in 0..<16 {
        for code1 in 0..<16 {
            let packed = pq_pack_u4_pair(UInt8(code0), UInt8(code1))

            var unpacked0: UInt8 = 0
            var unpacked1: UInt8 = 0
            pq_unpack_u4_pair(packed, &unpacked0, &unpacked1)

            assert(unpacked0 == code0, "Unpack failed: expected \(code0), got \(unpacked0)")
            assert(unpacked1 == code1, "Unpack failed: expected \(code1), got \(unpacked1)")
        }
    }
}
```

**Test 4: Bulk Packing**
```swift
func testU4BulkPacking() {
    let m = 8
    var codes = [UInt8](repeating: 0, count: m)
    for i in 0..<m {
        codes[i] = UInt8(i % 16)
    }

    var packed = [UInt8](repeating: 0, count: m / 2)
    pq_pack_u4_bulk(codes, m, &packed)

    var unpacked = [UInt8](repeating: 0, count: m)
    pq_unpack_u4_bulk(packed, m, &unpacked)

    for i in 0..<m {
        assert(unpacked[i] == codes[i], "Bulk pack/unpack failed at index \(i)")
    }
}
```

### 4. Residual Encoding Tests

**Test 5: Residual vs Direct Consistency**
```swift
func testResidualEncoding() {
    let n = 5_000
    let d = 768
    let kc = 100
    let m = 8
    let ks = 256

    let x = generateClusteredVectors(n: n, d: d, num_clusters: kc)

    // Train IVF and PQ
    let coarse_centroids = trainIVFCentroids(x, n: n, d: d, k: kc)
    var assignments = [Int32](repeating: 0, count: n)
    assignVectorsToCentroids(x, coarse_centroids, &assignments, n: n, d: d, kc: kc)

    let pq_codebooks_residual = trainPQCodebooks(x, coarse_centroids, assignments, m: m, ks: ks)

    // Materialize residuals
    var residuals = [Float](repeating: 0, count: n * d)
    for i in 0..<n {
        let a = Int(assignments[i])
        for j in 0..<d {
            residuals[i*d + j] = x[i*d + j] - coarse_centroids[a*d + j]
        }
    }

    // Encode: direct vs fused residual
    var codes_direct = [UInt8](repeating: 0, count: n * m)
    pq_encode_u8_f32(residuals, n, d, m, ks, pq_codebooks_residual, &codes_direct, nil)

    var codes_residual = [UInt8](repeating: 0, count: n * m)
    pq_encode_residual_u8_f32(x, n, d, m, ks, pq_codebooks_residual,
                              coarse_centroids, assignments, &codes_residual, nil)

    // Should be identical
    for i in 0..<codes_direct.count {
        assert(codes_direct[i] == codes_residual[i],
               "Residual encoding mismatch at index \(i)")
    }
}
```

### 5. Performance Regression

**Test 6: Throughput Benchmark**
```swift
func testEncodingThroughput() {
    let n = 1_000_000
    let d = 1024
    let m = 8
    let ks = 256

    let x = generateRandomVectors(n: n, d: d)
    let codebooks = trainPQCodebooks(x, n: 100_000, d: d, m: m, ks: ks)

    var codes = [UInt8](repeating: 0, count: n * m)

    let start = Date()
    pq_encode_u8_f32(x, n, d, m, ks, codebooks, &codes, nil)
    let elapsed = Date().timeIntervalSince(start)

    let throughput = Double(n) / elapsed
    print("Encoding throughput: \(throughput) vectors/sec")

    // Expect at least 1M vectors/sec on M2 Max
    assert(throughput > 1_000_000, "Throughput \(throughput) below target")
}
```

---

## Integration Patterns

### 1. IVF-PQ Index Construction

**Complete Workflow**:
```swift
// 1. Train IVF coarse quantizer
let coarse_centroids = trainIVFCentroids(data, n: n, d: d, k: num_coarse)

// 2. Assign vectors to coarse centroids
var assignments = [Int32](repeating: 0, count: n)
assignVectorsToCentroids(data, coarse_centroids, &assignments, n: n, d: d, kc: num_coarse)

// 3. Train PQ codebooks on residuals (kernel #19)
var pq_codebooks = [Float](repeating: 0, count: m * ks * (d/m))
pq_train_f32(data, n, d, m, ks, coarse_centroids, assignments, &cfg, &pq_codebooks, nil, nil)

// 4. Encode all vectors (kernel #20)
var codes = [UInt8](repeating: 0, count: n * m)
pq_encode_residual_u8_f32(data, n, d, m, ks, pq_codebooks, coarse_centroids, assignments, &codes, nil)

// 5. Organize codes by IVF list
var ivf_lists = [[UInt8]](repeating: [], count: num_coarse)
for i in 0..<n {
    let list_id = Int(assignments[i])
    ivf_lists[list_id].append(contentsOf: codes[i*m..<(i+1)*m])
}

// 6. Build index
let index = IVFPQIndex(
    coarse_centroids: coarse_centroids,
    pq_codebooks: pq_codebooks,
    ivf_lists: ivf_lists,
    dimension: d,
    m: m,
    ks: ks
)
```

### 2. Flat PQ Index

**Without IVF**:
```swift
// 1. Train PQ codebooks on original vectors
var pq_codebooks = [Float](repeating: 0, count: m * ks * (d/m))
pq_train_f32(data, n, d, m, ks, nil, nil, &cfg, &pq_codebooks, nil, nil)

// 2. Encode all vectors
var codes = [UInt8](repeating: 0, count: n * m)
pq_encode_u8_f32(data, n, d, m, ks, pq_codebooks, &codes, nil)

// 3. Build flat index
let index = FlatPQIndex(
    pq_codebooks: pq_codebooks,
    codes: codes,
    dimension: d,
    m: m,
    ks: ks
)
```

### 3. Incremental Encoding

**Add new vectors to existing index**:
```c
// Existing index with trained codebooks
IVFPQIndex* index = load_index("index.bin");

// New vectors to add
float* new_vectors = load_vectors("new_vectors.bin", n_new, d);

// Assign to coarse centroids
int32_t* assignments = malloc(n_new * sizeof(int32_t));
ivf_assign_vectors(new_vectors, n_new, d, index->coarse_centroids, index->num_coarse, assignments);

// Encode with residual PQ
uint8_t* codes = malloc(n_new * m);
pq_encode_residual_u8_f32(
    new_vectors, n_new, d, m, ks,
    index->pq_codebooks,
    index->coarse_centroids,
    assignments,
    codes,
    NULL
);

// Append to IVF lists
for (int64_t i = 0; i < n_new; i++) {
    int list_id = assignments[i];
    ivf_list_append(&index->lists[list_id], codes + i*m, m);
}
```

### 4. Integration with ADC Search (Kernel #22)

**Query Encoding** (not PQ-encoded, kept as float32):
```c
// Query remains in float32 for ADC
float* query = load_query(d);

// Build distance lookup tables (kernel #21)
float* luts = build_pq_luts(query, codebooks, m, ks, d);

// Scan codes using ADC (kernel #22)
int32_t* results = adc_scan(codes, luts, n, m, ks, k);
```

**Workflow**:
1. Query stays in float32
2. Build lookup tables: dist(query_j, codebook_j[k]) for all j,k
3. Approximate distance via table lookups: dist(query, x) ≈ Σⱼ LUT[j][code[j]]

---

## Coding Guidelines

### 1. API Design

**Consistent Naming**:
```c
// All PQ encoding functions prefixed with pq_encode_
void pq_encode_u8_f32(...);
void pq_encode_u4_f32(...);
void pq_encode_residual_u8_f32(...);
void pq_encode_residual_u4_f32(...);

// Packing helpers
uint8_t pq_pack_u4_pair(...);
void pq_unpack_u4_pair(...);
```

**Error Handling** (via assertions for debug builds):
```c
assert(d % m == 0 && "d must be divisible by m");
assert(ks == 256 || ks == 16 && "only ks=256 (u8) or ks=16 (u4) supported");
assert(m % 2 == 0 && "m must be even for u4 encoding");
```

### 2. Memory Layout

**AoS Codes** (standard):
```c
// codes[i*m + j] = code for vector i, subspace j
uint8_t codes[n * m];
```

**SoA Blocked Codes** (for ADC):
```c
// codes[j*n + i] = code for vector i, subspace j
uint8_t codes[m * n];
```

### 3. SIMD Patterns

**Vectorized Distance**:
```c
SIMD4<Float> compute_l2_simd(const float* x, const float* c, int dsub) {
    SIMD4<Float> acc = 0;
    for (int i = 0; i < dsub; i += 4) {
        SIMD4<Float> x_vec(x + i);
        SIMD4<Float> c_vec(c + i);
        SIMD4<Float> diff = x_vec - c_vec;
        acc += diff * diff;
    }
    return acc;
}
```

### 4. Telemetry Integration

**Metrics Collection** (#46):
```c
void emit_pq_encode_telemetry(int64_t n, int m, int ks, double time_sec) {
    telemetry_emit("pq_encode.vectors", n);
    telemetry_emit("pq_encode.subspaces", m);
    telemetry_emit("pq_encode.codebook_size", ks);
    telemetry_emit("pq_encode.time_sec", time_sec);
    telemetry_emit("pq_encode.throughput_vec_per_sec", (double)n / time_sec);
    telemetry_emit("pq_encode.bytes_written", n * m * (ks == 16 ? 0.5 : 1));
}
```

---

## Example Usage

### Example 1: Basic 8-bit Encoding

```c
#include "pq_encode.h"

int main() {
    int64_t n = 1000000;
    int d = 1024;
    int m = 8;
    int ks = 256;

    // Load data and codebooks
    float* x = load_vectors("vectors.bin", n, d);
    float* codebooks = load_codebooks("codebooks.bin", m, ks, d/m);

    // Allocate output
    uint8_t* codes = malloc(n * m);

    // Encode
    pq_encode_u8_f32(x, n, d, m, ks, codebooks, codes, NULL);

    // Save codes
    save_codes("codes.bin", codes, n, m);

    // Cleanup
    free(x);
    free(codebooks);
    free(codes);

    return 0;
}
```

### Example 2: 4-bit Encoding with Packing

```c
#include "pq_encode.h"

void encode_4bit(const char* input_file, const char* output_file) {
    int64_t n = 5000000;
    int d = 768;
    int m = 8;  // must be even
    int ks = 16;

    float* x = load_vectors(input_file, n, d);
    float* codebooks = load_codebooks("codebooks_4bit.bin", m, ks, d/m);

    // u4: m/2 bytes per vector
    uint8_t* codes = malloc(n * m / 2);

    // Encode with packing
    pq_encode_u4_f32(x, n, d, m, ks, codebooks, codes, NULL);

    printf("Encoded %ld vectors to %ld bytes (%.1fx compression)\n",
           n, n * m / 2, (double)(n * d * 4) / (n * m / 2));

    save_codes(output_file, codes, n, m / 2);

    free(x);
    free(codebooks);
    free(codes);
}
```

### Example 3: Residual Encoding for IVF-PQ

```c
#include "pq_encode.h"
#include "ivf.h"

void encode_ivf_pq(const char* data_file, IVFPQIndex* index) {
    int64_t n = 10000000;
    int d = 1024;
    int m = 8;
    int ks = 256;

    float* x = load_vectors(data_file, n, d);

    // Assign to IVF lists
    int32_t* assignments = malloc(n * sizeof(int32_t));
    ivf_assign(x, n, d, index->coarse_centroids, index->num_coarse, assignments);

    // Encode residuals
    uint8_t* codes = malloc(n * m);
    pq_encode_residual_u8_f32(
        x, n, d, m, ks,
        index->pq_codebooks,
        index->coarse_centroids,
        assignments,
        codes,
        NULL
    );

    // Organize by IVF list
    for (int64_t i = 0; i < n; i++) {
        int list_id = assignments[i];
        ivf_list_append(&index->lists[list_id], codes + i*m, m);
    }

    printf("Encoded %ld vectors into %d IVF lists\n", n, index->num_coarse);

    free(x);
    free(assignments);
    free(codes);
}
```

### Example 4: Swift Integration

```swift
import Foundation

func encodePQCodes(
    data: [Float],
    n: Int,
    d: Int,
    m: Int,
    ks: Int,
    codebooks: [Float]
) -> [UInt8] {
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 256 || ks == 16, "ks must be 256 or 16")

    let codeSize = (ks == 256) ? m : (m / 2)
    var codes = [UInt8](repeating: 0, count: n * codeSize)

    data.withUnsafeBufferPointer { dataPtr in
        codebooks.withUnsafeBufferPointer { codePtr in
            codes.withUnsafeMutableBufferPointer { codesPtr in
                if ks == 256 {
                    pq_encode_u8_f32(
                        dataPtr.baseAddress!,
                        Int64(n),
                        Int32(d),
                        Int32(m),
                        Int32(ks),
                        codePtr.baseAddress!,
                        codesPtr.baseAddress!,
                        nil
                    )
                } else {
                    pq_encode_u4_f32(
                        dataPtr.baseAddress!,
                        Int64(n),
                        Int32(d),
                        Int32(m),
                        Int32(ks),
                        codePtr.baseAddress!,
                        codesPtr.baseAddress!,
                        nil
                    )
                }
            }
        }
    }

    return codes
}
```

---

## Summary

**Kernel #20** provides efficient PQ encoding for vector compression:

1. **Functionality**: Convert float32 vectors to compact PQ codes via nearest-centroid quantization per subspace
2. **Formats**:
   - 8-bit (ks=256): 1 byte per subspace, m bytes per vector
   - 4-bit (ks=16): 4 bits per subspace, m/2 bytes per vector (packed)
3. **Performance**:
   - 1-10M vectors/sec on M2 Max (depending on ks and tiling)
   - 512-1024× compression ratios
4. **Key Features**:
   - Fused residual computation for IVF-PQ (no materialization)
   - Dot-product trick for 2× speedup with large ks
   - Centroid tiling for L1 cache efficiency
   - SIMD vectorization for distance computation
5. **Integration**:
   - Consumes codebooks from training (kernel #19)
   - Produces codes for ADC search (kernels #21, #22)
   - Supports incremental index construction

**Dependencies**:
- Kernel #01 (L2 distance, dot product)
- Kernel #19 (PQ training, codebooks)
- #48 (memory layouts)

**Typical Use**: Encode 10M vectors (1024-dim) with ks=256, m=8 in ~10 seconds, producing 8 bytes/vector (512× compression) for efficient IVF-PQ search.
