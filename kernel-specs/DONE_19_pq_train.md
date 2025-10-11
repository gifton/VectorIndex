# Kernel Specification #19: Product Quantization (PQ) Codebook Training

**ID**: 19
**Priority**: MUST
**Role**: B (Batch)
**Status**: Specification

---

## Purpose

Train Product Quantization (PQ) codebooks for efficient vector compression in IVF-PQ and flat PQ indexes. Product Quantization decomposes high-dimensional vectors into m independent subspaces, quantizing each subspace separately with a learned codebook of size ks. This enables:

1. **Compression**: Represent d-dimensional float32 vectors with m×log₂(ks) bits
2. **Fast distance computation**: Asymmetric Distance Computation (ADC) via lookup tables
3. **Residual quantization**: Quantize IVF residuals for accurate approximate search

**Typical Use Case**: Compress 1024-dim vectors using m=8 subspaces with ks=256 centroids per subspace, achieving 8 bytes/vector (256× compression from 4096 bytes) with 95%+ recall.

---

## Mathematical Foundations

### 1. Product Quantization

**Definition**: Product Quantization approximates a vector **x** ∈ ℝᵈ by decomposing it into m independent subspaces and quantizing each subspace separately.

**Subspace Decomposition**:
```
x = [x₁ ; x₂ ; ... ; xₘ] where xⱼ ∈ ℝ^(d/m)
```

Each subspace xⱼ has dimension dsub = d/m. We require d to be divisible by m.

**Per-Subspace Quantization**:

For each subspace j ∈ {1, ..., m}, learn a codebook **Cⱼ** = {**c**ⱼ,₁, ..., **c**ⱼ,ₖₛ} ⊂ ℝ^(dsub) of ks codewords.

Quantize subspace vector **x**ⱼ by nearest neighbor:
```
q(xⱼ) = argmin_{k ∈ {1,...,ks}} ‖xⱼ - cⱼ,ₖ‖²
```

**PQ Encoding**: The PQ code for **x** is the tuple of subspace indices:
```
PQ(x) = (q(x₁), q(x₂), ..., q(xₘ)) ∈ {1,...,ks}^m
```

**Storage**: m indices, each requiring log₂(ks) bits.
- For ks=256 (8 bits): m bytes total
- Example: 1024-dim vector with m=8, ks=256 → 8 bytes (vs 4096 bytes for f32)

**Reconstruction**:
```
x̂ = [cₘ,q(x₁) ; cₘ,q(x₂) ; ... ; cₘ,q(xₘ)]
```

**Quantization Error**:
```
‖x - x̂‖² = Σⱼ₌₁ᵐ ‖xⱼ - cⱼ,q(xⱼ)‖²
```

**Key Property**: The total squared error is the sum of per-subspace errors due to independence.

### 2. PQ Training Objective

For a training set **X** = {**x**₁, ..., **x**ₙ} ⊂ ℝᵈ, the PQ training objective is to minimize the total reconstruction error:

```
min_{C₁,...,Cₘ} Σᵢ₌₁ⁿ Σⱼ₌₁ᵐ min_{k ∈ {1,...,ks}} ‖xᵢ,ⱼ - cⱼ,ₖ‖²
```

**Decomposition**: This objective separates across subspaces:
```
min_{C₁,...,Cₘ} Σⱼ₌₁ᵐ [ Σᵢ₌₁ⁿ min_{k ∈ {1,...,ks}} ‖xᵢ,ⱼ - cⱼ,ₖ‖² ]
```

**Implication**: We can train each subspace codebook **Cⱼ** independently by solving a k-means problem on the subspace data {**x**₁,ⱼ, ..., **x**ₙ,ⱼ} with ks centroids.

**Per-Subspace Problem**:
```
min_{Cⱼ} Σᵢ₌₁ⁿ min_{k ∈ {1,...,ks}} ‖xᵢ,ⱼ - cⱼ,ₖ‖²
```

This is exactly the k-means objective with k=ks clusters.

### 3. Residual PQ (for IVF-PQ)

In IVF-PQ indexes, we quantize **residual vectors** rather than original vectors.

**IVF Assignment**: Each vector **x**ᵢ is assigned to a coarse centroid **c**ₐ₍ᵢ₎ where a(i) = argminⱼ ‖**x**ᵢ - **c**ⱼ‖².

**Residual**:
```
rᵢ = xᵢ - c_{a(i)}
```

**Residual PQ Training**: Train PQ codebooks on residuals {**r**₁, ..., **r**ₙ} instead of original vectors.

**Reconstruction**:
```
x̂ᵢ = c_{a(i)} + r̂ᵢ
```

where **r̂**ᵢ is the PQ reconstruction of **r**ᵢ.

**Advantage**: Residuals have lower variance than original vectors, leading to lower quantization error. Typical improvement: 5-10% better recall at same compression ratio.

### 4. Distortion Metric

**Distortion** measures the quality of quantization:

```
D = (1/n) Σᵢ₌₁ⁿ ‖xᵢ - x̂ᵢ‖²
  = (1/n) Σᵢ₌₁ⁿ Σⱼ₌₁ᵐ ‖xᵢ,ⱼ - cⱼ,q(xᵢ,ⱼ)‖²
```

**Normalized Distortion**:
```
D_norm = D / D_baseline
```

where D_baseline is the variance of the data:
```
D_baseline = (1/n) Σᵢ₌₁ⁿ ‖xᵢ - μ‖²
```

**Interpretation**: D_norm ∈ [0,1] measures the fraction of variance retained after quantization.
- D_norm = 0: perfect reconstruction
- D_norm = 0.05: 95% of variance captured (typical for well-tuned PQ)
- D_norm = 1.0: quantization is no better than replacing all vectors with the mean

### 5. Codebook Size vs Quality

**Trade-off**: Larger ks → lower distortion but more expensive ADC computation.

**Typical Configurations**:
- **ks=256** (8 bits): Most common, good balance of quality and speed
- **ks=16** (4 bits): 2× compression, acceptable for very high-dimensional vectors
- **ks=4096** (12 bits): Higher quality, rarely used due to ADC cost

**Bits per Vector**:
```
bits = m × log₂(ks)
```

**Examples**:
- m=8, ks=256: 64 bits = 8 bytes
- m=16, ks=256: 128 bits = 16 bytes
- m=32, ks=16: 128 bits = 16 bytes

**Subspace Dimension**:
```
dsub = d / m
```

**Constraint**: Effective codebook training requires n >> ks×dsub. Rule of thumb: n ≥ 100×ks for robust training.

---

## API Signatures

### 1. Batch PQ Training

```c
int pq_train_f32(
    const float* x,                    // [n × d] input vectors (AoS)
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int m,                             // number of subspaces
    int ks,                            // codebook size per subspace
    const float* coarse_centroids,     // [kc × d] IVF centroids (nullable)
    const int32_t* assign,             // [n] IVF assignments (nullable)
    const PQTrainConfig* cfg,          // configuration
    float* codebooks_out,              // [m × ks × dsub] output codebooks
    float* centroid_norms_out,         // [m × ks] output norms (nullable)
    PQTrainStats* stats_out            // output statistics (nullable)
);
```

**Parameters**:

- `x`: Input vectors in AoS layout `[n][d]`, **must be 64-byte aligned**
- `n`: Number of training vectors (recommended: n ≥ 100×ks)
- `d`: Dimension (must be divisible by m)
- `m`: Number of subspaces
  - Typical: 8, 16, 32
  - Trade-off: larger m → more flexibility, but each subspace has fewer dimensions
- `ks`: Codebook size per subspace
  - Typical: 256 (8-bit codes)
  - Also supported: 16 (4-bit codes), others
- `coarse_centroids`: IVF coarse centroids for residual PQ (nullable)
  - If non-null: train on residuals rᵢ = xᵢ - coarse_centroids[assign[i]]
  - If null: train on original vectors
- `assign`: IVF assignments for each vector (nullable, required if coarse_centroids is non-null)
- `cfg`: Configuration (detailed below)
- `codebooks_out`: Output buffer for codebooks, **must be preallocated** to m×ks×dsub floats
  - Layout: `codebooks_out[j*ks*dsub + k*dsub + i]` = dimension i of codeword k in subspace j
- `centroid_norms_out`: Optional precomputed squared norms ‖**c**ⱼ,ₖ‖² (nullable)
  - If non-null: fill with squared norms for dot-product-based distance computation
  - Layout: `centroid_norms_out[j*ks + k]` = ‖**c**ⱼ,ₖ‖²
- `stats_out`: Optional training statistics (nullable)

**Return Value**:
- `0`: Success
- `PQ_ERR_INVALID_DIM`: d not divisible by m
- `PQ_ERR_INVALID_K`: ks < 1 or ks > 65536
- `PQ_ERR_INSUFFICIENT_DATA`: n < ks
- `PQ_ERR_NULL_PTR`: required pointer is null
- `PQ_ERR_ALIGNMENT`: x is not 64-byte aligned

**Configuration** (`PQTrainConfig`):
```c
typedef struct {
    PQAlgorithm algo;              // Lloyd or MiniBatch
    int max_iters;                 // max k-means iterations per subspace (default: 25)
    float tol;                     // convergence tolerance (default: 1e-4)
    int batch_size;                // mini-batch size for MiniBatch algo (default: 1024)
    int64_t sample_n;              // optional: subsample training data (0 = use all)
    uint64_t seed;                 // RNG seed for k-means++ and mini-batch sampling
    int stream_id;                 // RNG stream ID (default: 0)
    EmptyClusterPolicy empty_policy; // how to handle empty clusters
    bool precompute_x_norm2;       // precompute ‖xᵢ,ⱼ‖² for dot-product distance (default: true)
    bool compute_centroid_norms;   // fill centroid_norms_out (default: true if non-null)
    int num_threads;               // parallelism (0 = auto, default: 0)
} PQTrainConfig;

typedef enum {
    PQ_ALGO_LLOYD,        // Full Lloyd's k-means
    PQ_ALGO_MINIBATCH     // Mini-batch k-means (faster for large n)
} PQAlgorithm;

typedef enum {
    EMPTY_POLICY_SPLIT,   // Split largest cluster
    EMPTY_POLICY_RESEED,  // Re-seed from farthest point
    EMPTY_POLICY_IGNORE   // Leave empty (not recommended)
} EmptyClusterPolicy;
```

**Statistics** (`PQTrainStats`):
```c
typedef struct {
    double distortion;             // total distortion (mean squared error)
    double distortion_per_subspace[MAX_SUBSPACES]; // per-subspace distortion
    int iters_per_subspace[MAX_SUBSPACES];         // iterations per subspace
    int empties_repaired;          // total empty clusters repaired
    double time_init_sec;          // time for k-means++ initialization
    double time_train_sec;         // time for k-means iterations
    int64_t bytes_read;            // total bytes read
} PQTrainStats;
```

### 2. Streaming PQ Training

For very large datasets that don't fit in memory, process data in chunks.

```c
int pq_train_streaming_f32(
    const float* x_chunked[],          // [chunks] array of chunk pointers
    const int64_t n_chunks[],          // [chunks] number of vectors per chunk
    int chunks,                        // number of chunks
    int d,                             // dimension
    int m,                             // number of subspaces
    int ks,                            // codebook size per subspace
    const float* coarse_centroids,     // [kc × d] IVF centroids (nullable)
    const int32_t* assign_chunked[],   // [chunks] array of assignment pointers (nullable)
    const PQTrainConfig* cfg,          // configuration
    float* codebooks_out,              // [m × ks × dsub] output codebooks
    float* centroid_norms_out,         // [m × ks] output norms (nullable)
    PQTrainStats* stats_out            // output statistics (nullable)
);
```

**Parameters**:
- `x_chunked`: Array of pointers to data chunks
- `n_chunks`: Number of vectors in each chunk
- `chunks`: Number of chunks
- Other parameters same as `pq_train_f32`

**Use Case**: When total dataset size exceeds RAM, process in chunks to avoid loading all data at once.

**Algorithm**: Use mini-batch k-means across chunks, maintaining codebook state between chunks.

---

## Algorithm Details

### 1. Overall Training Flow

```
pq_train_f32(x, n, d, m, ks, coarse_centroids, assign, cfg):
    dsub = d / m

    // Parallel over subspaces
    parallel for j in 0..m-1:
        // Extract subspace view (no copy)
        x_sub = view(x, offset=j*dsub, stride=d, count=n, dim=dsub)

        // Optionally compute residuals on-the-fly
        if coarse_centroids != NULL:
            residual_mode = true
            coarse_sub = view(coarse_centroids, offset=j*dsub, stride=d, dim=dsub)

        // k-means++ initialization
        seeds = kmeans_plusplus_seed(x_sub, n, dsub, ks, cfg.seed + j)

        // Lloyd's or mini-batch k-means
        if cfg.algo == LLOYD:
            codebook_j = lloyd_kmeans(x_sub, n, dsub, ks, seeds, cfg)
        else:
            codebook_j = minibatch_kmeans(x_sub, n, dsub, ks, seeds, cfg)

        // Store codebook
        codebooks_out[j*ks*dsub : (j+1)*ks*dsub] = codebook_j

        // Optionally compute norms
        if cfg.compute_centroid_norms:
            for k in 0..ks-1:
                centroid_norms_out[j*ks + k] = dot(codebook_j[k], codebook_j[k])

    // Compute total distortion
    if stats_out != NULL:
        stats_out.distortion = compute_distortion(x, codebooks_out, m, ks, dsub)
```

### 2. Subspace View (Zero-Copy)

**Challenge**: Extract subspace data without copying.

**Solution**: Use strided view with offset and stride.

```c
// Extract subspace j from x[n][d]
// Returns view of x_sub[n][dsub] where x_sub[i][k] = x[i][j*dsub + k]

typedef struct {
    const float* base;    // base pointer to x
    int64_t n;            // number of vectors
    int dsub;             // subspace dimension
    int d;                // original dimension (stride)
    int offset;           // starting offset (j*dsub)
} SubspaceView;

SubspaceView make_subspace_view(const float* x, int64_t n, int d, int m, int j) {
    int dsub = d / m;
    return (SubspaceView){
        .base = x,
        .n = n,
        .dsub = dsub,
        .d = d,
        .offset = j * dsub
    };
}

// Access element: x_sub[i][k] = x[i*d + offset + k]
float subspace_get(const SubspaceView* view, int64_t i, int k) {
    return view->base[i * view->d + view->offset + k];
}

// Copy subspace vector to contiguous buffer for distance computation
void subspace_copy(const SubspaceView* view, int64_t i, float* out) {
    const float* src = view->base + i * view->d + view->offset;
    memcpy(out, src, view->dsub * sizeof(float));
}
```

**SIMD-Friendly Access**:
```c
// Load subspace vector for SIMD distance computation
void subspace_load_simd(const SubspaceView* view, int64_t i, SIMD4<Float>* out, int num_simd) {
    const float* src = view->base + i * view->d + view->offset;
    for (int s = 0; s < num_simd; s++) {
        out[s] = SIMD4<Float>(src + s*4);
    }
}
```

### 3. On-the-Fly Residual Computation

For residual PQ, compute residuals during distance computation rather than materializing them.

```c
// Compute distance between x_sub[i] and codebook centroid c[k]
// with optional residual subtraction

float subspace_distance_with_residual(
    const SubspaceView* x_view,
    int64_t i,
    const float* codebook,
    int k,
    const SubspaceView* coarse_view,  // nullable
    const int32_t* assign              // nullable
) {
    int dsub = x_view->dsub;
    const float* x = x_view->base + i * x_view->d + x_view->offset;
    const float* c = codebook + k * dsub;

    float dist_sq = 0;

    if (coarse_view != NULL) {
        // Residual mode: distance to (x - coarse)
        int a = assign[i];
        const float* coarse = coarse_view->base + a * coarse_view->d + coarse_view->offset;

        for (int j = 0; j < dsub; j += 4) {
            SIMD4<Float> x_vec(x + j);
            SIMD4<Float> coarse_vec(coarse + j);
            SIMD4<Float> c_vec(c + j);
            SIMD4<Float> residual = x_vec - coarse_vec;
            SIMD4<Float> diff = residual - c_vec;
            dist_sq += reduce_add(diff * diff);
        }
    } else {
        // Direct mode: distance to x
        for (int j = 0; j < dsub; j += 4) {
            SIMD4<Float> x_vec(x + j);
            SIMD4<Float> c_vec(c + j);
            SIMD4<Float> diff = x_vec - c_vec;
            dist_sq += reduce_add(diff * diff);
        }
    }

    return dist_sq;
}
```

### 4. Lloyd's K-means per Subspace

**Pseudocode**:
```
lloyd_kmeans_subspace(x_view, n, dsub, ks, seeds, cfg):
    C = copy(seeds)  // [ks][dsub]

    for iter in 0..cfg.max_iters-1:
        // Assignment
        assignments = [0] * n
        S = zeros[ks][dsub]  // per-cluster sums
        N = zeros[ks]        // per-cluster counts

        parallel for i in 0..n-1:
            best_k = 0
            best_dist = distance(x_view, i, C[0])

            for k in 1..ks-1:
                dist = distance(x_view, i, C[k])
                if dist < best_dist or (dist == best_dist and k < best_k):
                    best_dist = dist
                    best_k = k

            assignments[i] = best_k
            atomic_add(N[best_k], 1)
            atomic_add_vector(S[best_k], x_view[i])

        // Update
        for k in 0..ks-1:
            if N[k] > 0:
                C[k] = S[k] / N[k]

        // Empty cluster repair
        empties = [k : N[k] == 0]
        if len(empties) > 0:
            repair_empty_clusters(C, empties, x_view, N, cfg.empty_policy)

        // Convergence check
        distortion = compute_distortion_subspace(x_view, C, assignments)
        improvement = (prev_distortion - distortion) / prev_distortion

        if improvement < cfg.tol and iter > 0:
            break

        prev_distortion = distortion

    return C
```

### 5. Mini-batch K-means per Subspace

For large n, use mini-batch k-means (see kernel #12 for details).

**Key Differences**:
- Process data in batches of size `cfg.batch_size`
- Update centroids after each batch rather than after full pass
- Faster convergence in wall-clock time for n > 100k

**Pseudocode**:
```
minibatch_kmeans_subspace(x_view, n, dsub, ks, seeds, cfg):
    C = copy(seeds)

    num_batches = ceil(n / cfg.batch_size)

    for iter in 0..cfg.max_iters-1:
        // Shuffle data indices
        indices = shuffle([0, 1, ..., n-1], cfg.seed + iter)

        for batch in 0..num_batches-1:
            start = batch * cfg.batch_size
            end = min(start + cfg.batch_size, n)
            batch_indices = indices[start:end]

            // Assignment + accumulation for batch
            S_batch = zeros[ks][dsub]
            N_batch = zeros[ks]

            for idx in batch_indices:
                k = assign(x_view, idx, C)
                N_batch[k] += 1
                S_batch[k] += x_view[idx]

            // Update centroids
            for k in 0..ks-1:
                if N_batch[k] > 0:
                    C[k] = S_batch[k] / N_batch[k]

            // Empty repair if needed
            empties = [k : N_batch[k] == 0 for all recent batches]
            if len(empties) > 0:
                repair_empty_clusters(C, empties, batch_data, N_batch, cfg.empty_policy)

    return C
```

### 6. Empty Cluster Repair

**Strategy 1: Split Largest Cluster** (default)
```
repair_empty_split(C, empties, x_view, N, empty_policy):
    for k_empty in empties (sorted ascending):
        // Find largest cluster
        k_max = argmax(N)

        // Find farthest point from k_max in current batch
        max_dist = -1
        i_far = -1
        for i in current_batch:
            if assign[i] == k_max:
                dist = distance(x_view, i, C[k_max])
                if dist > max_dist:
                    max_dist = dist
                    i_far = i

        // Assign to empty cluster
        if i_far >= 0:
            copy_vector(x_view, i_far, C[k_empty])
```

**Strategy 2: Re-seed from Farthest Point**
```
repair_empty_reseed(C, empties, x_view):
    for k_empty in empties (sorted ascending):
        // Find point farthest from all centroids
        max_min_dist = -1
        i_far = -1

        for i in current_batch:
            min_dist = min_k distance(x_view, i, C[k])
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                i_far = i

        if i_far >= 0:
            copy_vector(x_view, i_far, C[k_empty])
```

### 7. Distance Computation Optimization

**Dot-Product Trick**: For L2 distance, use:
```
‖x - c‖² = ‖x‖² + ‖c‖² - 2⟨x, c⟩
```

If we precompute ‖**x**ᵢ,ⱼ‖² and ‖**c**ⱼ,ₖ‖², distance computation reduces to:
```
dist² = query_norm + centroid_norm - 2 * dot_product(query, centroid)
```

**Implementation**:
```c
// Precompute query norms (once per iteration)
float* query_norms = malloc(n * sizeof(float));
parallel for i in 0..n-1:
    query_norms[i] = dot_product(x_view[i], x_view[i], dsub);

// Precompute centroid norms (once per iteration)
float* centroid_norms = malloc(ks * sizeof(float));
for k in 0..ks-1:
    centroid_norms[k] = dot_product(C[k], C[k], dsub);

// Assignment with dot-product trick
for i in 0..n-1:
    int k_best = 0;
    float dist_best = query_norms[i] + centroid_norms[0] - 2*dot_product(x_view[i], C[0], dsub);

    for k in 1..ks-1:
        float dist = query_norms[i] + centroid_norms[k] - 2*dot_product(x_view[i], C[k], dsub);
        if dist < dist_best:
            dist_best = dist;
            k_best = k;
    }

    assign[i] = k_best;
}
```

**Benefit**: Reduces distance computation from 2×dsub FLOPs (squared diff + sum) to dsub FLOPs (dot product only).

**Trade-off**: Requires O(n + ks) memory for norms and O(n + ks) time to precompute. Beneficial when ks is large (e.g., ks ≥ 64) and multiple iterations are performed.

---

## Implementation Strategies

### 1. Parallelism

**Parallel over Subspaces** (coarse-grained):
```c
#pragma omp parallel for num_threads(cfg->num_threads)
for (int j = 0; j < m; j++) {
    SubspaceView x_sub = make_subspace_view(x, n, d, m, j);

    float* codebook_j = codebooks_out + j * ks * dsub;
    float* norms_j = centroid_norms_out ? (centroid_norms_out + j * ks) : NULL;

    train_subspace_codebook(x_sub, ks, cfg, codebook_j, norms_j);
}
```

**Benefit**:
- Perfect load balance if all subspaces have similar convergence time
- No synchronization between subspaces
- Scales linearly with number of cores (up to m cores)

**Parallel over Samples** (fine-grained, within each subspace):
```c
// Within each subspace's k-means loop
#pragma omp parallel for schedule(static)
for (int64_t i = 0; i < n; i++) {
    int k_best = assign_nearest(x_sub, i, codebook, ks, dsub);

    // Thread-local accumulation
    accumulate_local(S_local, N_local, x_sub, i, k_best);
}

// Deterministic reduction
#pragma omp critical
{
    for (int k = 0; k < ks; k++) {
        N_global[k] += N_local[k];
        for (int d = 0; d < dsub; d++) {
            S_global[k][d] += S_local[k][d];
        }
    }
}
```

**Hybrid Parallelism**:
- If m ≥ num_threads: parallelize over subspaces only
- If m < num_threads: parallelize over subspaces, then over samples within each subspace

### 2. Vectorization

**SIMD Distance Computation**:
```c
float l2_squared_simd(const float* a, const float* b, int dsub) {
    SIMD4<Float> acc0 = 0, acc1 = 0;

    int d_vec = dsub & ~7;  // round down to multiple of 8
    for (int i = 0; i < d_vec; i += 8) {
        SIMD4<Float> a0(a + i), a1(a + i + 4);
        SIMD4<Float> b0(b + i), b1(b + i + 4);
        SIMD4<Float> diff0 = a0 - b0;
        SIMD4<Float> diff1 = a1 - b1;
        acc0 += diff0 * diff0;
        acc1 += diff1 * diff1;
    }

    float sum = reduce_add(acc0) + reduce_add(acc1);

    // Handle remainder
    for (int i = d_vec; i < dsub; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}
```

**Batched Assignment** (evaluate multiple centroids in parallel):
```c
// Assign one query to 4 centroids in parallel
void assign_batch_4(const float* query, const float* centroids, int dsub, float* dists) {
    const float* c0 = centroids + 0*dsub;
    const float* c1 = centroids + 1*dsub;
    const float* c2 = centroids + 2*dsub;
    const float* c3 = centroids + 3*dsub;

    SIMD4<Float> acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    for (int i = 0; i < dsub; i += 4) {
        SIMD4<Float> q(query + i);
        SIMD4<Float> d0 = q - SIMD4<Float>(c0 + i);
        SIMD4<Float> d1 = q - SIMD4<Float>(c1 + i);
        SIMD4<Float> d2 = q - SIMD4<Float>(c2 + i);
        SIMD4<Float> d3 = q - SIMD4<Float>(c3 + i);
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        acc2 += d2 * d2;
        acc3 += d3 * d3;
    }

    dists[0] = reduce_add(acc0);
    dists[1] = reduce_add(acc1);
    dists[2] = reduce_add(acc2);
    dists[3] = reduce_add(acc3);
}
```

### 3. Memory Layout

**Codebook Layout** (optimal for ADC):
```
// Layout 1: [m][ks][dsub] (contiguous per subspace)
codebooks[j][k][i] stored at: base + (j*ks*dsub + k*dsub + i) * sizeof(float)

// Access pattern for encoding:
for j in 0..m-1:
    k_best = argmin_k distance(x[j*dsub:(j+1)*dsub], codebooks[j][k])

// Good cache locality: all centroids for subspace j are contiguous
```

**Alternative Layout** (optimal for reconstruction):
```
// Layout 2: [ks][m][dsub] (interleaved by codeword)
// Rarely used, harder to train
```

**Recommendation**: Use Layout 1 (per-subspace contiguous) for both training and ADC.

### 4. Tiling

**Row Tiling** (for assignment):
```c
const int TILE_SIZE = 1024;  // process 1024 vectors at a time

for (int64_t tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
    int64_t tile_end = min(tile_start + TILE_SIZE, n);
    int tile_size = tile_end - tile_start;

    // Prefetch next tile
    if (tile_end < n) {
        prefetch_tile(x_view, tile_end, min(tile_end + TILE_SIZE, n));
    }

    // Process current tile
    for (int64_t i = tile_start; i < tile_end; i++) {
        assignments[i] = assign_nearest(x_view, i, codebook, ks, dsub);
    }
}
```

**Centroid Tiling** (for assignment):
```c
const int CENTROID_TILE = 32;  // evaluate 32 centroids at a time

for (int64_t i = 0; i < n; i++) {
    float dist_best = INFINITY;
    int k_best = 0;

    for (int k_tile = 0; k_tile < ks; k_tile += CENTROID_TILE) {
        int k_end = min(k_tile + CENTROID_TILE, ks);

        // Prefetch next tile
        if (k_end < ks) {
            prefetch_centroids(codebook, k_end, min(k_end + CENTROID_TILE, ks), dsub);
        }

        // Evaluate current tile
        for (int k = k_tile; k < k_end; k++) {
            float dist = l2_squared_simd(x_view[i], codebook + k*dsub, dsub);
            if (dist < dist_best) {
                dist_best = dist;
                k_best = k;
            }
        }
    }

    assignments[i] = k_best;
}
```

---

## Performance Characteristics

### 1. Computational Complexity

**Per Subspace** (Lloyd's k-means):
- Assignment: O(n×ks×dsub) FLOPs per iteration
- Update: O(n×dsub) FLOPs per iteration
- Iterations: typically 15-25
- **Total**: O(T×n×ks×dsub) where T ≈ 20

**Full PQ Training** (m subspaces):
- Serial: O(m×T×n×ks×dsub) = O(T×n×ks×d)
- Parallel (m threads): O(T×n×ks×d/m) per core

**Example** (d=1024, m=8, ks=256, n=10⁶, T=20):
- FLOPs per subspace: 20 × 10⁶ × 256 × 128 ≈ 6.5×10¹¹ FLOPs
- Total FLOPs: 8 × 6.5×10¹¹ ≈ 5.2×10¹² FLOPs
- Time at 1 TFLOP/s: ~5 seconds per subspace, ~40 seconds total (if serial)
- Time at 1 TFLOP/s with 8 cores: ~5 seconds total

### 2. Memory Bandwidth

**Reads per Assignment**:
- Query subvector: dsub×4 bytes
- All centroids: ks×dsub×4 bytes
- Total per query: (1+ks)×dsub×4 bytes

**Example** (dsub=128, ks=256):
- Bandwidth per query: 257×128×4 ≈ 131 KB
- Queries per second at 200 GB/s: 200×10⁹ / (131×10³) ≈ 1.5M queries/s

**With Centroid Tiling** (tile size = 32 centroids, all fit in L1):
- Bandwidth per query: dsub×4 bytes (query only, centroids in cache)
- Example: 128×4 = 512 bytes
- Queries per second: 200×10⁹ / 512 ≈ 390M queries/s
- **~260× speedup** from tiling

### 3. Performance Targets (Apple M2 Max, 8 P-cores)

| Configuration | Time per Subspace | Total Time (8 subspaces) | Notes |
|---------------|-------------------|--------------------------|-------|
| n=100K, ks=256, dsub=128 | 2.5 sec | 2.5 sec | Parallel over subspaces |
| n=1M, ks=256, dsub=128 | 25 sec | 25 sec | Parallel over subspaces |
| n=10M, ks=256, dsub=128 | 250 sec | 250 sec | Parallel over subspaces |
| n=1M, ks=256, dsub=64 | 12 sec | 12 sec | Smaller dsub → faster |
| n=1M, ks=16, dsub=128 | 2 sec | 2 sec | Fewer centroids → faster |

**Scaling**:
- **n**: Linear scaling (O(n))
- **ks**: Linear scaling (O(ks))
- **dsub**: Linear scaling (O(dsub))
- **m**: Constant time (parallel over subspaces, up to 8 cores)

**Bottleneck**: Memory bandwidth for large ks (ks ≥ 128), compute for small ks (ks ≤ 64).

### 4. Distortion vs Compression

**Typical Results** (1024-dim embeddings, m=8):
- **ks=256** (8 bytes): 5-10% distortion, 95-98% recall@10
- **ks=16** (4 bytes): 15-25% distortion, 85-92% recall@10
- **ks=4096** (12 bytes): 2-5% distortion, 98-99% recall@10

**Residual vs Direct**:
- Residual PQ: 5-10% better recall at same compression
- Cost: Requires IVF index and assignment computation

---

## Numerical Considerations

### 1. Floating-point Accumulation

**Centroid Update** (same as k-means #12):
```c
// Use f64 accumulation
double S[dsub];
memset(S, 0, dsub * sizeof(double));

for (int64_t i = 0; i < n; i++) {
    if (assign[i] == k) {
        for (int d = 0; d < dsub; d++) {
            S[d] += (double)x_sub[i][d];
        }
    }
}

// Convert to f32
for (int d = 0; d < dsub; d++) {
    centroid[d] = (float)(S[d] / count);
}
```

### 2. Deterministic Tie-breaking

**Distance Ties**:
```c
// Prefer smaller index on tie
if (dist < dist_best || (dist == dist_best && k < k_best)) {
    dist_best = dist;
    k_best = k;
}
```

### 3. Subspace Independence

**Correctness Property**: Training each subspace independently must be mathematically equivalent to joint training.

**Verification**:
```
Total distortion = Σⱼ₌₁ᵐ (distortion in subspace j)
```

**Test**:
```swift
func testSubspaceIndependence() {
    let (x, m, ks) = generateTestData()

    // Train all subspaces jointly
    var codebooks_joint = trainPQJoint(x, m, ks)
    let distortion_joint = computeDistortion(x, codebooks_joint, m, ks)

    // Train each subspace separately and sum distortions
    var distortion_separate: Double = 0
    for j in 0..<m {
        let x_sub = extractSubspace(x, j, m)
        let codebook_j = trainKMeans(x_sub, ks)
        distortion_separate += computeDistortion(x_sub, codebook_j, ks)
    }

    // Should be identical (within numerical tolerance)
    assert(abs(distortion_joint - distortion_separate) < 1e-5)
}
```

### 4. Norm Precomputation Accuracy

**Potential Issue**: Precomputed norms can become stale after centroid updates.

**Solution**: Recompute norms after each update.
```c
// After centroid update
for (int k = 0; k < ks; k++) {
    if (N[k] > 0) {
        // Update centroid
        for (int d = 0; d < dsub; d++) {
            C[k][d] = S[k][d] / N[k];
        }

        // Recompute norm
        float norm_sq = 0;
        for (int d = 0; d < dsub; d++) {
            norm_sq += C[k][d] * C[k][d];
        }
        centroid_norms[k] = norm_sq;
    }
}
```

---

## Correctness Testing

### 1. Distortion Tests

**Test 1: Distortion Decreases**
```swift
func testDistortionDecreases() {
    let x = generateRandomVectors(n: 50_000, d: 1024)
    let m = 8
    let ks = 256

    var codebooks = [Float](repeating: 0, count: m * ks * (1024/m))
    var stats = PQTrainStats()
    var cfg = PQTrainConfig(max_iters: 20, seed: 42)

    pq_train_f32(x, 50_000, 1024, m, ks, nil, nil, &cfg, &codebooks, nil, &stats)

    // Check per-subspace distortion
    for j in 0..<m {
        let distortions = stats.distortion_per_subspace[j]
        for i in 1..<distortions.count {
            assert(distortions[i] <= distortions[i-1],
                   "Distortion increased in subspace \(j) iter \(i)")
        }
    }
}
```

**Test 2: Distortion vs Baseline**
```swift
func testDistortionVsBaseline() {
    let x = generateRandomVectors(n: 10_000, d: 512)
    let m = 8
    let ks = 256

    var codebooks = [Float](repeating: 0, count: m * ks * (512/m))
    var stats = PQTrainStats()
    var cfg = PQTrainConfig(seed: 42)

    pq_train_f32(x, 10_000, 512, m, ks, nil, nil, &cfg, &codebooks, nil, &stats)

    // Baseline: replace all vectors with mean
    let mean = computeMean(x, n: 10_000, d: 512)
    var baseline_distortion: Double = 0
    for i in 0..<10_000 {
        baseline_distortion += l2SquaredDistance(x + i*512, mean, 512)
    }
    baseline_distortion /= Double(10_000)

    // PQ should achieve much better than baseline
    let improvement = 1.0 - stats.distortion / baseline_distortion
    assert(improvement > 0.5, "PQ only improved \(improvement*100)% over baseline")
}
```

### 2. Subspace Independence Tests

**Test 3: Per-Subspace Training Equivalence**
```swift
func testPerSubspaceEquivalence() {
    let x = generateRandomVectors(n: 5_000, d: 256)
    let m = 4
    let ks = 64
    let dsub = 256 / 4

    // Train all subspaces together
    var codebooks_all = [Float](repeating: 0, count: m * ks * dsub)
    var cfg = PQTrainConfig(seed: 42)
    pq_train_f32(x, 5_000, 256, m, ks, nil, nil, &cfg, &codebooks_all, nil, nil)

    // Train each subspace independently
    var codebooks_separate = [Float](repeating: 0, count: m * ks * dsub)
    for j in 0..<m {
        let x_sub = extractSubspace(x, n: 5_000, d: 256, m: m, j: j)
        var codebook_j = [Float](repeating: 0, count: ks * dsub)
        var cfg_j = PQTrainConfig(seed: 42)

        // Use k-means directly on subspace
        kmeans_minibatch_f32(x_sub, 5_000, Int32(dsub), Int32(ks), nil, &cfg_j,
                             &codebook_j, nil, nil)

        // Copy to combined codebook
        for k in 0..<ks {
            for d in 0..<dsub {
                codebooks_separate[j*ks*dsub + k*dsub + d] = codebook_j[k*dsub + d]
            }
        }
    }

    // Compare distortions
    let dist_all = computePQDistortion(x, codebooks_all, n: 5_000, d: 256, m: m, ks: ks)
    let dist_sep = computePQDistortion(x, codebooks_separate, n: 5_000, d: 256, m: m, ks: ks)

    assert(abs(dist_all - dist_sep) / dist_all < 0.01,
           "Distortion mismatch: \(dist_all) vs \(dist_sep)")
}
```

### 3. Residual PQ Tests

**Test 4: Residual vs Direct PQ**
```swift
func testResidualPQDistortion() {
    let n = 20_000
    let d = 768
    let kc = 100  // coarse centroids
    let m = 8
    let ks = 256

    let x = generateClusteredVectors(n: n, d: d, num_clusters: kc)

    // Train coarse quantizer
    let coarse_centroids = trainIVFCentroids(x, n: n, d: d, k: kc)
    var assignments = [Int32](repeating: 0, count: n)
    assignVectorsToCentroids(x, coarse_centroids, &assignments, n: n, d: d, kc: kc)

    // Direct PQ
    var codebooks_direct = [Float](repeating: 0, count: m * ks * (d/m))
    var stats_direct = PQTrainStats()
    var cfg = PQTrainConfig(seed: 42)
    pq_train_f32(x, n, d, m, ks, nil, nil, &cfg, &codebooks_direct, nil, &stats_direct)

    // Residual PQ
    var codebooks_residual = [Float](repeating: 0, count: m * ks * (d/m))
    var stats_residual = PQTrainStats()
    pq_train_f32(x, n, d, m, ks, coarse_centroids, assignments, &cfg,
                 &codebooks_residual, nil, &stats_residual)

    // Residual should have lower distortion
    assert(stats_residual.distortion < stats_direct.distortion,
           "Residual PQ distortion \(stats_residual.distortion) >= direct \(stats_direct.distortion)")

    let improvement = (stats_direct.distortion - stats_residual.distortion) / stats_direct.distortion
    print("Residual PQ improvement: \(improvement * 100)%")
}
```

### 4. Determinism Tests

**Test 5: Reproducibility**
```swift
func testPQDeterminism() {
    let x = generateRandomVectors(n: 10_000, d: 512)
    let m = 8
    let ks = 256
    let seed: UInt64 = 123456

    var codebooks1 = [Float](repeating: 0, count: m * ks * (512/m))
    var codebooks2 = [Float](repeating: 0, count: m * ks * (512/m))

    var cfg = PQTrainConfig(seed: seed, num_threads: 4)
    pq_train_f32(x, 10_000, 512, m, ks, nil, nil, &cfg, &codebooks1, nil, nil)
    pq_train_f32(x, 10_000, 512, m, ks, nil, nil, &cfg, &codebooks2, nil, nil)

    // Should be bitwise identical
    for i in 0..<codebooks1.count {
        let diff = abs(codebooks1[i] - codebooks2[i])
        assert(diff < 1e-6, "Codebook mismatch at index \(i): \(diff)")
    }
}
```

### 5. Encoding/Decoding Tests

**Test 6: Encode-Decode Consistency**
```swift
func testEncodeDecodeConsistency() {
    let x = generateRandomVectors(n: 1_000, d: 256)
    let m = 4
    let ks = 64

    // Train codebooks
    var codebooks = [Float](repeating: 0, count: m * ks * (256/m))
    var cfg = PQTrainConfig(seed: 42)
    pq_train_f32(x, 1_000, 256, m, ks, nil, nil, &cfg, &codebooks, nil, nil)

    // Encode and decode
    for i in 0..<1_000 {
        let x_vec = Array(x[i*256..<(i+1)*256])

        // Encode: find nearest codeword in each subspace
        var codes = [UInt8](repeating: 0, count: m)
        for j in 0..<m {
            let x_sub = Array(x_vec[j*(256/m)..<(j+1)*(256/m)])
            var min_dist = Float.infinity
            var best_k = 0

            for k in 0..<ks {
                let c_sub = Array(codebooks[j*ks*(256/m) + k*(256/m)..<j*ks*(256/m) + (k+1)*(256/m)])
                let dist = l2SquaredDistance(x_sub, c_sub)
                if dist < min_dist {
                    min_dist = dist
                    best_k = k
                }
            }
            codes[j] = UInt8(best_k)
        }

        // Decode: reconstruct from codes
        var x_reconstructed = [Float](repeating: 0, count: 256)
        for j in 0..<m {
            let k = Int(codes[j])
            let offset = j*ks*(256/m) + k*(256/m)
            for d in 0..<(256/m) {
                x_reconstructed[j*(256/m) + d] = codebooks[offset + d]
            }
        }

        // Verify reconstruction is nearest in each subspace
        for j in 0..<m {
            let x_sub = Array(x_vec[j*(256/m)..<(j+1)*(256/m)])
            let rec_sub = Array(x_reconstructed[j*(256/m)..<(j+1)*(256/m)])

            // Check that rec_sub is the nearest centroid
            let dist_rec = l2SquaredDistance(x_sub, rec_sub)
            for k in 0..<ks {
                let c_sub = Array(codebooks[j*ks*(256/m) + k*(256/m)..<j*ks*(256/m) + (k+1)*(256/m)])
                let dist_k = l2SquaredDistance(x_sub, c_sub)
                assert(dist_rec <= dist_k + 1e-5, "Reconstruction is not nearest centroid")
            }
        }
    }
}
```

---

## Integration Patterns

### 1. IVF-PQ Training Pipeline

**Complete Workflow**:
```swift
// 1. Train IVF coarse quantizer
let coarse_centroids = trainIVFCentroids(
    data: training_data,
    n: n_train,
    d: dimension,
    k: num_coarse_centroids
)

// 2. Assign training data to coarse centroids
var assignments = [Int32](repeating: 0, count: n_train)
assignVectorsToCentroids(training_data, coarse_centroids, &assignments,
                         n: n_train, d: dimension, kc: num_coarse_centroids)

// 3. Train PQ codebooks on residuals
var pq_codebooks = [Float](repeating: 0, count: m * ks * (dimension/m))
var pq_norms = [Float](repeating: 0, count: m * ks)
var cfg = PQTrainConfig(
    algo: .lloyd,
    max_iters: 25,
    seed: 42,
    compute_centroid_norms: true
)
var stats = PQTrainStats()

pq_train_f32(
    training_data,
    Int64(n_train),
    Int32(dimension),
    Int32(m),
    Int32(ks),
    coarse_centroids,
    assignments,
    &cfg,
    &pq_codebooks,
    &pq_norms,
    &stats
)

print("PQ training completed in \(stats.time_train_sec) sec")
print("Distortion: \(stats.distortion)")

// 4. Build IVF-PQ index
let index = IVFPQIndex(
    coarse_centroids: coarse_centroids,
    pq_codebooks: pq_codebooks,
    pq_norms: pq_norms,
    dimension: dimension,
    num_coarse: num_coarse_centroids,
    m: m,
    ks: ks
)
```

### 2. Flat PQ Training

**Without IVF** (quantize original vectors):
```swift
var pq_codebooks = [Float](repeating: 0, count: m * ks * (dimension/m))
var cfg = PQTrainConfig(
    algo: .minibatch,
    batch_size: 4096,
    max_iters: 20,
    seed: 42
)

pq_train_f32(
    training_data,
    Int64(n_train),
    Int32(dimension),
    Int32(m),
    Int32(ks),
    nil,              // no coarse centroids
    nil,              // no assignments
    &cfg,
    &pq_codebooks,
    nil,              // norms not needed for encoding
    nil
)

let index = FlatPQIndex(
    pq_codebooks: pq_codebooks,
    dimension: dimension,
    m: m,
    ks: ks
)
```

### 3. Streaming PQ Training

**For Very Large Datasets**:
```swift
let chunk_size = 1_000_000
let num_chunks = 10

var chunk_pointers = [UnsafePointer<Float>?](repeating: nil, count: num_chunks)
var chunk_sizes = [Int64](repeating: 0, count: num_chunks)

// Load chunks (or memory-map from disk)
for i in 0..<num_chunks {
    let chunk = loadChunk(i)
    chunk_pointers[i] = UnsafePointer(chunk)
    chunk_sizes[i] = Int64(chunk_size)
}

var pq_codebooks = [Float](repeating: 0, count: m * ks * (dimension/m))
var cfg = PQTrainConfig(
    algo: .minibatch,
    batch_size: 8192,
    max_iters: 15
)

pq_train_streaming_f32(
    &chunk_pointers,
    &chunk_sizes,
    Int32(num_chunks),
    Int32(dimension),
    Int32(m),
    Int32(ks),
    nil,
    nil,
    &cfg,
    &pq_codebooks,
    nil,
    nil
)
```

### 4. Integration with Encoding (Kernel #20)

**After training, encode vectors**:
```c
// Train codebooks
float* codebooks = train_pq_codebooks(x, n, d, m, ks);

// Encode dataset
uint8_t* codes = malloc(n * m);  // m bytes per vector (for ks=256)
pq_encode_f32(x, n, d, m, ks, codebooks, codes);  // kernel #20

// Build index with encoded vectors
IVFPQIndex* index = ivfpq_create(coarse_centroids, codebooks, codes, ...);
```

---

## Coding Guidelines

### 1. API Design

**Consistent Naming**:
```c
// All PQ training functions prefixed with pq_train_
int pq_train_f32(...);
int pq_train_streaming_f32(...);
int pq_train_subspace(...);  // internal helper
```

**Error Handling**:
```c
int result = pq_train_f32(...);
if (result != 0) {
    switch (result) {
        case PQ_ERR_INVALID_DIM:
            fprintf(stderr, "Dimension %d not divisible by m=%d\n", d, m);
            break;
        case PQ_ERR_INSUFFICIENT_DATA:
            fprintf(stderr, "Need n >= ks, got n=%ld, ks=%d\n", n, ks);
            break;
        // ...
    }
    return result;
}
```

### 2. Memory Management

**Alignment Requirements**:
```c
// Ensure x is 64-byte aligned for SIMD
if ((uintptr_t)x & 63) {
    return PQ_ERR_ALIGNMENT;
}

// Allocate aligned codebooks
float* codebooks = aligned_alloc(64, m * ks * dsub * sizeof(float));
if (!codebooks) {
    return PQ_ERR_ALLOC_FAILED;
}
```

**Subspace View** (zero-copy):
```c
// No allocation, just pointer arithmetic
SubspaceView view = make_subspace_view(x, n, d, m, j);
// Use view.base + view.offset for access
```

### 3. Parallelism

**OpenMP over Subspaces**:
```c
#pragma omp parallel for num_threads(cfg->num_threads) schedule(dynamic)
for (int j = 0; j < m; j++) {
    SubspaceView x_sub = make_subspace_view(x, n, d, m, j);
    SubspaceView coarse_sub = coarse_centroids ?
        make_subspace_view(coarse_centroids, kc, d, m, j) : (SubspaceView){0};

    float* codebook_j = codebooks_out + j * ks * dsub;
    float* norms_j = centroid_norms_out ? (centroid_norms_out + j * ks) : NULL;

    train_subspace_codebook(x_sub, coarse_sub, assign, ks, cfg, codebook_j, norms_j);
}
```

### 4. Telemetry Integration

**Metrics Collection** (#46):
```c
void emit_pq_train_telemetry(const PQTrainStats* stats, const PQTrainConfig* cfg, int m) {
    telemetry_emit("pq.distortion", stats->distortion);
    telemetry_emit("pq.empties_repaired", stats->empties_repaired);
    telemetry_emit("pq.time_init_sec", stats->time_init_sec);
    telemetry_emit("pq.time_train_sec", stats->time_train_sec);
    telemetry_emit("pq.bytes_read", stats->bytes_read);

    for (int j = 0; j < m; j++) {
        char key[64];
        snprintf(key, sizeof(key), "pq.distortion_subspace_%d", j);
        telemetry_emit(key, stats->distortion_per_subspace[j]);

        snprintf(key, sizeof(key), "pq.iters_subspace_%d", j);
        telemetry_emit(key, stats->iters_per_subspace[j]);
    }
}
```

---

## Example Usage

### Example 1: Basic PQ Training

```c
#include "pq_train.h"
#include <stdio.h>

int main() {
    int64_t n = 500000;
    int d = 1024;
    int m = 8;
    int ks = 256;

    // Load data
    float* x = load_vectors("embeddings.bin", n, d);

    // Configuration
    PQTrainConfig cfg = {
        .algo = PQ_ALGO_LLOYD,
        .max_iters = 25,
        .tol = 1e-4,
        .batch_size = 0,  // not used for Lloyd
        .sample_n = 0,
        .seed = 42,
        .stream_id = 0,
        .empty_policy = EMPTY_POLICY_SPLIT,
        .precompute_x_norm2 = true,
        .compute_centroid_norms = true,
        .num_threads = 0  // auto
    };

    // Allocate outputs
    int dsub = d / m;
    float* codebooks = aligned_alloc(64, m * ks * dsub * sizeof(float));
    float* norms = malloc(m * ks * sizeof(float));
    PQTrainStats stats = {0};

    // Train
    int result = pq_train_f32(
        x, n, d, m, ks,
        NULL,           // no residual (direct PQ)
        NULL,
        &cfg,
        codebooks,
        norms,
        &stats
    );

    if (result != 0) {
        fprintf(stderr, "PQ training failed: %d\n", result);
        return 1;
    }

    // Report
    printf("PQ training completed\n");
    printf("Total distortion: %.6f\n", stats.distortion);
    printf("Training time: %.2f sec\n", stats.time_train_sec);

    for (int j = 0; j < m; j++) {
        printf("Subspace %d: distortion=%.6f, iters=%d\n",
               j, stats.distortion_per_subspace[j], stats.iters_per_subspace[j]);
    }

    // Save codebooks
    save_pq_codebooks("pq_codebooks.bin", codebooks, m, ks, dsub);
    save_pq_norms("pq_norms.bin", norms, m, ks);

    // Cleanup
    free(x);
    free(codebooks);
    free(norms);

    return 0;
}
```

### Example 2: Residual PQ for IVF

```c
#include "pq_train.h"
#include "ivf_train.h"

void train_ivf_pq_index(const char* data_file, const char* output_file) {
    int64_t n = 10000000;
    int d = 768;
    int kc = 10000;  // coarse centroids
    int m = 8;
    int ks = 256;

    // Load data
    float* x = load_vectors(data_file, n, d);

    // 1. Train IVF coarse quantizer
    printf("Training IVF coarse quantizer...\n");
    float* coarse_centroids = malloc(kc * d * sizeof(float));
    train_ivf_centroids(x, n, d, kc, coarse_centroids);

    // 2. Assign vectors to coarse centroids
    printf("Assigning vectors...\n");
    int32_t* assignments = malloc(n * sizeof(int32_t));
    assign_vectors(x, n, d, coarse_centroids, kc, assignments);

    // 3. Train PQ on residuals
    printf("Training PQ codebooks on residuals...\n");
    PQTrainConfig cfg = {
        .algo = PQ_ALGO_MINIBATCH,
        .max_iters = 20,
        .tol = 1e-4,
        .batch_size = 8192,
        .seed = 42,
        .empty_policy = EMPTY_POLICY_SPLIT,
        .precompute_x_norm2 = true,
        .compute_centroid_norms = true,
        .num_threads = 0
    };

    int dsub = d / m;
    float* pq_codebooks = aligned_alloc(64, m * ks * dsub * sizeof(float));
    float* pq_norms = malloc(m * ks * sizeof(float));
    PQTrainStats stats = {0};

    pq_train_f32(
        x, n, d, m, ks,
        coarse_centroids,  // residual mode
        assignments,
        &cfg,
        pq_codebooks,
        pq_norms,
        &stats
    );

    printf("PQ distortion: %.6f\n", stats.distortion);
    printf("Training time: %.2f sec\n", stats.time_train_sec);

    // 4. Save index parameters
    save_ivfpq_index(output_file, coarse_centroids, kc, pq_codebooks, pq_norms,
                     d, m, ks);

    // Cleanup
    free(x);
    free(coarse_centroids);
    free(assignments);
    free(pq_codebooks);
    free(pq_norms);
}
```

### Example 3: Swift Integration

```swift
import Foundation

func trainPQCodebooks(
    data: [Float],
    n: Int,
    d: Int,
    m: Int,
    ks: Int,
    residual: Bool = false,
    coarseCentroids: [Float]? = nil,
    assignments: [Int32]? = nil
) -> [Float] {
    precondition(d % m == 0, "d must be divisible by m")
    precondition(!residual || (coarseCentroids != nil && assignments != nil),
                 "Residual mode requires coarse centroids and assignments")

    let dsub = d / m
    var codebooks = [Float](repeating: 0, count: m * ks * dsub)
    var norms = [Float](repeating: 0, count: m * ks)
    var stats = PQTrainStats()

    var cfg = PQTrainConfig(
        algo: PQ_ALGO_LLOYD,
        max_iters: 25,
        tol: 1e-4,
        batch_size: 0,
        sample_n: 0,
        seed: UInt64.random(in: 0...UInt64.max),
        stream_id: 0,
        empty_policy: EMPTY_POLICY_SPLIT,
        precompute_x_norm2: true,
        compute_centroid_norms: true,
        num_threads: 0
    )

    let result = data.withUnsafeBufferPointer { dataPtr in
        codebooks.withUnsafeMutableBufferPointer { codePtr in
            norms.withUnsafeMutableBufferPointer { normPtr in
                if residual {
                    return coarseCentroids!.withUnsafeBufferPointer { coarsePtr in
                        assignments!.withUnsafeBufferPointer { assignPtr in
                            pq_train_f32(
                                dataPtr.baseAddress!,
                                Int64(n),
                                Int32(d),
                                Int32(m),
                                Int32(ks),
                                coarsePtr.baseAddress!,
                                assignPtr.baseAddress!,
                                &cfg,
                                codePtr.baseAddress!,
                                normPtr.baseAddress!,
                                &stats
                            )
                        }
                    }
                } else {
                    return pq_train_f32(
                        dataPtr.baseAddress!,
                        Int64(n),
                        Int32(d),
                        Int32(m),
                        Int32(ks),
                        nil,
                        nil,
                        &cfg,
                        codePtr.baseAddress!,
                        normPtr.baseAddress!,
                        &stats
                    )
                }
            }
        }
    }

    guard result == 0 else {
        fatalError("PQ training failed with error \(result)")
    }

    print("PQ training completed in \(stats.time_train_sec) sec")
    print("Distortion: \(stats.distortion)")

    return codebooks
}
```

---

## Summary

**Kernel #19** provides efficient Product Quantization codebook training for vector compression:

1. **Algorithm**: Independent k-means per subspace with Lloyd's or mini-batch variants
2. **Key Features**:
   - Zero-copy subspace views for memory efficiency
   - On-the-fly residual computation for IVF-PQ
   - Parallel training across subspaces
   - Precomputed norms for fast distance computation
3. **Performance**:
   - ~25 seconds to train 8 subspaces with ks=256 on 1M vectors (M2 Max)
   - Achieves 95-98% recall@10 with 256× compression (1024-dim → 8 bytes)
4. **Integration**:
   - Works with IVF training (kernel #12) for residual PQ
   - Outputs used by PQ encoding (kernel #20) and ADC (kernel #22)
   - Supports streaming for very large datasets
5. **Numerical Robustness**:
   - f64 accumulation for centroid updates
   - Deterministic tie-breaking
   - Independent subspace verification

**Dependencies**:
- Kernel #01 (L2 distance)
- Kernel #11 (k-means++ seeding)
- Kernel #12 (k-means training, for mini-batch mode)
- S2 RNG (Squares generator)

**Typical Use**: Train PQ codebooks with m=8, ks=256 on 1M residual vectors (d=1024) in ~25 seconds, achieving 5-10% distortion and 95-98% search recall.
