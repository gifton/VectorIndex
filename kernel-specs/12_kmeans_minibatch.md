# Kernel Specification #12: Mini-batch & Streaming K-means Centroid Update

**ID**: 12
**Priority**: MUST
**Role**: B/M (Batch / Maintenance)
**Status**: Specification

---

## Purpose

Train and maintain coarse quantizer centroids for IVF (Inverted File) indexes using efficient mini-batch or streaming k-means updates. Supports:
1. **Initial training** from scratch or warm-start with k-means++ seeds (#11)
2. **Online drift adaptation** via streaming EWMA updates for dynamic datasets
3. **Multi-epoch refinement** with early stopping based on inertia convergence
4. **Empty cluster repair** to maintain full utilization of all centroids

**Target Use Case**: Training IVF coarse quantizers with 1,000–100,000 centroids on datasets with 1M–1B vectors, where full Lloyd's k-means is prohibitively expensive.

---

## Mathematical Foundations

### 1. K-means Objective Function

The k-means problem seeks to partition a dataset **X** = {**x**₁, …, **x**ₙ} ⊂ ℝᵈ into k clusters by minimizing the **inertia** (within-cluster sum of squared distances):

```
φ(C) = Σᵢ₌₁ⁿ min_c ‖xᵢ - cⱼ‖²₂
     = Σᵢ₌₁ⁿ ‖xᵢ - c_assign(i)‖²₂
```

where **C** = {**c**₁, …, **c**ₖ} is the set of k centroids and assign(i) = argminⱼ ‖**x**ᵢ - **c**ⱼ‖²₂.

**Complexity of Full Lloyd's Algorithm**:
- Assignment step: O(nkd) — compute distance from each of n points to each of k centroids
- Update step: O(nd) — recompute centroids as cluster means
- Iterations: Typically 10–100 iterations until convergence
- Total: O(Tnkd) where T is the number of iterations

**Scalability Challenge**: For n=10⁹, k=10⁵, d=1024, a single iteration requires ~10²⁰ FLOPs, which is impractical.

### 2. Mini-batch K-means Algorithm

**Reference**: Sculley, D. (2010). "Web-scale k-means clustering." *WWW 2010*.

Mini-batch k-means approximates Lloyd's algorithm by:
1. Sampling a random subset **B** ⊂ **X** of size b << n (the "mini-batch")
2. Assigning points in **B** to nearest centroids
3. Updating centroids using only the points in **B**

**Centroid Update Rule** (per mini-batch):
```
For each centroid c:
  N_c = number of points in B assigned to c
  S_c = sum of points in B assigned to c

  c_new = (c_old · n_c + S_c) / (n_c + N_c)
```

where n_c is a per-centroid counter tracking the cumulative number of points assigned to c across all mini-batches.

**Simplified Update** (ignoring history):
```
c_new = S_c / max(N_c, 1)
```

This is the update used in this implementation, equivalent to Lloyd's k-means on each mini-batch independently.

**Convergence Guarantee** (Bottou & Bengio, 1995):
- Mini-batch k-means converges to a local minimum of φ(C) under appropriate conditions
- Convergence rate: O(1/√T) for T mini-batches
- Quality: Typically within 1–5% of full Lloyd's k-means with b ≥ 256

**Computational Savings**:
- Full Lloyd's: O(Tnkd)
- Mini-batch: O(T'bkd) where T' is number of mini-batches
- For b=1024, n=10⁹, savings factor ≈ n/b = 10⁶

### 3. Online EWMA K-means

For streaming data with concept drift, maintain centroids using **Exponentially Weighted Moving Average** (EWMA) updates:

```
For each incoming point x:
  1. c = argmin_j ‖x - c_j‖²
  2. c_c ← (1-η)·c_c + η·x
```

where η ∈ (0,1] is the **learning rate** (decay parameter).

**Adaptive Learning Rate**:
```
η_c = η₀ / (1 + t_c)
```

where t_c is the number of times centroid c has been updated.

**Interpretation**:
- η₀ = 1.0: each point completely replaces the centroid (equivalent to nearest-neighbor tracking)
- η₀ = 0.01: each point contributes 1% to the centroid, providing smoothing over ~100 points
- Adaptive η: initial updates have large impact, later updates refine

**Convergence** (stationary distributions):
- For stationary data, EWMA converges to cluster means with bias O(η)
- For drifting data, EWMA tracks the drift with lag proportional to 1/η

**Use Case**: Incremental index maintenance where new vectors arrive continuously and cluster centers shift over time.

### 4. Empty Cluster Repair

During training, some centroids may become **empty** (N_c = 0) due to:
- Poor initialization
- Absorption of small clusters by larger neighbors
- Unlucky mini-batch sampling

**Repair Strategies**:

**Strategy 1: Split Largest Cluster**
```
1. Find centroid c_max with largest count N_max
2. Find farthest point x_far in current batch from c_max
3. Set empty centroid c_empty = x_far
4. Optionally nudge c_max slightly away from x_far
```

**Strategy 2: Re-seed via Farthest Point**
```
1. Find point x_far in current batch that is farthest from all centroids
2. Set c_empty = x_far
```

**Determinism**: When multiple centroids are empty, repair in ascending order of centroid index.

---

## API Signatures

### 1. One-shot Mini-batch Training

```c
int kmeans_minibatch_f32(
    const float* x,                    // [n × d] input vectors (AoS or AoSoA)
    int64_t n,                         // number of vectors
    int d,                             // dimension
    int kc,                            // number of centroids
    const float* init_centroids,       // [kc × d] initial centroids (nullable)
    const KMeansMBConfig* cfg,         // configuration
    float* centroids_out,              // [kc × d] output centroids
    int32_t* assign_out,               // [n] assignments (nullable)
    KMeansMBStats* stats_out           // output statistics (nullable)
);
```

**Parameters**:
- `x`: Input vectors in AoS layout `[n][d]` or AoSoA layout (specified in `cfg->layout`)
- `n`: Number of input vectors (1 ≤ n ≤ 10¹²)
- `d`: Dimension (typical: 512, 768, 1024, 1536; general: multiple of 4)
- `kc`: Number of centroids (1 ≤ kc ≤ 10⁶, typical: 10³–10⁵)
- `init_centroids`: Initial centroid positions
  - If non-null: use provided centroids (e.g., from k-means++ seeding #11)
  - If null: initialize internally using k-means++ (#11) with `cfg->seed`
- `cfg`: Configuration (detailed below)
- `centroids_out`: Output buffer for final centroids, **must be preallocated** to kc×d floats
- `assign_out`: Optional output buffer for final assignments
  - If non-null: compute and store assign(i) for each i ∈ [0,n)
  - If null: skip assignment computation (saves O(nkd) work)
  - **Warning**: Computing assignments for n > 10⁸ can be very expensive
- `stats_out`: Optional statistics (convergence history, timing, etc.)

**Return Value**:
- `0`: Success
- `KMEANS_ERR_INVALID_DIM`: d < 1 or d > 32768
- `KMEANS_ERR_INVALID_K`: kc < 1 or kc > n
- `KMEANS_ERR_NULL_PTR`: required pointer is null
- `KMEANS_ERR_NO_CONVERGENCE`: failed to converge within max epochs (non-fatal, centroids_out still valid)

**Configuration** (`KMeansMBConfig`):
```c
typedef struct {
    KMeansAlgorithm algo;         // LloydMiniBatch or OnlineEWMA
    int64_t batch_size;           // mini-batch size (default: 1024)
    int epochs;                   // number of passes over data (default: 10)
    int64_t subsample_n;          // optional: max rows per epoch (0 = use all n)
    float tol;                    // convergence tolerance (default: 1e-4)
    float decay;                  // EWMA decay η for OnlineEWMA (default: 0.01)
    uint64_t seed;                // RNG seed for mini-batch sampling
    int stream_id;                // RNG stream ID (default: 0)
    int prefetch_distance;        // prefetch lookahead (default: 8)
    VectorLayout layout;          // AoS or AoSoA_R (default: AoS)
    int aosoa_register_block;     // R for AoSoA (ignored if layout=AoS)
    bool compute_assignments;     // fill assign_out at end (default: false)
    int num_threads;              // parallelism (0 = auto, default: 0)
} KMeansMBConfig;

typedef enum {
    KMEANS_ALGO_LLOYD_MINIBATCH,  // Standard mini-batch Lloyd
    KMEANS_ALGO_ONLINE_EWMA       // Streaming EWMA updates
} KMeansAlgorithm;

typedef enum {
    LAYOUT_AOS,      // Array of Structures: [n][d]
    LAYOUT_AOSOA_R   // Array of Structures of Arrays: [n/R][d][R]
} VectorLayout;
```

**Configuration Details**:

- **algo**: Algorithm variant
  - `LLOYD_MINIBATCH`: Standard mini-batch k-means (default)
    - Updates centroids using per-batch means: c = S_c / N_c
    - Computes inertia each epoch for convergence monitoring
    - Best for initial training with finite dataset
  - `ONLINE_EWMA`: Streaming EWMA updates
    - Updates centroids using: c ← (1-η)c + η·x
    - Suitable for online drift adaptation
    - Does not compute inertia (expensive for streaming)

- **batch_size**: Mini-batch size (b)
  - Typical: 256–8192
  - Smaller: faster iterations, more noise, slower convergence
  - Larger: slower iterations, less noise, faster convergence
  - Recommended: 1024 for d=1024, k=10,000

- **epochs**: Number of full passes over the data
  - Typical: 5–20 epochs
  - Each epoch processes ⌈n/batch_size⌉ mini-batches
  - Early stopping via `tol` may terminate before reaching max epochs

- **subsample_n**: Optional subsampling limit
  - If subsample_n > 0 and subsample_n < n:
    - Randomly sample subsample_n points per epoch (without replacement)
    - Process in mini-batches of size batch_size
  - Use case: very large n (e.g., 10⁹), only need to see subset per epoch
  - Default: 0 (use all n points)

- **tol**: Relative inertia improvement threshold for early stopping
  - Stop if: (φ_prev - φ_curr) / φ_prev < tol
  - Typical: 1e-4 (0.01% improvement)
  - Only applies to LLOYD_MINIBATCH (ONLINE_EWMA does not track inertia)

- **decay**: Learning rate η for ONLINE_EWMA
  - Range: (0, 1]
  - Typical: 0.001–0.1
  - Smaller decay: more smoothing, slower adaptation to drift
  - Larger decay: less smoothing, faster adaptation
  - Ignored if algo = LLOYD_MINIBATCH

- **seed, stream_id**: RNG control for deterministic mini-batch sampling
  - Uses S2 generator (Squares RNG) as specified in project conventions
  - seed: 64-bit seed
  - stream_id: independent stream for parallel training

- **prefetch_distance**: Software prefetch lookahead
  - Prefetch x[i + prefetch_distance] while processing x[i]
  - Typical: 8–16
  - Coordination with #49 prefetch helpers

- **layout**: Vector memory layout
  - `AoS`: Standard row-major `[n][d]`, natural stride d
  - `AoSoA_R`: Blocked layout `[n/R][d][R]` for better cache/SIMD efficiency
  - If AoSoA, must match blocking used in #04 score_block kernel
  - Default: AoS (simpler, more general)

- **aosoa_register_block**: Register block size R for AoSoA
  - Typical: R=4 (SIMD4<Float>) or R=8 (dual accumulator)
  - Only used if layout=AoSoA_R
  - Must divide n evenly (or handle remainder separately)

- **compute_assignments**: Whether to compute final assignments
  - If true: after convergence, assign each point and fill assign_out
  - If false: skip final assignment pass (saves O(nkd) work)
  - **Warning**: For n=10⁹, k=10⁵, d=1024, assignment pass takes ~10–60 minutes
  - Recommendation: set false for very large n, compute assignments lazily as needed

- **num_threads**: Thread pool size
  - 0: auto-detect (use all available cores)
  - >0: use specified number of threads
  - Mini-batches are parallelized over points within each batch

**Statistics** (`KMeansMBStats`):
```c
typedef struct {
    int epochs_completed;          // actual number of epochs (may be < cfg->epochs due to early stop)
    int64_t batches_processed;     // total mini-batches processed
    int64_t rows_seen;             // total point-to-centroid assignments computed
    int64_t empties_repaired;      // number of empty cluster repairs
    double* inertia_per_epoch;     // [epochs_completed] inertia at end of each epoch (nullable)
    double final_inertia;          // inertia after final epoch
    double time_init_sec;          // time for initialization (seeding if needed)
    double time_training_sec;      // time for mini-batch iterations
    double time_assignment_sec;    // time for final assignment pass (0 if not computed)
    int64_t bytes_read;            // total bytes read from x and centroids
} KMeansMBStats;
```

### 2. Streaming / Stateful API

For online drift adaptation, maintain a persistent k-means state that can be updated incrementally.

**Initialization**:
```c
int kmeans_state_init(
    int d,                         // dimension
    int kc,                        // number of centroids
    const float* init_centroids,   // [kc × d] initial centroids (required)
    float decay,                   // EWMA decay η (0 < decay ≤ 1)
    KMeansState** state_out        // output state handle
);
```

**Incremental Update**:
```c
int kmeans_state_update_chunk(
    KMeansState* state,            // state handle
    const float* x_chunk,          // [m × d] new vectors
    int64_t m,                     // chunk size
    const KMeansUpdateOpts* opts,  // update options (nullable)
    float* centroids_out           // [kc × d] updated centroids (nullable)
);
```

**Finalization**:
```c
int kmeans_state_finalize(
    KMeansState* state,            // state handle
    float* centroids_out           // [kc × d] final centroids
);

void kmeans_state_destroy(KMeansState* state);
```

**State** (`KMeansState`):
```c
typedef struct KMeansState {
    int d;                         // dimension
    int kc;                        // number of centroids
    float* centroids;              // [kc × d] current centroids
    int64_t* counts;               // [kc] cumulative assignment counts per centroid
    float decay;                   // EWMA decay η
    int64_t total_updates;         // total number of points processed
} KMeansState;
```

**Update Options** (`KMeansUpdateOpts`):
```c
typedef struct {
    float decay;                   // override state decay (0 = use state default)
    bool normalize_centroids;      // re-center centroids to mean after chunk (default: false)
    int prefetch_distance;         // prefetch lookahead (default: 8)
    bool adaptive_decay;           // use η_c = decay / (1 + t_c) (default: false)
} KMeansUpdateOpts;
```

**Workflow**:
```c
// Initialize state with k-means++ seeds
float* seeds = kmeans_plusplus_seed(...);
KMeansState* state;
kmeans_state_init(d, kc, seeds, decay=0.01, &state);

// Stream updates
while (new_data_available) {
    float* chunk = get_next_chunk(&m);  // m vectors
    kmeans_state_update_chunk(state, chunk, m, NULL, NULL);
}

// Retrieve final centroids
float* final_centroids = malloc(kc * d * sizeof(float));
kmeans_state_finalize(state, final_centroids);
kmeans_state_destroy(state);
```

---

## Algorithm Details

### 1. Lloyd Mini-batch K-means (LLOYD_MINIBATCH)

**High-level Pseudocode**:
```
kmeans_minibatch(x[n][d], kc, init_centroids, cfg):
    // Initialization
    if init_centroids == NULL:
        C ← kmeans_plusplus_seed(x, n, d, kc, cfg.seed)  // kernel #11
    else:
        C ← copy(init_centroids)

    φ_prev ← ∞

    // Multi-epoch training
    for epoch in 1..cfg.epochs:
        // Optionally subsample
        if cfg.subsample_n > 0 and cfg.subsample_n < n:
            indices ← random_sample(n, cfg.subsample_n, cfg.seed, epoch)
            n_epoch ← cfg.subsample_n
        else:
            indices ← [0, 1, ..., n-1]
            n_epoch ← n

        // Shuffle indices for mini-batch sampling
        shuffle(indices, cfg.seed, epoch)

        // Process mini-batches
        num_batches ← ⌈n_epoch / cfg.batch_size⌉
        for batch_idx in 0..num_batches-1:
            // Extract mini-batch
            start ← batch_idx * cfg.batch_size
            end ← min(start + cfg.batch_size, n_epoch)
            B ← x[indices[start:end]]
            batch_size_actual ← end - start

            // Assignment + accumulation (parallel over points in B)
            S[kc][d] ← zeros  // per-centroid sums
            N[kc] ← zeros     // per-centroid counts

            parallel for i in 0..batch_size_actual-1:
                c_best ← assign(B[i], C, kc)  // argmin_c ‖B[i] - C[c]‖²
                atomic_add(N[c_best], 1)
                atomic_add_vector(S[c_best], B[i])  // S[c_best] += B[i]

            // Update centroids (serial, deterministic order)
            for c in 0..kc-1:
                if N[c] > 0:
                    C[c] ← S[c] / N[c]

            // Empty cluster repair
            empties ← [c : N[c] == 0]
            if len(empties) > 0:
                repair_empty_clusters(C, empties, B, N)

        // Convergence check
        φ_curr ← compute_inertia(x, indices[0:n_epoch], C, kc)
        improvement ← (φ_prev - φ_curr) / φ_prev

        if improvement < cfg.tol and epoch > 1:
            break  // early stop

        φ_prev ← φ_curr

    // Optional: compute final assignments
    if cfg.compute_assignments:
        parallel for i in 0..n-1:
            assign_out[i] ← assign(x[i], C, kc)

    centroids_out ← C
```

**Key Steps**:

**A. Assignment** (`assign(x, C, kc)`):
```
assign(x, C, kc) → int:
    c_best ← 0
    dist_best ← ‖x - C[0]‖²

    for c in 1..kc-1:
        dist ← ‖x - C[c]‖²
        if dist < dist_best or (dist == dist_best and c < c_best):
            dist_best ← dist
            c_best ← c

    return c_best
```

**Optimization**: Reuse kernel #01 (L2 squared distance):
```c
float dist_sq = l2_squared_distance_f32(x, C[c], d);
```

**Early-exit**: Track running minimum and skip centroids that cannot improve:
```c
// Precompute centroid norms if beneficial
float centroid_norms[kc];
for (int c = 0; c < kc; c++) {
    centroid_norms[c] = dot_product(C[c], C[c], d);
}

// Assignment with early exit
float query_norm = dot_product(x, x, d);
int c_best = 0;
float dist_best = query_norm + centroid_norms[0] - 2*dot_product(x, C[0], d);

for (int c = 1; c < kc; c++) {
    // Early exit: even with perfect alignment, cannot beat dist_best
    float max_ip = sqrtf(query_norm * centroid_norms[c]);
    float min_dist = query_norm + centroid_norms[c] - 2*max_ip;
    if (min_dist >= dist_best) continue;

    float ip = dot_product(x, C[c], d);
    float dist = query_norm + centroid_norms[c] - 2*ip;
    if (dist < dist_best) {
        dist_best = dist;
        c_best = c;
    }
}
```

**Note**: Early-exit is beneficial when kc is large (e.g., kc > 10,000) and queries are clustered.

**B. Empty Cluster Repair**:
```
repair_empty_clusters(C, empties, B, N):
    for c_empty in empties (sorted ascending):
        // Strategy 1: Split largest cluster
        c_max ← argmax_c N[c]

        // Find farthest point in B from c_max
        x_far ← B[0]
        dist_far ← ‖B[0] - C[c_max]‖²
        for i in 1..len(B)-1:
            dist ← ‖B[i] - C[c_max]‖²
            if dist > dist_far:
                dist_far ← dist
                x_far ← B[i]

        // Assign to empty centroid
        C[c_empty] ← x_far

        // Optional: nudge c_max away
        // C[c_max] ← C[c_max] + ε * (C[c_max] - x_far) / ‖C[c_max] - x_far‖
```

**Alternative Strategy 2**: Farthest point from all centroids:
```
repair_empty_clusters_v2(C, empties, B, N):
    for c_empty in empties (sorted ascending):
        // Find point in B farthest from all centroids
        x_far ← B[0]
        min_dist_far ← min_c ‖B[0] - C[c]‖²

        for i in 1..len(B)-1:
            min_dist ← min_c ‖B[i] - C[c]‖²
            if min_dist > min_dist_far:
                min_dist_far ← min_dist
                x_far ← B[i]

        C[c_empty] ← x_far
```

**C. Inertia Computation**:
```
compute_inertia(x, indices, C, kc) → float:
    φ ← 0.0 (f64 accumulator)

    parallel for idx in indices:
        c_best ← assign(x[idx], C, kc)
        dist_sq ← ‖x[idx] - C[c_best]‖²
        atomic_add(φ, dist_sq)

    return φ
```

**Optimization**: Compute inertia on a subsample (e.g., 10,000 random points) if n is very large:
```c
int64_t sample_size = min(n, 10000);
double inertia = compute_inertia_sampled(x, n, C, kc, sample_size, seed);
```

### 2. Online EWMA K-means (ONLINE_EWMA)

**Pseudocode**:
```
kmeans_online_ewma(x[n][d], kc, init_centroids, cfg):
    C ← init_centroids
    t[kc] ← zeros  // visit counts per centroid

    for i in 0..n-1:
        // Assignment
        c_best ← assign(x[i], C, kc)

        // EWMA update
        if cfg.adaptive_decay:
            η ← cfg.decay / (1 + t[c_best])
        else:
            η ← cfg.decay

        C[c_best] ← (1 - η) * C[c_best] + η * x[i]
        t[c_best] ← t[c_best] + 1

    centroids_out ← C
```

**Vectorized Update**:
```c
// C[c] ← (1-η)·C[c] + η·x
for (int j = 0; j < d; j += 4) {
    SIMD4<Float> c_vec = SIMD4<Float>(C[c] + j);
    SIMD4<Float> x_vec = SIMD4<Float>(x + j);
    SIMD4<Float> updated = (1 - eta) * c_vec + eta * x_vec;
    updated.store(C[c] + j);
}
```

**Parallel Update** (when processing batches):
- **Challenge**: Multiple threads may update the same centroid simultaneously
- **Solution 1** (per-thread shadow copy):
  ```
  Each thread maintains shadow_C[kc][d]
  After processing chunk:
      Merge shadow copies using weighted average based on per-thread counts
  ```
- **Solution 2** (atomic updates):
  ```
  Use atomic floating-point addition (if available) or lock-free CAS loop
  ```
- **Preferred**: Shadow copy + periodic merge (every 1000 points or per batch)

**Adaptive Decay**:
```
η_c = η₀ / (1 + t_c)
```

**Behavior**:
- First update (t_c=0): η = η₀ (e.g., 0.01)
- After 99 updates (t_c=99): η = 0.01/100 = 0.0001 (much smaller)
- Converges to stable centroid with diminishing updates

**Use Case**: Stationary data where initial estimates are noisy but later updates are refinements.

**Non-adaptive Decay** (constant η):
- Better for drifting data where recent points are more important
- Centroid "forgets" old data at exponential rate

### 3. Streaming State Updates

**Stateful Workflow**:
```
state_init(d, kc, init_centroids, decay):
    state.C ← copy(init_centroids)
    state.counts ← zeros[kc]
    state.decay ← decay
    state.total_updates ← 0
    return state

state_update_chunk(state, x_chunk[m][d], opts):
    for i in 0..m-1:
        c_best ← assign(x_chunk[i], state.C, kc)

        if opts.adaptive_decay:
            η ← opts.decay / (1 + state.counts[c_best])
        else:
            η ← opts.decay

        state.C[c_best] ← (1-η) * state.C[c_best] + η * x_chunk[i]
        state.counts[c_best] ← state.counts[c_best] + 1

    state.total_updates ← state.total_updates + m

    if opts.normalize_centroids:
        mean ← compute_mean(state.C, kc)
        for c in 0..kc-1:
            state.C[c] ← state.C[c] - mean

state_finalize(state):
    return copy(state.C)
```

**Normalization** (optional):
- After processing a chunk, re-center all centroids to have zero mean
- Prevents centroid drift in arbitrary directions
- Only necessary if data distribution is shifting

---

## Implementation Strategies

### 1. Vectorization & Tiling

**Centroid Tiling** (fit in L1 cache):
```
L1 cache size: 128 KB (typical Apple Silicon P-core)
Centroid size: d×4 bytes = 1024×4 = 4 KB
Tile size: 128 KB / 4 KB = 32 centroids per tile
```

**Algorithm** (assignment with tiling):
```c
const int TILE_SIZE = 32;
int num_tiles = (kc + TILE_SIZE - 1) / TILE_SIZE;

for (int i = 0; i < batch_size; i++) {
    const float* x = batch + i*d;
    float dist_best = INFINITY;
    int c_best = 0;

    for (int tile = 0; tile < num_tiles; tile++) {
        int c_start = tile * TILE_SIZE;
        int c_end = min(c_start + TILE_SIZE, kc);

        // Prefetch next tile
        if (tile + 1 < num_tiles) {
            prefetch_range(C + (c_start + TILE_SIZE)*d, TILE_SIZE*d*sizeof(float));
        }

        for (int c = c_start; c < c_end; c++) {
            float dist = l2_squared_f32(x, C + c*d, d);
            if (dist < dist_best) {
                dist_best = dist;
                c_best = c;
            }
        }
    }

    assignments[i] = c_best;
}
```

**SIMD Distance Computation**:
```c
float l2_squared_f32_simd(const float* a, const float* b, int d) {
    SIMD4<Float> acc0 = 0, acc1 = 0;

    for (int i = 0; i < d; i += 8) {
        SIMD4<Float> a0(a + i), a1(a + i + 4);
        SIMD4<Float> b0(b + i), b1(b + i + 4);
        SIMD4<Float> diff0 = a0 - b0;
        SIMD4<Float> diff1 = a1 - b1;
        acc0 += diff0 * diff0;
        acc1 += diff1 * diff1;
    }

    SIMD4<Float> total = acc0 + acc1;
    return total[0] + total[1] + total[2] + total[3];
}
```

**Batch Tiling** (fit in L2/L3 cache):
```
L2 cache size: 16 MB (shared across P-cores)
Batch of vectors: batch_size × d × 4 bytes = 1024 × 1024 × 4 = 4 MB
Fits comfortably in L2 along with centroid tile
```

### 2. Parallelism

**Parallel Assignment + Accumulation**:
```c
// Per-thread accumulators
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    float S_local[kc][d];  // thread-local sums
    int N_local[kc];       // thread-local counts
    memset(S_local, 0, kc*d*sizeof(float));
    memset(N_local, 0, kc*sizeof(int));

    // Process assigned rows
    #pragma omp for schedule(static)
    for (int i = 0; i < batch_size; i++) {
        int c_best = assign(batch[i], C, kc);
        N_local[c_best]++;
        for (int j = 0; j < d; j++) {
            S_local[c_best][j] += batch[i][j];
        }
    }

    // Reduction (deterministic order)
    #pragma omp critical
    {
        for (int c = 0; c < kc; c++) {
            N_global[c] += N_local[c];
            for (int j = 0; j < d; j++) {
                S_global[c][j] += S_local[c][j];
            }
        }
    }
}
```

**Avoiding False Sharing**:
```c
// Align per-centroid accumulators to cache line (64 bytes)
struct alignas(64) CentroidAccumulator {
    float sum[MAX_DIM];
    int count;
    char pad[64 - (MAX_DIM*4 + 4) % 64];
};

CentroidAccumulator accumulators[kc];
```

**Lock-free Updates** (for small updates):
```c
// Atomic increment (counts)
__atomic_fetch_add(&N[c], 1, __ATOMIC_RELAXED);

// Atomic float add (sums) - requires hardware support or CAS loop
void atomic_add_float(float* addr, float val) {
    uint32_t old_bits, new_bits;
    do {
        old_bits = __atomic_load_n((uint32_t*)addr, __ATOMIC_RELAXED);
        float old_val = *(float*)&old_bits;
        float new_val = old_val + val;
        new_bits = *(uint32_t*)&new_val;
    } while (!__atomic_compare_exchange_n((uint32_t*)addr, &old_bits, new_bits,
                                           false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}
```

**Recommendation**: Use per-thread accumulators for batch processing (avoids contention), atomic updates for true streaming (lower latency per point).

### 3. Memory Layout

**AoS Layout** (standard):
```
x[n][d]: Natural row-major layout
Access pattern for assignment: sequential over d for each row
Cache efficiency: Good if d fits in cache line (d ≤ 16 for f32)
```

**AoSoA Layout** (blocked):
```
x[n/R][d][R]: Blocked layout for SIMD
Example (R=4, d=8):
    x[0] = [x0[0..3], x1[0..3], x2[0..3], x3[0..3],  // first 4 dims of 4 vectors
            x0[4..7], x1[4..7], x2[4..7], x3[4..7]]  // next 4 dims of 4 vectors

Access pattern:
    for (int i = 0; i < n; i += R) {
        for (int j = 0; j < d; j += 4) {
            SIMD4<Float> x0 = load(x[i/R][j][0..3]);
            SIMD4<Float> c0 = load(C[c][j..j+3]);
            // Compute distance for x0
        }
    }
```

**Benefit**: Better SIMD utilization, reduced gather/scatter overhead.

**Cost**: More complex indexing, requires R to divide n.

**Recommendation**: Use AoS for general case, AoSoA for performance-critical high-throughput scenarios where R=4 or R=8.

### 4. Prefetching

**Software Prefetch** (x data):
```c
const int PREFETCH_DIST = 8;

for (int i = 0; i < batch_size; i++) {
    // Prefetch future row
    if (i + PREFETCH_DIST < batch_size) {
        __builtin_prefetch(batch + (i + PREFETCH_DIST)*d, 0, 3);  // read, high temporal locality
    }

    // Process current row
    int c_best = assign(batch + i*d, C, kc);
    // ...
}
```

**Centroid Prefetch** (during assignment):
```c
for (int c = 0; c < kc; c++) {
    // Prefetch next centroid
    if (c + 1 < kc) {
        __builtin_prefetch(C + (c+1)*d, 0, 3);
    }

    float dist = l2_squared_f32(x, C + c*d, d);
    // ...
}
```

**Integration with #49**: Use project-wide prefetch helpers for consistency.

---

## Performance Characteristics

### 1. Computational Complexity

**Per Mini-batch** (size b):
- **Assignment**: O(bkd) FLOPs
  - b points × k centroids × (d multiplications + d additions + 1 comparison)
  - FLOPs: ~2bkd
- **Update**: O(bd + kd) FLOPs
  - Accumulation: b points × d dimensions × 1 addition = bd FLOPs
  - Division: k centroids × d dimensions = kd FLOPs
- **Total per batch**: ~2bkd FLOPs

**Full Training** (T batches, E epochs):
- Total batches: T = E × ⌈n/b⌉
- Total FLOPs: ~2Ebkd × ⌈n/b⌉ ≈ 2Enkd
- Example: n=10⁹, k=10⁴, d=1024, E=10 → 2×10¹⁰×10⁴×10³ = 2×10²⁰ FLOPs

**FLOP/s on Apple M2 Max** (using SIMD, 8 threads):
- Peak f32: 3.5 TFLOP/s (P-cores)
- Achieved (assignment): ~1.5 TFLOP/s (43% of peak, memory-bound)
- Time per batch (b=1024, k=10⁴, d=1024): 2×1024×10⁴×10³ / 1.5×10¹² ≈ 14 ms
- Time per epoch (n=10⁹): 14 ms × ⌈10⁹/1024⌉ ≈ 13,700 sec ≈ 3.8 hours
- **Speedup vs full Lloyd's**: ~10⁶× (due to b/n ratio)

### 2. Memory Bandwidth

**Reads per Assignment**:
- Query vector: d×4 bytes
- All centroids: k×d×4 bytes
- Total: (1+k)×d×4 bytes per point

**Writes per Batch**:
- Centroid updates: k×d×4 bytes (per batch, amortized over b points)
- Per-point writes: ~k×d×4/b bytes per point

**Total Bandwidth per Point**: ~(1+k)×d×4 bytes (dominated by centroid reads)

**Example** (k=10⁴, d=1024):
- Bandwidth per point: 10⁴×1024×4 = 40 MB
- Points per second (BW=200 GB/s): 200×10⁹ / (40×10⁶) = 5,000 points/s
- **Memory-bound**: Even with infinite compute, limited to 5,000 points/s

**Optimization**: Centroid tiling to fit in cache reduces effective k for bandwidth calculation.

**Tile-based Bandwidth** (tile size = 32):
```
Centroid tile: 32×1024×4 = 128 KB (fits in L1)
Reads from DRAM per point: d×4 bytes (query only)
Reads from L1: 32×d×4 bytes per tile
Number of tiles: ⌈k/32⌉

Effective bandwidth: d×4 + ⌈k/32⌉×32×d×4 / (L1 latency factor)
≈ d×4 bytes (if L1 is fast enough)
```

**Achievable Throughput** (with tiling):
- Bandwidth per point: 1024×4 = 4 KB
- Points per second: 200×10⁹ / (4×10³) = 50 million points/s
- **10× improvement** from tiling

### 3. Performance Targets (Apple M2 Max, 8 threads)

| Configuration | Throughput | Latency per Batch | Notes |
|---------------|------------|-------------------|-------|
| k=1,000, d=512, b=1024 | 80K pts/s | 13 ms | Small k, fits in L1 |
| k=10,000, d=1024, b=1024 | 12K pts/s | 85 ms | Medium k, tiling required |
| k=100,000, d=1024, b=1024 | 1.5K pts/s | 680 ms | Large k, memory-bound |
| k=10,000, d=1536, b=512 | 8K pts/s | 64 ms | High dimension |

**Scaling**:
- **b (batch size)**: Larger b → better amortization of centroid reads, diminishing returns beyond b=2048
- **k (num centroids)**: Linear degradation, mitigated by tiling
- **d (dimension)**: Linear degradation in both compute and memory

**Comparison to Full Lloyd's**:
- Full Lloyd's (n=10⁹, k=10⁴, d=1024): ~1 iteration/hour
- Mini-batch (b=1024): ~1 epoch/4 hours, typically 10 epochs → 40 hours
- **Practical difference**: Mini-batch converges in 10–20 epochs, full Lloyd's may need 50–100 iterations
- **Net speedup**: 5–10× in wall-clock time

### 4. Convergence Rate

**Theoretical** (Bottou & Bengio, 1995):
- Mini-batch k-means converges at rate O(1/√T) where T is number of mini-batches
- Expected error: E[φ_mb - φ_opt] ≤ C/√T for some constant C

**Empirical**:
- With b=1024, typically 90% of final inertia reduction achieved in first 5 epochs
- Remaining 10% requires another 10–15 epochs
- Larger b → faster convergence (more information per update), but slower per-batch time

**Comparison to Online EWMA**:
- EWMA does not converge to global optimum (biased estimator)
- Suitable for tracking drift, not for minimizing inertia
- Bias: O(η) where η is decay parameter

---

## Numerical Considerations

### 1. Floating-point Precision

**Accumulation** (centroid sums):
```c
// BAD: f32 accumulation loses precision for large batches
float S[d];
for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < d; j++) {
        S[j] += batch[i][j];  // catastrophic cancellation if batch_size > 10^6
    }
}

// GOOD: f64 accumulation, convert to f32 at the end
double S[d];
for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < d; j++) {
        S[j] += (double)batch[i][j];
    }
}
float C_new[d];
for (int j = 0; j < d; j++) {
    C_new[j] = (float)(S[j] / N);
}
```

**Relative Error** (f32 vs f64 accumulation):
- f32 mantissa: 23 bits → ε ≈ 2⁻²³ ≈ 1.2×10⁻⁷
- Accumulated error over n additions: O(n·ε)
- For n=10⁶, error ≈ 0.1 (10% relative error!)
- f64 mantissa: 52 bits → ε ≈ 2⁻⁵² ≈ 2.2×10⁻¹⁶
- Accumulated error over n=10⁹: O(10⁹·10⁻¹⁶) = 10⁻⁷ (negligible)

**Recommendation**: Always use f64 for accumulation, f32 for storage.

### 2. Division by Zero

**Empty Clusters** (N[c] = 0):
```c
// BAD: undefined behavior
C[c][j] = S[c][j] / N[c];  // division by zero!

// GOOD: check before division
if (N[c] > 0) {
    for (int j = 0; j < d; j++) {
        C[c][j] = S[c][j] / N[c];
    }
} else {
    // C[c] remains unchanged (will be repaired later)
}
```

**Alternative**: Use `max(N[c], 1)` to avoid branch:
```c
int divisor = (N[c] > 0) ? N[c] : 1;
for (int j = 0; j < d; j++) {
    C[c][j] = S[c][j] / divisor;
}
// If N[c]==0, C[c] = S[c]/1 = 0 (invalid, but won't crash; repaired later)
```

### 3. Deterministic Tie-breaking

**Distance Ties**:
```c
// BAD: non-deterministic (depends on floating-point rounding)
if (dist < dist_best) {
    dist_best = dist;
    c_best = c;
}

// GOOD: deterministic (prefer smaller index on tie)
if (dist < dist_best || (dist == dist_best && c < c_best)) {
    dist_best = dist;
    c_best = c;
}
```

**Floating-point Equality**:
- `dist == dist_best` is exact when both computed identically (deterministic)
- If different code paths compute distances, may have rounding differences
- **Best practice**: Use exact equality for determinism, accept rare non-uniqueness

### 4. EWMA Numerical Stability

**Standard Update**:
```c
C[c] = (1 - eta) * C[c] + eta * x;
```

**Issue**: For very small η (e.g., η=10⁻⁶), `1-η` is close to 1, leading to loss of precision.

**Improved Update** (Kahan summation):
```c
// Maintain error compensation
float err[d] = {0};  // per-centroid error

for (int j = 0; j < d; j++) {
    float delta = eta * (x[j] - C[c][j]);
    float y = delta - err[j];
    float temp = C[c][j] + y;
    err[j] = (temp - C[c][j]) - y;
    C[c][j] = temp;
}
```

**Benefit**: Reduces accumulated rounding error for small η and many updates.

**Alternative**: Use f64 for EWMA centroids if precision is critical.

### 5. Reproducibility

**RNG Seeding**:
```c
// Deterministic mini-batch sampling
uint64_t batch_seed = hash(cfg->seed, epoch, batch_idx);
rng_init(&rng, batch_seed, cfg->stream_id);
```

**Thread Order**:
- Per-thread accumulators are reduced in deterministic order (centroid index 0→kc-1)
- Thread assignment to rows is deterministic via `#pragma omp for schedule(static)`

**Floating-point Reduction**:
- Sum of f32 values is not associative: (a+b)+c ≠ a+(b+c) due to rounding
- **Solution**: Reduce in fixed order to ensure bitwise identical results across runs

**Guarantee**: With identical seed, num_threads, and input data, produce bitwise identical centroids.

---

## Correctness Testing

### 1. Convergence Tests

**Test 1: Monotonic Inertia Decrease**
```swift
func testMonotonicInertiaDecrease() {
    let x = generateRandomVectors(n: 100_000, d: 128)
    var stats = KMeansMBStats()

    kmeans_minibatch_f32(x, n, d, kc: 100, nil, cfg, centroids, nil, &stats)

    for i in 1..<stats.epochs_completed {
        assert(stats.inertia_per_epoch[i] <= stats.inertia_per_epoch[i-1],
               "Inertia increased from epoch \(i-1) to \(i)")
    }
}
```

**Test 2: Convergence to Reference**
```swift
func testConvergenceToSklearn() {
    let x = generateSyntheticClusters(n: 10_000, k: 10, d: 64, separation: 5.0)

    // Reference: scikit-learn MiniBatchKMeans
    let sklearn_centroids = runSklearnMiniBatchKMeans(x, k: 10, batch_size: 1024, epochs: 20)

    // Implementation
    var cfg = KMeansMBConfig(batch_size: 1024, epochs: 20, seed: 42)
    var centroids = [Float](repeating: 0, count: 10*64)
    kmeans_minibatch_f32(x, 10_000, 64, 10, nil, &cfg, &centroids, nil, nil)

    // Compare: assign both centroid sets, measure Adjusted Rand Index (ARI)
    let ari = computeARI(x, centroids, sklearn_centroids)
    assert(ari > 0.95, "ARI = \(ari), expected > 0.95")
}
```

### 2. Determinism Tests

**Test 3: Single-threaded Reproducibility**
```swift
func testSingleThreadDeterminism() {
    let x = generateRandomVectors(n: 50_000, d: 256)
    let seed: UInt64 = 12345

    var centroids1 = [Float](repeating: 0, count: 100*256)
    var centroids2 = [Float](repeating: 0, count: 100*256)

    var cfg = KMeansMBConfig(seed: seed, num_threads: 1)
    kmeans_minibatch_f32(x, 50_000, 256, 100, nil, &cfg, &centroids1, nil, nil)
    kmeans_minibatch_f32(x, 50_000, 256, 100, nil, &cfg, &centroids2, nil, nil)

    for i in 0..<centroids1.count {
        assert(centroids1[i] == centroids2[i], "Mismatch at index \(i)")
    }
}
```

**Test 4: Multi-threaded Reproducibility**
```swift
func testMultiThreadDeterminism() {
    let x = generateRandomVectors(n: 100_000, d: 512)
    let seed: UInt64 = 67890

    var centroids_t1 = [Float](repeating: 0, count: 200*512)
    var centroids_t8 = [Float](repeating: 0, count: 200*512)

    var cfg_t1 = KMeansMBConfig(seed: seed, num_threads: 1)
    var cfg_t8 = KMeansMBConfig(seed: seed, num_threads: 8)

    kmeans_minibatch_f32(x, 100_000, 512, 200, nil, &cfg_t1, &centroids_t1, nil, nil)
    kmeans_minibatch_f32(x, 100_000, 512, 200, nil, &cfg_t8, &centroids_t8, nil, nil)

    // Allow small numerical differences due to f32 reduction order
    for i in 0..<centroids_t1.count {
        let diff = abs(centroids_t1[i] - centroids_t8[i])
        assert(diff < 1e-5, "Centroid \(i) differs by \(diff)")
    }
}
```

### 3. Online EWMA Tests

**Test 5: Stationary Data Convergence**
```swift
func testEWMAStationaryConvergence() {
    // Generate data from 10 fixed clusters
    let (x, true_centroids) = generateClusteredData(n: 100_000, k: 10, d: 128)

    // Initialize with k-means++
    let seeds = kMeansPlusPlusSeed(x, k: 10, d: 128)
    var state: KMeansState?
    kmeans_state_init(128, 10, seeds, decay: 0.01, &state)

    // Stream all points
    let chunk_size = 1000
    for i in stride(from: 0, to: 100_000, by: chunk_size) {
        let chunk = Array(x[i*128..<min((i+chunk_size)*128, 100_000*128)])
        kmeans_state_update_chunk(state, chunk, Int64(chunk.count/128), nil, nil)
    }

    var final_centroids = [Float](repeating: 0, count: 10*128)
    kmeans_state_finalize(state, &final_centroids)

    // Compare to true centroids
    let dist = centroidSetDistance(final_centroids, true_centroids, k: 10, d: 128)
    assert(dist < 0.1, "EWMA centroids deviate by \(dist) from true centroids")

    kmeans_state_destroy(state)
}
```

**Test 6: Drift Tracking**
```swift
func testEWMADriftTracking() {
    // Generate data with linearly drifting means
    var state: KMeansState?
    let init_centroids = generateRandomCentroids(k: 5, d: 64)
    kmeans_state_init(64, 5, init_centroids, decay: 0.05, &state)

    let num_chunks = 100
    for t in 0..<num_chunks {
        // Generate data with mean drifting: mean = [t/100, t/100, ...]
        let drift = Float(t) / Float(num_chunks)
        let chunk = generateDriftedData(k: 5, d: 64, m: 500, drift: drift)

        kmeans_state_update_chunk(state, chunk, 500, nil, nil)
    }

    var final_centroids = [Float](repeating: 0, count: 5*64)
    kmeans_state_finalize(state, &final_centroids)

    // Check that centroids tracked the drift (final mean ≈ 1.0)
    let mean_value = final_centroids.reduce(0, +) / Float(5*64)
    assert(abs(mean_value - 1.0) < 0.2, "Failed to track drift: mean = \(mean_value)")

    kmeans_state_destroy(state)
}
```

### 4. Empty Cluster Tests

**Test 7: Empty Cluster Repair**
```swift
func testEmptyClusterRepair() {
    // Create data with only 5 actual clusters, request 10 centroids
    let x = generateClusteredData(n: 10_000, k: 5, d: 64)

    var centroids = [Float](repeating: 0, count: 10*64)
    var assignments = [Int32](repeating: 0, count: 10_000)
    var cfg = KMeansMBConfig(compute_assignments: true, seed: 999)
    var stats = KMeansMBStats()

    kmeans_minibatch_f32(x, 10_000, 64, 10, nil, &cfg, &centroids, &assignments, &stats)

    // Count how many centroids are actually used
    var centroid_counts = [Int](repeating: 0, count: 10)
    for i in 0..<10_000 {
        centroid_counts[Int(assignments[i])] += 1
    }

    let num_empty = centroid_counts.filter { $0 == 0 }.count

    // At most 5 empty (since there are 5 true clusters), but repair should fill some
    assert(num_empty <= 5, "Too many empty clusters: \(num_empty)")
    assert(stats.empties_repaired > 0, "No repairs occurred despite potential empties")
}
```

### 5. Performance Regression Tests

**Test 8: Throughput Benchmark**
```swift
func testThroughputBenchmark() {
    let n = 100_000
    let d = 1024
    let k = 10_000
    let x = generateRandomVectors(n: n, d: d)

    var centroids = [Float](repeating: 0, count: k*d)
    var cfg = KMeansMBConfig(batch_size: 1024, epochs: 1, num_threads: 8)
    var stats = KMeansMBStats()

    kmeans_minibatch_f32(x, Int64(n), Int32(d), Int32(k), nil, &cfg, &centroids, nil, &stats)

    let points_per_sec = Double(n) / stats.time_training_sec
    let expected_min = 10_000.0  // 10K points/sec minimum

    assert(points_per_sec > expected_min,
           "Throughput \(points_per_sec) pts/s below threshold \(expected_min)")
}
```

---

## Integration Patterns

### 1. IVF Training Pipeline

**Full IVF Training** (from scratch):
```swift
// 1. Generate training sample (10% of dataset, at least 100k points)
let training_sample = sampleDataset(dataset, fraction: 0.1, min_count: 100_000)

// 2. k-means++ seeding
let seeds = kMeansPlusPlusSeed(
    training_sample,
    k: num_centroids,
    d: dimension,
    seed: 42
)

// 3. Mini-batch k-means refinement
var cfg = KMeansMBConfig(
    algo: .lloydMiniBatch,
    batch_size: 2048,
    epochs: 20,
    tol: 1e-4,
    seed: 42,
    num_threads: 0  // auto
)

var centroids = [Float](repeating: 0, count: num_centroids * dimension)
var stats = KMeansMBStats()

kmeans_minibatch_f32(
    training_sample,
    Int64(training_sample.count / dimension),
    Int32(dimension),
    Int32(num_centroids),
    seeds,
    &cfg,
    &centroids,
    nil,
    &stats
)

print("Converged in \(stats.epochs_completed) epochs")
print("Final inertia: \(stats.final_inertia)")

// 4. Build IVF index with learned centroids
let index = IVFIndex(centroids: centroids, num_centroids: num_centroids, dimension: dimension)
index.add(dataset)
```

### 2. Online Index Maintenance

**Incremental Centroid Updates** (for drifting data):
```swift
// Initialize IVF index with current centroids
var state: KMeansState?
kmeans_state_init(
    Int32(dimension),
    Int32(num_centroids),
    current_centroids,
    decay: 0.001,  // slow adaptation
    &state
)

// As new vectors arrive, update centroids periodically
var update_buffer: [Float] = []
let update_threshold = 10_000

while let new_vectors = streamNewVectors() {
    // Add to index
    index.add(new_vectors)

    // Buffer for centroid update
    update_buffer.append(contentsOf: new_vectors)

    // Update centroids when buffer is full
    if update_buffer.count >= update_threshold * dimension {
        var opts = KMeansUpdateOpts(adaptive_decay: true)
        kmeans_state_update_chunk(
            state,
            update_buffer,
            Int64(update_buffer.count / dimension),
            &opts,
            nil
        )

        // Retrieve updated centroids
        var updated_centroids = [Float](repeating: 0, count: num_centroids * dimension)
        kmeans_state_finalize(state, &updated_centroids)

        // Rebuild index coarse quantizer (expensive, do periodically)
        if should_rebuild_index {
            index.updateCoarseQuantizer(updated_centroids)
        }

        update_buffer.removeAll(keepingCapacity: true)
    }
}

kmeans_state_destroy(state)
```

### 3. Hierarchical Clustering

**Two-level IVF** (coarse + fine quantizers):
```swift
// Level 1: Coarse quantizer (1000 centroids)
let coarse_centroids = trainKMeans(
    data: training_sample,
    k: 1000,
    batch_size: 2048,
    epochs: 20
)

// Assign training data to coarse clusters
var assignments = [Int32](repeating: 0, count: training_sample.count / dimension)
assignVectorsToCentroids(training_sample, coarse_centroids, &assignments)

// Level 2: Fine quantizers (100 centroids per coarse cluster)
var fine_centroids: [[Float]] = []

for coarse_id in 0..<1000 {
    // Extract vectors assigned to this coarse cluster
    let cluster_vectors = extractCluster(training_sample, assignments, coarse_id)

    if cluster_vectors.isEmpty { continue }

    // Train fine quantizer for this cluster
    let fine = trainKMeans(
        data: cluster_vectors,
        k: 100,
        batch_size: 512,
        epochs: 15
    )

    fine_centroids.append(fine)
}

// Build two-level IVF index
let index = HierarchicalIVFIndex(
    coarse: coarse_centroids,
    fine: fine_centroids,
    dimension: dimension
)
```

### 4. Integration with Distance Kernels

**Reuse L2 Distance Kernel** (#01):
```c
// In assignment step, call optimized L2 kernel
float assign_nearest_centroid(const float* x, const float* C, int kc, int d) {
    int c_best = 0;
    float dist_best = l2_squared_f32(x, C, d);  // kernel #01

    for (int c = 1; c < kc; c++) {
        float dist = l2_squared_f32(x, C + c*d, d);  // kernel #01
        if (dist < dist_best) {
            dist_best = dist;
            c_best = c;
        }
    }

    return c_best;
}
```

**Batch Assignment** (via #04 score_block):
```c
// Score all points in batch against all centroids
score_block_l2_f32(
    batch,              // [batch_size × d]
    C,                  // [kc × d]
    batch_size,
    kc,
    d,
    scores_out          // [batch_size × kc]
);

// Find argmin for each point
for (int i = 0; i < batch_size; i++) {
    int c_best = 0;
    float dist_best = scores_out[i*kc];
    for (int c = 1; c < kc; c++) {
        if (scores_out[i*kc + c] < dist_best) {
            dist_best = scores_out[i*kc + c];
            c_best = c;
        }
    }
    assignments[i] = c_best;
}
```

---

## Coding Guidelines

### 1. API Design

**Consistent Naming**:
```c
// All mini-batch k-means functions prefixed with kmeans_
int kmeans_minibatch_f32(...);
int kmeans_state_init(...);
int kmeans_state_update_chunk(...);
int kmeans_state_finalize(...);
void kmeans_state_destroy(...);
```

**Error Handling**:
```c
// Return 0 on success, negative error code on failure
int result = kmeans_minibatch_f32(...);
if (result != 0) {
    fprintf(stderr, "K-means failed: %s\n", kmeans_error_string(result));
    return result;
}
```

**Nullability**:
```c
// Use _Nullable annotation for optional parameters
int kmeans_minibatch_f32(
    const float* x,
    int64_t n,
    int d,
    int kc,
    const float* _Nullable init_centroids,       // optional
    const KMeansMBConfig* cfg,
    float* centroids_out,
    int32_t* _Nullable assign_out,               // optional
    KMeansMBStats* _Nullable stats_out           // optional
);
```

### 2. Memory Management

**Buffer Allocation** (caller-allocated outputs):
```c
// Caller allocates output buffers
float* centroids_out = malloc(kc * d * sizeof(float));
int32_t* assign_out = malloc(n * sizeof(int32_t));
KMeansMBStats stats;

kmeans_minibatch_f32(x, n, d, kc, NULL, &cfg, centroids_out, assign_out, &stats);

// Caller frees
free(centroids_out);
free(assign_out);
if (stats.inertia_per_epoch) free(stats.inertia_per_epoch);
```

**Stateful API** (library-managed state):
```c
KMeansState* state = NULL;
kmeans_state_init(d, kc, init_centroids, decay, &state);

// ... updates ...

kmeans_state_destroy(state);  // library frees internal buffers
```

**Alignment**:
```c
// Align buffers to 64-byte cache lines for SIMD
float* centroids = aligned_alloc(64, kc * d * sizeof(float));
if (!centroids) {
    return KMEANS_ERR_ALLOC_FAILED;
}
```

### 3. Threading

**OpenMP Pragmas**:
```c
#pragma omp parallel num_threads(cfg->num_threads)
{
    // Per-thread accumulators
    float* S_local = calloc(kc * d, sizeof(float));
    int* N_local = calloc(kc, sizeof(int));

    #pragma omp for schedule(static)
    for (int i = 0; i < batch_size; i++) {
        int c = assign(batch + i*d, C, kc, d);
        N_local[c]++;
        for (int j = 0; j < d; j++) {
            S_local[c*d + j] += batch[i*d + j];
        }
    }

    // Reduction
    #pragma omp critical
    {
        for (int c = 0; c < kc; c++) {
            N_global[c] += N_local[c];
            for (int j = 0; j < d; j++) {
                S_global[c*d + j] += S_local[c*d + j];
            }
        }
    }

    free(S_local);
    free(N_local);
}
```

**GCD (Swift)**:
```swift
DispatchQueue.concurrentPerform(iterations: batchSize) { i in
    let c = assign(batch + i*d, centroids, kc, d)
    // Thread-safe accumulation
    lock.lock()
    N[c] += 1
    for j in 0..<d {
        S[c*d + j] += batch[i*d + j]
    }
    lock.unlock()
}
```

### 4. SIMD Patterns

**Centroid Update with SIMD**:
```c
// C[c] ← S[c] / N[c]
int N_c = N[c];
if (N_c > 0) {
    float inv_N = 1.0f / N_c;
    for (int j = 0; j < d; j += 4) {
        SIMD4<Float> s = SIMD4<Float>(S + c*d + j);
        SIMD4<Float> c_new = s * inv_N;
        c_new.store(C + c*d + j);
    }
}
```

**EWMA Update**:
```c
// C[c] ← (1-η)·C[c] + η·x
for (int j = 0; j < d; j += 4) {
    SIMD4<Float> c_vec = SIMD4<Float>(C + c*d + j);
    SIMD4<Float> x_vec = SIMD4<Float>(x + j);
    SIMD4<Float> updated = (1 - eta) * c_vec + eta * x_vec;
    updated.store(C + c*d + j);
}
```

### 5. Telemetry Integration

**Metrics Collection** (#46):
```c
void emit_kmeans_telemetry(const KMeansMBStats* stats, const KMeansMBConfig* cfg) {
    telemetry_emit("kmeans.epochs_completed", stats->epochs_completed);
    telemetry_emit("kmeans.batches_processed", stats->batches_processed);
    telemetry_emit("kmeans.rows_seen", stats->rows_seen);
    telemetry_emit("kmeans.empties_repaired", stats->empties_repaired);
    telemetry_emit("kmeans.final_inertia", stats->final_inertia);
    telemetry_emit("kmeans.time_training_sec", stats->time_training_sec);
    telemetry_emit("kmeans.bytes_read", stats->bytes_read);
    telemetry_emit("kmeans.throughput_pts_per_sec", stats->rows_seen / stats->time_training_sec);
}
```

---

## Example Usage

### Example 1: Basic Training

```c
#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
    // Parameters
    int64_t n = 1000000;
    int d = 1024;
    int kc = 10000;

    // Load data
    float* x = load_vectors("data.bin", n, d);

    // Configuration
    KMeansMBConfig cfg = {
        .algo = KMEANS_ALGO_LLOYD_MINIBATCH,
        .batch_size = 2048,
        .epochs = 20,
        .subsample_n = 0,
        .tol = 1e-4,
        .decay = 0,
        .seed = 42,
        .stream_id = 0,
        .prefetch_distance = 8,
        .layout = LAYOUT_AOS,
        .aosoa_register_block = 0,
        .compute_assignments = false,
        .num_threads = 0
    };

    // Allocate outputs
    float* centroids = aligned_alloc(64, kc * d * sizeof(float));
    KMeansMBStats stats = {0};

    // Train
    int result = kmeans_minibatch_f32(
        x, n, d, kc,
        NULL,           // auto-initialize with k-means++
        &cfg,
        centroids,
        NULL,           // skip assignments
        &stats
    );

    if (result != 0) {
        fprintf(stderr, "K-means failed: %d\n", result);
        return 1;
    }

    // Report
    printf("Converged in %d epochs\n", stats.epochs_completed);
    printf("Final inertia: %.2f\n", stats.final_inertia);
    printf("Training time: %.2f sec\n", stats.time_training_sec);
    printf("Throughput: %.0f points/sec\n", stats.rows_seen / stats.time_training_sec);

    // Save centroids
    save_vectors("centroids.bin", centroids, kc, d);

    // Cleanup
    free(x);
    free(centroids);
    if (stats.inertia_per_epoch) free(stats.inertia_per_epoch);

    return 0;
}
```

### Example 2: Streaming Updates

```c
#include "kmeans.h"

void online_clustering(const char* stream_file) {
    int d = 768;
    int kc = 1000;

    // Initialize with pre-trained centroids
    float* init_centroids = load_vectors("init_centroids.bin", kc, d);

    KMeansState* state = NULL;
    kmeans_state_init(d, kc, init_centroids, /*decay=*/0.01, &state);

    // Stream processing
    FILE* fp = fopen(stream_file, "rb");
    float chunk[1000 * 768];  // 1000 vectors per chunk

    while (!feof(fp)) {
        size_t count = fread(chunk, sizeof(float), 1000*768, fp);
        int64_t m = count / 768;

        if (m > 0) {
            KMeansUpdateOpts opts = {
                .decay = 0,               // use state default
                .normalize_centroids = false,
                .prefetch_distance = 8,
                .adaptive_decay = true
            };

            kmeans_state_update_chunk(state, chunk, m, &opts, NULL);
        }
    }

    fclose(fp);

    // Finalize and save
    float* final_centroids = malloc(kc * d * sizeof(float));
    kmeans_state_finalize(state, final_centroids);
    save_vectors("updated_centroids.bin", final_centroids, kc, d);

    // Cleanup
    kmeans_state_destroy(state);
    free(init_centroids);
    free(final_centroids);
}
```

### Example 3: Swift Integration

```swift
import Foundation
import Accelerate

func trainIVFCentroids(
    data: [Float],
    n: Int,
    d: Int,
    numCentroids: Int
) -> [Float] {
    // Configuration
    var cfg = KMeansMBConfig(
        algo: KMEANS_ALGO_LLOYD_MINIBATCH,
        batch_size: 1024,
        epochs: 15,
        subsample_n: 0,
        tol: 1e-4,
        decay: 0,
        seed: UInt64.random(in: 0...UInt64.max),
        stream_id: 0,
        prefetch_distance: 8,
        layout: LAYOUT_AOS,
        aosoa_register_block: 0,
        compute_assignments: false,
        num_threads: 0
    )

    // Allocate outputs
    var centroids = [Float](repeating: 0, count: numCentroids * d)
    var stats = KMeansMBStats()

    // Call C function
    let result = data.withUnsafeBufferPointer { dataPtr in
        centroids.withUnsafeMutableBufferPointer { centroidsPtr in
            kmeans_minibatch_f32(
                dataPtr.baseAddress!,
                Int64(n),
                Int32(d),
                Int32(numCentroids),
                nil,
                &cfg,
                centroidsPtr.baseAddress!,
                nil,
                &stats
            )
        }
    }

    guard result == 0 else {
        fatalError("K-means failed with error \(result)")
    }

    print("K-means completed in \(stats.epochs_completed) epochs")
    print("Final inertia: \(stats.final_inertia)")
    print("Time: \(stats.time_training_sec) sec")

    return centroids
}
```

---

## Summary

**Kernel #12** provides efficient mini-batch and streaming k-means algorithms for training and maintaining IVF coarse quantizers:

1. **Two Algorithm Modes**:
   - Lloyd mini-batch: standard mini-batch k-means for initial training
   - Online EWMA: streaming updates for drift adaptation

2. **Key Features**:
   - Automatic k-means++ initialization (#11) if no seeds provided
   - Empty cluster repair to maintain full centroid utilization
   - Early stopping based on inertia convergence
   - Parallel processing with deterministic reductions
   - Stateful API for incremental updates

3. **Performance**:
   - Throughput: 1.5K–80K points/sec (depending on k, d, batch_size)
   - Speedup vs full Lloyd's: 10⁶× per iteration (due to mini-batch sampling)
   - Wall-clock speedup: 5–10× (accounting for convergence rates)

4. **Integration**:
   - Consumes k-means++ seeds from #11
   - Uses L2 distance kernel #01 for assignment
   - Outputs centroids for IVF index construction (#29)
   - Supports online index maintenance for drifting datasets

5. **Numerical Robustness**:
   - f64 accumulation for centroid sums
   - Deterministic tie-breaking and reproducible RNG
   - Kahan summation for EWMA with small decay

**Dependencies**:
- Kernel #01 (L2 distance)
- Kernel #11 (k-means++ seeding)
- S2 RNG (Squares generator)
- #48 vector layouts (AoS/AoSoA)
- #49 prefetch helpers
- #46 telemetry

**Typical Use**: Train 10,000 centroids on 10⁸ vectors (d=1024) in ~2 hours on Apple M2 Max, achieving 95%+ of full Lloyd's k-means quality.
