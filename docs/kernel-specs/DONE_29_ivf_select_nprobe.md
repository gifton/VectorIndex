# Kernel Specification #29: IVF List Selection (nprobe routing)

**ID**: 29
**Priority**: MUST
**Role**: Q (Query)
**Status**: Specification

---

## Purpose

Select which IVF (Inverted File) lists to probe during query by finding the nprobe nearest coarse centroids. This is the critical routing decision that determines the recall/performance trade-off in IVF-based approximate search.

**Key Benefits**:
1. **Recall control**: More probed lists → higher recall but slower search
2. **Performance**: Prune search space from n vectors to nprobe×(n/kc) vectors
3. **Flexibility**: Supports L2, inner product, and cosine distance metrics

**Typical Use Case**: For an IVF index with 10K centroids, select nprobe=16 nearest centroids in ~50 μs, reducing search space by 625× while maintaining 95%+ recall.

---

## Mathematical Foundations

### 1. IVF Structure

**Inverted File Index**: Partition n database vectors into kc clusters (lists) using k-means:
```
Coarse centroids: C = {c₁, c₂, ..., c_kc}
IVF lists: L₁, L₂, ..., L_kc where L_j = {x : assign(x) = j}
```

**Coarse Assignment**:
```
assign(x) = argmin_j dist(x, c_j)
```

### 2. Query Routing

**Problem**: Given query **q**, which lists should we search?

**Optimal**: Probe lists that contain the true nearest neighbors.

**Approximate**: Probe nprobe lists with centroids nearest to **q**:
```
probe_lists = argmin_{|S|=nprobe} Σ_{j∈S} dist(q, c_j)
              = top-nprobe centroids by distance to q
```

**Distance Metrics**:
- **L2**: dist(q, c) = ‖q - c‖² (minimize)
- **Inner Product**: dist(q, c) = ⟨q, c⟩ (maximize)
- **Cosine**: dist(q, c) = ⟨q, c⟩ / (‖q‖·‖c‖) (maximize)

### 3. Recall vs Performance Trade-off

**Recall**: Probability that true nearest neighbors are in probed lists.

**Empirical Rule** (for well-distributed data):
```
Recall ≈ nprobe / kc for nprobe << kc
```

**Examples** (kc=10,000):
- nprobe=1: ~0.01% recall (very fast, terrible accuracy)
- nprobe=10: ~40% recall (fast, low accuracy)
- nprobe=50: ~80% recall (moderate speed, good accuracy)
- nprobe=100: ~90% recall (slower, high accuracy)
- nprobe=500: ~95% recall (slow, very high accuracy)

**Search Cost**:
```
Without IVF: O(n×d)
With IVF: O(kc×d + nprobe×(n/kc)×m)
         = O(kc×d + nprobe×n×m/kc)
```

where m is PQ code length (much smaller than d).

**Speedup**: For kc=10K, nprobe=50, m=8, d=1024:
```
Speedup ≈ n×d / (kc×d + nprobe×n×m/kc)
        ≈ n×1024 / (10000×1024 + 50×n×8/10000)
        ≈ 200× (for large n)
```

### 4. Beam Search Expansion (Optional)

**Motivation**: For high-dimensional data, nearest centroids by distance may not contain true neighbors due to curse of dimensionality.

**Graph-Based Search**: Build k-NN graph over centroids, use beam search to explore neighborhood:

```
1. Start: top beam_width centroids by distance
2. Expand: For each centroid in beam, visit unvisited neighbors from k-NN graph
3. Score: Compute distances to newly discovered centroids
4. Maintain: Keep best beam_width candidates in frontier
5. Repeat: Until nprobe unique centroids collected
```

**Benefit**: Can achieve higher recall with same nprobe by exploring centroid neighborhoods.

**Cost**: Additional graph traversal and scoring overhead.

---

## API Signatures

### 1. Standard nprobe Selection

```c
void ivf_select_nprobe(
    const float* q,                    // [d] query vector
    int d,                             // dimension
    const float* centroids,            // [kc × d] coarse centroids
    int kc,                            // number of centroids
    Metric metric,                     // distance metric (L2/IP/Cosine)
    int nprobe,                        // number of lists to probe
    const IVFSelectOpts* opts,         // options (nullable)
    int32_t* list_ids,                 // [nprobe] output list IDs
    float* list_scores                 // [nprobe] output scores (nullable)
);
```

**Parameters**:

- `q`: Query vector, length d
- `d`: Dimension
- `centroids`: Coarse centroids from IVF training, layout `[kc][d]`
- `kc`: Number of coarse centroids (IVF lists)
  - Typical: 100 (small), 1,000 (medium), 10,000 (large), 100,000 (very large)
- `metric`: Distance metric
  - `METRIC_L2`: L2 squared distance (minimize)
  - `METRIC_IP`: Inner product (maximize)
  - `METRIC_COSINE`: Cosine similarity (maximize)
- `nprobe`: Number of lists to probe
  - Typical: 1-500
  - Trade-off: higher nprobe → better recall, slower search
- `opts`: Optional configuration (nullable)
- `list_ids`: Output list IDs, **must be preallocated** to nprobe×4 bytes
  - Sorted by distance: list_ids[0] is nearest, list_ids[nprobe-1] is farthest
- `list_scores`: Optional output scores (nullable)
  - If non-null: fill with distances/similarities for each selected list

### 2. Beam Search Expansion

```c
void ivf_select_beam(
    const float* q,                    // [d] query vector
    int d,                             // dimension
    const float* centroids,            // [kc × d] coarse centroids
    int kc,                            // number of centroids
    const int32_t* knn_graph,          // [kc × deg] k-NN graph (nullable)
    int deg,                           // degree of k-NN graph
    Metric metric,                     // distance metric
    int nprobe,                        // number of lists to probe
    int beam_width,                    // beam width for search
    const IVFSelectOpts* opts,         // options (nullable)
    int32_t* list_ids,                 // [nprobe] output list IDs
    float* list_scores                 // [nprobe] output scores (nullable)
);
```

**Additional Parameters**:
- `knn_graph`: k-NN graph over centroids
  - Layout: `knn_graph[i*deg : (i+1)*deg]` = neighbor IDs for centroid i
  - If null: fall back to standard nprobe selection
- `deg`: Degree of k-NN graph (typical: 16-64)
- `beam_width`: Number of candidates to maintain during beam search
  - Typical: 2×nprobe to 4×nprobe
  - Larger beam_width → better recall, more overhead

### 3. Batch Query Processing

```c
void ivf_select_nprobe_batch(
    const float* Q,                    // [b × d] batch of queries
    int b,                             // batch size
    int d,                             // dimension
    const float* centroids,            // [kc × d] coarse centroids
    int kc,                            // number of centroids
    Metric metric,                     // distance metric
    int nprobe,                        // number of lists to probe
    const IVFSelectOpts* opts,         // options (nullable)
    int32_t* list_ids,                 // [b × nprobe] output list IDs
    float* list_scores                 // [b × nprobe] output scores (nullable)
);
```

**Batch Parameters**:
- `Q`: Batch of b queries, layout `[b][d]`
- `b`: Batch size
- `list_ids`: Output for all queries, layout `[b][nprobe]`
  - `list_ids[i*nprobe : (i+1)*nprobe]` = list IDs for query i

### 4. Options

```c
typedef struct {
    const uint64_t* disabled_lists;    // bitset of disabled lists (nullable)
    const float* centroid_norms;       // [kc] precomputed norms (nullable)
    const float* centroid_inv_norms;   // [kc] inverse norms for cosine (nullable)
    bool use_dot_trick;                // use dot-product trick for L2 (default: auto)
    int prefetch_distance;             // prefetch lookahead (default: 8)
    bool strict_fp;                    // strict floating-point mode (default: false)
    int num_threads;                   // parallelism for single query (0 = auto, default: 0)
} IVFSelectOpts;

typedef enum {
    METRIC_L2,          // L2 squared distance (minimize)
    METRIC_IP,          // Inner product (maximize)
    METRIC_COSINE       // Cosine similarity (maximize)
} Metric;
```

**Options Explained**:

- **disabled_lists**: Bitset of disabled lists (for incremental updates or filtering)
  - Bit i set → exclude centroid i from selection
  - Size: ⌈kc/64⌉ uint64_t words
  - Check: `(disabled_lists[i >> 6] & (1ULL << (i & 63))) != 0`

- **centroid_norms**: Precomputed squared norms ‖cᵢ‖²
  - Used for dot-product trick in L2 distance
  - Used as denominator in cosine similarity

- **centroid_inv_norms**: Precomputed inverse norms 1/‖cᵢ‖
  - Used for cosine similarity: cos(q, c) = ⟨q, c⟩ / (‖q‖·‖c‖) = ⟨q, c⟩ × inv_q_norm × inv_c_norm
  - Can be stored as f16 for 2× memory savings

- **use_dot_trick**: Force dot-product trick for L2 (requires centroid_norms)

- **prefetch_distance**: Software prefetch lookahead

- **strict_fp**: Strict floating-point reproducibility

- **num_threads**: Parallelism for single query
  - Typically used only for very large kc (kc > 100,000)
  - Partition centroids across threads, merge top-k results

---

## Algorithm Details

### 1. Standard nprobe Selection (Brute-Force)

**Pseudocode**:
```
ivf_select_nprobe(q, d, centroids, kc, metric, nprobe, opts, list_ids, list_scores):
    // 1. Compute distances to all centroids (kernel #04)
    scores = allocate(kc)
    score_block(q, centroids, kc, d, metric, scores, opts)

    // 2. Apply disabled list mask (if provided)
    if opts.disabled_lists:
        for i in 0..kc-1:
            if is_disabled(opts.disabled_lists, i):
                scores[i] = INFINITY (for L2) or -INFINITY (for IP/Cosine)

    // 3. Select top-nprobe (kernel #05)
    if metric == L2:
        partial_topk_min(scores, kc, nprobe, list_ids, list_scores)  // min-heap
    else:
        partial_topk_max(scores, kc, nprobe, list_ids, list_scores)  // max-heap

    // 4. Sort results by score (best-first)
    sort(list_ids, list_scores, nprobe)
```

**Implementation**:
```c
void ivf_select_nprobe(const float* q, int d, const float* centroids, int kc,
                        Metric metric, int nprobe, const IVFSelectOpts* opts,
                        int32_t* list_ids, float* list_scores) {
    // Allocate temporary score array
    float* scores = malloc(kc * sizeof(float));

    // Compute distances to all centroids using score_block kernel (#04)
    score_block_f32(q, centroids, kc, d, metric, scores, opts);

    // Apply disabled list mask
    if (opts && opts->disabled_lists) {
        float mask_value = (metric == METRIC_L2) ? INFINITY : -INFINITY;
        for (int i = 0; i < kc; i++) {
            int word_idx = i >> 6;
            int bit_idx = i & 63;
            if (opts->disabled_lists[word_idx] & (1ULL << bit_idx)) {
                scores[i] = mask_value;
            }
        }
    }

    // Select top-nprobe using partial top-k kernel (#05)
    if (metric == METRIC_L2) {
        partial_topk_min_f32(scores, kc, nprobe, list_ids, list_scores);
    } else {
        partial_topk_max_f32(scores, kc, nprobe, list_ids, list_scores);
    }

    // Sort by score (best-first)
    if (metric == METRIC_L2) {
        sort_ascending(list_ids, list_scores, nprobe);
    } else {
        sort_descending(list_ids, list_scores, nprobe);
    }

    free(scores);
}
```

### 2. Beam Search Expansion

**Pseudocode**:
```
ivf_select_beam(q, d, centroids, kc, knn_graph, deg, metric, nprobe, beam_width, opts, list_ids, list_scores):
    // 1. Initialize: score all centroids to get starting beam
    scores = allocate(kc)
    score_block(q, centroids, kc, d, metric, scores, opts)

    // 2. Select initial beam (top beam_width)
    beam = allocate(beam_width)
    visited = allocate_bitset(kc)
    partial_topk(scores, kc, beam_width, beam, NULL)

    for i in beam:
        mark_visited(visited, i)

    // 3. Beam search expansion
    result_set = priority_queue(capacity: nprobe)
    while size(result_set) < nprobe:
        // Collect unvisited neighbors
        neighbors = []
        for c in beam:
            for neighbor_id in knn_graph[c*deg : (c+1)*deg]:
                if not is_visited(visited, neighbor_id):
                    neighbors.append(neighbor_id)
                    mark_visited(visited, neighbor_id)

        if len(neighbors) == 0:
            break  // No more candidates

        // Score new candidates
        neighbor_scores = allocate(len(neighbors))
        for i, neighbor_id in enumerate(neighbors):
            neighbor_scores[i] = compute_distance(q, centroids[neighbor_id], d, metric)

        // Add to result set
        for i, neighbor_id in enumerate(neighbors):
            result_set.push(neighbor_id, neighbor_scores[i])

        // Update beam: keep best beam_width from result_set
        beam = result_set.top(beam_width)

    // 4. Extract top nprobe
    for i in 0..nprobe-1:
        (list_ids[i], list_scores[i]) = result_set.pop()
```

### 3. Batch Processing

**Parallel over Queries**:
```c
void ivf_select_nprobe_batch(const float* Q, int b, int d, const float* centroids,
                              int kc, Metric metric, int nprobe, const IVFSelectOpts* opts,
                              int32_t* list_ids, float* list_scores) {
    #pragma omp parallel for
    for (int i = 0; i < b; i++) {
        const float* q = Q + i*d;
        int32_t* ids_out = list_ids + i*nprobe;
        float* scores_out = list_scores ? (list_scores + i*nprobe) : NULL;

        ivf_select_nprobe(q, d, centroids, kc, metric, nprobe, opts,
                          ids_out, scores_out);
    }
}
```

### 4. Deterministic Tie-Breaking

**Issue**: When multiple centroids have identical distances, selection must be deterministic.

**Solution**: Break ties by centroid ID (prefer smaller ID):

```c
bool compare_l2(float dist_a, int id_a, float dist_b, int id_b) {
    if (dist_a < dist_b) return true;
    if (dist_a > dist_b) return false;
    return id_a < id_b;  // Tie-breaker
}

bool compare_ip(float dist_a, int id_a, float dist_b, int id_b) {
    if (dist_a > dist_b) return true;  // Maximize
    if (dist_a < dist_b) return false;
    return id_a < id_b;  // Tie-breaker
}
```

---

## Implementation Strategies

### 1. Metric-Specific Optimizations

**L2 Distance**:
```c
// Use kernel #01 (L2 squared distance)
l2sqr_f32_block(q, centroids, kc, d, scores, centroid_norms, q_norm, opts);
```

**Inner Product**:
```c
// Use kernel #02 (inner product)
ip_f32_block(q, centroids, kc, d, scores);
```

**Cosine Similarity**:
```c
// Compute inner products
ip_f32_block(q, centroids, kc, d, scores);

// Divide by norms
float q_norm = compute_norm(q, d);
for (int i = 0; i < kc; i++) {
    scores[i] = scores[i] / (q_norm * centroid_norms[i]);
}

// Alternative: use precomputed inverse norms
for (int i = 0; i < kc; i++) {
    scores[i] = scores[i] * q_inv_norm * centroid_inv_norms[i];
}
```

### 2. Heap Selection

**Min-Heap** (for L2):
```c
// Use kernel #05 (partial top-k min-heap)
partial_topk_min_f32(scores, kc, nprobe, list_ids, list_scores);
```

**Max-Heap** (for IP/Cosine):
```c
// Use kernel #05 (partial top-k max-heap)
partial_topk_max_f32(scores, kc, nprobe, list_ids, list_scores);
```

### 3. Large kc Parallelism

**Partition Centroids**:
```c
const int BLOCK_SIZE = 1024;
int num_blocks = (kc + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Allocate per-thread heaps
float** block_scores = malloc(num_blocks * sizeof(float*));
int32_t** block_ids = malloc(num_blocks * sizeof(int32_t*));

#pragma omp parallel for
for (int block = 0; block < num_blocks; block++) {
    int start = block * BLOCK_SIZE;
    int end = min(start + BLOCK_SIZE, kc);
    int block_kc = end - start;

    // Compute scores for this block
    block_scores[block] = malloc(block_kc * sizeof(float));
    score_block_f32(q, centroids + start*d, block_kc, d, metric,
                    block_scores[block], opts);

    // Partial top-k within block
    block_ids[block] = malloc(nprobe * sizeof(int32_t));
    partial_topk(block_scores[block], block_kc, nprobe,
                 block_ids[block], NULL);

    // Adjust IDs to global range
    for (int i = 0; i < nprobe; i++) {
        block_ids[block][i] += start;
    }
}

// Merge block results using kernel #06 (k-way merge)
merge_topk(block_ids, block_scores, num_blocks, nprobe, list_ids, list_scores);
```

---

## Performance Characteristics

### 1. Computational Complexity

**Standard Selection**:
- Score computation: O(kc×d) FLOPs (kernel #04)
- Top-k selection: O(kc log nprobe) comparisons (kernel #05)
- **Total**: O(kc×d + kc log nprobe) ≈ O(kc×d)

**Beam Search**:
- Initial scoring: O(kc×d)
- Beam expansions: O(beam_expansions × beam_width × deg × d)
- Typical beam_expansions: 2-5
- **Total**: O(kc×d + expansions×beam×deg×d)

**Example** (kc=10K, d=1024, nprobe=50):
- Standard: 10,000 × 1024 × 2 + 10,000 × log(50) ≈ 20M FLOPs
- Beam (beam=100, expansions=3, deg=32): 20M + 3×100×32×1024×2 ≈ 40M FLOPs

### 2. Memory Bandwidth

**Reads**:
- Query: d floats = 4d bytes
- Centroids: kc×d floats = 4kc×d bytes
- Norms (if used): kc floats = 4kc bytes
- **Total**: 4d + 4kc×d + 4kc ≈ 4kc×d bytes

**Writes**:
- Scores (temporary): kc floats = 4kc bytes
- Output: nprobe IDs + scores = 8×nprobe bytes

**Example** (kc=10K, d=1024):
- Reads: 4×1024 + 4×10000×1024 + 4×10000 ≈ 40 MB
- Writes: 4×10000 + 8×50 = 40 KB
- **Memory-bound**: Dominated by centroid reads

### 3. Performance Targets (Apple M2 Max, 1 P-core)

| Configuration | Latency | Throughput (batch) | Notes |
|---------------|---------|-------------------|-------|
| kc=1K, d=1024, nprobe=10 | 20 μs | 50K queries/sec | Small kc |
| kc=10K, d=1024, nprobe=50 | 50 μs | 20K queries/sec | Medium kc |
| kc=100K, d=1024, nprobe=100 | 500 μs | 2K queries/sec | Large kc |
| kc=10K, d=1024, nprobe=50, beam | 150 μs | 6.7K queries/sec | 3× overhead |

**Scaling**:
- **kc**: Linear (O(kc))
- **d**: Linear (O(d))
- **nprobe**: Logarithmic (O(log nprobe)), negligible
- **Batch**: Linear scaling with parallelism

### 4. Recall vs nprobe

**Empirical Results** (1M vectors, 10K centroids, k=10):

| nprobe | Recall@10 | Search Time | Speedup vs Exact |
|--------|-----------|-------------|------------------|
| 1 | 25% | 0.5 ms | 500× |
| 10 | 75% | 2 ms | 125× |
| 50 | 92% | 8 ms | 31× |
| 100 | 95% | 15 ms | 17× |
| 500 | 98% | 60 ms | 4× |

**Rule of Thumb**: nprobe ≈ 0.5% to 5% of kc for good recall/performance balance.

---

## Numerical Considerations

### 1. Cosine Similarity Normalization

**Computation**:
```c
// Compute query inverse norm
float q_norm_sq = 0;
for (int j = 0; j < d; j++) {
    q_norm_sq += q[j] * q[j];
}
float q_inv_norm = 1.0f / sqrtf(q_norm_sq);

// Compute cosine similarities
for (int i = 0; i < kc; i++) {
    float ip = dot_product(q, centroids + i*d, d);
    scores[i] = ip * q_inv_norm * centroid_inv_norms[i];
}
```

**Numerical Issue**: Division by near-zero norm.

**Mitigation**: Clamp norms to minimum value:
```c
float q_inv_norm = 1.0f / sqrtf(fmax(q_norm_sq, 1e-10f));
```

### 2. Tie-Breaking Determinism

**Guarantee**: When scores are identical, tie-break by centroid ID to ensure deterministic results.

**Implementation**: Use stable sorting or explicit comparison function with ID tie-breaker.

### 3. Disabled List Masking

**Correctness**: Set masked scores to sentinel values:
- L2: INFINITY (will be excluded from min-heap)
- IP/Cosine: -INFINITY (will be excluded from max-heap)

---

## Correctness Testing

### 1. Brute-Force Parity

**Test 1: Exact Match**
```swift
func testIVFSelectParity() {
    let kc = 1_000
    let d = 512
    let nprobe = 20

    let q = generateRandomVector(d: d)
    let centroids = generateRandomVectors(n: kc, d: d)

    // Fast implementation
    var list_ids_fast = [Int32](repeating: 0, count: nprobe)
    ivf_select_nprobe(q, d, centroids, kc, METRIC_L2, nprobe, nil, &list_ids_fast, nil)

    // Brute-force reference
    var distances = [Float](repeating: 0, count: kc)
    for i in 0..<kc {
        distances[i] = l2SquaredDistance(q, centroids[i*d..<(i+1)*d])
    }

    var sorted_ids = Array(0..<kc).sorted(by: { distances[$0] < distances[$1] })
    var list_ids_ref = sorted_ids.prefix(nprobe).map { Int32($0) }

    // Should match exactly
    for i in 0..<nprobe {
        assert(list_ids_fast[i] == list_ids_ref[i],
               "Mismatch at \(i): fast=\(list_ids_fast[i]) ref=\(list_ids_ref[i])")
    }
}
```

### 2. Metric Equivalence

**Test 2: Cosine = IP × Norms**
```swift
func testCosineEquivalence() {
    let kc
<!-- moved to docs/kernel-specs/ -->
