# ✅ DONE — Kernel Specification #39: Candidate Reservoir Buffer

**ID**: 39
**Priority**: MUST
**Role**: Q (Query)
**Status**: Specification

---

## Purpose

Buffer candidate (id, score) pairs during vector search scans with minimal contention, maintaining a bounded top-C collection for downstream top-K selection and exact reranking. This kernel provides a high-throughput, low-overhead intermediate storage mechanism for search candidates before final result computation.

**Key Benefits**:
1. **Low overhead**: Near-memcpy throughput when acceptance rate is low (~1-5 ns per candidate)
2. **Bounded memory**: Fixed O(C) space regardless of scan size
3. **Per-thread isolation**: No locks or atomic operations during scan (lock-free reads)
4. **Adaptive strategy**: Dynamically switches between modes based on occupancy and deduplication rate
5. **Metric-agnostic**: Handles L2 (minimize), inner product (maximize), and cosine similarity

**Typical Use Case**: During IVF-PQ scan of 1M vectors, buffer top-C=1000 candidates per thread at ~200M candidates/sec evaluation rate, then extract top-K=10 results for exact reranking.

**Critical Path**: This kernel sits on the latency-critical query path between ADC scan (kernel #22) and top-K extraction (kernels #05/#06). Every nanosecond of overhead directly impacts query latency.

---

## Mathematical Foundations

### 1. Reservoir Problem Statement

**Definition**: Given a stream of n candidate pairs (id_i, score_i) for i ∈ [0, n), maintain a bounded collection R of at most C candidates such that at any point, R contains the C best candidates seen so far according to metric M.

**Metric Comparator**: Define ⪯_M as the metric-specific ordering:
- **L2 distance**: a ⪯_M b ⟺ a ≤ b (smaller is better)
- **Inner product**: a ⪯_M b ⟺ a ≥ b (larger is better)
- **Cosine similarity**: a ⪯_M b ⟺ a ≥ b (larger is better)

**Best Candidate**: Candidate (id_a, s_a) is better than (id_b, s_b) if:
```
s_a ≺_M s_b  OR  (s_a = s_b AND id_a < id_b)
```
The tie-breaking by ID ensures deterministic ordering.

**Theorem 1 (Reservoir Correctness)**: After processing all n candidates, if |R| ≤ C, then R contains exactly the C best candidates from the stream.

**Proof**: By induction on stream position:
- **Base case** (i=0): R = ∅ trivially satisfies the property
- **Inductive step**: Assume R contains C best candidates from first i elements. For candidate (i+1):
  - If (i+1) is worse than all in R, reject → R still contains C best
  - If (i+1) is better than worst in R, replace worst → R now contains C best from i+1 elements
∎

### 2. Threshold Maintenance

**Threshold τ**: The score of the worst candidate currently in R.
- **L2**: τ = max(scores in R)
- **IP/Cosine**: τ = min(scores in R)

**Acceptance Condition**: Accept candidate (id, s) if:
```
s ≺_M τ  OR  (s = τ AND id < worst_id)
```

**Theorem 2 (Threshold Pruning)**: For a stream with ε fraction of acceptable candidates (score better than τ), the expected number of threshold updates is O(C log(εn/C)).

**Proof**: Threshold updates occur when accepting a candidate that replaces the current worst. Using order statistics theory, the k-th best element from n samples has expected rank C + O(√C log n). The number of updates follows the harmonic series truncated at εn/C, giving O(C log(εn/C)). ∎

**Practical Implication**: For typical ε ≈ 0.01 (1% acceptance rate), n=1M, C=1000, expected updates ≈ 1000 × log(10) ≈ 2300, or ~0.23% of candidates trigger updates.

### 3. Heap Maintenance Complexity

**Heap Property**: Binary min-heap (L2) or max-heap (IP) where each parent is better than its children.

**Operations**:
- **Insert** (size < C): O(log C) sift-up
- **Replace root**: O(log C) sift-down
- **Extract-min/max**: O(log C) pop root and sift-down

**Theorem 3 (Amortized Heap Cost)**: For n candidates with acceptance rate ε, total heap operations:
```
Cost = ε × n × O(log C)
```

**Example**: n=1M, ε=0.01, C=1000 → ~10K accepts × log₂(1000) ≈ 100K comparisons

### 4. Block Mode Quickselect

**Strategy**: Append candidates unsorted until buffer exceeds C + headroom, then run quickselect to find C-th best element and partition buffer.

**Quickselect Complexity**: Average O(n) with std::nth_element or partition-based selection.

**Theorem 4 (Block Mode Amortization)**: With headroom h = αC (typical α=0.1), the amortized cost per accepted candidate is O(1 + (1+α)/α) = O(1) for large buffers.

**Proof**:
- Fill buffer: C + h = C(1+α) appends at O(1) each
- Prune: Quickselect on C(1+α) elements costs O(C)
- Amortized cost per append: O(C) / (C×α) = O(1/α) = O(1) for fixed α
∎

**Practical Benefit**: Block mode achieves near-memcpy throughput (~5 GB/s) when acceptance rate is low, as appends are sequential writes with no comparisons.

### 5. Adaptive Mode Switching

**Decision Function**: Switch from Block to Heap mode when:
```
occupancy = |R| / C > threshold  (typically 0.75)
```

**Theorem 5 (Adaptive Optimality)**: Adaptive mode minimizes total cost by using Block mode for low-selectivity phases and Heap mode for high-selectivity phases.

**Proof Sketch**: Block mode has O(1) amortized insert but O(C) periodic prune. Heap mode has O(log C) insert but no prune. The crossover occurs when:
```
n × O(1) + O(C) ≈ n × O(log C)
n ≈ C / log C
```
For C=1000, crossover at n≈145. Adaptive switching triggers near this optimal point. ∎

### 6. Deduplication Integration

**Problem**: During multi-list IVF scan (nprobe > 1), same vector may appear in multiple lists.

**Solution**: Integrate with VisitedSet (kernel #32) to check before insertion:
```
if (!visited_set_contains(vs, id)) {
    reservoir_push(r, id, score);
    visited_set_insert(vs, id);
}
```

**Theorem 6 (Dedup Correctness)**: With VisitedSet integration, reservoir contains at most C distinct candidates, with no duplicates.

**Proof**: By construction, each ID is checked against VisitedSet before insertion. Once inserted, ID is marked visited. Future occurrences are rejected. ∎

### 7. Memory Bandwidth Analysis

**Data Size**:
- Per candidate: 8 bytes (ID) + 4 bytes (score) = 12 bytes
- Buffer capacity C=1000: 12 KB (fits in L1 cache)

**Bandwidth Requirements**:
- **Block mode (append)**: 12 bytes write per candidate
- **Heap mode**: 12 bytes read + 12 bytes write (swap) + O(log C) compares
- **Quickselect prune**: O(C) reads and writes = 24 KB

**Memory-Bound**: Reservoir operations are typically memory-bound, limited by L1/L2 cache bandwidth (~200-500 GB/s on Apple M2).

**Theoretical Throughput**:
```
Throughput (Block) = L1_bandwidth / bytes_per_candidate
                   = 200 GB/s / 12 bytes
                   ≈ 16.7B candidates/sec
```

**Practical**: Achieves 5-10B candidates/sec due to prefetch delays and branch mispredictions.

---

## API Signatures

### 1. Reservoir Initialization

```c
/// Initialize candidate reservoir with specified capacity and metric
///
/// Creates a per-thread reservoir for buffering search candidates. The reservoir
/// maintains the C best candidates according to the specified metric.
///
/// @param r           Output handle (allocated on success)
/// @param capacity_C  Maximum number of candidates to maintain (typically 100-10000)
/// @param metric      Distance metric (L2_DISTANCE, INNER_PRODUCT, COSINE_SIMILARITY)
/// @param opts        Configuration options (mode, headroom, etc.), nullable
///
/// @return 0 on success, error code on failure
///
/// @note Thread safety: Each thread must have its own reservoir instance
/// @note Memory: Allocates O(C × 12 bytes) for buffer storage
/// @note Performance: Initialization cost ~1 μs for typical C=1000
void reservoir_init(
    Reservoir** r,                     // output handle
    int capacity_C,                    // max candidates to maintain
    Metric metric,                     // distance metric
    const ReservoirOpts* opts          // options (nullable)
);
```

**Metrics**:
```c
typedef enum {
    L2_DISTANCE,          // Minimize: smaller distance is better
    INNER_PRODUCT,        // Maximize: larger dot product is better
    COSINE_SIMILARITY     // Maximize: larger cosine is better
} Metric;
```

### 2. Reservoir Reset

```c
/// Reset reservoir for new query, optionally changing capacity
///
/// Clears all buffered candidates and resets internal state. This is much
/// faster than destroying and recreating the reservoir.
///
/// @param r           Reservoir handle
/// @param capacity_C  New capacity (0 to keep current capacity)
///
/// @note Performance: ~10 ns (just resets counters, no memory deallocation)
/// @note Use case: Reuse same reservoir across multiple queries
void reservoir_reset(
    Reservoir* r,                      // reservoir handle
    int capacity_C                     // new capacity (0=keep current)
);
```

### 3. Push Batch

```c
/// Push a batch of candidates into reservoir
///
/// Evaluates n candidates and inserts those that qualify for the top-C collection.
/// Integrates with VisitedSet for deduplication if provided.
///
/// @param r           Reservoir handle
/// @param ids         Candidate IDs [n]
/// @param scores      Candidate scores [n]
/// @param n           Number of candidates
/// @param vs          Optional VisitedSet for deduplication (kernel #32), nullable
///
/// @return Number of candidates accepted (0 ≤ accepted ≤ n)
///
/// @note Performance: 5-50 ns per candidate depending on acceptance rate and mode
/// @note Thread safety: Not thread-safe; use one reservoir per thread
/// @note Deduplication: If vs is non-null, checks and marks visited IDs
int reservoir_push_batch(
    Reservoir* r,                      // reservoir handle
    const int64_t* ids,                // [n] candidate IDs
    const float* scores,               // [n] candidate scores
    int n,                             // number of candidates
    VisitedSet* vs                     // optional visited set (nullable)
);
```

**Return Value**: Number of accepted candidates, useful for telemetry:
```c
int accepted = reservoir_push_batch(r, ids, scores, 1000, vs);
double accept_rate = (double)accepted / 1000.0;
```

### 4. Reservoir Size

```c
/// Get current number of candidates in reservoir
///
/// @param r           Reservoir handle
///
/// @return Current size (0 ≤ size ≤ capacity_C)
///
/// @note Performance: O(1), just returns counter
int reservoir_size(const Reservoir* r);
```

### 5. Reservoir Snapshot

```c
/// Get read-only view of current reservoir contents
///
/// Returns SoA (Structure-of-Arrays) pointers to internal buffers. The returned
/// pointers are valid until next reservoir operation.
///
/// @param r           Reservoir handle
/// @param scores      Output: pointer to scores array [size]
/// @param ids         Output: pointer to IDs array [size]
/// @param count       Output: current reservoir size
///
/// @note Lifetime: Pointers invalid after next reservoir operation
/// @note Order: Elements are in internal storage order (not sorted)
/// @note Thread safety: Read-only; safe for concurrent reads
void reservoir_snapshot(
    const Reservoir* r,                // reservoir handle
    const float** scores,              // output: scores array
    const int64_t** ids,               // output: IDs array
    int* count                         // output: size
);
```

**Usage Pattern**:
```c
const float* scores;
const int64_t* ids;
int count;
reservoir_snapshot(r, &scores, &ids, &count);

for (int i = 0; i < count; i++) {
    printf("Candidate %lld: score=%.4f\n", ids[i], scores[i]);
}
```

### 6. Extract Top-K

```c
/// Extract top-K candidates from reservoir
///
/// Runs top-K selection (kernels #05/#06) on reservoir contents and returns
/// sorted results. This is typically the final step before exact reranking.
///
/// @param r           Reservoir handle
/// @param k           Number of results to extract (k ≤ reservoir_size)
/// @param top_scores  Output: top-K scores [k] in sorted order (best first)
/// @param top_ids     Output: top-K IDs [k] corresponding to scores
///
/// @note Performance: O(size × log k) for k-select + O(k log k) for sort
/// @note Ordering: Results are sorted from best to worst according to metric
/// @note Side effects: Does not modify reservoir (read-only operation)
void reservoir_extract_topk(
    const Reservoir* r,                // reservoir handle
    int k,                             // number of results to extract
    float* top_scores,                 // [k] output scores (pre-allocated)
    int64_t* top_ids                   // [k] output IDs (pre-allocated)
);
```

### 7. Reservoir Cleanup

```c
/// Free reservoir resources
///
/// @param r           Reservoir handle
///
/// @note Must be called to avoid memory leaks
void reservoir_free(Reservoir* r);
```

### 8. Configuration Options

```c
/// Configuration options for reservoir behavior
typedef struct {
    /// Buffering mode: Heap, Block, or Adaptive
    ReservoirMode mode;

    /// Extra headroom before pruning (fraction of C, typically 0.1-0.2)
    /// Only used in Block mode
    float reserve_extra;

    /// Threshold for adaptive mode switch (occupancy %, typically 0.75)
    /// When |R|/C exceeds this, switch from Block to Heap
    float adaptive_threshold;

    /// Enable deterministic tie-breaking by ID (default: true)
    /// Ensures reproducible results when scores are equal
    bool stable_ties;

    /// Initial mode for Adaptive (default: Block)
    ReservoirMode adaptive_initial_mode;

    /// Enable telemetry collection (default: false)
    bool telemetry;
} ReservoirOpts;

/// Reservoir buffering modes
typedef enum {
    RESERVOIR_MODE_HEAP,       // Min/max heap with O(log C) insert
    RESERVOIR_MODE_BLOCK,      // Block append + periodic quickselect
    RESERVOIR_MODE_ADAPTIVE    // Adaptive switching based on occupancy
} ReservoirMode;
```

**Default Options**:
```c
static const ReservoirOpts ReservoirOptsDefault = {
    .mode = RESERVOIR_MODE_ADAPTIVE,
    .reserve_extra = 0.1f,
    .adaptive_threshold = 0.75f,
    .stable_ties = true,
    .adaptive_initial_mode = RESERVOIR_MODE_BLOCK,
    .telemetry = false
};
```

### 9. Opaque Handle

```c
/// Opaque reservoir handle (implementation-defined)
typedef struct Reservoir Reservoir;
```

**Internal Structure** (not exposed to users):
```c
struct Reservoir {
    // Configuration
    int capacity;                  // Max capacity C
    Metric metric;                 // Distance metric
    ReservoirMode current_mode;    // Current buffering mode
    ReservoirOpts opts;            // Configuration options

    // Storage (SoA layout)
    float* scores;                 // [capacity'] score array
    int64_t* ids;                  // [capacity'] ID array
    int size;                      // Current number of candidates

    // Threshold
    float tau;                     // Current acceptance threshold
    int worst_idx;                 // Index of worst candidate (heap root)

    // Telemetry
    int64_t pushed;                // Total candidates evaluated
    int64_t accepted;              // Total accepted
    int64_t rejected_tau;          // Rejected by threshold
    int64_t rejected_dedup;        // Rejected by deduplication
    int64_t prunes;                // Number of quickselect prunes
    int64_t mode_switches;         // Adaptive mode switches
};
```

---

## Algorithm Details

### 1. Heap Mode: Binary Heap with Sift-Down

**Data Structure**: Binary min-heap (L2) or max-heap (IP/Cosine) stored in array.

**Heap Indexing**:
```
Parent of i:  (i - 1) / 2
Left child:   2*i + 1
Right child:  2*i + 2
```

**Insert Algorithm** (size < C):
```c
void heap_insert(Reservoir* r, int64_t id, float score) {
    int i = r->size;
    r->ids[i] = id;
    r->scores[i] = score;
    r->size++;

    // Sift up
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!is_better(score, r->scores[parent], r->metric)) {
            break;  // Heap property satisfied
        }

        // Swap with parent
        swap(&r->ids[i], &r->ids[parent]);
        swap(&r->scores[i], &r->scores[parent]);
        i = parent;
    }

    // Update threshold
    r->tau = r->scores[0];  // Root is worst element
}
```

**Replace Root Algorithm** (size == C, new candidate better than root):
```c
void heap_replace_root(Reservoir* r, int64_t id, float score) {
    r->ids[0] = id;
    r->scores[0] = score;

    // Sift down
    int i = 0;
    int size = r->size;

    while (true) {
        int left = 2*i + 1;
        int right = 2*i + 2;
        int worse = i;

        // Find worse child (we maintain min-heap/max-heap depending on metric)
        if (left < size && is_worse(r->scores[left], r->scores[worse], r->metric)) {
            worse = left;
        }
        if (right < size && is_worse(r->scores[right], r->scores[worse], r->metric)) {
            worse = right;
        }

        if (worse == i) {
            break;  // Heap property satisfied
        }

        // Swap with worse child
        swap(&r->ids[i], &r->ids[worse]);
        swap(&r->scores[i], &r->scores[worse]);
        i = worse;
    }

    // Update threshold
    r->tau = r->scores[0];
}
```

**Comparison Function**:
```c
bool is_better(float a, float b, Metric metric) {
    switch (metric) {
        case L2_DISTANCE:
            return a < b;  // Smaller is better
        case INNER_PRODUCT:
        case COSINE_SIMILARITY:
            return a > b;  // Larger is better
    }
}

bool is_worse(float a, float b, Metric metric) {
    return is_better(b, a, metric);
}
```

**Tie-Breaking**:
```c
bool is_better_with_tie(float score_a, int64_t id_a,
                       float score_b, int64_t id_b,
                       Metric metric) {
    if (score_a == score_b) {
        return id_a < id_b;  // Deterministic: smaller ID wins
    }
    return is_better(score_a, score_b, metric);
}
```

### 2. Block Mode: Append + Quickselect Prune

**Append Algorithm** (size < capacity + headroom):
```c
void block_append(Reservoir* r, int64_t id, float score) {
    int idx = r->size;
    r->ids[idx] = id;
    r->scores[idx] = score;
    r->size++;

    // Check if prune needed
    int capacity_with_headroom = r->capacity * (1.0f + r->opts.reserve_extra);
    if (r->size >= capacity_with_headroom) {
        block_prune(r);
    }
}
```

**Prune Algorithm** (quickselect to C-th element):
```c
void block_prune(Reservoir* r) {
    int target = r->capacity;

    // Partition around C-th best element using quickselect
    // This places C best elements in [0, C) (unordered)
    quickselect_partition(r->scores, r->ids, r->size, target, r->metric);

    // Compact: keep only first C elements
    r->size = target;

    // Update threshold: worst element among C best
    r->tau = find_worst_in_range(r->scores, r->size, r->metric);

    // Telemetry
    r->prunes++;
}
```

**Quickselect Implementation** (partition-based):
```c
void quickselect_partition(float* scores, int64_t* ids, int n, int k, Metric metric) {
    int left = 0;
    int right = n - 1;

    while (left < right) {
        // Choose pivot (median-of-three for better worst case)
        int mid = left + (right - left) / 2;
        float pivot = median_of_three(scores[left], scores[mid], scores[right], metric);

        // Partition
        int i = left;
        int j = right;
        while (i <= j) {
            while (i <= right && is_better(scores[i], pivot, metric)) i++;
            while (j >= left && is_worse(scores[j], pivot, metric)) j--;
            if (i <= j) {
                swap(&scores[i], &scores[j]);
                swap(&ids[i], &ids[j]);
                i++;
                j--;
            }
        }

        // Recurse into partition containing k
        if (k <= j) {
            right = j;
        } else if (k >= i) {
            left = i;
        } else {
            break;  // k-th element is at position j+1 or i-1
        }
    }
}
```

**Complexity**: Average O(n), worst-case O(n²) but rare with median-of-three pivot.

### 3. Adaptive Mode Switching

**Decision Logic**:
```c
void adaptive_check_switch(Reservoir* r) {
    if (r->current_mode == RESERVOIR_MODE_HEAP) {
        return;  // Once in Heap mode, stay in Heap mode
    }

    // Check occupancy
    float occupancy = (float)r->size / r->capacity;

    if (occupancy > r->opts.adaptive_threshold) {
        // Switch to Heap mode
        // Convert current buffer to heap
        heapify_in_place(r->scores, r->ids, r->size, r->metric);
        r->current_mode = RESERVOIR_MODE_HEAP;
        r->tau = r->scores[0];  // Root is worst element
        r->mode_switches++;
    }
}
```

**Heapify In-Place** (Floyd's algorithm):
```c
void heapify_in_place(float* scores, int64_t* ids, int size, Metric metric) {
    // Start from last non-leaf node and sift down
    for (int i = (size / 2) - 1; i >= 0; i--) {
        sift_down(scores, ids, size, i, metric);
    }
}
```

**Complexity**: O(n) heapify, much faster than n inserts at O(n log n).

### 4. Push Batch Implementation

**Main Algorithm**:
```c
int reservoir_push_batch(Reservoir* r, const int64_t* ids, const float* scores,
                        int n, VisitedSet* vs) {
    int accepted = 0;

    for (int i = 0; i < n; i++) {
        int64_t id = ids[i];
        float score = scores[i];

        // Deduplication check
        if (vs && visited_set_contains(vs, id)) {
            r->rejected_dedup++;
            continue;
        }

        // Acceptance check
        bool accept = false;
        if (r->size < r->capacity) {
            accept = true;  // Buffer not full
        } else {
            // Compare with threshold
            if (is_better(score, r->tau, r->metric)) {
                accept = true;
            } else if (score == r->tau) {
                // Tie-breaking: accept if ID is better
                int64_t worst_id = r->ids[r->worst_idx];
                if (id < worst_id) {
                    accept = true;
                }
            }
        }

        if (accept) {
            // Insert based on current mode
            if (r->current_mode == RESERVOIR_MODE_HEAP) {
                if (r->size < r->capacity) {
                    heap_insert(r, id, score);
                } else {
                    heap_replace_root(r, id, score);
                }
            } else {  // Block mode
                block_append(r, id, score);
            }

            // Mark as visited
            if (vs) {
                visited_set_insert(vs, id);
            }

            accepted++;
            r->accepted++;
        } else {
            r->rejected_tau++;
        }

        r->pushed++;

        // Adaptive mode check (every 64 candidates for efficiency)
        if (r->current_mode == RESERVOIR_MODE_ADAPTIVE && (i & 63) == 0) {
            adaptive_check_switch(r);
        }
    }

    return accepted;
}
```

**Optimization**: Batch processing amortizes branch overhead and enables prefetching.

### 5. Extract Top-K

**Algorithm** (using kernel #05 partial top-K selection):
```c
void reservoir_extract_topk(const Reservoir* r, int k,
                           float* top_scores, int64_t* top_ids) {
    assert(k <= r->size);

    if (r->current_mode == RESERVOIR_MODE_HEAP && k == r->size) {
        // Heap is already partially sorted, can extract directly
        // But need to sort for output
        memcpy(top_scores, r->scores, r->size * sizeof(float));
        memcpy(top_ids, r->ids, r->size * sizeof(int64_t));
        sort_by_score(top_scores, top_ids, k, r->metric);
    } else {
        // Use partial top-K selection (kernel #05)
        topk_partial_select(r->scores, r->ids, r->size, k,
                           top_scores, top_ids, r->metric);
    }
}
```

**Complexity**: O(size × log k) for selection + O(k log k) for final sort.

### 6. Vectorized Comparison (SIMD)

**Problem**: Compare batch of scores against threshold in parallel.

**SIMD Implementation** (NEON):
```c
int reservoir_push_batch_simd(Reservoir* r, const int64_t* ids,
                             const float* scores, int n, VisitedSet* vs) {
    int accepted = 0;
    float32x4_t tau_vec = vdupq_n_f32(r->tau);

    int i = 0;
    // Process 4 candidates at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t score_vec = vld1q_f32(&scores[i]);

        // Compare against threshold (L2: scores < tau, IP: scores > tau)
        uint32x4_t mask;
        if (r->metric == L2_DISTANCE) {
            mask = vcltq_f32(score_vec, tau_vec);  // scores < tau
        } else {
            mask = vcgtq_f32(score_vec, tau_vec);  // scores > tau
        }

        // Process candidates that passed threshold
        for (int j = 0; j < 4; j++) {
            if (vgetq_lane_u32(mask, j)) {
                // Dedup check and insert
                if (!vs || !visited_set_contains(vs, ids[i+j])) {
                    // Insert candidate
                    reservoir_push_single(r, ids[i+j], scores[i+j]);
                    if (vs) visited_set_insert(vs, ids[i+j]);
                    accepted++;
                }
            }
        }
    }

    // Scalar tail
    for (; i < n; i++) {
        // ... scalar processing
    }

    return accepted;
}
```

**Speedup**: ~2× for L2 distance due to vectorized comparisons and reduced branch mispredictions.

---

## Implementation Strategies

### 1. Cache-Friendly SoA Layout

**Structure-of-Arrays (SoA)** provides better cache utilization:
```c
// SoA: scores and IDs in separate arrays
float scores[1000];
int64_t ids[1000];

// AoS (avoid): candidates as structs
struct Candidate {
    int64_t id;
    float score;
} candidates[1000];
```

**Benefit**: When scanning only scores for comparison, SoA loads only score data into cache, avoiding unnecessary ID loads.

### 2. Alignment for SIMD

**Align Arrays to Cache Lines**:
```c
float* scores = aligned_alloc(64, capacity * sizeof(float));
int64_t* ids = aligned_alloc(64, capacity * sizeof(int64_t));
```

**Benefit**: 64-byte alignment ensures SIMD loads don't cross cache lines, improving throughput.

### 3. Prefetching

**Software Prefetch** for next batch:
```c
void reservoir_push_batch_prefetch(Reservoir* r, const int64_t* ids,
                                  const float* scores, int n, VisitedSet* vs) {
    const int PREFETCH_DISTANCE = 8;

    for (int i = 0; i < n; i++) {
        // Prefetch future candidates
        if (i + PREFETCH_DISTANCE < n) {
            __builtin_prefetch(&scores[i + PREFETCH_DISTANCE], 0, 3);
            __builtin_prefetch(&ids[i + PREFETCH_DISTANCE], 0, 3);
        }

        // Process current candidate
        reservoir_push_single(r, ids[i], scores[i]);
    }
}
```

**Benefit**: Hides memory latency (~100 cycles) by prefetching ahead.

### 4. Branch Prediction Hints

**Help Compiler with Likely/Unlikely**:
```c
if (__builtin_expect(r->size < r->capacity, 0)) {
    // Unlikely: buffer not full (only happens at start)
    heap_insert(r, id, score);
} else {
    // Likely: buffer full, check threshold
    if (__builtin_expect(is_better(score, r->tau, r->metric), 0)) {
        // Unlikely: accept (typically 1-5% acceptance rate)
        heap_replace_root(r, id, score);
    }
}
```

**Benefit**: Reduces branch misprediction penalties (~15 cycles per mispredict).

### 5. Batch Size Tuning

**Optimal Batch Size**:
- **Small batches** (< 64): High function call overhead
- **Large batches** (> 512): Poor cache locality
- **Sweet spot**: 128-256 candidates per batch

```c
// Example: Process ADC scan results in batches
const int BATCH_SIZE = 256;
for (int offset = 0; offset < n_candidates; offset += BATCH_SIZE) {
    int batch_n = min(BATCH_SIZE, n_candidates - offset);
    reservoir_push_batch(r, &ids[offset], &scores[offset], batch_n, vs);
}
```

---

## Performance Characteristics

### 1. Throughput Targets (Apple M2 Max)

**Heap Mode** (C=1000, ε=1% acceptance rate):

| Metric | Throughput (candidates/sec) | Latency per candidate | Notes |
|--------|------------------------------|----------------------|-------|
| L2 distance | 200M | 5 ns | Log₂(1000) ≈ 10 comparisons |
| Inner product | 180M | 5.5 ns | Slightly slower due to metric |
| With dedup | 150M | 6.7 ns | VisitedSet overhead |

**Block Mode** (C=1000, low occupancy):

| Metric | Throughput (candidates/sec) | Latency per candidate | Notes |
|--------|------------------------------|----------------------|-------|
| Append only | 5B | 0.2 ns | Near-memcpy speed |
| With threshold check | 1B | 1 ns | Single comparison |
| With prune (periodic) | Amortized 500M | 2 ns | O(C) prune amortized |

**Adaptive Mode** (C=1000):

| Phase | Throughput | Notes |
|-------|-----------|-------|
| Initial (Block) | 1-2B/sec | Low occupancy, append-only |
| After switch (Heap) | 200M/sec | High occupancy, heap maintenance |
| Overall | 500M-1B/sec | Depends on distribution |

**Key Observations**:
1. Block mode is 5-10× faster during low-occupancy phase
2. Heap mode degrades to ~200M/sec due to O(log C) overhead
3. Adaptive mode achieves best of both worlds

### 2. Memory Bandwidth Analysis

**Heap Mode**:
- Per insert: 12 bytes write (ID + score) + O(log C) × 24 bytes read/write (swaps)
- Average: ~150 bytes per insert
- Bandwidth: 200M inserts/sec × 150 bytes ≈ 30 GB/s

**Block Mode**:
- Per append: 12 bytes write (sequential)
- Bandwidth: 5B appends/sec × 12 bytes ≈ 60 GB/s (memory-bound)

**Cache Footprint**:
- C=1000: 12 KB (fits in 64 KB L1 cache)
- C=10000: 120 KB (spills to L2 cache, ~2× slower)

### 3. Scalability Analysis

**Capacity Scaling** (Heap mode):

| Capacity C | Insert latency | Speedup vs full sort |
|-----------|---------------|---------------------|
| 100 | 3.5 ns (log₂(100)≈6.6) | 100× |
| 1000 | 5 ns (log₂(1000)≈10) | 1000× |
| 10000 | 7 ns (log₂(10000)≈13.3) | 10000× |

**Stream Size Scaling**:

| Stream size n | Block prune overhead | Amortized cost |
|--------------|---------------------|----------------|
| 10K | 0.1% (1 prune) | 1.01 ns/candidate |
| 100K | 0.01% (10 prunes) | 1.001 ns/candidate |
| 1M | 0.001% (100 prunes) | 1.0001 ns/candidate |

**Conclusion**: Prune overhead is negligible for large streams.

### 4. Comparison with Alternatives

**Baseline: Full Sort**:
- Complexity: O(n log n)
- Time for n=1M: ~100 ms (std::sort)
- Extract top-K: O(K) trivial

**Reservoir (Heap)**:
- Complexity: O(n log C)
- Time for n=1M, C=1000: ~5 ms
- **Speedup**: 20×

**Reservoir (Adaptive)**:
- Complexity: O(n) average (Block phase) + O(n log C) (Heap phase)
- Time for n=1M, C=1000: ~2 ms
- **Speedup**: 50×

---

## Numerical Considerations

### 1. Floating-Point Comparison Stability

**Problem**: Floating-point equality is unstable due to rounding errors.

**Solution**: Use epsilon comparison for tie-breaking:
```c
bool scores_equal(float a, float b) {
    return fabs(a - b) < 1e-6f;  // Epsilon threshold
}

bool is_better_stable(float score_a, int64_t id_a,
                     float score_b, int64_t id_b,
                     Metric metric) {
    if (scores_equal(score_a, score_b)) {
        return id_a < id_b;  // Deterministic tie-break
    }
    return is_better(score_a, score_b, metric);
}
```

**Trade-off**: Epsilon too large → incorrect tie-breaking; too small → unstable comparisons.

### 2. NaN and Inf Handling

**Problem**: NaN scores can corrupt heap structure.

**Detection**:
```c
bool is_valid_score(float score) {
    return !isnan(score) && !isinf(score);
}
```

**Rejection**:
```c
int reservoir_push_batch(Reservoir* r, const int64_t* ids,
                        const float* scores, int n, VisitedSet* vs) {
    for (int i = 0; i < n; i++) {
        if (!is_valid_score(scores[i])) {
            r->rejected_invalid++;
            continue;
        }
        // ... normal processing
    }
}
```

### 3. Overflow Protection

**ID Overflow**: With int64_t IDs, overflow is not a concern (2^63 ≈ 9 × 10^18).

**Capacity Overflow**: Check before allocation:
```c
void reservoir_init(Reservoir** r, int capacity_C, Metric metric,
                   const ReservoirOpts* opts) {
    // Check for overflow
    if (capacity_C > INT_MAX / 2) {
        fprintf(stderr, "Error: capacity too large\n");
        *r = NULL;
        return;
    }

    int capacity_with_headroom = capacity_C * (1.0f + opts->reserve_extra);
    // ... allocate
}
```

---

## Correctness Testing

### 1. Correctness Against Full Sort

```swift
func testCorrectnessVsFullSort() throws {
    let n = 10000
    let C = 1000
    let k = 10

    // Generate random candidates
    var ids = [Int64](0..<Int64(n))
    var scores = (0..<n).map { _ in Float.random(in: 0...100) }

    // Reservoir extraction
    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

    reservoir_push_batch(reservoir!, ids, scores, Int32(n), nil)

    var res_scores = [Float](repeating: 0, count: k)
    var res_ids = [Int64](repeating: 0, count: k)
    reservoir_extract_topk(reservoir!, Int32(k), &res_scores, &res_ids)

    // Full sort reference
    var pairs = zip(ids, scores).sorted { $0.1 < $1.1 }
    let ref_ids = pairs.prefix(k).map { $0.0 }
    let ref_scores = pairs.prefix(k).map { $0.1 }

    // Compare
    for i in 0..<k {
        XCTAssertEqual(res_ids[i], ref_ids[i])
        XCTAssertEqual(res_scores[i], ref_scores[i], accuracy: 1e-6)
    }

    reservoir_free(reservoir!)
}
```

### 2. Determinism Under Equal Scores

```swift
func testDeterminism() throws {
    let n = 1000
    let C = 100

    // All scores equal, should order by ID
    var ids = [Int64]((0..<Int64(n)).shuffled())
    var scores = [Float](repeating: 42.0, count: n)

    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

    reservoir_push_batch(reservoir!, ids, scores, Int32(n), nil)

    var res_scores = [Float](repeating: 0, count: C)
    var res_ids = [Int64](repeating: 0, count: C)
    reservoir_extract_topk(reservoir!, Int32(C), &res_scores, &res_ids)

    // Should be sorted by ID (0, 1, 2, ..., 99)
    for i in 0..<C {
        XCTAssertEqual(res_ids[i], Int64(i))
        XCTAssertEqual(res_scores[i], 42.0)
    }

    reservoir_free(reservoir!)
}
```

### 3. Deduplication Integration

```swift
func testDeduplication() throws {
    let C = 100

    // Insert same ID with different scores
    var ids: [Int64] = [1, 2, 1, 3, 2, 1]  // IDs 1 and 2 repeated
    var scores: [Float] = [10.0, 20.0, 15.0, 30.0, 25.0, 5.0]

    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

    var visited: UnsafeMutablePointer<VisitedSet>?
    visited_set_init(&visited, 10000)

    let accepted = reservoir_push_batch(reservoir!, ids, scores, 6, visited)

    // Should accept only first occurrence of each ID: IDs 1, 2, 3
    XCTAssertEqual(accepted, 3)
    XCTAssertEqual(reservoir_size(reservoir!), 3)

    reservoir_free(reservoir!)
    visited_set_free(visited!)
}
```

### 4. Mode Switching (Adaptive)

```swift
func testAdaptiveModeSwitch() throws {
    var opts = ReservoirOpts.default
    opts.mode = .adaptive
    opts.adaptive_threshold = 0.75

    let C = 1000

    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, &opts)

    // Push candidates in two phases
    // Phase 1: Low occupancy (Block mode expected)
    var ids1 = [Int64]((0..<500).map { Int64($0) })
    var scores1 = [Float]((0..<500).map { Float($0) })
    reservoir_push_batch(reservoir!, ids1, scores1, 500, nil)

    // Should still be in Block mode
    XCTAssertEqual(reservoir!.pointee.current_mode, RESERVOIR_MODE_BLOCK)

    // Phase 2: High occupancy (should trigger switch to Heap)
    var ids2 = [Int64]((500..<1000).map { Int64($0) })
    var scores2 = [Float]((500..<1000).map { Float($0) })
    reservoir_push_batch(reservoir!, ids2, scores2, 500, nil)

    // Should have switched to Heap mode
    XCTAssertEqual(reservoir!.pointee.current_mode, RESERVOIR_MODE_HEAP)
    XCTAssertEqual(reservoir!.pointee.mode_switches, 1)

    reservoir_free(reservoir!)
}
```

### 5. Performance: Accept Rate

```swift
func testPerformanceAcceptRate() throws {
    let n = 1_000_000
    let C = 1000

    // Generate candidates with 1% acceptance rate
    var ids = [Int64]((0..<Int64(n)).shuffled())
    var scores = (0..<n).map { Float($0 % 100) }  // Scores 0-99

    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

    measure {
        reservoir_reset(reservoir!, Int32(C))
        reservoir_push_batch(reservoir!, ids, scores, Int32(n), nil)
    }

    let accepted = reservoir!.pointee.accepted
    let accept_rate = Double(accepted) / Double(n)

    print("Accept rate: \(accept_rate * 100)%")
    print("Throughput: \(Double(n) / executionTime) candidates/sec")

    reservoir_free(reservoir!)
}
```

### 6. Memory Footprint

```swift
func testMemoryFootprint() throws {
    let capacities = [100, 1000, 10000, 100000]

    for C in capacities {
        var reservoir: UnsafeMutablePointer<Reservoir>?
        reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

        let memory = C * (MemoryLayout<Float>.size + MemoryLayout<Int64>.size)
        print("C=\(C): \(memory / 1024) KB")

        reservoir_free(reservoir!)
    }
}
```

---

## Integration Patterns

### 1. IVF-PQ Query with Reservoir

```swift
import Foundation

/// Complete IVF-PQ query with reservoir buffering
func queryIVFPQWithReservoir(
    query: [Float],
    index: IVFPQIndex,
    k: Int,
    nprobe: Int,
    C: Int  // Reservoir capacity
) -> [(id: Int64, distance: Float)] {
    let d = index.dimension
    let m = index.m_subspaces
    let ks = index.ks_codebook_size

    // Step 1: Select top-nprobe IVF lists
    let probe_lists = selectNprobeIVFLists(query: query,
                                          coarse_centroids: index.coarse_centroids,
                                          nprobe: nprobe)

    // Step 2: Initialize reservoir and visited set
    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

    var visited: UnsafeMutablePointer<VisitedSet>?
    visited_set_init(&visited, Int32(index.num_vectors))

    // Step 3: Scan each probed IVF list
    for (list_id, _) in probe_lists.prefix(nprobe) {
        let ivf_list = index.ivf_lists[list_id]
        guard ivf_list.count > 0 else { continue }

        // Build LUT for this list
        var lut = [Float](repeating: 0, count: m * ks)
        buildResidualLUT(query: query,
                        coarse_centroid: index.coarse_centroids[list_id],
                        d: d, m: m, ks: ks,
                        codebooks: index.pq_codebooks,
                        lut: &lut,
                        centroid_norms: index.centroid_norms)

        // ADC scan (kernel #22)
        var scores = [Float](repeating: 0, count: ivf_list.count)
        adc_scan(codes: ivf_list.codes, n: ivf_list.count,
                m: m, ks: ks, lut: lut, scores: &scores)

        // Push to reservoir with deduplication
        reservoir_push_batch(reservoir!, ivf_list.ids, scores,
                           Int32(ivf_list.count), visited)
    }

    // Step 4: Extract top-K from reservoir
    var top_scores = [Float](repeating: 0, count: k)
    var top_ids = [Int64](repeating: 0, count: k)
    reservoir_extract_topk(reservoir!, Int32(k), &top_scores, &top_ids)

    // Step 5: Cleanup
    reservoir_free(reservoir!)
    visited_set_free(visited!)

    // Return results
    return zip(top_ids, top_scores).map { ($0, $1) }
}
```

### 2. Multi-Threaded Reservoir Merge

```swift
/// Query with per-thread reservoirs, merge results
func queryMultiThreaded(
    query: [Float],
    index: IVFPQIndex,
    k: Int,
    num_threads: Int
) -> [(id: Int64, distance: Float)] {
    let C = k * 10  // Reservoir capacity per thread

    // Partition IVF lists across threads
    let lists_per_thread = index.ivf_lists.count / num_threads

    // Per-thread results
    var thread_reservoirs = [UnsafeMutablePointer<Reservoir>?](repeating: nil,
                                                                count: num_threads)

    DispatchQueue.concurrentPerform(iterations: num_threads) { thread_id in
        // Initialize per-thread reservoir
        reservoir_init(&thread_reservoirs[thread_id], Int32(C), L2_DISTANCE, nil)

        // Scan assigned lists
        let start_list = thread_id * lists_per_thread
        let end_list = (thread_id == num_threads - 1) ? index.ivf_lists.count
                                                       : (thread_id + 1) * lists_per_thread

        for list_id in start_list..<end_list {
            // ... LUT build, ADC scan, reservoir push
        }
    }

    // Merge thread reservoirs
    var merged_scores = [Float]()
    var merged_ids = [Int64]()

    for r in thread_reservoirs {
        guard let reservoir = r else { continue }

        let size = reservoir_size(reservoir)
        var scores = [Float](repeating: 0, count: Int(size))
        var ids = [Int64](repeating: 0, count: Int(size))

        const float* scores_ptr;
        const int64_t* ids_ptr;
        int count;
        reservoir_snapshot(reservoir, &scores_ptr, &ids_ptr, &count)

        merged_scores.append(contentsOf: UnsafeBufferPointer(start: scores_ptr,
                                                             count: Int(count)))
        merged_ids.append(contentsOf: UnsafeBufferPointer(start: ids_ptr,
                                                          count: Int(count)))

        reservoir_free(reservoir)
    }

    // Final top-K merge (kernel #06)
    return topk_merge(ids: merged_ids, scores: merged_scores, k: k)
}
```

### 3. Exact Rerank Integration

```swift
/// Extract reservoir candidates for exact reranking
func extractForRerank(
    reservoir: UnsafeMutablePointer<Reservoir>,
    query: [Float],
    index: IVFPQIndex,
    k: Int
) -> [(id: Int64, distance: Float)] {
    // Extract top-C candidates from reservoir (C >> k)
    let C = reservoir_size(reservoir)
    var candidate_scores = [Float](repeating: 0, count: Int(C))
    var candidate_ids = [Int64](repeating: 0, count: Int(C))

    const float* scores_ptr;
    const int64_t* ids_ptr;
    int count;
    reservoir_snapshot(reservoir, &scores_ptr, &ids_ptr, &count)

    // Exact rerank (kernel #40)
    var reranked = exactRerank(query: query,
                              candidate_ids: Array(UnsafeBufferPointer(start: ids_ptr,
                                                                      count: Int(count))),
                              vectors: index.full_vectors,
                              k: k)

    return reranked
}
```

---

## Coding Guidelines

### 1. Memory Safety

**Always Check Allocation**:
```c
void reservoir_init(Reservoir** r, int capacity_C, Metric metric,
                   const ReservoirOpts* opts) {
    if (capacity_C <= 0) {
        *r = NULL;
        return;
    }

    Reservoir* reservoir = malloc(sizeof(Reservoir));
    if (!reservoir) {
        *r = NULL;
        return;
    }

    reservoir->scores = aligned_alloc(64, capacity_C * sizeof(float));
    reservoir->ids = aligned_alloc(64, capacity_C * sizeof(int64_t));

    if (!reservoir->scores || !reservoir->ids) {
        free(reservoir->scores);
        free(reservoir->ids);
        free(reservoir);
        *r = NULL;
        return;
    }

    // ... initialize
    *r = reservoir;
}
```

### 2. Telemetry Integration

```c
#include "telemetry.h"

int reservoir_push_batch(Reservoir* r, const int64_t* ids,
                        const float* scores, int n, VisitedSet* vs) {
    uint64_t start_time = clock_gettime_ns();

    int accepted = 0;
    // ... processing

    if (r->opts.telemetry) {
        uint64_t end_time = clock_gettime_ns();
        double time_us = (end_time - start_time) / 1000.0;

        telemetry_emit("reservoir.pushed", n);
        telemetry_emit("reservoir.accepted", accepted);
        telemetry_emit("reservoir.rejected_tau", r->rejected_tau);
        telemetry_emit("reservoir.rejected_dedup", r->rejected_dedup);
        telemetry_emit("reservoir.current_size", r->size);
        telemetry_emit("reservoir.current_tau", r->tau);
        telemetry_emit("reservoir.mode", r->current_mode);
        telemetry_emit("reservoir.time_us", time_us);
        telemetry_emit("reservoir.throughput_per_sec", n / (time_us / 1e6));
    }

    return accepted;
}
```

### 3. Const Correctness

```c
// Read-only functions should take const pointers
int reservoir_size(const Reservoir* r) {
    return r->size;
}

void reservoir_snapshot(const Reservoir* r, const float** scores,
                       const int64_t** ids, int* count) {
    *scores = r->scores;
    *ids = r->ids;
    *count = r->size;
}

// Mutating functions take non-const pointers
int reservoir_push_batch(Reservoir* r, const int64_t* ids,
                        const float* scores, int n, VisitedSet* vs);
```

---

## Example Usage

### Example 1: Basic Reservoir Usage (C)

```c
#include "reservoir.h"
#include <stdio.h>

int main() {
    int C = 1000;  // Capacity
    int k = 10;    // Top-K to extract

    // Initialize reservoir
    Reservoir* r;
    reservoir_init(&r, C, L2_DISTANCE, NULL);

    // Simulate scan: push 10K candidates
    int64_t ids[10000];
    float scores[10000];
    for (int i = 0; i < 10000; i++) {
        ids[i] = i;
        scores[i] = (float)(rand() % 1000) / 10.0f;
    }

    int accepted = reservoir_push_batch(r, ids, scores, 10000, NULL);
    printf("Accepted %d / 10000 candidates\n", accepted);

    // Extract top-K
    float top_scores[10];
    int64_t top_ids[10];
    reservoir_extract_topk(r, k, top_scores, top_ids);

    printf("Top-%d results:\n", k);
    for (int i = 0; i < k; i++) {
        printf("  [%d] id=%lld, score=%.2f\n", i, top_ids[i], top_scores[i]);
    }

    reservoir_free(r);
    return 0;
}
```

### Example 2: Adaptive Mode (C)

```c
#include "reservoir.h"

void benchmark_adaptive_mode() {
    ReservoirOpts opts = ReservoirOptsDefault;
    opts.mode = RESERVOIR_MODE_ADAPTIVE;
    opts.adaptive_threshold = 0.75f;

    Reservoir* r;
    reservoir_init(&r, 1000, L2_DISTANCE, &opts);

    // Push 1M candidates
    for (int batch = 0; batch < 1000; batch++) {
        int64_t ids[1000];
        float scores[1000];

        for (int i = 0; i < 1000; i++) {
            ids[i] = batch * 1000 + i;
            scores[i] = (float)(rand() % 10000) / 100.0f;
        }

        reservoir_push_batch(r, ids, scores, 1000, NULL);

        // Check mode switches
        if (batch % 100 == 0) {
            printf("Batch %d: mode=%s, size=%d, switches=%lld\n",
                   batch,
                   r->current_mode == RESERVOIR_MODE_BLOCK ? "Block" : "Heap",
                   r->size,
                   r->mode_switches);
        }
    }

    reservoir_free(r);
}
```

### Example 3: Deduplication (Swift)

```swift
import Foundation

func queryWithDeduplication() {
    let C = 1000
    let nprobe = 10

    var reservoir: UnsafeMutablePointer<Reservoir>?
    reservoir_init(&reservoir, Int32(C), L2_DISTANCE, nil)

    var visited: UnsafeMutablePointer<VisitedSet>?
    visited_set_init(&visited, 1_000_000)

    // Scan multiple IVF lists (may have duplicates)
    for list_id in 0..<nprobe {
        let list = ivf_lists[list_id]

        reservoir_push_batch(reservoir!, list.ids, list.scores,
                           Int32(list.count), visited)
    }

    // Extract unique top-K
    var top_scores = [Float](repeating: 0, count: 10)
    var top_ids = [Int64](repeating: 0, count: 10)
    reservoir_extract_topk(reservoir!, 10, &top_scores, &top_ids)

    print("Top-10 unique results extracted")

    reservoir_free(reservoir!)
    visited_set_free(visited!)
}
```

---

## Summary

**Kernel #39 (Candidate Reservoir Buffer)** is a high-performance intermediate storage mechanism for vector search candidates, providing bounded memory with adaptive buffering strategies.

### Key Characteristics

1. **Purpose**: Buffer candidate (id, score) pairs during scans with O(C) space and O(1) amortized insert
2. **Performance**: 200M-5B candidates/sec depending on mode and acceptance rate
3. **Modes**: Heap (O(log C) insert), Block (O(1) append), Adaptive (automatic switching)
4. **Memory**: 12 × C bytes (C=1000 → 12 KB, fits in L1 cache)

### Optimization Techniques

1. **Adaptive Mode**: Starts with Block mode for O(1) appends, switches to Heap at 75% occupancy
2. **SoA Layout**: Separate score/ID arrays improve cache utilization
3. **SIMD Comparison**: Vectorized threshold checks achieve 2× speedup
4. **Deterministic Ties**: Break score ties by ID for reproducible results

### Integration Points

- **Consumes**: Candidate scores from ADC scan (kernel #22)
- **Produces**: Top-C candidates for top-K selection (kernels #05/#06)
- **Coordinates**: With VisitedSet deduplication (kernel #32), exact rerank (kernel #40)

### Typical Use Case

Scan 1M vectors at 200M candidates/sec, maintaining top-C=1000 in 12 KB reservoir with <1% overhead, then extract top-K=10 for exact reranking with 50× speedup over full sort.

---

## Dependencies

**Kernel #39** depends on:
- **Kernel #05** (topk_partial): Extract top-K from reservoir
- **Kernel #06** (topk_merge): Merge multi-thread reservoir results
- **Kernel #32** (candidate_dedup): Deduplication via VisitedSet

**Kernel #39** is used by:
- **Kernel #22** (ADC scan): Buffers ADC scan results
- **Kernel #40** (exact_rerank): Provides candidates for reranking
- **Kernel #46** (telemetry): Emits reservoir statistics

---

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)

**NEON SIMD Optimization**:
- Vectorized threshold comparison: `vcltq_f32` / `vcgtq_f32`
- 4-wide processing achieves 2× speedup

**Cache Characteristics**:
- L1: 64 KB → C ≤ 5000 for full L1 residency
- L2: 256 KB → C ≤ 20000 for L2 residency
- Recommended: C = 1000-2000 for optimal performance

**Memory Bandwidth**:
- L1: ~200 GB/s → 16B candidates/sec theoretical
- L2: ~100 GB/s → 8B candidates/sec theoretical
- Practical: 5B candidates/sec (Block mode)
