# 33 — HNSW Traversal (Greedy + efSearch)

Version: 1.0 (Apple Silicon focused)

Scope
- Query-time traversal over HNSW graphs: greedy descent from top layers, then efSearch in layer 0 to produce a candidate set for final top‑k.
- Insertion uses the same traversal to find neighbors at each level before linking (see #34).

Dependencies (VectorIndex kernels and utilities)
- #01 L2, #02 IP, #03 Cosine microkernels; #04 ScoreBlock (batch scoring)
- #05/#06 TopK selection/merge; #08 IDFilter (bitset filtering)
- #09 Norm cache (cosine); #39 CandidateReservoir (optional frontier/candidate buffer)
- #49 Prefetch helpers (advisory); Metric semantics from VectorCore.SupportedDistanceMetric

Data Model
- Node IDs: 0‑based dense Int32/Int identifiers, unique per node.
- Layers: 0 = base layer; maxLevel ≥ 0; nodes store neighbors per layer.
- Graph representation for kernels (C‑friendly CSR by layer):
  - For each layer l, arrays: `neighbors_l: Int32[EL]`, `offsets_l: Int32[N+1]`.
  - For node u, its neighbors are `neighbors_l[offsets_l[u] ..< offsets_l[u+1]]`.
  - CSR arrays must be contiguous and 64‑byte aligned.
- Vector storage: contiguous AoS `[N × d]` f32 buffer `xb`, row‑major per node id, 64‑byte aligned.
  - If the index stores vectors sparsely, provide a gather callback that copies a neighbor batch into a small, contiguous staging buffer before scoring (see Performance).

API (C ABI + Swift wrappers)
- C (header suggestion: include/hnsw_traversal.h)
```
typedef enum { METRIC_L2 = 0, METRIC_IP = 1, METRIC_COSINE = 2 } HNSWMetric;

// Greedy descent from maxLevel→1; returns entry node for layer 0.
int32_t hnsw_greedy_descent_f32(
    const float* q, int d,
    int32_t entryPoint, int32_t maxLevel,
    const int32_t* const* offsetsPerLayer,   // per-layer [N+1]
    const int32_t* const* neighborsPerLayer, // per-layer [EL]
    const float* xb, int32_t N,
    HNSWMetric metric,
    const float* optionalInvNorms // for cosine; length N or NULL
);

// efSearch at layer 0; writes up to ef candidate ids sorted by distance asc.
int hnsw_efsearch_f32(
    const float* q, int d,
    int32_t enterL0,
    const int32_t* offsetsL0, const int32_t* neighborsL0,
    const float* xb, int32_t N,
    int ef, HNSWMetric metric,
    const uint64_t* allowBitset /*optional*/, int allowN /*domain size or 0*/,
    int32_t* idsOut, float* distsOut
);

// Convenience: full traversal (greedy + efSearch) returning up to ef candidates.
int hnsw_traverse_f32(
    const float* q, int d,
    int32_t entryPoint, int32_t maxLevel,
    const int32_t* const* offsetsPerLayer, const int32_t* const* neighborsPerLayer,
    const float* xb, int32_t N, int ef, HNSWMetric metric,
    const uint64_t* allowBitset /*optional*/, int allowN /*0 if none*/,
    int32_t* idsOut, float* distsOut
);
```
- Swift wrappers mirror the ABI and use ScoreBlock (#04) internally for batch scoring where possible.

Semantics
- Distance ordering: outputs are sorted by ascending “distance” (smaller is better). Cosine uses 1 − similarity; IP uses −dot if converted to distance.
- ef ≥ k for downstream top‑k. Larger ef increases recall with higher latency.
- Determinism: break ties by node id ascending; visited set is a bitset over [0, N).

Algorithms
1) GreedyDescent (layers maxLevel→1)
   - Start at entryPoint. Repeatedly scan neighbors at current node; if any neighbor is strictly closer to q than the current, move to the best neighbor and repeat. Stop at local minimum. Proceed to next lower layer using this node as the new enter.
2) efSearch (layer 0)
   - Maintain two structures:
     - candidates: min‑by‑distance structure (binary heap or linear min‑scan over a small vector)
     - result: bounded size ef “max‑by‑distance” heap (or sorted vector with tail pop)
   - Initialize with (enterL0, d(q, enterL0)).
   - Loop: pop the current best candidate. If result is at capacity (ef) and bestCandidate.dist > result.worstDist, break (early exit). Otherwise, for each unvisited neighbor v of candidate: compute d(q, v); if filter present, require pass; insert into candidates and, if result not full or dist < worst, into result (trim to ef). Mark visited.

Performance Guidance (Apple Silicon)
- Scoring:
  - Use #04 ScoreBlock to compute distances for a small batch of neighbors at once. Gather up to 32–64 neighbor vectors into a contiguous staging buffer (AoS `[batch×d]`) before calling ScoreBlock.
  - For cosine, prefer fused path with #09 Norm cache (precompute db inv‑norms; compute query inv‑norm once).
  - For IP and Cosine, ScoreBlock has optimized SIMD paths; for L2, it can use precomputed norms when available.
- Batch size: 32 or 64 typically balances L1 bandwidth and call overhead on M2/M3. Tune by d:
  - d ≤ 256: batch 64
  - 256 < d ≤ 1024: batch 32
  - d > 1024: batch 16–32 depending on cache pressure
- Alignment & layout:
  - Ensure `xb` is 64‑byte aligned; prefer compact AoS storage ordered by node id.
  - When adjacency neighbors are not contiguous in storage, gather into staging to reestablish sequential access for ScoreBlock and leverage prefetch.
- Prefetch:
  - Use #49 Prefetch helpers where available; otherwise, software prefetch is optional (Swift lacks portable intrinsic). Group neighbor ids before scoring to improve spatial locality.
- Heaps:
  - For ef ≤ 256, linear vectors with binary‑insert + tail prune can outperform binary heaps, avoiding allocations.
- Expected throughput (guidance, single core):
  - d=768, ef=200, M=16: 2–4 µs per query entry using batch ScoreBlock and gather buffer.
  - Memory is the limiter; prioritize contiguous loads.

Complexity
- Time: O(V · log ef + S) where V is visited edges and S is scoring time with batch kernels.
- Space: O(ef + N/64) for result and visited bitset. Staging buffers: O(batch×d).

Error Handling
- Invalid ids (out of range) → ignore neighbor; optional bounds check flag.
- Dimension mismatch (d≤0 or buffers too small) → error code.
- Filter bitset domain mismatch → treat as no‑pass (drop).

Telemetry Hooks (#46)
- Counters: edges_visited, neighbor_batches, candidates_pushed, early_exits.
- Timings: greedy_ns, efsearch_ns, scoring_ns, total_ns.

Validation & Tests
- Determinism: same inputs/seed produce identical outputs.
- Correctness vs. brute‑force on small graphs.
- Cosine with and without norm cache produces same ranking (within fp tolerance).
- Filter bitset gating behaves as allowlist.
- Stress: random graphs with varying M, ef; dimension sweep.

Integration Notes
- Downstream: feed `idsOut[0..ef)` + `distsOut` to #05/#06 for final top‑k, or directly slice first k.
- Insertion: use this traversal per layer to collect candidate set for neighbor selection (#34).

