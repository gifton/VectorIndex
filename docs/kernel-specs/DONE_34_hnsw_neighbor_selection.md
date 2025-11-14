# 34 — HNSW Neighbor Selection (Diversity + Prune)

Version: 1.0 (Apple Silicon focused)

Scope
- During insertion of a new node x_new, select up to M neighbors among candidates at layer l using HNSW’s diversity heuristic, then prune neighbor lists to enforce the degree cap M. Ensure symmetric linking.

Dependencies
- Candidate generation via traversal (#33)
- Distance microkernels #01/#02/#03 and #04 ScoreBlock (batch scoring)
- Optional: #05 partial TopK for fast top‑M; #09 Norm cache for cosine

Inputs
- x_new: f32[d] query vector (new node)
- candidates: set/vector of node ids (from #33 efSearch at layer l)
- vectors: contiguous AoS f32 buffer `[N × d]` (64‑byte aligned) or accessor/gather callback
- metric: L2/IP/Cosine (ranking semantics below)
- M: max neighbors per node per layer
- layer l: 0..maxLevel for where selection applies

Outputs
- selected: up to M node ids for edges (x_new ↔ id)
- pruned neighbor lists for all nodes touched (degree ≤ M), preserving diversity

Semantic Notes
- Ranking: smaller “distance” is better. For Cosine, use 1 − sim. For IP, use −dot if converting to distance; alternatively select by sim with reversed comparisons (preferred: convert to unified distance for consistency).
- Determinism: stable tie‑breaks (id asc). Exclude self and deleted nodes. Only consider neighbors whose level ≥ l.

Algorithms
1) Score candidates
   - Compute d(x_new, c) for all c ∈ candidates using #04 ScoreBlock in batches (32–64). For cosine, use #09 norm cache.
   - Produce an array `C = [(id, dist)]` sorted by increasing dist. For large candidate sets, use #05 partial TopK with k=M to reduce sort cost, but retain a short overflow buffer (e.g., 2×M) to improve diversity.

2) Diversity selection (SelectNeighborsHeuristic)
   - Initialize `selected = []`.
   - Iterate over `C` in ascending `dist` order. For each candidate c:
     - Accept c if ∀ s ∈ selected: d(c, x_new) ≤ d(c, s). This promotes angular spread (from HNSW paper).
     - Otherwise skip c.
     - Stop early if `selected.count == M`.
   - If `selected.count < M`, fill with nearest remaining from `C` (skipping already selected) until size == M or `C` exhausted.
   - Implementation detail: evaluating d(c, s) on the fly is O(|selected|). Since |selected| ≤ M (typically M ≤ 32), this is cheap. Use #04 ScoreBlock with a tiny batch to amortize when checking 2–4 candidates at a time.

3) Symmetric linking and pruning
   - Add edges bidirectionally: x_new↔s for all s ∈ selected.
   - For each affected node u ∈ {x_new} ∪ selected:
     - Dedupe the neighbor list at layer l; compute d(anchor, v) for all v; keep the nearest M (stable tie‑break by id). This is pruneNeighbors.
     - For pruning, use #05 partial TopK when degree ≫ M; otherwise small sort is fine.

Performance Guidance (Apple Silicon)
- Scoring:
  - Prefer #04 ScoreBlock with staging gathers of size 32–64 for candidate scoring. For `d≥256`, batch 32 or 64; for `d>1024`, batch 16–32.
  - Cosine: build/use #09 norm cache once per index build/update cycle. Compute query inv‑norm once.
- Selection:
  - When `|candidates| ≤ 2×M`, full sort is acceptable. When larger, use #05 partial TopK for the first M, then do the diversity pass on a 2×M window.
- Pruning:
  - Neighbor degree is typically small (≤2×M) after insertion. Simple sort by distance is often fastest. Avoid repeated allocations.
- Alignment & locality:
  - `vectors` AoS `[N×d]` 64‑byte aligned. For scattered ids, gather into a staging buffer to feed ScoreBlock and exploit contiguous reads.
- Expected:
  - d=768, M=16, |candidates|≈ef=200: scoring + selection in ~1–3 µs/core with batch ScoreBlock.

Complexity
- Scoring: O(|candidates| · d / SIMD) amortized with batch kernels.
- Diversity: O(M · |candidates|) worst‑case for checks, but |selected| ≤ M.
- Pruning: O(degree · log degree) or O(degree) with #05.

Error Handling
- Skip candidates with invalid ids or wrong layer; tolerate missing vectors by dropping that candidate.
- M ≤ 0 → empty selection; d ≤ 0 → error.

Telemetry Hooks (#46)
- Counters: candidates_in, candidates_scored, accepted, pruned_edges, symmetry_links.
- Timings: score_ns, select_ns, prune_ns.

Validation & Tests
- Compare against naïve exact graph construction (top‑M by distance only) for recall and graph metrics.
- Determinism under stable seeds and tie‑breaks.
- Cosine/IP/L2 parity on synthetic data.
- Bounds: M sweep {8,16,32}; d sweep; candidate set size sweep.

Implementation Interface (suggested)
```
// C ABI (neighbor selection for one layer)
int hnsw_select_neighbors_f32(
    const float* x_new, int d,
    const int32_t* candidates, int candCount,
    const float* xb, int32_t N,
    int M, int layer,
    HNSWMetric metric,
    const float* optionalInvNorms, // cosine
    int32_t* selectedOut           // length ≥ M
);

// C ABI (prune for an anchor u at layer l)
int hnsw_prune_neighbors_f32(
    int32_t u, const float* xb, int d,
    const int32_t* offsetsL, const int32_t* neighborsL,
    int M, HNSWMetric metric,
    const float* optionalInvNorms,
    int32_t* prunedOut // length ≥ M; also returns new degree
);
```
<!-- moved to docs/kernel-specs/ -->
