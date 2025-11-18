ID: 40 — Exact Re‑rank on Top‑C (original vectors) (MUST)

Purpose
- Compute exact scores for a query against a small candidate set of original vectors and return the top‑K; used after ANN scan (e.g., IVF‑PQ) to improve accuracy.

Role
- Q

Signatures
- Scores only (single query):
  - `void rerank_exact_scores(const float* q, int d, Metric metric, const int64_t* cand_ids /*[C]*/, int C, const VectorReader* reader, const RerankOpts* opts, float* scores_out /*[C]*/);`
- Top‑K (single query):
  - `void rerank_exact_topk(const float* q, int d, Metric metric, const int64_t* cand_ids /*[C]*/, int C, int K, const VectorReader* reader, const RerankOpts* opts, float* top_scores /*[K]*/, int64_t* top_ids /*[K]*/);`
- Batched (optional):
  - `void rerank_exact_topk_batch(const float* Q /*[b*d]*/, int b, int d, Metric metric, const int64_t* cand_offsets /*[b+1]*/, const int64_t* cand_ids /*[cand_offsets[b]]`, int K, const VectorReader* reader, const RerankOpts* opts, float* top_scores /*[sum b*K]*/, int64_t* top_ids /*[sum b*K]*/);`

Inputs / Outputs
- `cand_ids`: internal dense IDs (from #50). `C`: candidate count; `K ≤ C`.
- `reader`: abstraction over where original vectors live (see below). `scores_out`: aligned with `cand_ids`. `top_*`: best‑first ordered.

Metric
- `Metric ∈ {L2, IP, Cosine}`. Cosine uses #02 dot with #09 norms (base inv‑norms + query inv‑norm).

VectorReader (storage backends)
- DenseArray: contiguous matrix `xb /*[N][d]*/` where `id == row`.
- IVFListVecs: per‑list arrays `vecs[list] /*[len[list]][d]*/` with `idmap` (#50) mapping `id -> (list, offset)`.
- Callback: user callback to fetch/gather rows into a provided buffer: `size_t fetch(const int64_t* ids, int n, float* dst /*[n*d]*/);` Must return number of rows written; used for external stores.

Options (RerankOpts)
- `backend`: {DenseArray, IVFListVecs, Callback}.
- `gather_tile`: number of rows to gather into a contiguous staging buffer per iteration (typ. 32–256).
- `reorder_by_segment`: group `cand_ids` by physical locality before gather (stable within groups; default on).
- `have_inv_norms`: for Cosine, base `inv_norms /*[N]*/` available; else compute on the fly via #09 for gathered rows.
- `have_sq_norms`: for L2, base `||x||^2` available to enable MIPS→L2 trick (#01/#02 path) when profitable.
- `return_sorted`: ensure outputs sorted best‑first (default true). If false, caller may sort/merge later.
- `skip_missing`: when a vector is missing/tombstoned (#43), skip it; optionally write `+inf/−inf` into `scores_out`.
- `prefetch_distance`: cache prefetch tuning for source arrays; `strict_fp` toggles reassociation.

Constraints
- d ∈ {512,768,1024,1536} optimized; generic path pads to multiple of 16 for NEON.
- Typical ranges: C ∈ [50, 10k], K ∈ [1, 1024]. Best perf for C ≤ 2k.

Algorithm (single query)
- Precompute per‑query auxiliaries:
  - Cosine: `q_inv_norm = query_inv_norm_f32(q, d, eps)` (#09).
  - L2 with dot‑trick enabled and `have_sq_norms`: compute scalar `q_norm2` once.
- Partition `cand_ids` into gather tiles of size `T = gather_tile`.
  1) Gather stage:
     - For each tile, resolve source addresses via `reader` and copy into a contiguous scratch `xb_tile /*[T][d]*/` (64‑B aligned). If Cosine and `have_inv_norms`, gather `inv_norms_tile /*[T]*/`; else plan to compute.
  2) Score stage:
     - Call #04 `score_block(q, xb_tile, T, d, scores_tile, metric, xb_aux, q_aux)` where:
       - L2: `xb_aux = have_sq_norms ? sq_norms_tile : nullptr`, `q_aux = have_sq_norms ? q_norm2 : 0`. Auto‑select direct L2 vs dot‑trick per d.
       - IP: `xb_aux = nullptr`, `q_aux = 0`.
       - Cosine: `xb_aux = inv_norms_tile` (or compute via #09 on the tile), `q_aux = q_inv_norm`.
     - Write `scores_tile` into `scores_out` aligned with original `cand_ids` order.
- If `return_sorted` and `K < C`: run #05 `topk_partial(scores_out, cand_ids, C, K, &heap)` then output sorted results; else just copy and optionally sort.

Gather backends
- DenseArray: copy `xb[id]` rows directly; hardware prefetch by sequentializing ids via `reorder_by_segment` (bucket by `id >> page_bits`).
- IVFListVecs: resolve `(list, off)` via `idmap`; optionally group by `(list)` to maximize contiguous reads; gather segments per list.
- Callback: invoke `fetch` for current tile; if fewer than requested returned, compact and proceed.

Vectorization & Tiling
- Gather is memcpy‑like; use 16‑float vector loads/stores and prefetch next address in a small SW queue.
- Score uses #01/#02/#03 via #04; keep `q` in registers, tile T so `xb_tile` fits in L1.

Parallelism
- Single query: parallelize over tiles with per‑thread scratch and final atomic write into `scores_out` slots; then run a thread‑local top‑K or merge via #06 if sharded.
- Batched: parallel over queries; share read‑only source; per‑thread scratch buffers sized `T*d`.

Numeric
- f32 compute. Cosine with f16 inv‑norms: widen to f32. Deterministic tie‑break by id when selecting top‑K.

Missing / Tombstoned IDs (#43)
- If `skip_missing`: elide those candidates (reduce effective C). Else set sentinel scores (`+inf` for L2, `-inf` for IP/Cos) and keep the id.

Telemetry (#46)
- Candidates processed (C), top‑K (K), backend used, gather tiles, bytes copied, bytes scored, cache locality (avg segment length), path flags (dot‑trick, inv‑norms from cache vs on‑the‑fly), cycles per candidate.

Tests
- Correctness: compare `scores_out` to scalar reference over random ids and vectors for all metrics; verify ties stable by id.
- Equivalence: when `cand_ids` are contiguous rows, compare output to direct #04 call over the same block.
- Cosine paths: precomputed vs on‑the‑fly inv‑norms must match within tolerance (1e‑4 with f16 inv‑norms).
- Missing IDs: behavior matches `skip_missing` option; top‑K excludes sentinels when skipping.
- Performance: measure throughput vs C and T; ensure near‑memcpy gather cost and ≥85% of #04 throughput once in `xb_tile`.

Reuse / Integration
- Uses #04 scoring, #05/#06 top‑K, #09 for norms; relies on #50 for dense ID mapping and optionally list offsets for IVF‑Flat storage; interacts with #43 tombstones.
<!-- moved to docs/kernel-specs/ -->
