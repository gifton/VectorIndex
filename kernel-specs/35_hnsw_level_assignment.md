# 35 — HNSW Level Assignment (Randomized)

Version: 1.0 (Apple Silicon focused)

Scope
- Assign a hierarchical level to each inserted node in HNSW using a geometric‑like distribution controlled by M. This governs graph height and sparsity.

Background
- HNSW samples a level from an exponential tail so that the number of nodes per layer decreases geometrically. A common parameterization is P(level ≥ L) ∝ exp(−L · λ). We set λ = 1 / log(M) to tie sparsity to the configured max degree M.

Inputs
- M: max connections per node per layer (M ≥ 2); influences decay λ = 1/log(M)
- cap: integer ≥ 0; maximum allowed level (e.g., 16) to bound memory/time
- rng: deterministic RNG with seed+stream (see S2 RNG or RNGState.swift)

Output
- level: integer in [0, cap]

Distribution & Derivation
- Draw u ~ Uniform(0,1), map to L = floor(−log(u) / log(M)).
- Intuition: Larger M → larger log(M) → smaller λ → smaller expected L (shallower graphs as degree increases).
- Enforce bounds: min(L, cap).

Algorithm (reference)
```
// Inputs: M (>=2), cap (>=0), RNG with nextFloat()->[0,1)
float logm = logf((float)M);
if (logm <= 1e-9f) logm = 1e-9f;     // avoid div-by-zero for M pathologies
float u = max(1e-9f, min(1.0f - 1e-9f, rng.nextFloat())); // clip away from 0/1
int L = (int)floorf(-logf(u) / logm);
if (L > cap) L = cap;
return L;
```

Apple Silicon Notes
- The computation is negligible cost; favor branchless clipping and `logf()`.
- Keep RNG state in registers; avoid locks by using split streams per worker thread.

RNG Requirements
- Determinism across runs: given (seed, stream, index), the returned level is identical.
- Use S2 RNG (xoroshiro128** or Philox) or the existing `RNGState` (LCG) for a baseline. Prefer S2 for higher quality when available.
- For parallel insertion, derive per‑worker stream ids using a stable mapping (e.g., baseStream + workerId).

API (C + Swift)
```
// C ABI
int hnsw_sample_level(int M, int cap, uint64_t* rngState);

// Swift
@inlinable public func hnswSampleLevel(M: Int, cap: Int, rng: inout RNGState) -> Int
```

Validation & Tests
- Distribution sanity: simulate 1e6 samples and verify empirical P(level ≥ L) follows exp(−L/log(M)) within tolerance.
- Edge cases: M={2,16,32}, cap={0,1,16}; seed reproducibility; stream independence.

Integration
- Called once per insertion to determine top layer of the new node.
- Combined with traversal (#33) and neighbor selection/pruning (#34) per level.

