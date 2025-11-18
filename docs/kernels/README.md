# Kernel Overview

This folder contains CPU-first kernels used throughout VectorIndex for scoring, selection, training, and data layout. Kernels are written in Swift with optional C/Accelerate shims when that materially improves performance.

Key components (non-exhaustive):
- #04 ScoreBlock: batched distances/similarities (L2^2, Dot, Cosine)
- #05 TopK Selection: deterministic top-K with ordering by metric
- #11 K-means++ Seeding: D² sampling (dense and streaming variants)
- #12 Mini-batch / Streaming K-means: online updates and maintenance
- #19 PQTrain: Product Quantization codebook training (Lloyd + mini-batch)
- #23 Residual Kernel: residuals r = x − c for IVF-PQ
- #48 Layout Transforms: AoS ↔ AoSoA vector and PQ code interleave

## PQTrain (#19)

Trains Product Quantization codebooks per subspace. Two algorithms are available:
- Lloyd (batch): standard k-means with per-iteration recompute.
- Mini-batch: online incremental means with pass-level empty repair.

Defaults are conservative and tuned for correctness and CI-friendliness. For larger runs, you can adjust sampling and repair knobs via `PQTrainConfig`.

### PQTrainConfig (relevant fields)

- algo: `.lloyd` | `.minibatch` (default: `.lloyd`)
- maxIters: training passes/iterations (default: 25)
- tol: relative improvement threshold for early stop in Lloyd (default: 1e-4)
- batchSize: mini-batch size (default: 512 for mini-batch; otherwise unused)
- sampleN: rows per pass for mini-batch (0 ⇒ use `distEvalN` as pass budget)
- emptyPolicy:
  - Lloyd default: `.split` (if not overridden)
  - Mini-batch default is forced to `.reseed` for inexpensive repair
- precomputeXNorm2: gate for dot-trick in Lloyd (default: false)
- computeCentroidNorms: output centroid norms (default: true)
- numThreads: 0/auto, or set 1 for strict determinism
- verbose: prints high-level steps (default: false); DEBUG-only diagnostics remain independent
- warmStart: reuse provided codebooks as initial centroids when sizes match (default: false)
- distEvalN: sample size for distortion estimate if `sampleN == 0` (default: 2000)
- repairEvalN: mini-batch pass-level empty repair sample if `sampleN == 0` (default: 2000)
- streamingRepairEvalN: streaming pass-level empty repair sample (default: 512)

### Behavior highlights

- Mini-batch stability
  - Incremental running means per centroid (global counts persisted across batches).
  - Per-pass, bounded-cost empty repair using a sampled farthest-point strategy.
  - No per-batch reseed/split; empties are repaired once per pass.
- Lloyd correctness
  - Dot-trick is optional and recomputes centroid norms every iteration.
  - Negative distances (from numerical error) are clamped before accumulation.
- Seeding
  - Dense (single-chunk) flow: if training subset is large, seeding is capped to ~4×ks.
  - Streaming flow: seeding over chunks is also capped to ~4×ks using a global subset.
- Streaming training
  - Each pass thins rows across chunks toward a global pass budget (`sampleN` when set).
  - Pass-level empty repair uses `streamingRepairEvalN` samples (default 512).
- Determinism
  - All randomness comes from a split Xoroshiro128 stream; keep `numThreads <= 1` for bit‑exact runs.

### Minimal usage example

```swift
var cfg = PQTrainConfig()
cfg.algo = .minibatch
cfg.batchSize = 512
cfg.sampleN = 2000         // optional; else uses distEvalN
cfg.repairEvalN = 2000     // pass-level empty repair sample (minibatch)
cfg.streamingRepairEvalN = 512
cfg.precomputeXNorm2 = false
cfg.verbose = false

var codebooks: [Float] = []
var norms: [Float]? = []
let stats = try pq_train_f32(
  x: xb,
  n: Int64(xb.count / d),
  d: d,
  m: m,
  ks: ks,
  cfg: cfg,
  codebooksOut: &codebooks,
  centroidNormsOut: &norms
)
```

### Practical defaults

- Mini-batch: `batchSize=512`, `sampleN≈2000` (or use `distEvalN`), `emptyPolicy=.reseed`.
- Lloyd: `precomputeXNorm2=false` (use direct L2²), `emptyPolicy=.split` unless overridden.
- Streaming: pass-level empty repair uses `streamingRepairEvalN=512`; seeding is capped to ~4×ks.

### Notes

- DEBUG builds include safety assertions and progress logs (centroid finiteness, empty stats, dot‑trick sanity). These are disabled in release unless `verbose` is set (which only enables high‑level prints).
- On highly clustered data, consider increasing `repairEvalN` or performing a short Lloyd warm-up before mini‑batch for better initialization.
- Training stats include `warmStartSubspaces` indicating how many subspaces reused initial codebooks when `warmStart` is enabled.
