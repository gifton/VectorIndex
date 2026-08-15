# Baseline: 0.2.0 pre-Phase-3 (QUIET-MACHINE REQUALIFICATION)

Captured 2026-08-15 from a throwaway worktree at `8a512b2` (= `main` @ e71daae +
Phase-3 Task-1 harness-only changes; verified by `git diff --stat` — zero library
code differs from e71daae). Same six capture commands as
`.bench/baseline-0.2.0-pre-phase3/` (see that README for the exact CLI per file).

## Why this exists

The original 2026-07-31 baseline was captured while ~72 orphaned `yes` processes
had consumed ~16 cores for a week (load ≈ 290/16 cores). All ABSOLUTE times in
it are inflated 3-6x; only back-to-back relative deltas were ever trusted.
(A separate, later capture attempt on 2026-08-08 was additionally corrupted by
machine-sleep and a thermal-emergency sleep and was discarded entirely — that
episode postdates the Jul-31 baseline, which was load-contaminated only.) This directory replaces it as the absolute-number reference.
Phase-3's per-task gates were adjudicated against the old baseline's RELATIVE
deltas and remain valid; see the audit record in the Phase-3 ledger.

## Machine-quiet attestation

- Xcode closed (verified by process check), AC power, lid open, display allowed
  to sleep; every capture wrapped in `caffeinate -dims`.
- No-op `swift build` probe: 0.33 s (gate: <10 s). Load avg at capture ≈ 3.3
  (residual UI apps only; the contaminated era ran at ≈ 290).
- Cold release build of the harness took 79 s in this environment (same build:
  447 s+ compute under contamination) — the machine-effect yardstick.

## Provenance cross-checks (data-level, not just path-level)

- `ivf_search.json` recallAvg = 0.877 — a draw from the PRE-Task-8
  nondeterminism lottery (0.72–1.0), impossible on post-Phase-3 code (which is
  bit-identical at 0.9565000000000008). Confirms pre-Phase-3 binary.
- `mmap_append.json` shows the quadratic signature (commits/s halves as commits
  double: 544.7 → 291.8 → 151.9 → 77.2), impossible on post-Phase-3 near-linear
  code. Confirms pre-Phase-3 binary.
- `hnsw_search.json` recallAvg = 0.4145 and knn recall_at_k = 0.9556 / 0.9992
  match the contaminated-era values EXACTLY — correctness metrics are
  load-independent, as expected.

## Key numbers (quiet machine)

| Metric | Quiet value | (Contaminated-era value) |
|---|---|---|
| HNSW buildSeconds | 5.049 | (30.97) |
| HNSW recallAvg | 0.4145 | (0.4145 — exact match) |
| HNSW throughputQps / batchThroughputQps | 5393.7 / 16037.1 | (547.96 / 2922.29) |
| IVF optimizeSeconds | 0.0633 | (0.374) |
| IVF recallAvg | 0.877 (nondeterministic draw; pre-fix code) | (0.994 — also a draw) |
| knn uniform recall_at_k / insert_sec | 0.9556 / 2.779 | (0.9556 / 17.28) |
| knn clusters(8) recall_at_k / insert_sec | 0.9992 / 1.591 | (0.9992 / 9.65) |
| mmap append @1000/2000/4000/8000 | 544.7 / 291.8 / 151.9 / 77.2 c/s | (480.8 / 258.8 / 131.2 / 63.5) |

HNSW recallAvg 0.4145 is the known low-recall regime at these ef/M settings
(recorded as-is, not re-tuned — matches the original baseline's note).
