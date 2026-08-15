# Post-Phase-3 sweep (quiet machine)

Captured 2026-08-15 on `gifton/perf-0.2.0-phase3` @ 84df7d7 (Tasks 1-15 + audit
fold-ins 16a/16b/16c). Same six capture commands as the baseline (see
`.bench/baseline-0.2.0-pre-phase3/README.md` for exact CLI); compare against
`.bench/baseline-0.2.0-quiet/` (same machine, same session, same protocol —
the only clean absolute-number comparison in the phase record).

## Machine-quiet attestation

Same session and protocol as `.bench/baseline-0.2.0-quiet/`: Xcode closed, AC
power, `caffeinate -dims` per capture, no-op build probe 0.33 s. Release builds
took 79 s (both trees) — same thermal/frequency envelope on both sides.

## Methodology notes (read before quoting)

- **Single-run captures are noise-dominated for search throughput.** The raw
  single-run sweep showed HNSW single-query −31% and IVF −18% vs quiet
  baseline; interleaved median-of-3 per side showed **−1.3% and +2.7%
  (parity)** respectively. Both sides also
  show a consistent first-run-after-idle boost (~18% faster than runs 2-3).
  Where this README quotes a delta, it says which methodology produced it.
- IVF `recallAvg` is NOT comparable as a delta: the baseline binary predates the
  Task-8 determinism fix, so its value is one draw from a 0.72–1.0 lottery
  (this capture drew 0.877). Post-Phase-3 is bit-identical across processes at
  0.9565000000000008.

## Quiet-machine A/B vs `.bench/baseline-0.2.0-quiet/`

| Metric | Baseline (quiet) | Post-P3 (quiet) | Delta | Methodology |
|---|---|---|---|---|
| HNSW buildSeconds | 5.572 | 4.998 | **−10.3%** | median-of-3 per side |
| HNSW recallAvg | 0.4145 | 0.4145 | bit-identical | deterministic gate |
| HNSW single-query QPS | 3493 | 3448 | −1.3% (parity) | median-of-3 |
| HNSW batch QPS | 16844 | 17080 | +1.4% (parity) | median-of-3 |
| IVF optimizeSeconds | 0.0666 | 0.0622 | **−6.6%** | median-of-3 |
| IVF single-query QPS | 1049 | 1077 | +2.7% (parity) | median-of-3 |
| IVF batch QPS | 323.2 | 337.4 | +4.4% (parity) | median-of-3 |
| IVF recallAvg | 0.877 (lottery) | 0.9565000000000008 | now deterministic | see note above |
| knn uniform recall_at_k | 0.9556 | 0.9556 | exact | deterministic |
| knn clusters recall_at_k | 0.9992 | 0.9992 | exact | deterministic |
| knn insert_sec (uniform/clusters) | 2.779 / 1.591 | 2.724 / 1.676 | n=1, inside noise | single-run; see note |
| mmap append @1000 | 544.7 c/s | 9412.0 c/s | **17.3×** | full sweep, both sides |
| mmap append @8000 | 77.2 c/s | 10265.1 c/s | **133×** | full sweep, both sides |
| mmap shape | quadratic (halves per doubling) | near-flat (slightly rising) | **quadratic eliminated** | 4-point sweep |

Development-era gates (same-load back-to-back A/B under the contaminated
environment, recorded in the Phase-3 ledger) measured larger deltas on some
items (e.g. HNSW build −12.4%, IVF search −14% latency / 5-run). Those remain
valid as relative gates for what they measured; where a quiet median exists it
supersedes for quotation. knn insert (−14%/−8% same-load) has no quiet
multi-run measurement — treat as development-era evidence only.

Reservoir mode benchmark (quiet, median-of-7 per cell, for the
`ReservoirOptions.mode` doc guidance): ascending/random streams — heap/adaptive
12–26 ns/push vs block 90–487 (block 6–19× slower); descending — all three
within ~1.1–1.7×; block fastest in 0 of 18 cells; adaptive tracks heap within a
few percent and switches exactly once.
