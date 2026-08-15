# Baseline: 0.2.0 pre-Phase-3

Captured on `main` @ e71daae + Phase-3 Task-1 harness changes (harness-only;
no library code differs from e71daae). This supersedes `.bench/baseline-0.1.6/`
as the Phase-3 gate reference: it is post-A9 (topology fix) and post-Phase-2,
records host info inside each JSON, includes batchSearch QPS and the mmap
append sweep, and pins `--knn-clusters 8` (recorded in-file; the 0.1.6 value
was never recorded).

Gate rule: Phase-3 items compare against THESE numbers, same machine (see the
`host` block in each JSON), Release build, quiet machine.

Host (from the `host` block in each JSON): Apple M3 Max, Mac15,9, 48GB, macOS 26.5.2.

## Files

- `flat_search.json` — `--index flat --n 5000 --q 200 --dim 384 --k 10 --metric euclidean`
- `hnsw_search.json` — `--index hnsw --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --m 16 --efc 200 --efs 64`
  (also contains the always-built Flat baseline used for recall measurement)
- `ivf_search.json` — `--index ivf --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --nlist 64 --nprobe 4`
  (also contains the always-built Flat baseline used for recall measurement)
- `knn_graph_uniform.json` — `--knn-graph --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42`
- `knn_graph_clusters.json` — `--knn-graph --knn-clusters 8 --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42`
- `mmap_append.json` — `RUN_BENCHMARKS=1 MMAP_BENCH_OUT=... swift test -c release --filter MmapAppendBenchmark`.
  Measures per-commit whole-section CRC cost at final (pre-sized) container size — a faithful
  quadratic proxy, not a growth-path benchmark (the growth path has pre-existing defects tracked
  separately).

## Key numbers

| Metric | Value |
|---|---|
| Flat buildSeconds / recallAvg / throughputQps | ~0.0009s / 1.0 / ~500-580 qps |
| HNSW buildSeconds | 30.97s |
| HNSW recallAvg | 0.4145 |
| HNSW throughputQps / batchThroughputQps | 547.96 / 2922.29 |
| IVF optimizeSeconds / recallAvg | 0.374s / 0.994 |
| knn-graph uniform recall_at_k / insert_sec | 0.9556 / 17.28s |
| knn-graph clusters(8) recall_at_k / insert_sec | 0.9992 / 9.65s |
| mmap append @1000/2000/4000/8000 commits | 480.8 / 258.8 / 131.2 / 63.5 commits/s |

HNSW recallAvg (0.4145) sits in the same low-recall regime as the pre-existing ~0.41 baseline
at these ef/M settings; it is recorded here as-is (not re-tuned) since Phase-3 gates against this
exact number.
