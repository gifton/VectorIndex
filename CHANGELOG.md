# Changelog

All notable changes to VectorIndex are documented here. Versions follow the 0.x
convention: the minor digit signals breaking changes.

## [Unreleased] — 0.2.0

### Performance

Phase 3 (P1–P7 + carried items). Numbers below are quiet-machine measurements
(2026-08-15, M3 Max, `caffeinate`, attested in `.bench/baseline-0.2.0-quiet/`
and `.bench/post-phase3/` READMEs); items marked *dev-gate* were measured as
same-load back-to-back A/Bs during development and have no quiet-machine
multi-run requalification.

- **mmap ingestion: quadratic eliminated.** Deferred section CRCs (commits skip
  per-section CRC recomputation; `flush()`/unclean-open replay recompute) plus
  ranged page-aligned `msync`. Append throughput at the 4-point sweep went from
  544.7/291.8/151.9/77.2 commits/s (halving per size doubling) to
  9412/10030/10212/10265 (near-flat) — **17–133×** depending on container size.
- **HNSW build −10.3%** (median-of-3 per side: 5.572 s → 4.998 s at n=5000,
  d=384, M=16, efC=200): `searchLayer` rewritten on a (distance, arrival-order)
  binary heap with batched candidate scoring. Graphs are **bit-identical** to
  the previous implementation (determinism-gated; recallAvg 0.4145 exact match)
  and single-query/batch search throughput is at parity (−1.3%/+1.4%,
  median-of-3). knn-graph insert measured −14%/−8% as *dev-gate*.
- **IVF `optimize()` −6.6%** (median-of-3: 66.6 ms → 62.2 ms at n=5000,
  nlist=64) via single-pass materialization and assignment reuse — and now
  **deterministic across processes** (sorted store materialization; recall
  bit-identical at 0.9565000000000008 where the previous implementation drew
  randomly from a 0.72–1.0 range per process). Cosine/dot list assignment is
  GEMM-batched and tiled (bounded ~32 MiB transient, replacing an O(n·k)
  allocation). IVF search latency measured −14% / QPS +16% as *dev-gate*
  (5-run); quiet-machine single-query is at parity-to-slightly-better (+2.7%).
- **IVF batch centroid probes** via `cblas_sgemm` with exact score parity
  (60/60 + 40/40 including d=384); batch-QPS gains at nlist=64 are
  noise-dominated (honest 5-run mean +0.8%) — the win is structural headroom at
  larger nlist, not a headline number.
- **PQ training:** SIMD `l2Sq` (dual-SIMD4, 7.28× micro-benchmark, *dev-gate*)
  and the streaming k-means++ seeder reduced O(n·ks²) → O(n·ks), bit-identical
  seeding.
- **Allocation hoists** on ScoreBlock's fallback path, RangeQuery early-exit,
  ExactRerank batch scratch, and JournalFilter's per-call date formatters
  (thread-local caching), with first-ever direct test coverage for the first
  two paths.
- Cosine F16 inv-norms now delegate to the shared 16-wide `l2NormSquared`
  (removes the sixth hand-rolled sum-of-squares copy; parity-gated).
- Reservoir mode guidance re-measured on quiet hardware and documented on
  `ReservoirOptions.mode`: heap/adaptive are ~6–19× faster than block on
  early-stabilizing streams; all modes within ~1.7× on monotonic-improving
  streams; block fastest in 0 of 18 cells.

### Changed

- Telemetry consolidated onto the push-callback recorders and the dedup pull API,
  now accuracy-tested; the never-compiled `VINDEX_TELEM` histogram singleton and the
  internal `IDMapOpts.enableTelemetry` flag are gone outright. The public
  `VisitedOpts.enableTelemetry` flag (never read, gates nothing) is deprecated rather
  than deleted, since it's genuinely public API — non-breaking constraint. (B1, re-scoped)
- `CS2RNG` C target removed; its unique test coverage ported to the pure-Swift RNG API. (B2)
- `L2SqrMicrobench` dev target and stray `PQTrain.swift.new`/`.tmp` files removed. (B4)
- Dead/no-op internals deleted across kernels: `DispatchBK`, alignment/prefetch no-ops,
  the tiled-assign wrapper, unused `selectNeighbors`/`selectBatchSize`, unreachable
  `sumSquares`. (B5–B7, B10, B20)
- One sum-of-squares implementation (`Norms.l2NormSquared`); scalar distance-kernel
  families unified. (B8, B10)
- MIPSTransform internals routed to the canonical scoring kernels; its dead public
  surface is deprecated pending the 0.2.0 breaking phase. (B3)
- mmap tidy: single `CRC32`, shared disk-layout structs, in-place header hashing,
  WAL append-record CRC validated on replay. (B11, B13)
- `IDMap` keeps only the SwissTable backend. (B12)
- `IDFilter`/`CandidateReservoir`/sparse-paged dedup refactored under new
  characterization tests (previously untested); `BoundedTopKHeap` replaces the
  duplicated min/max heaps. (B14–B16)
- Rerank pointer smuggling now uses the shared `UnsafeSendable` box; HNSW neighbor
  selection and compaction stop reallocating distance arrays per comparison; IVF's
  `optimize`/`optimizeKMeans` are unified behind a new `optimize(maxIterations: Int = 20)`
  overload that carries the single real implementation (fixing `optimizeKMeans`'s
  missing `idToListIndex` population), with the zero-arg `optimize()` kept as the
  `VectorIndexProtocol` witness, forwarding to it (protocol requirement unchanged);
  HNSW init/metric/context-storage tidy. (B17–B19)

### Fixed

- Reservoir `.adaptive` mode now actually works: `currentMode` no longer collapses to
  the initial strategy at init (which made the block→heap switching logic structurally
  unreachable and left default-constructed reservoirs in pure Block mode forever); the
  adaptive block phase gained the same overflow-prune guard `.block` has (previously it
  grew the buffer without bound via the defensive-grow path); filling the buffer now
  completes the switch (the sampled occupancy check alone can miss for common shapes);
  and `reset()` re-seeds the mode before sizing buffers. `adaptiveInitialMode ==
  .adaptive` is now a precondition failure.

### Deprecated

- Scheduled for removal in the 0.2.0 breaking phase: the entire `MIPSTransform` public
  surface (9 symbols; dead, zero callers); `hnsw_prune_neighbors_f32_swift` and its
  `@_cdecl` shim; `IDMap`'s `.robinHood`/`.linearProbing` cases (now silently resolve to
  SwissTable); the array-based `topKIVF` overload, `scoresIVF`, and
  `IVFPostADC.rerankTopKFlat`; `RerankOpts.returnSorted`; `VisitedOpts.enableTelemetry`;
  and the 9 dead `Telemetry` public shells (`QueryCtx`, `TelemetryConfig`,
  `TelemetryGlobal`, `TelemetryCounter`, `TelemetryBytes`, `TelemetryDoubleField`,
  `TelemetryU64Field`, `TelemetryTimerGuard`, `TelemetryTimerToken`).

### Removed
<!-- breaking removals appended per task -->

## [0.1.7] - 2026-07-25

Correctness release: the nine fixes (A1–A9) from the 0.2.0 cleanup effort's
Phase 1, each guarded by a regression test. Non-breaking; the 0.2.0 cleanup
(Phases 2–5) continues on top of this.

### Fixed

- Durable `IVFListHandle.getListStats` returns real stats (was always throwing). (A2)
- `HNSWIndex.batchRemove` no longer corrupts the index on subset removal. (A5)
- Exact rerank preserves 64-bit candidate ids (was truncated to Int32); equal-score ties break deterministically by smallest id (full width); the top-k selection heap is freed instead of leaked. (A4)
- Correct TOC field offsets on the mmap grow/remap path. (A3)
- TOC region reservation covers all entries at the packed 36-byte size — previously the optional IDMap entry was written into unreserved space, clobbering list 0's descriptor whenever `includeIDMap: true`. (A7)
- Dedup forces a full clear when the touched-word ring saturates. (A8)
- PQ centroid squared-norm buffer is freed instead of leaked. (A6)
- HNSW traversal CSR pointers no longer escape their buffer scope. (A1)
- HNSW `pruneNeighbors` applies insertion's diversity heuristic; sequentially inserted well-separated clusters no longer disconnect the graph. (A9)

### Tests

- PQTrain: `testCompressionQuality` re-seeded with clusterable data and a calibrated threshold; `testLargeScaleTraining` scaled down for debug builds; `testStreamingPQTraining` seeded (all three pre-existing defects, unrelated to the fixes above).

Releases prior to 0.1.7 are recorded in `docs/CHANGELOG.md`.
