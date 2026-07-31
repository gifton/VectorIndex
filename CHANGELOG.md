# Changelog

All notable changes to VectorIndex are documented here. Versions follow the 0.x
convention: the minor digit signals breaking changes.

## [Unreleased] — 0.2.0

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
