# Changelog

All notable changes to VectorIndex are documented here. Versions follow the 0.x
convention: the minor digit signals breaking changes.

## [Unreleased] — 0.2.0

### Changed
<!-- cleanup / perf appended per task -->

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
