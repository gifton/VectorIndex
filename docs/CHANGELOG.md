> **Superseded:** releases from 0.1.7 onward are recorded in the root
> [`CHANGELOG.md`](../CHANGELOG.md). This file is the historical record of
> earlier releases.

## 0.1.0-alpha (2025-10-22)

Initial alpha release to unblock downstream packages.

### Highlights

- **Public Actors**: `FlatIndex`, `FlatIndexOptimized`, `IVFIndex`, `HNSWIndex` (all conform to `VectorIndexProtocol` and `AccelerableIndex`)
- **Shared Types**: `SearchResult`, `IndexStats`, `IndexStructure`, `AccelerationCandidates`, `AcceleratedResults`
- **Kernels**: Comprehensive kernel library under `IndexOps` namespace
  - Scoring (L2, Cosine, InnerProduct, ScoreBlock)
  - Selection (TopK), RangeQuery, Rerank (Exact)
  - Reservoir, Dedup, Filtering
  - Quantization (PQ, ADC/LUT, Post‑ADC)
  - Support (Norms, LayoutTransforms, Prefetch)
  - Transforms (MIPS), Telemetry
- **Persistence**: JSON-based index serialization
- **Swift 6**: Strict concurrency with actors and `@Sendable`
- **C ABI**: Performance-critical shims (HNSW traversal, scoring blocks)

### New Features

#### Error Infrastructure (Phase 1)

- ✅ **VectorIndexError System**: Comprehensive error handling with 23 error kinds across 6 categories
  - Input Validation, Data Integrity, Resource Constraints
  - Operation Failures, Configuration, Internal Errors
- ✅ **ErrorBuilder**: Fluent API for ergonomic error construction
  - Convenience builders for common patterns
  - Automatic source location capture (DEBUG builds)
  - Structured metadata for debugging
- ✅ **Error Chaining**: Multi-layer error propagation with root cause analysis
- ✅ **Documentation**: Complete guides (ERRORS.md, CONTRIBUTING.md, ERROR_HANDLING_INFRASTRUCTURE.md)

#### Error Migration (Phase 2)

- ✅ **IVFAppend**: Migrated 6 preconditions to structured errors
  - Parameter validation (k_c, m, d, group, format)
  - Comprehensive error messages with recovery guidance
  - 7 new test methods, 15 test cases
- ✅ **KMeansSeeding**: Migrated 2 preconditions + added 1 validation
  - Parameter validation (k, n, dimension)
  - 4 new test methods, 8 test cases

### Bug Fixes

- ✅ Fixed L2SqrKernel alignment crash on unaligned data
- ✅ Fixed K-means++ crash on identical data points (zero-weight fallback)
- ✅ Fixed Sendable conformance warnings (PartitionAccumulator, SubspaceAccumulator)
- ✅ Fixed K-means assignment computation flag

### API Changes

- **Made Internal** (Phase 1 API narrowing):
  - Telemetry system (internal implementation detail)
  - VIndexMmap and VIndexContainerBuilder (low-level persistence)
  - IDMap functions and types (internal ID management)

### Test Improvements

- ✅ 43 error infrastructure tests (100% passing)
- ✅ 26 benchmark tests skipped by default (enable manually for profiling)
- ✅ Fixed test suite hanging issues

### Known Issues

⚠️ **Pre-Existing Test Failures** (not caused by this release):
- IVFSelectTests: 281 failures (batch query operations)
- IVFListVecsReaderRerankTests: 1 failure
- IVFRecallTests: 1 failure
- IVFTests: 1 failure
- KMeansMiniBatchTests: 1 failure

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for detailed descriptions and planned fixes.

### What Works

✅ Error infrastructure (43/43 tests passing)
✅ Core IVF operations (single-query mode)
✅ K-means++ seeding
✅ L2/Cosine/InnerProduct distance metrics
✅ HNSW implementation
✅ Flat index operations
✅ PQ quantization
✅ JSON persistence

### Notes

- API is alpha and may evolve. Syntactic and structural parity with VectorCore maintained where applicable (e.g., `SupportedDistanceMetric`, typed overloads, provider seams)
- Linux is not currently a declared platform. Conditional imports added for future portability
- Phase 3 error migrations planned for 0.1.1 (PQTrain, VIndexMmap, remaining kernels)

### Migration Guide

If upgrading from pre-0.1.0 internal builds:

**IVFAppend** now throws:
```swift
// Before
let index = IVFListHandle(k_c: 10, m: 0, d: 128, opts: .default)

// After
let index = try IVFListHandle(k_c: 10, m: 0, d: 128, opts: .default)
```

**kmeansPlusPlusSeed** now throws:
```swift
// Before
let stats = kmeansPlusPlusSeed(data: ptr, count: n, dimension: d, k: k, ...)

// After
let stats = try kmeansPlusPlusSeed(data: ptr, count: n, dimension: d, k: k, ...)
```

See [ERRORS.md](ERRORS.md) for complete error handling guide.
## 0.1.1 (2025-11-17)

### Fixes
- IVFSelect batch query clobbering: refactored batch path to stage per‑query results and perform serial copy into outputs (disjoint writes; Swift 6 Sendable‑safe).
- IVFSelect tests: cosine equivalence compares scores by ID; L2 parity tolerance adjusted for vDSP vs scalar accumulation differences.
- VIndexMmap: added explicit version check (major==1) and strengthened open/init error paths; structured errors for header CRC, section CRC, endianness, mmap/file I/O maintained.

### Features
- Journaling Filter DSL (`JournalFilter`) with date/tags/custom predicates + comprehensive tests.
- IVFSelect batch sugar: `IndexOps.Batch.ivfSelectNprobe` returning per‑query `[nprobe]` arrays of IDs/scores.

### Kernel: PQTrain (#19)
- Mini-batch stability: switched to incremental running means with global counts; removed per-batch reseed/split; added per‑pass, bounded-cost empty repair (sampled farthest‑point).
- Seeding caps: dense and streaming seeding capped to ~4×ks candidates to bound initialization time.
- Streaming per-pass sampling: probabilistic thinning across chunks toward a pass budget (uses `sampleN` when set).
- Distortion safeguards: clamp negatives to zero; handle non‑finite totals; robust sampled evaluation.
- Lloyd correctness: optional dot‑trick gated (`precomputeXNorm2=false` by default) with per‑iteration centroid‑norm refresh and negative‑distance clamping.
- Warm-start (opt-in): `PQTrainConfig.warmStart` reuses provided codebooks as initial centroids when shaped; default unchanged.
- Tunables surfaced in `PQTrainConfig`:
  - `distEvalN` (default 2000): minibatch distortion eval sample when `sampleN == 0`.
  - `repairEvalN` (default 2000): minibatch pass-level empty repair sample when `sampleN == 0`.
  - `streamingRepairEvalN` (default 512): streaming pass-level empty repair sample.
  - `verbose` (default false): gate high-level prints; DEBUG-only diagnostics unaffected.

### Docs
- Added Kernel Overview at `docs/kernels/README.md` with a PQTrain quick guide, defaults, and tunables.

### Docs
- README: added Journaling Quickstart and IVF batch sugar examples.

### Tests
- Added VIndexMmap error tests: header CRC mismatch, version mismatch, section CRC mismatch, missing file open, and a pragmatic growth/remap failure case (asserts `.fileIOError` or `.mmapError` depending on environment).

## 0.1.4 (2026-04-11)

### Breaking Changes

- **VectorCore 0.2.0 adoption.** Minimum `VectorCore` dependency bumped from `0.1.6` to `0.2.0`. Public search APIs now return VectorCore's `StringSearchResult` (alias for `SearchResult<String>`) instead of the previously-local `SearchResult` struct, which has been removed.
  - Field rename: `.score` → `.distance` on all returned results.
  - Affected protocols: `VectorIndexProtocol.search` / `.batchSearch`, `AccelerableIndex.finalizeResults` / `.finalizeBatchResults`.
  - Affected actors: `FlatIndex`, `FlatIndexOptimized`, `HNSWIndex`, `IVFIndex`.

### Features

- **HNSW WAL / crash recovery** (new `HNSWWAL.swift`). Append-only sidecar write-ahead log for `HNSWIndex`. Every `insert` / `remove` / `batchInsert` is durably recorded (with CRC32 validation) before the in-memory state is touched. WAL records carry the exact level sampled by `randomLevel()` so replay is deterministic without re-advancing the RNG. New public API: `enableWAL(directory:)`, `checkpointWAL(to:)`, `disableWAL(checkpointTo:)`, and `HNSWIndex.openDurable(snapshotURL:walDirectory:...)` factory. `batchInsert` is a single atomic frame — a torn write mid-batch replays zero items.
- **Typed insert hint propagation.** The `IndexableVector` insert overload on `HNSWIndex` now reads `isNormalized` and `cachedMagnitude` from the vector and feeds a pre-computed inverse norm into the cosine cache. The `invNormsCache` is populated incrementally on insert (O(1) with hint, O(d) without) instead of being marked dirty and fully rebuilt O(N·d) on the next search. Added typed `batchInsert<V: IndexableVector>` overload that routes through the single-frame WAL path.
- **`searchWithMetadata` / `batchSearchWithMetadata`** (new `SearchResultsAdapter.swift`). Convenience extensions on `VectorIndexProtocol` that wrap results in VectorCore's `StringSearchResults`, surfacing `candidatesSearched`, `searchTimeNanos`, and `isExhaustive`. Per-index overrides report accurate `isExhaustive` semantics: `true` for `FlatIndex` / `FlatIndexOptimized`, `false` for `HNSWIndex` / `IVFIndex`.
- **Cosine fast path.** `distance(_:_:metric:queryIsNormalized:)` accepts a `queryIsNormalized` flag that skips the query-side `sumOfSquares` when the caller guarantees a unit-norm query. `HNSWIndex.search` gains an internal `qInvNorm:` overload that threads the pre-computed inverse norm into the traversal kernel.
- **Manhattan distance: SIMD4 rewrite.** Eliminates the temporary difference allocation; uses a `SIMD4<Float>` accumulator with a scalar tail.

### Bug Fixes

- **HNSW distance-insertion bounds.** Fixed an out-of-bounds issue in HNSW distance insertions (commit 9ee23d2).

### Internal

- `CRC32` struct in `VIndexMmap.swift` promoted from `private` to `internal` so the new HNSW WAL can reuse the hash implementation without duplication.
- `HNSWIndex.internalInsert` split into a live path (draws `randomLevel()`, appends WAL) and a replay-friendly `internalInsertAtLevel` (takes a pre-sampled level, no RNG involvement).
- `HNSWIndex.remove` split into a public WAL-aware path and a private `internalRemove` used by replay.

### Tests

- Updated `AccelerableIndexTests`, `CosineFusedCacheIntegrationTests`, `MicrokernelIntegrationTests`, and `PQTrainTests` for the new `StringSearchResult` / `.distance` field naming and related API changes.
- New `HNSWTypedInsertHintTests` (6 tests): cache coherence across inserts, hint vs computed path equivalence, non-cosine safety, reinsert stability.
- New `HNSWWALTests` (7 tests): clean-close replay, remove persistence, torn-frame truncation, checkpoint-race recovery, replay determinism across two fresh indices, batch-insert atomicity, frame codec round-trip.
