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
## 0.1.1 (Unreleased)

### Fixes
- IVFSelect batch query clobbering: refactored batch path to stage per‑query results and perform serial copy into outputs (disjoint writes; Swift 6 Sendable‑safe).
- IVFSelect tests: cosine equivalence compares scores by ID; L2 parity tolerance adjusted for vDSP vs scalar accumulation differences.
- VIndexMmap: added explicit version check (major==1) and strengthened open/init error paths; structured errors for header CRC, section CRC, endianness, mmap/file I/O maintained.

### Features
- Journaling Filter DSL (`JournalFilter`) with date/tags/custom predicates + comprehensive tests.
- IVFSelect batch sugar: `IndexOps.Batch.ivfSelectNprobe` returning per‑query `[nprobe]` arrays of IDs/scores.

### Docs
- README: added Journaling Quickstart and IVF batch sugar examples.

### Tests
- Added VIndexMmap error tests: header CRC mismatch, version mismatch, section CRC mismatch, missing file open, and a pragmatic growth/remap failure case (asserts `.fileIOError` or `.mmapError` depending on environment).

---
<!-- moved to docs/ -->
