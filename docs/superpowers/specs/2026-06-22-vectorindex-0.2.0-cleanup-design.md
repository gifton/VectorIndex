# VectorIndex 0.2.0 — Cleanup, Correctness & Performance Release (Design Spec)

**Date:** 2026-06-22
**Status:** Draft for approval
**Supersedes:** `docs/cleanup-0.2.0-plan.md` (backlog draft)
**Source:** 5-agent code review (2026-06-22, post-v0.1.6) + workspace verification pass.

---

## 1. Context & Goal

v0.1.6 shipped the additive `buildKNNGraph` producer. The review that followed surfaced
three classes of work that are best bundled into one *breaking* minor release:
correctness bugs, dead-code/internal cleanup (incl. a perf rework), and removal of dead
public API. Per SemVer-for-0.x convention, the public-surface removals make this **0.2.0**.

**Goal:** ship a VectorIndex that is correct, lean, faster on the documented hot paths, and
free of dead public surface — without regressing recall or breaking the one real downstream
consumer.

**Decisions locked (from brainstorming):**

| Topic | Decision |
|---|---|
| Scope | **Everything in 0.2.0** — correctness + all cleanup + perf rework + breaking removals. |
| C-ABI / CS2RNG | Removal gated on a verification step (now **satisfied**, see §3). |
| RNG | **Keep both Swift RNGs** (`RNGState` LCG + `S2Xoroshiro128`/`Philox`); delete only the dead CS2RNG C target. |
| Telemetry | **Wire to opt-in build config, fix bugs, AND fully validate with tests** this release. |
| FlatIndex | **Keep public** — consumer depends on it (see §3). |
| Acceleration surface | **Untouched in 0.2.0** — the VectorCore-`SoA` alignment workstream (which would also remove the home-grown L2 types) is deferred & documented in Appendix B. |

## 2. Non-Goals

- No 1.0 API freeze; this is still 0.x.
- No new index types or query features.
- No EmbedKit / VectorCore changes (VectorCore stays pinned at 0.3.1).
- No rewrite of the genuinely-justified custom block kernels (L2/cosine/IP register-blocked
  paths, the `TopKHeap` streaming selector) — review confirmed these should stay.

## 3. Verified external-consumer facts (the airtight gate)

Grepped across `/Users/goftin/dev/gsuite`:

- The **only** real importer of `VectorIndex` is `VSK/future/VectorIndexAccelerated`
  (single init commit 2025-11-26). `deprecated/Vector Explorer` is deprecated; **EmbedKit
  does not import VectorIndex**.
- `VectorIndexAccelerated` uses **none** of the symbols slated for removal in §7 — verified
  by per-symbol grep (`AccelerableIndexEnhanced`, `VectorReferenceCollection`,
  `UnifiedVectorStorage`, `SafeAccelerationCandidates`, `ReferenceAccelerationCandidates`,
  `searchWithMetadata`/`batchSearchWithMetadata`/`StringSearchResults`, `useDotTrick`,
  `ResidualError`, `vecsInterleave_f32_SIMD`, `atomicMultiWriter`).
- `VectorIndexAccelerated` **does** use `FlatIndex` (CPU baseline in ~8 test files) → **keep
  `FlatIndex` public.**
- The `@_cdecl` HNSW C-ABI shims (`hnsw_traverse_f32`, `hnsw_greedy_descent_f32`,
  `hnsw_efsearch_f32`) have **zero** consumers anywhere → safe to demote to `internal`/remove.
- `CS2RNG` is declared as a `VectorIndex` target dependency in `Package.swift` but **no Swift
  source `import`s it or calls its symbols** → dead; removable without touching the S2 Swift
  RNG types.

**Residual gate (Phase 0):** before tagging, build `VectorIndexAccelerated` against the
0.2.0 branch to confirm green. This is the only outstanding external check.

## 4. Approach & Sequencing

Single branch `gifton/cleanup-0.2.0`, work in strict phases, each phase a reviewable commit
group. Ordering rationale:

1. **Correctness before perf** — perf rewrites touch the same code as several correctness
   bugs; fixing first avoids rework. **Specific coupling:** A1 (escaping CSR pointers) lives
   in `searchLayer`/traversal, which Phase 3 rewrites. **Resolution:** Phase 1 fixes A1 only
   at the traversal-kernel call sites that survive the rewrite; the Phase-3 `searchLayer`
   replacement is written correct-by-construction (no escaping pointers), so we never
   fix-then-discard.
2. **Safe cleanup before perf** — shrinks the surface the perf work has to reason about.
3. **Perf behind a benchmark gate** — measured against a Phase-0 baseline.
4. **Breaking removals last** — smallest blast radius once everything else is green.

**TDD discipline (airtightness rule):** every correctness fix in Phase 1 starts with a
*failing test that reproduces the bug*. If a finding cannot be reproduced by a test, it is
reclassified (downgraded/removed) rather than "fixed" speculatively. This guarantees no fix
lands on a phantom bug.

---

## 5. Phase 0 — Gates & baseline

- **P0.1** Branch `gifton/cleanup-0.2.0` off `main` (v0.1.6).
- **P0.2** Capture perf baseline on `main`: run `VectorIndexBenchmarks` for HNSW search,
  `--knn-graph` (uniform + `--knn-clusters`), IVF search, Flat search, and the mmap append
  path. Save JSON under `.bench/baseline-0.1.6/`. These numbers are the Phase-3 gate.
- **P0.3** Build `VectorIndexAccelerated` against the branch (baseline green check; repeat
  before tag).
- **P0.4** Add `CHANGELOG.md` (repo has none) with an `## [Unreleased]` section; append per phase.

**Exit:** branch exists, baseline JSON committed, consumer builds green.

## 6. Phases 1–3 — line items

Legend — **CAT**: correctness / dead / perf / dep. **BRK**: API-breaking (yes/no).

### Phase 1 — Correctness (all start with a reproducing test)

| ID | File:Line | CAT | BRK | Bug → Fix | Test |
|----|-----------|-----|-----|-----------|------|
| A1 | `HNSWIndex.swift:190,330`; `HNSWKNNGraph.swift:130` | correctness | no | CSR base pointers escape `withUnsafeBufferPointer` closures, used after they return (UB). → Nest the pointer scopes around the `traverse` call, or wrap traversal in `withExtendedLifetime([csrOffsetsCache, csrNeighborsCache])`. Surviving call sites only; `searchLayer` site handled in A1∩P3. | Stress test under `-Onone` + ASan: large index, repeated search/buildKNNGraph; assert stable results. |
| A2 | `VIndexMmap.swift:477` → `IVFAppend.swift:345` | correctness | no | `mmapLists()` returns `nil` unconditionally → `get_list_stats_durable` always throws `contractViolation`. → Reimplement `mmapLists()` on `getListDescriptor`, or rewrite the caller to use `getListDescriptor` and delete `mmapLists`/`ListDesc`. | Test: durable IVF, call `get_list_stats_durable`, assert real stats (currently throws). |
| A3 | `VIndexMmap.swift:952` (vs `:388`) | correctness | no | grow/remap parses TOC at offsets @8/@16/@24 while writer/init use @4/@12/@20 → corruption after capacity growth. → Extract `parseTOC()`/`rebindSections()` used by both sites; single correct offset set. | Test: build mmap index, force `ensureFileCapacity` growth, reopen, assert all sections valid. |
| A4 | `ExactRerank.swift:682` | correctness | no | `Int64→Int32 truncatingIfNeeded` on candidate IDs silently corrupts IDs > 2³¹. → Pass `nil` ids to `selectTopK_streaming`, map heap indices back to `candIDs[idx]` after extraction (removes `ids32All`/`filteredIDs32`). | Test: rerank with IDs > 2³¹, assert returned IDs exact. |
| A5 | `HNSWIndex.swift:483` | correctness | no | `batchRemove` zeros `entryPoint`/`maxLevel`/`activeCount` after a *subset* remove → index returns `[]` thereafter. → Delete the three reset lines (and redundant cache-dirty marks); `remove()` already maintains state. | Test: insert N, `batchRemove` a subset, assert search still returns live points and `count == N - removed`. |
| A6 | `PQEncode.swift:529` | correctness | no | `ensureCentroidSqNorms` deliberately leaks the `m·ks` buffer on every Swift-fallback encode without precomputed norms. → Return an owned buffer with `defer { dealloc }` (pattern already used a few lines below). | Test: repeated encodes without norms; assert no unbounded growth (allocation count / heap watermark). |
| A7 | `VIndexContainerBuilder.swift:91` | correctness | no | `MemoryLayout<_TOCEntry>.stride` (~40, padded) used where writer emits packed 36-byte entries → layout drift. → Use explicit `36` constant; share one disk-layout struct set with the reader (folds into A3/B-dedup). | Covered by A3 round-trip test + an explicit `tocSize == 36` assertion. |
| A8 | `CandidateDedup.swift:472` | correctness | no | Touched-word tracking silently stops recording at `touchedCapacity` (1M); sparse-clear can then leave stale bits under adversarial ID spread. → Set a `touchedOverflowed` flag on saturation; force full-clear in reset when set. | Test: drive > capacity distinct touched words, reset, assert no stale membership. |

### Phase 2 — Non-breaking internal cleanup

| ID | File:Line | CAT | BRK | Change |
|----|-----------|-----|-----|--------|
| B1 | `Telemetry.swift` (all), `ExactRerank.swift:644`, `RangeQuery.swift:207` | dead→live | no | **Wire telemetry in.** Add a SwiftPM trait (or `-DVINDEX_TELEM` debug config) so the impl compiles in CI; fix the `used_prefetch` hardcoded-true surrogate (`Telemetry.swift:50`, `ExactRerank.swift:646`); make `RangeQuery.recordTelemetry` actually record. **Validate** with unit tests asserting counter/histogram accuracy (candidate counts, distance-eval counts) against a hand-computed scenario. |
| B2 | `Sources/CS2RNG/*`, `Package.swift:32,43` | dead | no | Remove the `CS2RNG` C target and its `VectorIndex` dependency entry (verified unused, §3). Keep S2 Swift RNG types + `S2RNGDtypeTests`. |
| B3 | `MIPSTransform.swift:266-409` | dead/dep | no | Delete `l2NormSquaredSIMD`, `innerProductSIMD`, `generic_l2sqrBlock`, and the sham `l2sqrBlock_dispatch` `#if`; route to `IndexOps.Scoring.InnerProduct`, `l2sqr_f32_block`, `Norms.l2NormSquared`. (~140 lines.) |
| B4 | `Sources/L2SqrMicrobench/*`, `Package.swift` | dead | no | Delete the `L2SqrMicrobench` executable target (dev scaffolding, no assertions). |
| B5 | `L2Sqr.swift:22`; `L2SqrKernel.swift:60,78` | dead | no | Delete always-`false` `DispatchBK` enum + guard; delete no-op `_verifyAlignment`/`_prefetchRow` and their call sites. |
| B6 | `HNSWIndex.swift:1165` | dead | no | Delete unused private `selectNeighbors` (insertion uses the `#34` kernel). |
| B7 | `PQLUT.swift:35`, `ResidualKernel.swift:115`, `KMeansMiniBatch.swift:193` | dead | no | Collapse the three identical no-op prefetch helpers into one internal symbol or remove. |
| B8 | `Norms.l2NormSquared` + `L2SqrKernel.swift:485`, `Cosine.swift:183,427`, `ScoreBlock.swift:67`, `MIPSTransform.swift:269` | dep | no | Make `Norms.l2NormSquared` the single sum-of-squares impl; other four delegate. (Keep inlined SIMD body; Accelerate `sumOfSquares` only if measured win for large d.) |
| B9 | `ResidualKernel.swift` | dead | no | Strip changelog-style "✅ Fixed" comments; unify residual out-of-range error path (`residuals_f32` vs `_inplace`) to one error kind. (`ResidualError` *removal* is breaking → §7.) |
| B10 | `HNSWTraversal.swift:99`, `HNSWNeighborSelection.swift` | dead | no | Delete unused `selectBatchSize`; de-dup the two byte-identical scalar distance-kernel families into one internal file. |

### Phase 3 — Performance rework (each gated against P0.2 baseline)

**Gate rule:** for every item, re-run the relevant baseline benchmark; require **recall@k
unchanged (±0)** for graph/search correctness and **no throughput regression**; record the
delta in CHANGELOG. Determinism (fixed seed → identical graph) must hold.

| ID | File:Line | CAT | BRK | Change | Gate metric |
|----|-----------|-----|-----|--------|-------------|
| P1 | `HNSWIndex.swift:754` (`searchLayer`) | perf | no | Replace the O(n²) hand-rolled construction traversal (linear candidate scan + `insertSorted` + per-candidate `vectorArray(at:)` copy) with the existing `HNSWTraversal` min-heap + `ScoreBlock` kernel. Written correct-by-construction re: A1. | Build time ↓; recall unchanged; determinism held. |
| P2 | `HNSWIndex.swift:680` (`pruneNeighbors`) | perf | no | Stop allocating an `N+1` offsets array per edge update; pass a 2-entry window / count to the prune kernel. | Build-time allocation traffic ↓ (Instruments alloc count). |
| P3 | `IVFSelect.swift:462`, `IVFIndex.swift:299` | perf/dep | no | Replace per-centroid `vDSP_dotpr`/`vDSP_distancesq` loop and scalar `[[Float]]` distance loops with one `cblas_sgemm` cross-term (`-2·X·Cᵀ`, norms already maintained); cache `MatrixDistance.prepare(centroids)` on `optimize()`. **Re-verify** `MatrixDistance` raw-pointer vs typed-vector wrap cost against pinned VectorCore 0.3.1 first. | IVF search QPS ↑; recall unchanged. |
| P4 | `VIndexMmap.swift:748` | perf | no | Per-commit full-section CRC recompute → defer to flush/close or make incremental/rolling; quadratic ingestion → linear. | mmap append throughput vs N (slope linear). |
| P5 | `VIndexMmap.swift:359` (`msyncPageAligned`) | perf | no | Honor ptr/len for a page-aligned partial `msync`, or batch one flush per commit instead of ~6 full-mapping flushes. | Per-commit syscall count ↓. |
| P6 | `PQTrain.swift:755,1416`; `KMeansSeeding.swift:363`; `ScoreBlock.swift:53`; `RangeQuery.swift:705`; `ExactRerank.swift:664`; `InnerProduct.swift:114`; `IDMap.swift:335`; `JournalFilter.swift:93` | perf | no | Lower-risk per-site fixes: SIMD `l2Sq` + incremental D² in PQ training; O(k²)→O(k) streaming seeder; hoist per-row/per-call allocations; `IDMap.erase` return internalID instead of O(count) rescan; reuse one `ISO8601DateFormatter` in `JournalFilter`. Batch as one commit; no API change. | Targeted micro-bench / alloc counts per site. |
| P7 | `pq_encode.c:396` | perf/dead | no | Remove the GNU statement-expression dead duplicate computation; plain locals. | Build clean; PQ encode parity test. |

## 7. Phase 4 — Breaking public removals (0.2.0 break)

All verified unused by the real consumer (§3). Direct delete (no deprecate-first: pre-1.0,
consumer-clean). Each removal: delete symbol + any internal-only references + tests that only
exist to exercise the dead symbol (move pointer-safety tests to test target if still wanted).

> **Acceleration surface intentionally retained.** Items C1–C4 in earlier drafts removed the
> home-grown Layer-2 "zero-copy" types (`AccelerableIndexEnhanced`, `VectorReferenceCollection`,
> `UnifiedVectorStorage`, `SafeAccelerationCandidates`, `ReferenceAccelerationCandidates`).
> **Decision: do NOT touch the acceleration surface in 0.2.0.** Removing those types is only
> sensible *paired with* adding the correct VectorCore-`SoA` producer that supersedes them, and
> that whole alignment workstream is **deferred and documented** (Appendix B). So the entire
> acceleration bridge — base `AccelerableIndex`/`AccelerationCandidates` (L1) **and** the L2
> types — is left exactly as-is this release.

| ID | Symbol(s) | File | Note |
|----|-----------|------|------|
| C5 | `searchWithMetadata` / `batchSearchWithMetadata` + 4 per-index overrides + `StringSearchResults` bridge | `SearchResultsAdapter.swift` | No callers anywhere; independent of the acceleration bridge (uses the CPU `search` path, not `getCandidates`). Delete file. |
| C6 | Collapse `ErrorHandling/` (4 files, ~1100 lines) → ~150 | `ErrorHandling/*` | Keep `IndexErrorKind`, flat `VectorIndexError{kind,message,operation,info,#if DEBUG file/line}`, `ErrorBuilder.info/message/build`. Drop chaining/`rootCause`/`isTransient`/`shouldReport`/`recoveryMessage`/`logMetadata`/`withAdditionalInfo`/`threadID`/`memoryPressure` and per-error `Date()`/`ISO8601DateFormatter`. Migrate existing throw sites (mechanical; they already use `.info/.message/.build`). |
| C7 | `vecsInterleave_f32_SIMD` (no-op public wrapper); vestigial `scratch` param on `vecsInterleaveInPlace_f32` | `LayoutTransforms.swift:264,363` | Remove or implement NEON. Recommend remove. |
| C8 | `L2SqrOpts.useDotTrick` | `L2SqrKernel.swift:21` | Redundant with `algo`. Remove field; update `IVFSelect`/`PQEncode`/`PQLUT` call sites to `algo`. |
| C9 | `ResidualError` | `ResidualKernel.swift:47` | Unify on `ErrorBuilder` (pairs with B9). |
| C10 | `CandidateDedup.ConcurrencyMode.atomicMultiWriter` + unused `concurrency`/`atomicConflicts` | `CandidateDedup.swift:36` | "Not implemented" mode; remove or `precondition`-reject. |
| C11 | Delete the `VectorProtocol` typed-overload family (16 methods) | `TypedOverloads.swift:7-85` | **Non-breaking in practice** — shadowed by the strictly-better `IndexableVector` family for all concrete types. Keep the `IndexableVector` family. |

**Explicitly NOT removed:**
- `FlatIndex` (consumer depends on it) — instead add a doc comment clarifying the `FlatIndex`
  (baseline) vs `FlatIndexOptimized` (production) split.
- The **entire acceleration surface** (`AccelerableIndex`, `AccelerationCandidates`,
  `AccelerableIndexEnhanced`, `VectorReferenceCollection`, `UnifiedVectorStorage`,
  `SafeAccelerationCandidates`, `ReferenceAccelerationCandidates`) — retained pending the
  deferred SoA-alignment workstream (Appendix B).

## 8. Phase 5 — Release

- Full `swift build -c release` + `swift test` green (incl. new correctness + telemetry tests).
- Re-run all P0.2 benchmarks; paste deltas into CHANGELOG; confirm no recall regression.
- Build `VectorIndexAccelerated` against the branch — green (residual gate).
- Finalize `CHANGELOG.md` `## [0.2.0]`; tag + push `v0.2.0`; GitHub release.

## 9. Testing strategy

- **Correctness:** one reproducing test per A1–A8 (fail on `main`, pass after). ASan/`-Onone`
  run for A1.
- **Telemetry:** accuracy unit tests (B1) — the validation the user asked for.
- **Perf:** P0.2 baseline vs post-Phase-3; recall asserted unchanged; throughput deltas recorded.
- **No-regression:** full existing suite green at each phase boundary; `VectorIndexAccelerated`
  builds at P0.3 and P5.

## 10. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| A3 TOC offset fix touches on-disk format | Round-trip + reopen tests; no format-version bump (same layout, fixing a parser bug). |
| P3 GEMM wrap cost may not beat scalar for small `kc` | Re-verify against pinned VectorCore 0.3.1; keep scalar path if measured slower. |
| Perf rewrite changes graph determinism | Determinism test must pass byte-identical; gate blocks tag otherwise. |
| `VectorIndexAccelerated` is early/unstable | It's the only consumer and uses no removed symbols except `FlatIndex` (kept); build-check both ends. |
| Telemetry overhead leaks into timing | Off by default; never enabled during P0.2/P5 timing runs. |

## 11. Success criteria

1. All 8 correctness bugs fixed with reproducing tests.
2. Dead code removed (telemetry wired+tested instead); net large line reduction.
3. Documented hot paths measurably faster, recall unchanged, determinism held.
4. Dead public surface gone (acceleration bridge intentionally retained per Appendix B);
   `VectorIndexAccelerated` still builds.
5. `v0.2.0` tagged with a CHANGELOG capturing the break + perf deltas.

---

## Appendix A — Complete finding ledger (traceability)

Every review finding has an explicit home (phase/line-item) or a deliberate disposition. No
finding is silently dropped. "Defer" = real but low-value; revisit post-0.2.0 only if cheap.

### Additional explicit line items (added in self-review)

| ID | File:Line | CAT | BRK | Change |
|----|-----------|-----|-----|--------|
| B11 | `VIndexContainerBuilder.swift:16,35` | dep/dead | no | Delete `_CRC32` (use existing `CRC32`); share one set of disk-layout structs with the reader (pairs with A3/A7). |
| B12 | `IDMap.swift:63-123,8` | simplify/perf | no | Keep only the default `swissTable`; drop RobinHood/LinearProbing and the `enum`-dispatch wrapper (removes per-op value copy-in/out). Fix `IDMapOpts` fields wrongly marked `public` on an internal type. |
| B13 | `VIndexMmap.swift:44,145,283,830,992` | perf/simplify | no | mmap low-sev tidy: merge byte-identical `toHost`/`fromHost`; hash header CRC in-place (drop 256-byte copy); validate WAL append-record CRC on replay (or stop computing it) + reuse one scratch buffer; leave inert `version_minor` plumbing (single format version). |
| B14 | `IDFilter.swift:322,108` | perf/dead | no | Inline the per-id test in `idFilterCompactN` (drop the size-`n` mask alloc); delete unused `FilterMode.shouldKeep`. |
| B15 | `CandidateReservoir.swift:305,255` | perf | no | `extractTopK` via existing `quickselectTop` (sort only k, drop the 3 allocs); add `size >= bufferCapacity` overflow prune to the `.adaptive` branch. |
| B16 | `CandidateDedup.swift:434`; `IVFSelect.swift:778` | perf/simplify | no | SparsePaged: avoid `Page` value-struct write-back (class wrapper or `[Int64:UInt32]` epoch side-table). Delete the stream-of-consciousness comments in `MinHeap.insert`; factor `MinHeap`/`MaxHeap` into one comparator-parameterized `BoundedTopKHeap`. |
| B17 | `ExactRerank.swift:304,559`; `HNSWIndex.swift:642,1088` | perf/simplify | no | Replace pointer-laundering via `UInt(bitPattern:)`+force-unwrap with a `@unchecked Sendable` box; flatten the recursive nested-closure IVF builder. HNSW: cache distances in `selected.min(by:)` (`:642`) and in `compact` re-prune (`:1088`) instead of re-allocating `[Float]` per comparison. |
| B18 | `IVFIndex.swift:265,311` | perf/simplify | no | Have `optimize()` delegate to `optimizeKMeans(maxIterations:)`; build one flat `ContiguousArray<Float>` shared between seeding and training (drop the double `[[Float]]` materialization); reuse minibatch assignments to build inverted lists (removes the per-vector nearest-centroid rescan — overlaps P3). |
| B19 | `HNSWIndex.swift:62,36,267`; `HNSWWAL.swift:247` | simplify/perf | no | `init(dimension:metric:)` delegates to the designated init; compute `hnswMetric` once in `init` (stop recomputing `toHNSWMetric`); make `BatchSearchContext.vectorStorage` a COW `ContiguousArray` (drop the eager `Array(...)` deep copy, matching `KNNBuildContext`); either exercise or drop the never-emitted WAL `.update`/`.clear` codec. |
| B20 | `KMeansMiniBatch.swift:355` | perf | no | Drop the no-op "tiled" wrapper in `_vi_km12_assignAOS_tiled` (it's a flat argmin once prefetch is removed). |

### Deferred (low value — explicitly out of 0.2.0)

| Finding | Reason to defer |
|---------|-----------------|
| `HNSWTraversal.greedyDescent`/`efSearch` public→internal demotion | Bundled with the C-ABI decision; harmless if left. Revisit only if trimming public surface further. |
| Accelerate `sumOfSquares`/CRC slice-by-8 micro-optimizations (B8/CRC) | Keep inlined SIMD; only adopt if a measured large-`d` win appears. Not worth bench effort now. |
| `streamingKMeansppSeed` O(k²) (separate from P6 main seeder) | Streaming seed path is rarely hot; fix opportunistically if P6 touches it. |

### Coverage check

- **Agent A (HNSW):** F1→A1; F2→P1; F3→P2; F4→B6; F5/F7→B17; F6→A5; F8→B10; F9→P1/P6; F10→Defer; F11/F14/F13/F12→B19. ✅
- **Agent B (IVF/PQ/KMeans):** F1/F2→P3; F3→P6; F4→P6; F5→P7; F6→A6; F7→B18; F8→B9/B7; F9→B16; F10→B20/Defer. ✅
- **Agent C (scoring/math):** F1→B4; F2→B5; F3→C8; F4→B5; F5→B3; F6→P6; F7→P6; F8→B8; F9→P6; F10→(fold into B-dedup, NormsCosineAdapter); F11→P6; F12→C7. ✅
- **Agent D (public surface):** #1→C11; #2 (`AccelerableIndexEnhanced`)→**Deferred, Appendix B**; #3 (`SafeAccelerationCandidates`)→**Deferred, Appendix B**; #4 (`VectorReferenceCollection`)→**Deferred, Appendix B**; #5 (`UnifiedVectorStorage`/`ReferenceAccelerationCandidates`)→**Deferred, Appendix B**; #6→C5; #7→keep FlatIndex (§3/§7); #8→A2/B13. ✅
- **Agent E (ops/error/util/persist):** ErrorHandling→C6; IndexErrorContext perf→C6; threadID/memoryPressure→C6; RNG/CS2RNG→B2; Telemetry→B1; mmap A2/A3/P4/P5/B11/B13; RangeQuery→P6/B1; ExactRerank→A4/B17; Dedup→A8/B16; Reservoir→B15; IDFilter→B14; IDMap→B12/P6; JournalFilter→P6. ✅

All ~40 findings accounted for.

---

## Appendix B — Acceleration ↔ VectorCore SoA alignment (DEFERRED — future release, documented only)

**Status:** NOT in 0.2.0. Captured here so the design isn't lost. Decision (2026-06-22): defer
the whole workstream; leave VectorIndex's acceleration surface untouched this release.

### B.1 Why this exists

VectorIndex exposes an `AccelerableIndex` bridge so a GPU consumer (`VectorIndexAccelerated`,
via `VectorAccelerate`) can compute distances on-device without VectorIndex duplicating GPU
code. Today that bridge is only partly used and not aligned with how VectorCore/VectorAccelerate
actually do zero-copy GPU handoff. This appendix records the correct alignment target.

### B.2 Verified facts (investigation 2026-06-22)

- **The real zero-copy contract VectorAccelerate consumes is VectorCore `SoA<V>` / `SoALayout`
  via `SoACandidateSet<V>` → `makeBuffer(bytesNoCopy:)`** — *not* `PageAlignedBuffer` /
  `UnifiedVectorBuffer` (those exist in VectorCore but are unused by VectorAccelerate).
  Tested path: `ZeroCopyBridgeTests`, `SoAKernelGoldenTests`, parity at 512/768/1536.
- `SoACompatible` requires a fixed VectorCore optimized type (`Vector{384,512,768,1536}Optimized`,
  `storage: ContiguousArray<SIMD4<Float>>`). **EmbedKit's 384-dim embeddings → `Vector384Optimized`
  is SoACompatible**, so the production use case qualifies.
- Layout is **lane-major (transposed)**: element `(lane ℓ, candidate j)` at `buf[ℓ*count + j]`,
  each `SIMD4<Float>`. Page-aligned base + page-rounded length when `SoA(pageAligned: true)`.
- **Zero-copy is `SoA → Metal`, not `VectorIndex → SoA`.** VectorIndex stores row-major
  `[Float]`; building `SoA<Vector384Optimized>` is a transpose/repack (one copy). VectorCore
  documents this as unavoidable ("GPU consumers must copy into page-aligned storage once").
- VectorAccelerate's legacy `Metal4ComputeEngine.batchEuclideanDistance([Float],[[Float]])`
  always copies; the zero-copy entry is `MetalComputeProvider.batchDistance(query:against:
  SoACandidateSet)`.
- VectorIndex's home-grown L2 types (`AccelerableIndexEnhanced`, `ReferenceAccelerationCandidates`,
  `UnifiedVectorStorage`, `SafeAccelerationCandidates`, `VectorReferenceCollection`) are
  row-major and align with **none** of this — they are the wrong abstraction for the SoA contract
  and have no conformers/consumers.
- Consumer state: `VectorIndexAccelerated` is mid-rework/partly non-compiling; its maintained
  `FlatIndexAccelerated`/`IVFIndexAccelerated` duplicate the vector store and use the *old*
  copying engine; only its broken `HNSWIndexAccelerated` calls the L1 bridge.

### B.3 Target design (when undeferred)

1. **VectorIndex (producer):** add to `AccelerableIndex` a method that yields candidates as a
   page-aligned VectorCore `SoA<V>` (+ its `SoALayout`) for `SoACompatible` dims — e.g.
   `func getAccelerationSoA<V: SoACompatible>(query:k:filter:as:) async throws -> SoA<V>`.
   Keep the row-major `AccelerationCandidates` path for non-SoACompatible dims and the CPU/HNSW
   consumer.
2. **VectorIndex (cleanup, paired):** once the SoA producer exists, **remove** the superseded
   home-grown L2 types (the old C1–C4) — *that* is when they go, replaced by the correct vehicle.
3. **VectorAccelerate (small gap):** add `SoACandidateSet(soa: SoA<V>, device:)` so a
   VectorIndex-built `SoA` is wrapped with no re-pack (today `SoACandidateSet.init` only takes
   `[V]` and builds the `SoA` itself).
4. **VectorIndexAccelerated (consumer):** migrate `FlatIndexAccelerated`/`IVFIndexAccelerated`
   off their duplicate stores onto the SoA path; fix/retire `HNSWIndexAccelerated`.

### B.4 Sequencing note

This is multi-repo feature work, gated on `VectorIndexAccelerated` being un-broken. Natural
home: a dedicated release **after** 0.2.0 (and after EmbedKit/VectorCore correctness settles),
ideally co-scheduled with the FAISS/Accelerate benchmark harness so the zero-copy path is
measured against the copying baseline. Each repo gets its own spec → plan cycle.
