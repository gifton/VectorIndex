# VectorIndex 0.2.0 — Cleanup & Correctness Plan

Source: 5-agent code review run 2026-06-22 (post-v0.1.6). Findings are review-level;
**verify each before acting**. Cross-cutting caveat: the sibling `../VectorCore` checkout
is OLDER than the pinned `0.3.1` in `.build/checkouts/VectorCore` — re-check every
"delegate to VectorCore" item against the pinned API, not the sibling dir.

Release strategy: 0.2.0 is the *breaking* cleanup. Where practical, land
`@available(*, deprecated)` on doomed public symbols in a 0.1.x patch first so the break
is pre-announced, then delete in 0.2.0.

---

## A. Correctness bugs — fix regardless of version (most are non-breaking)

| # | Bug | Location | Notes |
|---|-----|----------|-------|
| A1 | Escaping `withUnsafeBufferPointer` base pointers captured into arrays used after closure returns (UB) | `HNSWIndex.swift:190,330`; `HNSWKNNGraph.swift:130` | Lands in just-shipped code. Fix via nested scopes or `withExtendedLifetime`. |
| A2 | `mmapLists()` returns `nil` unconditionally → `get_list_stats_durable` always throws | `VIndexMmap.swift:477` → `IVFAppend.swift:345` | Reimplement on `getListDescriptor` or rewrite caller. |
| A3 | grow/remap parses TOC at wrong offsets (@8/@16/@24 vs @4/@12/@20) | `VIndexMmap.swift:952` | Extract shared `parseTOC()` and fix. Latent on-disk corruption. |
| A4 | Int64→Int32 ID truncation silently corrupts IDs > 2³¹ | `ExactRerank.swift:682` | Pass `nil` ids to `selectTopK_streaming`, map back post-extract. |
| A5 | `batchRemove` zeroes `activeCount`/`entryPoint` → corrupts index on partial removal | `HNSWIndex.swift:483` | Delete the reset lines; `remove()` already maintains state. |
| A6 | Deliberate buffer leak per Swift-fallback PQ encode without precomputed norms | `PQEncode.swift:529` | Return owned buffer w/ `defer dealloc` (pattern already used below it). |
| A7 | `MemoryLayout.stride` (~40) used where packed 36-byte TOC entries written | `VIndexContainerBuilder.swift:91` | Use explicit 36-byte constant; share disk structs with reader. |
| A8 | Dedup touched-word saturation can leave stale bits (adversarial ID dist) | `CandidateDedup.swift:472` | Set overflow flag → force full-clear on reset. |

## B. Non-breaking cleanup (internal — could even ship in a 0.1.x patch)

**Dead code / scaffolding**
- `Telemetry.swift` (764 lines) entirely behind never-defined `VINDEX_TELEM` → wire to a real config or delete; also `RangeQuery.swift:207` no-op `recordTelemetry`.
- CS2RNG C target + `S2_RNGDtype.swift` S2 RNG types are test-only; modulemap header path is broken/inert; plus unused legacy `RNG` struct (`Utilities/RNG.swift:106`). **Pick ONE RNG** (`RNGState`), delete the rest (~1700 lines). Grep dtype-conversion helpers separately — they may have real users.
- `MIPSTransform.swift:266-409` duplicate kernels (~140 lines) + sham `l2sqrBlock_dispatch` `#if` → route to `InnerProduct`/`l2sqr_f32_block`/`Norms.l2NormSquared`.
- `L2SqrMicrobench` target (delete), `DispatchBK` always-false placeholder (`L2Sqr.swift:22`), no-op `_verifyAlignment`/`_prefetchRow` (`L2SqrKernel.swift:60`), dead `selectNeighbors` (`HNSWIndex.swift:1165`), no-op prefetch helpers across PQLUT/ResidualKernel/KMeansMiniBatch.
- 5× duplicated sum-of-squares → consolidate on `Norms.l2NormSquared`.

**Performance**
- Construction `searchLayer` is O(n²) hand-rolled path → use existing `HNSWTraversal` min-heap + `ScoreBlock` kernel (biggest build-time win). `HNSWIndex.swift:754`.
- `pruneNeighbors` allocates N+1 offsets per edge → O(N²) alloc traffic. `HNSWIndex.swift:680`.
- Per-commit full-section CRC recompute → quadratic ingestion; `msyncPageAligned` flushes whole mapping ~6×/commit. `VIndexMmap.swift:748,359`.
- IVF coarse-quantizer scalar `[[Float]]`/per-centroid vDSP loops → single `cblas_sgemm` (norms already maintained). `IVFSelect.swift:462`, `IVFIndex.swift:299`.
- PQ training scalar `l2Sq`, double O(n) D² sampling pass, O(k²) streaming seeder. `PQTrain.swift:755,1416`, `KMeansSeeding.swift:363`.
- Per-row/per-call allocs: `ScoreBlock.swift:53`, `RangeQuery.swift:705`, `ExactRerank.swift:664`, `InnerProduct.swift:114`.
- `JournalFilter.swift:93` allocates `ISO8601DateFormatter` per item; `IDMap.swift:335` O(count) linear scan per erase.
- `pq_encode.c:396` GNU statement-expression with dead duplicate computation.

## C. Breaking removals (the reason this is 0.2.0)

Dead/unused **public** surface — remove (deprecate-first in 0.1.x where possible):
- `AccelerableIndexEnhanced` protocol + `withCandidateReferences` — zero conformers/callers.
- `VectorReferenceCollection` protocol — zero conformers/callers.
- `UnifiedVectorStorage` (dead) ; demote/inline `ReferenceAccelerationCandidates` ; `SafeAccelerationCandidates` → move to test target.
- Entire `SearchResultsAdapter.swift` (5 near-identical copies, no callers) — remove or collapse to one default impl.
- `ErrorHandling/` (~1100 lines, ~85% unused) → collapse to ~150 lines (keep `IndexErrorKind` + flat error + `ErrorBuilder.info/message/build`; drop chaining/`rootCause`/`isTransient`/`shouldReport`/`recoveryMessage`/`logMetadata`/`threadID`/`memoryPressure` + per-error `Date()`/`ISO8601DateFormatter`).
- Public no-op/redundant shims: `vecsInterleave_f32_SIMD` (`LayoutTransforms.swift:264`), `L2SqrOpts.useDotTrick`, `ResidualError`, `CandidateDedup.atomicMultiWriter` mode + dead `prefetchDistance` option fields.
- **Non-breaking-in-practice:** delete the `VectorProtocol` typed-overload family (16 methods, `TypedOverloads.swift:7-85`) — shadowed by the strictly-better `IndexableVector` family for all real types.
- Consider demoting `FlatIndex` out of public API (superset `FlatIndexOptimized` exists) — optional; `FlatIndex` is still used by tests/benchmarks as the baseline.

## Suggested sequencing
1. Branch `gifton/cleanup-0.2.0` off `main` (v0.1.6).
2. Land Section A correctness fixes first (each with a regression test), incl. A1 in the shipped kNN-graph code.
3. Section B internal cleanup (safe, big line reductions; verify telemetry/RNG/C-ABI aren't external contracts before deleting).
4. Section C breaking removals; bump version; write CHANGELOG (repo currently has none).
5. Verify: `swift build -c release` + full `swift test` green; re-run kNN-graph bench for parity.
