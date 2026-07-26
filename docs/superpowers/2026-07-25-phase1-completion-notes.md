# 0.2.0 Phase 0–1 completion notes (2026-07-25)

Branch `gifton/cleanup-0.2.0`, `47a2bc2` (v0.1.6) → `45c2a0e`: 17 commits — Phase-0 gates, correctness fixes A1–A9 (each with regression coverage), three PQTrain test repairs, final-review fix wave. Full-suite verification: 376 run / 338 pass / 38 skipped-by-design / 0 reproducible failures. Whole-branch review verdict: ready to merge (post-fix-wave re-review clean).

## Deviations from the plan (all reviewed)

- **A7 scope expansion:** the plan's "harmless slack" premise was wrong — `tocSize` was frozen from `tocCount=3` before the `includeIDMap` increment, so the 4th TOC entry clobbered list 0's descriptor (`capacity` observed as 4058561182). Fixed as final-count × packed 36; red→green.
- **A4 follow-up (user-approved):** equal-score rerank ties now break by smallest full-width candidate id via an id-rank permutation fed to the heap (boundary-correct), superseding the interim by-position policy.
- **Out-of-plan, user-approved:** PQTrain test repairs (`testCompressionQuality` reseeded/clusterable + recalibrated threshold 0.05 vs measured 0.00244; `testLargeScaleTraining` debug-scaled n=1000/d=8/batchSize=256; `testStreamingPQTraining` seeded).
- **Final-review fix wave:** pre-existing `TopKHeap` leak in `rerank_exact_topk` fixed (`defer deallocate()`); `skipMissing=false` path now has coverage (missing candidates keep their slot with the metric sentinel score); stale comments corrected.

## Carried forward — Phase 2 (cleanup) inventory additions

Beyond the spec's B1–B10 list:
- Dead code orphaned by A9: `hnsw_prune_neighbors_f32_swift` (only its `@_cdecl` shim + direct unit test remain — decide disposition), `HNSWIndex.selectNeighbors` (~line 1174, never called).
- `_TOCEntry` struct in `VIndexContainerBuilder.swift` (~35) — dead since A7 removed its only use.
- `Kernels/PQTrain.swift.new` — dead file, excluded in Package.swift (already on the Phase-2 menu).
- `RerankOpts.returnSorted` appears unreferenced in `rerank_exact_topk` (output always best-first) — wire or remove.
- Stale sparse-clear header comment `CandidateDedup.swift` (~291, doesn't mention overflow-forced full clear); A1 wrapped bodies re-indent.
- Executable-coverage gaps accepted under the hardening rule: A6 defer sites 3/5/6 (u4 + residual Swift fallbacks).

## Carried forward — Phase 3 (perf)

- **Baseline annotation:** `.bench/baseline-0.1.6/README.md` — recall baselines predate A9's topology change; treat recall movement as expected improvement or re-capture a post-Phase-1 reference.
- **pruneNeighbors P2 rewrite:** starts ahead (A9's kernel swap already dropped a per-prune ephemeral CSR allocation); add `assert(!current.contains(idx))` to make the no-self/no-dup invariant executable.
- `internalRemove` never decrements `maxLevel` (stale-high after removing the max-level node; efficiency-only, masked by `neighbors[safe:]`).
- **TICKET — streaming-trainer numerical stability:** pre-seed, `testStreamingPQTraining` produced distortion=0.0 once and 3.2e23 on rerun. Seeding fixed the *test*; the possible divergence in `minibatchKMeansSubspaceChunk` on unlucky data was never investigated. Potentially a real production bug.

## Carried forward — Phase 5 (release) / §2.3 P1

- Consumer check caveat: wrap-up's VectorIndexAccelerated build resolved *published* VectorIndex 0.1.3, not this branch — the real consumer-vs-branch check is a release gate.
- Parked (Task 4): durable growth is structurally unsatisfiable for non-last sections (`mmap_append_begin` growth policy vs the `VIndexMmap.swift` ~692-702 sanity check) — IDs-section growth via the durable path is dead code until the §2.3 P1 mmap-persistence work fixes the policy. A3's corrected remap path stays production-unreachable until then.
- Re-verify test-suite wall times on a quiet machine (Phase 0–1 runs happened under load average 88–195).
