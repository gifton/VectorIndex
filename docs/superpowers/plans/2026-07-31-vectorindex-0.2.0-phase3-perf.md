# VectorIndex 0.2.0 Phase 3 — Performance (P1–P7 + carried-forward) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the documented hot paths measurably faster — HNSW construction, IVF centroid scoring, mmap append durability, PQ-train inner loops — each behind a benchmark gate, with recall preserved and determinism held, plus close out the perf items carried forward from Phase 2.

**Architecture:** Every perf change is gated against a fresh baseline captured on `main` @ `e71daae` (Tasks 1–2 build the harness additions and capture it before any library change). Rewrites that change FP accumulation order (P1, P3b, P6a, F16) get ordering-level parity gates and a bounded recall tolerance; purely mechanical changes (P4, P5, P6c, P7) get exact-behavior gates (bit-parity tests, syscall/CRC counters). No public API changes; all new state and helpers are internal.

**Tech Stack:** Swift 6 (StrictConcurrency), Accelerate (`vDSP`, first use of `cblas_sgemm`), XCTest, SwiftPM benchmarks executable + `RUN_BENCHMARKS=1`-gated test-target benchmarks.

## Global Constraints

- **Branch:** `gifton/perf-0.2.0-phase3` off `main` @ `e71daae` (Phase-2 merge commit). Local commits only; no pushes unless the user authorizes.
- **NON-BREAKING:** no public API removals or signature changes. Additive internal helpers, internal-class members, new benchmark-CLI flags, and access-level raises (`private` → `internal` for test hooks) are all allowed. Deprecations are not expected this phase.
- **VectorCore stays pinned** at 0.3.1, revision `b26909e98b6a9c6b83f19904ea0072646a4920fd`. `Package.swift` / `Package.resolved` dependency entries untouched.
- **Benchmark gate rule** (spec §6 Phase 3): every perf item re-runs its relevant benchmark(s) against `.bench/baseline-0.2.0-pre-phase3/` (captured in Task 2), Release configuration, same machine, quiet machine (probe first: `time swift build` no-op must complete in <10 s — if not, the machine is loaded; wait, don't benchmark). Record deltas in the ledger as you go; Task 16 consolidates them into CHANGELOG.
- **Recall gate:** for changes that cannot alter FP results (P4, P5, P6c, P7): recall must be bit-identical (±0). For changes that alter FP accumulation order (P1, P3b, P6a, Task 13): `recallAvg` within **±0.01** of the fresh baseline, and any nonzero delta gets a one-line explanation in the ledger. This tolerance is a documented deviation from the spec's blanket "±0" (see Deviations table below).
- **Determinism gate:** `HNSWDeterminismTests.testIndependentBuildsProduceIdenticalGraphs` (added in Task 2) must be green at every task boundary — fixed seed → byte-identical graph across two independently constructed indices in the same binary.
- **TDD / coverage-first:** sites with zero live coverage (`streamingKMeansppSeed`, `rangeScanL2_earlyExit`, `ScoreBlock` default path, `compact()` re-prune) get a test before the change. A finding that cannot be reproduced by a test is reclassified and recorded, not speculatively "fixed".
- **Test running:** always foreground with explicit large timeouts; never end a turn mid-command. `swift test --filter` with **≥6 `|` alternation terms silently falls back to the full suite** — use single-suite or single-method filters only. `PQTrainTests` takes ~60 minutes — never run it as part of per-task verification; it runs once, in Task 16.
- **Commits:** end every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Never `git add` any `.superpowers/` path.

## Deviations from the 2026-06-22 spec (verified against live HEAD `e71daae`, 2026-07-31)

Research verified every spec site against current code. These deviations are baked into the tasks below; implementers and reviewers should treat this table (and the task briefs) as governing, not the spec's stale line numbers.

| Spec item | Deviation | Evidence |
|---|---|---|
| P1 "use the existing HNSWTraversal min-heap" | `HNSWTraversal.efSearch` requires whole-graph CSR snapshots; construction dirties the CSR cache every layer (`markCSRDirty` fires in `connect`/`pruneNeighbors`), so a literal swap-in forces an O(N+edges) rebuild per layer per insert — a regression. Task 5 instead reuses the **ScoreBlock batching pattern** against the live adjacency and adds a small arrival-order-tie-break min-heap (HNSWTraversal's private heap breaks ties by id, which would change semantics). | `HNSWIndex.swift:781-817`, `HNSWTraversal.swift:252-298`, research §3c |
| P2 "stop allocating an N+1 offsets array per edge update" | **Already done** — the A9 fix (`da605ad`, Phase 1) replaced the CSR-shaped `hnsw_prune_neighbors_f32_swift` call with flat-array `hnsw_select_neighbors_f32_swift`; no offsets array exists anymore. Remaining per-call allocs are O(m). Task 5 records closure and adds the completion-notes' suggested invariant assert. | `HNSWIndex.swift:707-751`, `docs/superpowers/2026-07-25-phase1-completion-notes.md:25` |
| P3 "norms already maintained" | **False** — `IVFIndex` never populates `IVFSelectOpts.centroidNorms`/`centroidInvNorms` anywhere. Task 6 builds the cache as new internal state. | zero grep hits in `IVFIndex.swift` |
| P3 "MatrixDistance.prepare(centroids)" | **Rejected after the spec's own mandated re-verification**: VectorCore 0.3.1's `MatrixDistance` has no raw-pointer entry point — every path is generic over `UnifiedVectorBuffer` and re-copies via `DynamicVector` wrap + `packRows` memcpy (2 extra copies minimum). Tasks 6–7 call `cblas_sgemm` directly on the already-contiguous internal buffers instead — same cross-term math, zero wrap cost. | `MatrixDistance.swift:188-200`, `DynamicVector.swift:45-48` |
| P6 "KMeansSeeding.swift:363 O(k²) streaming seeder" | **Stale** — `KMeansSeeding.swift` is already O(ndk)-correct. The real O(n·ks²) seeder is `PQTrain.swift:1416` `streamingKMeansppSeed` (which the spec's own Deferred table lists as "fix opportunistically if P6 touches it" — P6 does touch PQTrain, so Task 10 fixes it, coverage-first: it currently has **zero** test coverage). | research §1b/§2 |
| P6 "ExactRerank.swift:664" | Now inside deprecated dead code (`scoresIVF`, B17b). The live allocation site is `scoreBlock`'s three per-call buffers at `ExactRerank.swift:269-274`. Task 11 targets those. | research §5 |
| P6 "IDMap.erase return internalID instead of O(count) rescan" | **Deferred to Phase 4** — `idmapErase` has zero callers anywhere in `Sources/` (the public `IDMap` class exposes no erase at all); perf-fixing unreachable code is unverifiable. Phase 4's removal pass decides wire-up vs delete. Recorded in ledger. | research §6 |
| P6 "InnerProduct.swift:114" | **No change needed** — the `qPack` allocation is already hoisted per-call outside the row loop; only per-row allocations are worth chasing. Recorded in ledger. | research §12 |
| P6 "batch as one commit" | Relaxed to three grouped tasks (9/10/11) with one commit each — the coverage-first tests (streaming seeder, RangeQuery early-exit, ScoreBlock default path) make a single 8-site commit unreviewable. |
| Phase-3 gate "recall@k unchanged (±0)" | Relaxed to ±0.01-with-explanation for accumulation-order-changing rewrites only (P1/P3b/P6a/T13): swapping per-pair `distance()` for batched kernels physically changes FP rounding, which can flip near-tie neighbor choices. Exact ±0 is kept for all mechanical changes. The A9 precedent (baseline README) already acknowledges recall movement from topology-affecting fixes. |
| Baseline | `.bench/baseline-0.1.6/` predates the A9 topology change (its own README says recall comparisons need a re-capture), records no machine info, has no mmap-append mode, and the `--knn-clusters` value used for `knn_graph_clusters.json` is unrecoverable. Tasks 1–2 rebuild the harness and capture a fresh baseline; the cluster count is fixed at **8** and now recorded in the JSON. |

**Naming collision warning:** `docs/verification-gap-analysis-p0.md` uses "P1"/"P2" for unrelated items (mmap persistence wiring, filter pushdown). This plan's P-numbers come from `docs/superpowers/specs/2026-06-22-vectorindex-0.2.0-cleanup-design.md` §6 only.

## File Structure

| File | Role in this plan |
|---|---|
| `Sources/VectorIndexBenchmarks/main.swift` | Task 1: `--out` support for knn-graph mode, batch-QPS measurement, config completeness |
| `Tests/VectorIndexTests/MmapAppendBenchmark.swift` (new) | Task 1: `RUN_BENCHMARKS=1`-gated mmap append throughput sweep (test target because `IndexMmap` is internal) |
| `Tests/VectorIndexTests/HNSWDeterminismTests.swift` (new) | Task 2: cross-instance construction determinism |
| `Sources/VectorIndex/Kernels/VIndexMmap.swift` | Tasks 3–4: deferred section CRCs, ranged msync, instrumentation counters |
| `Sources/VectorIndex/HNSWIndex.swift` | Task 5: `searchLayer`/`greedySearchLayer` rewrite, `CandidateHeap`, graph-snapshot test hook (Task 2) |
| `Sources/VectorIndex/IVFIndex.swift` | Tasks 6–8: centroid cache, unified scoring, batch probes, optimize() rework |
| `Sources/VectorIndex/Kernels/CentroidBatchScore.swift` (new) | Task 7: `cblas_sgemm` batched query×centroid scoring |
| `Sources/VectorIndex/Kernels/PQTrain.swift` | Tasks 9–10: SIMD `l2Sq`, streaming seeder O(k) fix |
| `Sources/VectorIndex/Operations/Scoring/ScoreBlock.swift`, `.../RangeQuery/RangeQuery.swift`, `.../Rerank/ExactRerank.swift`, `Sources/VectorIndex/Filters/JournalFilter.swift` | Task 11: allocation hoists |
| `Sources/CPQEncode/pq_encode.c` | Task 12: statement-expression removal |
| `Sources/VectorIndex/Operations/Scoring/Cosine.swift` | Task 13: `precomputeInvNormsF16` delegation |
| `Tests/VectorIndexTests/HNSWAlignmentTest.swift` | Task 14: un-skip + API update |
| `Sources/VectorIndex/Operations/Reservoir/CandidateReservoir.swift` + `Tests/.../ReservoirModeBenchmarks.swift` (new) | Task 15: telemetry completeness + mode benchmark |
| `CHANGELOG.md`, `.bench/` | Tasks 2, 16 |

---

### Task 1: Benchmark harness upgrades

The gates in every later task need measurement capability the harness lacks: knn-graph mode ignores `--out` and omits config values from its JSON; nothing measures `batchSearch`; nothing measures mmap append at all. All harness changes land **before** the baseline capture (Task 2) so baseline and post-change runs use identical instrumentation. No library-source changes in this task (`Sources/VectorIndex/` untouched) — that is what makes the Task-2 capture a valid "before" picture.

**Files:**
- Modify: `Sources/VectorIndexBenchmarks/main.swift`
- Create: `Tests/VectorIndexTests/MmapAppendBenchmark.swift`
- Delete: `benchmark.swift` (repo root — dead orphan script, not referenced by any `Package.swift` target; verify with `grep -n benchmark.swift Package.swift` → no matches, then delete)

**Interfaces:**
- Produces (Task 2 depends on these): `--knn-graph` honoring `--out`; knn-graph JSON gaining `knn_clusters`, `recall_at_k`, `recall_sample`, `host` fields; `BenchResult.batchThroughputQps: Double`; test-target `MmapAppendBenchmark` writing JSON to the path in env `MMAP_BENCH_OUT`.
- Consumes: existing `outputData(_:to:)`, `HostInfo` (main.swift:535-575), `IndexMmap`/`VIndexContainerBuilder` via `@testable import`.

- [ ] **Step 1: knn-graph mode honors `--out` and emits complete config**

In `runKNNGraphBenchmark` (main.swift:99-204), replace the `print("""...""")` JSON emission (lines ~193-199) with a dictionary serialized through the same `outputData(_:to:)` path the other mode uses, adding the missing fields. Use stable key names (the old dynamic `recall_at_15_sample1000` key becomes two fields):

```swift
let payload: [String: Any] = [
    "benchmark": "hnsw_knn_graph",
    "n": config.n, "dim": config.dim, "k": config.k,
    "m": config.m, "efc": config.efc, "efs": config.efs,
    "seed": config.seed,
    "knn_clusters": config.knnClusters,
    "insert_sec": insertSec, "build_sec": buildSec,
    "points_per_sec": pointsPerSec, "edges": edges,
    "recall_at_k": recall, "recall_sample": sampleCount,
    "host": ["device": HostInfo.machineModel, "os": HostInfo.osVersion,
             "cpu": HostInfo.cpuBrand, "memoryGB": HostInfo.memoryGB]
]
let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
try outputData(data, to: config.output)   // nil → stdout, same as the other mode
```

Adapt variable names to what the function actually computes (read it first) — the values are already all in scope; only the emission changes. Keep the existing measurement logic untouched.

- [ ] **Step 2: add `batchThroughputQps` to `BenchResult` and measure it**

Add `var batchThroughputQps: Double = 0` to `BenchResult` (main.swift:491-505). In `benchHNSW` and `benchIVF` (and `benchFlat` if `FlatIndex` exposes a `batchSearch(queries:k:filter:)` — check; if it doesn't, leave flat at 0), after the existing per-query loop, add one timed batch call:

```swift
let batchStart = DispatchTime.now()
_ = try await idx.batchSearch(queries: queries, k: config.k, filter: nil)
let batchSec = Double(DispatchTime.now().uptimeNanoseconds - batchStart.uptimeNanoseconds) / 1e9
result.batchThroughputQps = batchSec > 0 ? Double(config.q) / batchSec : 0
```

Match the actual `batchSearch` signature in each index type (read the call sites in `Tests/` for the current shape).

- [ ] **Step 3: add host info to the simple-format output**

Add a `host: [String: String]` field to `BenchSuiteResult` populated from `HostInfo` (same four values as Step 1). This closes the "no record of which machine produced the baseline" gap for all future captures.

- [ ] **Step 4: create the mmap append benchmark (test target)**

`IndexMmap` and `VIndexContainerBuilder` are `internal`, so this benchmark lives in the test target behind `@testable import`, gated like `IVFSelectBenchmarks`:

```swift
import XCTest
@testable import VectorIndex

/// P4/P5 gate instrument: mmap append throughput vs commit count.
/// Enabled with RUN_BENCHMARKS=1; writes JSON to $MMAP_BENCH_OUT if set.
final class MmapAppendBenchmark: XCTestCase {
    override func setUpWithError() throws {
        if ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] != "1" {
            throw XCTSkip("Benchmarks disabled by default. Set RUN_BENCHMARKS=1 to enable.")
        }
    }

    func testAppendThroughputSweep() throws {
        let m = 16              // PQ subspaces → 16 code bytes/record
        let batch = 32          // records per commit
        let sweeps = [1_000, 2_000, 4_000, 8_000]   // commits per run
        var points: [[String: Any]] = []
        for commits in sweeps {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("mmap-bench-\(commits)-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let path = dir.appendingPathComponent("bench.vindex").path
            // Build a minimal single-list PQ container, then open it.
            // Mirror the fixture setup in Kernel30AppendTests.testDurablePQ8AppendWithRemap
            // (read that test first and reuse its builder invocation verbatim).
            let mmap = try makeDurableContainer(path: path, kc: 1, m: m)
            var ids = [UInt64](repeating: 0, count: batch)
            var codes = [UInt8](repeating: 0, count: batch * m)
            let t0 = DispatchTime.now()
            for c in 0..<commits {
                for r in 0..<batch { ids[r] = UInt64(c * batch + r) }
                for r in 0..<(batch * m) { codes[r] = UInt8((c + r) & 0xFF) }
                let res = try mmap.mmap_append_begin(listID: 0, addLen: batch)
                try ids.withUnsafeBytes { ib in
                    try codes.withUnsafeBytes { cb in
                        try mmap.mmap_append_commit(res, idsSrc: ib.baseAddress,
                                                    codesSrc: cb.baseAddress, vecsSrc: nil)
                    }
                }
            }
            let sec = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
            mmap.close()
            points.append(["commits": commits, "seconds": sec,
                           "commitsPerSec": Double(commits) / sec])
            print("mmap-append commits=\(commits) sec=\(sec) rate=\(Double(commits)/sec)/s")
        }
        if let out = ProcessInfo.processInfo.environment["MMAP_BENCH_OUT"] {
            let payload: [String: Any] = ["benchmark": "mmap_append", "batch": batch, "m": m,
                                          "points": points]
            let data = try JSONSerialization.data(withJSONObject: payload,
                                                  options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: out))
        }
    }
}
```

`makeDurableContainer` is a small file-local helper wrapping whatever `Kernel30AppendTests.testDurablePQ8AppendWithRemap` does to create + open a durable container — copy that fixture code, do not invent a new container shape. If the existing fixture creates multiple lists or a different `idCap`/`payloadCap`, keep its values; the sweep only needs *relative* timing across commit counts.

- [ ] **Step 5: smoke-verify both modes**

```bash
swift build -c release 2>&1 | tail -3
swift run -c release VectorIndexBenchmarks --knn-graph --n 200 --dim 16 --k 5 --seed 42 --out /tmp/knn-smoke.json && cat /tmp/knn-smoke.json
swift run -c release VectorIndexBenchmarks --index ivf --n 500 --q 20 --dim 32 --out /tmp/ivf-smoke.json && grep batchThroughputQps /tmp/ivf-smoke.json
RUN_BENCHMARKS=1 MMAP_BENCH_OUT=/tmp/mmap-smoke.json swift test -c release --filter MmapAppendBenchmark 2>&1 | tail -5 && cat /tmp/mmap-smoke.json
```
Expected: knn JSON contains `knn_clusters`, `host`, `recall_at_k`; ivf JSON contains a nonzero `batchThroughputQps` for the IVF row; mmap JSON has 4 sweep points. (For the smoke run, temporarily reduce the sweep via an env override or just accept the ~minutes it takes at current quadratic-CRC cost — record how long it took, that IS the "before" datapoint.)

- [ ] **Step 6: full existing-suite sanity (fast subset) + commit**

```bash
swift test --filter VIndexMmapErrorTests 2>&1 | tail -3
swift test --filter Kernel30AppendTests 2>&1 | tail -3
git add Sources/VectorIndexBenchmarks/main.swift Tests/VectorIndexTests/MmapAppendBenchmark.swift
git rm benchmark.swift
git commit -m "bench: knn-graph --out + config/host fields, batchSearch QPS, mmap append sweep

Harness-only changes ahead of the Phase-3 baseline capture; no library
source touched. Deletes the orphaned root benchmark.swift (never wired
into any Package.swift target).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Construction-determinism test + fresh baseline capture

Two deliverables: (a) the determinism gate every later task relies on — the current suite only proves *query-time* determinism (`buildKNNGraph` twice on one index); nothing asserts two independently constructed indices are identical; (b) the fresh baseline `.bench/baseline-0.2.0-pre-phase3/`, which replaces `.bench/baseline-0.1.6/` as the Phase-3 comparison point (the old one predates A9's topology change and lacks machine provenance).

**Files:**
- Modify: `Sources/VectorIndex/HNSWIndex.swift` (internal test hook only)
- Create: `Tests/VectorIndexTests/HNSWDeterminismTests.swift`
- Create: `.bench/baseline-0.2.0-pre-phase3/` (7 JSONs + README)

**Interfaces:**
- Produces: `internal struct HNSWGraphSnapshot: Equatable { let entryPoint: Int?; let maxLevel: Int; let levels: [Int]; let adjacency: [[[Int]]] }` and `internal func _testGraphSnapshot() -> HNSWGraphSnapshot` on `HNSWIndex` (Task 5's gate uses the same test).
- Consumes: Task 1's harness.

- [ ] **Step 1: add the graph snapshot hook**

In `HNSWIndex.swift`, near the other internal helpers:

```swift
// Test-only structural snapshot (internal, reached via @testable): full
// adjacency so determinism failures diff meaningfully instead of hashing.
internal struct HNSWGraphSnapshot: Equatable, Sendable {
    let entryPoint: Int?
    let maxLevel: Int
    let levels: [Int]
    let adjacency: [[[Int]]]   // [node][level][neighbor]
}

internal func _testGraphSnapshot() -> HNSWGraphSnapshot {
    HNSWGraphSnapshot(
        entryPoint: entryPoint,
        maxLevel: maxLevel,
        levels: nodes.map { $0.level },
        adjacency: nodes.map { $0.neighbors }
    )
}
```

Adapt member names to the actual stored properties (`entryPoint`/`maxLevel` — read the class header; if the entry point is stored differently, snapshot whatever uniquely identifies it). If `HNSWIndex` is an actor, the func is `internal func` and callers `await` it.

- [ ] **Step 2: write the determinism test**

```swift
import XCTest
@testable import VectorIndex

final class HNSWDeterminismTests: XCTestCase {
    // Same LCG data-gen pattern as HNSWKNNGraphTests — copy its generateDataset helper.
    func testIndependentBuildsProduceIdenticalGraphs() async throws {
        let dim = 32
        let data = generateDataset(count: 600, dim: dim, seed: 987)
        func build() async throws -> HNSWIndex {
            let idx = HNSWIndex(dimension: dim, metric: .euclidean,
                                config: .init(m: 8, efConstruction: 64, efSearch: 32))
            for (i, v) in data.enumerated() {
                try await idx.insert(id: "id\(i)", vector: v)
            }
            return idx
        }
        let a = try await build()
        let b = try await build()
        let sa = await a._testGraphSnapshot()
        let sb = await b._testGraphSnapshot()
        XCTAssertEqual(sa.entryPoint, sb.entryPoint)
        XCTAssertEqual(sa.maxLevel, sb.maxLevel)
        XCTAssertEqual(sa.levels, sb.levels)
        XCTAssertEqual(sa.adjacency, sb.adjacency,
            "same seed + same insertion order must produce a byte-identical graph")
    }
}
```

Mirror the exact `insert`/init signatures used by `HNSWKNNGraphTests.buildIndex` (read it first — the config initializer takes `rngSeed:` too; pass an explicit `rngSeed: 42` if the parameter exists so the test doesn't rely on the default constant).

- [ ] **Step 3: run it**

```bash
swift test --filter HNSWDeterminismTests 2>&1 | tail -3
```
Expected: PASS (construction is deterministic today — the level RNG is seeded and insertion is sequential). If it FAILS, stop: that is a real pre-existing nondeterminism discovery — report it to the controller instead of proceeding (Task 5's gate depends on this test being meaningful).

- [ ] **Step 4: capture the baseline (quiet machine)**

Probe first: `time swift build` (no-op) must be <10 s. Then:

```bash
mkdir -p .bench/baseline-0.2.0-pre-phase3
B=.bench/baseline-0.2.0-pre-phase3
swift run -c release VectorIndexBenchmarks --index flat --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --out $B/flat_search.json
swift run -c release VectorIndexBenchmarks --index hnsw --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --m 16 --efc 200 --efs 64 --out $B/hnsw_search.json
swift run -c release VectorIndexBenchmarks --index ivf --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --nlist 64 --nprobe 4 --out $B/ivf_search.json
swift run -c release VectorIndexBenchmarks --knn-graph --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42 --out $B/knn_graph_uniform.json
swift run -c release VectorIndexBenchmarks --knn-graph --knn-clusters 8 --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42 --out $B/knn_graph_clusters.json
RUN_BENCHMARKS=1 MMAP_BENCH_OUT=$PWD/$B/mmap_append.json swift test -c release --filter MmapAppendBenchmark 2>&1 | tail -3
```

Note the HNSW run takes ~15+ s build alone and the mmap sweep is quadratic-slow today (that's the point) — set a generous Bash timeout (600000) per command. The `--knn-clusters 8` value is a fresh choice (the 0.1.6 value is unrecoverable); it is now recorded inside the JSON.

- [ ] **Step 5: write the README and commit**

`.bench/baseline-0.2.0-pre-phase3/README.md`:

```markdown
# Baseline: 0.2.0 pre-Phase-3

Captured on `main` @ e71daae + Phase-3 Task-1 harness changes (harness-only;
no library code differs from e71daae). This supersedes `.bench/baseline-0.1.6/`
as the Phase-3 gate reference: it is post-A9 (topology fix) and post-Phase-2,
records host info inside each JSON, includes batchSearch QPS and the mmap
append sweep, and pins `--knn-clusters 8` (recorded in-file; the 0.1.6 value
was never recorded).

Gate rule: Phase-3 items compare against THESE numbers, same machine (see the
`host` block in each JSON), Release build, quiet machine.
```

```bash
git add .bench/baseline-0.2.0-pre-phase3 Sources/VectorIndex/HNSWIndex.swift Tests/VectorIndexTests/HNSWDeterminismTests.swift
git commit -m "bench: capture 0.2.0 pre-Phase-3 baseline + construction-determinism test

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: P4 — defer per-commit section CRCs (quadratic → linear ingestion)

`mmap_append_commit` calls `updateSectionCRC` 3–4× per commit, and each call is `CRC32.hash(p, sz)` over the **entire** section's allocated capacity (`VIndexMmap.swift:575-600`) — N appends into a growing section cost O(N²) total CRC work, and sections are container-wide (one TOC entry covers all lists), so a commit to list 0 re-hashes every list's storage. Sections receive writes at arbitrary per-list offsets, so a rolling/incremental section CRC is not possible; the spec's other sanctioned option — **defer to flush/close** — is the design here. Durability is preserved because the WAL already provides per-commit crash consistency (append + commit records, both CRC'd, fsync'd — `docs/kernel-specs/DONE_S_serialization_mmap.md:71-75`): payload bytes stay msync'd per commit; only the *checksum freshness* moves to close/flush, with WAL-replay recompute covering the crash window.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/VIndexMmap.swift`
- Test: `Tests/VectorIndexTests/VIndexMmapErrorTests.swift` (extend), `Tests/VectorIndexTests/Kernel30AppendTests.swift` (keep green)

**Interfaces:**
- Produces: `internal private(set) var crcBytesHashed: Int` on `IndexMmap` (Task 16 and this task's tests read it); `func flush() throws` on `IndexMmap` (internal class, so no public-surface impact); clean/dirty on-disk marker semantics described below.
- Consumes: Task 1's `MmapAppendBenchmark` for the throughput gate.

**Design (implementer: verify each premise against the code before writing — STOP and report if any premise is wrong):**

1. **Marker:** the WAL file itself is the unclean marker. Verify what `close()` and `mmap_wal_replay` do with the WAL file today (is it truncated on clean close? left in place?). If clean-close WAL truncation does not exist, add it: `close()` recomputes all section CRCs, msyncs, then truncates the WAL to 0 (`ftruncate(walFD, 0)` + fsync). An empty/absent WAL ⇒ clean shutdown ⇒ section CRCs are trustworthy.
2. **Commit path:** in `mmap_append_commit`, drop the three/four `updateSectionCRC(...)` calls. Everything else (WAL records, memcpys, msyncs, ListsDesc length update) stays.
3. **Open path:** in `open()`/`indexInit`, when the WAL is non-empty (unclean shutdown): skip the per-section CRC *verification* (header CRC still verified), run WAL replay as today, then recompute + persist all section CRCs and truncate the WAL. When the WAL is empty (clean shutdown): verify section CRCs exactly as today.
4. **Instrumentation:** add `internal private(set) var crcBytesHashed: Int = 0` incremented inside `updateSectionCRC` (and any new recompute helper) by `sz` before hashing. This is the deterministic gate: after Step-2's change, a run of K commits must add **zero** to `crcBytesHashed`.

- [ ] **Step 1: write the failing counter test**

Add to `VIndexMmapErrorTests.swift` (reuse its container fixture helpers):

```swift
func testCommitPathDefersSectionCRCs() throws {
    let (mmap, _) = try makeFixtureContainer()   // reuse/extract the existing fixture helper
    let before = mmap.crcBytesHashed
    // 50 small commits
    for c in 0..<50 {
        let res = try mmap.mmap_append_begin(listID: 0, addLen: 4)
        // ... ids/codes buffers as in the existing append tests ...
        try mmap.mmap_append_commit(res, idsSrc: ids, codesSrc: codes, vecsSrc: nil)
    }
    XCTAssertEqual(mmap.crcBytesHashed - before, 0,
        "commits must not hash section bytes; CRCs are deferred to flush/close")
    try mmap.flush()
    XCTAssertGreaterThan(mmap.crcBytesHashed - before, 0,
        "flush recomputes and persists section CRCs")
}
```

Also write the crash-window test:

```swift
func testUncleanCloseThenReopenRecomputesCRCsViaWAL() throws {
    // Append + commit, then drop the instance WITHOUT close() (simulate crash;
    // if IndexMmap's deinit closes cleanly, add an internal test-only
    // `_abandonWithoutClose()` that munmaps without CRC recompute/WAL truncate).
    // Reopen: must NOT throw section-CRC mismatch; replay applies lengths;
    // data reads back; a SECOND reopen (now clean) verifies CRCs strictly.
}
```

- [ ] **Step 2: run both to verify they fail**

```bash
swift test --filter VIndexMmapErrorTests 2>&1 | tail -5
```
Expected: the new tests fail (`crcBytesHashed`/`flush()` don't exist yet → compile error first; stub them, then assert-fail).

- [ ] **Step 3: implement the design (points 1–4 above)**

Keep `updateSectionCRC` itself (flush/close/replay call it); the change is *who calls it when*. The recompute-all helper:

```swift
private func recomputeAllSectionCRCs() throws {
    for ty in tocByType.keys { try updateSectionCRC(ty) }
}

func flush() throws {
    try recomputeAllSectionCRCs()
    msyncPageAligned(base, Int(fileSize))
    try truncateWAL()
}
```

`close()` calls `flush()` before unmapping. `truncateWAL()` = `ftruncate(walFD, 0)` + `fsync(walFD)` (guard walFD ≥ 0). Open-path logic per design point 3 — find where section CRCs are verified (`indexInit`, `VIndexMmap.swift:415-429`) and branch on WAL emptiness (`fstat(walFD).st_size > 0`).

- [ ] **Step 4: run the full mmap test surface**

```bash
swift test --filter VIndexMmapErrorTests 2>&1 | tail -5
swift test --filter Kernel30AppendTests 2>&1 | tail -3
swift test --filter RegressionA3_RemapTOCTests 2>&1 | tail -3
swift test --filter RegressionA2_DurableListStatsTests 2>&1 | tail -3
swift test --filter RegressionA7_TOCReservationTests 2>&1 | tail -3
```
(Separate invocations — remember the ≥6-term `--filter` alternation bug.) Expected: all green, including the pre-existing corruption tests: `testSectionCRCMismatchThrows` corrupts *after a clean close*, so strict verification still fires; `testWalReplay*` tests still pass because replay behavior is unchanged apart from the added CRC recompute at the end (verify those two tests' fixtures don't now hit the recompute in a way that masks their corruption — read them and adjust only if a test's *setup* relied on commit-time CRC persistence).

- [ ] **Step 5: measure the gate**

```bash
RUN_BENCHMARKS=1 MMAP_BENCH_OUT=/tmp/mmap_post_p4.json swift test -c release --filter MmapAppendBenchmark 2>&1 | tail -3
```
Gate: compare `/tmp/mmap_post_p4.json` against `.bench/baseline-0.2.0-pre-phase3/mmap_append.json`. Baseline slope is quadratic (time(8k)/time(1k) ≈ 40–64×); post-change must be near-linear (ratio < 12×) and absolute `commitsPerSec` must improve at every sweep point. Record both ratios in the ledger.

- [ ] **Step 6: commit**

```bash
git add Sources/VectorIndex/Kernels/VIndexMmap.swift Tests/VectorIndexTests/VIndexMmapErrorTests.swift
git commit -m "perf(mmap): defer section CRCs to flush/close (P4) — linear ingestion

Per-commit full-section CRC recompute made N appends O(N²). Sections take
writes at arbitrary per-list offsets, so rolling CRCs can't work; instead
commits skip section-CRC updates entirely and flush()/close() recompute +
persist them, truncating the WAL as the clean-shutdown marker. Unclean
open skips strict section verification, replays the WAL (records are
individually CRC'd), then recomputes. Durability per commit is unchanged:
payload msyncs and WAL fsyncs still happen in the same order.

Gate: mmap append sweep time(8k)/time(1k) <baseline ratio> → <new ratio>.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: P5 — ranged page-aligned msync + per-commit flush accounting

`msyncPageAligned` (`VIndexMmap.swift:382-385`) ignores its ptr/len parameters and always runs `msync(base, fileSize, MS_SYNC)` — a full-mapping synchronous flush per call, 6–8×/commit before Task 3, still 3–4×/commit after (ids, codes/vecs, ListsDesc). Fix: honor ptr/len with explicit page alignment (that was clearly the function's original intent, hence its name), and count calls/bytes for a deterministic gate.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/VIndexMmap.swift`
- Test: `Tests/VectorIndexTests/VIndexMmapErrorTests.swift` (extend)

**Interfaces:**
- Produces: `internal private(set) var msyncCallCount: Int`, `internal private(set) var msyncBytesFlushed: Int` on `IndexMmap`.
- Consumes: Task 3's shape of the commit path (this task runs after it; per-commit msync sites are ids + codes/vecs + ListsDesc only).

- [ ] **Step 1: write the failing counter test**

```swift
func testCommitFlushesOnlyTouchedPages() throws {
    let (mmap, _) = try makeFixtureContainer()
    let pageSize = Int(getpagesize())
    let callsBefore = mmap.msyncCallCount
    let bytesBefore = mmap.msyncBytesFlushed
    let res = try mmap.mmap_append_begin(listID: 0, addLen: 4)
    try mmap.mmap_append_commit(res, idsSrc: ids, codesSrc: codes, vecsSrc: nil)
    let calls = mmap.msyncCallCount - callsBefore
    let bytes = mmap.msyncBytesFlushed - bytesBefore
    XCTAssertEqual(calls, 3, "PQ commit = ids + codes + listsDesc flushes only")
    XCTAssertLessThanOrEqual(bytes, 3 * 2 * pageSize,
        "each flush covers only the touched range rounded to page boundaries, not fileSize")
}
```

(3 flushes × at most 2 pages each for a tiny write spanning a boundary. If the fixture is flat-format, expect 4 calls — match the fixture.)

- [ ] **Step 2: run to verify it fails** — `swift test --filter VIndexMmapErrorTests 2>&1 | tail -5` (compile error → stub counters → assert-fail on `bytes`, since today each call flushes `fileSize`).

- [ ] **Step 3: implement**

```swift
private(set) var msyncCallCount: Int = 0
private(set) var msyncBytesFlushed: Int = 0

@inline(__always) private func msyncPageAligned(_ ptr: UnsafeMutableRawPointer, _ length: Int) {
    let pageSize = Int(getpagesize())
    let baseAddr = Int(bitPattern: base)
    let start = Int(bitPattern: ptr)
    // Round start down to a page boundary, end up, and clamp to the mapping.
    let alignedStart = max(start & ~(pageSize - 1), baseAddr)
    let alignedEnd = min((start + length + pageSize - 1) & ~(pageSize - 1),
                         baseAddr + Int(fileSize))
    let len = alignedEnd - alignedStart
    guard len > 0 else { return }
    msyncCallCount &+= 1
    msyncBytesFlushed &+= len
    _ = msync(UnsafeMutableRawPointer(bitPattern: alignedStart)!, len, MS_SYNC)
}
```

The old body's comment cited "sub-page msync pitfalls on macOS" — the explicit page alignment above is the standard remedy (msync requires a page-aligned address; length rounding avoids EINVAL). Verify every existing call site passes the *actual touched* ptr/len (they do — the parameters were always correct, just ignored). `flush()`/`close()` from Task 3 intentionally still flush the whole mapping — that is one full flush per session, not per commit.

- [ ] **Step 4: verify durability semantics survive**

Same five suites as Task 3 Step 4, one filter per invocation, all green. The WAL-replay and reopen tests are the real durability check here: committed bytes must be on disk when `mmap_append_commit` returns even though only touched pages were flushed.

- [ ] **Step 5: measure + record**

```bash
RUN_BENCHMARKS=1 MMAP_BENCH_OUT=/tmp/mmap_post_p5.json swift test -c release --filter MmapAppendBenchmark 2>&1 | tail -3
```
Gate: `commitsPerSec` ≥ post-Task-3 numbers at every point (syscall reduction should show up most at small commit sizes). Record the delta chain (baseline → P4 → P5) in the ledger.

- [ ] **Step 6: commit** (same file set pattern as Task 3; message `perf(mmap): honor ptr/len in msyncPageAligned (P5) — 3 ranged flushes per commit` with the measured numbers and trailer).

---

### Task 5: P1 — heap + batched-ScoreBlock construction traversal (and P2 closure)

The construction hot path `searchLayer` (`HNSWIndex.swift:781-817`) pops candidates with an O(c) linear scan, maintains its result set with an O(ef) linear-scan insert, and — worst — scores **one neighbor at a time** through `vectorArray(at:)`, which heap-allocates a fresh `[Float]` copy per candidate before calling the generic VectorCore `distance()`. `greedySearchLayer` (`:763-779`) has the same per-neighbor copy+call shape. Query-time `search()` already does this right (`HNSWTraversal`: min-heap + batched `ScoreBlock` over pinned contiguous storage); construction never got the treatment. This task rewrites both functions in place against the **live adjacency** (see Deviations — the whole-graph CSR route is a regression), reusing `IndexOps.Scoring.ScoreBlock` for batched scoring and adding a small min-heap whose tie-break reproduces the current arrival-order semantics exactly.

**Semantics contract (the reviewer checks the code against this):**
- Old pop rule: first index with strictly smallest distance in an insertion-ordered array ⇒ **(distance asc, arrival-sequence asc)** total order. The new heap keys on exactly that pair.
- Old result insert: ascending by distance, ties placed **after** equal entries (upper-bound insertion) ⇒ binary-search upper-bound insert preserves it.
- Old acceptance: `result.count < ef || d < worst` (strict `<`, ties with the current worst are rejected when full) — preserved.
- Old visited rule: node is marked visited before the deleted-check; deleted nodes are consumed from `visited` but never scored — preserved.
- Distance values change representation (euclidean: L2² instead of `sqrt`; cosine/dot: same transforms ScoreBlock+conversion produce) — **ordering-equivalent**, and no distance value escapes `searchLayer`/`greedySearchLayer` (they return ids only; verify at both call sites).

**Files:**
- Modify: `Sources/VectorIndex/HNSWIndex.swift` (`searchLayer`, `greedySearchLayer`, delete `insertSorted`, delete `vectorArray(at:)` if unreferenced afterward — `update`/`save`/post-searchLayer-closest still use it at `:489/:667/:1092`; leave those, they're cold paths)
- Test: existing suites are the gate (no new test file; Task 2's determinism test + A9 + recall suites)

**Interfaces:**
- Consumes: `IndexOps.Scoring.ScoreBlock.run(q:xb:n:d:metric:out:cosineNorms:)` (`Operations/Scoring/ScoreBlock.swift:24`), `rebuildInvNormsIfNeededForCosine()` (existing, used by `pruneNeighbors`), `HNSWGraphSnapshot` test hook (Task 2).
- Produces: no interface changes — both functions keep their exact signatures (`private func searchLayer(_ query: [Float], enter: Int, ef: Int, level: Int) -> [Int]`, `private func greedySearchLayer(_ query: [Float], enter: Int, level: Int) -> Int`).

- [ ] **Step 1: record the before numbers**

```bash
swift run -c release VectorIndexBenchmarks --index hnsw --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --m 16 --efc 200 --efs 64 --out /tmp/hnsw_before_p1.json
```
(Should match the Task-2 baseline within noise; this re-run is the same-session anchor.)

- [ ] **Step 2: add the candidate heap (file-private, in HNSWIndex.swift)**

```swift
// P1: bounded-free min-heap over (distance, arrival sequence). Pop order
// reproduces the old linear-scan pop exactly: strictly smallest distance
// first; among equal distances, the earliest-pushed entry.
private struct CandidateHeap {
    private var dists: [Float] = []
    private var seqs: [Int32] = []
    private var ids: [Int32] = []
    private var nextSeq: Int32 = 0

    var isEmpty: Bool { dists.isEmpty }

    mutating func reserveCapacity(_ n: Int) {
        dists.reserveCapacity(n); seqs.reserveCapacity(n); ids.reserveCapacity(n)
    }

    @inline(__always) private func precedes(_ i: Int, _ j: Int) -> Bool {
        dists[i] != dists[j] ? dists[i] < dists[j] : seqs[i] < seqs[j]
    }

    @inline(__always) private mutating func swapAt(_ a: Int, _ b: Int) {
        dists.swapAt(a, b); seqs.swapAt(a, b); ids.swapAt(a, b)
    }

    mutating func push(id: Int32, dist: Float) {
        dists.append(dist); seqs.append(nextSeq); ids.append(id); nextSeq &+= 1
        var c = dists.count - 1
        while c > 0 {
            let p = (c - 1) >> 1
            if precedes(c, p) { swapAt(c, p); c = p } else { break }
        }
    }

    mutating func popMin() -> (id: Int32, dist: Float) {
        let outID = ids[0], outDist = dists[0]
        let last = dists.count - 1
        swapAt(0, last)
        dists.removeLast(); seqs.removeLast(); ids.removeLast()
        var p = 0
        while true {
            let l = 2 * p + 1, r = l + 1
            var m = p
            if l < dists.count, precedes(l, m) { m = l }
            if r < dists.count, precedes(r, m) { m = r }
            if m == p { break }
            swapAt(p, m); p = m
        }
        return (outID, outDist)
    }
}
```

- [ ] **Step 3: rewrite `searchLayer`**

```swift
private func searchLayer(_ query: [Float], enter: Int, ef: Int, level: Int) -> [Int] {
    let d = dimension
    let invNormsPtr: UnsafePointer<Float>? =
        (metric == .cosine) ? rebuildInvNormsIfNeededForCosine() : nil

    return query.withUnsafeBufferPointer { qbp -> [Int] in
        vectorStorage.withUnsafeBufferPointer { xbp -> [Int] in
            let qPtr = qbp.baseAddress!
            let xBase = xbp.baseAddress!

            var heap = CandidateHeap()
            heap.reserveCapacity(2 * ef)
            var resIDs: [Int] = []; var resDists: [Float] = []
            resIDs.reserveCapacity(ef + 1); resDists.reserveCapacity(ef + 1)
            var visited = Set<Int>()

            // Reused per-expansion scratch (grown on demand, never per-neighbor).
            var batchIDs: [Int] = []
            var gather: [Float] = []
            var gatherInv: [Float] = []          // cosine only
            var scores: [Float] = []
            let qInvNorm: Float? = (metric == .cosine)
                ? 1.0 / (IndexOps.Support.Norms.l2NormSquared(vector: qPtr, dimension: d)
                             .squareRoot() + 1e-12)
                : nil

            // Scores `batchIDs` against the query in ONE ScoreBlock call; writes
            // "smaller is better" distances into `scores[0..<count]`.
            func scoreBatch() {
                let n = batchIDs.count
                if gather.count < n * d { gather = [Float](repeating: 0, count: n * d) }
                if scores.count < n { scores = [Float](repeating: 0, count: n) }
                gather.withUnsafeMutableBufferPointer { gb in
                    for (i, id) in batchIDs.enumerated() {
                        (gb.baseAddress! + i * d)
                            .update(from: xBase + nodes[id].vectorOffset, count: d)
                    }
                }
                if metric == .cosine, let inv = invNormsPtr {
                    if gatherInv.count < n { gatherInv = [Float](repeating: 0, count: n) }
                    for (i, id) in batchIDs.enumerated() { gatherInv[i] = inv[id] }
                }
                // POINTER-LIFETIME: the CosineNormsHandle wraps a pointer into
                // gatherInv — it must be constructed AND consumed inside
                // gatherInv's withUnsafeBufferPointer scope, never stored.
                func runScore(_ handle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle?) {
                    gather.withUnsafeBufferPointer { gb in
                        scores.withUnsafeMutableBufferPointer { sb in
                            IndexOps.Scoring.ScoreBlock.run(
                                q: qPtr, xb: gb.baseAddress!, n: n, d: d,
                                metric: metric, out: sb.baseAddress!, cosineNorms: handle)
                            // Convert to "smaller is better", ordering-equivalent to
                            // the old distance(): euclidean L2² needs nothing (monotone
                            // in L2); dot/cosine flip higher-is-better scores.
                            switch metric {
                            case .euclidean: break
                            case .dotProduct: for i in 0..<n { sb[i] = -sb[i] }
                            case .cosine:     for i in 0..<n { sb[i] = 1 - sb[i] }
                            default: break   // ScoreBlock fallback already returns distances
                            }
                        }
                    }
                }
                if metric == .cosine {
                    gatherInv.withUnsafeBufferPointer { ib in
                        runScore(.init(dbInvNormsF32: ib.baseAddress, dbInvNormsF16: nil,
                                       queryInvNorm: qInvNorm, epsilon: 1e-12))
                    }
                } else {
                    runScore(nil)
                }
            }

            @inline(__always) func insertResult(_ id: Int, _ dist: Float) {
                // Upper-bound binary search: ties land AFTER equal entries,
                // matching the old insertSorted's arrival-order tie-break.
                var lo = 0, hi = resDists.count
                while lo < hi {
                    let mid = (lo + hi) >> 1
                    if resDists[mid] > dist { hi = mid } else { lo = mid + 1 }
                }
                resIDs.insert(id, at: lo); resDists.insert(dist, at: lo)
                if resIDs.count > ef { resIDs.removeLast(); resDists.removeLast() }
            }

            batchIDs = [enter]
            scoreBatch()
            let enterDist = scores[0]
            heap.push(id: Int32(enter), dist: enterDist)
            insertResult(enter, enterDist)
            visited.insert(enter)

            while !heap.isEmpty {
                let (cand32, candDist) = heap.popMin()
                if resIDs.count >= ef, let worst = resDists.last, candDist > worst { break }

                batchIDs.removeAll(keepingCapacity: true)
                for n in nodes[Int(cand32)].neighbors[safe: level] ?? [] {
                    if visited.insert(n).inserted, !nodes[n].isDeleted {
                        batchIDs.append(n)
                    }
                }
                guard !batchIDs.isEmpty else { continue }
                scoreBatch()
                for (i, nid) in batchIDs.enumerated() {
                    let nd = scores[i]
                    if resIDs.count < ef || nd < (resDists.last ?? .infinity) {
                        heap.push(id: Int32(nid), dist: nd)
                        insertResult(nid, nd)
                    }
                }
            }
            return resIDs
        }
    }
}
```

Delete `insertSorted` (now unreferenced). Adapt names (`nodes`, `vectorStorage`, `IndexOps.Support.Norms.l2NormSquared` — verify the exact norms helper name via `Operations/Support/Norms.swift:104`). **Pointer-lifetime rule (A1 lesson):** `qPtr`/`xBase`/`invNormsPtr` never escape the two `withUnsafeBufferPointer` closures, and `searchLayer` performs no mutation of `vectorStorage`/`nodes` — confirm no code path inside can call `insert`/`connect`.

- [ ] **Step 4: rewrite `greedySearchLayer` with the same batched scoring**

Same scratch pattern, no heap needed: score all of `cur`'s neighbors in one `scoreBatch()` call, then scan `scores` in neighbor order with strict `<` (preserves first-best-on-tie), loop until no improvement. Keep the signature and the deleted-node behavior identical to the current code (the current version does **not** skip deleted nodes here — preserve that; changing it is out of scope).

- [ ] **Step 5: P2 closure — invariant assert in `pruneNeighbors`**

From the Phase-1 completion notes' carried suggestion: in `pruneNeighbors`, after `let current = nodes[idx].neighbors[level]`, add:

```swift
assert(!current.contains(idx), "neighbor list must never contain the node itself")
```

- [ ] **Step 6: build + full HNSW test surface**

```bash
swift build 2>&1 | tail -3
swift test --filter HNSWDeterminismTests 2>&1 | tail -3
swift test --filter HNSWKNNGraphTests 2>&1 | tail -5
swift test --filter HNSWRecallTests 2>&1 | tail -3
swift test --filter RegressionA1_TraversalLifetimeTests 2>&1 | tail -3
swift test --filter HNSWWALTests 2>&1 | tail -3
swift test --filter HNSWNeighborSelectionTests 2>&1 | tail -3
swift test --filter HNSWTraversalKernelTests 2>&1 | tail -3
swift test --filter PersistenceEdgeTests 2>&1 | tail -3
```
All green, including `testKnownIssue_SequentialClusterInsertDisconnectsGraph` (A9, hard assertion) and the WAL replay-determinism test.

- [ ] **Step 7: benchmark gate**

```bash
swift run -c release VectorIndexBenchmarks --index hnsw --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --m 16 --efc 200 --efs 64 --out /tmp/hnsw_after_p1.json
swift run -c release VectorIndexBenchmarks --knn-graph --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42 --out /tmp/knn_uniform_after_p1.json
swift run -c release VectorIndexBenchmarks --knn-graph --knn-clusters 8 --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42 --out /tmp/knn_clusters_after_p1.json
```
Gates vs `.bench/baseline-0.2.0-pre-phase3/`:
- `buildSeconds` (hnsw) and `insert_sec` (both knn-graph runs) **down** — this is the headline number; the baseline builds ~15 s at n=5000.
- `recallAvg` / `recall_at_k` within ±0.01 of baseline; any nonzero delta explained in the ledger (expected cause: FP-rounding tie flips from the batched scorer).
- `searchAvgMs`/`throughputQps` not regressed (query path is untouched, but `getCandidates` shares `searchLayer` — confirm no regression).
- Determinism test green (Step 6) = fixed seed still yields identical graphs.
Also run one cosine-metric sanity: `swift run -c release VectorIndexBenchmarks --index hnsw --n 2000 --q 50 --dim 64 --metric cosine` (stdout) — recall sane (>0.3), no crash: this exercises the `CosineNormsHandle` path.

- [ ] **Step 8: commit**

```bash
git add Sources/VectorIndex/HNSWIndex.swift
git commit -m "perf(hnsw): heap + batched-ScoreBlock construction traversal (P1); P2 closed by A9

searchLayer: O(c) linear pop → (dist, arrival-seq) min-heap; per-neighbor
vectorArray(at:) copies + scalar distance() calls → one gathered ScoreBlock
batch per expansion; result set → binary-search upper-bound insert. Tie-break
and acceptance semantics preserved exactly (arrival-order ties, strict-<
acceptance). greedySearchLayer batched the same way. P2's N+1 offsets array
was already removed by A9 (da605ad); adds the neighbor-list self-reference
assert from the Phase-1 completion notes.

Gate: build <before>s → <after>s; knn insert <before> → <after>; recall
delta <x>; determinism + A9 + WAL replay green.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: P3a — contiguous centroid cache + one batched centroid-scoring path

`IVFIndex` stores centroids as `[[Float]]` and has **four** independent scalar per-centroid `distance()` loops (`nearestCentroidIndex` :322-332, `search` :454-465, `performIVFSearch` :589-596, `getCandidates` :780-785) that never touch the batched `IVFSelect` kernel. Additionally, its one `IVFSelect` call (`searchKernel30Flat` :911) passes default `IVFSelectOpts()` — no centroid norms — so the kernel's already-implemented dot-trick branch never activates (the spec's "norms already maintained" was false). This task adds the contiguous mirror + cached norms, funnels all four loops through one batched helper, and wires the norms into the Kernel-30 path.

**Files:**
- Modify: `Sources/VectorIndex/IVFIndex.swift`
- Test: existing IVF suites (listed in Step 5) are the gate; no new test file

**Interfaces:**
- Produces (Tasks 7–8 depend on these exact names):
  - `private var centroidsFlat: ContiguousArray<Float>` (kc×d row-major)
  - `private var centroidNormsSq: [Float]` (‖c‖² per centroid)
  - `private var centroidInvNorms: [Float]` (1/(‖c‖+1e-12) per centroid)
  - `private func rebuildCentroidCache()`
  - `private func centroidDistances(for query: [Float], queryIsNormalized: Bool = false) -> [Float]` — kc "smaller is better" values, ordering-equivalent to the old per-pair `distance()` (euclidean returns L2² — callers only argmin/sort, verify no value escapes into results)
- Consumes: `IndexOps.Scoring.ScoreBlock.run`, `IndexOps.Support.Norms.l2NormSquared`.

- [ ] **Step 1: add the cache**

```swift
// P3a: contiguous mirror of `centroids` + cached norms. Rebuilt whenever
// `centroids` is reassigned; single source for all centroid scoring.
private var centroidsFlat: ContiguousArray<Float> = []
private var centroidNormsSq: [Float] = []
private var centroidInvNorms: [Float] = []

private func rebuildCentroidCache() {
    let d = dimension, kc = centroids.count
    centroidsFlat.removeAll(keepingCapacity: true)
    centroidsFlat.reserveCapacity(kc * d)
    for c in centroids { centroidsFlat.append(contentsOf: c) }
    centroidNormsSq = [Float](repeating: 0, count: kc)
    centroidInvNorms = [Float](repeating: 0, count: kc)
    centroidsFlat.withUnsafeBufferPointer { cb in
        guard let base = cb.baseAddress else { return }
        for i in 0..<kc {
            let n2 = IndexOps.Support.Norms.l2NormSquared(vector: base + i * d, dimension: d)
            centroidNormsSq[i] = n2
            centroidInvNorms[i] = 1.0 / (n2.squareRoot() + 1e-12)
        }
    }
}
```

Call it after **every** assignment to `centroids`: `grep -n "centroids = \|centroids.removeAll\|centroids\.append" Sources/VectorIndex/IVFIndex.swift` and add `rebuildCentroidCache()` after each mutation site (optimize's `centroids = try await kmeans(...)`, the empty-store `centroids.removeAll()` path, `clear()`, and any load/deserialize path — enumerate them all in the report).

- [ ] **Step 2: add the batched scorer**

```swift
private func centroidDistances(for query: [Float], queryIsNormalized: Bool = false) -> [Float] {
    let kc = centroids.count, d = dimension
    var out = [Float](repeating: 0, count: kc)
    guard kc > 0 else { return out }
    query.withUnsafeBufferPointer { qb in
        centroidsFlat.withUnsafeBufferPointer { cb in
            // POINTER-LIFETIME: the CosineNormsHandle wraps a pointer into
            // centroidInvNorms — construct AND consume it inside that array's
            // withUnsafeBufferPointer scope, never store it beyond.
            func runScore(_ handle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle?) {
                out.withUnsafeMutableBufferPointer { ob in
                    IndexOps.Scoring.ScoreBlock.run(
                        q: qb.baseAddress!, xb: cb.baseAddress!, n: kc, d: d,
                        metric: metric, out: ob.baseAddress!, cosineNorms: handle)
                    switch metric {
                    case .euclidean: break            // L2², monotone-equivalent
                    case .dotProduct: for i in 0..<kc { ob[i] = -ob[i] }
                    case .cosine:     for i in 0..<kc { ob[i] = 1 - ob[i] }
                    default: break
                    }
                }
            }
            if metric == .cosine {
                let qInv: Float = queryIsNormalized
                    ? 1.0
                    : 1.0 / (IndexOps.Support.Norms.l2NormSquared(
                          vector: qb.baseAddress!, dimension: d).squareRoot() + 1e-12)
                centroidInvNorms.withUnsafeBufferPointer { ib in
                    runScore(.init(dbInvNormsF32: ib.baseAddress, dbInvNormsF16: nil,
                                   queryInvNorm: qInv, epsilon: 1e-12))
                }
            } else {
                runScore(nil)
            }
        }
    }
    return out
}
```

- [ ] **Step 3: rewrite the four call sites**

- `nearestCentroidIndex(for:)` → `let dists = centroidDistances(for: vector); return argmin` (empty → nil, preserve exact current nil semantics).
- `search` probe loop → `let dists = centroidDistances(for: query, queryIsNormalized: queryIsNormalized); var centroidDists = Array(dists.enumerated().map { ($0.offset, $0.element) })` then the existing sort/probe slice unchanged.
- `performIVFSearch` (static, context-based): add the flat cache + norms (+ metric) to the batch context struct so the static helper can score without actor state; replicate the helper as a static function or inline the same ScoreBlock call. Keep the context `Sendable`.
- `getCandidates` probe loop → same as `search`.
Verify with `grep -n "distance(" Sources/VectorIndex/IVFIndex.swift` that **no remaining per-centroid scalar loop survives** (per-vector `distance()` calls in list scanning are a different path — leave those; they are the rerank/list-scan cost, not the coarse quantizer).

- [ ] **Step 4: wire norms into the Kernel-30 path**

In `searchKernel30Flat` (:911 area), replace the default `IVFSelectOpts()` with one carrying the cache: `var opts = IVFSelectOpts(); opts.centroidNorms = centroidNormsSq; opts.centroidInvNorms = centroidInvNorms` (match the real field names/types in `IVFSelect.swift` — read `IVFSelectOpts` first). Also replace that path's transient `centroids.flatMap { $0 }` (:910) with `centroidsFlat` — one less O(kc·d) materialization per search.

- [ ] **Step 5: run the IVF test surface**

```bash
swift test --filter IVFRecallTests 2>&1 | tail -3
swift test --filter IVFKMeansPlusPlusTests 2>&1 | tail -3
swift test --filter IVFTests 2>&1 | tail -3
swift test --filter IVFMoreTests 2>&1 | tail -3
swift test --filter IVFListMaintenanceTests 2>&1 | tail -3
swift test --filter IVFProbeMonotonicTests 2>&1 | tail -3
swift test --filter IVFFlatRerankTests 2>&1 | tail -3
swift test --filter IVFListVecsReaderRerankTests 2>&1 | tail -3
swift test --filter IVFSelectTests 2>&1 | tail -3
```
All green. `testDotTrickEquivalence` (IVFSelectTests) is the pre-existing guard that the now-activated dot-trick branch matches direct L2.

- [ ] **Step 6: benchmark + commit**

```bash
swift run -c release VectorIndexBenchmarks --index ivf --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --nlist 64 --nprobe 4 --out /tmp/ivf_after_p3a.json
```
Gate: `recallAvg` bit-close to baseline (this change is ordering-equivalent scoring; small FP tie flips possible → ±0.01 rule, explain in ledger), `throughputQps` and `searchAvgMs` not regressed (expect modest improvement; the probe is a small share of IVF search at nlist=64 — say so honestly in the ledger rather than overclaiming). Commit as `perf(ivf): contiguous centroid cache + batched coarse scoring (P3a)` with the standard trailer.

---

### Task 7: P3b — one `cblas_sgemm` cross-term for batch probes

The batch path (`batchSearch`) still probes centroids once per query. With the Task-6 cache in place, all q×kc probe scores can come from a single GEMM. `MatrixDistance` was rejected (see Deviations — mandatory double copy); this calls `cblas_sgemm` directly, VectorIndex's first direct BLAS use (`import Accelerate` is already present in 4 kernel files; this adds a 5th site).

**Files:**
- Create: `Sources/VectorIndex/Kernels/CentroidBatchScore.swift`
- Modify: `Sources/VectorIndex/IVFIndex.swift` (`batchSearch` + batch context)
- Test: `Tests/VectorIndexTests/IVFBatchGEMMParityTests.swift` (new)

**Interfaces:**
- Produces (Task 8 uses this for assignment fallback):

```swift
internal enum CentroidBatchScore {
    /// One sgemm cross-term for q queries × kc centroids, row-major.
    /// out[qi*kc + ci] is "smaller is better", ordering-equivalent per row to
    /// DistanceUtils.distance:
    ///   euclidean  → ‖c‖² − 2⟨q,c⟩          (‖q‖² omitted: constant per row)
    ///   dotProduct → −⟨q,c⟩
    ///   cosine     → 1 − ⟨q,c⟩·qInv·cInv
    /// Returns false (out untouched) for metrics without a GEMM form
    /// (manhattan/chebyshev) — caller falls back to the per-query path.
    static func run(
        queries: UnsafePointer<Float>, q: Int,
        centroids: UnsafePointer<Float>, kc: Int, d: Int,
        metric: SupportedDistanceMetric,
        centroidNormsSq: [Float], centroidInvNorms: [Float],
        queriesAreNormalized: Bool,
        out: inout [Float]
    ) -> Bool
}
```

- Consumes: Task 6's `centroidsFlat`/`centroidNormsSq`/`centroidInvNorms`.

- [ ] **Step 1: write the parity test first**

```swift
import XCTest
@testable import VectorIndex

final class IVFBatchGEMMParityTests: XCTestCase {
    // Seeded fixture: 800 vectors / 24 dims / nlist 16, 60 queries.
    // batchSearch must agree with per-query search() almost everywhere;
    // FP-margin probe flips are tolerated but bounded, and the fixture is
    // fully seeded so any drift is deterministic, not flaky.
    func testBatchMatchesSingleQuerySearch() async throws {
        let idx = try await makeOptimizedIVF(n: 800, dim: 24, nlist: 16, seed: 4242)
        let queries = generateDataset(count: 60, dim: 24, seed: 777)
        let batch = try await idx.batchSearch(queries: queries, k: 5, filter: nil)
        var exact = 0
        for (qi, q) in queries.enumerated() {
            let single = try await idx.search(query: q, k: 5, filter: nil)
            let bIDs = batch[qi].map(\.id), sIDs = single.map(\.id)
            if bIDs == sIDs { exact += 1 }
            else {
                XCTAssertGreaterThanOrEqual(Set(bIDs).intersection(sIDs).count, 4,
                    "query \(qi): batch/single may differ only at FP-margin probe ties")
            }
        }
        XCTAssertGreaterThanOrEqual(exact, 57, "≥95% of queries must match exactly")
    }
}
```

(Helpers `makeOptimizedIVF`/`generateDataset`: copy the fixture patterns from `IVFRecallTests`. Adapt result-element field names to the real `SearchResult` shape.) Run: passes **today** (batch currently calls the same per-query path) — this is a characterization test that must survive the GEMM swap. `swift test --filter IVFBatchGEMMParityTests 2>&1 | tail -3` → PASS.

- [ ] **Step 2: implement `CentroidBatchScore`**

```swift
import Accelerate

internal enum CentroidBatchScore {
    static func run(/* signature above */) -> Bool {
        switch metric {
        case .euclidean, .dotProduct, .cosine: break
        default: return false
        }
        precondition(out.count >= q * kc)
        // out ← −2·Q·Cᵀ  (euclidean) / −1·Q·Cᵀ (dot, cosine pre-scale)
        let alpha: Float = (metric == .euclidean) ? -2 : -1
        out.withUnsafeMutableBufferPointer { ob in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(q), Int32(kc), Int32(d),
                        alpha, queries, Int32(d),
                        centroids, Int32(d),
                        0, ob.baseAddress!, Int32(kc))
            switch metric {
            case .euclidean:
                for qi in 0..<q {
                    let row = ob.baseAddress! + qi * kc
                    for ci in 0..<kc { row[ci] += centroidNormsSq[ci] }
                }
            case .dotProduct:
                break   // −⟨q,c⟩ already
            case .cosine:
                for qi in 0..<q {
                    let qInv: Float = queriesAreNormalized ? 1.0 :
                        1.0 / (IndexOps.Support.Norms.l2NormSquared(
                            vector: queries + qi * d, dimension: d).squareRoot() + 1e-12)
                    let row = ob.baseAddress! + qi * kc
                    // row currently holds −⟨q,c⟩ ⇒ 1 − dot·qInv·cInv = 1 + row·qInv·cInv
                    for ci in 0..<kc { row[ci] = 1 + row[ci] * qInv * centroidInvNorms[ci] }
                }
            default: break
            }
        }
        return true
    }
}
```

- [ ] **Step 3: use it in `batchSearch`**

Before the TaskGroup: flatten `queries` once into a row-major buffer, call `CentroidBatchScore.run`; on `true`, derive each query's sorted probe list (top `min(nprobe, kc)` by score, index-ascending tie-break to match the existing sort's stability — verify what `centroidDists.sort { $0.1 < $1.1 }` does with ties today: Swift's sort is not stable, so ties are already unspecified; use `(score, index)` tuple sort for determinism) and pass the precomputed probe lists into `performIVFSearch` via a new optional context field. On `false` (unsupported metric / kc==0), keep the existing per-query probe path — do not remove it.

- [ ] **Step 4: tests + benchmark gate**

```bash
swift test --filter IVFBatchGEMMParityTests 2>&1 | tail -3
swift test --filter IVFRecallTests 2>&1 | tail -3
swift test --filter IVFTests 2>&1 | tail -3
swift run -c release VectorIndexBenchmarks --index ivf --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --nlist 64 --nprobe 4 --out /tmp/ivf_after_p3b.json
```
Gate: `batchThroughputQps` (Task-1 metric) up vs baseline; single-query `throughputQps` unchanged; parity test green; recall within ±0.01 rule.

- [ ] **Step 5: commit** — `perf(ivf): sgemm cross-term batch centroid probes (P3b)` + measured deltas + trailer.

---

### Task 8: P3c — `optimize()` single materialization + assignment reuse

`optimize(maxIterations:)` (IVFIndex.swift:279-309) flattens the entire store **twice** (`kmeansPlusPlusInitRandom` :391-396 and `kmeans` :335-343 each do `store.map{...}` + `flatMap`) and then re-derives every point's assignment via the O(N·k) scalar `nearestCentroidIndex` rescan — even though `kmeans_minibatch_f32` already ran (with `computeAssignments: false`, discarding the assignments it computed). The in-code P3 note (:286-297) documents exactly this. Carried forward from Phase 2 / spec B18.

**Files:**
- Modify: `Sources/VectorIndex/IVFIndex.swift` (`optimize`, `kmeansPlusPlusInitRandom`, `kmeans`)
- Test: `Tests/VectorIndexTests/IVFKMeansPlusPlusTests.swift` (extend with an assignment-parity assertion)

**Interfaces:**
- Consumes: `kmeans_minibatch_f32` (`KMeansMiniBatch.swift` — read its `computeAssignments`/`assignOut` semantics FIRST, see Step 1), `CentroidBatchScore` (Task 7) for the fallback assignment pass.
- Produces: no signature changes; `kmeansPlusPlusInitRandom`/`kmeans` become `(flatData:count:...)`-taking private helpers.

- [ ] **Step 1: establish the assignment-semantics fact (decides the design)**

Read `kmeans_minibatch_f32` in `KMeansMiniBatch.swift`: when `computeAssignments: true` and `assignOut` is wired, are the emitted assignments computed against the **final** centroids (a dedicated final pass) or against the last iteration's pre-update centroids? Record the answer with a line reference in your report.
- **Final-pass semantics** → use `assignOut` directly for list building (behavior-identical to the rescan up to FP).
- **Stale-assignment semantics** → keep a final assignment pass, but do it in ONE batched call: `CentroidBatchScore.run(queries: flatData, q: n, centroids: finalCentroidsFlat, ...)` + per-row argmin (index-ascending tie-break, matching `nearestCentroidIndex`'s first-min-wins). Either way the O(N·k) *scalar per-pair* rescan and the second store materialization die.

- [ ] **Step 2: single materialization**

Restructure `optimize`:

```swift
public func optimize(maxIterations: Int = 20) async throws {
    guard !store.isEmpty else {
        centroids.removeAll(); lists.removeAll(); rebuildCentroidCache(); return
    }
    let k = max(1, min(config.nlist, store.count))
    let d = dimension
    // ONE store materialization shared by seeding, training, and list build.
    // Dictionary iteration order is stable within an unmutated instance, so
    // capturing ids and vectors in the same pass preserves the exact
    // per-process behavior the old double-materialization had.
    var orderedIDs: [VectorID] = []; orderedIDs.reserveCapacity(store.count)
    var flatData = [Float](); flatData.reserveCapacity(store.count * d)
    for (id, (vec, _)) in store {
        orderedIDs.append(id); flatData.append(contentsOf: vec)
    }
    let n = orderedIDs.count
    let initialCentroids = try kmeansPlusPlusInitRandom(k: k, seed: 42,
                                                        flatData: flatData, count: n)
    var assignments = [Int32](repeating: -1, count: n)
    centroids = try await kmeans(centroids: initialCentroids,
                                 maxIterations: maxIterations,
                                 flatData: flatData, count: n,
                                 assignOut: &assignments)
    rebuildCentroidCache()
    lists = Array(repeating: [], count: centroids.count)
    idToListIndex.removeAll(keepingCapacity: false)
    for (i, id) in orderedIDs.enumerated() {
        let ci = Int(assignments[i])
        guard lists.indices.contains(ci) else { continue }
        lists[ci].append(id)
        idToListIndex[id] = ci
    }
    // (delete the old in-code P3 overlap note comment — it is now done)
}
```

Thread `flatData`/`count` through `kmeansPlusPlusInitRandom` and `kmeans` as parameters (delete their internal `store.map`/`flatMap` lines; keep everything else — especially `seed: 42` and the RNG usage — byte-identical). `kmeans` wires `computeAssignments: true` + `assignOut` per the Step-1 finding (with the batched final pass appended if needed). Preserve the old rescan's edge semantics: points whose assignment is invalid are skipped from lists (old code: `if let ci = nearestCentroidIndex(...), lists.indices.contains(ci)`).

- [ ] **Step 3: extend the tests**

Add to `IVFKMeansPlusPlusTests`:

```swift
func testOptimizeAssignmentsMatchNearestCentroid() async throws {
    let idx = try await makeOptimizedIVF(n: 300, dim: 8, nlist: 8, seed: 99)
    // Every stored vector must sit in the list of its nearest final centroid
    // (ties allowed to either side — compare distances, not indices).
    let check = await idx._testAssignmentConsistency()
    XCTAssertEqual(check.mismatches, 0,
        "post-optimize lists must reflect nearest-final-centroid assignment; \(check.detail)")
}
```

with a small internal `_testAssignmentConsistency()` hook on `IVFIndex` that, for each id, compares the distance to its assigned centroid vs the min distance over all centroids (allow `<= min + 1e-5` to absorb FP) and reports mismatches. This pins the Step-1 semantics decision as an executable fact.

- [ ] **Step 4: run + gate**

```bash
swift test --filter IVFKMeansPlusPlusTests 2>&1 | tail -3
swift test --filter IVFRecallTests 2>&1 | tail -3
swift test --filter IVFTests 2>&1 | tail -3
swift test --filter IVFMoreTests 2>&1 | tail -3
swift run -c release VectorIndexBenchmarks --index ivf --n 5000 --q 200 --dim 384 --k 10 --metric euclidean --nlist 64 --nprobe 4 --out /tmp/ivf_after_p3c.json
```
Gate: `optimizeSeconds` down vs baseline (baseline: 0.268 s at n=5000/k=64 — the rescan is O(N·k·d), a large share); `testOptimizeAssignsAll`/`testOptimizeKMeansPopulatesIdToListIndex` green (assigned == count); recall unchanged within rule.

- [ ] **Step 5: commit** — `perf(ivf): optimize() single materialization + assignment reuse (P3c/B18)` + deltas + trailer.

---

### Task 9: P6a — SIMD `l2Sq` in PQ training

`PQTrain.swift`'s scalar `l2Sq` (:755-762) and its residual-subtracting overload (:774-782) are the hot inner primitive of every k-means path in the file, called O(n·ks) per iteration per subspace. `KMeansSeeding.swift:321-348` already demonstrates the target dual-SIMD4 pattern in the same codebase.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift`
- Test: existing `KMeansMiniBatchTests` + named-method PQTrain subset (full `PQTrainTests` runs in Task 16); micro-bench added to `Tests/VectorIndexTests/KMeansKernelBenchmarks.swift`

**Interfaces:** none — both overloads keep exact signatures; callers untouched.

- [ ] **Step 1: add the micro-bench and record the "before"**

Append to `KMeansKernelBenchmarks.swift` (match its existing gating/style):

```swift
func testPerformance_PQTrainL2SqMicro() throws {
    // RUN_BENCHMARKS-gated like the rest of this file.
    // 1M pairs at dsub=32: representative PQ-subspace shape.
    let dsub = 32, pairs = 1_000_000
    var a = [Float](repeating: 0, count: dsub), b = [Float](repeating: 0, count: dsub)
    for i in 0..<dsub { a[i] = Float(i) * 0.5; b[i] = Float(i) * 0.25 + 1 }
    var sink: Float = 0
    let t0 = DispatchTime.now()
    a.withUnsafeBufferPointer { ap in b.withUnsafeBufferPointer { bp in
        for _ in 0..<pairs { sink += pqTrainL2SqForBench(ap.baseAddress!, bp.baseAddress!, dsub) }
    }}
    let sec = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
    print("l2Sq micro: \(pairs) pairs dsub=\(dsub) in \(sec)s (sink \(sink))")
}
```

This needs `l2Sq` visible to tests: add a thin `@usableFromInline internal func pqTrainL2SqForBench(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float { l2Sq(a, b, len) }` next to `l2Sq` in PQTrain.swift. Run `RUN_BENCHMARKS=1 swift test -c release --filter KMeansKernelBenchmarks/testPerformance_PQTrainL2SqMicro` and record the seconds in the ledger.

- [ ] **Step 2: implement both SIMD overloads**

```swift
@inline(__always) private func l2Sq(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float {
    var acc0 = SIMD4<Float>.zero, acc1 = SIMD4<Float>.zero
    let l8 = len & ~7
    var i = 0
    while i < l8 {
        let a0 = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
        let b0 = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
        let d0 = a0 - b0; acc0 += d0 * d0
        let a1 = SIMD4<Float>(a[i+4], a[i+5], a[i+6], a[i+7])
        let b1 = SIMD4<Float>(b[i+4], b[i+5], b[i+6], b[i+7])
        let d1 = a1 - b1; acc1 += d1 * d1
        i += 8
    }
    var acc = (acc0 + acc1).sum()
    while i < len { let d = a[i] - b[i]; acc += d * d; i += 1 }
    return acc
}

@inline(__always)
private func l2Sq(_ x: UnsafePointer<Float>, _ c: UnsafePointer<Float>, _ dsub: Int, subtract g: UnsafePointer<Float>) -> Float {
    var acc0 = SIMD4<Float>.zero, acc1 = SIMD4<Float>.zero
    let l8 = dsub & ~7
    var i = 0
    while i < l8 {
        let r0 = (SIMD4<Float>(x[i], x[i+1], x[i+2], x[i+3])
                  - SIMD4<Float>(g[i], g[i+1], g[i+2], g[i+3]))
                 - SIMD4<Float>(c[i], c[i+1], c[i+2], c[i+3])
        acc0 += r0 * r0
        let r1 = (SIMD4<Float>(x[i+4], x[i+5], x[i+6], x[i+7])
                  - SIMD4<Float>(g[i+4], g[i+5], g[i+6], g[i+7]))
                 - SIMD4<Float>(c[i+4], c[i+5], c[i+6], c[i+7])
        acc1 += r1 * r1
        i += 8
    }
    var acc = (acc0 + acc1).sum()
    while i < dsub { let r = (x[i] - g[i]) - c[i]; acc += r * r; i += 1 }
    return acc
}
```

Note: this changes FP accumulation order → k-means trajectories can shift slightly. The PQTrain suite asserts distortion *thresholds*, not exact values (verify while running Step 3; if any test asserts exact floats, report it — do not silently retune).

- [ ] **Step 3: run the affected fast suites + a PQTrain subset**

```bash
swift test --filter KMeansMiniBatchTests 2>&1 | tail -3
swift test --filter KMeansPPSeedingTests 2>&1 | tail -3
swift test --filter PQEncodeParity_AoS_C_vs_Swift_Tests 2>&1 | tail -3
swift test --filter PQTrainTests/testStreamingPQTraining 2>&1 | tail -3
swift test --filter PQTrainTests/testBasicTraining 2>&1 | tail -3
```
(Single-method PQTrain filters; pick `testBasicTraining` or whatever the two shortest-looking methods are after reading the file — record which. Timeout 600000 each; if a method alone exceeds it, report rather than background.) Encode-parity is unaffected (l2Sq is Swift-training-side only) but run it anyway as the cheap cross-check.

- [ ] **Step 4: re-run the micro-bench** — same command as Step 1. Gate: ≥1.5× faster (dual-SIMD4 vs scalar at dsub=32 typically lands 2–4×). Record before/after in the ledger.

- [ ] **Step 5: commit** — `perf(pq): SIMD l2Sq in PQ training (P6a)` + micro numbers + note that full PQTrainTests defers to the phase gate + trailer.

---

### Task 10: P6b — streaming k-means++ seeder: coverage, O(ks²)→O(ks), stability ticket

`streamingKMeansppSeed` (`PQTrain.swift:1416-1482`) recomputes each point's min-distance over ALL chosen centroids from scratch, twice per new centroid — O(n·ks²) where its two sibling seeders in the same file maintain a running `dmin[i]` at O(n·ks). It has **zero test coverage** (the only streaming test's n exceeds the `totalN <= 4*ks` trigger cap, so it always takes the sampled path). The same region owns the carried-forward stability ticket: `minibatchKMeansSubspaceChunk` (:1335-1412) once produced distortion=0.0 / 3.2e23 on unseeded data, and its streaming update line (:1405) lacks the `v.isFinite ? v : 0` guard its non-streaming sibling has (:1227). Root cause was never diagnosed; only the test was made deterministic.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`streamingKMeansppSeed` only, unless Step 5 reproduces)
- Test: `Tests/VectorIndexTests/PQTrainTests.swift` (extend — the new tests must be fast: tiny n/ks)

**Interfaces:** none — private function, same signature.

- [ ] **Step 1: coverage-first characterization test**

Add to `PQTrainTests` (fast — small n; make sure it is NOT inside any slow-marked region and respects the suite's `CI_SKIP_PQTRAIN` gate structure):

```swift
func testStreamingSeederSmallNTakesStreamingBranch() throws {
    // totalN <= 4*ks forces streamingKMeansppSeed (PQTrain.swift seedingCap).
    let ks = 16, d = 16, m = 2, n = 40    // 40 <= 64 = 4*16
    // Build 2 chunks of seeded data; call pq_train_streaming_f32 with the
    // same config shape testStreamingPQTraining uses (copy its setup, shrink
    // the sizes, keep its seeded LCG fill).
    // Assertions:
    //  - status == OK
    //  - every output centroid component is finite
    //  - distortion (statsOut) is finite and < the trivial all-zero-centroid
    //    distortion computed inline over the same data
    //  - SNAPSHOT: record centroids[0..<4] values observed on the current
    //    (pre-fix) code into exact XCTAssertEqual constants — the O(ks) fix
    //    is FP-identical (same distances, same min-reduction), so these
    //    constants must survive Step 3 unchanged. That IS the parity gate.
}
```

Run once to confirm the streaming branch is hit (temporarily add a `print` or set a breakpoint assertion inside `streamingKMeansppSeed`; remove before commit) and to harvest the snapshot constants. `swift test --filter PQTrainTests/testStreamingSeederSmallNTakesStreamingBranch 2>&1 | tail -3` → PASS with snapshot pinned.

- [ ] **Step 2: implement the O(ks) fix**

Restructure the `for k in 1..<ks` body to maintain `dmin` (one `Float` per point across all chunks) updated against only the newest centroid — mirroring `kmeansppSeedSubspaceDense` (:900-949) in the same file:

```swift
// dminChunks[c][i]: running min distance² of point i (chunk c) to any chosen
// centroid. Seed against centroid 0, then per new centroid k: one l2Sq per
// point against centroid k−1 only, min-fold into dmin, then the weighted
// pick reads dmin directly. Same floats, same min-reduction → bit-identical
// output to the old full recompute.
```

Both passes (sum pass and weighted-pick pass) read `dmin` instead of re-scanning `0..<k`. Total: O(n·ks) `l2Sq` calls.

- [ ] **Step 3: verify parity** — `swift test --filter PQTrainTests/testStreamingSeederSmallNTakesStreamingBranch 2>&1 | tail -3`: the Step-1 snapshot constants must pass unchanged (bit-identical seeding proves the rewrite is pure). Also `swift test --filter PQTrainTests/testStreamingPQTraining 2>&1 | tail -3` (sampled branch untouched).

- [ ] **Step 4: stability-ticket reproduction attempt (timeboxed)**

Write a throwaway test (do not commit it if it fails to reproduce) feeding `pq_train_streaming_f32` adversarial data through the **minibatch** path: magnitudes around `1e19`, a chunk of all-identical points, and a chunk of zeros, seeded. Check `statsOut` distortion and centroids for NaN/Inf/absurd values across ~5 seeds.
- **Reproduced** → keep the test (named `testStreamingMinibatchSurvivesExtremeData`), apply the minimal guard mirroring :1227 (`let v = oldW*oldVal + newW*batchMean; C[baseC+u] = v.isFinite ? Float(v) : 0` — match the sibling's exact idiom), verify red→green.
- **Not reproduced** → revert the throwaway, change no production code, record in the ledger: "stability ticket: reproduction attempted (magnitudes 1e19/identical/zero chunks, 5 seeds) — not reproduced; ticket remains open." This is the spec's reclassification rule working as intended.

- [ ] **Step 5: commit** — `perf(pq): O(ks) streaming k-means++ seeder with first coverage (P6b)`; report states the stability-ticket outcome explicitly; trailer.

---

### Task 11: P6c — allocation hoists (ScoreBlock, RangeQuery, ExactRerank, JournalFilter)

Four live per-call/per-row allocation sites, two of which have zero test coverage today (tests first). One commit per site is fine; the task is one review unit.

**Files:**
- Modify: `Sources/VectorIndex/Operations/Scoring/ScoreBlock.swift` (:53-64 default branch)
- Modify: `Sources/VectorIndex/Operations/RangeQuery/RangeQuery.swift` (:705 area, `rangeScanL2_earlyExit`)
- Modify: `Sources/VectorIndex/Operations/Rerank/ExactRerank.swift` (:246-274 `scoreBlock` + batch entry)
- Modify: `Sources/VectorIndex/Filters/JournalFilter.swift` (:86-100)
- Test: `Tests/VectorIndexTests/ScoreBlockTests.swift` (new), extend `TelemetryRecorderTests` or new `RangeQueryEarlyExitTests.swift`, existing rerank + journal suites

**Interfaces:** `ExactRerank` gains an internal `_impl` variant threading scratch (public signatures unchanged); everything else is body-only.

- [ ] **Step 1 (ScoreBlock, test first):** create `ScoreBlockTests.swift` — first direct coverage of the kernel: for each of euclidean/dotProduct/cosine/manhattan, score a seeded 32×16 block via `ScoreBlock.run` and compare per-row against scalar reference math computed in the test (`accuracy: 1e-4`; manhattan exercises the `default:` fallback). Run → PASS (characterization). Then hoist in the `default:` branch: `qArr` (`Array(UnsafeBufferPointer(...))`) and `tmp` move above the `while` loop, `tmp` refilled per row. Re-run → PASS.
- [ ] **Step 2 (RangeQuery, test first):** new test forcing the early-exit path: `RangeScanConfig(earlyExit: .on)`, L2, seeded 500×16 data, threshold chosen to accept ~10%; oracle = the same call with `earlyExit: .off` (generic path) — assert identical ids+scores sets. Run → PASS (characterization of a previously-uncovered path; if it FAILS, you found a real early-exit bug: STOP and report, do not fix perf on top of a broken path). Then hoist `part` (`UnsafeMutableBufferPointer.allocate(capacity: R)`) above the `while blockStart < n` loop with a single `defer { part.deallocate() }`, zero-fill per block. Re-run → PASS.
- [ ] **Step 3 (ExactRerank):** extract the body of `rerank_exact_topk` into `internal func _rerank_exact_topk_impl(..., scratch: UnsafeMutablePointer<Float>?, present: UnsafeMutablePointer<UInt8>?, tileScores: UnsafeMutablePointer<Float>?)` where nil ⇒ allocate-locally (current behavior). Public `rerank_exact_topk` passes nil. `rerank_exact_topk_batch`'s `for qi in 0..<b` loop allocates the three buffers ONCE (sized `tile*d`/`tile`/`tile`) and passes them through. Verify: `swift test --filter RegressionA4_RerankIDWidthTests`, `--filter RegressionB17a_ParallelRerankTests`, `--filter IVFListVecsReaderRerankTests` (three invocations) all green.
- [ ] **Step 4 (JournalFilter):** replace the two per-invocation `ISO8601DateFormatter()` constructions with thread-local cached instances (preserves the documented concurrency rationale — the closure is `@Sendable` and may run concurrently):

```swift
private func cachedISO8601Formatter(fractional: Bool) -> ISO8601DateFormatter {
    let key = fractional ? "vindex.journal.iso8601.frac" : "vindex.journal.iso8601.plain"
    let dict = Thread.current.threadDictionary
    if let f = dict[key] as? ISO8601DateFormatter { return f }
    let f = ISO8601DateFormatter()
    if fractional { f.formatOptions = [.withInternetDateTime, .withFractionalSeconds] }
    dict[key] = f
    return f
}
```

The closure body becomes `let date = cachedISO8601Formatter(fractional: true).date(from: dateStr) ?? cachedISO8601Formatter(fractional: false).date(from: dateStr)`. Verify: `swift test --filter JournalFilterTests` and `--filter JournalFilterAdvancedTests` green.
- [ ] **Step 5: commit(s)** — one commit per site or one combined `perf: hoist per-row/per-call allocations (P6c: ScoreBlock, RangeQuery, ExactRerank, JournalFilter)`; report lists the two new coverage suites; trailer.

---

### Task 12: P7 — remove the GNU statement-expression in `pq_encode.c`

`encode_subspace_u8_residual_with_csq` (`Sources/CPQEncode/pq_encode.c:396-399`) computes the k=0 best-distance via a nonstandard `({ ... })` statement-expression duplicating the loop body's expression form. Bit-exactness matters: the value feeds an argmin tie-break, and `PQEncodeParity_AoS_C_vs_Swift_Tests.testResidualU8_AoS_C_vs_Swift_WithCSQ` gates C-vs-Swift equality.

**Files:**
- Modify: `Sources/CPQEncode/pq_encode.c:394-399`
- Test: existing parity suite

- [ ] **Step 1: replace with plain locals (identical operation order):**

```c
int   best_k = 0;
float dot_r_c0 = dot_only(x_sub, cb_j, dsub) - dot_only(coarse_sub, cb_j, dsub);
float best_d = r2 + csq_j[0] - 2.0f * dot_r_c0;
```

- [ ] **Step 2: verify** — `swift build 2>&1 | tail -3` (clean, no new warnings) and `swift test --filter PQEncodeParity_AoS_C_vs_Swift_Tests 2>&1 | tail -3` (both parity tests green — bit-exact).
- [ ] **Step 3: commit** — `cleanup(cpq): plain locals replace GNU statement-expression (P7)` + trailer.

---

### Task 13: F16 inv-norms — delegate the 6th sum-of-squares copy (benchmark-gated decision)

`Cosine.precomputeInvNormsF16` (`Operations/Scoring/Cosine.swift:430-446`) hand-rolls a 4-wide SIMD sum-of-squares instead of delegating to `IndexOps.Support.Norms.l2NormSquared` (16-wide) like every other call site in the file. Phase 2 deferred this because delegation changes the FP reduction tree on a numerically sensitive path (values round-trip through `Float16`). The carried-forward decision rule: delegate **iff** the existing parity gate holds.

**Files:**
- Modify: `Sources/VectorIndex/Operations/Scoring/Cosine.swift:430-446`
- Test: `Tests/VectorIndexTests/CosineKernelTests.swift` (`testCosineF16NormsParity`, 1e-3 tolerance — the pre-existing gate)

- [ ] **Step 1: run the gate before changing anything** — `swift test --filter CosineKernelTests 2>&1 | tail -3` → green (anchor).
- [ ] **Step 2: delegate:**

```swift
public static func precomputeInvNormsF16(
    xb: UnsafePointer<Float>, n: Int, d: Int, out: UnsafeMutablePointer<Float16>, epsilon: Float = 1e-12
) {
    if n == 0 { return }
    if d == 0 { for i in 0..<n { out[i] = Float16(1.0 / epsilon) }; return }
    let limit = Float(Float16.greatestFiniteMagnitude)
    for i in 0..<n {
        let s = IndexOps.Support.Norms.l2NormSquared(vector: xb.advanced(by: i * d), dimension: d)
        let inv = 1.0 / (s.squareRoot() + epsilon)
        out[i] = Float16(max(-limit, min(limit, inv)))
    }
}
```

(Preserves the d==0 and clamp semantics exactly; only the accumulation tree changes — expected delta ~1e-7 relative, three orders under the 1e-3 gate.)
- [ ] **Step 3: decide by measurement** — `swift test --filter CosineKernelTests 2>&1 | tail -3`. Green → keep (record "delegated; parity gate green"). Red → `git checkout -- Sources/VectorIndex/Operations/Scoring/Cosine.swift`, record "delegation exceeds F16 parity tolerance; duplicate retained deliberately" in the ledger, and add a code comment on the function saying exactly that. Either outcome completes the task.
- [ ] **Step 4: broader cosine surface** — `swift test --filter CosineFusedCacheIntegrationTests 2>&1 | tail -3` (and any other Cosine* suite found via `ls Tests/VectorIndexTests | grep -i cosine`).
- [ ] **Step 5: commit** — `cleanup(cosine): precomputeInvNormsF16 delegates to shared l2NormSquared` (or the keep-duplicate documentation commit) + trailer.

---

### Task 14: Revive `HNSWAlignmentTest` + `compact()` re-prune reachability verdict

The whole class is skipped by an unconditional `XCTSkip` in `setUpWithError` ("candidates API changed") — collateral damage that removed the only test covering `compact()` with a nontrivial deletion pattern. Separately, analysis suggests the re-prune branch (`HNSWIndex.swift:1151`, `mapped.count > config.m`) may be unreachable: `pruneNeighbors` bounds every list to ≤ m during construction, and remapping only shrinks lists. This task revives the class and settles reachability with evidence.

**Files:**
- Modify: `Tests/VectorIndexTests/HNSWAlignmentTest.swift`
- Possibly modify: `Sources/VectorIndex/HNSWIndex.swift` (comment only, if the branch proves defensive-only)

- [ ] **Step 1: remove the blanket skip** — delete the `setUpWithError` override; run `swift test --filter HNSWAlignmentTest 2>&1 | tail -10` and catalogue the compile/assert failures (expected: `getCandidates`/`AccelerationCandidates` API drift).
- [ ] **Step 2: update the three tests to the current candidates API** — mirror the current call shape from working call sites (`grep -rn "getCandidates" Tests/ Sources/ | grep -v AlignmentTest` for live patterns). Keep each test's original intent (structure-with-deletions, candidates alignment, post-compaction consistency); modernize mechanics only.
- [ ] **Step 3: reachability verdict for the re-prune branch** — attempt to construct, via public API only, an index where some surviving node's remapped neighbor list exceeds `config.m` at `compact()` time (try: tiny m (2–4), interleaved insert/delete/re-insert, `update()` calls). Instrument temporarily (local `print` in the branch) to observe firing.
  - **Fires** → extend `testStructureConsistencyAfterCompaction` with that fixture so the branch has live coverage; remove the instrumentation.
  - **Cannot fire** → add a comment on the branch: `// Defensive: pruneNeighbors bounds lists to ≤ m during construction, so remap cannot exceed m; kept as a guard for future insert-path changes. No public-API fixture reaches this (verified 2026-07-31, Phase 3 Task 14).` and record the attempted fixtures in the ledger.
- [ ] **Step 4: run** — `swift test --filter HNSWAlignmentTest 2>&1 | tail -3` (green, 3 un-skipped tests) plus `swift test --filter PersistenceEdgeTests 2>&1 | tail -3`.
- [ ] **Step 5: commit** — `test(hnsw): revive HNSWAlignmentTest (compact coverage); re-prune branch reachability verdict` + trailer.

---

### Task 15: Reservoir telemetry completeness + adaptive-vs-upfront mode benchmark

Carried from the `.adaptive` enablement (Task 19, Phase 2): benchmark `.adaptive` against upfront `.heap`/`.block` so users of the now-working mode have guidance. Blocker found in research: `ReservoirTelemetry.accepted` and `.prunes` are declared but never incremented — fix that first so the benchmark can report them.

**Files:**
- Modify: `Sources/VectorIndex/Operations/Reservoir/CandidateReservoir.swift`
- Create: `Tests/VectorIndexTests/ReservoirModeBenchmarks.swift`
- Test: extend `Tests/VectorIndexTests/CandidateReservoirTests.swift`

- [ ] **Step 1: failing telemetry test** — extend `CandidateReservoirTests`: after a pushBatch stream into a small `.block` reservoir with `telemetry: true` that forces ≥1 prune, assert `telemetry.prunes >= 1` and `telemetry.accepted == pushed - rejectedTau - rejectedDedup - rejectedInvalid`. Run → FAIL (fields never written).
- [ ] **Step 2: wire the two fields** — in `pushBatch`, increment `telemetry.accepted` where an element is actually admitted (locate the accept paths in both heap and block branches — the function's accepted-count bookkeeping already exists for the return value; mirror it); in `pruneToTopC()` (:399), `telemetry.prunes &+= 1`. Run → PASS. Also re-run the two adaptive tests: `swift test --filter CandidateReservoirTests 2>&1 | tail -3`.
- [ ] **Step 3: the mode benchmark** — new `ReservoirModeBenchmarks.swift`, `RUN_BENCHMARKS=1`-gated (copy the `IVFSelectBenchmarks` gate): for C ∈ {64, 1024}, stream shapes {descending (block-worst), ascending (block-best), random seeded} × modes {.heap, .block, .adaptive}: push 100k scored ids in 1k batches, measure wall time, print a table row per cell with ns/op, `modeSwitches`, `prunes`, `rejectedTau`. No assertions — informational.
- [ ] **Step 4: run it once, capture the table** — `RUN_BENCHMARKS=1 swift test -c release --filter ReservoirModeBenchmarks 2>&1 | tail -30`; paste the table into the ledger.
- [ ] **Step 5: document** — extend the doc comment on `ReservoirOptions.mode` with 3–4 lines of measured guidance (e.g. "descending-score streams: .heap ≈ …× faster than .block; .adaptive tracks the better of the two within …% — measured 2026-07-31, see ReservoirModeBenchmarks"). Use the actual numbers from Step 4.
- [ ] **Step 6: commit** — `feat(reservoir): complete telemetry (accepted/prunes) + mode benchmark & guidance` + trailer.

---

### Task 16: Phase close-out — full benchmark sweep, CHANGELOG, full-suite gate, consumer build

**Files:**
- Create: `.bench/post-phase3/` (7 JSONs + README with host info)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: full benchmark sweep (quiet machine, probe first)** — re-run all seven Task-2 capture commands with output into `.bench/post-phase3/`. Same machine as the baseline (verify the `host` blocks match).
- [ ] **Step 2: CHANGELOG** — under `## [Unreleased] — 0.2.0`, add a `### Performance` subsection: one bullet per item (P1, P3a/b/c, P4, P5, P6a/b/c, P7, F16 outcome) with the measured before→after numbers from the sweep and ledger (build seconds, optimizeSeconds, mmap commits/sec + slope ratio, batch QPS, micro-bench ratios) and the recall deltas with their one-line explanations. Also note under `### Fixed` if the stability ticket was reproduced+fixed in Task 10 (otherwise it stays out of the CHANGELOG — open tickets aren't release notes).
- [ ] **Step 3: full test-suite gate** — run every suite green. Composed single-suite runs (the `--filter` alternation bug forbids big alternations); `PQTrainTests` runs HERE, once (~60 min — coordinate with the controller: this command exceeds the 600 s Bash cap and will be auto-backgrounded; the controller collects the task notification, the implementer does NOT wait on it mid-turn). Skips must match the known set (38 skipped in 6 suites as of Phase 2, plus/minus the suites this phase un-skipped: `HNSWAlignmentTest` now runs — expected skip count decreases; document the new expected number).
- [ ] **Step 4: consumer residual gate** — build `VectorIndexAccelerated` against the branch: `swift build --package-path ../future/VectorIndexAccelerated` (adjust the path after `ls ../future/`; use an explicit `--package-path`, do NOT `cd`). Green required. It must pick up the local VectorIndex — check how its Package.swift references VectorIndex (path dependency vs remote); if remote-pinned, note that the check validates API-compat only after a temporary path override, then revert the override.
- [ ] **Step 5: determinism + recall final check** — `swift test --filter HNSWDeterminismTests` green; recall deltas across the sweep all within the ±0.01 rule with ledger explanations.
- [ ] **Step 6: commit** — `bench: post-Phase-3 sweep + CHANGELOG performance deltas` + trailer. Then hand off to the final whole-branch review per the SDD skill.

---

## Self-Review (performed at plan-writing time)

1. **Spec coverage:** P1→Task 5; P2→Task 5 (closed-by-A9, documented); P3→Tasks 6/7/8; P4→Task 3; P5→Task 4; P6→Tasks 9/10/11 (+2 recorded no-ops: IDMap-erase deferred, InnerProduct already-hoisted); P7→Task 12. Carried-forward: F16→13, compact coverage→14, reservoir benchmark→15, stability ticket→10, IVF optimize→8. Gates/baseline→1/2/16. Spec's Phase-3 gate rule (recall/determinism/CHANGELOG deltas) → Global Constraints + Task 16.
2. **Placeholder scan:** no TBDs; the two "read X first" steps (kmeans assignOut semantics, WAL close behavior) are explicit fact-establishment steps with decision rules for each outcome, not deferred design.
3. **Type consistency:** `HNSWGraphSnapshot`/`_testGraphSnapshot` (T2→T5), `centroidsFlat`/`centroidNormsSq`/`centroidInvNorms`/`centroidDistances(for:queryIsNormalized:)` (T6→T7/T8), `CentroidBatchScore.run` (T7→T8), `crcBytesHashed`/`flush()` (T3→T16), `msyncCallCount`/`msyncBytesFlushed` (T4), `batchThroughputQps`/`MMAP_BENCH_OUT` (T1→T2/T3/T4/T16) — names match across tasks.
