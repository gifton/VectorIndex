# VectorIndex 0.2.0 — Plan 1: Phase 0 (Gates) + Phase 1 (Correctness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the 8 correctness fixes (A1–A8) from the 0.2.0 cleanup spec on a fresh branch, each guarded by a test, after establishing the Phase-0 baseline gates.

**Architecture:** One branch `gifton/cleanup-0.2.0`. Phase 0 captures a perf/consumer baseline so later phases can prove no regression. Phase 1 fixes each correctness bug in isolation, TDD where a red test is achievable, with an explicit airtightness fallback where a bug is real-by-inspection but not deterministically reproducible (UB / latent-on-disk / leak).

**Tech Stack:** Swift 6, SwiftPM, XCTest (`@testable import VectorIndex`), Accelerate. Tests run via `swift test --filter <Name>`. Actors (`HNSWIndex`), unsafe-pointer kernels, mmap persistence, C interop (`CPQEncode`).

**Spec:** `docs/superpowers/specs/2026-06-22-vectorindex-0.2.0-cleanup-design.md` (§5 Phase 0, §6 Phase 1). This plan covers Phases 0–1 only; Plans 2–5 (cleanup / perf / breaking removals / release) are generated when reached.

**Airtightness rule (spec §4):** every correctness fix starts with a reproducing test. Where UB/latent/leak defects cannot be deterministically reproduced, the task says so explicitly, provides the strongest feasible guard, and applies the fix as documented hardening rather than claiming a red→green cycle.

---

## File map

**Source files modified (fixes):**
- `Sources/VectorIndex/Kernels/IVFAppend.swift` — A2 (`getListStats` durable path)
- `Sources/VectorIndex/HNSWIndex.swift` — A5 (`batchRemove`), A1 (search + batch traversal pointer lifetime)
- `Sources/VectorIndex/HNSWKNNGraph.swift` — A1 (`buildKNNRows` pointer lifetime)
- `Sources/VectorIndex/Kernels/VIndexMmap.swift` — A3 (remap TOC offsets)
- `Sources/VectorIndex/Kernels/VIndexContainerBuilder.swift` — A7 (`tocSize` stride)
- `Sources/VectorIndex/Operations/Rerank/ExactRerank.swift` — A4 (Int64 ID preservation)
- `Sources/VectorIndex/Operations/Quantization/PQEncode.swift` — A6 (leak)
- `Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift` — A8 (touched-word overflow)

**Test files created (one per bug, self-contained):**
- `Tests/VectorIndexTests/RegressionA2_DurableListStatsTests.swift`
- `Tests/VectorIndexTests/RegressionA5_BatchRemoveTests.swift`
- `Tests/VectorIndexTests/RegressionA3_RemapTOCTests.swift`
- `Tests/VectorIndexTests/RegressionA4_RerankIDWidthTests.swift`
- `Tests/VectorIndexTests/RegressionA8_DedupOverflowTests.swift`
- `Tests/VectorIndexTests/RegressionA1_TraversalLifetimeTests.swift`
- (A6, A7: no new behavioral test — see their tasks)

---

## Task 0: Phase 0 — Branch, baseline, CHANGELOG, consumer check

**Files:**
- Create: `CHANGELOG.md`, `.bench/baseline-0.1.6/` (JSON outputs)

- [ ] **Step 1: Create the branch off v0.1.6**

Run:
```bash
cd /Users/goftin/dev/gsuite/VSK/VectorIndex
git checkout main && git pull --ff-only
git checkout -b gifton/cleanup-0.2.0
```
Expected: `Switched to a new branch 'gifton/cleanup-0.2.0'`.

- [ ] **Step 2: Confirm a clean green baseline**

Run: `swift build -c release 2>&1 | tail -5`
Expected: `Build complete!`

- [ ] **Step 3: Capture the perf baseline (Phase-3 gate input)**

Run:
```bash
mkdir -p .bench/baseline-0.1.6
BIN=.build/release/VectorIndexBenchmarks
$BIN --knn-graph --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42 > .bench/baseline-0.1.6/knn_graph_uniform.json
$BIN --knn-graph --knn-clusters 50 --n 3000 --dim 384 --k 15 --m 16 --efc 200 --efs 96 --seed 42 > .bench/baseline-0.1.6/knn_graph_clusters.json
$BIN --index hnsw --n 5000 --q 200 --dim 384 --k 10 --seed 42 > .bench/baseline-0.1.6/hnsw_search.json
$BIN --index ivf --n 5000 --q 200 --dim 384 --k 10 --seed 42 > .bench/baseline-0.1.6/ivf_search.json
$BIN --index flat --n 5000 --q 200 --dim 384 --k 10 --seed 42 > .bench/baseline-0.1.6/flat_search.json
```
Expected: 5 JSON files, each a single-line result object. If an `--index` value errors, run `$BIN` with no args to print usage and adjust; record any benchmark that cannot run in the commit message.

- [ ] **Step 4: Create CHANGELOG.md**

Create `CHANGELOG.md`:
```markdown
# Changelog

All notable changes to VectorIndex are documented here. Versions follow the 0.x
convention: the minor digit signals breaking changes.

## [Unreleased] — 0.2.0

### Fixed
<!-- correctness fixes appended per task -->

### Changed
<!-- cleanup / perf appended per task -->

### Removed
<!-- breaking removals appended per task -->
```

- [ ] **Step 5: Confirm the downstream consumer builds against the branch (residual gate)**

Run:
```bash
cd /Users/goftin/dev/gsuite/VSK/future/VectorIndexAccelerated
swift build 2>&1 | tail -15
```
Expected: record the result. If it already fails to build on `main`'s VectorIndex (the consumer is known mid-rework), note the pre-existing failure now so later phases can distinguish new breakage from old. Do NOT block on a pre-existing failure; the gate is "we did not make it worse."

- [ ] **Step 6: Commit Phase-0 setup**

```bash
cd /Users/goftin/dev/gsuite/VSK/VectorIndex
git add CHANGELOG.md .bench/baseline-0.1.6 docs/superpowers
git commit -m "$(cat <<'EOF'
chore(0.2.0): Phase 0 — branch, perf baseline, CHANGELOG

Baseline JSON under .bench/baseline-0.1.6 is the Phase-3 no-regression gate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1 (spec A2): durable `getListStats` returns real stats instead of throwing

`IVFListHandle.getListStats(listID:durable:)` calls `mmap.mmapLists()`, which is a legacy
shim that unconditionally returns `nil`, so the durable path always throws `.contractViolation`.
Fix: read the descriptor via `mmap.getListDescriptor(listID:)` (the accessor `readList`
already uses).

**Files:**
- Test: `Tests/VectorIndexTests/RegressionA2_DurableListStatsTests.swift` (create)
- Modify: `Sources/VectorIndex/Kernels/IVFAppend.swift:345-355`

- [ ] **Step 1: Write the failing test**

Create `Tests/VectorIndexTests/RegressionA2_DurableListStatsTests.swift`:
```swift
import XCTest
@testable import VectorIndex

final class RegressionA2_DurableListStatsTests: XCTestCase {
    func testDurableGetListStatsReturnsCapacity() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vindex_a2_\(UUID().uuidString).vindex").path
        let k_c = 1, m = 8
        let mmap = try VIndexContainerBuilder.createMinimalContainer(
            path: tmp, format: .pq8, k_c: k_c, m: m, d: 0,
            idBits: 64, group: 4, idCap: 32, payloadCap: 16)
        defer { try? mmap.close(); _ = try? FileManager.default.removeItem(atPath: tmp) }

        var opts = IVFAppendOpts.default
        opts.format = .pq8
        opts.durable = true
        let h = try ivf_create_mmap(k_c: k_c, m: m, d: 0, mmap: mmap, opts: opts)

        let n = 5
        let listIDs = [Int32](repeating: 0, count: n)
        let extIDs = (0..<n).map { UInt64($0 + 100) }
        var codes = [UInt8](repeating: 0, count: n * m)
        for i in 0..<n { for j in 0..<m { codes[i*m + j] = UInt8(1 + j) } }
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: codes,
                       n: n, m: m, index: h, opts: opts, internalIDsOut: nil)

        // Before the fix this throws .contractViolation ("mmap list descriptors unavailable").
        let stats = try h.getListStats(listID: 0, durable: true)
        XCTAssertEqual(stats.length, n)
        XCTAssertGreaterThanOrEqual(stats.capacity, n)
        XCTAssertGreaterThan(stats.bytesIDs, 0)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RegressionA2_DurableListStatsTests 2>&1 | tail -20`
Expected: FAIL — throws `VectorIndexError` with message "mmap list descriptors unavailable (internal error)".

- [ ] **Step 3: Apply the fix**

In `Sources/VectorIndex/Kernels/IVFAppend.swift`, replace lines 345-355 (the `guard let (descs, _) = mmap.mmapLists()` block through `out.capacity = ...`):
```swift
            guard let desc = mmap.getListDescriptor(listID: Int(listID)) else {
                throw ErrorBuilder(.contractViolation, operation: "get_list_stats_durable")
                    .message("mmap list descriptors unavailable (internal error)")
                    .build()
            }
            let len = mmap.snapshotListLength(listID: Int(listID))
            var out = IVFListStats()
            out.length = len
            out.capacity = desc.capacity
```
(This drops the now-unused `let i = Int(listID)` / `let dsc = descs[i]` lines; the remaining `out.bytesIDs`/`switch format` block below is unchanged and already uses `out.capacity`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter RegressionA2_DurableListStatsTests 2>&1 | tail -20`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorIndex/Kernels/IVFAppend.swift Tests/VectorIndexTests/RegressionA2_DurableListStatsTests.swift
git commit -m "$(cat <<'EOF'
fix(ivf): durable getListStats reads descriptor instead of dead mmapLists()

mmapLists() is a legacy shim that always returns nil, so the durable stats
path always threw .contractViolation. Read via getListDescriptor(listID:).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 (spec A5): `batchRemove` no longer corrupts the index on partial removal

`batchRemove` calls `remove()` per id (which correctly maintains `entryPoint`/`activeCount`),
then unconditionally zeroes `entryPoint`/`maxLevel`/`activeCount` — corrupting the index when
only a subset is removed. Fix: delete those three reset lines; keep the cache-dirty marks.

**Files:**
- Test: `Tests/VectorIndexTests/RegressionA5_BatchRemoveTests.swift` (create)
- Modify: `Sources/VectorIndex/HNSWIndex.swift:481-487`

- [ ] **Step 1: Write the failing test**

Create `Tests/VectorIndexTests/RegressionA5_BatchRemoveTests.swift`:
```swift
import XCTest
@testable import VectorIndex

final class RegressionA5_BatchRemoveTests: XCTestCase {
    func testBatchRemoveSubsetKeepsIndexSearchable() async throws {
        let idx = HNSWIndex(dimension: 2)
        try await idx.insert(id: "a", vector: [0, 0], metadata: nil)
        try await idx.insert(id: "b", vector: [1, 0], metadata: nil)
        try await idx.insert(id: "c", vector: [0, 1], metadata: nil)
        try await idx.insert(id: "d", vector: [1, 1], metadata: nil)

        // Remove a subset only.
        try await idx.batchRemove(["b", "d"])

        // Survivors must still be findable (pre-fix: entryPoint=nil => empty results).
        let res = try await idx.search(query: [0, 0], k: 2, filter: nil)
        let ids = Set(res.map { $0.id })
        XCTAssertTrue(ids.contains("a"), "surviving point 'a' should be found")
        XCTAssertFalse(ids.contains("b"), "removed point 'b' should not be found")
        XCTAssertEqual(await idx.count, 2, "count should reflect 2 survivors")
    }
}
```
(If `count` is not `async`, drop the `await`; confirm by grepping `var count` / `func count` in `HNSWIndex.swift`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RegressionA5_BatchRemoveTests 2>&1 | tail -20`
Expected: FAIL — search returns empty (entryPoint was nilled), so `ids.contains("a")` is false; and/or `count` is 0.

- [ ] **Step 3: Apply the fix**

In `Sources/VectorIndex/HNSWIndex.swift`, change `batchRemove` (lines 481-487) from:
```swift
    public func batchRemove(_ ids: [VectorID]) async throws {
        for id in ids { try await remove(id: id) }
        entryPoint = nil
        maxLevel = 0
        activeCount = 0
        markCSRDirty(); markInvNormsDirty()
    }
```
to:
```swift
    public func batchRemove(_ ids: [VectorID]) async throws {
        for id in ids { try await remove(id: id) }
        // `remove()` -> `internalRemove()` already maintains entryPoint/activeCount
        // per id; only the CSR/invNorms caches need invalidating for the batch.
        markCSRDirty(); markInvNormsDirty()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter RegressionA5_BatchRemoveTests 2>&1 | tail -20`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorIndex/HNSWIndex.swift Tests/VectorIndexTests/RegressionA5_BatchRemoveTests.swift
git commit -m "$(cat <<'EOF'
fix(hnsw): batchRemove no longer zeroes entryPoint/activeCount

remove() already maintains entryPoint/maxLevel/activeCount per id; the
unconditional reset corrupted the index on subset removal (searches returned
empty). Keep only the CSR/invNorms cache invalidation.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 (spec A4): exact rerank preserves 64-bit candidate IDs

`rerank_exact_topk` narrows `Int64` candidate IDs to `Int32` (`truncatingIfNeeded`) to feed the
top-k heap, then widens back — silently corrupting IDs > 2³¹. Fix: pass `nil` ids to the heap
(it substitutes positional indices), then map the selected positions back through the original
`Int64` `candIDs`.

**Files:**
- Test: `Tests/VectorIndexTests/RegressionA4_RerankIDWidthTests.swift` (create)
- Modify: `Sources/VectorIndex/Operations/Rerank/ExactRerank.swift:672-722`

- [ ] **Step 1: Write the failing test**

Create `Tests/VectorIndexTests/RegressionA4_RerankIDWidthTests.swift`. Uses `CallbackReader` (handles arbitrary `Int64` ids) with two candidates whose IDs exceed `Int32.max`:
```swift
import XCTest
@testable import VectorIndex

final class RegressionA4_RerankIDWidthTests: XCTestCase {
    func testRerankPreservesIDsAboveInt32Max() {
        let d = 2
        // Two candidates with large 64-bit IDs; id A is the exact match for the query.
        let idA: Int64 = (1 << 31) + 10          // > Int32.max
        let idB: Int64 = (1 << 32) + 7           // > UInt32.max
        let vecA: [Float] = [1, 0]
        let vecB: [Float] = [0, 1]

        let reader = IndexOps.Rerank.CallbackReader(dim: d) { ids, count, dst, present in
            var found = 0
            for i in 0..<count {
                let id = ids[i]
                let row: [Float]? = (id == idA) ? vecA : (id == idB ? vecB : nil)
                if let r = row {
                    dst[i*d + 0] = r[0]; dst[i*d + 1] = r[1]
                    present[i] = 1; found += 1
                } else { present[i] = 0 }
            }
            return found
        }

        let q: [Float] = [1, 0]                  // closest to vecA => idA
        let candIDs: [Int64] = [idA, idB]
        var scores = [Float](repeating: 0, count: 1)
        var outIDs = [Int64](repeating: -1, count: 1)

        q.withUnsafeBufferPointer { qb in
            candIDs.withUnsafeBufferPointer { cb in
                scores.withUnsafeMutableBufferPointer { sb in
                    outIDs.withUnsafeMutableBufferPointer { ib in
                        let opts = IndexOps.Rerank.RerankOpts(backend: .callback)
                        IndexOps.Rerank.rerank_exact_topk(
                            q: qb.baseAddress!, d: d, metric: .euclidean,
                            candIDs: cb.baseAddress!, C: candIDs.count, K: 1,
                            reader: reader, opts: opts,
                            topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
                    }
                }
            }
        }
        // Pre-fix: the returned id is Int32(truncatingIfNeeded: idA) widened back => corrupted.
        XCTAssertEqual(outIDs[0], idA, "top-1 id must be the exact 64-bit candidate id")
    }
}
```
Note: confirm the exact `RerankOpts` initializer/`backend` enum case for the callback reader by grepping `RerankOpts(` and `enum RerankBackend`/`backend` in `ExactRerank.swift`; adjust the `opts` line to the real case (e.g. `.callback` vs `.denseArray`). If `skipMissing` defaults matter, set `opts.skipMissing = false`.

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RegressionA4_RerankIDWidthTests 2>&1 | tail -20`
Expected: FAIL — `outIDs[0]` equals the truncated/round-tripped value, not `idA`.

- [ ] **Step 3: Apply the fix**

In `Sources/VectorIndex/Operations/Rerank/ExactRerank.swift`, replace the filtered/unfiltered
Int32 id construction and the heap calls so the heap runs with `nil` ids (positional), and keep
a position→original-Int64 map. Replace lines 672-710 (`let useFiltered ...` through the `}()`)
with:
```swift
        // Select top-K using #05 TopK (deterministic tie-break by position).
        // IMPORTANT: never narrow the Int64 candidate ids; run the heap on positional
        // indices (ids: nil) and map winners back through candIDs to preserve full width.
        let useFiltered = opts.skipMissing
        let ordering = IndexOps.Selection.ordering(for: metric)
        // positions[i] is the original candIDs index for the i-th scored entry fed to the heap.
        var positions: [Int32] = []
        var heapScores: [Float] = []
        if useFiltered {
            positions.reserveCapacity(C)
            heapScores.reserveCapacity(C)
            for i in 0..<C where presentMask[i] != 0 {
                heapScores.append(scores[i])
                positions.append(Int32(i))
            }
        }
        let selHeap: IndexOps.Selection.TopKHeap = {
            if useFiltered {
                guard !heapScores.isEmpty else { return IndexOps.Selection.TopKHeap(capacity: K, ordering: ordering) }
                return IndexOps.Selection.selectTopK_streaming(
                    scores: heapScores, ids: nil, count: heapScores.count, k: K, ordering: ordering)
            } else {
                return IndexOps.Selection.selectTopK_streaming(
                    scores: scores, ids: nil, count: C, k: K, ordering: ordering)
            }
        }()
```
Then change the emit loop (lines 712-718) to map heap ids (which are now positions) back to the
original 64-bit candidate id:
```swift
        let pairs = selHeap.extractSorted()
        let actual = min(K, pairs.count)
        for i in 0..<actual {
            let pos = useFiltered ? Int(positions[Int(pairs[i].id)]) : Int(pairs[i].id)
            topScores[i] = pairs[i].score
            topIDs[i]    = candIDs.advanced(by: pos).pointee
        }
```
Delete the now-unused `filteredScores`/`filteredIDs32`/`ids32All` declarations (old lines 674-687).
The `if actual < K { ... }` padding block and the `#if VINDEX_TELEM` block below stay unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter RegressionA4_RerankIDWidthTests 2>&1 | tail -20`
Expected: PASS. Then run the existing rerank tests to confirm no regression:
`swift test --filter IVFListVecsReaderRerank 2>&1 | tail -20` → PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorIndex/Operations/Rerank/ExactRerank.swift Tests/VectorIndexTests/RegressionA4_RerankIDWidthTests.swift
git commit -m "$(cat <<'EOF'
fix(rerank): preserve 64-bit candidate ids in exact top-k

rerank_exact_topk narrowed Int64 ids to Int32 to feed the heap, corrupting
ids > 2^31. Run the heap on positional indices (ids: nil) and map winners back
through the original Int64 candIDs.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 (spec A3): fix TOC field offsets on the mmap grow/remap path

`ensureFileCapacity`'s TOC re-parse reads fields at offsets 8/16/24/28/32, but the writer and
`indexInit` use the packed layout 4/12/20/24/28. After a file-growth remap, every section's
offset/size is corrupted. **Testability caveat:** the existing `testDurablePQ8AppendWithRemap`
passes today, so section pointers may be rebound from the 64-byte per-list records rather than
`tocByType`. Attempt a reproducing test first; if none can be made to go red, apply the fix as a
documented consistency correction per spec §4 (do NOT claim a red→green cycle).

**Files:**
- Test: `Tests/VectorIndexTests/RegressionA3_RemapTOCTests.swift` (create)
- Modify: `Sources/VectorIndex/Kernels/VIndexMmap.swift:952-956`

- [ ] **Step 1: Attempt a reproducing test**

Create `Tests/VectorIndexTests/RegressionA3_RemapTOCTests.swift`. Force a remap, then reopen the
file fresh (a fresh `indexInit` parse must agree with post-remap section data):
```swift
import XCTest
@testable import VectorIndex

final class RegressionA3_RemapTOCTests: XCTestCase {
    func testRemapThenReopenPreservesSections() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vindex_a3_\(UUID().uuidString).vindex").path
        let k_c = 1, m = 8
        let mmap = try VIndexContainerBuilder.createMinimalContainer(
            path: tmp, format: .pq8, k_c: k_c, m: m, d: 0,
            idBits: 64, group: 4, idCap: 32, payloadCap: 4)  // payloadCap=4 forces remap
        var opts = IVFAppendOpts.default; opts.format = .pq8; opts.durable = true
        let h = try ivf_create_mmap(k_c: k_c, m: m, d: 0, mmap: mmap, opts: opts)

        let n = 10   // > payloadCap => triggers ensureFileCapacity remap
        let listIDs = [Int32](repeating: 0, count: n)
        let extIDs = (0..<n).map { UInt64($0 + 100) }
        var codes = [UInt8](repeating: 0, count: n * m)
        for i in 0..<n { for j in 0..<m { codes[i*m + j] = UInt8(1 + j) } }
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: codes,
                       n: n, m: m, index: h, opts: opts, internalIDsOut: nil)
        try mmap.close()

        // Reopen fresh: indexInit parses the TOC. If the remap wrote a self-consistent
        // file, a fresh read of list 0 must return the appended codes.
        let reopened = try IndexMmap(path: tmp)   // confirm the exact opener API by grep
        defer { try? reopened.close(); _ = try? FileManager.default.removeItem(atPath: tmp) }
        let h2 = try ivf_open_mmap(k_c: k_c, m: m, d: 0, mmap: reopened, opts: opts) // confirm API
        let (len, _, _, codesPtr, _) = try h2.readList(listID: 0)
        XCTAssertEqual(len, n)
        let first = Array(UnsafeBufferPointer<UInt8>(start: codesPtr!, count: m))
        XCTAssertEqual(first, (0..<m).map { j in UInt8(1 + j) })
    }
}
```
Confirm the reopen APIs (`IndexMmap(path:)`, `ivf_open_mmap`/equivalent) by grepping `Kernel30AppendTests.swift`, `IDMapPersistenceTests.swift`, and `VIndexMmapErrorTests.swift`; substitute the real opener. Run it:
`swift test --filter RegressionA3_RemapTOCTests 2>&1 | tail -20`.

- [ ] **Step 2: Decide the path based on the result**

- If it **FAILS** (garbage codes / wrong len / throw after reopen): you have a red test → proceed to Step 3 (TDD).
- If it **PASSES** despite the offset bug: keep the test as a guard, and treat Step 3 as a consistency fix. Add a one-line note to the commit body: "remap TOC offsets corrected for on-disk consistency; not independently reproducible via reopen because section pointers rebind from per-list records."

- [ ] **Step 3: Apply the fix**

In `Sources/VectorIndex/Kernels/VIndexMmap.swift`, change the remap TOC parse (lines 952-956) to
match the canonical writer/`indexInit` offsets (4/12/20/24/28):
```swift
                let off = readLE64(te.advanced(by: 4))
                let sz  = readLE64(te.advanced(by: 12))
                let al  = readLE32(te.advanced(by: 20))
                let flags = readLE32(te.advanced(by: 24))
                let crc = readLE32(te.advanced(by: 28))
```

- [ ] **Step 4: Run the test + the existing remap test**

Run:
```bash
swift test --filter RegressionA3_RemapTOCTests 2>&1 | tail -20
swift test --filter Kernel30Append 2>&1 | tail -20
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorIndex/Kernels/VIndexMmap.swift Tests/VectorIndexTests/RegressionA3_RemapTOCTests.swift
git commit -m "$(cat <<'EOF'
fix(mmap): correct TOC field offsets on grow/remap path

ensureFileCapacity re-parsed TOC entries at offsets 8/16/24/28/32; the writer
and indexInit use the packed layout 4/12/20/24/28. Align them so section
offset/size are not corrupted after a file-growth remap.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 (spec A7): TOC stride uses the packed 36-byte constant

`tocSize` is computed with `MemoryLayout<_TOCEntry>.stride` (padded to ~48), while every reader
and the writer use `DISK_TOC_ENTRY_SIZE = 36`. The region is over-reserved (harmless slack) but
the constant is wrong. This is a consistency fix; a `tocOffset`/`tocSize`-shape assertion is the
guard (no behavioral red test — the slack is currently benign).

**Files:**
- Modify: `Sources/VectorIndex/Kernels/VIndexContainerBuilder.swift:91`

- [ ] **Step 1: Apply the fix**

In `Sources/VectorIndex/Kernels/VIndexContainerBuilder.swift`, change line 91 from:
```swift
        let tocSize = UInt64(tocCount * MemoryLayout<_TOCEntry>.stride)
```
to:
```swift
        // On-disk TOC entries are packed 36 bytes (see writeTOCEntry / indexInit);
        // MemoryLayout.stride would over-reserve due to struct padding.
        let DISK_TOC_ENTRY_SIZE = 36
        let tocSize = UInt64(tocCount * DISK_TOC_ENTRY_SIZE)
```

- [ ] **Step 2: Verify build + full mmap/persistence tests still pass**

Run:
```bash
swift build 2>&1 | tail -5
swift test --filter Kernel30Append 2>&1 | tail -10
swift test --filter VIndexMmap 2>&1 | tail -10
```
Expected: build OK; both test groups PASS (the writer already used 36, so container round-trips are unaffected; this only shrinks the reserved region to match).

- [ ] **Step 3: Commit**

```bash
git add Sources/VectorIndex/Kernels/VIndexContainerBuilder.swift
git commit -m "$(cat <<'EOF'
fix(mmap): compute tocSize from packed 36-byte entry size

Was MemoryLayout<_TOCEntry>.stride (~48 with padding) while readers/writer use
the packed 36-byte on-disk layout. Use the canonical DISK_TOC_ENTRY_SIZE.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 (spec A8): dedup touched-word tracking survives saturation

In `.fixedBitset` mode, once `touchedCount` reaches `touchedCapacity` the touched-word ring
stops recording, but bits keep being set. On reset the sparse-clear only clears recorded words,
leaving stale bits → false duplicates in the next query. Fix: set a `touchedOverflowed` flag on
saturation and force a full clear on reset when set.

**Files:**
- Test: `Tests/VectorIndexTests/RegressionA8_DedupOverflowTests.swift` (create)
- Modify: `Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift` (fields ~151-153; sites 472, 630; reset 289-303)

- [ ] **Step 1: Write the failing test**

The bug requires `touchedCapacity < wordCount` and saturation below `wordCount/4`. `touchedCapacity
= max(16_384, min(wCount, 1_000_000))`, so we need `wCount > 1_000_000` ⇒ `idCapacity > 64M`. That
is an ~8MB bitset — acceptable for a single test. Create
`Tests/VectorIndexTests/RegressionA8_DedupOverflowTests.swift`:
```swift
import XCTest
@testable import VectorIndex

final class RegressionA8_DedupOverflowTests: XCTestCase {
    func testFixedBitsetResetClearsPostSaturationBits() {
        // wCount = ceil(idCapacity/64) must exceed 1_000_000 so touchedCapacity
        // caps at 1_000_000 < wCount, allowing saturation before the dense-clear threshold.
        let idCapacity: Int64 = 70_000_000            // wCount ~= 1_093_750 > 1_000_000
        let vs = DefaultVisitedSet(idCapacity: idCapacity, mode: .fixedBitset) // confirm init label

        // Touch enough distinct words to saturate the ring (1_000_000) plus a few more.
        // Word index = id >> 6; step by 64 to hit distinct words.
        let cap = 1_000_000
        let extra = 5
        for w in 0..<(cap + extra) {
            let id = Int64(w) << 6
            _ = vs.testAndSet(id)                      // confirm public test-and-set method name
        }
        vs.resetForNewQuery()

        // A word touched AFTER saturation (index cap+1) must be cleared: testAndSet
        // returns true (newly inserted) if reset cleared it; false (stale) exposes the bug.
        let postSaturationID = Int64(cap + 1) << 6
        XCTAssertTrue(vs.testAndSet(postSaturationID),
                      "post-saturation bit must be cleared on reset")
    }
}
```
Confirm the public API names by grepping `CandidateDedup.swift`: the `DefaultVisitedSet` initializer
label (`idCapacity:mode:` or similar), the public test-and-set entry (the `VisitedSet` protocol
method — likely `testAndSet(_:) -> Bool`), and the `.fixedBitset` mode case. Substitute exact names.

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter RegressionA8_DedupOverflowTests 2>&1 | tail -20`
Expected: FAIL — `testAndSet(postSaturationID)` returns `false` (bit still set from before reset).
(If it unexpectedly passes, the ring never saturated: increase `idCapacity` so `wCount` clears
1M by a wider margin, and verify the mode is `.fixedBitset` not `.denseEpoch`.)

- [ ] **Step 3: Apply the fix**

3a. Add the flag field near lines 151-153 in `CandidateDedup.swift`:
```swift
    @usableFromInline internal var touchedOverflowed: Bool = false
```
3b. In `_testAndSet_bitset` (line 472 area), record saturation. Change:
```swift
            if word == 0, touchedCount < touchedCapacity, let tw = touchedWords {
                tw[touchedCount] = w
                touchedCount &+= 1
            }
```
to:
```swift
            if word == 0 {
                if touchedCount < touchedCapacity, let tw = touchedWords {
                    tw[touchedCount] = w
                    touchedCount &+= 1
                } else {
                    touchedOverflowed = true
                }
            }
```
3c. In `_trackTouchedWord` (line 628-634), mirror it:
```swift
    @usableFromInline
    internal func _trackTouchedWord(_ w: Int) {
        if touchedCount < touchedCapacity, let tw = touchedWords {
            tw[touchedCount] = w
            touchedCount &+= 1
        } else {
            touchedOverflowed = true
        }
    }
```
3d. In `resetForNewQuery` fixedBitset branch (lines 289-303), force a full clear when overflowed.
Change the clear decision:
```swift
        } else if mode == .fixedBitset {
            let tc = touchedCount
            if touchedOverflowed, let bw = bitWords {
                for i in 0..<wordCount { bw[i] = 0 }   // ring saturated: sparse set is incomplete
            } else if tc > 0, let bw = bitWords {
                if tc < wordCount / 4, let tw = touchedWords {
                    for i in 0..<tc { bw[tw[i]] = 0 }
                } else {
                    for i in 0..<wordCount { bw[i] = 0 }
                }
            }
            touchedCount = 0
            touchedOverflowed = false
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter RegressionA8_DedupOverflowTests 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift Tests/VectorIndexTests/RegressionA8_DedupOverflowTests.swift
git commit -m "$(cat <<'EOF'
fix(dedup): force full clear when touched-word ring saturates

Once touchedCount hit touchedCapacity the ring stopped recording while bits
kept being set; sparse reset then left stale bits => false duplicates. Track a
touchedOverflowed flag and full-clear on reset when set.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 (spec A6): PQ centroid-norm buffer is freed, not leaked

`ensureCentroidSqNorms` allocates `[m*ks]` and returns it without deallocation when
`opts.centroidSqNorms == nil`, leaking on every Swift-fallback encode. Fix: at each of the six
call sites, add a `defer` that deallocates only when the buffer was actually allocated (i.e. when
`opts.centroidSqNorms == nil`). **Testability:** a leak is not a behavioral red test; the guard is
"encode still produces identical codes" (existing PQ parity tests) plus a repeated-encode smoke
test that must not crash. This is documented hardening per spec §4.

**Files:**
- Test: reuse existing `PQEncodeParity_SwiftOnly_Tests` (no new behavioral test)
- Modify: `Sources/VectorIndex/Operations/Quantization/PQEncode.swift` at lines 88, 104, 208, 265, 281, 402

- [ ] **Step 1: Apply the fix at all six call sites**

At each site, immediately after the `let centroidSq = ensureCentroidSqNorms(maybeSq: opts.centroidSqNorms, ...)`
binding, insert an ownership-aware `defer`. The pattern (identical at every site; the `maybeSq`
argument is `opts.centroidSqNorms` at all six):
```swift
            let ownsCSq = (opts.centroidSqNorms == nil)
            let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
                maybeSq: opts.centroidSqNorms, codebooks: <codebooks-at-site>, m: m, ks: ks, dsub: dsub
            )
            defer { if ownsCSq { UnsafeMutablePointer(mutating: centroidSq).deallocate() } }
```
- Sites 1 (line 88) and 4 (line 265): the binding is inside an `if opts.useDotTrick {` block; place
  the `let ownsCSq` and `defer` inside that same block (the block closes before the enclosing
  `return`, so the defer fires while `centroidSq` is still needed only within the block — correct,
  because `centroidSq` is used before the block closes).
- Sites 2 (104), 3 (208), 5 (281), 6 (402): top-level in the function body; the `defer` fires at
  function exit — after all uses. At sites 5 and 6, place it right after the existing binding and
  before the existing `let rBuf = ... ; defer { rBuf.deallocate() }` lines (order of independent
  defers does not matter).
- Use `<codebooks-at-site>` = `codebooks` at sites 1/2/3, `residualCodebooks` at sites 4/5/6 (as in
  the current code — do not change that argument).

Because `ensureCentroidSqNorms` returns `opts.centroidSqNorms` unchanged when non-nil, `ownsCSq`
guarantees we only free memory we allocated.

- [ ] **Step 2: Verify correctness is unchanged (parity tests)**

Run:
```bash
swift build 2>&1 | tail -5
swift test --filter PQEncodeParity 2>&1 | tail -20
swift test --filter PQTrain 2>&1 | tail -10
```
Expected: build OK; all PQ parity/train tests PASS (codes byte-identical — the fix only frees a
buffer after use). If any PQ test now crashes/double-frees, the culprit is a site where
`centroidSq` outlives the `defer` scope — re-check that site's block boundaries.

- [ ] **Step 3: Add a repeated-encode smoke test (guard)**

Append to the existing `Tests/VectorIndexTests/PQEncodeParity_SwiftOnly_Tests.swift` a test that
encodes many times without precomputed norms and asserts stable output (drives the previously-
leaking path repeatedly). Model it on an existing test in that file (grep the file for its encode
setup and reuse the same fixtures/entry point); the assertion is that two runs produce identical
codes and the loop completes:
```swift
    func testRepeatedEncodeWithoutPrecomputedNormsIsStable() throws {
        // Reuse this file's existing fixture builders for x/codebooks/opts with
        // opts.centroidSqNorms == nil and opts.useDotTrick == true.
        // <build x, codebooks, opts exactly as the neighbouring parity test does>
        // var first: [UInt8] = ...; encode once into `first`.
        // for _ in 0..<200 { encode into `codes`; XCTAssertEqual(codes, first) }
    }
```
Fill the `<...>` from the sibling test verbatim (the plan intentionally points to that file's
existing setup rather than duplicating fixtures that must match the codebook shape). Run:
`swift test --filter PQEncodeParity_SwiftOnly 2>&1 | tail -20` → PASS.

- [ ] **Step 4: Commit**

```bash
git add Sources/VectorIndex/Operations/Quantization/PQEncode.swift Tests/VectorIndexTests/PQEncodeParity_SwiftOnly_Tests.swift
git commit -m "$(cat <<'EOF'
fix(pq): free centroid squared-norm buffer instead of leaking it

ensureCentroidSqNorms leaked its [m*ks] allocation on every Swift-fallback
encode without precomputed norms. Add an ownership-aware defer at all six call
sites (free only when opts.centroidSqNorms == nil).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 (spec A1): HNSW traversal pointers no longer escape their buffer scope

At three sites, per-layer CSR base pointers (`[[Int32]]` element storage) are captured out of
`withUnsafeBufferPointer` closures into arrays consumed after the closures return — formally UB.
The backing `[[Int32]]` (a stored property / retained ctx field) stays alive, so it "works" today,
but the optimizer may break it. Fix: wrap the pointer-assembly-and-`traverse` region in
`withExtendedLifetime` of the source arrays so their storage is provably retained across the call.
**Testability:** UB is not deterministically reproducible; the guard is a stress test (must PASS)
plus an AddressSanitizer run. This is documented hardening per spec §4, not a red→green cycle.

**Files:**
- Test: `Tests/VectorIndexTests/RegressionA1_TraversalLifetimeTests.swift` (create)
- Modify: `Sources/VectorIndex/HNSWIndex.swift` (search ~186-207; performSingleSearch ~321-368), `Sources/VectorIndex/HNSWKNNGraph.swift` (~123-153)

- [ ] **Step 1: Write the stress guard test**

Create `Tests/VectorIndexTests/RegressionA1_TraversalLifetimeTests.swift`:
```swift
import XCTest
@testable import VectorIndex

final class RegressionA1_TraversalLifetimeTests: XCTestCase {
    func testRepeatedSearchAndKNNGraphStable() async throws {
        let dim = 16, n = 400
        let idx = HNSWIndex(dimension: dim)
        var rng: UInt64 = 0xDEADBEEF
        func rnd() -> Float { rng = 2862933555777941757 &* rng &+ 3037000493; return Float(rng >> 40) / Float(1 << 24) }
        var vecs: [[Float]] = []
        for i in 0..<n {
            let v = (0..<dim).map { _ in rnd() * 2 - 1 }
            vecs.append(v)
            try await idx.insert(id: "id\(i)", vector: v, metadata: nil)
        }
        // Many searches: results must be stable and non-empty across repeats.
        let q = vecs[0]
        let baseline = try await idx.search(query: q, k: 10, filter: nil).map { $0.id }
        XCTAssertFalse(baseline.isEmpty)
        for _ in 0..<200 {
            let r = try await idx.search(query: q, k: 10, filter: nil).map { $0.id }
            XCTAssertEqual(r, baseline, "search results must be deterministic/stable")
        }
        // buildKNNGraph exercises the third escaping site.
        let (g1, _) = try await idx.buildKNNGraph(k: 10)
        let (g2, _) = try await idx.buildKNNGraph(k: 10)
        XCTAssertEqual(g1.neighborIndices, g2.neighborIndices, "kNN graph must be deterministic")
    }
}
```

- [ ] **Step 2: Run the guard (expected PASS before and after; ASan is the real detector)**

Run: `swift test --filter RegressionA1_TraversalLifetimeTests 2>&1 | tail -20`
Expected: PASS (this documents intended behavior; it is a guard, not a red test).
Also run once under AddressSanitizer to look for the use-after-free:
`swift test --filter RegressionA1_TraversalLifetimeTests -Xswiftc -sanitize=address 2>&1 | tail -30`
Record whether ASan flags anything on the pre-fix code.

- [ ] **Step 3: Apply the fix at site 1 (`search`, HNSWIndex.swift ~186-207)**

Wrap the CSR pointer assembly + `traverse` in `withExtendedLifetime` of the two caches. Immediately
inside `allowBits.withUnsafeBufferPointer { allowBP in` (line 188), wrap the body:
```swift
                allowBits.withUnsafeBufferPointer { allowBP in
                  withExtendedLifetime(csrOffsetsCache) { withExtendedLifetime(csrNeighborsCache) {
                    // Build pointer arrays for layers from cached CSR
                    let offPtrsOpt = csrOffsetsCache.map { arr in arr.withUnsafeBufferPointer { Optional($0.baseAddress!) } }
                    let nbrPtrsOpt = csrNeighborsCache.map { arr in arr.withUnsafeBufferPointer { Optional($0.baseAddress!) } }
                    return offPtrsOpt.withUnsafeBufferPointer { offArr in
                        // ... existing body unchanged through the traverse(...) call and its return ...
                    }
                  } }
                }
```
Keep the existing inner body verbatim; only the two `withExtendedLifetime` wrappers and their
closing `} }` are added. Ensure the wrapper `return`s the inner result (the `withExtendedLifetime`
closures return their body value).

- [ ] **Step 4: Apply the fix at site 2 (`performSingleSearch`, ~321-368)**

Immediately inside `ctx.allowBits.withUnsafeBufferPointer { allowBP in` (line 323), wrap the
`offPtrs`/`nbrPtrs` build + `traverse` region:
```swift
                ctx.allowBits.withUnsafeBufferPointer { allowBP in
                  withExtendedLifetime(ctx.csrOffsets) { withExtendedLifetime(ctx.csrNeighbors) {
                    var offPtrs = [UnsafePointer<Int32>?]()
                    // ... existing body unchanged through the returned traverse result ...
                  } }
                }
```

- [ ] **Step 5: Apply the fix at site 3 (`buildKNNRows`, HNSWKNNGraph.swift ~123-153)**

Immediately inside `ctx.allowBits.withUnsafeBufferPointer { allowBP in` (line 124), wrap:
```swift
            ctx.allowBits.withUnsafeBufferPointer { allowBP in
              withExtendedLifetime(ctx.csrOffsets) { withExtendedLifetime(ctx.csrNeighbors) {
                var offPtrs = [UnsafePointer<Int32>?]()
                // ... existing body unchanged through the row loop / traverse ...
              } }
            }
```

- [ ] **Step 6: Build + run the guard + a broad HNSW test sweep**

Run:
```bash
swift build 2>&1 | tail -5
swift test --filter RegressionA1_TraversalLifetimeTests 2>&1 | tail -10
swift test --filter HNSW 2>&1 | tail -15
swift test --filter HNSWKNNGraph 2>&1 | tail -15
```
Expected: build OK; all PASS. Re-run the ASan command from Step 2 and confirm it is clean.

- [ ] **Step 7: Commit**

```bash
git add Sources/VectorIndex/HNSWIndex.swift Sources/VectorIndex/HNSWKNNGraph.swift Tests/VectorIndexTests/RegressionA1_TraversalLifetimeTests.swift
git commit -m "$(cat <<'EOF'
fix(hnsw): keep CSR backing storage alive across traverse (pointer lifetime)

Per-layer CSR base pointers were captured out of withUnsafeBufferPointer scopes
and used after they returned (formal UB). Wrap the assembly + traverse in
withExtendedLifetime of the source [[Int32]] arrays at all three sites.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 wrap-up

- [ ] **Step 1: Full suite green**

Run: `swift test 2>&1 | tail -20`
Expected: entire suite PASS (0 unexpected failures).

- [ ] **Step 2: Update CHANGELOG `### Fixed`**

Append under `## [Unreleased] — 0.2.0` → `### Fixed`:
```markdown
- Durable `IVFListHandle.getListStats` returns real stats (was always throwing). (A2)
- `HNSWIndex.batchRemove` no longer corrupts the index on subset removal. (A5)
- Exact rerank preserves 64-bit candidate ids (was truncated to Int32). (A4)
- Correct TOC field offsets on the mmap grow/remap path. (A3)
- `tocSize` uses the packed 36-byte entry size. (A7)
- Dedup forces a full clear when the touched-word ring saturates. (A8)
- PQ centroid squared-norm buffer is freed instead of leaked. (A6)
- HNSW traversal CSR pointers no longer escape their buffer scope. (A1)
```

- [ ] **Step 3: Commit the CHANGELOG and confirm consumer still builds**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record Phase 1 correctness fixes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
cd /Users/goftin/dev/gsuite/VSK/future/VectorIndexAccelerated && swift build 2>&1 | tail -10
```
Expected: consumer build result no worse than the Phase-0 baseline (Task 0 Step 5).

---

## Self-review notes (author)

- **Spec coverage:** A1–A8 each have a task; Phase-0 gates (branch/baseline/CHANGELOG/consumer)
  are Task 0. Phases 2–5 are out of scope for this plan (separate plans).
- **Airtightness honesty:** A2/A5/A4/A8 are true red→green TDD. A3 is red→green *if* reproducible,
  else a documented consistency fix (Step 2 decision gate). A7/A6/A1 are documented hardening with
  guards (no deterministic red test), each stating why.
- **API confirmations flagged inline** (not logic placeholders): exact `RerankOpts.backend` case
  (Task 3), `DefaultVisitedSet` init/`testAndSet`/mode labels (Task 6), mmap reopen API (Task 4),
  `HNSWIndex.count` async-ness (Task 2), and PQ sibling fixture reuse (Task 7). Each names the exact
  grep to confirm before editing.
- **Type consistency:** CSR caches/ctx fields are `[[Int32]]` (verified); `getListDescriptor`
  returns a labeled tuple with `.capacity: Int`; `selectTopK_streaming` accepts `ids: nil`.
