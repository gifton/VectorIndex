# VectorIndex Phase 4: Pointer-Safety Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ASan-confirmed stack-buffer-overflow reads in `PQTrain.swift`'s minibatch k-means family by replacing the unsafe `&array[index]` pointer idiom with buffer-pointer + offset arithmetic, and add the sanitizer coverage that would have caught it.

**Architecture:** Every defect is the same shape: a pointer obtained from `&array[index]` (valid for exactly one element in Swift) is passed to `l2Sq`, which reads `len` contiguous floats from it. The compiler is free to materialize that single element into a 4-byte stack temporary, so `l2Sq` walks off the end of it. The correct pattern already exists in the same file — `lloydKMeansSubspace` (`:849-971`) and `kmeansppSeedSubspaceDense` (`:972-1021`) hoist `withUnsafeBufferPointer` / `withUnsafeMutableBufferPointer` once and do offset arithmetic on the base address. Phase 4 converts the remaining 22 call sites to that pattern, function by function, each verified under AddressSanitizer.

**Tech Stack:** Swift 6, SwiftPM, XCTest, AddressSanitizer (`--sanitize=address`), GitHub Actions.

## Background — REQUIRED READING before Task 1

Two committed-to-be-committed investigation documents are the spec for this phase. Read both:

- `docs/superpowers/2026-08-08-pq-streaming-distortion-blowup-investigation.md` — root cause, ASan proof, the decisive same-process-vs-fresh-process control experiment, and why every existing guard misses it.
- `docs/superpowers/2026-08-09-pointer-escape-scope-assessment.md` — the full call-site inventory, which paths are safe, the Task-9 A/B non-regression proof, and CI gaps.

Three findings from those documents bind every task here:

1. **The corrupted value is finite.** Garbage bytes reinterpreted as `Float` decode to huge-but-finite values, so `isFinite` checks never fire. Do not add `isFinite` guards expecting them to fix anything (see Task 8 for the one place a guard is still wanted, on its own merits).
2. **Print-instrumentation MASKS this bug.** Adding prints near the hot loop perturbs stack layout enough that the overflowing read lands on benign memory — the previous investigation was healthy 12/12 through its own diagnostics. **Use ASan, never prints, to verify these fixes.**
3. **A single green run proves nothing.** The garbage depends on process history, so correctness must be demonstrated under ASan (which is deterministic about detecting the overflow), not by observing a sane distortion once.

## Global Constraints

- **NON-BREAKING.** `l2Sq` is `private`; every call site in scope is internal. No public API changes. Despite Phase 4 originally being earmarked for breaking work, this phase ships without any.
- **VectorCore stays pinned** at 0.3.1 (`b26909e98b6a9c6b83f19904ea0072646a4920fd`). Never modify `Package.swift` / `Package.resolved`.
- **Behavior-preserving in exact arithmetic.** These are pointer-provenance fixes, not algorithm changes. The same operands must be compared in the same order. Where a fix changes results, it is because the old results were garbage — say so explicitly rather than adjusting a test to match.
- **ASan is the verification instrument.** Every fix task's gate is `swift test --filter <Suite>/<method> --sanitize=address` passing clean where it currently faults. First ASan build is slow; run it FOREGROUND and wait.
- **No print-based debugging in these functions.** See Background item 2.
- All builds/tests run FOREGROUND with explicit large timeouts (600000). Never background, never end a turn mid-command.
- Single-suite or single-method `--filter` values only. `swift test --filter` with 6+ `|` alternation terms silently runs the FULL suite.
- `PQTrainTests` unfiltered takes ~3.4 minutes on a quiet machine (it was ~60 min under the load contention documented in Phase 3; that contention is resolved). Running it unfiltered is now acceptable, but prefer method filters while iterating.
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Never `git add` any path under `.superpowers/`. Local commits only; no pushes without authorization.

## File Structure

- `Sources/VectorIndex/Kernels/PQTrain.swift` — all 22 unsafe call sites. Every fix task touches this one file; tasks are separated by *function*, so they do not overlap.
- `Tests/VectorIndexTests/PQTrainTests.swift` — regression assertions (Tasks 3, 5).
- `.github/workflows/ci.yml` — sanitizer job and test-matrix gap (Task 6).
- `docs/superpowers/` — the two investigation documents get committed in Task 0.

---

### Task 0: Commit the investigation record

The two investigation documents are currently untracked. They are the spec for this phase and the durable record of a subtle bug; they must be in git before any code changes.

**Files:**
- Commit (already written, do not edit): `docs/superpowers/2026-08-08-pq-streaming-distortion-blowup-investigation.md`
- Commit (already written, do not edit): `docs/superpowers/2026-08-09-pointer-escape-scope-assessment.md`

- [ ] **Step 1: verify both files exist and are untracked**

Run: `git status --short docs/superpowers/`
Expected: both files listed with `??`.

- [ ] **Step 2: commit them by explicit path**

```bash
git add docs/superpowers/2026-08-08-pq-streaming-distortion-blowup-investigation.md \
        docs/superpowers/2026-08-09-pointer-escape-scope-assessment.md
git commit -m "docs: record PQTrain pointer-escape investigation and scope assessment

ASan-confirmed stack-buffer-overflow READ in l2Sq, reached from the
minibatch k-means family via the &array[index] single-element pointer
idiom. These two documents are the spec for Phase 4.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Do NOT stage `.bench/post-phase3/` or anything else.

---

### Task 1: Fix `minibatchKMeansSubspaceChunk` (P0, ASan-confirmed)

The originally-confirmed crash site. Unsafe calls at `:1405, :1407, :1412, :1414`, plus the `sums` accumulation reads at `:1409, :1417` which index `xChunk`/`coarse` directly (safe, but land inside the region you are wrapping).

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`minibatchKMeansSubspaceChunk`, `:1363-1440`)

**Interfaces:** none — `private` function, signature unchanged.

- [ ] **Step 1: confirm the failing state (RED)**

Run: `swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address 2>&1 | tail -30`
Expected: ASan aborts with `stack-buffer-overflow`, `#0 l2Sq PQTrain.swift:760`, `#1 minibatchKMeansSubspaceChunk PQTrain.swift:1412`. Capture this output verbatim for the report — it is the RED evidence.

- [ ] **Step 2: hoist buffer pointers around the batch loop**

The function currently does `var xChunk = xChunk` and takes `C: inout [Float]`. Restructure the `while s < nI` body so both arrays are accessed through base pointers held for the whole loop. `C` is read (distance) and written (update) in the same scope, so it needs `withUnsafeMutableBufferPointer`; read `C` through the same pointer rather than through the array to avoid overlapping access.

```swift
// Replace `var xChunk = xChunk` and the direct &-indexing with:
xChunk.withUnsafeBufferPointer { xbuf in
    let xptr = xbuf.baseAddress!
    C.withUnsafeMutableBufferPointer { cbuf in
        let cptr = cbuf.baseAddress!
        // ... existing while s < nI { ... } body, with every
        // l2Sq(&xChunk[base], &C[k*dsub], dsub)  ->  l2Sq(xptr + base, cptr + k*dsub, dsub)
        // and every  C[baseC + u]  ->  cptr[baseC + u]
        // and every  xChunk[base+u]  ->  xptr[base+u]
    }
}
```

For the `coarse` branch, wrap `coarse` in its own `withUnsafeBufferPointer` (it is a `let` copy) and use `gptr + gbase` for the `subtract:` argument — exactly as `lloydKMeansSubspace` does at `:849`.

Keep the operand order in every `l2Sq` call identical to the current code so the comparison sequence is unchanged.

- [ ] **Step 3: verify (GREEN)**

Run: `swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address 2>&1 | tail -10`
Expected: PASS, no ASan report.

- [ ] **Step 4: verify the distortion is now sane and stable across processes**

Run this three times as three separate invocations (the bug was process-history dependent):
`swift test --filter PQTrainTests/testStreamingPQTraining 2>&1 | grep "Streaming training"`
Expected: three finite values of the same small magnitude (the investigation's in-process control observed `2.6725087868710102`; expect that order, not `1e25`). Record all three. They need not be bit-identical to each other yet — Task 3 fixes the remaining unsafe reads in the distortion evaluation itself.

- [ ] **Step 5: commit**

```bash
git add Sources/VectorIndex/Kernels/PQTrain.swift
git commit -m "fix(pq): buffer-pointer provenance in minibatchKMeansSubspaceChunk

l2Sq reads dsub contiguous floats, but &array[index] is only valid for
one element; the compiler materialized a 4-byte stack temporary and l2Sq
read past it. ASan-confirmed stack-buffer-overflow at PQTrain.swift:760.
Hoist withUnsafe[Mutable]BufferPointer once and use offset arithmetic,
matching lloydKMeansSubspace's existing safe pattern.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Fix `minibatchKMeansSubspace` (P0, ASan-confirmed)

The non-chunked in-memory sibling — same defect, ten call sites across three blocks: assignment (`:1220, :1222, :1227, :1229`), empty-cluster repair (`:1284, :1286, :1290, :1292`), and the distortion-estimate block (`:1343, :1345`).

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`minibatchKMeansSubspace`, `:1164-1361`)

**Interfaces:** none — `private` function, signature unchanged.

- [ ] **Step 1: confirm the failing state (RED)**

Run: `swift test --filter PQTrainTests/testMiniBatchPQTraining --sanitize=address 2>&1 | tail -30`
Expected: ASan abort, `#1 minibatchKMeansSubspace PQTrain.swift:1227`. Capture verbatim.

- [ ] **Step 2: apply the same hoist to all three blocks**

Same transformation as Task 1. All three blocks live in one function, so a single pair of `withUnsafeBufferPointer` (for `x`) / `withUnsafeMutableBufferPointer` (for `C`) wrapping the function body is preferable to three separate wraps — but only if that does not force awkward nesting around the existing `emptyKs` logic. If one wrap is unwieldy, wrap each block separately and say so in the report.

Preserve the existing tie-break `if dk < bestD || (dk == bestD && k < bestK)` exactly.

- [ ] **Step 3: verify (GREEN)**

Run each as a separate invocation:
- `swift test --filter PQTrainTests/testMiniBatchPQTraining --sanitize=address 2>&1 | tail -10` → PASS
- `swift test --filter PQTrainTests/testWarmStartMinibatchImprovesOnePass --sanitize=address 2>&1 | tail -10` → PASS
- `swift test --filter PQTrainTests/testWarmStartDeterministic --sanitize=address 2>&1 | tail -10` → PASS

- [ ] **Step 4: check the false-negative case**

`testLargeScaleTraining` uses `dsub=1`, so `l2Sq` reads exactly one element and never overflows — it passed ASan before the fix for that reason, not because the path was safe. Run it under ASan now to confirm it still passes: `swift test --filter PQTrainTests/testLargeScaleTraining --sanitize=address 2>&1 | tail -10`. Note in the report that this test cannot detect this bug class and should not be cited as coverage.

- [ ] **Step 5: commit** — `fix(pq): buffer-pointer provenance in minibatchKMeansSubspace` + trailer.

---

### Task 3: Fix the streaming distortion evaluation + add a real assertion (P1)

`pq_train_streaming_f32`'s final distortion loop (`:637-654`) uses the same unsafe idiom at `:644, :646`. This is the code that actually produces the `3.47e25` number. Fixing it makes the reported statistic trustworthy; the test must then assert something that would have caught the old behavior.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`:637-654`)
- Modify: `Tests/VectorIndexTests/PQTrainTests.swift` (`testStreamingPQTraining`)

- [ ] **Step 1: strengthen the assertion FIRST, and watch it fail (RED)**

The current assertions are `XCTAssertEqual(codebooks.count, m*ks*(d/m))` and `XCTAssert(stats.distortion > 0)` (`:670-671`). Add a data-scale upper bound, following the `trivialDistortion` precedent already in `testStreamingSeederSmallNTakesStreamingBranch` (`:728-736`) — compute the all-zero-centroid baseline inline over the same data and require the trained distortion to be below it:

```swift
// A trained codebook must beat the trivial all-zero-centroid baseline.
// The pre-fix code reported ~1e25 here, which this bound rejects.
var trivial = 0.0
for i in 0..<Int(n) {
    for u in 0..<d { let v = Double(fullData[i*d + u]); trivial += v*v }
}
trivial /= Double(n)
XCTAssertLessThan(stats.distortion, trivial,
                  "distortion must beat the all-zero-centroid baseline")
XCTAssertTrue(stats.distortion.isFinite)
```

Run: `swift test --filter PQTrainTests/testStreamingPQTraining 2>&1 | tail -5`
Expected after Task 1 but before this task's Step 2: this may already PASS if Task 1 removed enough corruption. **If it passes, that is fine — record it and proceed.** The RED evidence for this bug class is the ASan report, not this assertion; the assertion exists to prevent silent regression.

- [ ] **Step 2: apply the buffer-pointer hoist to the distortion loop**

```swift
var Dj: Double = 0
var seen: Int64 = 0
for (c, nc) in nChunks.enumerated() {
    guard nc > 0 else { continue }
    xChunks[c].withUnsafeBufferPointer { xbuf in
        let xptr = xbuf.baseAddress!
        Cj.withUnsafeBufferPointer { cbuf in
            let cptr = cbuf.baseAddress!
            for i in 0..<Int(nc) {
                let base = i * d + j * dsub
                var best = l2Sq(xptr + base, cptr, dsub)
                for k in 1..<ks {
                    let dval = l2Sq(xptr + base, cptr + k*dsub, dsub)
                    if dval < best { best = dval }
                }
                if best < 0 { best = 0 }
                if best.isFinite { Dj += Double(best) }
            }
        }
    }
    seen += nc
}
```

Keep the `best < 0` clamp and the `isFinite` filter — they are cheap and harmless, even though the investigation showed they cannot catch this defect.

- [ ] **Step 3: verify** — `swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address 2>&1 | tail -10` → PASS.

- [ ] **Step 4: verify cross-process determinism**

Run three separate invocations of `swift test --filter PQTrainTests/testStreamingPQTraining 2>&1 | grep "Streaming training"`. With the reads now in-bounds, all three values must be **bit-identical** (the inputs are fully seeded). Report all three. If they differ, STOP — an unfixed unsafe read remains somewhere in the path.

- [ ] **Step 5: commit** — `fix(pq): buffer-pointer provenance in streaming distortion eval; assert distortion bound` + trailer.

---

### Task 4: Fix the streaming empty-cluster repair block (P1, NOT mechanical)

`pq_train_streaming_f32`'s inline empty-repair block (`:546-605`, unsafe calls at `:576, :578, :582, :584`) indexes randomly across chunks per iteration, so there is no single array to wrap for the whole loop. This needs a design decision, not a copy-paste — which is why it is its own review unit.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`:546-605`)

- [ ] **Step 1: read the block and characterize the access pattern**

Determine, and state in your report before changing anything: which arrays are indexed, whether the chunk index varies per iteration, and whether the candidate selection can be reordered to group accesses by chunk without changing which candidate is chosen. The RNG draw order must not change — that would alter results.

- [ ] **Step 2: choose and document an approach**

Two viable shapes, pick one and justify:
- **(a) Per-iteration wrap:** wrap `withUnsafeBufferPointer` inside the loop body around just the chosen chunk. Simplest and provably correct; costs one closure entry per repair candidate, which is negligible since repairs are rare.
- **(b) Pre-resolved base pointers:** collect all chunk base addresses up front via nested `withUnsafeBufferPointer` calls. Faster but the nesting depth equals the chunk count, which is not statically known — likely impractical. Reject this unless you find a clean formulation.

Default to (a) unless there is a concrete reason not to. Repair frequency is low; correctness dominates.

- [ ] **Step 3: implement, preserving RNG draw order exactly**

- [ ] **Step 4: verify** — ASan run of `swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address`, plus confirm from the `#if DEBUG` output that the repair path actually executed (`emptyPass > 0` in the trace). If the repair path did not execute, the ASan run did not exercise your change — construct a config that produces empty clusters and say how.

- [ ] **Step 5: commit** — `fix(pq): buffer-pointer provenance in streaming empty-cluster repair` + trailer.

---

### Task 5: Fix `streamingKMeansppSeed` + give it dsub>1 coverage (P1)

Unsafe calls at `:1481, :1518`. Harder because `outC` is both read and written in the scope needing the wrap. Additionally, the only existing test for this function (`testStreamingSeederSmallNTakesStreamingBranch`, added in Phase 3) uses `d=16, m=2` → `dsub=8`, so it does exercise a multi-element read — but `ks=16` with `n=40`, a much smaller shape than production. Confirm whether it faults.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`streamingKMeansppSeed`, `:1444-1523`)
- Modify: `Tests/VectorIndexTests/PQTrainTests.swift` if a larger-shape test is needed

- [ ] **Step 1: does it currently fault? (RED or documented no-repro)**

Run: `swift test --filter PQTrainTests/testStreamingSeederSmallNTakesStreamingBranch --sanitize=address 2>&1 | tail -30`
Record the result. If it faults, that is your RED. If it does NOT fault, the existing shape is too small to overflow into poisoned memory — note that explicitly and proceed anyway, because the idiom is unsafe regardless of whether this particular shape trips the detector.

- [ ] **Step 2: apply the hoist**

`outC` is `inout [Float]`. Use `outC.withUnsafeMutableBufferPointer` for the whole seeding body and read through the same pointer. Wrap each `xChunks[c]` per-chunk as in Task 4's approach (a) if a single wrap is not possible.

**Critical:** Task 10 of Phase 3 established that this function's output is bit-identical to its pre-rewrite form, pinned by `testStreamingSeederSmallNTakesStreamingBranch`'s `.bitPattern` snapshot assertions (`PQTrainTests.swift:750-753`). Your change must keep those constants passing — it is a pointer-provenance fix, not an arithmetic one. If the snapshot breaks, the old values were garbage-influenced; STOP and report rather than re-baselining the constants.

- [ ] **Step 3: verify** — `swift test --filter PQTrainTests/testStreamingSeederSmallNTakesStreamingBranch --sanitize=address` → PASS with snapshot constants intact.

- [ ] **Step 4: commit** — `fix(pq): buffer-pointer provenance in streamingKMeansppSeed` + trailer.

---

### Task 6: Close the CI gap (P1)

Today CI has no sanitizer job at all, and `PQTrainTests` is in **no** filter group while `CI_SKIP_PQTRAIN: '1'` is set (`.github/workflows/ci.yml:109`) — so this entire suite has never run in CI. That is why an ASan-detectable bug survived indefinitely.

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: add `PQTrainTests` to the regular matrix**

Add `PQTrainTests` to the `kernels` filter group (`:103`) and remove the `CI_SKIP_PQTRAIN: '1'` env (`:109`), or set it to `'0'`. The suite runs in ~3.4 minutes on unloaded hardware, which is acceptable for CI.

While editing, note in your report whether the existing filter regexes — single `^VectorIndexTests\.(A|B|C|…)` groups with many alternatives — are affected by the alternation quirk documented in Phase 3 (6+ `|` terms causing the full suite to run). If CI has silently been running the full suite on every job, that is worth flagging even though it is not this task's job to fix.

- [ ] **Step 2: add an ASan job**

Add a job that runs at least `PQTrainTests` under `--sanitize=address`. Model it on the existing job structure. ASan builds are slow; scope it to the kernels group rather than the whole suite, and consider `schedule:` plus `workflow_dispatch:` rather than every push if runtime is a concern — state your choice and why.

- [ ] **Step 3: prove the gate actually works**

This is the important step. Verify the new ASan job would have caught the original bug: check out `PQTrain.swift` at a pre-fix commit (`git stash` your work or use `git show <pre-fix-sha>:...` into a scratch copy), run the ASan command the CI job runs, and confirm it FAILS. Then restore and confirm it PASSES. A gate that has never been observed failing is not a gate.

- [ ] **Step 4: commit** — `ci: run PQTrainTests and add AddressSanitizer job` + trailer.

---

### Task 7: Scope the HNSW pointer escape (P2, investigation only)

`HNSWIndex.swift`'s `rebuildInvNormsIfNeededForCosine` (`:1163-1185`, called from `:206, :289, :448, :640, :726, :909, :960`) returns a pointer that escapes its own closure. Flagged by a Phase 3 reviewer and re-flagged by the scope assessment as a **different** defect mechanism from the `l2Sq` idiom — not yet ASan-verified either way.

**Files:**
- Investigate: `Sources/VectorIndex/HNSWIndex.swift`
- Create: `docs/superpowers/<date>-hnsw-pointer-escape-assessment.md`

- [ ] **Step 1: run the HNSW suites under ASan**

One invocation per suite: `HNSWTests`, `HNSWRecallTests`, `HNSWBatchAndErrorsTests`, `HNSWAlignmentTest`. Record faults or clean results with full stack traces.

- [ ] **Step 2: analyze the escape**

Determine whether the returned pointer is dereferenced after the closure exits, and whether the backing array is guaranteed alive and unmutated across that window. Phase 3's reviewer judged it "safe in practice (no mutation in scope)" — verify or refute that with evidence.

- [ ] **Step 3: write the assessment and STOP**

Do not fix in this task. Produce a document with the same structure as the Phase 4 scope assessment: evidence, verdict, and a recommended fix direction sized as tasks. Fixing becomes its own follow-up if warranted.

- [ ] **Step 4: commit the document** — `docs: assess HNSW cosine inv-norms pointer escape` + trailer.

---

### Task 8: Close the centroid-write guard asymmetry (P2, hardening)

`minibatchKMeansSubspaceChunk`'s centroid write (`:1433`) lacks the `v.isFinite ? v : 0` clamp its in-memory sibling has (`:1255`), and lacks the sibling's final full-array sanitization sweep (`:1330-1333`).

**This does NOT fix the ASan bug** — the investigation proved corrupted `l2Sq` output never reaches this write. It is a defensive asymmetry worth closing while the code is already being touched. Do not describe it in the commit message as fixing the overflow.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/PQTrain.swift` (`:1433` area)

- [ ] **Step 1: apply the guard, matching the sibling's exact idiom**

```swift
let v = Float(oldW * oldVal + newW * batchMean)
cptr[baseC + u] = v.isFinite ? v : 0
```

(Adjusted for the buffer pointer introduced in Task 1.)

- [ ] **Step 2: verify no behavior change on healthy input**

Run `swift test --filter PQTrainTests/testStreamingPQTraining` and confirm the distortion matches the bit-identical value established in Task 3 Step 4. The guard must be inert on well-behaved data.

- [ ] **Step 3: commit** — `fix(pq): match sibling's isFinite clamp on streaming centroid write (defensive)` + trailer.

---

### Task 9: Phase close-out

- [ ] **Step 1: full suite, no filter** — `swift test 2>&1 | tail -20`. All green. Record pass/skip counts.
- [ ] **Step 2: full ASan run over the kernels group** — one invocation, `--sanitize=address`, covering `PQTrainTests` and the kernel suites. Zero sanitizer reports.
- [ ] **Step 3: cross-process determinism check** — three separate invocations of `testStreamingPQTraining`; distortion bit-identical across all three.
- [ ] **Step 4: CHANGELOG** — under `## [Unreleased] — 0.2.0`, add a `### Fixed` entry. This one IS release-notes material (unlike the open ticket it replaces): describe the stack-buffer-overflow read, that it produced wildly wrong `distortion` statistics and potentially suboptimal cluster assignments, and that it is fixed. Note that PQ codebook *values* were not corrupted in any traced path.
- [ ] **Step 5: commit** — `chore: Phase 4 close-out — CHANGELOG and full-suite gate` + trailer.

---

## Self-Review (performed at plan-writing time)

1. **Spec coverage:** Scope-assessment items 1→Task 1, 2→Task 2, 3→Task 3, 4→Task 4, 5→Task 5, 6→Task 6, 7→Task 7, 8→Task 8. Task 0 (commit the investigation) and Task 9 (close-out) added — the scope doc assumed the documents were already committed, and had no close-out step.
2. **Placeholder scan:** no TBDs. Task 4's design choice is presented as two concrete named options with a stated default and a justification requirement, not a deferred decision. Task 5 Step 1 has documented handling for both the fault and no-fault outcomes.
3. **Type consistency:** `l2Sq(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float` and its `subtract:` overload are unchanged throughout; all fixes change only how the two pointer arguments are obtained. `xptr`/`cptr`/`gptr` naming follows `lloydKMeansSubspace`'s existing `xbase`/`cbase`/`gptr` convention.
4. **Ordering:** Tasks 1–3 are the ASan-confirmed and statistic-producing paths, so they come first and deliver the user-visible fix early. Tasks 4–5 are lower-frequency paths. Task 6 (CI) deliberately comes after the fixes so its "prove the gate fails pre-fix" step has a real pre-fix state to test against.
