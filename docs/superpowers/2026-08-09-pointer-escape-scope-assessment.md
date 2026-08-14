# Scope Assessment: `&array[index]` pointer-escape defect in PQTrain.swift

Status: **scoping complete**. Investigation and enumeration only — no production code was
changed as a result of this work (one temporary edit was made for an A/B experiment in Q3 and
was reverted with `git checkout --` before finishing; see Q3 for details and proof of revert).

Repo: `/Users/goftin/dev/gsuite/VSK/VectorIndex`, branch `gifton/perf-0.2.0-phase3`, HEAD `31b97d7`.

This builds directly on `docs/superpowers/2026-08-08-pq-streaming-distortion-blowup-investigation.md`,
which ASan-confirmed a stack-buffer-overflow READ in `l2Sq` (`PQTrain.swift:755-770`), reached
from `minibatchKMeansSubspaceChunk` at `:1412`, and flagged several other call sites as
structurally identical but unverified. This document verifies (or refutes) each of those
suspicions with evidence.

**Note on line numbers**: the prior doc cited `kmeansppSeedSubspaceDense` at `:972-1021` and the
Lloyd hot loop at `:849-971`. At the current HEAD these live at `:928-977` and `:1026-1076`
respectively — the functions and their content are otherwise as described; only the line
numbers drifted (unclear why, since HEAD is the same commit `31b97d7`, but immaterial to the
findings). All line numbers below are re-verified against the current file.

---

## Q1 — Does the non-streaming Lloyd path also trip ASan?

Ran each test individually, foreground, single-method filter, under `swift test --sanitize=address`.

| Test | Algo exercised | Result | Notes |
|---|---|---|---|
| `testBasicPQTraining` | `.lloyd` (default) | **CLEAN** — passed in 4.894s | Seeds via `kmeansppSeedSubspaceDense` (safe), trains via `lloydKMeansSubspace` (safe) |
| `testMiniBatchPQTraining` | `.minibatch` explicit | **FAULTED** | New ASan-confirmed site, see below |
| `testResidualPQ` | `.lloyd` (default), coarse/residual branch | **CLEAN** — passed in 58.086s | Exercises the `coarse`-branch of both seeding and Lloyd training |
| `testWarmStartLloydImprovesOneIter` | `.lloyd` explicit (cold + warm) | **CLEAN** — passed in 2.452s | Both cold-start and warm-start runs clean |
| `testLargeScaleTraining` | `.minibatch` explicit, but `d=8,m=8` → **dsub=1** | **CLEAN** — passed in 14.597s | Same unsafe idiom exercised, but see caveat below — this is *not* evidence the call site is safe |

### New ASan-confirmed fault: `testMiniBatchPQTraining`

```
==74740==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x00016b3d7cb4
READ of size 4 at 0x00016b3d7cb4 thread T0
    #0 l2Sq(_:_:_:) PQTrain.swift:760
    #1 minibatchKMeansSubspace(x:n:d:j:dsub:ks:coarse:assign:cfg:rng:C:didWarmStart:outDistortion:outIters:outEmpties:) PQTrain.swift:1227
    #2 runOneSubspace #1 @Sendable (_:) in pq_train_f32(...) PQTrain.swift:277
    #3 pq_train_f32(...) PQTrain.swift:329
    #4 PQTrainTests.testMiniBatchPQTraining() PQTrainTests.swift:84
SUMMARY: AddressSanitizer: stack-buffer-overflow PQTrain.swift:760 in l2Sq(_:_:_:)
```

This is a **second, independent confirmed crash site**, structurally identical to the
already-known one but in a *different function*: `minibatchKMeansSubspace` (the **non-chunked,
in-memory** minibatch path used by `pq_train_f32` when `cfg.algo = .minibatch`), as opposed to
`minibatchKMeansSubspaceChunk` (the **streaming/chunked** path used by
`pq_train_streaming_f32`). Both are reached via the exact same unsafe idiom at their respective
assignment loops.

**Important clarification of what "Lloyd path" means here.** The prior doc's phrase "non-streaming
Lloyd path" for line ranges `:1220-1292, :1343-1345` is a slight misnomer: those lines are inside
`minibatchKMeansSubspace`, not `lloydKMeansSubspace`. The *actual* Lloyd algorithm
(`lloydKMeansSubspace`, `:981-1160`) is fully safe (confirmed clean by three passing ASan runs
above) — it already uses the `withUnsafeBufferPointer`/`xbase+base` pattern throughout, including
in its `coarse`-branch and its `.split` empty-cluster-repair branch. The defect is confined to the
**minibatch family** (`minibatchKMeansSubspace` and `minibatchKMeansSubspaceChunk`), not Lloyd.

### The `testLargeScaleTraining` clean result is a false-negative, not a safety proof

`testLargeScaleTraining` uses `algo = .minibatch` with `d=8, m=8` → `dsub = d/m = 1`. In `l2Sq`,
`l8 = len & ~7 = 1 & ~7 = 0`, so the 8-wide SIMD loop body never executes; the function falls
straight into the scalar remainder loop `while i < len { ... }`, which for `len=1` reads exactly
element `0` — the one element Swift's `&array[index]` conversion actually guarantees. The call
site still uses the identical unsafe idiom (`l2Sq(&x[base], &C[0], dsub)` at the same lines
1220-1345); it simply doesn't overflow *at this particular `dsub`*. This is dsub-dependent luck,
not a structural difference, and is direct empirical confirmation of the prior doc's warning that
this idiom's safety depends on values the compiler gives no guarantee about. Every other test in
this repo that exercises `.minibatch` with `dsub > 1` (i.e. essentially every realistic
configuration, since `d` is typically ≥ 64 and `m` ≤ `d/2`) is exposed.

---

## Q2 — Is this repo-wide, or confined to PQTrain.swift?

```
grep -rn '&[A-Za-z_][A-Za-z0-9_]*\[' Sources/ --include="*.swift"   → 30 raw hits, all in PQTrain.swift
find . -name "*.swift" -not -path "./.build/*" -not -path "./.git/*" | xargs grep -l '&[A-Za-z_]...\['
    → only ./Sources/VectorIndex/Kernels/PQTrain.swift
```

**The `&arr[index]`-to-multi-element-read idiom is confined to `PQTrain.swift`.** No other Swift
file in the repo (checked across all of `Sources/` and `Tests/`, not just `VectorIndex`) contains
this pattern. Of the 30 raw grep hits, 8 are comments or plain (non-`&`) array-copy declarations
(`var xc = xChunks[c]  // var required for &xc[index] syntax`), leaving **22 real call-site
lines** that pass an unsafe pointer into `l2Sq`. Each line passes 2 or 3 such pointers (the
`subtract:` overload takes three), so the total count of individual unsafe-pointer *arguments* is
higher than 22, but 22 is the count of vulnerable call *expressions*.

### Full call-site table

`l2Sq` reads `len` (i.e. `dsub = d/m`, a runtime-configured value, typically 8–128+) contiguous
`Float`s from **each** `UnsafePointer` argument. Every row below is therefore unsafe whenever
`dsub > 1`, which is the overwhelmingly common case.

| Cluster (function) | Lines | Call pattern | Elements read (per arg) | ASan status | Verdict |
|---|---|---|---|---|---|
| `pq_train_streaming_f32` (inline, empty-cluster repair) | 576, 578 | `l2Sq(&xc[base], &Cj[k*dsub], dsub, subtract: &coarse[gbase])` | dsub × 3 args | not independently triggered by required tests | **unsafe** |
| `pq_train_streaming_f32` (inline, empty-cluster repair) | 582, 584 | `l2Sq(&xc[base], &Cj[k*dsub], dsub)` | dsub × 2 args | not independently triggered | **unsafe** |
| `pq_train_streaming_f32` (inline, final distortion eval) | 644, 646 | `l2Sq(&xc[base], &Cj[k*dsub], dsub)` | dsub × 2 args | not independently triggered | **unsafe** |
| `minibatchKMeansSubspace` (assignment loop) | 1220, 1222 | `l2Sq(&x[base], &C[k*dsub], dsub, subtract: &coarse[gbase])` | dsub × 3 args | coarse-branch not hit by `testMiniBatchPQTraining` (no coarse arg) | **unsafe** |
| `minibatchKMeansSubspace` (assignment loop) | 1227, 1229 | `l2Sq(&x[base], &C[k*dsub], dsub)` | dsub × 2 args | **ASan-CONFIRMED** (`testMiniBatchPQTraining`, this doc, Q1) | **CONFIRMED UNSAFE** |
| `minibatchKMeansSubspace` (empty repair) | 1284, 1286 | subtract variant | dsub × 3 args | not independently triggered | **unsafe** |
| `minibatchKMeansSubspace` (empty repair) | 1290, 1292 | plain variant | dsub × 2 args | not independently triggered | **unsafe** |
| `minibatchKMeansSubspace` (final distortion eval) | 1343, 1345 | plain variant | dsub × 2 args | not independently triggered | **unsafe** |
| `minibatchKMeansSubspaceChunk` (assignment loop) | 1405, 1407 | subtract variant | dsub × 3 args | not independently triggered (no coarse in test configs used) | **unsafe** |
| `minibatchKMeansSubspaceChunk` (assignment loop) | 1412, 1414 | plain variant | dsub × 2 args | **ASan-CONFIRMED** (original investigation doc + reconfirmed here) | **CONFIRMED UNSAFE** |
| `streamingKMeansppSeed` (seed-distance fold) | 1481, 1518 | `l2Sq(&xc[base], &outC[k*dsub], dsub)` | dsub × 2 args | only reached when `totalN ≤ 4×ks` (not hit by `testStreamingPQTraining`, n=5000 > 1024); not independently triggered | **unsafe** |

All 22 lines: **1 file** (`PQTrain.swift`).

### Confirmed-safe sites (for contrast, all in the same file)

- `kmeansppSeedSubspaceDense` (`:928-977`) — uses `xDense.withUnsafeBufferPointer` /
  `outC.withUnsafeMutableBufferPointer` + `xbase + i*dsub` offset arithmetic throughout. Exercised
  cleanly under ASan by every one of the 6 tests run in this investigation (it's the seeding path
  taken whenever `n`/`totalN` exceeds `4×ks`, which all of `testBasicPQTraining`,
  `testMiniBatchPQTraining`, `testResidualPQ`, `testStreamingPQTraining` hit). **Confirmed safe,
  both by code inspection and by ASan.**
- `kmeansppSeedSubspace` (`:814-926`, the non-"Dense" sibling, used only when `n ≤ 4×ks`) — also
  uses the same `withUnsafeBufferPointer` pattern (`:836-925`); its one non-pointer array access
  (`:825-831`) is a plain read, not `&x[...]`. **Safe by code inspection**, but not directly
  ASan-exercised by any of the 6 required/run tests in this investigation (none of them have
  `n ≤ 4×ks`) — flagged as inspection-only verification, not test-verified, for completeness.
- `lloydKMeansSubspace` (`:981-1160`, both the assignment loop `:1026-1076` and the `.split`
  empty-repair branch `:1113-1129`) — uses the same safe pattern. **Confirmed safe by ASan**
  (3 clean passing runs above).

### Related-but-distinct finding: `HNSWIndex.swift:rebuildInvNormsIfNeededForCosine`

Per the task brief, checked the reviewer-flagged function at `Sources/VectorIndex/HNSWIndex.swift:1163-1185`:

```swift
func rebuildInvNormsIfNeededForCosine() -> UnsafePointer<Float>? {
    ...
    return invNormsCache!.withUnsafeBufferPointer { $0.baseAddress }
}
```

This **is a real pointer-safety defect, but it is a different mechanism than the `l2Sq` bug**, not
the same defect class:

- The `l2Sq` bug: `&array[index]` inout-to-pointer conversion is guaranteed valid for exactly one
  element for the duration of one call; `l2Sq` then reads `len` elements past it. The unsafe
  window is "one call, N-element read."
- The HNSW bug: `withUnsafeBufferPointer { $0.baseAddress }` returns a pointer whose validity is
  *documented by Apple* to be scoped to the closure's execution — extracting `baseAddress` and
  returning it from the closure, then using it after `withUnsafeBufferPointer` has returned (at
  all 7 call sites: `:206, :289, :448, :640, :726, :909, :960`, each of which passes the escaped
  pointer into a traversal/selection kernel that reads it across an entire search/insert
  operation, i.e. many elements over an extended lifetime) violates that contract. In practice it
  tends to "work" because `invNormsCache` is a class-stored property (`self.invNormsCache`) whose
  backing buffer stays allocated as long as the property isn't reassigned or COW-triggered between
  the call and the pointer's last use — but this is an implementation detail, not a guarantee, and
  is exactly the kind of assumption ASan exists to catch.

**This was not run under ASan in this investigation** — it requires HNSW-suite tests, which were
out of the Q1 target list and the stated test-selection constraints (single-method/suite filters
on `PQTrainTests`), and doing so responsibly would need its own reproduction plan (an ASan run
across the HNSW test matrix, or a targeted new test that stresses concurrent search + mutation).
**Flagged as a separate Phase 4 (or later) investigation item, not folded into the `l2Sq`
remediation**, since the fix pattern is also different (return an index-based accessor, or use
`ManagedBuffer`/`ContiguousArray` with an explicit long-lived pointer, rather than
`withUnsafeBufferPointer` + escaped `baseAddress`).

---

## Q3 — Did Phase 3's Task 9 make this worse?

**Short answer: No. The defect fully predates Task 9. Task 9 neither introduced nor amplified it.**

### Evidence 1 — the commit diff itself

`git show 40c02bb -- Sources/VectorIndex/Kernels/PQTrain.swift` shows Task 9 rewrote **only the
bodies** of both `l2Sq` overloads (scalar loop → dual-SIMD4-accumulator with an 8-wide unroll).
The commit message states explicitly: *"Both overloads keep their exact private signatures and
callers untouched"* — confirmed by diff: zero changes to any call site. The pre-Task-9 scalar
`l2Sq` (`git show 40c02bb^:...`) is:

```swift
@inline(__always) private func l2Sq(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float {
    var acc: Float = 0
    for i in 0..<len {
        let d = a[i] - b[i]
        acc += d * d
    }
    return acc
}
```

This reads the exact same `len` contiguous elements from each pointer as the current version —
just in a different order/grouping (one at a time vs. 8-wide SIMD batches). The vulnerable
contract ("reads `len` elements from a pointer only guaranteed valid for 1") is unchanged.

### Evidence 2 — the commit message's own admission

Commit `40c02bb`'s message states: *"testStreamingPQTraining showed run-to-run distortion
divergence (1e16 / an assertion failure / 1e25) across reruns of the identical SIMD binary against
fully deterministic seeded inputs. **This predates this change** — the same failure signature is
documented in commit 225bf37 (2026-07-25), observed against the original scalar l2Sq."* Commit
`225bf37` ("test(pqtrain): seed testStreamingPQTraining data", 2026-07-25, six days before Task 9)
documents *"this test failing once with distortion=0.0 and passing on rerun (unseeded input
data)"* — process-nondeterministic distortion values against the pre-Task-9 scalar code, six days
before the SIMD rewrite landed.

### Evidence 3 — direct empirical A/B test (this investigation)

To settle this conclusively rather than rely on commit-message narrative, I temporarily replaced
**only** the two `l2Sq` function bodies in the current HEAD file with the pre-Task-9 scalar bodies
(everything else — call sites, later Phase-3 comments/fixes, API — held constant at HEAD), rebuilt
under ASan, and re-ran both faulting tests:

```
git diff --stat  →  1 file changed, 8 insertions(+), 28 deletions(-)   (isolated to the two l2Sq bodies)
```

`testStreamingPQTraining` against the **scalar** `l2Sq`:
```
==74980==ERROR: AddressSanitizer: stack-buffer-overflow ...
READ of size 4 at 0x00016f8113a4
    #0 l2Sq(_:_:_:) PQTrain.swift:758
    #1 minibatchKMeansSubspaceChunk(...) PQTrain.swift:1392
    #2 pq_train_streaming_f32(...) PQTrain.swift:538
    #3 PQTrainTests.testStreamingPQTraining() PQTrainTests.swift:662
==74997==ABORTING
```

`testMiniBatchPQTraining` against the **scalar** `l2Sq`:
```
==75028==ERROR: AddressSanitizer: stack-buffer-overflow ...
READ of size 4 at 0x00016f6b3cb4
    #0 l2Sq(_:_:_:) PQTrain.swift:758
    #1 minibatchKMeansSubspace(...) PQTrain.swift:1207
    #2 runOneSubspace #1 @Sendable (_:) in pq_train_f32(...) PQTrain.swift:277
    #3 pq_train_f32(...) PQTrain.swift:329
    #4 PQTrainTests.testMiniBatchPQTraining() PQTrainTests.swift:84
```

**Both tests fault identically** (same call chain, same functions, same crash class — line
numbers shifted only because the scalar bodies are 20 lines shorter) whether the SIMD or scalar
`l2Sq` is in place. This is direct, reproducible proof that the SIMD rewrite is not a prerequisite
for the crash.

The temporary edit was reverted immediately after capturing this evidence:
```
git checkout -- Sources/VectorIndex/Kernels/PQTrain.swift
git diff --stat  → (empty)
git status       → only the two untracked files, nothing else
```
(confirmed again at the end of this document, see final `git status`).

### Did the 8-wide unroll change how far past the pointer the code reads?

**No, not in terms of maximum extent.** Both the old scalar loop (`for i in 0..<len`) and the new
SIMD loop (`while i < l8 { ...8 elements... }; while i < len { ...remainder... }`) read every
index from `0` to `len-1` over the course of a full call — that's the function's job in both
versions. The difference is *access grouping*: the old code touches offsets one at a time in
strict order; the new code touches up to 8 offsets (32 bytes) per unrolled iteration before the
next bounds-irrelevant check. Since the A/B test shows **both versions fault at the same call site
with the same overflow classification**, this grouping difference does not change whether or where
the fundamental defect manifests under ASan.

What *can* differ between the two versions, as a side effect of any code change (not something
specific to SIMD): the compiled function's local-variable/register layout shifts, which shifts
*which* adjacent stack bytes get reinterpreted as the corrupted `Float` in **non-ASan** builds —
i.e., it can change the *specific garbage value* (and therefore the specific blown-up distortion
number) observed run-to-run, but not the presence of the defect. This is the same "diagnostic
prints mask the bug by perturbing stack layout" phenomenon the original investigation documented,
generalized: Task 9's rewrite is just another layout perturbation, unrelated to sanitizer-visible
correctness.

**Conclusion for Q3: the defect predates Task 9 (confirmed via commit diff, commit message, and
direct empirical A/B ASan test). Task 9 did not amplify it — it neither changed the maximum
out-of-bounds read extent nor changed whether ASan flags the call sites.**

---

## Q4 — What is the correct fix pattern, and what does it cost?

### Confirmed-safe reference implementations

- `kmeansppSeedSubspaceDense` (`:928-977`) — confirmed safe by ASan (Q1/Q2 above; every test run
  in this investigation exercises it cleanly).
- `lloydKMeansSubspace`'s hot loop (`:1026-1076`, plus its `.split` empty-repair branch
  `:1113-1129`) — confirmed safe by ASan.

Both use the identical pattern: `array.withUnsafeBufferPointer { buf in ... let base = buf.baseAddress! ... }` (or `withUnsafeMutableBufferPointer` for the output array), then pass
`base + offset` pointers into `l2Sq` instead of `&array[offset]`. For the `coarse`-branch, an
additional nested `coarseArr.withUnsafeBufferPointer { gbuf in ... }` wraps just that branch — this
exact 2-level nesting is already used successfully 3 times in the file (`kmeansppSeedSubspace`,
`kmeansppSeedSubspaceDense`'s sibling, `lloydKMeansSubspace`).

### Per-cluster mechanical-difficulty assessment

| Cluster | Files touched | Difficulty | Why |
|---|---|---|---|
| `minibatchKMeansSubspace` (`:1164-1361`) | `PQTrain.swift` | **Trivial-to-moderate, fully mechanical** | Nearly line-for-line structural twin of the already-safe `lloydKMeansSubspace` in the same file — same parameter shapes (`x: [Float]`, `C: inout [Float]`, optional `coarse`/`assign`), same three internal blocks (assignment loop, empty repair, final distortion eval). Copy the proven pattern from its sibling function. |
| `minibatchKMeansSubspaceChunk` (`:1363-1440`) | `PQTrain.swift` | **Trivial, fully mechanical** | Single block (assignment loop only, no internal empty-repair/final-eval). `xChunk: [Float]` and `C: inout [Float]` are plain value/inout params — wrap both in one `withUnsafeBufferPointer`/`withUnsafeMutableBufferPointer` pair, same as above. |
| `pq_train_streaming_f32` final distortion eval (`:637-654`, inline) | `PQTrain.swift` | **Mechanical** | Single chunk `xc = xChunks[c]` per outer-loop iteration (the `for (c, nc) in nChunks.enumerated()` loop) — wrap `xc.withUnsafeBufferPointer` once per chunk-iteration, straightforward. |
| `pq_train_streaming_f32` empty-cluster repair (`:546-605`, inline) | `PQTrain.swift` | **NOT mechanical — needs judgment** | The `for t in 0..<evalN` loop samples a **random global index per iteration**, mapped via `mapIndex(g)` to a `(chunk, local)` pair that **varies unpredictably across iterations of the same loop** (`pts[t] = (c, i)`, `var xc = xChunks[c]` re-fetched per `t`). A single hoisted `withUnsafeBufferPointer` can't cover this because different iterations need different chunks' buffers live simultaneously in principle (though not actually concurrently). Correct fix needs either (a) opening/closing a `xChunks[c].withUnsafeBufferPointer` scope *inside* the per-`t` loop body (mechanical but more verbose, and must not naively cache `baseAddress` across chunk boundaries), or (b) precomputing an array of `UnsafeBufferPointer`s up front — which itself must avoid repeating the HNSW-style escaped-pointer mistake (Q2). Flag for careful review, not a copy-paste job. |
| `streamingKMeansppSeed` (`:1444-1523`) | `PQTrain.swift` | **Moderate** | Also per-chunk (`for (c, nc) in nChunks.enumerated()`), so chunk-scoping is simpler than the empty-repair case above (one chunk per outer iteration, not a random global sample) — mechanical on that front. But `outC` (the `inout` output array) is both read (`&outC[0]`, `&outC[k*dsub]` as `l2Sq` args) *and* written via plain subscript (`outC[k*dsub+u] = ...`) within the *same* k-loop scope that would need wrapping. Requires care to keep the mutable-buffer-pointer scope correctly bounding all of `outC`'s uses in that loop without holding it open across the chunk-iteration where a *different* array (`dminChunks`) also gets mutated — doable, but not a blind copy-paste. |

**Summary: 2 of 5 clusters (the two ASan-confirmed ones) are straightforward mechanical fixes with
a proven in-file template to copy. 1 cluster is mechanical. 2 clusters require actual engineering
judgment** — most notably the streaming empty-cluster-repair block, whose per-iteration
cross-chunk indexing pattern doesn't fit the simple "wrap once, use everywhere" template used
elsewhere in this file.

---

## Q5 — Is there a cheap global safety net?

**No ASan CI job exists today, and — independently of that — the entire `PQTrainTests` suite does
not run in CI at all**, for two separate reasons:

1. `.github/workflows/ci.yml`'s `build` step is `swift build -v --build-tests` (no `--sanitize=`
   flag) and its `test` step is `swift test -v --parallel --skip-build --filter '${{
   steps.select.outputs.filter }}'` (also no `--sanitize=` flag). `grep -rn "sanitize\|ASan\|asan"
   .github` → 0 hits. No sanitizer of any kind runs in CI.
2. The `test` job's matrix has 4 groups (`core`, `hnsw`, `ivf`, `kernels`), each with an explicit
   regex filter listing test classes. `grep -c "PQTrainTests" .github/workflows/ci.yml` → **0**.
   `PQTrainTests` is not named in any of the four filter regexes, so it never matches and never
   runs — independent of and in addition to the fact that `PQTrainTests.setUpWithError()` also
   explicitly does `throw XCTSkip(...)` when `CI_SKIP_PQTRAIN=1` is set (which the `kernels` group
   step does set, though moot since the filter wouldn't have matched anyway).

Only `ThreadSanitizer` is mentioned anywhere in the repo, and only as a manually-run, non-CI
recommendation in two migration docs (`docs/migration-docs/SWIFT6_COMPLIANCE_PLAN.md:387-390`,
`docs/migration-docs/S2_implementation_summary.md:339`). AddressSanitizer is not mentioned in any
workflow, script, or doc outside the investigation docs.

**What it would take to add one**: a 5th CI job (or an extra step in the existing `build`/`test`
jobs) running `swift build --sanitize=address` + `swift test --sanitize=address --filter
PQTrainTests` (plus ideally the `kernels`/`hnsw` groups once the HNSW finding in Q2 is resolved).
ASan roughly doubles build time and meaningfully slows test execution (this investigation's runs
ranged ~2.5s–58s per single-method filter; a full-suite ASan run would be substantially slower
than the existing non-ASan `kernels`/`core` groups) — likely needs its own timeout budget and
possibly its own scheduled/nightly job rather than blocking every PR, given `ci.yml`'s existing
30-minute per-job timeout. This is a cheap, high-value addition given this bug class is
**completely invisible without it** — every non-ASan `swift test` run of `testMiniBatchPQTraining`
or `testStreamingPQTraining` passes today, silently.

---

## Proposed Phase 4 task breakdown

Ordered by severity/urgency; each item sized as one review unit.

1. **[P0] Fix `minibatchKMeansSubspaceChunk` assignment loop** (`PQTrain.swift:1391-1420`,
   specifically the unsafe calls at `:1405,1407,1412,1414`). ASan-confirmed crash (original
   investigation + this doc). Mechanical — copy `lloydKMeansSubspace`'s
   `withUnsafeBufferPointer`/offset-arithmetic pattern. **Verify**:
   `swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address` must pass clean
   (currently faults).

2. **[P0] Fix `minibatchKMeansSubspace`** (`PQTrain.swift:1164-1361`; unsafe calls at
   `:1220,1222,1227,1229,1284,1286,1290,1292,1343,1345`). ASan-confirmed crash (this doc, Q1).
   Mechanical — same template, three blocks in one function. **Verify**:
   `swift test --filter PQTrainTests/testMiniBatchPQTraining --sanitize=address` must pass clean
   (currently faults); also re-run `testLargeScaleTraining`, `testWarmStartMinibatchImprovesOnePass`,
   `testWarmStartDeterministic` under ASan since they weren't independently confirmed to hit this
   exact function's dsub>1 path in this investigation.

3. **[P1] Fix `pq_train_streaming_f32`'s inline final-distortion-eval block**
   (`PQTrain.swift:637-654`, unsafe calls at `:644,646`). Not yet independently ASan-triggered but
   structurally identical to items 1/2. Mechanical (single chunk per outer iteration). **Verify**:
   `swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address` clean, plus a
   dedicated regression test whose `stats.distortion` is asserted against a data-scale-appropriate
   upper bound (not just `> 0`) as the original investigation doc recommended.

4. **[P1] Fix `pq_train_streaming_f32`'s inline empty-cluster-repair block**
   (`PQTrain.swift:546-605`, unsafe calls at `:576,578,582,584`). **Not mechanical** — the
   per-iteration random cross-chunk indexing (see Q4) needs a design decision, not a copy-paste.
   Size this as its own review unit separate from item 3 even though both are in the same
   function, because the risk/complexity profile differs. **Verify**: ASan run of
   `testStreamingPQTraining` with a config that reliably produces empty clusters (this doc's Q1
   run showed `emptyPass` activity in the debug logs for related tests), plus manual code review
   of pointer-lifetime correctness given the loop restructuring required.

5. **[P1] Fix `streamingKMeansppSeed`** (`PQTrain.swift:1444-1523`, unsafe calls at `:1481,1518`).
   Moderate difficulty (see Q4 — `outC` is both read and written in the same scope that needs
   wrapping). Only reachable when `totalN ≤ 4×ks`, so needs a small dedicated test (existing tests
   don't hit this branch with n=5000/ks=256; would need e.g. n≤1024,ks=256 or similar). **Verify**:
   new/adjusted test with `totalN ≤ 4×ks` run under ASan.

6. **[P1] Add ASan to CI** (`.github/workflows/ci.yml`). Currently zero sanitizer coverage and
   `PQTrainTests` is entirely excluded from the existing test matrix (Q5). At minimum: (a) add
   `PQTrainTests` to the `kernels` filter group (remove/reconsider the `CI_SKIP_PQTRAIN` skip), and
   (b) add a scheduled or opt-in ASan job covering at least `PQTrainTests`, ideally `kernels` and
   `hnsw` too pending item 7. **Verify**: the new job fails on the current (pre-fix) `PQTrain.swift`
   and passes after items 1-5 land — i.e. confirm the gate actually catches this class of bug
   before trusting it going forward.

7. **[P2] Investigate `HNSWIndex.swift:rebuildInvNormsIfNeededForCosine`** (`:1163-1185`, 7 call
   sites at `:206,289,448,640,726,909,960`) as its own effort. Different defect mechanism
   (escaped closure-scoped pointer, not the `l2Sq` single-element idiom — see Q2), different test
   surface (HNSW suite, not PQTrain), not yet ASan-verified either way. Needs its own
   reproduction/verification plan before a fix pattern can be chosen.

8. **[P2, hardening, not correctness-blocking] Add the missing per-element `isFinite` guard on
   `minibatchKMeansSubspaceChunk`'s centroid write** (`PQTrain.swift:1433`,
   `C[baseC + u] = Float(oldW * oldVal + newW * batchMean)`) — the sibling in-memory function
   `minibatchKMeansSubspace` has this guard (`:1255`, `v.isFinite ? v : 0`) plus a final full-array
   sanitization sweep (`:1330-1333`); the streaming per-chunk function has neither. Per the
   original investigation, this gap is **not implicated in the confirmed bug** (corrupted `l2Sq`
   output never reaches the centroid write path in any traced case), but it's a real asymmetry
   worth closing defensively while touching this code anyway.

---

## Shippability verdict on `gifton/perf-0.2.0-phase3`

**Safe to merge as-is; does not need to wait on Phase 4.**

Justification, directly from Q3's evidence:

- The defect (unsafe `&array[index]` pointers fed to `l2Sq`'s multi-element read) exists
  identically in the pre-Phase-3 scalar `l2Sq` and the current Phase-3 SIMD `l2Sq`. The direct A/B
  ASan test in this document proves both versions fault at the exact same call sites with the
  exact same crash signature, for both the streaming (`testStreamingPQTraining`) and non-chunked
  (`testMiniBatchPQTraining`) minibatch paths.
- Task 9 (`40c02bb`) changed *only* the internal arithmetic of `l2Sq`; it did not touch any call
  site, and the maximum out-of-bounds read extent is unchanged (both versions read the same `len`
  elements over a full call — see Q3's "how far past" analysis).
- Merging Phase 3 therefore does not make this defect newly present, newly reachable, or newly
  severe relative to whatever branch Phase 3 was cut from. It was already there.
- The confirmed blast radius (per the original investigation's evidence #4, and re-confirmed here
  for `minibatchKMeansSubspace`) is a **READ** overflow that only ever corrupts ephemeral
  comparison values (`bestD`/`dk`/`best`/`dval`) used for cluster-assignment decisions and the
  `stats.distortion` statistic — never the values written back into the shipped codebook array in
  any traced path. It is not a WRITE overflow, so it does not corrupt unrelated program state; in
  a non-ASan (ordinary release) build the practical failure mode is "misleading distortion
  statistic and possibly a suboptimal cluster assignment," not a crash or heap corruption.
- The bug is additionally invisible to CI today regardless of Phase 3 (Q5: no ASan job, and
  `PQTrainTests` isn't even in CI's test matrix) — so merging Phase 3 does not change what CI does
  or does not catch here either.

This verdict is specifically about *not regressing* — it is not a certification that the
underlying UB is acceptable to leave unfixed. Given ASan-confirmed undefined behavior in a shipped
numerical kernel, Phase 4 (items 1-6 above) should be prioritized promptly and treated as
P0/P1 work in its own right; it just doesn't need to gate Phase 3's merge.

---

## `git status` confirmation (end of investigation)

```
On branch gifton/perf-0.2.0-phase3
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.bench/post-phase3/
	docs/superpowers/2026-08-08-pq-streaming-distortion-blowup-investigation.md
	docs/superpowers/2026-08-09-pointer-escape-scope-assessment.md

nothing added to commit but untracked files present (use "git add" to track)
```

No tracked file was left modified. The one temporary edit made during Q3 (swapping `l2Sq`'s
bodies for the pre-Task-9 scalar versions, to run the A/B ASan test) was reverted with
`git checkout -- Sources/VectorIndex/Kernels/PQTrain.swift` immediately after capturing evidence,
confirmed via `git diff --stat` returning empty before proceeding to Q4/Q5.
