# Assessment: `HNSWIndex.rebuildInvNormsIfNeededForCosine` pointer escape

Status: **investigation complete, no fix applied**. No production code was changed as a result of
this work; no temporary instrumentation was added (see "Instrumentation" below for why, and for
the one-line proof).

Repo: `/Users/goftin/dev/gsuite/VSK/VectorIndex`, branch `gifton/fix-0.2.0-phase4-pointer-safety`,
BASE `257f903`.

This is Task 7 of `docs/superpowers/plans/2026-08-09-vectorindex-phase4-pointer-safety.md`,
following directly on `docs/superpowers/2026-08-09-pointer-escape-scope-assessment.md`'s "Related
but distinct finding" section, which flagged
`Sources/VectorIndex/HNSWIndex.swift:rebuildInvNormsIfNeededForCosine` (`:1163-1185`, 7 call sites)
as a **different defect mechanism** from the `l2Sq` single-element idiom that Phase 4's other tasks
fix — an escaped closure-scoped pointer, not yet ASan-verified either way, and a Phase 3 reviewer's
unverified claim that it is "safe in practice (no mutation in scope)". This document verifies that
claim with evidence (ASan + static analysis) rather than accepting it on faith.

**Line-number note**: the brief cites call sites at `:206, 289, 448, 640, 726, 909, 960`. At the
current BASE commit these are unchanged — verified by direct `grep` against
`Sources/VectorIndex/HNSWIndex.swift` (see the per-call-site table below). No drift this time.

---

## Summary

- **All 7 call sites are SAFE under the current codebase**, but by two different mechanisms, and
  the safety of 5 of them rests on an invariant that is **true today but not compiler-enforced** at
  two of those five sites (`:206`, `:640`) — see "Overall verdict" for the precise breakdown.
- **7 ASan suite runs, 0 faults, 26 tests, all clean**: the 5 suites named in the brief (all of
  which construct every `HNSWIndex` with `.euclidean`, confirmed by grep — see below) plus 2
  supplementary suites I added specifically because they exercise `.cosine`, which is the metric
  gate that must be true for `rebuildInvNormsIfNeededForCosine`'s body (and therefore the escape
  itself) to ever execute at all.
- **Critical caveat on the 5 required suites**: none of them ever call the escaping code path.
  `rebuildInvNormsIfNeededForCosine` early-returns `nil` at `:1164` unless `metric == .cosine`, and
  `HNSWTests`, `HNSWRecallTests`, `HNSWBatchAndErrorsTests`, `HNSWAlignmentTest`, and
  `HNSWDeterminismTests` all construct their `HNSWIndex` instances with `.euclidean` (or the
  default, which is also `.euclidean`). Their clean ASan results are evidence about the
  *euclidean/dot-product* paths through these same functions, not about the escape under
  investigation — a false-negative risk directly analogous to `testLargeScaleTraining`'s
  `dsub=1` false negative in the 2026-08-09 doc's Q1. I ran 2 additional suites
  (`HNSWKNNGraphTests`, `HNSWWALTests`) that do construct cosine indices — these are the runs that
  actually exercise the pointer escape, and they are also clean.
- Static analysis explains *why* they're clean, not just *that* they're clean: `HNSWIndex` is a
  Swift `actor`, and every one of the 7 call sites' "escape → last dereference" windows is either
  (a) inside a function with no `async` keyword at all (so the Swift compiler statically forbids a
  suspension point from existing in that window — `greedySearchLayer`, `searchLayer`,
  `pruneNeighbors`), or (b) inside an `async` function with no `await` between the call and the
  last use (current-code-inspection fact, not compiler-enforced — `search`,
  `internalInsertAtLevel`), or (c) the escape is immediately collapsed into an owned copy within a
  single Swift expression, before any `await` is reachable (`batchSearch`'s context builder,
  `makeKNNBuildContext`). Since actor reentrancy can only occur at `await`, none of these windows
  can be interrupted by a concurrent mutation of `invNormsCache`.
- The two `async`-but-currently-await-free sites (`:206`, `:640`) are the ones I'd flag as
  **structurally sound today but brittle to future edits** — nothing stops a future maintainer from
  adding an `await` (e.g. a cooperative-cancellation yield) inside `search()` or
  `internalInsertAtLevel()`'s neighbor-selection loop without the compiler raising any error,
  silently reintroducing a real reentrancy window.
- The file already contains the fix pattern, half-applied: `search()` wraps its CSR-cache pointer
  escapes (`csrOffsetsCache`/`csrNeighborsCache`, same `withUnsafeBufferPointer { $0.baseAddress }`
  idiom) in `withExtendedLifetime(csrOffsetsCache) { withExtendedLifetime(csrNeighborsCache) { ... } }`
  at `:199` — but never applies the equivalent `withExtendedLifetime(invNormsCache)` around the
  `invNormsPtr` escape one call deeper in the very same closure nest. This is a real asymmetry, not
  a design choice, and it points directly at the cheapest fix (see "Recommended fix direction").

---

## ASan results

Method: `swift test -v --sanitize=address --filter "^VectorIndexTests\.<Class>"`, one class per
invocation, foreground, no backgrounding, `timeout: 600000` per the task's binding constraints.
First invocation (`HNSWTests`) rebuilt the ASan binary (13.22s — warm from earlier Phase 4 tasks,
per the brief); all subsequent invocations used `--skip-build` since nothing under `Sources/` or
`Tests/` changed between runs.

| # | Suite | Required by brief? | Metric(s) exercised | Tests | Result | Wall time |
|---|---|---|---|---|---|---|
| 1 | `HNSWTests` | yes | `.euclidean` only (grep-confirmed, `:6`) | 2 | **CLEAN** | 0.384s |
| 2 | `HNSWRecallTests` | yes | `.euclidean` only (`:66`) | 1 | **CLEAN** | 1.525s |
| 3 | `HNSWBatchAndErrorsTests` | yes | default ctor → `.euclidean` (no explicit `metric:` anywhere in file) | 3 | **CLEAN** | 0.004s |
| 4 | `HNSWAlignmentTest` | yes | `.euclidean` only (`:14,60,91`); includes `testStructureConsistencyAfterCompaction`, which exercises `compact()` — but under euclidean, so `invNormsCache` is never touched | 3 | **CLEAN** | 0.040s |
| 5 | `HNSWDeterminismTests` | yes (brief: "exercises the build path heavily") | `.euclidean` only (`:43`) | 1 | **CLEAN** | 4.202s |
| 6 | `HNSWKNNGraphTests` | **no — supplementary, added by me** | `.cosine` (`:138`, `testCosineMatchesEuclideanOnUnitVectors`); also exercises `compact()` under euclidean/cosine mix via `testDeletionsCompaction`, and `testDeterminism` | 9 | **CLEAN** | 88.168s |
| 7 | `HNSWWALTests` | **no — supplementary, added by me** | `.cosine` throughout (13 occurrences, `:124` onward) — insert, batch-insert, checkpoint, and WAL-replay paths all under cosine | 7 | **CLEAN** | 1.941s |

**Total: 26 tests across 7 suites, 0 AddressSanitizer faults, 0 test failures.** No trace to report
— `grep -n "AddressSanitizer"` against every captured log returned zero matches in all 7 runs.

**Why I ran 2 suites beyond the brief's list**: the task's stated goal is to determine "whether
ASan observes anything at all" for the escape. Suites 1–5, exactly as named in the brief, cannot
answer that question on their own — they never take the `metric == .cosine` branch that gates
`rebuildInvNormsIfNeededForCosine`'s body (see `:1164`), so the pointer that escapes
`withUnsafeBufferPointer` is never created, let alone dereferenced, in any of those 10
required-suite test cases. `grep -rln "HNSWIndex" Tests/VectorIndexTests/ | xargs grep -l
"\.cosine"` turned up `TypedOverloadsTests`, `HNSWWALTests`, `HNSWKNNGraphTests`, and
`HNSWTypedInsertHintTests` as the cosine-exercising suites in the repo. I ran the two with the
heaviest and most structurally relevant cosine coverage (`HNSWKNNGraphTests` directly exercises
call site `:448`'s `makeKNNBuildContext`, plus `compact()` interaction; `HNSWWALTests` exercises
the insert/replay path that hits call sites `:640`, `:726`, `:909`, `:960`) as supplementary,
single-suite, foreground invocations under the same binding constraints (no filter alternation,
`timeout: 600000`, no backgrounding). I did not run `TypedOverloadsTests` or
`HNSWTypedInsertHintTests` — see "Open questions" for why that's a gap I'm flagging rather than
closing myself, given the brief's explicit suite list.

---

## Per-call-site analysis

All 7 sites call `rebuildInvNormsIfNeededForCosine()` (`HNSWIndex.swift:1163-1185`), whose last line
is the escape itself:

```swift
return invNormsCache!.withUnsafeBufferPointer { $0.baseAddress }
```

`invNormsCache: [Float]?` (`:557`) is the sole backing array for every site — there is no other
array in play. The question at each site is only: what is the window between this return and the
pointer's last dereference, and can anything run in that window that mutates or reallocates
`invNormsCache`?

| Site | Enclosing function | `async`? | Escaped into | Last use | Window contains `await`? | Backing-array mutation path in window | Verdict | Protecting invariant |
|---|---|---|---|---|---|---|---|---|
| `:206` | `search(query:k:filter:qInvNorm:)` | yes | 6-level-deep nested `withUnsafeBufferPointer`/`withExtendedLifetime` closure (`:196-238`) | Same closure, `HNSWTraversal.traverse(...)` call at `:207-216`, single synchronous kernel call | **No** — grep of the full function body (`:168-239`) shows zero `await` | None found — `internalRemove`/`connect`/`appendInvNormIncremental` etc. cannot run concurrently with `self`'s own in-flight synchronous call (see below) | **SAFE** | `HNSWIndex` is a Swift `actor`; actor-isolated code runs to completion between suspension points (SE-0306). `search()`'s entire body from `rebuildInvNormsIfNeededForCosine()`'s call to `HNSWTraversal.traverse`'s return is `await`-free, so no other actor job (insert, remove, compact, another search) can interleave and mutate `invNormsCache` during this window — current-code-inspection fact, not compiler-enforced (the function *is* `async`, so a future `await` is legal Swift and would silently reopen the window). |
| `:289` | `batchSearch(queries:k:filter:)`, `BatchSearchContext` initializer | yes | Not a closure — collapsed inline: `rebuildInvNormsIfNeededForCosine().map { Array(UnsafeBufferPointer(start: $0, count: N)) }` | `Array.init(UnsafeBufferPointer:)`'s copy loop, same expression | **No** — this is a single Swift expression; the enclosing `BatchSearchContext(...)` initializer call has no `await` in any of its argument expressions | N/A — window is bounded to one expression, no suspension is even syntactically reachable inside it | **SAFE** | The raw pointer never survives past the expression that produced it. By the time `withThrowingTaskGroup` (`:313`, the function's first `await`) is reached, `ctx.invNorms` is an independently owned `[Float]?` copy; `Self.performSingleSearch` (the concurrent TaskGroup worker, `:330-408`) only ever touches that copy via a proper non-escaping `invNorms.withUnsafeBufferPointer` at `:359`, never the actor's raw pointer. |
| `:448` | `makeKNNBuildContext(k:ef:)` | **no** (plain sync func returning a tuple) | Same collapsed-inline pattern as `:289` | Same | N/A — function has no `async` keyword, cannot suspend | N/A | **SAFE** | Same "collapsed into an owned copy within one expression" argument as `:289`, reinforced by the enclosing function being syntactically non-suspendable. Confirmed by `HNSWKNNGraph.swift:61`: `makeKNNBuildContext` is called with no `await`, and its `ctx.invNorms` is handed to `buildKNNRows` TaskGroup workers (`HNSWKNNGraph.swift:71-80`) as an already-copied value, exactly mirroring `:289`'s pattern. |
| `:640` | `internalInsertAtLevel(...)`, neighbor-selection block | yes | Nested `vector.withUnsafeBufferPointer { vbp in vectorStorage.withUnsafeBufferPointer { xbbp in ... } }` (`:635-657`) | Same closure, `hnsw_select_neighbors_f32_swift(..., optionalInvNorms: invPtr, ...)` at `:643-651`, single synchronous C-interop call | **No** — the function's only `await` (`try await remove(id:)`, `:601`) is in the "replace existing id" branch, which completes (or is skipped) *before* `newIndex`/`vectorStorage.append`/the neighbor loop begins; zero `await` from `:603` through the end of the function | None — `appendInvNormIncremental` (`:613`) runs *before* this call, not during it; its mutation and this call's fresh read are strictly sequenced, never overlapping | **SAFE** | Same actor-reentrancy argument as `:206`. Same brittleness caveat: `internalInsertAtLevel` is `async`; nothing prevents a future `await` inside the `for l in stride(...)` loop. |
| `:726` | `pruneNeighbors(of:level:)` | **no** | `vectorStorage.withUnsafeBufferPointer { xbbp in ... }` (`:730-745`) | Same closure, `hnsw_select_neighbors_f32_swift(...)` call | N/A — function has no `async` keyword | N/A | **SAFE** | Compiler-enforced: no suspension point can exist inside a non-`async` function. `pruneNeighbors` is called synchronously from `connect()` (`:694,701`), itself called synchronously from `internalInsertAtLevel`'s already-`await`-free region. |
| `:909` | `greedySearchLayer(_:enter:level:)` | **no** | Obtained *before* any closure (`:908-909`), then read repeatedly across a `while changed` loop (`:930-951`) via `scoreBatch(...)` (an ordinary actor-isolated method call, not a nested closure — see file comment `:824-829` on why) | Last `scoreBatch` call inside the loop, `:941-942` | N/A — function has no `async` keyword | N/A | **SAFE** | Compiler-enforced, same as `:726`. This is the strongest of the 7: `greedySearchLayer` is syntactically incapable of suspending, so the pointer's validity across the entire multi-iteration `while` loop is guaranteed regardless of how the rest of the codebase evolves, *as long as `greedySearchLayer` itself never becomes `async`*. |
| `:960` | `searchLayer(_:enter:ef:level:)` | **no** | Same pattern as `:909` — obtained before the outer closures (`:959-960`), read across a `while !heap.isEmpty` loop (`:1003-1027`) via repeated `scoreBatch` calls (`:996,1017`) | Last `scoreBatch` call, `:1017-1018` | N/A — no `async` keyword | N/A | **SAFE** | Compiler-enforced, identical to `:909`. `searchLayer`/`greedySearchLayer` are called from two places: `internalInsertAtLevel`'s neighbor-descent loop (`:620,627`, construction-time) **and** the public `AccelerableIndex.getCandidates`/`getBatchCandidates` query-time path (`:1452,1458`) — `search()` itself bypasses them and calls `HNSWTraversal.traverse` directly (`:207`), but `getCandidates` is a real, separate query-time caller. `getCandidates`'s own body (`:1423-1487`) is also fully `await`-free (grep-confirmed), consistent with the pattern, but this makes `searchLayer`/`greedySearchLayer` hotter than "construction-only" — both an insert-time and a query-time kernel — which raises the stakes on Task B's perf-tension caveat below. |

### The `withExtendedLifetime` asymmetry

`search()` (`:199`) already wraps its CSR-cache pointer escapes in
`withExtendedLifetime(csrOffsetsCache) { withExtendedLifetime(csrNeighborsCache) { ... } }` — the
same `arr.withUnsafeBufferPointer { Optional($0.baseAddress!) }`-then-store-`baseAddress` pattern
as `rebuildInvNormsIfNeededForCosine`, applied to `csrOffsetsCache`/`csrNeighborsCache` instead of
`invNormsCache`. `rebuildInvNormsIfNeededForCosine()` is called *inside* that same
`withExtendedLifetime` nest (`:206`) but its own backing array, `invNormsCache`, is never separately
protected by `withExtendedLifetime(invNormsCache)`. Given our actor-reentrancy analysis, this
doesn't currently matter (nothing can mutate `invNormsCache` in this window regardless), but it is
concrete evidence that whoever wrote `search()` was already aware of and applying the mitigation
pattern for this exact hazard class — just not consistently to all three caches in the same
function. This is the basis for the cheapest fix option below.

---

## Overall verdict

**7 / 7 call sites: SAFE.** Zero UNSAFE, zero UNPROVEN.

But "SAFE" decomposes into three distinct strength levels, worth keeping separate rather than
flattening into one bucket:

1. **Compiler-enforced safe** (`:726`, `:909`, `:960` — 3 sites): the enclosing functions
   (`pruneNeighbors`, `greedySearchLayer`, `searchLayer`) are not `async`. Swift's type system makes
   it a compile error to suspend inside them, so the "no reentrancy in this window" invariant cannot
   silently regress without the compiler also having to accept a new `async` keyword on these
   functions — a visible, reviewable change.
2. **Safe-by-construction, suspension-independent** (`:289`, `:448` — 2 sites): the raw pointer is
   converted to an owned copy within a single synchronous Swift expression, before any `await` in
   the enclosing function is even reachable. Safety here doesn't depend on the *absence* of
   `await` elsewhere in the function — it depends only on Swift's argument-evaluation-order
   guarantee, which is a language guarantee, not a code-inspection fact.
3. **Currently safe, code-inspection-only** (`:206`, `:640` — 2 sites): the enclosing functions
   (`search`, `internalInsertAtLevel`) are `async` and *could* legally contain an `await` between
   the escape and last use; today they don't, verified by direct `grep` of the full function bodies,
   but nothing enforces this going forward. This is the one category worth hardening even though
   nothing is broken today — see below.

Phase 3's reviewer's claim ("safe in practice, no mutation in scope") is **confirmed**, but the
reasoning given was incomplete: "no mutation in scope" is true, but *why* it's true is the
actor-isolation/suspension-point argument above, not merely an observation that nothing happened to
mutate the array in the code as currently written. That distinction matters because it's exactly
what separates category 3 (currently-true-by-luck-of-omission) from categories 1–2
(true-by-structural-guarantee) — the reviewer's phrasing doesn't distinguish them, and a future
maintainer reading only "safe in practice" has no signal that 2 of the 7 sites are one stray
`await` away from becoming genuinely unsafe.

**On the prompt's risk framing — (a) reallocation/mutation vs. (b) compiler exploiting the lifetime
rule:**
- **(a) is fully closed for all 7 sites today**, by the actor-isolation argument above (categories
  1–3) or by copy semantics (category 2).
- **(b) is not closed for any of the 5 raw-pointer-reuse sites** (`:206, 640, 726, 909, 960`) —
  Apple's documented contract ("valid only during the closure") is still being violated at the
  language level, even though nothing in the *current* Swift/Clang toolchain (confirmed: Swift
  6.3.3, `swiftlang-6.3.3.1.3 clang-2100.1.1.101`, this run) exploits it. ASan cannot detect (b) at
  all in a build where (a) doesn't independently trigger a real use-after-free/reallocation —
  ASan instruments actual heap operations (alloc/free/asan-poison boundaries), not language-level
  contract violations that the current allocator happens not to act on. **A clean ASan run here is
  evidence against (a), not evidence against (b).** This is the single most important limitation of
  this investigation's method, and it's why "not yet ASan-verified either way" in the original
  flag was the right caution even though the answer turned out to be "safe."

---

## Recommended fix direction (sized as tasks)

None of these are implemented in this investigation. Sizing assumes a reviewer familiar with this
file (consistent with the 2026-08-09 doc's sizing convention).

### Task A — [P1] Close risk (a) explicitly with `withExtendedLifetime`, matching the file's own existing pattern

Wrap each of the 7 call sites' escape-to-last-use window in `withExtendedLifetime(invNormsCache) {
... }`, exactly mirroring `search()`'s existing `withExtendedLifetime(csrOffsetsCache) { ... }` at
`:199`. For the 3 compiler-enforced-safe sites this is defense-in-depth (no observable behavior
change possible, since nothing can suspend there anyway); for `:206` and `:640` it converts the
"safe because no `await` happens to be here today" invariant into "safe because the specific array
snapshot is kept alive via an explicit extra strong reference, independent of whether an `await`
gets added later" — it does **not** fully close risk (a) forward-looking either (a future `await`
could still let a *different* Task mutate `self.invNormsCache` to point at fresh data while the old
snapshot stays validly-alive-but-stale — a correctness bug, not a memory-safety one, and arguably an
improvement over a crash), but it removes the "silent UAF" failure mode entirely, which is the one
ASan would have caught.

- **Blast radius**: `Sources/VectorIndex/HNSWIndex.swift` only. 7 mechanical edits (wrap existing
  code, no restructuring of the closures or the kernel calls themselves). No signature changes.
- **Perf tension**: negligible. `withExtendedLifetime(_:)` on an `Array` argument is one extra
  atomic retain/release per call site invocation (once per `search()` call, once per
  `internalInsertAtLevel()` layer-descent iteration, etc. — not per-neighbor, not per-comparison).
  This should not be measurable against Phase 3's search/build benchmarks; no new allocations, no
  copies.
- **Verify**: re-run this doc's 7 ASan suites (should remain clean); additionally verify via a
  targeted review that each `withExtendedLifetime` wraps the *entire* window through the true last
  dereference (getting the wrap boundary wrong is the main way this task could be done incorrectly).

### Task B — [P2] Close risk (b) fully: restructure so the pointer never leaves its closure

Change `rebuildInvNormsIfNeededForCosine()`'s call convention from "return an escaped pointer" to
"take a non-escaping closure", e.g. `func withInvNormsForCosine<R>(_ body: (UnsafePointer<Float>?)
throws -> R) rethrows -> R`, and move each call site's downstream kernel invocation inside that
closure. This is the textbook-correct fix — the pointer's lifetime is then lexically bounded by
`withUnsafeBufferPointer`'s own closure, satisfying Apple's documented contract exactly, closing
both (a) and (b) unconditionally (no longer contingent on actor-isolation reasoning at all).

- **Blast radius**: `Sources/VectorIndex/HNSWIndex.swift` only, but touches all 7 call sites'
  *structure*, not just wrapping:
  - `:289`, `:448` (`batchSearch`/`makeKNNBuildContext`): trivial — the `.map { Array(...) }` copy
    already happens synchronously; move it inside the new closure (`withInvNormsForCosine { ptr in
    ptr.map { Array(UnsafeBufferPointer(start: $0, count: N)) } }`). No structural change to
    anything downstream of that line.
  - `:206`, `:640`, `:726` (`search`, `internalInsertAtLevel`, `pruneNeighbors`): one additional
    closure-nesting level around an already-single synchronous kernel call each. Mechanical.
  - `:909`, `:960` (`greedySearchLayer`, `searchLayer`): **the nontrivial ones.** The pointer is
    currently obtained *before* the outer `query.withUnsafeBufferPointer`/
    `vectorStorage.withUnsafeBufferPointer` nest and reused across a multi-iteration `while` loop
    that calls `scoreBatch` (a plain method, not a closure — see the file's own comment at
    `:824-829` explaining that `scoreBatch` takes every pointer as an explicit parameter
    specifically to avoid "spurious capture diagnostics" from nested `withUnsafeBufferPointer`
    closures capturing an `UnsafePointer`-bearing optional under this package's StrictConcurrency
    setting). Restructuring to close risk (b) here means making
    `withInvNormsForCosine { invNormsPtr in ... }` the outermost wrapper around the whole function
    body (or at least around the `while` loop), which risks reintroducing exactly the capture
    diagnostics that comment describes needing to route around. `scoreBatch`'s own signature would
    not need to change (it already takes `invNormsPtr` as a plain parameter) — only the two call
    sites that *obtain* the pointer need restructuring — but this is real engineering risk, not a
    copy-paste job, and should be reviewed by whoever wrote that `:824-829` comment or with
    equivalent care.
- **Perf tension**: expected zero-cost (no new allocations or copies — purely a lexical-scoping
  change; Swift generally inlines non-escaping closures, and this file already nests 4-6 levels deep
  in `search()` without a documented perf regression). But `searchLayer`/`greedySearchLayer` are
  explicitly Phase 3's optimized hot path (per this task's brief) — and, per the per-call-site table
  above, they're not construction-only: they're also on the `AccelerableIndex.getCandidates`/
  `getBatchCandidates` query-time path. So **this option should not ship without re-running Phase
  3's benchmark baseline** (`.bench/post-phase3/`
  has `hnsw_search.json`/`knn_graph_*.json` — the right comparison points) rather than relying on
  "should be zero-cost" reasoning alone. This is the one place in this fix direction where the
  brief's "−12.4% build win" regression concern is a real, non-theoretical risk if the restructuring
  is done clumsily (e.g. if it forces a capture that turns a currently-stack-allocated closure
  context into a heap-allocated one under StrictConcurrency).
- **Verify**: same 7 ASan suites, plus the Phase 3 benchmark re-run above, plus a manual review
  focused specifically on `:909`/`:960`'s restructuring given the capture-diagnostic risk noted.

### Task C — [P3, hardening, not correctness-blocking] Extend CI's ASan job to cover HNSW

`.github/workflows/ci.yml`'s `asan` job (`:140-234`) explicitly excludes HNSW today — its own
comment (`:204-205`) says folding in HNSW/mmap/persistence suites would be doing so "with no prior
baseline, for the first time, unreviewed." This investigation *is* that baseline now (7 suites, 26
tests, 0 faults, including 2 that specifically exercise the cosine escape). Add
`HNSWTests, HNSWRecallTests, HNSWBatchAndErrorsTests, HNSWAlignmentTest, HNSWDeterminismTests,
HNSWKNNGraphTests, HNSWWALTests` to the job's `classes=(...)` array (`:211-228`) so future changes
to this code get continuous ASan coverage instead of relying on point-in-time manual investigations
like this one.

- **Blast radius**: `.github/workflows/ci.yml` only, one array literal. `HNSWKNNGraphTests` alone
  measured 88s in this run (dominated by `testCosineMatchesEuclideanOnUnitVectors` at 30s and
  `testRecallVsBruteForce` at 31s) — adds meaningfully to the nightly job's runtime but the job
  already budgets for a ~25-30 min total against a 45-min timeout (per its own comments), so this
  should fit without needing a timeout increase; worth confirming empirically on the actual CI
  runner rather than assuming local-machine timing transfers directly.
- **Perf tension**: none (nightly/on-demand job only, not on the push/PR path per its `if:`
  condition).

### Explicitly rejected direction

Making every call site copy `invNormsCache` to an owned `[Float]` before use (i.e. applying the
`:289`/`:448` pattern everywhere) was considered and rejected for `:206, 640, 726, 909, 960`. Those
5 sites are reached from the hot search/construction path — `searchLayer`/`greedySearchLayer` run
once per layer per insert, and `search()`'s traversal runs once per user query. Forcing an O(N) copy
of the entire inverse-norms array on every such call (as opposed to `:289`/`:448`'s
once-per-batch/once-per-graph-build amortization, where the copy is already unavoidable for
`Sendable` TaskGroup handoff regardless of this investigation) is exactly the kind of regression
risk the task brief flagged against Phase 3's −12.4% build-time win. Task A/B above both avoid this
by construction (no new copies introduced at any of the 5 hot-path sites).

---

## Open questions

1. **`TypedOverloadsTests` and `HNSWTypedInsertHintTests`** also construct cosine `HNSWIndex`
   instances (8 `.cosine` references each, per grep) and were not run under ASan in this
   investigation — I stayed within the brief's explicit 5-suite list plus 2 self-added supplements
   chosen for direct call-site relevance, rather than unilaterally expanding further. If a truly
   exhaustive cosine-path ASan sweep is wanted before closing this out, these two are the remaining
   gap.
2. **No ThreadSanitizer run.** This investigation's actor-isolation argument (no reentrancy without
   `await`) is a well-established Swift concurrency guarantee (SE-0306), not something I
   independently re-verified with TSan. If there's residual doubt about the actor-isolation
   reasoning itself (as opposed to the specific 7 call sites), a TSan pass would be the next
   escalation — the 2026-08-09 doc's Q5 notes TSan is already mentioned (manually-run, non-CI) in
   `docs/migration-docs/SWIFT6_COMPLIANCE_PLAN.md` and `S2_implementation_summary.md`.
3. **The CSR-cache escape (`csrOffsetsCache`/`csrNeighborsCache`, `:201-202`) uses the identical
   `withUnsafeBufferPointer { $0.baseAddress }`-then-store pattern**, just with `withExtendedLifetime`
   already applied. I did not audit whether that protection is itself complete (e.g. whether it's
   applied at every call site that reads `csrOffsetsCache`/`csrNeighborsCache`, not just `search()`)
   — out of scope for this task, which is specifically about `rebuildInvNormsIfNeededForCosine`'s 7
   sites, but a repo-wide grep for this idiom (à la the 2026-08-09 doc's Q2) would be a reasonable
   follow-up if Task A/B above are picked up, since the fix pattern would be identical.
4. **Task B's risk at `:909`/`:960`** (the StrictConcurrency capture-diagnostic concern) is a
   prediction based on the file's own comment at `:824-829`, not something I test-compiled. Whoever
   picks up Task B should treat that comment as a warning to prototype the restructuring early
   rather than assume it'll compile cleanly.

---

## Instrumentation

**None was added.** Per this phase's proven lesson (print-instrumentation can mask memory bugs by
perturbing stack/heap layout — see the 2026-08-09 doc's Q3 "diagnostic prints mask the bug" finding,
generalized), and because ASan plus static/actor-isolation analysis was sufficient to reach a
verdict without needing to perturb the code under test, I did not add, and therefore did not need to
revert, any temporary logging, assertions, or test code. Confirmed by the final `git status` below
showing zero modifications to any tracked file.

---

## `git status` confirmation (end of investigation, before this doc/commit)

```
On branch gifton/fix-0.2.0-phase4-pointer-safety
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.bench/post-phase3/

nothing added to commit but untracked files present (use "git add" to track)
```

`git diff --stat` returns empty. No tracked file was modified at any point during this
investigation.
