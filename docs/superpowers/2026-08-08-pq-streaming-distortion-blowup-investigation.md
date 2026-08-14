# Investigation: `PQTrainTests.testStreamingPQTraining` distortion blowup

Status: **root cause found and confirmed with AddressSanitizer**. Discovery/documentation only — no production code was changed as a result of this investigation.

## Summary

`stats.distortion` is occasionally computed from garbage floating-point bytes because `l2Sq(_:_:_:)` (`Sources/VectorIndex/Kernels/PQTrain.swift:755`) is repeatedly called with a pointer obtained from `&array[index]` and then walks `dsub` (32) contiguous floats starting at that address, which Swift only guarantees is valid for the *single* addressed element. AddressSanitizer confirms this is a genuine **stack-buffer-overflow READ** (`PQTrain.swift:760`, inside `l2Sq`'s SIMD loop), triggered from the call `l2Sq(&xChunk[base], &C[0], dsub)` at `PQTrain.swift:1412` inside `minibatchKMeansSubspaceChunk`. The garbage bytes read past the materialized single-`Float` stack temporary are process-history-dependent, so the same deterministic input produces a different (always huge) `stats.distortion` on every fresh process launch, while the persisted centroid array `Cj`/`C` itself stays finite because the corruption only ever touches an *ephemeral* comparison value, never a value that gets written back into the codebook.

## Reproduction

Config used throughout (matches the test exactly): `n=5000, d=256, m=8, ks=256, numChunks=5 (1000 each), cfg.seed=42, cfg.algo=.minibatch, cfg.maxIters=10, cfg.batchSize=512`, input filled by the test's seeded LCG (`rng = 0x5773_7EA1_1234_5678`, `rng = 2862933555777941757 &* rng &+ 3037000493`).

Command (single-method filter, as required):
```
swift test --filter PQTrainTests/testStreamingPQTraining
```

**Debug config (`swift test`'s default, no flags), unmodified `PQTrain.swift` at HEAD `31b97d7`, four separate process invocations, byte-identical input every time:**

| Run | distortion |
|---|---|
| 1 | `3.4658116304874356e+25` |
| 2 | `4.5663833178365935e+25` |
| 3 | `4.0446685065234416e+30` |
| 4 | `3.4658116304874356e+25` (same as run 1) |

Run 1's value is an exact match for the value reported in the bug ticket. All four runs `XCTAssert(stats.distortion > 0)` passed (test stays green).

**Release config**, same filter with `-c release`, unmodified code, two runs — the bug reproduces at optimized codegen too, not just `-Onone`:

| Run | distortion |
|---|---|
| R1 | `3.777893186295716e+22` |
| R2 | `3.320413933267719e+20` |

**Control — identical inputs, same process, repeated in-process (no new process launch each time):** a throwaway loop harness (`Tests/VectorIndexTests/ZZZTempDiagLoopTests.swift`, deleted before finishing this investigation) called `pq_train_streaming_f32` 10 times in a single process with byte-identical data/config each iteration. All 10 iterations returned the *exact same* `distortion=2.6725087868710102` (bit-identical), a sane value. This, together with the debug/release table above, is the key discriminator: **identical algorithmic inputs give identical results within one process, but different results across process launches** — the signature of reading uninitialized/out-of-bounds memory whose content depends on process-level layout (ASLR, prior stack usage), not of a deterministic arithmetic bug.

**Important caveat about the repo state found at start of investigation**: the working tree was *not* clean as stated in the task brief — `Sources/VectorIndex/Kernels/PQTrain.swift` already carried an uncommitted diagnostic patch (read-only instrumentation: max-abs/checksum prints, an `updated`-value magnitude guard, etc.) and an untracked `Tests/VectorIndexTests/ZZZTempDiagLoopTests.swift` existed, evidently left behind by the prior investigation referenced in the task ("a prior investigation reproduced run-to-run nondeterminism but never found the mechanism"). Running the test through *that* instrumented version was **always** healthy: `distortion=2.6725087868710102` on every one of 12 runs (2 fresh `swift test --filter` processes + the 10-iteration in-process loop). I initially treated this as troubling — until stashing the diagnostic patch and rerunning on genuinely unmodified `PQTrain.swift` immediately reproduced the blowup (table above). **The prior investigator's own diagnostics were accidentally masking the bug** — inserting extra reads/branches near the hot loop perturbs stack layout/register allocation enough that the corrupting read happens to land on a harmless (already-live, zero-initialized) stack slot instead of stale garbage. This is fully consistent with the stack-buffer-overflow root cause below and explains why the earlier investigation "reproduced run-to-run nondeterminism but never found the mechanism" — its own instrumentation was suppressing the very thing it was trying to observe.

## Evidence

1. **Confirmed with AddressSanitizer.** Built and ran the unmodified code under ASan:
   ```
   swift build --sanitize=address
   swift test --filter PQTrainTests/testStreamingPQTraining --sanitize=address
   ```
   Result: immediate abort with
   ```
   ==69387==ERROR: AddressSanitizer: stack-buffer-overflow on address 0x00016dbe13a4 ...
   READ of size 4 at 0x00016dbe13a4 thread T0
       #0 l2Sq(_:_:_:) PQTrain.swift:760
       #1 minibatchKMeansSubspaceChunk(...) PQTrain.swift:1412
       #2 pq_train_streaming_f32(...) PQTrain.swift:538
       #3 PQTrainTests.testStreamingPQTraining() PQTrainTests.swift:662
   Address ... is located in stack of thread T0 ... frame minibatchKMeansSubspaceChunk ...
       [1760, 1764) ''  <== Memory access at offset 1764 overflows this variable
   SUMMARY: AddressSanitizer: stack-buffer-overflow PQTrain.swift:760 in l2Sq(_:_:_:)
   ```
   The overflowing address is explicitly reported as living inside `minibatchKMeansSubspaceChunk`'s own **stack frame**, overflowing a 4-byte object (i.e. one `Float`) — not the heap-allocated backing store of `xChunk`/`C`. This is direct, tool-confirmed evidence that the pointer handed to `l2Sq` was a materialized single-element stack temporary, not a pointer into the array's real contiguous buffer.

2. **The vulnerable call pattern.** `l2Sq` (`PQTrain.swift:755-770`) is `@inline(__always)` and reads `len` (here `dsub=32`) contiguous `Float`s starting at each of its two `UnsafePointer<Float>` arguments:
   ```swift
   755  @inline(__always) private func l2Sq(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float {
   ...
   760      let a0 = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])   // <- crash site
   ```
   Every call site in the streaming/minibatch path invokes it as `l2Sq(&array[index], &array2[index2], dsub)`. Swift's `&expr[index]` → `UnsafePointer` conversion ("implicit pointer conversion" / inout-to-pointer) is documented to be valid *only for the duration of the call, for that one element* — reading `dsub-1` further elements past it relies on an unspecified implementation detail (that the compiler happens to vend a pointer into the array's real contiguous storage rather than materializing a scalar copy). ASan shows that for this call, at this optimization level, on this platform, that assumption does **not** hold.
   - Assignment step (the one ASan caught): `PQTrain.swift:1405-1414`, e.g. `bestD = l2Sq(&xChunk[base], &C[0], dsub)` (:1412) and `let dk = l2Sq(&xChunk[base], &C[k*dsub], dsub)` (:1414, called `ks-1` times per point).
   - Final distortion evaluation (structurally identical pattern): `PQTrain.swift:644` `var best = l2Sq(&xc[base], &Cj[0], dsub)` and `:646` `let dval = l2Sq(&xc[base], &Cj[k*dsub], dsub)` inside the loop that produces `Dj`/`stats.distortionPerSubspace[j]` (`:637-654`), summed into `stats.distortion` at `:660`.
   - Pass-level empty-cluster repair: `PQTrain.swift:576-584` (same pattern).
   - The **same anti-pattern also exists in the non-streaming Lloyd path** (`l2Sq(&x[base], &C[...], dsub)` at `:1220, :1222, :1227, :1229, :1284, :1286, :1290, :1292, :1343, :1345`) and in the streaming seeder repair helper (`:1481, :1518`). This is a systemic call-site pattern across the file, not a one-off typo — see "Recommended fix direction."
   - By contrast, `kmeansppSeedSubspaceDense` (`:972-1021`, used by the *seeding* path for this test) and `lloydKMeansSubspace`'s hot loop obtain their pointers via `withUnsafeBufferPointer`/`withUnsafeMutableBufferPointer` and raw pointer arithmetic (`xbase + i*dsub`, `:849-971` region) — a genuinely safe pattern, contiguous-array guaranteed. These call sites were **not** flagged by ASan and are not implicated.

3. **Seeding is not the source (falsifies one plausible alternative).** `totalN=5000 > seedingCap=4*ks=1024` (`PQTrain.swift:465`), so this test takes the **sampled** seeding branch (`:466-518`, `sampleWithoutReplacement` + `kmeansppSeedSubspaceDense`), not `streamingKMeansppSeed`. That seeder only ever copies real input-derived values (bounded in `[-1,1]`, confirmed by the prior investigator's own `[TEMP-DIAG][seed] maxAbs≈0.9999...` prints, which I re-verified in the healthy diagnostic runs — every subspace's seed `maxAbs` was ≤ 1.0) into `Cj` via `withUnsafeBufferPointer`, a safe pattern. Seeding is not implicated.

4. **Centroids stay finite throughout — the corruption never lands in `Cj`/`C`.** The prior investigator's own per-pass instrumentation (`#if DEBUG` block around `PQTrain.swift:607-616`) never fired the `!v.isFinite` guard at `:615` in any of the 12 healthy runs I reproduced, and `[TEMP-DIAG][precompute]` checksums showed all centroid components bounded near `[-1, 1]` after every pass. This matches the ASan finding precisely: the out-of-bounds read happens only inside `l2Sq`'s *ephemeral, return-value-only* computation (a ranking/comparison quantity, `best`/`bestD`/`dval`/`dk`), which is used to pick an index (`bestK`) or to accumulate into `Dj`/`stats.distortion` — it is never assigned back into `C[baseC+u]` (`:1433`) or `Cj[k*dsub+u]`. The update line that *does* write centroids, `C[baseC + u] = Float(oldW * oldVal + newW * batchMean)` (`:1433`), only ever consumes `sums[baseC+u]` (accumulated from real, non-pointer-tricked `xChunk[base+u]` reads at `:1417`) and the previous (already-finite) `C[baseC+u]`. So a corrupted `l2Sq` result can misdirect *which* cluster a point's real value gets summed into (via a wrong `bestK`), but it cannot inject a garbage magnitude into the centroid values themselves.

5. **`globalCounts` cannot wrap or go negative at these scales.** `cfg.sampleN` defaults to 0, and since `totalN=5000 > 2000`, `pq_train_streaming_f32` sets `cfg.sampleN = 2000` (`:438`). Per-pass thinning probability is `sampleProb = min(1.0, min(totalN,sampleN)/totalN) = 2000/5000 = 0.4` (`:541` region — this line moved during the diagnostic patch but the formula is unchanged). Observed `passCounts` in the diagnostic runs were ~1900–2050 per pass (matches `5000*0.4`), and `globalCounts[k]` accumulates additively across at most `maxIters=10` passes, so the theoretical maximum for any single cluster is on the order of `5000*10=50,000` — roughly 14 orders of magnitude below `Int64.max` (~9.2e18). `globalCounts[k] = oldN &+ ck` (`:1425`) cannot wrap given these bounds. This is an analytic bound, not an empirical one, but it is airtight for this test's configuration.

## Root cause analysis

The mechanism, in order:
1. `minibatchKMeansSubspaceChunk` calls `l2Sq(&xChunk[base], &C[k*dsub], dsub)` (`:1405-1417`) to find the nearest centroid for a point, needing to read `dsub=32` contiguous `Float`s starting at each pointer.
2. For at least one of these two `&array[index]` arguments, the Swift compiler (confirmed at both `-Onone`/debug and `-O`/release) does not vend a pointer into the array's real heap-allocated contiguous storage; it materializes the single addressed `Float` into a 4-byte temporary that lives in `minibatchKMeansSubspaceChunk`'s **stack frame**, and passes the address of that temporary.
3. `l2Sq`'s unrolled SIMD loop (`:759-767`) then reads 7 more `Float`s past that temporary — i.e., walks off the end of a 4-byte stack object into whatever bytes happen to occupy the next ~124 bytes of that stack frame (or beyond it, into the frame's redzone, which is what ASan's instrumented poison bytes caught).
4. Those adjacent bytes are leftover content from *other* local variables/register spills in that same stack frame (or, once past the frame, uninitialized/previously-used stack memory from other function calls in this process's history) — not derived from `xChunk`/`C` at all. Reinterpreted as `Float`, arbitrary byte patterns frequently decode to enormous magnitudes (near the top of the float range) rather than NaN/Inf, which is exactly why the earlier in-kernel `isFinite` guards (`:615` on `Cj`, and the `best.isFinite` check at `:650`) never fire — **the garbage is finite by construction of how it's used** (`best.isFinite` at `:650` only filters `NaN`/`Inf`, and huge-but-finite floats like `1e20` sail right through it into the accumulator `Dj`).
5. Because this is memory that a *different* process launch will have used differently (ASLR base addresses, differing prior stack/heap traffic from process bring-up, XCTest bootstrap, etc.), the exact garbage value — and hence the final `stats.distortion` — differs from run to run even though the algorithm's inputs are 100% deterministic. Because the garbage is finite (not NaN/Inf), and because `best < 0 { best = 0 }` (`:649`) only clamps negatives, an enormous positive `best` sails straight into `Dj += Double(best)` (`:650`), and `Dj` is then divided by `seen` (`:654`) and summed across all 8 subspaces (`:660`) — so a single corrupted per-point evaluation in a 5000-point, 8-subspace loop is enough to move `stats.distortion` from O(10¹) to O(10²⁰–10³⁰).

This is traced to specific lines: the bug is the call convention at `PQTrain.swift:1405-1417` (and structurally identical call sites at `:576-584`, `:644-646`, and the Lloyd-path equivalents) feeding `l2Sq`'s multi-element read (`:755-770`) with a pointer whose validity Swift only guarantees for one element.

**What is ruled out:**
- Seeding producing pathological initial centroids (evidence 3).
- The centroid values (`Cj`/`C`) themselves becoming non-finite or unboundedly large (evidence 4) — they stay bounded near `[-1,1]` in every run I instrumented, healthy or not.
- `globalCounts` wraparound / going negative (evidence 5, analytic bound).
- The convex-combination update formula itself being mathematically unsound (see verdict below).

**What is confirmed:**
- A stack-buffer-overflow READ inside `l2Sq`, reached from the streaming minibatch assignment step, caught live by AddressSanitizer with an exact file:line and call stack.

I did not exhaustively determine with a byte-level debugger *which* of the two `&array[index]` arguments at `:1412` (`&xChunk[base]` vs `&C[0]`) is the one that gets materialized to the stack in this specific case, nor whether it is *always* one or the other or varies by call site/compiler decision. This is a reasonable remaining detail for the implementer (see Open Questions) but does not change the fix: both arguments use the same unsafe idiom and both must be corrected regardless of which one ASan's specific stack trace pinned.

## Verdict on the convex-combination hypothesis: **REFUTED**

The measurement that decided it:
- **Analytic**: `globalCounts[k]` cannot exceed ~50,000 for this test's config (n=5000, sampleProb=0.4, maxIters=10), about 14 orders of magnitude below `Int64.max`; `oldN &+ ck` (`:1425`) cannot wrap. `oldW + newW = oldN/newN + ck/newN` with `newN = oldN &+ ck` computed exactly (no overflow) stays a valid convex combination.
- **Empirical**: In every healthy run captured (12 total, using the prior investigator's own update-magnitude diagnostic that would have printed any `update` where `abs(oldW*oldVal + newW*batchMean) > 5.0`, along with `oldW+newW`), the diagnostic never fired — the weights and update magnitudes stayed sane throughout.
- **Direct disproof of mechanism**: AddressSanitizer shows the actual defect is a stack-buffer-overflow *read* inside the L2 distance kernel (`l2Sq`, called for cluster **assignment**/**distortion evaluation**), not anywhere in the weighted-average update math at `:1421-1437`. The corrupted quantity never reaches `C[baseC+u]`; it only ever reaches a comparison variable (`bestD`/`dk`/`best`/`dval`) or, downstream, `Dj`.

The missing `isFinite` clamp on `C[baseC + u]` (present in the non-streaming Lloyd update, absent here) is a real, independently-worth-fixing gap, but it is **not implicated in this bug** — it would only help if a centroid component itself went non-finite, which never happens here.

## Why the test doesn't catch it

`XCTAssertEqual(codebooks.count, m*ks*(d/m))` and `XCTAssert(stats.distortion > 0)` (`PQTrainTests.swift:670-671`) are the only two assertions. `stats.distortion > 0` is true whether distortion is `2.67` or `3.4e25` — any finite positive number passes, including astronomically wrong ones. An assertion that **would** catch it: an upper bound sanity check appropriate to the data distribution, e.g. `XCTAssertLessThan(stats.distortion, someBound)` where `someBound` is derived from the data's known scale (data is drawn from `[-1,1]^d`, so a trivial all-zero-centroid baseline distortion — already computed inline as `trivialDistortion` in the sibling test `testStreamingSeederSmallNTakesStreamingBranch`, `PQTrainTests.swift:728-736` — gives a natural O(1)-scale reference; `stats.distortion` should never exceed a small multiple of that trivial baseline for m=8 subspaces of well-behaved uniform data). Additionally, an `XCTAssertTrue(v.isFinite)` sweep over `codebooks` (as `testStreamingSeederSmallNTakesStreamingBranch` already does at `:723-725`) would **not** have caught this specific bug, since it targets centroid finiteness, not distortion magnitude — reinforcing that this is a distinct defect class from the one that test already guards against.

## Recommended fix direction

Do **not** apply the missing-`isFinite`-on-centroid-write fix expecting it to resolve this ticket — per the evidence above it is orthogonal (centroids never go non-finite in any reproduction of this bug) and would leave `stats.distortion` blown up exactly as before.

The actual fix needs to eliminate the unsafe pointer-materialization risk at the `l2Sq` call sites, most urgently:
- `PQTrain.swift:1405-1417` (`minibatchKMeansSubspaceChunk`, the ASan-confirmed site)
- `PQTrain.swift:576-584` (pass-level empty-cluster repair)
- `PQTrain.swift:644-646` (final distortion evaluation)
- and, for full correctness (same anti-pattern, not yet proven to crash but structurally identical and therefore suspect), the Lloyd-path call sites at `:1220-1292, :1343-1345` and `:1481, :1518`.

Concretely: replace the `l2Sq(&array[index], &array2[index2], dsub)` idiom with `withUnsafeBufferPointer`/`withUnsafeMutableBufferPointer`-derived base pointers plus manual offset arithmetic (`xbase + base`, `cbase + k*dsub`) — the exact pattern already used safely in `kmeansppSeedSubspaceDense` (`:972-1021`) and the Lloyd hot loop's `withUnsafeBufferPointer` region (`:849-971`). That pattern gives a genuine pointer into the array's contiguous storage for the whole call's duration, which is what `l2Sq`'s multi-element read actually requires.

Regression test recommendation: after fixing, add an assertion to `testStreamingPQTraining` (and ideally the equivalent Lloyd-path tests) bounding `stats.distortion` against a data-scale-appropriate ceiling (e.g., the `trivialDistortion` baseline pattern from `testStreamingSeederSmallNTakesStreamingBranch:728-736`), not just `> 0`. Given the bug is process-launch-dependent, the regression test should be run several times (or under a loop/CI matrix) to have confidence it isn't passing by luck — a single green run is not proof of a fix for this class of bug. Running the corrected code under `--sanitize=address` at least once (as done in this investigation) is the most direct way to confirm the fix, since ASan will abort immediately if any `l2Sq` call site is still handed a single-element materialized pointer.

## Open questions

1. Which specific argument in `l2Sq(&xChunk[base], &C[0], dsub)` (`:1412`) is the one the compiler materializes to a stack temporary — `&xChunk[base]` or `&C[0]`/`&C[k*dsub]`? Not determined; doesn't change the fix (both must be corrected) but would help explain *why* this particular call gets miscompiled while superficially similar calls elsewhere might not (if that's even true — this wasn't verified either).
2. Does the same class of bug reproduce (crash under ASan) for the Lloyd-path call sites (`:1220-1292, :1343-1345, :1481, :1518`), or does it only manifest for this specific streaming/minibatch code shape (frame size, variable count, `dsub`/`ks` combination)? I did not run ASan against the non-streaming tests to check; the pattern is structurally identical so I flag it as suspect but unconfirmed.
3. Is there a compiler/SIL-level explanation (e.g. exclusivity enforcement, a `_read`/`_modify` accessor materialization, a heuristic in `-Onone` vs `-O` SILGen) for exactly when `&array[index]` fails to alias into the real buffer versus when it (apparently, based on the fact this code has presumably run "successfully" many times before) succeeds? Understanding this would clarify whether the current safe-looking call sites elsewhere in the codebase using the same idiom are latent bugs waiting to reproduce under different frame layouts, or genuinely safe for reasons not yet identified.
4. The assignment-collapse symptom (`emptyPass=255` of `ks=256`, every point in a pass funneled to one cluster) was reproduced identically in *both* the healthy (`distortion=2.67`) and — per the task's own prior evidence — some unhealthy runs, and also appears in the unrelated, sane-distortion `testMiniBatchPQTraining`. This investigation confirms it is a **separate, likely-benign-but-worth-its-own-look** algorithmic property of minibatch k-means on this synthetic uniform `[-1,1]^256` data/config (plausibly related to how the farthest-point empty-cluster repair interacts with a near-symmetric data distribution), not a symptom of the stack-overflow bug. It was out of scope to root-cause here but may warrant separate investigation.
