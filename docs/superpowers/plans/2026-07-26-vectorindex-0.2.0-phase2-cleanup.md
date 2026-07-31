# VectorIndex 0.2.0 — Plan 2: Phase 2 (Non-Breaking Cleanup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the 0.2.0 cleanup spec's Phase 2 — every non-breaking internal cleanup item (B1–B20 plus the Phase-1-orphaned dead code) — leaving the package smaller, telemetry consolidated-and-tested, and every touched-but-untested structure under new characterization tests.

**Architecture:** One branch `gifton/cleanup-0.2.0-phase2` off `main` @ `ee67895` (v0.1.7). Eighteen tasks ordered safest-first: dead targets/files, then the re-scoped CS2RNG and telemetry work, then kernel/no-op deletions and consolidations, then the mmap/data-structure and index-level simplifications (tests-first wherever current coverage is zero), then wrap-up. Each task is one reviewable commit group ending with a green build and its named test filters.

**Tech Stack:** Swift 6 / SwiftPM, XCTest (`@testable import VectorIndex`), Accelerate. Tests run via `swift test --filter <Name>`.

**Spec:** `docs/superpowers/specs/2026-06-22-vectorindex-0.2.0-cleanup-design.md` §6 Phase 2 + Appendix A (B11–B20). Re-scopes verified 2026-07-26 against `main` @ `ee67895` (post-Phase-1; the spec's line numbers predate Phase 1 — this plan's numbers are current). Carried-forward items from `docs/superpowers/2026-07-25-phase1-completion-notes.md` are folded in.

**Verified premise corrections this plan builds on (do not "fix" a task back to the spec's wording):**
1. `CS2RNG` is NOT import-free: three test files (674 lines) import it — B2 ports/triages them first (Task 2).
2. The `VINDEX_TELEM`-gated telemetry singleton has never compiled under Swift 6 (12 errors). USER DECISION 2026-07-26: consolidate on the six working push-callback recorders + the dedup pull API; delete the dead singleton (Task 3).
3. All of `MIPSTransform.swift`'s public API is dead — but public removal is Phase 4; Task 5 deprecates and consolidates internals only.
4. `IDFilter`, `CandidateReservoir`, and dedup's `.sparsePaged` mode have zero existing tests — Tasks 13–14 are tests-first.
5. The two "byte-identical" scalar kernel families (B10) differ textually — Task 7 is a real refactor with parity coverage, not copy-delete.

## Global Constraints

- **Non-breaking, absolutely:** no `public` symbol may be removed or have its signature changed. Where a task's deletion target turns out `public`, mark it `@available(*, deprecated, message: "...")` instead and record it in Appendix P4 for the Phase-4 breaking pass. Access-level *raises* (fileprivate→internal) are fine; `public`→lower is not.
- **VectorCore pin untouched** (0.3.1, revision `b26909e…`): no `swift package update`, no `Package.swift` dependency edits beyond the target deletions specified in Tasks 1–2.
- **Every task ends green:** `swift build` clean + the task's named test filters passing, run in the FOREGROUND with explicit generous timeouts (≤600000 ms per command; split filters that could exceed it). Never end a working turn while a command is in flight; never background test runs.
- **Determinism holds:** fixed-seed HNSW/IVF construction must stay byte-identical wherever a task touches index code — the named suites (incl. `RegressionA1_TraversalLifetimeTests`) are the gate.
- **Tests-first where coverage is zero:** a refactor of untested code lands only after a characterization test of current behavior is green (Tasks 13, 14, and marked sub-steps elsewhere).
- **Commit style:** conventional-commit subjects as given per task; end every commit message with the executing assistant's standard Co-Authored-By trailer (do not copy a hardcoded model name from older plans).
- **No pushes** unless the controlling session says so; local commits only. Never `git add` any `.superpowers/` path.
- **Benchmarks are out of scope:** Phase 3 owns perf measurement. Tasks here marked "perf/simplify" are behavior-neutral refactors gated by tests, not benchmarks (see `.bench/baseline-0.1.6/README.md` for the Phase-3 gate note).

## File map (files touched, by task)

- Task 1: `Package.swift`, `Sources/L2SqrMicrobench/` (delete), `Sources/VectorIndex/Kernels/PQTrain.swift.new` (delete), stray `.xcscheme`/`.tmp` files
- Task 2: `Package.swift`, `Sources/CS2RNG/` (delete), `Tests/VectorIndexTests/{S2EdgeCaseTests,RNGDeterminismTests,DTypeConversionTests}.swift`
- Task 3: `Sources/VectorIndex/.../Telemetry.swift`, `ExactRerank.swift`, `RangeQuery.swift`, `IDMap.swift`, `CandidateDedup.swift`, new `Tests/VectorIndexTests/TelemetryRecorderTests.swift`
- Task 4: `L2Sqr.swift`, `L2SqrKernel.swift`, `PQLUT.swift`, `ResidualKernel.swift`, `KMeansMiniBatch.swift`, `HNSWIndex.swift`, `HNSWTraversal.swift`, `ScoreBlock.swift`
- Task 5: `MIPSTransform.swift`
- Task 6: `Norms` + `L2SqrKernel.swift`, `Cosine.swift`, (`MIPSTransform.swift`)
- Task 7: `HNSWTraversal.swift`, `HNSWNeighborSelection.swift`
- Task 8: `HNSWNeighborSelection.swift` (+ its direct test file)
- Task 9: `ResidualKernel.swift`, `Tests/.../PQEncodeParity_SwiftOnly_Tests.swift` (or new u4 test file)
- Task 10: `VIndexContainerBuilder.swift`, `VIndexMmap.swift`, `Tests/.../VIndexMmapErrorTests.swift`
- Task 11: `VIndexMmap.swift`
- Task 12: `IDMap.swift`
- Task 13: `IDFilter.swift`, `CandidateReservoir.swift`, new characterization test files
- Task 14: `CandidateDedup.swift`, `IVFSelect.swift`, new characterization test file
- Task 15: `ExactRerank.swift`, `IVFPostADC.swift`, `HNSWIndex.swift`
- Task 16: `IVFIndex.swift`
- Task 17: `HNSWIndex.swift`, `HNSWWAL.swift`
- Task 18: `CHANGELOG.md`

---
## Tasks 1–3: Dead targets, CS2RNG, Telemetry consolidation

All three tasks are branched off `main @ ee67895`. Toolchain used to verify every command in
this fragment: `swift-driver 1.148.6 / Apple Swift 6.3.2`, target `arm64-apple-macosx26.0`,
run from `/Users/goftin/dev/gsuite/VSK/VectorIndex`.

---

### Task 1: Delete dead targets and stray files (B4 + inert files)

**Files:**
- Create: none
- Modify: `Package.swift` (remove the `exclude:` block from the `VectorIndex` target, lines
  46–50; remove the entire `L2SqrMicrobench` executable-target block, lines 56–66; remove the
  `exclude:` block from the `VectorIndexTests` target, lines 81–84)
- Delete: `Sources/L2SqrMicrobench/main.swift` (113 lines, whole target)
- Delete: `.swiftpm/xcode/xcshareddata/xcschemes/L2SqrMicrobench.xcscheme`
- Delete: `Sources/VectorIndex/Kernels/PQTrain.swift.new` (6-line scratch placeholder)
- Delete: `Tests/VectorIndexTests/PQTrainTests.swift.tmp` (0 bytes, empty scratch file)
- Test: none new; `Tests/VectorIndexTests/PQTrainTests.swift` (18 existing tests, untouched)
  must remain green as a regression check.

**Interfaces:** Consumes: none. Produces: none. No public symbol in the `VectorIndex` library
product is touched — `L2SqrMicrobench` is a separate `executableTarget`, not part of the
`VectorIndex` library target, and `PQTrain.swift.new` / `PQTrainTests.swift.tmp` are both
already excluded from compilation (they are inert text files as far as the compiler is
concerned, not source that could declare a public symbol).

- [ ] **Step 1: Sync `main` and create the phase-2 branch.**
  ```bash
  git checkout main
  git pull --ff-only
  git checkout -b gifton/cleanup-0.2.0-phase2
  ```
  Expected final line: `Switched to a new branch 'gifton/cleanup-0.2.0-phase2'`

- [ ] **Step 2: Delete the `L2SqrMicrobench` target's source and its stale Xcode scheme.**
  ```bash
  git rm -r Sources/L2SqrMicrobench
  git rm .swiftpm/xcode/xcshareddata/xcschemes/L2SqrMicrobench.xcscheme
  ```
  Expected output includes:
  ```
  rm 'Sources/L2SqrMicrobench/main.swift'
  rm '.swiftpm/xcode/xcshareddata/xcschemes/L2SqrMicrobench.xcscheme'
  ```

- [ ] **Step 3: Delete the two dead scratch files.**
  ```bash
  git rm Sources/VectorIndex/Kernels/PQTrain.swift.new
  git rm Tests/VectorIndexTests/PQTrainTests.swift.tmp
  ```
  Expected output:
  ```
  rm 'Sources/VectorIndex/Kernels/PQTrain.swift.new'
  rm 'Tests/VectorIndexTests/PQTrainTests.swift.tmp'
  ```
  (`PQTrain.swift.new` is a 6-line placeholder comment block, not real code — confirmed by
  reading it in full. `PQTrainTests.swift.tmp` is 0 bytes. Neither is compiled today because
  both are listed in `Package.swift`'s `exclude:` arrays, removed in the next step. The real,
  active implementation and test suite are the adjacent `Kernels/PQTrain.swift` (1490 lines)
  and `Tests/VectorIndexTests/PQTrainTests.swift` (18 tests), which are untouched by this task.)

- [ ] **Step 4: Edit `Package.swift` — drop the `VectorIndex` target's now-empty-purpose
  `exclude:` block.**
  In the `.target(name: "VectorIndex", ...)` block, remove the `exclude:` array (its only
  entry pointed at the file just deleted):
  ```diff
           .target(
               name: "VectorIndex",
               dependencies: [
                   "CAtomicsShim",
                   "CPQEncode",
                   "CS2RNG",
                   .product(name: "VectorCore", package: "VectorCore")
               ],
  -            exclude: [
  -                // Exclude scratch files relative to Sources/VectorIndex
  -                // Note: residual kernel docs moved to /docs; no longer under Sources.
  -                "Kernels/PQTrain.swift.new"
  -            ],
               swiftSettings: [
                   .enableExperimentalFeature("StrictConcurrency"),
                   .enableUpcomingFeature("ExistentialAny")
               ]
           ),
  ```
  (Leave the `"CS2RNG",` dependency line exactly as-is — that is Task 2's job, not this one.)

- [ ] **Step 5: Edit `Package.swift` — delete the entire `L2SqrMicrobench` target block.**
  ```diff
           ),
  -        .executableTarget(
  -            name: "L2SqrMicrobench",
  -            dependencies: [
  -                "VectorIndex",
  -                .product(name: "VectorCore", package: "VectorCore")
  -            ],
  -            swiftSettings: [
  -                .enableExperimentalFeature("StrictConcurrency"),
  -                .enableUpcomingFeature("ExistentialAny")
  -            ]
  -        ),
           .executableTarget(
               name: "VectorIndexBenchmarks",
  ```

- [ ] **Step 6: Edit `Package.swift` — drop the `VectorIndexTests` target's now-empty-purpose
  `exclude:` block.**
  ```diff
           .testTarget(
               name: "VectorIndexTests",
               dependencies: ["VectorIndex"],
  -            exclude: [
  -                // Exclude temporary scratch tests (relative to Tests/VectorIndexTests)
  -                "PQTrainTests.swift.tmp"
  -            ],
               swiftSettings: [ .enableExperimentalFeature("StrictConcurrency") ]
           ),
  ```

- [ ] **Step 7: Verify `Package.swift` has no remaining reference to the deleted target/files.**
  ```bash
  grep -n "L2SqrMicrobench\|PQTrain.swift.new\|PQTrainTests.swift.tmp" Package.swift
  ```
  Expected output: nothing (empty; the command exits 1 with zero matches — that is correct,
  not a failure).

- [ ] **Step 8: Clean build.**
  ```bash
  swift build 2>&1 | tail -5
  ```
  Expected final line: `Build complete!` with zero errors, and no mention of
  `L2SqrMicrobench` anywhere in the output (it no longer exists as a target to build).

- [ ] **Step 9: Confirm `PQTrainTests` is unaffected (18 tests, all passing).**
  ```bash
  swift test --filter PQTrainTests 2>&1 | tail -25
  ```
  Expected final lines:
  ```
  Test Suite 'PQTrainTests' passed at ...
       Executed 18 tests, with 0 failures (0 unexpected) in ...
  Test Suite 'All tests' passed at ...
       Executed 18 tests, with 0 failures (0 unexpected) in ...
  ```

- [ ] **Step 10: Confirm the full suite still builds/links (no test filter) — sanity check
  that removing the `L2SqrMicrobench` executable target didn't orphan anything.**
  ```bash
  swift build --target VectorIndexBenchmarks 2>&1 | tail -5
  ```
  Expected final line: `Build complete!` (the remaining `VectorIndexBenchmarks` executable
  target depends only on `VectorIndex` + `VectorCore`, never on `L2SqrMicrobench`).

- [ ] **Step 11: Commit.**
  ```bash
  git add Package.swift Sources/L2SqrMicrobench Sources/VectorIndex/Kernels/PQTrain.swift.new \
    Tests/VectorIndexTests/PQTrainTests.swift.tmp \
    .swiftpm/xcode/xcshareddata/xcschemes/L2SqrMicrobench.xcscheme \
    docs/superpowers/plans/2026-07-26-vectorindex-0.2.0-phase2-cleanup.md
  git commit -m "$(cat <<'EOF'
  chore(cleanup): remove dead L2SqrMicrobench target and scratch files

  Also versions this phase's execution plan (docs/superpowers/plans/).

  L2SqrMicrobench was an assertion-free dev timing harness with no test-target
  or CI dependency (Package.swift, docs only). PQTrain.swift.new and
  PQTrainTests.swift.tmp were stale scratch placeholders already excluded from
  compilation; the real implementation/tests (PQTrain.swift, PQTrainTests.swift)
  are untouched and still pass (18/18).

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  git status --short
  ```
  Expected: commit succeeds; `git status --short` prints nothing (clean tree).

---

### Task 2: Port CS2RNG-dependent tests, then delete the CS2RNG target (B2, re-scoped)

**Premise correction (verified live against `main @ ee67895`, not just the brief):** three test
files import `CS2RNG` and call its C symbols directly — `Tests/VectorIndexTests/S2EdgeCaseTests.swift`
(275 lines, 10 tests), `Tests/VectorIndexTests/RNGDeterminismTests.swift` (85 lines, 4 tests),
`Tests/VectorIndexTests/DTypeConversionTests.swift` (314 lines, 13 tests) — 27 tests / 674 lines
total. Deleting `Sources/CS2RNG` and its two `Package.swift` entries without touching these files
does not build (`no such module 'CS2RNG'`). Every one of the 27 tests was individually triaged
below into exactly one of three buckets. Two real, verified findings drove several bucket calls:

1. `quantizeSymmetric`/`quantizeAffine` in `Sources/VectorIndex/Kernels/S2_RNGDtype.swift:522-534`
   do `Int32(r)` on an unclamped rounded `Float` before calling `clampI8`. Verified live:
   `Int32(Float.infinity)` and `Int32(Float.nan)` **trap** (fatal error, kills the whole test
   process), unlike the C `quantize_i8_symmetric`, which saturates correctly on those inputs.
   This is a real, pre-existing bug in already-shipped Swift code — out of scope to fix in a
   dead-code-removal task, so any C-side test that exercises Inf/NaN quantization is deleted
   (not ported) rather than silently made to crash `swift test`.
2. `packNibblesU4` in the same file (`S2_RNGDtype.swift:636`) has
   `precondition(indices.count % 2 == 0, "Indices count must be even")` — the C
   `pack_nibbles_u4` supports odd-length input; the Swift port does not. Porting the one C test
   that exercises this (`testNibblePackOddLength`) would trap on the precondition. Also flagged,
   not ported, not fixed here.

**Files:**
- Modify: `Package.swift` (remove the `.target(name: "CS2RNG", ...)` block and the `"CS2RNG",`
  dependency line inside the `VectorIndex` target — post-Task-1 line numbers: target block at
  lines 31–37, dependency line at line 43)
- Modify: `Tests/VectorIndexTests/S2RNGDtypeTests.swift` (append 6 new test methods before the
  closing `}`, currently at line 539)
- Delete: `Tests/VectorIndexTests/S2EdgeCaseTests.swift`
- Delete: `Tests/VectorIndexTests/RNGDeterminismTests.swift`
- Delete: `Tests/VectorIndexTests/DTypeConversionTests.swift`
- Delete: `Sources/CS2RNG/` (`s_rng_dtype_helpers.c`, `include/s_rng_dtype_helpers.h`)
- Test: `Tests/VectorIndexTests/S2RNGDtypeTests.swift` (22 existing tests + 6 new = 28, all green)

**Interfaces:** Consumes: `Sources/VectorIndex/Kernels/S2_RNGDtype.swift`'s existing public API
(`S2Xoroshiro128`, `packNibblesU4`, `quantizeSymmetric`, `store64LE`/`load64LE`, `f32ToF16Batch`/
`f16ToF32Batch` — all already public, all unchanged by this task). Produces: none (test-only
change; no source symbol is added, removed, or changed in `Sources/`).

**Per-file triage (bucket → disposition):**

| File | Test | Bucket | Disposition |
|---|---|---|---|
| S2EdgeCaseTests | `testQuantizationWithNaN` | A: dies with target | Delete — only assertion is `XCTAssertTrue(true)` (vacuous); porting would also trap (`Int32(Float.nan)`, finding #1 above) |
| S2EdgeCaseTests | `testQuantizationWithInfinity` | A: dies with target | Delete — porting would trap (`Int32(Float.infinity)`, finding #1) |
| S2EdgeCaseTests | `testQuantizationWithLargeValues` | C: unique | **Ported** → `testInt8QuantizationExtremeSaturation` (values stay in Int32 range, verified safe) |
| S2EdgeCaseTests | `testQuantizationNearMaxFloat` | C: unique | **Ported** → merged into `testInt8QuantizationExtremeSaturation` |
| S2EdgeCaseTests | `testSaturationCountingAccuracy` | A: dies with target | Delete — only assertion is inside `#if S2_ENABLE_TELEMETRY`, which has no Swift `-D` equivalent anywhere in `Package.swift`; already dead code today (verified: `grep` of `swiftSettings` shows no such define) |
| S2EdgeCaseTests | `testXoroWithMultipleStreams` | B: redundant | Delete — covered by `S2RNGDtypeTests.testXoroshiro128StreamIndependence` |
| S2EdgeCaseTests | `testPhiloxDeterminism` | B: redundant | Delete — covered by `S2RNGDtypeTests.testPhilox4x32Reproducibility` |
| S2EdgeCaseTests | `testF16ConversionWithSpecialValues` | B (NaN/Inf/±0 parts) + C (overflow/underflow parts) | NaN/Inf/±0 → delete, covered by `testF16RoundTrip`/`testF16NaNPreservation`/`testF16SignOfZero`; overflow/underflow → **ported**, merged into `testF16BoundaryValues` |
| S2EdgeCaseTests | `testTelemetryResetThreadSafety` | A: dies with target | Delete — entire body is inside `#if S2_ENABLE_TELEMETRY`; function is already a no-op |
| S2EdgeCaseTests | `testUnalignedQuantization` | A: dies with target | Delete — tests C SIMD "scalar tail" handling; Swift port has one uniform scalar loop for every length, no separate path to test; round-trip property redundant with `testInt8QuantizationSymmetric` |
| RNGDeterminismTests | `testXoroReproducibility` | B: redundant | Delete — covered by `S2RNGDtypeTests.testXoroshiro128Reproducibility` (same property, different iteration count) |
| RNGDeterminismTests | `testStreamIndependence` | C: unique (stronger) | **Ported** → `testXoroshiro128StreamIndependenceQuantitative` (quantitative collision bound vs. existing test's boolean check) |
| RNGDeterminismTests | `testUniformityChiSquare` | C: unique | **Ported** → `testXoroshiro128UniformityChiSquare` (rigorous chi-square test; existing coverage is only a loose ±20% tolerance check) |
| RNGDeterminismTests | `testPhiloxReproducibility` | B: redundant | Delete — covered by `S2RNGDtypeTests.testPhilox4x32Reproducibility` |
| DTypeConversionTests | `testF32ToF16RoundTrip` | B (bulk) + C (2 boundary values) | Bulk → delete, covered by `testF16RoundTrip`; min-normal/subnormal values → **ported**, merged into `testF16BoundaryValues` |
| DTypeConversionTests | `testF16NaNPayloadPreservation` | B: redundant | Delete — covered by `S2RNGDtypeTests.testF16NaNPreservation` |
| DTypeConversionTests | `testF32ToF16Saturation` | C, but subsumed | Delete — same overflow-to-Inf property as the new `testF16BoundaryValues`; not double-ported |
| DTypeConversionTests | `testF32ToBF16RoundTrip` | B: redundant | Delete — `S2RNGDtypeTests.testBF16RoundTrip` already tests the identical value set incl. the same `3.38e38` case |
| DTypeConversionTests | `testSymmetricQuantizationRoundTrip` | B: redundant | Delete — covered by `testInt8QuantizationSymmetric` |
| DTypeConversionTests | `testSymmetricQuantizationSaturation` | B: redundant | Delete — `testInt8QuantizationSymmetric` already exercises the ±200 saturation boundary |
| DTypeConversionTests | `testAffineQuantizationWithZeroPoint` | B: redundant | Delete — same code path as `testInt8QuantizationAffine`, different parameters only |
| DTypeConversionTests | `testNibblePackUnpackRoundTrip` | B: redundant | Delete — identical value set to `testU4PackUnpack` |
| DTypeConversionTests | `testNibblePackingOrderLowFirst` | C: unique | **Ported** → `testNibblePackingOrderLowFirst` (asserts a hand-computed byte `0xA3`, independent of `packPair()`, unlike `testU4PackUnpack`'s tautological check) |
| DTypeConversionTests | `testNibblePackOddLength` | A: dies with target | Delete — porting would trap (`packNibblesU4`'s even-count precondition, finding #2 above) |
| DTypeConversionTests | `testLittleEndianRoundTrip` | B (16/32-bit) + C (64-bit) | 16/32-bit → delete, covered by `testEndianHelpers`; 64-bit → **ported** as `testEndianHelpers64Bit` (existing test never exercises `store64LE`/`load64LE`) |
| DTypeConversionTests | `testAlignedNEONPath` | A: dies with target | Delete — NEON-vs-scalar distinction doesn't exist in the Swift port; round-trip property redundant with `testF16RoundTrip` |
| DTypeConversionTests | `testUnalignedScalarPath` | A: dies with target | Delete — same reasoning as above |

Net: 8 tests die with the target (bucket A), 13 are redundant deletions (bucket B), 6 are ported
as 6 new test methods (bucket C, some merging 2–3 source tests each). All 6 ported tests below
were compiled and run against the real `S2_RNGDtype.swift` API on this branch's toolchain before
being written into this task — the numbers are observed, not estimated.

- [ ] **Step 1: Add the 6 ported tests to `S2RNGDtypeTests.swift` first, while `CS2RNG` still
  exists, so they can be verified in isolation before anything is deleted.**
  ```diff
       func testPadTo() throws {
           XCTAssertEqual(padTo(0, multiple: 16), 0)
           XCTAssertEqual(padTo(1, multiple: 16), 16)
           XCTAssertEqual(padTo(16, multiple: 16), 16)
           XCTAssertEqual(padTo(17, multiple: 16), 32)
       }
  +
  +    // MARK: - Ported from CS2RNG-dependent tests (deleted with the CS2RNG C target;
  +    // see Task 2 of the phase-2 cleanup plan for the full per-test triage).
  +
  +    func testXoroshiro128UniformityChiSquare() throws {
  +        // Ported from RNGDeterminismTests.testUniformityChiSquare.
  +        // Chi-square goodness-of-fit test: chi2 = sum((observed-expected)^2 / expected).
  +        var rng = S2Xoroshiro128(seed: 0xDEADBEEF, streamID: 0, taskID: 0)
  +
  +        let bins = 100
  +        var counts = [Int](repeating: 0, count: bins)
  +        let samples = 100_000
  +
  +        for _ in 0..<samples {
  +            let u = rng.nextUniform()
  +            let bin = min(Int(u * Float(bins)), bins - 1)
  +            counts[bin] += 1
  +        }
  +
  +        let expected = Double(samples) / Double(bins)
  +        let chiSquare = counts.reduce(0.0) { sum, count in
  +            let diff = Double(count) - expected
  +            return sum + (diff * diff) / expected
  +        }
  +
  +        // Critical value for 99 degrees of freedom at alpha=0.001 is ~149.
  +        // Verified observed value on this seed: ~99.06.
  +        XCTAssertLessThan(chiSquare, 149.0,
  +                           "Chi-square test failed: S2Xoroshiro128.nextUniform() is not uniform")
  +    }
  +
  +    func testXoroshiro128StreamIndependenceQuantitative() throws {
  +        // Ported from RNGDeterminismTests.testStreamIndependence. Strengthens
  +        // testXoroshiro128StreamIndependence's boolean check with a quantitative bound.
  +        let seed: UInt64 = 42
  +
  +        var rng0 = S2Xoroshiro128(seed: seed, streamID: 0, taskID: 0)
  +        var rng1 = S2Xoroshiro128(seed: seed, streamID: 1, taskID: 0)
  +
  +        let seq0 = (0..<1000).map { _ in rng0.nextUniform() }
  +        let seq1 = (0..<1000).map { _ in rng1.nextUniform() }
  +
  +        // Verified observed value on this seed: 0 collisions.
  +        let collisions = zip(seq0, seq1).filter { $0 == $1 }.count
  +        XCTAssertLessThan(collisions, 10,
  +                           "Independent streams should have <1% collision rate")
  +    }
  +
  +    func testInt8QuantizationExtremeSaturation() throws {
  +        // Ported from S2EdgeCaseTests.testQuantizationWithLargeValues +
  +        // testQuantizationNearMaxFloat. Values are chosen to stay within Int32 range
  +        // after division by scale, so quantizeSymmetric's internal `Int32(r)` conversion
  +        // does not trap (see this task's finding #1 for the Inf/NaN case that does trap
  +        // and was deliberately NOT ported).
  +        let largeValues: [Float] = [
  +            16_777_216.0,   // 2^24
  +            33_554_432.0,   // 2^25
  +            1.0e10,
  +            -1.0e10
  +        ]
  +        var largeQuantized = [Int8](repeating: 0, count: largeValues.count)
  +        largeValues.withUnsafeBufferPointer { src in
  +            largeQuantized.withUnsafeMutableBufferPointer { dst in
  +                quantizeSymmetric(x: src, scale: 127.0, y: dst)
  +            }
  +        }
  +        // Verified observed output: [127, 127, 127, -128]
  +        XCTAssertEqual(largeQuantized, [127, 127, 127, -128],
  +                       "Values far beyond scale*127 must saturate to int8 bounds")
  +
  +        let maxVal = Float.greatestFiniteMagnitude
  +        let nearMaxValues: [Float] = [maxVal, -maxVal, maxVal * 0.5, -maxVal * 0.5]
  +        let nearMaxScale = maxVal / 127.0
  +        var nearMaxQuantized = [Int8](repeating: 0, count: nearMaxValues.count)
  +        nearMaxValues.withUnsafeBufferPointer { src in
  +            nearMaxQuantized.withUnsafeMutableBufferPointer { dst in
  +                quantizeSymmetric(x: src, scale: nearMaxScale, y: dst)
  +            }
  +        }
  +        // Verified observed output: [127, -127, 63, -63]
  +        XCTAssertEqual(abs(nearMaxQuantized[0]), 127, "Max float should saturate to +-127")
  +        XCTAssertEqual(abs(nearMaxQuantized[1]), 127, "-Max float should saturate to +-127")
  +        XCTAssertLessThanOrEqual(abs(Int(nearMaxQuantized[2]) - 64), 1,
  +                                  "0.5*max should quantize to ~64")
  +        XCTAssertLessThanOrEqual(abs(Int(nearMaxQuantized[3]) + 64), 1,
  +                                  "-0.5*max should quantize to ~-64")
  +    }
  +
  +    func testNibblePackingOrderLowFirst() throws {
  +        // Ported from DTypeConversionTests.testNibblePackingOrderLowFirst. Asserts a
  +        // hand-computed byte value rather than round-tripping through packPair(), so it
  +        // independently documents the "low nibble = first code" contract (testU4PackUnpack
  +        // only checks packed[i] == packPair(...), which is tautological w.r.t. packPair
  +        // itself).
  +        let input: [UInt8] = [0x3, 0xA]  // codes 3 and 10
  +        var packed = [UInt8](repeating: 0, count: 1)
  +
  +        input.withUnsafeBufferPointer { src in
  +            packed.withUnsafeMutableBufferPointer { dst in
  +                packNibblesU4(indices: src, packed: dst)
  +            }
  +        }
  +
  +        XCTAssertEqual(packed[0], 0xA3,
  +                       "Low nibble=0x3 (first code), high nibble=0xA (second code)")
  +    }
  +
  +    func testEndianHelpers64Bit() throws {
  +        // Ported from DTypeConversionTests.testLittleEndianRoundTrip's 64-bit case.
  +        // testEndianHelpers only covers the 16/32-bit helpers.
  +        var buffer = ContiguousArray<UInt64>(repeating: 0, count: 1)
  +
  +        buffer.withUnsafeMutableBytes { ptr in
  +            store64LE(ptr.baseAddress!, 0xFEDCBA9876543210)
  +        }
  +
  +        let loaded64 = buffer.withUnsafeBytes { ptr in
  +            load64LE(ptr.baseAddress!)
  +        }
  +
  +        XCTAssertEqual(loaded64, 0xFEDCBA9876543210)
  +    }
  +
  +    func testF16BoundaryValues() throws {
  +        // Merged from S2EdgeCaseTests.testF16ConversionWithSpecialValues (overflow/
  +        // underflow) and DTypeConversionTests.testF32ToF16RoundTrip (min-normal/subnormal
  +        // boundary values). Verified live: Float16(Float.greatestFiniteMagnitude) saturates
  +        // to +Inf (no trap) -- narrowing float-to-float conversions saturate per IEEE 754,
  +        // unlike the float-to-Int32 trap in testInt8QuantizationExtremeSaturation's comment.
  +        let maxVal = Float.greatestFiniteMagnitude
  +        let testValues: [Float] = [
  +            maxVal,          // overflows f16 range -> +Inf
  +            -maxVal,         // overflows f16 range -> -Inf
  +            1.0e-20,         // underflows f16 range -> 0
  +            6.10352e-5,      // smallest f16 normal
  +            5.96046e-8       // smallest f16 subnormal (2^-24)
  +        ]
  +
  +        var f16Bits = [UInt16](repeating: 0, count: testValues.count)
  +        var roundTrip = [Float](repeating: 0, count: testValues.count)
  +
  +        testValues.withUnsafeBufferPointer { src in
  +            f16Bits.withUnsafeMutableBufferPointer { dst in
  +                f32ToF16Batch(src.baseAddress!, dst.baseAddress!, testValues.count)
  +            }
  +        }
  +        f16Bits.withUnsafeBufferPointer { src in
  +            roundTrip.withUnsafeMutableBufferPointer { dst in
  +                f16ToF32Batch(src.baseAddress!, dst.baseAddress!, testValues.count)
  +            }
  +        }
  +
  +        XCTAssertTrue(roundTrip[0].isInfinite && roundTrip[0] > 0, "Max f32 should overflow to +Inf in f16")
  +        XCTAssertTrue(roundTrip[1].isInfinite && roundTrip[1] < 0, "-Max f32 should overflow to -Inf in f16")
  +        XCTAssertEqual(roundTrip[2], 0.0, "1e-20 should underflow to zero in f16")
  +
  +        let minNormalError = abs(roundTrip[3] - testValues[3]) / testValues[3]
  +        XCTAssertLessThan(minNormalError, 0.001, "Smallest f16 normal should round-trip accurately")
  +
  +        let subnormalError = abs(roundTrip[4] - testValues[4])
  +        XCTAssertLessThan(subnormalError, 1e-8, "Smallest f16 subnormal should round-trip accurately")
  +    }
   }
  ```

- [ ] **Step 2: Verify the 6 new tests pass while `CS2RNG` still exists (isolates any failure
  to the new tests themselves, not to the deletions in later steps).**
  ```bash
  swift test --filter "S2RNGDtypeTests/testXoroshiro128UniformityChiSquare|S2RNGDtypeTests/testXoroshiro128StreamIndependenceQuantitative|S2RNGDtypeTests/testInt8QuantizationExtremeSaturation|S2RNGDtypeTests/testNibblePackingOrderLowFirst|S2RNGDtypeTests/testEndianHelpers64Bit|S2RNGDtypeTests/testF16BoundaryValues" 2>&1 | tail -20
  ```
  Expected final lines:
  ```
  Test Suite 'S2RNGDtypeTests' passed at ...
       Executed 6 tests, with 0 failures (0 unexpected) in ...
  ```

- [ ] **Step 3: Delete the three CS2RNG-importing test files.**
  ```bash
  git rm Tests/VectorIndexTests/S2EdgeCaseTests.swift
  git rm Tests/VectorIndexTests/RNGDeterminismTests.swift
  git rm Tests/VectorIndexTests/DTypeConversionTests.swift
  ```
  Expected output: three `rm '...'` lines, one per file.

- [ ] **Step 4: Delete the `CS2RNG` C target's sources.**
  ```bash
  git rm -r Sources/CS2RNG
  ```
  Expected output includes `rm 'Sources/CS2RNG/s_rng_dtype_helpers.c'` and
  `rm 'Sources/CS2RNG/include/s_rng_dtype_helpers.h'`.

- [ ] **Step 5: Edit `Package.swift` — remove the `CS2RNG` target block.**
  ```diff
           .target(
               name: "CPQEncode",
               publicHeadersPath: "include"
           ),
  -        .target(
  -            name: "CS2RNG",
  -            publicHeadersPath: "include",
  -            cSettings: [
  -                .define("S2_ENABLE_TELEMETRY", to: "1")
  -            ]
  -        ),
           .target(
               name: "VectorIndex",
  ```

- [ ] **Step 6: Edit `Package.swift` — remove the `"CS2RNG"` dependency entry.**
  ```diff
           .target(
               name: "VectorIndex",
               dependencies: [
                   "CAtomicsShim",
                   "CPQEncode",
  -                "CS2RNG",
                   .product(name: "VectorCore", package: "VectorCore")
               ],
  ```

- [ ] **Step 7: Verify no remaining reference to CS2RNG anywhere in the repo.**
  ```bash
  grep -rn "CS2RNG" --include="*.swift" Package.swift Sources Tests 2>/dev/null
  ```
  Expected output: nothing (empty).

- [ ] **Step 8: Clean build.**
  ```bash
  swift build 2>&1 | tail -5
  ```
  Expected final line: `Build complete!` with zero errors.

- [ ] **Step 9: Full `S2` test filter green (covers `S2RNGDtypeTests`, the only remaining `S2*`
  suite once the three deleted files are gone).**
  ```bash
  swift test --filter S2RNGDtypeTests 2>&1 | tail -10
  ```
  Expected final lines:
  ```
  Test Suite 'S2RNGDtypeTests' passed at ...
       Executed 28 tests, with 0 failures (0 unexpected) in ...
  ```
  (22 pre-existing + 6 ported = 28.)

- [ ] **Step 10: Full test suite green (confirms the 3 deleted files aren't referenced by
  anything else, e.g. no shared XCTestCase subclass or helper).**
  ```bash
  swift test 2>&1 | tail -15
  ```
  Expected final lines: `** TEST SUCCEEDED **`-equivalent SwiftPM summary — `Executed N tests,
  with 0 failures` and no mention of `S2EdgeCaseTests`, `RNGDeterminismTests`, or
  `DTypeConversionTests` (they no longer exist).

- [ ] **Step 11: Commit.**
  ```bash
  git add Package.swift Sources/CS2RNG Tests/VectorIndexTests/S2RNGDtypeTests.swift \
    Tests/VectorIndexTests/S2EdgeCaseTests.swift Tests/VectorIndexTests/RNGDeterminismTests.swift \
    Tests/VectorIndexTests/DTypeConversionTests.swift
  git commit -m "$(cat <<'EOF'
  chore(cleanup): remove dead CS2RNG C target; port its unique test coverage

  CS2RNG has no production Swift caller (Sources/VectorIndex/Kernels/S2_RNGDtype.swift
  is a complete, already-shipped pure-Swift reimplementation of its RNG/dtype
  surface), but 3 test files (674 lines) imported it directly and would have
  broken the build. Each of their 27 tests was triaged: 8 die with the C target
  (untestable without it, incl. two that would trap the Swift port on Inf/NaN/
  odd-length input -- pre-existing bugs, out of scope here), 13 were already
  redundant with S2RNGDtypeTests, and 6 covered unique behavior now ported onto
  the pure-Swift API (chi-square uniformity, quantitative stream independence,
  extreme-value int8 saturation, nibble-packing byte order, 64-bit endian
  helpers, f16 overflow/underflow/subnormal boundaries).

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  git status --short
  ```
  Expected: commit succeeds; `git status --short` prints nothing (clean tree).

---

### Task 3: Telemetry consolidation (B1, re-scoped per user decision 2026-07-26)

**Premise corrections (verified live against `main @ ee67895`, beyond what the brief covered):**

1. The brief already found that `Telemetry.swift`'s `#if VINDEX_TELEM` branch fails with 12
   errors when the flag is turned on (10 Swift-6-strict-concurrency violations on bare mutable
   `static var` globals, 2 stale pthread-TLS-destructor API calls) and is never selected in any
   normal build (no Swift `-D VINDEX_TELEM` exists anywhere in `Package.swift`). Re-verified:
   `swift build --target VectorIndex -Xswiftc -DVINDEX_TELEM` still fails with exactly those 12
   errors on this branch.
2. **New finding, not in the brief:** two of the brief's "six already-working" recorders are
   *also* dead. `HNSWTelemetryRecorder.record?(t)` (`HNSWTraversal.swift:289,299`) and all 6 of
   `GlobalTelemetryRecorder.record?(telemetry)`'s call sites (`LayoutTransforms.swift`) are each
   individually wrapped in `#if ENABLE_TELEMETRY` — a *third*, independent dead compile flag
   (distinct from both `VINDEX_TELEM` and `S2_ENABLE_TELEMETRY`) that is never defined anywhere
   in `Package.swift`. Verified live: `swift build --target VectorIndex -Xswiftc -DENABLE_TELEMETRY`
   fails with 13 errors — 7 Swift-6-strict-concurrency violations on bare `private var
   greedy_ns_accum`/`scoringAccum_ns`/`efsearch_ns_accum`/`earlyExitCount`/`edgesVisitedCount`/
   `neighborBatchesCount`/`candidatesPushedCount` globals in `HNSWTraversal.swift`, plus 6
   "initializer ... is internal and cannot be referenced from an '@inlinable' function" errors
   in `LayoutTransforms.swift` (`LayoutTransformTelemetry`'s synthesized memberwise init is not
   `public`/`@usableFromInline`, but its 6 callers are `@inlinable`). This is the same disease as
   `VINDEX_TELEM` (dead flag + doesn't compile if enabled), on a different pair of recorders.
   Fixing `ENABLE_TELEMETRY` is a real, separate bug fix (rewrite 7 globals + widen 1
   initializer's access level, then decide whether to actually flip the flag on) — **out of
   scope for this task**; routed to a follow-up (see the note after Step 11). The other four
   recorders (`Cosine`/`InnerProduct`/`L2Sqr`/`TopK`) are genuinely gated only by a runtime
   `Bool` (or, for `L2Sqr`, not gated at all) with no `#if` involved — confirmed by reading
   every call site — so the brief's "already-working" claim holds for those four.
3. `IDMapOpts.enableTelemetry` (`IDMap.swift:15`) carries a `public` keyword but `IDMapOpts`
   itself is declared `internal struct` (`IDMap.swift:8`) — Swift caps the property's effective
   access at its container's level, so this was never actually part of the public API despite
   the keyword. Safe to delete outright, no deprecation needed.
4. `VisitedOpts.enableTelemetry` (`CandidateDedup.swift:49`) is genuinely public — `VisitedOpts`
   is a `public struct` with a `public init`. Deleting it would change the public initializer's
   signature (a breaking change), so it is deprecated in place, not deleted.

**Files:**
- Modify: `Sources/VectorIndex/Kernels/Telemetry.swift` (full-file rewrite — see Step 1)
- Modify: `Sources/VectorIndex/Operations/Rerank/ExactRerank.swift` (delete the two
  `#if VINDEX_TELEM` blocks at the original lines ~644–649 and ~748–751)
- Modify: `Sources/VectorIndex/Kernels/IDMap.swift` (delete line 15)
- Modify: `Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift` (deprecate line 49)
- Modify: `Sources/VectorIndex/Operations/RangeQuery/RangeQuery.swift` (replace the
  `recordTelemetry` free function with `RangeScanTelemetryRecorder`; update 3 call sites)
- Create: `Tests/VectorIndexTests/TelemetryRecorderTests.swift` (8 new tests)

**Interfaces:**
- Consumes: `HNSWTelemetryRecorder.record`, `GlobalTelemetryRecorder.record`,
  `IndexOps.Scoring.Cosine.TelemetryRecorder.sink`,
  `IndexOps.Scoring.InnerProduct.TelemetryRecorder.sink`, `L2SqrTelemetryRecorder.sink`,
  `IndexOps.Selection.TopKTelemetryRecorder.sink` (all pre-existing, unchanged),
  `DefaultVisitedSet.getTelemetry(elapsedNanos:) -> VisitedTelemetry` (pre-existing, unchanged).
- Produces: `public enum RangeScanTelemetryRecorder { public nonisolated(unsafe) static var
  sink: ((RangeScanTelemetry) -> Void)?; public static func record(_ t: RangeScanTelemetry) }` —
  new public symbol, additive only (no existing symbol's signature changes). The internal
  `recordTelemetry(_:)` free function it replaces was `@usableFromInline internal`, never
  public, so removing it is not a public-API break.

This task is 2 commits (deletion, then wiring+tests); both leave `swift build` / the relevant
`swift test` filter green.

#### Commit 1: delete the dead VINDEX_TELEM implementation and vestigial flags

- [ ] **Step 1: Replace `Sources/VectorIndex/Kernels/Telemetry.swift` in full.** The
  `#if VINDEX_TELEM` branch (the broken TLS/histogram/JSON-snapshot singleton) and its
  `#else` stub are collapsed into one unconditional stub; the 8 `telem_*` wrapper functions and
  5 `TELEM_*` event-recording functions are deleted outright (internal, zero remaining callers
  once Step 2 removes their only call sites); `QueryStatsLight` is deleted (internal, was only
  used by the deleted ring-buffer code). The 9 symbols that are `public` — `QueryCtx`,
  `TelemetryConfig`, `TelemetryGlobal`, `TelemetryCounter`, `TelemetryBytes`,
  `TelemetryDoubleField`, `TelemetryU64Field`, `TelemetryTimerGuard`, `TelemetryTimerToken` —
  are kept (cannot be deleted per the non-breaking rule) and marked
  `@available(*, deprecated, ...)`. `TelemetryTimerId`/`TelemetryFlags`/`QueryStats` (all
  `internal`) are kept because the deprecated public types still reference them
  (`TelemetryGlobal.time_ns`'s default value needs `TelemetryTimerId.allCases.count`;
  `TelemetryTimerGuard`/`TelemetryTimerToken` store a `TelemetryTimerId`; `TelemetryConfig.sink`
  closure parameter type is `QueryStats`, whose `flags` field needs `TelemetryFlags`).

  Overwrite the file with:
  ```swift
  //
  //  Telemetry.swift
  //  VectorIndex
  //
  //  Kernel #46: Index Stats & Telemetry
  //
  //  Phase-2 cleanup (2026-07): the VINDEX_TELEM-gated TLS/histogram/JSON-snapshot
  //  implementation below never compiled (12 Swift 6 strict-concurrency + stale-API
  //  errors under -D VINDEX_TELEM) and was never reachable from any call site in a
  //  shipping build. It has been removed. The types below are kept -- deprecated,
  //  not deleted, because they are `public` -- for source compatibility only; see
  //  the `@available` message on each for the Phase-4 removal plan.
  //
  //  The project's actual telemetry surface is the per-kernel push-callback
  //  recorders (HNSWTelemetryRecorder, GlobalTelemetryRecorder,
  //  IndexOps.Scoring.Cosine/InnerProduct.TelemetryRecorder, L2SqrTelemetryRecorder,
  //  IndexOps.Selection.TopKTelemetryRecorder, RangeScanTelemetryRecorder) plus
  //  CandidateDedup.DefaultVisitedSet.getTelemetry(). See docs/cleanup-0.2.0-plan.md.
  //

  import Foundation

  // MARK: - Internal Types (dead; retained only as storage for the deprecated public types below)

  /// Timer identifiers for different query stages
  internal enum TelemetryTimerId: Int, CaseIterable {
    case t_lut_build = 0
    case t_scan_adc
    case t_score_flat
    case t_topk
    case t_merge
    case t_dedup
    case t_reservoir
    case t_rerank
    case t_total
  }

  /// Optimization flags tracking which code paths were used
  internal struct TelemetryFlags: OptionSet, Sendable {
    let rawValue: UInt64
    static let used_dot_trick         = TelemetryFlags(rawValue: 1 << 0)
    static let used_cosine            = TelemetryFlags(rawValue: 1 << 1)
    static let used_interleaved_codes = TelemetryFlags(rawValue: 1 << 2)
    static let used_u4                = TelemetryFlags(rawValue: 1 << 3)
    static let used_prefetch          = TelemetryFlags(rawValue: 1 << 4)
    static let used_heap_merge        = TelemetryFlags(rawValue: 1 << 5)
  }

  /// Per-query statistics (returned to caller after query completion)
  internal struct QueryStats {
    // Identity / configuration
    var metric: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) // 16 bytes
    var d: Int32 = 0
    var m: Int32 = 0
    var ks: Int32 = 0
    var nprobe: Int32 = 0
    var C: Int32 = 0
    var K: Int32 = 0

    // Work
    var kc_scored: UInt64 = 0
    var lists_routed: UInt64 = 0
    var lists_scanned: UInt64 = 0
    var codes_scanned: UInt64 = 0
    var vecs_scored: UInt64 = 0
    var candidates_emitted: UInt64 = 0
    var candidates_unique: UInt64 = 0
    var candidates_kept: UInt64 = 0
    var topk_selected: UInt64 = 0

    // Saturation / quality
    var reservoir_tau: Double = 0
    var heap_sifts: UInt64 = 0
    var quickselect_calls: UInt64 = 0
    var dup_ratio: Double = 0
    var beam_expansions: UInt64 = 0

    // Bytes
    var bytes_lut: UInt64 = 0
    var bytes_codes: UInt64 = 0
    var bytes_vecs: UInt64 = 0
    var bytes_ids: UInt64 = 0
    var bytes_norms: UInt64 = 0

    // Timers (ns)
    var t_lut_build: UInt64 = 0
    var t_scan_adc: UInt64 = 0
    var t_score_flat: UInt64 = 0
    var t_topk: UInt64 = 0
    var t_merge: UInt64 = 0
    var t_dedup: UInt64 = 0
    var t_reservoir: UInt64 = 0
    var t_rerank: UInt64 = 0
    var t_total: UInt64 = 0

    // Flags
    var flags: TelemetryFlags = []
  }

  /// Query context (passed at begin_query)
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: the VINDEX_TELEM implementation that consumed this never compiled and had no reachable call site. Scheduled for removal in Phase 4.")
  public struct QueryCtx {
    var metric: String?
    var d: Int32 = 0
    var m: Int32 = 0
    var ks: Int32 = 0
    var nprobe: Int32 = 0
    var C: Int32 = 0
    var K: Int32 = 0
    init(metric: String? = nil, d: Int32, m: Int32, ks: Int32, nprobe: Int32, C: Int32, K: Int32) {
      self.metric = metric; self.d = d; self.m = m; self.ks = ks; self.nprobe = nprobe; self.C = C; self.K = K
    }
  }

  /// Telemetry configuration
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: the VINDEX_TELEM implementation that consumed this never compiled and had no reachable call site. Scheduled for removal in Phase 4.")
  public struct TelemetryConfig {
    var enabled: Bool
    var sampleRate: Double           // [0,1]
    var maxHistBuckets: Int          // default 64, capped at 128
    var sink: ((QueryStats) -> Void)?// optional callback per query
    var persistSnapshot: Bool
    var persistPath: String?
    init(enabled: Bool = false, sampleRate: Double = 0.0, maxHistBuckets: Int = 64,
                sink: ((QueryStats) -> Void)? = nil, persistSnapshot: Bool = false, persistPath: String? = nil) {
      self.enabled = enabled; self.sampleRate = sampleRate; self.maxHistBuckets = maxHistBuckets
      self.sink = sink; self.persistSnapshot = persistSnapshot; self.persistPath = persistPath
    }
  }

  /// Global telemetry aggregates (snapshot-able)
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: the VINDEX_TELEM implementation that consumed this never compiled and had no reachable call site. Scheduled for removal in Phase 4.")
  public struct TelemetryGlobal {
    // Totals
    var queries_total: UInt64 = 0
    var queries_sampled: UInt64 = 0

    // Work sums
    var work_kc_scored: UInt64 = 0
    var work_lists_routed: UInt64 = 0
    var work_lists_scanned: UInt64 = 0
    var work_codes_scanned: UInt64 = 0
    var work_vecs_scored: UInt64 = 0
    var work_candidates_emitted: UInt64 = 0
    var work_candidates_unique: UInt64 = 0
    var work_candidates_kept: UInt64 = 0
    var work_topk_selected: UInt64 = 0

    // Bytes sums
    var bytes_lut: UInt64 = 0
    var bytes_codes: UInt64 = 0
    var bytes_vecs: UInt64 = 0
    var bytes_ids: UInt64 = 0
    var bytes_norms: UInt64 = 0

    // Time sums
    var time_ns: [UInt64] = Array(repeating: 0, count: TelemetryTimerId.allCases.count)

    // Flags counters
    var flag_used_dot_trick: UInt64 = 0
    var flag_used_cosine: UInt64 = 0
    var flag_used_interleaved_codes: UInt64 = 0
    var flag_used_u4: UInt64 = 0
    var flag_used_prefetch: UInt64 = 0
    var flag_used_heap_merge: UInt64 = 0

    // Ring
    var ring_cap: UInt32 = 1024
  }

  // MARK: - Event Helpers (dead; retained only as storage for the deprecated public types below)

  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever incremented this. Scheduled for removal in Phase 4.")
  public enum TelemetryCounter {
    case kc_scored, lists_routed, lists_scanned, codes_scanned, vecs_scored
    case candidates_emitted, candidates_unique, candidates_kept, topk_selected
    case heap_sifts, quickselect_calls, beam_expansions
  }

  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever incremented this. Scheduled for removal in Phase 4.")
  public enum TelemetryBytes { case lut, codes, vecs, ids, norms }
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever set this. Scheduled for removal in Phase 4.")
  public enum TelemetryDoubleField { case reservoir_tau, dup_ratio }
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever set this. Scheduled for removal in Phase 4.")
  public enum TelemetryU64Field { case candidates_emitted, candidates_unique, candidates_kept }

  /// RAII timer guard (automatically stops timer on deinit)
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever constructed this. Scheduled for removal in Phase 4.")
  public struct TelemetryTimerGuard: ~Copyable {
    internal let id: TelemetryTimerId
    internal let t0: UInt64
    init(_ id: TelemetryTimerId) { self.id = id; self.t0 = Telemetry._nowNs() }
    deinit { Telemetry._addTimer(id, delta: Telemetry._nowNs() &- t0) }
  }

  /// Manual timer token (start/end pair)
  @available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever constructed this. Scheduled for removal in Phase 4.")
  public struct TelemetryTimerToken {
    internal let id: TelemetryTimerId
    internal let t0: UInt64
  }

  // MARK: - Implementation (stub only; the VINDEX_TELEM-gated real implementation
  // and its 12 compile errors -- 10 Swift 6 strict-concurrency violations on bare
  // mutable global state, 2 stale pthread-TLS-destructor API calls -- were removed
  // in the Phase 2 cleanup. `_nowNs`/`_addTimer` are kept only because the
  // deprecated-but-still-public `TelemetryTimerGuard` above references them.)

  @usableFromInline
  internal enum Telemetry {
    @usableFromInline
    @inline(__always)
    static func _nowNs() -> UInt64 { 0 }
    @inline(__always) static func _addTimer(_ id: TelemetryTimerId, delta: UInt64) {}
  }
  ```

- [ ] **Step 2: Delete the two dead `#if VINDEX_TELEM` blocks in `ExactRerank.swift`, plus the
  now-stale header comment that references `VINDEX_TELEM`.**
  ```diff
       // - Honors options from kernel-spec (#40) including gather tiling and locality reorder
  -    // - Emits telemetry counters/timers (#46) when compiled with VINDEX_TELEM

       public extension IndexOps {
           enum Rerank {}
  ```
  ```diff
           reader: any VectorReader, opts: RerankOpts,
           scoresOut: UnsafeMutablePointer<Float>
       ) {
  -        #if VINDEX_TELEM
  -        _ = TELEM_TIMER_GUARD(.t_rerank)
  -        TELEM_FLAG(.used_prefetch) // advisory; locality reorder acts as prefetch surrogate
  -        if metric == .cosine { TELEM_FLAG(.used_cosine) }
  -        TELEM_INC(.vecs_scored, UInt64(C))
  -        TELEM_ADD_BYTES(.vecs, UInt64(C * d * MemoryLayout<Float>.stride))
  -        #endif
           scoreBlock(q: q, d: d, metric: metric, ids: candIDs, C: C, reader: reader, opts: opts, scoresOut: scoresOut, presentMaskOut: nil)
       }
  ```
  ```diff
           if actual < K {
               let sentinel = _missingSentinel(metric)
               for i in actual..<K { topScores[i] = sentinel; topIDs[i] = -1 }
           }
  -
  -        #if VINDEX_TELEM
  -        TELEM_SET64(.candidates_kept, UInt64(actual))
  -        TELEM_INC(.topk_selected, UInt64(K))
  -        #endif
       }
  ```

- [ ] **Step 3: Delete the vestigial `IDMapOpts.enableTelemetry` (not public — see premise
  correction #3 above).**
  ```diff
           public var enableBloom: Bool = false
  -        public var enableTelemetry: Bool = false
           public static var `default`: IDMapOpts { IDMapOpts() }
  ```

- [ ] **Step 4: Deprecate (do not delete) `VisitedOpts.enableTelemetry` in
  `CandidateDedup.swift` — see premise correction #4 above.**
  ```diff
           public let epochBits: Int         // For DenseEpoch wrap testing (8..32)
  +        @available(*, deprecated, message: "Never read anywhere in DefaultVisitedSet; gates nothing. Scheduled for removal in Phase 4.")
           public let enableTelemetry: Bool
  ```
  This produces exactly one expected warning, at `VisitedOpts.init`'s own assignment
  (`self.enableTelemetry = enableTelemetry`) — verified live. This is normal for deprecating a
  stored property that a memberwise-style init must still assign, and does not fail the build.
  It is the only new warning either of these two edits introduces.

- [ ] **Step 5: Verify no remaining reference to the deleted `TELEM_*`/`telem_*`
  symbols anywhere in the repo.**
  ```bash
  grep -rn "TELEM_INC\|TELEM_ADD_BYTES\|TELEM_SET\|TELEM_FLAG\|TELEM_TIMER_GUARD\|TELEM_TIMER_START\|TELEM_TIMER_END\|telem_init\|telem_shutdown\|telem_thread_init\|telem_begin_query\|telem_end_query\|telem_snapshot\|QueryStatsLight" --include="*.swift" Sources Tests
  ```
  Expected output: nothing (empty).

- [ ] **Step 6: Clean build.**
  ```bash
  swift build 2>&1 | tail -10
  ```
  Expected: `Build complete!`, plus exactly one warning line mentioning
  `'enableTelemetry' is deprecated` at `CandidateDedup.swift`'s `VisitedOpts.init` (Step 4) and
  no other new warnings or errors.

- [ ] **Step 7: Full test suite green (nothing else touched anything in this commit).**
  ```bash
  swift test 2>&1 | tail -15
  ```
  Expected: `Executed N tests, with 0 failures` (same `N` as on `main` before this task, since
  no test file changed yet).

- [ ] **Step 8: Commit 1.**
  ```bash
  git add Sources/VectorIndex/Kernels/Telemetry.swift \
    Sources/VectorIndex/Operations/Rerank/ExactRerank.swift \
    Sources/VectorIndex/Kernels/IDMap.swift \
    Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift
  git commit -m "$(cat <<'EOF'
  chore(cleanup): remove the never-compiled VINDEX_TELEM telemetry singleton

  Telemetry.swift's #if VINDEX_TELEM branch (TLS + lock-striped histograms +
  JSON snapshot) fails with 12 errors when the flag is turned on and is never
  selected in any build today (no Swift -D VINDEX_TELEM exists anywhere in
  Package.swift) -- it and its two call sites in ExactRerank.swift (incl. the
  permanently-true used_prefetch surrogate flag) are removed. The file's public
  types (QueryCtx, TelemetryConfig, TelemetryGlobal, TelemetryCounter,
  TelemetryBytes, TelemetryDoubleField, TelemetryU64Field, TelemetryTimerGuard,
  TelemetryTimerToken) are kept and marked deprecated rather than deleted, per
  the non-breaking rule. Also drops IDMapOpts.enableTelemetry (not actually
  public -- its container is an internal struct) and deprecates
  VisitedOpts.enableTelemetry (genuinely public, never read; CandidateDedup.swift),
  both of which gated nothing.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  git status --short
  ```
  Expected: commit succeeds; `git status --short` prints nothing.

#### Commit 2: wire RangeQuery's telemetry, add accuracy tests for the 7 recorders + 1 pull API

- [ ] **Step 9: Replace `RangeQuery.swift`'s no-op `recordTelemetry` free function with a
  push-callback recorder, following the exact pattern of the other five (`HNSWTelemetryRecorder`
  / `Cosine`/`InnerProduct`/`TopK`'s `TelemetryRecorder` `sink`+`record` wrapper).**
  ```diff
  -// Lightweight recorder hook — replace with your Telemetry (#46) integration.
  -@usableFromInline
  -@inline(__always) internal func recordTelemetry(_ t: RangeScanTelemetry) {
  -    _ = t  // TODO: Wire to global telemetry
  +/// Push-callback telemetry recorder, following the same pattern as
  +/// `HNSWTelemetryRecorder`, `IndexOps.Scoring.Cosine.TelemetryRecorder`,
  +/// `IndexOps.Scoring.InnerProduct.TelemetryRecorder`, `L2SqrTelemetryRecorder`,
  +/// and `IndexOps.Selection.TopKTelemetryRecorder`. `sink` defaults to `nil`, so
  +/// this is a no-op until a host application opts in.
  +public enum RangeScanTelemetryRecorder {
  +    public nonisolated(unsafe) static var sink: ((RangeScanTelemetry) -> Void)?
  +    @inline(__always) public static func record(_ t: RangeScanTelemetry) { sink?(t) }
     }
  ```
  `rangeScanBlock`, `rangeScanADC_u8`, and `rangeScanADC_u4` (the 3 callers) are all
  `@inlinable public func`, so `record` must be `public` (matching the other 5 recorders) — a
  `@usableFromInline internal` function, as the old `recordTelemetry` was, would also have
  worked, but `public` matches the established pattern exactly, per the task's instruction to
  follow it.

- [ ] **Step 10: Update the 3 call sites (`rangeScanBlock` line ~309, `rangeScanADC_u8` line
  ~465, `rangeScanADC_u4` line ~555 — post-Step-9 line numbers).**
  ```bash
  grep -rn "recordTelemetry(telem)" Sources/VectorIndex/Operations/RangeQuery/RangeQuery.swift
  ```
  Expected output: 3 lines, each reading `        recordTelemetry(telem)`. Replace all 3
  occurrences (identical text at each site) with:
  ```diff
  -        recordTelemetry(telem)
  +        RangeScanTelemetryRecorder.record(telem)
  ```

- [ ] **Step 11: Verify no remaining reference to the old free function.**
  ```bash
  grep -rn "recordTelemetry(" --include="*.swift" Sources Tests
  ```
  Expected output: nothing (empty; only `RangeScanTelemetryRecorder.record(...)` remains).

  **Follow-up note (not part of this task, do not action here):** the `ENABLE_TELEMETRY`
  discovery from this task's premise-correction #2 — `HNSWTelemetryRecorder` and
  `GlobalTelemetryRecorder`'s production call sites are dead behind a flag that also doesn't
  compile when enabled — is a real defect distinct from anything B1 originally scoped. It is
  not a public-symbol deprecation (no public API is affected either way), so it does not belong
  in the Phase-4 deprecation-removal list; it belongs on the project's regular bug backlog as
  its own fix: (a) mark the 7 `HNSWTraversal.swift` accumulator globals `nonisolated(unsafe)`
  (matching every other recorder's pattern in this codebase) or move them into an actor/struct,
  (b) mark `LayoutTransformTelemetry`'s initializer `public` (or `@usableFromInline`) so its 6
  `@inlinable` callers can reach it, (c) only then decide, separately, whether `ENABLE_TELEMETRY`
  should default on, stay opt-in via a trait, or be replaced by unconditional recording (the
  `L2SqrTelemetryRecorder` pattern) like the other four.

- [ ] **Step 12: Create `Tests/VectorIndexTests/TelemetryRecorderTests.swift` with all 8
  tests.** Every numeric expectation below (bytes-read formulas, TopK's comparisons/heapPushes/
  siftOperations, RangeQuery's kept/scores) was produced by actually compiling and running this
  exact test file against this branch's kernels before being written here — none are estimated.
  ```swift
  //
  //  TelemetryRecorderTests.swift
  //  VectorIndexTests
  //
  //  Accuracy tests for the project's actual telemetry surface (Phase 2 cleanup,
  //  Task 3): the six pre-existing `nonisolated(unsafe)` push-callback recorders,
  //  the newly-wired RangeScanTelemetryRecorder, and CandidateDedup's pull-style
  //  getTelemetry() API. Each test hand-computes the expected counters/fields for
  //  a small, fully deterministic input and asserts them exactly.
  //
  //  Concurrency note: every recorder here is a `nonisolated(unsafe) static var`
  //  callback (or, for CandidateDedup, an instance method) with no built-in
  //  synchronization. All assertions below pin the deterministic single-threaded
  //  path -- one call in, one recorded value out, on the calling thread -- and
  //  never run kernels concurrently against a shared recorder. Do not add
  //  parallel/XCTestCase-concurrent variants of these tests without adding your
  //  own synchronization around `sink`/`record`, since nothing here does.
  //

  import XCTest
  @testable import VectorIndex

  final class TelemetryRecorderTests: XCTestCase {

      // MARK: - HNSWTelemetryRecorder
      //
      // NOTE: HNSWTraversal.greedyDescent/efSearch's calls to
      // `HNSWTelemetryRecorder.record?(t)` are wrapped in `#if ENABLE_TELEMETRY`
      // (HNSWTraversal.swift:284-299), a *second*, independent dead compile flag
      // from VINDEX_TELEM -- `ENABLE_TELEMETRY` is never defined anywhere in
      // Package.swift, and turning it on does not compile (verified live:
      // 7 Swift 6 strict-concurrency errors on bare `private var ..._accum`/
      // `...Count` globals in HNSWTraversal.swift). That make-it-compile-and-flip-
      // it-on fix is out of scope for this telemetry-consolidation task (see the
      // task's premise-correction note) and is routed to a follow-up. This test
      // therefore exercises the recorder's sink mechanism directly -- the part of
      // the system that does work today -- rather than through the (currently
      // unreachable) production call site.
      func testHNSWTelemetryRecorderSink() throws {
          var captured: HNSWTraversalTelemetry?
          HNSWTelemetryRecorder.record = { captured = $0 }
          defer { HNSWTelemetryRecorder.record = nil }

          var t = HNSWTraversalTelemetry()
          t.edgesVisited = 12
          t.neighborBatches = 3
          t.candidatesPushed = 7
          t.earlyExits = 1
          t.greedy_ns = 100
          t.efsearch_ns = 200
          t.scoring_ns = 50
          t.total_ns = 350

          HNSWTelemetryRecorder.record?(t)

          XCTAssertEqual(captured?.edgesVisited, 12)
          XCTAssertEqual(captured?.neighborBatches, 3)
          XCTAssertEqual(captured?.candidatesPushed, 7)
          XCTAssertEqual(captured?.earlyExits, 1)
          XCTAssertEqual(captured?.greedy_ns, 100)
          XCTAssertEqual(captured?.efsearch_ns, 200)
          XCTAssertEqual(captured?.scoring_ns, 50)
          XCTAssertEqual(captured?.total_ns, 350)
      }

      // MARK: - GlobalTelemetryRecorder (LayoutTransforms)
      //
      // Same caveat as HNSW above: all 6 of LayoutTransforms.swift's
      // `GlobalTelemetryRecorder.record?(telemetry)` call sites are individually
      // wrapped in `#if ENABLE_TELEMETRY` and are dead for the same reason
      // (verified live: turning the flag on additionally fails with 6 "internal
      // initializer referenced from an '@inlinable' function" errors, since
      // `LayoutTransformTelemetry`'s memberwise init is not `public`/
      // `@usableFromInline` but the callers are `@inlinable`). Tested directly.
      func testGlobalTelemetryRecorderSink() throws {
          var captured: LayoutTransformTelemetry?
          GlobalTelemetryRecorder.record = { captured = $0 }
          defer { GlobalTelemetryRecorder.record = nil }

          let t = LayoutTransformTelemetry(
              transformType: "vec_interleave",
              vectorCount: 100,
              dimension: 128,
              subquantizers: 0,
              rowBlockSize: 8,
              groupSize: 0,
              bytesTransformed: 51200,
              executionTimeNanos: 5000
          )

          GlobalTelemetryRecorder.record?(t)

          XCTAssertEqual(captured?.transformType, "vec_interleave")
          XCTAssertEqual(captured?.vectorCount, 100)
          XCTAssertEqual(captured?.dimension, 128)
          XCTAssertEqual(captured?.rowBlockSize, 8)
          XCTAssertEqual(captured?.bytesTransformed, 51200)
          XCTAssertEqual(captured?.executionTimeNanos, 5000)
      }

      // MARK: - IndexOps.Scoring.Cosine.TelemetryRecorder
      //
      // Real call: this recorder's call site is gated only by a runtime
      // `options.enableTelemetry` bool (default false), not by any `#if` -- it
      // already works today. n=1, d=4, q=xb=[1,0,0,0] (unit vectors, cosine=1),
      // explicit dbInvNorms/queryInvNorm=1.0 avoid relying on internally-computed
      // norms. d=4 is not one of the specialized dims (512/768/1024/1536), so the
      // fused-generic path runs (kernelVariant "generic_fused").
      func testCosineTelemetryRecorder() throws {
          var captured: IndexOps.Scoring.Cosine.Telemetry?
          IndexOps.Scoring.Cosine.TelemetryRecorder.sink = { captured = $0 }
          defer { IndexOps.Scoring.Cosine.TelemetryRecorder.sink = nil }

          let q: [Float] = [1, 0, 0, 0]
          let xb: [Float] = [1, 0, 0, 0]
          let invNorms: [Float] = [1.0]
          var out: [Float] = [0]
          var opts = IndexOps.Scoring.Cosine.Options()
          opts.enableTelemetry = true

          q.withUnsafeBufferPointer { qp in
              xb.withUnsafeBufferPointer { xbp in
                  invNorms.withUnsafeBufferPointer { np in
                      out.withUnsafeMutableBufferPointer { op in
                          IndexOps.Scoring.Cosine.run(
                              q: qp.baseAddress!, xb: xbp.baseAddress!, n: 1, d: 4,
                              out: op.baseAddress!, dbInvNorms: np.baseAddress!,
                              queryInvNorm: 1.0, options: opts)
                      }
                  }
              }
          }

          XCTAssertEqual(out[0], 1.0, accuracy: 1e-6)
          XCTAssertEqual(captured?.kernelVariant, "generic_fused")
          XCTAssertEqual(captured?.rowsProcessed, 1)
          // bytesRead = n*d*4 (xb) + d*4 (q) + n*4 (dbInvNorms) = 16 + 16 + 4
          XCTAssertEqual(captured?.bytesRead, 36)
          XCTAssertEqual(captured?.usedFusedPath, true)
          XCTAssertEqual(captured?.usedF16Norms, false)
          XCTAssertEqual(captured?.zeroNormCount, 0)
          XCTAssertEqual(captured?.clampedCount, 0)
      }

      // MARK: - IndexOps.Scoring.InnerProduct.TelemetryRecorder
      //
      // Real call: same runtime-bool gating pattern as Cosine. n=1, d=4,
      // q=[1,2,3,4], xb=[1,1,1,1] -> dot product 1+2+3+4=10. d=4 is not
      // specialized, so the generic (non-fast) path runs.
      func testInnerProductTelemetryRecorder() throws {
          var captured: IndexOps.Scoring.InnerProduct.Telemetry?
          IndexOps.Scoring.InnerProduct.TelemetryRecorder.sink = { captured = $0 }
          defer { IndexOps.Scoring.InnerProduct.TelemetryRecorder.sink = nil }

          let q: [Float] = [1, 2, 3, 4]
          let xb: [Float] = [1, 1, 1, 1]
          var out: [Float] = [0]
          var opts = IndexOps.Scoring.InnerProduct.Options()
          opts.enableTelemetry = true

          q.withUnsafeBufferPointer { qp in
              xb.withUnsafeBufferPointer { xbp in
                  out.withUnsafeMutableBufferPointer { op in
                      IndexOps.Scoring.InnerProduct.run(
                          q: qp.baseAddress!, xb: xbp.baseAddress!, n: 1, d: 4,
                          out: op.baseAddress!, options: opts)
                  }
              }
          }

          XCTAssertEqual(out[0], 10.0, accuracy: 1e-6)
          XCTAssertEqual(captured?.kernelVariant, "generic")
          XCTAssertEqual(captured?.rowsProcessed, 1)
          // bytesRead = (n*d + d) * 4 = (4 + 4) * 4
          XCTAssertEqual(captured?.bytesRead, 32)
          XCTAssertEqual(captured?.fastPathHit, false)
          XCTAssertEqual(captured?.vectorWidth, 4)
      }

      // MARK: - L2SqrTelemetryRecorder
      //
      // Real call: `l2sqr_f32_block` records unconditionally (no runtime gate at
      // all, unlike the other five -- verified by reading the source: the
      // `L2SqrTelemetryRecorder.record(...)` call has no enclosing `if`). q=zero
      // vector, xb=[1,1,1,1] -> squared L2 distance = 4. `opts.algo = .direct`
      // forces the direct path so `usedDotTrick` is deterministically false.
      func testL2SqrTelemetryRecorder() throws {
          var captured: L2SqrTelemetry?
          L2SqrTelemetryRecorder.sink = { captured = $0 }
          defer { L2SqrTelemetryRecorder.sink = nil }

          let q: [Float] = [0, 0, 0, 0]
          let xb: [Float] = [1, 1, 1, 1]
          var out: [Float] = [0]
          var opts = L2SqrOpts.default
          opts.algo = .direct

          q.withUnsafeBufferPointer { qp in
              xb.withUnsafeBufferPointer { xbp in
                  out.withUnsafeMutableBufferPointer { op in
                      withUnsafePointer(to: opts) { optsP in
                          l2sqr_f32_block(qp.baseAddress!, xbp.baseAddress!, 1, 4, op.baseAddress!, nil, .nan, optsP)
                      }
                  }
              }
          }

          XCTAssertEqual(out[0], 4.0, accuracy: 1e-6)
          XCTAssertEqual(captured?.rows, 1)
          XCTAssertEqual(captured?.dim, 4)
          XCTAssertEqual(captured?.usedDotTrick, false)
          XCTAssertNil(captured?.specializedDim)
          // bytesRead = n*d*4 (xb) + d*4 (q) + 0 (no norms) = 16 + 16
          XCTAssertEqual(captured?.bytesRead, 32)
      }

      // MARK: - IndexOps.Selection.TopKTelemetryRecorder
      //
      // Real call: 5 candidates, k=3, max-ordering, `forceAlgorithm: .streaming`
      // for a deterministic algorithm choice (n=5 is far under the default
      // hybridThreshold of 16,384 anyway, so `.streaming` would also be chosen
      // automatically). Expected top-3 by score: 9(id 14), 8(id 12), 5(id 10).
      func testTopKTelemetryRecorder() throws {
          var captured: IndexOps.Selection.TopKTelemetry?
          IndexOps.Selection.TopKTelemetryRecorder.sink = { captured = $0 }
          defer { IndexOps.Selection.TopKTelemetryRecorder.sink = nil }

          let scores: [Float] = [5, 3, 8, 1, 9]
          let ids: [Int32] = [10, 11, 12, 13, 14]
          let config = IndexOps.Selection.TopKConfig(enableTelemetry: true, forceAlgorithm: .streaming, hybridThreshold: 16_384)

          var pairs: [(score: Float, id: Int32)] = []
          scores.withUnsafeBufferPointer { sp in
              ids.withUnsafeBufferPointer { ip in
                  let heap = IndexOps.Selection.selectTopK(scores: sp.baseAddress!, ids: ip.baseAddress, count: 5, k: 3, ordering: .max, config: config)
                  pairs = heap.extractSorted().map { ($0.score, $0.id) }
                  heap.deallocate()
              }
          }

          XCTAssertEqual(pairs.map(\.score), [9, 8, 5])
          XCTAssertEqual(pairs.map(\.id), [14, 12, 10])

          XCTAssertEqual(captured?.candidatesProcessed, 5)
          XCTAssertEqual(captured?.k, 3)
          // Verified against the real streaming-heap implementation on this input.
          XCTAssertEqual(captured?.comparisons, 2)
          XCTAssertEqual(captured?.heapPushes, 4)
          XCTAssertEqual(captured?.siftOperations, 2)
      }

      // MARK: - RangeScanTelemetryRecorder (newly wired in this task)
      //
      // Real call: 3 flat vectors, d=2, query=[0,0], database rows at distance²
      // 0, 1, and 25 from the query. threshold=2.0 (compared against the raw
      // L2Sqr score, i.e. squared distance, not sqrt'd -- verified: scoresOut
      // for the kept rows come back as 0.0 and 1.0, not 0.0 and 1.0's square
      // roots). `config.earlyExit = .off` forces the flat non-early-exit path
      // deterministically.
      func testRangeScanTelemetryRecorder() throws {
          var captured: RangeScanTelemetry?
          RangeScanTelemetryRecorder.sink = { captured = $0 }
          defer { RangeScanTelemetryRecorder.sink = nil }

          let query: [Float] = [0, 0]
          let database: [Float] = [0, 0, /* row 0 */ 1, 0, /* row 1 */ 5, 0 /* row 2 */]
          let ids: [Int64] = [100, 101, 102]
          var idsOut = [Int64](repeating: 0, count: 3)
          var scoresOut = [Float](repeating: 0, count: 3)
          let config = RangeScanConfig(earlyExit: .off, outputScores: true, idFilter: nil,
                                       visitedSet: nil, reservoir: nil, outputMode: .compacted,
                                       tileSize: 1024, enableTelemetry: true)

          let kept = query.withUnsafeBufferPointer { qp in
              database.withUnsafeBufferPointer { dbp in
                  ids.withUnsafeBufferPointer { idp in
                      idsOut.withUnsafeMutableBufferPointer { iop in
                          scoresOut.withUnsafeMutableBufferPointer { sop in
                              rangeScanBlock(query: qp.baseAddress!, database: dbp.baseAddress!,
                                             ids: idp.baseAddress, vectorCount: 3, dimension: 2,
                                             metric: .l2, threshold: 2.0, idsOut: iop.baseAddress!,
                                             scoresOut: sop.baseAddress!, maxOut: 3, config: config)
                          }
                      }
                  }
              }
          }

          XCTAssertEqual(kept, 2)
          XCTAssertEqual(Array(idsOut.prefix(2)), [100, 101])
          XCTAssertEqual(Array(scoresOut.prefix(2)), [0.0, 1.0])

          XCTAssertNotNil(captured, "RangeScanTelemetryRecorder.sink must fire when config.enableTelemetry is true")
          XCTAssertEqual(captured?.vectorsScanned, 3)
          XCTAssertEqual(captured?.vectorsKept, 2)
          XCTAssertEqual(captured?.usedEarlyExit, false)
          XCTAssertEqual(captured?.earlyExitHits, 0)
          XCTAssertEqual(captured?.usedADCPath, false)
          // bytesScored = n*d*4 = 3*2*4
          XCTAssertEqual(captured?.bytesScored, 24)
          XCTAssertEqual(captured?.bytesCodes, 0)
      }

      // MARK: - CandidateDedup pull API: DefaultVisitedSet.getTelemetry()
      //
      // Not a push callback -- exercised directly. Sequence [5,7,5,9,7,5] against
      // a fresh denseEpoch set: 5(new),7(new),5(dup),9(new),7(dup),5(dup) ->
      // totalChecks=6, uniqueCount=3, duplicateCount=3.
      func testCandidateDedupGetTelemetry() throws {
          let vs = DefaultVisitedSet(idCapacity: 100, opts: VisitedOpts(mode: .denseEpoch))
          vs.resetForNewQuery()

          let sequence: [Int64] = [5, 7, 5, 9, 7, 5]
          let results = sequence.map { vs.testAndSet(id: $0) }

          XCTAssertEqual(results, [true, true, false, true, false, false])

          let telemetry = vs.getTelemetry(elapsedNanos: 1234)
          XCTAssertEqual(telemetry.mode, .denseEpoch)
          XCTAssertEqual(telemetry.totalChecks, 6)
          XCTAssertEqual(telemetry.uniqueCount, 3)
          XCTAssertEqual(telemetry.duplicateCount, 3)
          XCTAssertEqual(telemetry.checkTimeNanos, 1234)
          XCTAssertEqual(telemetry.deduplicationRate, 0.5, accuracy: 1e-9)
      }
  }
  ```

- [ ] **Step 13: Clean build.**
  ```bash
  swift build 2>&1 | tail -5
  ```
  Expected final line: `Build complete!`

- [ ] **Step 14: Run the new test file — expect all 8 green.**
  ```bash
  swift test --filter TelemetryRecorderTests 2>&1 | tail -20
  ```
  Expected final lines:
  ```
  Test Suite 'TelemetryRecorderTests' passed at ...
       Executed 8 tests, with 0 failures (0 unexpected) in ...
  ```

- [ ] **Step 15: Full test suite green.**
  ```bash
  swift test 2>&1 | tail -15
  ```
  Expected: `Executed N tests, with 0 failures` (N = the Step 7 count + 8 new tests).

- [ ] **Step 16: Commit 2.**
  ```bash
  git add Sources/VectorIndex/Operations/RangeQuery/RangeQuery.swift \
    Tests/VectorIndexTests/TelemetryRecorderTests.swift
  git commit -m "$(cat <<'EOF'
  feat(telemetry): wire RangeQuery's discarded telemetry into a push recorder

  RangeScanTelemetry was built correctly at all 3 call sites in RangeQuery.swift
  but thrown away by a literal no-op recordTelemetry() ("TODO: Wire to global
  telemetry"). Replaced with RangeScanTelemetryRecorder, matching the sink+record
  pattern already used by Cosine/InnerProduct/L2Sqr/TopK's recorders. Adds
  TelemetryRecorderTests.swift: one hand-computed accuracy test per recorder
  (HNSW, LayoutTransforms/Global, Cosine, InnerProduct, L2Sqr, TopK, RangeQuery)
  plus one for CandidateDedup's pull-style getTelemetry() API -- the first tests
  this telemetry surface has ever had.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  git status --short
  ```
  Expected: commit succeeds; `git status --short` prints nothing (clean tree).

### Task 4: Delete no-op/dead kernel helpers (B5 + B7 + B20 + B6 + B10a + dead `sumSquares`)

**Files:**
- Modify `Sources/VectorIndex/Operations/Scoring/L2Sqr.swift` (L22-25 guard, L65-85 `DispatchBK` enum; 102 lines total)
- Modify `Sources/VectorIndex/Operations/Scoring/L2SqrKernel.swift` (L60-81 helper defs, L111-114 + L281 call sites; 503 lines total)
- Modify `Sources/VectorIndex/Operations/Quantization/PQLUT.swift` (L35-42 `_prefetch` def, 6 call sites at L247/257/268/276/363/407)
- Modify `Sources/VectorIndex/Kernels/ResidualKernel.swift` (L115-121 `_prefetchRead` def, guarded prefetch blocks in both `residuals_f32` and `residuals_f32_inplace`)
- Modify `Sources/VectorIndex/Kernels/KMeansMiniBatchKernel.swift` (L192-207 `_vi_km12_prefetch` def, L349-382 `_vi_km12_assignAOS_tiled`, L539 dead `tile` local, call sites L572/721/804)
- Modify `Sources/VectorIndex/HNSWIndex.swift` (L1168-1203 dead `selectNeighbors` extension; file is 1357 lines)
- Modify `Sources/VectorIndex/Kernels/HNSWTraversal.swift` (L99 dead `selectBatchSize`)
- Modify `Sources/VectorIndex/Operations/Scoring/ScoreBlock.swift` (L66-91 dead `sumSquares` + orphaned `load4`; file is 93 lines)

**Interfaces:** Consumes: none new. Produces: `_vi_km12_assignAOS(xVec:C:kc:d:) -> (cBest: Int, distBest: Float)` (internal, replaces `_vi_km12_assignAOS_tiled(xVec:C:kc:d:tile:)`). All other symbols touched are `internal`/`private`/`@usableFromInline internal` — no `public` symbol is deleted or changed in this task.

This task has no test/coverage risk on its own (every symbol removed is either a hard-coded-`false` sham, a documented no-op, or has zero callers, per the research brief) — the verification step just needs to prove the deletions didn't typo-break anything that compiles and that the broad kernel/HNSW suites still pass.

- [ ] **Step 1: Confirm branch.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex branch --show-current
  ```
  Expected output: `gifton/cleanup-0.2.0-phase2`

- [ ] **Step 2: `L2Sqr.swift` — delete the `DispatchBK` no-op guard inside `run`.**
  In `Sources/VectorIndex/Operations/Scoring/L2Sqr.swift`, delete this exact block (currently lines 22-26, immediately before `// Delegate to the microkernel implementation`):
  ```swift
                  // Future: dispatch to BK kernels if available (no-op placeholders remain)
                  if DispatchBK.dispatchIfAvailable(q: q, xb: xb, n: n, d: d, out: out, xb_norm: xb_norm, q_norm: q_norm) {
                      return
                  }

  ```
  Result: `run`'s body goes straight from `if d == 0 { for i in 0..<n { out[i] = 0 } ; return }` to `// Delegate to the microkernel implementation`.

- [ ] **Step 3: `L2Sqr.swift` — delete the now-orphaned `DispatchBK` enum.**
  Delete this exact block (currently lines 64-85, between the closing `}` of `Telemetry` and the `// MARK: - Scalar reference (testing)` comment):
  ```swift

              // MARK: BK dispatch hooks (placeholders)
              enum DispatchBK {
                  @inline(__always) static var hasBK512: Bool { false }
                  @inline(__always) static var hasBK768: Bool { false }
                  @inline(__always) static var hasBK1024: Bool { false }
                  @inline(__always) static var hasBK1536: Bool { false }

                  @inline(__always)
                  static func dispatchIfAvailable(
                      q: UnsafePointer<Float>, xb: UnsafePointer<Float>, n: Int, d: Int,
                      out: UnsafeMutablePointer<Float>, xb_norm: UnsafePointer<Float>?, q_norm: Float?
                  ) -> Bool {
                      switch d {
                      case 512 where hasBK512: return false
                      case 768 where hasBK768: return false
                      case 1024 where hasBK1024: return false
                      case 1536 where hasBK1536: return false
                      default: return false
                      }
                  }
              }
  ```
  (`DispatchBK` is nested inside plain `enum L2Sqr { ... }`, not directly inside the `public extension`, so it is `internal` by Swift's default-access rule — safe to delete outright, not a public-API deletion.)

- [ ] **Step 4: `L2SqrKernel.swift` — delete the `_verifyAlignment`/`_prefetchRow` no-op definitions.**
  Delete this exact block (currently lines 60-82):
  ```swift
  // MARK: - Helper: Alignment verification (performance hint)
  //
  // Note: 16-byte alignment enables optimal SIMD performance, but unaligned data
  // is handled correctly (just slower). We verify alignment in debug builds for
  // performance profiling, but don't enforce it since Swift [Float] arrays are
  // not guaranteed to be aligned. The SIMD4<Float> operations handle unaligned
  // loads correctly, they're just slightly slower due to extra memory ops.

  @inline(__always)
  func _verifyAlignment(_ ptr: UnsafeRawPointer?, _ label: String, alignment: Int = 16) {
      // Alignment check removed - Swift arrays are not guaranteed to be aligned,
      // and SIMD4 operations handle unaligned data correctly (with minor perf impact)
      // For optimal performance, users can pre-align data using:
      //   UnsafeMutableRawBufferPointer.allocate(byteCount:alignment:)
  }

  // MARK: - Helper: Prefetch (hint-only; no-op on Swift)

  @inline(__always)
  func _prefetchRow(_ base: UnsafeRawPointer, _ byteStride: Int) {
      _ = base; _ = byteStride
  }
  ```
  **Do not touch** `PQLUT.swift`'s own `_verifyAlignment(_:_:_:)` (different file, different parameter order, has a real DEBUG-only precondition body — out of scope here). These two functions share a name but are separate top-level `internal func`s in separate files; only the `L2SqrKernel.swift` copy is a genuine no-op.

- [ ] **Step 5: `L2SqrKernel.swift` — delete the 4 `_verifyAlignment` call sites in `l2sqr_f32_block`.**
  Delete this exact block (currently lines 111-115, the first 4 statements of `l2sqr_f32_block`'s body, immediately before `guard n > 0, d > 0 else { return }`):
  ```swift
      _verifyAlignment(q, "q")
      _verifyAlignment(xb, "xb")
      _verifyAlignment(out, "out")
      if let xbNorm = xb_norm { _verifyAlignment(xbNorm, "xb_norm") }

  ```

- [ ] **Step 6: `L2SqrKernel.swift` — remove the `_prefetchRow` call site and its now-dead locals in `_l2sqr_direct_generic`.**
  Replace:
  ```swift
  func _l2sqr_direct_generic(
      _ q: UnsafePointer<Float>,
      _ xb: UnsafePointer<Float>,
      _ n: Int, _ d: Int,
      _ out: UnsafeMutablePointer<Float>,
      _ strict: Bool,
      _ prefetchDistance: Int
  ) {
      let rowBytes = d * MemoryLayout<Float>.stride
      for i in 0..<n {
          // Prefetch next row (hint only)
          let pfRow = i + prefetchDistance
          if pfRow < n { _prefetchRow(UnsafeRawPointer(xb + pfRow * d), rowBytes) }
          let x = xb + i * d
          out[i] = _l2sqr_single_direct(q: q, x: x, d: d, kahan: strict)
      }
  }
  ```
  with:
  ```swift
  func _l2sqr_direct_generic(
      _ q: UnsafePointer<Float>,
      _ xb: UnsafePointer<Float>,
      _ n: Int, _ d: Int,
      _ out: UnsafeMutablePointer<Float>,
      _ strict: Bool,
      _ prefetchDistance: Int
  ) {
      for i in 0..<n {
          let x = xb + i * d
          out[i] = _l2sqr_single_direct(q: q, x: x, d: d, kahan: strict)
      }
  }
  ```
  (`prefetchDistance` stays in the signature — it's called positionally from `_l2sqr_block_direct` at two sites — but is now unused in the body; Swift does not warn on unused function parameters, so this stays warning-free.)

- [ ] **Step 7: `PQLUT.swift` — delete the `_prefetch` no-op definition.**
  Delete this exact block (currently lines 34-42, leaves `_verifyAlignment`'s closing `}` directly followed by a blank line then `// MARK: - Options`):
  ```swift

  @inline(__always)
  @usableFromInline
  internal func _prefetch(_ ptr: UnsafeRawPointer?) {
      // Advisory only; left as a no-op in portable Swift.
      // On Apple Silicon, compilers often auto-prefetch. Spec includes the
      // knob for tuning, so we keep the API for future inline asm if desired.
      _ = ptr
  }
  ```

- [ ] **Step 8: `PQLUT.swift` — delete all 6 call-site guards in one `replace_all` edit.**
  This exact 3-line block appears 6 times, byte-identical, at (current) lines 246-248, 256-258, 267-269, 275-277, 362-364, 406-408 — use `replace_all: true` with `new_string: ""`:
  ```swift
                      if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                          _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                      }
  ```
  After the edit, verify the replacement count:
  ```bash
  grep -c "_prefetch(" /Users/goftin/dev/gsuite/VSK/VectorIndex/Sources/VectorIndex/Operations/Quantization/PQLUT.swift
  ```
  Expected output: `0`

- [ ] **Step 9: `ResidualKernel.swift` — delete the `_prefetchRead` no-op definition.**
  Delete this exact block (currently lines 114-121, leaves `_storeSIMD4`'s closing `}` directly followed by a blank line then `// MARK: - Core residual kernels`):
  ```swift

  @usableFromInline
  @inline(__always)
  internal func _prefetchRead(_ ptr: UnsafeRawPointer?) {
      // Swift does not expose a portable prefetch intrinsic.
      // Left intentionally as a no-op to keep option parity with spec.
      _ = ptr
  }
  ```

- [ ] **Step 10: `ResidualKernel.swift` — delete the dead prefetch block in `residuals_f32`.**
  Replace:
  ```swift
      // Ungrouped / original order
      let pd = max(0, opts.prefetchDistance)
      let nInt = Int(n)
      let d8 = (d / 8) * 8

      for i in 0..<nInt {
          // Prefetch next vector & centroid (advisory)
          if pd > 0 {
              let f = i + pd
              if f < nInt {
                  _prefetchRead(UnsafeRawPointer(x.advanced(by: f * d)))
                  let fa = Int(coarseIDs[f])
                  if !opts.checkBounds || (fa >= 0 && fa < opts.kc) {
                      _prefetchRead(UnsafeRawPointer(coarseCentroids.advanced(by: fa * d)))
                  }
              }
          }

          let a = Int(coarseIDs[i])
  ```
  with:
  ```swift
      // Ungrouped / original order
      let nInt = Int(n)
      let d8 = (d / 8) * 8

      for i in 0..<nInt {
          let a = Int(coarseIDs[i])
  ```
  (`pd`, `f`, `fa` were computed only to feed the no-op `_prefetchRead`; all three are dropped together to avoid an unused-variable warning.)

- [ ] **Step 11: `ResidualKernel.swift` — delete the identical dead prefetch block in `residuals_f32_inplace`.**
  Replace:
  ```swift
      let pd = max(0, opts.prefetchDistance)
      let nInt = Int(n)
      let d8 = (d / 8) * 8

      for i in 0..<nInt {
          if pd > 0 {
              let f = i + pd
              if f < nInt {
                  _prefetchRead(UnsafeRawPointer(x_io.advanced(by: f * d)))
                  let fa = Int(coarseIDs[f])
                  if !opts.checkBounds || (fa >= 0 && fa < opts.kc) {
                      _prefetchRead(UnsafeRawPointer(coarseCentroids.advanced(by: fa * d)))
                  }
              }
          }

          let a = Int(coarseIDs[i])
  ```
  with:
  ```swift
      let nInt = Int(n)
      let d8 = (d / 8) * 8

      for i in 0..<nInt {
          let a = Int(coarseIDs[i])
  ```

- [ ] **Step 12: `KMeansMiniBatchKernel.swift` — delete the `_vi_km12_prefetch` no-op definition.**
  Delete this exact block (currently lines 192-207, leaves `_vi_km12_sum4`'s closing `}` directly followed by a blank line then the `/// L2 squared distance...` doc comment):
  ```swift

  /// Prefetch hint (no-op for now; infrastructure for future platform-specific optimization)
  ///
  /// Future: Could use __builtin_prefetch (ARM) or _mm_prefetch (x86) via C bridge.
  /// Current: Documentation placeholder to preserve API surface.
  @usableFromInline
  @inline(__always)
  internal func _vi_km12_prefetch(_ ptr: UnsafeRawPointer?) {
      _ = ptr
      // TODO: Implement platform-specific prefetch
      // #if arch(arm64)
      //   __builtin_prefetch(ptr, 0, 0)
      // #elseif arch(x86_64)
      //   _mm_prefetch(ptr, _MM_HINT_T0)
      // #endif
  }
  ```

- [ ] **Step 13: `KMeansMiniBatchKernel.swift` — collapse `_vi_km12_assignAOS_tiled` into a flat `_vi_km12_assignAOS` scan (B20).**
  Replace:
  ```swift
  // MARK: - Assignment (tiling over centroids)

  /// Assign vector to nearest centroid using tiled scan
  ///
  /// Cache-friendly: processes centroids in tiles of size `tile` to maintain
  /// hot data in L1/L2 cache. Tile size 32 chosen empirically for typical d.
  @usableFromInline
  internal func _vi_km12_assignAOS_tiled(
      xVec: UnsafePointer<Float>,
      C: UnsafePointer<Float>, kc: Int, d: Int,
      tile: Int
  ) -> (cBest: Int, distBest: Float) {
      var cBest = 0
      var distBest = _vi_km12_l2sq_aos(xVec, C, d)

      var c = 1
      while c < kc {
          let end = min(c + tile, kc)
          // (Optional) prefetch next tile
          if end < kc {
              _vi_km12_prefetch(UnsafeRawPointer(C.advanced(by: end * d)))
          }
          while c < end {
              let dist = _vi_km12_l2sq_aos(xVec, C.advanced(by: c * d), d)
              // Deterministic tie-breaking: prefer lower centroid index
              if dist < distBest || (dist == distBest && c < cBest) {
                  distBest = dist
                  cBest = c
              }
              c += 1
          }
      }
      return (cBest, distBest)
  }
  ```
  with:
  ```swift
  // MARK: - Assignment (flat scan; tiling removed — see Phase 2 B20 cleanup)

  /// Assign vector to nearest centroid via a flat linear scan over all centroids.
  ///
  /// This used to be tiled with an inter-tile prefetch hint, but the prefetch
  /// (`_vi_km12_prefetch`) was a documented no-op, so the tiling added loop
  /// overhead with no cache benefit. Collapsed to a plain scan; same result.
  @usableFromInline
  internal func _vi_km12_assignAOS(
      xVec: UnsafePointer<Float>,
      C: UnsafePointer<Float>, kc: Int, d: Int
  ) -> (cBest: Int, distBest: Float) {
      var cBest = 0
      var distBest = _vi_km12_l2sq_aos(xVec, C, d)

      var c = 1
      while c < kc {
          let dist = _vi_km12_l2sq_aos(xVec, C.advanced(by: c * d), d)
          // Deterministic tie-breaking: prefer lower centroid index
          if dist < distBest || (dist == distBest && c < cBest) {
              distBest = dist
              cBest = c
          }
          c += 1
      }
      return (cBest, distBest)
  }
  ```

- [ ] **Step 14: `KMeansMiniBatchKernel.swift` — update the 3 call sites to the new flat signature.**
  Edit 1 (currently line 572):
  ```swift
                          return _vi_km12_assignAOS_tiled(xVec: vec, C: centroidsOut, kc: kc, d: d, tile: tile)
  ```
  becomes:
  ```swift
                          return _vi_km12_assignAOS(xVec: vec, C: centroidsOut, kc: kc, d: d)
  ```
  Edit 2 (currently line 721):
  ```swift
                      return _vi_km12_assignAOS_tiled(xVec: vec, C: centroidsOut, kc: kc, d: d, tile: 32)
  ```
  becomes:
  ```swift
                      return _vi_km12_assignAOS(xVec: vec, C: centroidsOut, kc: kc, d: d)
  ```
  Edit 3 (currently line 804):
  ```swift
                  return _vi_km12_assignAOS_tiled(xVec: vec, C: state.centroids.withUnsafeBufferPointer { $0.baseAddress! }, kc: kc, d: d, tile: 32)
  ```
  becomes:
  ```swift
                  return _vi_km12_assignAOS(xVec: vec, C: state.centroids.withUnsafeBufferPointer { $0.baseAddress! }, kc: kc, d: d)
  ```

- [ ] **Step 15: `KMeansMiniBatchKernel.swift` — delete the now-dead `tile` local that fed call site 1.**
  Delete this exact 2-line block (currently lines 539-540, the only use of `tile` was call site 1 above):
  ```swift
          let tile = 32  // Cache tile size (empirically optimal for d ∈ [128, 2048])

  ```
  Confirm no leftover references:
  ```bash
  grep -n "_vi_km12_assignAOS_tiled\|_vi_km12_prefetch\b" /Users/goftin/dev/gsuite/VSK/VectorIndex/Sources/VectorIndex/Kernels/KMeansMiniBatchKernel.swift
  ```
  Expected output: (empty — no matches)

- [ ] **Step 16: `HNSWIndex.swift` — delete the dead `selectNeighbors` extension (B6).**
  Delete this exact block (currently lines 1168-1203; confirmed the `private extension HNSWIndex { ... }` opened at line 1169 contains only this one method, so the whole extension is deletable, leaving the `}` of the previous extension directly followed by a blank line then `// MARK: - AccelerableIndex Implementation`):
  ```swift

  // MARK: - Neighbor selection heuristic
  private extension HNSWIndex {
      // Select up to maxM diverse neighbors among candidate node indices for a given vector at level.
      // Implements a simple diversity heuristic from HNSW: candidates are considered in increasing
      // distance order; a candidate is selected if it is closer to the new point than to any already
      // selected neighbor, promoting angular diversity.
      func selectNeighbors(for vec: [Float], among candidates: [Int], level: Int, maxM: Int) -> [Int] {
          // Sort candidates by distance to vec
          var sorted: [(Int, Float)] = candidates.map { ($0, distance(vec, vectorArray(at: $0), metric: metric)) }
          sorted.sort { $0.1 < $1.1 }
          var selected: [Int] = []
          selected.reserveCapacity(min(maxM, sorted.count))
          for (cand, _) in sorted {
              var good = true
              let candVec = vectorArray(at: cand)
              for s in selected {
                  // If candidate is much closer to an already selected neighbor than to the new point,
                  // skip it (encourage spread). Criterion: d(cand, s) < d(cand, new)
                  let d_cs = distance(candVec, vectorArray(at: s), metric: metric)
                  let d_cx = distance(candVec, vec, metric: metric)
                  if d_cs < d_cx { good = false; break }
              }
              if good { selected.append(cand) }
              if selected.count >= maxM { break }
          }
          // Fallback: if too few selected, fill with nearest remaining
          if selected.count < maxM {
              for (cand, _) in sorted where !selected.contains(cand) {
                  selected.append(cand)
                  if selected.count >= maxM { break }
              }
          }
          return selected
      }
  }
  ```
  (Real neighbor selection during insertion already goes through `hnsw_select_neighbors_f32_swift` — Kernel #34 — called from `HNSWIndex.swift:627,709`; this free function was superseded and had zero callers.)

- [ ] **Step 17: `HNSWTraversal.swift` — delete the dead `selectBatchSize` (B10a).**
  Delete this exact line (currently line 99, immediately before `private func scoreNeighborsBatch_f32(`):
  ```swift
  @inline(__always) private func selectBatchSize(_ d: Int) -> Int { if d <= 256 { return 64 }; if d <= 1024 { return 32 }; return 16 }
  ```
  **Do not touch** `HNSWNeighborSelection.swift`'s `ns_selectBatchSize` — that one is called at `HNSWNeighborSelection.swift:156` and is live.

- [ ] **Step 18: `ScoreBlock.swift` — delete the dead `sumSquares` and the now-orphaned `load4`.**
  Delete this exact block (currently lines 66-91, leaves `run`'s closing `}` directly followed by a blank line then the two closing `}` of `enum ScoreBlock` / the extension):
  ```swift

          @inline(__always)
          private static func sumSquares(ptr: UnsafePointer<Float>, d: Int) -> Float {
              let d4 = d & ~3
              var acc = SIMD4<Float>.zero
              var j = 0
              while j < d4 {
                  let v = load4(ptr.advanced(by: j))
                  acc += v * v
                  j += 4
              }
              var s = acc[0] + acc[1] + acc[2] + acc[3]
              if d - d4 >= 1 { s += ptr[d4+0] * ptr[d4+0] }
              if d - d4 >= 2 { s += ptr[d4+1] * ptr[d4+1] }
              if d - d4 >= 3 { s += ptr[d4+2] * ptr[d4+2] }
              return s
          }

          @inline(__always)
          private static func load4(_ p: UnsafePointer<Float>) -> SIMD4<Float> {
              if Int(bitPattern: p) & (MemoryLayout<SIMD4<Float>>.alignment - 1) == 0 {
                  return p.withMemoryRebound(to: SIMD4<Float>.self, capacity: 1) { $0.pointee }
              } else {
                  return SIMD4<Float>(p[0], p[1], p[2], p[3])
              }
          }
  ```
  (`sumSquares` has zero callers in this file — `run` never calls it, per the brief's grep. `load4` was only ever called by `sumSquares`, so it becomes dead too once `sumSquares` is gone; both go together.)

- [ ] **Step 19: Build.**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build --build-tests
  ```
  Expected output ends with: `Build complete!` (no errors; pre-existing unrelated warnings such as the `VIndexMmap.swift:366` `withUnsafeBytes` unused-result warning are fine).

- [ ] **Step 20: Run the named test filters (foreground).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --skip-build --filter '^VectorIndexTests\.(MicrokernelIntegrationTests|ResidualKernelTests|KMeansMiniBatchTests|HNSWTests|HNSWRecallTests|HNSWKNNGraphTests|HNSWParamSweepTests|HNSWMoreTests|HNSWBatchAndErrorsTests|HNSWNeighborSelectionTests|HNSWTraversalKernelTests|RegressionA4_RerankIDWidthTests)'
  ```
  Expected output ends with (measured baseline before this task's edits — takes ~5 minutes; counts must match after the edits since no test was added or removed):
  ```
  Test Suite 'Selected tests' passed at ...
  	 Executed 54 tests, with 2 tests skipped and 0 failures (0 unexpected) in ... seconds
  ```

- [ ] **Step 21: Commit.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex add \
    Sources/VectorIndex/Operations/Scoring/L2Sqr.swift \
    Sources/VectorIndex/Operations/Scoring/L2SqrKernel.swift \
    Sources/VectorIndex/Operations/Quantization/PQLUT.swift \
    Sources/VectorIndex/Kernels/ResidualKernel.swift \
    Sources/VectorIndex/Kernels/KMeansMiniBatchKernel.swift \
    Sources/VectorIndex/HNSWIndex.swift \
    Sources/VectorIndex/Kernels/HNSWTraversal.swift \
    Sources/VectorIndex/Operations/Scoring/ScoreBlock.swift
  ```
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex commit -m "$(cat <<'EOF'
  refactor(kernels): delete dead/no-op scoring & kernel helpers (B5,B7,B20,B6,B10a)

  Removes code with zero behavioral effect, confirmed by grep across
  Sources/+Tests/ for each symbol:
  - L2Sqr.DispatchBK: hardcoded-false dispatch enum, always falls through (B5)
  - L2SqrKernel._verifyAlignment/_prefetchRow: empty-bodied no-ops (B5)
  - PQLUT._prefetch, ResidualKernel._prefetchRead,
    KMeansMiniBatch._vi_km12_prefetch: three identical no-op prefetch hints (B7)
  - KMeansMiniBatch's tiled centroid-assignment wrapper collapsed to a flat
    scan now that its only "benefit" (the prefetch call) is gone (B20)
  - HNSWIndex.selectNeighbors: superseded, zero-caller array-based neighbor
    heuristic; hnsw_select_neighbors_f32_swift is the live path (B6)
  - HNSWTraversal.selectBatchSize: zero-caller dead code, unlike its live
    HNSWNeighborSelection.ns_selectBatchSize sibling (B10a)
  - ScoreBlock.sumSquares (+ its only caller, ScoreBlock.load4): unreachable

  All touched symbols are internal/private/@usableFromInline internal; no
  public API changes. Non-breaking per Phase 2 policy.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  ```

---

### Task 5: B3 — `MIPSTransform.swift` internal-helper consolidation + deprecate dead public API

**Files:** Modify `Sources/VectorIndex/Operations/Transform/MIPSTransform.swift` (409 lines total)

**Interfaces:** Consumes: `IndexOps.Support.Norms.l2NormSquared(vector x: UnsafePointer<Float>, dimension d: Int) -> Float` (`Sources/VectorIndex/Operations/Support/Norms.swift:105`), `IndexOps.Scoring.InnerProduct.run(q:xb:n:d:out:stride:options:)` (`Sources/VectorIndex/Operations/Scoring/InnerProduct.swift:8-16`), `l2sqr_f32_block(_:_:_:_:_:_:_:_:)` (`Sources/VectorIndex/Operations/Scoring/L2SqrKernel.swift:101-110`). Produces: none new — every public declaration in the file keeps its existing signature, only gaining an `@available(*, deprecated, ...)` attribute.

**Premise correction (per brief):** this isn't just 4 dead helpers — the *entire public API* of this file (`MIPSTransformMode`, `R2Parameter`, `AugmentedVectorStorage`, `MIPSTransformTelemetry`, `computeR2Parameter`, `mipsMaterializeAugmentation`, `mipsAugmentQuery`, `mipsVirtualToL2Scores`, `mipsHybridScoreBlock`) has **zero callers outside this file and zero tests** (`grep -rli "mips" Tests/` → no hits at all). The spec scoped B3 narrowly ("delete these 4 helpers, route to canonical kernels"); this task does exactly that scoped rewire, and — since Phase 2 forbids deleting `public` symbols — additionally marks the file's whole dead public surface `@available(*, deprecated)` rather than leaving it silently rotting. Full removal is Phase 4's job (see the `PHASE4-ROUTING` block at the end of this fragment). No test file exists for MIPS and none is added — the verification for this task is build success + the full existing suite staying green, explicitly (there is nothing dead-code-specific to assert).

- [ ] **Step 1: Confirm branch.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex branch --show-current
  ```
  Expected output: `gifton/cleanup-0.2.0-phase2`

- [ ] **Step 2: Delete the 4 dead/sham helpers in one shot (contiguous tail of the file).**
  In `Sources/VectorIndex/Operations/Transform/MIPSTransform.swift`, delete this exact block — everything from the `l2NormSquaredSIMD` MARK comment through end-of-file (currently lines 266-409/410, i.e. `l2NormSquaredSIMD`, `innerProductSIMD`, the `l2sqrBlock_dispatch` sham, and its `generic_l2sqrBlock` fallback):
  ```swift
  // MARK: - Helper: SIMD L2 norm (sum of squares)

  @inlinable
  internal func l2NormSquaredSIMD(_ x: UnsafePointer<Float>, _ d: Int) -> Float {
      let w = 4
      let dv = (d / w) * w
      var a0 = SIMD4<Float>(repeating: 0), a1 = SIMD4<Float>(repeating: 0),
          a2 = SIMD4<Float>(repeating: 0), a3 = SIMD4<Float>(repeating: 0)
      var j = 0
      while j + 15 < d {
          let v0 = SIMD4<Float>(x[j + 0], x[j + 1], x[j + 2], x[j + 3])
          let v1 = SIMD4<Float>(x[j + 4], x[j + 5], x[j + 6], x[j + 7])
          let v2 = SIMD4<Float>(x[j + 8], x[j + 9], x[j + 10], x[j + 11])
          let v3 = SIMD4<Float>(x[j + 12], x[j + 13], x[j + 14], x[j + 15])
          // FIXED: Use regular operators, not wrapping operators
          a0 += v0 * v0
          a1 += v1 * v1
          a2 += v2 * v2
          a3 += v3 * v3
          j += 16
      }
      let combined = a0 + a1 + a2 + a3
      var acc = combined[0] + combined[1] + combined[2] + combined[3]
      while j < dv {
          let v = SIMD4<Float>(x[j + 0], x[j + 1], x[j + 2], x[j + 3])
          let vSq = v * v
          acc += vSq[0] + vSq[1] + vSq[2] + vSq[3]
          j += w
      }
      while j < d {
          let v = x[j]
          acc += v * v
          j += 1
      }
      return acc
  }

  // MARK: - Helper: SIMD inner product (dot)

  @inlinable
  internal func innerProductSIMD(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
      let w = 4
      let dv = (d / w) * w
      var a0 = SIMD4<Float>(repeating: 0), a1 = SIMD4<Float>(repeating: 0),
          a2 = SIMD4<Float>(repeating: 0), a3 = SIMD4<Float>(repeating: 0)
      var j = 0
      while j + 15 < d {
          let q0 = SIMD4<Float>(a[j + 0], a[j + 1], a[j + 2], a[j + 3])
          let x0 = SIMD4<Float>(b[j + 0], b[j + 1], b[j + 2], b[j + 3])
          let q1 = SIMD4<Float>(a[j + 4], a[j + 5], a[j + 6], a[j + 7])
          let x1 = SIMD4<Float>(b[j + 4], b[j + 5], b[j + 6], b[j + 7])
          let q2 = SIMD4<Float>(a[j + 8], a[j + 9], a[j + 10], a[j + 11])
          let x2 = SIMD4<Float>(b[j + 8], b[j + 9], b[j + 10], b[j + 11])
          let q3 = SIMD4<Float>(a[j + 12], a[j + 13], a[j + 14], a[j + 15])
          let x3 = SIMD4<Float>(b[j + 12], b[j + 13], b[j + 14], b[j + 15])
          // FIXED: Use regular operators, not wrapping operators
          a0 += q0 * x0
          a1 += q1 * x1
          a2 += q2 * x2
          a3 += q3 * x3
          j += 16
      }
      let combined = a0 + a1 + a2 + a3
      var acc = combined[0] + combined[1] + combined[2] + combined[3]
      while j < dv {
          let qv = SIMD4<Float>(a[j + 0], a[j + 1], a[j + 2], a[j + 3])
          let xv = SIMD4<Float>(b[j + 0], b[j + 1], b[j + 2], b[j + 3])
          let prod = qv * xv
          acc += prod[0] + prod[1] + prod[2] + prod[3]
          j += w
      }
      while j < d {
          acc += a[j] * b[j]
          j += 1
      }
      return acc
  }

  // MARK: - L2² block dispatcher (prefers high-perf kernel #01 when present)

  /// If your project already includes the L2 microkernel (#01) this shim will
  /// naturally inline to it; otherwise a generic fallback is used.
  @inlinable
  internal func l2sqrBlock_dispatch(
      query: UnsafePointer<Float>,
      database: UnsafePointer<Float>,
      vectorCount n: Int,
      dimension d: Int,
      output: UnsafeMutablePointer<Float>
  ) {
      // Try to use the real kernel #01 if available
      #if canImport(VectorCore)
      // Wire to your actual entrypoint if available
      generic_l2sqrBlock(query: query, database: database, n: n, d: d, out: output)
      #else
      // Fall back to generic implementation
      generic_l2sqrBlock(query: query, database: database, n: n, d: d, out: output)
      #endif
  }

  @inlinable
  internal func generic_l2sqrBlock(
      query q: UnsafePointer<Float>,
      database xb: UnsafePointer<Float>,
      n: Int,
      d: Int,
      out: UnsafeMutablePointer<Float>
  ) {
      // Simple, cache-friendly row loop with SIMD inner diff² accumulation
      for i in 0..<n {
          let x = xb + i * d
          var acc0 = SIMD4<Float>(repeating: 0), acc1 = SIMD4<Float>(repeating: 0),
              acc2 = SIMD4<Float>(repeating: 0), acc3 = SIMD4<Float>(repeating: 0)
          var j = 0
          while j + 15 < d {
              let q0 = SIMD4<Float>(q[j + 0], q[j + 1], q[j + 2], q[j + 3])
              let x0 = SIMD4<Float>(x[j + 0], x[j + 1], x[j + 2], x[j + 3])
              let q1 = SIMD4<Float>(q[j + 4], q[j + 5], q[j + 6], q[j + 7])
              let x1 = SIMD4<Float>(x[j + 4], x[j + 5], x[j + 6], x[j + 7])
              let q2 = SIMD4<Float>(q[j + 8], q[j + 9], q[j + 10], q[j + 11])
              let x2 = SIMD4<Float>(x[j + 8], x[j + 9], x[j + 10], x[j + 11])
              let q3 = SIMD4<Float>(q[j + 12], q[j + 13], q[j + 14], q[j + 15])
              let x3 = SIMD4<Float>(x[j + 12], x[j + 13], x[j + 14], x[j + 15])
              // FIXED: Use regular operators, not wrapping operators
              let d0 = q0 - x0
              let d1 = q1 - x1
              let d2 = q2 - x2
              let d3 = q3 - x3
              acc0 += d0 * d0
              acc1 += d1 * d1
              acc2 += d2 * d2
              acc3 += d3 * d3
              j += 16
          }
          let combined = acc0 + acc1 + acc2 + acc3
          var s = combined[0] + combined[1] + combined[2] + combined[3]
          while j < d {
              let diff = q[j] - x[j]
              s += diff * diff
              j += 1
          }
          out[i] = s
      }
  }
  ```
  New content: (nothing — the file now ends at `mipsHybridScoreBlock`'s closing `}`).

- [ ] **Step 3: Rewire `computeR2Parameter` to the canonical norm kernel.**
  Replace:
  ```swift
      for i in 0..<n {
          let row = vectors + i * d
          let s = l2NormSquaredSIMD(row, d)
          if s > maxSq { maxSq = s }
      }
  ```
  with:
  ```swift
      for i in 0..<n {
          let row = vectors + i * d
          let s = IndexOps.Support.Norms.l2NormSquared(vector: row, dimension: d)
          if s > maxSq { maxSq = s }
      }
  ```

- [ ] **Step 4: Rewire `mipsMaterializeAugmentation` to the canonical norm kernel.**
  Replace:
  ```swift
          // Compute sqrt(max(0, R² - ‖x‖²)) for slot d
          let normSq = l2NormSquaredSIMD(x, d)
  ```
  with:
  ```swift
          // Compute sqrt(max(0, R² - ‖x‖²)) for slot d
          let normSq = IndexOps.Support.Norms.l2NormSquared(vector: x, dimension: d)
  ```

- [ ] **Step 5: Rewire `mipsVirtualToL2Scores` to the canonical norm + inner-product kernels.**
  Replace:
  ```swift
  @inlinable
  public func mipsVirtualToL2Scores(
      query: UnsafePointer<Float>,
      baseVectors: UnsafePointer<Float>,
      count n: Int,
      dimension d: Int,
      r2: R2Parameter,
      scoresOut: UnsafeMutablePointer<Float>
  ) {
      let qSq = l2NormSquaredSIMD(query, d)
      let r2v = r2.value
      // Compute dot and apply fused epilogue (no temporary buffer)
      for i in 0..<n {
          let x = baseVectors + i * d
          let dot = innerProductSIMD(query, x, d)
          scoresOut[i] = qSq + r2v - 2.0 * dot
      }
  }
  ```
  with:
  ```swift
  @inlinable
  public func mipsVirtualToL2Scores(
      query: UnsafePointer<Float>,
      baseVectors: UnsafePointer<Float>,
      count n: Int,
      dimension d: Int,
      r2: R2Parameter,
      scoresOut: UnsafeMutablePointer<Float>
  ) {
      let qSq = IndexOps.Support.Norms.l2NormSquared(vector: query, dimension: d)
      let r2v = r2.value
      // Batch-compute dot products via the canonical kernel, then apply the
      // fused epilogue in place (no temporary buffer needed).
      IndexOps.Scoring.InnerProduct.run(q: query, xb: baseVectors, n: n, d: d, out: scoresOut)
      for i in 0..<n {
          scoresOut[i] = qSq + r2v - 2.0 * scoresOut[i]
      }
  }
  ```

- [ ] **Step 6: Rewire `mipsHybridScoreBlock` to the canonical L2² kernel.**
  Replace:
  ```swift
          // Prefer your high-performance L2 microkernel (#01) if available; otherwise fallback.
          l2sqrBlock_dispatch(
              query: augQ,
              database: augBase,
              vectorCount: n,
              dimension: storage.paddedDim,
              output: scoresOut
          )
  ```
  with:
  ```swift
          // Canonical L2² microkernel (#01).
          l2sqr_f32_block(augQ, augBase, n, storage.paddedDim, scoresOut)
  ```

- [ ] **Step 7: Deprecate the file's 4 dead public types.**
  ```swift
  @frozen
  public enum MIPSTransformMode: Sendable {
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @frozen
  public enum MIPSTransformMode: Sendable {
  ```
  ```swift
  @frozen
  public struct R2Parameter {
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @frozen
  public struct R2Parameter {
  ```
  ```swift
  public struct AugmentedVectorStorage {
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  public struct AugmentedVectorStorage {
  ```
  ```swift
  @frozen
  public struct MIPSTransformTelemetry {
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @frozen
  public struct MIPSTransformTelemetry {
  ```

- [ ] **Step 8: Deprecate the file's 5 dead public functions.**
  ```swift
  @inlinable
  public func computeR2Parameter(
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @inlinable
  public func computeR2Parameter(
  ```
  ```swift
  @inlinable
  public func mipsMaterializeAugmentation(
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @inlinable
  public func mipsMaterializeAugmentation(
  ```
  ```swift
  @inlinable
  public func mipsAugmentQuery(
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @inlinable
  public func mipsAugmentQuery(
  ```
  ```swift
  @inlinable
  public func mipsVirtualToL2Scores(
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @inlinable
  public func mipsVirtualToL2Scores(
  ```
  ```swift
  @inlinable
  public func mipsHybridScoreBlock(
  ```
  becomes:
  ```swift
  @available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
  @inlinable
  public func mipsHybridScoreBlock(
  ```

- [ ] **Step 9: Build (expect new deprecation warnings, zero errors).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build --build-tests 2>&1 | tail -20
  ```
  Expected: several new `warning: 'X' is deprecated` lines for `MIPSTransform.swift` (harmless — this file has no callers so nothing else emits these warnings), ending with `Build complete!`. `Package.swift` has no `-warnings-as-errors`/`-Werror` setting, so these do not fail the build.

- [ ] **Step 10: Run the full test suite (foreground) — no MIPS-specific test exists, so this is the whole verification.**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && CI_SKIP_PQTRAIN=1 swift test --skip-build
  ```
  Expected output ends with: `Executed <N> tests, ... with 0 failures (0 unexpected)` — same pass count as the pre-Task-5 baseline (this task touches zero test-covered call paths; `mipsVirtualToL2Scores`/`mipsHybridScoreBlock` have no tests to regress, and no other file imports anything from `MIPSTransform.swift`).

- [ ] **Step 11: Commit.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex add Sources/VectorIndex/Operations/Transform/MIPSTransform.swift
  ```
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex commit -m "$(cat <<'EOF'
  refactor(mips): route internal helpers to canonical kernels; deprecate dead API (B3)

  l2NormSquaredSIMD/innerProductSIMD/l2sqrBlock_dispatch/generic_l2sqrBlock were
  private duplicates of Norms.l2NormSquared / InnerProduct.run / l2sqr_f32_block,
  called only from within this file. Deleted them and rewired computeR2Parameter,
  mipsMaterializeAugmentation, mipsVirtualToL2Scores, and mipsHybridScoreBlock to
  call the canonical kernels directly.

  Per the Phase 2 research brief, MIPSTransform.swift's entire public API (the
  4 rewired functions plus mipsAugmentQuery, MIPSTransformMode, R2Parameter,
  AugmentedVectorStorage, MIPSTransformTelemetry) has zero callers and zero
  tests anywhere in the repo (grep -rli "mips" Tests/ → no hits). Since Phase 2
  forbids removing public symbols, the whole dead surface is marked
  @available(*, deprecated) instead of deleted; full removal is routed to
  Phase 4 (see PHASE4-ROUTING in the Phase 2 plan).

  No dedicated test exists for this file (none added — nothing here is worth
  testing ahead of its Phase-4 removal); verified via full-suite green + clean
  build.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  ```

---

### Task 6: B8 — one sum-of-squares implementation

**Files:**
- Modify `Sources/VectorIndex/Operations/Scoring/L2SqrKernel.swift` (L484-502, `_normSquared`)
- Modify `Sources/VectorIndex/Operations/Scoring/Cosine.swift` (L183-193 `sumSquares`, L427-441 `precomputeInvNorms`)
- Test `Tests/VectorIndexTests/MicrokernelIntegrationTests.swift` (add 1 new test; currently 47 lines / 2 tests)

**Interfaces:** Consumes: `IndexOps.Support.Norms.l2NormSquared(vector x: UnsafePointer<Float>, dimension d: Int) -> Float` (`Sources/VectorIndex/Operations/Support/Norms.swift:105`, already the canonical impl — no change to this function). Produces: none new — `_normSquared`, `Cosine.sumSquares`, `Cosine.precomputeInvNorms` all keep their existing signatures; bodies only.

**Scope note:** `ScoreBlock.swift`'s duplicate `sumSquares` was dead code (zero callers), already deleted in Task 4 — not touched again here. `MIPSTransform.swift`'s `l2NormSquaredSIMD` duplicate was already deleted in Task 5 (its 3 call sites rewired directly to `Norms.l2NormSquared`) — by the time this task runs, that symbol no longer exists, so there is nothing left to delegate there. That leaves exactly 3 real delegation targets, all in this task.

**Numeric-parity risk (per brief):** `_normSquared`'s old body does a 16-wide SIMD main loop then falls straight to a **scalar** tail loop (no intermediate 4-wide step), whereas `Norms.l2NormSquared` does 16-wide → 4-wide → scalar. For `d` that is an exact multiple of 16 (every fixed dim used elsewhere in this codebase — 64, 128, 256, 512, 768, 1024, 1536 — plus the small ad hoc test dims like `d=2`), both produce byte-identical results (no remainder ever executes). The two implementations only disagree in accumulation grouping when `4 ≤ (d mod 16) `, which can shift the last few ULPs of the result. This exact code path — `l2sqr_f32_block`'s auto-selected dot-trick branch (`d >= 256`, no precomputed norms) — is **not exercised by any existing test** (confirmed: `Tests/VectorIndexTests/MicrokernelIntegrationTests.swift` only uses `d=64`, `Tests/VectorIndexTests/HNSWNeighborSelectionTests.swift` only uses `d=2`). Existing tolerance-based tests do **not** currently absorb this because they never reach the branch at all. Step 4 below adds a small parity test at `d=257` (both `≥256` to force dot-trick auto-selection, and not a multiple of 16 to force the remainder path) that would have caught a real regression here, closing that gap.

- [ ] **Step 1: Confirm branch.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex branch --show-current
  ```
  Expected output: `gifton/cleanup-0.2.0-phase2`

- [ ] **Step 2: `L2SqrKernel.swift` — delegate `_normSquared` to `Norms.l2NormSquared`.**
  Replace (currently lines 484-502):
  ```swift
  @inline(__always)
  func _normSquared(_ v: UnsafePointer<Float>, _ d: Int) -> Float {
      var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
      var j = 0
      while j + 15 < d {
          let v0 = SIMD4<Float>(v + j + 0)
          let v1 = SIMD4<Float>(v + j + 4)
          let v2 = SIMD4<Float>(v + j + 8)
          let v3 = SIMD4<Float>(v + j + 12)
          a0 += v0 * v0
          a1 += v1 * v1
          a2 += v2 * v2
          a3 += v3 * v3
          j += 16
      }
      var sum = (a0 + a1 + a2 + a3).sum()
      for t in j..<d { let val = v[t]; sum += val * val }
      return sum
  }
  ```
  with:
  ```swift
  @inline(__always)
  func _normSquared(_ v: UnsafePointer<Float>, _ d: Int) -> Float {
      IndexOps.Support.Norms.l2NormSquared(vector: v, dimension: d)
  }
  ```
  (Callers at `L2SqrKernel.swift:406` inside `_l2sqr_block_dot_fused` and `:475` inside `_l2sqr_block_dot_fused_serial` are unaffected — same call syntax, new body.)

- [ ] **Step 3: `Cosine.swift` — delegate `sumSquares` to `Norms.l2NormSquared`.**
  Replace (currently lines 183-193):
  ```swift
          @inline(__always) private static func sumSquares(ptr: UnsafePointer<Float>, d: Int) -> Float {
              let d4 = d & ~3
              var acc = SIMD4<Float>.zero
              var j = 0
              while j < d4 { let v = load4(ptr.advanced(by: j)); acc += v * v; j += 4 }
              var s = hsum4(acc)
              if d - d4 >= 1 { s += ptr[d4+0] * ptr[d4+0] }
              if d - d4 >= 2 { s += ptr[d4+1] * ptr[d4+1] }
              if d - d4 >= 3 { s += ptr[d4+2] * ptr[d4+2] }
              return s
          }
  ```
  with:
  ```swift
          @inline(__always) private static func sumSquares(ptr: UnsafePointer<Float>, d: Int) -> Float {
              IndexOps.Support.Norms.l2NormSquared(vector: ptr, dimension: d)
          }
  ```
  (Callers at `Cosine.swift:114` in `run`'s on-the-fly-norms branch and `:196` in `computeQueryInvNorm_impl` are unaffected.)

- [ ] **Step 4: `Cosine.swift` — fold `precomputeInvNorms`'s separately-inlined loop into the same canonical call (the spec-unlisted 5th duplicate).**
  Replace (currently lines 427-441):
  ```swift
          public static func precomputeInvNorms(
              xb: UnsafePointer<Float>, n: Int, d: Int, out: UnsafeMutablePointer<Float>, epsilon: Float = 1e-12
          ) {
              if n == 0 { return }
              if d == 0 { for i in 0..<n { out[i] = 1.0 / epsilon }; return }
              let d4 = d & ~3
              for i in 0..<n {
                  let row = xb.advanced(by: i * d)
                  var acc = SIMD4<Float>.zero
                  var j = 0
                  while j < d4 { let v = load4(row.advanced(by: j)); acc += v * v; j += 4 }
                  var s = hsum4(acc); while j < d { let v = row[j]; s += v * v; j += 1 }
                  out[i] = 1.0 / (sqrt(s) + epsilon)
              }
          }
  ```
  with:
  ```swift
          public static func precomputeInvNorms(
              xb: UnsafePointer<Float>, n: Int, d: Int, out: UnsafeMutablePointer<Float>, epsilon: Float = 1e-12
          ) {
              if n == 0 { return }
              if d == 0 { for i in 0..<n { out[i] = 1.0 / epsilon }; return }
              for i in 0..<n {
                  let row = xb.advanced(by: i * d)
                  let s = IndexOps.Support.Norms.l2NormSquared(vector: row, dimension: d)
                  out[i] = 1.0 / (sqrt(s) + epsilon)
              }
          }
  ```
  (`precomputeInvNorms` stays `public` with the same signature — this is a body-only refactor, not a deletion, so no deprecation is needed. It has zero internal callers per the brief, so this cannot regress any in-repo caller; it's still worth fixing since the whole point of B8 is "one impl.")

- [ ] **Step 5: Add the numeric-parity spot test for the dot-trick branch (closes the coverage gap noted above).**
  In `Tests/VectorIndexTests/MicrokernelIntegrationTests.swift`, insert this new test method immediately before the final closing `}` of `MicrokernelIntegrationTests`:
  ```swift

      func testL2SqrDotTrickNormSquaredParity_NonMultipleOf16() {
          // d=257 forces both (a) auto dot-trick selection (d >= 256, per
          // L2SqrKernel.swift's useDotTrick heuristic) and (b) a non-multiple-
          // of-16 remainder in the on-the-fly ‖·‖² computation inside
          // _normSquared, which now delegates to Norms.l2NormSquared. Guards
          // against accumulation-order drift introduced by that delegation
          // (Task 6 / B8) — this exact path had no prior test coverage.
          let d = 257, n = 5
          var q = [Float](repeating: 0, count: d)
          var xb = [Float](repeating: 0, count: n * d)
          for i in 0..<d { q[i] = Float.random(in: -1...1) }
          for i in 0..<(n * d) { xb[i] = Float.random(in: -1...1) }

          var dotTrickOut = [Float](repeating: 0, count: n)
          var scalarOut = [Float](repeating: 0, count: n)

          q.withUnsafeBufferPointer { qb in
              xb.withUnsafeBufferPointer { xbb in
                  var opts = L2SqrOpts(algo: .dotTrick, useDotTrick: true, prefetchDistance: 8, strictFP: false, numThreads: 1)
                  dotTrickOut.withUnsafeMutableBufferPointer { out in
                      withUnsafePointer(to: &opts) { optsPtr in
                          l2sqr_f32_block(qb.baseAddress!, xbb.baseAddress!, n, d, out.baseAddress!, nil, .nan, optsPtr)
                      }
                  }
                  scalarOut.withUnsafeMutableBufferPointer { out in
                      IndexOps.Scoring.L2Sqr.runScalarRef(q: qb.baseAddress!, xb: xbb.baseAddress!, n: n, d: d, out: out.baseAddress!)
                  }
              }
          }

          for i in 0..<n {
              XCTAssertEqual(dotTrickOut[i], scalarOut[i], accuracy: 1e-2,
                             "row \(i): dot-trick vs scalar mismatch at d=\(d) (exercises _normSquared's non-16-aligned remainder path)")
          }
      }
  ```

- [ ] **Step 6: Build.**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build --build-tests
  ```
  Expected output ends with: `Build complete!`

- [ ] **Step 7: Run the named test filters (foreground).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --skip-build --filter '^VectorIndexTests\.(MicrokernelIntegrationTests|CosineKernelTests)'
  ```
  Expected output ends with:
  ```
  Test Suite 'Selected tests' passed at ...
  	 Executed 8 tests, with 0 failures (0 unexpected) in ... seconds
  ```
  (5 pre-existing `CosineKernelTests` + 2 pre-existing `MicrokernelIntegrationTests` + 1 new `testL2SqrDotTrickNormSquaredParity_NonMultipleOf16` = 8; measured baseline before this task's edits was 5+2=7.)

- [ ] **Step 8: Commit.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex add \
    Sources/VectorIndex/Operations/Scoring/L2SqrKernel.swift \
    Sources/VectorIndex/Operations/Scoring/Cosine.swift \
    Tests/VectorIndexTests/MicrokernelIntegrationTests.swift
  ```
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex commit -m "$(cat <<'EOF'
  refactor(scoring): consolidate sum-of-squares onto Norms.l2NormSquared (B8)

  L2SqrKernel._normSquared, Cosine.sumSquares, and Cosine.precomputeInvNorms's
  separately-inlined accumulation loop (a 5th, spec-unlisted duplicate) now all
  delegate to the one canonical implementation, Norms.l2NormSquared. (The other
  two spec-listed duplicates are already handled: ScoreBlock.sumSquares was
  dead code deleted in the prior cleanup commit, and MIPSTransform.l2NormSquaredSIMD
  was deleted and rewired in the commit before that.)

  _normSquared's old body summed its 0-15-element remainder scalar-only, while
  Norms.l2NormSquared does an extra 4-wide SIMD tier first; the two only
  disagree in accumulation order (ULP-level) when d mod 16 is in [4,15], and
  no existing test exercised that combined with the dot-trick auto-selection
  path (d >= 256, no precomputed norms). Added
  testL2SqrDotTrickNormSquaredParity_NonMultipleOf16 (d=257) to cover it.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  ```

---

### Task 7: B10b — merge the two scalar HNSW distance-kernel families

**Files:** Modify `Sources/VectorIndex/Kernels/HNSWTraversal.swift` (L41-70 deleted; call sites at L179, L188, L224, L235 updated; 329 lines total). No changes needed in `Sources/VectorIndex/Kernels/HNSWNeighborSelection.swift` — its `ns_*` family becomes the sole surviving implementation as-is.

**Interfaces:** Consumes: `ns_invnorm_f32(_ x: UnsafePointer<Float>, _ d: Int) -> Float` and `ns_distance_f32(a: UnsafePointer<Float>, b: UnsafePointer<Float>, d: Int, metric: HNSWMetric, invA: Float?, invB: Float?) -> Float` (both `@usableFromInline internal`, `Sources/VectorIndex/Kernels/HNSWNeighborSelection.swift:76-95`; visible module-wide with no import needed since both files are in target `VectorIndex`). Produces: none new.

**Premise correction (per brief):** the two families are **not byte-identical text**, contra the spec's phrasing — `distance_f32(q:d:x:metric:qInv:xInv:)` (Family A, `HNSWTraversal.swift`) and `ns_distance_f32(a:b:d:metric:invA:invB:)` (Family B, `HNSWNeighborSelection.swift`) have different parameter order/labels, so this is a real multi-call-site refactor, not a delete-the-duplicate copy-paste. It is, however, **low-risk**: `dot_f32`/`l2sq_f32` (and their `ns_` twins) are mathematically symmetric in their two vector arguments, so a parameter-order mistake at a call site is either (a) caught immediately by the type checker (an `Int` passed where an `UnsafePointer<Float>` is expected won't compile), or (b) numerically inert (swapping the two vector pointers of a symmetric function changes nothing). Family A's quartet is also used far more narrowly in practice than it looks: `HNSWTraversal.swift`'s batched neighbor scoring already goes through the shared `ScoreBlock.run` (via `scoreNeighborsBatch_f32`), so the private `dot_f32`/`l2sq_f32`/`invnorm_f32`/`distance_f32` are only reachable via 4 call sites, all single-pair "entry-point self-distance" calculations inside `greedyDescent_core`/`efSearch_core`.

**Coverage check (no new test needed — reasoning below):** `Tests/VectorIndexTests/HNSWTraversalKernelTests.swift` only exercises the L2 metric at `d=2`, so it doesn't cover the COSINE branch of the merged call sites directly. However, `HNSWTraversal.traverse`/`greedyDescent_core` is the production path called from `HNSWIndex.swift:197,344,356`, and cosine-metric HNSW builds/searches are exercised broadly by `Tests/VectorIndexTests/HNSWKNNGraphTests.swift`, `HNSWTypedInsertHintTests.swift`, and `HNSWWALTests.swift` (all reference cosine). Combined with the type-safety argument above, this is sufficient parity coverage — no new test is added for this task.

- [ ] **Step 1: Confirm branch.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex branch --show-current
  ```
  Expected output: `gifton/cleanup-0.2.0-phase2`

- [ ] **Step 2: Delete Family A's 4-function quartet from `HNSWTraversal.swift`.**
  Delete this exact block (currently lines 41-70, leaves `clampID`'s closing `}` directly followed by a blank line then `private struct HeapNode`):
  ```swift

  @inline(__always) private func dot_f32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
      var acc: Float = 0; var i = 0; let u = d & ~3
      while i < u { acc += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]; i += 4 }
      while i < d { acc += a[i] * b[i]; i += 1 }
      return acc
  }
  @inline(__always) private func l2sq_f32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
      var acc: Float = 0; var i = 0; let u = d & ~3
      while i < u {
          let d0 = a[i]-b[i], d1 = a[i+1]-b[i+1], d2 = a[i+2]-b[i+2], d3 = a[i+3]-b[i+3]
          acc += d0*d0; acc += d1*d1; acc += d2*d2; acc += d3*d3; i += 4
      }
      while i < d { let dv = a[i]-b[i]; acc += dv*dv; i += 1 }
      return acc
  }
  @inline(__always) private func invnorm_f32(_ a: UnsafePointer<Float>, _ d: Int) -> Float {
      let n = sqrtf(max(1e-12, dot_f32(a, a, d))); return 1.0 / n
  }
  @inline(__always) private func distance_f32(q: UnsafePointer<Float>, d: Int, x: UnsafePointer<Float>, metric: HNSWMetric, qInv: Float?, xInv: Float?) -> Float {
      switch metric {
      case .L2: return l2sq_f32(q, x, d)
      case .IP: return -dot_f32(q, x, d)
      case .COSINE:
          let qi = qInv ?? invnorm_f32(q, d)
          let xi = xInv ?? invnorm_f32(x, d)
          let sim = dot_f32(q, x, d) * qi * xi
          return 1.0 - sim
      }
  }
  ```

- [ ] **Step 3: Update the 2 identical `invnorm_f32` call sites (`replace_all`).**
  This exact line appears twice, byte-identical — once in `greedyDescent_core`, once in `efSearch_core` — use `replace_all: true`:
  ```swift
      let qInv: Float? = (metric == .COSINE) ? (qInvNorm ?? invnorm_f32(q, d)) : nil
  ```
  becomes:
  ```swift
      let qInv: Float? = (metric == .COSINE) ? (qInvNorm ?? ns_invnorm_f32(q, d)) : nil
  ```

- [ ] **Step 4: Update the `distance_f32` call site in `greedyDescent_core` (self-distance vs. current best).**
  Replace:
  ```swift
              var bestDist = distance_f32(q: q, d: d, x: xb.advanced(by: current * d), metric: metric, qInv: qInv, xInv: (metric == .COSINE ? invNorms?.advanced(by: current).pointee : nil))
  ```
  with:
  ```swift
              var bestDist = ns_distance_f32(a: q, b: xb.advanced(by: current * d), d: d, metric: metric, invA: qInv, invB: (metric == .COSINE ? invNorms?.advanced(by: current).pointee : nil))
  ```

- [ ] **Step 5: Update the `distance_f32` call site in `efSearch_core` (entry-point distance).**
  Replace:
  ```swift
      let enterDist = distance_f32(q: q, d: d, x: enterX, metric: metric, qInv: qInv, xInv: nil)
  ```
  with:
  ```swift
      let enterDist = ns_distance_f32(a: q, b: enterX, d: d, metric: metric, invA: qInv, invB: nil)
  ```

- [ ] **Step 6: Verify no stray references to the deleted Family A remain.**
  ```bash
  grep -n "\bdot_f32(\|\bl2sq_f32(\|\binvnorm_f32(\|\bdistance_f32(" /Users/goftin/dev/gsuite/VSK/VectorIndex/Sources/VectorIndex/Kernels/HNSWTraversal.swift
  ```
  Expected output: (empty — no matches; all remaining calls in the file now use the `ns_`-prefixed names)

- [ ] **Step 7: Build.**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build --build-tests
  ```
  Expected output ends with: `Build complete!` (a real type mismatch at any of the 4 call sites would fail here, which is the primary safety net for this refactor per the risk analysis above).

- [ ] **Step 8: Run the named test filters (foreground).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --skip-build --filter '^VectorIndexTests\.(HNSWTraversalKernelTests|HNSWNeighborSelectionTests|HNSWTests|HNSWRecallTests|HNSWKNNGraphTests|HNSWTypedInsertHintTests|HNSWWALTests)'
  ```
  Expected output ends with:
  ```
  Test Suite 'Selected tests' passed at ...
  	 Executed <N> tests, with 0 failures (0 unexpected) in ... seconds
  ```
  (Same `<N>` as the pre-Task-7 baseline — no test added or removed by this task.)

- [ ] **Step 9: Commit.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex add Sources/VectorIndex/Kernels/HNSWTraversal.swift
  ```
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex commit -m "$(cat <<'EOF'
  refactor(hnsw): merge duplicate scalar distance-kernel families onto ns_* (B10b)

  HNSWTraversal.swift's private {dot_f32, l2sq_f32, invnorm_f32, distance_f32}
  quartet was algorithmically identical to (but not textually identical to,
  contra the spec's "byte-identical" framing) HNSWNeighborSelection.swift's
  {ns_dot_f32, ns_l2sq_f32, ns_invnorm_f32, ns_distance_f32}. Deleted Family A
  and repointed its 4 call sites (2x self-distance in greedyDescent_core /
  efSearch_core, both only reachable for the single entry-point/current-best
  distance calc — the batched neighbor loop already goes through the shared
  ScoreBlock.run) to the ns_* family, which keeps its existing signature
  (a/b/d/metric/invA/invB) unchanged.

  No new test added: the merge is compile-checked (dot/l2sq are symmetric in
  their two vector args, so any call-site param-order mistake either fails to
  typecheck or is numerically inert), and the COSINE branch this touches is
  already exercised transitively via HNSWKNNGraphTests/HNSWTypedInsertHintTests/
  HNSWWALTests through the production HNSWIndex -> HNSWTraversal.traverse path.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  ```

---

### Task 8: A9-orphan — `hnsw_prune_neighbors_f32_swift` disposition

**Files:** Modify `Sources/VectorIndex/Kernels/HNSWNeighborSelection.swift` (L253, L365-366; 387 lines total)

**Interfaces:** Consumes: none new. Produces: `@available(*, deprecated, ...)` added to `public func hnsw_prune_neighbors_f32_swift(u: Int32, xb: UnsafePointer<Float>, d: Int, offsetsL: UnsafePointer<Int32>, neighborsL: UnsafePointer<Int32>, M: Int, metric: HNSWMetric, optionalInvNorms: UnsafePointer<Float>?, N: Int, prunedOut: UnsafeMutablePointer<Int32>) -> Int` and to its `@_cdecl("hnsw_prune_neighbors_f32") public func c_hnsw_prune_neighbors_f32(...)` shim — no signature changes.

**Disposition (per brief, access-level check):** both `hnsw_prune_neighbors_f32_swift` (line 253) and its C-ABI shim `c_hnsw_prune_neighbors_f32` (line 366, `@_cdecl("hnsw_prune_neighbors_f32")`) are declared `public` — confirmed by direct read, not `internal`. Per the Phase 2 non-breaking rule, a `public` deletion target becomes `@available(*, deprecated)` instead of being deleted, so **both** get deprecated (not just the kernel) — the shim is `public` too, even though it's a C-ABI export, so it falls under the same "no public symbol removed" rule. The Swift kernel is dead only in *production wiring*: `HNSWIndex.swift`'s insertion/maintenance path calls `hnsw_select_neighbors_f32_swift` (its sibling) but never `hnsw_prune_neighbors_f32_swift` — grep across the whole `/Users/goftin/dev/gsuite` tree (all repos) for the C-exported symbol name `hnsw_prune_neighbors_f32` finds zero hits outside this repo's own `Sources`/`Tests`, including the real downstream consumer `VectorIndexAccelerated`. It is **not**, however, fully dead: it still has one legitimate consumer, its own direct unit test `Tests/VectorIndexTests/HNSWNeighborSelectionTests.swift:60` (`testPruneNeighborsKeepsTopM_L2`). That test is left in place unmodified — it continues to validate the (now-deprecated, still-correct) kernel body, which is exactly the disposition the brief itself recommends ("keep the kernel + test, demote the shim").

- [ ] **Step 1: Confirm branch.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex branch --show-current
  ```
  Expected output: `gifton/cleanup-0.2.0-phase2`

- [ ] **Step 2: Deprecate `hnsw_prune_neighbors_f32_swift`.**
  In `Sources/VectorIndex/Kernels/HNSWNeighborSelection.swift`, replace (currently line 253):
  ```swift
  public func hnsw_prune_neighbors_f32_swift(
  ```
  with:
  ```swift
  @available(*, deprecated, message: "Dead in production wiring since A9 — HNSWIndex never calls this (only hnsw_select_neighbors_f32_swift is wired into insertion/maintenance). Scheduled for removal in 0.2.0's breaking phase.")
  public func hnsw_prune_neighbors_f32_swift(
  ```

- [ ] **Step 3: Deprecate the `@_cdecl` shim `c_hnsw_prune_neighbors_f32`.**
  Replace (currently lines 365-366):
  ```swift
  @_cdecl("hnsw_prune_neighbors_f32")
  public func c_hnsw_prune_neighbors_f32(
  ```
  with:
  ```swift
  @available(*, deprecated, message: "Exports a dead-in-production kernel (see hnsw_prune_neighbors_f32_swift); zero consumers found repo-wide, including VectorIndexAccelerated. Scheduled for removal in 0.2.0's breaking phase.")
  @_cdecl("hnsw_prune_neighbors_f32")
  public func c_hnsw_prune_neighbors_f32(
  ```
  (No change needed at the shim's internal call to `hnsw_prune_neighbors_f32_swift` on line ~378 — Swift does not re-warn when a deprecated declaration calls another deprecated declaration.)

- [ ] **Step 4: Build (expect exactly one new deprecation warning outside this file, at the test call site).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build --build-tests 2>&1 | grep -A3 "hnsw_prune_neighbors_f32_swift.*deprecated"
  ```
  Expected output: one warning block pointing at `Tests/VectorIndexTests/HNSWNeighborSelectionTests.swift:60` (`testPruneNeighborsKeepsTopM_L2`'s call to `hnsw_prune_neighbors_f32_swift`). This is expected and non-blocking (`Package.swift` sets no `-warnings-as-errors`); it exists precisely because that test is the kernel's one remaining legitimate consumer.

- [ ] **Step 5: Run the named test filter (foreground).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --skip-build --filter '^VectorIndexTests\.(HNSWNeighborSelectionTests)'
  ```
  Expected output ends with:
  ```
  Test Suite 'HNSWNeighborSelectionTests' passed at ...
  	 Executed 2 tests, with 0 failures (0 unexpected) in ... seconds
  ```
  (Matches the measured pre-Task-8 baseline: `testPruneNeighborsKeepsTopM_L2` + `testSelectNeighborsDiversityAndFill_L2`, both still passing — only a compiler warning changed, not behavior.)

- [ ] **Step 6: Commit.**
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex add Sources/VectorIndex/Kernels/HNSWNeighborSelection.swift
  ```
  ```bash
  git -C /Users/goftin/dev/gsuite/VSK/VectorIndex commit -m "$(cat <<'EOF'
  refactor(hnsw): deprecate hnsw_prune_neighbors_f32_swift + its C shim (A9-orphan)

  Confirmed both hnsw_prune_neighbors_f32_swift and its @_cdecl("hnsw_prune_
  neighbors_f32") shim c_hnsw_prune_neighbors_f32 are public, so Phase 2's
  non-breaking rule applies to both, not just the kernel. Neither is called
  from HNSWIndex's insertion/maintenance path (only its sibling
  hnsw_select_neighbors_f32_swift is wired in there since A9), and a whole-tree
  grep across /Users/goftin/dev/gsuite (including VectorIndexAccelerated) finds
  zero consumers of the C-exported symbol name.

  The kernel is not fully dead, though: its own direct unit test,
  testPruneNeighborsKeepsTopM_L2, is a legitimate consumer and is left
  unchanged — it continues to guard the (now-deprecated) implementation.
  hnsw_select_neighbors_f32_swift's own tests + HNSWIndex's broad
  insertion/build suite already cover the live neighbor-maintenance path this
  kernel was apparently meant to feed. Full removal routed to Phase 4.

  Co-Authored-By: <use the executing assistant's standard Co-Authored-By trailer>
  EOF
  )"
  ```

---

### Task 9: B9 non-breaking half + A6 coverage

**Files:**
- Modify: `Sources/VectorIndex/Kernels/ResidualKernel.swift` (lines 293–303, 352–367, 376–381, 438–452, 456–463, 500–508 — current-tree at `ee67895`; Tasks 3/4 touch this file earlier in the plan sequence, so re-grep the anchor snippets below before editing if they've drifted)
- Modify: `Tests/VectorIndexTests/PQEncodeParity_SwiftOnly_Tests.swift` (append after line 135, inside the existing `final class PQEncodeParity_SwiftOnly_Tests`)
- Test only (no change expected): `Tests/VectorIndexTests/ResidualKernelTests.swift:405-441` (`testErrorHandling`, the pinned test this task must keep green)

**Interfaces:**
Consumes: `ResidualError.invalidCoarseID` (`Sources/VectorIndex/Kernels/ResidualKernel.swift:47`, existing, unchanged); `pq_encode_u4_f32(_:_:_:_:_:_:_:_:)`, `pq_encode_residual_u8_f32(_:_:_:_:_:_:_:_:_:_:)`, `pq_encode_residual_u4_f32(_:_:_:_:_:_:_:_:_:_:)`, `PQEncodeOpts.init(useDotTrick:outputLayout:centroidSqNorms:)` (all `Sources/VectorIndex/Operations/Quantization/PQEncode.swift`, existing, unchanged).
Produces: none. No public symbol is added or changed — `ResidualError` (public, line 47) is untouched and simply becomes the error every `checkBounds` site throws (it stays around regardless; full removal is C9/Phase 4).

---

- [ ] **Step 1: Unify `_residuals_grouped`'s bounds-check error onto `ResidualError.invalidCoarseID` and drop its inline comment**

  `_residuals_grouped` is the function starting `internal func _residuals_grouped(` a few lines above line 339. Find:
  ```swift
    var counts = [Int](repeating: 0, count: kc)
    for i in 0..<nInt {
        let a = Int(coarseIDs[i])
        if opts.checkBounds {
            guard a >= 0 && a < kc else {
                throw ErrorBuilder(.invalidRange, operation: "residuals_grouped")
                    .message("Coarse assignment ID out of valid range")
                    .info("coarse_id", "\(a)")
                    .info("valid_range", "0..<\(kc)")
                    .info("vector_index", "\(i)")
                    .build()
            }
        }
        counts[a] += 1  // ✅ Fixed: regular += instead of &+=
    }
  ```
  Replace with:
  ```swift
    var counts = [Int](repeating: 0, count: kc)
    for i in 0..<nInt {
        let a = Int(coarseIDs[i])
        if opts.checkBounds {
            guard a >= 0 && a < kc else {
                throw ResidualError.invalidCoarseID
            }
        }
        counts[a] += 1
    }
  ```

- [ ] **Step 2: Drop `_residuals_grouped`'s second inline comment**

  A few lines below (still in `_residuals_grouped`), find:
  ```swift
        grouped[pos] = i
        cursor[a] += 1  // ✅ Fixed: regular += instead of &+=
    }
  ```
  Replace with:
  ```swift
        grouped[pos] = i
        cursor[a] += 1
    }
  ```

- [ ] **Step 3: Unify `_residuals_grouped_inplace`'s bounds-check error and drop its inline comment**

  `_residuals_grouped_inplace` is the next function down (`internal func _residuals_grouped_inplace(`). Find:
  ```swift
    var counts = [Int](repeating: 0, count: kc)
    for i in 0..<nInt {
        let a = Int(coarseIDs[i])
        if opts.checkBounds {
            guard a >= 0 && a < kc else {
                throw ErrorBuilder(.invalidRange, operation: "residuals_grouped_inplace")
                    .message("Coarse assignment ID out of valid range")
                    .info("coarse_id", "\(a)")
                    .info("valid_range", "0..<\(kc)")
                    .info("vector_index", "\(i)")
                    .build()
            }
        }
        counts[a] += 1  // ✅ Fixed: regular += instead of &+=
    }
  ```
  Replace with:
  ```swift
    var counts = [Int](repeating: 0, count: kc)
    for i in 0..<nInt {
        let a = Int(coarseIDs[i])
        if opts.checkBounds {
            guard a >= 0 && a < kc else {
                throw ResidualError.invalidCoarseID
            }
        }
        counts[a] += 1
    }
  ```

- [ ] **Step 4: Drop `_residuals_grouped_inplace`'s second inline comment**

  Find:
  ```swift
        grouped[pos] = i
        cursor[a] += 1  // ✅ Fixed: regular += instead of &+=
    }
  ```
  Replace with:
  ```swift
        grouped[pos] = i
        cursor[a] += 1
    }
  ```
  (This text is identical to Step 2's — it's the second occurrence, inside `_residuals_grouped_inplace` instead of `_residuals_grouped`. Edit the one in the function you're currently in.)

- [ ] **Step 5: Unify `residuals_f32_inplace`'s (ungrouped path) bounds-check error — this is the one non-`_grouped` outlier**

  `residuals_f32_inplace` is the public function starting at (current-tree) line 244. Find, inside its ungrouped branch:
  ```swift
        let a = Int(coarseIDs[i])
        if opts.checkBounds {
            guard a >= 0 && a < opts.kc else {
                throw ErrorBuilder(.invalidRange, operation: "residuals_compute")
                    .message("Coarse assignment ID out of valid range")
                    .info("coarse_id", "\(a)")
                    .info("valid_range", "0..<\(opts.kc)")
                    .info("vector_index", "\(i)")
                    .build()
            }
        }
  ```
  Replace with:
  ```swift
        let a = Int(coarseIDs[i])
        if opts.checkBounds {
            guard a >= 0 && a < opts.kc else {
                throw ResidualError.invalidCoarseID
            }
        }
  ```
  Do **not** touch `residuals_f32`'s own ungrouped bounds check (lines 201–206) — it already throws `ResidualError.invalidCoarseID` and is the one the pinned test (`testErrorHandling`) exercises directly.

- [ ] **Step 6: Delete the changelog-style "Fixed Issues" block**

  Near the end of the file (current-tree lines 500–508), find:
  ```swift
  // MARK: - Notes / Integration
  //
  // ✅ **Fixed Issues**:
  // 1. Replaced &+= with += (no overflow expected)
  // 2. Added proper throws error handling
  // 3. Integrated Accelerate framework (vDSP) for d >= 256
  // 4. Removed hot-path allocations (fused functions now call existing PQ kernels)
  // 5. Mathematical documentation added
  //
  // **Integration with existing kernels**:
  ```
  Replace with (keep the section header and the still-useful integration notes that follow; drop only the itemized changelog):
  ```swift
  // MARK: - Notes / Integration
  //
  // **Integration with existing kernels**:
  ```

- [ ] **Step 7: Verify the bounds-check unification is green**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'ResidualKernelTests' 2>&1 | tail -30
  ```
  Expected: all `ResidualKernelTests` pass, including `testErrorHandling` (still asserts `error as? ResidualError == .invalidCoarseID` against `residuals_f32` — untouched) and the grouped-vs-ungrouped parity test (`residuals_grouped` compared against `residuals_standard`, output values unaffected by the error-path change). 0 failures.

- [ ] **Step 8: Add A6 coverage — `pq_encode_u4_f32` allocate-then-free stability (site 3)**

  Append to `Tests/VectorIndexTests/PQEncodeParity_SwiftOnly_Tests.swift`, inside the class, after `testRepeatedEncodeWithoutPrecomputedNormsIsStable` (reuses `makeData` from the same file):
  ```swift
    /// Guards Task 9 (spec A6, site 3): `pq_encode_u4_f32`'s only
    /// `ensureCentroidSqNorms` call site (PQEncode.swift:213) is reached whenever the
    /// C fast path is skipped (`.sOA` layout here, since the C branch only ever
    /// triggers for `.aOS`) -- previously called from zero test files. Drives the
    /// allocate-then-free path 51 times (1 baseline + 50 repeats) and asserts
    /// byte-identical packed output each time, the same idiom
    /// `testRepeatedEncodeWithoutPrecomputedNormsIsStable` uses for the u8 path.
    func testU4RepeatedEncodeWithoutPrecomputedNormsIsStable() {
        let n = 12, d = 24, m = 6, ks = 16
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: nil)

        func encodeOnce() -> [UInt8] {
            var codes = [UInt8](repeating: 0, count: n * (m / 2))
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    codes.withUnsafeMutableBufferPointer { out in
                        pq_encode_u4_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, out.baseAddress!, &opts)
                    }
                }
            }
            return codes
        }

        let first = encodeOnce()
        for _ in 0..<50 {
            XCTAssertEqual(encodeOnce(), first)
        }
    }
  ```

- [ ] **Step 9: Add A6 coverage — `pq_encode_residual_u8_f32` Swift-fallback stability (site 5)**

  Append directly after Step 8's test:
  ```swift
    /// Guards Task 9 (spec A6, site 5): `pq_encode_residual_u8_f32`'s Swift-fallback
    /// `ensureCentroidSqNorms` call site (PQEncode.swift:290), reached when the
    /// C-fast-path condition (`_useCPQEncode && layout == .aOS`) is false. The only
    /// existing residual-u8 test (`testResidualU8_SoA_WithCSQ_vs_Default`) always
    /// passes a precomputed `csq`, so it never reaches this site -- forcing
    /// `centroidSqNorms: nil` here does. Drives 50 repeats, asserts stability.
    func testResidualU8RepeatedEncodeWithoutPrecomputedNormsIsStable() {
        let n = 8, d = 32, m = 8, ks = 256, kc = 4
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var coarse = [Float](repeating: 0, count: kc * d)
        for i in 0..<(kc * d) { coarse[i] = Float(cos(Double(i * 19 % 4096)) * 0.33) }
        var assignments = [Int32](repeating: 0, count: n)
        for i in 0..<n { assignments[i] = Int32(i % kc) }
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: nil)

        func encodeOnce() -> [UInt8] {
            var codes = [UInt8](repeating: 0, count: n * m)
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    coarse.withUnsafeBufferPointer { gb in
                        assignments.withUnsafeBufferPointer { asg in
                            codes.withUnsafeMutableBufferPointer { out in
                                pq_encode_residual_u8_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, gb.baseAddress!, asg.baseAddress!, out.baseAddress!, &opts)
                            }
                        }
                    }
                }
            }
            return codes
        }

        let first = encodeOnce()
        for _ in 0..<50 {
            XCTAssertEqual(encodeOnce(), first)
        }
    }
  ```

- [ ] **Step 10: Add A6 coverage — `pq_encode_residual_u4_f32` allocate-then-free stability (site 6)**

  Append directly after Step 9's test:
  ```swift
    /// Guards Task 9 (spec A6, site 6): `pq_encode_residual_u4_f32`'s only
    /// `ensureCentroidSqNorms` call site (PQEncode.swift:413) -- same shape as site 3,
    /// the C path here never branches on `useDotTrick`/csq internally either.
    /// Previously called from zero test files. Drives 50 repeats, asserts stability.
    func testResidualU4RepeatedEncodeWithoutPrecomputedNormsIsStable() {
        let n = 8, d = 32, m = 8, ks = 16, kc = 4
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var coarse = [Float](repeating: 0, count: kc * d)
        for i in 0..<(kc * d) { coarse[i] = Float(cos(Double(i * 19 % 4096)) * 0.33) }
        var assignments = [Int32](repeating: 0, count: n)
        for i in 0..<n { assignments[i] = Int32(i % kc) }
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: nil)

        func encodeOnce() -> [UInt8] {
            var codes = [UInt8](repeating: 0, count: n * (m / 2))
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    coarse.withUnsafeBufferPointer { gb in
                        assignments.withUnsafeBufferPointer { asg in
                            codes.withUnsafeMutableBufferPointer { out in
                                pq_encode_residual_u4_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, gb.baseAddress!, asg.baseAddress!, out.baseAddress!, &opts)
                            }
                        }
                    }
                }
            }
            return codes
        }

        let first = encodeOnce()
        for _ in 0..<50 {
            XCTAssertEqual(encodeOnce(), first)
        }
    }
  ```

- [ ] **Step 11: Full task verification**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | tail -20
  ```
  Expected: `Build complete!`, exit 0, no new warnings.
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'ResidualKernelTests|PQEncodeParity_SwiftOnly_Tests' 2>&1 | tail -40
  ```
  Expected: all tests in both suites pass, 0 failures, including the 3 new `*RepeatedEncodeWithoutPrecomputedNormsIsStable` tests.

- [ ] **Commit**

  ```bash
  git add Sources/VectorIndex/Kernels/ResidualKernel.swift Tests/VectorIndexTests/PQEncodeParity_SwiftOnly_Tests.swift
  git commit -m "$(cat <<'EOF'
  fix(residuals): unify coarse-ID bounds-check error; cover PQ u4/residual encode paths

  residuals_f32_inplace, _residuals_grouped, and _residuals_grouped_inplace now all
  throw ResidualError.invalidCoarseID on out-of-range coarse IDs instead of
  ErrorBuilder(.invalidRange), matching residuals_f32 and the one pinned error-type
  test (ResidualKernelTests.testErrorHandling) -- zero test changes needed since that
  test only exercises residuals_f32, which already threw ResidualError.

  Also strips the four inline "Fixed" comments and the changelog-style "Fixed Issues"
  block (pure historical narration, no information content beyond the code itself),
  and adds stability coverage for pq_encode_u4_f32, pq_encode_residual_u8_f32's Swift
  fallback, and pq_encode_residual_u4_f32 -- all three previously reachable by zero
  tests at their ensureCentroidSqNorms allocate-then-free call sites (A6).
  EOF
  )"
  ```
  (with the executing assistant's standard Co-Authored-By trailer)

---

### Task 10: B11 — one CRC32, shared disk-layout structs (`VIndexContainerBuilder.swift`)

**Files:**
- Modify: `Sources/VectorIndex/Kernels/VIndexContainerBuilder.swift` (currently 311 lines; edits touch lines 13-56, 90-92, 260-266, 272-300)
- Modify: `Sources/VectorIndex/Kernels/VIndexMmap.swift` (lines 104, 120 — access-level only)
- Modify: `Tests/VectorIndexTests/VIndexMmapErrorTests.swift` (lines 21-36 — delete local `CRC32` mirror)

**Interfaces:**
- Consumes: `internal struct CRC32` and `internal struct VIndexHeader` (both defined in `VIndexMmap.swift`; `VIndexHeader` is raised from `private` to `internal` in this task; `CRC32` is already `internal`).
- Produces: `VIndexContainerBuilder.createMinimalContainer(...) throws -> IndexMmap` (signature unchanged — this task is a byte-identical internal dedup, no public API changes).
- `internal struct TOCEntry` (`VIndexMmap.swift:104`) is also raised to `internal` per the task brief's literal instruction, for symmetry with `VIndexHeader`, but **has no live call site added in the builder in this task** — the builder still writes TOC entries via raw packed offsets (`writeTOCEntry`, unchanged), because binding memory to `TOCEntry`'s natural Swift layout would NOT match the packed 36-byte on-disk format (same padding hazard the file's own comment warns about for the header). Do not "helpfully" wire `TOCEntry` into `writeTOCEntry` — that is out of scope and unsafe.

---

- [ ] **Step 1 — Checkout the phase-2 branch (idempotent: works whether earlier phase-2 tasks already created it).**
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex
  git fetch origin
  git checkout gifton/cleanup-0.2.0-phase2 2>/dev/null || git checkout -b gifton/cleanup-0.2.0-phase2 origin/main
  git status
  ```
  Expected output: `On branch gifton/cleanup-0.2.0-phase2` and a clean working tree.

- [ ] **Step 2 — Raise `TOCEntry` and `VIndexHeader` from `private` to `internal` in `VIndexMmap.swift`.**
  In `Sources/VectorIndex/Kernels/VIndexMmap.swift`, change:
  ```swift
  private struct TOCEntry {
  ```
  to:
  ```swift
  internal struct TOCEntry {
  ```
  and change:
  ```swift
  private struct VIndexHeader {
  ```
  to:
  ```swift
  internal struct VIndexHeader {
  ```
  (Both structs' bodies are unchanged — only the leading access keyword.)

- [ ] **Step 3 — Delete the builder's `_CRC32`, `_TOCEntry`, and `_Header` mirrors in `VIndexContainerBuilder.swift`.**
  Replace this whole block (current lines 13-56):
  ```swift
  @inline(__always) private func alignUpU64(_ x: UInt64, _ a: UInt64) -> UInt64 { let m = a &- 1; return (x &+ m) & ~m }

  // Local CRC32 for builder (duplicate of VIndexMmap.swift but self-contained)
  private struct _CRC32 {
      private static let table: [UInt32] = {
          (0..<256).map { i -> UInt32 in
              var c = UInt32(i)
              for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
              return c
          }
      }()
      @inline(__always) static func hash(_ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
          var c: UInt32 = 0xFFFF_FFFF
          let p = data.bindMemory(to: UInt8.self, capacity: len)
          for i in 0..<len {
              c = _CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
          }
          return c ^ 0xFFFF_FFFF
      }
  }

  // These mirror VIndexMmap.swift disk structs
  private struct _TOCEntry { var type: UInt32; var offset: UInt64; var size: UInt64; var align: UInt32; var flags: UInt32; var crc32: UInt32; var reserved: UInt32 }
  private struct _Header {
      var magic: UInt64
      var version_major: UInt16
      var version_minor: UInt16
      var endianness: UInt8
      var arch: UInt8
      var flags: UInt32
      var d: UInt32
      var m: UInt16
      var ks: UInt16
      var kc: UInt32
      var id_bits: UInt8
      var code_group_g: UInt8
      var reservedA: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
      var N_total: UInt64
      var generation: UInt64
      var toc_offset: UInt64
      var toc_entries: UInt32
      var header_crc32: UInt32
      var reservedRest: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64)
  }
  ```
  with:
  ```swift
  @inline(__always) private func alignUpU64(_ x: UInt64, _ a: UInt64) -> UInt64 { let m = a &- 1; return (x &+ m) & ~m }

  // CRC32 and the disk-layout structs (VIndexHeader, TOCEntry) are shared with
  // VIndexMmap.swift (Sources/VectorIndex/Kernels/VIndexMmap.swift) — both `internal` in
  // this module. Phase-2 (B11) dedup of what used to be byte-identical `_CRC32`/`_Header`/
  // `_TOCEntry` mirrors defined locally in this file. `ListDesc` has no builder-side mirror
  // to dedup (the builder writes ListsDesc records via raw packed offsets, never via a
  // named struct) — that is unchanged, out of scope for B11.
  ```
  This is a pure deletion (~44 lines removed); no other code moves.

- [ ] **Step 4 — Point the builder's CRC computation at the canonical `CRC32`.**
  In the same file, in `writeCRC(at:offset:size:)`, change:
  ```swift
  func writeCRC(at index: Int, offset: UInt64, size: UInt64) {
      let p = UnsafeRawPointer(base).advanced(by: Int(offset))
      let c = _CRC32.hash(p, Int(size))
  ```
  to:
  ```swift
  func writeCRC(at index: Int, offset: UInt64, size: UInt64) {
      let p = UnsafeRawPointer(base).advanced(by: Int(offset))
      let c = CRC32.hash(p, Int(size))
  ```

- [ ] **Step 5 — Point the builder's header construction/CRC at `VIndexHeader`/`CRC32`.**
  Change:
  ```swift
          // Write header with CRC
          let hdr = _Header(
  ```
  to:
  ```swift
          // Write header with CRC
          let hdr = VIndexHeader(
  ```
  Change:
  ```swift
          // Compute header CRC over 256 bytes with crc field zeroed
          let hdrPtr = UnsafeMutableRawPointer(base).assumingMemoryBound(to: _Header.self)
          hdrPtr.pointee = hdr
          // Zero CRC field in place then compute
          hdrPtr.pointee.header_crc32 = 0
          let crc = _CRC32.hash(UnsafeRawPointer(hdrPtr), 256)
          hdrPtr.pointee.header_crc32 = crc
  ```
  to:
  ```swift
          // Compute header CRC over 256 bytes with crc field zeroed
          let hdrPtr = UnsafeMutableRawPointer(base).assumingMemoryBound(to: VIndexHeader.self)
          hdrPtr.pointee = hdr
          // Zero CRC field in place then compute
          hdrPtr.pointee.header_crc32 = 0
          let crc = CRC32.hash(UnsafeRawPointer(hdrPtr), 256)
          hdrPtr.pointee.header_crc32 = crc
  ```
  This is a pure rename — the actual on-disk bytes produced are bit-for-bit identical to before (`VIndexHeader`'s field layout is identical to the deleted `_Header`'s), so existing containers and their stored CRCs remain valid.

- [ ] **Step 6 — Reword the `DISK_TOC_ENTRY_SIZE` comment that referenced the now-deleted `_TOCEntry`.**
  Change:
  ```swift
          // On-disk TOC entries are packed 36 bytes (see writeTOCEntry below); MemoryLayout.stride
          // on the in-memory-only _TOCEntry mirror would over-reserve due to struct padding.
          let DISK_TOC_ENTRY_SIZE: UInt64 = 36
  ```
  to:
  ```swift
          // On-disk TOC entries are packed 36 bytes (see writeTOCEntry below); MemoryLayout.stride
          // on an in-memory struct mirror would over-reserve due to struct padding, so entries
          // are written via raw offsets rather than by binding memory to VIndexMmap.swift's
          // `TOCEntry` (which is `internal` for read-side symmetry but not used here — see the
          // file-header dedup note above).
          let DISK_TOC_ENTRY_SIZE: UInt64 = 36
  ```

- [ ] **Step 7 — Delete the test file's third CRC32 copy, letting `@testable import VectorIndex` resolve to the canonical one.**
  `Tests/VectorIndexTests/VIndexMmapErrorTests.swift` already has `@testable import VectorIndex` at the top (line 2), which exposes `internal` module symbols — including `CRC32` — unqualified in this file. Delete the local shadowing copy (current lines 21-36):
  ```swift
      // Local CRC32 for tests (matches builder logic)
      private struct CRC32 {
          static let table: [UInt32] = {
              (0..<256).map { i -> UInt32 in
                  var c = UInt32(i)
                  for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
                  return c
              }
          }()
          @inline(__always) static func hash(_ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
              var c: UInt32 = 0xFFFF_FFFF
              let p = data.bindMemory(to: UInt8.self, capacity: len)
              for i in 0..<len { c = CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8) }
              return c ^ 0xFFFF_FFFF
          }
      }

  ```
  Do not touch anything else in the file — its one call site (`CRC32.hash(UnsafeRawPointer(base), hdrSize)`, in `testVersionMismatchThrows`) now resolves to the module's internal `CRC32` and needs no edit.

- [ ] **Step 8 — Build.**
  ```bash
  swift build 2>&1 | tail -30
  ```
  Expected: `Build complete!` with no errors (pre-existing `#no-usage` warnings in this file are unrelated and fine to see).

- [ ] **Step 9 — Run the named regression filters for this task.**
  ```bash
  swift test --filter 'VIndexMmapErrorTests|RegressionA2_DurableListStatsTests|RegressionA3_RemapTOCTests|RegressionA7_TOCReservationTests|Kernel30AppendTests' 2>&1 | tail -60
  ```
  Expected: all of `testHeaderCRCMismatchThrows`, `testVersionMismatchThrows`, `testSectionCRCMismatchThrows`, `testOpenMissingFileThrows`, `testEnsureCapacityGrowOrRemapFailure` (VIndexMmapErrorTests), `testDurableGetListStatsReturnsCapacity` (RegressionA2), `testRemapThenReopenPreservesSections` (RegressionA3), `testListZeroDescriptorCapacityNotClobberedByIDMapTOCEntry` (RegressionA7), and all 4 `Kernel30AppendTests` methods report `** TEST SUCCEEDED **` / `Executed N tests, with 0 failures`.

- [ ] **Step 10 — Commit.**
  ```bash
  git add Sources/VectorIndex/Kernels/VIndexContainerBuilder.swift \
          Sources/VectorIndex/Kernels/VIndexMmap.swift \
          Tests/VectorIndexTests/VIndexMmapErrorTests.swift
  git commit -m "$(cat <<'EOF'
  refactor(mmap): dedup CRC32 and header/TOC structs between builder and reader (B11)

  VIndexContainerBuilder.swift carried byte-identical private mirrors of VIndexMmap.swift's
  CRC32 implementation and disk-layout header struct (plus a dead, never-instantiated
  _TOCEntry mirror). Raise VIndexHeader/TOCEntry to internal and delete the builder's
  _CRC32/_Header/_TOCEntry copies in favor of the canonical ones; the on-disk bytes and CRCs
  produced are unchanged (pure rename), so existing containers remain valid.

  Co-Authored-By: use the executing assistant's standard Co-Authored-By trailer
  EOF
  )"
  ```

---

### Task 11: B13 — mmap low-severity tidy (`VIndexMmap.swift`)

**Files:**
- Modify: `Sources/VectorIndex/Kernels/VIndexMmap.swift` (edits at lines ~44-68 `toHost`/`fromHost`/`CRC32`, ~145-155 `computeHeaderCRC`, ~219-220 WAL scratch property, ~849-865 `mmap_wal_replay`, ~994-1041 `writeWalAppend`/`writeWalCommit`)
- Modify: `Tests/VectorIndexTests/VIndexMmapErrorTests.swift` (append 2 new test methods + 1 helper)

**Interfaces:**
- Consumes: none new.
- Produces (all `internal`/`private`, no public API change): `CRC32.update(_:_:_:) -> UInt32`, `CRC32.finalize(_:) -> UInt32` (new); `computeHeaderCRC(_:) -> UInt32` (signature unchanged, body rewritten); `IndexMmap.mmap_wal_replay() throws` (signature unchanged, `WAL_APPEND_TAG` branch behavior changed from "discard" to "validate CRC, halt replay on mismatch" — this is a **behavior change on an until-now-untested internal recovery path**, not a public API change).

This task depends on Task 10 having already landed (Task 10 deletes `VIndexMmapErrorTests.swift`'s local `CRC32` struct; this task's new tests call the module's `CRC32` the same way the file's existing tests already do). Do not run this task against a tree where Task 10 hasn't landed.

---

- [ ] **Step 1 — Delete the dead `fromHost` function.**
  In `Sources/VectorIndex/Kernels/VIndexMmap.swift`, delete (current lines 48-50):
  ```swift
  @inline(__always) private func fromHost<T: FixedWidthInteger>(_ v: T, fileEndian: Endian) -> T {
      ((fileEndian == .little) == hostIsLittleEndian()) ? v : v.byteSwapped
  }

  ```
  Confirmed zero call sites anywhere in `Sources/` or `Tests/` (`grep -rn "fromHost"` matches only the declaration). `toHost` (the one actually used, ~20 call sites) is untouched.

- [ ] **Step 2 — Extend `CRC32` with a resumable update/finalize API, preserving `hash(_:_:)` byte-for-byte.**
  Replace the `CRC32` struct (current lines 52-68):
  ```swift
  internal struct CRC32 {
      private static let table: [UInt32] = {
          (0..<256).map { i -> UInt32 in
              var c = UInt32(i)
              for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
              return c
          }
      }()
      @inline(__always) static func hash(_ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
          var c: UInt32 = 0xFFFF_FFFF
          let p = data.bindMemory(to: UInt8.self, capacity: len)
          for i in 0..<len {
              c = CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
          }
          return c ^ 0xFFFF_FFFF
      }
  }
  ```
  with:
  ```swift
  internal struct CRC32 {
      private static let table: [UInt32] = {
          (0..<256).map { i -> UInt32 in
              var c = UInt32(i)
              for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
              return c
          }
      }()

      /// Resumable CRC32 update: feeds `data`/`len` into a running `state` so callers can
      /// compute one CRC32 over several non-contiguous byte ranges (e.g. "all of a struct
      /// except one 4-byte field") without first copying them into one contiguous buffer.
      /// Start a fresh computation with `state = 0xFFFF_FFFF`; call `finalize(_:)` on the
      /// last state returned to get the standard CRC32 value.
      @inline(__always) static func update(_ state: UInt32, _ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
          var c = state
          let p = data.bindMemory(to: UInt8.self, capacity: len)
          for i in 0..<len {
              c = CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
          }
          return c
      }

      /// Applies CRC32's final XOR to a running `update(_:_:_:)` state.
      @inline(__always) static func finalize(_ state: UInt32) -> UInt32 { state ^ 0xFFFF_FFFF }

      @inline(__always) static func hash(_ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
          finalize(update(0xFFFF_FFFF, data, len))
      }
  }
  ```
  `hash(_:_:)`'s signature and output are unchanged, so all ~20 existing call sites elsewhere in this file (and the 2 in `VIndexContainerBuilder.swift` from Task 10) need no edits.

- [ ] **Step 3 — Rewrite `computeHeaderCRC` to hash in place (no 256-byte copy, no write access needed).**
  Replace (current lines 145-155):
  ```swift
  @inline(__always) private func computeHeaderCRC(_ raw: UnsafeRawPointer) -> UInt32 {
      // Copy header and zero the CRC field using struct field access (same as builder)
      var buf = [UInt8](repeating: 0, count: 256)
      memcpy(&buf, raw, 256)
      // Zero the CRC field at its actual offset (68-71) via struct overlay
      return buf.withUnsafeMutableBytes { bufPtr in
          let hdrPtr = bufPtr.baseAddress!.assumingMemoryBound(to: VIndexHeader.self)
          hdrPtr.pointee.header_crc32 = 0
          return CRC32.hash(bufPtr.baseAddress!, 256)
      }
  }
  ```
  with:
  ```swift
  @inline(__always) private func computeHeaderCRC(_ raw: UnsafeRawPointer) -> UInt32 {
      // Two-region hash directly against the mapped header bytes: no 256-byte copy, and no
      // write access needed (the mapping may be PROT_READ-only when opts.readOnly is set, so
      // we must never mutate it). Must stay byte-for-byte equivalent to "zero the CRC field,
      // then CRC32 all 256 bytes" (what the builder computes at write time in
      // VIndexContainerBuilder.swift) — the 4 zeroed bytes still have to be fed into the
      // running CRC, just from a local zero value instead of the live memory.
      let crcOffset = MemoryLayout<VIndexHeader>.offset(of: \.header_crc32)! // 68
      let crcSize = 4
      var state = CRC32.update(0xFFFF_FFFF, raw, crcOffset)
      var zero: UInt32 = 0
      state = withUnsafeBytes(of: &zero) { CRC32.update(state, $0.baseAddress!, crcSize) }
      let tailOffset = crcOffset + crcSize
      state = CRC32.update(state, raw.advanced(by: tailOffset), 256 - tailOffset)
      return CRC32.finalize(state)
  }
  ```
  This fully eliminates the allocation (rather than merely reusing a scratch buffer for it, which is what the brief's "reuse one scratch buffer" framing suggested) — there is nothing left to reuse a buffer *for* in this function once the copy itself is gone. `computeHeaderCRC` is on the hot `IndexMmap.open` path whenever `opts.verifyCRCs` is set (the default), so this is exercised by essentially every mmap test in the suite; a correctness bug here would fail nearly all of them, which is why no dedicated unit test for this function exists or is needed beyond the broad regression run in Step 8.

- [ ] **Step 4 — Add a reusable WAL scratch buffer and rewire `writeWalAppend`/`writeWalCommit` to use it.**
  In the `IndexMmap` class, right after:
  ```swift
      private var walFD: Int32 = -1
      private var walPath: String
  ```
  add:
  ```swift
      /// Reusable scratch buffer for staging WAL record bytes before CRC32 hashing. Sized to
      /// the larger of the two records' CRC'd prefixes (WalAppend's 40 bytes: tag..vecsOff).
      /// Avoids a fresh heap allocation on every append/commit call.
      private var walScratch = [UInt8](repeating: 0, count: 40)
  ```
  Then in `writeWalAppend`, replace:
  ```swift
      private func writeWalAppend(listID: Int, oldLen: Int, delta: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64) throws {
          var rec = WalAppend(tag: WAL_APPEND_TAG.littleEndian, listID: UInt32(listID).littleEndian, oldLen: UInt32(oldLen).littleEndian, delta: UInt32(delta).littleEndian, idsOff: idsOff.littleEndian, codesOff: codesOff.littleEndian, vecsOff: vecsOff.littleEndian, crc32: 0)
          var tmp = [UInt8](repeating: 0, count: MemoryLayout<WalAppend>.size - 4)
          withUnsafeBytes(of: rec.tag) { tmp.replaceSubrange(0..<4, with: $0) }
          withUnsafeBytes(of: rec.listID) { tmp.replaceSubrange(4..<8, with: $0) }
          withUnsafeBytes(of: rec.oldLen) { tmp.replaceSubrange(8..<12, with: $0) }
          withUnsafeBytes(of: rec.delta) { tmp.replaceSubrange(12..<16, with: $0) }
          withUnsafeBytes(of: rec.idsOff) { tmp.replaceSubrange(16..<24, with: $0) }
          withUnsafeBytes(of: rec.codesOff) { tmp.replaceSubrange(24..<32, with: $0) }
          withUnsafeBytes(of: rec.vecsOff) { tmp.replaceSubrange(32..<40, with: $0) }
          let crc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, tmp.count) }
          rec.crc32 = crc.littleEndian
  ```
  with:
  ```swift
      private func writeWalAppend(listID: Int, oldLen: Int, delta: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64) throws {
          var rec = WalAppend(tag: WAL_APPEND_TAG.littleEndian, listID: UInt32(listID).littleEndian, oldLen: UInt32(oldLen).littleEndian, delta: UInt32(delta).littleEndian, idsOff: idsOff.littleEndian, codesOff: codesOff.littleEndian, vecsOff: vecsOff.littleEndian, crc32: 0)
          withUnsafeBytes(of: rec.tag) { walScratch.replaceSubrange(0..<4, with: $0) }
          withUnsafeBytes(of: rec.listID) { walScratch.replaceSubrange(4..<8, with: $0) }
          withUnsafeBytes(of: rec.oldLen) { walScratch.replaceSubrange(8..<12, with: $0) }
          withUnsafeBytes(of: rec.delta) { walScratch.replaceSubrange(12..<16, with: $0) }
          withUnsafeBytes(of: rec.idsOff) { walScratch.replaceSubrange(16..<24, with: $0) }
          withUnsafeBytes(of: rec.codesOff) { walScratch.replaceSubrange(24..<32, with: $0) }
          withUnsafeBytes(of: rec.vecsOff) { walScratch.replaceSubrange(32..<40, with: $0) }
          let crc = walScratch.withUnsafeBytes { CRC32.hash($0.baseAddress!, 40) }
          rec.crc32 = crc.littleEndian
  ```
  (`MemoryLayout<WalAppend>.size - 4` is 40 — `WalAppend` is `tag:UInt32, listID:UInt32, oldLen:UInt32, delta:UInt32, idsOff:UInt64, codesOff:UInt64, vecsOff:UInt64, crc32:UInt32` = 44 bytes total, minus the 4-byte `crc32` field.)

  Then in `writeWalCommit`, replace:
  ```swift
      private func writeWalCommit(listID: Int, newLen: Int) throws {
          var rec = WalCommit(tag: WAL_COMMIT_TAG.littleEndian, listID: UInt32(listID).littleEndian, newLen: UInt32(newLen).littleEndian, crc32: 0)
          var tmp = [UInt8](repeating: 0, count: 8)
          withUnsafeBytes(of: rec.listID) { tmp.replaceSubrange(0..<4, with: $0) }
          withUnsafeBytes(of: rec.newLen) { tmp.replaceSubrange(4..<8, with: $0) }
          let crc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, 8) }
          rec.crc32 = crc.littleEndian
  ```
  with:
  ```swift
      private func writeWalCommit(listID: Int, newLen: Int) throws {
          var rec = WalCommit(tag: WAL_COMMIT_TAG.littleEndian, listID: UInt32(listID).littleEndian, newLen: UInt32(newLen).littleEndian, crc32: 0)
          withUnsafeBytes(of: rec.listID) { walScratch.replaceSubrange(0..<4, with: $0) }
          withUnsafeBytes(of: rec.newLen) { walScratch.replaceSubrange(4..<8, with: $0) }
          let crc = walScratch.withUnsafeBytes { CRC32.hash($0.baseAddress!, 8) }
          rec.crc32 = crc.littleEndian
  ```
  Both functions now reuse the same 40-byte instance-level buffer (only ever reading the first 8 or 40 bytes as needed); `replaceSubrange` with an equal-length replacement never changes `walScratch`'s count, so no reallocation happens after the one-time property initialization.

- [ ] **Step 5 — Validate the WAL append record's CRC on replay instead of discarding it.**
  In `mmap_wal_replay()`, replace:
  ```swift
              if tag == WAL_APPEND_TAG {
                  _ = readExact(recordSizeAppend - 4)
              } else if tag == WAL_COMMIT_TAG {
  ```
  with:
  ```swift
              if tag == WAL_APPEND_TAG {
                  guard let rest = readExact(recordSizeAppend - 4) else { break }
                  // rest = listID(4) oldLen(4) delta(4) idsOff(8) codesOff(8) vecsOff(8) crc32(4) = 40 bytes
                  let storedCRC = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 36, as: UInt32.self)) }
                  var payload = [UInt8](repeating: 0, count: recordSizeAppend - 4) // tag + listID..vecsOff = 40 bytes
                  tagBytes.withUnsafeBytes { payload.replaceSubrange(0..<4, with: $0) }
                  payload.replaceSubrange(4..<(recordSizeAppend - 4), with: rest[0..<(recordSizeAppend - 8)])
                  let calc = payload.withUnsafeBytes { CRC32.hash($0.baseAddress!, payload.count) }
                  // A torn/corrupt append record means the log is unreliable from here on —
                  // stop replay, exactly like the WAL_COMMIT_TAG branch already does on its
                  // own CRC mismatch. Append records don't themselves drive recovered state
                  // (only WAL_COMMIT_TAG's newLen does); this CRC is validated purely to
                  // detect corruption and halt before trusting anything that follows it.
                  if calc != storedCRC { break }
              } else if tag == WAL_COMMIT_TAG {
  ```
  This is the "wire it up" option (rather than "stop computing it"): the write side already computes and stores this CRC on every append (Step 4's `writeWalAppend`), so validating it on replay is a small, self-contained addition with a real payoff — it turns a previously-silent gap (a torn/corrupt append record was read and discarded with no check at all) into a detected-and-halted one, matching the WAL_COMMIT_TAG branch's existing policy.

- [ ] **Step 6 — Leave `version_minor` alone (already correctly out of scope).**
  No code change. `verMinor` (`VIndexMmap.swift`, version-check block) is read and placed into an error-message `.info(...)` field on the already-failing major-version-mismatch path and never gates any branch; the design's own "single format version" policy explicitly leaves this inert. Add a one-line comment for the next reader, immediately above the existing `let verMinor = ...` line:
  ```swift
          // NOTE(Phase-2 B13): verMinor is read and reported in error diagnostics only; it does
          // not gate any behavior under the current single-format-version policy. Left as-is
          // intentionally — not a bug, not wired up. See PHASE4-ROUTING if minor-version gating
          // is ever introduced.
  ```

- [ ] **Step 7 — Add WAL replay tests (none existed before this task).**
  Append to `Tests/VectorIndexTests/VIndexMmapErrorTests.swift`, just before the closing `}` of `final class VIndexMmapErrorTests`, reusing the file's existing `tempPath()`, `readUnalignedLE32`, `readUnalignedLE64` helpers:
  ```swift
      // MARK: - WAL replay (B13) — no prior coverage existed for mmap_wal_replay at all.

      /// Locates the ListsDesc section's file offset by parsing the header + TOC directly
      /// (same technique as testSectionCRCMismatchThrows above), then overwrites list
      /// `listID`'s packed `length` field (record-relative offset +4) in place.
      private func setListLength(path: String, listID: Int, newLength: UInt32) throws {
          let fd = Darwin.open(path, O_RDWR | O_CLOEXEC)
          XCTAssertGreaterThanOrEqual(fd, 0)
          defer { _ = Darwin.close(fd) }
          var hdrBuf = [UInt8](repeating: 0, count: 256)
          _ = hdrBuf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, 256, 0) }
          let tocOffset = hdrBuf.withUnsafeBytes { readUnalignedLE64($0.baseAddress!.advanced(by: 56)) }
          let tocEntries = Int(hdrBuf.withUnsafeBytes { readUnalignedLE32($0.baseAddress!.advanced(by: 64)) })
          let DISK_TOC_ENTRY_SIZE = 36
          var tocAll = [UInt8](repeating: 0, count: tocEntries * DISK_TOC_ENTRY_SIZE)
          _ = tocAll.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocAll.count, off_t(tocOffset)) }
          var listsDescOffset: UInt64 = 0
          var found = false
          tocAll.withUnsafeBytes { raw in
              for i in 0..<tocEntries {
                  let base = raw.baseAddress!.advanced(by: i * DISK_TOC_ENTRY_SIZE)
                  if readUnalignedLE32(base) == SectionType.listsDesc.rawValue {
                      listsDescOffset = readUnalignedLE64(base.advanced(by: 4))
                      found = true
                      break
                  }
              }
          }
          XCTAssertTrue(found, "ListsDesc TOC entry not found")
          var v = newLength.littleEndian
          let fieldOffset = off_t(listsDescOffset) + off_t(listID * 64 + 4)
          _ = withUnsafeBytes(of: &v) { pwrite(fd, $0.baseAddress, 4, fieldOffset) }
      }

      func testWalReplayAppliesLengthFromValidCommitRecord() throws {
          let path = tempPath()
          let m = 4
          let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 16, payloadCap: 16, includeIDMap: false)
          defer {
              try? mmap.close()
              _ = try? FileManager.default.removeItem(atPath: path)
              _ = try? FileManager.default.removeItem(atPath: path + ".wal")
          }

          let n = 3
          let ids: [UInt64] = [10, 11, 12]
          let codes = [UInt8](repeating: 7, count: n * m)
          let res = try mmap.mmap_append_begin(listID: 0, addLen: n)
          try ids.withUnsafeBufferPointer { idBuf in
              try codes.withUnsafeBufferPointer { codeBuf in
                  try mmap.mmap_append_commit(res, idsSrc: UnsafeRawPointer(idBuf.baseAddress!), codesSrc: UnsafeRawPointer(codeBuf.baseAddress!), vecsSrc: nil)
              }
          }
          XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, n)

          // Simulate a crash where the fsync'd WAL made it to disk but the separate,
          // synchronous listsDesc-length write did not: roll the on-disk length back to 0.
          try setListLength(path: path, listID: 0, newLength: 0)
          XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, 0, "sanity: rollback landed")

          try mmap.mmap_wal_replay()
          XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, n,
                         "replay must restore length from the validated WAL commit record")
      }

      func testWalReplayStopsAtCorruptAppendRecordCRC() throws {
          let path = tempPath()
          let m = 4
          let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 16, payloadCap: 16, includeIDMap: false)
          defer {
              try? mmap.close()
              _ = try? FileManager.default.removeItem(atPath: path)
              _ = try? FileManager.default.removeItem(atPath: path + ".wal")
          }

          let n = 3
          let ids: [UInt64] = [10, 11, 12]
          let codes = [UInt8](repeating: 7, count: n * m)
          let res = try mmap.mmap_append_begin(listID: 0, addLen: n)
          try ids.withUnsafeBufferPointer { idBuf in
              try codes.withUnsafeBufferPointer { codeBuf in
                  try mmap.mmap_append_commit(res, idsSrc: UnsafeRawPointer(idBuf.baseAddress!), codesSrc: UnsafeRawPointer(codeBuf.baseAddress!), vecsSrc: nil)
              }
          }
          try setListLength(path: path, listID: 0, newLength: 0)

          // Flip a byte inside the WAL append record's CRC field. writeWalAppend runs before
          // writeWalCommit, so this is the very first record in the .wal file: WalAppend is
          // 44 bytes (tag4+listID4+oldLen4+delta4+idsOff8+codesOff8+vecsOff8+crc32(4)), so its
          // crc32 field is at absolute file offset 40..43.
          let walPath = path + ".wal"
          let walFD = Darwin.open(walPath, O_RDWR | O_CLOEXEC)
          XCTAssertGreaterThanOrEqual(walFD, 0)
          defer { _ = Darwin.close(walFD) }
          var crcByte: UInt8 = 0
          _ = withUnsafeMutableBytes(of: &crcByte) { pread(walFD, $0.baseAddress, 1, 40) }
          crcByte ^= 0xFF
          _ = withUnsafeBytes(of: &crcByte) { pwrite(walFD, $0.baseAddress, 1, 40) }

          try mmap.mmap_wal_replay()

          XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, 0,
                         "corrupt append record must halt replay before the commit record is ever applied")
      }
  ```

- [ ] **Step 8 — Build.**
  ```bash
  swift build 2>&1 | tail -30
  ```
  Expected: `Build complete!`.

- [ ] **Step 9 — Run the named filters for this task.**
  ```bash
  swift test --filter 'VIndexMmapErrorTests|RegressionA3_RemapTOCTests|RegressionA2_DurableListStatsTests|Kernel30AppendTests' 2>&1 | tail -80
  ```
  Expected: all pre-existing tests still pass, plus the two new methods `testWalReplayAppliesLengthFromValidCommitRecord` and `testWalReplayStopsAtCorruptAppendRecordCRC` report `passed`.

- [ ] **Step 10 — Commit.**
  ```bash
  git add Sources/VectorIndex/Kernels/VIndexMmap.swift Tests/VectorIndexTests/VIndexMmapErrorTests.swift
  git commit -m "$(cat <<'EOF'
  refactor(mmap): tidy toHost/CRC/scratch paths, validate WAL append CRC on replay (B13)

  - Delete fromHost (zero call sites).
  - computeHeaderCRC now hashes the header in place via a resumable CRC32.update/finalize
    API instead of copying 256 bytes per open (also avoids needing write access to a
    possibly read-only mapping).
  - writeWalAppend/writeWalCommit now share one reusable scratch buffer instead of
    allocating a fresh one per call.
  - mmap_wal_replay now validates the WAL append record's CRC (previously computed on
    write but silently discarded on replay) and halts replay on mismatch, matching the
    commit record's existing policy. Added the first tests ever written against
    mmap_wal_replay to cover both the happy path and the halt-on-corruption path.

  Co-Authored-By: use the executing assistant's standard Co-Authored-By trailer
  EOF
  )"
  ```

---

### Task 12: B12 — IDMap single-backend (`IDMap.swift`)

**Files:**
- Modify: `Sources/VectorIndex/Kernels/IDMap.swift` (edits at lines 5-17 `HashTableImpl`/`IDMapOpts`, 63-85 `HashTable` enum, 105-123 `RobinHoodTable`/`LinearProbingTable`, 183-193 `Impl.init`, 342 `idmapRehash`)

**Interfaces:**
- Consumes: none new.
- Produces: `internal struct IDMapOpts` unchanged shape (same field names/types/defaults; only member access-level changes from `public` to implicit-`internal`); `IDMap.Impl.hashTable` changes type from the deleted `private enum HashTable` to `private struct SwissTable` directly (both are file-private/module-invisible types, so this is not observable from any caller — `IDMap`'s own public methods (`append`, `lookup`, `externalID(for:)`, etc.) are unchanged).
- `public enum HashTableImpl` keeps all 3 cases (`.swissTable` live; `.robinHood`/`.linearProbing` marked `@available(*, deprecated)` rather than deleted, since the enum itself is `public` — see PHASE4-ROUTING at the end of this document).

---

- [ ] **Step 1 — Deprecate (don't delete) the two unused `HashTableImpl` cases.**
  `HashTableImpl` is a `public enum` (verify: `grep -n "public enum HashTableImpl" Sources/VectorIndex/Kernels/IDMap.swift` → line 5). Because it's public, removing cases would be a breaking source change for any external consumer that references them (even though no in-repo code does — `grep -rn "robinHood\|linearProbing" Sources/ Tests/` outside `IDMap.swift` returns zero hits). Replace:
  ```swift
  public enum HashTableImpl: Sendable { case swissTable, robinHood, linearProbing }
  ```
  with:
  ```swift
  public enum HashTableImpl: Sendable {
      case swissTable
      @available(*, deprecated, message: "Robin Hood hashing backend removed in VectorIndex 0.2.0 (Phase-2 cleanup, B12): IDMap now always uses SwissTable. This case has no effect and is retained only for source compatibility; scheduled for removal in a future major version. See PHASE4-ROUTING.")
      case robinHood
      @available(*, deprecated, message: "Linear-probing hashing backend removed in VectorIndex 0.2.0 (Phase-2 cleanup, B12): IDMap now always uses SwissTable. This case has no effect and is retained only for source compatibility; scheduled for removal in a future major version. See PHASE4-ROUTING.")
      case linearProbing
  }
  ```

- [ ] **Step 2 — Fix `IDMapOpts`'s wrongly-`public` members (it is an `internal` type, so this is non-breaking).**
  Replace:
  ```swift
  internal struct IDMapOpts: Sendable {
      public var allowReplace: Bool = false
      public var hashTableImpl: HashTableImpl = .swissTable
      public var capacityHint: Int = 1_000
      public var maxLoadFactor: Double = 0.875
      public var concurrency: IDMapConcurrency = .singleWriter
      public var enableBloom: Bool = false
      public var enableTelemetry: Bool = false
      public static var `default`: IDMapOpts { IDMapOpts() }
  }
  ```
  with:
  ```swift
  internal struct IDMapOpts: Sendable {
      var allowReplace: Bool = false
      /// Inert since this B12 cleanup collapsed IDMap onto a single SwissTable backend
      /// (Robin Hood / linear-probing variants had zero production or test users — see
      /// commit history for B12). Retained only so callers constructing `IDMapOpts` with
      /// an explicit `hashTableImpl:` argument keep compiling; the value is no longer read.
      var hashTableImpl: HashTableImpl = .swissTable
      var capacityHint: Int = 1_000
      var maxLoadFactor: Double = 0.875
      var concurrency: IDMapConcurrency = .singleWriter
      var enableBloom: Bool = false
      var enableTelemetry: Bool = false
      static var `default`: IDMapOpts { IDMapOpts() }
  }
  ```

- [ ] **Step 3 — Delete the `HashTable` enum-dispatch wrapper and the `RobinHoodTable`/`LinearProbingTable` backends.**
  Delete the whole `HashTable` enum (current lines 63-85):
  ```swift
  private enum HashTable {
      case swiss(SwissTable), robin(RobinHoodTable), linear(LinearProbingTable)
      static func allocate(buckets: Int, impl: HashTableImpl) -> HashTable {
          switch impl {
          case .swissTable:
              let bc = max(16, (buckets + 15) & ~15)
              return .swiss(SwissTable(bucketCount: bc))
          case .robinHood:
              let bc = max(8, nextPow2(buckets))
              return .robin(RobinHoodTable(bucketCount: bc))
          case .linearProbing:
              let bc = max(8, nextPow2(buckets))
              return .linear(LinearProbingTable(bucketCount: bc))
          }
      }
      var bucketCount: Int { switch self { case .swiss(let t): return t.bucketCount; case .robin(let t): return t.bucketCount; case .linear(let t): return t.bucketCount } }
      var count: Int { switch self { case .swiss(let t): return t.count; case .robin(let t): return t.count; case .linear(let t): return t.count } }
      mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { switch self { case .swiss(var t): let r=t.lookup(key); self = .swiss(t); return r; case .robin(var t): let r=t.lookup(key); self = .robin(t); return r; case .linear(var t): let r=t.lookup(key); self = .linear(t); return r } }
      mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { switch self { case .swiss(var t): let p=try t.insert(key, value); self = .swiss(t); return p; case .robin(var t): let p=try t.insert(key, value); self = .robin(t); return p; case .linear(var t): let p=try t.insert(key, value); self = .linear(t); return p } }
      mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { switch self { case .swiss(var t): let p=t.updateValue(for: key, to: value); self = .swiss(t); return p; case .robin(var t): let p=t.updateValue(for: key, to: value); self = .robin(t); return p; case .linear(var t): let p=t.updateValue(for: key, to: value); self = .linear(t); return p } }
      mutating func erase(_ key: UInt64) -> (Bool, Int?) { switch self { case .swiss(var t): let r=t.erase(key); self = .swiss(t); return r; case .robin(var t): let r=t.erase(key); self = .robin(t); return r; case .linear(var t): let r=t.erase(key); self = .linear(t); return r } }
      func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { switch self { case .swiss(let t): try t.forEach(body); case .robin(let t): try t.forEach(body); case .linear(let t): try t.forEach(body) } }
  }
  ```
  Leave `private struct SwissTable { ... }` (current lines 87-103) completely untouched.

  Then delete `RobinHoodTable` and `LinearProbingTable` in their entirety (current lines 105-123):
  ```swift
  private struct RobinHoodTable { struct Entry { var externalID: UInt64 = 0; var internalID: Int64 = -1; var dib: UInt8 = 0 }
      var entries: [Entry]; var bucketCount: Int; var count: Int=0
      init(bucketCount: Int) { self.bucketCount=bucketCount; self.entries=[Entry](repeating: Entry(), count: bucketCount)}
      mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { var curKey=key; var curVal=value; var dib: UInt8=0; var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { entries[idx]=Entry(externalID: curKey, internalID: curVal, dib: dib); count &+= 1; return probes } ; if e.externalID==curKey { entries[idx].internalID=curVal; return probes } ; if e.dib < dib { entries[idx]=Entry(externalID: curKey, internalID: curVal, dib: dib); curKey=e.externalID; curVal=e.internalID; dib=e.dib } ; idx=(idx+1)&(bucketCount-1); if dib==255 { throw ErrorBuilder(.capacityExceeded, operation: "idmap_robin_insert").message("Excessive probing in hash table").info("dib", "255").build() } ; dib &+= 1 } ; throw ErrorBuilder(.capacityExceeded, operation: "idmap_robin_insert").message("Hash table full").info("bucket_count", "\(bucketCount)").info("count", "\(count)").build() }
      mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { let r=lookup(key); if r.0 { var idx=hashH1(key, bucketCount); var dib: UInt8=0; var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { return nil } ; if e.externalID==key { entries[idx].internalID=value; return probes } ; if e.dib < dib { return nil } ; idx=(idx+1)&(bucketCount-1); dib &+= 1 } } ; return nil }
      mutating func erase(_ key: UInt64) -> (Bool, Int?) { var idx=hashH1(key, bucketCount); var dib: UInt8=0; var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { return (false, probes) } ; if e.externalID==key { var j=idx; var k=(j+1)&(bucketCount-1); while entries[k].externalID != 0 && entries[k].dib > 0 { entries[j]=Entry(externalID: entries[k].externalID, internalID: entries[k].internalID, dib: entries[k].dib &- 1); j=k; k=(k+1)&(bucketCount-1) } ; entries[j]=Entry(); count &-= 1; return (true, probes) } ; if e.dib < dib { return (false, probes) } ; idx=(idx+1)&(bucketCount-1); dib &+= 1 } ; return (false, probes) }
      mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { var idx=hashH1(key, bucketCount); var dib: UInt8=0; var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { return (false, -1, probes) } ; if e.externalID==key { return (true, e.internalID, probes) } ; if e.dib < dib { return (false, -1, probes) } ; idx=(idx+1)&(bucketCount-1); dib &+= 1 } ; return (false, -1, probes) }
      func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { for e in entries where e.externalID != 0 { try body(e.externalID, e.internalID) } }
  }

  private struct LinearProbingTable { enum State: UInt8 { case empty=0, deleted=1, full=2 } ; struct Entry { var externalID: UInt64 = 0; var internalID: Int64 = -1; var st: State = .empty }
      var entries: [Entry]; var bucketCount: Int; var count: Int=0
      init(bucketCount: Int) { self.bucketCount=bucketCount; self.entries=[Entry](repeating: Entry(), count: bucketCount) }
      mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; if entries[idx].st != .full { entries[idx]=Entry(externalID: key, internalID: value, st: .full); count &+= 1; return probes } ; if entries[idx].externalID==key { entries[idx].internalID=value; return probes } ; idx=(idx+1)&(bucketCount-1) } ; throw ErrorBuilder(.capacityExceeded, operation: "idmap_linear_insert").message("Hash table full").info("bucket_count", "\(bucketCount)").info("count", "\(count)").build() }
      mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { let r=lookup(key); if r.0 { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.st == .empty { return nil } ; if e.st == .full && e.externalID==key { entries[idx].internalID=value; return probes } ; idx=(idx+1)&(bucketCount-1) } } ; return nil }
      mutating func erase(_ key: UInt64) -> (Bool, Int?) { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.st == .empty { return (false, probes) } ; if e.st == .full && e.externalID==key { entries[idx].st = .deleted; count &-= 1; return (true, probes) } ; idx=(idx+1)&(bucketCount-1) } ; return (false, probes) }
      mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.st == .empty { return (false, -1, probes) } ; if e.st == .full && e.externalID==key { return (true, e.internalID, probes) } ; idx=(idx+1)&(bucketCount-1) } ; return (false, -1, probes) }
      func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { for e in entries where e.st == .full { try body(e.externalID, e.internalID) } }
  }
  ```

- [ ] **Step 4 — Point `IDMap.Impl` at `SwissTable` directly.**
  In `IDMap.Impl`, change:
  ```swift
      fileprivate var hashTable: HashTable
  ```
  to:
  ```swift
      fileprivate var hashTable: SwissTable
  ```
  and change:
  ```swift
      fileprivate var retired: [HashTable] = []
  ```
  to:
  ```swift
      fileprivate var retired: [SwissTable] = []
  ```
  and in `Impl.init`, change:
  ```swift
          self.hashTable = HashTable.allocate(buckets: hashBuckets, impl: opts.hashTableImpl)
  ```
  to:
  ```swift
          self.hashTable = SwissTable(bucketCount: max(16, (hashBuckets + 15) & ~15))
  ```
  (This preserves `HashTable.allocate`'s `.swissTable` bucket-rounding formula exactly, so bucket-count sizing behavior is unchanged.)

- [ ] **Step 5 — Fix `idmapRehash` the same way.**
  Change:
  ```swift
  internal func idmapRehash(_ map: IDMap, newBucketCount: Int) throws { let impl = map.impl; let buckets = max(16, nextPow2(newBucketCount)); var newTable = HashTable.allocate(buckets: buckets, impl: impl.opts.hashTableImpl); for i in 0..<impl.count { if impl.tombstones?.isSet(i) == true { continue } ; let ext = impl.extByInt[Int(i)]; if ext == 0 && i != 0 { continue } ; _ = try newTable.insert(ext, i) } ; if let lock = impl.rwLock { lock.writeLock(); let old = impl.hashTable; impl.hashTable = newTable; lock.writeUnlock(); impl.retired.append(old) } else { let old = impl.hashTable; impl.hashTable = newTable; _ = old } }
  ```
  to:
  ```swift
  internal func idmapRehash(_ map: IDMap, newBucketCount: Int) throws { let impl = map.impl; let buckets = max(16, nextPow2(newBucketCount)); var newTable = SwissTable(bucketCount: max(16, (buckets + 15) & ~15)); for i in 0..<impl.count { if impl.tombstones?.isSet(i) == true { continue } ; let ext = impl.extByInt[Int(i)]; if ext == 0 && i != 0 { continue } ; _ = try newTable.insert(ext, i) } ; if let lock = impl.rwLock { lock.writeLock(); let old = impl.hashTable; impl.hashTable = newTable; lock.writeUnlock(); impl.retired.append(old) } else { let old = impl.hashTable; impl.hashTable = newTable; _ = old } }
  ```

- [ ] **Step 6 — Build.**
  ```bash
  swift build 2>&1 | tail -40
  ```
  Expected: `Build complete!`. You should see deprecation warnings at any remaining reference to `.robinHood`/`.linearProbing` (there should be none left in `Sources/`/`Tests/` per the Step-1 grep) — if any appear, that is a signal you missed a caller; investigate before proceeding.

- [ ] **Step 7 — Run the named filters for this task.**
  ```bash
  swift test --filter 'IDMapPersistenceTests|IVFTests|IVFMoreTests|IVFRecallTests|IVFListMaintenanceTests' 2>&1 | tail -80
  ```
  Expected: `IDMapPersistenceTests`'s two methods still report as **skipped** (unchanged pre-existing `XCTSkip`, not a regression — B12 does not touch the mmap-persistence CRC issue those skips are gated on) and all `IVFTests`/`IVFMoreTests`/`IVFRecallTests`/`IVFListMaintenanceTests` methods pass. These exercise `IVFIndex`'s `idMap50` (always constructed with `opts: .default`, i.e. `.swissTable` — confirmed the only backend any production code ever selects) end-to-end via insert/query/persist paths, which is the real regression coverage for this change since no dedicated IDMap unit-test suite runs today.

- [ ] **Step 8 — Commit.**
  ```bash
  git add Sources/VectorIndex/Kernels/IDMap.swift
  git commit -m "$(cat <<'EOF'
  refactor(idmap): collapse to a single SwissTable backend (B12)

  RobinHoodTable and LinearProbingTable had zero production or test callers (IVFIndex
  always constructs IDMapOpts.default, i.e. .swissTable) and existed only behind a
  3-way enum-dispatch wrapper that copied the active table in and out of `self` on every
  op. Delete both unused backends and the dispatch wrapper; IDMap.Impl now holds a
  SwissTable directly. HashTableImpl (public) keeps all 3 cases for source compatibility,
  with .robinHood/.linearProbing marked deprecated rather than removed. Also fixes
  IDMapOpts's stored properties being marked `public` on an `internal` type (dead access
  modifiers with no effect); safe since the type itself was never externally visible.

  Co-Authored-By: use the executing assistant's standard Co-Authored-By trailer
  EOF
  )"
  ```

---

### Task 13: B14 + B15 — IDFilter + CandidateReservoir, tests-first

**Files:**
- Create: `Tests/VectorIndexTests/IDFilterTests.swift`
- Create: `Tests/VectorIndexTests/CandidateReservoirTests.swift`
- Modify: `Sources/VectorIndex/Operations/Filtering/IDFilter.swift` (lines 104-115 `FilterMode.shouldKeep`, 316-356 `idFilterCompactN`)
- Modify: `Sources/VectorIndex/Operations/Reservoir/CandidateReservoir.swift` (lines 249-264 `.adaptive` branch of `pushBatch`, 289-327 `extractTopK`, new buffer-based quickselect helpers near line 453)

**Interfaces:**
- Consumes: none new.
- Produces: `public func idFilterCompactN(...) -> Int` (signature unchanged; body no longer allocates); `public func extractTopK(k:topScores:topIDs:)` (signature unchanged; body now partitions instead of fully sorting); new `@usableFromInline internal` buffer-based quickselect helpers on `CandidateReservoir` (`quickselectTopBuffer`, `partitionAroundPivotBuffer`, `medianOfThreeIndexBuffer`, `swapAtBuffer`) used only by `extractTopK` — the existing self-mutating `quickselectTop`/`partitionAroundPivot`/`medianOfThreeIndex`/`swapAt` used by the hot `pruneToTopC()` path are **left untouched, byte-for-byte**, to avoid any risk to that path; this duplicates ~40 lines but is the deliberate, documented trade-off (see Step 8 below).
- `FilterMode.shouldKeep(bit:)` (internal, zero call sites) is deleted outright — it has no access modifier (defaults to `internal`), so it is not part of the public API and needs no deprecation route.

Both `IDFilter.swift` and `CandidateReservoir.swift` have **zero existing test files** today (`grep -rln "IDFilter\|idFilterCompact\|idFilterMask\|idFilterPass" Tests/VectorIndexTests/` and `grep -rln "CandidateReservoir" Tests/VectorIndexTests/` both return nothing). Every sub-step below that touches behavior is preceded by a test that must be green against the **current, unmodified** code before you touch the implementation.

---

### 13a — Characterization tests (write first, run green against current code)

- [ ] **Step 1 — Create `Tests/VectorIndexTests/IDFilterTests.swift`.**
  ```swift
  import XCTest
  @testable import VectorIndex

  final class IDFilterTests: XCTestCase {
      // Characterization test pinning idFilterCompactN's current (mask-allocating) behavior
      // BEFORE the B14 refactor, so the refactor can be verified byte-for-byte identical.
      //
      // Fixture: 8 ids [0...7], one allowlist bitset (bit=1 keeps ids {0,2,4,6}), one
      // denylist bitset (bit=1 drops ids {2,6}). Composed keep-set = allow AND NOT deny:
      //   id 0: allow=1, deny=0 -> KEEP    id 2: allow=1, deny=1 -> DROP (denied)
      //   id 4: allow=1, deny=0 -> KEEP    id 6: allow=1, deny=1 -> DROP (denied)
      //   id 1,3,5,7: allow=0 -> DROP
      // Expected stable-order result: ids [0, 4], scores [10, 14].
      func testIdFilterCompactNComposedAllowAndDeny() {
          let n = 8
          let capacity = 64
          var allow0Word: UInt64 = 0
          var denyWord: UInt64 = 0
          for id in [0, 2, 4, 6] { allow0Word |= (1 << UInt64(id)) }
          for id in [2, 6] { denyWord |= (1 << UInt64(id)) }

          let ids: [Int64] = (0..<n).map { Int64($0) }
          let scores: [Float] = (0..<n).map { Float(10 + $0) }
          var idsOut = [Int64](repeating: -1, count: n)
          var scoresOut = [Float](repeating: .nan, count: n)

          let kept = withUnsafePointer(to: &allow0Word) { a0 in
              withUnsafePointer(to: &denyWord) { dp in
                  ids.withUnsafeBufferPointer { idsBuf in
                      scores.withUnsafeBufferPointer { scoresBuf in
                          idsOut.withUnsafeMutableBufferPointer { idsOutBuf in
                              scoresOut.withUnsafeMutableBufferPointer { scoresOutBuf in
                                  idFilterCompactN(
                                      filters: [a0, dp],
                                      modes: [.allowlist, .denylist],
                                      filterCount: 2,
                                      idsIn: idsBuf.baseAddress!,
                                      scoresIn: scoresBuf.baseAddress,
                                      count: n,
                                      capacity: capacity,
                                      idsOut: idsOutBuf.baseAddress!,
                                      scoresOut: scoresOutBuf.baseAddress
                                  )
                              }
                          }
                      }
                  }
              }
          }

          XCTAssertEqual(kept, 2)
          XCTAssertEqual(Array(idsOut[0..<kept]), [0, 4])
          XCTAssertEqual(Array(scoresOut[0..<kept]), [10, 14])
      }

      // F=0 (no filters at all) must keep every id — idFilterPassN with all-nil allow/deny
      // pointers returns true after the bounds check.
      func testIdFilterCompactNWithZeroFiltersKeepsAll() {
          let n = 4
          let ids: [Int64] = [5, 6, 7, 8]
          let scores: [Float] = [1, 2, 3, 4]
          var idsOut = [Int64](repeating: -1, count: n)
          var scoresOut = [Float](repeating: .nan, count: n)

          let kept = ids.withUnsafeBufferPointer { idsBuf in
              scores.withUnsafeBufferPointer { scoresBuf in
                  idsOut.withUnsafeMutableBufferPointer { idsOutBuf in
                      scoresOut.withUnsafeMutableBufferPointer { scoresOutBuf in
                          idFilterCompactN(
                              filters: [], modes: [], filterCount: 0,
                              idsIn: idsBuf.baseAddress!, scoresIn: scoresBuf.baseAddress,
                              count: n, capacity: 64,
                              idsOut: idsOutBuf.baseAddress!, scoresOut: scoresOutBuf.baseAddress
                          )
                      }
                  }
              }
          }

          XCTAssertEqual(kept, 4)
          XCTAssertEqual(idsOut, ids)
          XCTAssertEqual(scoresOut, scores)
      }
  }
  ```

- [ ] **Step 2 — Create `Tests/VectorIndexTests/CandidateReservoirTests.swift`.**
  ```swift
  import XCTest
  @testable import VectorIndex

  final class CandidateReservoirTests: XCTestCase {
      // Characterization test pinning extractTopK's current sort-based behavior BEFORE the
      // B15 quickselect-based rewrite, so the refactor can be verified for exact parity
      // (including tie-break-by-smaller-ID ordering).
      //
      // Fixture: 4 candidates pushed in .block mode with headroom 5 (capacity 4,
      // reserveExtra 0.10 -> bufferCapacity = 4 + ceil(4*0.10) = 5), so no auto-prune fires
      // during pushBatch (size only reaches 4, never >= bufferCapacity) and size/order are
      // exactly as pushed: id 10 score 5.0, id 20 score 3.0 (tie w/ id 30), id 30 score 3.0,
      // id 40 score 1.0. L2 metric (smaller is better). Expected best-first: 40, 20, 30, 10
      // (id 20 before 30 on the score tie, since 20 < 30).
      func testExtractTopKOrdersByScoreThenIDForL2() {
          let reservoir = CandidateReservoir(
              capacity: 4, metric: .l2,
              options: ReservoirOptions(mode: .block, reserveExtra: 0.10)
          )
          let ids: [Int64] = [10, 20, 30, 40]
          let scores: [Float] = [5.0, 3.0, 3.0, 1.0]
          ids.withUnsafeBufferPointer { idBuf in
              scores.withUnsafeBufferPointer { scoreBuf in
                  _ = reservoir.pushBatch(ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!, count: 4)
              }
          }
          XCTAssertEqual(reservoir.count, 4, "sanity: no auto-prune should have fired yet")

          var outScores = [Float](repeating: .nan, count: 4)
          var outIDs = [Int64](repeating: -1, count: 4)
          outScores.withUnsafeMutableBufferPointer { sp in
              outIDs.withUnsafeMutableBufferPointer { ip in
                  reservoir.extractTopK(k: 4, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
              }
          }
          XCTAssertEqual(outIDs, [40, 20, 30, 10])
          XCTAssertEqual(outScores, [1.0, 3.0, 3.0, 5.0])

          // Partial top-k must be a prefix of the full ordering.
          var outScores2 = [Float](repeating: .nan, count: 2)
          var outIDs2 = [Int64](repeating: -1, count: 2)
          outScores2.withUnsafeMutableBufferPointer { sp in
              outIDs2.withUnsafeMutableBufferPointer { ip in
                  reservoir.extractTopK(k: 2, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
              }
          }
          XCTAssertEqual(outIDs2, [40, 20])
          XCTAssertEqual(outScores2, [1.0, 3.0])
      }

      // extractTopK is documented as read-only; a second extraction must return the
      // identical result and must not change reservoir.count.
      func testExtractTopKIsReadOnly() {
          let reservoir = CandidateReservoir(capacity: 4, metric: .l2, options: ReservoirOptions(mode: .block, reserveExtra: 0.10))
          let ids: [Int64] = [10, 20, 30, 40]
          let scores: [Float] = [5.0, 3.0, 3.0, 1.0]
          ids.withUnsafeBufferPointer { idBuf in
              scores.withUnsafeBufferPointer { scoreBuf in
                  _ = reservoir.pushBatch(ids: idBuf.baseAddress!, scores: scoreBuf.baseAddress!, count: 4)
              }
          }
          func extract() -> ([Int64], [Float]) {
              var outScores = [Float](repeating: .nan, count: 4)
              var outIDs = [Int64](repeating: -1, count: 4)
              outScores.withUnsafeMutableBufferPointer { sp in
                  outIDs.withUnsafeMutableBufferPointer { ip in
                      reservoir.extractTopK(k: 4, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
                  }
              }
              return (outIDs, outScores)
          }
          let first = extract()
          let second = extract()
          XCTAssertEqual(first.0, second.0)
          XCTAssertEqual(first.1, second.1)
          XCTAssertEqual(reservoir.count, 4, "extractTopK must not change reservoir size")
      }

      // Regression test for the missing `.adaptive` overflow-prune guard (B15): with a small
      // enough capacity, the periodic (every-64-pushes) occupancy check can miss the
      // adaptiveThreshold crossing entirely before `size` reaches `bufferCapacity`, so
      // appendUnsorted's defensive fallback silently *grows* the buffer instead of pruning —
      // defeating the "no hot-path allocations" fixed-capacity design intent.
      //
      // capacity=100, reserveExtra=0.10 -> bufferCapacity = 100 + ceil(100*0.10) = 110.
      // adaptiveThreshold=0.75 -> switch-trigger occupancy = 75. The periodic check only
      // runs at size % 64 == 0, i.e. size=64 (occ 0.64, below threshold) then size=128
      // (already past bufferCapacity=110). Pushing 111 single-item batches crosses
      // bufferCapacity on push #111 without the periodic check ever tripping first.
      //
      // THIS TEST IS EXPECTED TO BE RED against the current, unmodified code (bufferCapacity
      // grows past 110 via appendUnsorted's defensive-grow branch) and MUST turn GREEN once
      // the `.adaptive` case gains the same `if size >= bufferCapacity { pruneToTopC() }`
      // guard `.block` already has. This is a standard red-green bugfix test, not a
      // characterization of current (buggy) behavior — do not "fix" the test to match the
      // bug.
      func testAdaptiveModePrunesBeforeBufferOverflow() {
          let capacity = 100
          let reservoir = CandidateReservoir(
              capacity: capacity, metric: .l2,
              options: ReservoirOptions(mode: .adaptive, reserveExtra: 0.10, adaptiveThreshold: 0.75, adaptiveInitialMode: .block)
          )
          let expectedBufferCapacity = capacity + Int(ceil(Double(capacity) * 0.10)) // 110

          let n = 111
          var oneID: Int64 = 0
          var oneScore: Float = 0
          for i in 0..<n {
              oneID = Int64(i)
              oneScore = Float(i)
              withUnsafePointer(to: &oneID) { idp in
                  withUnsafePointer(to: &oneScore) { sp in
                      _ = reservoir.pushBatch(ids: idp, scores: sp, count: 1)
                  }
              }
          }

          XCTAssertEqual(reservoir.bufferCapacity, expectedBufferCapacity,
                         "adaptive mode must prune at bufferCapacity instead of silently growing the buffer")
      }
  }
  ```

- [ ] **Step 3 — Build and confirm the new tests compile and give the expected pre-refactor signal.**
  ```bash
  swift build 2>&1 | tail -30
  swift test --filter 'IDFilterTests|CandidateReservoirTests' 2>&1 | tail -80
  ```
  Expected: `IDFilterTests` (both methods) and `CandidateReservoirTests.testExtractTopKOrdersByScoreThenIDForL2` / `testExtractTopKIsReadOnly` report **passed** (these characterize *current* behavior, so they must be green now). `testAdaptiveModePrunesBeforeBufferOverflow` is expected to **fail** right now (`XCTAssertEqual` reporting `bufferCapacity` as `220`, not `110`) — confirm the failure message shows the buggy grown value; this is the expected red state proving the repro is real, not a mistake in the test.

### 13b — B14: inline `idFilterCompactN`, delete dead `FilterMode.shouldKeep`

- [ ] **Step 4 — Delete the unused `FilterMode.shouldKeep` method.**
  In `Sources/VectorIndex/Operations/Filtering/IDFilter.swift`, replace:
  ```swift
  public enum FilterMode {
      case allowlist
      case denylist

      @inlinable
      func shouldKeep(bit: Bool) -> Bool {
          switch self {
          case .allowlist: return bit
          case .denylist:  return !bit
          }
      }
  }
  ```
  with:
  ```swift
  public enum FilterMode {
      case allowlist
      case denylist
  }
  ```
  (`shouldKeep` has no explicit access modifier, i.e. `internal`, and `grep -rn "shouldKeep(" Sources/ Tests/` matches only its own declaration — safe to delete outright, no deprecation needed.)

- [ ] **Step 5 — Inline the per-id test into `idFilterCompactN`, dropping the mask allocation.**
  Replace (current lines 316-356):
  ```swift
  /// Filter+compact with composed multi-filter using a precomputed mask.
  /// Generates mask first to amortize checks, then performs stable copy.
  ///
  /// ⚠️ **Allocation Note**: This function allocates a temporary `UInt8` mask buffer of size `n`.
  ///
  /// - Returns: count of kept elements.
  @inlinable
  public func idFilterCompactN(
      filters: [UnsafePointer<UInt64>?],
      modes: [FilterMode],
      filterCount F: Int,
      idsIn: UnsafePointer<Int64>,
      scoresIn: UnsafePointer<Float>?,
      count n: Int,
      capacity: Int,
      idsOut: UnsafeMutablePointer<Int64>,
      scoresOut: UnsafeMutablePointer<Float>?
  ) -> Int {
      // Build mask first (⚠️ allocates temporary buffer)
      var mask = [UInt8](repeating: 0, count: n)
      let kept = idFilterMaskN(
          filters: filters, modes: modes, filterCount: F,
          ids: idsIn, count: n, capacity: capacity,
          maskOut: &mask
      )

      // Stable compaction using mask
      var writeIdx = 0
      for i in 0..<n {
          if mask[i] == 1 {
              idsOut[writeIdx] = idsIn[i]
              if let sIn = scoresIn, let sOut = scoresOut {
                  sOut[writeIdx] = sIn[i]
              }
              writeIdx &+= 1
          }
      }
      assert(writeIdx == kept)
      return writeIdx
  }
  ```
  with:
  ```swift
  /// Filter+compact with composed multi-filter, single-pass (no mask allocation).
  /// Mirrors idFilterCompact's zero-allocation pattern for the single-filter case, but for
  /// up to 5 composed filters (4 allow + 1 deny).
  ///
  /// - Returns: count of kept elements.
  @inlinable
  public func idFilterCompactN(
      filters: [UnsafePointer<UInt64>?],
      modes: [FilterMode],
      filterCount F: Int,
      idsIn: UnsafePointer<Int64>,
      scoresIn: UnsafePointer<Float>?,
      count n: Int,
      capacity: Int,
      idsOut: UnsafeMutablePointer<Int64>,
      scoresOut: UnsafeMutablePointer<Float>?
  ) -> Int {
      precondition(F >= 0 && F <= 5, "Up to 5 filters supported (4 allow + 1 deny via mode)")

      // Map to at most 4 allows + optional deny (mirrors idFilterMaskN's own setup).
      var allowPtrs: [UnsafePointer<UInt64>?] = [nil, nil, nil, nil]
      var denyPtr: UnsafePointer<UInt64>?
      var aIdx = 0
      for f in 0..<F {
          let ptr = filters[f]
          switch modes[f] {
          case .allowlist:
              if aIdx < 4 { allowPtrs[aIdx] = ptr; aIdx &+= 1 }
          case .denylist:
              if denyPtr == nil { denyPtr = ptr }
          }
      }

      var writeIdx = 0
      for i in 0..<n {
          let id = idsIn[i]
          let pass = idFilterPassN(
              allow0: allowPtrs[0], allow1: allowPtrs[1], allow2: allowPtrs[2], allow3: allowPtrs[3],
              deny: denyPtr, id: id, capacity: capacity
          )
          if pass {
              idsOut[writeIdx] = id
              if let sIn = scoresIn, let sOut = scoresOut {
                  sOut[writeIdx] = sIn[i]
              }
              writeIdx &+= 1
          }
      }
      return writeIdx
  }
  ```

- [ ] **Step 6 — Build and re-run the IDFilter tests (must stay green — proves parity).**
  ```bash
  swift build 2>&1 | tail -30
  swift test --filter 'IDFilterTests' 2>&1 | tail -40
  ```
  Expected: both `IDFilterTests` methods still `passed`, unchanged from Step 3.

### 13c — B15: quickselect-based `extractTopK` + `.adaptive` overflow-prune fix

- [ ] **Step 7 — Add buffer-based quickselect helpers to `CandidateReservoir`, for `extractTopK`'s exclusive use.**
  In `Sources/VectorIndex/Operations/Reservoir/CandidateReservoir.swift`, immediately after the existing `medianOfThreeIndex(_:_:_:)` method (right before the `// MARK: - Ordering predicates` section), add:
  ```swift

    // MARK: - Selection: quickselect over caller-provided buffers (extractTopK only)

    /// Mirrors quickselectTop/partitionAroundPivot/medianOfThreeIndex/swapAt above exactly,
    /// but operates on explicit buffers instead of self.scores/self.ids, so extractTopK can
    /// reuse the same median-of-three quickselect on its own read-only copies without
    /// mutating the reservoir (which the self-based versions do, and which pruneToTopC()'s
    /// hot mutating path depends on — deliberately left untouched here to avoid any risk to
    /// that path; these two implementations must be kept in sync if the algorithm changes).
    @usableFromInline
    internal func quickselectTopBuffer(
      scoresBuf: UnsafeMutableBufferPointer<Float>,
      idsBuf: UnsafeMutableBufferPointer<Int64>,
      count: Int,
      countKeep k: Int
    ) {
      var left = 0
      var right = count &- 1
      let target = k &- 1

      while left <= right {
        let pivotIndex = medianOfThreeIndexBuffer(scoresBuf, idsBuf, left, (left &+ right) >> 1, right)
        let newPivot = partitionAroundPivotBuffer(scoresBuf, idsBuf, left: left, right: right, pivotIndex: pivotIndex)
        if newPivot == target { return }
        if target < newPivot {
          right = newPivot &- 1
        } else {
          left = newPivot &+ 1
        }
      }
    }

    @usableFromInline
    internal func partitionAroundPivotBuffer(
      _ scoresBuf: UnsafeMutableBufferPointer<Float>,
      _ idsBuf: UnsafeMutableBufferPointer<Int64>,
      left: Int, right: Int, pivotIndex: Int
    ) -> Int {
      swapAtBuffer(scoresBuf, idsBuf, pivotIndex, right)
      let pivotScore = scoresBuf[right]
      let pivotID = idsBuf[right]
      var store = left
      var i = left
      while i < right {
        if isBetter(scoreA: scoresBuf[i], idA: idsBuf[i], scoreB: pivotScore, idB: pivotID) {
          swapAtBuffer(scoresBuf, idsBuf, i, store)
          store &+= 1
        }
        i &+= 1
      }
      swapAtBuffer(scoresBuf, idsBuf, store, right)
      return store
    }

    @usableFromInline
    internal func medianOfThreeIndexBuffer(
      _ scoresBuf: UnsafeMutableBufferPointer<Float>,
      _ idsBuf: UnsafeMutableBufferPointer<Int64>,
      _ a: Int, _ b: Int, _ c: Int
    ) -> Int {
      let sa = scoresBuf[a], ia = idsBuf[a]
      let sb = scoresBuf[b], ib = idsBuf[b]
      let sc = scoresBuf[c], ic = idsBuf[c]

      let ab = isBetter(scoreA: sa, idA: ia, scoreB: sb, idB: ib)
      let bc = isBetter(scoreA: sb, idA: ib, scoreB: sc, idB: ic)
      let ac = isBetter(scoreA: sa, idA: ia, scoreB: sc, idB: ic)

      if ab {
        if bc { return b } else { return ac ? c : a }
      } else {
        if ac { return a } else { return bc ? c : b }
      }
    }

    @usableFromInline
    internal func swapAtBuffer(
      _ scoresBuf: UnsafeMutableBufferPointer<Float>,
      _ idsBuf: UnsafeMutableBufferPointer<Int64>,
      _ a: Int, _ b: Int
    ) {
      if a == b { return }
      let tmpS = scoresBuf[a]; scoresBuf[a] = scoresBuf[b]; scoresBuf[b] = tmpS
      let tmpI = idsBuf[a]; idsBuf[a] = idsBuf[b]; idsBuf[b] = tmpI
    }
  ```

- [ ] **Step 8 — Rewrite `extractTopK` to partition-then-sort-k instead of full-sort.**
  Replace (current lines 289-327):
  ```swift
    // MARK: - Extract Top‑K (read-only; does not modify reservoir)

    /// Extracts top‑K results (best-first) into caller-provided buffers.
    /// K must be ≤ current `count`.
    ///
    /// Complexity: O(count log count) for full sort; acceptable since it's off the hot path.
    /// If you need partial select, you can adapt this to a k‑select then sort K.
    @inlinable
    public func extractTopK(
      k: Int,
      topScores outScores: UnsafeMutablePointer<Float>,
      topIDs outIDs: UnsafeMutablePointer<Int64>
    ) {
      precondition(k >= 0 && k <= size, "k must be in [0, count]")

      // Copy to local work buffers (read-only operation per spec).
      var ws = [Float](repeating: 0, count: size)
      var wi = [Int64](repeating: 0, count: size)

      // Use .update(from:count:) per project deprecation guidance.
      scores.withUnsafeBufferPointer { sp in
        ws.withUnsafeMutableBufferPointer { wp in
          wp.baseAddress!.update(from: sp.baseAddress!, count: size)
        }
      }
      ids.withUnsafeBufferPointer { ip in
        wi.withUnsafeMutableBufferPointer { wp in
          wp.baseAddress!.update(from: ip.baseAddress!, count: size)
        }
      }

      // Sort entire set by "better first" comparator (deterministic).
      ws.indices.sorted { a, b in
        isBetter(scoreA: ws[a], idA: wi[a], scoreB: ws[b], idB: wi[b])
      }.prefix(k).enumerated().forEach { (j, idx) in
        outScores[j] = ws[idx]
        outIDs[j] = wi[idx]
      }
    }
  ```
  with:
  ```swift
    // MARK: - Extract Top‑K (read-only; does not modify reservoir)

    /// Extracts top‑K results (best-first) into caller-provided buffers.
    /// K must be ≤ current `count`.
    ///
    /// Complexity: O(count) expected quickselect partition (skipped entirely when k == count)
    /// followed by an O(k log k) sort of just the top k, vs. the previous O(count log count)
    /// full sort of every buffered candidate.
    @inlinable
    public func extractTopK(
      k: Int,
      topScores outScores: UnsafeMutablePointer<Float>,
      topIDs outIDs: UnsafeMutablePointer<Int64>
    ) {
      precondition(k >= 0 && k <= size, "k must be in [0, count]")
      guard k > 0 else { return }

      // Copy to local work buffers (read-only operation per spec: self.scores/self.ids are
      // never touched, so mode/heap/tau invariants are untouched too).
      var ws = [Float](repeating: 0, count: size)
      var wi = [Int64](repeating: 0, count: size)

      scores.withUnsafeBufferPointer { sp in
        ws.withUnsafeMutableBufferPointer { wp in
          wp.baseAddress!.update(from: sp.baseAddress!, count: size)
        }
      }
      ids.withUnsafeBufferPointer { ip in
        wi.withUnsafeMutableBufferPointer { wp in
          wp.baseAddress!.update(from: ip.baseAddress!, count: size)
        }
      }

      ws.withUnsafeMutableBufferPointer { wsBuf in
        wi.withUnsafeMutableBufferPointer { wiBuf in
          if k < size {
            quickselectTopBuffer(scoresBuf: wsBuf, idsBuf: wiBuf, count: size, countKeep: k)
          }
          // Sort just the first k by the same "better first" comparator (deterministic).
          let order = (0..<k).sorted { a, b in
            isBetter(scoreA: wsBuf[a], idA: wiBuf[a], scoreB: wsBuf[b], idB: wiBuf[b])
          }
          for (j, idx) in order.enumerated() {
            outScores[j] = wsBuf[idx]
            outIDs[j] = wiBuf[idx]
          }
        }
      }
    }
  ```
  Note the `guard k > 0 else { return }` early-out is a strict improvement (skips the size-length copy entirely for k=0) that is still contract-compatible: the original also wrote nothing to `outScores`/`outIDs` when `k == 0`.

- [ ] **Step 9 — Fix the `.adaptive` mode's missing overflow-prune guard.**
  In `pushBatch`, replace:
  ```swift
      case .adaptive:
        // In adaptive block phase until switch
        appendUnsorted(id: cid, score: s)
        acceptedInBatch &+= 1

        // Check occupancy periodically to keep overhead low.
        if (size & 63) == 0 {
          let occ = Float(size) / Float(C)
          if occ > opts.adaptiveThreshold {
            // Switch to heap: ensure we have exactly top‑C, then heapify (worst-at-root).
            if size > C { pruneToTopC() }
            heapifyWorstRoot()
            currentMode = .heap
            telemetry.modeSwitches &+= 1
          }
        }
  ```
  with:
  ```swift
      case .adaptive:
        // In adaptive block phase until switch
        appendUnsorted(id: cid, score: s)
        acceptedInBatch &+= 1

        // Defensive: never let size reach bufferCapacity without pruning first (mirrors
        // .block above). Without this, a push burst landing between two periodic occupancy
        // checks (every 64 pushes) can walk `size` past `bufferCapacity` before the ratio
        // check ever fires, silently triggering appendUnsorted's defensive buffer-growth
        // fallback instead of a prune — defeating the fixed-capacity design intent.
        if size >= bufferCapacity {
          pruneToTopC()
        }

        // Check occupancy periodically to keep overhead low.
        if (size & 63) == 0 {
          let occ = Float(size) / Float(C)
          if occ > opts.adaptiveThreshold {
            // Switch to heap: ensure we have exactly top‑C, then heapify (worst-at-root).
            if size > C { pruneToTopC() }
            heapifyWorstRoot()
            currentMode = .heap
            telemetry.modeSwitches &+= 1
          }
        }
  ```

- [ ] **Step 10 — Build and re-run all of this task's tests (must ALL be green now, including the previously-red one).**
  ```bash
  swift build 2>&1 | tail -30
  swift test --filter 'IDFilterTests|CandidateReservoirTests' 2>&1 | tail -100
  ```
  Expected: every method in both files reports `passed`, including `testAdaptiveModePrunesBeforeBufferOverflow` (now green — `bufferCapacity` stays `110`).

- [ ] **Step 11 — Run the broader named filters for this task (confirms no collateral regression).**
  ```bash
  swift test --filter 'IDFilterTests|CandidateReservoirTests|IVFTests|IVFMoreTests|IVFRecallTests' 2>&1 | tail -100
  ```
  Expected: all green.

- [ ] **Step 12 — Commit (one commit covering both B14 and B15, since they share the tests-first setup step).**
  ```bash
  git add Tests/VectorIndexTests/IDFilterTests.swift \
          Tests/VectorIndexTests/CandidateReservoirTests.swift \
          Sources/VectorIndex/Operations/Filtering/IDFilter.swift \
          Sources/VectorIndex/Operations/Reservoir/CandidateReservoir.swift
  git commit -m "$(cat <<'EOF'
  refactor(filter,reservoir): drop mask alloc in idFilterCompactN, quickselect extractTopK,
  fix .adaptive overflow prune (B14+B15)

  Neither IDFilter.swift nor CandidateReservoir.swift had any test coverage before this
  change, so both refactors are preceded by characterization tests pinning current
  behavior (idFilterCompactN's composed allow/deny output, extractTopK's tie-break order)
  plus one red-green bugfix test for a real, reachable bug: `.adaptive` mode's periodic
  (every-64-push) occupancy check can miss its threshold crossing before `size` reaches
  `bufferCapacity`, silently triggering a defensive buffer *growth* instead of a prune and
  defeating the fixed-capacity, no-hot-path-allocation design intent.

  - idFilterCompactN: inline the per-id test instead of building a size-n mask array first.
  - FilterMode.shouldKeep: deleted (internal, zero call sites).
  - extractTopK: quickselect-partition to the top k, then sort only the k-sized prefix,
    instead of fully sorting every buffered candidate. New buffer-parameterized quickselect
    helpers are added for this; the existing self-mutating ones used by pruneToTopC()'s hot
    path are left untouched to avoid any risk there.
  - CandidateReservoir.pushBatch's .adaptive case gains the same
    `if size >= bufferCapacity { pruneToTopC() }` guard .block already has.

  Co-Authored-By: use the executing assistant's standard Co-Authored-By trailer
  EOF
  )"
  ```

---

### Task 14: B16 — dedup/heap tidy (`CandidateDedup.swift`, `IVFSelect.swift`)

**Files:**
- Create: `Tests/VectorIndexTests/CandidateDedupSparsePagedTests.swift`
- Modify: `Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift` (lines 141-145 `Page`/`pageTable`, 272-321 `resetForNewQuery`, 421-459 `_testAndSet_sparse`)
- Modify: `Sources/VectorIndex/Kernels/IVFSelect.swift` (lines 768-954 `TopKHeap`/`MinHeap`/`MaxHeap`, call sites at ~377-383, ~549, ~564, ~690, ~730)

**Interfaces:**
- Consumes: none new.
- Produces: `@usableFromInline internal final class Page` (was `struct`) in `CandidateDedup.swift` — file-private-adjacent type (`@usableFromInline internal`, no external references per brief's grep), so the struct→class change is invisible to any caller; `private final class BoundedTopKHeap: TopKHeap` in `IVFSelect.swift` replaces `MinHeap`/`MaxHeap` (both `private`, zero external references — confirmed distinct from the unrelated `private struct MinHeap` in `HNSWTraversal.swift`, which is **not** touched by this task).

---

- [ ] **Step 1 — Add the SparsePaged characterization test (write first; `.sparsePaged` has zero existing coverage).**
  Create `Tests/VectorIndexTests/CandidateDedupSparsePagedTests.swift`:
  ```swift
  import XCTest
  @testable import VectorIndex

  final class CandidateDedupSparsePagedTests: XCTestCase {
      // Characterization test pinning `.sparsePaged` mode's current behavior BEFORE the B16
      // Page value-struct -> reference-type refactor. `.sparsePaged` has zero existing test
      // coverage (grep-confirmed), so this pins: cross-page allocation, first-seen/duplicate
      // semantics, page reuse across queries (only the FIRST-ever touch of a page counts
      // toward pagesAllocatedThisQuery), and per-touched-page clearing on resetForNewQuery().
      func testSparsePagedCrossPageAllocationDuplicateAndReset() {
          let idCapacity: Int64 = 200_000
          let vs = DefaultVisitedSet(idCapacity: idCapacity, opts: VisitedOpts(mode: .sparsePaged))

          // Three ids on three distinct pages (pageBits defaults to 15 -> 32,768 ids/page).
          let idPage0: Int64 = 10
          let idPage1: Int64 = 40_000   // 40_000 >> 15 == 1
          let idPage2: Int64 = 70_000   // 70_000 >> 15 == 2

          XCTAssertTrue(vs.testAndSet(id: idPage0), "first touch must be newly-seen")
          XCTAssertTrue(vs.testAndSet(id: idPage1))
          XCTAssertTrue(vs.testAndSet(id: idPage2))
          XCTAssertEqual(vs.pagesAllocatedThisQuery, 3, "three distinct pages must each allocate once")

          XCTAssertFalse(vs.testAndSet(id: idPage0), "second touch of the same id must be a duplicate")
          XCTAssertTrue(vs.contains(idPage1))
          XCTAssertFalse(vs.contains(Int64(41_000)), "an untouched id on the same page as idPage1 must not read as set")

          vs.resetForNewQuery()
          XCTAssertEqual(vs.pagesClearedThisQuery, 3, "reset must clear exactly the 3 pages touched last query")

          // Page reuse: touching a NEW id on the already-allocated page 0 must not re-count
          // as a fresh page allocation (the Page object persists across queries in pageTable).
          let idPage0Other: Int64 = 11
          XCTAssertTrue(vs.testAndSet(id: idPage0Other), "bits must be cleared after reset")
          XCTAssertEqual(vs.pagesAllocatedThisQuery, 0, "reusing an already-allocated page must not increment pagesAllocatedThisQuery")

          // Old id from the previous query must read as fresh again post-reset (bits cleared).
          XCTAssertTrue(vs.testAndSet(id: idPage0), "id from previous query must be newly-seen after reset")
      }
  }
  ```

- [ ] **Step 2 — Build and confirm the new test passes against current (struct-based) code.**
  ```bash
  swift build 2>&1 | tail -30
  swift test --filter 'CandidateDedupSparsePagedTests' 2>&1 | tail -30
  ```
  Expected: `testSparsePagedCrossPageAllocationDuplicateAndReset` reports `passed`.

- [ ] **Step 3 — Convert `Page` from a value struct to a reference class (removes all dictionary write-back dances).**
  In `Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift`, replace:
  ```swift
      @usableFromInline internal struct Page {
          var bits: UnsafeMutablePointer<UInt64>  // 4 KiB bitset
          var lastTouchedEpoch: UInt32            // dedup touched-pages per-query
      }
      @usableFromInline internal var pageTable: [Int64: Page] = [:] // pageID -> Page
  ```
  with:
  ```swift
      @usableFromInline internal final class Page {
          let bits: UnsafeMutablePointer<UInt64>  // 4 KiB bitset
          var lastTouchedEpoch: UInt32             // dedup touched-pages per-query
          init(bits: UnsafeMutablePointer<UInt64>, lastTouchedEpoch: UInt32) {
              self.bits = bits
              self.lastTouchedEpoch = lastTouchedEpoch
          }
      }
      @usableFromInline internal var pageTable: [Int64: Page] = [:] // pageID -> Page
  ```

- [ ] **Step 4 — Drop the now-unneeded write-back in the epoch-wrap branch of `resetForNewQuery`.**
  Replace:
  ```swift
          if mode == .sparsePaged, queryEpoch == 0 {
              for (pid, var page) in pageTable {
                  page.lastTouchedEpoch = 0
                  pageTable[pid] = page
                  let ptr = page.bits
                  for i in 0..<DefaultVisitedSet.wordsPerPage { ptr[i] = 0 }
              }
              queryEpoch = 1
              epochWraps &+= 1
          }
  ```
  with:
  ```swift
          if mode == .sparsePaged, queryEpoch == 0 {
              for (_, page) in pageTable {
                  page.lastTouchedEpoch = 0
                  let ptr = page.bits
                  for i in 0..<DefaultVisitedSet.wordsPerPage { ptr[i] = 0 }
              }
              queryEpoch = 1
              epochWraps &+= 1
          }
  ```
  (`page` is now a class reference, so mutating `page.lastTouchedEpoch` in place is visible through every other reference to the same `Page` — including the one already stored in `pageTable` — with no need to write it back into the dictionary.)

- [ ] **Step 5 — Drop the write-back dance in `_testAndSet_sparse`.**
  Replace:
  ```swift
      @usableFromInline
      internal func _testAndSet_sparse(_ id: Int64) -> Bool {
          assert(id >= 0, "ID must be non-negative")
          totalChecks &+= 1

          let pid = id >> Int64(pageBits)
          var page = pageTable[pid]

          // Allocate on demand
          if page == nil {
              let p = UnsafeMutablePointer<UInt64>.allocate(capacity: DefaultVisitedSet.wordsPerPage)
              p.initialize(repeating: 0, count: DefaultVisitedSet.wordsPerPage)
              page = Page(bits: p, lastTouchedEpoch: 0)
              pageTable[pid] = page
              pagesAllocatedThisQuery &+= 1
          }

          // If first touch this query, record for later clear
          if page!.lastTouchedEpoch != queryEpoch {
              page!.lastTouchedEpoch = queryEpoch
              pageTable[pid] = page
              touchedPages.append(pid)
          }

          // Bit test
          let inPage = id & pageMask
          let w = Int(inPage >> 6)
          let b = Int(inPage & 63)
          let mask: UInt64 = 1 &<< b

          let word = page!.bits[w]
          if (word & mask) == 0 {
              page!.bits[w] = word | mask
              uniqueCount &+= 1
              return true
          } else {
              duplicateCount &+= 1
              return false
          }
      }
  ```
  with:
  ```swift
      @usableFromInline
      internal func _testAndSet_sparse(_ id: Int64) -> Bool {
          assert(id >= 0, "ID must be non-negative")
          totalChecks &+= 1

          let pid = id >> Int64(pageBits)
          let page: Page
          if let existing = pageTable[pid] {
              page = existing
          } else {
              let p = UnsafeMutablePointer<UInt64>.allocate(capacity: DefaultVisitedSet.wordsPerPage)
              p.initialize(repeating: 0, count: DefaultVisitedSet.wordsPerPage)
              let newPage = Page(bits: p, lastTouchedEpoch: 0)
              pageTable[pid] = newPage
              pagesAllocatedThisQuery &+= 1
              page = newPage
          }

          // If first touch this query, record for later clear (mutate in place; class Page
          // needs no dictionary write-back since it is a reference type).
          if page.lastTouchedEpoch != queryEpoch {
              page.lastTouchedEpoch = queryEpoch
              touchedPages.append(pid)
          }

          // Bit test
          let inPage = id & pageMask
          let w = Int(inPage >> 6)
          let b = Int(inPage & 63)
          let mask: UInt64 = 1 &<< b

          let word = page.bits[w]
          if (word & mask) == 0 {
              page.bits[w] = word | mask
              uniqueCount &+= 1
              return true
          } else {
              duplicateCount &+= 1
              return false
          }
      }
  ```

- [ ] **Step 6 — Fix the stale sparse-clear header comment.**
  In `resetForNewQuery`, replace the comment immediately inside the `.fixedBitset` branch:
  ```swift
          } else if mode == .fixedBitset {
              // Clear only touched words if sparse, else full clear is faster.
              let tc = touchedCount
  ```
  with:
  ```swift
          } else if mode == .fixedBitset {
              // FixedBitset: clear only the touched words when the touched set is sparse
              // relative to total word count (< 25%, i.e. tc < wordCount / 4 below), else a
              // full clear is cheaper. Also handles the A8 ring-saturation case (touchedOverflowed)
              // by forcing a full clear, since the sparse touched-word list is then incomplete.
              let tc = touchedCount
  ```
  (The old wording's "sparse" read as referring to `.sparsePaged` *mode* — the comment immediately above it, at the top of this same `if/else if`, is about that mode — when it actually means "the touched-word set is sparse relative to `wordCount`," a `.fixedBitset`-only concept. The reworded comment also names the `touchedOverflowed`/A8 branch it sits above, which was added after the original comment was written.)

- [ ] **Step 7 — Build and re-run the SparsePaged test (must stay green — proves parity) plus the existing FixedBitset regression test.**
  ```bash
  swift build 2>&1 | tail -30
  swift test --filter 'CandidateDedupSparsePagedTests|RegressionA8_DedupOverflowTests' 2>&1 | tail -40
  ```
  Expected: `testSparsePagedCrossPageAllocationDuplicateAndReset` and `testFixedBitsetResetClearsPostSaturationBits` both `passed`.

- [ ] **Step 8 — Replace `MinHeap`/`MaxHeap` with one comparator-parameterized `BoundedTopKHeap`.**
  In `Sources/VectorIndex/Kernels/IVFSelect.swift`, replace the entire block from the `MinHeap` doc comment through the end of `MaxHeap` (current lines 774-954 — i.e. everything between the `TopKHeap` protocol above and the `// MARK: - Memory Pool for Score Buffers` section below; leave the `TopKHeap` protocol itself, lines 768-772, untouched):
  ```swift
  /// Min-heap for L2 distance (keep best = smallest).
  ///
  /// Maintains top-k smallest scores with O(log k) insertion.
  /// Tie-breaking: prefer smaller ID for deterministic results.
  private final class MinHeap: TopKHeap {
      ... (through the end of MaxHeap's extractSorted(), current line 954)
  }
  ```
  with:
  ```swift
  /// Bounded top-k heap parameterized by an ordering comparator, replacing the previously
  /// duplicated MinHeap (L2)/MaxHeap (IP/Cosine) — structurally identical apart from the
  /// flipped comparator direction.
  ///
  /// Maintains an internal heap of size ≤ capacity whose root always holds the WORST of the
  /// currently-kept top-k (by `isBetter`), so a new candidate can be rejected in O(1) against
  /// the root, or accepted and the heap restored in O(log k).
  private final class BoundedTopKHeap: TopKHeap {
      private var storage: [(id: Int32, score: Float)] = []
      private let capacity: Int
      /// True if `a` should be kept over `b` under the target metric (with deterministic
      /// tie-break by smaller ID) — used both to decide whether a new candidate displaces
      /// the current worst-of-kept, and to sort the final result best-first.
      private let isBetter: (_ scoreA: Float, _ idA: Int32, _ scoreB: Float, _ idB: Int32) -> Bool
      /// True if `a` should sit closer to the heap root than `b` — i.e. `a` is "worse" under
      /// `isBetter`, since the root always holds the worst of the currently-kept top-k.
      private let heapCompare: (_ a: (id: Int32, score: Float), _ b: (id: Int32, score: Float)) -> Bool

      init(
        capacity: Int,
        isBetter: @escaping (Float, Int32, Float, Int32) -> Bool,
        heapCompare: @escaping ((id: Int32, score: Float), (id: Int32, score: Float)) -> Bool
      ) {
          self.capacity = capacity
          self.isBetter = isBetter
          self.heapCompare = heapCompare
          storage.reserveCapacity(capacity)
      }

      var count: Int { storage.count }

      func insert(id: Int32, score: Float) {
          if storage.count < capacity {
              storage.append((id, score))
              bubbleUp(storage.count - 1)
          } else if let top = storage.first, isBetter(score, id, top.score, top.id) {
              storage[0] = (id, score)
              bubbleDown(0)
          }
      }

      private func bubbleUp(_ index: Int) {
          var idx = index
          while idx > 0 {
              let parent = (idx - 1) / 2
              if heapCompare(storage[idx], storage[parent]) {
                  storage.swapAt(idx, parent)
                  idx = parent
              } else {
                  break
              }
          }
      }

      private func bubbleDown(_ index: Int) {
          var idx = index
          while true {
              let left = 2 * idx + 1
              let right = 2 * idx + 2
              var extreme = idx
              if left < storage.count && heapCompare(storage[left], storage[extreme]) { extreme = left }
              if right < storage.count && heapCompare(storage[right], storage[extreme]) { extreme = right }
              if extreme != idx {
                  storage.swapAt(idx, extreme)
                  idx = extreme
              } else {
                  break
              }
          }
      }

      func extractSorted() -> [(id: Int32, score: Float)] {
          let result = storage.sorted { isBetter($0.score, $0.id, $1.score, $1.id) }
          storage.removeAll(keepingCapacity: true)
          return result
      }

      /// L2 (minimize): smaller score wins; tie-break smaller ID. Internally a max-heap
      /// (root = largest/worst of the kept top-k), matching the old MinHeap exactly.
      static func forL2(capacity: Int) -> BoundedTopKHeap {
          BoundedTopKHeap(
              capacity: capacity,
              isBetter: { scoreA, idA, scoreB, idB in
                  if scoreA < scoreB { return true }
                  if scoreA > scoreB { return false }
                  return idA < idB
              },
              heapCompare: { a, b in
                  if a.score > b.score { return true }
                  if a.score < b.score { return false }
                  return a.id > b.id
              }
          )
      }

      /// IP/Cosine (maximize): larger score wins; tie-break smaller ID. Internally a
      /// min-heap (root = smallest/worst of the kept top-k), matching the old MaxHeap exactly.
      static func forMaxMetric(capacity: Int) -> BoundedTopKHeap {
          BoundedTopKHeap(
              capacity: capacity,
              isBetter: { scoreA, idA, scoreB, idB in
                  if scoreA > scoreB { return true }
                  if scoreA < scoreB { return false }
                  return idA < idB
              },
              heapCompare: { a, b in
                  if a.score < b.score { return true }
                  if a.score > b.score { return false }
                  return a.id < b.id
              }
          )
      }
  }
  ```

- [ ] **Step 9 — Update all 5 call sites.**
  Replace:
  ```swift
      let heap: any TopKHeap
      switch metric {
      case .l2:
          heap = MinHeap(capacity: actualK)
      case .ip, .cosine:
          heap = MaxHeap(capacity: actualK)
      }
  ```
  with:
  ```swift
      let heap: any TopKHeap
      switch metric {
      case .l2:
          heap = BoundedTopKHeap.forL2(capacity: actualK)
      case .ip, .cosine:
          heap = BoundedTopKHeap.forMaxMetric(capacity: actualK)
      }
  ```
  Replace:
  ```swift
      let initialHeap: any TopKHeap = (metric == .l2) ? MinHeap(capacity: beamWidth) : MaxHeap(capacity: beamWidth)
  ```
  with:
  ```swift
      let initialHeap: any TopKHeap = (metric == .l2) ? BoundedTopKHeap.forL2(capacity: beamWidth) : BoundedTopKHeap.forMaxMetric(capacity: beamWidth)
  ```
  Replace:
  ```swift
      let resultHeap: any TopKHeap = (metric == .l2) ? MinHeap(capacity: max(beamWidth, nprobe)) : MaxHeap(capacity: max(beamWidth, nprobe))
  ```
  with:
  ```swift
      let resultHeap: any TopKHeap = (metric == .l2) ? BoundedTopKHeap.forL2(capacity: max(beamWidth, nprobe)) : BoundedTopKHeap.forMaxMetric(capacity: max(beamWidth, nprobe))
  ```
  Replace:
  ```swift
                  let heap: any TopKHeap = (metric == .l2) ? MinHeap(capacity: nprobe) : MaxHeap(capacity: nprobe)
  ```
  (inside `partitionAndSelectParallel`) with:
  ```swift
                  let heap: any TopKHeap = (metric == .l2) ? BoundedTopKHeap.forL2(capacity: nprobe) : BoundedTopKHeap.forMaxMetric(capacity: nprobe)
  ```
  Replace:
  ```swift
      let heap: any TopKHeap = (metric == .l2) ? MinHeap(capacity: nprobe) : MaxHeap(capacity: nprobe)
  ```
  (inside `mergePartitions`) with:
  ```swift
      let heap: any TopKHeap = (metric == .l2) ? BoundedTopKHeap.forL2(capacity: nprobe) : BoundedTopKHeap.forMaxMetric(capacity: nprobe)
  ```
  Do **not** touch `Sources/VectorIndex/Kernels/HNSWTraversal.swift:73`'s `private struct MinHeap` — that is an unrelated, separately-scoped candidate-frontier min-heap (different file, different protocol, a `struct` not a `class`) that happens to share a name; it is out of scope for B16.

- [ ] **Step 10 — Build.**
  ```bash
  swift build 2>&1 | tail -40
  ```
  Expected: `Build complete!`, with no remaining references to `MinHeap`/`MaxHeap` in `IVFSelect.swift` (`grep -n "MinHeap\|MaxHeap" Sources/VectorIndex/Kernels/IVFSelect.swift` should return nothing).

- [ ] **Step 11 — Run the named filters for this task.**
  ```bash
  swift test --filter 'CandidateDedupSparsePagedTests|RegressionA8_DedupOverflowTests|IVFSelectTests' 2>&1 | tail -120
  ```
  Expected: all green, including `IVFSelectTests.testTieBreakingDeterminism` and `testBatchVsSingleParity` (these two specifically exercise comparator/tie-break behavior through the public `ivf_select_*` entry points and are the primary regression guard for the `MinHeap`/`MaxHeap` → `BoundedTopKHeap` merge, since nothing constructs the private heap classes directly).

- [ ] **Step 12 — Commit.**
  ```bash
  git add Tests/VectorIndexTests/CandidateDedupSparsePagedTests.swift \
          Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift \
          Sources/VectorIndex/Kernels/IVFSelect.swift
  git commit -m "$(cat <<'EOF'
  refactor(dedup,ivfselect): Page value->reference type, unify MinHeap/MaxHeap, fix stale
  comment (B16)

  - CandidateDedup's SparsePaged Page was a value struct stored in a [Int64: Page]
    dictionary, forcing an explicit copy-out/mutate/copy-back dance on every epoch-stamp
    update; converting it to a class removes all of that write-back code. .sparsePaged had
    zero existing test coverage, so this is preceded by a characterization test pinning
    cross-page allocation, duplicate detection, page reuse across queries, and
    touched-page clearing on reset.
  - IVFSelect's MinHeap (L2) and MaxHeap (IP/Cosine) were structurally identical apart from
    a flipped comparator direction (and MinHeap carried several stream-of-consciousness
    "wait, let me fix this" comments left in from development). Unified into one
    comparator-parameterized BoundedTopKHeap; all 5 call sites updated.
  - Reworded CandidateDedup's stale "Clear only touched words if sparse" comment, which
    read as describing .sparsePaged mode when it actually means "the touched-word set is
    sparse relative to wordCount" (a .fixedBitset-only concept), and didn't mention the A8
    touchedOverflowed full-clear branch added after the comment was written.

  Co-Authored-By: use the executing assistant's standard Co-Authored-By trailer
  EOF
  )"
  ```

---

### Task 15: B17 (pointer-laundering, dead builder, HNSW distance caching) + returnSorted

**Files:**
- Modify: `Sources/VectorIndex/Operations/Rerank/ExactRerank.swift` (lines 21–64 `RerankOpts`; 300–357 `scoreBlock` parallel branch; 536–586 `topKIVF` array-based overload; 588–634 `scoresIVF`) — current-tree at `ee67895`; Task 3 touches this file earlier in the plan sequence, re-grep anchors below if drifted.
- Modify: `Sources/VectorIndex/Operations/Quantization/IVFPostADC.swift` (lines 20–37, `rerankTopKFlat`)
- Modify: `Sources/VectorIndex/IVFIndex.swift` (lines 952–965, the one `RerankOpts(...)` construction site — do not confuse with Task 16's `optimize()`/`optimizeKMeans` in the same file)
- Modify: `Sources/VectorIndex/HNSWIndex.swift` (lines 618–648 `internalInsertAtLevel`'s `selected.min(by:)`; 1115–1141 `compact()`'s re-prune) — Tasks 3/4/7/8/17 also touch this file; re-grep anchors if drifted, and land this task in the same session as (or after) Task 17 to minimize merge friction within this file.
- Create: `Tests/VectorIndexTests/RegressionB17a_ParallelRerankTests.swift`

**Interfaces:**
Consumes: `UnsafeSendablePtr<T>` / `UnsafeSendableMutPtr<T>` (`Sources/VectorIndex/Operations/Quantization/ADCScan.swift:162-168` — file-top-level, default/internal access, already module-visible; the only one of the three existing definitions *not* local to a function, so it's the one to reuse directly without redeclaring); `IndexOps.Rerank.rerank_exact_topk(q:d:metric:candIDs:C:K:reader:opts:topScores:topIDs:)` (existing, used by the new test).
Produces: `IndexOps.Rerank.RerankOpts.returnSorted: Bool` becomes `@available(*, deprecated, ...)` (same external `{ get set }` shape, now computed over a private backing field — no call-site changes needed); `IndexOps.Rerank.topKIVF(q:d:metric:candInternalIDs:id2List:id2Offset:lists:K:opts:) -> (scores:[Float], ids:[Int64])` (array-based overload) becomes `@available(*, deprecated, ...)` (signature unchanged); `IndexOps.Rerank.scoresIVF(q:d:metric:candInternalIDs:id2List:id2Offset:lists:opts:) -> [Float]` becomes `@available(*, deprecated, ...)` (signature unchanged); `IVFPostADC.rerankTopKFlat(q:d:metric:candInternalIDs:id2List:id2Offset:lists:K:opts:) -> (scores:[Float], ids:[Int64])` becomes `@available(*, deprecated, ...)` (signature unchanged).

---

#### (a) Pointer-laundering fix, tests-first

- [ ] **Step 1: Add the parallel-branch characterization test (against the current, unmodified `scoreBlock`)**

  Create `Tests/VectorIndexTests/RegressionB17a_ParallelRerankTests.swift`:
  ```swift
  import XCTest
  @testable import VectorIndex

  /// Regression guard for B17a (`ExactRerank.swift` `scoreBlock`'s parallel branch).
  /// Per the Phase-2 research brief: no existing test drives `C` above
  /// `opts.parallelThreshold` (default 8192, with default `gatherTile` 128 so
  /// `tile*2 == 256`), so the `DispatchQueue.concurrentPerform` branch -- including
  /// the pointer-laundering code being replaced in this task -- was never exercised
  /// by any test. Written *before* the fix to characterize current (correct)
  /// behavior; must stay green after the `UnsafeSendablePtr`/`UnsafeSendableMutPtr`
  /// rewrite.
  ///
  /// Drives C = 8300 (> default parallelThreshold 8192) through `rerank_exact_topk`
  /// twice -- once with `enableParallel: true` (parallel path) and once with
  /// `enableParallel: false` (sequential path, already covered elsewhere) -- and
  /// asserts identical top-K output, proving the parallel branch's scatter-back
  /// writes to the same slots as the trusted sequential path.
  final class RegressionB17a_ParallelRerankTests: XCTestCase {
      func testParallelPathAboveThresholdMatchesSequential() {
          let d = 4
          let C = 8300
          let K = 16

          // Candidate id i maps to vector [Float(i), 0, 0, 0]; query is the origin,
          // so euclidean distance increases monotonically with id -- smallest ids win.
          let reader = IndexOps.Rerank.CallbackReader(dim: d) { ids, count, dst, present in
              for i in 0..<count {
                  let v = Float(ids[i])
                  dst[i*d + 0] = v; dst[i*d + 1] = 0; dst[i*d + 2] = 0; dst[i*d + 3] = 0
                  present[i] = 1
              }
              return count
          }

          let q: [Float] = [0, 0, 0, 0]
          let candIDs: [Int64] = (0..<C).map { Int64($0) }

          func run(enableParallel: Bool) -> (scores: [Float], ids: [Int64]) {
              var scores = [Float](repeating: 0, count: K)
              var outIDs = [Int64](repeating: -1, count: K)
              q.withUnsafeBufferPointer { qb in
                  candIDs.withUnsafeBufferPointer { cb in
                      scores.withUnsafeMutableBufferPointer { sb in
                          outIDs.withUnsafeMutableBufferPointer { ib in
                              let opts = IndexOps.Rerank.RerankOpts(
                                  backend: .callback,
                                  enableParallel: enableParallel,
                                  parallelThreshold: 8192
                              )
                              IndexOps.Rerank.rerank_exact_topk(
                                  q: qb.baseAddress!, d: d, metric: .euclidean,
                                  candIDs: cb.baseAddress!, C: C, K: K,
                                  reader: reader, opts: opts,
                                  topScores: sb.baseAddress!, topIDs: ib.baseAddress!)
                          }
                      }
                  }
              }
              return (scores, outIDs)
          }

          let parallel = run(enableParallel: true)
          let sequential = run(enableParallel: false)

          XCTAssertEqual(parallel.ids, sequential.ids,
                          "parallel scoreBlock branch must scatter results to the same slots as the sequential branch")
          XCTAssertEqual(parallel.scores, sequential.scores)
          // Smallest-id candidates (closest to the origin) must win.
          XCTAssertEqual(parallel.ids, Array((0..<Int64(K))))
      }
  }
  ```

- [ ] **Step 2: Confirm the characterization test passes against unmodified code**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'RegressionB17a_ParallelRerankTests' 2>&1 | tail -20
  ```
  Expected: 1 test, 0 failures. This establishes the safety net before touching `scoreBlock`.

- [ ] **Step 3: Replace the `UInt(bitPattern:)` box construction with `UnsafeSendablePtr`/`UnsafeSendableMutPtr`**

  In `ExactRerank.swift`, inside `scoreBlock`'s parallel branch, find:
  ```swift
          // Wrap non-Sendable references safely for capture
          struct SendableBox<T>: @unchecked Sendable { let value: T }
          let readerBox = SendableBox(value: reader)
          let qAddr = UInt(bitPattern: q)
          let outAddr = UInt(bitPattern: scoresOut)
          let maskAddr = presentMaskOut.map { UInt(bitPattern: $0) } ?? 0
  ```
  Replace with:
  ```swift
          // Wrap non-Sendable references safely for capture. UnsafeSendablePtr/
          // UnsafeSendableMutPtr already exist module-wide (ADCScan.swift:162-168);
          // reused here instead of the previous UInt(bitPattern:) round-trip +
          // force-unwrap.
          struct SendableBox<T>: @unchecked Sendable { let value: T }
          let readerBox = SendableBox(value: reader)
          let qBox = UnsafeSendablePtr(ptr: q)
          let outBox = UnsafeSendableMutPtr(ptr: scoresOut)
          let maskBox: UnsafeSendableMutPtr<UInt8>? = presentMaskOut.map { UnsafeSendableMutPtr(ptr: $0) }
  ```

- [ ] **Step 4: Replace the `bitPattern:` reconstruction inside the closure**

  A few lines further down (still inside the `DispatchQueue.concurrentPerform` closure), find:
  ```swift
                  let qLocal = UnsafePointer<Float>(bitPattern: qAddr)!
                  scoreTile(q: qLocal, d: d, metric: metric, n: chunk, qNorm: qNorm, xb: scratch, out: tileScores, reader: readerLocal, opts: opts)

                  // Scatter back to global outputs (disjoint indices across tiles)
                  let scoresOutLocal = UnsafeMutablePointer<Float>(bitPattern: outAddr)!
                  let presentMaskLocal: UnsafeMutablePointer<UInt8>? = (maskAddr != 0) ? UnsafeMutablePointer<UInt8>(bitPattern: maskAddr)! : nil
  ```
  Replace with:
  ```swift
                  let qLocal = qBox.ptr
                  scoreTile(q: qLocal, d: d, metric: metric, n: chunk, qNorm: qNorm, xb: scratch, out: tileScores, reader: readerLocal, opts: opts)

                  // Scatter back to global outputs (disjoint indices across tiles)
                  let scoresOutLocal = outBox.ptr
                  let presentMaskLocal: UnsafeMutablePointer<UInt8>? = maskBox?.ptr
  ```

- [ ] **Step 5: Re-confirm the parallel-path test after the rewrite**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'RegressionB17a_ParallelRerankTests' 2>&1 | tail -20
  ```
  Expected: still 1 test, 0 failures — proves the box replacement is behavior-preserving.

#### (b) Deprecate the dead recursive-builder wrappers (public → deprecate, not delete)

- [ ] **Step 6: Deprecate `topKIVF`'s array-based overload (contains the recursive `buildAndRun`)**

  In `ExactRerank.swift`, find:
  ```swift
      // Array-based lists version (convenience)
      static func topKIVF(
  ```
  Replace with:
  ```swift
      // Array-based lists version (convenience).
      //
      // Phase-2 (B17b) finding: dead code -- zero callers besides
      // IVFPostADC.rerankTopKFlat (Sources/VectorIndex/Operations/Quantization/
      // IVFPostADC.swift:20), which itself has zero callers and zero test coverage
      // anywhere in Sources/ or Tests/ (grep-verified). Its body's recursive
      // `buildAndRun` closure (a latent stack-depth risk for large `nlist`) is left
      // as-is rather than flattened, since the whole function is scheduled for
      // deletion, not improvement. Kept (not deleted) because it is public API
      // (implicit via `public extension IndexOps.Rerank`) and Phase 2 is
      // non-breaking; see the PHASE4-ROUTING appendix.
      @available(*, deprecated, message: "Dead code: no callers besides the equally-dead IVFPostADC.rerankTopKFlat. Scheduled for removal in 0.2.0's breaking phase.")
      static func topKIVF(
  ```

- [ ] **Step 7: Deprecate `scoresIVF` (same recursive-builder shape, zero callers at all)**

  Find:
  ```swift
      static func scoresIVF(
  ```
  Replace with:
  ```swift
      // Phase-2 (B17b) finding: dead code -- zero callers anywhere in Sources/ or
      // Tests/ (grep-verified). Kept (not deleted) because it is public API
      // (implicit via `public extension IndexOps.Rerank`); see PHASE4-ROUTING.
      @available(*, deprecated, message: "Dead code: zero callers anywhere in the package. Scheduled for removal in 0.2.0's breaking phase.")
      static func scoresIVF(
  ```

- [ ] **Step 8: Deprecate `IVFPostADC.rerankTopKFlat`**

  In `Sources/VectorIndex/Operations/Quantization/IVFPostADC.swift`, find:
  ```swift
      public static func rerankTopKFlat(
  ```
  Replace with:
  ```swift
      // Phase-2 (B17b) finding: zero callers anywhere in Sources/ or Tests/
      // (grep-verified). Kept (not deleted) -- public API, non-breaking
      // constraint; see PHASE4-ROUTING.
      @available(*, deprecated, message: "Dead code: no callers found anywhere in the package. Scheduled for removal in 0.2.0's breaking phase.")
      public static func rerankTopKFlat(
  ```

- [ ] **Step 9: Confirm zero new warnings from the deprecations**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | grep -i "deprecat" ; echo "exit: $?"
  ```
  Expected: no output before `exit: 1` (grep finds nothing) — `rerankTopKFlat`'s body calls the now-equally-deprecated `topKIVF`, and Swift does not warn when a deprecated declaration calls another deprecated declaration; since both have zero other callers, no warnings fire anywhere else either.

#### (c) HNSW distance caching (`selected.min(by:)` and `compact()`'s re-prune sort)

- [ ] **Step 10: Precompute distances once in `internalInsertAtLevel`'s neighbor-selection**

  In `HNSWIndex.swift`, inside `internalInsertAtLevel`, find:
  ```swift
                  connect(newIndex, with: selected, level: l)
                  // Update cur to closest among selected for next lower layer
                  if let best = selected.min(by: { distance(vector, vectorArray(at: $0), metric: metric) < distance(vector, vectorArray(at: $1), metric: metric) }) {
                      cur = best
                  }
              }
  ```
  Replace with:
  ```swift
                  connect(newIndex, with: selected, level: l)
                  // Update cur to closest among selected for next lower layer.
                  // Precompute each candidate's vector + distance once (B17c): the
                  // previous min(by:) comparator called vectorArray(at:)/distance(...)
                  // twice per comparison for both operands, with zero caching across
                  // comparisons -- up to 2(M-1) redundant allocations and full-d
                  // distance recomputations for an M-candidate selection.
                  if !selected.isEmpty {
                      let dists = selected.map { distance(vector, vectorArray(at: $0), metric: metric) }
                      var bestIdx = 0
                      for i in 1..<dists.count where dists[i] < dists[bestIdx] { bestIdx = i }
                      cur = selected[bestIdx]
                  }
              }
  ```
  (A9's `pruneNeighbors` rewrite does not touch this site -- it runs *before* `connect()`, picking `cur` for the next-lower layer, a different call site than `pruneNeighbors`'s reverse-edge shrink.)

- [ ] **Step 11: Precompute distances once in `compact()`'s re-prune sort**

  In `HNSWIndex.swift`'s `compact()`, find:
  ```swift
                  // prune to M
                  if mapped.count > config.m {
                      let nodeOffset = newNodes[i].vectorOffset
                      let nodeVec = Array(newVectorStorage[nodeOffset..<(nodeOffset + dim)])
                      mapped.sort {
                          let off0 = newNodes[$0].vectorOffset
                          let off1 = newNodes[$1].vectorOffset
                          let vec0 = Array(newVectorStorage[off0..<(off0 + dim)])
                          let vec1 = Array(newVectorStorage[off1..<(off1 + dim)])
                          return distance(nodeVec, vec0, metric: metric) < distance(nodeVec, vec1, metric: metric)
                      }
                      mapped = Array(mapped.prefix(config.m))
                  }
  ```
  Replace with:
  ```swift
                  // prune to M
                  if mapped.count > config.m {
                      let nodeOffset = newNodes[i].vectorOffset
                      let nodeVec = Array(newVectorStorage[nodeOffset..<(nodeOffset + dim)])
                      // Precompute each candidate's distance once (B17c): the previous
                      // sort comparator recomputed vec0/vec1 (fresh Array allocations)
                      // on every pairwise comparison, O(mapped.count log mapped.count)
                      // times, re-copying the same handful of candidates repeatedly.
                      let withDist: [(id: Int, dist: Float)] = mapped.map { cand in
                          let off = newNodes[cand].vectorOffset
                          let vec = Array(newVectorStorage[off..<(off + dim)])
                          return (cand, distance(nodeVec, vec, metric: metric))
                      }
                      mapped = withDist.sorted { $0.dist < $1.dist }.prefix(config.m).map { $0.id }
                  }
  ```

- [ ] **Step 12: Determinism check for (c)**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'HNSWTests|HNSWRecallTests|HNSWParamSweepTests|HNSWKNNGraphTests|RegressionA1_TraversalLifetimeTests' 2>&1 | tail -40
  ```
  Expected: 0 failures. `RegressionA1_TraversalLifetimeTests` in particular must show byte-identical results to before the change (Swift's `sorted()`/`sort()` are stable, matching the prior comparator's tie behavior, so this is a correctness-preserving rewrite, not a behavior change).

#### (d) `RerankOpts.returnSorted` — dead field, deprecate (public, non-breaking)

- [ ] **Step 13: Convert `returnSorted` to a deprecated computed property over a private backing field**

  In `ExactRerank.swift`'s `RerankOpts`, find:
  ```swift
          public var returnSorted: Bool              // outputs sorted best-first
  ```
  Replace with:
  ```swift
          // B17/returnSorted finding: never read by scoreBlock/rerank_exact_topk/
          // rerank_exact_scores -- rerank_exact_topk unconditionally calls
          // selHeap.extractSorted(), so output is always best-first regardless of
          // this flag. Kept (not deleted) as public API; see PHASE4-ROUTING.
          // Backed by a private stored field so this type's own init can still
          // assign it without tripping its own deprecation warning (Swift warns on
          // deprecated-member use even from within the declaring type's own init).
          @available(*, deprecated, message: "Never honored; results are always emitted best-first. Removal scheduled for 0.2.0's breaking phase.")
          public var returnSorted: Bool {
              get { _returnSorted }
              set { _returnSorted = newValue }
          }
          private var _returnSorted: Bool
  ```
  Then, inside `RerankOpts.init`, find:
  ```swift
              self.returnSorted = returnSorted
  ```
  Replace with:
  ```swift
              self._returnSorted = returnSorted
  ```

- [ ] **Step 14: Remove the one internal caller's redundant explicit `returnSorted: true`**

  In `Sources/VectorIndex/IVFIndex.swift`, find (inside `searchKernel30Flat`'s `RerankOpts(...)` construction):
  ```swift
                              let opts = IndexOps.Rerank.RerankOpts(
                                  backend: .ivfListVecs,
                                  gatherTile: 128,
                                  reorderBySegment: true,
                                  haveInvNorms: false,
                                  haveSqNorms: false,
                                  returnSorted: true,
                                  skipMissing: true,
                                  prefetchDistance: 8,
                                  strictFP: false,
                                  enableParallel: true,
                                  parallelThreshold: 2048,
                                  maxConcurrency: 0
                              )
  ```
  Replace with:
  ```swift
                              let opts = IndexOps.Rerank.RerankOpts(
                                  backend: .ivfListVecs,
                                  gatherTile: 128,
                                  reorderBySegment: true,
                                  haveInvNorms: false,
                                  haveSqNorms: false,
                                  skipMissing: true,
                                  prefetchDistance: 8,
                                  strictFP: false,
                                  enableParallel: true,
                                  parallelThreshold: 2048,
                                  maxConcurrency: 0
                              )
  ```
  (`returnSorted` defaults to `true` in the initializer, so this is a behavior no-op; it also avoids IVFIndex.swift picking up a needless deprecation warning at its own call site.)

- [ ] **Step 15: Confirm build is clean of new warnings from (d)**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | grep -i "returnSorted" ; echo "exit: $?"
  ```
  Expected: no output before `exit: 1` — the only remaining internal use of `returnSorted` is the private-backed `init`, which no longer touches the deprecated property directly.

- [ ] **Step 16: Full task verification**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | tail -20
  ```
  Expected: `Build complete!`, exit 0.
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'RegressionB17a_ParallelRerankTests|IVFFlatRerankTests|IVFListVecsReaderRerankTests|RegressionA4_RerankIDWidthTests|HNSWTests|HNSWRecallTests|HNSWParamSweepTests|HNSWKNNGraphTests|RegressionA1_TraversalLifetimeTests' 2>&1 | tail -60
  ```
  Expected: 0 failures across all listed suites. (Split into two `--filter` invocations if this exceeds the 600000 ms command timeout in your environment.)

- [ ] **Commit**

  ```bash
  git add Sources/VectorIndex/Operations/Rerank/ExactRerank.swift Sources/VectorIndex/Operations/Quantization/IVFPostADC.swift Sources/VectorIndex/IVFIndex.swift Sources/VectorIndex/HNSWIndex.swift Tests/VectorIndexTests/RegressionB17a_ParallelRerankTests.swift
  git commit -m "$(cat <<'EOF'
  refactor(rerank,hnsw): shared Sendable pointer box, deprecate dead IVF wrappers, cache HNSW distances

  ExactRerank.scoreBlock's parallel branch now wraps q/scoresOut/presentMaskOut with
  the existing UnsafeSendablePtr/UnsafeSendableMutPtr types (already used in
  ADCScan.swift/LayoutTransforms.swift/L2SqrKernel.swift) instead of the
  UInt(bitPattern:) round-trip + force-unwrap dance; a new large-candidate-count test
  (RegressionB17a_ParallelRerankTests) closes the gap where this branch had zero
  coverage of any kind.

  topKIVF's array-based overload, scoresIVF, and IVFPostADC.rerankTopKFlat are all
  confirmed dead (the recursive buildAndRun closure they share has zero callers
  besides each other and zero test coverage) -- deprecated in place rather than
  deleted since all three are public API; scheduled for removal in the 0.2.0
  breaking phase.

  HNSWIndex stops reallocating candidate vectors/distances per pairwise comparison
  in internalInsertAtLevel's neighbor-selection and compact()'s re-prune sort --
  both now precompute each candidate's distance exactly once.

  RerankOpts.returnSorted (never read by any rerank code path -- output is always
  best-first) is deprecated rather than deleted, backed by a private stored field so
  its own initializer doesn't trip the deprecation warning on itself.
  EOF
  )"
  ```
  (with the executing assistant's standard Co-Authored-By trailer)

---

### Task 16: B18 — `IVFIndex.optimize()`/`optimizeKMeans(maxIterations:)` consolidation

**Files:**
- Modify: `Sources/VectorIndex/IVFIndex.swift` (lines 265–296, current-tree at `ee67895`)
- Modify: `Tests/VectorIndexTests/IVFKMeansPlusPlusTests.swift` (append after `testOptimizeAssignsAll`, line 19)

**Interfaces:**
Consumes: none new (uses existing private `kmeans(centroids:maxIterations:)`, `kmeansPlusPlusInitRandom(k:seed:)`, `nearestCentroidIndex(for:)`).
Produces: `IVFIndex.optimize() async throws` becomes `IVFIndex.optimize(maxIterations: Int = 20) async throws` — additive defaulted parameter; every existing zero-argument call site (`try await ivf.optimize()`, ~16 test call sites plus `IVFIndex.swift:677`'s `load()`) continues to compile and behave identically, since the default (20) matches the value it replaces. `IVFIndex.optimizeKMeans(maxIterations: Int = 15) async throws` — signature unchanged, body now delegates.

---

- [ ] **Step 1: Give `optimize()` a `maxIterations` parameter and make it the single real implementation**

  In `IVFIndex.swift`, find:
  ```swift
      public func optimize() async throws {
          // Build centroids with CPU Lloyd's KMeans and assign points to lists
          // Use k = min(nlist, store.count)
          guard !store.isEmpty else {
              centroids.removeAll(); lists.removeAll(); return
          }
          let k = max(1, min(config.nlist, store.count))
          // Initialize centroids using deterministic k‑means++ (farthest‑point) seeding
          let initialCentroids = try kmeansPlusPlusInitRandom(k: k, seed: 42)
          centroids = try await kmeans(centroids: initialCentroids, maxIterations: 20)
          // Build inverted lists
          lists = Array(repeating: [], count: centroids.count)
          idToListIndex.removeAll(keepingCapacity: false)
          for (id, (vec, _)) in store {
              if let ci = nearestCentroidIndex(for: vec), lists.indices.contains(ci) {
                  lists[ci].append(id)
                  idToListIndex[id] = ci
              }
          }
      }

      // MARK: - KMeans scaffolding (to be implemented)
      public func optimizeKMeans(maxIterations: Int = 15) async throws {
          guard !store.isEmpty else { centroids.removeAll(); lists.removeAll(); return }
          let k = max(1, min(config.nlist, store.count))
          let initC = try kmeansPlusPlusInitRandom(k: k, seed: 42)
          centroids = try await kmeans(centroids: initC, maxIterations: maxIterations)
          lists = Array(repeating: [], count: centroids.count)
          for (id, (vec, _)) in store {
              if let ci = nearestCentroidIndex(for: vec) { lists[ci].append(id) }
          }
      }
  ```
  Replace with:
  ```swift
      // B18: maxIterations defaults to 20 (this method's previous hardcoded value),
      // so every existing zero-argument call site (ivf.optimize()) is unaffected.
      // optimizeKMeans(maxIterations:) below now delegates here instead of running
      // its own divergent copy.
      public func optimize(maxIterations: Int = 20) async throws {
          // Build centroids with CPU Lloyd's KMeans and assign points to lists
          // Use k = min(nlist, store.count)
          guard !store.isEmpty else {
              centroids.removeAll(); lists.removeAll(); return
          }
          let k = max(1, min(config.nlist, store.count))
          // Initialize centroids using deterministic k‑means++ (farthest‑point) seeding
          //
          // Phase-3 (P3) overlap note, not fixed here: kmeansPlusPlusInitRandom and
          // kmeans each independently re-derive the same flat [Float] from store (two
          // O(N*d) materializations where one shared buffer would do), and
          // nearestCentroidIndex below re-scans every vector against every centroid
          // (O(N*k)) even though kmeans's underlying kmeans_minibatch_f32 call already
          // ran with computeAssignments: false -- flipping that to true and wiring
          // assignOut: would eliminate the rescan entirely. Both are real perf
          // opportunities the Phase-2 research brief flagged as overlapping Phase 3's
          // mandate (behavior-neutral cleanup is Phase 2's job, perf is Phase 3's);
          // left for Phase 3 to pick up under its benchmark gate.
          let initialCentroids = try kmeansPlusPlusInitRandom(k: k, seed: 42)
          centroids = try await kmeans(centroids: initialCentroids, maxIterations: maxIterations)
          // Build inverted lists
          lists = Array(repeating: [], count: centroids.count)
          idToListIndex.removeAll(keepingCapacity: false)
          for (id, (vec, _)) in store {
              if let ci = nearestCentroidIndex(for: vec), lists.indices.contains(ci) {
                  lists[ci].append(id)
                  idToListIndex[id] = ci
              }
          }
      }

      // MARK: - KMeans scaffolding
      // B18: was a standalone copy that diverged non-cosmetically from optimize() --
      // it never populated idToListIndex and never bounds-guarded the list-array
      // write (lists[ci].append with no lists.indices.contains(ci) guard). Zero
      // callers and zero test coverage existed for this method before this task (see
      // the Phase-2 research brief, B18). Now a thin delegating wrapper, which fixes
      // both bugs by construction since there's no separate body left to diverge in.
      public func optimizeKMeans(maxIterations: Int = 15) async throws {
          try await optimize(maxIterations: maxIterations)
      }
  ```

- [ ] **Step 2: Add the characterization test proving the delegation fix**

  Append to `Tests/VectorIndexTests/IVFKMeansPlusPlusTests.swift`, inside the class, after `testOptimizeAssignsAll`:
  ```swift
      /// Guards Task 16 (B18): before this task, optimizeKMeans(maxIterations:) ran
      /// its own standalone body that never populated idToListIndex. Same
      /// three-cluster fixture as testOptimizeAssignsAll, driven through
      /// optimizeKMeans() instead of optimize() -- "assigned" (backed by
      /// idToListIndex.count, see IVFIndex.statistics()) must now read 6, proving
      /// the delegation to optimize(maxIterations:) actually populates the map.
      func testOptimizeKMeansPopulatesIdToListIndex() async throws {
          let ivf = IVFIndex(dimension: 3, metric: .euclidean, config: .init(nlist: 3, nprobe: 1))
          try await ivf.batchInsert([
              ("a1", [1, 0, 0], nil), ("a2", [0.9, 0, 0], nil),
              ("b1", [0, 1, 0], nil), ("b2", [0, 0.95, 0], nil),
              ("c1", [0, 0, 1], nil), ("c2", [0, 0, 0.9], nil)
          ])
          try await ivf.optimizeKMeans()
          let stats = await ivf.statistics()
          XCTAssertEqual(stats.vectorCount, 6)
          XCTAssertEqual(Int(stats.details["assigned"] ?? "0"), 6,
                         "optimizeKMeans must populate idToListIndex for every assigned vector")
      }
  ```

- [ ] **Step 3: Full task verification**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | tail -20
  ```
  Expected: `Build complete!`, exit 0.
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'IVFKMeansPlusPlusTests|APIPolishTests|StatsTests|IVFMoreTests|TypedOverloadsTests|PersistenceTests|IVFRecallTests|IVFListMaintenanceTests|IVFTests|IVFProbeMonotonicTests|AccelerableIndexTests' 2>&1 | tail -60
  ```
  Expected: 0 failures across all listed suites — this is `optimize()`'s full existing coverage (per the research brief's grep), all of which must stay green since `optimize()` is now the sole implementation `optimizeKMeans` delegates to. Split into two invocations if it exceeds the 600000 ms timeout.

- [ ] **Commit**

  ```bash
  git add Sources/VectorIndex/IVFIndex.swift Tests/VectorIndexTests/IVFKMeansPlusPlusTests.swift
  git commit -m "$(cat <<'EOF'
  refactor(ivf): unify optimize()/optimizeKMeans(), fix idToListIndex population bug

  optimizeKMeans(maxIterations:) ran its own standalone body that diverged
  non-cosmetically from optimize() -- it never populated idToListIndex and never
  bounds-guarded the list-array write, and had zero callers or test coverage
  anywhere in the package. optimize() gains a maxIterations parameter (default 20,
  matching its previous hardcoded value, so every existing zero-argument call site
  is unaffected) and becomes the single real implementation; optimizeKMeans is now a
  thin delegating wrapper, which fixes both bugs by construction. A new test proves
  optimizeKMeans now populates idToListIndex.

  The double [[Float]] materialization between seeding and training, and the
  post-kmeans O(N*k) nearest-centroid rescan (kmeans_minibatch_f32 already runs with
  computeAssignments: false), are left as a documented Phase-3 (P3) overlap --
  real perf opportunities out of scope for this behavior-neutral consolidation.
  EOF
  )"
  ```
  (with the executing assistant's standard Co-Authored-By trailer)

---

### Task 17: B19 — HNSW init delegation, cached metric, COW batch context, WAL `.update`/`.clear` coverage

**Files:**
- Modify: `Sources/VectorIndex/HNSWIndex.swift` (lines 29–70 stored properties + both inits; call sites at 178, 266, 447, 623, 700; lines 231–233 & 268–270 `BatchSearchContext`) — current-tree at `ee67895`; Tasks 3/4/7/8/15 also touch this file, re-grep anchors if drifted.
- Modify: `Sources/VectorIndex/HNSWWAL.swift` (lines 44–53, add a comment; no logic change)
- Modify: `Tests/VectorIndexTests/HNSWWALTests.swift` (lines 54–103, `testFrameRoundTripAllRecordTypes`)

**Interfaces:**
Consumes: none new (uses existing `Self.toHNSWMetric(_:)`, `HNSWXoroRNGState.from(seed:stream:)`).
Produces: none public. `HNSWIndex.init(dimension:metric:)`'s external signature is unchanged (still resolves for the exact two-labeled-argument call shape); its body now genuinely delegates. The new `hnswMetric` stored property is `private`.

---

- [ ] **Step 1: Add the cached `hnswMetric` property and compute it once in the designated init**

  In `HNSWIndex.swift`, find:
  ```swift
      // MARK: - Public API
      public let dimension: Int
      public let metric: SupportedDistanceMetric
      public let config: Configuration
      public var count: Int { activeCount }

      /// Supported metrics for HNSW traversal kernel
      private static let supportedMetrics: Set<SupportedDistanceMetric> = [.euclidean, .dotProduct, .cosine]

      /// Convert SupportedDistanceMetric to HNSWMetric (validated at init)
      @inline(__always)
      private static func toHNSWMetric(_ metric: SupportedDistanceMetric) -> HNSWMetric {
          switch metric {
          case .euclidean: return .L2
          case .dotProduct: return .IP
          case .cosine: return .COSINE
          case .manhattan, .chebyshev:
              // Should never reach here due to init validation
              preconditionFailure("Unsupported metric '\(metric)' in HNSWIndex")
          }
      }

      public init(dimension: Int, metric: SupportedDistanceMetric = .euclidean, config: Configuration = .init()) {
          guard Self.supportedMetrics.contains(metric) else {
              preconditionFailure("HNSWIndex does not support metric '\(metric)'. Supported metrics: euclidean, dotProduct, cosine. Use FlatIndex for manhattan/chebyshev.")
          }
          self.dimension = dimension
          self.metric = metric
          self.config = config
          self.rng35 = HNSWXoroRNGState.from(seed: config.rngSeed, stream: config.rngStream)
      }

      // Protocol-required initializer (delegates to designated one)
      public init(dimension: Int, metric: SupportedDistanceMetric) {
          guard Self.supportedMetrics.contains(metric) else {
              preconditionFailure("HNSWIndex does not support metric '\(metric)'. Supported metrics: euclidean, dotProduct, cosine. Use FlatIndex for manhattan/chebyshev.")
          }
          self.dimension = dimension
          self.metric = metric
          self.config = .init()
          self.rng35 = HNSWXoroRNGState.from(seed: self.config.rngSeed, stream: self.config.rngStream)
      }
  ```
  Replace with:
  ```swift
      // MARK: - Public API
      public let dimension: Int
      public let metric: SupportedDistanceMetric
      public let config: Configuration
      public var count: Int { activeCount }

      /// `toHNSWMetric(metric)` computed once at init and cached here -- `metric` is
      /// an invariant `let` validated at init, so recomputing this pure mapping on
      /// every search/batchSearch/makeKNNBuildContext/insert/prune call (B19) was
      /// wasted work on the query hot path.
      private let hnswMetric: HNSWMetric

      /// Supported metrics for HNSW traversal kernel
      private static let supportedMetrics: Set<SupportedDistanceMetric> = [.euclidean, .dotProduct, .cosine]

      /// Convert SupportedDistanceMetric to HNSWMetric (validated at init).
      /// Only called once per instance now, from the designated initializer below.
      @inline(__always)
      private static func toHNSWMetric(_ metric: SupportedDistanceMetric) -> HNSWMetric {
          switch metric {
          case .euclidean: return .L2
          case .dotProduct: return .IP
          case .cosine: return .COSINE
          case .manhattan, .chebyshev:
              // Should never reach here due to init validation
              preconditionFailure("Unsupported metric '\(metric)' in HNSWIndex")
          }
      }

      public init(dimension: Int, metric: SupportedDistanceMetric = .euclidean, config: Configuration = .init()) {
          guard Self.supportedMetrics.contains(metric) else {
              preconditionFailure("HNSWIndex does not support metric '\(metric)'. Supported metrics: euclidean, dotProduct, cosine. Use FlatIndex for manhattan/chebyshev.")
          }
          self.dimension = dimension
          self.metric = metric
          self.config = config
          self.hnswMetric = Self.toHNSWMetric(metric)
          self.rng35 = HNSWXoroRNGState.from(seed: config.rngSeed, stream: config.rngStream)
      }

      // Protocol-required initializer. B19: this comment already claimed "delegates
      // to the designated one" but the body actually duplicated the precondition
      // check and RNG init instead -- a real bug (comment vs. code), not just style,
      // since this exact two-labeled-argument call shape (HNSWIndex(dimension:
      // metric:), no config:) is what resolves to *this* initializer rather than the
      // designated one with a defaulted config:, and it's the shape ~15+ call sites
      // across HNSWWALTests, TypedOverloadsTests, AccelerableIndexTests,
      // HNSWKNNGraphTests, and ArrayCopyOptimizationBenchmark use. Now it actually
      // delegates.
      public init(dimension: Int, metric: SupportedDistanceMetric) {
          self.init(dimension: dimension, metric: metric, config: .init())
      }
  ```

- [ ] **Step 2: Stop recomputing `toHNSWMetric` in `search` and `batchSearch`**

  Find (appears twice, in `search` around line 178 and `batchSearch` around line 266 -- edit both occurrences):
  ```swift
          // Map metric (validated at init)
          let m33 = Self.toHNSWMetric(metric)
  ```
  Replace each with:
  ```swift
          // Map metric (validated at init, cached in hnswMetric)
          let m33 = hnswMetric
  ```

- [ ] **Step 3: Stop recomputing `toHNSWMetric` in `makeKNNBuildContext`**

  Find (inside the `KNNBuildContext(...)` construction, around line 447):
  ```swift
              metric: Self.toHNSWMetric(metric)
          )
          return (ctx, ids)
  ```
  Replace with:
  ```swift
              metric: hnswMetric
          )
          return (ctx, ids)
  ```

- [ ] **Step 4: Stop recomputing `toHNSWMetric` in insertion's neighbor selection and in `pruneNeighbors`**

  Find (appears twice, once in `internalInsertAtLevel` around line 623 and once in `pruneNeighbors` around line 700 -- edit both occurrences):
  ```swift
          let metric34 = Self.toHNSWMetric(metric)
  ```
  Replace each with:
  ```swift
          let metric34 = hnswMetric
  ```

- [ ] **Step 5: Switch `BatchSearchContext.vectorStorage` to `ContiguousArray<Float>` (COW share, matching `KNNBuildContext`)**

  Find:
  ```swift
      /// Context for parallel batch search - bundles all data needed by worker tasks
      private struct BatchSearchContext: @unchecked Sendable {
          let vectorStorage: [Float]
  ```
  Replace with:
  ```swift
      /// Context for parallel batch search - bundles all data needed by worker tasks.
      /// vectorStorage is ContiguousArray<Float> (B19), matching KNNBuildContext and
      /// the actor's own `private var vectorStorage: ContiguousArray<Float>` storage
      /// -- a plain value assignment is COW-shared (zero bytes copied unless the
      /// actor later mutates mid-search), whereas the previous [Float] forced an
      /// eager Array(vectorStorage) buffer copy of the entire vector store on every
      /// batchSearch() call, even though the TaskGroup workers only ever read it.
      private struct BatchSearchContext: @unchecked Sendable {
          let vectorStorage: ContiguousArray<Float>
  ```
  Then find the construction site:
  ```swift
          let ctx = BatchSearchContext(
              vectorStorage: Array(vectorStorage),
  ```
  Replace with:
  ```swift
          let ctx = BatchSearchContext(
              vectorStorage: vectorStorage,
  ```
  No other call site needs a change -- `ctx.vectorStorage.withUnsafeBufferPointer` (`performSingleSearch`) works identically on `ContiguousArray`.

- [ ] **Step 6: Verify (a)-(c) are behavior-preserving**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | tail -20
  ```
  Expected: `Build complete!`, exit 0.
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'HNSWBatchAndErrorsTests|HNSWKNNGraphTests|HNSWTests|HNSWRecallTests|HNSWParamSweepTests|RegressionA1_TraversalLifetimeTests' 2>&1 | tail -60
  ```
  Expected: 0 failures. `RegressionA1_TraversalLifetimeTests` and the recall/param-sweep suites confirm the init delegation, cached-metric, and COW-context changes produced no observable behavior difference.

- [ ] **Step 7: Add the reserved-but-unemitted comment for `.update`/`.clear`**

  In `HNSWWAL.swift`, find:
  ```swift
  internal enum HNSWWALRecordType: UInt32 {
      case insert = 1
      case remove = 2
      case update = 3
      case clear = 4
      case batchInsert = 5
  }
  ```
  Replace with:
  ```swift
  // B19 finding: .update and .clear are fully implemented, codec-complete record
  // kinds that no current write path emits. HNSWIndex.update(id:vector:metadata:)
  // logs itself as a .remove + .insert pair (see HNSWIndex.swift's update(...)), and
  // clear() does not touch the WAL at all -- applyReplayRecord already has
  // dedicated handling for both (HNSWIndex.swift, the .update/.clear cases), kept
  // for forward compatibility with a possible future single-frame update/batch-clear
  // format. Deliberately not deleted here: this is an on-disk format decision, out
  // of scope for this cleanup pass. .update previously had zero test coverage of any
  // kind (encode, decode, or replay-apply); see HNSWWALTests.swift's
  // testFrameRoundTripAllRecordTypes for its encode/decode round-trip coverage,
  // added in this task.
  internal enum HNSWWALRecordType: UInt32 {
      case insert = 1
      case remove = 2
      case update = 3
      case clear = 4
      case batchInsert = 5
  }
  ```

- [ ] **Step 8: Extend the WAL round-trip test to cover `.update`**

  In `Tests/VectorIndexTests/HNSWWALTests.swift`'s `testFrameRoundTripAllRecordTypes`, find:
  ```swift
          let cases: [HNSWWALRecord] = [
              .insert(insertItem),
              .remove(id: "bravo"),
              .clear,
              .batchInsert([
                  insertItem,
                  HNSWWALInsertItem(id: "beta", level: 0, vector: [1, 0, 0, 0], metadata: nil, knownInvNorm: nil)
              ])
          ]
  ```
  Replace with:
  ```swift
          let cases: [HNSWWALRecord] = [
              .insert(insertItem),
              .remove(id: "bravo"),
              .update(id: "gamma", vector: [0.5, -0.5, 0.25, -0.25], metadata: ["u": "1"]),
              .clear,
              .batchInsert([
                  insertItem,
                  HNSWWALInsertItem(id: "beta", level: 0, vector: [1, 0, 0, 0], metadata: nil, knownInvNorm: nil)
              ])
          ]
  ```
  Then find:
  ```swift
              case (.remove(let a), .remove(let b)):
                  XCTAssertEqual(a, b)
              case (.clear, .clear):
                  break
  ```
  Replace with:
  ```swift
              case (.remove(let a), .remove(let b)):
                  XCTAssertEqual(a, b)
              case (.update(let aid, let avec, let ameta), .update(let bid, let bvec, let bmeta)):
                  // B19: .update had zero test coverage of any kind before this case --
                  // codec-complete but never emitted by any current write path (see the
                  // reserved-but-unemitted note in HNSWWAL.swift).
                  XCTAssertEqual(aid, bid)
                  XCTAssertEqual(avec, bvec)
                  XCTAssertEqual(ameta, bmeta)
              case (.clear, .clear):
                  break
  ```

- [ ] **Step 9: Full task verification**

  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift build 2>&1 | tail -20
  ```
  Expected: `Build complete!`, exit 0.
  ```bash
  cd /Users/goftin/dev/gsuite/VSK/VectorIndex && swift test --filter 'HNSWWALTests|HNSWBatchAndErrorsTests|HNSWKNNGraphTests|HNSWTests|HNSWRecallTests|HNSWParamSweepTests|RegressionA1_TraversalLifetimeTests' 2>&1 | tail -60
  ```
  Expected: 0 failures across all listed suites, including the extended `testFrameRoundTripAllRecordTypes` (now 5 cases instead of 4).

- [ ] **Commit**

  ```bash
  git add Sources/VectorIndex/HNSWIndex.swift Sources/VectorIndex/HNSWWAL.swift Tests/VectorIndexTests/HNSWWALTests.swift
  git commit -m "$(cat <<'EOF'
  refactor(hnsw): fix init delegation, cache hnswMetric, COW batch-search context; cover WAL .update

  HNSWIndex.init(dimension:metric:)'s doc comment already claimed it "delegates to
  the designated one" but the body actually duplicated the precondition check and
  RNG init -- a real bug (comment vs. code). It now genuinely delegates via
  self.init(...). This is the initializer that resolves for the exact
  two-labeled-argument call shape used by ~15+ existing call sites across
  HNSWWALTests, TypedOverloadsTests, AccelerableIndexTests, and HNSWKNNGraphTests,
  so the fix is exercised immediately by the existing suite.

  toHNSWMetric(metric) -- a pure function of an init-validated invariant -- is now
  computed once at init (new private hnswMetric field) instead of on every
  search/batchSearch/makeKNNBuildContext/insert/prune call, including the query hot
  path. BatchSearchContext.vectorStorage switches from [Float] to
  ContiguousArray<Float> (matching KNNBuildContext), removing an eager full-buffer
  copy on every batchSearch() call in favor of a COW share.

  HNSWWAL's .update/.clear record kinds are fully implemented but never emitted by
  any current write path; kept (not deleted, a deliberate on-disk format decision)
  with a comment explaining why, and .update gets its first test coverage
  (encode/decode round-trip, extending the existing .clear round-trip test).
  EOF
  )"
  ```
  (with the executing assistant's standard Co-Authored-By trailer)

---

### Task 18: Phase 2 wrap-up — CHANGELOG, full suite, consumer check

**Files:**
- Modify: `CHANGELOG.md` (the `## [Unreleased] — 0.2.0` → `### Changed` section)

**Interfaces:** none

- [ ] **Step 1: Full suite green**

Run the complete suite in filter groups sized to finish under the per-command timeout (start from the group split the Phase-1 wrap-up used; check `uptime` first and note the load average). Expected: 0 unexpected failures; skipped-by-design counts consistent with the Phase-1 wrap-up record (38 ± the tests this plan added/removed — reconcile the delta explicitly in the report).

- [ ] **Step 2: Append the Changed entries**

Replace the `<!-- cleanup / perf appended per task -->` placeholder under `### Changed` with:

```markdown
- Telemetry consolidated onto the push-callback recorders and the dedup pull API,
  now accuracy-tested; the never-compiled `VINDEX_TELEM` histogram singleton and two
  vestigial `enableTelemetry` flags are gone. (B1, re-scoped)
- `CS2RNG` C target removed; its unique test coverage ported to the pure-Swift RNG API. (B2)
- `L2SqrMicrobench` dev target and stray `PQTrain.swift.new`/`.tmp` files removed. (B4)
- Dead/no-op internals deleted across kernels: `DispatchBK`, alignment/prefetch no-ops,
  the tiled-assign wrapper, unused `selectNeighbors`/`selectBatchSize`, unreachable
  `sumSquares`. (B5–B7, B10, B20)
- One sum-of-squares implementation (`Norms.l2NormSquared`); scalar distance-kernel
  families unified. (B8, B10)
- MIPSTransform internals routed to the canonical scoring kernels; its dead public
  surface is deprecated pending the 0.2.0 breaking phase. (B3)
- mmap tidy: single `CRC32`, shared disk-layout structs, in-place header hashing,
  WAL append-record CRC validated on replay. (B11, B13)
- `IDMap` keeps only the SwissTable backend. (B12)
- `IDFilter`/`CandidateReservoir`/sparse-paged dedup refactored under new
  characterization tests (previously untested); reservoir `.adaptive` overflow prune
  added; `BoundedTopKHeap` replaces the duplicated min/max heaps. (B14–B16)
- Rerank pointer smuggling now uses the shared `UnsafeSendable` box; HNSW neighbor
  selection and compaction stop reallocating distance arrays per comparison; IVF
  `optimize`/`optimizeKMeans` unified (fixing `optimizeKMeans`'s missing
  `idToListIndex` population); HNSW init/metric/context-storage tidy. (B17–B19)
```

Adjust any line whose task was re-scoped during execution so the CHANGELOG states what actually happened — the entries above are the plan's expectation, not a template to keep verbatim against reality.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record Phase 2 cleanup"
```
(with the executing assistant's standard Co-Authored-By trailer)

- [ ] **Step 4: Consumer check**

Run: `cd /Users/goftin/dev/gsuite/VSK/future/VectorIndexAccelerated && swift build 2>&1 | tail -10`
Expected: builds, no worse than the Phase-1 record (clean, ≤1 pre-existing warning). Note: this still resolves the *published* VectorIndex, so it is a nothing-external-regressed check; the real consumer-vs-branch check remains a Phase-5 gate.

- [ ] **Step 5: Line-count delta (report only)**

Run: `git diff --stat main...HEAD | tail -3`
Record the net line reduction in the execution report (spec success criterion #2: "net large line reduction").

---

## Appendix P4 — Phase-4 routing (deprecated this phase, delete in the breaking phase)

Collected from the tasks above at execution time: every symbol a task deprecated instead of deleting (public access), plus the spec's standing C5–C11 list and `ResidualError` (C9). The Phase-4 plan consumes this appendix.

**Routing block 1:**

<!--
Public symbols this plan had to deprecate instead of delete (Phase 2's non-breaking rule),
for the Phase-4 removal appendix:

1. `QueryCtx` (Sources/VectorIndex/Kernels/Telemetry.swift) — dead, VINDEX_TELEM never compiled.
2. `TelemetryConfig` (same file) — dead, same reason.
3. `TelemetryGlobal` (same file) — dead, same reason.
4. `TelemetryCounter` (same file) — dead, same reason.
5. `TelemetryBytes` (same file) — dead, same reason.
6. `TelemetryDoubleField` (same file) — dead, same reason.
7. `TelemetryU64Field` (same file) — dead, same reason.
8. `TelemetryTimerGuard` (same file, `~Copyable` struct) — dead, same reason; its `init`/`deinit`
   still reference the internal `Telemetry` stub enum's `_nowNs()`/`_addTimer()` no-ops, which
   must be removed together with this type in Phase 4.
9. `TelemetryTimerToken` (same file) — dead, same reason.
10. `VisitedOpts.enableTelemetry` (Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift) —
    genuinely public stored property + public-init parameter; never read anywhere in
    `DefaultVisitedSet`, gates nothing. Removing it in Phase 4 also means dropping the
    `enableTelemetry:` parameter from `VisitedOpts.init`, which IS a breaking signature change
    at that point (acceptable in a Phase explicitly scoped for breaking removals).

Not on this list (verified NOT public, so already fully deleted rather than deprecated):
`IDMapOpts.enableTelemetry` — carried a `public` keyword but its container, `IDMapOpts`, is an
`internal struct`, capping the effective access level at internal. Deleted outright in Task 3,
Step 3.

Separately, NOT a deprecation candidate (no public symbol involved either way) but flagged for
the same Phase-4/follow-up planning pass: the `ENABLE_TELEMETRY` compile flag discovered in
Task 3's premise correction #2 (dead + doesn't compile when enabled, gating
`HNSWTelemetryRecorder` and `GlobalTelemetryRecorder`'s only production call sites) needs its
own fix, tracked separately from this deprecation list — see the note after Task 3, Step 11.
-->

**Routing block 2:**

Every public symbol deprecated (not deleted) by Tasks 4-8, for Phase 4 to actually remove:

**From Task 5 — `Sources/VectorIndex/Operations/Transform/MIPSTransform.swift`** (entire file's public surface is dead; zero callers/tests found anywhere in the repo):
- `public enum MIPSTransformMode: Sendable { case explicit, virtual, hybrid }`
- `public struct R2Parameter { var value, isStale, maxNormSquared; let margin; init(maxNormSquared:margin:); mutating func observe(normSquared:); mutating func refresh() }`
- `public struct AugmentedVectorStorage { let originalDim, paddedDim, count; var r2; init(count:originalDim:r2:); mutating func allocate(); func deallocate() }`
- `public struct MIPSTransformTelemetry { let mode, vectorsProcessed, dimension, r2Value, r2Stale, materialized, executionTimeNanos; var throughputVecsPerSec }`
- `public func computeR2Parameter(vectors:count:dimension:margin:) -> R2Parameter`
- `public func mipsMaterializeAugmentation(baseVectors:count:dimension:r2:augmentedOut:paddedDim:)`
- `public func mipsAugmentQuery(query:dimension:augmentedOut:paddedDim:)`
- `public func mipsVirtualToL2Scores(query:baseVectors:count:dimension:r2:scoresOut:)`
- `public func mipsHybridScoreBlock(query:storage:baseVectors:scoresOut:)`

Phase 4 action: delete the entire file (`Sources/VectorIndex/Operations/Transform/MIPSTransform.swift`) — confirmed zero callers/tests for any symbol in it, so no other file needs updating.

**From Task 8 — `Sources/VectorIndex/Kernels/HNSWNeighborSelection.swift`** (dead in production wiring since A9; one direct unit test remains):
- `public func hnsw_prune_neighbors_f32_swift(u:xb:d:offsetsL:neighborsL:M:metric:optionalInvNorms:N:prunedOut:) -> Int` (line 253)
- `@_cdecl("hnsw_prune_neighbors_f32") public func c_hnsw_prune_neighbors_f32(...) -> Int32` (line 366)

Phase 4 action: delete both functions **and** their direct unit test `Tests/VectorIndexTests/HNSWNeighborSelectionTests.swift` → `testPruneNeighborsKeepsTopM_L2` (lines ~50-70-ish at Phase-4 time; re-locate by name) at the same time — that test exists solely to cover this dead-in-production kernel, and `hnsw_select_neighbors_f32_swift`'s own tests plus the broad HNSW insertion/build suite already cover the live neighbor-maintenance path.

**Routing block 3:**

## Phase-4 routing (public symbols deprecated instead of deleted in Tasks 10-14)

Per this plan's NON-BREAKING constraint, no public symbol is removed in Tasks 10-14. The following are deprecated in place and routed to Phase-4 (the first phase where a breaking/major version bump is in scope) for actual deletion:

1. **`HashTableImpl.robinHood`** (`Sources/VectorIndex/Kernels/IDMap.swift`) — Task 12 marks `@available(*, deprecated)`. Dead since B12 collapsed `IDMap` onto a single `SwissTable` backend; zero production or test callers ever selected this case. Phase-4: delete the case; if `HashTableImpl` then has only one case (`.swissTable`) with no remaining reason to exist as a type, consider deleting `HashTableImpl` and `IDMapOpts.hashTableImpl` together at that point (note `IDMapOpts` itself is `internal`, so removing its field is non-breaking whenever it happens — only the enum's cases are the public-facing constraint).
2. **`HashTableImpl.linearProbing`** (same file) — same rationale and same Phase-4 disposition as `.robinHood`.

No other public symbol was touched by Tasks 10-14. Everything else identified as dead/unused in this fragment's brief (`VIndexContainerBuilder.swift`'s `_CRC32`/`_TOCEntry`/`_Header`, `VIndexMmap.swift`'s `fromHost`, `IDMap.swift`'s `RobinHoodTable`/`LinearProbingTable`/`HashTable` enum wrapper, `IDFilter.swift`'s `FilterMode.shouldKeep`, `IVFSelect.swift`'s `MinHeap`/`MaxHeap`) had **no explicit access modifier or was already `private`/`internal`**, i.e. was never part of the package's public API surface, and was therefore deleted outright rather than deprecated.

### Not concretizable / left for a later pass
- **`ListDesc` shared builder-writer** (B11): the design brief's "shared disk-layout structs" framing suggested 3 shared structs, but `ListDesc` has no builder-side mirror to dedup today (the builder writes ListsDesc records via raw packed offsets, never via a named type) — introducing a `writeListDescRecord` helper would be new work, not a dedup of an existing duplicate, and is out of scope for Task 10.
- **`alignUpU64` (builder) vs. `alignUp` (reader)** (adjacent to B11, not in its line list): same-shape byte-identical duplication, deliberately left alone since it wasn't explicitly named in scope.
- **`version_minor` plumbing** (B13): confirmed inert by design (single-format-version policy); documented in place with a comment (Task 11, Step 6), not wired up or removed.
- **Unifying `IVFSelect.swift`'s new `BoundedTopKHeap` with the already-existing, more general `IndexOps.Selection.TopKHeap`** (`Sources/VectorIndex/Operations/Selection/TopK.swift:54`, used by `FlatIndexOptimized.swift`/`ExactRerank.swift`/`TopKMerge.swift`) — a bigger, higher-value "one less heap implementation" change, but not what B16 as scoped asks for (a local, comparator-parameterized replacement of `MinHeap`/`MaxHeap` only). Flagged as a scoping question for whoever owns Phase-3's `IVFSelect.swift` GEMM-rewrite item, since that item already touches this file for other reasons.
- **`IDMapPersistenceTests`'s two `XCTSkip`'d tests** (B12 side-note): their skip reason ("CRC validation needs refactoring for mmap persistence") looks arguably addressed indirectly by Tasks 10/11's CRC cleanup, but re-enabling and fixing them was not attempted here — it's a distinct investigation (confirm the underlying mmap-persistence CRC issue is actually resolved, not just the code it was pointing at being tidied) that risks scope creep into this fragment's five tasks. Left for a follow-up task.

**Routing block 4:**

- `IndexOps.Rerank.RerankOpts.returnSorted` (`Sources/VectorIndex/Operations/Rerank/ExactRerank.swift`) — deprecated Task 15d; never honored, results are always emitted best-first. Delete the property (and its now-unused `_returnSorted` backing field + `returnSorted` init parameter) in Phase 4.
- `IndexOps.Rerank.topKIVF(q:d:metric:candInternalIDs:id2List:id2Offset:lists:K:opts:)` (array-based overload, `ExactRerank.swift:536`) — deprecated Task 15b; zero callers besides the equally-dead `IVFPostADC.rerankTopKFlat`. Delete in Phase 4 (its recursive `buildAndRun` closure goes with it — no need to flatten it first).
- `IndexOps.Rerank.scoresIVF(q:d:metric:candInternalIDs:id2List:id2Offset:lists:opts:)` (`ExactRerank.swift:588`) — deprecated Task 15b; zero callers anywhere in the package. Delete in Phase 4.
- `IVFPostADC.rerankTopKFlat(q:d:metric:candInternalIDs:id2List:id2Offset:lists:K:opts:)` (`Sources/VectorIndex/Operations/Quantization/IVFPostADC.swift:20`) — deprecated Task 15b; zero callers anywhere in the package. Delete in Phase 4; consider deleting the whole `IVFPostADC.swift` file if nothing else ever lands in that enum.
