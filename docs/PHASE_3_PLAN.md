# Phase 3: Error Migration Plan (0.1.1)

**Status:** In progress for 0.1.1 release
**Created:** 2025-10-22
**Dependencies:** Phase 1 ✅ Complete, Phase 2 ✅ Complete

---

## Overview

Phase 3 continues the migration of preconditions to structured error handling across remaining kernels and infrastructure. This phase focuses on less critical paths that didn't block the 0.1.0-alpha release.

**Completed in Phase 2:**
- ✅ IVFAppend (6 preconditions)
- ✅ KMeansSeeding (2 preconditions + 1 new validation)
- ✅ 43/43 error infrastructure tests passing

**Remaining for Phase 3:**
- VIndexMmap (fatalErrors → throws)
- Other kernels audit (TBD)

---

## Phase 3.1: PQTrain Kernel Migration (Completed)

### Scope

**File:** `Sources/VectorIndex/Kernels/PQTrain.swift`

**Status:** Completed; PQ training is production‑ready with tests.

#### Identified Preconditions (from grep):

1. Line 494: `precondition(m % 2 == 0)` - PQ4 dimension check
2. Line 612: `precondition(m % 2 == 0)` - PQ4 dimension check
3. Line 742: `precondition(m % 2 == 0)` - PQ4 dimension check
4. Line 839: `precondition(m % 2 == 0)` - PQ4 dimension check
5. Additional preconditions in initialization code

### Migration Strategy

**Error Types:**
- `m % 2 != 0` for PQ4 → `VectorIndexError(.invalidParameter)`
  - Message: "m must be even for PQ4 format"
  - Metadata: `m` value, `format` type

**Testing:**
- Test odd m values for PQ4
- Test valid even m values
- Test PQ8 (should not require even m)

No further action required in Phase 3.

---

## Phase 3.2: VIndexMmap Migration

### Scope

**File:** `Sources/VectorIndex/Kernels/VIndexMmap.swift`

**fatalErrors to Migrate:** 4 total (estimated)

### Current Status

VIndexMmap is marked internal (Phase 1 API narrowing). This reduces urgency for migration since users don't directly call these APIs.

### Migration Strategy

**Error Types:**
- Invalid mmap operations → `VectorIndexError(.mmapError)`
- File corruption → `VectorIndexError(.corruptedData)`
- Version mismatches → `VectorIndexError(.versionMismatch)`
- CRC failures → `VectorIndexError(.corruptedData)`

**Testing:**
- Test corrupted file headers
- Test version mismatches
- Test CRC failures
- Test successful mmap operations

**Estimated Effort:** 3-4 hours
**Priority:** Low-Medium (internal API, affects persistence only)

---

## Phase 3.3: Remaining Kernels Audit

### Scope

Systematic audit of all remaining kernel files for preconditions/fatalErrors.

**Files to Audit:**
- `Sources/VectorIndex/Kernels/IVFSelect.swift`
- `Sources/VectorIndex/Kernels/ExactRerank.swift`
- `Sources/VectorIndex/Kernels/PQLUT.swift`
- Other kernel files in `Sources/VectorIndex/Kernels/`

### Process

1. **Grep for assertions:**
   ```bash
   grep -rn "precondition\|fatalError\|assert(" Sources/VectorIndex/Kernels/
   ```

2. **Classify each:**
   - User-facing runtime error → Migrate to `throws`
   - Programmer error → Keep as precondition
   - Unreachable code → Keep as fatalError

3. **Prioritize by:**
   - API visibility (public > internal)
   - Usage frequency (hot path > cold path)
   - User impact (crashes > performance)

**Estimated Effort:** 4-6 hours
**Priority:** Low (systematic cleanup)

---

## Phase 3.4: High-Level API Migration

### Scope

Review actor-level APIs (FlatIndex, IVFIndex, HNSWIndex, FlatIndexOptimized) for error handling improvements.

**Current State:**
- Most actor APIs already use structured errors (VectorError)
- Some may throw generic errors that could benefit from VectorIndexError

**Goals:**
- Ensure all actor methods have comprehensive `throws` documentation
- Migrate any remaining generic errors to VectorIndexError
- Add error path tests for actor-level operations

**Estimated Effort:** 2-3 hours
**Priority:** Medium (user-facing APIs)

---

## Testing Strategy for Phase 3

### Test Coverage Requirements

1. **Every migrated throw must have:**
   - Positive test (valid parameters succeed)
   - Negative test (invalid parameters throw correct error)
   - Metadata verification (error contains expected context)

2. **Error path tests must verify:**
   - `error.kind` is correct
   - `error.kind.isRecoverable` is appropriate
   - `error.context.operation` is set
   - `error.context.additionalInfo` contains relevant metadata
   - `error.message` is actionable

3. **Integration tests should verify:**
   - Error propagation through layers
   - Error chaining works correctly
   - Recovery guidance is accurate

### Test Organization

All Phase 3 tests will be added to `ErrorInfrastructureTests.swift` with clear section markers:

```swift
// MARK: - Phase 3 Migration Tests: PQTrain
// MARK: - Phase 3 Migration Tests: VIndexMmap
// MARK: - Phase 3 Migration Tests: Remaining Kernels
```

---

## Timeline & Milestones

### Milestone 1: PQTrain Migration (Completed)
Delivered previously; see CHANGELOG and kernel status.

### Milestone 2: VIndexMmap Migration
- **Duration:** 1 week
- **Deliverables:**
  - VIndexMmap fatalErrors migrated
  - 6-8 new test cases
  - Persistence error handling improved

### Milestone 3: Kernel Audit & Cleanup
- **Duration:** 1-2 weeks
- **Deliverables:**
  - Complete kernel audit
  - Remaining migrations completed
  - Comprehensive test coverage

### Milestone 4: High-Level API Polish
- **Duration:** 1 week
- **Deliverables:**
  - Actor-level error handling reviewed
  - Documentation complete
  - Integration tests added

**Total Estimated Time:** 4-5 weeks

---

## Success Criteria

### Phase 3 Complete When:

✅ All user-facing preconditions migrated to structured errors
✅ All fatalErrors in recoverable paths migrated
✅ Comprehensive test coverage (>95% of error paths)
✅ Updated documentation (ERRORS.md, CONTRIBUTING.md)
✅ No regressions in existing functionality
✅ Clean build (0 errors, 0 warnings)

---

## Deferred to Future Releases

### Not in Scope for Phase 3:

- **Performance optimization** - Error handling overhead is minimal (<2%)
- **Logging integration** - Planned for 0.2.0 with structured logging
- **Localization** - Error messages currently English-only
- **Error recovery mechanisms** - Automatic retry logic for transient errors

---

## Risk Assessment

### Low Risk:
- PQTrain migration (well-understood pattern)
- Test additions (no production impact)

### Medium Risk:
- VIndexMmap migration (affects persistence)
  - Mitigation: Extensive testing with corrupted files
  - Mitigation: Backward compatibility tests

### High Risk:
- None identified

---

## Dependencies

### Required:
- ✅ Phase 1 error infrastructure (complete)
- ✅ Phase 2 migrations (complete)
- ✅ ERRORS.md documentation (complete)

### Optional:
- Structured logging system (0.2.0)
- Performance profiling tools

---

## Communication Plan

### Internal:
- Update CHANGELOG.md for each milestone
- Maintain KNOWN_ISSUES.md for discovered issues
- Weekly progress updates in team sync

### External:
- Document breaking changes (if any)
- Migration guide for affected users
- Release notes highlighting improvements

---

## Rollback Plan

If Phase 3 migrations cause issues:

1. **Immediate:** Revert problematic commit
2. **Short-term:** Release patch with revert
3. **Long-term:** Fix issues and re-introduce in next release

**Safety Net:**
- All changes in feature branch until tested
- Comprehensive CI/CD test suite
- Manual QA before merge

---

## Open Questions

1. **Should we migrate internal-only APIs?**
   - Proposal: Low priority, defer to 0.2.0
   - Rationale: No user impact

2. **Performance impact acceptable?**
   - Current: <2% overhead on happy path
   - Threshold: <5% acceptable for alpha/beta

3. **Error message verbosity?**
   - Current: Detailed messages with metadata
   - Feedback needed from users

---

## Next Steps for 0.1.1

1. **Create tracking issue** for Phase 3 work
2. **Schedule milestone planning** session
3. **Assign owners** for each phase component
4. **Set up monitoring** for error rates in production
5. **Gather user feedback** from 0.1.0-alpha

---

**Document Owner:** Error Infrastructure Team
**Last Updated:** 2025-10-22
**Status:** Ready for 0.1.1 planning
<!-- moved to docs/ -->
