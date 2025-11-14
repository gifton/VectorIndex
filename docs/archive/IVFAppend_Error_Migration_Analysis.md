> NOTE (Archived 2025-11-12): This document is archived. IVFAppend precondition
> migrations have been implemented and validated with tests. See CHANGELOG and
> KNOWN_ISSUES for current status.

# IVFAppend.swift Error Migration Analysis (Archived)
**Deep Technical Analysis of Remaining 7 precondition/fatalError Calls**

---

## Executive Summary

After deep analysis, **4 out of 7 calls should be migrated** to proper error handling.
The other 3 are legitimate programming invariants that should remain as preconditions.

---

## Detailed Analysis

### ✅ KEEP: Line 24 - `alignedAlloc` alignment precondition

```swift
@inline(__always) private func alignedAlloc(_ size: Int, alignment: Int = 64) -> UnsafeMutableRawPointer? {
    precondition(isPowerOfTwo(alignment))  // ← Line 24
    var p: UnsafeMutableRawPointer? = nil
    let err = posix_memalign(&p, alignment, size)
    return err == 0 ? p : nil
}
```

**Type:** Programming invariant
**Rationale:**
- `alignment` parameter is **hardcoded to 64** at all call sites
- This validates a POSIX `posix_memalign` API contract (must be power-of-two)
- Private internal helper, never exposed to users
- If this fails, it's a VectorIndex bug, not user error

**Decision:** **KEEP as precondition** ✅

---

### ✅ KEEP: Line 71 - `packNibblesU4` even count precondition

```swift
@inline(__always)
private func packNibblesU4(idx4: UnsafePointer<UInt8>, n: Int, out: UnsafeMutablePointer<UInt8>) {
    precondition(n % 2 == 0)  // ← Line 71
    // ... nibble packing logic ...
}
```

**Type:** Mathematical invariant
**Rationale:**
- Nibble packing requires even count (pack pairs of 4-bit values into bytes)
- Parameter `n` is always `m` (number of PQ subspaces)
- We already validate: `m % group == 0` where `group ∈ {4, 8}`
- Therefore: `m % 2 == 0` is **mathematically guaranteed** by initialization checks
- Private helper, defensive check on established invariant

**Decision:** **KEEP as precondition** ✅

---

### ✅ KEEP: Line 174 - `codeBytesPerVector` PQ4 precondition

```swift
@inline(__always)
public var codeBytesPerVector: Int {
    switch format {
    case .pq8: return m
    case .pq4: precondition(m % 2 == 0); return m >> 1  // ← Line 174
    case .flat: return d * MemoryLayout<Float>.stride
    }
}
```

**Type:** Initialization invariant
**Rationale:**
- Only executes when `format == .pq4`
- Initialization validates: `m % group == 0` where `group ∈ {4, 8}` for PQ formats
- Therefore: `m % 2 == 0` is guaranteed by init validation
- Defensive check on immutable property established at construction

**Decision:** **KEEP as precondition** ✅

---

### 🔧 MIGRATE: Lines 63, 65 - `storeExternalID` storage mismatch

```swift
@inline(__always)
private func storeExternalID(_ storage: inout IDStorage, _ index: Int, _ val64: UInt64, opts: IVFAppendOpts) throws {
    if opts.id_bits == 32 {
        guard val64 <= UInt64(UInt32.max) else { throw IVFError.idWidthUnsupported }
        if case .u32(let p) = storage {
            p![index] = UInt32(truncatingIfNeeded: val64)
        } else {
            fatalError("ID storage kind mismatch")  // ← Line 63
        }
    } else {
        if case .u64(let p) = storage {
            p![index] = val64
        } else {
            fatalError("ID storage kind mismatch")  // ← Line 65
        }
    }
}
```

**Type:** Internal inconsistency (detectable runtime error)
**Root Causes:**
1. `opts.id_bits` modified after `IDStorage` construction
2. Memory corruption of enum discriminant
3. Concurrent access race condition
4. Bug in storage allocation logic

**Impact Analysis:**
- **Call sites:** 6 locations (all append/insert operations)
- **Already throws:** Function already throws `IVFError.idWidthUnsupported`
- **Propagation:** All callers already handle `throws`

**Migration Strategy:**
```swift
// Current: Crashes on mismatch
fatalError("ID storage kind mismatch")

// Proposed: Throw with context
throw ErrorBuilder(.internalInconsistency, operation: "store_id")
    .message("ID storage type mismatch (internal bug)")
    .info("expected_bits", "\(opts.id_bits)")
    .info("storage_variant", storage == .u32 ? "u32" : "u64")
    .build()
```

**Decision:** **MIGRATE to `.internalInconsistency`** 🔧

---

### 🔧 MIGRATE: Line 372 - `growList` storage mismatch

```swift
private func growList(...) throws {
    // ... allocate new buffers ...

    if n > 0 {
        switch (list.ids, newIDs) {
        case (.u64(let old), .u64(let neu)):
            memcpy(neu, old, n * MemoryLayout<UInt64>.stride)
        case (.u32(let old), .u32(let neu)):
            memcpy(neu, old, n * MemoryLayout<UInt32>.stride)
        default:
            fatalError("ID storage kind mismatch on grow")  // ← Line 372
        }
        // ...
    }
}
```

**Type:** Internal inconsistency (detectable runtime error)
**Root Causes:**
1. Old and new `IDStorage` constructed with different `opts.id_bits`
2. `opts.id_bits` modified during growth operation
3. Memory corruption

**Impact Analysis:**
- **Call sites:** 5 locations (all append operations that trigger growth)
- **Already throws:** Function already throws for allocation failures
- **Context:** Error path only (no performance impact on success)

**Migration Strategy:**
```swift
// Current: Crashes on mismatch
default: fatalError("ID storage kind mismatch on grow")

// Proposed: Throw with diagnostic info
default:
    let oldBits = (list.ids == .u32) ? 32 : 64
    let newBits = (newIDs == .u32) ? 32 : 64
    throw ErrorBuilder(.internalInconsistency, operation: "grow_list")
        .message("ID storage type mismatch during list growth (internal bug)")
        .info("old_id_bits", "\(oldBits)")
        .info("new_id_bits", "\(newBits)")
        .info("expected_bits", "\(opts.id_bits)")
        .build()
```

**Decision:** **MIGRATE to `.internalInconsistency`** 🔧

---

### 🔧 MIGRATE: Line 593 - `ivf_insert_at` format mismatch

```swift
public func ivf_insert_at(list_id: Int32, pos: Int, external_ids: UnsafePointer<UInt64>,
                          codes: UnsafePointer<UInt8>, n: Int, index: IVFListHandle) throws {
    guard list_id >= 0 && list_id < Int32(index.k_c) else { throw IVFError.invalidListID }
    guard n >= 0 else { throw IVFError.invalidInput }
    // ... NO FORMAT VALIDATION HERE ...

    // ... later in function ...
    for i in 0..<n {
        try storeExternalID(&L.ids, pos + i, external_ids[i], opts: index.opts)
        switch index.format {
        case .pq8: /* ... */
        case .pq4: /* ... */
        case .flat:
            fatalError("Use ivf_insert_at_flat for IVF-Flat")  // ← Line 593
        }
    }
}
```

**Type:** API contract violation (user error, detectable)
**Root Cause:**
- User called `ivf_insert_at()` on a **flat format** index
- Should have called `ivf_insert_at_flat()` instead
- Function lacks format validation at entry

**Impact Analysis:**
- **Public API:** User-facing function
- **Recoverable:** User can call correct function
- **Documentation:** API docs don't specify format restriction

**Migration Strategy:**
```swift
// Add format validation at function entry
public func ivf_insert_at(...) throws {
    guard index.format == .pq8 || index.format == .pq4 else {
        throw ErrorBuilder(.unsupportedLayout, operation: "ivf_insert_at")
            .message("ivf_insert_at requires PQ format; use ivf_insert_at_flat for flat format")
            .info("actual_format", "\(index.format)")
            .build()
    }
    // ... rest of function ...
}

// Remove fatalError from switch (now unreachable)
switch index.format {
case .pq8: /* ... */
case .pq4: /* ... */
case .flat:
    break  // Unreachable due to guard above
}
```

**Additional Work:**
- Update API documentation with format requirement
- Add test for error path

**Decision:** **MIGRATE to `.unsupportedLayout`** 🔧

---

## Implementation Phases

### Phase 1: `storeExternalID` Migration (Lines 63, 65)
**Scope:** 2 fatalError calls in same function
**Impact:** All append/insert operations
**Complexity:** Medium (6 call sites already handle throws)
**Testing:** Add tests for ID storage corruption scenarios

### Phase 2: `growList` Migration (Line 372)
**Scope:** 1 fatalError in growth logic
**Impact:** Append operations that trigger capacity increase
**Complexity:** Low (already in error path)
**Testing:** Add test for storage type mismatch during growth

### Phase 3: `ivf_insert_at` Migration (Line 593)
**Scope:** 1 fatalError + add format guard
**Impact:** Public API, user-facing
**Complexity:** Low (straightforward validation)
**Testing:** Add test for calling with flat format

---

## Risk Assessment

### Low Risk Migrations
- ✅ **Line 372 (`growList`)**: Already throws, rare error path
- ✅ **Line 593 (`ivf_insert_at`)**: Clear user error with recovery

### Medium Risk Migrations
- ⚠️ **Lines 63, 65 (`storeExternalID`)**: Called in hot path, but errors should be extremely rare

### Performance Impact
- **Happy path:** Zero overhead (errors only thrown on corruption)
- **Error path:** ~100ns for error construction (acceptable)

---

## Testing Strategy

### Unit Tests Required
1. **`storeExternalID` mismatch:**
   - Manually corrupt `IDStorage` enum (unsafe code)
   - Verify throws `.internalInconsistency`
   - Check error metadata includes storage types

2. **`growList` mismatch:**
   - Mock scenario where old/new storage types differ
   - Verify throws `.internalInconsistency`
   - Check error includes diagnostic info

3. **`ivf_insert_at` format error:**
   - Create flat format index
   - Call `ivf_insert_at` (not `_flat` variant)
   - Verify throws `.unsupportedLayout`
   - Check recovery message suggests correct API

### Integration Tests
- Verify all append/insert operations handle new error types
- Confirm no performance regression on happy path

---

## Documentation Updates

### API Documentation
```swift
/// Inserts vectors at specified position in PQ-format IVF list
///
/// - Important: This function requires PQ8 or PQ4 format.
///              For flat format indices, use `ivf_insert_at_flat()`.
///
/// - Throws:
///   - `VectorIndexError(.unsupportedLayout)`: If index format is flat
///   - `VectorIndexError(.internalInconsistency)`: If internal storage corruption detected
///   - `IVFError.invalidListID`: If list_id out of range
///   - `IVFError.outOfRange`: If position invalid
///   - `IVFError.allocationFailed`: If capacity increase fails
public func ivf_insert_at(...) throws
```

---

## Estimated Effort

- **Phase 1:** 2 hours (storeExternalID + tests)
- **Phase 2:** 1 hour (growList + tests)
- **Phase 3:** 1.5 hours (ivf_insert_at + tests + docs)

**Total:** ~4.5 hours for complete migration

---

**Status:** Analysis Complete - Ready for Implementation
**Next Action:** Begin Phase 1 (storeExternalID migration)
