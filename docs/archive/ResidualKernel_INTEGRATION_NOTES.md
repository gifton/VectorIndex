> NOTE (Archived 2025-11-12): Superseded by ResidualKernel_FINAL_EVALUATION.md.

# Residual Kernel Integration Notes (Archived)

## Summary
External agent provided algorithmically correct implementation of Kernel #23 (Residual Computation).

**Rating**: 6.5/10 - Good foundation but needs production hardening

## Critical Fixes Required

### 1. Operator Consistency
**Lines**: ~353, 369, 427, 445
**Issue**: Uses `&+=` despite comment claiming "no wrapping operators"
**Fix**:
```swift
// Before:
counts[a] &+= 1

// After:
counts[a] += 1
```

### 2. Hot-Path Allocation
**Function**: `residual_pq_encode_u8_f32`
**Issue**: Allocates scratch buffer per call (4KB-6KB typical)
**Fix**: Add parameter:
```swift
public func residual_pq_encode_u8_f32(
    ...,
    scratch: UnsafeMutablePointer<Float>  // Caller-provided [d] buffer
)
```

### 3. Error Handling
**Issue**: `precondition()` crashes on invalid input
**Fix**: Introduce error types:
```swift
public enum ResidualError: Error, Sendable {
    case invalidCoarseID(index: Int, value: Int32, validRange: Range<Int>)
    case dimensionMismatch(expected: Int, actual: Int)
    case invalidAlignment(address: UInt, required: Int)
}

public func residuals_f32(...) throws {
    guard a >= 0 && a < opts.kc else {
        throw ResidualError.invalidCoarseID(index: i, value: coarseIDs[i], validRange: 0..<opts.kc)
    }
    ...
}
```

### 4. PQ Kernel Integration
**Issue**: Fused functions duplicate logic from kernels #20 and #21
**Fix**: Refactor to compose:
```swift
// In residual_pq_encode_u8_f32:
for i in 0..<nInt {
    // Compute residual into scratch
    let vec = x.advanced(by: i * d)
    let cen = coarseCentroids.advanced(by: Int(coarseIDs[i]) * d)
    vDSP_vsub(cen, 1, vec, 1, scratch, 1, vDSP_Length(d))

    // Call existing PQ encode kernel (don't reimplement!)
    try pq_encode_u8_f32_single(
        scratch, d: d, m: m, ks: ks,
        codebooks: codebooks,
        codeOut: codesOut.advanced(by: i * m)
    )
}
```

## Performance Optimizations

### 5. Accelerate Framework
Add fast path for large dimensions:
```swift
@inlinable
func _residual_subtract_vdsp(
    _ vec: UnsafePointer<Float>,
    _ cen: UnsafePointer<Float>,
    _ out: UnsafeMutablePointer<Float>,
    _ d: Int
) {
    vDSP_vsub(cen, 1, vec, 1, out, 1, vDSP_Length(d))
}

// In residuals_f32:
if d >= 256 && !opts.groupByCentroid {
    _residual_subtract_vdsp(vec, cen, out, d)
} else {
    // Existing SIMD4 path for small d or grouped processing
}
```

### 6. Alignment Validation
Add debug-mode checks:
```swift
#if DEBUG
func _checkAlignment(_ ptr: UnsafeRawPointer, _ alignment: Int) {
    assert(Int(bitPattern: ptr) % alignment == 0,
           "Pointer \(ptr) not aligned to \(alignment) bytes")
}
#endif

// In residuals_f32:
_checkAlignment(UnsafeRawPointer(x), 16)
_checkAlignment(UnsafeRawPointer(coarseCentroids), 16)
```

## Telemetry Integration

### 7. Connect to System #46
Wrap kernels with instrumentation:
```swift
public func residuals_f32_instrumented(
    ...,
    telemetry: TelemetryCollector?
) throws {
    let startTime = DispatchTime.now().uptimeNanoseconds

    try residuals_f32(x, coarseIDs, coarseCentroids, n, d, rOut, opts)

    let elapsed = DispatchTime.now().uptimeNanoseconds - startTime
    telemetry?.emit(ResidualTelemetry(
        n: n, d: d, fused: false, grouped: opts.groupByCentroid,
        timeNanos: elapsed, bytesWritten: Int64(n) * Int64(d) * 4
    ))
}
```

## Testing Requirements

### 8. Test Suite (from spec)
Implement all 6 tests from kernel-specs/23_residuals.md:
- [ ] `testResidualsCorrectness` - scalar reference comparison
- [ ] `testInPlaceCorrectness` - in-place vs out-of-place parity
- [ ] `testFusedEncodingParity` - fused vs non-fused encoding
- [ ] `testFusedLUTParity` - fused vs non-fused LUT
- [ ] `testGroupedParity` - grouped vs ungrouped processing
- [ ] `testResidualThroughput` - performance benchmark (>30M vec/s)

Location: `Tests/VectorIndexTests/ResidualKernelTests.swift`

## API Surface Changes

### 9. Consistency with Existing Kernels
Match naming conventions from PQEncode/PQTrain:
```swift
// Current:
residuals_f32(...)  // ✅ Good (matches pq_encode_u8_f32)

// But options should match project style:
public struct ResidualConfiguration: Sendable {  // Not "Opts"
    public let grouping: GroupingStrategy  // Not "groupByCentroid: Bool"
    public let validation: ValidationMode  // Not "checkBounds: Bool"
}

public enum GroupingStrategy: Sendable {
    case none
    case byCentroid
}

public enum ValidationMode: Sendable {
    case disabled
    case debug      // Assertions only
    case production // Throws errors
}
```

## Documentation Additions

### 10. Mathematical Rigor
Add proofs per CLAUDE.md requirements:
```swift
/// Computes residual vectors **r** = **x** - **c**_{a(i)} for IVF-PQ.
///
/// ## Mathematical Formulation
/// Given:
/// - Input vectors **X** ∈ ℝ^{n×d}
/// - Coarse centroids **C** ∈ ℝ^{k_c×d}
/// - Assignment function a: [n] → [k_c]
///
/// Compute:
///   **r**_i = **x**_i - **c**_{a(i)}  ∀i ∈ [0, n)
///
/// ## Complexity Analysis
/// - Time: O(n·d) subtractions
/// - Space: O(n·d) output (materialized) or O(d) scratch (fused)
/// - Cache: O(k_c·d) centroid working set
///
/// ## Numerical Properties
/// - **Stability**: Subtraction is numerically stable for similar-magnitude operands
/// - **Error bound**: |fl(x - c) - (x - c)| ≤ ε|x - c| where ε = 2^-24 (Float32)
/// - **Determinism**: Bitwise identical results for identical inputs
///
/// ## Performance Characteristics (Apple M2 Max)
/// - d=512:  50M vec/s (200ms for 10M vectors)
/// - d=1024: 40M vec/s (250ms for 10M vectors)
/// - d=1536: 30M vec/s (333ms for 10M vectors)
///
/// - Complexity: O(n) time, O(1) space per vector
/// - Throws: `ResidualError` if validation fails
@inlinable
public func residuals_f32(...) throws { ... }
```

## Integration Steps

1. **Create feature branch**: `kernel-23-residuals`
2. **Fix critical issues**: Operators, allocations, error handling
3. **Write tests**: All 6 from spec
4. **Run benchmarks**: Verify >30M vec/s on M2
5. **Integrate telemetry**: Connect to system #46
6. **Code review**: Focus on SIMD correctness and alignment
7. **Merge**: After all tests pass + benchmark validation

## Dependencies

- ✅ Kernel #12 (IVF training) - for coarse centroids
- ⚠️ Kernel #20 (PQ encode) - needs refactoring for composition
- ⚠️ Kernel #21 (PQ LUT) - needs refactoring for composition
- ❌ Kernel #46 (Telemetry) - must integrate before merge

## Performance Validation

Run benchmarks with:
```bash
swift test --configuration release --filter ResidualKernelPerformanceTests
```

Expected results (M2 Max, 8 P-cores):
- d=512, n=10M: >50M vec/s
- d=1024, n=10M: >40M vec/s
- Grouped (kc=10K): 10-15% improvement
- Fused vs non-fused: Same throughput, -40GB memory

## Risk Assessment

**Low Risk**:
- Core SIMD algorithm is correct
- No complex concurrency patterns
- Well-documented spec compliance

**Medium Risk**:
- Alignment assumptions not validated
- Hot-path allocations may impact latency
- Fused paths need integration testing

**High Risk**:
- No error recovery (crashes on bad input)
- Duplicated PQ logic creates maintenance burden
- Missing comprehensive test coverage

## Recommendation

**Proceed with integration** after addressing critical fixes 1-4.

The algorithmic foundation is solid, but production readiness requires:
1. Error handling
2. Memory allocation fixes
3. PQ kernel composition
4. Comprehensive tests

Estimated effort: 2-3 days for hardening + testing.
