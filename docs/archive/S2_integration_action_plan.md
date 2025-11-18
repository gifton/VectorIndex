> NOTE (Archived 2025-11-12): This plan is archived. S2 integration has been
> completed; see S2_implementation_summary.md and S2_phase3_testing_summary.md
> for the final results.

# S2 (RNG & Dtype Helpers) Integration Action Plan (Archived)

## Status: ⚠️ **BLOCKED** - Critical bugs must be fixed before merge

---

## Critical Bugs (MUST FIX)

### 🔴 P0: Determinism Violation in NEON Quantization
**File**: `s_rng_dtype_helpers.c:581-600`
**Issue**: When `__ARM_FEATURE_FRINT` is undefined (M1, older ARM), `vcvtq_s32_f32` uses round-toward-zero instead of ties-to-even, breaking determinism guarantee.

**Fix**:
```c
#if !defined(__ARM_FEATURE_FRINT)
static inline float32x4_t vrndnq_f32_compat(float32x4_t v) {
    // Implement software tie-to-even using floor + correction
    float32x4_t abs_v = vabsq_f32(v);
    float32x4_t floor_v = vrndmq_f32(abs_v);  // floor
    float32x4_t frac = vsubq_f32(abs_v, floor_v);

    // frac > 0.5 → ceil
    uint32x4_t ceil_mask = vcgtq_f32(frac, vdupq_n_f32(0.5f));

    // frac == 0.5 → round to even (check LSB of floor_v)
    uint32x4_t tie_mask = vceqq_f32(frac, vdupq_n_f32(0.5f));
    uint32x4_t floor_i = vcvtq_u32_f32(floor_v);
    uint32x4_t is_odd = vandq_u32(floor_i, vdupq_n_u32(1));
    uint32x4_t round_up = vorrq_u32(ceil_mask, vandq_u32(tie_mask, is_odd));

    float32x4_t result = vbslq_f32(round_up,
                                   vaddq_f32(floor_v, vdupq_n_f32(1.0f)),
                                   floor_v);

    // Restore sign
    return vbslq_f32(vreinterpretq_u32_f32(v) & 0x80000000u,
                     vnegq_f32(result),
                     result);
}
#define vrndnq_f32(v) vrndnq_f32_compat(v)
#endif
```

**Estimated effort**: 2-3 hours (implement + test on M1/M2/M3)

---

### 🔴 P0: Thread-Unsafe Telemetry Counters
**File**: `s_rng_dtype_helpers.c:26-38`
**Issue**: Global `g_tel` has data races in multi-threaded workloads.

**Fix**:
```c
#include <stdatomic.h>

typedef struct {
    atomic_uint_fast64_t bytes_f32_to_f16;
    atomic_uint_fast64_t bytes_f16_to_f32;
    atomic_uint_fast64_t bytes_f32_to_bf16;
    atomic_uint_fast64_t bytes_bf16_to_f32;
    atomic_uint_fast64_t bytes_q_i8;
    atomic_uint_fast64_t bytes_dq_i8;
    atomic_uint_fast64_t bytes_pack_u4;
    atomic_uint_fast64_t bytes_unpack_u4;
    atomic_uint_fast64_t saturations_i8;
    atomic_uint_fast64_t rounding_mode_last;  // Non-atomic OK (advisory only)
} S2Telemetry;

static inline void tel_add_u64(atomic_uint_fast64_t* p, uint64_t v) {
#if S2_ENABLE_TELEMETRY
    atomic_fetch_add_explicit(p, v, memory_order_relaxed);
#else
    (void)p; (void)v;
#endif
}
```

**Estimated effort**: 1 hour

---

### 🟡 P1: Missing Saturation Telemetry in NEON Path
**File**: `s_rng_dtype_helpers.c:598` (and other NEON quantization paths)
**Issue**: `vqmovn` saturates silently; telemetry only counts scalar path saturations.

**Fix**:
```c
// Before vqmovn, detect saturation
int16x8_t max_s8 = vdupq_n_s16(127);
int16x8_t min_s8 = vdupq_n_s16(-128);
uint16x8_t sat_hi = vcgtq_s16(p0, max_s8);
uint16x8_t sat_lo = vcltq_s16(p0, min_s8);
uint16x8_t sat_mask = vorrq_u16(sat_hi, sat_lo);

// Count saturations (horizontal sum)
uint16x4_t sat_low = vget_low_u16(sat_mask);
uint16x4_t sat_high = vget_high_u16(sat_mask);
uint32_t sat_count = vaddv_u16(sat_low) + vaddv_u16(sat_high);
tel_add_u64(&g_tel.saturations_i8, sat_count);

int8x8_t q0 = vqmovn_s16(p0);
```

**Estimated effort**: 1 hour (apply to all quantize paths)

---

## Major Issues (SHOULD FIX)

### 🟡 P1: Document Alignment Requirements
**Action**: Add alignment assertions or use explicit unaligned API.

```c
void f32_to_f16(const float* src, uint16_t* dst, int n, RoundingMode rm) {
    // Assert alignment if spec requires it
    assert(((uintptr_t)src & 0x3F) == 0 && "src must be 64-byte aligned");
    assert(((uintptr_t)dst & 0x3F) == 0 && "dst must be 64-byte aligned");

    // OR: Use unaligned API and update spec
    // vld1q_f32(src) is already unaligned-safe on modern ARM
}
```

**Estimated effort**: 30 minutes

---

### 🟡 P1: Vectorize `unpack_nibbles_u4` with NEON
**Current**: Scalar bit ops at ~10 GB/s
**Target**: ≥40 GB/s (spec requirement)

**Approach**: Use NEON bit manipulation:
```c
// Process 16 packed bytes → 32 nibbles
uint8x16_t packed = vld1q_u8(in + i);
uint8x16_t low_nibbles = vandq_u8(packed, vdupq_n_u8(0x0F));
uint8x16_t high_nibbles = vshrq_n_u8(packed, 4);
// Interleave with vzip
```

**Estimated effort**: 2 hours

---

### 🟢 P2: BF16 NaN Handling Code Clarity
**File**: `s_rng_dtype_helpers.c:304-308`
**Issue**: Comment doesn't clarify that check happens *after* rounding.

**Fix**: Add comment explaining post-rounding NaN check.

**Estimated effort**: 5 minutes

---

## Integration Steps

### Phase 1: Pre-merge Fixes (Block merge until complete)
- [ ] Fix P0: Determinism violation (NEON tie-to-even)
- [ ] Fix P0: Atomic telemetry counters
- [ ] Fix P1: NEON saturation counting
- [ ] Fix P1: Document/assert alignment requirements
- [ ] Verify fixes on M1, M2, M3 hardware

### Phase 2: Integration
- [ ] Create `Sources/VectorIndex/Operations/` directory
- [ ] Move `s_rng_dtype_helpers.{c,h}` into Operations/
- [ ] Update Package.swift with C sources
- [ ] Create Swift wrappers (`RNGHelpers.swift`, `DTypeConversions.swift`)
- [ ] Add bridging header or module map

### Phase 3: Testing
- [ ] Run `RNGDeterminismTests` on 3+ different seeds
- [ ] Run `DTypeConversionTests` with edge cases
- [ ] Benchmark on M2/M3: verify ≥30 GB/s for f32↔f16
- [ ] Multi-threaded stress test (ThreadSanitizer enabled)
- [ ] Cross-platform test (Intel Mac via Rosetta 2 if available)

### Phase 4: Integration with Existing Kernels
- [ ] Replace ad-hoc RNG in K-means++ (#11) with xoroshiro128**
- [ ] Use f16 conversions for memory-compressed indices (if applicable)
- [ ] Integrate PQ 4-bit packing with existing PQ implementation (#19, #20)
- [ ] Add telemetry hooks to Swift Metrics or os_signpost

### Phase 5: Documentation
- [ ] Add usage examples to README
- [ ] Document determinism guarantees and testing methodology
- [ ] Add performance benchmarks to docs
- [ ] Create migration guide for existing code

---

## Validation Checklist

Before marking S2 as "complete":
- [ ] All P0/P1 bugs fixed
- [ ] Tests pass on M1, M2, M3 with ThreadSanitizer
- [ ] Performance targets met (≥30 GB/s f32↔f16, ≥40 GB/s u4 unpack)
- [ ] Determinism verified: same seed → same output across 10 runs
- [ ] Multi-threaded test: 8 threads × 1M operations, no data races
- [ ] Code review by second engineer
- [ ] Integration tests pass with existing VectorIndex operations

---

## Open Questions

1. **Alignment policy**: Require 64-byte alignment (assert/crash) or support unaligned (slower)?
   - **Recommendation**: Support unaligned, document that 64-byte alignment gives +10% perf

2. **Error handling**: Return error codes or assert on invalid inputs?
   - **Recommendation**: Assert in debug, UB in release (match C library conventions)

3. **Telemetry in release builds**: Enable by default or opt-in?
   - **Recommendation**: Disabled by default, enable via `S2_ENABLE_TELEMETRY=1`

4. **Quantization scale computation**: Provide helper or leave to caller?
   - **Recommendation**: Add `float compute_scale_i8_symmetric(const float* x, int n)`

---

## Timeline Estimate

- **P0 fixes**: 4-5 hours
- **P1 fixes**: 3-4 hours
- **Integration + wrappers**: 4 hours
- **Testing + validation**: 6 hours
- **Documentation**: 2 hours

**Total**: ~2-3 days of focused work

---

## Risk Assessment

**High Risk**:
- Determinism fix is non-trivial; must be validated on multiple ARM generations
- Performance regression if NEON paths disabled due to bugs

**Medium Risk**:
- Atomic telemetry may have 5-10% overhead; measure and document
- Alignment assertions may break existing code

**Low Risk**:
- Integration with Swift is straightforward (C interop is mature)
- RNG algorithms are well-tested (xoroshiro, Philox are standard)

---

## Dependencies

- **Toolchain**: Xcode 15+ (C11 `<stdatomic.h>` support)
- **Hardware access**: M1, M2, M3 for validation
- **Existing code**: Audit for RNG usage that needs migration

---

## Success Criteria

S2 is "done" when:
1. ✅ All P0 bugs fixed and validated
2. ✅ Performance targets met on M2/M3
3. ✅ Determinism proven via 100+ runs with different seeds
4. ✅ Thread-safe under ThreadSanitizer
5. ✅ Integrated with at least one existing kernel (K-means++ or PQ)
6. ✅ Documentation complete with examples

---

**Next Steps**: Review this plan, then proceed with P0 fixes.
