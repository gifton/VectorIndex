/*
 * s_rng_dtype_helpers.h
 * ID: S2 — RNG & Dtype Helpers (MUST)
 *
 * Purpose:
 *   Fast, deterministic RNG primitives and robust dtype conversion/packing
 *   utilities for training/seeding and cross-kernel usage.
 *
 * Provides:
 *   - RNG: xoroshiro128**, Philox4x32-10, utilities for K-means++, sampling
 *   - Dtype: f32↔f16/bf16 casts, int8 quantize/dequant, PQ 4-bit pack/unpack
 *   - Endian helpers, alignment macros, telemetry hooks
 *
 * Thread Safety:
 *   - RNG state is NOT thread-safe; use separate Xoro128 per thread
 *   - All conversion functions are thread-safe and reentrant
 *   - Telemetry counters use atomics (thread-safe)
 *
 * Determinism Guarantee:
 *   Given (seed, stream_id, worker_id), outcomes are identical across runs,
 *   chips, and thread counts for the same partitioning.
 */

#ifndef S_RNG_DTYPE_HELPERS_H
#define S_RNG_DTYPE_HELPERS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * RNG: xoroshiro128** (fast stateful stream)
 * ========================================================================== */

/// xoroshiro128** state (128 bits)
typedef struct {
    uint64_t s0, s1;
} Xoro128;

/// Initialize xoroshiro128** with seed and stream ID.
/// - seed: 64-bit seed value
/// - stream_id: 64-bit stream identifier for independent sequences
/// - Ensures state is never (0, 0)
void xoro128_init(Xoro128* x, uint64_t seed, uint64_t stream_id);

/// Generate next 64-bit random value
uint64_t xoro128_next_u64(Xoro128* x);

/// Generate next 32-bit random value (uses upper 32 bits of u64)
uint32_t xoro128_next_u32(Xoro128* x);

/// Generate uniform float in [0, 1) with 24-bit precision
float xoro128_next_uniform(Xoro128* x);

/// Generate uniform double in [0, 1) with 53-bit precision
double xoro128_next_uniform_f64(Xoro128* x);

/// Skip ahead n steps (simple forward iteration)
void rng_skip_ahead_xoro(Xoro128* x, uint64_t n);

/* ============================================================================
 * RNG: Philox4x32-10 (counter-based, stateless)
 * ========================================================================== */

/// Derive 128-bit Philox key from seed and stream ID
void philox_key(uint64_t seed, uint64_t stream_id, uint64_t* k0, uint64_t* k1);

/// Generate 4×32-bit random values from key and counter
/// - k0, k1: 128-bit key from philox_key()
/// - ctr_lo, ctr_hi: 128-bit counter
/// - out: array of 4 uint32_t values
void philox_next4(uint64_t k0, uint64_t k1,
                  uint64_t ctr_lo, uint64_t ctr_hi,
                  uint32_t out[4]);

/* ============================================================================
 * RNG: Utilities
 * ========================================================================== */

/// Stable seed/stream derivation for parallel workers
void rng_split(uint64_t seed, int worker_id, int task_id,
               uint64_t* out_seed, uint64_t* out_stream);

/// Fisher-Yates shuffle in-place
void randperm_inplace(uint32_t* a, int n, Xoro128* x);

/// Sample k indices from [0, n) without replacement (Vitter's Algorithm S)
void sample_without_replacement(uint32_t n, uint32_t k, uint32_t* out, Xoro128* x);

/// Box-Muller transform for Gaussian N(0,1) samples
void gaussian_box_muller(float* out, int n, Xoro128* x);

/// Weighted random choice (returns index in [0, n), or -1 if all weights ≤0)
int weighted_pick(const float* w, int n, Xoro128* x);

/// Subsample m indices from [0, n) without replacement (returns actual count)
int subsample_indices(uint32_t n, uint32_t m, uint32_t* out, Xoro128* x);

/* ============================================================================
 * Dtype: f32 ↔ f16/bf16 conversion
 * ========================================================================== */

/// Rounding modes for float conversions
typedef enum {
    NearestTiesToEven = 0,  ///< IEEE 754 default (RNE)
    TowardZero = 1,         ///< Truncate
} RoundingMode;

/// Convert f32 → f16 (IEEE 754 binary16)
/// - Preserves NaN payload, signed zeros
/// - Saturates overflow to ±Inf
/// - Uses NEON fcvtn on AArch64 with fp16 support
/// - Alignment: Optimal with 64-byte aligned buffers; supports unaligned
void f32_to_f16(const float* src, uint16_t* dst, int n, RoundingMode rm);

/// Convert f16 → f32 (lossless)
void f16_to_f32(const uint16_t* src, float* dst, int n);

/// Convert f32 → bf16 (bfloat16)
/// - Preserves NaN payload, signed zeros
/// - Saturates overflow to ±Inf
void f32_to_bf16(const float* src, uint16_t* dst, int n, RoundingMode rm);

/// Convert bf16 → f32 (lossless)
void bf16_to_f32(const uint16_t* src, float* dst, int n);

/* ============================================================================
 * Dtype: int8 quantization
 * ========================================================================== */

/// Symmetric quantization: x → round(x / scale), clamped to [-128, 127]
/// - scale: typically max(|x|) / 127 (computed by caller)
/// - Rounding: nearest ties-to-even
/// - Updates telemetry saturation counter
void quantize_i8_symmetric(const float* x, int n, float scale, int8_t* y);

/// Symmetric dequantization: y → y * scale
void dequantize_i8_symmetric(const int8_t* y, int n, float scale, float* x);

/// Affine (asymmetric) quantization: x → round(x / scale) + zero_point
/// - zero_point: typically -128 to map [0, 510] to [-128, 127]
void quantize_i8_affine(const float* x, int n, float scale, int32_t zero_point, int8_t* y);

/// Affine dequantization: y → (y - zero_point) * scale
void dequantize_i8_affine(const int8_t* y, int n, float scale, int32_t zero_point, float* x);

/* ============================================================================
 * Dtype: PQ 4-bit pack/unpack
 * ========================================================================== */

/// Pack 4-bit codes into bytes (low nibble = first code)
/// - idx4[n]: array of n values in [0, 15]
/// - out[(n+1)/2]: packed bytes
void pack_nibbles_u4(const uint8_t* idx4, int n, uint8_t* out);

/// Unpack bytes into 4-bit codes (low nibble = first code)
/// - in[(n+1)/2]: packed bytes
/// - idx4[n]: array of n unpacked values
void unpack_nibbles_u4(const uint8_t* in, int n, uint8_t* idx4);

/* ============================================================================
 * Endian helpers (little-endian serialization)
 * ========================================================================== */

/// Load little-endian values from memory (unaligned-safe)
uint16_t le16(const void* p);
uint32_t le32(const void* p);
uint64_t le64(const void* p);

/// Store little-endian values to memory (unaligned-safe)
void store_le16(void* p, uint16_t v);
void store_le32(void* p, uint32_t v);
void store_le64(void* p, uint64_t v);

/* ============================================================================
 * Alignment & stride macros
 * ========================================================================== */

#define ALIGN_UP(x, a)       (((x) + ((a) - 1)) & ~((a) - 1))
#define IS_ALIGNED(ptr, a)   (((uintptr_t)(ptr) & ((a) - 1)) == 0)
#define PAD_TO(len, mult)    (((len) + (mult) - 1) / (mult) * (mult))

/* ============================================================================
 * Telemetry (optional counters, thread-safe)
 * ========================================================================== */

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdatomic.h>
typedef atomic_uint_fast64_t S2TelemetryCounter;
#else
// Fallback for pre-C11: use plain uint64_t (not thread-safe)
typedef uint64_t S2TelemetryCounter;
#endif

typedef struct {
    S2TelemetryCounter bytes_f32_to_f16;
    S2TelemetryCounter bytes_f16_to_f32;
    S2TelemetryCounter bytes_f32_to_bf16;
    S2TelemetryCounter bytes_bf16_to_f32;
    S2TelemetryCounter bytes_q_i8;
    S2TelemetryCounter bytes_dq_i8;
    S2TelemetryCounter bytes_pack_u4;
    S2TelemetryCounter bytes_unpack_u4;
    S2TelemetryCounter saturations_i8;
    uint64_t rounding_mode_last;  ///< Non-atomic (advisory only)
} S2Telemetry;

/// Get current telemetry snapshot (read-only)
const S2Telemetry* s2_get_telemetry(void);

/// Reset all telemetry counters to zero
void s2_reset_telemetry(void);

#ifdef __cplusplus
}
#endif

#endif /* S_RNG_DTYPE_HELPERS_H */
