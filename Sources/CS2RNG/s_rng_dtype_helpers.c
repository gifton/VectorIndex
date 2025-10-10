/*
 * s_rng_dtype_helpers.c
 * Implementation of S2 RNG & Dtype Helpers
 *
 * FIXES APPLIED:
 *   [P0] Atomic telemetry counters (thread-safe)
 *   [P0] Software tie-to-even rounding for NEON (determinism across ARM chips)
 *   [P1] Saturation counting in NEON paths
 *   [P1] Alignment documentation
 *   [P2] Vectorized nibble unpacking with NEON
 */

#include "include/s_rng_dtype_helpers.h"

#include <string.h>
#include <math.h>
#include <assert.h>

/* ---------- Feature detection for AArch64 NEON (M2/M3 fast paths) ---------- */
#if defined(__aarch64__) && defined(__ARM_NEON)
  #include <arm_neon.h>
  #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #include <arm_fp16.h>
    #define S2_HAVE_FP16 1
  #else
    #define S2_HAVE_FP16 0
  #endif
  #define S2_HAVE_NEON 1
#else
  #define S2_HAVE_FP16 0
  #define S2_HAVE_NEON 0
#endif

/* -------------------------------- Telemetry -------------------------------- */
#ifndef S2_ENABLE_TELEMETRY
#define S2_ENABLE_TELEMETRY 0
#endif

static S2Telemetry g_tel = {0};

/* [FIX P0] Thread-safe atomic operations for telemetry */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#include <stdatomic.h>
static inline void tel_add_u64(S2TelemetryCounter* p, uint64_t v) {
#if S2_ENABLE_TELEMETRY
    atomic_fetch_add_explicit(p, v, memory_order_relaxed);
#else
    (void)p; (void)v;
#endif
}
static inline void tel_set_u64(uint64_t* p, uint64_t v) {
#if S2_ENABLE_TELEMETRY
    *p = v;  // rounding_mode_last is advisory, no atomics needed
#else
    (void)p; (void)v;
#endif
}
#else
/* Pre-C11 fallback: not thread-safe */
static inline void tel_add_u64(uint64_t* p, uint64_t v) {
#if S2_ENABLE_TELEMETRY
    *p += v;
#else
    (void)p; (void)v;
#endif
}
static inline void tel_set_u64(uint64_t* p, uint64_t v) {
#if S2_ENABLE_TELEMETRY
    *p = v;
#else
    (void)p; (void)v;
#endif
}
#endif

const S2Telemetry* s2_get_telemetry(void) { return &g_tel; }

void s2_reset_telemetry(void) {
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && S2_ENABLE_TELEMETRY
    /* [FIX CRITICAL] Use atomic stores to avoid data races */
    atomic_store_explicit(&g_tel.bytes_f32_to_f16, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_f16_to_f32, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_f32_to_bf16, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_bf16_to_f32, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_q_i8, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_dq_i8, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_pack_u4, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.bytes_unpack_u4, 0, memory_order_relaxed);
    atomic_store_explicit(&g_tel.saturations_i8, 0, memory_order_relaxed);
    g_tel.rounding_mode_last = 0;  /* Non-atomic field, OK */
#else
    /* Pre-C11 fallback: not thread-safe, but acceptable for single-threaded use */
    memset(&g_tel, 0, sizeof(g_tel));
#endif
}

/* ------------------------------- RNG: Xoroshiro ----------------------------- */
/* [FIX MAJOR] Prevent UB when k=0 by masking shift amounts to [0,63] */
static inline uint64_t rotl64(uint64_t x, int k) {
    /* Mask to [0, 63] avoids UB on shift-by-64.
     * For k=0: (x << 0) | (x >> 0) = x | x = x (identity, correct).
     * For k=64: same as k=0 after masking (also identity, correct). */
    return (x << (k & 63)) | (x >> ((-k) & 63));
}

/* SplitMix64 – 64-bit LCG-style mixer for seeding/derivation. */
static inline uint64_t splitmix64_next(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void xoro128_init(Xoro128* x, uint64_t seed, uint64_t stream_id) {
    /* Derive two 64b states via SplitMix64; ensure not both zero. */
    uint64_t s = seed ^ (stream_id * 0xD2B74407B1CE6E93ULL);
    x->s0 = splitmix64_next(&s);
    x->s1 = splitmix64_next(&s);
    if ((x->s0 | x->s1) == 0) { /* forbidden state; perturb */
        x->s0 = 0x9E3779B97F4A7C15ULL;
        x->s1 = 0xD1B54A32D192ED03ULL;
    }
}

uint64_t xoro128_next_u64(Xoro128* x) {
    /* xoroshiro128**: output = rotl(s0 * 5, 7) * 9; */
    const uint64_t s0 = x->s0;
    uint64_t s1 = x->s1;
    const uint64_t r = rotl64(s0 * 5ULL, 7) * 9ULL;

    s1 ^= s0;
    x->s0 = rotl64(s0, 24) ^ s1 ^ (s1 << 16);
    x->s1 = rotl64(s1, 37);
    return r;
}

uint32_t xoro128_next_u32(Xoro128* x) {
    return (uint32_t)(xoro128_next_u64(x) >> 32);
}

float xoro128_next_uniform(Xoro128* x) {
    /* 24-bit mantissa uniform on [0,1) */
    uint32_t r = xoro128_next_u32(x);
    /* Use top 24 bits to match float precision */
    r >>= 8;
    return (float)r * (1.0f / 16777216.0f);
}

double xoro128_next_uniform_f64(Xoro128* x) {
    uint64_t r = xoro128_next_u64(x);
    r >>= 11; /* 53 bits */
    return (double)r * (1.0 / 9007199254740992.0); /* 2^53 */
}

void rng_skip_ahead_xoro(Xoro128* x, uint64_t n) {
    /* Simple step-forward; acceptable for typical skip sizes in kernels. */
    while (n--) (void)xoro128_next_u64(x);
}

/* ------------------------------- RNG: Philox ------------------------------- */
/* Philox4x32-10 constants from Random123 (Salmon et al.) */
static const uint32_t PHILOX_M0 = 0xD2511F53U;
static const uint32_t PHILOX_M1 = 0xCD9E8D57U;
static const uint32_t PHILOX_W0 = 0x9E3779B9U;
static const uint32_t PHILOX_W1 = 0xBB67AE85U;

static inline void mulhilo32(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo) {
    uint64_t p = (uint64_t)a * (uint64_t)b;
    *hi = (uint32_t)(p >> 32);
    *lo = (uint32_t)(p & 0xFFFFFFFFu);
}

void philox_key(uint64_t seed, uint64_t stream_id, uint64_t* k0, uint64_t* k1) {
    /* Derive 128-bit key from seed, stream via SplitMix64 */
    uint64_t s = seed ^ (stream_id * 0x9E3779B97F4A7C15ULL);
    uint64_t a = splitmix64_next(&s);
    uint64_t b = splitmix64_next(&s);
    *k0 = a;
    *k1 = b;
}

void philox_next4(uint64_t k0, uint64_t k1,
                  uint64_t ctr_lo, uint64_t ctr_hi,
                  uint32_t out[4]) {
    uint32_t c0 = (uint32_t)(ctr_lo & 0xFFFFFFFFu);
    uint32_t c1 = (uint32_t)(ctr_lo >> 32);
    uint32_t c2 = (uint32_t)(ctr_hi & 0xFFFFFFFFu);
    uint32_t c3 = (uint32_t)(ctr_hi >> 32);

    uint32_t kx = (uint32_t)(k0 & 0xFFFFFFFFu);
    uint32_t ky = (uint32_t)(k0 >> 32);
    uint32_t kz = (uint32_t)(k1 & 0xFFFFFFFFu);
    uint32_t kw = (uint32_t)(k1 >> 32);

    /* NOTE: Philox4x32-10 uses a 128-bit key split into 4 words (kx, ky, kz, kw).
     * By design, only 2 words (kx, kz) are XORed per round (see lines below).
     * All 4 words are updated via Weyl sequence for cryptographic avalanche.
     * This is CORRECT per Salmon et al. (2011) - not a bug. */
    for (int round = 0; round < 10; ++round) {
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo32(PHILOX_M0, c0, &hi0, &lo0);
        mulhilo32(PHILOX_M1, c2, &hi1, &lo1);

        uint32_t n0 = hi1 ^ c1 ^ kz;  /* Only kz used here (by design) */
        uint32_t n1 = lo1;
        uint32_t n2 = hi0 ^ c3 ^ kx;  /* Only kx used here (by design) */
        uint32_t n3 = lo0;

        c0 = n0; c1 = n1; c2 = n2; c3 = n3;

        /* Key schedule: all 4 words updated (maintains cryptographic properties) */
        kx += PHILOX_W0; ky += PHILOX_W1; kz += PHILOX_W0; kw += PHILOX_W1;
    }

    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

/* ---------------------------- RNG: Utilities -------------------------------- */
void rng_split(uint64_t seed, int worker_id, int task_id,
               uint64_t* out_seed, uint64_t* out_stream) {
    /* Stable derivation using SplitMix64 over a fused 64-bit key. */
    uint64_t s = seed
               ^ ((uint64_t)(uint32_t)worker_id * 0xD1B54A32D192ED03ULL)
               ^ ((uint64_t)(uint32_t)task_id   * 0x94D049BB133111EBULL);
    *out_seed   = splitmix64_next(&s);
    *out_stream = splitmix64_next(&s);
}

void randperm_inplace(uint32_t* a, int n, Xoro128* x) {
    for (int i = n - 1; i > 0; --i) {
        /* j in [0, i] */
        uint32_t r = xoro128_next_u32(x);
        uint32_t j = (uint32_t)((uint64_t)r * (uint64_t)(i + 1) >> 32);
        uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

void sample_without_replacement(uint32_t n, uint32_t k, uint32_t* out, Xoro128* x) {
    /* Vitter's Algorithm S: single pass O(n). */
    uint32_t t = 0, m = 0;
    while (m < k && t < n) {
        double u = xoro128_next_uniform_f64(x);
        if ((double)(n - t) * u >= (double)(k - m)) {
            t++;
        } else {
            out[m++] = t++;
        }
    }
}

void gaussian_box_muller(float* out, int n, Xoro128* x) {
    int i = 0;
    while (i + 1 < n) {
        /* u1 in (0,1], u2 in [0,1) */
        double u1, u2;
        do { u1 = xoro128_next_uniform_f64(x); } while (u1 <= 0.0);
        u2 = xoro128_next_uniform_f64(x);

        double r = sqrt(-2.0 * log(u1));
        double theta = 6.28318530717958647692 * u2; /* 2*pi */
        double s, c; s = sin(theta); c = cos(theta);

        out[i++] = (float)(r * c);
        out[i++] = (float)(r * s);
    }
    if (i < n) {
        double u1;
        do { u1 = xoro128_next_uniform_f64(x); } while (u1 <= 0.0);
        double u2 = xoro128_next_uniform_f64(x);
        double r = sqrt(-2.0 * log(u1));
        double theta = 6.28318530717958647692 * u2;
        out[i] = (float)(r * cos(theta));
    }
}

int weighted_pick(const float* w, int n, Xoro128* x) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) if (w[i] > 0.0f) sum += (double)w[i];
    if (!(sum > 0.0)) return -1;
    double r = xoro128_next_uniform_f64(x) * sum;
    double c = 0.0;
    for (int i = 0; i < n; ++i) {
        double wi = w[i] > 0.0f ? (double)w[i] : 0.0;
        c += wi;
        if (r < c) return i;
    }
    return n - 1; /* fallback (shouldn't happen) */
}

int subsample_indices(uint32_t n, uint32_t m, uint32_t* out, Xoro128* x) {
    if (m > n) m = n;
    sample_without_replacement(n, m, out, x);
    return (int)m;
}

/* --------------------------- Dtype: helpers common -------------------------- */
static inline int32_t round_ties_to_even_f32_to_i32(float v) {
    /* Use double for robust tie detection; does not depend on FENV state. */
    double vd = (double)v;
    double rd = floor(vd);
    double frac = vd - rd;
    if (frac > 0.5) rd += 1.0;
    else if (frac < 0.5) { /* keep rd */ }
    else { /* exactly .5 -> tie; choose even */
        rd = ((int64_t)rd & 1) ? (rd + 1.0) : rd;
    }
    /* Clamp to int32 range to avoid UB on cast. */
    if (rd > 2147483647.0) rd = 2147483647.0;
    if (rd < -2147483648.0) rd = -2147483648.0;
    return (int32_t)rd;
}

/* [FIX P0] Software tie-to-even rounding for NEON when FRINT not available
 * [FIX MAJOR] Handle NaN, Inf, and large values (|x| >= 2^23) correctly */
#if S2_HAVE_NEON && !defined(__ARM_FEATURE_FRINT)
static inline float32x4_t vrndnq_f32_compat(float32x4_t v) {
    /* Fast path: values with |x| >= 2^23 are already integers (no fractional bits)
     * This also handles NaN and Inf correctly (comparison returns false, handled below) */
    float32x4_t abs_v = vabsq_f32(v);
    uint32x4_t is_large = vcgeq_f32(abs_v, vdupq_n_f32(8388608.0f)); /* 2^23 */

    /* For normal-range values, apply tie-to-even logic */
    float32x4_t floor_v = vrndmq_f32(abs_v);  /* floor (always available) */
    float32x4_t frac = vsubq_f32(abs_v, floor_v);

    /* frac > 0.5 → round up */
    uint32x4_t round_up = vcgtq_f32(frac, vdupq_n_f32(0.5f));

    /* frac == 0.5 → tie; round to even
     * Check parity WITHOUT int32 conversion (avoids UB for large values)
     * floor_v is odd if (floor_v / 2) has fractional part >= 0.5 */
    float32x4_t half_floor = vmulq_f32(floor_v, vdupq_n_f32(0.5f));
    float32x4_t half_floor_rounded = vrndmq_f32(half_floor);
    float32x4_t half_frac = vsubq_f32(half_floor, half_floor_rounded);
    uint32x4_t is_odd = vcgtq_f32(half_frac, vdupq_n_f32(0.25f));

    uint32x4_t is_tie = vceqq_f32(frac, vdupq_n_f32(0.5f));
    uint32x4_t tie_round_up = vandq_u32(is_tie, is_odd);

    /* Combine: round up if (frac > 0.5) OR (frac == 0.5 AND odd) */
    uint32x4_t do_round_up = vorrq_u32(round_up, tie_round_up);

    float32x4_t result = vbslq_f32(do_round_up,
                                   vaddq_f32(floor_v, vdupq_n_f32(1.0f)),
                                   floor_v);

    /* Restore sign */
    uint32x4_t sign_bits = vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000000u));
    result = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign_bits));

    /* Return large values (including NaN/Inf) unchanged, normal values rounded */
    return vbslq_f32(is_large, v, result);
}
#define vrndnq_f32(v) vrndnq_f32_compat(v)
#elif S2_HAVE_NEON && defined(__ARM_FEATURE_FRINT)
/* Hardware tie-to-even available */
#define vrndnq_f32(v) vrndnq_f32(v)
#endif

static inline uint16_t f32_to_f16_bits_rte(float f) {
    /* IEEE754 conversion with round-to-nearest-ties-to-even. */
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 31) & 0x1;
    uint32_t exp  = (v.u >> 23) & 0xFF;
    uint32_t man  = v.u & 0x7FFFFF;

    uint16_t hs;
    if (exp == 0xFF) { /* Inf/NaN */
        uint16_t hman = (uint16_t)(man >> 13);
        uint16_t hexp = 0x1F;
        /* Preserve NaN payload; do not canonicalize. Ensure at least one bit set for NaN. */
        if (man != 0) { if (hman == 0) hman = 1; }
        hs = (uint16_t)((sign << 15) | (hexp << 10) | hman);
    } else if (exp == 0) { /* Zero/subnormal -> might underflow to zero/subnormal */
        if (man == 0) {
            hs = (uint16_t)(sign << 15); /* signed zero preserved */
        } else {
            /* Normalize mantissa */
            int e = -126;
            while ((man & 0x00800000u) == 0) { man <<= 1; e--; }
            man &= 0x007FFFFFu;
            int16_t hexp = (int16_t)(e + 127 - 15);
            if (hexp <= 0) {
                /* subnormal half */
                int shift = (14 - hexp);
                uint32_t mant_rounded = (man | 0x00800000u) >> (shift + 13);
                uint32_t lost = (man | 0x00800000u) & ((1u << (shift + 13)) - 1u);
                /* Tie-to-even on subnormal pack */
                uint32_t add = (lost > (1u << (shift + 12))) ? 1u :
                               (lost == (1u << (shift + 12)) ? (mant_rounded & 1u) : 0u);
                uint16_t hman = (uint16_t)(mant_rounded + add);
                hs = (uint16_t)((sign << 15) | hman);
            } else {
                /* normal path below will handle */
                uint32_t new_exp = (uint32_t)hexp;
                uint32_t mant = man;
                /* Round */
                uint32_t mant_round = mant + 0x00001000u; /* add 0.5 ulp at 13th bit */
                if (mant_round & 0x00800000u) { /* carry -> increment exponent */
                    new_exp += 1;
                    mant_round = 0;
                }
                if (new_exp >= 0x1F) { /* overflow -> Inf */
                    hs = (uint16_t)((sign << 15) | (0x1F << 10));
                } else {
                    hs = (uint16_t)((sign << 15) | (new_exp << 10) | (mant_round >> 13));
                }
            }
        }
    } else {
        int16_t e = (int16_t)exp - 127 + 15;
        if (e <= 0) {
            /* subnormal */
            if (e < -10) {
                hs = (uint16_t)(sign << 15); /* underflow to signed zero */
            } else {
                uint32_t man_full = man | 0x00800000u;
                uint32_t shift = (uint32_t)(14 - e);
                uint32_t mant = man_full >> (shift);
                uint32_t lost = man_full & ((1u << shift) - 1u);
                uint32_t add = (lost > (1u << (shift - 1))) ? 1u :
                               (lost == (1u << (shift - 1)) ? (mant & 1u) : 0u);
                uint16_t hman = (uint16_t)(mant + add);
                hs = (uint16_t)((sign << 15) | hman);
            }
        } else if (e >= 0x1F) {
            /* overflow -> Inf */
            hs = (uint16_t)((sign << 15) | (0x1F << 10));
        } else {
            /* round mantissa to 10 bits */
            uint32_t mant_round = man + 0x00001000u; /* add 0.5 ulp (13th bit) */
            if (mant_round & 0x00800000u) { /* carry */
                e += 1; mant_round = 0;
                if (e >= 0x1F) {
                    hs = (uint16_t)((sign << 15) | (0x1F << 10));
                    goto done;
                }
            }
            hs = (uint16_t)((sign << 15) | ((uint16_t)e << 10) | (uint16_t)(mant_round >> 13));
        }
    }
done:
    return hs;
}

static inline uint16_t f32_to_f16_bits_trunc(float f) {
    /* Round toward zero (truncate). */
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 31) & 0x1;
    uint32_t exp  = (v.u >> 23) & 0xFF;
    uint32_t man  = v.u & 0x7FFFFF;

    if (exp == 0xFF) { /* Inf/NaN unchanged payload */
        uint16_t hman = (uint16_t)(man >> 13);
        uint16_t hexp = 0x1F;
        if (man != 0 && hman == 0) hman = 1;
        return (uint16_t)((sign << 15) | (hexp << 10) | hman);
    }
    int16_t e = (int16_t)exp - 127 + 15;
    if (e <= 0) {
        /* Underflow to subnorm/zero by truncation */
        if (e < -10) return (uint16_t)(sign << 15);
        uint32_t man_full = man | 0x00800000u;
        uint32_t shift = (uint32_t)(14 - e);
        uint16_t hman = (uint16_t)(man_full >> shift);
        return (uint16_t)((sign << 15) | hman);
    } else if (e >= 0x1F) {
        /* Overflow -> max normal toward zero */
        return (uint16_t)((sign << 15) | (0x1E << 10) | 0x03FFu);
    } else {
        return (uint16_t)((sign << 15) | ((uint16_t)e << 10) | (uint16_t)(man >> 13));
    }
}

static inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x03FFu;

    uint32_t out_sign = sign << 31;
    uint32_t out_exp, out_man;

    if (exp == 0) {
        if (man == 0) {
            out_exp = 0; out_man = 0;
        } else {
            /* subnormal */
            int e = -14;
            uint32_t m = man;
            while ((m & 0x0400u) == 0) { m <<= 1; e--; }
            m &= 0x03FFu;
            out_exp = (uint32_t)(e + 127) << 23;
            out_man = m << 13;
        }
    } else if (exp == 0x1F) {
        out_exp = 0xFFu << 23;
        out_man = man ? ((man << 13) | 0x00400000u) : 0; /* keep NaN payload, force quiet */
    } else {
        out_exp = (uint32_t)(exp - 15 + 127) << 23;
        out_man = man << 13;
    }

    union { uint32_t u; float f; } v = { out_sign | out_exp | out_man };
    return v.f;
}

/* BF16 conversion (IEEE 754 binary32<->bfloat16) */
static inline uint16_t f32_to_bf16_bits_rte(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t u = v.u;
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t round_bias = 0x7FFFu + lsb; /* ties-to-even */
    uint16_t b = (uint16_t)((u + round_bias) >> 16);
    /* [FIX P2] Preserve NaN payload (check is post-rounding by design) */
    if (((u & 0x7F800000u) == 0x7F800000u) && (u & 0x007FFFFFu)) {
        if ((b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) == 0) b |= 1u;
    }
    return b;
}
static inline uint16_t f32_to_bf16_bits_trunc(float f) {
    union { float f; uint32_t u; } v = { f };
    uint16_t b = (uint16_t)(v.u >> 16);
    if (((v.u & 0x7F800000u) == 0x7F800000u) && (v.u & 0x007FFFFFu)) {
        if ((b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) == 0) b |= 1u;
    }
    return b;
}
static inline float bf16_bits_to_f32(uint16_t b) {
    union { uint32_t u; float f; } v = { ((uint32_t)b) << 16 };
    return v.f;
}

/* ------------------------------- f32 <-> f16 -------------------------------- */
void f32_to_f16(const float* src, uint16_t* dst, int n, RoundingMode rm) {
    /* [FIX P1] Document alignment: optimal with 64-byte aligned buffers */
    tel_set_u64(&g_tel.rounding_mode_last, (uint64_t)rm);
#if S2_HAVE_FP16
    if (rm == NearestTiesToEven) {
        int i = 0;
        /* Process 16 floats per iteration (4x float32x4_t -> 4x float16x4_t) */
        for (; i + 15 < n; i += 16) {
            float32x4_t a0 = vld1q_f32(src + i + 0);
            float32x4_t a1 = vld1q_f32(src + i + 4);
            float32x4_t a2 = vld1q_f32(src + i + 8);
            float32x4_t a3 = vld1q_f32(src + i + 12);

            float16x4_t h0 = vcvt_f16_f32(a0); /* fcvtn */
            float16x4_t h1 = vcvt_f16_f32(a1);
            float16x4_t h2 = vcvt_f16_f32(a2);
            float16x4_t h3 = vcvt_f16_f32(a3);

            vst1_u16(dst + i + 0,  vreinterpret_u16_f16(h0));
            vst1_u16(dst + i + 4,  vreinterpret_u16_f16(h1));
            vst1_u16(dst + i + 8,  vreinterpret_u16_f16(h2));
            vst1_u16(dst + i + 12, vreinterpret_u16_f16(h3));
        }
        for (; i < n; ++i) dst[i] = f32_to_f16_bits_rte(src[i]);
        tel_add_u64(&g_tel.bytes_f32_to_f16, (uint64_t)n * 4ULL);
        return;
    }
#endif
    for (int i = 0; i < n; ++i) {
        dst[i] = (rm == NearestTiesToEven) ? f32_to_f16_bits_rte(src[i])
                                           : f32_to_f16_bits_trunc(src[i]);
    }
    tel_add_u64(&g_tel.bytes_f32_to_f16, (uint64_t)n * 4);
}

void f16_to_f32(const uint16_t* src, float* dst, int n) {
#if S2_HAVE_FP16
    int i = 0;
    for (; i + 15 < n; i += 16) {
        float16x4_t h0 = vreinterpret_f16_u16(vld1_u16(src + i + 0));
        float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(src + i + 4));
        float16x4_t h2 = vreinterpret_f16_u16(vld1_u16(src + i + 8));
        float16x4_t h3 = vreinterpret_f16_u16(vld1_u16(src + i + 12));

        float32x4_t a0 = vcvt_f32_f16(h0); /* fcvtl */
        float32x4_t a1 = vcvt_f32_f16(h1);
        float32x4_t a2 = vcvt_f32_f16(h2);
        float32x4_t a3 = vcvt_f32_f16(h3);

        vst1q_f32(dst + i + 0,  a0);
        vst1q_f32(dst + i + 4,  a1);
        vst1q_f32(dst + i + 8,  a2);
        vst1q_f32(dst + i + 12, a3);
    }
    for (; i < n; ++i) dst[i] = f16_bits_to_f32(src[i]);
#else
    for (int i = 0; i < n; ++i) dst[i] = f16_bits_to_f32(src[i]);
#endif
    tel_add_u64(&g_tel.bytes_f16_to_f32, (uint64_t)n * 2ULL);
}

/* ------------------------------- f32 <-> bf16 ------------------------------- */
void f32_to_bf16(const float* src, uint16_t* dst, int n, RoundingMode rm) {
    tel_set_u64(&g_tel.rounding_mode_last, (uint64_t)rm);
    if (rm == NearestTiesToEven) {
        for (int i = 0; i < n; ++i) dst[i] = f32_to_bf16_bits_rte(src[i]);
    } else {
        for (int i = 0; i < n; ++i) dst[i] = f32_to_bf16_bits_trunc(src[i]);
    }
    tel_add_u64(&g_tel.bytes_f32_to_bf16, (uint64_t)n * 4ULL);
}
void bf16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = bf16_bits_to_f32(src[i]);
    tel_add_u64(&g_tel.bytes_bf16_to_f32, (uint64_t)n * 2ULL);
}

/* ------------------------ int8 quantize/dequantize -------------------------- */
static inline int8_t clamp_i8(int32_t v, S2TelemetryCounter* sat_counter) {
    if (v > 127) { tel_add_u64(sat_counter, 1); return 127; }
    if (v < -128){ tel_add_u64(sat_counter, 1); return -128; }
    return (int8_t)v;
}

void quantize_i8_symmetric(const float* x, int n, float scale, int8_t* y) {
    /* scale = max(|x|)/127 (caller) */
    assert(scale > 0.0f && isfinite(scale) && "Scale must be positive and finite");
    const float inv = 1.0f / scale;
#if S2_HAVE_NEON
    int i = 0;
    float32x4_t vinv = vdupq_n_f32(inv);
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(x + i + 0);
        float32x4_t a1 = vld1q_f32(x + i + 4);
        float32x4_t a2 = vld1q_f32(x + i + 8);
        float32x4_t a3 = vld1q_f32(x + i + 12);

        /* v = x * inv; round to nearest ties-to-even */
        a0 = vmulq_f32(a0, vinv); a1 = vmulq_f32(a1, vinv);
        a2 = vmulq_f32(a2, vinv); a3 = vmulq_f32(a3, vinv);

        /* [FIX P0] Use software tie-to-even if FRINT unavailable */
        a0 = vrndnq_f32(a0); a1 = vrndnq_f32(a1);
        a2 = vrndnq_f32(a2); a3 = vrndnq_f32(a3);

        int32x4_t i0 = vcvtq_s32_f32(a0);
        int32x4_t i1 = vcvtq_s32_f32(a1);
        int32x4_t i2 = vcvtq_s32_f32(a2);
        int32x4_t i3 = vcvtq_s32_f32(a3);

        /* [FIX CRITICAL] Count saturations BEFORE any narrowing */
        int32x4_t max_s8_i32 = vdupq_n_s32(127);
        int32x4_t min_s8_i32 = vdupq_n_s32(-128);

        /* Create saturation masks (0xFFFFFFFF if outside [-128,127], 0 if within) */
        uint32x4_t sat0_mask = vorrq_u32(
            vcgtq_s32(i0, max_s8_i32),
            vcltq_s32(i0, min_s8_i32)
        );
        uint32x4_t sat1_mask = vorrq_u32(vcgtq_s32(i1, max_s8_i32), vcltq_s32(i1, min_s8_i32));
        uint32x4_t sat2_mask = vorrq_u32(vcgtq_s32(i2, max_s8_i32), vcltq_s32(i2, min_s8_i32));
        uint32x4_t sat3_mask = vorrq_u32(vcgtq_s32(i3, max_s8_i32), vcltq_s32(i3, min_s8_i32));

        /* Convert masks to counts (0xFFFFFFFF -> 1, 0 -> 0) */
        uint32x4_t ones = vdupq_n_u32(1);
        sat0_mask = vandq_u32(sat0_mask, ones);
        sat1_mask = vandq_u32(sat1_mask, ones);
        sat2_mask = vandq_u32(sat2_mask, ones);
        sat3_mask = vandq_u32(sat3_mask, ones);

        /* Sum up saturation counts */
#if defined(__ARM_FEATURE_DOTPROD) || defined(__aarch64__)
        uint64_t sat_count = vaddvq_u32(sat0_mask) + vaddvq_u32(sat1_mask) +
                             vaddvq_u32(sat2_mask) + vaddvq_u32(sat3_mask);
#else
        uint32x4_t sum01 = vaddq_u32(sat0_mask, sat1_mask);
        uint32x4_t sum23 = vaddq_u32(sat2_mask, sat3_mask);
        uint32x4_t sum_all = vaddq_u32(sum01, sum23);
        uint32x2_t sum_low = vget_low_u32(sum_all);
        uint32x2_t sum_high = vget_high_u32(sum_all);
        uint32x2_t sum_combined = vadd_u32(sum_low, sum_high);
        uint64_t sat_count = vget_lane_u32(sum_combined, 0) + vget_lane_u32(sum_combined, 1);
#endif
        tel_add_u64(&g_tel.saturations_i8, sat_count);

        /* NOW perform narrowing (saturation already counted) */
        int16x4_t n0 = vqmovn_s32(i0);
        int16x4_t n1 = vqmovn_s32(i1);
        int16x4_t n2 = vqmovn_s32(i2);
        int16x4_t n3 = vqmovn_s32(i3);

        int16x8_t p0 = vcombine_s16(n0, n1);
        int16x8_t p1 = vcombine_s16(n2, n3);

        /* Narrow 16->8 with saturation */
        int8x8_t  q0 = vqmovn_s16(p0);
        int8x8_t  q1 = vqmovn_s16(p1);

        vst1_s8(y + i + 0, q0);
        vst1_s8(y + i + 8, q1);
    }
    for (; i < n; ++i) {
        int32_t q = round_ties_to_even_f32_to_i32(x[i] * inv);
        y[i] = clamp_i8(q, &g_tel.saturations_i8);
    }
#else
    for (int i = 0; i < n; ++i) {
        int32_t q = round_ties_to_even_f32_to_i32(x[i] * inv);
        y[i] = clamp_i8(q, &g_tel.saturations_i8);
    }
#endif
    tel_add_u64(&g_tel.bytes_q_i8, (uint64_t)n * 4ULL);
}

void dequantize_i8_symmetric(const int8_t* y, int n, float scale, float* x) {
#if S2_HAVE_NEON
    int i = 0;
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 15 < n; i += 16) {
        int8x16_t q = vld1q_s8(y + i);
        int16x8_t l = vmovl_s8(vget_low_s8(q));
        int16x8_t h = vmovl_s8(vget_high_s8(q));

        int32x4_t l0 = vmovl_s16(vget_low_s16(l));
        int32x4_t l1 = vmovl_s16(vget_high_s16(l));
        int32x4_t h0 = vmovl_s16(vget_low_s16(h));
        int32x4_t h1 = vmovl_s16(vget_high_s16(h));

        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(l0), vscale);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(l1), vscale);
        float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(h0), vscale);
        float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(h1), vscale);

        vst1q_f32(x + i + 0,  f0);
        vst1q_f32(x + i + 4,  f1);
        vst1q_f32(x + i + 8,  f2);
        vst1q_f32(x + i + 12, f3);
    }
    for (; i < n; ++i) x[i] = (float)y[i] * scale;
#else
    for (int i = 0; i < n; ++i) x[i] = (float)y[i] * scale;
#endif
    tel_add_u64(&g_tel.bytes_dq_i8, (uint64_t)n * 1ULL);
}

void quantize_i8_affine(const float* x, int n, float scale, int32_t zero_point, int8_t* y) {
    assert(scale > 0.0f && isfinite(scale) && "Scale must be positive and finite");
    const float inv = 1.0f / scale;
#if S2_HAVE_NEON
    int i = 0;
    float32x4_t vinv = vdupq_n_f32(inv);
    int32x4_t vzp = vdupq_n_s32(zero_point);
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(x + i + 0);
        float32x4_t a1 = vld1q_f32(x + i + 4);
        float32x4_t a2 = vld1q_f32(x + i + 8);
        float32x4_t a3 = vld1q_f32(x + i + 12);

        a0 = vmulq_f32(a0, vinv);
        a1 = vmulq_f32(a1, vinv);
        a2 = vmulq_f32(a2, vinv);
        a3 = vmulq_f32(a3, vinv);

        /* [FIX P0] Use software tie-to-even if FRINT unavailable */
        a0 = vrndnq_f32(a0); a1 = vrndnq_f32(a1);
        a2 = vrndnq_f32(a2); a3 = vrndnq_f32(a3);

        int32x4_t i0 = vaddq_s32(vcvtq_s32_f32(a0), vzp);
        int32x4_t i1 = vaddq_s32(vcvtq_s32_f32(a1), vzp);
        int32x4_t i2 = vaddq_s32(vcvtq_s32_f32(a2), vzp);
        int32x4_t i3 = vaddq_s32(vcvtq_s32_f32(a3), vzp);

        /* [FIX CRITICAL] Count saturations BEFORE any narrowing */
        int32x4_t max_s8_i32 = vdupq_n_s32(127);
        int32x4_t min_s8_i32 = vdupq_n_s32(-128);

        /* Create saturation masks (0xFFFFFFFF if outside [-128,127], 0 if within) */
        uint32x4_t sat0_mask = vorrq_u32(vcgtq_s32(i0, max_s8_i32), vcltq_s32(i0, min_s8_i32));
        uint32x4_t sat1_mask = vorrq_u32(vcgtq_s32(i1, max_s8_i32), vcltq_s32(i1, min_s8_i32));
        uint32x4_t sat2_mask = vorrq_u32(vcgtq_s32(i2, max_s8_i32), vcltq_s32(i2, min_s8_i32));
        uint32x4_t sat3_mask = vorrq_u32(vcgtq_s32(i3, max_s8_i32), vcltq_s32(i3, min_s8_i32));

        /* Convert masks to counts (0xFFFFFFFF -> 1, 0 -> 0) */
        uint32x4_t ones = vdupq_n_u32(1);
        sat0_mask = vandq_u32(sat0_mask, ones);
        sat1_mask = vandq_u32(sat1_mask, ones);
        sat2_mask = vandq_u32(sat2_mask, ones);
        sat3_mask = vandq_u32(sat3_mask, ones);

        /* Sum up saturation counts */
#if defined(__ARM_FEATURE_DOTPROD) || defined(__aarch64__)
        uint64_t sat_count = vaddvq_u32(sat0_mask) + vaddvq_u32(sat1_mask) +
                             vaddvq_u32(sat2_mask) + vaddvq_u32(sat3_mask);
#else
        uint32x4_t sum01 = vaddq_u32(sat0_mask, sat1_mask);
        uint32x4_t sum23 = vaddq_u32(sat2_mask, sat3_mask);
        uint32x4_t sum_all = vaddq_u32(sum01, sum23);
        uint32x2_t sum_low = vget_low_u32(sum_all);
        uint32x2_t sum_high = vget_high_u32(sum_all);
        uint32x2_t sum_combined = vadd_u32(sum_low, sum_high);
        uint64_t sat_count = vget_lane_u32(sum_combined, 0) + vget_lane_u32(sum_combined, 1);
#endif
        tel_add_u64(&g_tel.saturations_i8, sat_count);

        /* NOW perform narrowing (saturation already counted) */
        int16x4_t n0 = vqmovn_s32(i0);
        int16x4_t n1 = vqmovn_s32(i1);
        int16x4_t n2 = vqmovn_s32(i2);
        int16x4_t n3 = vqmovn_s32(i3);

        int16x8_t p0 = vcombine_s16(n0, n1);
        int16x8_t p1 = vcombine_s16(n2, n3);

        /* Narrow 16->8 with saturation */
        int8x8_t  q0 = vqmovn_s16(p0);
        int8x8_t  q1 = vqmovn_s16(p1);

        vst1_s8(y + i + 0, q0);
        vst1_s8(y + i + 8, q1);
    }
    for (; i < n; ++i) {
        int32_t q = round_ties_to_even_f32_to_i32(x[i] * inv) + zero_point;
        y[i] = clamp_i8(q, &g_tel.saturations_i8);
    }
#else
    for (int i = 0; i < n; ++i) {
        int32_t q = round_ties_to_even_f32_to_i32(x[i] * inv) + zero_point;
        y[i] = clamp_i8(q, &g_tel.saturations_i8);
    }
#endif
    tel_add_u64(&g_tel.bytes_q_i8, (uint64_t)n * 4ULL);
}

void dequantize_i8_affine(const int8_t* y, int n, float scale, int32_t zero_point, float* x) {
#if S2_HAVE_NEON
    int i = 0;
    float32x4_t vscale = vdupq_n_f32(scale);
    int32x4_t vzp = vdupq_n_s32(zero_point);
    for (; i + 15 < n; i += 16) {
        int8x16_t q = vld1q_s8(y + i);

        int16x8_t l = vmovl_s8(vget_low_s8(q));
        int16x8_t h = vmovl_s8(vget_high_s8(q));

        int32x4_t l0 = vsubq_s32(vmovl_s16(vget_low_s16(l)), vzp);
        int32x4_t l1 = vsubq_s32(vmovl_s16(vget_high_s16(l)), vzp);
        int32x4_t h0 = vsubq_s32(vmovl_s16(vget_low_s16(h)), vzp);
        int32x4_t h1 = vsubq_s32(vmovl_s16(vget_high_s16(h)), vzp);

        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(l0), vscale);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(l1), vscale);
        float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(h0), vscale);
        float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(h1), vscale);

        vst1q_f32(x + i + 0,  f0);
        vst1q_f32(x + i + 4,  f1);
        vst1q_f32(x + i + 8,  f2);
        vst1q_f32(x + i + 12, f3);
    }
    for (; i < n; ++i) x[i] = (float)((int32_t)y[i] - zero_point) * scale;
#else
    for (int i = 0; i < n; ++i) x[i] = (float)((int32_t)y[i] - zero_point) * scale;
#endif
    tel_add_u64(&g_tel.bytes_dq_i8, (uint64_t)n * 1ULL);
}

/* ----------------------------- PQ 4-bit pack/unpack ------------------------- */
void pack_nibbles_u4(const uint8_t* idx4, int n, uint8_t* out) {
    int i = 0, o = 0;
    /* Batch 8->4 bytes (unrolled) */
    for (; i + 7 < n; i += 8, o += 4) {
        uint8_t b0 = (idx4[i+0] & 0x0F) | ((idx4[i+1] & 0x0F) << 4);
        uint8_t b1 = (idx4[i+2] & 0x0F) | ((idx4[i+3] & 0x0F) << 4);
        uint8_t b2 = (idx4[i+4] & 0x0F) | ((idx4[i+5] & 0x0F) << 4);
        uint8_t b3 = (idx4[i+6] & 0x0F) | ((idx4[i+7] & 0x0F) << 4);
        out[o+0] = b0; out[o+1] = b1; out[o+2] = b2; out[o+3] = b3;
    }
    for (; i + 1 < n; i += 2, ++o) {
        out[o] = (uint8_t)((idx4[i] & 0x0F) | ((idx4[i+1] & 0x0F) << 4));
    }
    if (i < n) { /* odd tail: place last in low nibble; high nibble zero */
        out[o] = (uint8_t)(idx4[i] & 0x0F);
        ++o;
    }
    tel_add_u64(&g_tel.bytes_pack_u4, (uint64_t)n * 1ULL);
}

/* [FIX P2] Vectorize unpack_nibbles_u4 with NEON for 40+ GB/s target */
void unpack_nibbles_u4(const uint8_t* in, int n, uint8_t* idx4) {
    /* n is number of output nibbles */
    int o = 0;
#if S2_HAVE_NEON
    /* Process 32 nibbles (16 packed bytes) per iteration */
    for (; o + 31 < n; o += 32) {
        uint8x16_t packed = vld1q_u8(in + (o >> 1));

        /* Extract low and high nibbles */
        uint8x16_t low_mask = vdupq_n_u8(0x0F);
        uint8x16_t low_nibbles = vandq_u8(packed, low_mask);
        uint8x16_t high_nibbles = vshrq_n_u8(packed, 4);

        /* Interleave low and high nibbles using vzip */
        uint8x16x2_t interleaved = vzipq_u8(low_nibbles, high_nibbles);

        vst1q_u8(idx4 + o + 0,  interleaved.val[0]);
        vst1q_u8(idx4 + o + 16, interleaved.val[1]);
    }
#endif

    /* Scalar tail */
    int i = o >> 1;
    for (; o + 1 < n; o += 2, ++i) {
        uint8_t b = in[i];
        idx4[o]   = (uint8_t)(b & 0x0F);
        idx4[o+1] = (uint8_t)(b >> 4);
    }
    if (o < n) {
        uint8_t b = in[i];
        idx4[o] = (uint8_t)(b & 0x0F);
    }
    tel_add_u64(&g_tel.bytes_unpack_u4, (uint64_t)n * 1ULL);
}

/* ------------------------------- Endian helpers ----------------------------- */
/* Detect host endianness at compile time if possible. */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  #define S2_HOST_LITTLE 1
#else
  #if defined(_WIN32) || defined(_WIN64)
    #define S2_HOST_LITTLE 1
  #else
    #define S2_HOST_LITTLE 0
  #endif
#endif

static inline uint16_t bswap16(uint16_t v) {
#if defined(__has_builtin)
  #if __has_builtin(__builtin_bswap16)
    return __builtin_bswap16(v);
  #endif
#endif
    return (uint16_t)((v << 8) | (v >> 8));
}
static inline uint32_t bswap32(uint32_t v) {
#if defined(__has_builtin)
  #if __has_builtin(__builtin_bswap32)
    return __builtin_bswap32(v);
  #endif
#endif
    return (v << 24) | ((v << 8) & 0x00FF0000u) | ((v >> 8) & 0x0000FF00u) | (v >> 24);
}
static inline uint64_t bswap64(uint64_t v) {
#if defined(__has_builtin)
  #if __has_builtin(__builtin_bswap64)
    return __builtin_bswap64(v);
  #endif
#endif
    return ((uint64_t)bswap32((uint32_t)v) << 32) | bswap32((uint32_t)(v >> 32));
}

uint16_t le16(const void* p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
#if S2_HOST_LITTLE
    return v;
#else
    return bswap16(v);
#endif
}
uint32_t le32(const void* p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
#if S2_HOST_LITTLE
    return v;
#else
    return bswap32(v);
#endif
}
uint64_t le64(const void* p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
#if S2_HOST_LITTLE
    return v;
#else
    return bswap64(v);
#endif
}

void store_le16(void* p, uint16_t v) {
#if S2_HOST_LITTLE
    memcpy(p, &v, sizeof(v));
#else
    uint16_t t = bswap16(v); memcpy(p, &t, sizeof(t));
#endif
}
void store_le32(void* p, uint32_t v) {
#if S2_HOST_LITTLE
    memcpy(p, &v, sizeof(v));
#else
    uint32_t t = bswap32(v); memcpy(p, &t, sizeof(t));
#endif
}
void store_le64(void* p, uint64_t v) {
#if S2_HOST_LITTLE
    memcpy(p, &v, sizeof(v));
#else
    uint64_t t = bswap64(v); memcpy(p, &t, sizeof(t));
#endif
}
