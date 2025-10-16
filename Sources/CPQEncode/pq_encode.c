#include "cpq_encode.h"

#include <string.h>
#include <math.h>
#include <assert.h>

/* --------------------------- Optional OpenMP support ------------------------ */
#if defined(_OPENMP)
  #include <omp.h>
  #ifndef PQ_USE_OPENMP
    #define PQ_USE_OPENMP 1
  #endif
#else
  #define PQ_USE_OPENMP 0
#endif

/* ------------------------------- SIMD (NEON) -------------------------------- */
#if defined(__aarch64__) && defined(__ARM_NEON)
  #include <arm_neon.h>
  #define PQ_HAVE_NEON 1
#else
  #define PQ_HAVE_NEON 0
#endif

/* ------------------------------- Defaults ---------------------------------- */
#ifndef PQ_DEFAULT_PREFETCH_DIST
#define PQ_DEFAULT_PREFETCH_DIST 8
#endif
#ifndef PQ_DEFAULT_TILE_K
/* Tile of centroids per subspace to improve L1 locality (32 works well for ks=256). */
#define PQ_DEFAULT_TILE_K 32
#endif
#ifndef PQ_DEFAULT_SOA_BLOCK_B
#define PQ_DEFAULT_SOA_BLOCK_B 64
#endif
#ifndef PQ_DEFAULT_INTERLEAVE_G
#define PQ_DEFAULT_INTERLEAVE_G 8
#endif

/* ---------------------------- Utility: prefetch ----------------------------- */
#if defined(__GNUC__) || defined(__clang__)
  #define PQ_PREFETCH_R(addr) __builtin_prefetch((addr), 0, 3)
#else
  #define PQ_PREFETCH_R(addr) ((void)0)
#endif

/* ------------------------------ Utility: hadd ------------------------------- */
#if PQ_HAVE_NEON
static inline float haddq_f32(float32x4_t v) {
    #if defined(__aarch64__)
        return vaddvq_f32(v);
    #else
        return vgetq_lane_f32(v,0)+vgetq_lane_f32(v,1)+vgetq_lane_f32(v,2)+vgetq_lane_f32(v,3);
    #endif
}
#endif

/* ------------------------------ Tile chooser ------------------------------- */
static inline int choose_tile_k(int ks, int dsub) {
    /* Aim ~2–4KB working set per tile: tile_k * dsub * 4 bytes ≈ 2048–4096. */
    int t = 32;
    int target_bytes = 3072;
    if (dsub > 0) {
        int by = target_bytes / (dsub * 4);
        if (by >= 8) t = by & ~7; /* multiple of 8 */
        if (t < 16) t = 16;
        if (t > 64) t = 64;
    }
    if (t > ks) t = ks;
    return t;
}

/* ----------------------------- Local min helper ----------------------------- */
static inline void pq_argmin_update(float dist, int k, float* best_dist, int* best_k) {
    /* Deterministic tie-breaking: prefer smaller k on equal distance. */
    if (dist < *best_dist || (dist == *best_dist && k < *best_k)) {
        *best_dist = dist;
        *best_k = k;
    }
}

/* ------------------------------ Distance: L2 -------------------------------- */
static inline float l2_sq_scalar(const float* a, const float* b, int dsub) {
    float acc = 0.0f;
    for (int i = 0; i < dsub; ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

#if PQ_HAVE_NEON
static inline float l2_sq_neon(const float* a, const float* b, int dsub) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    for (; i + 7 < dsub; i += 8) {
        float32x4_t av0 = vld1q_f32(a + i);
        float32x4_t bv0 = vld1q_f32(b + i);
        float32x4_t av1 = vld1q_f32(a + i + 4);
        float32x4_t bv1 = vld1q_f32(b + i + 4);
        float32x4_t d0 = vsubq_f32(av0, bv0);
        float32x4_t d1 = vsubq_f32(av1, bv1);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    float sum = haddq_f32(acc);
    for (; i < dsub; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

static inline float l2_sq(const float* a, const float* b, int dsub) {
#if PQ_HAVE_NEON
    return l2_sq_neon(a, b, dsub);
#else
    return l2_sq_scalar(a, b, dsub);
#endif
}

/* ---------------------- Distance via dot-trick (no norms) ------------------- */
static inline float dist_dp_scalar(const float* x, const float* c, int dsub, float x_norm2) {
    float dot = 0.0f, c2 = 0.0f;
    for (int i = 0; i < dsub; ++i) {
        float ci = c[i];
        dot += x[i] * ci;
        c2  += ci * ci;
    }
    return x_norm2 + c2 - 2.0f * dot;
}

#if PQ_HAVE_NEON
static inline float dist_dp_neon(const float* x, const float* c, int dsub, float x_norm2) {
    int i = 0;
    float32x4_t acc_dot = vdupq_n_f32(0.f);
    float32x4_t acc_c2  = vdupq_n_f32(0.f);
    for (; i + 7 < dsub; i += 8) {
        float32x4_t xv0 = vld1q_f32(x + i);
        float32x4_t cv0 = vld1q_f32(c + i);
        float32x4_t xv1 = vld1q_f32(x + i + 4);
        float32x4_t cv1 = vld1q_f32(c + i + 4);
        acc_dot = vfmaq_f32(acc_dot, xv0, cv0);
        acc_dot = vfmaq_f32(acc_dot, xv1, cv1);
        acc_c2  = vfmaq_f32(acc_c2,  cv0, cv0);
        acc_c2  = vfmaq_f32(acc_c2,  cv1, cv1);
    }
    float dot = haddq_f32(acc_dot);
    float c2  = haddq_f32(acc_c2);
    for (; i < dsub; ++i) {
        float ci = c[i];
        dot += x[i] * ci;
        c2  += ci * ci;
    }
    return x_norm2 + c2 - 2.0f * dot;
}
#endif

static inline float dist_dp(const float* x, const float* c, int dsub, float x_norm2) {
#if PQ_HAVE_NEON
    return dist_dp_neon(x, c, dsub, x_norm2);
#else
    return dist_dp_scalar(x, c, dsub, x_norm2);
#endif
}

/* Dot only (no c^2), for use with precomputed centroid^2 */
#if PQ_HAVE_NEON
static inline float dot_only_neon(const float* x, const float* c, int dsub) {
    int i = 0;
    float32x4_t acc = vdupq_n_f32(0.f);
    for (; i + 7 < dsub; i += 8) {
        float32x4_t xv0 = vld1q_f32(x + i);
        float32x4_t cv0 = vld1q_f32(c + i);
        float32x4_t xv1 = vld1q_f32(x + i + 4);
        float32x4_t cv1 = vld1q_f32(c + i + 4);
        acc = vfmaq_f32(acc, xv0, cv0);
        acc = vfmaq_f32(acc, xv1, cv1);
    }
    float sum = haddq_f32(acc);
    for (; i < dsub; ++i) sum += x[i] * c[i];
    return sum;
}
#endif
static inline float dot_only(const float* x, const float* c, int dsub) {
#if PQ_HAVE_NEON
    return dot_only_neon(x, c, dsub);
#else
    float dot = 0.0f;
    for (int i = 0; i < dsub; ++i) dot += x[i] * c[i];
    return dot;
#endif
}

/* ----------------------- Residual distance (fused L2) ---------------------- */
static inline float l2_sq_residual_scalar(const float* x, const float* coarse,
                                          const float* c, int dsub) {
    float acc = 0.0f;
    for (int i = 0; i < dsub; ++i) {
        float r = (x[i] - coarse[i]) - c[i];
        acc += r * r;
    }
    return acc;
}

#if PQ_HAVE_NEON
static inline float l2_sq_residual_neon(const float* x, const float* coarse,
                                        const float* c, int dsub) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    for (; i + 7 < dsub; i += 8) {
        float32x4_t xv0 = vld1q_f32(x + i);
        float32x4_t gv0 = vld1q_f32(coarse + i);
        float32x4_t cv0 = vld1q_f32(c + i);
        float32x4_t xv1 = vld1q_f32(x + i + 4);
        float32x4_t gv1 = vld1q_f32(coarse + i + 4);
        float32x4_t cv1 = vld1q_f32(c + i + 4);
        float32x4_t r0 = vsubq_f32(vsubq_f32(xv0, gv0), cv0);
        float32x4_t r1 = vsubq_f32(vsubq_f32(xv1, gv1), cv1);
        acc0 = vfmaq_f32(acc0, r0, r0);
        acc1 = vfmaq_f32(acc1, r1, r1);
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    float sum = haddq_f32(acc);
    for (; i < dsub; ++i) {
        float r = (x[i] - coarse[i]) - c[i];
        sum += r * r;
    }
    return sum;
}
#endif

static inline float l2_sq_residual(const float* x, const float* coarse,
                                   const float* c, int dsub) {
#if PQ_HAVE_NEON
    return l2_sq_residual_neon(x, coarse, c, dsub);
#else
    return l2_sq_residual_scalar(x, coarse, c, dsub);
#endif
}

/* -------------- Dot-trick residual variant (optional, no norms) ------------ */
static inline float dist_dp_residual_scalar(const float* x, const float* coarse,
                                            const float* c, int dsub, float r_norm2) {
    float dot = 0.0f, c2 = 0.0f;
    for (int i = 0; i < dsub; ++i) {
        float ri = x[i] - coarse[i];
        float ci = c[i];
        dot += ri * ci;
        c2  += ci * ci;
    }
    return r_norm2 + c2 - 2.0f * dot;
}

/* -------------------------- Layout index computation ------------------------ */
static inline size_t idx_layout_u8(int64_t i, int j,
                                   int64_t n, int m,
                                   PQLayout layout, int B, int g) {
    (void)n; (void)B; (void)g;
    switch (layout) {
        case PQ_LAYOUT_AOS:
            return (size_t)(i * (int64_t)m + j);
        case PQ_LAYOUT_SOA_BLOCKED: {
            int64_t blocks = (n + B - 1) / B;
            return (size_t)(j * blocks * B + (i / B) * B + (i % B));
        }
        case PQ_LAYOUT_INTERLEAVED_BLOCK: {
            return (size_t)((i / g) * (int64_t)m * g + j * g + (i % g));
        }
        default: return (size_t)(i * (int64_t)m + j);
    }
}

/* ---------------------------- Encode (inner loops) -------------------------- */
static inline uint8_t encode_subspace_u8_direct(
    const float* x_sub, const float* cb_j, int ks, int dsub, int tile_k)
{
    int best_k = 0;
    float best_d = l2_sq(x_sub, cb_j + 0 * dsub, dsub);

    for (int t = 0; t < ks; t += tile_k) {
        int kend = (t + tile_k < ks) ? (t + tile_k) : ks;
        if (kend < ks) PQ_PREFETCH_R(cb_j + kend * dsub);
        for (int k = (t == 0 ? 1 : t); k < kend; ++k) {
            float d = l2_sq(x_sub, cb_j + k * dsub, dsub);
            pq_argmin_update(d, k, &best_d, &best_k);
        }
    }
    return (uint8_t)best_k;
}

static inline uint8_t encode_subspace_u8_dot(
    const float* x_sub, const float* cb_j, int ks, int dsub, int tile_k)
{
    float x2 = 0.0f;
#if PQ_HAVE_NEON
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
        for (; i + 7 < dsub; i += 8) {
            float32x4_t xv0 = vld1q_f32(x_sub + i);
            float32x4_t xv1 = vld1q_f32(x_sub + i + 4);
            acc0 = vfmaq_f32(acc0, xv0, xv0);
            acc1 = vfmaq_f32(acc1, xv1, xv1);
        }
        float32x4_t acc = vaddq_f32(acc0, acc1);
        x2 = haddq_f32(acc);
        for (; i < dsub; ++i) x2 += x_sub[i] * x_sub[i];
    }
#else
    for (int i = 0; i < dsub; ++i) x2 += x_sub[i] * x_sub[i];
#endif

    int   best_k = 0;
    float best_d = dist_dp(x_sub, cb_j + 0 * dsub, dsub, x2);

    for (int t = 0; t < ks; t += tile_k) {
        int kend = (t + tile_k < ks) ? (t + tile_k) : ks;
        if (kend < ks) PQ_PREFETCH_R(cb_j + kend * dsub);
        for (int k = (t == 0 ? 1 : t); k < kend; ++k) {
            float d = dist_dp(x_sub, cb_j + k * dsub, dsub, x2);
            pq_argmin_update(d, k, &best_d, &best_k);
        }
    }
    return (uint8_t)best_k;
}

static inline uint8_t encode_subspace_u8_dot_with_csq(
    const float* x_sub, const float* cb_j, const float* csq_j, int ks, int dsub, int tile_k)
{
    float x2 = 0.0f;
#if PQ_HAVE_NEON
    {
        int i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
        for (; i + 7 < dsub; i += 8) {
            float32x4_t xv0 = vld1q_f32(x_sub + i);
            float32x4_t xv1 = vld1q_f32(x_sub + i + 4);
            acc0 = vfmaq_f32(acc0, xv0, xv0);
            acc1 = vfmaq_f32(acc1, xv1, xv1);
        }
        float32x4_t acc = vaddq_f32(acc0, acc1);
        x2 = haddq_f32(acc);
        for (; i < dsub; ++i) x2 += x_sub[i] * x_sub[i];
    }
#else
    for (int i = 0; i < dsub; ++i) x2 += x_sub[i] * x_sub[i];
#endif

    int   best_k = 0;
    float best_d = x2 + csq_j[0] - 2.0f * dot_only(x_sub, cb_j + 0 * dsub, dsub);

    for (int t = 0; t < ks; t += tile_k) {
        int kend = (t + tile_k < ks) ? (t + tile_k) : ks;
        if (kend < ks) PQ_PREFETCH_R(cb_j + kend * dsub);
        for (int k = (t == 0 ? 1 : t); k < kend; ++k) {
            float d = x2 + csq_j[k] - 2.0f * dot_only(x_sub, cb_j + k * dsub, dsub);
            pq_argmin_update(d, k, &best_d, &best_k);
        }
    }
    return (uint8_t)best_k;
}

static inline uint8_t encode_subspace_u8_residual_with_csq(
    const float* x_sub, const float* coarse_sub, const float* cb_j,
    const float* csq_j, int ks, int dsub, int tile_k)
{
    /* Compute r2 = ||x - g||^2 once */
    float r2 = 0.0f;
#if PQ_HAVE_NEON
    {
        int i = 0;
        float32x4_t acc = vdupq_n_f32(0.f);
        for (; i + 7 < dsub; i += 8) {
            float32x4_t xv0 = vld1q_f32(x_sub + i);
            float32x4_t gv0 = vld1q_f32(coarse_sub + i);
            float32x4_t xv1 = vld1q_f32(x_sub + i + 4);
            float32x4_t gv1 = vld1q_f32(coarse_sub + i + 4);
            float32x4_t r0 = vsubq_f32(xv0, gv0);
            float32x4_t r1 = vsubq_f32(xv1, gv1);
            acc = vfmaq_f32(acc, r0, r0);
            acc = vfmaq_f32(acc, r1, r1);
        }
        r2 = haddq_f32(acc);
        for (; i < dsub; ++i) { float ri = x_sub[i] - coarse_sub[i]; r2 += ri * ri; }
    }
#else
    for (int i = 0; i < dsub; ++i) { float ri = x_sub[i] - coarse_sub[i]; r2 += ri * ri; }
#endif

    int   best_k = 0;
    float best_d = r2 + csq_j[0] - 2.0f * /* dot(r, c0) */ ({
        /* compute dot(r,c) on the fly */
        dot_only(x_sub, cb_j + 0 * dsub, dsub) - dot_only(coarse_sub, cb_j + 0 * dsub, dsub);
    });

    for (int t = 0; t < ks; t += tile_k) {
        int kend = (t + tile_k < ks) ? (t + tile_k) : ks;
        if (kend < ks) PQ_PREFETCH_R(cb_j + kend * dsub);
        for (int k = (t == 0 ? 1 : t); k < kend; ++k) {
            /* dot(r, c_k) = dot(x, c_k) - dot(g, c_k) */
            float dot_r_ck = dot_only(x_sub, cb_j + k * dsub, dsub) - dot_only(coarse_sub, cb_j + k * dsub, dsub);
            float d = r2 + csq_j[k] - 2.0f * dot_r_ck;
            pq_argmin_update(d, k, &best_d, &best_k);
        }
    }
    return (uint8_t)best_k;
}

static inline uint8_t encode_subspace_u8_residual(
    const float* x_sub, const float* coarse_sub, const float* cb_j,
    int ks, int dsub, int tile_k, bool use_dot)
{
    int   best_k = 0;
    float best_d;

    if (!use_dot) {
        best_d = l2_sq_residual(x_sub, coarse_sub, cb_j + 0 * dsub, dsub);
        for (int t = 0; t < ks; t += tile_k) {
            int kend = (t + tile_k < ks) ? (t + tile_k) : ks;
            if (kend < ks) PQ_PREFETCH_R(cb_j + kend * dsub);
            for (int k = (t == 0 ? 1 : t); k < kend; ++k) {
                float d = l2_sq_residual(x_sub, coarse_sub, cb_j + k * dsub, dsub);
                pq_argmin_update(d, k, &best_d, &best_k);
            }
        }
        return (uint8_t)best_k;
    }

    float r2 = 0.0f;
    for (int i = 0; i < dsub; ++i) {
        float ri = x_sub[i] - coarse_sub[i];
        r2 += ri * ri;
    }
    best_d = dist_dp_residual_scalar(x_sub, coarse_sub, cb_j + 0 * dsub, dsub, r2);
    for (int t = 0; t < ks; t += tile_k) {
        int kend = (t + tile_k < ks) ? (t + tile_k) : ks;
        if (kend < ks) PQ_PREFETCH_R(cb_j + kend * dsub);
        for (int k = (t == 0 ? 1 : t); k < kend; ++k) {
            float d = dist_dp_residual_scalar(x_sub, coarse_sub, cb_j + k * dsub, dsub, r2);
            pq_argmin_update(d, k, &best_d, &best_k);
        }
    }
    return (uint8_t)best_k;
}

/* ------------------------------ u4 pack helpers ----------------------------- */
void cpq_pack_u4_bulk(const uint8_t* codes, int m, uint8_t* packed) {
    assert((m & 1) == 0);
    int o = 0;
    for (int j = 0; j < m; j += 2) {
        packed[o++] = (uint8_t)((codes[j] & 0x0F) | ((codes[j + 1] & 0x0F) << 4));
    }
}
void cpq_unpack_u4_bulk(const uint8_t* packed, int m, uint8_t* codes) {
    assert((m & 1) == 0);
    int o = 0, j = 0;
    for (; j < m; j += 2, ++o) {
        uint8_t b = packed[o];
        codes[j]   = (uint8_t)(b & 0x0F);
        codes[j+1] = (uint8_t)((b >> 4) & 0x0F);
    }
}

/* ------------------------------- Defaults init ------------------------------ */
static inline void pq_opts_default(PQEncodeOpts* o, int ks) {
    o->layout              = PQ_LAYOUT_AOS;
    o->use_dot_trick       = (ks >= 64); /* auto */
    o->precompute_x_norm2  = o->use_dot_trick;
    o->prefetch_distance   = PQ_DEFAULT_PREFETCH_DIST;
    o->num_threads         = 0;
    o->soa_block_B         = PQ_DEFAULT_SOA_BLOCK_B;
    o->interleave_g        = PQ_DEFAULT_INTERLEAVE_G;
}

/* --------------------------------- u8 encode -------------------------------- */
void cpq_encode_u8_f32(const float* x, int64_t n, int d, int m, int ks,
                      const float* codebooks, uint8_t* codes,
                      const PQEncodeOpts* opts_in)
{
    assert(x && codebooks && codes);
    assert(n >= 0 && d > 0 && m > 0 && (d % m) == 0);
    assert(ks == 256);

    PQEncodeOpts opts;
    if (opts_in) { opts = *opts_in; }
    else         { pq_opts_default(&opts, ks); }

    const int dsub    = d / m;
    const int tile_k  = choose_tile_k(ks, dsub);
    const PQLayout L  = opts.layout;
    const int B       = (opts.soa_block_B > 0) ? opts.soa_block_B : PQ_DEFAULT_SOA_BLOCK_B;
    const int g       = (opts.interleave_g > 0) ? opts.interleave_g : PQ_DEFAULT_INTERLEAVE_G;
    const int pf_dist = (opts.prefetch_distance > 0) ? opts.prefetch_distance : PQ_DEFAULT_PREFETCH_DIST;

    #if PQ_USE_OPENMP
    int nthr = (opts.num_threads > 0) ? opts.num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 256) num_threads(nthr)
    #endif
    for (int64_t i = 0; i < n; ++i) {
        if (pf_dist > 0 && (i + pf_dist) < n) PQ_PREFETCH_R(x + (i + pf_dist) * (int64_t)d);

        const float* xi = x + i * (int64_t)d;
        for (int j = 0; j < m; ++j) {
            const float* x_sub  = xi + (int64_t)j * dsub;
            const float* cb_j   = codebooks + ((int64_t)j * ks) * dsub;

            uint8_t code = opts.use_dot_trick
                         ? encode_subspace_u8_dot(x_sub, cb_j, ks, dsub, tile_k)
                         : encode_subspace_u8_direct(x_sub, cb_j, ks, dsub, tile_k);

            size_t idx = idx_layout_u8(i, j, n, m, L, B, g);
            codes[idx] = code;
        }
    }
}

void cpq_encode_u8_f32_with_csq(const float* x, int64_t n, int d, int m, int ks,
                      const float* codebooks, const float* centroid_sq, uint8_t* codes,
                      const PQEncodeOpts* opts_in)
{
    assert(x && codebooks && centroid_sq && codes);
    assert(n >= 0 && d > 0 && m > 0 && (d % m) == 0);
    assert(ks == 256);

    PQEncodeOpts opts;
    if (opts_in) { opts = *opts_in; }
    else         { pq_opts_default(&opts, ks); }

    const int dsub    = d / m;
    const int tile_k  = choose_tile_k(ks, dsub);
    const PQLayout L  = opts.layout;
    const int B       = (opts.soa_block_B > 0) ? opts.soa_block_B : PQ_DEFAULT_SOA_BLOCK_B;
    const int g       = (opts.interleave_g > 0) ? opts.interleave_g : PQ_DEFAULT_INTERLEAVE_G;
    const int pf_dist = (opts.prefetch_distance > 0) ? opts.prefetch_distance : PQ_DEFAULT_PREFETCH_DIST;

    #if PQ_USE_OPENMP
    int nthr = (opts.num_threads > 0) ? opts.num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 256) num_threads(nthr)
    #endif
    for (int64_t i = 0; i < n; ++i) {
        if (pf_dist > 0 && (i + pf_dist) < n) PQ_PREFETCH_R(x + (i + pf_dist) * (int64_t)d);
        const float* xi = x + i * (int64_t)d;
        for (int j = 0; j < m; ++j) {
            const float* x_sub = xi + (int64_t)j * dsub;
            const float* cb_j  = codebooks + ((int64_t)j * ks) * dsub;
            const float* csq_j = centroid_sq + (int64_t)j * ks;
            uint8_t code = encode_subspace_u8_dot_with_csq(x_sub, cb_j, csq_j, ks, dsub, tile_k);
            size_t idx = idx_layout_u8(i, j, n, m, L, B, g);
            codes[idx] = code;
        }
    }
}

/* --------------------------------- u4 encode -------------------------------- */
void cpq_encode_u4_f32(const float* x, int64_t n, int d, int m, int ks,
                      const float* codebooks, uint8_t* codes,
                      const PQEncodeOpts* opts_in)
{
    assert(x && codebooks && codes);
    assert(n >= 0 && d > 0 && m > 0 && (d % m) == 0);
    assert((m & 1) == 0);
    assert(ks == 16);

    PQEncodeOpts opts;
    if (opts_in) { opts = *opts_in; }
    else         { pq_opts_default(&opts, ks); }
    (void)opts; /* u4: AoS enforced for packed output */

    const int dsub   = d / m;
    const int tile_k = choose_tile_k(ks, dsub);
    const int pf_dist = PQ_DEFAULT_PREFETCH_DIST;

    #if PQ_USE_OPENMP
    int nthr = (opts.num_threads > 0) ? opts.num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 256) num_threads(nthr)
    #endif
    for (int64_t i = 0; i < n; ++i) {
        if (pf_dist > 0 && (i + pf_dist) < n) PQ_PREFETCH_R(x + (i + pf_dist) * (int64_t)d);

        const float* xi = x + i * (int64_t)d;
        uint8_t* out_i  = codes + i * (int64_t)(m >> 1);

        for (int j = 0; j < m; j += 2) {
            const float* x0 = xi + (int64_t)j * dsub;
            const float* x1 = xi + (int64_t)(j + 1) * dsub;

            const float* cb0 = codebooks + ((int64_t)j * ks) * dsub;
            const float* cb1 = codebooks + ((int64_t)(j + 1) * ks) * dsub;

            uint8_t c0 = encode_subspace_u8_direct(x0, cb0, ks, dsub, tile_k);
            uint8_t c1 = encode_subspace_u8_direct(x1, cb1, ks, dsub, tile_k);

            out_i[j >> 1] = (uint8_t)((c0 & 0x0F) | ((c1 & 0x0F) << 4));
        }
    }
}

/* ---------------------------- residual (u8, IVF-PQ) ------------------------- */
void cpq_encode_residual_u8_f32(const float* x, int64_t n, int d, int m, int ks,
                               const float* codebooks,
                               const float* coarse_centroids,
                               const int32_t* assignments,
                               uint8_t* codes,
                               const PQEncodeOpts* opts_in)
{
    assert(x && codebooks && coarse_centroids && assignments && codes);
    assert(n >= 0 && d > 0 && m > 0 && (d % m) == 0);
    assert(ks == 256);

    PQEncodeOpts opts;
    if (opts_in) { opts = *opts_in; }
    else         { pq_opts_default(&opts, ks); }

    const int dsub    = d / m;
    const int tile_k  = choose_tile_k(ks, dsub);
    const PQLayout L  = opts.layout;
    const int B       = (opts.soa_block_B > 0) ? opts.soa_block_B : PQ_DEFAULT_SOA_BLOCK_B;
    const int g       = (opts.interleave_g > 0) ? opts.interleave_g : PQ_DEFAULT_INTERLEAVE_G;
    const int pf_dist = (opts.prefetch_distance > 0) ? opts.prefetch_distance : PQ_DEFAULT_PREFETCH_DIST;

    #if PQ_USE_OPENMP
    int nthr = (opts.num_threads > 0) ? opts.num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 256) num_threads(nthr)
    #endif
    for (int64_t i = 0; i < n; ++i) {
        if (pf_dist > 0 && (i + pf_dist) < n) PQ_PREFETCH_R(x + (i + pf_dist) * (int64_t)d);

        const float* xi   = x + i * (int64_t)d;
        const float* gc   = coarse_centroids + (int64_t)assignments[i] * d;

        for (int j = 0; j < m; ++j) {
            const float* x_sub  = xi + (int64_t)j * dsub;
            const float* g_sub  = gc + (int64_t)j * dsub;
            const float* cb_j   = codebooks + ((int64_t)j * ks) * dsub;

            uint8_t code = encode_subspace_u8_residual(
                x_sub, g_sub, cb_j, ks, dsub, tile_k,
                opts.use_dot_trick
            );

            size_t idx = idx_layout_u8(i, j, n, m, L, B, g);
            codes[idx] = code;
        }
    }
}

void cpq_encode_residual_u8_f32_with_csq(const float* x, int64_t n, int d, int m, int ks,
                               const float* codebooks, const float* centroid_sq,
                               const float* coarse_centroids, const int32_t* assignments,
                               uint8_t* codes, const PQEncodeOpts* opts_in)
{
    assert(x && codebooks && centroid_sq && coarse_centroids && assignments && codes);
    assert(n >= 0 && d > 0 && m > 0 && (d % m) == 0);
    assert(ks == 256);

    PQEncodeOpts opts;
    if (opts_in) { opts = *opts_in; }
    else         { pq_opts_default(&opts, ks); }

    const int dsub    = d / m;
    const int tile_k  = choose_tile_k(ks, dsub);
    const PQLayout L  = opts.layout;
    const int B       = (opts.soa_block_B > 0) ? opts.soa_block_B : PQ_DEFAULT_SOA_BLOCK_B;
    const int g       = (opts.interleave_g > 0) ? opts.interleave_g : PQ_DEFAULT_INTERLEAVE_G;
    const int pf_dist = (opts.prefetch_distance > 0) ? opts.prefetch_distance : PQ_DEFAULT_PREFETCH_DIST;

    #if PQ_USE_OPENMP
    int nthr = (opts.num_threads > 0) ? opts.num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 256) num_threads(nthr)
    #endif
    for (int64_t i = 0; i < n; ++i) {
        if (pf_dist > 0 && (i + pf_dist) < n) PQ_PREFETCH_R(x + (i + pf_dist) * (int64_t)d);

        const float* xi = x + i * (int64_t)d;
        const float* gc = coarse_centroids + (int64_t)assignments[i] * d;
        for (int j = 0; j < m; ++j) {
            const float* x_sub = xi + (int64_t)j * dsub;
            const float* g_sub = gc + (int64_t)j * dsub;
            const float* cb_j  = codebooks + ((int64_t)j * ks) * dsub;
            const float* csq_j = centroid_sq + (int64_t)j * ks;
            uint8_t code = encode_subspace_u8_residual_with_csq(x_sub, g_sub, cb_j, csq_j, ks, dsub, tile_k);
            size_t idx = idx_layout_u8(i, j, n, m, L, B, g);
            codes[idx] = code;
        }
    }
}

/* ---------------------------- residual (u4, IVF-PQ) ------------------------- */
void cpq_encode_residual_u4_f32(const float* x, int64_t n, int d, int m, int ks,
                               const float* codebooks,
                               const float* coarse_centroids,
                               const int32_t* assignments,
                               uint8_t* codes,
                               const PQEncodeOpts* opts_in)
{
    assert(x && codebooks && coarse_centroids && assignments && codes);
    assert(n >= 0 && d > 0 && m > 0 && (d % m) == 0);
    assert((m & 1) == 0);
    assert(ks == 16);

    PQEncodeOpts opts;
    if (opts_in) { opts = *opts_in; }
    else         { pq_opts_default(&opts, ks); }
    (void)opts; /* u4 packed AoS only here */

    const int dsub   = d / m;
    const int tile_k = choose_tile_k(ks, dsub);
    const int pf_dist = PQ_DEFAULT_PREFETCH_DIST;

    #if PQ_USE_OPENMP
    int nthr = (opts.num_threads > 0) ? opts.num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 256) num_threads(nthr)
    #endif
    for (int64_t i = 0; i < n; ++i) {
        if (pf_dist > 0 && (i + pf_dist) < n) PQ_PREFETCH_R(x + (i + pf_dist) * (int64_t)d);

        const float* xi = x + i * (int64_t)d;
        const float* gc = coarse_centroids + (int64_t)assignments[i] * d;
        uint8_t* out_i  = codes + i * (int64_t)(m >> 1);

        for (int j = 0; j < m; j += 2) {
            const float* x0  = xi + (int64_t)j * dsub;
            const float* g0  = gc + (int64_t)j * dsub;
            const float* cb0 = codebooks + ((int64_t)j * ks) * dsub;

            const float* x1  = xi + (int64_t)(j + 1) * dsub;
            const float* g1  = gc + (int64_t)(j + 1) * dsub;
            const float* cb1 = codebooks + ((int64_t)(j + 1) * ks) * dsub;

            uint8_t c0 = encode_subspace_u8_residual(x0, g0, cb0, ks, dsub, tile_k, /*use_dot=*/false);
            uint8_t c1 = encode_subspace_u8_residual(x1, g1, cb1, ks, dsub, tile_k, /*use_dot=*/false);

            out_i[j >> 1] = (uint8_t)((c0 & 0x0F) | ((c1 & 0x0F) << 4));
        }
    }
}
