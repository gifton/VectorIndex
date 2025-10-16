// CPQEncode: C PQ encoder interface (namespaced functions)
#ifndef CPQ_ENCODE_H
#define CPQ_ENCODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Restrict qualifier macro for better aliasing assumptions */
#if !defined(CPQ_RESTRICT)
  #if defined(__clang__) || defined(__GNUC__)
    #define CPQ_RESTRICT __restrict
  #else
    #define CPQ_RESTRICT restrict
  #endif
#endif

/* ------------------------------ Layout options ------------------------------ */
typedef enum {
    PQ_LAYOUT_AOS = 0,         /* codes[i*m + j] */
    PQ_LAYOUT_SOA_BLOCKED = 1, /* requires padded storage */
    PQ_LAYOUT_INTERLEAVED_BLOCK = 2 /* requires padded storage */
} PQLayout;

/* ------------------------------ Encode options ------------------------------ */
typedef struct {
    PQLayout layout;            /* default: PQ_LAYOUT_AOS */
    bool     use_dot_trick;     /* default: auto (ks >= 64) */
    bool     precompute_x_norm2;/* default: true if dot-trick */
    int      prefetch_distance; /* default: 8 (vectors ahead) */
    int      num_threads;       /* default: 0 => library decides */
    int      soa_block_B;       /* block size B for SOA_BLOCKED (default 64) */
    int      interleave_g;      /* group size g for INTERLEAVED_BLOCK (default 8) */
} PQEncodeOpts;

/* --------------------------------- API (u8) --------------------------------- */
void cpq_encode_u8_f32(
    const float* CPQ_RESTRICT x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,
    int ks,                        /* must be 256 */
    const float* CPQ_RESTRICT codebooks,        /* [m * ks * dsub], row-major by (j,k,idx) */
    uint8_t* CPQ_RESTRICT codes,                /* output: [n * m] for AoS layout */
    const PQEncodeOpts* opts       /* nullable (use defaults) */
);

/* u8 encode with precomputed centroid squared norms (AoS layout).
   'centroid_sq' must point to [m * ks] floats laid out as [j * ks + k]. */
void cpq_encode_u8_f32_with_csq(
    const float* CPQ_RESTRICT x,
    int64_t n,
    int d,
    int m,
    int ks,
    const float* CPQ_RESTRICT codebooks,
    const float* CPQ_RESTRICT centroid_sq,
    uint8_t* CPQ_RESTRICT codes,
    const PQEncodeOpts* opts
);

/* --------------------------------- API (u4) --------------------------------- */
void cpq_encode_u4_f32(
    const float* CPQ_RESTRICT x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,                         /* must be even */
    int ks,                        /* must be 16 */
    const float* CPQ_RESTRICT codebooks,        /* [m * ks * dsub] */
    uint8_t* CPQ_RESTRICT codes,                /* output: [n * (m/2)] packed */
    const PQEncodeOpts* opts       /* nullable */
);

/* --------------------------- Residual (IVF-PQ) APIs ------------------------- */
void cpq_encode_residual_u8_f32(
    const float* CPQ_RESTRICT x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,
    int ks,                        /* 256 */
    const float* CPQ_RESTRICT codebooks,        /* [m * ks * dsub] */
    const float* CPQ_RESTRICT coarse_centroids, /* [kc * d] */
    const int32_t* CPQ_RESTRICT assignments,    /* [n] (index into coarse_centroids) */
    uint8_t* CPQ_RESTRICT codes,                /* [n * m] */
    const PQEncodeOpts* opts
);

/* residual u8 encode with precomputed centroid squared norms (AoS layout). */
void cpq_encode_residual_u8_f32_with_csq(
    const float* CPQ_RESTRICT x,
    int64_t n,
    int d,
    int m,
    int ks,
    const float* CPQ_RESTRICT codebooks,
    const float* CPQ_RESTRICT centroid_sq,
    const float* CPQ_RESTRICT coarse_centroids,
    const int32_t* CPQ_RESTRICT assignments,
    uint8_t* CPQ_RESTRICT codes,
    const PQEncodeOpts* opts
);

void cpq_encode_residual_u4_f32(
    const float* CPQ_RESTRICT x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,                         /* even */
    int ks,                        /* 16 */
    const float* CPQ_RESTRICT codebooks,        /* [m * ks * dsub] */
    const float* CPQ_RESTRICT coarse_centroids, /* [kc * d] */
    const int32_t* CPQ_RESTRICT assignments,    /* [n] */
    uint8_t* CPQ_RESTRICT codes,                /* [n * (m/2)] packed */
    const PQEncodeOpts* opts
);

/* ---------------------------- u4 pack/unpack helpers ------------------------ */
void cpq_pack_u4_bulk(const uint8_t* CPQ_RESTRICT codes, int m, uint8_t* CPQ_RESTRICT packed);   /* m must be even */
void cpq_unpack_u4_bulk(const uint8_t* CPQ_RESTRICT packed, int m, uint8_t* CPQ_RESTRICT codes); /* m must be even */

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CPQ_ENCODE_H */
