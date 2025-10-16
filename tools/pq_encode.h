// Copied from user-provided header for alignment verification
#ifndef PQ_ENCODE_H
#define PQ_ENCODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------ Layout options ------------------------------ */
typedef enum {
    PQ_LAYOUT_AOS = 0,         /* codes[i*m + j] */
    PQ_LAYOUT_SOA_BLOCKED = 1, /* requires padded storage, see notes in .c */
    PQ_LAYOUT_INTERLEAVED_BLOCK = 2 /* requires padded storage, see notes */
} PQLayout;

/* ------------------------------ Encode options ------------------------------ */
typedef struct {
    PQLayout layout;            /* default: PQ_LAYOUT_AOS */
    bool     use_dot_trick;     /* default: auto (ks >= 64) */
    bool     precompute_x_norm2;/* default: true if dot-trick */
    int      prefetch_distance; /* default: 8 (vectors ahead) */
    int      num_threads;       /* default: 0 => library decides / single-thread if no OpenMP */
    /* Advanced (optional, compile-time defaults used if 0):
       For non-AoS layouts you typically pad storage. These affect indexing only. */
    int      soa_block_B;       /* block size B for SOA_BLOCKED (default 64) */
    int      interleave_g;      /* group size g for INTERLEAVED_BLOCK (default 8) */
} PQEncodeOpts;

/* --------------------------------- API (u8) --------------------------------- */
void pq_encode_u8_f32(
    const float* x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,
    int ks,                        /* must be 256 */
    const float* codebooks,        /* [m * ks * dsub], row-major by (j,k,idx) */
    uint8_t* codes,                /* output: [n * m] for AoS layout */
    const PQEncodeOpts* opts       /* nullable (use defaults) */
);

/* --------------------------------- API (u4) --------------------------------- */
/* Packed outputs: two 4-bit codes per byte (low nibble first).
   Note: layout is AoS for u4 in this kernel. */
void pq_encode_u4_f32(
    const float* x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,                         /* must be even */
    int ks,                        /* must be 16 */
    const float* codebooks,        /* [m * ks * dsub] */
    uint8_t* codes,                /* output: [n * (m/2)] packed */
    const PQEncodeOpts* opts       /* nullable */
);

/* --------------------------- Residual (IVF-PQ) APIs ------------------------- */
void pq_encode_residual_u8_f32(
    const float* x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,
    int ks,                        /* 256 */
    const float* codebooks,        /* [m * ks * dsub] */
    const float* coarse_centroids, /* [kc * d] */
    const int32_t* assignments,    /* [n] (index into coarse_centroids) */
    uint8_t* codes,                /* [n * m] */
    const PQEncodeOpts* opts
);

void pq_encode_residual_u4_f32(
    const float* x,                /* [n * d] AoS */
    int64_t n,
    int d,
    int m,                         /* even */
    int ks,                        /* 16 */
    const float* codebooks,        /* [m * ks * dsub] */
    const float* coarse_centroids, /* [kc * d] */
    const int32_t* assignments,    /* [n] */
    uint8_t* codes,                /* [n * (m/2)] packed */
    const PQEncodeOpts* opts
);

/* ---------------------------- u4 pack/unpack helpers ------------------------ */
static inline uint8_t pq_pack_u4_pair(uint8_t code0, uint8_t code1) {
    return (uint8_t)((code0 & 0x0F) | ((code1 & 0x0F) << 4));
}
static inline void pq_unpack_u4_pair(uint8_t byte, uint8_t* code0, uint8_t* code1) {
    *code0 = (uint8_t)(byte & 0x0F);
    *code1 = (uint8_t)((byte >> 4) & 0x0F);
}
void pq_pack_u4_bulk(const uint8_t* codes, int m, uint8_t* packed);   /* m must be even */
void pq_unpack_u4_bulk(const uint8_t* packed, int m, uint8_t* codes); /* m must be even */

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* PQ_ENCODE_H */

