//  include/hnsw_traversal.h
//  HNSW traversal (Greedy + efSearch) — C ABI per spec #33

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { METRIC_L2 = 0, METRIC_IP = 1, METRIC_COSINE = 2 } HNSWMetric;

// Greedy descent from maxLevel→1; returns entry node for layer 0 (or -1 on error).
int32_t hnsw_greedy_descent_f32(
    const float* q, int d,
    int32_t entryPoint, int32_t maxLevel,
    const int32_t* const* offsetsPerLayer,   // per-layer [N+1]
    const int32_t* const* neighborsPerLayer, // per-layer [EL]
    const float* xb, int32_t N,
    HNSWMetric metric,
    const float* optionalInvNorms // for cosine; length N or NULL
);

// efSearch at layer 0; writes up to ef candidate ids/dists sorted by distance asc.
// Returns the number of candidates written (≤ ef), or <0 on error.
int hnsw_efsearch_f32(
    const float* q, int d,
    int32_t enterL0,
    const int32_t* offsetsL0, const int32_t* neighborsL0,
    const float* xb, int32_t N,
    int ef, HNSWMetric metric,
    const uint64_t* allowBitset /*optional*/, int allowN /*domain size or 0*/,
    int32_t* idsOut, float* distsOut
);

// Convenience: greedy + efSearch. Returns written count, or <0 on error.
int hnsw_traverse_f32(
    const float* q, int d,
    int32_t entryPoint, int32_t maxLevel,
    const int32_t* const* offsetsPerLayer, const int32_t* const* neighborsPerLayer,
    const float* xb, int32_t N, int ef, HNSWMetric metric,
    const uint64_t* allowBitset /*optional*/, int allowN /*0 if none*/,
    int32_t* idsOut, float* distsOut
);

#ifdef __cplusplus
}
#endif

