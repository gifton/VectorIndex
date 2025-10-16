// CAtomicsShim.h
#pragma once
#include <stdatomic.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Acquire-load a 32-bit unsigned integer
static inline uint32_t atomic_load_u32_acquire(const uint32_t* p) {
    const _Atomic(uint32_t)* ap = (const _Atomic(uint32_t)*)p;
    return atomic_load_explicit(ap, memory_order_acquire);
}

// Release-store a 32-bit unsigned integer
static inline void atomic_store_u32_release(uint32_t* p, uint32_t v) {
    _Atomic(uint32_t)* ap = (_Atomic(uint32_t)*)p;
    atomic_store_explicit(ap, v, memory_order_release);
}

#ifdef __cplusplus
}
#endif

