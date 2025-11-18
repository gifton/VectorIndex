ID: S2 — RNG & Dtype Helpers (MUST)

Purpose
- Provide fast, deterministic RNG primitives for training/seeding and robust dtype conversion/packing utilities used across kernels.

Scope
- RNG: counter‑based and xoroshiro streams, splittable per thread/task, utilities for K‑means++ (#11), mini‑batch K‑means (#12), PQ training (#19), and sampling.
- Dtype: f32↔f16/bf16 casts, int8 quantize/dequant, PQ 4‑bit pack/unpack, endian helpers for serialization, and alignment macros.

RNG
- Streams
  - `Philox4x32-10` (counter‑based): stateless function of `(key, counter)`, great for reproducible parallelism.
  - `xoroshiro128**` (stateful): very fast per‑thread stream for inner‑loop randoms.
- API
  - `typedef struct { uint64_t s0, s1; } Xoro128;`
  - `void xoro128_init(Xoro128* x, uint64_t seed, uint64_t stream_id);`
  - `uint64_t xoro128_next_u64(Xoro128* x);`
  - `uint32_t xoro128_next_u32(Xoro128* x);`
  - `float xoro128_next_uniform(Xoro128* x);` // [0,1)
  - `double xoro128_next_uniform_f64(Xoro128* x);`
  - `void philox_key(uint64_t seed, uint64_t stream_id, uint64_t* k0, uint64_t* k1);`
  - `void philox_next4(uint64_t k0, uint64_t k1, uint64_t ctr_lo, uint64_t ctr_hi, uint32_t out[4]);` // counter→4x32
- Utilities
  - `void rng_split(uint64_t seed, int worker_id, int task_id, uint64_t* out_seed, uint64_t* out_stream);` // stable derivation
  - `void rng_skip_ahead_xoro(Xoro128* x, uint64_t n);` and counter bump for Philox via `(ctr + n)`.
  - `void randperm_inplace(uint32_t* a, int n, Xoro128* x);` // Fisher‑Yates
  - `void sample_without_replacement(uint32_t n, uint32_t k, uint32_t* out, Xoro128* x);` // Algorithm S/Reservoir
  - `void gaussian_box_muller(float* out, int n, Xoro128* x);`
- K‑means++ helpers (#11/#19)
  - Weighted choice by distances: `int weighted_pick(const float* w, int n, Xoro128* x);` (stable alias method optional P1)
  - Subsampling for mini‑batch: `int subsample_indices(uint32_t n, uint32_t m, uint32_t* out, Xoro128* x);`
- Determinism
  - Define that given `(seed, stream_id, worker_id)`, outcomes are identical across runs, chips, and thread counts for the same partitioning.

Dtype Helpers
- f32 ↔ f16/bf16
  - API: `void f32_to_f16(const float* src, uint16_t* dst, int n, RoundingMode rm);`
         `void f16_to_f32(const uint16_t* src, float* dst, int n);`
         `void f32_to_bf16(const float* src, uint16_t* dst, int n, RoundingMode rm);`
         `void bf16_to_f32(const uint16_t* src, float* dst, int n);`
  - Rounding: `NearestTiesToEven` (default), `TowardZero`. Saturation: N/A for floats (pass through NaN/Inf); preserve sign of zero.
  - Vectorization: use NEON `fcvtn`/`fcvtl` where available; process 16 floats per iteration; ensure 64‑B alignment.
- Quantize to int8 (symmetric/asymmetric)
  - API: `void quantize_i8_symmetric(const float* x, int n, float scale, int8_t* y);` where `scale = max(|x|)/127` (caller may precompute per‑tensor/subvector).
         `void dequantize_i8_symmetric(const int8_t* y, int n, float scale, float* x);`
         `void quantize_i8_affine(const float* x, int n, float scale, int32_t zero_point, int8_t* y);` and inverse.
  - Behavior: clamp to [-128,127]; rounding nearest ties‑to‑even; vectorize with NEON `vqmovn`‑style narrowing.
- PQ 4‑bit pack/unpack
  - API: `void pack_nibbles_u4(const uint8_t* idx4 /*[n]*/, int n, uint8_t* out /*[n/2]*/);`
         `void unpack_nibbles_u4(const uint8_t* in /*[n/2]*/, int n, uint8_t* idx4 /*[n]*/);`
  - Order: low nibble is first code, high nibble is second; batch 8→4 bytes for throughput; branchless bit ops.
- Endian helpers (serialization)
  - API: `uint16_t le16(const void* p); uint32_t le32(const void* p); uint64_t le64(const void* p);` and `void store_le16(void* p, uint16_t v); ...`
  - Implement via byte‑swaps when host endianness != little; used by S1.
- Alignment & stride
  - Macros: `ALIGN_UP(x, a)`, `IS_ALIGNED(ptr, a)`, `PAD_TO(vec_len, multiple)`; used across kernels to satisfy 64‑B alignment and d%16 padding.

Performance Targets
- f32↔f16: ≥30 GB/s on M2/M3 for aligned buffers; i8 quantize: ≥20 GB/s with vectorized clamp; u4 pack/unpack: ≥40 GB/s on L1‑resident data.

Telemetry (#46)
- Optional counters: bytes converted, rounding mode, saturation counts (i8), pack/unpack throughput.

Tests
- Converters: round‑trip f32→f16→f32 and bf16 within expected epsilon; NaN payload preserved (don’t canonicalize); sign of zero preserved.
- Quantize: reference vs scalar for random and adversarial distributions; saturation counts correct.
- RNG: reproducibility across runs/threads for fixed seeds; uniformity tests (Chi‑square) on large samples; correctness of permutations and sampling without replacement.

Integration
- RNG used in #11/#12/#19; dtype in #09 (inv norms), #20/#22 (PQ pack/unpack), S1 (endianness), and throughout for alignment/padding.
<!-- moved to docs/kernel-specs/ -->
