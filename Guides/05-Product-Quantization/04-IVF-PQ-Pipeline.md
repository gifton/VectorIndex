# IVF-PQ Pipeline

> **Reading time:** 10 minutes
> **Prerequisites:** [ADC and Lookup Tables](./03-ADC-And-LUT.md)

---

## The Concept

**IVF-PQ** combines IVF's space partitioning with PQ's compression:

1. **IVF**: Reduce candidates by searching only nprobe clusters
2. **PQ**: Store compressed vectors, use ADC for fast distance

This achieves both sublinear search AND massive memory reduction.

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IVF-PQ Search Pipeline                           │
│                                                                         │
│  Query q                                                                │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────────────────────────────────┐                           │
│  │ 1. Find nprobe nearest centroids        │  O(nlist × d)             │
│  │    (using full-precision centroids)     │                           │
│  └─────────────────────────────────────────┘                           │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────────────────────────────────┐                           │
│  │ 2. Build LUT for PQ subspaces           │  O(m × k × dsub)          │
│  │    (once per query)                     │                           │
│  └─────────────────────────────────────────┘                           │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────────────────────────────────┐                           │
│  │ 3. ADC scan of candidate lists          │  O(candidates × m)        │
│  │    (lookup-based distance)              │                           │
│  └─────────────────────────────────────────┘                           │
│     │                                                                   │
│     ▼                                                                   │
│  ┌─────────────────────────────────────────┐                           │
│  │ 4. Optional: Exact rerank top-k'        │  O(k' × d)                │
│  │    (using stored full vectors)          │                           │
│  └─────────────────────────────────────────┘                           │
│     │                                                                   │
│     ▼                                                                   │
│  Return top-k results                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Analysis

For 100 million vectors, 512 dimensions:

```
Full vectors:     100M × 512 × 4 = 200 GB
PQ codes (m=64):  100M × 64 = 6.4 GB
IVF centroids:    4096 × 512 × 4 = 8 MB
PQ codebooks:     64 × 256 × 8 × 4 = 512 KB

Total IVF-PQ:     ~6.5 GB (vs 200 GB)
Compression:      31×
```

This fits in a single machine's RAM!

---

## Residual Encoding

For better accuracy, encode **residuals** instead of raw vectors:

```
Standard PQ:
  encode(x)

Residual PQ (in IVF-PQ):
  residual = x - centroid[assigned_list]
  encode(residual)

Why residuals?
  - Removes cluster-level variation
  - Remaining variation is smaller, easier to quantize
  - Typically 5-10% better recall
```

---

## In VectorIndex

```swift
// 📍 See: Sources/VectorIndex/Kernels/IVFAppend.swift

public enum IVFFormat: UInt8 {
    case flat = 0    // Full vectors
    case pq8 = 1     // 8-bit PQ codes
    case pq4 = 2     // 4-bit PQ codes (packed)
}

// 📍 See: Sources/VectorIndex/Operations/Quantization/PQEncode.swift

// Residual encoding for IVF-PQ
public func pq_encode_residual_u8_f32(
    _ x: UnsafePointer<Float>,          // Original vectors
    _ centroids: UnsafePointer<Float>,  // IVF centroids
    _ assignments: UnsafePointer<Int32>, // Which list each vector belongs to
    _ n: Int64,
    _ d: Int32,
    _ m: Int32,
    _ ks: Int32,
    _ codebooks: UnsafePointer<Float>,
    _ codes: UnsafeMutablePointer<UInt8>,
    _ opts: UnsafePointer<PQEncodeOpts>?
)
```

---

## Performance Comparison

| Index Type | Storage (100M × 512D) | Search Latency | Recall |
|------------|----------------------|----------------|--------|
| Flat | 200 GB | 500ms | 100% |
| IVF-Flat | 200 GB | 10ms | 95% |
| HNSW | 300 GB | 1ms | 98% |
| **IVF-PQ** | **6.5 GB** | **5ms** | **90%** |

IVF-PQ trades recall for massive memory savings.

---

## When to Use IVF-PQ

**Ideal for:**
- 100M+ vectors
- Memory-constrained environments
- Throughput-focused (many QPS)
- 85-95% recall is acceptable

**Avoid when:**
- Need >98% recall (use HNSW or IVF-Flat)
- Small datasets (<1M vectors)
- Cannot afford training time

---

## 🔗 VectorCore Connection

The full pipeline uses VectorCore at every stage:

```swift
// 🔗 VectorCore: Complete IVF-PQ search

// 1. Centroid search (VectorCore distances)
for (i, centroid) in centroids.enumerated() {
    centroidDists[i] = distance(query, centroid)  // ← SIMD
}

// 2. LUT construction (VectorCore for subspace distances)
for j in 0..<m {
    for code in 0..<256 {
        lut[j][code] = l2_squared(query_sub[j], codebook[j][code])  // ← SIMD
    }
}

// 3. ADC scan (lookups, not VectorCore-heavy)
for code in candidateCodes {
    dist = Σ lut[j][code[j]]  // Fast table lookups
}

// 4. Optional rerank (VectorCore for exact distances)
for candidate in top_k_prime {
    exact_dist = distance(query, original_vector[candidate])  // ← SIMD
}
```

---

## Key Takeaways

1. **IVF-PQ = IVF + PQ.** Partition space, compress vectors.

2. **31× compression typical.** 200GB → 6.5GB for 100M × 512D.

3. **Residual encoding helps.** Encode x - centroid, not x.

4. **Trade recall for memory.** ~90% recall vs ~98% for HNSW.

5. **Best for very large scale.** 100M+ vectors, memory-constrained.

---

## Next Up

How do we measure and optimize search quality?

**[→ Chapter 6: Performance & Tuning](../06-Performance-And-Tuning/README.md)**
