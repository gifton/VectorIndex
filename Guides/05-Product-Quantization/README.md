# Chapter 5: Product Quantization

> **Compress vectors, search faster.**

Product Quantization (PQ) is a lossy compression technique that can reduce vector storage by 10-50× while maintaining reasonable search quality. This chapter explores how PQ works and when to use it.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Compression Intuition](./01-Compression-Intuition.md) | 10 min | Why and how we compress vectors |
| [2. Subspace Codebooks](./02-Subspace-Codebooks.md) | 15 min | Splitting dimensions, training codebooks |
| [3. ADC and Lookup Tables](./03-ADC-And-LUT.md) | 12 min | Fast distance approximation |
| [4. IVF-PQ Pipeline](./04-IVF-PQ-Pipeline.md) | 10 min | Combining IVF with PQ for scale |

---

## The Big Picture

Instead of storing 512 floats (2KB), store 64 bytes that approximate the original:

```
Original vector (512 × 4 bytes = 2048 bytes):
┌────────────────────────────────────────────────────────────────────────┐
│ 0.12 │ 0.45 │ -0.33 │ ... (512 floats) ... │ 0.78 │ -0.21 │ 0.56 │   │
└────────────────────────────────────────────────────────────────────────┘

PQ-encoded (64 × 1 byte = 64 bytes):
┌──────────────────────────────────────────────────────────────────┐
│ 42 │ 17 │ 203 │ 88 │ ... (64 code indices) ... │ 156 │ 91 │ 12 │
└──────────────────────────────────────────────────────────────────┘
     ↓    ↓    ↓
    Look up in codebook to approximate original subvector
```

**Compression ratio:** 2048 / 64 = **32×**

---

## How PQ Works

### Step 1: Split Into Subspaces

Divide the d-dimensional vector into m subvectors:

```
d = 512, m = 64 → dsub = 8

Original: [dim0, dim1, ..., dim511]

Subspace 0: [dim0, dim1, ..., dim7]
Subspace 1: [dim8, dim9, ..., dim15]
...
Subspace 63: [dim504, dim505, ..., dim511]
```

### Step 2: Train Codebooks

For each subspace, cluster training vectors and keep k centroids (typically k=256):

```
Subspace 0 codebook:          Subspace 1 codebook:
  Code 0: [0.12, -0.45, ...]    Code 0: [0.78, 0.33, ...]
  Code 1: [0.33, 0.21, ...]     Code 1: [-0.12, 0.56, ...]
  ...                            ...
  Code 255: [-0.67, 0.89, ...]  Code 255: [0.11, -0.44, ...]
```

### Step 3: Encode

Replace each subvector with the index of its nearest centroid:

```
Vector subspace 0: [0.11, -0.43, ...]
Nearest centroid: Code 42
Encoded value: 42 (1 byte)
```

---

## The Tradeoff

| Aspect | Full Vectors | PQ Encoded |
|--------|-------------|------------|
| Storage | 2KB per vector | 64 bytes per vector |
| Distance accuracy | Exact | Approximate |
| Distance speed | O(d) | O(m) with LUT |
| Training needed | No | Yes (k-means per subspace) |

---

## When to Use PQ

**Good fit:**
- Very large datasets (100M+ vectors)
- Memory is the bottleneck
- Can tolerate ~90-95% recall
- Combined with IVF (IVF-PQ)

**Less suitable:**
- High recall requirements (>99%)
- Small datasets (full vectors fit in RAM)
- Frequently changing data (retraining is expensive)

---

## 🔗 VectorCore Connection

PQ leverages VectorCore for:

| Operation | VectorCore Usage |
|-----------|------------------|
| Codebook training | k-means uses SIMD distances |
| Encoding | Nearest centroid search |
| ADC distance | Lookup table construction |

```swift
// 🔗 VectorCore: Encoding uses distance to find nearest centroid

for subspace in 0..<m {
    let subvec = vector[subspace * dsub ..< (subspace + 1) * dsub]
    var bestCode = 0
    var bestDist = Float.infinity

    for code in 0..<256 {
        let centroid = codebook[subspace][code]
        let d = distance(subvec, centroid)  // ← VectorCore
        if d < bestDist {
            bestDist = d
            bestCode = code
        }
    }

    encoded[subspace] = UInt8(bestCode)
}
```

---

## VectorIndex PQ Implementation

```swift
// 📍 See: Sources/VectorIndex/Operations/Quantization/PQEncode.swift

public func pq_encode_u8_f32(
    _ x: UnsafePointer<Float>,     // Input vectors [n × d]
    _ n: Int64,
    _ d: Int32,
    _ m: Int32,                     // Number of subspaces
    _ ks: Int32,                    // Codebook size (256)
    _ codebooks: UnsafePointer<Float>,  // [m × ks × dsub]
    _ codes: UnsafeMutablePointer<UInt8>, // Output [n × m]
    _ opts: UnsafePointer<PQEncodeOpts>?
)
```

---

## Start Here

**[→ Compression Intuition](./01-Compression-Intuition.md)**

---

*Chapter 5 of 7 • [← HNSW Graph Index](../04-HNSW-Graph-Index/README.md) | [Next: Performance & Tuning →](../06-Performance-And-Tuning/README.md)*
