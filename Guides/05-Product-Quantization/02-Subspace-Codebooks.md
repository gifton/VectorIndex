# Subspace Codebooks

> **Reading time:** 15 minutes
> **Prerequisites:** [Compression Intuition](./01-Compression-Intuition.md)

---

## The Concept

A **codebook** is a dictionary of representative vectors for a subspace. Training learns these representatives from data.

---

## Codebook Structure

For m subspaces with k centroids each:

```
Codebook[subspace][code] = centroid vector of length dsub

Total codebook size: m × k × dsub floats

Example (m=64, k=256, dsub=8):
  64 × 256 × 8 × 4 bytes = 512 KB
```

This is small compared to the vectors themselves—worth keeping in memory!

---

## Training Algorithm

Each subspace uses independent k-means:

```
Algorithm: Train PQ Codebooks

Input: training_vectors [n × d], m subspaces, k centroids

For subspace j in 0..<m:
    1. Extract subvectors: subdata = training_vectors[:, j*dsub : (j+1)*dsub]
    2. Run k-means on subdata with k clusters
    3. Store centroids as codebook[j]

Return codebook [m × k × dsub]
```

### Training Data Requirements

```
Rule of thumb: At least 1000 × k training samples

For k=256:
  Minimum: 256,000 training vectors
  Recommended: 1,000,000+ for stable codebooks

If training data is limited:
  - Use fewer subspaces (larger dsub)
  - Use smaller k (e.g., k=64)
  - Sample with replacement
```

---

## Encoding Process

```swift
// 📍 See: Sources/VectorIndex/Operations/Quantization/PQEncode.swift:66-124

@inlinable
public func pq_encode_u8_f32(
    _ x: UnsafePointer<Float>,          // [n × d] input vectors
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,
    _ ks32: Int32,                       // k=256 for u8
    _ codebooks: UnsafePointer<Float>,   // [m × ks × dsub]
    _ codes: UnsafeMutablePointer<UInt8>, // [n × m] output
    _ optsPtr: UnsafePointer<PQEncodeOpts>?
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    let dsub = d / m

    // Precompute centroid squared norms for dot-product trick
    let centroidSq = ensureCentroidSqNorms(...)

    for i in 0..<n {
        let xRow = x + i * d
        for j in 0..<m {
            // Find nearest centroid using dot-product trick
            let code = argminCode_u8(
                xSub: xRow + j * dsub,
                codebook_j: codebooks + j * ks * dsub,
                centroidSq_j: centroidSq + j * ks,  // Precomputed ||c||²
                ks: ks,
                dsub: dsub,
                useDot: opts.useDotTrick
            )
            codes[i * m + j] = UInt8(code)
        }
    }
}
```

---

## The Dot-Product Trick

Naive encoding computes L2 distance to each centroid. The dot-product trick is faster:

```
||x - c||² = ||x||² + ||c||² - 2⟨x, c⟩

Since ||c||² is constant (precomputed), we only need:
  - ||x||² (compute once per subvector)
  - ⟨x, c⟩ (dot product, faster than full L2)
```

```swift
// 📍 See: Sources/VectorIndex/Operations/Quantization/PQEncode.swift:35-45

@frozen
public struct PQEncodeOpts {
    /// Use dot-product trick when ks is large (default: true)
    public var useDotTrick: Bool = true

    /// Precomputed centroid squared norms [m × ks]
    public var centroidSqNorms: UnsafePointer<Float>?
}
```

---

## Code Layout Options

Two layouts for encoded vectors:

### AoS (Array of Structures)
```
codes[i * m + j] = code for vector i, subspace j

Memory: [v0s0, v0s1, ..., v0sm, v1s0, v1s1, ...]

Good for: Encoding, decoding single vectors
```

### SoA (Structure of Arrays)
```
codes[j * n + i] = code for subspace j, vector i

Memory: [s0v0, s0v1, ..., s0vn, s1v0, s1v1, ...]

Good for: ADC scan (batch distance computation)
```

---

## 🔗 VectorCore Connection

Codebook training uses VectorCore's k-means:

```swift
// 🔗 VectorCore: K-means for each subspace

for subspace in 0..<m {
    let subdata = extractSubspace(training_data, subspace, dsub)

    // VectorCore's SIMD-accelerated k-means
    let centroids = kmeans(subdata, k: 256, maxIterations: 20)

    codebook[subspace] = centroids
}
```

Encoding uses SIMD distance for nearest centroid:

```swift
// 🔗 VectorCore: Distance to 256 centroids

for code in 0..<256 {
    let d = distance(subvector, centroid[code])  // ← SIMD
    if d < bestDist { bestDist = d; bestCode = code }
}
```

---

## Key Takeaways

1. **One codebook per subspace.** Each has k centroids of dimension dsub.

2. **Training is independent k-means.** Run separately for each subspace.

3. **Need sufficient training data.** At least 1000×k samples recommended.

4. **Dot-product trick speeds encoding.** Avoid redundant computation.

5. **SoA layout for search, AoS for access.** Choose based on use case.

---

## Next Up

How do we compute distances with encoded vectors?

**[→ ADC and Lookup Tables](./03-ADC-And-LUT.md)**
