# Compression Intuition

> **Reading time:** 10 minutes
> **Prerequisites:** [Chapter 5 Introduction](./README.md)

---

## The Concept

**Product Quantization** exploits the fact that high-dimensional vectors have redundant structure. Instead of storing every float, we store indices into a learned dictionary.

---

## The Memory Problem

Consider a 10-million vector dataset:

```
10M vectors × 512 dimensions × 4 bytes = 20 GB

That's just the vectors—index structures add more!

Problem: Doesn't fit in RAM on most machines
Solution: Compress vectors to ~1/32 of original size
```

---

## Quantization Basics

**Scalar quantization**: Replace each float with a smaller representation

```
Float32: 4 bytes, ~7 decimal digits precision
Int8:    1 byte, 256 discrete values

Naive approach: Quantize each dimension to int8
  Compression: 4× (4 bytes → 1 byte)
  Quality: Poor for semantic similarity
```

**Vector quantization**: Replace entire vector with nearest centroid

```
Cluster all vectors into k groups
Store just the cluster index

For k = 65536 (16 bits):
  Compression: 1024× (2048 bytes → 2 bytes)
  Quality: Very poor (only 65K possible vectors)
```

**Product quantization**: The sweet spot

```
Split vector into subspaces
Vector-quantize each subspace independently

For m=64 subspaces, k=256 centroids each:
  Compression: 32× (2048 bytes → 64 bytes)
  Quality: Good (256^64 ≈ 10^154 possible vectors)
```

---

## The Key Insight

The "product" in Product Quantization refers to the Cartesian product:

```
Total possible encoded vectors = k^m

For k=256, m=64:
  256^64 = 2^512 ≈ 10^154 different vectors

This is astronomically large—far more than the number of atoms
in the universe (~10^80). PQ can represent essentially any vector.
```

Compare to simple vector quantization:

```
VQ with 65536 centroids: 65536 possible vectors
PQ with m=64, k=256:     10^154 possible vectors

Same storage (2 bytes for VQ, 64 bytes for PQ)
Vastly more representational power
```

---

## Subspace Independence

PQ assumes subspaces are somewhat independent:

```
Good case (embedding structure):
  Dimensions 0-7:   Encode "topic A" features
  Dimensions 8-15:  Encode "topic B" features
  ...
  Each subspace captures different aspects

Bad case (correlated):
  All 512 dimensions: Encode the same thing
  Splitting loses important relationships
```

For most learned embeddings, subspace independence holds reasonably well.

---

## Visualization

```
Original 512D vector:

[█ █ ░ █ ░ ░ █ ░ │ ░ █ █ ░ █ ░ █ │ ... │ █ ░ ░ █ ░ █ █ ░]
 ←── subspace 0 ──→ ←── subspace 1 ──→     ←── subspace 63 ──→

Each subspace (8 dims) is quantized to one of 256 centroids:

Subspace 0: nearest centroid = 42
Subspace 1: nearest centroid = 17
...
Subspace 63: nearest centroid = 203

Encoded: [42, 17, ..., 203]  (64 bytes)
```

---

## Reconstruction Error

Encoding is lossy—the reconstructed vector differs from original:

```
Original:     [0.12, 0.45, -0.33, 0.78, ...]
Encoded:      [42, 17, ...]
Reconstructed: [0.14, 0.43, -0.31, 0.81, ...]  (from codebook lookup)

Error: ||original - reconstructed||

Typical reconstruction error: 5-15% of original magnitude
```

---

## Impact on Search

Distance between query and PQ-encoded vector is approximate:

```
Exact distance:  d(query, original_vector)
PQ distance:     d(query, reconstructed_vector)

Error: |exact - PQ| / exact ≈ 5-15%

This causes recall loss:
  Some true neighbors have overestimated distances → miss them
  Some non-neighbors have underestimated distances → rank too high
```

---

## Compression Ratio Trade-off

| m | dsub | Bytes/Vector | Compression | Typical Recall |
|---|------|--------------|-------------|----------------|
| 8 | 64 | 8 | 256× | 70-80% |
| 16 | 32 | 16 | 128× | 80-85% |
| 32 | 16 | 32 | 64× | 85-90% |
| 64 | 8 | 64 | 32× | 90-95% |
| 128 | 4 | 128 | 16× | 95-98% |

More subspaces = more bytes = better quality.

---

## In VectorIndex

```swift
// 📍 See: Sources/VectorIndex/Operations/Quantization/PQEncode.swift

public enum PQCodeLayout: UInt8 {
    case aOS = 0  // [vec0_sub0, vec0_sub1, ..., vec1_sub0, ...]
    case sOA = 1  // [sub0_vec0, sub0_vec1, ..., sub1_vec0, ...]
}

@frozen
public struct PQEncodeOpts {
    public var useDotTrick: Bool = true
    public var outputLayout: PQCodeLayout = .aOS
    public var centroidSqNorms: UnsafePointer<Float>?
}
```

---

## Key Takeaways

1. **PQ compresses by dictionary lookup.** Store indices, not floats.

2. **Product structure enables huge codebook.** k^m possible vectors with only m bytes.

3. **Subspace independence assumed.** Works well for learned embeddings.

4. **Lossy compression.** ~5-15% reconstruction error is typical.

5. **More subspaces = better quality.** Trade storage for accuracy.

---

## Next Up

How do we train these codebooks?

**[→ Subspace Codebooks](./02-Subspace-Codebooks.md)**
