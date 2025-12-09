# ADC and Lookup Tables

> **Reading time:** 12 minutes
> **Prerequisites:** [Subspace Codebooks](./02-Subspace-Codebooks.md)

---

## The Concept

**Asymmetric Distance Computation (ADC)** computes approximate distances between a full-precision query and PQ-encoded database vectors using precomputed lookup tables.

---

## Why "Asymmetric"?

```
Symmetric:  Both vectors are PQ-encoded
            d(encode(q), encode(x))
            Fast but loses information from query

Asymmetric: Query is full precision, database is encoded
            d(q, decode(encode(x)))
            Slower but more accurate (query not quantized)
```

ADC is the standard for search—keep query precise, compress only the database.

---

## The Lookup Table

For a query q, precompute distances to all centroids in all subspaces:

```
LUT[subspace][code] = ||q_subspace - centroid[subspace][code]||²

Size: m × k × 4 bytes = 64 × 256 × 4 = 64 KB per query
```

### Construction

```swift
// Pseudocode for LUT construction

func buildLUT(query: [Float], codebooks: [[[Float]]]) -> [[Float]] {
    var lut = [[Float]](repeating: [Float](repeating: 0, count: k), count: m)

    for j in 0..<m {
        let q_sub = query[j * dsub ..< (j+1) * dsub]

        for code in 0..<k {
            let centroid = codebooks[j][code]
            lut[j][code] = l2_squared(q_sub, centroid)
        }
    }

    return lut
}
```

---

## ADC Distance Computation

Once LUT is built, distance to any encoded vector is just table lookups:

```
Encoded vector: [c₀, c₁, c₂, ..., c_{m-1}]

ADC distance = LUT[0][c₀] + LUT[1][c₁] + ... + LUT[m-1][c_{m-1}]
             = Σⱼ LUT[j][cⱼ]
```

### Complexity

```
Full distance:  O(d) = O(512) operations
ADC distance:   O(m) = O(64) lookups + additions

Speedup: 8× (for m=64, d=512)
```

---

## Implementation

```swift
// 📍 See: Sources/VectorIndex/Operations/Quantization/ADCScan.swift

@inlinable
public func adc_scan_u8(
    query: UnsafePointer<Float>,
    codes: UnsafePointer<UInt8>,   // [n × m] encoded vectors
    n: Int,
    m: Int,
    lut: UnsafePointer<Float>,     // [m × 256] precomputed
    distances: UnsafeMutablePointer<Float>
) {
    for i in 0..<n {
        var dist: Float = 0

        for j in 0..<m {
            let code = Int(codes[i * m + j])
            dist += lut[j * 256 + code]  // Table lookup
        }

        distances[i] = dist
    }
}
```

---

## SoA Layout for SIMD

With SoA layout, we can process multiple vectors simultaneously:

```
SoA layout: codes[j * n + i] = code for subspace j, vector i

SIMD4 processing:
  Load 4 codes from subspace j: [c0, c1, c2, c3]
  Gather 4 LUT values: [lut[j][c0], lut[j][c1], lut[j][c2], lut[j][c3]]
  Add to 4 accumulators
```

```swift
// SIMD-optimized ADC (pseudocode)

for j in 0..<m {
    let lutRow = lut[j]  // LUT for subspace j

    for i in stride(from: 0, to: n, by: 4) {
        let codes4 = loadSIMD4(codes, offset: j * n + i)

        // Gather: look up 4 values simultaneously
        let dists4 = gather(lutRow, indices: codes4)

        // Accumulate
        accumulators[i..<i+4] += dists4
    }
}
```

---

## Accuracy Analysis

ADC error comes from quantization:

```
True distance:   d(q, x)
ADC distance:    d(q, decode(encode(x)))
               = d(q, x̂)  where x̂ ≈ x

Error: |d(q, x) - d(q, x̂)|

Bound: ||x - x̂|| (reconstruction error)
```

For well-trained codebooks:
- Mean error: 5-10% of true distance
- Variance: Depends on data distribution

---

## 🔗 VectorCore Connection

LUT construction uses VectorCore for subspace distances:

```swift
// 🔗 VectorCore: Building lookup table

for j in 0..<m {
    let q_sub = query[j * dsub ..< (j+1) * dsub]

    for code in 0..<256 {
        // SIMD-accelerated L2² for dsub dimensions
        lut[j][code] = l2_squared(q_sub, codebook[j][code])
    }
}

// This is O(m × k × dsub) = O(d × k) per query
// For d=512, k=256: ~130K FLOPs (fast with SIMD)
```

---

## Key Takeaways

1. **ADC keeps query full-precision.** Only database vectors are quantized.

2. **LUT precomputes distances.** O(m × k) table, built once per query.

3. **Distance is m table lookups.** O(m) vs O(d) for full distance.

4. **SoA layout enables SIMD.** Process multiple vectors simultaneously.

5. **Error bounded by reconstruction.** Good codebooks = good ADC accuracy.

---

## Next Up

How do we combine PQ with IVF for maximum efficiency?

**[→ IVF-PQ Pipeline](./04-IVF-PQ-Pipeline.md)**
