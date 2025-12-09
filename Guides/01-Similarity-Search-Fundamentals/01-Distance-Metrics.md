# Distance Metrics

> **Reading time:** 12 minutes
> **Prerequisites:** [VectorCore Chapter 3: Numerical Computing](../../../VectorCore/Guides/03-Numerical-Computing/README.md)

---

## The Concept

A **distance metric** (or similarity measure) defines what "close" means for vectors. Different metrics capture different notions of similarity:

```
Vector A: [1, 0, 0]
Vector B: [0, 1, 0]
Vector C: [2, 0, 0]

L2 Distance:
  d(A, B) = √2 ≈ 1.41    (perpendicular)
  d(A, C) = 1.0          (same direction, different magnitude)

Cosine Similarity:
  cos(A, B) = 0          (perpendicular = dissimilar)
  cos(A, C) = 1          (same direction = identical)
```

The choice of metric fundamentally changes which vectors are considered "nearest."

---

## The Three Core Metrics

### 1. Euclidean Distance (L2)

The straight-line distance between two points:

```
d(a, b) = √(Σᵢ (aᵢ - bᵢ)²)
```

**Geometric interpretation:**

```
        b = (3, 4)
        ●
       /│
      / │ Δy = 3
     /  │
    ●───┘
   a = (1, 1)
     Δx = 2

d(a, b) = √(2² + 3²) = √13 ≈ 3.6
```

**When to use:**
- Pixel-level image similarity
- Physical measurements (coordinates, sensor data)
- When magnitude matters

**VectorIndex implementation:**

```swift
// 📍 See: Sources/VectorIndex/Operations/Scoring/L2Sqr.swift
//
// We often compute L2² (squared) to avoid the sqrt:
// - Preserves ordering (if d₁ < d₂, then d₁² < d₂²)
// - Faster computation
// - Take sqrt only for final results if needed
```

### 2. Cosine Similarity

The cosine of the angle between vectors:

```
cos(a, b) = (a · b) / (||a|| × ||b||)
```

**Geometric interpretation:**

```
              ↗ a
             /
            / θ
           /
          ●─────────→ b

cos(θ) = 1   when θ = 0°   (same direction)
cos(θ) = 0   when θ = 90°  (perpendicular)
cos(θ) = -1  when θ = 180° (opposite)
```

**When to use:**
- Text embeddings (semantic similarity)
- Document comparison
- When direction matters, not magnitude

**Converting to distance:**
Cosine is a *similarity* (higher = more similar). To use it with nearest-neighbor search:

```swift
// Option 1: Cosine distance
cosine_distance = 1 - cosine_similarity

// Option 2: Angular distance
angular_distance = arccos(cosine_similarity) / π
```

**VectorIndex implementation:**

```swift
// 📍 See: Sources/VectorIndex/Operations/Scoring/Cosine.swift
//
// We precompute inverse norms for efficiency:
// cos(a, b) = (a · b) × invNorm(a) × invNorm(b)
//
// This avoids computing ||a|| per query when searching.
```

### 3. Inner Product (Dot Product)

The raw dot product without normalization:

```
ip(a, b) = Σᵢ (aᵢ × bᵢ)
```

**Geometric interpretation:**

```
ip(a, b) = ||a|| × ||b|| × cos(θ)
         = (magnitude of a) × (projection of b onto a)
```

**When to use:**
- Maximum Inner Product Search (MIPS)
- Recommendation systems (user-item affinity)
- When pre-normalized embeddings are used

**⚠️ Important:** Inner product is a *similarity*, not a distance. Larger = more similar. VectorIndex handles this by negating scores internally for consistent "lower is better" ordering.

---

## Comparison Table

| Metric | Formula | Range | Handles Magnitude? | Common Use |
|--------|---------|-------|-------------------|------------|
| L2 | √Σ(aᵢ-bᵢ)² | [0, ∞) | Yes | Images, coordinates |
| Cosine | (a·b)/(‖a‖‖b‖) | [-1, 1] | No (normalized) | Text, embeddings |
| Dot Product | Σ(aᵢ×bᵢ) | (-∞, ∞) | Yes | Recommendations, MIPS |

---

## Why This Matters

### Wrong Metric = Wrong Results

Consider searching for semantically similar sentences:

```
Query:     "The quick brown fox" → [0.5, 0.3, 0.8, ...]
Document A: "A fast auburn fox"  → [0.4, 0.25, 0.7, ...]  (similar meaning)
Document B: "Foxes are mammals"  → [1.0, 0.6, 1.6, ...]  (same direction, 2× magnitude)
```

With **L2 distance**:
- d(Query, A) ≈ 0.18
- d(Query, B) ≈ 0.85

With **Cosine similarity**:
- cos(Query, A) ≈ 0.99
- cos(Query, B) ≈ 0.99 (identical direction!)

If embeddings aren't normalized, L2 might rank the semantically similar document correctly by accident, while cosine would tie them. The right choice depends on how your embeddings were trained.

### Pre-Normalized Embeddings

Many embedding models output L2-normalized vectors (||v|| = 1). In this case:

```
Cosine similarity = Dot product
L2² = 2 - 2 × (dot product)
```

This means for normalized vectors, all three metrics give equivalent rankings! You can use whichever is fastest (usually dot product).

---

## In VectorIndex

VectorIndex supports multiple metrics through `SupportedDistanceMetric`:

```swift
// 📍 See: Sources/VectorIndex/IndexProtocols.swift

public enum SupportedDistanceMetric: String, Codable, Sendable {
    case euclidean    // L2 distance
    case dotProduct   // Inner product (negated for distance ordering)
    case cosine       // 1 - cosine similarity
    case manhattan    // L1 distance (FlatIndex only)
    case chebyshev    // L∞ distance (FlatIndex only)
}
```

Each index type validates which metrics it supports:

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:36

private static let supportedMetrics: Set<SupportedDistanceMetric> =
    [.euclidean, .dotProduct, .cosine]
```

---

## 🔗 VectorCore Connection

Distance computation is where VectorCore's SIMD optimizations shine:

```swift
// 🔗 VectorCore: SIMD4 storage enables 4-wide distance computation
//
// Instead of:
//   for i in 0..<d { sum += (a[i] - b[i])² }
//
// We process 4 dimensions at once:
//   let diff = a_simd4 - b_simd4
//   let sq = diff * diff
//   sum += sq  // 4 partial sums in one SIMD4
```

For a 512-dimensional vector:
- Scalar: 512 subtractions + 512 multiplications + 511 additions
- SIMD4: 128 SIMD subtractions + 128 SIMD multiplications + ~128 additions

That's a **4× reduction** in operations.

---

## Numerical Stability

🔗 **VectorCore Connection:** Recall from [Numerical Computing](../../../VectorCore/Guides/03-Numerical-Computing/README.md) that floating-point math can overflow.

**Problem with L2:**

```swift
let a: [Float] = [1e20, 1e20, ...]
let b: [Float] = [0, 0, ...]

// Naive L2²:
let diff = a[0] - b[0]  // 1e20
let sq = diff * diff     // 1e40 → OVERFLOW (Float max ≈ 3.4e38)
```

**Solution:** Compute in double precision or use scaled algorithms:

```swift
// 📍 See: Sources/VectorIndex/Operations/Scoring/L2SqrKernel.swift
//
// VectorIndex uses Float32 for storage but can accumulate in Float64
// for long vectors, or use two-pass scaling like VectorCore's normalize.
```

---

## Key Takeaways

1. **L2 distance** measures straight-line distance. Use when magnitude matters (images, coordinates).

2. **Cosine similarity** measures directional alignment. Use for text embeddings and semantic search.

3. **Dot product** combines magnitude and direction. Use for recommendations and pre-normalized vectors.

4. **For normalized vectors, all three are equivalent.** Choose the fastest (usually dot product).

5. **Wrong metric = wrong results.** Match your metric to how your embeddings were trained.

---

## Next Up

Now that you understand *what* we're measuring, let's explore why measuring everything is fundamentally hard:

**[→ The Curse of Dimensionality](./02-Curse-Of-Dimensionality.md)**
