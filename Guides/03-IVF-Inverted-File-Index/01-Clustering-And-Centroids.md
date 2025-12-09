# Clustering and Centroids

> **Reading time:** 15 minutes
> **Prerequisites:** [Chapter 2: Flat Index Baseline](../02-Flat-Index-Baseline/README.md)

---

## The Concept

**Clustering** partitions vectors into groups where similar vectors belong to the same group. The **centroid** is the "center" of each group—a representative vector that summarizes its members.

```
Before clustering:                After clustering:

    ●  ●                             ┌─────────┐
  ●   ●  ●                           │ ●  ●    │
    ●  ●                             │●   ●  ● │
                     ─────────►      │  ★  ●   │ ← Cell 0, centroid ★
  ●                                  │ ●  ●    │
    ●   ●  ●                         └─────────┘
  ●    ●
    ●                                ┌─────────┐
                                     │●        │
                                     │  ●   ●  │
                                     │ ★   ●   │ ← Cell 1, centroid ★
                                     │ ●    ●  │
                                     └─────────┘
```

The key insight: **if a query is near a centroid, it's likely near vectors in that cell.**

---

## K-Means Algorithm

The standard algorithm for IVF clustering is **k-means** (specifically, Lloyd's algorithm):

```
Algorithm: Lloyd's K-Means

Input:  vectors X = {x₁, ..., xₙ}
        k (number of clusters)
        max_iterations

1. Initialize k centroids (see below)

2. Repeat until convergence:

   a. Assignment step:
      For each vector xᵢ:
        Assign xᵢ to the cluster with nearest centroid

   b. Update step:
      For each cluster j:
        centroid_j = mean of all vectors assigned to j

3. Return centroids and assignments
```

### Convergence

K-means converges when assignments stop changing:

```
Iteration 1:  Assignments: [0,0,1,1,2,0,1,2,...]
Iteration 2:  Assignments: [0,0,1,1,2,0,2,2,...]  (changed)
Iteration 3:  Assignments: [0,0,1,1,2,0,2,2,...]  (same → converged)
```

Or when a maximum iteration count is reached.

---

## Initialization Strategies

Centroid initialization significantly impacts clustering quality.

### Random Initialization

Pick k random vectors as initial centroids:

```swift
var centroids: [[Float]] = []
for _ in 0..<k {
    let randomIndex = Int.random(in: 0..<n)
    centroids.append(data[randomIndex])
}
```

**Problem:** Can pick outliers or clustered points, leading to poor convergence.

### K-Means++ Initialization

Pick centroids that are spread out using distance-weighted sampling:

```
Algorithm: K-Means++

1. Pick first centroid uniformly at random

2. For each remaining centroid:
   a. Compute distance d(x) from each point x to nearest existing centroid
   b. Sample next centroid with probability ∝ d(x)²
   c. (Points far from existing centroids are more likely to be chosen)

3. Return k centroids
```

This produces well-spread initial centroids:

```
Random init:                    K-means++ init:

    ★                              ★
    ★  ★                                   ★
      ★                                         ★

  (all in one region)           (spread across space)
```

---

## In VectorIndex

VectorIndex uses k-means++ and mini-batch k-means for efficiency:

```swift
// 📍 See: Sources/VectorIndex/Kernels/KMeansSeeding.swift:167

// K-means++ initialization (C-style API for performance)
public func kmeansPlusPlusSeed(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    k: Int,
    config: KMeansSeedConfig,
    centroidsOut: UnsafeMutablePointer<Float>,
    chosenIndicesOut: UnsafeMutablePointer<Int32>?
) throws -> KMeansSeedStats
```

```swift
// 📍 See: Sources/VectorIndex/Kernels/KMeansMiniBatchKernel.swift:424

// Mini-batch k-means for large datasets
public func kmeans_minibatch_f32(
    x: UnsafePointer<Float>,    // [n × d] row-major
    n: Int64,
    d: Int,
    kc: Int,                     // number of centroids
    initCentroids: UnsafePointer<Float>,
    cfg: KMeansMBConfig,
    centroidsOut: UnsafeMutablePointer<Float>,
    assignOut: UnsafeMutablePointer<Int32>?,
    statsOut: UnsafeMutablePointer<KMeansMBStats>?
) -> KMeansStatus
```

### Mini-Batch K-Means

For large datasets, updating centroids using all points is expensive. Mini-batch k-means uses random samples:

```
Standard K-Means:           Mini-Batch K-Means:
─────────────────           ──────────────────
Per iteration:              Per iteration:
  - Look at ALL n points    - Sample batch of b points
  - O(n × k × d)            - O(b × k × d)

Total: O(n × k × d × iter)  Total: O(b × k × d × iter)

For n=1M, b=1024, iter=20:
  Standard: 20M × k ops     Mini-batch: 20k × k ops
  (1000× faster)
```

---

## Voronoi Cells

The regions defined by centroids are called **Voronoi cells**:

```
Each point belongs to the cell of its nearest centroid:

    ╱ ╲           ╱
   ╱   ╲    ★    ╱
  ╱  ★  ╲       ╱
 ╱       ╲     ╱
──────────╲───╱──────
           ╲ ╱
            ╳
           ╱ ╲
          ╱   ╲
         ╱  ★  ╲
        ╱       ╲

★ = Centroid
Lines = Cell boundaries (equidistant from neighboring centroids)
```

In high dimensions, these cells become **high-dimensional polytopes** (not easy to visualize, but the math works the same).

---

## Choosing K (nlist)

How many clusters should you create?

### Rule of Thumb

```
nlist ≈ √n  (for balanced cells)

Examples:
  n = 10,000   → nlist ≈ 100
  n = 100,000  → nlist ≈ 316
  n = 1,000,000 → nlist ≈ 1,000
```

### Considerations

**Too few clusters (nlist too small):**
- Each cell has many vectors
- Must search many vectors per probe
- Lower speedup

**Too many clusters (nlist too large):**
- Each cell has few vectors
- Centroid comparison becomes expensive
- Some cells may be empty

**Practical range:**
```
nlist ∈ [4 × √n, 16 × √n]

For n = 1M:
  Range: 4,000 to 16,000 clusters
  Common choice: 4,096 or 8,192 (powers of 2)
```

---

## 🔗 VectorCore Connection

K-means is distance-heavy—VectorCore optimizations directly apply:

```swift
// 🔗 VectorCore: Assignment step uses batch distances

// For each vector, find nearest centroid
// This is exactly like brute-force search over centroids

for vec in batch {
    var bestCentroid = 0
    var bestDist = Float.infinity

    for (i, centroid) in centroids.enumerated() {
        // SIMD-accelerated distance from VectorCore
        let d = distance(vec, centroid, metric: .euclidean)
        if d < bestDist {
            bestDist = d
            bestCentroid = i
        }
    }

    assignments.append(bestCentroid)
}
```

The update step (computing means) also benefits from SIMD for summing vectors.

---

## Training Time Expectations

K-means clustering takes time. Plan accordingly:

```
Dataset Size    nlist    Approximate Training Time
─────────────────────────────────────────────────
   100,000       256          1-2 seconds
   500,000      1024          5-10 seconds
 1,000,000      4096          30-60 seconds
10,000,000      8192          5-10 minutes
```

**Training is a one-time cost** — do it once when building the index, then search many times.

---

## Key Takeaways

1. **K-means partitions vectors into clusters.** Each cluster has a centroid (center point).

2. **K-means++ initialization matters.** Well-spread initial centroids converge faster and better.

3. **Mini-batch k-means scales.** Sample-based updates enable million-vector datasets.

4. **nlist ≈ √n is a starting point.** Adjust based on your recall/speed requirements.

5. **Training is one-time.** Invest time upfront for fast searches later.

---

## Next Up

Now that we understand clustering, let's see how vectors are stored and retrieved:

**[→ Inverted Lists](./02-Inverted-Lists.md)**
