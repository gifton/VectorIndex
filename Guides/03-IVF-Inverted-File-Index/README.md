# Chapter 3: IVF — Inverted File Index

> **Divide the space, conquer the search.**

IVF (Inverted File Index) is the workhorse of large-scale vector search. The idea is simple: partition vectors into groups, then search only the relevant groups. This chapter explores how clustering enables sublinear search.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Clustering and Centroids](./01-Clustering-And-Centroids.md) | 15 min | K-means algorithm, Voronoi cells, training |
| [2. Inverted Lists](./02-Inverted-Lists.md) | 10 min | How vectors are assigned and stored |
| [3. The nprobe Tradeoff](./03-Nprobe-Tradeoff.md) | 12 min | Balancing recall and latency |
| [4. IVF in VectorIndex](./04-IVF-In-VectorIndex.md) | 10 min | Implementation details and API |

---

## The Big Picture

IVF partitions the vector space using k-means clustering:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Vector Space                                     │
│                                                                         │
│    ┌───────────┐      ┌───────────┐      ┌───────────┐                 │
│    │  ● ●      │      │    ★      │      │      ●    │                 │
│    │ ●  ★      │      │  ● ● ●    │      │  ★  ●     │                 │
│    │    ●      │      │●    ●     │      │ ●    ●    │                 │
│    │  ●   ●    │      │  ●        │      │   ● ●     │                 │
│    └───────────┘      └───────────┘      └───────────┘                 │
│       Cell 0             Cell 1             Cell 2                      │
│                                                                         │
│    ★ = Centroid (cluster center)                                       │
│    ● = Vectors assigned to that cell                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

At query time:
1. Find the `nprobe` nearest centroids to the query
2. Search only the vectors in those cells
3. Return top-k from the combined results

---

## Why IVF Works

### The Math

```
Full scan:    Check n vectors
IVF scan:     Check n × (nprobe / nlist) vectors

Example:
  n = 1,000,000 vectors
  nlist = 1,000 centroids
  nprobe = 8 lists

  Vectors checked: 1,000,000 × 8/1,000 = 8,000
  Speedup: 125×
```

### The Intuition

Similar vectors cluster together. If your query is about "machine learning," you don't need to check cells containing "cooking recipes":

```
Query: "How does backpropagation work?"

Nearest centroids:
  1. ML/Neural Networks cell     ← search this
  2. Optimization/Calculus cell  ← search this
  3. Computer Science cell       ← search this
  ...
  Far centroids:
  998. Culinary Arts cell        ← skip
  999. Fashion Design cell       ← skip
  1000. Sports cell              ← skip
```

---

## The Tradeoff

IVF introduces a recall-speed tradeoff controlled by `nprobe`:

```
nprobe = 1:   Very fast, low recall (~50-70%)
              Only check the single nearest cell

nprobe = nlist/10: Balanced (~90-95% recall)
                   Check 10% of cells

nprobe = nlist: Maximum recall (100%)
                Equivalent to flat search (defeats the purpose)
```

The optimal `nprobe` depends on:
- Your recall requirements
- The clustering quality
- The distribution of your data

---

## 🔗 VectorCore Connection

IVF leverages VectorCore at multiple levels:

| Operation | VectorCore Technique |
|-----------|---------------------|
| Centroid distances | SIMD batch distance to find nearest centroids |
| Cell scanning | SIMD distance for candidate vectors |
| K-means training | SIMD distance for cluster assignment |

```swift
// 🔗 VectorCore: K-means assignment step

// For each vector, find nearest centroid:
for vec in dataset {
    var bestCentroid = 0
    var bestDist = Float.infinity

    for (i, centroid) in centroids.enumerated() {
        let d = distance(vec, centroid)  // ← SIMD-accelerated
        if d < bestDist {
            bestDist = d
            bestCentroid = i
        }
    }

    assignments[vec] = bestCentroid
}
```

---

## IVF Variants

VectorIndex implements the base IVF algorithm. Common variants include:

| Variant | Description |
|---------|-------------|
| **IVF-Flat** | Stores raw vectors in cells (what VectorIndex uses) |
| **IVF-PQ** | Compresses vectors with Product Quantization (Chapter 5) |
| **IVF-SQ** | Scalar quantization (8-bit vectors) |
| **IVF-HNSW** | Uses HNSW to find nearest centroids |

---

## When to Use IVF

**Good fit:**
- Large datasets (100k - 100M vectors)
- Moderate recall requirements (85-95%)
- Need to balance speed and accuracy
- Data can be meaningfully clustered

**Less suitable:**
- Very high recall requirements (>99%)
- Streaming data (frequent inserts)
- Very small datasets (FlatIndex is better)
- Data with no natural clusters

---

## Start Here

**[→ Clustering and Centroids](./01-Clustering-And-Centroids.md)**

---

*Chapter 3 of 7 • [← Flat Index Baseline](../02-Flat-Index-Baseline/README.md) | [Next: HNSW Graph Index →](../04-HNSW-Graph-Index/README.md)*
