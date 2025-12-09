# Approximate Nearest Neighbors

> **Reading time:** 8 minutes
> **Prerequisites:** [The Curse of Dimensionality](./02-Curse-Of-Dimensionality.md)

---

## The Concept

**Approximate Nearest Neighbor (ANN)** search trades accuracy for speed:

```
Exact k-NN:   Return the TRUE k closest vectors
              Guarantee: 100% recall
              Cost: O(n) per query

Approximate:  Return k vectors that are PROBABLY among the closest
              Guarantee: High recall (e.g., 95%)
              Cost: O(log n) to O(√n) per query
```

The insight is that for most applications, finding a "good enough" result quickly is more valuable than finding the perfect result slowly.

---

## Defining "Approximate"

### Recall@K

The standard metric for ANN quality:

```
Recall@K = |ANN results ∩ True k-NN| / k
```

**Example:**

```
True 5-NN:   {A, B, C, D, E}
ANN result:  {A, B, C, F, G}

Recall@5 = 3/5 = 60%
```

Three of the true top-5 were found; two were missed.

### The Recall-Speed Tradeoff

Every ANN index has tunable parameters that control this tradeoff:

```
              │
              │            ●  Flat (exact)
    Recall    │          ●
              │        ●
              │      ●    ← "knee" of the curve
              │    ●
              │  ●
              │●
              └────────────────────────────
                         QPS (queries per second)
```

Typical targets:
- **High-precision apps** (medical, legal): 99%+ recall
- **Consumer search** (e-commerce, content): 90-95% recall
- **Recommendations** (engagement-driven): 80-90% recall

---

## The Two Families of ANN

### Family 1: Partitioning Methods

Divide the vector space into regions; search only relevant regions.

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│    ┌─────────┐        ┌─────────┐        ┌─────────┐     │
│    │  ● ●    │        │    ●    │        │ ●   ●   │     │
│    │ ●  ●●   │        │  ●  ●   │        │   ●     │     │
│    │    ●    │        │●    ●   │        │  ●  ●   │     │
│    └─────────┘        └─────────┘        └─────────┘     │
│       Cell 1             Cell 2             Cell 3       │
│                                                           │
│    Query: Find which cell(s) to search                    │
│    Then: Scan only those cells                           │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Examples:** IVF (Inverted File Index), LSH (Locality Sensitive Hashing)

**Tradeoff:** More cells searched → higher recall, lower speed

### Family 2: Graph Methods

Build a navigable graph; traverse toward the query.

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│         ●─────●                                          │
│        ╱│     │╲                                         │
│       ╱ │     │ ╲                                        │
│      ●──●─────●──●                                       │
│       ╲ │     │ ╱                                        │
│        ╲│     │╱                                         │
│         ●─────●                                          │
│                     Query: Start at entry point          │
│                     Navigate: Greedily move closer       │
│                     Stop: When no closer neighbor exists │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Examples:** HNSW (Hierarchical Navigable Small World), NSG (Navigating Spreading-out Graph)

**Tradeoff:** More neighbors explored → higher recall, lower speed

---

## Why Approximation Works

### Reason 1: Embedding Structure

Real embeddings aren't random—they have semantic structure:

```
"Dog" ── "Cat" ── "Pet"
   │        │
"Wolf"   "Tiger"

Semantically similar items cluster together.
Approximation exploits this clustering.
```

### Reason 2: Top-K Robustness

For most applications, the exact ranking doesn't matter:

```
True ranking:     1. Doc_A (0.95)
                  2. Doc_B (0.94)
                  3. Doc_C (0.93)

Approximate:      1. Doc_B (0.94)
                  2. Doc_A (0.95)
                  3. Doc_C (0.93)

User sees: Same three documents, slightly different order
User experience: Identical
```

### Reason 3: Error Tolerance

Applications downstream of search often have their own filtering and ranking:

```
Vector Search → Re-ranking Model → Business Rules → User

The vector search doesn't need to be perfect;
it just needs to find a good candidate set.
```

---

## The ANN Landscape

| Method | Recall | Speed | Memory | Build Time | Updates |
|--------|--------|-------|--------|------------|---------|
| **Flat** | 100% | Slow | Baseline | None | O(1) |
| **IVF** | 90-99% | Medium | Low | Fast | Medium |
| **HNSW** | 95-99.9% | Fast | High | Medium | Hard |
| **IVF-PQ** | 80-95% | Very Fast | Very Low | Slow | Medium |

No single method dominates. Choice depends on:
- Dataset size
- Memory budget
- Query latency requirements
- Recall requirements
- Update frequency

---

## In VectorIndex

VectorIndex provides three index types representing different points on the tradeoff curve:

```swift
// 📍 See: Sources/VectorIndex/IndexProtocols.swift

// Maximum recall, minimum complexity
let flat = FlatIndex(dimension: 768, metric: .cosine)

// Balanced: partitioning-based
let ivf = IVFIndex(dimension: 768, metric: .cosine,
                   config: .init(nlist: 1024, nprobe: 32))

// High recall, graph-based
let hnsw = HNSWIndex(dimension: 768, metric: .cosine,
                     config: .init(m: 32, efConstruction: 200, efSearch: 128))
```

The remaining chapters explore each in detail:
- **Chapter 2**: FlatIndex (the baseline)
- **Chapter 3**: IVFIndex (partitioning)
- **Chapter 4**: HNSWIndex (graph navigation)
- **Chapter 5**: Product Quantization (compression)

---

## 🔗 VectorCore Connection

ANN methods still rely on VectorCore primitives for the "inner loop":

```swift
// 🔗 VectorCore: Every ANN method eventually computes distances
//
// HNSW greedy search:
//   for neighbor in currentNode.neighbors {
//       let dist = distance(query, neighbor.vector)  // ← VectorCore
//       if dist < bestDist { bestDist = dist; best = neighbor }
//   }
//
// IVF cell scanning:
//   for vec in cell.vectors {
//       let dist = distance(query, vec)  // ← VectorCore
//       heap.pushIfBetter(dist, vec.id)
//   }
```

VectorCore optimizations directly speed up ANN. A 2× faster distance kernel means 2× faster search.

---

## Key Takeaways

1. **ANN trades accuracy for speed.** Accept small recall loss for large latency gains.

2. **Recall@K measures quality.** What fraction of true neighbors did we find?

3. **Two main families:** Partitioning (IVF) divides space; graphs (HNSW) navigate structure.

4. **Approximation works because:**
   - Real data has structure (clustering)
   - Exact ranking often doesn't matter
   - Downstream systems tolerate noise

5. **No free lunch.** Different methods excel in different scenarios. Understand your requirements.

---

## Next Up

Now that we understand the fundamental tradeoff, let's start with the simplest approach—and learn when it's actually the best choice:

**[→ Chapter 2: Flat Index Baseline](../02-Flat-Index-Baseline/README.md)**
