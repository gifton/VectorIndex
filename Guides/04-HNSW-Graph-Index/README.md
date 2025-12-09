# Chapter 4: HNSW — Graph-Based Search

> **Navigate the graph, find the neighbors.**

HNSW (Hierarchical Navigable Small World) achieves remarkable recall with logarithmic search complexity. Instead of partitioning space, it builds a navigable graph where greedy traversal quickly converges to the nearest neighbors.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Skip Lists and Layers](./01-Skip-Lists-And-Layers.md) | 12 min | The hierarchical structure |
| [2. Greedy Search](./02-Greedy-Search.md) | 10 min | How navigation works |
| [3. M and efConstruction](./03-M-And-EfConstruction.md) | 12 min | Graph construction parameters |
| [4. efSearch and Recall](./04-EfSearch-And-Recall.md) | 10 min | Query-time accuracy control |
| [5. HNSW in VectorIndex](./05-HNSW-In-VectorIndex.md) | 12 min | Implementation and API |

---

## The Big Picture

HNSW organizes vectors into a multi-layer graph:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HNSW Structure                                │
│                                                                         │
│  Layer 2:   ●───────────────────────────●                              │
│              ↓                           ↓                              │
│  Layer 1:   ●───────●───────────●───────●───────●                      │
│              ↓       ↓           ↓       ↓       ↓                      │
│  Layer 0:   ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●                    │
│                                                                         │
│  Navigation: Start at top layer, descend while getting closer          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Upper layers act as "express lanes" for long-distance navigation. Lower layers provide fine-grained local search.

---

## Why HNSW Works

### The Small-World Property

Real-world networks (social, neural, etc.) exhibit **small-world** properties:
- High clustering (neighbors of neighbors are likely connected)
- Short path lengths (any two nodes are ~log(n) hops apart)

HNSW artificially constructs a graph with these properties:

```
Random graph:           Small-world graph:
   ●───●                    ●───●
  ╱│╲ ╱│╲                   │╲ ╱│
 ● │ ● │ ●                  ● ╳ ●
  ╲│╱ ╲│╱                   │╱ ╲│
   ●───●                    ●───●

 Many random               Short paths +
 long edges                local clusters
```

### The Hierarchy Enables O(log n)

Without layers: O(√n) expected hops (like skip list without hierarchy)

With layers: O(log n) expected hops (like skip list)

```
Layer 0: 1,000,000 nodes  (fine-grained)
Layer 1:    10,000 nodes  (coarser)
Layer 2:       100 nodes  (coarsest)

Start at Layer 2: Quick coarse positioning
Descend to Layer 0: Fine-grained refinement
```

---

## The Tradeoffs

### Advantages

| Property | HNSW | IVF |
|----------|------|-----|
| Recall at fixed latency | Higher | Lower |
| Search complexity | O(log n × ef) | O(nprobe × n/nlist) |
| No training phase | ✓ (incremental) | ✗ (needs k-means) |
| Query-time tuning | efSearch | nprobe |

### Disadvantages

| Property | HNSW | IVF |
|----------|------|-----|
| Memory overhead | High (~2× vectors) | Low |
| Deletion support | Hard (tombstones) | Easy |
| Build time | O(n log n) | O(n) after training |
| Parallelization | Hard (graph updates) | Easy (independent cells) |

---

## When to Use HNSW

**Good fit:**
- High recall requirements (95%+)
- Moderate memory budget
- Relatively static data
- Need fast query-time tuning

**Less suitable:**
- Very tight memory constraints
- Heavy insert/delete workloads
- Need exact results
- Very large scale (>100M vectors on single node)

---

## 🔗 VectorCore Connection

HNSW uses VectorCore throughout:

| Operation | VectorCore Usage |
|-----------|------------------|
| Distance computation | SIMD-accelerated for each edge traversal |
| Neighbor selection | Batch distance computation |
| Graph construction | Distance to all candidates |

```swift
// 🔗 VectorCore: Every edge traversal computes distance

while searchNotComplete {
    for neighbor in currentNode.neighbors {
        let dist = distance(query, neighbor.vector)  // ← VectorCore
        if dist < bestSoFar {
            candidates.add(neighbor)
        }
    }
}
```

---

## Parameters Overview

| Parameter | Purpose | Typical Range |
|-----------|---------|---------------|
| `M` | Max connections per node | 8-64 |
| `efConstruction` | Build-time search width | 100-400 |
| `efSearch` | Query-time search width | 10-200 |

Higher values = better quality, higher cost. Chapters 3-4 explore these in detail.

---

## Start Here

**[→ Skip Lists and Layers](./01-Skip-Lists-And-Layers.md)**

---

*Chapter 4 of 7 • [← IVF Index](../03-IVF-Inverted-File-Index/README.md) | [Next: Product Quantization →](../05-Product-Quantization/README.md)*
