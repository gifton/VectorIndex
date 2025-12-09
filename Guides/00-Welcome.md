# VectorIndex Learning Guide

> **From fast vectors to fast search—building on VectorCore foundations.**

Welcome to the VectorIndex Learning Guide. This is the **second volume** in the VSK (Vector Search Kit) educational series, designed to take you from understanding high-performance vector operations to building production-ready similarity search systems.

---

## Prerequisites: VectorCore Foundations

**This guide assumes you have completed the [VectorCore Learning Guide](../../VectorCore/Guides/00-Welcome.md)** or have equivalent knowledge of:

| VectorCore Chapter | Concepts You'll Use Here |
|-------------------|-------------------------|
| [1. Memory Fundamentals](../../VectorCore/Guides/01-Memory-Fundamentals/README.md) | Contiguous storage, cache-friendly access patterns |
| [2. SIMD Demystified](../../VectorCore/Guides/02-SIMD-Demystified/README.md) | Why SIMD4 storage enables fast distance computation |
| [3. Numerical Computing](../../VectorCore/Guides/03-Numerical-Computing/README.md) | Floating-point stability in distance calculations |
| [4. Unsafe Swift](../../VectorCore/Guides/04-Unsafe-Swift/README.md) | Pointer-based kernels for maximum performance |
| [5. Performance Patterns](../../VectorCore/Guides/05-Performance-Patterns/README.md) | Measuring and optimizing hot paths |

If you haven't worked through VectorCore's guides, we strongly recommend doing so first. The concepts build directly on each other:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VectorCore (Volume 1)                            │
│                                                                         │
│   Memory → SIMD → Numerical → Unsafe → Performance → Capstone           │
│                                                                         │
│   "How do I make individual vector operations fast?"                    │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       VectorIndex (Volume 2)                            │
│                                                                         │
│   Similarity → Flat → IVF → HNSW → PQ → Tuning → Capstone               │
│                                                                         │
│   "How do I search millions of vectors efficiently?"                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What You'll Learn

This guide teaches **the algorithms and data structures** that power modern similarity search:

| Chapter | You'll Learn | Why It Matters |
|---------|-------------|----------------|
| [1. Similarity Search Fundamentals](./01-Similarity-Search-Fundamentals/README.md) | Distance metrics, the curse of dimensionality | Understanding the problem we're solving |
| [2. Flat Index Baseline](./02-Flat-Index-Baseline/README.md) | Brute-force search, top-k selection | The simple approach that's sometimes best |
| [3. IVF: Inverted File Index](./03-IVF-Inverted-File-Index/README.md) | Clustering, inverted lists, nprobe | Partitioning space to search less |
| [4. HNSW: Graph-Based Search](./04-HNSW-Graph-Index/README.md) | Hierarchical graphs, greedy navigation | Logarithmic search through graph structure |
| [5. Product Quantization](./05-Product-Quantization/README.md) | Vector compression, codebooks, ADC | Trading precision for memory and speed |
| [6. Performance & Tuning](./06-Performance-And-Tuning/README.md) | Recall@K, QPS, index selection | Measuring what matters, choosing wisely |
| [7. Capstone](./07-Capstone/README.md) | End-to-end system design | Putting it all together |

---

## The Core Problem

VectorCore taught you how to compute a single dot product in ~100 nanoseconds. That's fast—but what if you have **10 million vectors**?

```
One dot product:     ~100 ns
Ten million:         ~1 second

Query rate needed:   100 QPS
Time budget:         10 ms per query
Vectors you can check: 100,000

Gap: 100× too slow for exact search
```

This is where **indexing** comes in. Instead of checking every vector, we build data structures that let us find the *approximate* nearest neighbors by examining only a tiny fraction of the dataset.

---

## The Fundamental Tradeoff

Every index in this guide navigates the same tradeoff:

```
                    ┌─────────────────────────────────────┐
                    │                                     │
        Recall      │    ●  Flat (exact)                 │
        (accuracy)  │                                     │
           ▲        │         ●  HNSW (high recall)      │
           │        │                                     │
           │        │              ●  IVF (balanced)     │
           │        │                                     │
           │        │                   ●  IVF-PQ        │
           │        │                      (compressed)   │
           │        │                                     │
           └────────┴─────────────────────────────────────►
                              Speed / Memory Efficiency
```

- **Flat**: Perfect recall, but O(n) per query
- **HNSW**: Near-perfect recall, O(log n) search, high memory
- **IVF**: Tunable recall via nprobe, moderate memory
- **IVF-PQ**: Good recall, very low memory, highest throughput

Understanding *when* to use each approach is as important as understanding *how* they work.

---

## How to Use This Guide

### The Sequential Path

Each chapter builds on the previous. If you're new to similarity search:

```
Chapter 1 ──→ Chapter 2 ──→ Chapter 3 ──→ Chapter 4 ──→ Chapter 5 ──→ Chapter 6 ──→ Chapter 7
 Fundamentals    Flat        IVF          HNSW          PQ          Tuning       Capstone
```

### The Reference Path

If you're already familiar with ANN concepts:

- **"I need to understand HNSW parameters"** → [Chapter 4](./04-HNSW-Graph-Index/README.md)
- **"How does product quantization work?"** → [Chapter 5](./05-Product-Quantization/README.md)
- **"Which index should I use?"** → [Chapter 6](./06-Performance-And-Tuning/README.md)

### Each Guide Follows This Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  THE CONCEPT                                                │
│  What's the algorithm? Plain English, diagrams.             │
├─────────────────────────────────────────────────────────────┤
│  WHY IT MATTERS                                             │
│  What problem does this solve? When would you use it?       │
├─────────────────────────────────────────────────────────────┤
│  THE ALGORITHM                                              │
│  Step-by-step walkthrough with examples.                    │
├─────────────────────────────────────────────────────────────┤
│  IN VECTORINDEX                                             │
│  Where is this implemented? Links to actual source.         │
├─────────────────────────────────────────────────────────────┤
│  VECTORCORE CONNECTION                                      │
│  How do VectorCore primitives power this?                   │
├─────────────────────────────────────────────────────────────┤
│  KEY TAKEAWAYS                                              │
│  What should stick? The transferable lessons.               │
└─────────────────────────────────────────────────────────────┘
```

---

## VectorIndex Source Locations

Throughout this guide, we reference actual implementation code:

| Topic | File Path |
|-------|-----------|
| Index Protocol | `Sources/VectorIndex/IndexProtocols.swift` |
| Flat Index | `Sources/VectorIndex/FlatIndex.swift` |
| IVF Index | `Sources/VectorIndex/IVFIndex.swift` |
| HNSW Index | `Sources/VectorIndex/HNSWIndex.swift` |
| K-Means Clustering | `Sources/VectorIndex/Kernels/KMeansMiniBatchKernel.swift` |
| HNSW Traversal | `Sources/VectorIndex/Kernels/HNSWTraversal.swift` |
| PQ Encoding | `Sources/VectorIndex/Operations/Quantization/PQEncode.swift` |
| Distance Kernels | `Sources/VectorIndex/Operations/Scoring/L2Sqr.swift` |
| Top-K Selection | `Sources/VectorIndex/Operations/Selection/TopK.swift` |

---

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| `n` | Number of vectors in the index |
| `d` | Dimensionality of vectors |
| `k` | Number of nearest neighbors to return |
| `📍 See:` | Link to VectorIndex source code |
| `🔗 VectorCore:` | Connection to VectorCore concept |
| `⚠️` | Common mistake or pitfall |
| `💡` | Key insight or tip |

---

## Let's Begin

Ready to learn how similarity search actually works?

**[→ Chapter 1: Similarity Search Fundamentals](./01-Similarity-Search-Fundamentals/README.md)**

---

*VectorIndex Learning Guide • Volume 2 of the VSK Educational Series • Dec 2024*
