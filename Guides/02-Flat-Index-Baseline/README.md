# Chapter 2: Flat Index Baseline

> **Sometimes the simplest approach is the best approach.**

Before diving into sophisticated approximate methods, we need to understand the baseline: brute-force exact search. This chapter explores when linear scan is not just acceptable but optimal, and how to implement it efficiently.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Brute-Force Search](./01-Brute-Force-Search.md) | 10 min | The algorithm, complexity, and when it's optimal |
| [2. Top-K Selection](./02-Top-K-Selection.md) | 12 min | Heaps, partial sorts, and early termination |
| [3. When Flat Is Enough](./03-When-Flat-Is-Enough.md) | 8 min | Decision framework for choosing FlatIndex |

---

## The Big Picture

FlatIndex represents the simplest possible vector index:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FlatIndex                                     │
│                                                                         │
│   Storage: Dictionary mapping VectorID → (vector, metadata)             │
│                                                                         │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  "doc_001" → ([0.12, 0.45, ...], {"title": "..."})             │   │
│   │  "doc_002" → ([0.33, -0.21, ...], {"title": "..."})            │   │
│   │  "doc_003" → ([0.78, 0.11, ...], {"title": "..."})             │   │
│   │  ...                                                            │   │
│   │  "doc_n"   → ([0.56, 0.89, ...], {"title": "..."})             │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   Search:  Scan ALL vectors, compute distance, keep top-k              │
│                                                                         │
│   Complexity: O(n × d) per query                                       │
│   Recall: 100% (exact)                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why Study the Baseline?

### Reason 1: It's Often Optimal

For small datasets, FlatIndex beats everything else:

```
Dataset size: 10,000 vectors
Dimension: 512
Target: 10 QPS

FlatIndex latency: ~5ms (achievable with SIMD)
HNSW overhead: Build graph, maintain connections...

FlatIndex wins: Simpler, faster to build, exact results
```

### Reason 2: It's the Correctness Reference

When testing IVF or HNSW, how do you know they're returning good results?

```swift
// Test approximate index against exact baseline
let flatResults = await flat.search(query: q, k: 10, filter: nil)
let hnswResults = await hnsw.search(query: q, k: 10, filter: nil)

let recall = computeRecall(approximate: hnswResults, exact: flatResults)
XCTAssert(recall >= 0.95)
```

### Reason 3: It Teaches Core Patterns

The techniques in FlatIndex—batch processing, parallel search, top-k selection—appear in every other index type.

---

## 🔗 VectorCore Connection

FlatIndex is where VectorCore's optimizations have the most direct impact:

| VectorCore Concept | FlatIndex Application |
|-------------------|----------------------|
| [SIMD distance kernels](../../VectorCore/Guides/02-SIMD-Demystified/README.md) | Every distance computation uses SIMD4 |
| [Contiguous storage](../../VectorCore/Guides/01-Memory-Fundamentals/README.md) | Batch operations benefit from cache locality |
| [Parallel processing](../../VectorCore/Guides/05-Performance-Patterns/README.md) | batchSearch uses TaskGroup for parallelism |

The inner loop of FlatIndex is pure VectorCore:

```swift
// 📍 See: Sources/VectorIndex/FlatIndex.swift:48-66

for (id, (vec, meta)) in vectors {
    let d = distance(query, vec, metric: metric)  // ← VectorCore primitive
    results.append(SearchResult(id: id, score: d))
}
```

---

## Performance Expectations

On Apple Silicon (M1/M2/M3):

| Dataset Size | Dimension | Single Query | Batch (100) |
|-------------|-----------|--------------|-------------|
| 1,000 | 512 | ~0.5ms | ~10ms |
| 10,000 | 512 | ~5ms | ~50ms |
| 100,000 | 512 | ~50ms | ~500ms |
| 1,000,000 | 512 | ~500ms | ~5s |

FlatIndex scales linearly. The question is whether that linear scaling is acceptable for your use case.

---

## Start Here

**[→ Brute-Force Search](./01-Brute-Force-Search.md)**

---

*Chapter 2 of 7 • [← Similarity Search Fundamentals](../01-Similarity-Search-Fundamentals/README.md) | [Next: IVF Index →](../03-IVF-Inverted-File-Index/README.md)*
