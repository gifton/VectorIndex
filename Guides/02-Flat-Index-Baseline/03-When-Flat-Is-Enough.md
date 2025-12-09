# When Flat Is Enough

> **Reading time:** 8 minutes
> **Prerequisites:** [Top-K Selection](./02-Top-K-Selection.md)

---

## The Concept

FlatIndex should be your **default choice** until proven insufficient. It's simpler to understand, simpler to debug, and guarantees exact results.

This guide provides a decision framework for when to stick with FlatIndex versus when to graduate to approximate indices.

---

## Decision Framework

### The Quick Checklist

```
Use FlatIndex when ANY of these are true:

□ Dataset < 50,000 vectors
□ Query latency budget > 50ms
□ 100% recall is required
□ Dataset changes frequently (many inserts/deletes)
□ Prototype or development phase
□ Memory is very constrained (no room for graph overhead)
```

### The Detailed Decision Tree

```
                          START
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Dataset size < 10,000?     │
              └─────────────────────────────┘
                     │              │
                    Yes             No
                     │              │
                     ▼              ▼
              ┌──────────┐   ┌─────────────────────────────┐
              │ FlatIndex │   │  Latency budget > 100ms?   │
              └──────────┘   └─────────────────────────────┘
                                   │              │
                                  Yes             No
                                   │              │
                                   ▼              ▼
                            ┌──────────┐   ┌─────────────────────────────┐
                            │ FlatIndex │   │  100% recall required?     │
                            └──────────┘   └─────────────────────────────┘
                                                 │              │
                                                Yes             No
                                                 │              │
                                                 ▼              ▼
                                          ┌──────────┐   ┌─────────────────┐
                                          │ FlatIndex │   │  Use ANN Index  │
                                          └──────────┘   │  (IVF or HNSW)  │
                                                         └─────────────────┘
```

---

## Case Studies

### Case 1: Small Product Catalog

```
Scenario:
  - E-commerce site with 5,000 products
  - Product embeddings from CLIP (512-dim)
  - Visual similarity search
  - 10 QPS expected

Analysis:
  FlatIndex latency: ~2-3ms per query
  Capacity: 10 QPS × 3ms = 3% CPU utilization

Decision: FlatIndex ✓
  - Well under performance requirements
  - Simple to maintain
  - Exact results for product matching
```

### Case 2: Document Search for Legal Discovery

```
Scenario:
  - 100,000 legal documents
  - Sentence embeddings (768-dim)
  - Must find ALL relevant documents
  - Batch queries (not real-time)

Analysis:
  FlatIndex latency: ~50ms per query
  Recall: 100% (required for legal compliance)

Decision: FlatIndex ✓
  - Recall requirement mandates exact search
  - Batch processing tolerates higher latency
  - Missing a relevant document = liability
```

### Case 3: Real-Time Recommendations

```
Scenario:
  - 10 million user embeddings
  - Item embeddings (256-dim)
  - <10ms latency requirement
  - 1000 QPS

Analysis:
  FlatIndex latency: ~100ms per query (too slow)
  FlatIndex capacity: 10 QPS max (100× under requirement)

Decision: FlatIndex ✗ → Use HNSW or IVF
  - Scale requires approximate methods
  - 95% recall is acceptable for recommendations
```

### Case 4: Rapid Prototyping

```
Scenario:
  - Building a demo for stakeholders
  - 50,000 vectors (will grow to 5M in production)
  - Need something working today

Analysis:
  FlatIndex: Works immediately, no tuning
  HNSW: Need to choose M, efConstruction, efSearch...

Decision: FlatIndex ✓ (for now)
  - Get the demo working first
  - Swap index type later when scale demands it
  - VectorIndexProtocol makes this a one-line change
```

---

## Performance Boundaries

### Latency vs. Dataset Size

```
Dimension: 512

Dataset Size    FlatIndex Latency    Verdict
───────────────────────────────────────────────
     1,000            ~0.5ms         Always OK
    10,000            ~5ms           Usually OK
    50,000            ~25ms          Often OK
   100,000            ~50ms          Sometimes OK
   500,000            ~250ms         Rarely OK
 1,000,000            ~500ms         Almost never OK
```

### Throughput Limits

```
Assuming single-threaded:

Dataset Size    Max QPS (FlatIndex)
──────────────────────────────────
    10,000          200 QPS
    50,000           40 QPS
   100,000           20 QPS
   500,000            4 QPS
 1,000,000            2 QPS

With 8-core parallelism (batch search):
  Multiply above by ~6-8×
```

---

## The Index Swap Pattern

VectorIndex is designed for easy index swapping:

```swift
// 📍 See: Sources/VectorIndex/IndexProtocols.swift

// All indices conform to VectorIndexProtocol
public protocol VectorIndexProtocol: Actor {
    func search(query: [Float], k: Int, filter: ...) async throws -> [SearchResult]
    // ...
}

// Your code uses the protocol, not concrete types:
class SearchService {
    private let index: any VectorIndexProtocol

    init(index: any VectorIndexProtocol) {
        self.index = index
    }

    func search(query: [Float]) async throws -> [SearchResult] {
        return try await index.search(query: query, k: 10, filter: nil)
    }
}

// Start with FlatIndex:
let service = SearchService(index: FlatIndex(dimension: 512, metric: .cosine))

// Later, swap to HNSW with no other code changes:
let service = SearchService(index: HNSWIndex(dimension: 512, metric: .cosine))
```

---

## Hidden Costs of ANN Indices

Before graduating from FlatIndex, consider these overheads:

### HNSW Hidden Costs

```
Build time:       O(n log n) — can take minutes for large datasets
Memory overhead:  ~1.5-2× raw vector size (graph edges)
Update cost:      Insertions require graph maintenance
Tuning required:  M, efConstruction, efSearch all affect recall/speed
```

### IVF Hidden Costs

```
Training time:    k-means clustering can be slow
Rebuild needed:   Adding vectors may require re-clustering
Parameter tuning: nlist, nprobe affect recall/speed tradeoff
```

### FlatIndex: No Hidden Costs

```
Build time:       O(1) per insert
Memory overhead:  0 (just the vectors)
Update cost:      O(1) insert, O(1) delete
Tuning required:  None
```

---

## Key Takeaways

1. **Start with FlatIndex.** It's the simplest option that might just work.

2. **For < 50k vectors, FlatIndex often wins.** Index overhead exceeds benefits.

3. **For exact recall requirements, FlatIndex is mandatory.** ANN cannot guarantee 100%.

4. **For rapid iteration, FlatIndex is fastest to set up.** No parameters to tune.

5. **The protocol abstraction enables easy swapping.** Graduate to ANN when needed.

6. **Consider hidden costs.** Build time, memory, tuning complexity all add up.

---

## Next Up

When FlatIndex isn't enough, the first step is partitioning. Let's explore how IVF divides the search space:

**[→ Chapter 3: IVF Index](../03-IVF-Inverted-File-Index/README.md)**
