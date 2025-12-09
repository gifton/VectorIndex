# Index Selection Guide

> **Reading time:** 12 minutes
> **Prerequisites:** [QPS and Latency](./02-QPS-And-Latency.md)

---

## The Decision Framework

Choosing an index depends on your requirements across four dimensions:

1. **Dataset size** (n)
2. **Recall requirements**
3. **Latency budget**
4. **Memory budget**

---

## The Decision Tree

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │  n < 50,000?          │
                    └───────────────────────┘
                           │           │
                          Yes          No
                           │           │
                           ▼           ▼
                    ┌──────────┐   ┌───────────────────────┐
                    │ FlatIndex │   │  100% recall needed?  │
                    └──────────┘   └───────────────────────┘
                                          │           │
                                         Yes          No
                                          │           │
                                          ▼           ▼
                                   ┌──────────┐   ┌───────────────────────┐
                                   │ FlatIndex │   │  Memory constrained?  │
                                   └──────────┘   └───────────────────────┘
                                                         │           │
                                                        Yes          No
                                                         │           │
                                                         ▼           ▼
                                           ┌─────────────────┐   ┌───────────────────────┐
                                           │     IVF-PQ      │   │  Recall > 95%?        │
                                           └─────────────────┘   └───────────────────────┘
                                                                        │           │
                                                                       Yes          No
                                                                        │           │
                                                                        ▼           ▼
                                                                 ┌──────────┐   ┌──────────┐
                                                                 │   HNSW   │   │   IVF    │
                                                                 └──────────┘   └──────────┘
```

---

## Index Comparison Matrix

| Property | FlatIndex | IVF | HNSW | IVF-PQ |
|----------|-----------|-----|------|--------|
| **Max practical size** | 50K | 10M | 50M | 1B |
| **Recall range** | 100% | 85-99% | 95-99.9% | 80-95% |
| **Typical latency** | O(n) | O(√n) | O(log n) | O(√n) |
| **Memory overhead** | 0% | ~5% | 50-100% | -90% |
| **Build time** | O(1) | O(n) | O(n log n) | O(n) |
| **Update cost** | O(1) | O(1)* | O(log n) | O(1)* |
| **Tuning complexity** | None | Low | Medium | Medium |

*IVF may need periodic re-clustering

---

## Detailed Recommendations

### FlatIndex

**Use when:**
```
✓ n < 50,000
✓ Need 100% recall
✓ Data changes frequently
✓ Simplicity is priority
✓ Development/prototyping
```

**Avoid when:**
```
✗ n > 100,000 (too slow)
✗ Latency < 10ms required (can't achieve)
```

### IVF (IVFIndex)

**Use when:**
```
✓ 50K < n < 10M
✓ 85-95% recall acceptable
✓ Balanced speed/recall needed
✓ Can afford training time
```

**Configuration:**
```swift
let config = IVFIndex.Configuration(
    nlist: Int(sqrt(Double(n))),  // Start here
    nprobe: nlist / 10            // Tune for recall
)
```

### HNSW (HNSWIndex)

**Use when:**
```
✓ 50K < n < 50M
✓ Need 95%+ recall
✓ Memory is available
✓ Data is relatively static
```

**Configuration:**
```swift
let config = HNSWIndex.Configuration(
    m: 16,              // 16-32 typical
    efConstruction: 200, // Higher = better graph
    efSearch: 64        // Tune for recall
)
```

### IVF-PQ

**Use when:**
```
✓ n > 10M (memory constrained)
✓ 80-95% recall acceptable
✓ Throughput prioritized
✓ Can afford training time
```

---

## Sizing Guidelines

### Memory Estimation

```swift
func estimateMemory(n: Int, d: Int, indexType: String) -> Int {
    let vectorBytes = n * d * 4  // Float32

    switch indexType {
    case "Flat":
        return vectorBytes
    case "IVF":
        let nlist = Int(sqrt(Double(n)))
        let centroidBytes = nlist * d * 4
        return vectorBytes + centroidBytes
    case "HNSW":
        let graphBytes = n * 32 * 4 * 2  // M=16, 2 layers avg
        return vectorBytes + graphBytes
    case "IVF-PQ":
        let m = 64
        let codeBytes = n * m
        let codebookBytes = m * 256 * (d/m) * 4
        return codeBytes + codebookBytes
    default:
        return vectorBytes
    }
}
```

### Latency Estimation

```
FlatIndex:  latency ≈ n × d × 2ns
IVF:        latency ≈ nprobe × (n/nlist) × d × 2ns
HNSW:       latency ≈ efSearch × M × log(n) × d × 2ns
```

---

## Migration Path

Start simple, scale up:

```
Phase 1: Development
  └── FlatIndex (simple, exact)

Phase 2: Initial Production
  └── HNSW or IVF (based on requirements)

Phase 3: Scale
  └── IVF-PQ or sharded HNSW

The VectorIndexProtocol abstraction makes migration easy:
  - Same search API
  - Just swap index type
```

---

## Real-World Scenarios

### Scenario 1: E-commerce Product Search

```
Requirements:
  - 500K products
  - 90% recall acceptable
  - <20ms p99 latency
  - 16 GB RAM available

Recommendation: HNSW
  - Fits in memory with overhead
  - High recall with low latency
  - M=16, efSearch=64
```

### Scenario 2: Document Retrieval at Scale

```
Requirements:
  - 50M documents
  - 85% recall acceptable
  - <50ms p99 latency
  - 32 GB RAM limit

Recommendation: IVF-PQ
  - 50M × 768D full = 150 GB (too big)
  - IVF-PQ: ~5 GB
  - nlist=8192, nprobe=64, m=64
```

### Scenario 3: High-Precision Legal Search

```
Requirements:
  - 2M documents
  - 99%+ recall required
  - Latency flexible
  - 64 GB RAM available

Recommendation: HNSW with high ef
  - 2M × 768D + graph ≈ 12 GB
  - M=32, efConstruction=400, efSearch=256
  - Or FlatIndex if latency truly doesn't matter
```

---

## Key Takeaways

1. **Start with FlatIndex.** Graduate to ANN when needed.

2. **HNSW for high recall.** Best quality at cost of memory.

3. **IVF for balanced workloads.** Tunable recall/speed.

4. **IVF-PQ for massive scale.** Trade recall for memory.

5. **Protocol abstraction enables migration.** Easy to swap indices.

---

## Next Up

How much memory will your index actually use?

**[→ Memory Footprint](./04-Memory-Footprint.md)**
