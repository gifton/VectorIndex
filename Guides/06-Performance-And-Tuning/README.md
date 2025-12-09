# Chapter 6: Performance & Tuning

> **Measure what matters. Optimize what hurts.**

This chapter covers how to evaluate vector search systems, choose the right index for your needs, and tune parameters for optimal performance.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Recall@K](./01-Recall-At-K.md) | 10 min | Measuring search quality |
| [2. QPS and Latency](./02-QPS-And-Latency.md) | 10 min | Measuring search speed |
| [3. Index Selection Guide](./03-Index-Selection-Guide.md) | 12 min | Choosing the right index |
| [4. Memory Footprint](./04-Memory-Footprint.md) | 8 min | Estimating index size |

---

## The Big Picture

Vector search optimization involves three competing goals:

```
                    Recall
                      ▲
                     ╱ ╲
                    ╱   ╲
                   ╱     ╲
                  ╱       ╲
                 ╱         ╲
                ◀───────────▶
           Latency          Memory

You can optimize two at the expense of the third:
  - High recall + Low latency → High memory (HNSW)
  - High recall + Low memory → High latency (IVF-PQ + rerank)
  - Low latency + Low memory → Lower recall (IVF-PQ, small nprobe)
```

---

## Key Metrics

### Quality Metrics

| Metric | Definition | When to Use |
|--------|------------|-------------|
| **Recall@K** | Fraction of true top-k found | Primary quality metric |
| **Precision@K** | Fraction of results that are true top-k | When false positives matter |
| **MRR** | Mean Reciprocal Rank | When top-1 matters most |
| **NDCG** | Normalized DCG | When ranking order matters |

### Performance Metrics

| Metric | Definition | When to Use |
|--------|------------|-------------|
| **QPS** | Queries per second | Throughput-focused |
| **p50 Latency** | Median query time | Typical performance |
| **p99 Latency** | 99th percentile | Tail latency SLOs |
| **Build Time** | Index construction time | Batch systems |

### Resource Metrics

| Metric | Definition | When to Use |
|--------|------------|-------------|
| **Memory** | RAM usage | Capacity planning |
| **Disk** | Storage for persistence | Durability requirements |
| **CPU** | Utilization during search | Cost optimization |

---

## The Tuning Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Tuning Workflow                                  │
│                                                                         │
│  1. Define Requirements                                                 │
│     ├── Target recall (e.g., 95%)                                      │
│     ├── Latency SLO (e.g., p99 < 10ms)                                │
│     └── Memory budget (e.g., 32 GB)                                    │
│                                                                         │
│  2. Choose Index Type                                                   │
│     ├── Start with decision tree (Chapter 6.3)                         │
│     └── Consider future scale                                          │
│                                                                         │
│  3. Calibrate Parameters                                                │
│     ├── Sample representative queries                                  │
│     ├── Measure recall vs. latency curve                               │
│     └── Find knee of curve that meets requirements                     │
│                                                                         │
│  4. Validate in Production                                              │
│     ├── A/B test against baseline                                      │
│     └── Monitor metrics over time                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Index Selection

| Scenario | Recommended Index | Key Parameters |
|----------|------------------|----------------|
| < 50k vectors | FlatIndex | — |
| 50k-1M, high recall | HNSW | M=32, efSearch=128 |
| 50k-1M, balanced | IVF | nlist=√n, nprobe=nlist/10 |
| 1M-100M, high recall | HNSW | M=16-32, efSearch=64-128 |
| 1M-100M, memory constrained | IVF-PQ | m=64, nlist=4096 |
| > 100M, single node | IVF-PQ | m=64-128, rerank=true |

---

## 🔗 VectorCore Connection

Performance depends on VectorCore's distance throughput:

```swift
// 🔗 VectorCore: Distance computation is the critical path

// Benchmark VectorCore distance speed:
let throughput = measureDistanceQPS(dimension: 512)
// Typical: 10-50M distances/sec on Apple Silicon

// This bounds index performance:
// - Flat: QPS ≈ throughput / n
// - HNSW: QPS ≈ throughput / (efSearch × M × log(n))
// - IVF: QPS ≈ throughput / (nprobe × n/nlist)
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Testing on training data | Overly optimistic recall | Use held-out test queries |
| Ignoring tail latency | p99 blows up | Measure and optimize p99, not just average |
| Over-provisioning memory | Wasted resources | Right-size based on actual needs |
| Under-training IVF | Poor cluster quality | Use sufficient training data |
| Wrong metric | Silent quality degradation | Match metric to embedding model |

---

## Start Here

**[→ Recall@K](./01-Recall-At-K.md)**

---

*Chapter 6 of 7 • [← Product Quantization](../05-Product-Quantization/README.md) | [Next: Capstone →](../07-Capstone/README.md)*
