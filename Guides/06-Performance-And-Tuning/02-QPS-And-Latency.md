# QPS and Latency

> **Reading time:** 10 minutes
> **Prerequisites:** [Recall@K](./01-Recall-At-K.md)

---

## The Concept

**QPS** (Queries Per Second) measures throughput. **Latency** measures response time. They're related but distinct:

```
QPS = 1 / mean_latency  (for single-threaded)
QPS = parallelism / mean_latency  (for multi-threaded)
```

---

## Measuring Latency

### Single Query Latency

```swift
func measureLatency(query: [Float], index: VectorIndexProtocol) async -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    _ = try await index.search(query: query, k: 10, filter: nil)
    let end = CFAbsoluteTimeGetCurrent()
    return (end - start) * 1000  // milliseconds
}
```

### Percentiles Matter

```
1000 queries:
  p50 (median):  2.1ms
  p90:           3.5ms
  p95:           5.2ms
  p99:           12.3ms  ← Tail latency!
  max:           45.0ms

If SLO is "p99 < 10ms", this index FAILS.
Average (2.8ms) would hide the problem.
```

### Why Percentiles Vary

```
Latency variation comes from:
  - Cache misses (cold queries)
  - GC pauses (Swift ARC overhead)
  - Graph structure (some queries need more hops)
  - OS scheduling (other processes)
```

---

## Measuring QPS

### Single-Threaded QPS

```swift
func measureQPS(queries: [[Float]], index: VectorIndexProtocol, duration: Double = 5.0) async -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    var completed = 0
    var i = 0

    while CFAbsoluteTimeGetCurrent() - start < duration {
        _ = try await index.search(query: queries[i % queries.count], k: 10, filter: nil)
        completed += 1
        i += 1
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - start
    return Double(completed) / elapsed
}
```

### Multi-Threaded QPS

```swift
func measureParallelQPS(queries: [[Float]], index: VectorIndexProtocol, concurrency: Int = 8) async -> Double {
    let start = CFAbsoluteTimeGetCurrent()

    let results = try await index.batchSearch(
        queries: Array(queries.prefix(1000)),
        k: 10,
        filter: nil
    )

    let elapsed = CFAbsoluteTimeGetCurrent() - start
    return Double(results.count) / elapsed
}
```

---

## Latency Breakdown

Understand where time goes:

```
HNSW Search Breakdown:

┌────────────────────────────────────────────┐
│ Total: 2.5ms                               │
├────────────────────────────────────────────┤
│ Layer descent:        0.3ms (12%)          │
│ Layer 0 search:       1.8ms (72%)          │ ← Dominant
│ Result construction:  0.2ms (8%)           │
│ Protocol overhead:    0.2ms (8%)           │
└────────────────────────────────────────────┘
```

Optimize the dominant component (Layer 0 search = distance computations).

---

## Throughput vs. Latency Tradeoff

```
                        │
   QPS (throughput)     │                  ● Batch processing
                        │            ●
                        │      ●
                        │  ●
                        │●
                        └────────────────────────────
                                    Latency (ms)

Batching increases throughput but also latency.
Choose based on your requirements:
  - Interactive: Optimize latency
  - Batch jobs: Optimize throughput
```

---

## Benchmarking Best Practices

### Warm Up

```swift
// Run warm-up queries before measuring
for _ in 0..<100 {
    _ = try await index.search(query: randomQuery(), k: 10, filter: nil)
}

// Now measure
let results = measureLatency(...)
```

### Avoid Caching Effects

```swift
// Use different queries each time
let queries = generateRandomQueries(count: 10000)

// Or cycle through a large query set
for i in 0..<numMeasurements {
    let query = queries[i % queries.count]
    // ...
}
```

### Multiple Runs

```swift
// Run multiple trials
var latencies: [Double] = []
for _ in 0..<5 {
    latencies.append(measureLatency(...))
}

let median = latencies.sorted()[2]
let stddev = standardDeviation(latencies)
```

---

## Expected Performance

On Apple Silicon (M1/M2/M3):

| Index | Dataset | k | Latency (p50) | QPS (single) |
|-------|---------|---|---------------|--------------|
| Flat | 10K | 10 | 1ms | 1000 |
| Flat | 100K | 10 | 10ms | 100 |
| HNSW | 100K | 10 | 0.3ms | 3000 |
| HNSW | 1M | 10 | 0.5ms | 2000 |
| IVF | 1M | 10 | 2ms | 500 |

These are rough estimates—measure on your hardware!

---

## 🔗 VectorCore Connection

Performance is bounded by VectorCore's distance throughput:

```swift
// 🔗 VectorCore: Distance computation speed

// Measure raw distance throughput:
let vectorsPerSecond = benchmarkDistances(dimension: 512)
// Typical: 10-50M distances/sec on Apple Silicon

// This bounds index QPS:
// - Flat: QPS ≤ vectorsPerSecond / n
// - HNSW: QPS ≤ vectorsPerSecond / (nodes_visited)
// - IVF: QPS ≤ vectorsPerSecond / (candidates_scanned)
```

---

## Key Takeaways

1. **Measure percentiles, not just averages.** p99 reveals tail latency.

2. **QPS and latency are linked.** QPS = concurrency / latency.

3. **Warm up before measuring.** Avoid cold-cache effects.

4. **Use diverse queries.** Avoid caching artifacts.

5. **Understand the breakdown.** Optimize the dominant component.

---

## Next Up

How do we choose the right index for our requirements?

**[→ Index Selection Guide](./03-Index-Selection-Guide.md)**
