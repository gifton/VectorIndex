# efSearch and Recall

> **Reading time:** 10 minutes
> **Prerequisites:** [M and efConstruction](./03-M-And-EfConstruction.md)

---

## The Concept

**efSearch** controls the beam width during query-time search. Unlike M and efConstruction (which are fixed at build time), efSearch can be adjusted per query.

```
efSearch = 10:               efSearch = 100:

Explore 10 candidates         Explore 100 candidates
     ↓                             ↓
Fast but may miss            Thorough but slower
some true neighbors          Higher recall
     ↓                             ↓
~85% recall                   ~98% recall
```

This is HNSW's primary knob for the recall-latency tradeoff.

---

## How efSearch Affects Search

### The Search Process

```
Layer 0 beam search with ef=efSearch:

1. Start with entry node in candidates
2. While candidates not empty:
   - Pop closest candidate
   - If worse than ef-th best result, STOP
   - Add all unvisited neighbors to candidates
   - Keep top-ef results

Higher ef = more candidates explored = higher recall
```

### The Stopping Condition

```
ef = 10:                      ef = 100:

Results: [d=0.1, 0.2, ...,    Results: [d=0.1, 0.2, ...,
          0.5, 0.6, 0.7,                0.5, 0.6, ...,
          0.8, 0.9, 1.0]                0.95, 0.96, 0.97...]
                ↑                                    ↑
         10th best = 1.0              100th best = 0.97

Stop when candidate > 1.0    Stop when candidate > 0.97
(earlier stopping)           (later stopping, more thorough)
```

---

## The Recall-Latency Curve

```
           │
   Recall  │                         ●────●
           │                   ●────
           │             ●────
           │       ●────
           │ ●────
           └──────────────────────────────────────
             10   20   50   100  200  500  efSearch

           │
   Latency │                              ●
           │                        ●
           │                  ●
           │            ●
           │      ●
           │ ●
           └──────────────────────────────────────
             10   20   50   100  200  500  efSearch
```

**Key insight:** Recall has diminishing returns; latency grows roughly linearly.

---

## Finding the Right efSearch

### The Calibration Process

```swift
// Calibration algorithm

let testQueries = sampleQueries(n: 1000)
let exactResults = testQueries.map { flatIndex.search($0, k: 10) }

var calibration: [(ef: Int, recall: Float, latency: Double)] = []

for ef in [10, 20, 50, 100, 200, 500] {
    hnsw.config.efSearch = ef  // Adjust efSearch

    var totalRecall = 0.0
    var totalLatency = 0.0

    for (query, exact) in zip(testQueries, exactResults) {
        let start = clock()
        let approx = hnsw.search(query, k: 10)
        let elapsed = clock() - start

        let recall = intersectionSize(approx, exact) / 10.0
        totalRecall += recall
        totalLatency += elapsed
    }

    calibration.append((
        ef: ef,
        recall: Float(totalRecall / 1000),
        latency: totalLatency / 1000
    ))
}

// Find smallest ef that meets recall target
let target = 0.95
let optimal = calibration.first { $0.recall >= target }
```

### Typical Results

```
efSearch   Recall@10   Latency
──────────────────────────────
   10       72%        0.05ms
   20       85%        0.08ms
   50       93%        0.15ms
  100       97%        0.28ms   ← Often the sweet spot
  200       99%        0.52ms
  500       99.8%      1.20ms
```

---

## Dynamic efSearch

Different queries may need different thoroughness:

```swift
// Adaptive efSearch based on result confidence

func adaptiveSearch(query: [Float], k: Int, targetRecall: Float) -> [SearchResult] {
    var ef = 20  // Start low

    while ef <= 500 {
        let results = hnsw.search(query, k: k, efSearch: ef)

        // Heuristic: if top results are very close, we're confident
        let topScore = results[0].score
        let kthScore = results[k-1].score
        let spread = kthScore - topScore

        if spread < threshold {
            // Results are tight, likely found true neighbors
            return results
        }

        ef *= 2  // Try harder
    }

    return hnsw.search(query, k: k, efSearch: 500)
}
```

---

## efSearch vs. k

A common question: what if k > efSearch?

```
efSearch = 50, k = 100

Problem: We only keep 50 candidates, but need 100 results!

Solution: efSearch must be >= k

Rule: efSearch = max(efSearch, k)
```

VectorIndex handles this automatically:

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:121

let ef = max(config.efSearch, k)  // Ensure ef >= k
```

---

## In VectorIndex

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:102-170

public func search(query: [Float], k: Int, filter: ...) async throws -> [SearchResult] {
    guard k > 0 else { return [] }
    try checkVector(query)
    guard let ep = entryPoint else { return [] }

    // Ensure ef >= k
    let ef = max(config.efSearch, k)

    // ... (layer descent and search)
}
```

### Per-Query Override (Pattern)

While VectorIndex uses fixed efSearch, you can implement per-query control:

```swift
// Pattern: Create index variants for different recall requirements

class AdaptiveHNSWService {
    private let lowRecallIndex: HNSWIndex   // efSearch = 32
    private let highRecallIndex: HNSWIndex  // efSearch = 200

    func search(query: [Float], k: Int, highRecall: Bool) async throws -> [SearchResult] {
        if highRecall {
            return try await highRecallIndex.search(query: query, k: k, filter: nil)
        } else {
            return try await lowRecallIndex.search(query: query, k: k, filter: nil)
        }
    }
}
```

---

## 🔗 VectorCore Connection

Search cost is proportional to efSearch:

```swift
// 🔗 VectorCore: Search distance computations

// Nodes explored ≈ efSearch × avg_degree
// Distance computations ≈ efSearch × M

// For efSearch=100, M=16:
//   ~1,600 distance computations per query
//   Each distance: SIMD-accelerated (d operations)
//   Total: 1,600 × d ≈ 800K FLOPs for d=512
```

---

## Guidelines

| Recall Target | efSearch (starting point) |
|---------------|---------------------------|
| 80% | k × 2 |
| 90% | k × 4 |
| 95% | k × 8 |
| 99% | k × 16 |
| 99.9% | k × 32 |

**Always calibrate on your data!** These are rough starting points.

---

## Key Takeaways

1. **efSearch is the query-time recall knob.** Larger = higher recall, higher latency.

2. **efSearch must be >= k.** VectorIndex handles this automatically.

3. **Diminishing returns on recall.** Going from 95% to 99% costs more than 90% to 95%.

4. **Calibrate empirically.** Measure on representative queries.

5. **Can be dynamic per query.** Adjust based on confidence or importance.

---

## Next Up

Let's see how all these pieces come together in VectorIndex's HNSW implementation:

**[→ HNSW in VectorIndex](./05-HNSW-In-VectorIndex.md)**
