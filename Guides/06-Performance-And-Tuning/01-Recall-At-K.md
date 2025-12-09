# Recall@K

> **Reading time:** 10 minutes
> **Prerequisites:** [Chapter 6 Introduction](./README.md)

---

## The Concept

**Recall@K** measures what fraction of the true k nearest neighbors your index finds:

```
Recall@K = |ANN_results ∩ True_k-NN| / k
```

This is the primary quality metric for approximate nearest neighbor search.

---

## Computing Recall

### Step 1: Get Ground Truth

Use exact search (FlatIndex) to find true top-k:

```swift
let flat = FlatIndex(dimension: d, metric: .cosine)
// ... insert all vectors

let trueTopK = try await flat.search(query: q, k: k, filter: nil)
```

### Step 2: Get Approximate Results

```swift
let hnsw = HNSWIndex(dimension: d, metric: .cosine)
// ... insert all vectors

let approxTopK = try await hnsw.search(query: q, k: k, filter: nil)
```

### Step 3: Compute Overlap

```swift
func recallAtK(approx: [SearchResult], exact: [SearchResult]) -> Float {
    let approxIDs = Set(approx.map { $0.id })
    let exactIDs = Set(exact.map { $0.id })
    let intersection = approxIDs.intersection(exactIDs)
    return Float(intersection.count) / Float(exact.count)
}
```

---

## Interpreting Recall

```
Recall@10 = 90%

Meaning: 9 of the true 10 nearest neighbors were found
         1 true neighbor was missed
         (may have returned a non-neighbor instead)
```

### Typical Targets

| Application | Recall Target | Rationale |
|-------------|---------------|-----------|
| Legal/Medical search | 99%+ | Missing relevant docs = liability |
| E-commerce | 90-95% | Good enough for recommendations |
| Content feed | 80-90% | Engagement-driven, exact doesn't matter |
| Development | 70%+ | Just need rough results for testing |

---

## Recall Across Parameter Ranges

Sweep parameters to find the recall-latency curve:

```swift
// Calibration script

let testQueries = sampleQueries(count: 1000)
let exactResults = testQueries.map { flat.search($0, k: 10) }

var calibration: [(param: Int, recall: Float, latency: Double)] = []

// For HNSW: sweep efSearch
for ef in [10, 20, 50, 100, 200, 500] {
    let results = testQueries.map { hnsw.search($0, k: 10, efSearch: ef) }

    let recall = zip(results, exactResults).map { recallAtK($0, $1) }.average()
    let latency = measureLatency { ... }

    calibration.append((ef, recall, latency))
}

// Find parameter that meets recall target with minimum latency
let target = 0.95
let optimal = calibration.first { $0.recall >= target }
```

---

## Visualizing the Curve

```
           │
   Recall  │                         ●────●
           │                   ●────
           │             ●────
           │       ●──── ← "knee" - best tradeoff
           │ ●────
           │
           └──────────────────────────────────────
                              Latency (ms)

The "knee" is where small latency increases give diminishing recall gains.
Often a good operating point.
```

---

## Recall vs. K

Recall can vary with k:

```
k=1:   Recall might be 98% (easier to find single best)
k=10:  Recall might be 95% (harder to find all 10)
k=100: Recall might be 92% (even harder)

Always measure at your actual k value!
```

---

## Common Mistakes

### Mistake 1: Testing on Training Data

```
Wrong: Use same vectors for training IVF and measuring recall
Right: Hold out test vectors, or use separate query set
```

### Mistake 2: Too Few Test Queries

```
Wrong: Measure recall on 10 queries
Right: Use 1000+ queries for stable estimates

Standard error of recall estimate:
  SE ≈ √(recall × (1-recall) / n_queries)

For recall=0.95, n=1000: SE ≈ 0.007 (±0.7%)
For recall=0.95, n=10:   SE ≈ 0.07 (±7%)
```

### Mistake 3: Ignoring Metric Mismatch

```
Wrong: Build index with L2, measure recall with cosine similarity
Right: Use same metric for index and ground truth
```

---

## 🔗 VectorCore Connection

Recall depends on distance computation accuracy:

```swift
// 🔗 VectorCore: Numerical stability affects recall

// If distance computation has errors:
true_distance:   0.12345678
computed:        0.12345679 (floating-point rounding)

// Usually not a problem, but for ties:
vec_A distance: 0.123456
vec_B distance: 0.123456 (tie)

// Tie-breaking can affect which makes it into top-k
// VectorCore uses deterministic tie-breaking for reproducibility
```

---

## Key Takeaways

1. **Recall@K = overlap with true top-k.** Primary quality metric.

2. **Use FlatIndex for ground truth.** Exact search provides the reference.

3. **Sweep parameters to find the knee.** Balance recall vs. latency.

4. **Use 1000+ test queries.** Fewer gives unstable estimates.

5. **Match k to your actual use case.** Recall varies with k.

---

## Next Up

How do we measure the speed side of the tradeoff?

**[→ QPS and Latency](./02-QPS-And-Latency.md)**
