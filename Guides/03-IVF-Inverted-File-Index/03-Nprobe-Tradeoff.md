# The nprobe Tradeoff

> **Reading time:** 12 minutes
> **Prerequisites:** [Inverted Lists](./02-Inverted-Lists.md)

---

## The Concept

**nprobe** is the number of inverted lists to search at query time. It's the primary knob for controlling IVF's recall-speed tradeoff.

```
nprobe = 1:    Fast, low recall    (search 1 list)
nprobe = 10:   Balanced            (search 10 lists)
nprobe = 100:  Slow, high recall   (search 100 lists)
nprobe = nlist: Exact (≈ flat)     (search all lists)
```

---

## Why Recall Isn't 100%

A query's true nearest neighbors might not all be in the closest cluster:

```
Query: ●

                True nearest neighbors:
    ┌─────────┐  ┌─────────┐
    │    ○    │  │ ★  ★    │   ★ = True top-3 NN
    │  ○   ○  │  │   ★     │   ○ = Other vectors
    │    ○    │  │  ○  ○   │
    │ ★   ●   │  │         │   ● = Query
    │         │  │         │
    └─────────┘  └─────────┘
       Cell A       Cell B

With nprobe=1 (Cell A only):
  - Find 1 of 3 true NN
  - Recall = 33%

With nprobe=2 (Cells A and B):
  - Find 3 of 3 true NN
  - Recall = 100%
```

Vectors near cell boundaries can "belong" to the wrong cell from the query's perspective.

---

## The Mathematics

### Expected Recall Model

For a simplified model with uniform distribution:

```
Probability that a true NN is in one of the nprobe closest cells:

P(hit) ≈ 1 - (1 - nprobe/nlist)^k

For k=10 nearest neighbors:
  nprobe=1,  nlist=1000:  P ≈ 1%
  nprobe=10, nlist=1000:  P ≈ 10%
  nprobe=100, nlist=1000: P ≈ 63%
```

Real data is clustered, so actual recall is higher than this uniform model suggests.

### Empirical Results

Typical recall curves on embedding datasets:

```
         │
   100%  │                    ●────●────●
         │               ●────
         │          ●────
  Recall │     ●────
         │ ●───
         │●
    50%  │
         │
         └──────────────────────────────────
           1    8   16   32   64   128  nprobe

Typical values for 95% recall:
  nlist=256:   nprobe ≈ 8-16
  nlist=1024:  nprobe ≈ 16-32
  nlist=4096:  nprobe ≈ 32-64
```

---

## Latency Impact

More probes = more vectors to scan:

```
Vectors scanned ≈ n × (nprobe / nlist)

Example (n=1M, nlist=1000):
  nprobe=1:   ~1,000 vectors scanned
  nprobe=10:  ~10,000 vectors scanned
  nprobe=100: ~100,000 vectors scanned
```

Latency scales linearly with nprobe (approximately):

```
         │
         │                          ●
         │                     ●
 Latency │                ●
         │           ●
         │      ●
         │ ●
         └──────────────────────────────────
           1    8   16   32   64   128  nprobe
```

---

## Finding the Right nprobe

### The Systematic Approach

```
Algorithm: nprobe Calibration

1. Sample 1000 queries from your query distribution
2. For each query, compute exact top-k (using FlatIndex)
3. For nprobe in [1, 2, 4, 8, 16, 32, 64, 128, ...]:
   a. Run IVF search
   b. Compute recall = intersection(IVF_result, exact_result) / k
   c. Record (nprobe, mean_recall, mean_latency)
4. Choose smallest nprobe that meets recall target
```

### Example Calibration Results

```
nprobe   Recall@10   Latency
────────────────────────────
   1       52%        0.5ms
   2       68%        0.8ms
   4       81%        1.4ms
   8       91%        2.5ms   ← Target: 90% recall
  16       96%        4.8ms
  32       99%        9.2ms
  64       99.8%     18.0ms
```

For 90% recall target: **nprobe=8**

---

## Dynamic nprobe

Some systems adjust nprobe based on query characteristics:

```swift
// Pseudocode for adaptive nprobe

func search(query: [Float], k: Int) -> [SearchResult] {
    // Start with low nprobe
    var nprobe = 4
    var results = ivfSearch(query, k: k, nprobe: nprobe)

    // If results look uncertain, probe more
    while results.last!.score > confidenceThreshold && nprobe < maxNprobe {
        nprobe *= 2
        results = ivfSearch(query, k: k, nprobe: nprobe)
    }

    return results
}
```

VectorIndex uses fixed nprobe, but the protocol allows implementing adaptive strategies on top.

---

## In VectorIndex

nprobe is configured at index creation:

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift

public struct Configuration: Sendable {
    public let nlist: Int    // Number of clusters
    public let nprobe: Int   // Lists to search per query

    public init(nlist: Int = 256, nprobe: Int = 8) {
        self.nlist = nlist
        self.nprobe = nprobe
    }
}
```

Usage:

```swift
let ivf = IVFIndex(
    dimension: 512,
    metric: .cosine,
    config: .init(nlist: 1024, nprobe: 16)  // 16 lists per query
)
```

### The Search Implementation

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:413-462

public func search(query: [Float], k: Int, filter: ...) async throws -> [SearchResult] {
    // Find nprobe nearest centroids
    let probe = min(config.nprobe, centroids.count)
    var centroidDists: [(Int, Float)] = []
    for (i, c) in centroids.enumerated() {
        centroidDists.append((i, distance(query, c, metric: metric)))
    }
    centroidDists.sort { $0.1 < $1.1 }

    // Collect candidates from top nprobe lists
    var candidates = Set<VectorID>()
    for (ci, _) in centroidDists.prefix(probe) {
        for id in lists[ci] { candidates.insert(id) }
    }

    // Score and return top-k
    // ...
}
```

---

## 🔗 VectorCore Connection

Finding nearest centroids is a mini-search problem:

```swift
// 🔗 VectorCore: Centroid distance computation

// For nlist centroids, compute distances to query
for (i, centroid) in centroids.enumerated() {
    let dist = distance(query, centroid, metric: metric)  // ← SIMD
    centroidDists.append((i, dist))
}

// Then sort and take top nprobe
// This is O(nlist × d) for distances + O(nlist log nprobe) for selection
```

For large nlist, this overhead can be significant. Some systems use a coarse index over centroids (e.g., HNSW over centroids).

---

## nprobe Guidelines

| Recall Target | nprobe (rule of thumb) |
|---------------|------------------------|
| 80% | nlist / 32 |
| 90% | nlist / 16 |
| 95% | nlist / 8 |
| 99% | nlist / 4 |
| 99.9% | nlist / 2 |

**Always calibrate on your data!** These are starting points only.

---

## Key Takeaways

1. **nprobe controls recall vs. speed.** More probes = higher recall, higher latency.

2. **Boundary effects cause recall loss.** True neighbors may be in adjacent cells.

3. **Calibrate empirically.** Measure recall on your query distribution.

4. **Typical sweet spot: 5-15% of nlist.** Achieves 90-95% recall for most datasets.

5. **nprobe can be dynamic.** Adjust based on confidence in results.

---

## Next Up

Let's see how all these pieces come together in VectorIndex's implementation:

**[→ IVF in VectorIndex](./04-IVF-In-VectorIndex.md)**
