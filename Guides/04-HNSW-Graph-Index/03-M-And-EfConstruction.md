# M and efConstruction

> **Reading time:** 12 minutes
> **Prerequisites:** [Greedy Search](./02-Greedy-Search.md)

---

## The Concept

Graph construction has two key parameters:

- **M**: Maximum number of connections per node per layer
- **efConstruction**: Search width when finding neighbors for a new node

These control the **quality-cost tradeoff during index building**.

---

## Parameter M

### What It Controls

M determines graph connectivity:

```
M = 4:                        M = 16:

  ●───●                       ●───●───●───●
  │   │                      ╱│╲ ╱│╲ ╱│╲ ╱│╲
  ●───●                     ● │ ● │ ● │ ● │ ●
  │   │                      ╲│╱ ╲│╱ ╲│╱ ╲│╱
  ●───●                       ●───●───●───●

Sparse graph                 Dense graph
Low memory                   High memory
Lower recall                 Higher recall
```

### Memory Impact

```
Memory per node ≈ M × sizeof(neighbor_index) × avg_layers

For n = 1,000,000 nodes, M = 16:
  avg_layers ≈ 1.07 (exponential distribution)
  neighbors_L0 ≈ 2M = 32
  neighbors_L1+ ≈ M = 16

  Memory ≈ n × (32 + 16×0.07) × 4 bytes
        ≈ 1M × 33 × 4 = 132 MB (just for graph)
```

### Recall Impact

Higher M = more paths to explore = higher probability of finding true neighbors:

```
           │
   Recall  │                    ●────●
           │              ●────
           │        ●────
           │   ●────
           │●──
           └──────────────────────────────
              4    8   16   32   64     M

Typical values:
  M = 8:   ~90% recall
  M = 16:  ~95% recall
  M = 32:  ~98% recall
  M = 64:  ~99% recall
```

---

## Parameter efConstruction

### What It Controls

When inserting a node, we search for its neighbors using beam width `efConstruction`:

```
efConstruction = 10:          efConstruction = 200:

Search explores 10            Search explores 200
candidates per layer          candidates per layer
    ↓                             ↓
May miss some good           Finds better neighbors
neighbors                     ↓
    ↓                         Higher quality graph
Lower quality graph
```

### Build Time Impact

```
Build time ∝ n × efConstruction × log(n) × d

For n = 1,000,000, d = 512:
  efConstruction = 40:   ~2 minutes
  efConstruction = 100:  ~5 minutes
  efConstruction = 200:  ~10 minutes
  efConstruction = 400:  ~20 minutes
```

### Recall Impact

Better neighbors = better graph = higher recall:

```
           │
   Recall  │                         ●────●
           │                   ●────
           │             ●────
           │       ●────
           │ ●────
           └──────────────────────────────────
             40   100   200   400   800  efConstruction
```

---

## The Interaction

M and efConstruction work together:

```
                    Low efConstruction    High efConstruction
                    ───────────────────   ───────────────────
Low M (8-12)        Fast build, low       Moderate build,
                    quality, low recall   medium quality

High M (32-64)      Fast build, good      Slow build,
                    quality, high recall  best quality
```

### Recommended Starting Points

| Use Case | M | efConstruction | Notes |
|----------|---|----------------|-------|
| Development/Testing | 8 | 40 | Fast iteration |
| Balanced | 16 | 200 | Good all-around |
| High Recall | 32 | 400 | Production quality |
| Maximum Recall | 64 | 800 | When recall > 99% needed |

---

## The Neighbor Selection Heuristic

Not all candidate neighbors are equal. HNSW uses a **diversity heuristic**:

```
Algorithm: Select Neighbors (Heuristic)

Input: new_node, candidates (sorted by distance), M

1. selected = []
2. For each candidate C in order of distance:
   a. If selected is empty, add C
   b. Else:
      - For each already-selected S:
        - If dist(C, S) < dist(C, new_node):
          - C is closer to an existing neighbor than to new_node
          - SKIP C (would create redundancy)
      - If not skipped, add C to selected
3. If |selected| < M, fill with nearest unselected candidates

Return selected
```

This prevents clustering of similar neighbors:

```
Without heuristic:            With heuristic:

  New node: N                   New node: N

  Selected: A, B, C             Selected: A, D, G
  (all in same direction)       (spread around N)

      A●  B●  C●                  D●
           ╲  │ ╱                  │
            ╲│╱                  A●────N●────G●
             N●                       │
                                      E●

Poor coverage                   Good coverage
```

---

## In VectorIndex

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:14-27

public struct Configuration: Sendable {
    public let m: Int              // Max connections per node
    public let efConstruction: Int // Build-time search width
    public let efSearch: Int       // Query-time search width
    public let rngSeed: UInt64     // Deterministic layer assignment

    public init(
        m: Int = 16,
        efConstruction: Int = 200,
        efSearch: Int = 64,
        rngSeed: UInt64 = 0xDEADBEEFCAFEBABE,
        rngStream: UInt64 = 0
    ) { ... }
}
```

### Neighbor Selection Implementation

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:861-894

func selectNeighbors(for vec: [Float], among candidates: [Int], level: Int, maxM: Int) -> [Int] {
    // Sort candidates by distance
    var sorted: [(Int, Float)] = candidates.map {
        ($0, distance(vec, vectorArray(at: $0), metric: metric))
    }
    sorted.sort { $0.1 < $1.1 }

    // Apply diversity heuristic
    var selected: [Int] = []
    for (cand, _) in sorted {
        var good = true
        let candVec = vectorArray(at: cand)

        for s in selected {
            let d_cs = distance(candVec, vectorArray(at: s), metric: metric)
            let d_cx = distance(candVec, vec, metric: metric)
            if d_cs < d_cx {
                good = false
                break
            }
        }

        if good { selected.append(cand) }
        if selected.count >= maxM { break }
    }

    // Fallback: fill with nearest if needed
    if selected.count < maxM {
        for (cand, _) in sorted where !selected.contains(cand) {
            selected.append(cand)
            if selected.count >= maxM { break }
        }
    }

    return selected
}
```

---

## 🔗 VectorCore Connection

Construction is extremely distance-heavy:

```swift
// 🔗 VectorCore: For each insertion

// 1. Descent through layers: O(log n × M) distances
// 2. Layer 0 search: O(efConstruction) distances
// 3. Neighbor selection: O(candidates²) distances for heuristic

// Total per insertion: O(efConstruction + M²) distance computations
// Total for n insertions: O(n × efConstruction) VectorCore calls
```

Build time is dominated by VectorCore distance computations.

---

## Key Takeaways

1. **M controls graph density.** Higher M = more memory, higher recall.

2. **efConstruction controls build quality.** Higher ef = slower build, better graph.

3. **They interact.** High M with low efConstruction can still produce good graphs.

4. **Diversity heuristic improves quality.** Avoids redundant neighbors.

5. **Start with M=16, efConstruction=200.** Adjust based on recall measurements.

---

## Next Up

Once built, how do we tune search quality at query time?

**[→ efSearch and Recall](./04-EfSearch-And-Recall.md)**
