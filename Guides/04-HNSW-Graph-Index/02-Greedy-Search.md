# Greedy Search

> **Reading time:** 10 minutes
> **Prerequisites:** [Skip Lists and Layers](./01-Skip-Lists-And-Layers.md)

---

## The Concept

HNSW search uses **greedy best-first traversal**: always move toward the neighbor closest to the query.

```
Query: Q (target)

Start: Entry point E

Step 1: At E, check neighbors → N1 is closest to Q → move to N1
Step 2: At N1, check neighbors → N4 is closest to Q → move to N4
Step 3: At N4, check neighbors → none closer than N4 → STOP

Result: N4 is the local minimum (best approximation)
```

---

## The Algorithm

### Phase 1: Descent Through Layers

```
Algorithm: HNSW Descent (Layers L_max to 1)

For layer L from max_level down to 1:
    1. At current node, examine all neighbors at layer L
    2. Move to the neighbor closest to query
    3. Repeat until no closer neighbor exists
    4. Descend to layer L-1, keeping current node
```

This quickly positions us near the target region.

### Phase 2: Layer 0 Search

```
Algorithm: HNSW Layer 0 Search

1. Initialize:
   - candidates = priority queue (min by distance)
   - results = priority queue (max by distance, capacity ef)
   - visited = set of seen nodes

2. Add entry node to candidates and results

3. While candidates not empty:
   a. Pop closest candidate C
   b. If C is farther than worst in results (and results full), STOP
   c. For each neighbor N of C at layer 0:
      - If N not visited:
        - Mark visited
        - Compute distance(query, N)
        - Add to candidates
        - Add to results (maintaining top-ef)

4. Return top-k from results
```

---

## Walkthrough Example

```
Query Q wants 3 nearest neighbors (k=3), efSearch=5

Layer 2:
        Entry ●
             ╱ ╲
           ●     ●  ← Greedy: pick closer one
                  ↓
Layer 1:
      ●──●──●──●
      (continue greedy descent)
           ↓
Layer 0:
      Start thorough search at landing point
      Expand outward, keeping 5 best (ef=5)

      ●──●──●──●──●──●──●
         ↑  ↑  ↑
        Best 3 returned
```

---

## Why Greedy Works

### The Small-World Structure

HNSW constructs graphs with high clustering coefficient:

```
Local neighborhood:           Query enters from here
                                    ↓
    ●───●───●               ●───●───●───Q
    │ ╲ │ ╱ │               │ ╲ │ ╱ │
    ●───●───●    ───→       ●───●───●
    │ ╱ │ ╲ │               │ ╱ │ ╲ │
    ●───●───●               ●───●───●

    Clusters are well-connected internally
    → Greedy search finds local optima quickly
```

### The Hierarchy Prevents Dead Ends

Upper layers have long-range connections:

```
Without hierarchy:           With hierarchy:

Query lands in               Query uses express lanes
wrong region, stuck          to reach correct region

    ●───●                    Layer 2: ●────────────●
    │╲ ╱│                                          ↓
    ●─Q─● stuck!             Layer 1:     ●────●───●
    │╱ ╲│                                      ↓
    ●───●                    Layer 0: ●─●─●─●─●─●─Q
```

---

## In VectorIndex

### Greedy Layer Search

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:595-610

private func greedySearchLayer(_ query: [Float], enter: Int, level: Int) -> Int {
    var cur = enter
    var curDist = distance(query, vectorArray(at: cur), metric: metric)
    var changed = true

    while changed {
        changed = false
        for n in nodes[cur].neighbors[safe: level] ?? [] {
            let d = distance(query, vectorArray(at: n), metric: metric)
            if d < curDist {
                curDist = d
                cur = n
                changed = true
            }
        }
    }
    return cur
}
```

### Layer 0 Beam Search

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:613-649

private func searchLayer(_ query: [Float], enter: Int, ef: Int, level: Int) -> [Int] {
    var candidates: [(Int, Float)] = []  // Min-heap by distance
    var result: [(Int, Float)] = []      // Top-ef by distance
    var visited = Set<Int>()

    let enterDist = distance(query, vectorArray(at: enter), metric: metric)
    candidates.append((enter, enterDist))
    result.append((enter, enterDist))
    visited.insert(enter)

    while !candidates.isEmpty {
        // Pop best candidate
        let (cand, candDist) = candidates.removeFirst()  // Simplified

        // If worse than worst in result and result is full, stop
        if result.count >= ef, let worst = result.last, candDist > worst.1 {
            break
        }

        // Explore neighbors
        for n in nodes[cand].neighbors[safe: level] ?? [] {
            if visited.insert(n).inserted, !nodes[n].isDeleted {
                let d = distance(query, vectorArray(at: n), metric: metric)
                if result.count < ef || d < (result.last?.1 ?? .infinity) {
                    candidates.append((n, d))
                    insertSorted(&result, (n, d))
                    if result.count > ef { result.removeLast() }
                }
            }
        }
    }

    return result.map { $0.0 }
}
```

---

## Visited Set Optimization

The visited set prevents re-exploring nodes:

```
Without visited tracking:     With visited tracking:

    A → B → A → B → A...     A → B → C → D → E (done)
    (infinite loop!)          (each node seen once)
```

For large graphs, the visited set can use:
- `Set<Int>`: Simple, O(1) lookup
- Bitset: Memory-efficient for large n
- Bloom filter: Approximate, very fast

VectorIndex uses `Set<Int>` for simplicity.

---

## 🔗 VectorCore Connection

Search is distance-dominated:

```swift
// 🔗 VectorCore: Every neighbor examination

for n in nodes[cand].neighbors[level] {
    let d = distance(query, vectorArray(at: n), metric: metric)  // ← VectorCore
    // ... update candidates/results
}
```

Optimizations:
- Batch distance computation when exploring multiple neighbors
- Prefetching next neighbor's vector while computing current distance
- SIMD-accelerated distance kernels

---

## Key Takeaways

1. **Two-phase search:** Fast descent through layers, then thorough Layer 0 search.

2. **Greedy navigation:** Always move toward closer neighbors.

3. **ef controls thoroughness:** Larger ef = more candidates explored = higher recall.

4. **Visited set prevents loops:** Essential for correctness.

5. **Small-world structure enables convergence:** Graph construction ensures greedy finds good solutions.

---

## Next Up

How do we build this graph in the first place? Let's explore construction parameters:

**[→ M and efConstruction](./03-M-And-EfConstruction.md)**
