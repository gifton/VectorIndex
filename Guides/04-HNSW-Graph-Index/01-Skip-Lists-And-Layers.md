# Skip Lists and Layers

> **Reading time:** 12 minutes
> **Prerequisites:** [Chapter 4 Introduction](./README.md)

---

## The Concept

HNSW is inspired by **skip lists**—a probabilistic data structure that achieves O(log n) search by maintaining multiple "express lane" layers above a base layer.

```
Skip List Structure:

Layer 3:  ●─────────────────────────────────────────●
          ↓                                         ↓
Layer 2:  ●─────────────●─────────────────────────●─●
          ↓             ↓                         ↓
Layer 1:  ●───────●─────●───────●───────●─────────●─●
          ↓       ↓     ↓       ↓       ↓         ↓
Layer 0:  ●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●─●
```

**Key property:** Each element has a random "height"—most elements are only in Layer 0, but a few reach higher layers.

---

## From Skip Lists to HNSW

Skip lists work for 1D ordered data. HNSW extends the concept to high-dimensional vectors:

| Skip List | HNSW |
|-----------|------|
| Linear chain | Graph with M neighbors |
| Comparison-based navigation | Distance-based greedy search |
| Probabilistic height | Probabilistic layer assignment |
| O(log n) search | O(log n) expected hops |

---

## Layer Assignment

Each node is assigned a layer using an exponential distribution:

```swift
// 📍 See: Sources/VectorIndex/Kernels/HNSWLevelAssignment.swift

func randomLevel(M: Int) -> Int {
    // Probability of being in layer L = (1/M)^L
    // Expected max layer for n elements ≈ log_M(n)

    let mL = 1.0 / log(Double(M))
    let r = Double.random(in: 0..<1)
    return Int(floor(-log(r) * mL))
}
```

**Distribution example (M=16):**

```
Layer 0:  100%   of nodes
Layer 1:   6.25% of nodes
Layer 2:   0.39% of nodes
Layer 3:   0.024% of nodes
...

For n = 1,000,000:
  Layer 0: 1,000,000 nodes
  Layer 1:    62,500 nodes
  Layer 2:     3,906 nodes
  Layer 3:       244 nodes
  Layer 4:        15 nodes
  Layer 5:         1 node (entry point)
```

---

## The Graph Structure

Each node has connections at each layer it participates in:

```
Node A (assigned to Layer 2):

  Layer 2:  A ───→ [B, C]           (2 connections)
  Layer 1:  A ───→ [B, C, D, E]     (4 connections)
  Layer 0:  A ───→ [B, C, D, E, F, G, H, I]  (up to 2M connections)

Different M values per layer:
  Layer 0: Up to 2M connections (denser graph)
  Layer 1+: Up to M connections (sparser graph)
```

### Why Different M for Layer 0?

Layer 0 is where all vectors live and where the final search happens. More connections improve recall:

```
Layer 0 connectivity:        Search paths:

  2M connections:              Many paths to target
  ●─●─●─●─●─●─●                ●═══●═══●
  │╲│╱│╲│╱│╲│                  ║   ╲   ║
  ●─●─●─●─●─●─●                ●═══●═══●
  │╲│╱│╲│╱│╲│                  (redundancy helps)
  ●─●─●─●─●─●─●
```

---

## The Entry Point

The **entry point** is the single node at the highest layer. All searches start here:

```
Entry point: Node 42 (happens to be at Layer 5)

Search: "Find neighbors of query Q"

1. Start at Node 42, Layer 5
2. Greedily descend through layers
3. At Layer 0, perform thorough local search
4. Return best candidates
```

The entry point changes when a new node is inserted with a higher layer than the current maximum.

---

## Visualization

Let's trace a 3-layer HNSW with 12 nodes:

```
Layer 2:        [3]─────────────[9]
                 │               │
Layer 1:    [1]─[3]─────[6]────[9]─[11]
             │   │       │       │   │
Layer 0: [0][1][2][3][4][5][6][7][8][9][10][11]
          ═══════════════════════════════════

Node assignments:
  Layer 0 only: 0, 2, 4, 5, 7, 8, 10
  Up to Layer 1: 1, 6, 11
  Up to Layer 2: 3, 9

Entry point: Node 3 or 9 (whichever was inserted last at Layer 2)
```

---

## In VectorIndex

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift

private struct Node {
    let id: VectorID
    let vectorOffset: Int     // Position in contiguous storage
    var metadata: [String: String]?
    let level: Int            // Highest layer this node appears in
    var neighbors: [[Int]]    // Neighbors per layer: neighbors[layer] = [nodeIndex, ...]
    var isDeleted: Bool
}

private var nodes: [Node] = []
private var entryPoint: Int?
private var maxLevel: Int = 0
```

### Layer Construction

When inserting a new node:

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:438-516

private func internalInsert(id: VectorID, vector: [Float], metadata: [String: String]?) async throws {
    // 1. Assign random layer
    let level = randomLevel()
    let newIndex = nodes.count

    // 2. Create node with empty neighbor lists per layer
    let node = Node(
        id: id,
        vectorOffset: vectorStorage.count,
        level: level,
        neighbors: Array(repeating: [], count: level + 1)
    )
    nodes.append(node)

    // 3. Navigate from entry point to find neighbors at each layer
    if let oldEP = entryPoint {
        var cur = oldEP

        // Descend from max layer to new node's layer + 1
        for l in stride(from: maxLevel, to: level, by: -1) {
            cur = greedySearchLayer(vector, enter: cur, level: l)
        }

        // At each layer from level down to 0: connect to neighbors
        for l in stride(from: min(level, maxLevel), through: 0, by: -1) {
            let candidates = searchLayer(vector, enter: cur, ef: efConstruction, level: l)
            let selected = selectNeighbors(for: vector, among: candidates, maxM: config.m)
            connect(newIndex, with: selected, level: l)
        }

        // 4. Update entry point if needed
        if level > maxLevel {
            maxLevel = level
            entryPoint = newIndex
        }
    } else {
        // First node becomes entry point
        entryPoint = newIndex
        maxLevel = level
    }
}
```

---

## 🔗 VectorCore Connection

Layer navigation involves many distance computations:

```swift
// 🔗 VectorCore: Each layer descent computes distances

for l in stride(from: maxLevel, to: 0, by: -1) {
    // Greedy search at this layer
    var cur = currentNode
    var bestDist = distance(query, cur.vector)  // ← VectorCore

    for neighbor in cur.neighbors[l] {
        let d = distance(query, neighbor.vector)  // ← VectorCore
        if d < bestDist {
            bestDist = d
            cur = neighbor
        }
    }
}
```

---

## Key Takeaways

1. **Layers create express lanes.** Upper layers enable O(log n) navigation.

2. **Exponential layer assignment.** Most nodes are Layer 0 only; few reach high layers.

3. **Entry point is the highest node.** All searches start from this single node.

4. **Layer 0 is denser (2M edges).** Better recall for the fine-grained final search.

5. **Each node stores per-layer neighbors.** Memory: O(n × M × avg_layers).

---

## Next Up

Now let's see how search actually traverses this structure:

**[→ Greedy Search](./02-Greedy-Search.md)**
