# HNSW in VectorIndex

> **Reading time:** 12 minutes
> **Prerequisites:** [efSearch and Recall](./04-EfSearch-And-Recall.md)

---

## The Implementation

This guide walks through VectorIndex's HNSW implementation, connecting concepts from previous guides to actual code.

---

## Configuration

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:14-27

public struct Configuration: Sendable {
    public let m: Int              // Max connections per node per layer
    public let efConstruction: Int // Build-time search width
    public let efSearch: Int       // Query-time search width
    public let rngSeed: UInt64     // Deterministic layer assignment
    public let rngStream: UInt64   // Stream ID for sharding

    public init(
        m: Int = 16,
        efConstruction: Int = 200,
        efSearch: Int = 64,
        rngSeed: UInt64 = 0xDEADBEEFCAFEBABE,
        rngStream: UInt64 = 0
    ) {
        self.m = m
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.rngSeed = rngSeed
        self.rngStream = rngStream
    }
}
```

### Creating an HNSW Index

```swift
// Basic usage
let hnsw = HNSWIndex(dimension: 768, metric: .cosine)

// With custom configuration
let hnsw = HNSWIndex(
    dimension: 768,
    metric: .cosine,
    config: .init(m: 32, efConstruction: 400, efSearch: 128)
)
```

---

## Internal Data Structures

### Node Representation

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:398-405

private struct Node {
    let id: VectorID
    let vectorOffset: Int      // Offset into vectorStorage
    var metadata: [String: String]?
    let level: Int             // Highest layer this node appears in
    var neighbors: [[Int]]     // Per-layer neighbor indices
    var isDeleted: Bool
}
```

### Contiguous Vector Storage

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:417

private var vectorStorage: ContiguousArray<Float> = []
```

Vectors are stored contiguously for cache efficiency:

```
vectorStorage layout:

[vec0_dim0, vec0_dim1, ..., vec0_dimD, vec1_dim0, vec1_dim1, ...]
     ↑                                      ↑
     Node 0 starts here                     Node 1 starts here
     (offset = 0)                           (offset = D)
```

### CSR Caching for Traversal

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:421-426

private var csrOffsetsCache: [[Int32]] = []     // Per-layer CSR offsets
private var csrNeighborsCache: [[Int32]] = []   // Per-layer flat neighbor lists
private var invNormsCache: [Float]?              // For cosine metric
```

The CSR (Compressed Sparse Row) format enables efficient kernel traversal:

```
Original:                      CSR Format:
Node 0: neighbors [2, 5, 7]    offsets:   [0, 3, 5, 8, ...]
Node 1: neighbors [3, 4]       neighbors: [2, 5, 7, 3, 4, 1, 6, 8, ...]
Node 2: neighbors [1, 6, 8]
                               Node i's neighbors: neighbors[offsets[i]:offsets[i+1]]
```

---

## Search Implementation

### The Full Search Flow

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:102-170

public func search(query: [Float], k: Int, filter: ...) async throws -> [SearchResult] {
    guard k > 0 else { return [] }
    try checkVector(query)
    guard let ep = entryPoint else { return [] }

    // Rebuild caches if needed
    rebuildCSRIfNeeded()

    // Build allow bitset (exclude deleted nodes)
    let N = nodes.count
    var allowBits = [UInt64](repeating: 0, count: (N + 63) >> 6)
    for i in 0..<N {
        if !nodes[i].isDeleted {
            let w = i >> 6, b = i & 63
            allowBits[w] |= (1 << UInt64(b))
        }
    }

    // Call optimized traversal kernel
    let ef = max(config.efSearch, k)
    var idsOut = [Int32](repeating: -1, count: ef)
    var distsOut = [Float](repeating: .infinity, count: ef)

    let written = HNSWTraversal.traverse(
        q: query, d: dimension,
        entryPoint: ep, maxLevel: maxLevel,
        offsetsPerLayer: csrOffsetsCache,
        neighborsPerLayer: csrNeighborsCache,
        xb: vectorStorage, N: N,
        ef: ef, metric: metric,
        allowBits: allowBits,
        idsOut: &idsOut, distsOut: &distsOut
    )

    // Build results with optional filter
    var results: [SearchResult] = []
    for i in 0..<written {
        let idx = Int(idsOut[i])
        let node = nodes[idx]
        if let filter = filter, !filter(node.metadata) { continue }
        var score = distsOut[i]
        if metric == .euclidean { score = sqrt(score) }  // L2² → L2
        results.append(SearchResult(id: node.id, score: score))
        if results.count == k { break }
    }

    return results
}
```

### The Traversal Kernel

```swift
// 📍 See: Sources/VectorIndex/Kernels/HNSWTraversal.swift

public struct HNSWTraversal {
    public static func traverse(
        q: UnsafePointer<Float>, d: Int,
        entryPoint: Int32, maxLevel: Int32,
        offsetsPerLayer: [UnsafePointer<Int32>?],
        neighborsPerLayer: [UnsafePointer<Int32>?],
        xb: UnsafePointer<Float>, N: Int,
        ef: Int, metric: HNSWMetric,
        allowBits: UnsafePointer<UInt64>?, allowN: Int,
        invNorms: UnsafePointer<Float>?,
        idsOut: inout [Int32], distsOut: inout [Float]
    ) -> Int {
        // Optimized C-style traversal for maximum performance
        // Uses unsafe pointers throughout
        // ...
    }
}
```

---

## Insertion Implementation

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:438-516

private func internalInsert(id: VectorID, vector: [Float], metadata: [String: String]?) async throws {
    // Handle update case
    if idToIndex[id] != nil {
        try await remove(id: id)
    }

    // 1. Assign random level
    let level = randomLevel()
    let newIndex = nodes.count

    // 2. Store vector contiguously
    let vectorOffset = vectorStorage.count
    vectorStorage.append(contentsOf: vector)

    // 3. Create node
    let node = Node(
        id: id, vectorOffset: vectorOffset, metadata: metadata,
        level: level, neighbors: Array(repeating: [], count: level + 1),
        isDeleted: false
    )
    nodes.append(node)
    idToIndex[id] = newIndex
    activeCount += 1
    markInvNormsDirty()
    markCSRDirty()

    // 4. Connect to graph
    if let oldEP = entryPoint {
        var cur = oldEP

        // Descend to new node's level
        for l in stride(from: maxLevel, to: level, by: -1) {
            cur = greedySearchLayer(vector, enter: cur, level: l)
        }

        // Connect at each level
        let ef = max(config.efConstruction, config.m)
        for l in stride(from: min(level, maxLevel), through: 0, by: -1) {
            let candidates = searchLayer(vector, enter: cur, ef: ef, level: l)
            let selected = selectNeighbors(for: vector, among: candidates, maxM: config.m)
            connect(newIndex, with: selected, level: l)
        }

        // Update entry point if needed
        if level > maxLevel {
            maxLevel = level
            entryPoint = newIndex
        }
    } else {
        entryPoint = newIndex
        maxLevel = level
    }
}
```

---

## Deletion with Tombstones

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:77-92

public func remove(id: VectorID) async throws {
    if let idx = idToIndex[id] {
        nodes[idx].isDeleted = true  // Tombstone
        idToIndex.removeValue(forKey: id)

        // Detach from neighbors
        let lvl = nodes[idx].level
        for l in 0...lvl {
            for n in nodes[idx].neighbors[l] {
                removeNeighbor(n, idx, level: l)
            }
            nodes[idx].neighbors[l].removeAll()
        }

        // Update entry point if needed
        if entryPoint == idx {
            entryPoint = findAnyActiveIndex()
        }
        activeCount -= 1
    }
}
```

Tombstones remain until `compact()` is called.

---

## Parallel Batch Search

```swift
// 📍 See: Sources/VectorIndex/HNSWIndex.swift:190-252

public func batchSearch(queries: [[Float]], k: Int, filter: ...) async throws -> [[SearchResult]] {
    // Snapshot data for parallel access
    let ctx = BatchSearchContext(
        vectorStorage: Array(vectorStorage),
        csrOffsets: csrOffsetsCache,
        csrNeighbors: csrNeighborsCache,
        // ... other immutable data
    )

    // Parallel execution
    return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
        for (queryIndex, query) in queries.enumerated() {
            group.addTask {
                Self.performSingleSearch(query: query, queryIndex: queryIndex, ctx: ctx, filter: filter)
            }
        }

        var results = [[SearchResult]](repeating: [], count: queries.count)
        for try await (index, result) in group {
            results[index] = result
        }
        return results
    }
}
```

---

## 🔗 VectorCore Connection

HNSW relies heavily on VectorCore for distance computation:

```swift
// 🔗 VectorCore: Distance is the critical path

// Greedy layer search: O(M × levels) distances per query
// Layer 0 search: O(ef × M) distances per query
// Insertion: O(efConstruction × M × levels) distances per insert

// All these use VectorCore's SIMD-accelerated kernels
```

The contiguous storage layout enables efficient VectorCore access:

```swift
// Accessing vector at node index
@inline(__always)
private func vectorArray(at nodeIndex: Int) -> [Float] {
    let offset = nodes[nodeIndex].vectorOffset
    return Array(vectorStorage[offset..<(offset + dimension)])
}

// 🔗 VectorCore: Contiguous layout enables SIMD prefetching
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert | O(efConstruction × log n × M) | Graph updates |
| Search | O(ef × log n × M) | Graph traversal |
| Delete | O(level × M) | Tombstone + neighbor detach |
| Compact | O(n × level × M) | Rebuild without tombstones |

---

## Key Takeaways

1. **Contiguous vector storage.** Cache-friendly layout for traversal.

2. **CSR caching.** Enables efficient kernel-based search.

3. **Tombstone deletion.** Logical delete; physical removal via compact().

4. **Parallel batch search.** Multiple queries execute concurrently.

5. **Optimized traversal kernel.** Unsafe pointers for maximum performance.

---

## Next Up

Both IVF and HNSW store full vectors. What if we need to compress them?

**[→ Chapter 5: Product Quantization](../05-Product-Quantization/README.md)**
