# IVF in VectorIndex

> **Reading time:** 10 minutes
> **Prerequisites:** [The nprobe Tradeoff](./03-Nprobe-Tradeoff.md)

---

## The Implementation

This guide walks through VectorIndex's IVF implementation, connecting the concepts from previous guides to actual code.

---

## Configuration

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:14-22

public struct Configuration: Sendable {
    public let nlist: Int      // Number of coarse centroids
    public let nprobe: Int     // Lists to probe at search time

    public init(nlist: Int = 256, nprobe: Int = 8) {
        self.nlist = nlist
        self.nprobe = nprobe
    }
}
```

### Creating an IVF Index

```swift
// Basic usage
let ivf = IVFIndex(dimension: 768, metric: .cosine)

// With custom configuration
let ivf = IVFIndex(
    dimension: 768,
    metric: .cosine,
    config: .init(nlist: 1024, nprobe: 32)
)
```

---

## The Training Flow

IVF requires a training phase to compute centroids:

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:265-284

public func optimize() async throws {
    guard !store.isEmpty else {
        centroids.removeAll()
        lists.removeAll()
        return
    }

    // 1. Determine cluster count
    let k = max(1, min(config.nlist, store.count))

    // 2. Initialize with k-means++
    let initialCentroids = try kmeansPlusPlusInitRandom(k: k, seed: 42)

    // 3. Run Lloyd's algorithm
    centroids = try await kmeans(centroids: initialCentroids, maxIterations: 20)

    // 4. Assign all vectors to clusters
    lists = Array(repeating: [], count: centroids.count)
    idToListIndex.removeAll(keepingCapacity: false)
    for (id, (vec, _)) in store {
        if let ci = nearestCentroidIndex(for: vec) {
            lists[ci].append(id)
            idToListIndex[id] = ci
        }
    }
}
```

### Usage Pattern

```swift
// Build index
let ivf = IVFIndex(dimension: 768, metric: .cosine, config: .init(nlist: 256))

// Add vectors
for doc in documents {
    try await ivf.insert(id: doc.id, vector: doc.embedding, metadata: doc.meta)
}

// IMPORTANT: Call optimize() to build centroids
try await ivf.optimize()

// Now search works efficiently
let results = try await ivf.search(query: queryVec, k: 10, filter: nil)
```

---

## The Search Flow

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:413-462

public func search(query: [Float], k: Int, filter: ...) async throws -> [SearchResult] {
    // Validation
    guard k > 0 else { return [] }
    guard query.count == dimension else { throw VectorError.dimensionMismatch(...) }

    // If not trained, fall back to linear scan
    if centroids.isEmpty || lists.isEmpty {
        return linearScan(query: query, k: k, filter: filter)
    }

    // 1. Find nprobe nearest centroids
    let probe = min(config.nprobe, centroids.count)
    var centroidDists: [(Int, Float)] = []
    for (i, c) in centroids.enumerated() {
        centroidDists.append((i, distance(query, c, metric: metric)))
    }
    centroidDists.sort { $0.1 < $1.1 }

    // 2. Collect candidates from probed lists
    var candidates = Set<VectorID>()
    for (ci, _) in centroidDists.prefix(probe) {
        for id in lists[ci] { candidates.insert(id) }
    }

    // 3. Score candidates
    var results: [SearchResult] = []
    for id in candidates {
        guard let (vec, meta) = store[id] else { continue }
        if let filter = filter, !filter(meta) { continue }
        let d = distance(query, vec, metric: metric)
        results.append(SearchResult(id: id, score: d))
    }

    // 4. Return top-k
    results.sort { $0.score < $1.score }
    if results.count > k { results.removeLast(results.count - k) }
    return results
}
```

---

## Kernel #30: High-Performance Storage

For larger scale, VectorIndex provides optimized storage via Kernel #30:

```swift
// 📍 See: Sources/VectorIndex/Kernels/IVFAppend.swift

// Enable kernel #30 storage with optional persistence
try await ivf.enableKernel30Storage(
    format: .flat,
    k_c: 1024,
    durablePath: "/path/to/index.bin"
)

// Ingest vectors in bulk
try await ivf.ingestFlat(
    listIDs: assignedLists,
    externalIDs: vectorIDs,
    vectors: flatVectors
)
```

Benefits:
- Contiguous memory layout
- Memory-mapped persistence
- Optimized for batch operations

---

## Parallel Batch Search

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:475-543

public func batchSearch(queries: [[Float]], k: Int, filter: ...) async throws -> [[SearchResult]] {
    // Parallel execution using TaskGroup
    return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
        for (queryIndex, query) in queries.enumerated() {
            group.addTask {
                Self.performIVFSearch(query: query, queryIndex: queryIndex, ...)
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

## AccelerableIndex Protocol

IVF implements `AccelerableIndex` for GPU/accelerator integration:

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:702-830

extension IVFIndex: AccelerableIndex {
    public func getCandidates(query: [Float], k: Int, filter: ...) async throws -> AccelerationCandidates {
        // 1. Find nearest centroids
        // 2. Collect candidate vectors from those lists
        // 3. Return as contiguous buffer for GPU processing
    }

    public func getIndexStructure() async -> IndexStructure {
        if centroids.isEmpty { return .flat }

        let structure = IVFStructure(
            centroids: centroids,
            invertedLists: lists,
            nprobe: config.nprobe
        )
        return .ivf(structure)
    }
}
```

---

## 🔗 VectorCore Connection

IVF uses VectorCore's distance functions throughout:

```swift
// 🔗 VectorCore: Distance computation

// Centroid distances
for (i, c) in centroids.enumerated() {
    let d = distance(query, c, metric: metric)  // ← VectorCore
    centroidDists.append((i, d))
}

// Candidate scoring
for id in candidates {
    let d = distance(query, vec, metric: metric)  // ← VectorCore
    results.append(SearchResult(id: id, score: d))
}
```

The k-means training also leverages VectorCore:

```swift
// 📍 See: Sources/VectorIndex/Kernels/KMeansMiniBatchKernel.swift

// Assignment step: find nearest centroid for each vector
// Uses SIMD-accelerated distance computation
```

---

## Persistence

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift:642-666

public func save(to url: URL) async throws {
    let recs: [PersistedRecord] = store.map { ... }
    let payload = PersistedIndex(
        type: "IVF",
        version: 1,
        dimension: dimension,
        metric: metric.rawValue,
        records: recs
    )
    let data = try JSONEncoder().encode(payload)
    try data.write(to: url, options: .atomic)
}

public static func load(from url: URL) async throws -> IVFIndex {
    let data = try Data(contentsOf: url)
    let payload = try JSONDecoder().decode(PersistedIndex.self, from: data)
    let idx = IVFIndex(dimension: payload.dimension, metric: .from(raw: payload.metric))
    try await idx.batchInsert(payload.records.map { ... })
    try await idx.optimize()  // Rebuild centroids
    return idx
}
```

Note: `optimize()` is called after loading to rebuild centroids.

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert | O(nlist × d) | Find nearest centroid |
| Search | O(nlist × d + candidates × d) | Centroid search + candidate scoring |
| Optimize | O(n × nlist × d × iterations) | K-means training |
| Delete | O(list_size) | Linear scan of assigned list |
| Save/Load | O(n × d) | Serialize all vectors |

---

## Key Takeaways

1. **Call `optimize()` after bulk inserts.** Training builds the centroids that enable fast search.

2. **Kernel #30 for scale.** Memory-mapped storage for larger datasets.

3. **Parallel batch search.** Multiple queries execute concurrently.

4. **AccelerableIndex enables GPU.** Get candidates for off-CPU processing.

5. **Persistence rebuilds centroids.** `load()` calls `optimize()` automatically.

---

## Next Up

IVF is powerful, but graph-based methods often achieve better recall. Let's explore HNSW:

**[→ Chapter 4: HNSW Graph Index](../04-HNSW-Graph-Index/README.md)**
