# Inverted Lists

> **Reading time:** 10 minutes
> **Prerequisites:** [Clustering and Centroids](./01-Clustering-And-Centroids.md)

---

## The Concept

An **inverted list** stores all vectors that belong to a particular cluster. The name comes from information retrieval, where term→document mappings are "inverted" from the document→term view.

```
Traditional view:              Inverted view:
────────────────               ──────────────
doc1 → cluster_5              cluster_0 → [doc7, doc12, doc45, ...]
doc2 → cluster_2              cluster_1 → [doc3, doc9, doc22, ...]
doc3 → cluster_1              cluster_2 → [doc2, doc8, doc31, ...]
...                           ...
```

This structure enables efficient subset search: given a cluster ID, immediately access all its vectors.

---

## The Data Structure

Each cluster maintains a list of (vector_id, vector_data):

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IVF Index Structure                              │
│                                                                         │
│  Centroids: [c₀, c₁, c₂, ..., c_{nlist-1}]                             │
│                                                                         │
│  Inverted Lists:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ List 0: [(id_7, vec_7), (id_12, vec_12), (id_45, vec_45), ...]  │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ List 1: [(id_3, vec_3), (id_9, vec_9), (id_22, vec_22), ...]    │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ List 2: [(id_2, vec_2), (id_8, vec_8), (id_31, vec_31), ...]    │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ ...                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ID→List mapping: {id_7: 0, id_3: 1, id_2: 2, ...}                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Operations

### Insertion

When inserting a new vector:

```swift
func insert(id: VectorID, vector: [Float]) {
    // 1. Find nearest centroid
    let nearestList = findNearestCentroid(vector)

    // 2. Add to that list
    lists[nearestList].append((id, vector))

    // 3. Update mapping
    idToList[id] = nearestList
}
```

**Complexity:** O(nlist × d) to find nearest centroid

### Search

When searching:

```swift
func search(query: [Float], k: Int, nprobe: Int) -> [SearchResult] {
    // 1. Find nprobe nearest centroids
    let probeLists = findNearestCentroids(query, n: nprobe)

    // 2. Collect candidates from those lists
    var candidates: [SearchResult] = []
    for listIndex in probeLists {
        for (id, vec) in lists[listIndex] {
            let dist = distance(query, vec)
            candidates.append(SearchResult(id: id, score: dist))
        }
    }

    // 3. Return top-k
    return topK(candidates, k: k)
}
```

**Complexity:** O(nprobe × d) for centroids + O(candidates × d) for vectors

### Deletion

When removing a vector:

```swift
func remove(id: VectorID) {
    guard let listIndex = idToList[id] else { return }

    // Remove from list
    lists[listIndex].removeAll { $0.id == id }

    // Update mapping
    idToList.removeValue(forKey: id)
}
```

**Complexity:** O(list_length) to find and remove

---

## List Size Distribution

Ideally, vectors distribute evenly across lists:

```
Ideal (uniform):              Real (skewed):

List 0: ████████ (100)       List 0: ████████████████ (160)
List 1: ████████ (100)       List 1: ████████ (80)
List 2: ████████ (100)       List 2: ███ (30)
List 3: ████████ (100)       List 3: ████████████████████ (200)
List 4: ████████ (100)       List 4: ████ (30)
```

**Skewed distribution causes problems:**
- Large lists: Slow to scan
- Small lists: Wasted centroids, missed vectors

### Causes of Skew

1. **Non-uniform data**: Embedding spaces often have dense regions
2. **Poor centroid placement**: K-means can converge to local optima
3. **Insufficient training data**: Centroids don't represent full distribution

### Mitigation

```swift
// VectorIndex: Re-assign after optimize()
public func optimize() async throws {
    // Rebuild centroids with current data
    let k = max(1, min(config.nlist, store.count))
    centroids = try await kmeans(centroids: initialCentroids, maxIterations: 20)

    // Re-assign all vectors to new centroids
    lists = Array(repeating: [], count: centroids.count)
    for (id, (vec, _)) in store {
        if let ci = nearestCentroidIndex(for: vec) {
            lists[ci].append(id)
        }
    }
}
```

---

## Memory Layout

For cache efficiency, vectors within a list should be contiguous:

```
Poor layout (scattered):       Good layout (contiguous):

List 0 vectors:                List 0 vectors:
  vec_7  at address 0x1000      vec_7  at address 0x1000
  vec_12 at address 0x5000      vec_12 at address 0x1200
  vec_45 at address 0x9000      vec_45 at address 0x1400
        ↑                             ↑
  Cache misses galore!          Sequential access = prefetch friendly
```

VectorIndex achieves this through careful storage:

```swift
// 📍 See: Sources/VectorIndex/Kernels/IVFAppend.swift

// Vectors are appended contiguously within each list
// List storage: [list0_vecs...][list1_vecs...][list2_vecs...]
```

---

## In VectorIndex

VectorIndex's IVF implementation:

```swift
// 📍 See: Sources/VectorIndex/IVFIndex.swift

public actor IVFIndex: VectorIndexProtocol {
    // Centroids: [[Float]] - k-means cluster centers
    private var centroids: [[Float]] = []

    // Inverted lists: [[VectorID]] - IDs per cluster
    private var lists: [[VectorID]] = []

    // ID to list mapping for fast deletion
    private var idToListIndex: [VectorID: Int] = [:]

    // Vector storage (separate from lists)
    private var store: [VectorID: ([Float], [String: String]?)] = [:]
}
```

The Kernel #30 path provides optimized storage:

```swift
// 📍 See: Sources/VectorIndex/Kernels/IVFAppend.swift

// High-performance ingestion with optional persistence
public func ingestFlat(
    listIDs: [Int32],
    externalIDs: [UInt64],
    vectors: [Float],
    opts: IVFAppendOpts?
) async throws
```

---

## 🔗 VectorCore Connection

List scanning is where VectorCore shines:

```swift
// 🔗 VectorCore: Scanning a list is like mini brute-force

for listIndex in probeLists {
    // Each list is a small flat index
    // VectorCore's SIMD distance kernels apply directly

    for (id, vec) in lists[listIndex] {
        let dist = distance(query, vec, metric: metric)
        candidates.append((id, dist))
    }
}
```

The contiguous layout enables:
- **Prefetching**: CPU predicts sequential access
- **SIMD batching**: Process multiple vectors per iteration
- **Cache efficiency**: Minimize memory stalls

---

## Key Takeaways

1. **Inverted lists map clusters to vectors.** Given a cluster, get all its members instantly.

2. **Insertion assigns to nearest centroid.** O(nlist × d) to find, O(1) to append.

3. **Search scans selected lists.** Only examine vectors in the nprobe nearest clusters.

4. **Even distribution matters.** Skewed lists cause performance variance.

5. **Contiguous storage helps.** Memory layout affects cache performance.

---

## Next Up

Now let's explore the key tradeoff in IVF—how nprobe controls recall and speed:

**[→ The nprobe Tradeoff](./03-Nprobe-Tradeoff.md)**
