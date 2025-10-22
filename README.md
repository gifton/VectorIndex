# VectorIndex

CPU‑first vector index algorithms for the VectorStoreKit suite. Depends only on `VectorCore` and provides clean seams for optional GPU acceleration later.

## Features

- Flat index (exact search)
- HNSW index (approximate; hierarchical graph)
- IVF index (approximate; inverted file w/ KMeans centroids)
- Seeded KMeans++ initialization (IVF)
- JSON persistence (versioned) for all indices
- Compaction (`compact()`) for HNSW/IVF
- Strict Swift 6 concurrency (actors, @Sendable)

## Usage

```swift
import VectorIndex

// Flat exact index
let flat = FlatIndex(dimension: 768, metric: .euclidean)
try await flat.insert(id: "a", vector: embeddingA, metadata: ["label":"x"])
let res = try await flat.search(query: queryEmbedding, k: 10, filter: nil)

// HNSW approximate index
let hnsw = HNSWIndex(dimension: 768, metric: .euclidean, config: .init(m: 16, efConstruction: 200, efSearch: 64))
try await hnsw.batchInsert(items)
let nn = try await hnsw.search(query: queryEmbedding, k: 10, filter: nil)

// IVF approximate index
let ivf = IVFIndex(dimension: 768, metric: .euclidean, config: .init(nlist: 256, nprobe: 8))
try await ivf.batchInsert(items)
try await ivf.optimize() // builds centroids + lists
let ivfRes = try await ivf.search(query: queryEmbedding, k: 10, filter: nil)
```

## Persistence

```swift
let url = URL(fileURLWithPath: "/tmp/index.json")
try await hnsw.save(to: url)
let loaded = try await HNSWIndex.load(from: url)
```

Notes:
- Format is JSON (versioned). HNSW graph structure is rebuilt on load via re‑insertion. IVF centroids/lists are rebuilt via `optimize()` in `load()`.

## API Surface

- `insert(id:vector:metadata:)`, `batchInsert(...)`
- `remove(id:)`, `batchRemove([...])`
- `search(query:k:filter:)`, `batchSearch(queries:k:filter:)`
- `contains(id:)`, `update(id:vector:metadata:)`
- `statistics() -> IndexStats`
- `save(to:)`, `static load(from:)`
- `compact()`

## Parameter Tuning

HNSW:
- `m` (max connections): higher → better recall, more memory
- `efConstruction`: higher → better graph quality, slower build
- `efSearch`: higher → better recall, higher query latency

IVF:
- `nlist`: number of centroids; larger partitions data more finely
- `nprobe`: probed lists per query; higher → better recall, slower search
- `seed`: RNG seed for KMeans++ (deterministic tests)

## Error Handling

VectorIndex 0.1.0+ uses structured error handling with `VectorIndexError`:

```swift
do {
    let index = try IVFListHandle(k_c: 10, m: 0, d: 128, opts: .default)
    let stats = try kmeansPlusPlusSeed(data: vectors, count: n, dimension: d, k: 10, ...)
} catch let error as VectorIndexError {
    // Rich error information
    print("Error: \(error.message)")
    print("Recovery: \(error.recoveryMessage)")
    print("Recoverable: \(error.kind.isRecoverable)")

    // Structured metadata for debugging
    print("Operation: \(error.context.operation)")
    print("Details: \(error.context.additionalInfo)")
}
```

See [ERRORS.md](ERRORS.md) for complete error handling guide.

## Concurrency

- All indices are actors; methods are `async` and actor‑isolated.
- Filter closures are `@Sendable`; avoid blocking IO inside filters.

## Acceleration

- The package is CPU‑only by design. A future bridge (e.g., `VectorIndexAccelerate`) can inject GPU‑accelerated distance providers for IVF building and scoring.

## Known Issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for documented issues in 0.1.0-alpha.
