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

## Journaling Quickstart

VectorIndex supports metadata-aware filtering for journaling workflows via a small filter DSL.

```swift
import VectorIndex

// 1) Build an index and ingest entries with metadata (ISO-8601 date, comma-separated tags)
let index = HNSWIndex(dimension: 768, metric: .cosine, config: .init(m: 16, efConstruction: 200, efSearch: 64))
try await index.batchInsert([
    (id: "2025-01-01", vector: v1, metadata: [
        "date": "2025-01-01T09:12:00Z",
        "tags": "journal, work",
        "title": "New Year, new notes"
    ]),
    (id: "2025-01-15", vector: v2, metadata: [
        "date": "2025-01-15T18:05:00Z",
        "tags": "journal, mood",
        "title": "Mid-month reflections"
    ]),
])

// 2) Create a filter for January entries tagged with "journal" but not "private"
let fmt = ISO8601DateFormatter(); fmt.formatOptions = [.withInternetDateTime]
let start = fmt.date(from: "2025-01-01T00:00:00Z")!
let end   = fmt.date(from: "2025-01-31T23:59:59Z")!

let filter = JournalFilter()
    .dateBetween(start, end)
    .includingTags(["journal"])    // any by default; pass requireAll: true to require all tags
    .excludingTags(["private"])     // reject entries containing these tags
    .build()

// 3) Use the filter for both single-query and batch search
let top = try await index.search(query: someEmbedding, k: 20, filter: filter)
let batched = try await index.batchSearch(queries: [q1, q2, q3], k: 10, filter: filter)
```

Notes:
- Default metadata keys are `date` and `tags` (comma-delimited). Override with `setKeys(dateKey:tagsKey:delimiter:)` if your schema differs.
- `allowMissingKeys(true)` includes entries missing keys; invalid dates are treated as non-matching.
- The closure from `JournalFilter.build()` is `@Sendable` and safe to pass into `search`/`batchSearch`.

## Advanced: IVF Select Batch Sugar

For lower-level pipelines (e.g., custom IVF workflows), a convenience API returns per‑query arrays for IVF centroid selection:

```swift
// Queries Q [b×d] and coarse centroids [kc×d]
let b = 8, d = 128, kc = 2048, nprobe = 32
let (ids2D, scores2D) = IndexOps.Batch.ivfSelectNprobe(
    Q: queriesFlat, b: b, d: d,
    centroids: centroids, kc: kc,
    metric: .l2, nprobe: nprobe,
    opts: IVFSelectOpts(),
    gatherScores: true
)

// ids2D: [[Int32]] sized [b][nprobe]; scores2D: [[Float]]? (nil if gatherScores=false)
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
