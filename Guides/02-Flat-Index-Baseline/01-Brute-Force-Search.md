# Brute-Force Search

> **Reading time:** 10 minutes
> **Prerequisites:** [Chapter 1: Similarity Search Fundamentals](../01-Similarity-Search-Fundamentals/README.md)

---

## The Concept

**Brute-force search** (also called linear scan or exhaustive search) is the simplest nearest neighbor algorithm:

```
Algorithm: Brute-Force k-NN Search

Input:  query vector q
        dataset X = {x₁, x₂, ..., xₙ}
        k (number of neighbors)

1. For each vector xᵢ in X:
     - Compute distance d(q, xᵢ)
     - Add (xᵢ, distance) to results

2. Sort results by distance (ascending)

3. Return top k results
```

That's it. No index structure, no preprocessing—just compute all distances and sort.

---

## Complexity Analysis

### Time Complexity

```
Distance computation: O(d) per vector
Total distances:      O(n × d)
Sorting:              O(n log n)

Overall: O(n × d + n log n) = O(n × d) when d > log n
```

For typical embedding dimensions (d = 512-1536), the distance computation dominates.

### Space Complexity

```
Storage: O(n × d) for the vectors
Query:   O(n) for temporary results (before top-k selection)
```

### Practical Numbers

```
n = 1,000,000 vectors
d = 512 dimensions
Single query: 512 million FLOPs

At 100 GFLOP/s (typical SIMD throughput):
  = 5.12 ms per query
  = ~200 QPS theoretical max
```

---

## Why It's Not Always Bad

### Small Dataset Advantage

For small datasets, the overhead of building an index exceeds its benefit:

```
Dataset: 10,000 vectors × 512 dimensions

Brute-force search: ~0.5ms per query
HNSW build time:    ~5-10 seconds
HNSW search:        ~0.1ms per query

Break-even point: 10 seconds / 0.4ms = 25,000 queries
```

If you're running fewer than 25,000 queries, FlatIndex is faster *overall*.

### Exact Guarantee

When precision matters more than speed:

```
Medical diagnosis: Wrong result = potential harm
Legal discovery:   Missing document = liability
Financial audit:   Must examine all matches

FlatIndex: Guaranteed 100% recall
HNSW:      99.5% recall (0.5% might include critical result)
```

### No Build Phase

FlatIndex is immediately ready after inserts:

```swift
let index = FlatIndex(dimension: 512, metric: .cosine)

// Insert and immediately search - no optimize() needed
await index.insert(id: "doc1", vector: embedding, metadata: nil)
let results = await index.search(query: q, k: 10, filter: nil)
```

Other indices require training or graph construction before they're effective.

---

## The Algorithm in Detail

### Step 1: Distance Computation

The inner loop dominates runtime:

```swift
// 📍 See: Sources/VectorIndex/FlatIndex.swift:48-66

public func search(query: [Float], k: Int, filter: ...) async throws -> [SearchResult] {
    guard k > 0 else { return [] }
    var results: [SearchResult] = []
    results.reserveCapacity(min(k, vectors.count))

    for (id, (vec, meta)) in vectors {
        // Optional filter check
        if let filter = filter, !filter(meta) { continue }

        // Distance computation - this is the hot path
        let d = distance(query, vec, metric: metric)

        results.append(SearchResult(id: id, score: d))
    }

    // Sort and truncate
    results.sort { $0.score < $1.score }
    if results.count > k { results.removeLast(results.count - k) }
    return results
}
```

### Step 2: The Distance Function

```swift
// 📍 See: Sources/VectorIndex/DistanceUtils.swift

@inlinable
public func distance(_ a: [Float], _ b: [Float], metric: SupportedDistanceMetric) -> Float {
    switch metric {
    case .euclidean:
        return euclideanDistance(a, b)
    case .cosine:
        return cosineDistance(a, b)
    case .dotProduct:
        return -dotProduct(a, b)  // Negate for "lower is better"
    // ...
    }
}
```

### Step 3: Sorting

Swift's built-in sort is highly optimized (introsort hybrid):

```swift
results.sort { $0.score < $1.score }

// For n = 1,000,000:
//   O(n log n) = 20 million comparisons
//   ~10-20ms on modern CPU
```

---

## Optimizations in VectorIndex

### Parallel Batch Search

```swift
// 📍 See: Sources/VectorIndex/FlatIndex.swift:76-101

public func batchSearch(queries: [[Float]], k: Int, ...) async throws -> [[SearchResult]] {
    return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
        for (queryIndex, query) in queries.enumerated() {
            group.addTask {
                try Self.performFlatSearch(query: query, queryIndex: queryIndex, ...)
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

Each query runs on a separate task, utilizing all CPU cores.

### Early Filter Elimination

```swift
for (id, (vec, meta)) in vectors {
    // Check filter BEFORE computing distance
    if let filter = filter, !filter(meta) { continue }

    let d = distance(query, vec, metric: metric)  // Expensive
    // ...
}
```

Filtering before distance computation avoids wasted work.

---

## 🔗 VectorCore Connection

The distance computation is where VectorCore shines:

```swift
// 🔗 VectorCore: SIMD-accelerated distance

// Instead of scalar loop:
var sum: Float = 0
for i in 0..<d {
    let diff = a[i] - b[i]
    sum += diff * diff
}

// VectorCore uses SIMD4:
let chunks = d / 4
var acc = SIMD4<Float>.zero
for i in 0..<chunks {
    let diff = a_simd4[i] - b_simd4[i]
    acc += diff * diff
}
let sum = acc.sum()

// 4× fewer loop iterations, same result
```

For a 512-dimensional vector:
- Scalar: 512 subtractions + 512 multiplies + 511 additions
- SIMD4: 128 SIMD ops total

---

## Key Takeaways

1. **Brute-force is O(n × d).** Linear in dataset size, linear in dimension.

2. **It's optimal for small datasets.** Index build overhead exceeds search savings below ~10-50k vectors.

3. **It guarantees 100% recall.** When exactness matters, FlatIndex is the only choice.

4. **Distance computation dominates.** This is where SIMD optimization has the biggest impact.

5. **Parallel batch search helps.** Multiple queries can run concurrently across CPU cores.

---

## Next Up

The full sort in brute-force is wasteful—we only need top-k, not full ordering. Let's optimize:

**[→ Top-K Selection](./02-Top-K-Selection.md)**
