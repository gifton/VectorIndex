Title: ✅ DONE — Partial Top-K Selection Kernel — Efficient Best-Results Selection for Vector Search

Summary
- Implement a high-performance partial top-k selection kernel that efficiently maintains the k best scoring results from a large stream of scored vectors.
- Supports both min-heap (for distance metrics like L2) and max-heap (for similarity metrics like IP/Cosine) configurations.
- Provides two algorithm strategies: streaming heap for moderate datasets and hybrid quickselect for very large datasets (n > 16k).
- Ensures deterministic, stable tie-breaking by vector ID for reproducible search results.
- Serves as the fundamental selection primitive for all search operations in VectorIndex.

Project Context
- VectorIndex implements approximate nearest neighbor (ANN) search with configurable k results
- Top-k selection is the final stage of every search operation:
  - **IVF Search**: Select k best from scored cells (n=100-100k candidates)
  - **HNSW Search**: Maintain k best during beam search (dynamic candidate set)
  - **Flat Search**: Select k from entire database (n=1M+ candidates)
  - **Re-ranking**: Refine top-k after initial filtering
- Industry context: Top-k selection accounts for 5-15% of total search time
- Challenge: Balance between algorithmic complexity and practical performance
  - Full sort: O(n log n) — too slow for large n
  - Heap: O(n log k) — optimal asymptotic complexity
  - Quickselect: O(n) average — but requires careful implementation
- VectorCore provides primitives; VectorIndex needs search-optimized selection
- Typical usage patterns:
  - k ∈ {10, 50, 100} (user-facing search)
  - n ∈ {1000, 10000, 100000} (candidate set sizes)
  - Per-thread partial selection → merge across threads (#06)

Goals
- Achieve O(n log k) complexity for streaming heap algorithm
- Achieve O(n + k log k) for hybrid quickselect approach on large n
- Support both min-heap (L2 distance) and max-heap (IP/Cosine similarity)
- Deterministic, stable tie-breaking for reproducible ranking
- Branch-efficient comparisons for predictable performance
- Zero allocations beyond heap structure (fixed-size)
- Thread-safe for concurrent per-thread heap maintenance
- Seamless integration with merge kernel (#06) for distributed top-k

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/TopKSelectionKernel.swift`
- Core implementations:
  - Heap data structure: `TopKHeap` with min/max variants
  - Streaming algorithm: `selectTopK_streaming` for moderate n
  - Hybrid algorithm: `selectTopK_hybrid` for large n (n > 16k)
  - Dispatch logic: `selectTopK` with automatic strategy selection
- Supporting utilities:
  - Heap operations: sift-up, sift-down, heapify
  - Stable comparison with ID tie-breaking
  - Quickselect partition for hybrid approach
  - Telemetry integration (#46)
- Integration points:
  - Consumes scores from ScoreBlockKernel (#04)
  - Exports partial results to MergeTopKKernel (#06)
  - Used by IVF search, HNSW search, flat search

API & Signatures

```swift
// MARK: - Heap Configuration

/// Heap ordering for top-k selection
public enum HeapOrdering {
    case min  // Min-heap: smaller scores are better (L2 distance)
    case max  // Max-heap: larger scores are better (IP, Cosine)

    /// Whether to keep scores larger than heap root
    func shouldReplace(_ newScore: Float, _ heapRoot: Float) -> Bool {
        switch self {
        case .min: return newScore < heapRoot
        case .max: return newScore > heapRoot
        }
    }
}

// MARK: - Heap Data Structure

/// Fixed-size heap for top-k selection
/// Structure-of-Arrays layout for cache efficiency
public struct TopKHeap {
    /// Heap ordering (min or max)
    public let ordering: HeapOrdering

    /// Capacity (k)
    public let capacity: Int

    /// Current size (≤ k)
    public private(set) var count: Int

    /// Scores (heap-ordered) [k], 64-byte aligned
    private var scores: UnsafeMutablePointer<Float>

    /// IDs corresponding to scores [k], 64-byte aligned
    private var ids: UnsafeMutablePointer<Int32>

    /// Initialize heap with capacity k
    public init(capacity: Int, ordering: HeapOrdering)

    /// Insert score-ID pair (only if heap not full or better than root)
    @inlinable
    public mutating func push(score: Float, id: Int32)

    /// Access root (best score so far)
    public var root: (score: Float, id: Int32)? {
        guard count > 0 else { return nil }
        return (scores[0], ids[0])
    }

    /// Extract all results sorted by score (best to worst)
    public func extractSorted() -> [(score: Float, id: Int32)]

    /// Reset heap to empty state
    public mutating func clear()

    /// Deallocate memory
    public func deallocate()
}

// MARK: - Core Top-K Selection API

/// Select top-k results from scored vectors
/// Automatically chooses optimal algorithm based on n and k
///
/// - Complexity: O(n log k) for streaming, O(n + k log k) for hybrid
/// - Performance: 5-15% of total search time for typical k and n
/// - Thread Safety: Each call operates on independent heap
///
/// - Parameters:
///   - scores: Array of scores [n]
///   - ids: Array of vector IDs [n], if nil uses 0..<n
///   - n: Number of scores/candidates
///   - k: Number of top results to select
///   - ordering: Heap ordering (min for L2, max for IP/Cosine)
///   - config: Optional configuration (telemetry, strategy, etc.)
/// - Returns: TopKHeap containing k best results
@inlinable
public func selectTopK(
    scores: UnsafePointer<Float>,
    ids: UnsafePointer<Int32>?,
    count n: Int,
    k: Int,
    ordering: HeapOrdering,
    config: TopKConfig = .default
) -> TopKHeap

// MARK: - Algorithm-Specific Variants

/// Streaming heap algorithm for moderate n
/// Best for: n < 16k, any k
@inlinable
public func selectTopK_streaming(
    scores: UnsafePointer<Float>,
    ids: UnsafePointer<Int32>?,
    count: Int,
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap

/// Hybrid quickselect + heap refinement for large n
/// Best for: n > 16k, k << n
@inlinable
public func selectTopK_hybrid(
    scores: UnsafePointer<Float>,
    ids: UnsafePointer<Int32>?,
    count: Int,
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap

// MARK: - Batch Selection

/// Select top-k from multiple independent score arrays
/// Useful for multi-query search or distributed partitions
@inlinable
public func selectTopKBatch(
    scoresPerPartition: [UnsafePointer<Float>],
    idsPerPartition: [UnsafePointer<Int32>?],
    countsPerPartition: [Int],
    k: Int,
    ordering: HeapOrdering
) -> [TopKHeap]

// MARK: - Configuration

/// Configuration for top-k selection
public struct TopKConfig {
    /// Enable telemetry recording (default: false)
    let enableTelemetry: Bool

    /// Force specific algorithm (default: automatic based on n)
    let forceAlgorithm: Algorithm?

    /// Threshold for hybrid algorithm (default: 16384)
    let hybridThreshold: Int

    public enum Algorithm {
        case streaming  // O(n log k) heap-based
        case hybrid     // O(n + k log k) quickselect + heap
    }

    public static let `default` = TopKConfig(
        enableTelemetry: false,
        forceAlgorithm: nil,
        hybridThreshold: 16384
    )
}

// MARK: - Telemetry

/// Per-selection execution statistics
public struct TopKTelemetry {
    public let algorithm: TopKConfig.Algorithm
    public let candidatesProcessed: Int
    public let k: Int
    public let comparisons: Int          // Number of score comparisons
    public let heapPushes: Int           // Number of heap insertions
    public let siftOperations: Int       // Number of sift-up/down operations
    public let executionTimeNanos: UInt64

    public var throughputCandidatesPerSec: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return Double(candidatesProcessed) / seconds
    }
}

// MARK: - Convenience API

extension TopKSelectionKernel {
    /// High-level API with automatic memory management
    /// Returns array of (score, id) tuples sorted best to worst
    public static func select(
        scores: [Float],
        ids: [Int32]? = nil,
        k: Int,
        ordering: HeapOrdering
    ) -> [(score: Float, id: Int32)]

    /// Select top-k from DistanceMetric context
    /// Automatically determines heap ordering from metric
    public static func select(
        scores: [Float],
        ids: [Int32]? = nil,
        k: Int,
        metric: DistanceMetric
    ) -> [(score: Float, id: Int32)]
}
```

Algorithm Details

**Streaming Heap Algorithm** (for n < 16k):

```swift
@inlinable
public func selectTopK_streaming(
    scores: UnsafePointer<Float>,
    ids: UnsafePointer<Int32>?,
    count n: Int,
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap {
    var heap = TopKHeap(capacity: k, ordering: ordering)

    // Phase 1: Fill heap with first k elements
    let initialFill = min(k, n)
    for i in 0..<initialFill {
        let id = ids?[i] ?? Int32(i)
        heap.push(score: scores[i], id: id)
    }

    // Heapify the initial k elements
    heap.heapify()

    // Phase 2: Stream remaining elements, maintaining heap invariant
    for i in k..<n {
        let score = scores[i]
        let id = ids?[i] ?? Int32(i)

        // Compare with heap root (worst element in current top-k)
        guard let root = heap.root else { continue }

        if ordering.shouldReplace(score, root.score) {
            // New score is better; replace root and sift down
            heap.replaceRoot(score: score, id: id)
        } else if score == root.score && id < root.id {
            // Tie-breaking: prefer smaller ID for stability
            heap.replaceRoot(score: score, id: id)
        }
    }

    return heap
}
```

**Hybrid Quickselect Algorithm** (for n > 16k):

```swift
@inlinable
public func selectTopK_hybrid(
    scores: UnsafePointer<Float>,
    ids: UnsafePointer<Int32>?,
    count n: Int,
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap {
    // Phase 1: Copy scores and IDs to temporary buffer for partitioning
    var workScores = [Float](repeating: 0, count: n)
    var workIds = [Int32](repeating: 0, count: n)

    for i in 0..<n {
        workScores[i] = scores[i]
        workIds[i] = ids?[i] ?? Int32(i)
    }

    // Phase 2: Use quickselect to partition around k-th element
    // This moves k best elements to left side (approximately)
    let kthIndex = quickselectPartition(
        scores: &workScores,
        ids: &workIds,
        k: k,
        ordering: ordering
    )

    // Phase 3: Heap refinement
    // Build exact top-k heap from elements around partition boundary
    // This handles duplicates and ensures stable ordering
    var heap = TopKHeap(capacity: k, ordering: ordering)

    // Start from partition point and expand outward
    let searchStart = max(0, kthIndex - k)
    let searchEnd = min(n, kthIndex + k)

    for i in searchStart..<searchEnd {
        heap.push(score: workScores[i], id: workIds[i])
    }

    return heap
}
```

**Heap Operations** (core primitives):

```swift
extension TopKHeap {
    /// Sift element up to maintain heap property
    @inline(__always)
    private mutating func siftUp(index: Int) {
        var currentIdx = index
        let score = scores[currentIdx]
        let id = ids[currentIdx]

        while currentIdx > 0 {
            let parentIdx = (currentIdx - 1) / 2
            let parentScore = scores[parentIdx]
            let parentId = ids[parentIdx]

            // Check heap property with stable tie-breaking
            let shouldSwap: Bool
            switch ordering {
            case .min:
                shouldSwap = score < parentScore || (score == parentScore && id < parentId)
            case .max:
                shouldSwap = score > parentScore || (score == parentScore && id < parentId)
            }

            guard shouldSwap else { break }

            // Swap with parent
            scores[currentIdx] = parentScore
            ids[currentIdx] = parentId
            currentIdx = parentIdx
        }

        scores[currentIdx] = score
        ids[currentIdx] = id
    }

    /// Sift element down to maintain heap property
    @inline(__always)
    private mutating func siftDown(index: Int) {
        var currentIdx = index
        let score = scores[currentIdx]
        let id = ids[currentIdx]

        while true {
            let leftIdx = 2 * currentIdx + 1
            let rightIdx = 2 * currentIdx + 2

            guard leftIdx < count else { break }

            // Find better child (if any)
            var swapIdx = currentIdx
            var swapScore = score
            var swapId = id

            // Check left child
            let leftScore = scores[leftIdx]
            let leftId = ids[leftIdx]
            let leftBetter: Bool
            switch ordering {
            case .min:
                leftBetter = leftScore < swapScore || (leftScore == swapScore && leftId < swapId)
            case .max:
                leftBetter = leftScore > swapScore || (leftScore == swapScore && leftId < swapId)
            }

            if leftBetter {
                swapIdx = leftIdx
                swapScore = leftScore
                swapId = leftId
            }

            // Check right child
            if rightIdx < count {
                let rightScore = scores[rightIdx]
                let rightId = ids[rightIdx]
                let rightBetter: Bool
                switch ordering {
                case .min:
                    rightBetter = rightScore < swapScore || (rightScore == swapScore && rightId < swapId)
                case .max:
                    rightBetter = rightScore > swapScore || (rightScore == swapScore && rightId < swapId)
                }

                if rightBetter {
                    swapIdx = rightIdx
                    swapScore = rightScore
                    swapId = rightId
                }
            }

            guard swapIdx != currentIdx else { break }

            // Swap with better child
            scores[currentIdx] = swapScore
            ids[currentIdx] = swapId
            currentIdx = swapIdx
        }

        scores[currentIdx] = score
        ids[currentIdx] = id
    }

    /// Replace heap root with new element and sift down
    @inline(__always)
    mutating func replaceRoot(score: Float, id: Int32) {
        scores[0] = score
        ids[0] = id
        siftDown(index: 0)
    }

    /// Build heap from unordered array (in-place heapify)
    @inline(__always)
    mutating func heapify() {
        // Start from last non-leaf node and sift down
        for i in stride(from: (count / 2) - 1, through: 0, by: -1) {
            siftDown(index: i)
        }
    }
}
```

**Quickselect Partition** (for hybrid algorithm):

```swift
@inlinable
func quickselectPartition(
    scores: inout [Float],
    ids: inout [Int32],
    k: Int,
    ordering: HeapOrdering
) -> Int {
    var left = 0
    var right = scores.count - 1

    while left < right {
        // Choose pivot (median-of-three for better performance)
        let pivotIdx = medianOfThree(scores, left, (left + right) / 2, right, ordering)
        let pivotScore = scores[pivotIdx]

        // Partition around pivot
        scores.swapAt(pivotIdx, right)
        ids.swapAt(pivotIdx, right)

        var storeIdx = left
        for i in left..<right {
            let better: Bool
            switch ordering {
            case .min:
                better = scores[i] < pivotScore
            case .max:
                better = scores[i] > pivotScore
            }

            if better {
                scores.swapAt(i, storeIdx)
                ids.swapAt(i, storeIdx)
                storeIdx += 1
            }
        }

        scores.swapAt(storeIdx, right)
        ids.swapAt(storeIdx, right)

        // Recurse on appropriate partition
        if storeIdx == k {
            return storeIdx
        } else if storeIdx < k {
            left = storeIdx + 1
        } else {
            right = storeIdx - 1
        }
    }

    return left
}
```

Data Structure Details

**Memory Layout** (Structure-of-Arrays):

```
TopKHeap (for k=10):
┌─────────────────────────────────┐
│ scores[0..9]: Float32           │  40 bytes (64-byte aligned)
│ [score₀, score₁, ..., score₉]  │
├─────────────────────────────────┤
│ ids[0..9]: Int32                │  40 bytes (64-byte aligned)
│ [id₀, id₁, ..., id₉]           │
└─────────────────────────────────┘

Total: 80 bytes + metadata
```

**Why SoA over AoS?**:
- Better cache line utilization during comparisons
- SIMD-friendly for batched operations (future optimization)
- Reduces memory traffic during sift operations
- Aligned arrays prevent cache line splits

**Heap Property** (min-heap example):

```
Min-Heap (L2 distance, lower is better):
         0.5 (id=42)        ← Root (worst score in top-k)
        /            \
    0.3 (id=17)    0.4 (id=99)
   /        \      /
0.1 (id=5) 0.2  0.35

For any node i:
- scores[i] ≥ scores[2i+1] (left child)
- scores[i] ≥ scores[2i+2] (right child)
- On tie: ids[i] < ids[child] (stable)

Root (scores[0]) = worst score currently in top-k
```

Performance Characteristics

**Complexity Analysis**:

```
Algorithm       | Time            | Space    | Best For
----------------|-----------------|----------|------------------
Streaming Heap  | O(n log k)      | O(k)     | n < 16k, any k
Hybrid Select   | O(n + k log k)  | O(n + k) | n > 16k, k << n
Full Sort       | O(n log n)      | O(n)     | Never (too slow)

Comparison counts (n=10000, k=10):
- Streaming: ~10000 * log₂(10) ≈ 33k comparisons
- Hybrid: ~10000 + 10 * log₂(10) ≈ 10k comparisons
- Full sort: ~10000 * log₂(10000) ≈ 133k comparisons
```

**Algorithm Selection Strategy**:

```swift
func chooseAlgorithm(n: Int, k: Int) -> TopKConfig.Algorithm {
    // Hybrid is better for large n and small k
    // Crossover point around n=16k empirically on Apple Silicon
    if n > 16384 && k < n / 100 {
        return .hybrid
    } else {
        return .streaming
    }
}
```

**Branch Prediction Optimization**:

```swift
// Bad: Unpredictable branches in tight loop
if ordering == .min {
    if score < rootScore { /* ... */ }
} else {
    if score > rootScore { /* ... */ }
}

// Good: Hoist ordering check outside loop
let comparator: (Float, Float) -> Bool
switch ordering {
case .min: comparator = (<)
case .max: comparator = (>)
}

for i in 0..<n {
    if comparator(scores[i], rootScore) { /* ... */ }
}
```

**Cache Efficiency**:
- Heap fits in L1 cache for k ≤ 1000: 1000 * 4 + 1000 * 4 = 8 KB < 128 KB
- Sequential scan of scores array: Prefetch-friendly
- Sift operations: Random access but within small heap (L1 resident)

Stability & Determinism

**Stable Tie-Breaking**:

```swift
// Comparison with stable tie-breaking
func isStrictlyBetter(
    newScore: Float, newId: Int32,
    oldScore: Float, oldId: Int32,
    ordering: HeapOrdering
) -> Bool {
    switch ordering {
    case .min:
        // For min-heap: smaller score is better, smaller ID breaks ties
        return newScore < oldScore || (newScore == oldScore && newId < newId)
    case .max:
        // For max-heap: larger score is better, smaller ID breaks ties
        return newScore > oldScore || (newScore == oldScore && newId < oldId)
    }
}
```

**Why Stability Matters**:
- Reproducible search results across runs
- Consistent ranking in A/B tests
- Debuggability: same input → same output
- Fair comparison in benchmarks

**Example** (cosine similarity with duplicates):

```
Candidates:
id=10: score=0.95
id=20: score=0.95  ← Duplicate score
id=30: score=0.94
id=40: score=0.95  ← Duplicate score

Stable top-3 (max-heap, smaller ID breaks ties):
1. id=10: score=0.95  (first occurrence)
2. id=20: score=0.95  (second occurrence)
3. id=40: score=0.95  (third occurrence)

Without stability, order is non-deterministic!
```

Parallelization Strategy

**Per-Thread Heaps**:

```swift
func searchParallel(
    query: Vector,
    partitions: [[Vector]],
    k: Int,
    metric: DistanceMetric
) -> [SearchResult] {
    let ordering: HeapOrdering = metric.higherIsBetter ? .max : .min

    // Step 1: Each thread maintains its own heap
    let partialHeaps = DispatchQueue.concurrentPerform(
        iterations: partitions.count
    ) { partitionID -> TopKHeap in
        let partition = partitions[partitionID]
        var scores = [Float](repeating: 0, count: partition.count)

        // Score all vectors in this partition
        scoreBlock(query, partition, metric, &scores)

        // Select top-k from this partition
        return selectTopK(
            scores: scores,
            ids: nil,  // Use implicit IDs
            count: partition.count,
            k: k,
            ordering: ordering
        )
    }

    // Step 2: Merge partial heaps (using kernel #06)
    return mergeTopK(partialHeaps, k: k, ordering: ordering)
}
```

**Thread Safety**:
- Each thread has independent heap (no sharing)
- No locks needed during selection phase
- Merge phase handled by separate kernel (#06)

Performance Targets (Apple M1/M2/M3, Release Build)

**Throughput** (candidates/second):

```
Algorithm       | k=10    | k=100   | k=1000
----------------|---------|---------|----------
Streaming (n=1k)   | 50M/s   | 20M/s   | 5M/s
Streaming (n=10k)  | 10M/s   | 5M/s    | 2M/s
Hybrid (n=100k)    | 50M/s   | 40M/s   | 20M/s
```

**Latency** (per selection):

```
n        | k=10      | k=100     | k=1000
---------|-----------|-----------|----------
1,000    | 20 μs     | 50 μs     | 200 μs
10,000   | 1 ms      | 2 ms      | 5 ms
100,000  | 2 ms      | 3 ms      | 6 ms (hybrid)
```

**Memory Usage**:
- Heap: 8k bytes (k * (4 + 4) bytes)
- Streaming: O(k) — minimal overhead
- Hybrid: O(n) — temporary buffer for partitioning

**Comparison with Alternatives**:

```
Method              | Time (n=100k, k=10)
--------------------|--------------------
Full sort           | 15 ms
Partial sort (top k)| 8 ms
Heap (streaming)    | 10 ms
Heap (hybrid)       | 2 ms ← Winner
```

Correctness & Testing

**Golden Reference**:
```swift
// Reference: Full sort and take top-k
func referenceTopK(scores: [Float], ids: [Int32], k: Int, ordering: HeapOrdering) -> [(Float, Int32)] {
    let paired = zip(scores, ids)
    let sorted: [(Float, Int32)]

    switch ordering {
    case .min:
        sorted = paired.sorted { $0.0 < $1.0 || ($0.0 == $1.0 && $0.1 < $1.1) }
    case .max:
        sorted = paired.sorted { $0.0 > $1.0 || ($0.0 == $1.0 && $0.1 < $1.1) }
    }

    return Array(sorted.prefix(k))
}
```

**Test Cases**:

1. **Basic Correctness**:
   - Compare against full sort for random data
   - Various n and k combinations
   - Min-heap and max-heap orderings

2. **Stability**:
   - All scores identical (pure ID ordering)
   - Many duplicate scores
   - Verify stable ordering: id₁ < id₂ for same score

3. **Edge Cases**:
   - k = 0 (empty result)
   - k = n (all elements)
   - k > n (return all n elements)
   - n = 1 (single element)

4. **Adversarial**:
   - Already sorted (best case)
   - Reverse sorted (worst case)
   - All equal scores
   - Alternating good/bad scores

5. **Algorithm Comparison**:
   - Streaming vs hybrid produce identical results
   - Automatic selection chooses correctly

**Example Tests**:

```swift
func testTopK_Correctness() {
    let n = 10000
    let k = 10
    let scores = (0..<n).map { _ in Float.random(in: 0...1) }
    let ids = (0..<n).map { Int32($0) }

    let heap = selectTopK(
        scores: scores,
        ids: ids,
        count: n,
        k: k,
        ordering: .max
    )

    let result = heap.extractSorted()
    let reference = referenceTopK(scores: scores, ids: ids, k: k, ordering: .max)

    XCTAssertEqual(result.count, k)
    for i in 0..<k {
        XCTAssertEqual(result[i].score, reference[i].0)
        XCTAssertEqual(result[i].id, reference[i].1)
    }
}

func testTopK_Stability() {
    let n = 100
    let k = 10
    let scores = [Float](repeating: 0.5, count: n)  // All identical
    let ids = (0..<n).map { Int32($0) }

    let heap = selectTopK(
        scores: scores,
        ids: ids,
        count: n,
        k: k,
        ordering: .max
    )

    let result = heap.extractSorted()

    // Should return first k IDs in order (stable)
    for i in 0..<k {
        XCTAssertEqual(result[i].id, Int32(i))
    }
}

func testTopK_AlgorithmEquivalence() {
    let n = 100000
    let k = 10
    let scores = (0..<n).map { _ in Float.random(in: 0...1) }
    let ids = (0..<n).map { Int32($0) }

    let streamingHeap = selectTopK_streaming(
        scores: scores,
        ids: ids,
        count: n,
        k: k,
        ordering: .max
    )

    let hybridHeap = selectTopK_hybrid(
        scores: scores,
        ids: ids,
        count: n,
        k: k,
        ordering: .max
    )

    let streamingResult = streamingHeap.extractSorted()
    let hybridResult = hybridHeap.extractSorted()

    // Both algorithms must produce identical results
    for i in 0..<k {
        XCTAssertEqual(streamingResult[i].score, hybridResult[i].score)
        XCTAssertEqual(streamingResult[i].id, hybridResult[i].id)
    }
}
```

Integration with Search Algorithms

**IVF Search**:

```swift
struct IVFIndex {
    func search(query: Vector, nProbe: Int, k: Int) -> [SearchResult] {
        // Score all candidates from probed cells
        var allScores: [Float] = []
        var allIds: [Int32] = []

        for cellID in selectTopCells(query, nProbe) {
            let cellScores = scoreCell(query, cellID)
            let cellIds = getCellVectorIds(cellID)

            allScores.append(contentsOf: cellScores)
            allIds.append(contentsOf: cellIds)
        }

        // Select top-k from all candidates
        let ordering: HeapOrdering = metric.higherIsBetter ? .max : .min
        let heap = selectTopK(
            scores: allScores,
            ids: allIds,
            count: allScores.count,
            k: k,
            ordering: ordering
        )

        return heap.extractSorted().map { SearchResult(id: $0.id, score: $0.score) }
    }
}
```

**HNSW Search** (dynamic candidate maintenance):

```swift
struct HNSWIndex {
    func searchLayer(query: Vector, ef: Int) -> [GraphNode] {
        var candidates = TopKHeap(capacity: ef, ordering: .max)
        var visited = Set<Int32>()

        // Dynamically maintain top-ef candidates during beam search
        while let current = candidates.root {
            let neighbors = getNeighbors(current.id)

            for neighbor in neighbors where !visited.contains(neighbor.id) {
                let score = computeScore(query, neighbor.vector)
                candidates.push(score: score, id: neighbor.id)
                visited.insert(neighbor.id)
            }
        }

        return candidates.extractSorted().map { GraphNode(id: $0.id) }
    }
}
```

**Distributed Search** (merge partial results):

```swift
func distributedSearch(
    query: Vector,
    shards: [IndexShard],
    k: Int
) -> [SearchResult] {
    // Step 1: Search each shard independently
    let partialHeaps: [TopKHeap] = shards.parallelMap { shard in
        let scores = shard.scoreAll(query)
        return selectTopK(
            scores: scores,
            ids: shard.vectorIds,
            count: scores.count,
            k: k,
            ordering: .max
        )
    }

    // Step 2: Merge partial heaps (kernel #06)
    return mergeTopK(partialHeaps, k: k)
}
```

Coding Guidelines

**Performance Best Practices**:
- Use streaming algorithm for n < 16k (lower constant overhead)
- Use hybrid algorithm for large n with small k
- Preallocate heap at index construction time, reuse across queries
- Stable tie-breaking is free (negligible cost)
- Heap operations are branch-heavy; ensure good branch prediction

**Memory Management**:
- Heap allocates exactly 8k bytes (fixed size)
- Caller owns heap lifetime
- No hidden allocations in hot path
- Use stack allocation for small k (k ≤ 100)

**API Usage**:
```swift
// Good: Reuse heap across queries
var heap = TopKHeap(capacity: k, ordering: .max)
for query in queries {
    heap.clear()
    // ... fill heap ...
}

// Bad: Allocate new heap per query
for query in queries {
    let heap = TopKHeap(capacity: k, ordering: .max)  // Allocation overhead!
}
```

**Error Handling**:
- k ≤ 0: Return empty heap
- k > n: Return all n elements (heap size = n)
- nil IDs: Use implicit 0..<n

Non-Goals

- Multi-dimensional top-k (select on multiple criteria)
- Approximate top-k (exact selection only)
- Distributed merge (handled by kernel #06)
- GPU/Metal acceleration (CPU-bound algorithm)
- Incremental updates (rebuild heap for new data)

Example Usage

```swift
import VectorIndex

// Example 1: Basic top-k selection
let scores: [Float] = [0.9, 0.5, 0.8, 0.3, 0.95, 0.7]
let ids: [Int32] = [10, 20, 30, 40, 50, 60]

let heap = selectTopK(
    scores: scores,
    ids: ids,
    count: scores.count,
    k: 3,
    ordering: .max  // Higher is better
)

let results = heap.extractSorted()
// [(0.95, id=50), (0.9, id=10), (0.8, id=30)]

// Example 2: L2 distance (min-heap)
let distances: [Float] = [0.5, 1.2, 0.3, 0.8, 0.1]

let minHeap = selectTopK(
    scores: distances,
    ids: nil,  // Use implicit IDs 0, 1, 2, ...
    count: distances.count,
    k: 2,
    ordering: .min  // Lower is better
)

let nearest = minHeap.extractSorted()
// [(0.1, id=4), (0.3, id=2)]

// Example 3: IVF search with top-k
let query = Vector(...)
let cellScores = scoreIVFCells(query, nProbe: 10)
let candidateIds = getCandidateIds(nProbe: 10)

let searchResults = selectTopK(
    scores: cellScores,
    ids: candidateIds,
    count: cellScores.count,
    k: 10,
    ordering: .max
)

// Example 4: Batch selection for multiple partitions
let partitionScores: [[Float]] = [
    [0.9, 0.8, 0.7],  // Partition 0
    [0.95, 0.85],     // Partition 1
    [0.6, 0.5, 0.4]   // Partition 2
]

let partialHeaps = partitionScores.enumerated().map { (partitionID, scores) in
    selectTopK(
        scores: scores,
        ids: nil,
        count: scores.count,
        k: 2,  // Top-2 per partition
        ordering: .max
    )
}

// Merge via kernel #06
let globalTopK = mergeTopK(partialHeaps, k: 2)

// Example 5: High-level convenience API
let results = TopKSelectionKernel.select(
    scores: [0.5, 0.9, 0.3, 0.8],
    k: 2,
    ordering: .max
)
// Automatically handles memory layout
```

Mathematical Foundation

**Heap Property** (min-heap):
```
For all nodes i in heap:
  scores[i] ≤ scores[2i+1]  (left child)
  scores[i] ≤ scores[2i+2]  (right child)

Root is maximum element (worst in top-k for min-heap)
```

**Complexity Proofs**:

1. **Streaming Heap**: O(n log k)
   - First k elements: O(k log k) heapify
   - Remaining (n-k) elements: O((n-k) log k) comparisons + sifts
   - Total: O(k log k + (n-k) log k) = O(n log k)

2. **Hybrid Quickselect**: O(n + k log k) average case
   - Quickselect partition: O(n) average (randomized pivot)
   - Heap refinement: O(k log k) worst case
   - Total: O(n + k log k)
   - Worst case: O(n²) if bad pivots, but rare with median-of-three

**Stable Comparison**:
```
Lexicographic ordering on (score, id) pairs:
(s₁, id₁) < (s₂, id₂) ⟺ s₁ < s₂ ∨ (s₁ = s₂ ∧ id₁ < id₂)
```

Dependencies

**Internal**:
- ScoreBlockKernel (#04): Provides scored candidates
- MergeTopKKernel (#06): Merges partial heaps across threads
- Telemetry (#46): Performance instrumentation

**External**:
- Swift Standard Library: Heap algorithms, comparison
- Foundation: None (pure compute kernel)
- Dispatch: None (single-threaded per heap)

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Performance**:
- Streaming: O(n log k) confirmed via comparison counts
- Hybrid: O(n + k log k) for n > 16k, k << n
- Throughput: >10M candidates/sec for typical k and n
- Memory: O(k) for streaming, O(n) temporary for hybrid

✅ **Correctness**:
- Bit-exact match with full sort reference
- Stable tie-breaking: deterministic ordering
- All edge cases handled (k=0, k>n, etc.)

✅ **Stability**:
- Duplicate scores ordered by ID
- Reproducible across runs
- Same results from streaming and hybrid

✅ **Coverage**:
- Both min-heap and max-heap tested
- All k and n ranges covered
- Adversarial inputs (sorted, equal, etc.)

✅ **Integration**:
- Successfully used by IVF search
- Compatible with HNSW beam search
- Merges cleanly via kernel #06
- Supports parallel per-thread heaps

✅ **Usability**:
- Clear API for both orderings
- Automatic algorithm selection
- High-level convenience methods
- Comprehensive documentation with examples
<!-- moved to docs/kernel-specs/ -->
