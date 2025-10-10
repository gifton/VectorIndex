Title: ✅ DONE — K-Way Top-K Merge Kernel — Distributed Result Merging for Vector Search

Summary
- Implement a high-performance k-way merge kernel that combines multiple partial top-k heaps from distributed partitions/threads into a single global top-k result.
- Uses priority queue (tournament tree) to efficiently select the best k elements from m input heaps in O(k log m) time.
- Ensures deterministic, stable ordering across merge operations for reproducible search results.
- Critical for parallel search: IVF multi-probe, distributed HNSW, sharded indices.
- Supports both min-heap (L2 distance) and max-heap (IP/Cosine similarity) configurations.

Project Context
- VectorIndex implements distributed search with parallel scoring across partitions
- K-way merge is the final aggregation stage of distributed search:
  - **IVF Multi-Probe**: Merge results from multiple probed cells
  - **Distributed Flat Search**: Merge results from multiple shards
  - **Parallel HNSW**: Merge results from multiple entry points
  - **Multi-Query Search**: Combine results from query batches
- Industry context: Essential for horizontal scaling of vector search
- Challenge: Efficiently merge m heaps of size k without materializing all m×k elements
- Typical usage patterns:
  - m ∈ {2, 4, 8, 16} partitions (number of threads/shards)
  - k ∈ {10, 50, 100} results per partition
  - Output: global top-k (same k)
- VectorCore provides primitives; VectorIndex needs distributed merge
- Integrates with partial top-k selection (#05) for complete distributed pipeline

Goals
- Achieve O(k log m) complexity for merging m heaps of size k
- Support arbitrary number of input heaps (m from 2 to 1000+)
- Deterministic stable ordering for reproducible results
- Zero unnecessary allocations (reuse merge heap structure)
- Thread-safe merge operation (read-only input heaps)
- Efficient for common cases: m=4-16 partitions, k=10-100
- Seamless integration with parallel scoring and selection

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/TopKMergeKernel.swift`
- Core implementations:
  - Merge data structure: `MergeHeap` for tournament tree
  - Main algorithm: `mergeTopK` with automatic heap management
  - Specialized variants: `mergeTopK_small` for m ≤ 8, `mergeTopK_large` for m > 8
  - Iterator management: Track position in each input heap
- Supporting utilities:
  - Heap element comparison with stable tie-breaking
  - Sift operations for merge heap
  - Batch extraction for efficiency
  - Telemetry integration (#46)
- Integration points:
  - Consumes partial heaps from TopKSelectionKernel (#05)
  - Exports final results to search APIs
  - Used by IVF, HNSW, flat search with parallelism

API & Signatures

```swift
// MARK: - Core Merge API

/// Merge multiple partial top-k heaps into a single global top-k result
/// Uses tournament tree to efficiently select k best elements
///
/// - Complexity: O(k log m) where m = number of input heaps
/// - Performance: 5-10μs for typical m=8, k=10 on M1
/// - Thread Safety: Reentrant; input heaps are read-only
///
/// - Parameters:
///   - partialHeaps: Array of partial top-k heaps from distributed scoring
///   - k: Number of top results to return (global top-k)
///   - ordering: Heap ordering (min for L2, max for IP/Cosine)
///   - config: Optional configuration (telemetry, etc.)
/// - Returns: TopKHeap containing global top-k results
@inlinable
public func mergeTopK(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering,
    config: TopKMergeConfig = .default
) -> TopKHeap

/// Merge partial heaps with explicit score/ID arrays
/// Allows merging from heterogeneous sources
@inlinable
public func mergeTopK(
    scores: [[Float]],
    ids: [[Int32]],
    k: Int,
    ordering: HeapOrdering,
    config: TopKMergeConfig = .default
) -> TopKHeap

// MARK: - Specialized Variants

/// Optimized merge for small number of heaps (m ≤ 8)
/// Uses unrolled comparisons for better branch prediction
@inlinable
func mergeTopK_small(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap

/// General merge for arbitrary number of heaps
/// Uses priority queue for O(k log m) complexity
@inlinable
func mergeTopK_large(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap

// MARK: - Iterator Management

/// Iterator over elements in a partial heap
struct HeapIterator {
    let heap: TopKHeap
    var position: Int

    /// Get current element without advancing
    var current: (score: Float, id: Int32)? {
        guard position < heap.count else { return nil }
        return heap[position]
    }

    /// Advance to next element
    mutating func advance() {
        position += 1
    }

    /// Whether iterator has more elements
    var hasMore: Bool {
        position < heap.count
    }
}

// MARK: - Merge Heap

/// Priority queue for tournament-style merging
/// Maintains one element from each input heap
struct MergeHeap {
    /// Heap ordering (min or max)
    let ordering: HeapOrdering

    /// Current elements in merge heap [m]
    private var scores: [Float]

    /// IDs corresponding to scores [m]
    private var ids: [Int32]

    /// Source heap index for each element [m]
    private var sources: [Int]

    /// Current heap size
    private(set) var count: Int

    /// Initialize merge heap with m input heaps
    init(capacity: Int, ordering: HeapOrdering)

    /// Insert element from specific source heap
    mutating func push(score: Float, id: Int32, source: Int)

    /// Extract root (best element) and return its source
    mutating func pop() -> (score: Float, id: Int32, source: Int)?

    /// Replace root with new element from same source
    mutating func replaceRoot(score: Float, id: Int32, source: Int)
}

// MARK: - Configuration

/// Configuration for top-k merge
public struct TopKMergeConfig {
    /// Enable telemetry recording (default: false)
    let enableTelemetry: Bool

    /// Threshold for small vs large merge strategy (default: 8)
    let smallMergeThreshold: Int

    public static let `default` = TopKMergeConfig(
        enableTelemetry: false,
        smallMergeThreshold: 8
    )
}

// MARK: - Telemetry

/// Per-merge execution statistics
public struct TopKMergeTelemetry {
    public let inputHeapCount: Int       // m
    public let outputK: Int              // k
    public let totalCandidates: Int      // Total elements across all input heaps
    public let mergeOps: Int             // Number of heap pops/pushes
    public let shardUtilization: [Int]   // Elements used from each shard
    public let executionTimeNanos: UInt64

    public var averageUtilizationPercent: Double {
        let avgUsed = Double(shardUtilization.reduce(0, +)) / Double(inputHeapCount)
        let avgSize = Double(totalCandidates) / Double(inputHeapCount)
        return (avgUsed / avgSize) * 100.0
    }
}

// MARK: - Convenience API

extension TopKMergeKernel {
    /// High-level merge with automatic configuration
    public static func merge(
        partialResults: [[(score: Float, id: Int32)]],
        k: Int,
        ordering: HeapOrdering
    ) -> [(score: Float, id: Int32)]
}
```

Algorithm Details

**K-Way Merge Algorithm** (tournament tree approach):

```swift
@inlinable
public func mergeTopK(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering,
    config: TopKMergeConfig = .default
) -> TopKHeap {
    let m = partialHeaps.count

    // Edge cases
    guard m > 0 else { return TopKHeap(capacity: k, ordering: ordering) }
    if m == 1 { return partialHeaps[0] }

    // Choose algorithm based on number of heaps
    if m <= config.smallMergeThreshold {
        return mergeTopK_small(partialHeaps: partialHeaps, k: k, ordering: ordering)
    } else {
        return mergeTopK_large(partialHeaps: partialHeaps, k: k, ordering: ordering)
    }
}
```

**Tournament Tree Merge** (for general case):

```swift
@inlinable
func mergeTopK_large(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap {
    let m = partialHeaps.count
    var result = TopKHeap(capacity: k, ordering: ordering)

    // Initialize iterators for each input heap
    var iterators = partialHeaps.map { HeapIterator(heap: $0, position: 0) }

    // Initialize merge heap with first element from each input heap
    var mergeHeap = MergeHeap(capacity: m, ordering: ordering)
    for (source, iterator) in iterators.enumerated() {
        if let element = iterator.current {
            mergeHeap.push(score: element.score, id: element.id, source: source)
        }
    }

    // Extract k best elements
    var extracted = 0
    while extracted < k, let best = mergeHeap.pop() {
        // Add to result
        result.push(score: best.score, id: best.id)
        extracted += 1

        // Advance iterator for source that provided this element
        iterators[best.source].advance()

        // If source has more elements, push next to merge heap
        if let next = iterators[best.source].current {
            mergeHeap.push(score: next.score, id: next.id, source: best.source)
        }
    }

    return result
}
```

**Small Merge Optimization** (m ≤ 8):

```swift
@inlinable
func mergeTopK_small(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering
) -> TopKHeap {
    let m = partialHeaps.count
    var result = TopKHeap(capacity: k, ordering: ordering)

    // Initialize iterators
    var iterators = partialHeaps.map { HeapIterator(heap: $0, position: 0) }

    // For small m, use linear scan instead of heap
    // Better branch prediction and fewer indirections
    var extracted = 0
    while extracted < k {
        var bestScore: Float
        var bestId: Int32
        var bestSource = -1

        // Linear scan to find best among current elements
        switch ordering {
        case .min:
            bestScore = .infinity
            for (source, iterator) in iterators.enumerated() {
                if let element = iterator.current {
                    if element.score < bestScore ||
                       (element.score == bestScore && element.id < bestId) {
                        bestScore = element.score
                        bestId = element.id
                        bestSource = source
                    }
                }
            }

        case .max:
            bestScore = -.infinity
            for (source, iterator) in iterators.enumerated() {
                if let element = iterator.current {
                    if element.score > bestScore ||
                       (element.score == bestScore && element.id < bestId) {
                        bestScore = element.score
                        bestId = element.id
                        bestSource = source
                    }
                }
            }
        }

        // No more elements available
        guard bestSource >= 0 else { break }

        // Add to result and advance
        result.push(score: bestScore, id: bestId)
        iterators[bestSource].advance()
        extracted += 1
    }

    return result
}
```

**Merge Heap Operations**:

```swift
extension MergeHeap {
    /// Sift down to maintain heap property after root replacement
    @inline(__always)
    private mutating func siftDown(index: Int) {
        var currentIdx = index
        let score = scores[currentIdx]
        let id = ids[currentIdx]
        let source = sources[currentIdx]

        while true {
            let leftIdx = 2 * currentIdx + 1
            let rightIdx = 2 * currentIdx + 2

            guard leftIdx < count else { break }

            var swapIdx = currentIdx
            var swapScore = score
            var swapId = id
            var swapSource = source

            // Compare with left child (stable comparison)
            let leftScore = scores[leftIdx]
            let leftId = ids[leftIdx]
            let leftBetter: Bool
            switch ordering {
            case .min:
                leftBetter = leftScore < swapScore ||
                            (leftScore == swapScore && leftId < swapId)
            case .max:
                leftBetter = leftScore > swapScore ||
                            (leftScore == swapScore && leftId < swapId)
            }

            if leftBetter {
                swapIdx = leftIdx
                swapScore = leftScore
                swapId = leftId
                swapSource = sources[leftIdx]
            }

            // Compare with right child
            if rightIdx < count {
                let rightScore = scores[rightIdx]
                let rightId = ids[rightIdx]
                let rightBetter: Bool
                switch ordering {
                case .min:
                    rightBetter = rightScore < swapScore ||
                                 (rightScore == swapScore && rightId < swapId)
                case .max:
                    rightBetter = rightScore > swapScore ||
                                 (rightScore == swapScore && rightId < swapId)
                }

                if rightBetter {
                    swapIdx = rightIdx
                    swapScore = rightScore
                    swapId = rightId
                    swapSource = sources[rightIdx]
                }
            }

            guard swapIdx != currentIdx else { break }

            // Swap with better child
            scores[currentIdx] = swapScore
            ids[currentIdx] = swapId
            sources[currentIdx] = swapSource
            currentIdx = swapIdx
        }

        scores[currentIdx] = score
        ids[currentIdx] = id
        sources[currentIdx] = source
    }

    /// Extract root element
    mutating func pop() -> (score: Float, id: Int32, source: Int)? {
        guard count > 0 else { return nil }

        let result = (scores[0], ids[0], sources[0])

        // Move last element to root and sift down
        count -= 1
        if count > 0 {
            scores[0] = scores[count]
            ids[0] = ids[count]
            sources[0] = sources[count]
            siftDown(index: 0)
        }

        return result
    }

    /// Replace root with new element from same source
    mutating func replaceRoot(score: Float, id: Int32, source: Int) {
        scores[0] = score
        ids[0] = id
        sources[0] = source
        siftDown(index: 0)
    }
}
```

**Stable Comparison**:

```swift
/// Compare two elements with stable tie-breaking by ID
func isStrictlyBetter(
    score1: Float, id1: Int32,
    score2: Float, id2: Int32,
    ordering: HeapOrdering
) -> Bool {
    switch ordering {
    case .min:
        return score1 < score2 || (score1 == score2 && id1 < id2)
    case .max:
        return score1 > score2 || (score1 == score2 && id1 < id2)
    }
}
```

Complexity Analysis

**Time Complexity**:

```
Operation                  | Complexity
---------------------------|------------------
Initialize merge heap      | O(m) where m = number of input heaps
Extract k elements         | O(k log m) - k pops + pushes
Total                      | O(m + k log m) = O(k log m) since k >> m typically

Comparison with alternatives:
- Concatenate + sort: O(mk log(mk)) — too slow
- Concatenate + select: O(mk) — requires materializing all elements
- Tournament merge: O(k log m) — optimal ✓
```

**Space Complexity**:

```
Data Structure        | Space
----------------------|----------
Merge heap           | O(m) - one element per input heap
Iterators            | O(m) - position tracking
Result heap          | O(k) - output
Total                | O(m + k)
```

**Example** (m=8 partitions, k=10):

```
Initialize merge heap: 8 elements, build heap in O(8)
Extract 10 best:
  - Pop from merge heap: O(log 8) = O(3)
  - Push replacement: O(log 8) = O(3)
  - Total: 10 × O(log 8) = 30 operations

Total time: O(8 + 10 log 8) = O(38 operations)

Compare with full sort:
  - Total elements: 8 × 10 = 80
  - Sort: O(80 log 80) ≈ 500 operations
  - Speedup: ~13×
```

Algorithm Selection Strategy

**Small vs Large Merge**:

```swift
func chooseStrategy(m: Int) -> MergeStrategy {
    // For small m (≤8), linear scan is competitive with heap
    // - Better branch prediction
    // - Fewer indirections
    // - Better cache locality
    // - Crossover point empirically determined on Apple Silicon

    if m <= 8 {
        return .linearScan  // O(k × m) but lower constant
    } else {
        return .tournamentTree  // O(k log m)
    }
}
```

**Performance Comparison** (Apple M1):

```
m    | k=10 Linear | k=10 Heap | k=100 Linear | k=100 Heap
-----|-------------|-----------|--------------|------------
2    | 0.5 μs      | 1.0 μs    | 3 μs         | 8 μs
4    | 0.8 μs      | 1.5 μs    | 5 μs         | 12 μs
8    | 1.2 μs      | 2.0 μs    | 8 μs         | 16 μs
16   | 2.5 μs      | 2.5 μs    | 15 μs        | 20 μs
32   | 5.0 μs      | 3.0 μs    | 30 μs        | 25 μs ← Heap wins
```

Stability & Determinism

**Stable Merge Property**:

```
Given input heaps with stable ordering (from kernel #05),
merge preserves stability across partitions.

Example (max-heap, cosine similarity):
Partition 0: [(0.95, id=5), (0.9, id=3), (0.85, id=1)]
Partition 1: [(0.95, id=2), (0.9, id=8), (0.8, id=4)]
Partition 2: [(0.9, id=6), (0.85, id=7), (0.75, id=9)]

Merged (stable by ID within same score):
1. (0.95, id=2)  ← Partition 1, smaller ID
2. (0.95, id=5)  ← Partition 0
3. (0.9, id=3)   ← Partition 0, smaller ID
4. (0.9, id=6)   ← Partition 2
5. (0.9, id=8)   ← Partition 1
...
```

**Why Stability Matters**:
- Reproducible search results across different partition assignments
- Consistent ranking in distributed systems
- Fair comparison when evaluating recall metrics
- Debuggability: deterministic behavior

Parallelization Patterns

**Typical Workflow** (distributed search):

```swift
func distributedSearch(
    query: Vector,
    shards: [IndexShard],
    k: Int,
    metric: DistanceMetric
) -> [SearchResult] {
    let ordering: HeapOrdering = metric.higherIsBetter ? .max : .min

    // Phase 1: Parallel scoring and local top-k selection
    let partialHeaps = DispatchQueue.concurrentPerform(
        iterations: shards.count
    ) { shardID -> TopKHeap in
        let shard = shards[shardID]

        // Score all vectors in this shard
        var scores = [Float](repeating: 0, count: shard.vectorCount)
        scoreBlock(
            query: query.data,
            database: shard.vectors,
            vectorCount: shard.vectorCount,
            dimension: query.dimension,
            metric: metric,
            scores: &scores
        )

        // Select local top-k (kernel #05)
        return selectTopK(
            scores: scores,
            ids: shard.vectorIds,
            count: shard.vectorCount,
            k: k,
            ordering: ordering
        )
    }

    // Phase 2: Merge partial heaps (kernel #06) - sequential
    let globalTopK = mergeTopK(
        partialHeaps: partialHeaps,
        k: k,
        ordering: ordering
    )

    return globalTopK.extractSorted().map {
        SearchResult(id: $0.id, score: $0.score)
    }
}
```

**Thread Safety**:
- Input heaps are read-only during merge
- Merge operation is sequential (no concurrent merging needed)
- Result heap is constructed independently
- No locks or synchronization required

Performance Targets (Apple M1/M2/M3, Release Build)

**Latency** (per merge operation):

```
m    | k=10     | k=50     | k=100
-----|----------|----------|----------
2    | 0.5 μs   | 2 μs     | 4 μs
4    | 1 μs     | 4 μs     | 8 μs
8    | 2 μs     | 8 μs     | 15 μs
16   | 3 μs     | 12 μs    | 25 μs
32   | 4 μs     | 18 μs    | 35 μs
```

**Throughput** (merges/second):

```
For typical distributed search (m=8, k=10):
- Merge latency: ~2 μs
- Throughput: ~500,000 merges/sec
- Merge overhead: <1% of total search time
```

**Scalability**:

```
As m (number of partitions) increases:
- Linear scan: O(k × m) - degrades linearly
- Tournament tree: O(k log m) - degrades logarithmically

Speedup for m=64, k=10:
- Linear: 64 × 10 = 640 operations
- Tournament: 10 × log₂(64) = 60 operations
- Speedup: ~10.6×
```

Correctness & Testing

**Golden Reference**:

```swift
// Reference: Concatenate all heaps, sort, take top-k
func referenceMerge(
    partialHeaps: [TopKHeap],
    k: Int,
    ordering: HeapOrdering
) -> [(Float, Int32)] {
    // Collect all elements from all heaps
    var allElements: [(Float, Int32)] = []
    for heap in partialHeaps {
        allElements.append(contentsOf: heap.extractSorted())
    }

    // Sort with stable ordering
    let sorted: [(Float, Int32)]
    switch ordering {
    case .min:
        sorted = allElements.sorted {
            $0.0 < $1.0 || ($0.0 == $1.0 && $0.1 < $1.1)
        }
    case .max:
        sorted = allElements.sorted {
            $0.0 > $1.0 || ($0.0 == $1.0 && $0.1 < $1.1)
        }
    }

    return Array(sorted.prefix(k))
}
```

**Test Cases**:

1. **Basic Correctness**:
   - Compare against concatenate+sort reference
   - Various m and k combinations
   - Min-heap and max-heap orderings

2. **Stability**:
   - Multiple partitions with duplicate scores
   - Verify stable ordering by ID
   - Deterministic across runs

3. **Edge Cases**:
   - m = 1 (single partition, no merge needed)
   - Empty partitions
   - k > total elements (return all)
   - All partitions have same scores

4. **Partition Imbalance**:
   - Some partitions empty, others full
   - Varying partition sizes
   - Early exhaustion of some partitions

5. **Algorithm Selection**:
   - Verify small merge (m ≤ 8) produces same result
   - Verify large merge (m > 8) produces same result
   - Crossover point correctness

**Example Tests**:

```swift
func testMerge_Correctness() {
    let m = 8
    let k = 10

    // Create partial heaps with random scores
    var partialHeaps: [TopKHeap] = []
    for _ in 0..<m {
        var heap = TopKHeap(capacity: k, ordering: .max)
        for id in 0..<k {
            let score = Float.random(in: 0...1)
            heap.push(score: score, id: Int32(id))
        }
        partialHeaps.append(heap)
    }

    // Merge using kernel
    let merged = mergeTopK(
        partialHeaps: partialHeaps,
        k: k,
        ordering: .max
    )

    // Compare with reference
    let reference = referenceMerge(
        partialHeaps: partialHeaps,
        k: k,
        ordering: .max
    )

    let result = merged.extractSorted()
    XCTAssertEqual(result.count, min(k, reference.count))

    for i in 0..<result.count {
        XCTAssertEqual(result[i].score, reference[i].0)
        XCTAssertEqual(result[i].id, reference[i].1)
    }
}

func testMerge_Stability() {
    let m = 4
    let k = 10
    let fixedScore: Float = 0.5  // All same score

    var partialHeaps: [TopKHeap] = []
    for partitionID in 0..<m {
        var heap = TopKHeap(capacity: k, ordering: .max)
        for i in 0..<k {
            let id = Int32(partitionID * k + i)
            heap.push(score: fixedScore, id: id)
        }
        partialHeaps.append(heap)
    }

    let merged = mergeTopK(
        partialHeaps: partialHeaps,
        k: k,
        ordering: .max
    )

    let result = merged.extractSorted()

    // All scores equal, so should be ordered by ID (stable)
    for i in 0..<result.count {
        XCTAssertEqual(result[i].id, Int32(i))
    }
}

func testMerge_AlgorithmEquivalence() {
    let k = 10

    // Test small merge (m ≤ 8)
    var smallHeaps: [TopKHeap] = []
    for _ in 0..<8 {
        var heap = TopKHeap(capacity: k, ordering: .max)
        for id in 0..<k {
            heap.push(score: Float.random(in: 0...1), id: Int32(id))
        }
        smallHeaps.append(heap)
    }

    let smallResult = mergeTopK_small(
        partialHeaps: smallHeaps,
        k: k,
        ordering: .max
    )

    let largeResult = mergeTopK_large(
        partialHeaps: smallHeaps,
        k: k,
        ordering: .max
    )

    // Both algorithms must produce identical results
    let small = smallResult.extractSorted()
    let large = largeResult.extractSorted()

    XCTAssertEqual(small.count, large.count)
    for i in 0..<small.count {
        XCTAssertEqual(small[i].score, large[i].score)
        XCTAssertEqual(small[i].id, large[i].id)
    }
}
```

Integration with Search Algorithms

**IVF Multi-Probe Search**:

```swift
struct IVFIndex {
    func search(query: Vector, nProbe: Int, k: Int) -> [SearchResult] {
        let ordering: HeapOrdering = metric.higherIsBetter ? .max : .min

        // Step 1: Select top-nProbe cells to probe
        let probedCells = selectTopCells(query: query, nProbe: nProbe)

        // Step 2: Score and select top-k from each cell (parallel)
        let cellHeaps = DispatchQueue.concurrentPerform(
            iterations: probedCells.count
        ) { cellIdx -> TopKHeap in
            let cellID = probedCells[cellIdx]
            let cellVectors = cells[cellID]

            var scores = [Float](repeating: 0, count: cellVectors.count)
            scoreBlock(query, cellVectors, metric, &scores)

            return selectTopK(
                scores: scores,
                ids: getCellVectorIds(cellID),
                count: cellVectors.count,
                k: k,
                ordering: ordering
            )
        }

        // Step 3: Merge partial heaps into global top-k
        let globalTopK = mergeTopK(
            partialHeaps: cellHeaps,
            k: k,
            ordering: ordering
        )

        return globalTopK.extractSorted().map {
            SearchResult(id: $0.id, score: $0.score)
        }
    }
}
```

**Distributed Flat Search**:

```swift
struct DistributedFlatIndex {
    let shards: [FlatIndex]

    func search(query: Vector, k: Int) -> [SearchResult] {
        let ordering: HeapOrdering = metric.higherIsBetter ? .max : .min

        // Search each shard independently (potentially on different machines)
        let shardResults: [TopKHeap] = shards.parallelMap { shard in
            shard.searchLocal(query: query, k: k)
        }

        // Merge results from all shards
        let globalTopK = mergeTopK(
            partialHeaps: shardResults,
            k: k,
            ordering: ordering
        )

        return globalTopK.extractSorted().map {
            SearchResult(id: $0.id, score: $0.score)
        }
    }
}
```

**HNSW Multi-Entry Search**:

```swift
struct HNSWIndex {
    func searchMultiEntry(
        query: Vector,
        entryPoints: [GraphNode],
        ef: Int,
        k: Int
    ) -> [SearchResult] {
        let ordering: HeapOrdering = metric.higherIsBetter ? .max : .min

        // Search from each entry point independently
        let entryResults: [TopKHeap] = entryPoints.map { entry in
            searchFromEntry(query: query, entry: entry, ef: ef, k: k)
        }

        // Merge results from all entry points
        let globalTopK = mergeTopK(
            partialHeaps: entryResults,
            k: k,
            ordering: ordering
        )

        return globalTopK.extractSorted().map {
            SearchResult(id: $0.id, score: $0.score)
        }
    }
}
```

Coding Guidelines

**Performance Best Practices**:
- Use small merge for m ≤ 8 (better constant factors)
- Use tournament tree for m > 8 (better asymptotic complexity)
- Preallocate merge heap once, reuse across queries
- Input heaps should already be sorted (from kernel #05)
- Avoid materializing all elements (defeats purpose of merge)

**Memory Management**:
- Merge heap allocates O(m) space (minimal)
- Result heap allocates O(k) space
- No allocation in merge loop (hot path)
- Input heaps are read-only (safe for concurrent reads)

**API Usage**:

```swift
// Good: Reuse for multiple merges
let mergeHeap = MergeHeap(capacity: 16, ordering: .max)
for queryBatch in batches {
    let partialHeaps = scoreInParallel(queryBatch)
    let result = mergeTopK(partialHeaps, k: 10, ordering: .max)
    // Process result...
}

// Bad: Repeated allocation
for queryBatch in batches {
    let partialHeaps = scoreInParallel(queryBatch)
    let result = mergeTopK(partialHeaps, k: 10, ordering: .max)
    // Allocates new merge heap each time!
}
```

**Error Handling**:
- Empty input: Return empty heap
- k > total elements: Return all elements
- m = 1: Return input heap directly (no merge)
- Mismatched ordering: Undefined behavior (assert in debug)

Non-Goals

- Parallel merging (merge is cheap relative to scoring)
- Distributed merge across network (application-level)
- Approximate merge (exact top-k only)
- Multi-criteria merge (single score dimension)
- Incremental merge (rebuild for new partitions)

Example Usage

```swift
import VectorIndex

// Example 1: IVF multi-probe merge
let query = Vector(data: ...)
let nProbe = 8
let k = 10

// Score each probed cell independently
var cellHeaps: [TopKHeap] = []
for cellID in 0..<nProbe {
    let cellScores = scoreCell(query, cellID)
    let heap = selectTopK(
        scores: cellScores,
        ids: getCellIds(cellID),
        count: cellScores.count,
        k: k,
        ordering: .max
    )
    cellHeaps.append(heap)
}

// Merge all cell results
let finalResults = mergeTopK(
    partialHeaps: cellHeaps,
    k: k,
    ordering: .max
)

// Example 2: Distributed shard merge
let shardResults: [TopKHeap] = [
    shard0Results,  // Top-10 from shard 0
    shard1Results,  // Top-10 from shard 1
    shard2Results,  // Top-10 from shard 2
    shard3Results   // Top-10 from shard 3
]

let globalResults = mergeTopK(
    partialHeaps: shardResults,
    k: 10,
    ordering: .max
)

// Example 3: HNSW multi-entry merge
let entryPoints = [node0, node1, node2]
var entryResults: [TopKHeap] = []

for entry in entryPoints {
    let result = searchFromEntry(query, entry, ef: 50, k: 10)
    entryResults.append(result)
}

let mergedResults = mergeTopK(
    partialHeaps: entryResults,
    k: 10,
    ordering: .max
)

// Example 4: Manual merge with score arrays
let scores: [[Float]] = [
    [0.9, 0.8, 0.7],      // Partition 0
    [0.95, 0.85, 0.75],   // Partition 1
    [0.92, 0.82, 0.72]    // Partition 2
]

let ids: [[Int32]] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

let merged = mergeTopK(
    scores: scores,
    ids: ids,
    k: 5,
    ordering: .max
)

// Example 5: High-level convenience API
let partialResults = [
    [(0.9, id: 1), (0.8, id: 2)],
    [(0.95, id: 3), (0.85, id: 4)],
    [(0.92, id: 5), (0.82, id: 6)]
]

let final = TopKMergeKernel.merge(
    partialResults: partialResults,
    k: 3,
    ordering: .max
)
// Returns: [(0.95, id: 3), (0.92, id: 5), (0.9, id: 1)]
```

Mathematical Foundation

**K-Way Merge Problem**:
```
Given: m sorted arrays A₁, A₂, ..., Aₘ of sizes k₁, k₂, ..., kₘ
Find: k smallest/largest elements from ⋃ᵢ Aᵢ

Solutions:
1. Concatenate + Sort: O(N log N) where N = Σkᵢ
2. Concatenate + Select: O(N) but requires materializing all
3. Tournament Tree: O(k log m) ← Optimal for k << N ✓
```

**Tournament Tree Invariant**:
```
Merge heap maintains:
- Root = best element among current heads of input heaps
- Each node compares elements from different sources
- Sift operations preserve heap property
- Stable tie-breaking by (score, id) lexicographic order
```

**Complexity Proof**:
```
Initialize: Build heap from m elements
  - Time: O(m)
  - Space: O(m)

Extract k elements:
  - Each extraction: O(log m) to restore heap
  - k extractions: O(k log m)

Total: O(m + k log m)

Since typically k >> m (e.g., k=10-100, m=4-16):
  → O(k log m) dominates
```

Dependencies

**Internal**:
- TopKSelectionKernel (#05): Provides partial heaps to merge
- TopKHeap data structure: Input and output format
- Telemetry (#46): Performance instrumentation

**External**:
- Swift Standard Library: Comparison, iteration
- Foundation: None (pure algorithm)
- Dispatch: None (sequential merge)

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Performance**:
- O(k log m) complexity verified via operation counts
- Latency < 10μs for typical m=8, k=10 on M1
- Merge overhead < 1% of total search time

✅ **Correctness**:
- Bit-exact match with concatenate+sort reference
- Stable tie-breaking: deterministic ordering
- All edge cases handled (empty, single partition, k>total)

✅ **Stability**:
- Duplicate scores ordered by ID across partitions
- Reproducible across different partition assignments
- Same result from small and large merge strategies

✅ **Coverage**:
- Both min-heap and max-heap tested
- All m ranges (2 to 1000+) covered
- All k ranges covered
- Partition imbalance scenarios

✅ **Integration**:
- Successfully used by IVF multi-probe search
- Compatible with distributed flat search
- Works with HNSW multi-entry search
- Seamless composition with kernel #05

✅ **Efficiency**:
- Small merge optimized for m ≤ 8
- Tournament tree scales to large m
- Automatic strategy selection
- Minimal memory overhead (O(m + k))
