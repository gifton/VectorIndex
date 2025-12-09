# Top-K Selection

> **Reading time:** 12 minutes
> **Prerequisites:** [Brute-Force Search](./01-Brute-Force-Search.md)

---

## The Concept

In brute-force search, we compute n distances but only need the k smallest. Sorting all n values is wasteful:

```
Full sort:   O(n log n) — orders ALL elements
Top-k:       O(n log k) — maintains only k best

For n = 1,000,000 and k = 10:
  Full sort: 20 million comparisons
  Top-k:     3.3 million comparisons (6× fewer)
```

**Top-k selection** uses a max-heap to efficiently track the k smallest values seen so far.

---

## The Heap-Based Algorithm

### Data Structure: Max-Heap

A max-heap keeps the **largest** element at the root:

```
Max-heap of size k=4 (tracking 4 smallest distances):

         0.95  ← Root (largest of the k smallest)
        /    \
     0.72    0.88
     /
   0.45

Property: Every parent ≥ its children
Benefit:  O(1) access to the largest element we're keeping
```

### Algorithm

```
Algorithm: Top-K Selection via Max-Heap

Input:  stream of (id, distance) pairs
        k (number to keep)

1. Initialize empty max-heap of capacity k

2. For each (id, distance):
     If heap.size < k:
         heap.push(id, distance)
     Else if distance < heap.peek():  // Better than worst in heap
         heap.pop()                    // Remove worst
         heap.push(id, distance)       // Add new

3. Return heap contents (sorted if needed)
```

### Walkthrough Example

Finding top-3 smallest from stream: [0.5, 0.2, 0.8, 0.1, 0.6, 0.3]

```
Step 1: 0.5 → heap = [0.5]                    (size < 3, just add)
Step 2: 0.2 → heap = [0.5, 0.2]               (size < 3, just add)
Step 3: 0.8 → heap = [0.8, 0.2, 0.5]          (size = 3, 0.8 is root)
Step 4: 0.1 → 0.1 < 0.8? Yes!
              pop 0.8, push 0.1
              heap = [0.5, 0.2, 0.1]          (0.5 is now root)
Step 5: 0.6 → 0.6 < 0.5? No, skip
Step 6: 0.3 → 0.3 < 0.5? Yes!
              pop 0.5, push 0.3
              heap = [0.3, 0.2, 0.1]

Result: {0.1, 0.2, 0.3} ✓
```

---

## Complexity Analysis

### Time

```
Each element: O(log k) for heap operations
Total:        O(n log k)

Compare to full sort: O(n log n)

Speedup: log(n) / log(k)
  n=1M, k=10: log(1M)/log(10) ≈ 6×
  n=1M, k=100: log(1M)/log(100) ≈ 3×
```

### Space

```
Heap storage: O(k) — only keep k elements
vs. full sort: O(n) — must store all before sorting
```

### When Heap Wins

```
            │
    Time    │    Full Sort
            │         ╱
            │        ╱
            │       ╱ ←── Crossover at k ≈ n/log(n)
            │      ╱
            │     ╱  Heap-based
            │    ╱
            └────────────────────────────
                     k (results needed)

For n = 1,000,000:
  Heap wins when k < ~50,000
  (covers virtually all practical cases)
```

---

## Implementation Strategies

### Strategy 1: Swift's Built-in Heap

Swift doesn't have a heap in the standard library, but you can use `CFBinaryHeap` or implement one:

```swift
struct MaxHeap<T> {
    private var elements: [(T, Float)] = []

    var count: Int { elements.count }

    func peek() -> Float? { elements.first?.1 }

    mutating func push(_ item: T, score: Float) {
        elements.append((item, score))
        siftUp(elements.count - 1)
    }

    mutating func pop() -> (T, Float)? {
        guard !elements.isEmpty else { return nil }
        elements.swapAt(0, elements.count - 1)
        let result = elements.removeLast()
        if !elements.isEmpty { siftDown(0) }
        return result
    }

    // ... siftUp, siftDown implementations
}
```

### Strategy 2: Partial Sort

For small k, Swift's sort with early termination:

```swift
// Swift's sort is adaptive - for small k, we can:
var results: [(String, Float)] = ...

// Get k smallest using partial sort
let topK = results.sorted { $0.1 < $1.1 }.prefix(k)

// Or use nth_element-style algorithm (not in stdlib)
```

### Strategy 3: Reservoir with Threshold

Maintain a threshold based on the k-th best seen:

```swift
// 📍 See: Sources/VectorIndex/Operations/Selection/TopK.swift

var topK: [(id: VectorID, score: Float)] = []
var threshold: Float = .infinity

for (id, vec, _) in vectors {
    let d = distance(query, vec, metric: metric)

    // Early skip: if worse than current k-th, don't bother
    if d >= threshold { continue }

    // Insert and maintain sorted order
    insertSorted(&topK, (id, d))
    if topK.count > k {
        topK.removeLast()
        threshold = topK.last!.score
    }
}
```

---

## The Threshold Optimization

The threshold provides **early termination** benefits:

```
Without threshold:
  - Compute distance for ALL n vectors
  - Insert all into heap

With threshold:
  - Compute distance for ALL n vectors
  - Only insert if distance < threshold
  - As threshold tightens, fewer insertions

Best case (data already sorted by distance):
  - First k elements set threshold
  - Remaining n-k elements skip heap entirely
  - Only k heap insertions total!
```

---

## In VectorIndex

VectorIndex uses straightforward sorted array for simplicity:

```swift
// 📍 See: Sources/VectorIndex/FlatIndex.swift:48-66

var results: [SearchResult] = []
results.reserveCapacity(min(k, vectors.count))

for (id, (vec, meta)) in vectors {
    if let filter = filter, !filter(meta) { continue }
    let d = distance(query, vec, metric: metric)
    results.append(SearchResult(id: id, score: d))
}

// Sort once at the end
results.sort { $0.score < $1.score }
if results.count > k { results.removeLast(results.count - k) }
```

For the typical case where vectors.count is moderate, this is efficient enough. The dedicated TopK kernel is used for larger-scale operations:

```swift
// 📍 See: Sources/VectorIndex/Operations/Selection/TopK.swift:54-106

// Fixed-size heap for top-k selection (Structure of Arrays layout)
public struct TopKHeap {
    public let ordering: HeapOrdering  // .min or .max
    public let capacity: Int           // k
    private var scores: UnsafeMutablePointer<Float>
    private var ids: UnsafeMutablePointer<Int32>

    public mutating func push(score: Float, id: Int32) {
        if count < capacity {
            // Add to heap
            _directWrite(at: count, score: score, id: id)
            _siftUp(from: count)
            count += 1
        } else if ordering.shouldReplace(score, id, rootScore: scores[0], rootId: ids[0]) {
            // Replace root and re-heapify
            replaceRoot(score: score, id: id)
        }
    }
}
```

The heap uses Structure-of-Arrays (SoA) layout for cache efficiency: separate arrays for scores and IDs.

---

## 🔗 VectorCore Connection

The top-k problem appears throughout VectorCore too:

```swift
// 🔗 VectorCore: Batch operations often need top-k

// Finding maximum element in SIMD4 (horizontal max)
let v = SIMD4<Float>(0.5, 0.2, 0.8, 0.1)
let maxVal = v.max()  // 0.8

// This is the k=1 case of top-k selection
// SIMD operations help when scanning many vectors
```

---

## Advanced: Approximate Top-K

For very large candidate sets, even O(n log k) can be slow. Approximate methods exist:

```
Approach: Random sampling + top-k

1. Sample √n candidates uniformly
2. Find top-k in sample
3. Use k-th sample value as threshold
4. Scan full data, keeping only below threshold

Expected candidates: O(k√n) instead of O(n)
Recall: ~99% for well-behaved distributions
```

VectorIndex doesn't currently implement this, as ANN indices (IVF, HNSW) already reduce candidate sets.

---

## Key Takeaways

1. **Don't sort when you only need top-k.** O(n log k) beats O(n log n).

2. **Max-heap tracks k smallest.** Root is the largest of the k smallest (easy to compare against).

3. **Threshold optimization.** Skip heap operations for obviously-worse candidates.

4. **For small k, simple is fine.** Heap overhead isn't worth it for k < 100 on modern CPUs.

5. **This pattern recurs everywhere.** IVF and HNSW both need top-k for their results.

---

## Next Up

Now let's establish when FlatIndex is the right choice:

**[→ When Flat Is Enough](./03-When-Flat-Is-Enough.md)**
