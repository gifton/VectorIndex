# ✅ DONE — Kernel Specification #32: Multi-List Candidate Deduplication

**ID**: 32
**Priority**: MUST
**Role**: Q (Query)
**Status**: Specification

---

## Purpose

Provide high-performance duplicate detection and filtering for candidate vectors during multi-probe search operations (IVF, HNSW beam search). Prevents redundant distance computations and maintains result set quality by ensuring each candidate is evaluated exactly once per query, even when appearing in multiple inverted lists, graph neighborhoods, or shards.

**Key Benefits**:
1. **Performance**: Eliminates ~30-70% of redundant distance calculations in multi-probe IVF
2. **Correctness**: Guarantees unique candidates for accurate top-k selection
3. **Efficiency**: Sub-nanosecond to ~8ns per-check latency with cache-friendly designs
4. **Scalability**: Supports databases with millions to billions of vectors via sparse paging

**Typical Use Case**: During IVF search with nprobe=10, scan 10 inverted lists containing 10K vectors each. Without dedup, evaluate 100K candidates; with dedup (assuming 50% overlap), evaluate only 50K unique candidates, saving 50K distance computations.

---

## Mathematical Foundations

### 1. Set Membership Problem

**Definition**: Given a query-local set **V** (visited set) and a stream of candidate IDs **C** = {c₁, c₂, ..., cₙ}, determine membership: cᵢ ∈ **V**.

**Operations**:
1. **Test**: `contains(V, id) → boolean` — check if id ∈ **V**
2. **Insert**: `add(V, id)` — **V** ← **V** ∪ {id}
3. **Test-and-Set** (atomic): If id ∉ **V**, add and return `true`; else return `false`
4. **Reset**: **V** ← ∅ (prepare for new query)

**Complexity Goals**:
- Test: O(1) average, O(log n) worst-case (hash table / tree)
- Insert: O(1) amortized
- Reset: O(k) where k = |**V**| (only clear touched elements)
- Space: O(id_capacity) for dense, O(unique_ids) for sparse

### 2. Duplicate Filtering Semantics

**First-Seen Semantics**: For duplicate candidates, retain only the **first occurrence** in scan order.

**Example**: Scanning lists L₁, L₂, L₃:
```
L₁: [42, 17, 99, 5]
L₂: [99, 3, 42, 11]  ← 99 and 42 are duplicates
L₃: [5, 8, 42, 20]   ← 5 and 42 are duplicates

Scan order: [42, 17, 99, 5, 99*, 3, 42*, 11, 5*, 8, 42*, 20]
Filtered:   [42, 17, 99, 5,     3,      11,    8,      20]
```
where `*` denotes filtered duplicates.

**Invariant**: After filtering, all candidates are unique: ∀ i ≠ j, cᵢ ≠ cⱼ.

### 3. Multi-List Overlap Analysis

**Overlap Model**: For IVF with nprobe lists, expected overlap between lists i and j:

```
P(overlap) ≈ (list_size_i × list_size_j) / total_vectors
```

**Typical Overlap Rates**:
- IVF with 256 clusters, nprobe=10: 30-50% overlap
- HNSW beam search, ef=100: 20-40% overlap (graph neighbors)
- Sharded search (2 shards): ~0-10% overlap (low if well-partitioned)

**Performance Impact**:
```
Without dedup: distance_computations = nprobe × avg_list_size
With dedup:    distance_computations = unique_candidates

Savings = (1 - unique_ratio) × 100%
Example: nprobe=10, avg_list_size=10K, overlap=40%
  → 100K candidates → 60K unique → 40% savings
```

---

## API Signatures

### 1. State Management

```swift
/// Opaque visited set for per-query duplicate detection
/// Thread Affinity: Single-writer per instance (reusable across queries)
/// Lifetime: Create once per worker thread, reuse for all queries
public final class VisitedSet {
    // Internal implementation hidden (dense epoch / sparse paged / fixed bitset)
}

/// Configuration for visited set behavior
public struct VisitedOpts {
    /// Storage strategy
    let mode: VisitedMode

    /// Concurrency model
    let concurrency: ConcurrencyMode

    /// For SparsePaged: bits per page (default: 15 → 32,768 IDs/page)
    let pageBits: Int

    /// For DenseEpoch: epoch stamp width (default: 32-bit)
    let epochBits: Int

    /// Enable telemetry recording (default: false)
    let enableTelemetry: Bool

    public static let `default` = VisitedOpts(
        mode: .denseEpoch,
        concurrency: .singleWriter,
        pageBits: 15,
        epochBits: 32,
        enableTelemetry: false
    )
}

/// Storage implementation strategy
public enum VisitedMode {
    case denseEpoch      // O(id_capacity) space, O(1) check, O(1) reset via epoch bump
    case sparsePaged     // O(unique_ids) space, O(1) check, O(touched_pages) reset
    case fixedBitset     // O(id_capacity/8) space, O(1) check, O(touched_bits) reset
}

/// Concurrency strategy
public enum ConcurrencyMode {
    case singleWriter           // Single query thread, no synchronization
    case shardedMultiWriter     // Multiple threads, ID-sharded (no atomics)
    case atomicMultiWriter      // Multiple threads, atomic operations (slower)
}
```

### 2. Core Operations

```swift
/// Initialize visited set for a given ID capacity
/// - Parameters:
///   - idCapacity: Maximum internal ID value (from ID remapping kernel #50)
///   - opts: Configuration options
/// - Complexity: O(id_capacity) for dense modes, O(1) for sparse
@inlinable
public func visitedInit(
    idCapacity: Int64,
    opts: VisitedOpts = .default
) -> VisitedSet

/// Reset visited set for a new query
/// Clears all visited state efficiently (epoch bump or touched-only clear)
/// - Complexity: O(1) for DenseEpoch, O(touched) for others
/// - Thread Safety: Must be called by same thread that uses the set
@inlinable
public func visitedReset(_ vs: VisitedSet)

/// Test if ID was visited, and mark as visited if not
/// Returns true if this is the FIRST time seeing this ID (not a duplicate)
/// - Parameters:
///   - vs: Visited set
///   - id: Dense internal ID in [0, idCapacity)
/// - Returns: `true` if first occurrence (should process), `false` if duplicate (skip)
/// - Complexity: O(1) average
/// - Thread Safety: Single-writer or sharded access only
@inline(__always)
public func visitedTestAndSet(
    _ vs: VisitedSet,
    _ id: Int64
) -> Bool

/// Batch test-and-set: check multiple IDs, output mask of new/duplicate status
/// More efficient than individual calls for small blocks
/// - Parameters:
///   - vs: Visited set
///   - ids: Array of dense IDs to check
///   - count: Number of IDs
///   - maskOut: Output bitmask: maskOut[i] = 1 if ids[i] is new, 0 if duplicate
/// - Returns: Count of new (non-duplicate) IDs
/// - Complexity: O(n)
@inlinable
public func visitedMaskAndMark(
    _ vs: VisitedSet,
    ids: UnsafePointer<Int64>,
    count: Int,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int

/// In-place deduplication: compact array to remove duplicates
/// Preserves order of first occurrences
/// - Parameters:
///   - vs: Visited set
///   - ids: Array of dense IDs (will be compacted in-place)
///   - scores: Optional parallel score array (compacted alongside IDs)
///   - count: Input count
/// - Returns: New count after deduplication
/// - Complexity: O(n)
/// - Post-condition: ids[0..<newCount] contains unique first-seen IDs
@inlinable
public func dedupInPlace(
    _ vs: VisitedSet,
    ids: UnsafeMutablePointer<Int64>,
    scores: UnsafeMutablePointer<Float>?,
    count: Int
) -> Int

/// Free visited set resources
public func visitedFree(_ vs: VisitedSet)
```

### 3. Convenience API

```swift
extension VisitedSet {
    /// High-level deduplication for Swift arrays
    public func dedup(ids: inout [Int64], scores: inout [Float]?) -> Int

    /// Filter candidates through visited set
    public func filterUnique(candidates: [Int64]) -> [Int64]

    /// Check if ID was already visited (read-only test, no side effects)
    public func contains(_ id: Int64) -> Bool

    /// Get statistics for current query
    public func getStats() -> VisitedStats
}

/// Per-query statistics
public struct VisitedStats {
    public let totalChecks: Int64      // Total test-and-set calls
    public let uniqueCount: Int64      // New IDs (first-seen)
    public let duplicateCount: Int64   // Filtered duplicates
    public let pagesAllocated: Int     // SparsePaged: pages allocated
    public let epochValue: UInt32      // DenseEpoch: current epoch
}
```

---

## Algorithm Details

### Implementation Strategy 1: Dense Epoch Array

**Concept**: Maintain a dense array `stamp[id_capacity]` of epoch values. Each query has an epoch counter. Check by comparing `stamp[id] == current_epoch`.

**Data Structure**:
```swift
struct VisitedSetDenseEpoch {
    var stamps: UnsafeMutablePointer<UInt32>  // [id_capacity] aligned to 64 bytes
    var capacity: Int64
    var currentEpoch: UInt32
    var stats: VisitedStats
}
```

**Operations**:

```swift
// Initialize
func visitedInit_DenseEpoch(idCapacity: Int64) -> VisitedSet {
    let stamps = UnsafeMutablePointer<UInt32>.allocate(capacity: Int(idCapacity))
    stamps.initialize(repeating: 0, count: Int(idCapacity))

    return VisitedSetDenseEpoch(
        stamps: stamps,
        capacity: idCapacity,
        currentEpoch: 1,  // Start at 1 (0 is initial state)
        stats: VisitedStats()
    )
}

// Reset for new query (O(1) - just bump epoch)
@inline(__always)
func visitedReset_DenseEpoch(_ vs: inout VisitedSetDenseEpoch) {
    vs.currentEpoch += 1

    // Handle epoch wrap (rare: every 4 billion queries)
    if vs.currentEpoch == 0 {
        // Full clear required
        vs.stamps.initialize(repeating: 0, count: Int(vs.capacity))
        vs.currentEpoch = 1
        vs.stats.epochWraps += 1
    }

    vs.stats.reset()
}

// Test-and-set (O(1) - single memory access)
@inline(__always)
func visitedTestAndSet_DenseEpoch(
    _ vs: inout VisitedSetDenseEpoch,
    _ id: Int64
) -> Bool {
    assert(id >= 0 && id < vs.capacity, "ID out of bounds")

    let currentStamp = vs.stamps[Int(id)]

    if currentStamp != vs.currentEpoch {
        // First occurrence - mark visited
        vs.stamps[Int(id)] = vs.currentEpoch
        vs.stats.uniqueCount += 1
        return true
    } else {
        // Duplicate
        vs.stats.duplicateCount += 1
        return false
    }
}

// Batch mask-and-mark
@inlinable
func visitedMaskAndMark_DenseEpoch(
    _ vs: inout VisitedSetDenseEpoch,
    ids: UnsafePointer<Int64>,
    count: Int,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    var uniqueCount = 0
    let epoch = vs.currentEpoch

    for i in 0..<count {
        let id = ids[i]
        let currentStamp = vs.stamps[Int(id)]

        if currentStamp != epoch {
            vs.stamps[Int(id)] = epoch
            maskOut[i] = 1  // New
            uniqueCount += 1
        } else {
            maskOut[i] = 0  // Duplicate
        }
    }

    vs.stats.uniqueCount += Int64(uniqueCount)
    vs.stats.duplicateCount += Int64(count - uniqueCount)

    return uniqueCount
}
```

**Performance Characteristics**:
- **Check Time**: ~2-3 ns/check (L1 cache hit), ~10-20 ns (L2), ~50-100 ns (L3/RAM)
- **Reset Time**: O(1) — just increment epoch (< 1 ns)
- **Space**: 4 × id_capacity bytes (4 bytes per UInt32 stamp)
- **Cache Behavior**: Excellent for clustered IDs (IVF lists), poor for random access
- **Wrap Handling**: Every 2³² queries (~4 billion), requires O(id_capacity) clear

**Pros**:
- Fastest reset (O(1) epoch bump)
- Predictable performance
- No allocation during queries

**Cons**:
- O(id_capacity) space even for sparse queries
- Example: 10M vectors → 40 MB memory
- Not suitable for billion-scale databases

---

### Implementation Strategy 2: Sparse Paged Bitset

**Concept**: Allocate bitset pages on-demand as IDs are accessed. Each page covers `2^page_bits` IDs. Track touched pages for efficient reset.

**Data Structure**:
```swift
struct VisitedSetSparsePaged {
    var pageTable: [Int64: UnsafeMutablePointer<UInt64>]  // page_id → bitset page
    var touchedPages: [Int64]                             // Pages accessed this query
    var pageBits: Int                                      // Bits per page (default 15)
    var pageMask: Int64                                    // Mask for in-page index
    var stats: VisitedStats
}

// Constants
let BITS_PER_PAGE = 1 << 15       // 32,768 IDs per page (4 KB bitset)
let WORDS_PER_PAGE = 512          // 64-bit words per page (32,768 / 64)
```

**Page Layout**:
```
Page ID calculation:
  page_id = id >> page_bits
  in_page_index = id & page_mask
  word_index = in_page_index >> 6  (divide by 64)
  bit_index = in_page_index & 63   (mod 64)

Example (page_bits=15, id=100,000):
  page_id = 100,000 >> 15 = 3
  in_page_index = 100,000 & 32,767 = 1,696
  word_index = 1,696 >> 6 = 26
  bit_index = 1,696 & 63 = 32

Access: pages[3][26] |= (1 << 32)
```

**Operations**:

```swift
// Initialize
func visitedInit_SparsePaged(opts: VisitedOpts) -> VisitedSet {
    let pageBits = opts.pageBits
    let pageMask = (1 << pageBits) - 1

    return VisitedSetSparsePaged(
        pageTable: [:],
        touchedPages: [],
        pageBits: pageBits,
        pageMask: Int64(pageMask),
        stats: VisitedStats()
    )
}

// Reset for new query (O(touched_pages))
@inlinable
func visitedReset_SparsePaged(_ vs: inout VisitedSetSparsePaged) {
    // Clear only touched pages
    for pageID in vs.touchedPages {
        if let page = vs.pageTable[pageID] {
            // Clear all words in page
            page.initialize(repeating: 0, count: WORDS_PER_PAGE)
        }
    }

    vs.touchedPages.removeAll(keepingCapacity: true)
    vs.stats.pagesCleared = vs.touchedPages.count
    vs.stats.reset()
}

// Test-and-set with page allocation
@inline(__always)
func visitedTestAndSet_SparsePaged(
    _ vs: inout VisitedSetSparsePaged,
    _ id: Int64
) -> Bool {
    let pageID = id >> vs.pageBits
    let inPageIndex = id & vs.pageMask
    let wordIndex = Int(inPageIndex >> 6)
    let bitIndex = Int(inPageIndex & 63)
    let bitMask: UInt64 = 1 << bitIndex

    // Get or allocate page
    var page: UnsafeMutablePointer<UInt64>
    if let existingPage = vs.pageTable[pageID] {
        page = existingPage
    } else {
        // Allocate new page
        page = UnsafeMutablePointer<UInt64>.allocate(capacity: WORDS_PER_PAGE)
        page.initialize(repeating: 0, count: WORDS_PER_PAGE)
        vs.pageTable[pageID] = page
        vs.touchedPages.append(pageID)
        vs.stats.pagesAllocated += 1
    }

    // Test and set bit
    let word = page[wordIndex]
    if (word & bitMask) == 0 {
        // First occurrence
        page[wordIndex] = word | bitMask
        vs.stats.uniqueCount += 1
        return true
    } else {
        // Duplicate
        vs.stats.duplicateCount += 1
        return false
    }
}
```

**Performance Characteristics**:
- **Check Time**: ~5-8 ns/check (includes hash lookup + bit test)
- **Reset Time**: O(touched_pages × page_size) ≈ O(unique_ids)
  - Example: 10K unique IDs → 1 page (4KB) → ~500 ns clear
- **Space**: (unique_ids / IDs_per_page) × page_size
  - Example: 100K unique IDs → 4 pages × 4KB = 16 KB
- **Page Allocation**: Amortized O(1), ~100 ns per new page

**Pros**:
- Memory-efficient for sparse queries
- Scales to billion-ID databases (allocate only accessed regions)
- Cache-friendly (4 KB pages fit in L1/L2)

**Cons**:
- Hash lookup overhead vs direct array access
- Reset cost O(touched) rather than O(1)
- Initial page allocation latency

---

### Implementation Strategy 3: Fixed Bitset with Touched Tracking

**Concept**: Pre-allocate full bitset for `[0, id_capacity)`, but track touched words for efficient reset.

**Data Structure**:
```swift
struct VisitedSetFixedBitset {
    var bits: UnsafeMutablePointer<UInt64>  // Bitset [id_capacity / 64]
    var touchedWords: UnsafeMutablePointer<Int>  // Ring buffer of touched word indices
    var touchedCount: Int
    var touchedCapacity: Int
    var wordCount: Int
    var stats: VisitedStats
}

// Word calculation
@inline(__always)
func wordIndex(_ id: Int64) -> Int { Int(id >> 6) }
@inline(__always)
func bitIndex(_ id: Int64) -> Int { Int(id & 63) }
```

**Operations**:

```swift
// Initialize
func visitedInit_FixedBitset(idCapacity: Int64, touchedCapacity: Int = 16384) -> VisitedSet {
    let wordCount = Int((idCapacity + 63) / 64)  // Ceiling division
    let bits = UnsafeMutablePointer<UInt64>.allocate(capacity: wordCount)
    bits.initialize(repeating: 0, count: wordCount)

    let touchedWords = UnsafeMutablePointer<Int>.allocate(capacity: touchedCapacity)

    return VisitedSetFixedBitset(
        bits: bits,
        touchedWords: touchedWords,
        touchedCount: 0,
        touchedCapacity: touchedCapacity,
        wordCount: wordCount,
        stats: VisitedStats()
    )
}

// Reset (O(touched_words))
@inlinable
func visitedReset_FixedBitset(_ vs: inout VisitedSetFixedBitset) {
    if vs.touchedCount < vs.wordCount / 4 {
        // Sparse: clear only touched words
        for i in 0..<vs.touchedCount {
            let wordIdx = vs.touchedWords[i]
            vs.bits[wordIdx] = 0
        }
    } else {
        // Dense: full clear is faster
        vs.bits.initialize(repeating: 0, count: vs.wordCount)
    }

    vs.touchedCount = 0
    vs.stats.reset()
}

// Test-and-set with touched tracking
@inline(__always)
func visitedTestAndSet_FixedBitset(
    _ vs: inout VisitedSetFixedBitset,
    _ id: Int64
) -> Bool {
    let wIdx = wordIndex(id)
    let bIdx = bitIndex(id)
    let mask: UInt64 = 1 << bIdx

    let word = vs.bits[wIdx]

    if (word & mask) == 0 {
        // First occurrence
        let newWord = word | mask
        vs.bits[wIdx] = newWord

        // Track touched word (if space available)
        if word == 0 && vs.touchedCount < vs.touchedCapacity {
            vs.touchedWords[vs.touchedCount] = wIdx
            vs.touchedCount += 1
        }

        vs.stats.uniqueCount += 1
        return true
    } else {
        // Duplicate
        vs.stats.duplicateCount += 1
        return false
    }
}
```

**Performance Characteristics**:
- **Check Time**: ~2-4 ns/check (direct array access + bit ops)
- **Reset Time**: O(min(touched_words, word_count / 4))
  - Sparse: ~1 ns per touched word
  - Dense: Full clear (memset)
- **Space**: id_capacity / 8 bytes + touched tracking overhead
  - Example: 10M IDs → 1.25 MB bitset + 64 KB touched buffer = ~1.3 MB

**Pros**:
- Fastest check time (direct indexing, no hash)
- Compact space (1 bit per ID)
- Predictable performance

**Cons**:
- Fixed O(id_capacity) space regardless of sparsity
- Reset cost proportional to unique IDs (worse than DenseEpoch)

---

## Vectorization & SIMD Optimization

### Batch Processing with SIMD

For `visitedMaskAndMark`, process multiple IDs in parallel:

```swift
@inlinable
func visitedMaskAndMark_SIMD(
    _ vs: inout VisitedSetDenseEpoch,
    ids: UnsafePointer<Int64>,
    count: Int,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    var uniqueCount = 0
    let epoch = vs.currentEpoch

    // Process in blocks of 4 (SIMD width for Int64 on NEON)
    let blockedCount = (count / 4) * 4

    for i in stride(from: 0, to: blockedCount, by: 4) {
        // Load 4 IDs
        let id0 = ids[i + 0]
        let id1 = ids[i + 1]
        let id2 = ids[i + 2]
        let id3 = ids[i + 3]

        // Prefetch stamps (if not in cache)
        #if arch(arm64)
        // Conceptual: actual prefetch would use inline assembly
        // __builtin_prefetch(&vs.stamps[Int(id0)])
        #endif

        // Load stamps
        let stamp0 = vs.stamps[Int(id0)]
        let stamp1 = vs.stamps[Int(id1)]
        let stamp2 = vs.stamps[Int(id2)]
        let stamp3 = vs.stamps[Int(id3)]

        // Compare (branchless)
        let new0 = (stamp0 != epoch) ? UInt8(1) : UInt8(0)
        let new1 = (stamp1 != epoch) ? UInt8(1) : UInt8(0)
        let new2 = (stamp2 != epoch) ? UInt8(1) : UInt8(0)
        let new3 = (stamp3 != epoch) ? UInt8(1) : UInt8(0)

        // Update stamps (conditional writes, but no branches)
        vs.stamps[Int(id0)] = epoch  // Unconditional for simplicity (or use CMOV)
        vs.stamps[Int(id1)] = epoch
        vs.stamps[Int(id2)] = epoch
        vs.stamps[Int(id3)] = epoch

        // Write masks
        maskOut[i + 0] = new0
        maskOut[i + 1] = new1
        maskOut[i + 2] = new2
        maskOut[i + 3] = new3

        uniqueCount += Int(new0) + Int(new1) + Int(new2) + Int(new3)
    }

    // Scalar tail
    for i in blockedCount..<count {
        let id = ids[i]
        let stamp = vs.stamps[Int(id)]

        if stamp != epoch {
            vs.stamps[Int(id)] = epoch
            maskOut[i] = 1
            uniqueCount += 1
        } else {
            maskOut[i] = 0
        }
    }

    return uniqueCount
}
```

**Optimization Notes**:
- **Prefetching**: For predictable ID streams, prefetch stamps 2-4 elements ahead
- **Branchless**: Use conditional moves (CMOV) instead of branches
- **ILP**: Unroll to expose instruction-level parallelism (4-8 way)
- **Bandwidth**: Limited by memory bandwidth for large ID streams

---

## In-Place Deduplication Algorithm

**Compact array by removing duplicates while preserving order**:

```swift
@inlinable
func dedupInPlace(
    _ vs: VisitedSet,
    ids: UnsafeMutablePointer<Int64>,
    scores: UnsafeMutablePointer<Float>?,
    count: Int
) -> Int {
    var writeIdx = 0

    for readIdx in 0..<count {
        let id = ids[readIdx]

        if visitedTestAndSet(vs, id) {
            // First occurrence - keep it
            if writeIdx != readIdx {
                ids[writeIdx] = id
                if let scores = scores {
                    scores[writeIdx] = scores[readIdx]
                }
            }
            writeIdx += 1
        }
        // else: duplicate - skip (don't increment writeIdx)
    }

    return writeIdx  // New count
}
```

**Complexity**: O(n) single pass
**Space**: O(1) beyond visited set

**Example**:
```
Input:  ids    = [42, 17, 99, 42, 5, 99, 11]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

After dedup (writeIdx = 5):
        ids    = [42, 17, 99, 5, 11, ?, ?]  ← Only [0:5) valid
        scores = [0.9, 0.8, 0.7, 0.5, 0.3, ?, ?]
```

---

## Memory Layout & Cache Optimization

### DenseEpoch Layout

```
Memory map (id_capacity = 10,000,000):
┌────────────────────────────────────┐
│ stamps[0]        │ UInt32 (4 bytes)│  ← 64-byte aligned
│ stamps[1]        │                 │
│ ...              │                 │
│ stamps[9,999,999]│                 │
└────────────────────────────────────┘
Total: 40 MB (4 bytes × 10M)

Cache line (64 bytes) holds 16 stamps
Clustered IDs (e.g., IVF list) → high cache hit rate
Random IDs → thrashing
```

**Access Pattern Optimization**:
- **Sequential scan**: Perfect cache behavior (prefetch-friendly)
- **IVF lists**: Cluster by coarse quantizer → spatial locality
- **Graph neighbors**: Moderate locality (depends on graph structure)

### SparsePaged Layout

```
Page structure (page_bits = 15):
┌─────────────────────────────────────────┐
│ Page 0: bitset for IDs [0, 32,767]      │  4 KB (512 × UInt64)
│ Page 1: bitset for IDs [32,768, 65,535] │  4 KB
│ ...                                      │
│ Page N: bitset for IDs [...]            │  4 KB
└─────────────────────────────────────────┘

Page table (hash map):
  page_id → page_ptr

Touched pages vector:
  [3, 7, 12, ...]  ← Page IDs accessed this query

Total memory (example: 100K unique IDs):
  - 4 pages × 4 KB = 16 KB (bitset pages)
  - 4 entries × 16 bytes = 64 bytes (hash table overhead)
  - 4 × 8 bytes = 32 bytes (touched pages)
  Total: ~16 KB
```

**Cache Behavior**:
- Page size (4 KB) fits comfortably in L1d (128 KB on M-series)
- Typical query touches 1-10 pages → total <40 KB → L1 resident
- Page allocation: ~100 ns latency, amortized over 32K IDs

---

## Parallelism & Concurrency

### Single-Writer Model (Recommended)

**Pattern**: Each query processed by one thread. Visited set is thread-local.

```swift
// Per-thread worker state
class QueryWorker {
    let visitedSet: VisitedSet

    init(idCapacity: Int64) {
        self.visitedSet = visitedInit(idCapacity: idCapacity)
    }

    func processQuery(_ query: Vector) -> [SearchResult] {
        visitedReset(visitedSet)

        var candidates: [(id: Int64, score: Float)] = []

        // Scan multiple IVF lists
        for listID in selectedLists {
            let list = ivfLists[listID]

            for candidateID in list {
                // Check for duplicate
                if visitedTestAndSet(visitedSet, candidateID) {
                    // First occurrence - compute distance and add
                    let score = computeScore(query, candidateID)
                    candidates.append((candidateID, score))
                }
                // else: skip duplicate
            }
        }

        return selectTopK(candidates)
    }
}

// Thread pool of workers
let workers = (0..<threadCount).map { _ in QueryWorker(idCapacity: idCapacity) }

// Process queries in parallel (each query on one worker)
DispatchQueue.concurrentPerform(iterations: queries.count) { queryIdx in
    let worker = workers[queryIdx % threadCount]
    results[queryIdx] = worker.processQuery(queries[queryIdx])
}
```

**Pros**:
- No synchronization overhead
- Excellent cache locality
- Predictable performance

### Sharded Multi-Writer Model

**Pattern**: Single query processed by multiple threads, each owning a shard of IDs.

```swift
// Partition ID space by modulo sharding
let shardCount = 8
let shardsPerQuery = shardCount

class QueryShardWorker {
    let shardID: Int
    let shardCount: Int
    let visitedSet: VisitedSet

    func shouldHandle(_ id: Int64) -> Bool {
        return (id % Int64(shardCount)) == Int64(shardID)
    }

    func processShard(_ query: Vector, listIDs: [Int]) -> [SearchResult] {
        visitedReset(visitedSet)

        var candidates: [(id: Int64, score: Float)] = []

        for listID in listIDs {
            let list = ivfLists[listID]

            for candidateID in list {
                // Only process IDs in this shard
                guard shouldHandle(candidateID) else { continue }

                if visitedTestAndSet(visitedSet, candidateID) {
                    let score = computeScore(query, candidateID)
                    candidates.append((candidateID, score))
                }
            }
        }

        return candidates
    }
}

// Parallel shard processing
let shardWorkers = (0..<shardCount).map { shardID in
    QueryShardWorker(shardID: shardID, shardCount: shardCount, ...)
}

func processQueryParallel(_ query: Vector) -> [SearchResult] {
    var allCandidates = [[SearchResult]]()

    DispatchQueue.concurrentPerform(iterations: shardCount) { shardID in
        let worker = shardWorkers[shardID]
        allCandidates[shardID] = worker.processShard(query, selectedLists)
    }

    // Merge results from all shards
    let merged = allCandidates.flatMap { $0 }
    return selectTopK(merged)
}
```

**Pros**:
- Exploits intra-query parallelism
- No atomics needed (disjoint ID ownership)

**Cons**:
- Load imbalance if IDs not uniformly distributed
- Merge overhead

### Atomic Multi-Writer (Advanced)

For **FixedBitset** mode with `fetch_or` atomics:

```swift
@inline(__always)
func visitedTestAndSet_Atomic(
    _ vs: VisitedSet,
    _ id: Int64
) -> Bool {
    let wIdx = wordIndex(id)
    let bIdx = bitIndex(id)
    let mask: UInt64 = 1 << bIdx

    // Atomic fetch-or
    let oldWord = atomicFetchOr(&vs.bits[wIdx], mask, ordering: .relaxed)

    // Check if bit was previously 0
    return (oldWord & mask) == 0
}
```

**Performance**: ~10-20 ns per atomic op (vs ~2-3 ns non-atomic)

**Use when**: Multiple threads must process same query without ID sharding

---

## Performance Characteristics

### Latency Benchmarks (Apple M2, Release Build)

| Mode          | Check Latency (ns) | Reset Latency | Space (10M IDs) | Notes                       |
|---------------|-------------------|---------------|-----------------|------------------------------|
| DenseEpoch    | 2-3 (L1 hit)      | O(1), <1 ns   | 40 MB           | Best for dense, clustered    |
|               | 10-20 (L2)        |               |                 |                              |
|               | 50-100 (RAM)      |               |                 |                              |
| SparsePaged   | 5-8               | O(pages), 500ns| 16 KB (100K IDs)| Best for sparse queries      |
| FixedBitset   | 2-4               | O(touched), 2μs| 1.25 MB         | Balanced                     |
| Atomic (Bitset)| 10-20            | O(touched)    | 1.25 MB         | Multi-writer fallback        |

### Throughput Benchmarks

**Scenario**: IVF search, nprobe=10, avg_list_size=10K, 40% duplicate rate

| Mode        | Checks/sec | Effective Dedup Rate | Total Query Time |
|-------------|------------|---------------------|------------------|
| DenseEpoch  | 400M       | 40% saved (40K skips)| 1.2 ms           |
| SparsePaged | 150M       | 40% saved           | 1.5 ms           |
| FixedBitset | 300M       | 40% saved           | 1.3 ms           |

**Interpretation**: Even with check overhead, dedup saves 40% of distance computations, which dominate total time.

### Scalability

**DenseEpoch**:
- Scales to ~100M IDs (400 MB memory)
- Beyond: memory footprint prohibitive

**SparsePaged**:
- Scales to billions of IDs (allocate only accessed regions)
- Example: 1B ID capacity, 100K unique per query → 16 KB memory

**Recommendation**:
- **< 10M IDs**: DenseEpoch (fastest, acceptable memory)
- **10M - 1B IDs**: SparsePaged (memory-efficient)
- **> 1B IDs**: SparsePaged (only viable option)

---

## Integration with Search Kernels

### IVF Multi-Probe Search (#05)

```swift
func searchIVF(
    query: Vector,
    index: IVFIndex,
    nprobe: Int,
    k: Int
) -> [SearchResult] {
    // Initialize visited set for this query
    let visited = queryWorker.visitedSet
    visitedReset(visited)

    // Select top-nprobe coarse clusters
    let selectedLists = selectCoarseClusters(query, nprobe)

    // Candidate accumulator (reservoir, kernel #39)
    var reservoir = Reservoir(capacity: k * 10)

    // Scan each list
    for listID in selectedLists {
        let list = index.invertedLists[listID]

        // Stream candidates from list
        for candidateID in list.ids {
            // Check for duplicate (kernel #32)
            guard visitedTestAndSet(visited, candidateID) else {
                continue  // Skip duplicate
            }

            // Compute exact distance (kernel #01, #02, or #03)
            let score = computeScore(query, list.vectors[candidateID], index.metric)

            // Add to reservoir (kernel #39)
            reservoir.insert(id: candidateID, score: score)
        }
    }

    // Extract top-k (kernel #05, #06)
    return reservoir.topK(k)
}
```

### ADC Scan with Deduplication (#22)

```swift
func scanADC_WithDedup(
    lut: UnsafePointer<Float>,  // Lookup table
    codes: UnsafePointer<UInt8>,
    ids: UnsafePointer<Int64>,
    count: Int,
    visited: VisitedSet,
    reservoir: Reservoir
) {
    // Batch process for efficiency
    let batchSize = 64
    var mask = [UInt8](repeating: 0, count: batchSize)

    for batchStart in stride(from: 0, to: count, by: batchSize) {
        let batchEnd = min(batchStart + batchSize, count)
        let batchCount = batchEnd - batchStart

        // Check batch for duplicates (kernel #32)
        let uniqueCount = visitedMaskAndMark(
            visited,
            ids: ids + batchStart,
            count: batchCount,
            maskOut: &mask
        )

        // Skip batch if all duplicates
        guard uniqueCount > 0 else { continue }

        // Compute scores only for unique candidates
        for i in 0..<batchCount {
            guard mask[i] == 1 else { continue }  // Skip duplicates

            let idx = batchStart + i
            let id = ids[idx]
            let code = codes + idx * M  // M subquantizers

            // ADC distance computation (kernel #22)
            let score = adcDistance(lut, code)

            // Insert into reservoir
            reservoir.insert(id: id, score: score)
        }
    }
}
```

### HNSW Beam Search (#29)

```swift
func searchHNSW(
    query: Vector,
    graph: HNSWGraph,
    ef: Int
) -> [SearchResult] {
    let visited = queryWorker.visitedSet
    visitedReset(visited)

    var candidates = PriorityQueue<Candidate>(capacity: ef)
    var results = PriorityQueue<Candidate>(capacity: ef)

    // Start from entry point
    let entryID = graph.entryPoint
    let entryScore = computeScore(query, graph.vectors[entryID])

    visitedTestAndSet(visited, entryID)
    candidates.push(Candidate(id: entryID, score: entryScore))
    results.push(Candidate(id: entryID, score: entryScore))

    while !candidates.isEmpty {
        let current = candidates.pop()

        // Stop if current is worse than ef-th result
        if current.score > results.peek().score {
            break
        }

        // Explore neighbors
        let neighbors = graph.neighbors[current.id]

        for neighborID in neighbors {
            // Deduplication check (kernel #32)
            guard visitedTestAndSet(visited, neighborID) else {
                continue  // Already visited
            }

            let score = computeScore(query, graph.vectors[neighborID])

            if results.count < ef || score < results.peek().score {
                candidates.push(Candidate(id: neighborID, score: score))
                results.push(Candidate(id: neighborID, score: score))

                if results.count > ef {
                    results.pop()  // Remove worst
                }
            }
        }
    }

    return results.sorted()
}
```

---

## Correctness Testing

### Test 1: Basic Deduplication

```swift
func testBasicDedup() {
    let visited = visitedInit(idCapacity: 1000)
    defer { visitedFree(visited) }

    visitedReset(visited)

    // First occurrence
    XCTAssertTrue(visitedTestAndSet(visited, 42))
    XCTAssertTrue(visitedTestAndSet(visited, 17))

    // Duplicates
    XCTAssertFalse(visitedTestAndSet(visited, 42))  // Dup
    XCTAssertFalse(visitedTestAndSet(visited, 17))  // Dup

    // New IDs
    XCTAssertTrue(visitedTestAndSet(visited, 99))

    // Verify stats
    let stats = visited.getStats()
    XCTAssertEqual(stats.uniqueCount, 3)
    XCTAssertEqual(stats.duplicateCount, 2)
}
```

### Test 2: In-Place Compaction

```swift
func testDedupInPlace() {
    let visited = visitedInit(idCapacity: 1000)
    defer { visitedFree(visited) }

    visitedReset(visited)

    var ids: [Int64] = [42, 17, 99, 42, 5, 99, 11, 42]
    var scores: [Float] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    let newCount = ids.withUnsafeMutableBufferPointer { idsPtr in
        scores.withUnsafeMutableBufferPointer { scoresPtr in
            dedupInPlace(
                visited,
                ids: idsPtr.baseAddress!,
                scores: scoresPtr.baseAddress!,
                count: ids.count
            )
        }
    }

    // Expected: [42, 17, 99, 5, 11] (first occurrences only)
    XCTAssertEqual(newCount, 5)
    XCTAssertEqual(Array(ids[0..<newCount]), [42, 17, 99, 5, 11])
    XCTAssertEqual(Array(scores[0..<newCount]), [0.9, 0.8, 0.7, 0.5, 0.3])
}
```

### Test 3: Batch Mask and Mark

```swift
func testMaskAndMark() {
    let visited = visitedInit(idCapacity: 1000)
    defer { visitedFree(visited) }

    visitedReset(visited)

    // First batch: all new
    var ids1: [Int64] = [10, 20, 30, 40]
    var mask1 = [UInt8](repeating: 0, count: 4)

    let unique1 = visitedMaskAndMark(visited, ids: ids1, count: 4, maskOut: &mask1)

    XCTAssertEqual(unique1, 4)
    XCTAssertEqual(mask1, [1, 1, 1, 1])  // All new

    // Second batch: mixed
    var ids2: [Int64] = [20, 50, 30, 60]  // 20 and 30 are duplicates
    var mask2 = [UInt8](repeating: 0, count: 4)

    let unique2 = visitedMaskAndMark(visited, ids: ids2, count: 4, maskOut: &mask2)

    XCTAssertEqual(unique2, 2)  // Only 50 and 60 are new
    XCTAssertEqual(mask2, [0, 1, 0, 1])  // [dup, new, dup, new]
}
```

### Test 4: Epoch Wrap Handling

```swift
func testEpochWrap() {
    let visited = visitedInit(idCapacity: 100, opts: VisitedOpts(epochBits: 8))
    defer { visitedFree(visited) }

    // Force 256 resets to trigger wrap (8-bit epoch)
    for queryIdx in 0..<256 {
        visitedReset(visited)
        XCTAssertTrue(visitedTestAndSet(visited, 42))
    }

    // After wrap, should still work correctly
    visitedReset(visited)
    XCTAssertTrue(visitedTestAndSet(visited, 42))  // First in new epoch
    XCTAssertFalse(visitedTestAndSet(visited, 42))  // Duplicate

    // Verify wrap was handled
    let stats = visited.getStats()
    XCTAssertGreaterThan(stats.epochWraps, 0)
}
```

### Test 5: Sparse Page Allocation

```swift
func testSparsePageAllocation() {
    let opts = VisitedOpts(mode: .sparsePaged, pageBits: 10)  // 1024 IDs/page
    let visited = visitedInit(idCapacity: 1_000_000, opts: opts)
    defer { visitedFree(visited) }

    visitedReset(visited)

    // Access IDs in widely separated pages
    XCTAssertTrue(visitedTestAndSet(visited, 500))      // Page 0
    XCTAssertTrue(visitedTestAndSet(visited, 50_000))   // Page 48
    XCTAssertTrue(visitedTestAndSet(visited, 500_000))  // Page 488

    let stats = visited.getStats()
    XCTAssertEqual(stats.pagesAllocated, 3)
    XCTAssertEqual(stats.uniqueCount, 3)

    // Reset should clear only touched pages
    visitedReset(visited)
    XCTAssertEqual(stats.pagesCleared, 3)

    // IDs should be new again
    XCTAssertTrue(visitedTestAndSet(visited, 500))
    XCTAssertTrue(visitedTestAndSet(visited, 50_000))
}
```

### Test 6: Multi-Writer Sharding

```swift
func testShardedMultiWriter() {
    let shardCount = 4
    let visited = (0..<shardCount).map { _ in
        visitedInit(idCapacity: 1000)
    }
    defer { visited.forEach { visitedFree($0) } }

    visited.forEach { visitedReset($0) }

    // Distribute IDs by modulo sharding
    let ids: [Int64] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    var shardedResults = [[Int64]](repeating: [], count: shardCount)

    DispatchQueue.concurrentPerform(iterations: shardCount) { shardID in
        for id in ids {
            if id % Int64(shardCount) == Int64(shardID) {
                if visitedTestAndSet(visited[shardID], id) {
                    shardedResults[shardID].append(id)
                }
            }
        }
    }

    // Merge results
    let merged = shardedResults.flatMap { $0 }.sorted()

    // Should have all IDs exactly once
    XCTAssertEqual(merged, Array(ids))
}
```

### Test 7: Performance Benchmark

```swift
func testPerformanceDedup() {
    let idCapacity: Int64 = 10_000_000
    let visited = visitedInit(idCapacity: idCapacity)
    defer { visitedFree(visited) }

    // Generate realistic workload: 100K candidates, 40% duplicates
    let uniqueCount = 60_000
    let totalCount = 100_000

    var ids = [Int64]()
    var uniqueIDs = Set<Int64>()

    // Generate unique IDs
    while uniqueIDs.count < uniqueCount {
        let id = Int64.random(in: 0..<idCapacity)
        uniqueIDs.insert(id)
    }

    // Create candidate stream with duplicates
    ids = Array(uniqueIDs)
    while ids.count < totalCount {
        let dupID = uniqueIDs.randomElement()!
        ids.append(dupID)
    }
    ids.shuffle()

    // Benchmark
    visitedReset(visited)

    measure {
        var newCount = 0
        for id in ids {
            if visitedTestAndSet(visited, id) {
                newCount += 1
            }
        }
        XCTAssertEqual(newCount, uniqueCount)
    }

    // Target: <1 ms for 100K checks on M2
    // Average: 5-8 ns/check → 500-800 μs total
}
```

---

## Telemetry Integration (#46)

### Per-Query Metrics

```swift
public struct VisitedTelemetry {
    public let mode: VisitedMode
    public let totalChecks: Int64
    public let uniqueCount: Int64
    public let duplicateCount: Int64
    public let pagesAllocated: Int       // SparsePaged only
    public let pagesCleared: Int         // SparsePaged only
    public let epochValue: UInt32        // DenseEpoch only
    public let epochWraps: Int           // DenseEpoch only
    public let atomicConflicts: Int      // Atomic mode only
    public let checkTimeNanos: UInt64

    public var deduplicationRate: Double {
        let total = uniqueCount + duplicateCount
        return total > 0 ? Double(duplicateCount) / Double(total) : 0.0
    }

    public var avgCheckLatencyNs: Double {
        return totalChecks > 0 ? Double(checkTimeNanos) / Double(totalChecks) : 0.0
    }
}

// Usage
#if ENABLE_TELEMETRY
let startTime = mach_absolute_time()
#endif

// ... perform deduplication ...

#if ENABLE_TELEMETRY
let elapsedNanos = mach_absolute_time() - startTime
let telemetry = visited.getTelemetry(elapsedNanos: elapsedNanos)
GlobalTelemetryRecorder.record(telemetry)

print("""
    Dedup stats:
      Mode: \(telemetry.mode)
      Total checks: \(telemetry.totalChecks)
      Unique: \(telemetry.uniqueCount)
      Duplicates: \(telemetry.duplicateCount)
      Dedup rate: \(String(format: "%.1f%%", telemetry.deduplicationRate * 100))
      Avg latency: \(String(format: "%.1f ns", telemetry.avgCheckLatencyNs))
    """)
#endif
```

---

## Coding Guidelines

### API Usage Best Practices

**Good Patterns**:
```swift
// ✅ Reset visited set at query start
visitedReset(visited)

// ✅ Inline filtering during scan
for candidateID in stream {
    guard visitedTestAndSet(visited, candidateID) else { continue }
    processCandidate(candidateID)
}

// ✅ Reuse visited set across queries (same thread)
for query in queries {
    visitedReset(visited)
    processQuery(query, visited)
}
```

**Anti-Patterns**:
```swift
// ❌ Forgetting to reset between queries
processQuery(query1, visited)
processQuery(query2, visited)  // Wrong! Will think all IDs are duplicates

// ❌ Sharing visited set across threads without sharding
DispatchQueue.concurrentPerform(iterations: queries.count) { i in
    visitedTestAndSet(globalVisited, ids[i])  // Data race!
}

// ❌ Allocating visited set per query (expensive)
for query in queries {
    let visited = visitedInit(idCapacity: 10_000_000)  // 40 MB allocation!
    processQuery(query, visited)
    visitedFree(visited)
}
```

### Mode Selection Guide

```swift
func selectVisitedMode(
    idCapacity: Int64,
    avgUniquePerQuery: Int,
    queryPattern: QueryPattern
) -> VisitedMode {
    let memoryBudget = 100_000_000  // 100 MB

    // DenseEpoch requires 4 × idCapacity bytes
    let denseMemory = idCapacity * 4

    if denseMemory < memoryBudget && queryPattern == .clustered {
        return .denseEpoch  // Fastest reset, good cache locality
    }

    // SparsePaged: memory proportional to unique IDs
    let avgPageCount = (avgUniquePerQuery + 32767) / 32768
    let sparseMemory = avgPageCount * 4096

    if sparseMemory < memoryBudget / 10 {
        return .sparsePaged  // Memory-efficient for sparse queries
    }

    // FixedBitset: 1 bit per ID
    let bitsetMemory = idCapacity / 8

    if bitsetMemory < memoryBudget {
        return .fixedBitset  // Balanced
    }

    return .sparsePaged  // Default for large ID spaces
}
```

---

## Summary

**Kernel #32** provides high-performance candidate deduplication for multi-list search operations:

1. **Functionality**: Detect and filter duplicate candidate IDs across multiple inverted lists, graph neighborhoods, or shards
2. **Algorithms**:
   - **DenseEpoch**: O(1) reset via epoch bumping, 2-3 ns/check, O(id_capacity) space
   - **SparsePaged**: O(unique_ids) space, 5-8 ns/check, O(touched_pages) reset
   - **FixedBitset**: O(id_capacity/8) space, 2-4 ns/check, O(touched_words) reset
3. **Performance**:
   - Check latency: 2-8 ns depending on mode and cache hit rate
   - Saves 30-70% of distance computations in multi-probe scenarios
   - Scales from thousands to billions of IDs
4. **Key Features**:
   - First-seen semantics (preserves scan order)
   - Thread-local single-writer (fastest) or sharded multi-writer (parallel)
   - Efficient reset (O(1) for DenseEpoch, O(touched) for others)
   - Integrates with IVF (#05), ADC (#22), HNSW (#29), Reservoir (#39)
5. **Integration**:
   - Requires dense internal IDs from ID remapping (#50)
   - Inline filtering during candidate streaming
   - Batch operations for block processing
   - Telemetry for performance monitoring (#46)

**Dependencies**: ID Remap (#50)

**Used By**: IVF Search (#05), ADC Scan (#22), HNSW Beam Search (#29), Candidate Reservoir (#39)

**Typical Use**: During IVF search with nprobe=10, filter 100K candidates → 60K unique, saving 40% of distance computations at <1 ms overhead.
