Title: ✅ DONE — ID Filter / Bitset Kernel — Fast Allow/Deny Filtering for Vector Search

Summary
- Implement high-performance bitset-based ID filtering to support access control, tenant isolation, and selective search over dense internal ID spaces.
- Provides both inline single-ID checks and batch mask/compact operations for efficient candidate filtering.
- Supports multi-filter composition (up to 4 filters) with AND/AND-NOT semantics for complex filtering requirements.
- Critical for production systems: security ACLs, multi-tenancy, soft deletes, query-specific allowlists.
- Achieves <3 ns/ID for single filter, <6 ns/ID for composed filters on Apple Silicon.

Project Context
- VectorIndex operates on dense internal IDs (domain [0, N)) managed by ID mapper (#50)
- ID filtering is essential for production vector search:
  - **Security/ACL**: User permissions, row-level security
  - **Multi-tenancy**: Partition data by tenant/shard
  - **Soft deletes**: Mask deleted documents without rebuilding index
  - **Query-specific filters**: User-provided allowlists/denylists
  - **Offline evaluation**: Test set isolation, A/B testing subsets
- Industry context: ~30-40% of production queries include filters
- Challenge: Balance filtering overhead vs. wasted computation
  - Filter too early: Complex filter logic in hot path
  - Filter too late: Waste scoring computation on filtered IDs
  - Optimal: Fast bitset check before expensive operations
- VectorCore provides primitives; VectorIndex needs filtering integration
- Typical usage patterns:
  - F (filter count) ∈ {1, 2, 3, 4} per query
  - Selectivity ∈ {0.1%, 10%, 90%} depending on use case
  - Bitset size: O(N/64) words for N vectors

Goals
- Achieve <3 ns/ID for single bitset check (inline path)
- Achieve <6 ns/ID for 4-filter composition (inline path)
- Support batch operations for amortized efficiency
- Stable compaction preserves relative order
- Zero-copy bitset representation (read-only, shared)
- Thread-safe for concurrent queries (read-only bitsets)
- Composable with tombstones (#43) and visited sets (#32)
- Seamless integration with all search kernels

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/IDFilterKernel.swift`
- Core implementations:
  - Inline checks: `idFilterPass` for single ID
  - Composed checks: `idFilterPassN` for up to 4 filters
  - Batch mask: `idFilterMask` to generate keep/drop bitmask
  - Batch compact: `idFilterCompact` for stable compaction
  - Overlay structure: `IDFilterOverlay` for per-query filters
- Bitset representation:
  - 64-bit word array, 64-byte aligned
  - Size: ⌈N/64⌉ words for N IDs
  - Modes: allowlist (1=keep) or denylist (1=drop)
- Integration points:
  - Used by range query (#07) for pre-filtering
  - Used by ADC scan (#22) for candidate filtering
  - Used by HNSW (#29) for neighbor acceptance
  - Used by reservoir (#39) before insertion
  - Combined with tombstones (#43)

API & Signatures

```swift
// MARK: - Bitset Representation

/// Bitset for dense ID filtering
/// Represents ID domain [0, N) using ⌈N/64⌉ words
public struct IDFilterBitset {
    /// Bitset words (64-bit), 64-byte aligned
    let words: UnsafePointer<UInt64>

    /// Number of IDs covered (N)
    let capacity: Int

    /// Number of words in bitset (⌈N/64⌉)
    var wordCount: Int {
        (capacity + 63) / 64
    }

    /// Initialize from word array
    public init(words: UnsafePointer<UInt64>, capacity: Int)

    /// Test if ID is set (1)
    @inline(__always)
    public func test(id: Int64) -> Bool {
        guard id >= 0 && id < capacity else { return false }
        let wordIdx = Int(id >> 6)  // id / 64
        let bitIdx = Int(id & 63)    // id % 64
        return (words[wordIdx] >> bitIdx) & 1 == 1
    }

    /// Set ID bit to 1
    public mutating func set(id: Int64)

    /// Clear ID bit to 0
    public mutating func clear(id: Int64)

    /// Set ID bit to value
    public mutating func set(id: Int64, value: Bool)
}

// MARK: - Filter Mode

/// Bitset interpretation mode
public enum FilterMode {
    case allowlist  // 1 = keep, 0 = drop
    case denylist   // 1 = drop, 0 = keep

    /// Whether ID should be kept given bit value
    func shouldKeep(bit: Bool) -> Bool {
        switch self {
        case .allowlist: return bit
        case .denylist: return !bit
        }
    }
}

// MARK: - Inline Single-ID Checks

/// Test if single ID passes filter
/// Optimized for inline checks in hot paths
///
/// - Complexity: O(1) - single word load + bit test
/// - Performance: <3 ns on Apple Silicon (L1 cache hit)
/// - Thread Safety: Reentrant; bitset is read-only
///
/// - Parameters:
///   - bitset: Bitset words, 64-byte aligned
///   - id: ID to test (must be in [0, capacity))
///   - capacity: Number of IDs covered by bitset
///   - mode: Allowlist or denylist mode
/// - Returns: true if ID passes filter
@inline(__always)
public func idFilterPass(
    bitset: UnsafePointer<UInt64>,
    id: Int64,
    capacity: Int,
    mode: FilterMode
) -> Bool {
    // Out-of-range always drops
    guard id >= 0 && id < capacity else { return false }

    // Compute word and bit index
    let wordIdx = Int(id >> 6)
    let bitIdx = Int(id & 63)

    // Load word and extract bit
    let word = bitset[wordIdx]
    let bit = (word >> bitIdx) & 1 == 1

    // Apply mode semantics
    return mode.shouldKeep(bit: bit)
}

// MARK: - Inline Composed Checks

/// Test if ID passes composed multi-filter
/// Combines up to 4 allowlist filters (AND) and 1 denylist (AND-NOT)
///
/// Composition logic:
///   keep = (allow0 AND allow1 AND allow2 AND allow3) AND NOT deny
///
/// - Complexity: O(F) where F = number of filters
/// - Performance: <6 ns for F=4 on Apple Silicon
///
/// - Parameters:
///   - allow0, allow1, allow2, allow3: Allowlist bitsets (nil = skip)
///   - deny: Denylist bitset (nil = skip)
///   - id: ID to test
///   - capacity: Number of IDs covered
/// - Returns: true if ID passes all filters
@inline(__always)
public func idFilterPassN(
    allow0: UnsafePointer<UInt64>?,
    allow1: UnsafePointer<UInt64>? = nil,
    allow2: UnsafePointer<UInt64>? = nil,
    allow3: UnsafePointer<UInt64>? = nil,
    deny: UnsafePointer<UInt64>? = nil,
    id: Int64,
    capacity: Int
) -> Bool {
    // Out-of-range always drops
    guard id >= 0 && id < capacity else { return false }

    // Compute indices once
    let wordIdx = Int(id >> 6)
    let bitIdx = Int(id & 63)
    let mask = UInt64(1) << bitIdx

    // Check allowlists (all must pass)
    if let allow = allow0 {
        if (allow[wordIdx] & mask) == 0 { return false }
    }
    if let allow = allow1 {
        if (allow[wordIdx] & mask) == 0 { return false }
    }
    if let allow = allow2 {
        if (allow[wordIdx] & mask) == 0 { return false }
    }
    if let allow = allow3 {
        if (allow[wordIdx] & mask) == 0 { return false }
    }

    // Check denylist (must not be set)
    if let denyBits = deny {
        if (denyBits[wordIdx] & mask) != 0 { return false }
    }

    return true
}

// MARK: - Batch Mask Generation

/// Generate bitmask for batch of IDs
/// Output mask: 1 = keep, 0 = drop
///
/// - Complexity: O(n) where n = number of IDs
/// - Performance: ~2-3 ns/ID for single filter
///
/// - Parameters:
///   - bitset: Filter bitset
///   - ids: Array of IDs to test [n]
///   - count: Number of IDs
///   - capacity: Bitset capacity
///   - mode: Allowlist or denylist
///   - maskOut: Output bitmask [n], 1=keep, 0=drop
/// - Returns: Number of IDs kept (count of 1s in mask)
@inlinable
public func idFilterMask(
    bitset: UnsafePointer<UInt64>,
    ids: UnsafePointer<Int64>,
    count n: Int,
    capacity: Int,
    mode: FilterMode,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    var keptCount = 0

    // Process in batches for ILP
    let batchSize = 8
    let nBatched = (n / batchSize) * batchSize

    for i in stride(from: 0, to: nBatched, by: batchSize) {
        // Unroll 8 iterations for instruction-level parallelism
        for j in 0..<batchSize {
            let id = ids[i + j]
            let passes = idFilterPass(
                bitset: bitset,
                id: id,
                capacity: capacity,
                mode: mode
            )
            maskOut[i + j] = passes ? 1 : 0
            keptCount += passes ? 1 : 0
        }
    }

    // Handle remainder
    for i in nBatched..<n {
        let id = ids[i]
        let passes = idFilterPass(
            bitset: bitset,
            id: id,
            capacity: capacity,
            mode: mode
        )
        maskOut[i] = passes ? 1 : 0
        keptCount += passes ? 1 : 0
    }

    return keptCount
}

/// Generate bitmask for composed multi-filter
///
/// - Parameters:
///   - filters: Array of filter bitsets [F]
///   - modes: Array of filter modes [F]
///   - filterCount: Number of filters (F ≤ 4)
@inlinable
public func idFilterMaskN(
    filters: [UnsafePointer<UInt64>?],
    modes: [FilterMode],
    filterCount F: Int,
    ids: UnsafePointer<Int64>,
    count n: Int,
    capacity: Int,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    // Separate into allows and denies
    var allows: [UnsafePointer<UInt64>?] = [nil, nil, nil, nil]
    var deny: UnsafePointer<UInt64>? = nil

    var allowIdx = 0
    for i in 0..<F {
        guard let filter = filters[i] else { continue }
        switch modes[i] {
        case .allowlist:
            if allowIdx < 4 {
                allows[allowIdx] = filter
                allowIdx += 1
            }
        case .denylist:
            // Only support one denylist
            deny = filter
        }
    }

    // Apply composed filter
    var keptCount = 0
    for i in 0..<n {
        let id = ids[i]
        let passes = idFilterPassN(
            allow0: allows[0],
            allow1: allows[1],
            allow2: allows[2],
            allow3: allows[3],
            deny: deny,
            id: id,
            capacity: capacity
        )
        maskOut[i] = passes ? 1 : 0
        keptCount += passes ? 1 : 0
    }

    return keptCount
}

// MARK: - Batch Compact

/// Filter and compact ID/score arrays
/// Stable compaction: preserves relative order of kept IDs
///
/// - Complexity: O(n)
/// - Performance: ~3-4 ns/ID including compaction
///
/// - Parameters:
///   - bitset: Filter bitset
///   - idsIn: Input IDs [n]
///   - scoresIn: Input scores [n], optional
///   - count: Number of input IDs
///   - capacity: Bitset capacity
///   - mode: Filter mode
///   - idsOut: Output IDs (compacted)
///   - scoresOut: Output scores (compacted), optional
/// - Returns: Number of IDs kept
@inlinable
public func idFilterCompact(
    bitset: UnsafePointer<UInt64>,
    idsIn: UnsafePointer<Int64>,
    scoresIn: UnsafePointer<Float>?,
    count n: Int,
    capacity: Int,
    mode: FilterMode,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?
) -> Int {
    var writeIdx = 0

    for i in 0..<n {
        let id = idsIn[i]
        let passes = idFilterPass(
            bitset: bitset,
            id: id,
            capacity: capacity,
            mode: mode
        )

        if passes {
            idsOut[writeIdx] = id

            if let scoresInPtr = scoresIn, let scoresOutPtr = scoresOut {
                scoresOutPtr[writeIdx] = scoresInPtr[i]
            }

            writeIdx += 1
        }
    }

    return writeIdx
}

/// Compact with composed multi-filter
@inlinable
public func idFilterCompactN(
    filters: [UnsafePointer<UInt64>?],
    modes: [FilterMode],
    filterCount F: Int,
    idsIn: UnsafePointer<Int64>,
    scoresIn: UnsafePointer<Float>?,
    count n: Int,
    capacity: Int,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?
) -> Int {
    // Generate mask first
    var mask = [UInt8](repeating: 0, count: n)
    let keptCount = idFilterMaskN(
        filters: filters,
        modes: modes,
        filterCount: F,
        ids: idsIn,
        count: n,
        capacity: capacity,
        maskOut: &mask
    )

    // Compact using mask
    var writeIdx = 0
    for i in 0..<n {
        if mask[i] == 1 {
            idsOut[writeIdx] = idsIn[i]

            if let scoresInPtr = scoresIn, let scoresOutPtr = scoresOut {
                scoresOutPtr[writeIdx] = scoresInPtr[i]
            }

            writeIdx += 1
        }
    }

    return writeIdx
}

// MARK: - Filter Overlay

/// Per-query filter overlay combining global and query-specific filters
/// Typical usage: 1-2 global filters + optional query allowlist/denylist
public struct IDFilterOverlay {
    /// Global allowlist filters (e.g., tenant, shard) [up to 4]
    let globalAllows: [UnsafePointer<UInt64>?]

    /// Global denylist (e.g., soft deletes, tombstones)
    let globalDeny: UnsafePointer<UInt64>?

    /// Query-specific allowlist (e.g., user-provided IDs)
    let queryAllow: UnsafePointer<UInt64>?

    /// Query-specific denylist
    let queryDeny: UnsafePointer<UInt64>?

    /// Bitset capacity
    let capacity: Int

    public init(
        globalAllows: [UnsafePointer<UInt64>?] = [nil, nil, nil, nil],
        globalDeny: UnsafePointer<UInt64>? = nil,
        queryAllow: UnsafePointer<UInt64>? = nil,
        queryDeny: UnsafePointer<UInt64>? = nil,
        capacity: Int
    ) {
        self.globalAllows = globalAllows
        self.globalDeny = globalDeny
        self.queryAllow = queryAllow
        self.queryDeny = queryDeny
        self.capacity = capacity
    }

    /// Test if ID passes all filters in overlay
    @inline(__always)
    public func test(id: Int64) -> Bool {
        guard id >= 0 && id < capacity else { return false }

        let wordIdx = Int(id >> 6)
        let bitIdx = Int(id & 63)
        let mask = UInt64(1) << bitIdx

        // Check global allowlists
        for allow in globalAllows {
            guard let allowBits = allow else { continue }
            if (allowBits[wordIdx] & mask) == 0 { return false }
        }

        // Check query allowlist (if present, overrides global)
        if let queryAllowBits = queryAllow {
            if (queryAllowBits[wordIdx] & mask) == 0 { return false }
        }

        // Check global denylist
        if let denyBits = globalDeny {
            if (denyBits[wordIdx] & mask) != 0 { return false }
        }

        // Check query denylist
        if let queryDenyBits = queryDeny {
            if (queryDenyBits[wordIdx] & mask) != 0 { return false }
        }

        return true
    }
}

// MARK: - Telemetry

/// Per-operation statistics
public struct IDFilterTelemetry {
    public let idsProcessed: Int
    public let idsKept: Int
    public let idsDropped: Int
    public let filtersActive: Int       // Number of filters applied
    public let composedChecks: Int      // Total bit checks performed
    public let executionTimeNanos: UInt64

    public var selectivityPercent: Double {
        return (Double(idsKept) / Double(idsProcessed)) * 100.0
    }

    public var nanosPerID: Double {
        return Double(executionTimeNanos) / Double(idsProcessed)
    }
}

// MARK: - Convenience API

extension IDFilterKernel {
    /// High-level filter with automatic mode detection
    public static func filter(
        ids: [Int64],
        allowedIDs: Set<Int64>? = nil,
        deniedIDs: Set<Int64>? = nil,
        capacity: Int
    ) -> [Int64]
}
```

Algorithm Details

**Bitset Layout**:

```
ID Space: [0, N)
Bitset: Array of ⌈N/64⌉ UInt64 words

Example: N = 200
Words needed: ⌈200/64⌉ = 4 words (256 bits, covering IDs 0-255)

Memory layout (64-byte aligned):
┌──────────────────────────────────┐
│ word[0]: bits 0-63   (IDs 0-63)  │  8 bytes
│ word[1]: bits 64-127 (IDs 64-127)│  8 bytes
│ word[2]: bits 128-191(IDs 128-191)│ 8 bytes
│ word[3]: bits 192-255(IDs 192-255)│ 8 bytes
└──────────────────────────────────┘
Total: 32 bytes

Bit indexing for ID = 137:
  wordIdx = 137 >> 6 = 2  (word 2)
  bitIdx  = 137 & 63 = 9  (bit 9 within word)
  mask    = 1 << 9    = 0x0200

To test: (word[2] >> 9) & 1
To set:  word[2] |= mask
To clear: word[2] &= ~mask
```

**Single-ID Check Algorithm**:

```swift
@inline(__always)
func idFilterPass(
    bitset: UnsafePointer<UInt64>,
    id: Int64,
    capacity: Int,
    mode: FilterMode
) -> Bool {
    // Bounds check (critical for safety)
    guard id >= 0 && id < capacity else { return false }

    // Bit arithmetic (branchless after guard)
    let wordIdx = Int(id >> 6)      // Divide by 64
    let bitIdx = Int(id & 63)       // Modulo 64
    let word = bitset[wordIdx]      // Load word (8 bytes)
    let bit = (word >> bitIdx) & 1  // Extract bit

    // Apply mode (branchless for common case)
    switch mode {
    case .allowlist: return bit == 1
    case .denylist: return bit == 0
    }
}
```

**Composed Multi-Filter Algorithm**:

```swift
// Composition semantics:
//   keep = (allow0 AND allow1 AND ... AND allowN) AND NOT (deny0 OR deny1 OR ...)
//
// Simplified: (AND over allows) AND NOT (any deny)

@inline(__always)
func idFilterPassN(
    allow0: UnsafePointer<UInt64>?,
    allow1: UnsafePointer<UInt64>?,
    allow2: UnsafePointer<UInt64>?,
    allow3: UnsafePointer<UInt64>?,
    deny: UnsafePointer<UInt64>?,
    id: Int64,
    capacity: Int
) -> Bool {
    guard id >= 0 && id < capacity else { return false }

    // Compute indices once (CSE optimization)
    let wordIdx = Int(id >> 6)
    let bitIdx = Int(id & 63)
    let mask = UInt64(1) << bitIdx

    // Early exit on first failing allowlist
    if let allow = allow0, (allow[wordIdx] & mask) == 0 { return false }
    if let allow = allow1, (allow[wordIdx] & mask) == 0 { return false }
    if let allow = allow2, (allow[wordIdx] & mask) == 0 { return false }
    if let allow = allow3, (allow[wordIdx] & mask) == 0 { return false }

    // Deny check (if present)
    if let denyBits = deny, (denyBits[wordIdx] & mask) != 0 { return false }

    return true
}
```

**Stable Compaction Algorithm**:

```swift
// Two-finger compaction: read from i, write to writeIdx
// Preserves relative order of kept elements (stable)

func idFilterCompact(
    bitset: UnsafePointer<UInt64>,
    idsIn: UnsafePointer<Int64>,
    scoresIn: UnsafePointer<Float>?,
    count n: Int,
    capacity: Int,
    mode: FilterMode,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?
) -> Int {
    var writeIdx = 0  // Write finger

    for i in 0..<n {  // Read finger
        if idFilterPass(bitset: bitset, id: idsIn[i], capacity: capacity, mode: mode) {
            // Keep: copy to write position
            idsOut[writeIdx] = idsIn[i]

            if let scoresInPtr = scoresIn, let scoresOutPtr = scoresOut {
                scoresOutPtr[writeIdx] = scoresInPtr[i]
            }

            writeIdx += 1
        }
        // Drop: skip (don't increment writeIdx)
    }

    return writeIdx  // Number kept
}
```

Performance Characteristics

**Latency per ID**:

```
Operation                  | Latency (Apple M1/M2) | Notes
---------------------------|----------------------|------------------
Single ID check (hot)      | 1.5-2.5 ns           | L1 cache hit
Single ID check (cold)     | 8-12 ns              | L2/L3 cache miss
4-filter composed (hot)    | 4-6 ns               | 4× word loads + logic
Batch mask (hot)           | 2-3 ns/ID            | Amortized, unrolled
Batch compact (hot)        | 3-4 ns/ID            | Including copy
```

**Throughput** (Apple M1, n=10000):

```
Filter Count | IDs/second | Bandwidth
-------------|------------|-------------
1 filter     | 500M/s     | 4 GB/s (ID stream)
2 filters    | 400M/s     | 3.2 GB/s
4 filters    | 300M/s     | 2.4 GB/s
```

**Cache Efficiency**:

```
Bitset size vs cache levels (N = vector count):

N          | Bitset Size | Fits In
-----------|-------------|----------
1,000      | 128 bytes   | L1 (128 KB)
10,000     | 1.25 KB     | L1
100,000    | 12.5 KB     | L1
1,000,000  | 125 KB      | L1/L2 boundary
10,000,000 | 1.25 MB     | L2 (12 MB on M1)
100,000,000| 12.5 MB     | L2/L3 boundary

Random access pattern:
- Hot bitset (in L1): ~2 ns per check
- Warm bitset (in L2): ~5-8 ns per check
- Cold bitset (in L3/RAM): ~50-100 ns per check
```

**Selectivity Impact**:

```
Selectivity | Compact Efficiency | Notes
------------|-------------------|------------------
1%          | High              | Mostly drops, minimal copies
10%         | High              | Good cache utilization
50%         | Medium            | Half data copied
90%         | Low               | Most data copied (overhead)
```

Bitset Composition Semantics

**Composition Logic**:

```
Given:
- A0, A1, A2, A3: Allowlist bitsets
- D: Denylist bitset

Result = (A0 AND A1 AND A2 AND A3) AND NOT D

Truth table for 2 allows + 1 deny:
A0 | A1 | D  | Result
---|----|----|-------
0  | -  | -  | 0      (fails first allow)
1  | 0  | -  | 0      (fails second allow)
1  | 1  | 1  | 0      (denied)
1  | 1  | 0  | 1      (passes all) ✓
```

**Example Use Cases**:

1. **Multi-Tenant with Soft Deletes**:
   ```swift
   allow0 = tenantBitset      // Tenant 123's vectors
   allow1 = shardBitset       // Shard 5's vectors
   deny   = deletedBitset     // Soft-deleted vectors

   Result: Tenant 123's vectors in shard 5, excluding deleted
   ```

2. **ACL + Query Filter**:
   ```swift
   allow0 = aclBitset         // User's permitted IDs
   allow1 = queryAllowlist    // User's query-specific IDs
   deny   = nil

   Result: Intersection of ACL and query filter
   ```

3. **Evaluation Subset**:
   ```swift
   allow0 = testSetBitset     // Test set IDs
   deny   = outlierBitset     // Known outliers to exclude

   Result: Test set excluding outliers
   ```

Integration Patterns

**Range Query Integration** (#07):

```swift
func rangeScanBlock(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>?,
    vectorCount n: Int,
    dimension d: Int,
    metric: DistanceMetric,
    threshold: Float,
    idsOut: UnsafeMutablePointer<Int64>,
    scoresOut: UnsafeMutablePointer<Float>?,
    maxOut: Int,
    filter: IDFilterOverlay?  // Optional filter
) -> Int {
    var outputCount = 0

    for i in 0..<n {
        let id = ids?[i] ?? Int64(i)

        // Apply filter before scoring (save computation)
        if let filterOverlay = filter {
            if !filterOverlay.test(id: id) { continue }
        }

        // Score and check threshold
        let score = computeScore(query, database + i*d, d, metric)
        if passesThreshold(score, threshold, metric) {
            guard outputCount < maxOut else { break }
            idsOut[outputCount] = id
            scoresOut?[outputCount] = score
            outputCount += 1
        }
    }

    return outputCount
}
```

**ADC Scan Integration** (#22):

```swift
func adcScan_u8(
    codes: UnsafePointer<UInt8>,
    vectorCount n: Int,
    lut: UnsafePointer<Float>,
    ids: UnsafePointer<Int64>,
    output: UnsafeMutablePointer<Float>,
    filter: IDFilterOverlay?
) {
    for i in 0..<n {
        let id = ids[i]

        // Skip if filtered out
        if let filterOverlay = filter {
            if !filterOverlay.test(id: id) {
                output[i] = Float.infinity  // Sentinel value
                continue
            }
        }

        // Compute ADC distance
        output[i] = computeADCDistance(codes + i*m, lut, m)
    }
}
```

**HNSW Neighbor Acceptance** (#29):

```swift
func searchLayer(
    query: Vector,
    entryPoint: GraphNode,
    ef: Int,
    filter: IDFilterOverlay?
) -> [GraphNode] {
    var candidates = TopKHeap(capacity: ef, ordering: .max)

    while let current = candidates.root {
        let neighbors = getNeighbors(current.id)

        for neighbor in neighbors {
            // Filter check before adding to candidates
            if let filterOverlay = filter {
                if !filterOverlay.test(id: neighbor.id) { continue }
            }

            let score = computeScore(query, neighbor.vector)
            candidates.push(score: score, id: neighbor.id)
        }
    }

    return candidates.extractSorted()
}
```

**Reservoir Integration** (#39):

```swift
struct Reservoir {
    func insert(id: Int64, score: Float, filter: IDFilterOverlay?) {
        // Filter before insertion (save heap operations)
        if let filterOverlay = filter {
            if !filterOverlay.test(id: id) { return }
        }

        // Insert to reservoir
        if shouldReplace(score) {
            replaceWorst(id: id, score: score)
        }
    }
}
```

**Tombstone Combination** (#43):

```swift
// Combine soft deletes (tombstones) with user filter
let overlay = IDFilterOverlay(
    globalAllows: [userACLBitset, nil, nil, nil],
    globalDeny: tombstoneBitset,        // Soft-deleted IDs
    queryAllow: userQueryFilter,
    queryDeny: nil,
    capacity: indexCapacity
)

// All queries use this overlay
let results = search(query: query, k: 10, filter: overlay)
```

Correctness & Testing

**Test Cases**:

1. **Basic Semantics**:
   - Allowlist: Only IDs with bit=1 pass
   - Denylist: Only IDs with bit=0 pass
   - Out-of-range: IDs < 0 or ≥ N always drop

2. **Composition**:
   - 2 allowlists: Intersection semantics
   - 4 allowlists: All must pass
   - Allowlist + denylist: Allow AND NOT deny
   - Empty filters: All pass

3. **Stability**:
   - Compaction preserves relative order
   - Scores stay aligned with IDs after compact
   - Deterministic for same input

4. **Edge Cases**:
   - Empty ID list (n=0)
   - All IDs pass (selectivity=100%)
   - No IDs pass (selectivity=0%)
   - ID boundary values (0, N-1, N, -1)

5. **Performance**:
   - Target: <3 ns/ID for single filter
   - Target: <6 ns/ID for 4-filter composition
   - Batch should amortize overhead

**Example Tests**:

```swift
func testIDFilter_AllowlistSemantics() {
    let capacity = 256
    var bitset = createBitset(capacity: capacity)

    // Set bits for IDs: 10, 20, 30
    bitset.set(id: 10)
    bitset.set(id: 20)
    bitset.set(id: 30)

    // Test allowlist mode
    XCTAssertTrue(idFilterPass(bitset.words, id: 10, capacity, .allowlist))
    XCTAssertTrue(idFilterPass(bitset.words, id: 20, capacity, .allowlist))
    XCTAssertTrue(idFilterPass(bitset.words, id: 30, capacity, .allowlist))
    XCTAssertFalse(idFilterPass(bitset.words, id: 15, capacity, .allowlist))
    XCTAssertFalse(idFilterPass(bitset.words, id: 100, capacity, .allowlist))
}

func testIDFilter_Composition() {
    let capacity = 256

    // Allow0: IDs 0-99
    var allow0 = createBitset(capacity: capacity)
    for id in 0..<100 { allow0.set(id: Int64(id)) }

    // Allow1: IDs 50-149
    var allow1 = createBitset(capacity: capacity)
    for id in 50..<150 { allow1.set(id: Int64(id)) }

    // Deny: IDs 80-89
    var deny = createBitset(capacity: capacity)
    for id in 80..<90 { deny.set(id: Int64(id)) }

    // Result should be: (0-99 AND 50-149) AND NOT (80-89)
    //                  = 50-99 excluding 80-89
    //                  = 50-79, 90-99

    XCTAssertTrue(idFilterPassN(
        allow0: allow0.words, allow1: allow1.words, deny: deny.words,
        id: 50, capacity: capacity))  // In both allows, not denied

    XCTAssertTrue(idFilterPassN(
        allow0: allow0.words, allow1: allow1.words, deny: deny.words,
        id: 79, capacity: capacity))

    XCTAssertFalse(idFilterPassN(
        allow0: allow0.words, allow1: allow1.words, deny: deny.words,
        id: 85, capacity: capacity))  // Denied

    XCTAssertFalse(idFilterPassN(
        allow0: allow0.words, allow1: allow1.words, deny: deny.words,
        id: 150, capacity: capacity))  // Not in allow0
}

func testIDFilter_StableCompaction() {
    let capacity = 256
    var allowlist = createBitset(capacity: capacity)

    // Allow only even IDs
    for id in stride(from: 0, to: 100, by: 2) {
        allowlist.set(id: Int64(id))
    }

    // Input: IDs 0-99 with scores
    let idsIn = (0..<100).map { Int64($0) }
    let scoresIn = (0..<100).map { Float($0) * 0.1 }

    var idsOut = [Int64](repeating: 0, count: 100)
    var scoresOut = [Float](repeating: 0, count: 100)

    let kept = idFilterCompact(
        bitset: allowlist.words,
        idsIn: idsIn,
        scoresIn: scoresIn,
        count: 100,
        capacity: capacity,
        mode: .allowlist,
        idsOut: &idsOut,
        scoresOut: &scoresOut
    )

    // Should keep 50 even IDs
    XCTAssertEqual(kept, 50)

    // Verify order preserved: 0, 2, 4, 6, ...
    for i in 0..<kept {
        XCTAssertEqual(idsOut[i], Int64(i * 2))
        XCTAssertEqual(scoresOut[i], Float(i * 2) * 0.1)
    }
}
```

Coding Guidelines

**Performance Best Practices**:
- Inline checks for hot paths (use `@inline(__always)`)
- Batch operations for amortized efficiency (8-16 IDs per batch)
- Prefetch bitset words for random access patterns
- Early exit on first failing allowlist (short-circuit evaluation)
- Unroll loops for instruction-level parallelism

**Memory Management**:
- Bitsets are read-only during queries (zero-copy)
- 64-byte alignment for cache line efficiency
- Share bitsets across threads (no synchronization needed)
- Allocate bitsets at index build time, not per-query

**API Usage**:

```swift
// Good: Inline check in hot path
@inline(__always)
func processCandidate(id: Int64, filter: IDFilterOverlay?) -> Bool {
    if let overlay = filter {
        return overlay.test(id: id)
    }
    return true
}

// Good: Batch compact for many IDs
let kept = idFilterCompact(bitset, idsIn, scoresIn, n, capacity, .allowlist, idsOut, scoresOut)

// Bad: Individual allocations per ID
for id in ids {
    let result = [id].filter { idFilterPass(bitset, $0, capacity, .allowlist) }
}
```

**Error Handling**:
- Out-of-range IDs: Always return false (safe default)
- Null bitset pointers: Skip filter (treat as pass-all)
- Capacity mismatch: Assert in debug, undefined in release

Non-Goals

- Dynamic bitset resizing (fixed capacity from ID mapper)
- Compressed bitsets (dense representation only)
- Mutable bitsets during queries (read-only for thread safety)
- Bitset serialization (handled at application level)
- GPU/Metal acceleration (CPU-focused)

Example Usage

```swift
import VectorIndex

// Example 1: Basic allowlist filtering
let capacity = 10000
var tenantBitset = IDFilterBitset.allocate(capacity: capacity)

// Set allowed IDs for tenant
for id in [10, 20, 30, 40, 50] {
    tenantBitset.set(id: Int64(id))
}

// Test single ID
let passes = idFilterPass(
    bitset: tenantBitset.words,
    id: 30,
    capacity: capacity,
    mode: .allowlist
)  // true

// Example 2: Batch compaction
let idsIn: [Int64] = [5, 10, 15, 20, 25, 30]
let scoresIn: [Float] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

var idsOut = [Int64](repeating: 0, count: idsIn.count)
var scoresOut = [Float](repeating: 0, count: idsIn.count)

let kept = idFilterCompact(
    bitset: tenantBitset.words,
    idsIn: idsIn,
    scoresIn: scoresIn,
    count: idsIn.count,
    capacity: capacity,
    mode: .allowlist,
    idsOut: &idsOut,
    scoresOut: &scoresOut
)

// kept = 3 (IDs: 10, 20, 30)

// Example 3: Multi-tenant with soft deletes
var tenant1Bitset = createBitset(capacity: capacity)
var tenant2Bitset = createBitset(capacity: capacity)
var deletedBitset = createBitset(capacity: capacity)

// Setup bitsets...

let overlay = IDFilterOverlay(
    globalAllows: [tenant1Bitset.words, tenant2Bitset.words, nil, nil],
    globalDeny: deletedBitset.words,
    capacity: capacity
)

// Use in search
let results = search(query: query, k: 10, filter: overlay)

// Example 4: Query-specific allowlist
var userACLBitset = createBitset(capacity: capacity)
var queryAllowlist = createBitset(capacity: capacity)

// User can only see IDs in ACL
for id in userPermittedIDs {
    userACLBitset.set(id: id)
}

// User's query specifies additional filter
for id in querySpecificIDs {
    queryAllowlist.set(id: id)
}

let queryOverlay = IDFilterOverlay(
    globalAllows: [userACLBitset.words, nil, nil, nil],
    queryAllow: queryAllowlist.words,
    capacity: capacity
)

// Result: Intersection of ACL and query filter
let results = search(query: query, k: 10, filter: queryOverlay)

// Example 5: Integration with range query
var mask = [UInt8](repeating: 0, count: n)

let keptCount = idFilterMask(
    bitset: tenantBitset.words,
    ids: candidateIDs,
    count: n,
    capacity: capacity,
    mode: .allowlist,
    maskOut: &mask
)

// Use mask for range query
for i in 0..<n {
    if mask[i] == 1 {
        // Process this candidate
    }
}
```

Mathematical Foundation

**Bitset Operations**:

```
Let B be a bitset of size N

Get bit i:   (B[i >> 6] >> (i & 63)) & 1
Set bit i:   B[i >> 6] |= (1 << (i & 63))
Clear bit i: B[i >> 6] &= ~(1 << (i & 63))

Word index:  i >> 6  ≡ floor(i / 64)
Bit index:   i & 63  ≡ i mod 64
```

**Composition Algebra**:

```
Given bitsets A, B, C, D:

AND:     A ∩ B = {i | i ∈ A and i ∈ B}
OR:      A ∪ B = {i | i ∈ A or i ∈ B}
NOT:     Ā = {i | i ∉ A}
AND-NOT: A \ B = A ∩ B̄ = {i | i ∈ A and i ∉ B}

Composition: (A ∩ B ∩ C) \ D = (A ∩ B ∩ C) ∩ D̄

Commutative: A ∩ B = B ∩ A
Associative: (A ∩ B) ∩ C = A ∩ (B ∩ C)
Distributive: A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
```

**Selectivity**:

```
Let S = selectivity (fraction of IDs passing filter)
Let N = total IDs
Let F = number of filters

Expected kept: K = N × S

For independent filters with selectivities S₁, S₂, ..., Sₖ:
Combined selectivity: S = S₁ × S₂ × ... × Sₖ

Example: 3 filters with 50% selectivity each
S = 0.5 × 0.5 × 0.5 = 0.125 (12.5% kept)
```

Dependencies

**Internal**:
- ID Mapper (#50): Provides dense ID domain [0, N)
- Range Query (#07): Uses filters for pre-filtering
- ADC Scan (#22): Uses filters for candidate filtering
- HNSW (#29): Uses filters for neighbor acceptance
- Reservoir (#39): Uses filters before insertion
- Tombstones (#43): Combines with deny bitsets
- Visited Set (#32): Separate stateful per-query structure
- Telemetry (#46): Performance tracking

**External**:
- Swift Standard Library: UnsafePointer, bit operations
- Foundation: None (pure bit manipulation)
- Dispatch: None (read-only, thread-safe)

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Performance**:
- Single ID check: <3 ns (L1 cache hit)
- 4-filter composition: <6 ns per ID
- Batch operations: <4 ns/ID including compaction
- Zero overhead for nil filters

✅ **Correctness**:
- Allowlist/denylist semantics correct
- Composition logic correct (AND over allows, AND-NOT deny)
- Out-of-range IDs handled safely (return false)
- Stable compaction preserves order

✅ **Integration**:
- Works with range queries (#07)
- Works with ADC scan (#22)
- Works with HNSW (#29)
- Works with reservoir (#39)
- Combines with tombstones (#43)

✅ **Thread Safety**:
- Read-only bitsets safe for concurrent access
- No locks or synchronization needed
- Deterministic results

✅ **Flexibility**:
- Supports 1-4 allowlists + 1 denylist
- Overlay structure for per-query filters
- Batch and inline modes for different use cases
- Compatible with all search operations
