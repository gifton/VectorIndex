# Kernel Specification #50: ID Remapping (External ↔ Internal Dense Handles)

**ID**: 50
**Priority**: MUST
**Role**: B/M (Build/Maintenance)
**Status**: Specification

---

## Purpose

Provide high-performance bidirectional mapping between user-facing external IDs (UInt64, opaque) and internal dense IDs (Int64, range `[0, N_total)`). Enables compact kernel data structures (bitsets, arrays), contiguous storage layouts, and efficient operations across the entire vector database infrastructure.

**Key Benefits**:
1. **Compact Storage**: Dense IDs enable O(N) arrays vs O(capacity) sparse maps
2. **Fast Operations**: Bitset-based deduplication (#32), range checks, tombstone tracking (#43)
3. **User Flexibility**: External IDs can be arbitrary 64-bit values (UUIDs, hashes, database keys)
4. **Memory Efficiency**: Single dense array + hash table vs sparse external-keyed structures

**Typical Use Case**: User inserts vectors with external IDs `[uuid1, uuid2, ...]` → system assigns internal IDs `[0, 1, 2, ...]` → queries use dense IDs internally for O(1) bitset operations → results map back to external IDs for user consumption.

---

## Mathematical Foundations

### 1. Bijection Problem

**Definition**: Maintain a bijection **f**: External → Internal where:
- **External ID space**: **E** = {0, 1, ..., 2⁶⁴-1} (UInt64, sparse)
- **Internal ID space**: **I** = {0, 1, ..., N-1} (Int64, dense, N = current vector count)
- **Mapping**: **f**: **E** → **I** ∪ {-1}, where -1 denotes "not found"
- **Inverse**: **f⁻¹**: **I** → **E**, always defined for valid internal IDs

**Properties**:
- **Injective**: ∀ e₁, e₂ ∈ **E**, if f(e₁) = f(e₂) ≠ -1, then e₁ = e₂ (unique external IDs)
- **Surjective**: ∀ i ∈ **I**, ∃ e ∈ **E** such that f(e) = i (all internal IDs assigned)
- **Monotonic allocation**: Internal IDs assigned in increasing order: 0, 1, 2, ...
- **Append-only (P0)**: Internal IDs never reused (until compaction #31)

### 2. Hash Table Theory

**Open Addressing**: Resolve collisions by probing alternative locations in the table.

**Hash Function**: **h**: **E** → [0, M-1] where M = table size (power of 2)
```
h(e) = (e * GOLDEN_RATIO) >> (64 - log₂(M))
```

**Probe Sequence**: For key e with hash h₀ = h(e):
```
Linear: hᵢ = (h₀ + i) mod M
Quadratic: hᵢ = (h₀ + c₁·i + c₂·i²) mod M
Robin Hood: Linear with displacement tracking
```

**Load Factor**: α = N / M (target: α ≤ 0.8 for performance)

**Probe Length**: Average probes for successful lookup:
```
Expected (linear probing, α=0.5): ≈ 1.5
Expected (linear probing, α=0.75): ≈ 8.5
Expected (Robin Hood, α=0.9): ≈ 2.5 (variance reduction)
```

### 3. Swiss Table (Google's Abseil)

**Concept**: Group-based SIMD probing with control bytes.

**Layout**:
```
Control bytes: [g₀, g₁, ..., gₙ₋₁] (1 byte each, 16-byte groups)
  - 0b11111111 = empty
  - 0b11111110 = deleted (tombstone)
  - 0b0xxxxxxx = 7-bit hash fragment (H2)

Key-Value pairs: [(k₀, v₀), (k₁, v₁), ...]
```

**Lookup Algorithm**:
```
1. Compute H1(key) = primary hash → group index
2. Compute H2(key) = 7-bit hash fragment
3. Load control bytes for group (16 bytes, SIMD)
4. Match H2 against all 16 control bytes (parallel)
5. For each match: check key equality
6. If not found: linear probe to next group
```

**SIMD Matching** (ARM NEON):
```c
uint8x16_t ctrl = vld1q_u8(ctrl_bytes);
uint8x16_t h2_vec = vdupq_n_u8(h2);
uint8x16_t match = vceqq_u8(ctrl, h2_vec);
uint16_t mask = vget_lane_u16(vshrn_n_u16(match, 4), 0);
// mask has bits set for matching positions
```

### 4. Robin Hood Hashing

**Concept**: Minimize variance in probe lengths by "stealing from the rich."

**Invariant**: For each entry at position p with ideal position h:
- **Distance** d = (p - h) mod M (probe distance, DIB = Distance to Initial Bucket)
- When inserting, if existing entry has shorter DIB, swap and continue inserting displaced entry
- Result: More equitable probe distributions

**Insertion**:
```
1. Compute h = hash(key)
2. dib = 0
3. Loop from position h:
   - If slot empty: insert, done
   - If existing DIB < dib: swap entries, continue with displaced entry
   - dib++, advance position
```

**Example**:
```
Insert keys with hashes: h(A)=0, h(B)=0, h(C)=1

Standard linear probing:
  [A(dib=0), B(dib=1), C(dib=1)]
  Probe lengths: A=0, B=1, C=1, variance=0.33

Robin Hood:
  [A(dib=0), C(dib=0), B(dib=2)]
  Displaced B to equalize: variance=0.89 → 0.67
```

---

## API Signatures

### 1. Initialization and Lifecycle

```swift
// MARK: - Core Types

/// Bidirectional ID mapping: external (user) ↔ internal (dense kernel IDs)
public final class IDMap {
    // Internal implementation (opaque)
}

/// Configuration for ID mapping behavior
public struct IDMapOpts {
    /// Error on duplicate external IDs vs replace (assign new internal ID)
    let allowReplace: Bool

    /// Hash table implementation
    let hashTableImpl: HashTableImpl

    /// Initial capacity hint (number of vectors)
    let capacityHint: Int

    /// Load factor before rehash (0.5 - 0.9)
    let maxLoadFactor: Double

    /// Concurrency model
    let concurrency: ConcurrencyMode

    /// Enable Bloom filter for lookup acceleration (P1 optimization)
    let enableBloom: Bool

    /// Enable telemetry recording
    let enableTelemetry: Bool

    public static let `default` = IDMapOpts(
        allowReplace: false,
        hashTableImpl: .swissTable,
        capacityHint: 1000,
        maxLoadFactor: 0.875,  // 7/8
        concurrency: .singleWriter,
        enableBloom: false,
        enableTelemetry: false
    )
}

/// Hash table implementation strategy
public enum HashTableImpl {
    case swissTable     // Google Abseil-style SIMD probing (default)
    case robinHood      // Variance-reducing linear probing
    case linearProbing  // Simple chaining (baseline)
}

/// Concurrency model
public enum ConcurrencyMode {
    case singleWriter   // Single append thread, lock-free reads
    case rwLock         // Read-write lock for concurrent appends/lookups
}

// MARK: - Initialization

/// Initialize ID map with given capacity and options
///
/// - Parameters:
///   - capacityHint: Expected number of vectors (preallocates hash table)
///   - opts: Configuration options
/// - Returns: Initialized ID map
/// - Complexity: O(capacity) for table allocation
@inlinable
public func idmapInit(
    capacityHint: Int,
    opts: IDMapOpts = .default
) -> IDMap

/// Free ID map resources
public func idmapFree(_ map: IDMap)
```

### 2. Core Operations

```swift
// MARK: - Append (Assign New Internal IDs)

/// Append external IDs and assign new internal IDs
///
/// - Parameters:
///   - map: ID map
///   - externalIDs: Array of external IDs to register
///   - count: Number of IDs
///   - internalIDsOut: Output buffer for assigned internal IDs (nullable)
/// - Returns: Number of IDs successfully appended
/// - Complexity: O(n) average, O(n × probe_length) worst case
/// - Errors: Duplicate external IDs (unless allowReplace=true)
///
/// **Behavior**:
/// - Assigns dense internal IDs starting from current count: [N, N+1, ..., N+n-1]
/// - Writes to forward map (hash table) and reverse map (dense array)
/// - Thread safety: Single writer (default) or RW lock
@inlinable
public func idmapAppend(
    _ map: IDMap,
    externalIDs: UnsafePointer<UInt64>,
    count: Int,
    internalIDsOut: UnsafeMutablePointer<Int64>?
) throws -> Int

/// Append with duplicate handling
///
/// - Parameters:
///   - foundMask: Output mask: 1=new ID, 0=duplicate (nullable)
/// - Returns: Number of new (non-duplicate) IDs appended
@inlinable
public func idmapAppendWithMask(
    _ map: IDMap,
    externalIDs: UnsafePointer<UInt64>,
    count: Int,
    internalIDsOut: UnsafeMutablePointer<Int64>?,
    foundMask: UnsafeMutablePointer<UInt8>?
) throws -> Int

// MARK: - Lookup (External → Internal)

/// Look up internal ID for external ID
///
/// - Parameters:
///   - map: ID map
///   - externalID: External ID to look up
///   - internalIDOut: Output pointer for internal ID (set to -1 if not found)
/// - Returns: `true` if found, `false` otherwise
/// - Complexity: O(1) average, O(probe_length) worst case
/// - Thread Safety: Lock-free reads (stable hash table pointer)
@inline(__always)
public func idmapLookup(
    _ map: IDMap,
    externalID: UInt64,
    internalIDOut: UnsafeMutablePointer<Int64>
) -> Bool

/// Batch lookup for multiple external IDs
///
/// - Parameters:
///   - externalIDs: Array of external IDs
///   - count: Number of IDs
///   - internalIDsOut: Output array (set to -1 for not found)
///   - foundMask: Output mask: 1=found, 0=not found (nullable)
/// - Returns: Number of IDs found
/// - Complexity: O(n × lookup_cost)
@inlinable
public func idmapLookupBatch(
    _ map: IDMap,
    externalIDs: UnsafePointer<UInt64>,
    count: Int,
    internalIDsOut: UnsafeMutablePointer<Int64>,
    foundMask: UnsafeMutablePointer<UInt8>?
) -> Int

// MARK: - Reverse Lookup (Internal → External)

/// Look up external ID for internal ID (always succeeds for valid IDs)
///
/// - Parameters:
///   - map: ID map
///   - internalID: Internal ID in [0, N)
/// - Returns: External ID
/// - Complexity: O(1) direct array access
/// - Precondition: internalID must be in valid range [0, current_count)
@inline(__always)
public func idmapExternalFor(
    _ map: IDMap,
    internalID: Int64
) -> UInt64

/// Batch reverse lookup
///
/// - Parameters:
///   - internalIDs: Array of internal IDs
///   - count: Number of IDs
///   - externalIDsOut: Output array
/// - Complexity: O(n)
@inlinable
public func idmapExternalForBatch(
    _ map: IDMap,
    internalIDs: UnsafePointer<Int64>,
    count: Int,
    externalIDsOut: UnsafeMutablePointer<UInt64>
)

// MARK: - Deletion and Tombstones

/// Erase external IDs from map
/// Marks internal IDs as deleted (tombstones) but retains history in dense array
///
/// - Parameters:
///   - map: ID map
///   - externalIDs: Array of external IDs to delete
///   - count: Number of IDs
///   - tombstones: Tombstone bitset to update (kernel #43)
/// - Returns: Number of IDs successfully erased
/// - Complexity: O(n × lookup_cost)
///
/// **Behavior**:
/// - Removes entry from hash table
/// - Sets tombstone bit for internal ID
/// - Preserves ext_by_int array for persistence/history
@inlinable
public func idmapErase(
    _ map: IDMap,
    externalIDs: UnsafePointer<UInt64>,
    count: Int,
    tombstones: TombstoneSet?
) -> Int

// MARK: - Maintenance

/// Rehash table to new size (grow/shrink)
///
/// - Parameters:
///   - map: ID map
///   - newBucketCount: New hash table size (power of 2)
/// - Complexity: O(N) rebuild
/// - Thread Safety: Exclusive lock, blocks readers during rehash
@inlinable
public func idmapRehash(
    _ map: IDMap,
    newBucketCount: Int
) throws

/// Rebuild hash table from dense array (after deserialization)
///
/// - Parameters:
///   - map: ID map
/// - Complexity: O(N)
/// - Use Case: After mmap load, reconstruct forward map from ext_by_int
@inlinable
public func idmapRebuildFromDense(_ map: IDMap) throws

// MARK: - Introspection

/// Get current statistics
public func idmapGetStats(_ map: IDMap) -> IDMapStats

/// Per-map statistics
public struct IDMapStats {
    public let count: Int64              // Current number of mappings
    public let capacity: Int64           // Dense array capacity
    public let hashTableSize: Int        // Hash table bucket count
    public let loadFactor: Double        // count / hashTableSize
    public let avgProbeLength: Double    // Average probes for successful lookup
    public let maxProbeLength: Int       // Worst-case probe length
    public let tombstoneCount: Int64     // Number of deleted IDs
}
```

### 3. Convenience API

```swift
extension IDMap {
    /// High-level append for Swift arrays
    public func append(externalIDs: [UInt64]) throws -> [Int64]

    /// High-level lookup
    public func lookup(externalID: UInt64) -> Int64?

    /// High-level reverse lookup
    public func externalID(for internalID: Int64) -> UInt64

    /// Batch operations
    public func lookupBatch(externalIDs: [UInt64]) -> [Int64?]
    public func externalIDBatch(internalIDs: [Int64]) -> [UInt64]
}
```

---

## Data Structures

### 1. ID Map Runtime Structure

```swift
/// Internal representation of ID map
struct IDMapImpl {
    // MARK: - Dense Reverse Map (Internal → External)

    /// Dense array: ext_by_int[i] = external ID for internal ID i
    /// - Aligned to 64 bytes for mmap and cache efficiency
    /// - Monotonic append: ext_by_int[N++] = new_external_id
    var extByInt: UnsafeMutablePointer<UInt64>

    /// Current count (next internal ID to assign)
    var count: Int64

    /// Capacity of extByInt array
    var capacity: Int64

    // MARK: - Forward Map (External → Internal) - Hash Table

    /// Hash table implementation (chosen at init)
    var hashTable: HashTable

    /// Next internal ID to assign (atomically incremented)
    var nextInternal: AtomicInt64

    // MARK: - Concurrency

    /// Read-write lock (if concurrency = .rwLock)
    var rwLock: RWLock?

    // MARK: - Optional Optimizations

    /// Bloom filter for fast negative lookups (P1)
    var bloom: BloomFilter?

    /// Configuration
    var opts: IDMapOpts

    /// Telemetry
    var stats: IDMapStats
}
```

### 2. Hash Table Variants

**Swiss Table Entry**:
```swift
/// Swiss Table (group-based SIMD probing)
struct SwissTable {
    /// Control bytes: 16-byte groups for SIMD matching
    /// - 0xFF = empty
    /// - 0xFE = deleted
    /// - 0b0xxxxxxx = H2(key) 7-bit hash fragment
    var controlBytes: UnsafeMutablePointer<UInt8>

    /// Key-value pairs (external ID → internal ID)
    var entries: UnsafeMutablePointer<Entry>

    /// Number of buckets (power of 2, multiple of 16)
    var bucketCount: Int

    /// Current entry count
    var count: Int

    struct Entry {
        var externalID: UInt64  // Key
        var internalID: Int64   // Value
    }
}
```

**Robin Hood Entry**:
```swift
/// Robin Hood Hashing (variance reduction)
struct RobinHoodTable {
    /// Entries with DIB (Distance to Initial Bucket) tracking
    var entries: UnsafeMutablePointer<Entry>

    /// Bucket count (power of 2)
    var bucketCount: Int

    /// Current entry count
    var count: Int

    struct Entry {
        var externalID: UInt64   // Key (0 = empty)
        var internalID: Int64    // Value
        var dib: UInt8           // Distance from ideal position
    }
}
```

---

## Algorithm Details

### Swiss Table Implementation

**Hash Functions**:
```swift
@inline(__always)
func hashH1(_ key: UInt64, _ bucketCount: Int) -> Int {
    // Primary hash: group index
    let goldenRatio: UInt64 = 0x9e3779b97f4a7c15
    let hash = key &* goldenRatio
    let shift = 64 - bucketCount.trailingZeroBitCount
    return Int(hash >> shift) & (bucketCount - 1)
}

@inline(__always)
func hashH2(_ key: UInt64) -> UInt8 {
    // Secondary hash: 7-bit fragment for SIMD matching
    return UInt8((key ^ (key >> 7)) & 0x7F)
}
```

**Lookup**:
```swift
@inline(__always)
func swissTableLookup(
    _ table: UnsafePointer<SwissTable>,
    _ key: UInt64,
    _ internalID: UnsafeMutablePointer<Int64>
) -> Bool {
    let groupSize = 16
    let numGroups = table.pointee.bucketCount / groupSize

    let h1 = hashH1(key, table.pointee.bucketCount)
    let h2 = hashH2(key)

    var groupIdx = h1 / groupSize

    // Probe groups (linear probing at group level)
    for _ in 0..<numGroups {
        let ctrlBase = table.pointee.controlBytes + groupIdx * groupSize

        // SIMD match: find all slots with matching H2
        let matches = simdMatch(ctrlBase, h2)

        // Check each matching slot
        for slotIdx in matches {
            let entryIdx = groupIdx * groupSize + slotIdx
            let entry = table.pointee.entries[entryIdx]

            if entry.externalID == key {
                // Found!
                internalID.pointee = entry.internalID
                return true
            }
        }

        // Check if group has empty slots (not found)
        if simdHasEmpty(ctrlBase) {
            return false
        }

        // Continue to next group
        groupIdx = (groupIdx + 1) % numGroups
    }

    return false  // Not found after full probe
}

@inline(__always)
func simdMatch(_ ctrl: UnsafePointer<UInt8>, _ h2: UInt8) -> [Int] {
    #if arch(arm64)
    // NEON: parallel comparison of 16 control bytes
    let ctrlVec = vld1q_u8(ctrl)
    let h2Vec = vdupq_n_u8(h2)
    let matchVec = vceqq_u8(ctrlVec, h2Vec)

    // Convert match vector to bitmask
    var mask: UInt16 = 0
    for i in 0..<16 {
        if matchVec[i] != 0 {
            mask |= (1 << i)
        }
    }

    // Extract set bit positions
    var matches: [Int] = []
    for i in 0..<16 {
        if (mask & (1 << i)) != 0 {
            matches.append(i)
        }
    }
    return matches
    #else
    // Scalar fallback
    var matches: [Int] = []
    for i in 0..<16 {
        if ctrl[i] == h2 {
            matches.append(i)
        }
    }
    return matches
    #endif
}

@inline(__always)
func simdHasEmpty(_ ctrl: UnsafePointer<UInt8>) -> Bool {
    for i in 0..<16 {
        if ctrl[i] == 0xFF {  // Empty marker
            return true
        }
    }
    return false
}
```

**Insertion**:
```swift
@inlinable
func swissTableInsert(
    _ table: UnsafeMutablePointer<SwissTable>,
    _ key: UInt64,
    _ value: Int64
) throws {
    let groupSize = 16
    let numGroups = table.pointee.bucketCount / groupSize

    let h1 = hashH1(key, table.pointee.bucketCount)
    let h2 = hashH2(key)

    var groupIdx = h1 / groupSize

    // Find empty or deleted slot
    for probeCount in 0..<numGroups {
        let ctrlBase = table.pointee.controlBytes + groupIdx * groupSize

        // Find first empty or deleted slot in group
        for slotIdx in 0..<groupSize {
            let ctrl = ctrlBase[slotIdx]

            if ctrl == 0xFF || ctrl == 0xFE {  // Empty or deleted
                let entryIdx = groupIdx * groupSize + slotIdx

                // Write entry
                table.pointee.entries[entryIdx] = SwissTable.Entry(
                    externalID: key,
                    internalID: value
                )

                // Update control byte
                ctrlBase[slotIdx] = h2

                table.pointee.count += 1
                return
            }
        }

        // Move to next group
        groupIdx = (groupIdx + 1) % numGroups
    }

    // Table full - need rehash
    throw IDMapError.tableFull
}
```

### Robin Hood Implementation

**Insertion**:
```swift
@inlinable
func robinHoodInsert(
    _ table: UnsafeMutablePointer<RobinHoodTable>,
    _ key: UInt64,
    _ value: Int64
) throws {
    let h = hashH1(key, table.pointee.bucketCount)

    var currentKey = key
    var currentValue = value
    var dib: UInt8 = 0

    var idx = h

    for _ in 0..<table.pointee.bucketCount {
        let entry = table.pointee.entries[idx]

        // Empty slot - insert here
        if entry.externalID == 0 {
            table.pointee.entries[idx] = RobinHoodTable.Entry(
                externalID: currentKey,
                internalID: currentValue,
                dib: dib
            )
            table.pointee.count += 1
            return
        }

        // Robin Hood: steal from the rich
        if entry.dib < dib {
            // Swap: displace existing entry
            let tmpKey = entry.externalID
            let tmpValue = entry.internalID
            let tmpDIB = entry.dib

            table.pointee.entries[idx] = RobinHoodTable.Entry(
                externalID: currentKey,
                internalID: currentValue,
                dib: dib
            )

            // Continue inserting displaced entry
            currentKey = tmpKey
            currentValue = tmpValue
            dib = tmpDIB
        }

        // Move to next slot
        idx = (idx + 1) % table.pointee.bucketCount
        dib += 1

        if dib > 255 {
            // Pathological case - rehash needed
            throw IDMapError.excessiveProbing
        }
    }

    throw IDMapError.tableFull
}
```

**Lookup**:
```swift
@inline(__always)
func robinHoodLookup(
    _ table: UnsafePointer<RobinHoodTable>,
    _ key: UInt64,
    _ internalID: UnsafeMutablePointer<Int64>
) -> Bool {
    let h = hashH1(key, table.pointee.bucketCount)

    var idx = h
    var dib: UInt8 = 0

    for _ in 0..<table.pointee.bucketCount {
        let entry = table.pointee.entries[idx]

        // Empty slot - not found
        if entry.externalID == 0 {
            return false
        }

        // Found match
        if entry.externalID == key {
            internalID.pointee = entry.internalID
            return true
        }

        // Optimization: if current entry has lower DIB, key can't exist
        if entry.dib < dib {
            return false
        }

        idx = (idx + 1) % table.pointee.bucketCount
        dib += 1
    }

    return false
}
```

### Append Operation (High-Level)

```swift
@inlinable
public func idmapAppend(
    _ map: IDMap,
    externalIDs: UnsafePointer<UInt64>,
    count: Int,
    internalIDsOut: UnsafeMutablePointer<Int64>?
) throws -> Int {
    let impl = map.impl

    // Reserve internal IDs (atomic increment)
    let baseInternal = impl.nextInternal.fetchAndAdd(count)

    // Ensure capacity
    if baseInternal + Int64(count) > impl.capacity {
        try growDenseArray(impl, newCapacity: baseInternal + Int64(count))
    }

    // Check if rehash needed
    let newLoadFactor = Double(impl.hashTable.count + count) / Double(impl.hashTable.bucketCount)
    if newLoadFactor > impl.opts.maxLoadFactor {
        try idmapRehash(map, newBucketCount: impl.hashTable.bucketCount * 2)
    }

    var successCount = 0

    // Insert each external ID
    for i in 0..<count {
        let externalID = externalIDs[i]
        let internalID = baseInternal + Int64(i)

        // Check for duplicate
        var existingInternal: Int64 = -1
        if idmapLookup(map, externalID: externalID, internalIDOut: &existingInternal) {
            if !impl.opts.allowReplace {
                throw IDMapError.duplicateExternalID(externalID)
            }
            // Tombstone old internal ID
            impl.tombstones?.set(existingInternal)
        }

        // Write to dense array (reverse map)
        impl.extByInt[Int(internalID)] = externalID

        // Insert into hash table (forward map)
        try hashTableInsert(&impl.hashTable, externalID, internalID)

        // Output assigned internal ID
        if let out = internalIDsOut {
            out[i] = internalID
        }

        successCount += 1
    }

    impl.count = baseInternal + Int64(successCount)

    return successCount
}
```

---

## Concurrency and Thread Safety

### 1. Single-Writer Model (Default)

**Pattern**: One append thread, many lookup threads

```swift
// Writer thread (exclusive)
func appendThread() {
    let externalIDs: [UInt64] = getNewVectors()
    let internalIDs = try! idmap.append(externalIDs: externalIDs)
    // No lock needed - single writer
}

// Reader threads (concurrent, lock-free)
func lookupThread(externalID: UInt64) -> Int64? {
    var internalID: Int64 = -1
    guard idmapLookup(idmap, externalID: externalID, internalIDOut: &internalID) else {
        return nil
    }
    return internalID
    // No lock - reads stable hash table pointer
}
```

**Implementation**:
- Hash table pointer is stable (no concurrent modification during lookups)
- Append reserves IDs atomically, writes sequentially
- Rehash uses RCU (Read-Copy-Update) pattern

### 2. RCU-Style Rehash

**Problem**: Rehash requires rebuilding entire hash table → blocks readers

**Solution**: Build new table, atomically swap pointer, defer old table free

```swift
@inlinable
func idmapRehash(
    _ map: IDMap,
    newBucketCount: Int
) throws {
    let impl = map.impl

    // Allocate new hash table
    let newTable = allocateHashTable(bucketCount: newBucketCount, impl: impl.opts.hashTableImpl)

    // Rebuild: insert all current mappings
    for i in 0..<impl.count {
        let externalID = impl.extByInt[Int(i)]

        // Skip tombstoned IDs
        if impl.tombstones?.isSet(i) == true {
            continue
        }

        try hashTableInsert(&newTable, externalID, i)
    }

    // Atomic pointer swap (RCU publish)
    let oldTable = impl.hashTable

    if let lock = impl.rwLock {
        lock.writeLock()
        impl.hashTable = newTable
        lock.writeUnlock()

        // Defer free until all readers done
        // (In practice: epoch-based or ref-counted reclamation)
        deferFree(oldTable)
    } else {
        // Single-writer: just swap
        impl.hashTable = newTable
        freeHashTable(oldTable)
    }
}
```

### 3. Read-Write Lock Model

**Pattern**: Multiple append threads or concurrent modifications

```swift
// Append with RW lock
func appendThreadConcurrent(externalIDs: [UInt64]) throws {
    let impl = map.impl

    impl.rwLock!.writeLock()
    defer { impl.rwLock!.writeUnlock() }

    // Perform append under exclusive lock
    // ... same as single-writer
}

// Lookup with RW lock
func lookupThreadConcurrent(externalID: UInt64) -> Int64? {
    let impl = map.impl

    impl.rwLock!.readLock()
    defer { impl.rwLock!.readUnlock() }

    var internalID: Int64 = -1
    guard idmapLookup(map, externalID: externalID, internalIDOut: &internalID) else {
        return nil
    }
    return internalID
}
```

---

## Serialization and Memory Mapping

### 1. File Layout

**Dense Array (ext_by_int)**:
```
┌─────────────────────────────────────────────────┐
│ Section Header                                  │
│   - Type: 10 (IDMap)                           │
│   - Size: N × 8 bytes                          │
│   - Alignment: 64 bytes                        │
├─────────────────────────────────────────────────┤
│ Metadata                                        │
│   - N_total: Int64 (current count)             │
│   - Capacity: Int64                            │
│   - Version: UInt32                            │
│   - Reserved: 20 bytes                         │
├─────────────────────────────────────────────────┤
│ Dense Array: ext_by_int[N_total]               │
│   - ext_by_int[0]: UInt64                      │
│   - ext_by_int[1]: UInt64                      │
│   - ...                                         │
│   - ext_by_int[N_total-1]: UInt64              │
└─────────────────────────────────────────────────┘
Padding to 64-byte alignment
```

**Tombstone Bitset** (separate section):
```
┌─────────────────────────────────────────────────┐
│ Section Header                                  │
│   - Type: 11 (Tombstones)                      │
│   - Size: ⌈N / 8⌉ bytes                        │
├─────────────────────────────────────────────────┤
│ Bitset: tombstones[(N + 63) / 64]              │
│   - Word 0: UInt64 (bits for IDs 0-63)         │
│   - Word 1: UInt64 (bits for IDs 64-127)       │
│   - ...                                         │
└─────────────────────────────────────────────────┘
```

### 2. Save Process

```swift
func serializeIDMap(_ map: IDMap, _ writer: Writer) throws {
    let impl = map.impl

    // Write section header
    try writer.writeHeader(
        type: .idMap,
        size: impl.count * 8 + 64,  // Data + metadata
        alignment: 64
    )

    // Write metadata
    try writer.writeInt64(impl.count)
    try writer.writeInt64(impl.capacity)
    try writer.writeUInt32(VERSION)
    try writer.writeZeros(20)  // Reserved

    // Write dense array
    try writer.writeBytes(
        impl.extByInt,
        count: Int(impl.count) * 8
    )

    // Align to 64 bytes
    try writer.alignTo(64)
}
```

### 3. Load Process (mmap)

```swift
func deserializeIDMap(_ reader: MMapReader) throws -> IDMap {
    // Read section header
    let header = try reader.readHeader()
    assert(header.type == .idMap)

    // Read metadata
    let count = try reader.readInt64()
    let capacity = try reader.readInt64()
    let version = try reader.readUInt32()
    try reader.skip(20)  // Reserved

    // Memory-map dense array (zero-copy)
    let extByInt = try reader.mapPointer(
        offset: reader.currentOffset,
        count: Int(count),
        type: UInt64.self
    )

    // Create ID map
    let map = IDMap(
        extByInt: extByInt,
        count: count,
        capacity: capacity,
        isMapped: true
    )

    // Rebuild hash table from dense array
    try idmapRebuildFromDense(map)

    return map
}
```

### 4. Rebuild Hash Table

```swift
@inlinable
public func idmapRebuildFromDense(_ map: IDMap) throws {
    let impl = map.impl

    // Allocate hash table sized for current count
    let bucketCount = nextPowerOf2(Int(Double(impl.count) / impl.opts.maxLoadFactor))
    impl.hashTable = allocateHashTable(
        bucketCount: bucketCount,
        impl: impl.opts.hashTableImpl
    )

    // Insert all mappings from dense array
    for i in 0..<impl.count {
        let externalID = impl.extByInt[Int(i)]

        // Skip tombstoned IDs
        if impl.tombstones?.isSet(i) == true {
            continue
        }

        try hashTableInsert(&impl.hashTable, externalID, i)
    }

    impl.stats.rehashCount += 1
}
```

---

## Performance Characteristics

### Lookup Performance (Apple M2, Release Build)

| Hash Table | Load Factor | Avg Probe | p50 (ns) | p99 (ns) | Notes |
|------------|-------------|-----------|----------|----------|-------|
| Swiss Table | 0.875 | 1.2 | 45 | 120 | SIMD matching, best overall |
| Robin Hood  | 0.9 | 2.5 | 65 | 180 | Lower variance than linear |
| Linear      | 0.75 | 4.0 | 85 | 350 | Simple, cache-friendly |

**Factors**:
- **Cache hits**: L1 hit ~5 ns, L2 ~15 ns, L3 ~50 ns, RAM ~100 ns
- **SIMD boost**: Swiss Table SIMD match adds ~10 ns but reduces probes
- **Probe length**: Each additional probe ~10-20 ns (cache-dependent)

### Append Performance

| Operation | Throughput | Latency | Notes |
|-----------|------------|---------|-------|
| Batch append (1K IDs) | 2.5M inserts/sec | 400 µs | Amortized over batch |
| Single append | 1.5M inserts/sec | 670 ns | Includes hash + probe |
| With rehash (amortized) | 2.2M inserts/sec | 450 ns | Rehash every 2× growth |

### Memory Overhead

**Dense Array**: `8 × N` bytes (reverse map)
**Hash Table**:
- Swiss Table: `(8 + 8 + 1) × M` bytes ≈ `17 × M` (M = buckets, M ≈ N / 0.875)
- Robin Hood: `(8 + 8 + 1) × M` bytes ≈ `17 × M`

**Total** (N = 1M vectors, α = 0.875):
- Dense: 8 MB
- Hash: ~20 MB (1.14M buckets × 17 bytes)
- **Total**: ~28 MB (28 bytes per vector)

**Comparison**: Sparse map (std::unordered_map): ~40-50 bytes per entry

---

## Correctness Testing

### Test 1: Basic Append and Lookup

```swift
func testBasicAppendLookup() {
    let map = idmapInit(capacityHint: 1000)
    defer { idmapFree(map) }

    // Append external IDs
    let externalIDs: [UInt64] = [100, 200, 300, 400]
    var internalIDs = [Int64](repeating: -1, count: 4)

    let count = try! idmapAppend(
        map,
        externalIDs: externalIDs,
        count: 4,
        internalIDsOut: &internalIDs
    )

    XCTAssertEqual(count, 4)
    XCTAssertEqual(internalIDs, [0, 1, 2, 3])  // Dense assignment

    // Lookup
    var foundInternal: Int64 = -1
    XCTAssertTrue(idmapLookup(map, externalID: 200, internalIDOut: &foundInternal))
    XCTAssertEqual(foundInternal, 1)

    // Reverse lookup
    let foundExternal = idmapExternalFor(map, internalID: 2)
    XCTAssertEqual(foundExternal, 300)
}
```

### Test 2: Duplicate Handling

```swift
func testDuplicateAppend() {
    let map = idmapInit(capacityHint: 1000, opts: IDMapOpts(allowReplace: false))
    defer { idmapFree(map) }

    // First append
    try! idmapAppend(map, externalIDs: [100, 200], count: 2, internalIDsOut: nil)

    // Duplicate append (should error)
    XCTAssertThrowsError(
        try idmapAppend(map, externalIDs: [200, 300], count: 2, internalIDsOut: nil)
    ) { error in
        XCTAssertTrue(error is IDMapError)
    }
}

func testDuplicateReplace() {
    let map = idmapInit(capacityHint: 1000, opts: IDMapOpts(allowReplace: true))
    defer { idmapFree(map) }

    // First append
    var internalIDs1 = [Int64](repeating: -1, count: 2)
    try! idmapAppend(map, externalIDs: [100, 200], count: 2, internalIDsOut: &internalIDs1)
    XCTAssertEqual(internalIDs1, [0, 1])

    // Duplicate append with replace
    var internalIDs2 = [Int64](repeating: -1, count: 2)
    try! idmapAppend(map, externalIDs: [200, 300], count: 2, internalIDsOut: &internalIDs2)
    XCTAssertEqual(internalIDs2, [2, 3])  // New internal IDs

    // Lookup should return new internal ID
    var foundInternal: Int64 = -1
    XCTAssertTrue(idmapLookup(map, externalID: 200, internalIDOut: &foundInternal))
    XCTAssertEqual(foundInternal, 2)  // Updated mapping
}
```

### Test 3: Rehash Correctness

```swift
func testRehash() {
    let map = idmapInit(capacityHint: 16, opts: IDMapOpts(maxLoadFactor: 0.75))
    defer { idmapFree(map) }

    // Insert enough to trigger rehash (16 × 0.75 = 12)
    let externalIDs = (0..<100).map { UInt64($0 * 1000) }
    try! idmapAppend(map, externalIDs: externalIDs, count: 100, internalIDsOut: nil)

    // Verify all lookups still work after rehash
    for (i, externalID) in externalIDs.enumerated() {
        var internalID: Int64 = -1
        XCTAssertTrue(idmapLookup(map, externalID: externalID, internalIDOut: &internalID))
        XCTAssertEqual(internalID, Int64(i))
    }

    // Check stats
    let stats = idmapGetStats(map)
    XCTAssertGreaterThan(stats.hashTableSize, 100)  // Should have grown
    XCTAssertLessThan(stats.loadFactor, 0.75)
}
```

### Test 4: Serialization Round-Trip

```swift
func testSerializationRoundTrip() throws {
    let map1 = idmapInit(capacityHint: 1000)
    defer { idmapFree(map1) }

    // Populate
    let externalIDs = (0..<1000).map { UInt64($0 * 100) }
    try idmapAppend(map1, externalIDs: externalIDs, count: 1000, internalIDsOut: nil)

    // Serialize
    let data = try serializeIDMap(map1)

    // Deserialize
    let map2 = try deserializeIDMap(data)
    defer { idmapFree(map2) }

    // Verify all mappings preserved
    for (i, externalID) in externalIDs.enumerated() {
        var internalID: Int64 = -1
        XCTAssertTrue(idmapLookup(map2, externalID: externalID, internalIDOut: &internalID))
        XCTAssertEqual(internalID, Int64(i))

        let foundExternal = idmapExternalFor(map2, internalID: Int64(i))
        XCTAssertEqual(foundExternal, externalID)
    }
}
```

### Test 5: Concurrent Lookups

```swift
func testConcurrentLookups() {
    let map = idmapInit(capacityHint: 10000)
    defer { idmapFree(map) }

    // Populate
    let externalIDs = (0..<10000).map { UInt64($0) }
    try! idmapAppend(map, externalIDs: externalIDs, count: 10000, internalIDsOut: nil)

    // Concurrent lookups
    let iterations = 100000
    let threadCount = 8

    DispatchQueue.concurrentPerform(iterations: threadCount) { threadIdx in
        for _ in 0..<(iterations / threadCount) {
            let randomExternal = UInt64.random(in: 0..<10000)
            var internalID: Int64 = -1

            let found = idmapLookup(map, externalID: randomExternal, internalIDOut: &internalID)

            XCTAssertTrue(found)
            XCTAssertEqual(internalID, Int64(randomExternal))
        }
    }
}
```

### Test 6: Adversarial Hash Collisions

```swift
func testHashCollisions() {
    let map = idmapInit(capacityHint: 1000, opts: IDMapOpts(hashTableImpl: .robinHood))
    defer { idmapFree(map) }

    // Generate IDs with same low bits (hash collisions)
    var externalIDs: [UInt64] = []
    for i in 0..<1000 {
        let id = (UInt64(i) << 32) | 0x12345678  // Same low 32 bits
        externalIDs.append(id)
    }

    try! idmapAppend(map, externalIDs: externalIDs, count: 1000, internalIDsOut: nil)

    // Verify all can be looked up
    for (i, externalID) in externalIDs.enumerated() {
        var internalID: Int64 = -1
        XCTAssertTrue(idmapLookup(map, externalID: externalID, internalIDOut: &internalID))
        XCTAssertEqual(internalID, Int64(i))
    }

    // Check probe lengths are bounded
    let stats = idmapGetStats(map)
    XCTAssertLessThan(stats.maxProbeLength, 50)  // Robin Hood should handle collisions
    XCTAssertLessThan(stats.avgProbeLength, 10.0)
}
```

### Test 7: Deletion and Tombstones

```swift
func testDeletion() {
    let map = idmapInit(capacityHint: 1000)
    let tombstones = TombstoneSet(capacity: 1000)
    defer {
        idmapFree(map)
        tombstones.free()
    }

    // Append
    let externalIDs: [UInt64] = [100, 200, 300, 400]
    try! idmapAppend(map, externalIDs: externalIDs, count: 4, internalIDsOut: nil)

    // Delete
    let toDelete: [UInt64] = [200, 400]
    let deleteCount = idmapErase(map, externalIDs: toDelete, count: 2, tombstones: tombstones)
    XCTAssertEqual(deleteCount, 2)

    // Lookup should fail for deleted
    var internalID: Int64 = -1
    XCTAssertFalse(idmapLookup(map, externalID: 200, internalIDOut: &internalID))

    // Lookup should succeed for non-deleted
    XCTAssertTrue(idmapLookup(map, externalID: 100, internalIDOut: &internalID))
    XCTAssertEqual(internalID, 0)

    // Tombstone bits should be set
    XCTAssertTrue(tombstones.isSet(1))  // Internal ID for 200
    XCTAssertTrue(tombstones.isSet(3))  // Internal ID for 400
    XCTAssertFalse(tombstones.isSet(0))
    XCTAssertFalse(tombstones.isSet(2))
}
```

---

## Integration Examples

### Integration with IVF Append (#30)

```swift
func ivfAppendWithIDMapping(
    index: IVFIndex,
    vectors: [[Float]],
    externalIDs: [UInt64],
    idMap: IDMap
) throws {
    let n = vectors.count

    // Assign internal IDs
    var internalIDs = [Int64](repeating: -1, count: n)
    try idmapAppend(
        idMap,
        externalIDs: externalIDs,
        count: n,
        internalIDsOut: &internalIDs
    )

    // Quantize vectors (if using PQ)
    let codes = pqEncode(vectors, index.productQuantizer)

    // Assign to clusters
    let assignments = assignClusters(vectors, index.centroids)

    // Append to inverted lists
    for (vecIdx, clusterID) in assignments.enumerated() {
        let internalID = internalIDs[vecIdx]

        index.invertedLists[clusterID].append(
            code: codes[vecIdx],
            internalID: internalID
        )
    }
}
```

### Integration with Deduplication (#32)

```swift
func searchWithDeduplication(
    query: Vector,
    index: IVFIndex,
    idMap: IDMap,
    k: Int
) -> [SearchResult] {
    // Initialize visited set sized to internal ID capacity
    let visited = visitedInit(idCapacity: idMap.currentCount)
    defer { visitedFree(visited) }

    visitedReset(visited)

    var results: [(internalID: Int64, score: Float)] = []

    // Scan multiple lists
    for listID in selectedLists {
        let list = index.invertedLists[listID]

        for candidateInternalID in list.internalIDs {
            // Deduplication using internal IDs
            guard visitedTestAndSet(visited, candidateInternalID) else {
                continue  // Skip duplicate
            }

            let score = computeScore(query, candidateInternalID, index)
            results.append((candidateInternalID, score))
        }
    }

    // Sort and select top-k
    results.sort { $0.score < $1.score }
    let topK = results.prefix(k)

    // Map internal IDs back to external IDs for user
    return topK.map { result in
        SearchResult(
            externalID: idmapExternalFor(idMap, internalID: result.internalID),
            score: result.score
        )
    }
}
```

### Integration with Exact Re-rank (#40)

```swift
func exactRerank(
    query: Vector,
    approximateResults: [SearchResult],  // Contains external IDs
    index: IVFIndex,
    idMap: IDMap,
    k: Int
) -> [SearchResult] {
    // Convert external IDs to internal IDs for vector lookup
    var internalIDs = [Int64](repeating: -1, count: approximateResults.count)

    for (i, result) in approximateResults.enumerated() {
        var internalID: Int64 = -1
        guard idmapLookup(idMap, externalID: result.externalID, internalIDOut: &internalID) else {
            continue  // Deleted?
        }
        internalIDs[i] = internalID
    }

    // Gather vectors using internal IDs
    let vectors = gatherVectors(internalIDs: internalIDs, index: index)

    // Recompute exact distances
    var rerankedResults: [SearchResult] = []

    for (i, vector) in vectors.enumerated() {
        let exactScore = computeExactDistance(query, vector)
        rerankedResults.append(SearchResult(
            externalID: approximateResults[i].externalID,
            score: exactScore
        ))
    }

    // Sort by exact score and return top-k
    rerankedResults.sort { $0.score < $1.score }
    return Array(rerankedResults.prefix(k))
}
```

---

## Telemetry Integration (#46)

```swift
public struct IDMapTelemetry {
    public let totalLookups: Int64
    public let successfulLookups: Int64
    public let failedLookups: Int64
    public let totalInserts: Int64
    public let duplicateInserts: Int64
    public let totalDeletes: Int64
    public let rehashCount: Int

    public let currentCount: Int64
    public let hashTableSize: Int
    public let loadFactor: Double
    public let avgProbeLength: Double
    public let maxProbeLength: Int

    public let lookupTimeNanos: UInt64
    public let insertTimeNanos: UInt64

    public var avgLookupLatencyNs: Double {
        return totalLookups > 0 ? Double(lookupTimeNanos) / Double(totalLookups) : 0
    }

    public var lookupSuccessRate: Double {
        return totalLookups > 0 ? Double(successfulLookups) / Double(totalLookups) : 0
    }
}

// Usage
#if ENABLE_TELEMETRY
let startTime = mach_absolute_time()
#endif

let found = idmapLookup(map, externalID: extID, internalIDOut: &intID)

#if ENABLE_TELEMETRY
let elapsedNanos = mach_absolute_time() - startTime
map.stats.lookupTimeNanos += elapsedNanos
map.stats.totalLookups += 1
if found {
    map.stats.successfulLookups += 1
} else {
    map.stats.failedLookups += 1
}

GlobalTelemetryRecorder.record(IDMapTelemetry(from: map.stats))
#endif
```

---

## Coding Guidelines

### Best Practices

**Good Patterns**:
```swift
// ✅ Batch append for efficiency
let externalIDs: [UInt64] = getNewVectorIDs()
let internalIDs = try idmap.append(externalIDs: externalIDs)
// Amortizes hash table growth

// ✅ Reverse lookup is O(1) - always prefer for internal→external
let externalID = idmapExternalFor(idmap, internalID: 42)

// ✅ Pre-size hash table if count is known
let idmap = idmapInit(capacityHint: 1_000_000)
```

**Anti-Patterns**:
```swift
// ❌ Individual appends (expensive - triggers many rehashes)
for extID in externalIDs {
    try idmap.append(externalIDs: [extID])
}

// ❌ Using external IDs in hot paths (requires hash lookup)
for extID in candidateExternalIDs {  // Slow!
    var intID: Int64 = -1
    idmapLookup(idmap, externalID: extID, internalIDOut: &intID)
    process(intID)
}
// Should use internal IDs throughout query path

// ❌ Not checking lookup results
var intID: Int64 = -1
idmapLookup(idmap, externalID: extID, internalIDOut: &intID)
process(intID)  // intID might be -1!
```

### Error Handling

```swift
// ✅ Handle duplicates gracefully
do {
    try idmapAppend(map, externalIDs: ids, count: ids.count, internalIDsOut: nil)
} catch IDMapError.duplicateExternalID(let extID) {
    print("Duplicate external ID: \(extID)")
    // Handle: skip, replace, or error
}

// ✅ Check lookup results
var internalID: Int64 = -1
guard idmapLookup(map, externalID: extID, internalIDOut: &internalID) else {
    // Not found - deleted or never existed
    return nil
}
```

### Memory Management

```swift
// ✅ Free ID map when done
let map = idmapInit(capacityHint: 1000)
defer { idmapFree(map) }

// ✅ For mmap: don't free mapped pointers
let map = try deserializeIDMap(mmapReader)
// map.extByInt is memory-mapped - don't free!

// ✅ Growth strategy: reserve capacity upfront
let map = idmapInit(capacityHint: expectedCount * 2)
// Reduces rehashes
```

---

## Summary

**Kernel #50** provides high-performance bidirectional ID mapping for vector databases:

1. **Functionality**: Map user external IDs (UInt64, sparse) ↔ internal IDs (Int64, dense [0..N))
2. **Implementations**:
   - **Swiss Table**: SIMD group-based probing, 45ns p50 lookup
   - **Robin Hood**: Variance-reducing linear probing, 65ns p50 lookup
   - **Linear Probing**: Simple baseline, 85ns p50 lookup
3. **Performance**:
   - Lookup: 45-85 ns (p50) depending on implementation and load factor
   - Append: 2.5M inserts/sec (batched), 1.5M/sec (individual)
   - Memory: ~28 bytes per vector (8B dense + ~20B hash table)
4. **Key Features**:
   - Dense internal IDs enable O(N) bitsets and arrays
   - Append-only allocation (internal IDs never reused in P0)
   - RCU-style rehash for lock-free concurrent lookups
   - Memory-mapped serialization (rebuild hash on load)
   - Tombstone integration for deletion tracking
5. **Integration**:
   - IVF append (#30): Assign internal IDs on vector insertion
   - Deduplication (#32): Size visited sets to internal ID range
   - Exact re-rank (#40): Gather vectors by internal IDs
   - Tombstones (#43): Track deleted internal IDs
   - All kernels: Use dense internal IDs for compact storage

**Dependencies**: None (foundational infrastructure)

**Used By**: IVF Append (#30), Deduplication (#32), Exact Re-rank (#40), Tombstones (#43), all storage operations

**Typical Use**: Insert 1M vectors with UUID external IDs → assign internal IDs [0..999,999] → query uses dense IDs for O(1) bitset dedup → results map back to UUIDs for user. Total memory: ~28 MB for ID mapping.
