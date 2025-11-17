// ===----------------------------------------------------------------------===//
//  IDFilter.swift
//  VectorIndex
//
//  High-performance allow/deny bitset filtering over dense internal IDs.
//  CPU-only, no allocations in hot paths; stable compaction for (ids,scores).
// ===----------------------------------------------------------------------===//

import Foundation

// MARK: - Bitset Representation

/// Bitset for dense ID filtering over the domain [0, N).
/// Backed by ⌈N/64⌉ 64-bit words; 64-byte aligned for cache efficiency.
/// Query-time use is read-only; allocation helpers exist for tests/tools.
public struct IDFilterBitset {
    /// 64-bit words (may be shared across threads; treat as read-only in queries)
    public let words: UnsafeMutablePointer<UInt64>

    /// Number of IDs covered (N)
    public let capacity: Int

    /// Number of words in the bitset (⌈N/64⌉)
    @inlinable public var wordCount: Int { (capacity + 63) >> 6 }

    /// Read-only view for query-time use (thread-safe)
    @inlinable public var readOnly: UnsafePointer<UInt64> {
        UnsafePointer(words)
    }

    /// Construct from an existing (aligned) buffer.
    @inlinable
    public init(words: UnsafeMutablePointer<UInt64>, capacity: Int) {
        self.words = words
        self.capacity = capacity
    }

    // MARK: Query-time test

    /// Test whether the bit for `id` is set (1). Bounds-checked.
    @inlinable
    public func test(id: Int64) -> Bool {
        guard id >= 0 && id < capacity else { return false }
        let wordIdx = Int(id >> 6)      // id / 64
        let bitIdx  = Int(id & 63)      // id % 64
        return ((words[wordIdx] >> bitIdx) & 1) == 1
    }

    // MARK: Build-time helpers (not used in hot paths)

    /// Set the given ID bit to 1 (build/update-time only).
    @inlinable
    public mutating func set(id: Int64) {
        guard id >= 0 && id < capacity else { return }
        let w = Int(id >> 6), b = Int(id & 63)
        words[w] |= (1 as UInt64) << b
    }

    /// Clear the given ID bit to 0 (build/update-time only).
    @inlinable
    public mutating func clear(id: Int64) {
        guard id >= 0 && id < capacity else { return }
        let w = Int(id >> 6), b = Int(id & 63)
        words[w] &= ~((1 as UInt64) << b)
    }

    /// Set the given ID bit to the specified value (build/update-time only).
    @inlinable
    public mutating func set(id: Int64, value: Bool) {
        if value {
            set(id: id)
        } else {
            clear(id: id)
        }
    }

    // MARK: Allocation helpers (for tests/tools)

    /// Allocate a 64-byte aligned bitset buffer of size ⌈capacity/64⌉ words.
    /// Initializes all bits to `initialBit` (false by default).
    public static func allocate(capacity: Int, initialBit: Bool = false, alignment: Int = 64) -> IDFilterBitset {
        precondition(capacity >= 0)
        let wc = (capacity + 63) >> 6
        let raw = UnsafeMutableRawPointer.allocate(byteCount: wc &* MemoryLayout<UInt64>.stride,
                                                   alignment: max(alignment, MemoryLayout<UInt64>.alignment))
        let ptr = raw.bindMemory(to: UInt64.self, capacity: wc)
        let initVal: UInt64 = initialBit ? ~UInt64(0) : 0
        ptr.initialize(repeating: initVal, count: wc)
        return IDFilterBitset(words: ptr, capacity: capacity)
    }

    /// Deallocate the underlying storage.
    public func deallocate() {
        words.deinitialize(count: wordCount)
        words.deallocate()
    }
}

// MARK: - Filter Mode

/// Bitset interpretation mode.
/// - allowlist: bit 1 = keep, 0 = drop
/// - denylist:  bit 1 = drop, 0 = keep
public enum FilterMode {
    case allowlist
    case denylist

    @inlinable
    func shouldKeep(bit: Bool) -> Bool {
        switch self {
        case .allowlist: return bit
        case .denylist:  return !bit
        }
    }
}

// MARK: - Inline Single-ID Checks (hot path)

/// Fast, inline single-ID check against a single bitset.
/// Bounds-checked; branchless after guard.
/// - Returns: `true` if ID should be kept under `mode`.
@inline(__always)
public func idFilterPass(
    bitset: UnsafePointer<UInt64>,
    id: Int64,
    capacity: Int,
    mode: FilterMode
) -> Bool {
    // Bounds check (critical for safety)
    guard id >= 0 && id < capacity else { return false }

    // Bit arithmetic (branchless after guard)
    let wordIdx = Int(id >> 6)      // / 64
    let bitIdx  = Int(id & 63)      // % 64
    let word    = bitset[wordIdx]
    let bit     = ((word >> bitIdx) & 1) == 1

    // Apply mode
    switch mode {
    case .allowlist: return bit
    case .denylist:  return !bit
    }
}
// Spec: single-ID pass semantics, branchless core.  [oai_citation:3‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)  [oai_citation:4‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)

// MARK: - Composed Multi-Filter (up to 4 allowlists + 1 deny)

/// Composition semantics:
/// keep = (allow0 AND allow1 AND allow2 AND allow3) AND NOT(deny)
/// Any `allow*` pointer may be nil (skipped); `deny` may be nil.
/// Early-exits on the first failing allow; single-pass index computation.
@inline(__always)
public func idFilterPassN(
    allow0: UnsafePointer<UInt64>?,
    allow1: UnsafePointer<UInt64>?,
    allow2: UnsafePointer<UInt64>?,
    allow3: UnsafePointer<UInt64>?,
    deny: UnsafePointer<UInt64>?,
    id: Int64,
    capacity: Int
) -> Bool {
    guard id >= 0 && id < capacity else { return false }

    // Compute once; reuse (common subexpression elimination)
    let wordIdx = Int(id >> 6)
    let bitIdx  = Int(id & 63)
    let mask = (1 as UInt64) << bitIdx

    // Early exit on first failing allow
    if let a0 = allow0, (a0[wordIdx] & mask) == 0 { return false }
    if let a1 = allow1, (a1[wordIdx] & mask) == 0 { return false }
    if let a2 = allow2, (a2[wordIdx] & mask) == 0 { return false }
    if let a3 = allow3, (a3[wordIdx] & mask) == 0 { return false }

    // Deny check
    if let d = deny, (d[wordIdx] & mask) != 0 { return false }

    return true
}
// Spec: composed allow∧…∧allow ∧ ¬deny, with early-exit and shared indices.  [oai_citation:5‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)

// MARK: - Batch Mask

/// Generate a keep/drop mask for `n` IDs against a single bitset.
/// - Returns: number of kept IDs (mask[i] = 1 → keep, 0 → drop).
@inlinable
public func idFilterMask(
    bitset: UnsafePointer<UInt64>,
    ids: UnsafePointer<Int64>?,         // if nil, use implicit IDs 0..<n
    count n: Int,
    capacity: Int,
    mode: FilterMode,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    var kept = 0
    if let idsPtr = ids {
        for i in 0..<n {
            let id = idsPtr[i]
            let pass = idFilterPass(bitset: bitset, id: id, capacity: capacity, mode: mode)
            maskOut[i] = pass ? 1 : 0
            kept &+= pass ? 1 : 0
        }
    } else {
        // Implicit IDs: 0..<n
        for i in 0..<n {
            let id = Int64(i)
            let pass = idFilterPass(bitset: bitset, id: id, capacity: capacity, mode: mode)
            maskOut[i] = pass ? 1 : 0
            kept &+= pass ? 1 : 0
        }
    }
    return kept
}

/// Generate a keep/drop mask for `n` IDs against up to `F` composed filters.
/// `filters` and `modes` must have length ≥ F; nil pointers are ignored.
///
/// ⚠️ **Allocation Note**: This function allocates a temporary `UInt8` mask buffer of size `n`.
/// For zero-allocation operation, use `idFilterPassN` directly in a manual loop.
///
/// - Returns: number of kept IDs (mask[i] = 1 → keep).
@inlinable
public func idFilterMaskN(
    filters: [UnsafePointer<UInt64>?],
    modes: [FilterMode],
    filterCount F: Int,
    ids: UnsafePointer<Int64>?,
    count n: Int,
    capacity: Int,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    precondition(F >= 0 && F <= 5, "Up to 5 filters supported (4 allow + 1 deny via mode)")

    // Map to at most 4 allows + optional deny.
    var allowPtrs: [UnsafePointer<UInt64>?] = [nil, nil, nil, nil]
    var denyPtr: UnsafePointer<UInt64>?

    var aIdx = 0
    for f in 0..<F {
        let ptr = filters[f]
        switch modes[f] {
        case .allowlist:
            if aIdx < 4 { allowPtrs[aIdx] = ptr; aIdx &+= 1 }
        case .denylist:
            // Capture a single deny pointer (if multiple deny filters are provided, OR semantics are equivalent to one combined deny via word-wise OR at build time).
            if denyPtr == nil { denyPtr = ptr }
        }
    }

    var kept = 0
    if let idsPtr = ids {
        for i in 0..<n {
            let id = idsPtr[i]
            let pass = idFilterPassN(
                allow0: allowPtrs[0],
                allow1: allowPtrs[1],
                allow2: allowPtrs[2],
                allow3: allowPtrs[3],
                deny: denyPtr,
                id: id,
                capacity: capacity
            )
            maskOut[i] = pass ? 1 : 0
            kept &+= pass ? 1 : 0
        }
    } else {
        for i in 0..<n {
            let id = Int64(i)
            let pass = idFilterPassN(
                allow0: allowPtrs[0],
                allow1: allowPtrs[1],
                allow2: allowPtrs[2],
                allow3: allowPtrs[3],
                deny: denyPtr,
                id: id,
                capacity: capacity
            )
            maskOut[i] = pass ? 1 : 0
            kept &+= pass ? 1 : 0
        }
    }
    return kept
}

// MARK: - Batch Compact (stable)

/// Filter and compact ID/score arrays (stable; preserves order among kept IDs).
/// Writes compacted results to `idsOut` (and `scoresOut` if provided).
/// - Returns: count of kept elements.
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
        if idFilterPass(bitset: bitset, id: id, capacity: capacity, mode: mode) {
            idsOut[writeIdx] = id
            if let sIn = scoresIn, let sOut = scoresOut {
                sOut[writeIdx] = sIn[i]
            }
            writeIdx &+= 1
        }
    }
    return writeIdx
}
// Spec: stable two-finger compaction; scores kept aligned.  [oai_citation:6‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)  [oai_citation:7‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)

/// Filter+compact with composed multi-filter using a precomputed mask.
/// Generates mask first to amortize checks, then performs stable copy.
///
/// ⚠️ **Allocation Note**: This function allocates a temporary `UInt8` mask buffer of size `n`.
///
/// - Returns: count of kept elements.
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
    // Build mask first (⚠️ allocates temporary buffer)
    var mask = [UInt8](repeating: 0, count: n)
    let kept = idFilterMaskN(
        filters: filters, modes: modes, filterCount: F,
        ids: idsIn, count: n, capacity: capacity,
        maskOut: &mask
    )

    // Stable compaction using mask
    var writeIdx = 0
    for i in 0..<n {
        if mask[i] == 1 {
            idsOut[writeIdx] = idsIn[i]
            if let sIn = scoresIn, let sOut = scoresOut {
                sOut[writeIdx] = sIn[i]
            }
            writeIdx &+= 1
        }
    }
    assert(writeIdx == kept)
    return writeIdx
}
// Spec: mask + stable compact variant for composed filters.  [oai_citation:8‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)  [oai_citation:9‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)

// MARK: - Per-query Filter Overlay

/// Per-query overlay that composes global allow/deny with query allow/deny.
/// Typical usage: 1-2 global allows (tenant/shard), global deny (tombstones),
/// and optional query allow/deny (user filters).
public struct IDFilterOverlay {
    public let globalAllows: [UnsafePointer<UInt64>?]   // exactly 4 slots
    public let globalDeny: UnsafePointer<UInt64>?
    public let queryAllow: UnsafePointer<UInt64>?
    public let queryDeny: UnsafePointer<UInt64>?
    public let capacity: Int

    @inlinable
    public init(
        globalAllows: [UnsafePointer<UInt64>?] = [nil, nil, nil, nil],
        globalDeny: UnsafePointer<UInt64>? = nil,
        queryAllow: UnsafePointer<UInt64>? = nil,
        queryDeny: UnsafePointer<UInt64>? = nil,
        capacity: Int
    ) {
        precondition(globalAllows.count <= 4, "Maximum 4 global allowlists supported")

        // Store exactly 4 entries for `globalAllows` to keep tight loops simple.
        if globalAllows.count >= 4 {
            self.globalAllows = Array(globalAllows.prefix(4))
        } else {
            self.globalAllows = globalAllows + Array(repeating: nil, count: 4 - globalAllows.count)
        }
        self.globalDeny  = globalDeny
        self.queryAllow  = queryAllow
        self.queryDeny   = queryDeny
        self.capacity    = capacity
    }

    /// Test whether `id` passes all overlay filters.
    /// keep = (∧ globalAllows) ∧ (queryAllow? then it must pass) ∧ ¬globalDeny ∧ ¬queryDeny
    @inline(__always)
    public func test(id: Int64) -> Bool {
        guard id >= 0 && id < capacity else { return false }

        let wordIdx = Int(id >> 6)
        let bitIdx  = Int(id & 63)
        let mask    = (1 as UInt64) << bitIdx

        // Global allowlists (intersection)
        for allow in globalAllows {
            if let bits = allow, (bits[wordIdx] & mask) == 0 { return false }
        }

        // Query allowlist (if provided)
        if let qa = queryAllow, (qa[wordIdx] & mask) == 0 { return false }

        // Global deny
        if let gd = globalDeny, (gd[wordIdx] & mask) != 0 { return false }

        // Query deny
        if let qd = queryDeny, (qd[wordIdx] & mask) != 0 { return false }

        return true
    }
}
// Spec: overlay composition and test() semantics.  [oai_citation:10‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)

// MARK: - Telemetry

/// Per-operation statistics (optional; populate at call sites if needed).
public struct IDFilterTelemetry {
    public let idsProcessed: Int
    public let idsKept: Int
    public let idsDropped: Int
    public let filtersActive: Int
    public let composedChecks: Int
    public let executionTimeNanos: UInt64

    @inlinable public var selectivityPercent: Double {
        guard idsProcessed > 0 else { return 0 }
        return (Double(idsKept) / Double(idsProcessed)) * 100.0
    }

    @inlinable public var nanosPerID: Double {
        guard idsProcessed > 0 else { return 0 }
        return Double(executionTimeNanos) / Double(idsProcessed)
    }
}
// Spec: lightweight telemetry fields for profiling.  [oai_citation:11‡08_id_filter.md](file-service://file-TWS7TvB6rB2nnbH7j3o5hw)

// MARK: - Convenience Extensions

extension IDFilterBitset {
    /// Convenience: compact Swift arrays via `idFilterCompact`.
    @inlinable
    public func compact(
        ids: UnsafePointer<Int64>,
        scores: UnsafePointer<Float>?,
        count n: Int,
        mode: FilterMode,
        idsOut: UnsafeMutablePointer<Int64>,
        scoresOut: UnsafeMutablePointer<Float>?
    ) -> Int {
        idFilterCompact(
            bitset: readOnly,
            idsIn: ids,
            scoresIn: scores,
            count: n,
            capacity: capacity,
            mode: mode,
            idsOut: idsOut,
            scoresOut: scoresOut
        )
    }

    /// Convenience: test ID with mode using instance method.
    @inline(__always)
    public func passes(id: Int64, mode: FilterMode) -> Bool {
        idFilterPass(bitset: readOnly, id: id, capacity: capacity, mode: mode)
    }
}
