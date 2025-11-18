// Sources/VectorIndex/Operations/Dedup/CandidateDedup.swift
//
// Kernel #32: Multi-List Candidate Deduplication
// - Compatible with Range Query (#07) and Candidate Reservoir (#39)
// - Implements DenseEpoch, SparsePaged, and FixedBitset strategies
// - Swift 6 Sendable compliance and inlining/visibility annotations
//
// Thread-safety: single-writer per instance (query-local). See docs below.
//
// Spec reference: Kernel Specification #32 (Multi-List Candidate Deduplication)
// (Implements: visitedInit, visitedReset, visitedTestAndSet, visitedMaskAndMark,
//  dedupInPlace, visitedFree, plus convenience APIs)

import Foundation

// MARK: - Integration Protocol (used by kernels #07, #39)

/// Protocol used by existing kernels (e.g., Range Query #07) for dedup.
/// Single-writer per instance is assumed unless noted otherwise.
public protocol VisitedSet: Sendable {
    /// Returns true if this is the first time seeing `id` in the current query.
    @discardableResult
    func testAndSet(id: Int64) -> Bool
}

// MARK: - Public Options & Stats

/// Storage implementation strategy
public enum VisitedMode: Sendable {
    case denseEpoch      // UInt32 stamp-per-ID with epoch bump reset
    case sparsePaged     // On-demand bitset pages (2^pageBits IDs/page)
    case fixedBitset     // Full bitset for [0, idCapacity)
}

/// Concurrency strategy (documented; DefaultVisitedSet is single-writer)
public enum ConcurrencyMode: Sendable {
    case singleWriter           // Fastest, no synchronization
    case shardedMultiWriter     // App-level sharding by ID; one set per shard
    case atomicMultiWriter      // (Not implemented): would require atomics
}

/// Configuration for visited set behavior
@frozen
public struct VisitedOpts: Sendable {
    public let mode: VisitedMode
    public let concurrency: ConcurrencyMode
    public let pageBits: Int          // For SparsePaged (IDs per page = 1 << pageBits)
    public let epochBits: Int         // For DenseEpoch wrap testing (8..32)
    public let enableTelemetry: Bool

    public init(
        mode: VisitedMode = .denseEpoch,
        concurrency: ConcurrencyMode = .singleWriter,
        pageBits: Int = 15,
        epochBits: Int = 32,
        enableTelemetry: Bool = false
    ) {
        self.mode = mode
        self.concurrency = concurrency
        self.pageBits = pageBits
        self.epochBits = epochBits
        self.enableTelemetry = enableTelemetry
    }

    public static let `default` = VisitedOpts()
}

/// Per-query statistics (lightweight)
@frozen
public struct VisitedStats: Sendable {
    public let totalChecks: Int64
    public let uniqueCount: Int64
    public let duplicateCount: Int64
    public let pagesAllocated: Int   // SparsePaged only (this query)
    public let epochValue: UInt32    // DenseEpoch current epoch
}

/// Optional telemetry for perf tracking (#46)
@frozen
public struct VisitedTelemetry: Sendable {
    public let mode: VisitedMode
    public let totalChecks: Int64
    public let uniqueCount: Int64
    public let duplicateCount: Int64
    public let pagesAllocated: Int       // SparsePaged only
    public let pagesCleared: Int         // SparsePaged only
    public let epochValue: UInt32        // DenseEpoch only
    public let epochWraps: Int           // DenseEpoch only
    public let atomicConflicts: Int      // Atomic mode only (not used)
    public let checkTimeNanos: UInt64

    public var deduplicationRate: Double {
        let total = uniqueCount + duplicateCount
        return total > 0 ? Double(duplicateCount) / Double(total) : 0.0
    }
    public var avgCheckLatencyNs: Double {
        totalChecks > 0 ? Double(checkTimeNanos) / Double(totalChecks) : 0.0
    }
}

// MARK: - DefaultVisitedSet (drop-in replacement)

/// Default implementation of Kernel #32.
/// Single-writer per instance; reuse across queries with `resetForNewQuery()`.
///
/// Thread-safety:
/// - Intended for single writer per query thread (fast path).
/// - For parallel intra-query, shard the ID space and use one instance per shard.
/// - Atomic multi-writer not implemented here (would require atomics).
public final class DefaultVisitedSet: @unchecked Sendable, VisitedSet {

    // MARK: Public immutable configuration

    public let mode: VisitedMode
    public let concurrency: ConcurrencyMode
    public let opts: VisitedOpts
    public let idCapacity: Int64

    // MARK: Per-query counters (publicly exposed via `getStats()`)

    @usableFromInline internal var totalChecks: Int64 = 0
    @usableFromInline internal var uniqueCount: Int64 = 0
    @usableFromInline internal var duplicateCount: Int64 = 0
    @usableFromInline internal var pagesAllocatedThisQuery: Int = 0
    @usableFromInline internal var pagesClearedThisQuery: Int = 0

    // MARK: Epoch / query sequencing (used by all modes)
    // We use a shared "queryEpoch" that always increments at reset.
    @usableFromInline internal var queryEpoch: UInt32 = 1
    @usableFromInline internal let epochMask: UInt32 // (1 << epochBits) - 1
    @usableFromInline internal var epochWraps: Int = 0

    // MARK: DenseEpoch storage
    @usableFromInline internal var stamps: UnsafeMutablePointer<UInt32>? // [idCapacity]
    // stamps[i] == queryEpoch means "visited this query"

    // MARK: SparsePaged storage
    @usableFromInline internal let pageBits: Int
    @usableFromInline internal let pageMask: Int64
    @usableFromInline internal static let wordsPerPage: Int = 512 // 32,768 / 64
    @usableFromInline internal struct Page {
        var bits: UnsafeMutablePointer<UInt64>  // 4 KiB bitset
        var lastTouchedEpoch: UInt32            // dedup touched-pages per-query
    }
    @usableFromInline internal var pageTable: [Int64: Page] = [:] // pageID -> Page
    @usableFromInline internal var touchedPages: [Int64] = []     // pageIDs touched this query

    // MARK: FixedBitset storage
    @usableFromInline internal var bitWords: UnsafeMutablePointer<UInt64>?
    @usableFromInline internal var wordCount: Int = 0
    @usableFromInline internal var touchedWords: UnsafeMutablePointer<Int>?
    @usableFromInline internal var touchedCount: Int = 0
    @usableFromInline internal var touchedCapacity: Int = 0

    // MARK: Init / Deinit

    public init(idCapacity: Int64, opts: VisitedOpts = .default) {
        precondition(idCapacity > 0, "idCapacity must be > 0")

        self.idCapacity = idCapacity
        self.mode = opts.mode
        self.concurrency = opts.concurrency
        self.opts = opts

        // Epoch wrap control (limit to <= 32 bits)
        let eb = max(8, min(32, opts.epochBits))
        self.epochMask = eb == 32 ? 0xFFFF_FFFF : (1 << eb) - 1

        // SparsePaged derivations (safe defaults even if not used)
        self.pageBits = max(6, min(24, opts.pageBits)) // clamp [6,24]
        self.pageMask = Int64((1 << self.pageBits) - 1)

        switch self.mode {
        case .denseEpoch:
            // Allocate stamp array [idCapacity], initialize to 0
            let cnt = Int(idCapacity)
            let p = UnsafeMutablePointer<UInt32>.allocate(capacity: cnt)
            // Initialize once at construction (safe)
            p.initialize(repeating: 0, count: cnt)
            self.stamps = p

        case .sparsePaged:
            // Nothing to allocate yet; pages are on-demand
            self.stamps = nil
            // touchedPages reserve a small capacity (avoid realloc in hot path)
            self.touchedPages.reserveCapacity(16)

        case .fixedBitset:
            // Allocate bitset words and touched-word ring
            let wCount = Int((idCapacity + 63) >> 6) // ceil(capacity / 64)
            self.wordCount = wCount
            let p = UnsafeMutablePointer<UInt64>.allocate(capacity: wCount)
            p.initialize(repeating: 0, count: wCount)
            self.bitWords = p

            // Touched ring buffer (heuristic capacity)
            // Keep it reasonably large to remain sparse; grow if needed.
            let cap = max(16_384, min(wCount, 1_000_000))
            self.touchedCapacity = cap
            let ring = UnsafeMutablePointer<Int>.allocate(capacity: cap)
            self.touchedWords = ring
        }
    }

    deinit {
        freeAll()
    }

    // MARK: Memory free (also exposed via visitedFree)

    @usableFromInline internal func freeAll() {
        if let s = stamps {
            s.deallocate()
            stamps = nil
        }
        if let bw = bitWords {
            bw.deallocate()
            bitWords = nil
        }
        if let tw = touchedWords {
            tw.deallocate()
            touchedWords = nil
        }
        if !pageTable.isEmpty {
            for (_, page) in pageTable {
                page.bits.deallocate()
            }
            pageTable.removeAll(keepingCapacity: false)
        }
        touchedPages.removeAll(keepingCapacity: false)
    }

    // MARK: - Public API (VisitedSet protocol)

    /// Test-and-set (first-seen semantics)
    /// - Returns: true if first time seen this query; false if duplicate.
    @discardableResult
    public func testAndSet(id: Int64) -> Bool {
        switch mode {
        case .denseEpoch:   return _testAndSet_dense(id)
        case .sparsePaged:  return _testAndSet_sparse(id)
        case .fixedBitset:  return _testAndSet_bitset(id)
        }
    }

    // MARK: - Query lifecycle

    /// Reset visited state for a new query (O(1) for DenseEpoch, O(touched) otherwise).
    public func resetForNewQuery() {
        // Begin new query epoch (shared across modes)
        let next = (queryEpoch & epochMask) &+ 1
        queryEpoch = next
        pagesAllocatedThisQuery = 0
        pagesClearedThisQuery = 0
        totalChecks = 0
        uniqueCount = 0
        duplicateCount = 0

        // DenseEpoch epoch wrap handling
        if mode == .denseEpoch, queryEpoch == 0 {
            // Full clear required (rare)
            if let s = stamps {
                let n = Int(idCapacity)
                // Reinitialize by simple loop (safe for already-initialized memory)
                for i in 0..<n { s[i] = 0 }
            }
            queryEpoch = 1
            epochWraps &+= 1
        }

        // SparsePaged / FixedBitset do NOT need full clear now (cleared at end of previous query)
        // We clear pages/words during the transition to this new query below.
        if mode == .sparsePaged {
            // Clear only the pages touched in the previous query.
            // Note: touchedPages contains unique pageIDs (due to per-page epoch tagging).
            let count = touchedPages.count
            if !touchedPages.isEmpty {
                for pid in touchedPages {
                    if let page = pageTable[pid] {
                        let ptr = page.bits
                        // Clear page words (4KiB)
                        for i in 0..<DefaultVisitedSet.wordsPerPage { ptr[i] = 0 }
                        // keep page.lastTouchedEpoch as-is; will be updated on first touch
                    }
                }
                pagesClearedThisQuery = count
                touchedPages.removeAll(keepingCapacity: true)
            }
        } else if mode == .fixedBitset {
            // Clear only touched words if sparse, else full clear is faster.
            let tc = touchedCount
            if tc > 0, let bw = bitWords {
                if tc < wordCount / 4, let tw = touchedWords {
                    for i in 0..<tc {
                        bw[tw[i]] = 0
                    }
                } else {
                    // Dense: full clear
                    for i in 0..<wordCount { bw[i] = 0 }
                }
            }
            touchedCount = 0
        }

        // SparsePaged epoch wrap handling (rare): if epoch wrapped to 0,
        // ensure page.lastTouchedEpoch == 0 for all pages to avoid stale equality.
        if mode == .sparsePaged, queryEpoch == 0 {
            for (pid, var page) in pageTable {
                page.lastTouchedEpoch = 0
                pageTable[pid] = page
                let ptr = page.bits
                for i in 0..<DefaultVisitedSet.wordsPerPage { ptr[i] = 0 }
            }
            queryEpoch = 1
            epochWraps &+= 1
        }
    }

    // MARK: - Convenience APIs (spec)

    /// High-level deduplication for Swift arrays (preserves order)
    @discardableResult
    public func dedup(ids: inout [Int64], scores: inout [Float]?) -> Int {
        let count = ids.count
        let newCount = ids.withUnsafeMutableBufferPointer { idBuf -> Int in
            if let scoresPtr = scores?.withUnsafeMutableBufferPointer({ $0.baseAddress }) {
                return _dedupInPlace(ids: idBuf.baseAddress!, scores: scoresPtr, count: count)
            } else {
                return _dedupInPlace(ids: idBuf.baseAddress!, scores: nil, count: count)
            }
        }
        if var s = scores { s.removeSubrange(newCount..<s.count); scores = s }
        ids.removeSubrange(newCount..<ids.count)
        return newCount
    }

    /// Filter candidates through visited set (first-seen only)
    public func filterUnique(candidates: [Int64]) -> [Int64] {
        var out: [Int64] = []
        out.reserveCapacity(candidates.count)
        for id in candidates {
            if testAndSet(id: id) { out.append(id) }
        }
        return out
    }

    /// Read-only membership test (no side-effects)
    public func contains(_ id: Int64) -> Bool {
        switch mode {
        case .denseEpoch:
            guard id >= 0 && id < idCapacity, let s = stamps else { return false }
            return s[Int(id)] == queryEpoch
        case .sparsePaged:
            let pageID = id >> Int64(pageBits)
            let inPage = id & pageMask
            let w = Int(inPage >> 6), b = Int(inPage & 63)
            guard let page = pageTable[pageID] else { return false }
            let word = page.bits[w]
            return (word & (1 << b)) != 0
        case .fixedBitset:
            guard id >= 0 && id < idCapacity, let bw = bitWords else { return false }
            let w = _wordIndex(id), b = _bitIndex(id)
            return (bw[w] & (1 << b)) != 0
        }
    }

    /// Snapshot of current per-query statistics
    public func getStats() -> VisitedStats {
        VisitedStats(
            totalChecks: totalChecks,
            uniqueCount: uniqueCount,
            duplicateCount: duplicateCount,
            pagesAllocated: pagesAllocatedThisQuery,
            epochValue: queryEpoch
        )
    }

    /// Build telemetry record for this query
    public func getTelemetry(elapsedNanos: UInt64) -> VisitedTelemetry {
        VisitedTelemetry(
            mode: mode,
            totalChecks: totalChecks,
            uniqueCount: uniqueCount,
            duplicateCount: duplicateCount,
            pagesAllocated: pagesAllocatedThisQuery,
            pagesCleared: pagesClearedThisQuery,
            epochValue: queryEpoch,
            epochWraps: epochWraps,
            atomicConflicts: 0,
            checkTimeNanos: elapsedNanos
        )
    }

    // MARK: - Internal fast paths (called from inlinable wrappers)

    /// DenseEpoch test-and-set
    @usableFromInline
    internal func _testAndSet_dense(_ id: Int64) -> Bool {
        assert(id >= 0 && id < idCapacity, "ID out of bounds")
        totalChecks &+= 1
        guard let s = stamps else { return false }
        let idx = Int(id)
        let epoch = queryEpoch
        let stamp = s[idx]
        if stamp != epoch {
            s[idx] = epoch
            uniqueCount &+= 1
            return true
        } else {
            duplicateCount &+= 1
            return false
        }
    }

    /// SparsePaged test-and-set
    @usableFromInline
    internal func _testAndSet_sparse(_ id: Int64) -> Bool {
        assert(id >= 0, "ID must be non-negative")
        totalChecks &+= 1

        let pid = id >> Int64(pageBits)
        var page = pageTable[pid]

        // Allocate on demand
        if page == nil {
            let p = UnsafeMutablePointer<UInt64>.allocate(capacity: DefaultVisitedSet.wordsPerPage)
            p.initialize(repeating: 0, count: DefaultVisitedSet.wordsPerPage)
            page = Page(bits: p, lastTouchedEpoch: 0)
            pageTable[pid] = page
            pagesAllocatedThisQuery &+= 1
        }

        // If first touch this query, record for later clear
        if page!.lastTouchedEpoch != queryEpoch {
            page!.lastTouchedEpoch = queryEpoch
            pageTable[pid] = page
            touchedPages.append(pid)
        }

        // Bit test
        let inPage = id & pageMask
        let w = Int(inPage >> 6)
        let b = Int(inPage & 63)
        let mask: UInt64 = 1 &<< b

        let word = page!.bits[w]
        if (word & mask) == 0 {
            page!.bits[w] = word | mask
            uniqueCount &+= 1
            return true
        } else {
            duplicateCount &+= 1
            return false
        }
    }

    /// FixedBitset test-and-set
    @usableFromInline
    internal func _testAndSet_bitset(_ id: Int64) -> Bool {
        assert(id >= 0 && id < idCapacity, "ID out of bounds")
        totalChecks &+= 1

        guard let bw = bitWords else { return false }
        let w = _wordIndex(id)
        let b = _bitIndex(id)
        let mask: UInt64 = 1 &<< b

        let word = bw[w]
        if (word & mask) == 0 {
            bw[w] = word | mask
            // Track newly touched word (if word was previously 0)
            if word == 0, touchedCount < touchedCapacity, let tw = touchedWords {
                tw[touchedCount] = w
                touchedCount &+= 1
            }
            uniqueCount &+= 1
            return true
        } else {
            duplicateCount &+= 1
            return false
        }
    }

    /// Batch mask-and-mark (fast paths)
    @usableFromInline
    internal func _maskAndMark(
        ids: UnsafePointer<Int64>,
        count: Int,
        maskOut: UnsafeMutablePointer<UInt8>
    ) -> Int {
        switch mode {
        case .denseEpoch:
            return _maskAndMark_dense(ids: ids, count: count, maskOut: maskOut)
        case .fixedBitset:
            return _maskAndMark_bitset(ids: ids, count: count, maskOut: maskOut)
        case .sparsePaged:
            // Fallback to scalar for sparse (hash lookup dominates anyway)
            var uniques = 0
            for i in 0..<count {
                if _testAndSet_sparse(ids[i]) {
                    maskOut[i] = 1; uniques &+= 1
                } else {
                    maskOut[i] = 0
                }
            }
            return uniques
        }
    }

    @usableFromInline
    internal func _maskAndMark_dense(
        ids: UnsafePointer<Int64>,
        count: Int,
        maskOut: UnsafeMutablePointer<UInt8>
    ) -> Int {
        var uniqueLocal = 0
        let epoch = queryEpoch
        guard let s = stamps else { return 0 }

        // Unrolled processing in blocks of 4 (branch-lean)
        let blk = (count / 4) * 4
        var i = 0
        while i < blk {
            let i0 = Int(ids[i + 0])
            let i1 = Int(ids[i + 1])
            let i2 = Int(ids[i + 2])
            let i3 = Int(ids[i + 3])

            let st0 = s[i0], st1 = s[i1], st2 = s[i2], st3 = s[i3]

            let n0: UInt8 = st0 != epoch ? 1 : 0
            let n1: UInt8 = st1 != epoch ? 1 : 0
            let n2: UInt8 = st2 != epoch ? 1 : 0
            let n3: UInt8 = st3 != epoch ? 1 : 0

            if n0 == 1 { s[i0] = epoch; uniqueLocal &+= 1 }
            if n1 == 1 { s[i1] = epoch; uniqueLocal &+= 1 }
            if n2 == 1 { s[i2] = epoch; uniqueLocal &+= 1 }
            if n3 == 1 { s[i3] = epoch; uniqueLocal &+= 1 }

            maskOut[i + 0] = n0
            maskOut[i + 1] = n1
            maskOut[i + 2] = n2
            maskOut[i + 3] = n3

            i &+= 4
        }
        while i < count {
            let idx = Int(ids[i])
            let st = s[idx]
            if st != epoch {
                s[idx] = epoch
                maskOut[i] = 1
                uniqueLocal &+= 1
            } else {
                maskOut[i] = 0
            }
            i &+= 1
        }

        totalChecks &+= Int64(count)
        uniqueCount &+= Int64(uniqueLocal)
        duplicateCount &+= Int64(count - uniqueLocal)
        return uniqueLocal
    }

    @usableFromInline
    internal func _maskAndMark_bitset(
        ids: UnsafePointer<Int64>,
        count: Int,
        maskOut: UnsafeMutablePointer<UInt8>
    ) -> Int {
        guard let bw = bitWords else { return 0 }
        var uniques = 0

        var i = 0
        let blk = (count / 4) * 4
        while i < blk {
            // Manual unroll 4x
            let id0 = ids[i+0], id1 = ids[i+1], id2 = ids[i+2], id3 = ids[i+3]

            let w0 = _wordIndex(id0), b0 = _bitIndex(id0)
            let w1 = _wordIndex(id1), b1 = _bitIndex(id1)
            let w2 = _wordIndex(id2), b2 = _bitIndex(id2)
            let w3 = _wordIndex(id3), b3 = _bitIndex(id3)

            let m0: UInt64 = 1 &<< b0
            let m1: UInt64 = 1 &<< b1
            let m2: UInt64 = 1 &<< b2
            let m3: UInt64 = 1 &<< b3

            let x0 = bw[w0], x1 = bw[w1], x2 = bw[w2], x3 = bw[w3]

            let n0 = (x0 & m0) == 0
            let n1 = (x1 & m1) == 0
            let n2 = (x2 & m2) == 0
            let n3 = (x3 & m3) == 0

            if n0 { bw[w0] = x0 | m0; if x0 == 0 { _trackTouchedWord(w0) }; uniques &+= 1; maskOut[i+0] = 1 } else { maskOut[i+0] = 0 }
            if n1 { bw[w1] = x1 | m1; if x1 == 0 { _trackTouchedWord(w1) }; uniques &+= 1; maskOut[i+1] = 1 } else { maskOut[i+1] = 0 }
            if n2 { bw[w2] = x2 | m2; if x2 == 0 { _trackTouchedWord(w2) }; uniques &+= 1; maskOut[i+2] = 1 } else { maskOut[i+2] = 0 }
            if n3 { bw[w3] = x3 | m3; if x3 == 0 { _trackTouchedWord(w3) }; uniques &+= 1; maskOut[i+3] = 1 } else { maskOut[i+3] = 0 }

            i &+= 4
        }
        while i < count {
            let id = ids[i]
            let w = _wordIndex(id), b = _bitIndex(id)
            let m: UInt64 = 1 &<< b
            let x = bw[w]
            if (x & m) == 0 {
                bw[w] = x | m
                if x == 0 { _trackTouchedWord(w) }
                maskOut[i] = 1
                uniques &+= 1
            } else {
                maskOut[i] = 0
            }
            i &+= 1
        }

        totalChecks &+= Int64(count)
        uniqueCount &+= Int64(uniques)
        duplicateCount &+= Int64(count - uniques)
        return uniques
    }

    @usableFromInline
    internal func _trackTouchedWord(_ w: Int) {
        if touchedCount < touchedCapacity, let tw = touchedWords {
            tw[touchedCount] = w
            touchedCount &+= 1
        }
    }

    /// In-place compaction (preserve first-seen order)
    @usableFromInline
    internal func _dedupInPlace(
        ids: UnsafeMutablePointer<Int64>,
        scores: UnsafeMutablePointer<Float>?,
        count: Int
    ) -> Int {
        var writeIdx = 0
        for readIdx in 0..<count {
            let id = ids[readIdx]
            if testAndSet(id: id) {
                if writeIdx != readIdx {
                    ids[writeIdx] = id
                    if let s = scores { s[writeIdx] = s[readIdx] }
                }
                writeIdx &+= 1
            }
        }
        return writeIdx
    }
}

// MARK: - Small helpers

@usableFromInline
internal func _wordIndex(_ id: Int64) -> Int { Int(id >> 6) }

@usableFromInline
internal func _bitIndex(_ id: Int64) -> Int { Int(id & 63) }

// MARK: - Top-level free functions (spec API)

/**
 Initialize a visited set for a given ID capacity.
 - Complexity: O(idCapacity) for dense modes; O(1) for sparse.
 - Thread-safety: Create once per worker thread and reuse.
 */
@inlinable
public func visitedInit(
    idCapacity: Int64,
    opts: VisitedOpts = .default
) -> DefaultVisitedSet {
    DefaultVisitedSet(idCapacity: idCapacity, opts: opts)
}

/**
 Reset visited set for a new query.
 - Complexity: O(1) for DenseEpoch, O(touched) for others.
 - Thread-safety: Must be called by the thread that uses the set.
 */
@inlinable
public func visitedReset(_ vs: DefaultVisitedSet) {
    vs.resetForNewQuery()
}

/**
 Test-and-set (first-seen semantics).
 - Returns: true if first occurrence (should process), false if duplicate (skip).
 - Complexity: O(1) average.
 */
@inline(__always)
public func visitedTestAndSet(
    _ vs: any VisitedSet,
    _ id: Int64
) -> Bool {
    vs.testAndSet(id: id)
}

/**
 Batch mask-and-mark.
 Writes maskOut[i] = 1 if ids[i] is new in this query; 0 if duplicate.
 - Returns: count of new IDs.
 - Complexity: O(n).
 */
@inlinable
public func visitedMaskAndMark(
    _ vs: any VisitedSet,
    ids: UnsafePointer<Int64>,
    count: Int,
    maskOut: UnsafeMutablePointer<UInt8>
) -> Int {
    if let impl = vs as? DefaultVisitedSet {
        return impl._maskAndMark(ids: ids, count: count, maskOut: maskOut)
    } else {
        var uniques = 0
        for i in 0..<count {
            if vs.testAndSet(id: ids[i]) {
                maskOut[i] = 1; uniques &+= 1
            } else {
                maskOut[i] = 0
            }
        }
        return uniques
    }
}

/**
 In-place deduplication of `ids` (and optional parallel `scores`), preserving order.
 - Returns: New count after compaction.
 - Complexity: O(n).
 */
@inlinable
public func dedupInPlace(
    _ vs: any VisitedSet,
    ids: UnsafeMutablePointer<Int64>,
    scores: UnsafeMutablePointer<Float>?,
    count: Int
) -> Int {
    if let impl = vs as? DefaultVisitedSet {
        return impl._dedupInPlace(ids: ids, scores: scores, count: count)
    } else {
        var writeIdx = 0
        for readIdx in 0..<count {
            let id = ids[readIdx]
            if vs.testAndSet(id: id) {
                if writeIdx != readIdx {
                    ids[writeIdx] = id
                    if let s = scores { s[writeIdx] = s[readIdx] }
                }
                writeIdx &+= 1
            }
        }
        return writeIdx
    }
}

/**
 Free internal resources. Normally unnecessary if instances are deallocated.
 */
public func visitedFree(_ vs: DefaultVisitedSet) {
    vs.freeAll()
}

// MARK: - Integration Notes (Documentation)
//
// • Drop-in replacement:
//   - Use the `VisitedSet` protocol in kernels; construct with `visitedInit(...)` which
//     returns `DefaultVisitedSet` conforming to `VisitedSet`.
//   - Range Query (#07) that previously used `DefaultVisitedSet` can directly use this type.
//
// • Threading model:
//   - Single-writer per instance (recommended). Reuse one instance per worker thread.
//   - For intra-query parallelism, shard by ID (e.g., modulo) and use one instance per shard.
//   - Atomic multi-writer is not implemented in this kernel; if needed, isolate per-shard.
//
// • Determinism:
//   - First-seen semantics with stable scan order; `dedupInPlace` preserves order.
//
// • Hot path behavior:
//   - No heap allocations in DenseEpoch/FixedBitset checks.
//   - SparsePaged allocates new 4KiB pages lazily and reuses them across queries.
//   - Reset is O(1) for DenseEpoch (epoch bump) and O(touched) for others.
//
// • Visibility & inlining:
//   - Public hot wrappers are @inlinable or @inline(__always) where appropriate.
//   - Internal properties/methods used by inlinable wrappers are @usableFromInline.
//
// • SIMD / loads:
//   - Batch fast paths use manual unrolling; no reliance on non-public SIMD initializers.
//   - Memory loads/stores done via pointer indexing; no internal SIMD initializers.
//
// • Deprecations:
//   - No use of deprecated pointer copying; we avoid `.assign(from:count:)` entirely.
//   - Arithmetic uses standard `+ - *` operators.
//
// • Telemetry (#46):
//   - `getStats()` provides lightweight counters.
//   - `getTelemetry(elapsedNanos:)` fills the extended structure for external recorder.
