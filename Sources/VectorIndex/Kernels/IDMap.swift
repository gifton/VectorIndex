import Foundation

// Kernel #50: ID Remapping (External UInt64 ↔ Internal Dense Int64)

public enum HashTableImpl: Sendable { case swissTable, robinHood, linearProbing }
public enum IDMapConcurrency: Sendable { case singleWriter, rwLock }

internal struct IDMapOpts: Sendable {
    public var allowReplace: Bool = false
    public var hashTableImpl: HashTableImpl = .swissTable
    public var capacityHint: Int = 1_000
    public var maxLoadFactor: Double = 0.875
    public var concurrency: IDMapConcurrency = .singleWriter
    public var enableBloom: Bool = false
    public var enableTelemetry: Bool = false
    public static var `default`: IDMapOpts { IDMapOpts() }
}

public struct IDMapStats {
    public let count: Int64
    public let capacity: Int64
    public let hashTableSize: Int
    public let loadFactor: Double
    public let avgProbeLength: Double
    public let maxProbeLength: Int
    public let tombstoneCount: Int64
}

// IDMapError removed - migrated to VectorIndexError
// All throw sites now use ErrorBuilder with appropriate IndexErrorKind

public final class TombstoneSet {
    private var bits: [UInt64]
    public init(capacity: Int) { bits = [UInt64](repeating: 0, count: (capacity + 63) >> 6) }
    @inline(__always) private func iw(_ i: Int64) -> (Int, UInt64) { let idx = Int(i >> 6); return (idx, 1 &<< UInt64(i & 63)) }
    public func set(_ i: Int64) { let (w, m) = iw(i); if w < bits.count { bits[w] |= m } }
    public func isSet(_ i: Int64) -> Bool { let (w, m) = iw(i); return (w < bits.count) && ((bits[w] & m) != 0) }
}

// ---- Hash utilities and table implementations (placed before Impl for visibility)
@inline(__always) private func mix64(_ x: UInt64) -> UInt64 {
    var z = x &+ 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
}
@inline(__always) private func hashH1(_ key: UInt64, _ bucketCount: Int) -> Int {
    let golden: UInt64 = 0x9E3779B97F4A7C15
    let hash = key &* golden
    let shift = 64 - bucketCount.trailingZeroBitCount
    return Int((hash >> shift) & UInt64(bucketCount - 1))
}
@inline(__always) private func hashH2(_ key: UInt64) -> UInt8 { UInt8((key ^ (key >> 7)) & 0x7F) }
@usableFromInline @inline(__always) func nextPow2(_ x: Int) -> Int {
    var v = max(1, x)
    v -= 1; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16
    #if arch(x86_64) || arch(arm64)
    v |= v >> 32
    #endif
    return v + 1
}

private enum HashTable {
    case swiss(SwissTable), robin(RobinHoodTable), linear(LinearProbingTable)
    static func allocate(buckets: Int, impl: HashTableImpl) -> HashTable {
        switch impl {
        case .swissTable:
            let bc = max(16, (buckets + 15) & ~15)
            return .swiss(SwissTable(bucketCount: bc))
        case .robinHood:
            let bc = max(8, nextPow2(buckets))
            return .robin(RobinHoodTable(bucketCount: bc))
        case .linearProbing:
            let bc = max(8, nextPow2(buckets))
            return .linear(LinearProbingTable(bucketCount: bc))
        }
    }
    var bucketCount: Int { switch self { case .swiss(let t): return t.bucketCount; case .robin(let t): return t.bucketCount; case .linear(let t): return t.bucketCount } }
    var count: Int { switch self { case .swiss(let t): return t.count; case .robin(let t): return t.count; case .linear(let t): return t.count } }
    mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { switch self { case .swiss(var t): let r=t.lookup(key); self = .swiss(t); return r; case .robin(var t): let r=t.lookup(key); self = .robin(t); return r; case .linear(var t): let r=t.lookup(key); self = .linear(t); return r } }
    mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { switch self { case .swiss(var t): let p=try t.insert(key, value); self = .swiss(t); return p; case .robin(var t): let p=try t.insert(key, value); self = .robin(t); return p; case .linear(var t): let p=try t.insert(key, value); self = .linear(t); return p } }
    mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { switch self { case .swiss(var t): let p=t.updateValue(for: key, to: value); self = .swiss(t); return p; case .robin(var t): let p=t.updateValue(for: key, to: value); self = .robin(t); return p; case .linear(var t): let p=t.updateValue(for: key, to: value); self = .linear(t); return p } }
    mutating func erase(_ key: UInt64) -> (Bool, Int?) { switch self { case .swiss(var t): let r=t.erase(key); self = .swiss(t); return r; case .robin(var t): let r=t.erase(key); self = .robin(t); return r; case .linear(var t): let r=t.erase(key); self = .linear(t); return r } }
    func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { switch self { case .swiss(let t): try t.forEach(body); case .robin(let t): try t.forEach(body); case .linear(let t): try t.forEach(body) } }
}

private struct SwissTable {
    struct Entry { var externalID: UInt64 = 0; var internalID: Int64 = -1 }
    private static let groupSize = 16
    var control: [UInt8]
    var entries: [Entry]
    var bucketCount: Int
    var count: Int = 0
    init(bucketCount: Int) { self.bucketCount=bucketCount; self.control=[UInt8](repeating: 0xFF, count: bucketCount); self.entries=[Entry](repeating: Entry(), count: bucketCount) }
    @inline(__always) private func groupIndex(_ h1: Int) -> Int { h1 / SwissTable.groupSize }
    @inline(__always) private func matchesInGroup(_ base: Int, _ h2: UInt8) -> [Int] { var hits: [Int]=[]; hits.reserveCapacity(2); for i in 0..<SwissTable.groupSize where control[base+i] == h2 { hits.append(i) } ; return hits }
    @inline(__always) private func groupHasEmpty(_ base: Int) -> Bool { for i in 0..<SwissTable.groupSize where control[base+i] == 0xFF { return true } ; return false }
    mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { let h1=hashH1(key, bucketCount); let h2=hashH2(key); let groups=bucketCount/SwissTable.groupSize; var g=groupIndex(h1); var probes=0; for _ in 0..<groups { let base=g*SwissTable.groupSize; probes &+= 1; let matches = matchesInGroup(base, h2); if !matches.isEmpty { for s in matches { let idx=base+s; if entries[idx].externalID==key { return (true, entries[idx].internalID, probes) } } } ; if groupHasEmpty(base) { return (false, -1, probes) } ; g = (g+1) % groups } ; return (false, -1, probes) }
    mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { let h1=hashH1(key, bucketCount); let h2=hashH2(key); let groups=bucketCount/SwissTable.groupSize; var g=groupIndex(h1); var probes=0; for _ in 0..<groups { let base=g*SwissTable.groupSize; probes &+= 1; for s in 0..<SwissTable.groupSize { let c=control[base+s]; if c==0xFF || c==0xFE { let idx=base+s; entries[idx]=Entry(externalID: key, internalID: value); control[idx]=h2; count &+= 1; return probes } } ; g = (g+1) % groups } ; throw ErrorBuilder(.capacityExceeded, operation: "idmap_swiss_insert").message("Hash table full").info("bucket_count", "\(bucketCount)").info("count", "\(count)").build() }
    mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { let r=lookup(key); if r.0 { let h1=hashH1(key, bucketCount); let h2=hashH2(key); let groups=bucketCount/SwissTable.groupSize; var g=groupIndex(h1); for _ in 0..<groups { let base=g*SwissTable.groupSize; let matches=matchesInGroup(base, h2); if !matches.isEmpty { for s in matches { let idx=base+s; if entries[idx].externalID==key { entries[idx].internalID=value; return r.2 } } } ; if groupHasEmpty(base) { break } ; g=(g+1)%groups } } ; return nil }
    mutating func erase(_ key: UInt64) -> (Bool, Int?) { let h1=hashH1(key, bucketCount); let h2=hashH2(key); let groups=bucketCount/SwissTable.groupSize; var g=groupIndex(h1); var probes=0; for _ in 0..<groups { probes &+= 1; let base=g*SwissTable.groupSize; let matches=matchesInGroup(base, h2); for s in matches { let idx=base+s; if entries[idx].externalID==key { control[idx]=0xFE; count &-= 1; return (true, probes) } } ; if groupHasEmpty(base) { return (false, probes) } ; g=(g+1)%groups } ; return (false, probes) }
    func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { for i in 0..<bucketCount { let c=control[i]; if c != 0xFF && c != 0xFE { try body(entries[i].externalID, entries[i].internalID) } } }
}

private struct RobinHoodTable { struct Entry { var externalID: UInt64 = 0; var internalID: Int64 = -1; var dib: UInt8 = 0 }
    var entries: [Entry]; var bucketCount: Int; var count: Int=0
    init(bucketCount: Int) { self.bucketCount=bucketCount; self.entries=[Entry](repeating: Entry(), count: bucketCount)}
    mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { var curKey=key; var curVal=value; var dib: UInt8=0; var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { entries[idx]=Entry(externalID: curKey, internalID: curVal, dib: dib); count &+= 1; return probes } ; if e.externalID==curKey { entries[idx].internalID=curVal; return probes } ; if e.dib < dib { entries[idx]=Entry(externalID: curKey, internalID: curVal, dib: dib); curKey=e.externalID; curVal=e.internalID; dib=e.dib } ; idx=(idx+1)&(bucketCount-1); if dib==255 { throw ErrorBuilder(.capacityExceeded, operation: "idmap_robin_insert").message("Excessive probing in hash table").info("dib", "255").build() } ; dib &+= 1 } ; throw ErrorBuilder(.capacityExceeded, operation: "idmap_robin_insert").message("Hash table full").info("bucket_count", "\(bucketCount)").info("count", "\(count)").build() }
    mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { let r=lookup(key); if r.0 { var idx=hashH1(key, bucketCount); var dib: UInt8=0; var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { return nil } ; if e.externalID==key { entries[idx].internalID=value; return probes } ; if e.dib < dib { return nil } ; idx=(idx+1)&(bucketCount-1); dib &+= 1 } } ; return nil }
    mutating func erase(_ key: UInt64) -> (Bool, Int?) { var idx=hashH1(key, bucketCount); var dib: UInt8=0; var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { return (false, probes) } ; if e.externalID==key { var j=idx; var k=(j+1)&(bucketCount-1); while entries[k].externalID != 0 && entries[k].dib > 0 { entries[j]=Entry(externalID: entries[k].externalID, internalID: entries[k].internalID, dib: entries[k].dib &- 1); j=k; k=(k+1)&(bucketCount-1) } ; entries[j]=Entry(); count &-= 1; return (true, probes) } ; if e.dib < dib { return (false, probes) } ; idx=(idx+1)&(bucketCount-1); dib &+= 1 } ; return (false, probes) }
    mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { var idx=hashH1(key, bucketCount); var dib: UInt8=0; var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.externalID==0 { return (false, -1, probes) } ; if e.externalID==key { return (true, e.internalID, probes) } ; if e.dib < dib { return (false, -1, probes) } ; idx=(idx+1)&(bucketCount-1); dib &+= 1 } ; return (false, -1, probes) }
    func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { for e in entries where e.externalID != 0 { try body(e.externalID, e.internalID) } }
}

private struct LinearProbingTable { enum State: UInt8 { case empty=0, deleted=1, full=2 } ; struct Entry { var externalID: UInt64 = 0; var internalID: Int64 = -1; var st: State = .empty }
    var entries: [Entry]; var bucketCount: Int; var count: Int=0
    init(bucketCount: Int) { self.bucketCount=bucketCount; self.entries=[Entry](repeating: Entry(), count: bucketCount) }
    mutating func insert(_ key: UInt64, _ value: Int64) throws -> Int { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; if entries[idx].st != .full { entries[idx]=Entry(externalID: key, internalID: value, st: .full); count &+= 1; return probes } ; if entries[idx].externalID==key { entries[idx].internalID=value; return probes } ; idx=(idx+1)&(bucketCount-1) } ; throw ErrorBuilder(.capacityExceeded, operation: "idmap_linear_insert").message("Hash table full").info("bucket_count", "\(bucketCount)").info("count", "\(count)").build() }
    mutating func updateValue(for key: UInt64, to value: Int64) -> Int? { let r=lookup(key); if r.0 { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.st == .empty { return nil } ; if e.st == .full && e.externalID==key { entries[idx].internalID=value; return probes } ; idx=(idx+1)&(bucketCount-1) } } ; return nil }
    mutating func erase(_ key: UInt64) -> (Bool, Int?) { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.st == .empty { return (false, probes) } ; if e.st == .full && e.externalID==key { entries[idx].st = .deleted; count &-= 1; return (true, probes) } ; idx=(idx+1)&(bucketCount-1) } ; return (false, probes) }
    mutating func lookup(_ key: UInt64) -> (Bool, Int64, Int) { var idx=hashH1(key, bucketCount); var probes=0; for _ in 0..<bucketCount { probes &+= 1; let e=entries[idx]; if e.st == .empty { return (false, -1, probes) } ; if e.st == .full && e.externalID==key { return (true, e.internalID, probes) } ; idx=(idx+1)&(bucketCount-1) } ; return (false, -1, probes) }
    func forEach(_ body: (UInt64, Int64) throws -> Void) rethrows { for e in entries where e.st == .full { try body(e.externalID, e.internalID) } }
}

public final class IDMap {
    public func append(externalIDs: [UInt64]) throws -> [Int64] {
        try externalIDs.withUnsafeBufferPointer { buf -> [Int64] in
            var out = [Int64](repeating: -1, count: buf.count)
            try out.withUnsafeMutableBufferPointer { dst in
                _ = try idmapAppend(self, externalIDs: buf.baseAddress!, count: buf.count, internalIDsOut: dst.baseAddress)
            }
            return out
        }
    }
    public func lookup(externalID: UInt64) -> Int64? {
        let impl = self.impl
        let lock: RWLock? = impl.rwLock
        if let lock = lock { lock.readLock(); }
        defer { lock?.readUnlock() }
        if let bloom = impl.bloom, !bloom.mightContain(externalID) { return nil }
        let (found, val, probes) = impl.hashTable.lookup(externalID)
        impl.probeTotal &+= Int64(probes); impl.probeOps &+= 1; impl.probeMax = max(impl.probeMax, probes)
        return found ? val : nil
    }
    public func externalID(for internalID: Int64) -> UInt64 {
        let impl = self.impl
        precondition(internalID >= 0 && internalID < impl.count, "internalID out of range")
        return impl.extByInt[Int(internalID)]
    }
    public func lookupBatch(externalIDs: [UInt64]) -> [Int64?] {
        var outs = [Int64?](repeating: nil, count: externalIDs.count)
        for (i, k) in externalIDs.enumerated() {
            outs[i] = lookup(externalID: k)
        }
        return outs
    }
    public func externalIDBatch(internalIDs: [Int64]) -> [UInt64] {
        var outs = [UInt64](repeating: 0, count: internalIDs.count)
        let impl = self.impl
        for (i, id) in internalIDs.enumerated() {
            outs[i] = (id >= 0 && id < impl.count) ? impl.extByInt[Int(id)] : 0
        }
        return outs
    }

    let impl: Impl
    init(impl: Impl) { self.impl = impl }

    public final class Impl {
        var extByInt: [UInt64]
        var count: Int64
        var capacity: Int64
        fileprivate var hashTable: HashTable
        fileprivate var nextInternal: Counter
        fileprivate var rwLock: RWLock?
        fileprivate var bloom: Bloom?
        let opts: IDMapOpts
        var tombstones: TombstoneSet?
        var probeTotal: Int64 = 0
        var probeOps: Int64 = 0
        var probeMax: Int = 0
        fileprivate var retired: [HashTable] = []
        init(extByIntCap: Int, hashBuckets: Int, opts: IDMapOpts) {
            self.extByInt = [UInt64](repeating: 0, count: max(1, extByIntCap))
            self.count = 0
            self.capacity = Int64(extByInt.count)
            self.hashTable = HashTable.allocate(buckets: hashBuckets, impl: opts.hashTableImpl)
            self.nextInternal = Counter()
            self.opts = opts
            self.rwLock = (opts.concurrency == .rwLock) ? RWLock() : nil
            self.bloom = opts.enableBloom ? Bloom(capacity: max(1024, extByIntCap * 4)) : nil
        }
    }
}

private final class Counter {
    private var v: Int64 = 0
    private let lock = NSLock()
    @inline(__always) func fetchAndAdd(_ d: Int64) -> Int64 { lock.lock(); defer { lock.unlock() }; let o = v; v = o &+ d; return o }
    @inline(__always) func load() -> Int64 { lock.lock(); defer { lock.unlock() }; return v }
    @inline(__always) func store(_ x: Int64) { lock.lock(); v = x; lock.unlock() }
}

private final class RWLock {
    private var lock = pthread_rwlock_t()
    init() { pthread_rwlock_init(&lock, nil) }
    deinit { pthread_rwlock_destroy(&lock) }
    @inline(__always) func readLock() { pthread_rwlock_rdlock(&lock) }
    @inline(__always) func readUnlock() { pthread_rwlock_unlock(&lock) }
    @inline(__always) func writeLock() { pthread_rwlock_wrlock(&lock) }
    @inline(__always) func writeUnlock() { pthread_rwlock_unlock(&lock) }
}

private final class Bloom {
    private let m: Int
    private var bits: [UInt64]
    init(capacity: Int) {
        self.m = max(64, nextPow2(capacity))
        self.bits = [UInt64](repeating: 0, count: m >> 6)
    }
    @inline(__always) private func h1(_ x: UInt64) -> Int { Int(mix64(x) & UInt64(m - 1)) }
    @inline(__always) private func h2(_ x: UInt64) -> Int { Int(mix64(x ^ 0x9E3779B97F4A7C15) & UInt64(m - 1)) }
    @inline(__always) func add(_ x: UInt64) {
        let a = h1(x), b = h2(x)
        bits[a >> 6] |= (1 &<< UInt64(a & 63))
        bits[b >> 6] |= (1 &<< UInt64(b & 63))
    }
    @inline(__always) func mightContain(_ x: UInt64) -> Bool {
        let a = h1(x), b = h2(x)
        let ba = bits[a >> 6] & (1 &<< UInt64(a & 63))
        if ba == 0 { return false }
        let bb = bits[b >> 6] & (1 &<< UInt64(b & 63))
        return bb != 0
    }
}

// Initialization & lifecycle
internal func idmapInit(capacityHint: Int, opts: IDMapOpts = IDMapOpts()) -> IDMap {
    let denseCap = max(1, capacityHint)
    let buckets = max(16, nextPow2(Int(Double(denseCap) / max(0.1, min(opts.maxLoadFactor, 0.95)))))
    let impl = IDMap.Impl(extByIntCap: denseCap, hashBuckets: buckets, opts: opts)
    return IDMap(impl: impl)
}
internal func idmapFree(_ map: IDMap) { map.impl.rwLock = nil; map.impl.extByInt.removeAll(keepingCapacity: false); map.impl.retired.removeAll(keepingCapacity: false) }

// Core ops
internal func idmapAppend(_ map: IDMap, externalIDs: UnsafePointer<UInt64>, count n: Int, internalIDsOut: UnsafeMutablePointer<Int64>?) throws -> Int {
    var dummy = [UInt8](repeating: 0, count: n)
    return try idmapAppendWithMask(map, externalIDs: externalIDs, count: n, internalIDsOut: internalIDsOut, foundMask: &dummy)
}

internal func idmapAppendWithMask(_ map: IDMap, externalIDs: UnsafePointer<UInt64>, count n: Int, internalIDsOut: UnsafeMutablePointer<Int64>?, foundMask: UnsafeMutablePointer<UInt8>?) throws -> Int {
    let impl = map.impl
    if let lock = impl.rwLock { lock.writeLock(); defer { lock.writeUnlock() } }
    let base = impl.nextInternal.fetchAndAdd(Int64(n))
    let required = base &+ Int64(n)
    if required > impl.capacity {
        let reqInt = Int(required)
        let rounded = (reqInt + 63) & ~63
        try growDenseArray(impl, newCapacity: rounded)
    }
    let projected = Double(impl.hashTable.count + n) / Double(impl.hashTable.bucketCount)
    if projected > impl.opts.maxLoadFactor { try idmapRehash(map, newBucketCount: impl.hashTable.bucketCount << 1) }
    var newCount = 0
    for i in 0..<n {
        let ext = externalIDs[i]
        if let bloom = impl.bloom, !bloom.mightContain(ext) {
            // proceed to insert
        } else {
            let (found, existInt, probes) = impl.hashTable.lookup(ext)
            if found {
                impl.probeTotal &+= Int64(probes); impl.probeOps &+= 1; impl.probeMax = max(impl.probeMax, probes)
                if impl.opts.allowReplace {
                    let newInternal = base &+ Int64(i)
                    impl.extByInt[Int(newInternal)] = ext
                    if let ts = impl.tombstones { ts.set(existInt) }
                    if let up = impl.hashTable.updateValue(for: ext, to: newInternal) {
                        impl.probeTotal &+= Int64(up); impl.probeOps &+= 1; impl.probeMax = max(impl.probeMax, up)
                    } else {
                        throw ErrorBuilder(.capacityExceeded, operation: "idmap_update")
                            .message("Failed to update hash table entry")
                            .info("external_id", "\(ext)")
                            .build()
                    }
                    internalIDsOut?.advanced(by: i).pointee = newInternal
                    foundMask?.advanced(by: i).pointee = 1
                    newCount &+= 1
                    impl.bloom?.add(ext)
                    continue
                } else {
                    throw ErrorBuilder(.duplicateID, operation: "idmap_insert")
                        .message("Duplicate external ID not allowed")
                        .info("external_id", "\(ext)")
                        .info("existing_internal_id", "\(existInt)")
                        .build()
                }
            }
        }
        let internalID = base &+ Int64(i)
        impl.extByInt[Int(internalID)] = ext
        let probes = try impl.hashTable.insert(ext, internalID)
        impl.probeTotal &+= Int64(probes); impl.probeOps &+= 1; impl.probeMax = max(impl.probeMax, probes)
        internalIDsOut?.advanced(by: i).pointee = internalID
        foundMask?.advanced(by: i).pointee = 1
        newCount &+= 1
        impl.bloom?.add(ext)
    }
    let candidate = base &+ Int64(n)
    if candidate > impl.count { impl.count = candidate }
    return newCount
}

@inline(__always) public func idmapLookup(_ map: IDMap, externalID: UInt64, internalIDOut: UnsafeMutablePointer<Int64>) -> Bool {
    let impl = map.impl
    if let lock = impl.rwLock { lock.readLock(); defer { lock.readUnlock() } }
    if let bloom = impl.bloom, !bloom.mightContain(externalID) { internalIDOut.pointee = -1; return false }
    let (found, val, probes) = impl.hashTable.lookup(externalID)
    impl.probeTotal &+= Int64(probes); impl.probeOps &+= 1; impl.probeMax = max(impl.probeMax, probes)
    internalIDOut.pointee = found ? val : -1
    return found
}

internal func idmapLookupBatch(_ map: IDMap, externalIDs: UnsafePointer<UInt64>, count n: Int, internalIDsOut: UnsafeMutablePointer<Int64>, foundMask: UnsafeMutablePointer<UInt8>?) -> Int {
    let impl = map.impl
    if let lock = impl.rwLock { lock.readLock(); defer { lock.readUnlock() } }
    var foundCnt = 0
    for i in 0..<n { var v: Int64 = -1; let hit = idmapLookup(map, externalID: externalIDs[i], internalIDOut: &v); internalIDsOut.advanced(by: i).pointee = v; if let fm = foundMask { fm.advanced(by: i).pointee = hit ? 1 : 0 }; if hit { foundCnt &+= 1 } }
    return foundCnt
}

@inline(__always) public func idmapExternalFor(_ map: IDMap, internalID: Int64) -> UInt64 { let impl = map.impl; precondition(internalID >= 0 && internalID < impl.count, "internalID out of range"); return impl.extByInt[Int(internalID)] }

internal func idmapExternalForBatch(_ map: IDMap, internalIDs: UnsafePointer<Int64>, count n: Int, externalIDsOut: UnsafeMutablePointer<UInt64>) { let impl = map.impl; for i in 0..<n { let id = internalIDs[i]; externalIDsOut[i] = (id >= 0 && id < impl.count) ? impl.extByInt[Int(id)] : 0 } }

internal func idmapErase(_ map: IDMap, externalIDs: UnsafePointer<UInt64>, count n: Int, tombstones: TombstoneSet?) -> Int { let impl = map.impl; if let lock = impl.rwLock { lock.writeLock(); defer { lock.writeUnlock() } } ; var del=0; for i in 0..<n { let key = externalIDs[i]; let (ok, _) = impl.hashTable.erase(key); if ok { // fallback scan to find old internal ID
            var old: Int64 = -1
            for j in 0..<impl.count { if impl.extByInt[Int(j)] == key { old = j; break } }
            if old >= 0 { (tombstones ?? impl.tombstones)?.set(old) }
            del &+= 1 }
    } ; return del }

internal func idmapRehash(_ map: IDMap, newBucketCount: Int) throws { let impl = map.impl; let buckets = max(16, nextPow2(newBucketCount)); var newTable = HashTable.allocate(buckets: buckets, impl: impl.opts.hashTableImpl); for i in 0..<impl.count { if impl.tombstones?.isSet(i) == true { continue } ; let ext = impl.extByInt[Int(i)]; if ext == 0 && i != 0 { continue } ; _ = try newTable.insert(ext, i) } ; if let lock = impl.rwLock { lock.writeLock(); let old = impl.hashTable; impl.hashTable = newTable; lock.writeUnlock(); impl.retired.append(old) } else { let old = impl.hashTable; impl.hashTable = newTable; _ = old } }

internal func idmapRebuildFromDense(_ map: IDMap) throws { let impl = map.impl; let target = max(16, nextPow2(Int(Double(max(1, Int(impl.count))) / max(0.1, min(impl.opts.maxLoadFactor, 0.95))))); try idmapRehash(map, newBucketCount: target) }

internal func idmapGetStats(_ map: IDMap) -> IDMapStats { let impl = map.impl; let lf = Double(impl.hashTable.count) / Double(max(1, impl.hashTable.bucketCount)); let avg = impl.probeOps > 0 ? Double(impl.probeTotal) / Double(impl.probeOps) : 0.0; let tomb = (impl.tombstones != nil) ? estimateTombstones(impl) : 0; return IDMapStats(count: impl.count, capacity: impl.capacity, hashTableSize: impl.hashTable.bucketCount, loadFactor: lf, avgProbeLength: avg, maxProbeLength: impl.probeMax, tombstoneCount: Int64(tomb)) }
private func estimateTombstones(_ impl: IDMap.Impl) -> Int { var t=0; if let ts = impl.tombstones { for i in 0..<impl.count { if ts.isSet(i) { t &+= 1 } } } ; return t }

@inline(__always) private func growDenseArray(_ impl: IDMap.Impl, newCapacity: Int) throws { if Int64(newCapacity) <= impl.capacity { return } ; let newCap = max(Int(impl.capacity) << 1, newCapacity); impl.extByInt.append(contentsOf: repeatElement(0, count: newCap - impl.extByInt.count)); impl.capacity = Int64(impl.extByInt.count) }

private struct IDMapHeader { var nTotal: Int64; var capacity: Int64; var version: UInt32; var reserved: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8) }
internal func serializeIDMap(_ map: IDMap) throws -> Data {
    let impl = map.impl
    var data = Data()
    var header = IDMapHeader(nTotal: impl.count, capacity: impl.capacity, version: 1,
                             reserved: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    withUnsafeBytes(of: &header) { raw in data.append(contentsOf: raw) }
    let rem = data.count & 63
    if rem != 0 { data.append(contentsOf: [UInt8](repeating: 0, count: 64 - rem)) }
    let n = Int(impl.count)
    impl.extByInt.withUnsafeBytes { raw in
        let view = raw.bindMemory(to: UInt64.self)
        let buf = UnsafeRawBufferPointer(start: view.baseAddress, count: n * MemoryLayout<UInt64>.size)
        data.append(contentsOf: buf)
    }
    let rem2 = data.count & 63
    if rem2 != 0 { data.append(contentsOf: [UInt8](repeating: 0, count: 64 - rem2)) }
    return data
}
internal func deserializeIDMap(_ blob: Data, opts: IDMapOpts = IDMapOpts()) throws -> IDMap { var offs = 0; func read<T>(_ type: T.Type) -> T { let size = MemoryLayout<T>.size; let v: T = blob.withUnsafeBytes { raw in let ptr = raw.baseAddress!.advanced(by: offs).assumingMemoryBound(to: T.self); return ptr.pointee }; offs &+= size; return v } ; let _: IDMapHeader = read(IDMapHeader.self); let rem = offs & 63; if rem != 0 { offs &+= (64 - rem) } ; let n = Int(blob.count - offs) / MemoryLayout<UInt64>.size; var extByInt = [UInt64](repeating: 0, count: max(n, 1)); extByInt.withUnsafeMutableBytes { dst in let bytes = n * MemoryLayout<UInt64>.size; blob.copyBytes(to: dst, from: offs..<(offs + bytes)) } ; let buckets = max(16, nextPow2(Int(Double(max(1, n)) / max(0.1, min(opts.maxLoadFactor, 0.95))))); let impl = IDMap.Impl(extByIntCap: max(n, 1), hashBuckets: buckets, opts: opts); impl.extByInt = extByInt; impl.count = Int64(n); impl.capacity = Int64(extByInt.count); let map = IDMap(impl: impl); try idmapRebuildFromDense(map); return map }
