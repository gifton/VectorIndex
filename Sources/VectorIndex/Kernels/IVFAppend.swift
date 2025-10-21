//
//  IVFAppend.swift
//  Kernel #30: IVF List Append/Insert Operations (Swift)
//
import Foundation
#if canImport(Darwin)
import Darwin
import os.lock
#else
import Glibc
#endif
// Access S1 mmap API
// (IndexMmap is defined in VIndexMmap.swift in this target)


// The implementation is adapted for integration: durable mode is guarded and
// requires S1 mmap handle. For now, durable writes are not enabled from this path.

// Int alignment helper not needed here; UInt64 variant lives in VIndexMmap

@inline(__always) private func isPowerOfTwo(_ x: Int) -> Bool { (x & (x - 1)) == 0 && x > 0 }

@inline(__always) private func alignedAlloc(_ size: Int, alignment: Int = 64) -> UnsafeMutableRawPointer? {
    precondition(isPowerOfTwo(alignment))
    var p: UnsafeMutableRawPointer? = nil
    let err = posix_memalign(&p, alignment, size)
    return err == 0 ? p : nil
}
@inline(__always) private func alignedFree(_ p: UnsafeMutableRawPointer?) { free(p) }

public enum IVFFormat: Int32 { case pq8 = 0, pq4 = 1, flat = 2 }
public enum IVFConcurrencyMode: Int32 { case singleWriter = 0, perListMultiWriter = 1, globalMultiWriter = 2 }

public struct IVFAppendOpts {
    public var format: IVFFormat = .pq8
    public var group: Int = 4
    public var pack4_unpacked: Bool = false
    public var reserve_factor: Float = 2.0
    public var reserve_min: Int = 256
    public var id_bits: Int = 64
    public var timestamps: Bool = false
    public var concurrency: IVFConcurrencyMode = .perListMultiWriter
    public var durable: Bool = false
    public var allocator: IVFAllocator? = nil
    public static var `default`: IVFAppendOpts { IVFAppendOpts() }
}

public struct IVFAllocator {
    public var alloc: (_ bytes: Int, _ align: Int) -> UnsafeMutableRawPointer?
    public var free: (_ p: UnsafeMutableRawPointer?) -> Void
    public init(alloc: @escaping (Int, Int) -> UnsafeMutableRawPointer?, free: @escaping (UnsafeMutableRawPointer?) -> Void) { self.alloc = alloc; self.free = free }
}

public enum IVFError: Error { case invalidInput, invalidDimensions, invalidListID, invalidFormat, invalidGroup, idWidthUnsupported, capacityOverflow, allocationFailed, mmapRequiredForDurable, outOfRange }

private enum IDStorage { case u32(ptr: UnsafeMutablePointer<UInt32>?), u64(ptr: UnsafeMutablePointer<UInt64>?) }
@inline(__always) private func idStrideBytes(_ opts: IVFAppendOpts) -> Int { (opts.id_bits == 32) ? MemoryLayout<UInt32>.stride : MemoryLayout<UInt64>.stride }

@inline(__always)
private func storeExternalID(_ storage: inout IDStorage, _ index: Int, _ val64: UInt64, opts: IVFAppendOpts) throws {
    if opts.id_bits == 32 {
        guard val64 <= UInt64(UInt32.max) else { throw IVFError.idWidthUnsupported }
        if case .u32(let p) = storage { p![index] = UInt32(truncatingIfNeeded: val64) } else { fatalError("ID storage kind mismatch") }
    } else {
        if case .u64(let p) = storage { p![index] = val64 } else { fatalError("ID storage kind mismatch") }
    }
}

@inline(__always)
private func packNibblesU4(idx4: UnsafePointer<UInt8>, n: Int, out: UnsafeMutablePointer<UInt8>) {
    precondition(n % 2 == 0)
    var j = 0
    let n8 = (n >> 3) << 3
    while j < n8 {
        let c0 = idx4[j + 0] & 0x0F, c1 = idx4[j + 1] & 0x0F
        let c2 = idx4[j + 2] & 0x0F, c3 = idx4[j + 3] & 0x0F
        let c4 = idx4[j + 4] & 0x0F, c5 = idx4[j + 5] & 0x0F
        let c6 = idx4[j + 6] & 0x0F, c7 = idx4[j + 7] & 0x0F
        out[(j >> 1) + 0] = c0 | (c1 << 4)
        out[(j >> 1) + 1] = c2 | (c3 << 4)
        out[(j >> 1) + 2] = c4 | (c5 << 4)
        out[(j >> 1) + 3] = c6 | (c7 << 4)
        j += 8
    }
    while j < n { out[j >> 1] = (idx4[j] & 0x0F) | ((idx4[j+1] & 0x0F) << 4); j += 2 }
}

private protocol ListLock { func lock(); func unlock() }
#if canImport(Darwin)
private final class UnfairLock: ListLock {
    private var _l = os_unfair_lock()
    @inline(__always) func lock()   { os_unfair_lock_lock(&_l) }
    @inline(__always) func unlock() { os_unfair_lock_unlock(&_l) }
}
#else
private final class MutexLock: ListLock {
    private var m = pthread_mutex_t()
    init() {
        var a = pthread_mutexattr_t()
        pthread_mutexattr_init(&a)
        pthread_mutex_init(&m, &a)
        pthread_mutexattr_destroy(&a)
    }
    deinit { pthread_mutex_destroy(&m) }
    @inline(__always) func lock()   { pthread_mutex_lock(&m) }
    @inline(__always) func unlock() { pthread_mutex_unlock(&m) }
}
#endif
@inline(__always) private func makeLock() -> any ListLock {
    #if canImport(Darwin)
    return UnfairLock()
    #else
    return MutexLock()
    #endif
}

private enum StorageBackend { case heap, mmap }

public struct IVFListStats { public var length: Int = 0; public var capacity: Int = 0; public var bytesIDs: Int = 0; public var bytesCodesOrVecs: Int = 0 }

private final class IVFList {
    var length: Int = 0
    var capacity: Int = 0
    let lock: any ListLock
    var commitMarker: UInt64 = 0
    var ids: IDStorage
    var codes: UnsafeMutablePointer<UInt8>?
    var xb: UnsafeMutablePointer<Float>?
    var ts: UnsafeMutablePointer<UInt64>? = nil
    var idsRaw: UnsafeMutableRawPointer?
    var codesRaw: UnsafeMutableRawPointer?
    var xbRaw: UnsafeMutableRawPointer?
    var tsRaw: UnsafeMutableRawPointer?
    weak var index: IVFListHandle?

    init(capacity: Int, codeBytesPerVector: Int, dFlat: Int, opts: IVFAppendOpts, index: IVFListHandle) throws {
        self.lock = makeLock(); self.index = index; self.capacity = max(capacity, 0)
        let idBytes = capacity * idStrideBytes(opts)
        guard let idsBuf = (opts.allocator?.alloc(idBytes, 64) ?? alignedAlloc(idBytes, alignment: 64)) else { throw IVFError.allocationFailed }
        self.idsRaw = idsBuf
        if opts.id_bits == 32 { self.ids = .u32(ptr: idsBuf.bindMemory(to: UInt32.self, capacity: capacity)) }
        else { self.ids = .u64(ptr: idsBuf.bindMemory(to: UInt64.self, capacity: capacity)) }
        switch index.format {
        case .pq8, .pq4:
            let bytes = capacity * codeBytesPerVector
            guard let cbuf = (opts.allocator?.alloc(bytes, 1) ?? malloc(bytes)) else { throw IVFError.allocationFailed }
            self.codesRaw = cbuf; self.codes = cbuf.bindMemory(to: UInt8.self, capacity: bytes)
        case .flat:
            let elems = capacity * dFlat; let byteCount = elems * MemoryLayout<Float>.stride
            guard let xbuf = (opts.allocator?.alloc(byteCount, 64) ?? alignedAlloc(byteCount, alignment: 64)) else { throw IVFError.allocationFailed }
            self.xbRaw = xbuf; self.xb = xbuf.bindMemory(to: Float.self, capacity: elems)
        }
        if opts.timestamps {
            let bytes = capacity * MemoryLayout<UInt64>.stride
            guard let tbuf = (opts.allocator?.alloc(bytes, 64) ?? alignedAlloc(bytes, alignment: 64)) else { throw IVFError.allocationFailed }
            self.tsRaw = tbuf; self.ts = tbuf.bindMemory(to: UInt64.self, capacity: capacity)
        }
    }
    deinit { if let r = idsRaw { (index?.opts.allocator?.free(r) ?? alignedFree(r)) }; if let r = codesRaw { (index?.opts.allocator?.free(r) ?? free(r)) }; if let r = xbRaw { (index?.opts.allocator?.free(r) ?? alignedFree(r)) }; if let r = tsRaw { (index?.opts.allocator?.free(r) ?? alignedFree(r)) } }
}

public final class IVFListHandle {
    public let k_c: Int
    public let m: Int
    public let d: Int
    public let format: IVFFormat
    public var opts: IVFAppendOpts
    fileprivate var storage: StorageBackend
    // Optional mmap handle when durable mode is used (internal - low-level persistence)
    internal var mmapHandle: IndexMmap? = nil
    fileprivate var lists: [IVFList] = []
    private var nextInternalID: Int64 = 0
    fileprivate let globalLock: any ListLock = makeLock()
    @inline(__always) public var codeBytesPerVector: Int { switch format { case .pq8: return m; case .pq4: precondition(m % 2 == 0); return m >> 1; case .flat: return d * MemoryLayout<Float>.stride } }

    public init(k_c: Int, m: Int, d: Int, opts: IVFAppendOpts = .default) throws {
        precondition(k_c > 0)
        if opts.format == .flat { precondition(d > 0); precondition(m == 0) }
        else { precondition(m > 0 && d == 0); precondition(opts.group == 4 || opts.group == 8); precondition(m % opts.group == 0) }
        self.k_c = k_c; self.m = m; self.d = d; self.opts = opts; self.format = opts.format; self.storage = .heap
        self.lists.reserveCapacity(k_c)
        let initCap = max(0, opts.reserve_min)
        for _ in 0..<k_c { let list = try IVFList(capacity: initCap, codeBytesPerVector: self.codeBytesPerVector, dFlat: d, opts: opts, index: self); lists.append(list) }
    }
    @inline(__always) fileprivate func allocateInternalID(_ count: Int) -> Int64 { switch opts.concurrency { case .singleWriter: let b = nextInternalID; nextInternalID &+= Int64(count); return b; default: globalLock.lock(); let b = nextInternalID; nextInternalID &+= Int64(count); globalLock.unlock(); return b } }
    public func getListStats(listID: Int32) throws -> IVFListStats { guard listID >= 0 && listID < Int32(k_c) else { throw IVFError.invalidListID }; let L = lists[Int(listID)]; var out = IVFListStats(); out.length = L.length; out.capacity = L.capacity; out.bytesIDs = L.capacity * idStrideBytes(opts); switch format { case .pq8, .pq4: out.bytesCodesOrVecs = L.capacity * codeBytesPerVector; case .flat: out.bytesCodesOrVecs = L.capacity * d * MemoryLayout<Float>.stride }; return out }
    public func getListStats(listID: Int32, durable: Bool) throws -> IVFListStats {
        if durable, storage == .mmap, let mmap = mmapHandle {
            guard listID >= 0 && listID < Int32(k_c) else { throw IVFError.invalidListID }
            guard let (descs, _) = mmap.mmapLists() else { throw IVFError.invalidInput }
            let i = Int(listID); let dsc = descs[i]
            let len = mmap.snapshotListLength(listID: i)
            var out = IVFListStats()
            out.length = len
            out.capacity = dsc.capacityHost(mmap.fileEndianness)
            out.bytesIDs = out.capacity * idStrideBytes(opts)
            switch format { case .pq8, .pq4: out.bytesCodesOrVecs = out.capacity * codeBytesPerVector; case .flat: out.bytesCodesOrVecs = out.capacity * d * MemoryLayout<Float>.stride }
            return out
        }
        return try getListStats(listID: listID)
    }

    public func readList(listID: Int32) throws -> (length: Int, idsU64: UnsafePointer<UInt64>?, idsU32: UnsafePointer<UInt32>?, codes: UnsafePointer<UInt8>?, xb: UnsafePointer<Float>?) {
        guard listID >= 0 && listID < Int32(k_c) else { throw IVFError.invalidListID }
        if storage == .mmap, let mmap = mmapHandle {
            let i = Int(listID)
            let n = mmap.snapshotListLength(listID: i)
            guard let (descs, _) = mmap.mmapLists() else { throw IVFError.invalidInput }
            let dsc = descs[i]
            let idsOff = dsc.idsOffsetHost(mmap.fileEndianness)
            let codesOff = dsc.codesOffsetHost(mmap.fileEndianness)
            let vecsOff = dsc.vecsOffsetHost(mmap.fileEndianness)
            if opts.id_bits == 64 {
                guard let base = mmap.idsBase() else { throw IVFError.invalidInput }
                let ptr = base.advanced(by: Int(idsOff)).assumingMemoryBound(to: UInt64.self)
                if format == .flat {
                    guard let vbase = mmap.vecsBase() else { throw IVFError.invalidInput }
                    let vptr = vbase.advanced(by: Int(vecsOff)).assumingMemoryBound(to: Float.self)
                    return (n, UnsafePointer(ptr), nil, nil, UnsafePointer(vptr))
                } else {
                    guard let cbase = mmap.codesBase() else { throw IVFError.invalidInput }
                    let cptr = cbase.advanced(by: Int(codesOff)).assumingMemoryBound(to: UInt8.self)
                    return (n, UnsafePointer(ptr), nil, UnsafePointer(cptr), nil)
                }
            } else if opts.id_bits == 32 {
                guard let base = mmap.idsBase() else { throw IVFError.invalidInput }
                let ptr = base.advanced(by: Int(idsOff)).assumingMemoryBound(to: UInt32.self)
                if format == .flat {
                    guard let vbase = mmap.vecsBase() else { throw IVFError.invalidInput }
                    let vptr = vbase.advanced(by: Int(vecsOff)).assumingMemoryBound(to: Float.self)
                    return (n, nil, UnsafePointer(ptr), nil, UnsafePointer(vptr))
                } else {
                    guard let cbase = mmap.codesBase() else { throw IVFError.invalidInput }
                    let cptr = cbase.advanced(by: Int(codesOff)).assumingMemoryBound(to: UInt8.self)
                    return (n, nil, UnsafePointer(ptr), UnsafePointer(cptr), nil)
                }
            }
            throw IVFError.idWidthUnsupported
        }
        let L = lists[Int(listID)]
        let n = L.length
        switch opts.id_bits {
        case 64:
            if case .u64(let p) = L.ids { return (n, UnsafePointer(p), nil, UnsafePointer(L.codes), UnsafePointer(L.xb)) }
        case 32:
            if case .u32(let p) = L.ids { return (n, nil, UnsafePointer(p), UnsafePointer(L.codes), UnsafePointer(L.xb)) }
        default: break
        }
        throw IVFError.idWidthUnsupported
    }
}

@inline(__always) public func ivf_create(k_c: Int, m: Int, d: Int, opts: IVFAppendOpts? = nil) throws -> IVFListHandle {
    try IVFListHandle(k_c: k_c, m: m, d: d, opts: opts ?? .default)
}

@inline(__always) public func ivf_destroy(_ index: IVFListHandle) { /* ARC */ }

// Create an IVF handle attached to an existing mmap index for durable appends (internal - low-level mmap API).
@inline(__always)
internal func ivf_create_mmap(k_c: Int, m: Int, d: Int, mmap: IndexMmap, opts: IVFAppendOpts? = nil) throws -> IVFListHandle {
    var o = opts ?? .default
    let h = try IVFListHandle(k_c: k_c, m: m, d: d, opts: o)
    h.storage = .mmap
    h.mmapHandle = mmap
    return h
}

@inline(__always)
private func safeNewCapacity(oldCap: Int, need: Int, opts: IVFAppendOpts) throws -> Int {
    if oldCap == 0 { return max(opts.reserve_min, need) }
    let factor = max(1.1, Double(opts.reserve_factor)); let mult = Double(oldCap) * factor
    let candidate1 = Int(mult.rounded(.up)); let candidate2 = oldCap + max(opts.reserve_min, need - oldCap)
    let newCap = max(candidate1, candidate2); if newCap <= oldCap { throw IVFError.capacityOverflow }; return newCap
}

private func growList(_ list: IVFList, codeBytesPerVec: Int, dFlat: Int, opts: IVFAppendOpts, index: IVFListHandle, minNewLen: Int) throws {
    let newCap = try safeNewCapacity(oldCap: list.capacity, need: minNewLen, opts: opts)
    let idBytes = newCap * idStrideBytes(opts)
    guard let newIDsRaw = (opts.allocator?.alloc(idBytes, 64) ?? alignedAlloc(idBytes, alignment: 64)) else { throw IVFError.allocationFailed }
    var newIDs: IDStorage = (opts.id_bits == 32) ? .u32(ptr: newIDsRaw.bindMemory(to: UInt32.self, capacity: newCap)) : .u64(ptr: newIDsRaw.bindMemory(to: UInt64.self, capacity: newCap))
    var newCodesRaw: UnsafeMutableRawPointer? = nil; var newCodes: UnsafeMutablePointer<UInt8>? = nil; var newXBRaw: UnsafeMutableRawPointer? = nil; var newXB: UnsafeMutablePointer<Float>? = nil
    switch index.format {
    case .pq8, .pq4:
        let bytes = newCap * codeBytesPerVec; guard let c = (opts.allocator?.alloc(bytes, 1) ?? malloc(bytes)) else { (opts.allocator?.free(newIDsRaw) ?? alignedFree(newIDsRaw)); throw IVFError.allocationFailed }
        newCodesRaw = c; newCodes = c.bindMemory(to: UInt8.self, capacity: bytes)
    case .flat:
        let elems = newCap * dFlat; let byteCount = elems * MemoryLayout<Float>.stride
        guard let xraw = (opts.allocator?.alloc(byteCount, 64) ?? alignedAlloc(byteCount, alignment: 64)) else { (opts.allocator?.free(newIDsRaw) ?? alignedFree(newIDsRaw)); throw IVFError.allocationFailed }
        newXBRaw = xraw; newXB = xraw.bindMemory(to: Float.self, capacity: elems)
    }
    var newTSRaw: UnsafeMutableRawPointer? = nil; var newTS: UnsafeMutablePointer<UInt64>? = nil
    if opts.timestamps { let bytes = newCap * MemoryLayout<UInt64>.stride; guard let t = (opts.allocator?.alloc(bytes, 64) ?? alignedAlloc(bytes, alignment: 64)) else { (opts.allocator?.free(newIDsRaw) ?? alignedFree(newIDsRaw)); if let r = newCodesRaw { (opts.allocator?.free(r) ?? free(r)) }; if let r = newXBRaw { (opts.allocator?.free(r) ?? alignedFree(r)) }; throw IVFError.allocationFailed }; newTSRaw = t; newTS = t.bindMemory(to: UInt64.self, capacity: newCap) }
    let n = list.length
    if n > 0 {
        switch (list.ids, newIDs) {
        case (.u64(let old), .u64(let neu)): memcpy(neu, old, n * MemoryLayout<UInt64>.stride)
        case (.u32(let old), .u32(let neu)): memcpy(neu, old, n * MemoryLayout<UInt32>.stride)
        default: fatalError("ID storage kind mismatch on grow")
        }
        switch index.format {
        case .pq8, .pq4: if let old = list.codes, let neu = newCodes { memcpy(neu, old, n * codeBytesPerVec) }
        case .flat: if let old = list.xb, let neu = newXB { memcpy(neu, old, n * dFlat * MemoryLayout<Float>.stride) }
        }
        if opts.timestamps, let old = list.ts, let neu = newTS { memcpy(neu, old, n * MemoryLayout<UInt64>.stride) }
    }
    if let r = list.idsRaw { (index.opts.allocator?.free(r) ?? alignedFree(r)) }
    if let r = list.codesRaw { (index.opts.allocator?.free(r) ?? free(r)) }
    if let r = list.xbRaw { (index.opts.allocator?.free(r) ?? alignedFree(r)) }
    if let r = list.tsRaw { (index.opts.allocator?.free(r) ?? alignedFree(r)) }
    list.idsRaw = newIDsRaw; list.ids = newIDs; list.codesRaw = newCodesRaw; list.codes = newCodes; list.xbRaw = newXBRaw; list.xb = newXB; list.tsRaw = newTSRaw; list.ts = newTS; list.capacity = newCap
}

private struct PerListBatch { var count: Int = 0; var indices: [Int] = [] }
private func groupByListIDs(listIDs: UnsafePointer<Int32>, n: Int, k_c: Int) -> [PerListBatch] {
    var counts = [Int](repeating: 0, count: k_c); for i in 0..<n { counts[Int(listIDs[i])] &+= 1 }
    var batches = [PerListBatch](repeating: PerListBatch(), count: k_c)
    for l in 0..<k_c { batches[l].count = counts[l]; batches[l].indices.reserveCapacity(counts[l]) }
    for i in 0..<n { let lid = Int(listIDs[i]); batches[lid].indices.append(i) }
    return batches
}

public func ivf_append(list_ids: UnsafePointer<Int32>, external_ids: UnsafePointer<UInt64>, codes: UnsafePointer<UInt8>, n: Int, m: Int, index: IVFListHandle, opts inOpts: IVFAppendOpts?, internalIDsOut: UnsafeMutablePointer<Int64>?) throws {
    guard n >= 0, m == index.m, (index.format == .pq8 || index.format == .pq4) else { throw IVFError.invalidFormat }
    let opts = inOpts ?? index.opts
    guard (opts.group == 4 || opts.group == 8), m % opts.group == 0 else { throw IVFError.invalidGroup }
    if opts.durable {
        guard index.storage == .mmap, let mmap = index.mmapHandle else { throw IVFError.mmapRequiredForDurable }
        let baseInternal = index.allocateInternalID(n)
        // Group by list and perform mmap durable appends per list
        for listID in 0..<index.k_c {
            // Build per-list indices
            var localIndices: [Int] = []
            localIndices.reserveCapacity(n / max(1, index.k_c))
            for i in 0..<n { if Int(list_ids[i]) == listID { localIndices.append(i) } }
            if localIndices.isEmpty { continue }
            let count = localIndices.count
            // Prepare contiguous temp buffers
            let idStride = idStrideBytes(index.opts)
            let codesStride = (index.format == .pq8) ? m : (m >> 1)
            let idsBytes = count * idStride
            let codesBytes = count * codesStride
            guard let idsBuf = malloc(idsBytes) else { throw IVFError.allocationFailed }
            defer { free(idsBuf) }
            guard let codesBuf = malloc(codesBytes) else { throw IVFError.allocationFailed }
            defer { free(codesBuf) }
            // Fill buffers
            if index.opts.id_bits == 32 {
                let p = idsBuf.bindMemory(to: UInt32.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() {
                    let v = external_ids[srcIdx]
                    if v > UInt64(UInt32.max) { throw IVFError.idWidthUnsupported }
                    p[j] = UInt32(truncatingIfNeeded: v)
                }
            } else {
                let p = idsBuf.bindMemory(to: UInt64.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() { p[j] = external_ids[srcIdx] }
            }
            let codeDst = codesBuf.bindMemory(to: UInt8.self, capacity: codesBytes)
            if index.format == .pq8 {
                for (j, srcIdx) in localIndices.enumerated() { memcpy(codeDst.advanced(by: j * codesStride), codes.advanced(by: srcIdx * m), m) }
            } else { // pq4
                if index.opts.pack4_unpacked {
                    for (j, srcIdx) in localIndices.enumerated() { packNibblesU4(idx4: codes.advanced(by: srcIdx * m), n: m, out: codeDst.advanced(by: j * codesStride)) }
                } else {
                    for (j, srcIdx) in localIndices.enumerated() { memcpy(codeDst.advanced(by: j * codesStride), codes.advanced(by: srcIdx * (m >> 1)), (m >> 1)) }
                }
            }
            // Reserve and commit
            let res = try mmap.mmap_append_begin(listID: listID, addLen: count)
            // Sanity: ensure stride matches container
            if res.idsStride != idStride || res.codesStride != codesStride { throw IVFError.invalidFormat }
            try mmap.mmap_append_commit(res, idsSrc: idsBuf, codesSrc: codesBuf, vecsSrc: nil)
            // Internal IDs out (monotonic over entire batch)
            if let out = internalIDsOut { for srcIdx in localIndices { out[srcIdx] = baseInternal + Int64(srcIdx) } }
            // Update in-memory length mirror
            let L = index.lists[listID]
            L.length += count
            if L.capacity < L.length { L.capacity = L.length }
        }
        return
    }
    for i in 0..<n { let lid = Int(list_ids[i]); if lid < 0 || lid >= index.k_c { throw IVFError.invalidListID } }
    let batches = groupByListIDs(listIDs: list_ids, n: n, k_c: index.k_c)
    let baseInternal = index.allocateInternalID(n)
    let srcCodeStride: Int = (index.format == .pq8) ? m : (opts.pack4_unpacked ? m : (m >> 1))
    for listID in 0..<index.k_c {
        let batch = batches[listID]; if batch.count == 0 { continue }
        let L = index.lists[listID]
        switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
        let newLen = L.length + batch.count
        if newLen > L.capacity { try growList(L, codeBytesPerVec: index.codeBytesPerVector, dFlat: index.d, opts: index.opts, index: index, minNewLen: newLen) }
        let startPos = L.length; let codeBytesPerVec = index.codeBytesPerVector
        let codesDstBase = L.codes?.advanced(by: startPos * codeBytesPerVec)
        for (localIdx, srcIdx) in batch.indices.enumerated() {
            try storeExternalID(&L.ids, startPos + localIdx, external_ids[srcIdx], opts: index.opts)
            if let out = internalIDsOut { out[srcIdx] = baseInternal + Int64(srcIdx) }
            if let codesDstBase = codesDstBase { let dst = codesDstBase.advanced(by: localIdx * codeBytesPerVec); let src = codes.advanced(by: srcIdx * srcCodeStride); switch index.format { case .pq8: memcpy(dst, src, m); case .pq4: if index.opts.pack4_unpacked { packNibblesU4(idx4: src, n: m, out: dst) } else { memcpy(dst, src, m >> 1) }; case .flat: break } }
        }
        if index.opts.timestamps, let tsPtr = L.ts { let now = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0); for i in 0..<batch.count { tsPtr[startPos + i] = now } }
        L.length = newLen
        switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break }
    }
}

public func ivf_append_one_list(list_id: Int32, external_ids: UnsafePointer<UInt64>, codes: UnsafePointer<UInt8>, n: Int, m: Int, index: IVFListHandle, opts inOpts: IVFAppendOpts?, internalIDsOut: UnsafeMutablePointer<Int64>?) throws {
    guard n >= 0, m == index.m, (index.format == .pq8 || index.format == .pq4) else { throw IVFError.invalidFormat }
    guard list_id >= 0 && list_id < Int32(index.k_c) else { throw IVFError.invalidListID }
    let opts = inOpts ?? index.opts
    if opts.durable { throw IVFError.mmapRequiredForDurable }
    let L = index.lists[Int(list_id)]
    switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
    let newLen = L.length + n
    if newLen > L.capacity { try growList(L, codeBytesPerVec: index.codeBytesPerVector, dFlat: index.d, opts: index.opts, index: index, minNewLen: newLen) }
    let baseInternal = index.allocateInternalID(n)
    let srcCodeStride: Int = (index.format == .pq8) ? m : (opts.pack4_unpacked ? m : (m >> 1))
    let startPos = L.length
    for i in 0..<n {
        try storeExternalID(&L.ids, startPos + i, external_ids[i], opts: index.opts)
        if let out = internalIDsOut { out[i] = baseInternal + Int64(i) }
        if index.format == .pq8 || index.format == .pq4 {
            let dst = L.codes!.advanced(by: (startPos + i) * index.codeBytesPerVector)
            let src = codes.advanced(by: i * srcCodeStride)
            switch index.format { case .pq8: memcpy(dst, src, m); case .pq4: if opts.pack4_unpacked { packNibblesU4(idx4: src, n: m, out: dst) } else { memcpy(dst, src, m >> 1) }; case .flat: break }
        }
    }
    if index.opts.timestamps, let tsPtr = L.ts { let now = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0); for i in 0..<n { tsPtr[startPos + i] = now } }
    L.length = newLen
    switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break }
}

public func ivf_append_flat(list_ids: UnsafePointer<Int32>, external_ids: UnsafePointer<UInt64>, xb: UnsafePointer<Float>, n: Int, d: Int, index: IVFListHandle, opts inOpts: IVFAppendOpts?, internalIDsOut: UnsafeMutablePointer<Int64>?) throws {
    guard index.format == .flat, d == index.d else { throw IVFError.invalidDimensions }
    let opts = inOpts ?? index.opts
    if opts.durable {
        guard index.storage == .mmap, let mmap = index.mmapHandle else { throw IVFError.mmapRequiredForDurable }
        let baseInternal = index.allocateInternalID(n)
        for listID in 0..<index.k_c {
            var localIndices: [Int] = []
            for i in 0..<n { if Int(list_ids[i]) == listID { localIndices.append(i) } }
            if localIndices.isEmpty { continue }
            let count = localIndices.count
            let idStride = idStrideBytes(index.opts)
            let idsBytes = count * idStride
            let vecStride = d * MemoryLayout<Float>.stride
            let vecBytes = count * vecStride
            guard let idsBuf = malloc(idsBytes) else { throw IVFError.allocationFailed }
            defer { free(idsBuf) }
            guard let vecBuf = malloc(vecBytes) else { throw IVFError.allocationFailed }
            defer { free(vecBuf) }
            if index.opts.id_bits == 32 {
                let p = idsBuf.bindMemory(to: UInt32.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() { let v = external_ids[srcIdx]; if v > UInt64(UInt32.max) { throw IVFError.idWidthUnsupported }; p[j] = UInt32(truncatingIfNeeded: v) }
            } else {
                let p = idsBuf.bindMemory(to: UInt64.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() { p[j] = external_ids[srcIdx] }
            }
            let vdst = vecBuf.bindMemory(to: UInt8.self, capacity: vecBytes)
            for (j, srcIdx) in localIndices.enumerated() {
                memcpy(vdst.advanced(by: j * vecStride), xb.advanced(by: srcIdx * d), vecStride)
            }
            let res = try mmap.mmap_append_begin(listID: listID, addLen: count)
            if res.idsStride != idStride || res.vecsStride != vecStride { throw IVFError.invalidFormat }
            try mmap.mmap_append_commit(res, idsSrc: idsBuf, codesSrc: nil, vecsSrc: vecBuf)
            if let out = internalIDsOut { for srcIdx in localIndices { out[srcIdx] = baseInternal + Int64(srcIdx) } }
            let L = index.lists[listID]; L.length += count; if L.capacity < L.length { L.capacity = L.length }
        }
        return
    }
    for i in 0..<n { let lid = Int(list_ids[i]); if lid < 0 || lid >= index.k_c { throw IVFError.invalidListID } }
    let batches = groupByListIDs(listIDs: list_ids, n: n, k_c: index.k_c)
    let baseInternal = index.allocateInternalID(n)
    for listID in 0..<index.k_c {
        let batch = batches[listID]; if batch.count == 0 { continue }
        let L = index.lists[listID]
        switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
        let newLen = L.length + batch.count
        if newLen > L.capacity { try growList(L, codeBytesPerVec: index.codeBytesPerVector, dFlat: index.d, opts: index.opts, index: index, minNewLen: newLen) }
        let startPos = L.length
        for (localIdx, srcIdx) in batch.indices.enumerated() {
            try storeExternalID(&L.ids, startPos + localIdx, external_ids[srcIdx], opts: index.opts)
            if let out = internalIDsOut { out[srcIdx] = baseInternal + Int64(srcIdx) }
            let dst = L.xb!.advanced(by: (startPos + localIdx) * d)
            memcpy(dst, xb.advanced(by: srcIdx * d), d * MemoryLayout<Float>.stride)
        }
        if index.opts.timestamps, let tsPtr = L.ts { let now = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0); for i in 0..<batch.count { tsPtr[startPos + i] = now } }
        L.length = newLen
        switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break }
    }
}

public func ivf_insert_at(list_id: Int32, pos: Int, external_ids: UnsafePointer<UInt64>, codes: UnsafePointer<UInt8>, n: Int, index: IVFListHandle) throws {
    guard list_id >= 0 && list_id < Int32(index.k_c) else { throw IVFError.invalidListID }
    guard n >= 0 else { throw IVFError.invalidInput }
    if index.opts.durable { throw IVFError.mmapRequiredForDurable }
    let L = index.lists[Int(list_id)]
    switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
    defer { switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break } }
    guard pos >= 0 && pos <= L.length else { throw IVFError.outOfRange }
    let newLen = L.length + n
    if newLen > L.capacity { try growList(L, codeBytesPerVec: index.codeBytesPerVector, dFlat: index.d, opts: index.opts, index: index, minNewLen: newLen) }
    let tailCount = L.length - pos; let idB = idStrideBytes(index.opts)
    if tailCount > 0 {
        let dstIDs = L.idsRaw!.advanced(by: (pos + n) * idB); let srcIDs = L.idsRaw!.advanced(by: pos * idB); memmove(dstIDs, srcIDs, tailCount * idB)
        switch index.format {
        case .pq8, .pq4:
            let bVec = index.codeBytesPerVector; let dst = L.codes!.advanced(by: (pos + n) * bVec); let src = L.codes!.advanced(by: pos * bVec); memmove(dst, src, tailCount * bVec)
        case .flat:
            let elems = tailCount * index.d; let dst = L.xb!.advanced(by: (pos + n) * index.d); let src = L.xb!.advanced(by: pos * index.d); memmove(dst, src, elems * MemoryLayout<Float>.stride)
        }
        if index.opts.timestamps, let tsPtr = L.ts { let dst = tsPtr.advanced(by: pos + n); let src = tsPtr.advanced(by: pos); memmove(dst, src, tailCount * MemoryLayout<UInt64>.stride) }
    }
    for i in 0..<n {
        try storeExternalID(&L.ids, pos + i, external_ids[i], opts: index.opts)
        switch index.format {
        case .pq8:
            let dst = L.codes!.advanced(by: (pos + i) * index.codeBytesPerVector); let src = codes.advanced(by: i * index.m); memcpy(dst, src, index.m)
        case .pq4:
            let dst = L.codes!.advanced(by: (pos + i) * index.codeBytesPerVector); memcpy(dst, codes.advanced(by: i * (index.m >> 1)), index.m >> 1)
        case .flat: fatalError("Use ivf_insert_at_flat for IVF-Flat")
        }
        if index.opts.timestamps, let tsPtr = L.ts { tsPtr[pos + i] = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0) }
    }
    L.length = newLen
}

public func ivf_insert_at_flat(list_id: Int32, pos: Int, external_ids: UnsafePointer<UInt64>, xb: UnsafePointer<Float>, n: Int, index: IVFListHandle) throws {
    guard index.format == .flat else { throw IVFError.invalidFormat }
    guard list_id >= 0 && list_id < Int32(index.k_c) else { throw IVFError.invalidListID }
    guard n >= 0 else { throw IVFError.invalidInput }
    if index.opts.durable { throw IVFError.mmapRequiredForDurable }
    let L = index.lists[Int(list_id)]
    switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
    defer { switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break } }
    guard pos >= 0 && pos <= L.length else { throw IVFError.outOfRange }
    let newLen = L.length + n
    if newLen > L.capacity { try growList(L, codeBytesPerVec: index.codeBytesPerVector, dFlat: index.d, opts: index.opts, index: index, minNewLen: newLen) }
    let tail = L.length - pos; let idB = idStrideBytes(index.opts)
    if tail > 0 {
        let dstIDs = L.idsRaw!.advanced(by: (pos + n) * idB); let srcIDs = L.idsRaw!.advanced(by: pos * idB); memmove(dstIDs, srcIDs, tail * idB)
        let elems = tail * index.d; let dst = L.xb!.advanced(by: (pos + n) * index.d); let src = L.xb!.advanced(by: pos * index.d); memmove(dst, src, elems * MemoryLayout<Float>.stride)
        if index.opts.timestamps, let tsPtr = L.ts { let dst = tsPtr.advanced(by: pos + n); let src = tsPtr.advanced(by: pos); memmove(dst, src, tail * MemoryLayout<UInt64>.stride) }
    }
    for i in 0..<n { try storeExternalID(&L.ids, pos + i, external_ids[i], opts: index.opts); let dst = L.xb!.advanced(by: (pos + i) * index.d); memcpy(dst, xb.advanced(by: i * index.d), index.d * MemoryLayout<Float>.stride); if index.opts.timestamps, let tsPtr = L.ts { tsPtr[pos + i] = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0) } }
    L.length = newLen
}
