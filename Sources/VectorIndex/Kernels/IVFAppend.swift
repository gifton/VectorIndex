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
    var p: UnsafeMutableRawPointer?
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
    public var allocator: IVFAllocator?
    public static var `default`: IVFAppendOpts { IVFAppendOpts() }
}

public struct IVFAllocator {
    public var alloc: (_ bytes: Int, _ align: Int) -> UnsafeMutableRawPointer?
    public var free: (_ p: UnsafeMutableRawPointer?) -> Void
    public init(alloc: @escaping (Int, Int) -> UnsafeMutableRawPointer?, free: @escaping (UnsafeMutableRawPointer?) -> Void) { self.alloc = alloc; self.free = free }
}

// IVFError removed - migrated to VectorIndexError
// All throw sites now use ErrorBuilder with appropriate IndexErrorKind

private enum IDStorage { case u32(ptr: UnsafeMutablePointer<UInt32>?), u64(ptr: UnsafeMutablePointer<UInt64>?) }
@inline(__always) private func idStrideBytes(_ opts: IVFAppendOpts) -> Int { (opts.id_bits == 32) ? MemoryLayout<UInt32>.stride : MemoryLayout<UInt64>.stride }

/// Stores an external ID into the appropriate storage type
///
/// This function validates that the storage variant matches the configured ID bit width.
/// A mismatch indicates internal corruption or a programming error (opts.id_bits modified
/// after storage allocation, memory corruption, or bug in storage initialization).
///
/// - Throws:
///   - `IVFError.idWidthUnsupported`: If value exceeds UInt32.max when using 32-bit IDs
///   - `VectorIndexError(.internalInconsistency)`: If storage type doesn't match opts.id_bits
@inline(__always)
private func storeExternalID(_ storage: inout IDStorage, _ index: Int, _ val64: UInt64, opts: IVFAppendOpts) throws {
    if opts.id_bits == 32 {
        // Validate value fits in 32 bits
        guard val64 <= UInt64(UInt32.max) else {
            throw ErrorBuilder(.invalidParameter, operation: "store_external_id")
                .message("ID value exceeds 32-bit maximum")
                .info("id_value", "\(val64)")
                .info("max_value", "\(UInt32.max)")
                .build()
        }

        // Verify storage is actually u32 variant
        if case .u32(let p) = storage {
            p![index] = UInt32(truncatingIfNeeded: val64)
        } else {
            // Internal inconsistency: storage is u64 but opts specify 32-bit IDs
            throw ErrorBuilder(.internalInconsistency, operation: "store_external_id")
                .message("ID storage type mismatch: expected u32 storage but found u64")
                .info("expected_bits", "32")
                .info("storage_variant", "u64")
                .info("index", "\(index)")
                .build()
        }
    } else {
        // opts.id_bits == 64
        if case .u64(let p) = storage {
            p![index] = val64
        } else {
            // Internal inconsistency: storage is u32 but opts specify 64-bit IDs
            throw ErrorBuilder(.internalInconsistency, operation: "store_external_id")
                .message("ID storage type mismatch: expected u64 storage but found u32")
                .info("expected_bits", "64")
                .info("storage_variant", "u32")
                .info("index", "\(index)")
                .build()
        }
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
    @inline(__always) func lock() { os_unfair_lock_lock(&_l) }
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
    @inline(__always) func lock() { pthread_mutex_lock(&m) }
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
    var ts: UnsafeMutablePointer<UInt64>?
    var idsRaw: UnsafeMutableRawPointer?
    var codesRaw: UnsafeMutableRawPointer?
    var xbRaw: UnsafeMutableRawPointer?
    var tsRaw: UnsafeMutableRawPointer?
    weak var index: IVFListHandle?

    init(capacity: Int, codeBytesPerVector: Int, dFlat: Int, opts: IVFAppendOpts, index: IVFListHandle) throws {
        self.lock = makeLock(); self.index = index; self.capacity = max(capacity, 0)
        let idBytes = capacity * idStrideBytes(opts)
        guard let idsBuf = (opts.allocator?.alloc(idBytes, 64) ?? alignedAlloc(idBytes, alignment: 64)) else {
            throw ErrorBuilder(.memoryExhausted, operation: "ivf_list_init")
                .message("Failed to allocate ID storage")
                .info("requested_bytes", "\(idBytes)")
                .build()
        }
        self.idsRaw = idsBuf
        if opts.id_bits == 32 { self.ids = .u32(ptr: idsBuf.bindMemory(to: UInt32.self, capacity: capacity)) } else { self.ids = .u64(ptr: idsBuf.bindMemory(to: UInt64.self, capacity: capacity)) }
        switch index.format {
        case .pq8, .pq4:
            let bytes = capacity * codeBytesPerVector
            guard let cbuf = (opts.allocator?.alloc(bytes, 1) ?? malloc(bytes)) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_list_init")
                    .message("Failed to allocate code storage")
                    .info("requested_bytes", "\(bytes)")
                    .build()
            }
            self.codesRaw = cbuf; self.codes = cbuf.bindMemory(to: UInt8.self, capacity: bytes)
        case .flat:
            let elems = capacity * dFlat; let byteCount = elems * MemoryLayout<Float>.stride
            guard let xbuf = (opts.allocator?.alloc(byteCount, 64) ?? alignedAlloc(byteCount, alignment: 64)) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_list_init")
                    .message("Failed to allocate vector storage")
                    .info("requested_bytes", "\(byteCount)")
                    .build()
            }
            self.xbRaw = xbuf; self.xb = xbuf.bindMemory(to: Float.self, capacity: elems)
        }
        if opts.timestamps {
            let bytes = capacity * MemoryLayout<UInt64>.stride
            guard let tbuf = (opts.allocator?.alloc(bytes, 64) ?? alignedAlloc(bytes, alignment: 64)) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_list_init")
                    .message("Failed to allocate timestamp storage")
                    .info("requested_bytes", "\(bytes)")
                    .build()
            }
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
    internal var mmapHandle: IndexMmap?
    fileprivate var lists: [IVFList] = []
    private var nextInternalID: Int64 = 0
    fileprivate let globalLock: any ListLock = makeLock()
    @inline(__always) public var codeBytesPerVector: Int { switch format { case .pq8: return m; case .pq4: precondition(m % 2 == 0); return m >> 1; case .flat: return d * MemoryLayout<Float>.stride } }

    /// Create IVF index handle with specified configuration
    ///
    /// - Throws:
    ///   - `VectorIndexError(.invalidParameter)`: If k_c ≤ 0
    ///   - `VectorIndexError(.invalidParameter)`: If d ≤ 0 for flat format
    ///   - `VectorIndexError(.invalidParameter)`: If m ≠ 0 for flat format
    ///   - `VectorIndexError(.invalidParameter)`: If m ≤ 0 or d ≠ 0 for PQ format
    ///   - `VectorIndexError(.invalidParameter)`: If group not 4 or 8
    ///   - `VectorIndexError(.invalidParameter)`: If m not divisible by group
    ///   - `IVFError(.allocationFailed)`: If memory allocation fails
    ///
    /// - Parameters:
    ///   - k_c: Number of partitions (must be > 0)
    ///   - m: Number of PQ subspaces (must be 0 for flat, > 0 for PQ)
    ///   - d: Vector dimension (must be > 0 for flat, 0 for PQ)
    ///   - opts: Configuration options
    public init(k_c: Int, m: Int, d: Int, opts: IVFAppendOpts = .default) throws {
        // Validate k_c
        guard k_c > 0 else {
            throw ErrorBuilder.invalidParameter(
                operation: "ivf_create",
                name: "k_c",
                value: "\(k_c)",
                constraint: "must be > 0"
            )
        }

        // Validate format-specific parameters
        if opts.format == .flat {
            // Flat format: d > 0, m == 0
            guard d > 0 else {
                throw ErrorBuilder.invalidParameter(
                    operation: "ivf_create",
                    name: "d",
                    value: "\(d)",
                    constraint: "must be > 0 for flat format"
                )
            }
            guard m == 0 else {
                throw ErrorBuilder.invalidParameter(
                    operation: "ivf_create",
                    name: "m",
                    value: "\(m)",
                    constraint: "must be 0 for flat format"
                )
            }
        } else {
            // PQ format: m > 0, d == 0
            guard m > 0 && d == 0 else {
                throw ErrorBuilder(.invalidParameter, operation: "ivf_create")
                    .message("For PQ format: m must be > 0 and d must be 0")
                    .info("m", "\(m)")
                    .info("d", "\(d)")
                    .info("format", "\(opts.format)")
                    .build()
            }

            // Validate group size
            guard opts.group == 4 || opts.group == 8 else {
                throw ErrorBuilder.invalidParameter(
                    operation: "ivf_create",
                    name: "group",
                    value: "\(opts.group)",
                    constraint: "must be 4 or 8"
                )
            }

            // Validate m divisible by group
            guard m % opts.group == 0 else {
                throw ErrorBuilder(.invalidParameter, operation: "ivf_create")
                    .message("m must be divisible by group size")
                    .info("m", "\(m)")
                    .info("group", "\(opts.group)")
                    .build()
            }
        }

        self.k_c = k_c; self.m = m; self.d = d; self.opts = opts; self.format = opts.format; self.storage = .heap
        self.lists.reserveCapacity(k_c)
        let initCap = max(0, opts.reserve_min)
        for _ in 0..<k_c { let list = try IVFList(capacity: initCap, codeBytesPerVector: self.codeBytesPerVector, dFlat: d, opts: opts, index: self); lists.append(list) }
    }
    @inline(__always) fileprivate func allocateInternalID(_ count: Int) -> Int64 { switch opts.concurrency { case .singleWriter: let b = nextInternalID; nextInternalID &+= Int64(count); return b; default: globalLock.lock(); let b = nextInternalID; nextInternalID &+= Int64(count); globalLock.unlock(); return b } }
    public func getListStats(listID: Int32) throws -> IVFListStats {
        guard listID >= 0 && listID < Int32(k_c) else {
            throw ErrorBuilder(.invalidRange, operation: "get_list_stats")
                .message("List ID out of valid range")
                .info("list_id", "\(listID)")
                .info("valid_range", "0..<\(k_c)")
                .build()
        }
        let L = lists[Int(listID)]
        var out = IVFListStats()
        out.length = L.length
        out.capacity = L.capacity
        out.bytesIDs = L.capacity * idStrideBytes(opts)
        switch format {
        case .pq8, .pq4: out.bytesCodesOrVecs = L.capacity * codeBytesPerVector
        case .flat: out.bytesCodesOrVecs = L.capacity * d * MemoryLayout<Float>.stride
        }
        return out
    }
    public func getListStats(listID: Int32, durable: Bool) throws -> IVFListStats {
        if durable, storage == .mmap, let mmap = mmapHandle {
            guard listID >= 0 && listID < Int32(k_c) else {
                throw ErrorBuilder(.invalidRange, operation: "get_list_stats_durable")
                    .message("List ID out of valid range")
                    .info("list_id", "\(listID)")
                    .info("valid_range", "0..<\(k_c)")
                    .build()
            }
            guard let (descs, _) = mmap.mmapLists() else {
                throw ErrorBuilder(.contractViolation, operation: "get_list_stats_durable")
                    .message("mmap list descriptors unavailable (internal error)")
                    .build()
            }
            let i = Int(listID)
            let dsc = descs[i]
            let len = mmap.snapshotListLength(listID: i)
            var out = IVFListStats()
            out.length = len
            out.capacity = dsc.capacityHost(mmap.fileEndianness)
            out.bytesIDs = out.capacity * idStrideBytes(opts)
            switch format {
            case .pq8, .pq4: out.bytesCodesOrVecs = out.capacity * codeBytesPerVector
            case .flat: out.bytesCodesOrVecs = out.capacity * d * MemoryLayout<Float>.stride
            }
            return out
        }
        return try getListStats(listID: listID)
    }

    public func readList(listID: Int32) throws -> (length: Int, idsU64: UnsafePointer<UInt64>?, idsU32: UnsafePointer<UInt32>?, codes: UnsafePointer<UInt8>?, xb: UnsafePointer<Float>?) {
        guard listID >= 0 && listID < Int32(k_c) else {
            throw ErrorBuilder(.invalidRange, operation: "read_list")
                .message("List ID out of valid range")
                .info("list_id", "\(listID)")
                .info("valid_range", "0..<\(k_c)")
                .build()
        }

        if storage == .mmap, let mmap = mmapHandle {
            let i = Int(listID)
            let n = mmap.snapshotListLength(listID: i)
            guard let desc = mmap.getListDescriptor(listID: i) else {
                throw ErrorBuilder(.contractViolation, operation: "read_list")
                    .message("mmap list descriptors unavailable (internal error)")
                    .build()
            }
            let idsOff = desc.idsOff
            let codesOff = desc.codesOff
            let vecsOff = desc.vecsOff

            if opts.id_bits == 64 {
                guard let (idsBaseRaw, idsSize) = mmap.sectionSlice(.ids) else {
                    throw ErrorBuilder(.contractViolation, operation: "read_list")
                        .message("mmap IDs base pointer unavailable (internal error)")
                        .build()
                }
                let idStride = idStrideBytes(opts)
                let idsSize64 = UInt64(idsSize)
                let idsNeeded = idsOff &+ UInt64(n &* idStride)
                if idsNeeded > idsSize64 {
                    throw ErrorBuilder(.corruptedData, operation: "read_list")
                        .message("IDs section out of bounds for requested list")
                        .info("offset", "\(idsOff)")
                        .info("bytes", "\(n * idStride)")
                        .info("section_size", "\(idsSize)")
                        .build()
                }
                let ptr = UnsafeMutableRawPointer(mutating: idsBaseRaw).advanced(by: Int(idsOff)).assumingMemoryBound(to: UInt64.self)
                if format == .flat {
                    guard let (vecBaseRaw, vecSize) = mmap.sectionSlice(.vecs) else {
                        throw ErrorBuilder(.contractViolation, operation: "read_list")
                            .message("mmap vectors base pointer unavailable (internal error)")
                            .build()
                    }
                    let vecSize64 = UInt64(vecSize)
                    let vecBytes64 = UInt64(n) &* UInt64(d) &* UInt64(MemoryLayout<Float>.stride)
                    let vecNeeded = vecsOff &+ vecBytes64
                    if vecNeeded > vecSize64 {
                        throw ErrorBuilder(.corruptedData, operation: "read_list")
                            .message("Vecs section out of bounds for requested list")
                            .build()
                    }
                    let vptr = UnsafeMutableRawPointer(mutating: vecBaseRaw).advanced(by: Int(vecsOff)).assumingMemoryBound(to: Float.self)
                    return (n, UnsafePointer(ptr), nil, nil, UnsafePointer(vptr))
                } else {
                    guard let (codesBaseRaw, codesSize) = mmap.sectionSlice(.codes) else {
                        throw ErrorBuilder(.contractViolation, operation: "read_list")
                            .message("mmap codes base pointer unavailable (internal error)")
                            .build()
                    }
                    let codeStride = (format == .pq8) ? m : (m >> 1)
                    let codesSize64 = UInt64(codesSize)
                    let codeBytes64 = UInt64(n) &* UInt64(codeStride)
                    let codesNeeded = codesOff &+ codeBytes64
                    if codesNeeded > codesSize64 {
                        throw ErrorBuilder(.corruptedData, operation: "read_list")
                            .message("Codes section out of bounds for requested list")
                            .build()
                    }
                    let cptr = UnsafeMutableRawPointer(mutating: codesBaseRaw).advanced(by: Int(codesOff)).assumingMemoryBound(to: UInt8.self)
                    return (n, UnsafePointer(ptr), nil, UnsafePointer(cptr), nil)
                }
            } else if opts.id_bits == 32 {
                guard let (idsBaseRaw, idsSize) = mmap.sectionSlice(.ids) else {
                    throw ErrorBuilder(.contractViolation, operation: "read_list")
                        .message("mmap IDs base pointer unavailable (internal error)")
                        .build()
                }
                let idStride = idStrideBytes(opts)
                let idsSize64 = UInt64(idsSize)
                let idsNeeded = idsOff &+ UInt64(n &* idStride)
                if idsNeeded > idsSize64 {
                    throw ErrorBuilder(.corruptedData, operation: "read_list")
                        .message("IDs section out of bounds for requested list")
                        .build()
                    }
                let ptr = UnsafeMutableRawPointer(mutating: idsBaseRaw).advanced(by: Int(idsOff)).assumingMemoryBound(to: UInt32.self)
                if format == .flat {
                    guard let (vecBaseRaw, vecSize) = mmap.sectionSlice(.vecs) else {
                        throw ErrorBuilder(.contractViolation, operation: "read_list")
                            .message("mmap vectors base pointer unavailable (internal error)")
                            .build()
                    }
                    let vecSize64 = UInt64(vecSize)
                    let vecBytes64 = UInt64(n) &* UInt64(d) &* UInt64(MemoryLayout<Float>.stride)
                    let vecNeeded = vecsOff &+ vecBytes64
                    if vecNeeded > vecSize64 {
                        throw ErrorBuilder(.corruptedData, operation: "read_list")
                            .message("Vecs section out of bounds for requested list")
                            .build()
                    }
                    let vptr = UnsafeMutableRawPointer(mutating: vecBaseRaw).advanced(by: Int(vecsOff)).assumingMemoryBound(to: Float.self)
                    return (n, nil, UnsafePointer(ptr), nil, UnsafePointer(vptr))
                } else {
                    guard let (codesBaseRaw, codesSize) = mmap.sectionSlice(.codes) else {
                        throw ErrorBuilder(.contractViolation, operation: "read_list")
                            .message("mmap codes base pointer unavailable (internal error)")
                            .build()
                    }
                    let codeStride = (format == .pq8) ? m : (m >> 1)
                    let codesSize64 = UInt64(codesSize)
                    let codeBytes64 = UInt64(n) &* UInt64(codeStride)
                    let codesNeeded = codesOff &+ codeBytes64
                    if codesNeeded > codesSize64 {
                        throw ErrorBuilder(.corruptedData, operation: "read_list")
                            .message("Codes section out of bounds for requested list")
                            .build()
                    }
                    let cptr = UnsafeMutableRawPointer(mutating: codesBaseRaw).advanced(by: Int(codesOff)).assumingMemoryBound(to: UInt8.self)
                    return (n, nil, UnsafePointer(ptr), UnsafePointer(cptr), nil)
                }
            }
            throw ErrorBuilder(.invalidParameter, operation: "read_list")
                .message("Unsupported ID bit width")
                .info("id_bits", "\(opts.id_bits)")
                .info("supported_values", "32, 64")
                .build()
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
        throw ErrorBuilder(.invalidParameter, operation: "read_list")
            .message("Unsupported ID bit width")
            .info("id_bits", "\(opts.id_bits)")
            .info("supported_values", "32, 64")
            .build()
    }
}

@inline(__always) public func ivf_create(k_c: Int, m: Int, d: Int, opts: IVFAppendOpts? = nil) throws -> IVFListHandle {
    try IVFListHandle(k_c: k_c, m: m, d: d, opts: opts ?? .default)
}

@inline(__always) public func ivf_destroy(_ index: IVFListHandle) { /* ARC */ }

// Create an IVF handle attached to an existing mmap index for durable appends (internal - low-level mmap API).
@inline(__always)
internal func ivf_create_mmap(k_c: Int, m: Int, d: Int, mmap: IndexMmap, opts: IVFAppendOpts? = nil) throws -> IVFListHandle {
    let o = opts ?? .default
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
    let newCap = max(candidate1, candidate2)
    if newCap <= oldCap {
        throw ErrorBuilder(.capacityExceeded, operation: "calculate_capacity")
            .message("Capacity growth overflow")
            .info("old_capacity", "\(oldCap)")
            .info("needed", "\(need)")
            .build()
    }
    return newCap
}

private func growList(_ list: IVFList, codeBytesPerVec: Int, dFlat: Int, opts: IVFAppendOpts, index: IVFListHandle, minNewLen: Int) throws {
    let newCap = try safeNewCapacity(oldCap: list.capacity, need: minNewLen, opts: opts)
    let idBytes = newCap * idStrideBytes(opts)
    guard let newIDsRaw = (opts.allocator?.alloc(idBytes, 64) ?? alignedAlloc(idBytes, alignment: 64)) else {
        throw ErrorBuilder(.memoryExhausted, operation: "grow_list")
            .message("Failed to allocate ID storage during list growth")
            .info("requested_bytes", "\(idBytes)")
            .info("new_capacity", "\(newCap)")
            .build()
    }
    let newIDs: IDStorage = (opts.id_bits == 32) ? .u32(ptr: newIDsRaw.bindMemory(to: UInt32.self, capacity: newCap)) : .u64(ptr: newIDsRaw.bindMemory(to: UInt64.self, capacity: newCap))
    var newCodesRaw: UnsafeMutableRawPointer?; var newCodes: UnsafeMutablePointer<UInt8>?; var newXBRaw: UnsafeMutableRawPointer?; var newXB: UnsafeMutablePointer<Float>?
    switch index.format {
    case .pq8, .pq4:
        let bytes = newCap * codeBytesPerVec
        guard let c = (opts.allocator?.alloc(bytes, 1) ?? malloc(bytes)) else {
            (opts.allocator?.free(newIDsRaw) ?? alignedFree(newIDsRaw))
            throw ErrorBuilder(.memoryExhausted, operation: "grow_list")
                .message("Failed to allocate code storage during list growth")
                .info("requested_bytes", "\(bytes)")
                .info("new_capacity", "\(newCap)")
                .build()
        }
        newCodesRaw = c; newCodes = c.bindMemory(to: UInt8.self, capacity: bytes)
    case .flat:
        let elems = newCap * dFlat; let byteCount = elems * MemoryLayout<Float>.stride
        guard let xraw = (opts.allocator?.alloc(byteCount, 64) ?? alignedAlloc(byteCount, alignment: 64)) else {
            (opts.allocator?.free(newIDsRaw) ?? alignedFree(newIDsRaw))
            throw ErrorBuilder(.memoryExhausted, operation: "grow_list")
                .message("Failed to allocate vector storage during list growth")
                .info("requested_bytes", "\(byteCount)")
                .info("new_capacity", "\(newCap)")
                .build()
        }
        newXBRaw = xraw; newXB = xraw.bindMemory(to: Float.self, capacity: elems)
    }
    var newTSRaw: UnsafeMutableRawPointer?; var newTS: UnsafeMutablePointer<UInt64>?
    if opts.timestamps {
        let bytes = newCap * MemoryLayout<UInt64>.stride
        guard let t = (opts.allocator?.alloc(bytes, 64) ?? alignedAlloc(bytes, alignment: 64)) else {
            (opts.allocator?.free(newIDsRaw) ?? alignedFree(newIDsRaw))
            if let r = newCodesRaw { (opts.allocator?.free(r) ?? free(r)) }
            if let r = newXBRaw { (opts.allocator?.free(r) ?? alignedFree(r)) }
            throw ErrorBuilder(.memoryExhausted, operation: "grow_list")
                .message("Failed to allocate timestamp storage during list growth")
                .info("requested_bytes", "\(bytes)")
                .info("new_capacity", "\(newCap)")
                .build()
        }
        newTSRaw = t; newTS = t.bindMemory(to: UInt64.self, capacity: newCap)
    }
    let n = list.length
    if n > 0 {
        // Copy existing IDs to new storage
        // This validates that old and new storage types match opts.id_bits
        switch (list.ids, newIDs) {
        case (.u64(let old), .u64(let neu)):
            memcpy(neu, old, n * MemoryLayout<UInt64>.stride)
        case (.u32(let old), .u32(let neu)):
            memcpy(neu, old, n * MemoryLayout<UInt32>.stride)
        default:
            // Internal inconsistency: storage types don't match
            // This indicates opts.id_bits was modified after list creation,
            // memory corruption, or a bug in storage initialization
            var oldBits: Int
            var newBits: Int
            if case .u32 = list.ids {
                oldBits = 32
            } else {
                oldBits = 64
            }
            if case .u32 = newIDs {
                newBits = 32
            } else {
                newBits = 64
            }
            throw ErrorBuilder(.internalInconsistency, operation: "grow_list")
                .message("ID storage type mismatch during list growth")
                .info("old_id_bits", "\(oldBits)")
                .info("new_id_bits", "\(newBits)")
                .info("expected_bits", "\(opts.id_bits)")
                .info("list_length", "\(n)")
                .info("old_capacity", "\(list.capacity)")
                .info("new_capacity", "\(newCap)")
                .build()
        }

        // Copy existing codes or vectors
        switch index.format {
        case .pq8, .pq4:
            if let old = list.codes, let neu = newCodes {
                memcpy(neu, old, n * codeBytesPerVec)
            }
        case .flat:
            if let old = list.xb, let neu = newXB {
                memcpy(neu, old, n * dFlat * MemoryLayout<Float>.stride)
            }
        }

        // Copy existing timestamps if enabled
        if opts.timestamps, let old = list.ts, let neu = newTS {
            memcpy(neu, old, n * MemoryLayout<UInt64>.stride)
        }
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
    guard n >= 0, m == index.m, index.format == .pq8 || index.format == .pq4 else {
        throw ErrorBuilder(.unsupportedLayout, operation: "ivf_append")
            .message("ivf_append requires PQ format with matching m parameter")
            .info("format", "\(index.format)")
            .info("expected_m", "\(index.m)")
            .info("provided_m", "\(m)")
            .info("n", "\(n)")
            .build()
    }
    let opts = inOpts ?? index.opts
    guard opts.group == 4 || opts.group == 8, m % opts.group == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "ivf_append")
            .message("Invalid group size or m not divisible by group")
            .info("group", "\(opts.group)")
            .info("m", "\(m)")
            .info("valid_groups", "4, 8")
            .build()
    }
    if opts.durable {
        guard index.storage == .mmap, let mmap = index.mmapHandle else {
            throw ErrorBuilder(.missingDependency, operation: "ivf_append")
                .message("Durable mode requires mmap backend")
                .info("storage", "\(index.storage)")
                .info("has_mmap_handle", "\(index.mmapHandle != nil)")
                .build()
        }
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
            guard let idsBuf = malloc(idsBytes) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_append_durable")
                    .message("Failed to allocate temporary ID buffer for durable append")
                    .info("requested_bytes", "\(idsBytes)")
                    .info("list_id", "\(listID)")
                    .build()
            }
            defer { free(idsBuf) }
            guard let codesBuf = malloc(codesBytes) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_append_durable")
                    .message("Failed to allocate temporary code buffer for durable append")
                    .info("requested_bytes", "\(codesBytes)")
                    .info("list_id", "\(listID)")
                    .build()
            }
            defer { free(codesBuf) }
            // Fill buffers
            if index.opts.id_bits == 32 {
                let p = idsBuf.bindMemory(to: UInt32.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() {
                    let v = external_ids[srcIdx]
                    if v > UInt64(UInt32.max) {
                        throw ErrorBuilder(.invalidParameter, operation: "ivf_append_durable")
                            .message("ID value exceeds 32-bit maximum")
                            .info("id_value", "\(v)")
                            .info("max_value", "\(UInt32.max)")
                            .info("index", "\(srcIdx)")
                            .build()
                    }
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
            if res.idsStride != idStride || res.codesStride != codesStride {
                throw ErrorBuilder(.unsupportedLayout, operation: "ivf_append_durable")
                    .message("Stride mismatch between index and mmap container")
                    .info("expected_ids_stride", "\(idStride)")
                    .info("actual_ids_stride", "\(res.idsStride)")
                    .info("expected_codes_stride", "\(codesStride)")
                    .info("actual_codes_stride", "\(res.codesStride)")
                    .build()
            }
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
    for i in 0..<n {
        let lid = Int(list_ids[i])
        if lid < 0 || lid >= index.k_c {
            throw ErrorBuilder(.invalidRange, operation: "ivf_append")
                .message("List ID out of valid range")
                .info("list_id", "\(lid)")
                .info("valid_range", "0..<\(index.k_c)")
                .info("vector_index", "\(i)")
                .build()
        }
    }
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
            if let codesDstBase = codesDstBase {
                let dst = codesDstBase.advanced(by: localIdx * codeBytesPerVec)
                let src = codes.advanced(by: srcIdx * srcCodeStride)
                switch index.format {
                case .pq8:
                    memcpy(dst, src, m)
                case .pq4:
                    if opts.pack4_unpacked {
                        packNibblesU4(idx4: src, n: m, out: dst)
                    } else {
                        memcpy(dst, src, m >> 1)
                    }
                case .flat:
                    break
                }
            }
        }
        if index.opts.timestamps, let tsPtr = L.ts { let now = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0); for i in 0..<batch.count { tsPtr[startPos + i] = now } }
        L.length = newLen
        switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break }
    }
}

public func ivf_append_one_list(list_id: Int32, external_ids: UnsafePointer<UInt64>, codes: UnsafePointer<UInt8>, n: Int, m: Int, index: IVFListHandle, opts inOpts: IVFAppendOpts?, internalIDsOut: UnsafeMutablePointer<Int64>?) throws {
    guard n >= 0, m == index.m, index.format == .pq8 || index.format == .pq4 else {
        throw ErrorBuilder(.unsupportedLayout, operation: "ivf_append_one_list")
            .message("ivf_append_one_list requires PQ format with matching m parameter")
            .info("format", "\(index.format)")
            .info("expected_m", "\(index.m)")
            .info("provided_m", "\(m)")
            .info("n", "\(n)")
            .build()
    }
    guard list_id >= 0 && list_id < Int32(index.k_c) else {
        throw ErrorBuilder(.invalidRange, operation: "ivf_append_one_list")
            .message("List ID out of valid range")
            .info("list_id", "\(list_id)")
            .info("valid_range", "0..<\(index.k_c)")
            .build()
    }
    let opts = inOpts ?? index.opts
    if opts.durable {
        throw ErrorBuilder(.missingDependency, operation: "ivf_append_one_list")
            .message("Durable mode not supported in single-list append (use ivf_append)")
            .build()
    }
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
    guard index.format == .flat, d == index.d else {
        throw ErrorBuilder(.dimensionMismatch, operation: "ivf_append_flat")
            .message("ivf_append_flat requires flat format with matching dimension")
            .info("format", "\(index.format)")
            .info("expected_d", "\(index.d)")
            .info("provided_d", "\(d)")
            .build()
    }
    let opts = inOpts ?? index.opts
    if opts.durable {
        guard index.storage == .mmap, let mmap = index.mmapHandle else {
            throw ErrorBuilder(.missingDependency, operation: "ivf_append_flat")
                .message("Durable mode requires mmap backend")
                .info("storage", "\(index.storage)")
                .info("has_mmap_handle", "\(index.mmapHandle != nil)")
                .build()
        }
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
            guard let idsBuf = malloc(idsBytes) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_append_flat_durable")
                    .message("Failed to allocate temporary ID buffer for durable append")
                    .info("requested_bytes", "\(idsBytes)")
                    .info("list_id", "\(listID)")
                    .build()
            }
            defer { free(idsBuf) }
            guard let vecBuf = malloc(vecBytes) else {
                throw ErrorBuilder(.memoryExhausted, operation: "ivf_append_flat_durable")
                    .message("Failed to allocate temporary vector buffer for durable append")
                    .info("requested_bytes", "\(vecBytes)")
                    .info("list_id", "\(listID)")
                    .build()
            }
            defer { free(vecBuf) }
            if index.opts.id_bits == 32 {
                let p = idsBuf.bindMemory(to: UInt32.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() {
                    let v = external_ids[srcIdx]
                    if v > UInt64(UInt32.max) {
                        throw ErrorBuilder(.invalidParameter, operation: "ivf_append_flat_durable")
                            .message("ID value exceeds 32-bit maximum")
                            .info("id_value", "\(v)")
                            .info("max_value", "\(UInt32.max)")
                            .info("index", "\(srcIdx)")
                            .build()
                    }
                    p[j] = UInt32(truncatingIfNeeded: v)
                }
            } else {
                let p = idsBuf.bindMemory(to: UInt64.self, capacity: count)
                for (j, srcIdx) in localIndices.enumerated() { p[j] = external_ids[srcIdx] }
            }
            let vdst = vecBuf.bindMemory(to: UInt8.self, capacity: vecBytes)
            for (j, srcIdx) in localIndices.enumerated() {
                memcpy(vdst.advanced(by: j * vecStride), xb.advanced(by: srcIdx * d), vecStride)
            }
            let res = try mmap.mmap_append_begin(listID: listID, addLen: count)
            if res.idsStride != idStride || res.vecsStride != vecStride {
                throw ErrorBuilder(.unsupportedLayout, operation: "ivf_append_flat_durable")
                    .message("Stride mismatch between index and mmap container")
                    .info("expected_ids_stride", "\(idStride)")
                    .info("actual_ids_stride", "\(res.idsStride)")
                    .info("expected_vecs_stride", "\(vecStride)")
                    .info("actual_vecs_stride", "\(res.vecsStride)")
                    .build()
            }
            try mmap.mmap_append_commit(res, idsSrc: idsBuf, codesSrc: nil, vecsSrc: vecBuf)
            if let out = internalIDsOut { for srcIdx in localIndices { out[srcIdx] = baseInternal + Int64(srcIdx) } }
            let L = index.lists[listID]; L.length += count; if L.capacity < L.length { L.capacity = L.length }
        }
        return
    }
    for i in 0..<n {
        let lid = Int(list_ids[i])
        if lid < 0 || lid >= index.k_c {
            throw ErrorBuilder(.invalidRange, operation: "ivf_append_flat")
                .message("List ID out of valid range")
                .info("list_id", "\(lid)")
                .info("valid_range", "0..<\(index.k_c)")
                .info("vector_index", "\(i)")
                .build()
        }
    }
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

/// Inserts vectors at a specified position in a PQ-format IVF list
///
/// This function is specifically for PQ8 and PQ4 format indices. For flat format
/// indices, use `ivf_insert_at_flat()` instead.
///
/// - Important: This function requires PQ8 or PQ4 format. Attempting to use it
///              with a flat format index will result in an error.
///
/// - Parameters:
///   - list_id: Target list index (0..<k_c)
///   - pos: Position to insert at (0...list.length)
///   - external_ids: External IDs for inserted vectors
///   - codes: Quantized codes (PQ8: n×m bytes, PQ4: n×(m/2) bytes)
///   - n: Number of vectors to insert
///   - index: IVF list handle
///
/// - Throws:
///   - `VectorIndexError(.unsupportedLayout)`: If index format is flat (use ivf_insert_at_flat)
///   - `VectorIndexError(.internalInconsistency)`: If internal storage corruption detected
///   - `VectorIndexError(.invalidRange)`: If list_id out of range or position invalid
///   - `VectorIndexError(.contractViolation)`: If n < 0
///   - `VectorIndexError(.missingDependency)`: If durable mode not properly configured
///   - `VectorIndexError(.memoryExhausted)`: If capacity increase fails
public func ivf_insert_at(list_id: Int32, pos: Int, external_ids: UnsafePointer<UInt64>, codes: UnsafePointer<UInt8>, n: Int, index: IVFListHandle) throws {
    // Validate format compatibility
    guard index.format == .pq8 || index.format == .pq4 else {
        throw ErrorBuilder(.unsupportedLayout, operation: "ivf_insert_at")
            .message("ivf_insert_at requires PQ format; use ivf_insert_at_flat for flat format")
            .info("actual_format", "\(index.format)")
            .build()
    }

    guard list_id >= 0 && list_id < Int32(index.k_c) else {
        throw ErrorBuilder(.invalidRange, operation: "ivf_insert_at")
            .message("List ID out of valid range")
            .info("list_id", "\(list_id)")
            .info("valid_range", "0..<\(index.k_c)")
            .build()
    }
    guard n >= 0 else {
        throw ErrorBuilder(.contractViolation, operation: "ivf_insert_at")
            .message("Invalid vector count (must be non-negative)")
            .info("n", "\(n)")
            .build()
    }
    if index.opts.durable {
        throw ErrorBuilder(.missingDependency, operation: "ivf_insert_at")
            .message("Durable mode not supported for insert operations")
            .build()
    }
    let L = index.lists[Int(list_id)]
    switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
    defer { switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break } }
    guard pos >= 0 && pos <= L.length else {
        throw ErrorBuilder(.invalidRange, operation: "ivf_insert_at")
            .message("Insert position out of valid range")
            .info("position", "\(pos)")
            .info("valid_range", "0...\(L.length)")
            .info("list_length", "\(L.length)")
            .build()
    }
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
        case .flat:
            // Unreachable: format validation at function entry ensures only PQ formats reach here
            break
        }
        if index.opts.timestamps, let tsPtr = L.ts { tsPtr[pos + i] = UInt64(Date().timeIntervalSince1970 * 1_000_000_000.0) }
    }
    L.length = newLen
}

public func ivf_insert_at_flat(list_id: Int32, pos: Int, external_ids: UnsafePointer<UInt64>, xb: UnsafePointer<Float>, n: Int, index: IVFListHandle) throws {
    guard index.format == .flat else {
        throw ErrorBuilder(.unsupportedLayout, operation: "ivf_insert_at_flat")
            .message("ivf_insert_at_flat requires flat format; use ivf_insert_at for PQ format")
            .info("actual_format", "\(index.format)")
            .build()
    }
    guard list_id >= 0 && list_id < Int32(index.k_c) else {
        throw ErrorBuilder(.invalidRange, operation: "ivf_insert_at_flat")
            .message("List ID out of valid range")
            .info("list_id", "\(list_id)")
            .info("valid_range", "0..<\(index.k_c)")
            .build()
    }
    guard n >= 0 else {
        throw ErrorBuilder(.contractViolation, operation: "ivf_insert_at_flat")
            .message("Invalid vector count (must be non-negative)")
            .info("n", "\(n)")
            .build()
    }
    if index.opts.durable {
        throw ErrorBuilder(.missingDependency, operation: "ivf_insert_at_flat")
            .message("Durable mode not supported for insert operations")
            .build()
    }
    let L = index.lists[Int(list_id)]
    switch index.opts.concurrency { case .perListMultiWriter: L.lock.lock(); case .globalMultiWriter: index.globalLock.lock(); case .singleWriter: break }
    defer { switch index.opts.concurrency { case .perListMultiWriter: L.lock.unlock(); case .globalMultiWriter: index.globalLock.unlock(); case .singleWriter: break } }
    guard pos >= 0 && pos <= L.length else {
        throw ErrorBuilder(.invalidRange, operation: "ivf_insert_at_flat")
            .message("Insert position out of valid range")
            .info("position", "\(pos)")
            .info("valid_range", "0...\(L.length)")
            .info("list_length", "\(L.length)")
            .build()
    }
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
