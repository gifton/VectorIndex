//
//  VIndexMmap.swift
//
//  S1 — Serialization & Mmap Layout (Swift)
//
//  Implements:
//    - Header (256 B) + TOC parsing with CRC32
//    - Section table + typed accessors
//    - List descriptors and zero-copy pointers to IDs/Codes/Vecs
//    - Endianness handling for header/TOC/descriptors
//    - mmap open/close with alignment checks
//    - Durable append protocol: WAL (write-ahead log), begin/commit, replay
//    - msync + release/acquire ordering for lock-free readers
//

import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import CAtomicsShim

@inline(__always) private func alignUp(_ x: UInt64, _ a: UInt64) -> UInt64 {
    let m = a &- 1
    return (x &+ m) & ~m
}

public enum Endian: UInt8 { case little = 1, big = 2 }

@inline(__always) private func hostIsLittleEndian() -> Bool {
    var x: UInt16 = 0x0102
    return withUnsafeBytes(of: &x) { $0[0] == 0x02 }
}

@inline(__always) private func toHost<T: FixedWidthInteger>(_ v: T, fileEndian: Endian) -> T {
    ((fileEndian == .little) == hostIsLittleEndian()) ? v : v.byteSwapped
}

@inline(__always) private func fromHost<T: FixedWidthInteger>(_ v: T, fileEndian: Endian) -> T {
    ((fileEndian == .little) == hostIsLittleEndian()) ? v : v.byteSwapped
}

private struct CRC32 {
    private static let table: [UInt32] = {
        (0..<256).map { i -> UInt32 in
            var c = UInt32(i)
            for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
            return c
        }
    }()
    @inline(__always) static func hash(_ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        let p = data.bindMemory(to: UInt8.self, capacity: len)
        for i in 0..<len {
            c = CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
        }
        return c ^ 0xFFFF_FFFF
    }
}

// magic: "VINDEX\0\0" (LE)
private let VINDEX_MAGIC = UInt64(0x00585845444E4956)

internal enum SectionType: UInt32, Sendable {
    case centroids = 1, codebooks = 2, centroidNorms = 3, listsDesc = 4
    case ids = 5, codes = 6, vecs = 7, normsInv = 8, normsSq = 9
    case idMap = 10, tombstones = 11, telemetry = 12, freeList = 13, walAnchor = 14
}

internal struct ListDesc {
    var format: UInt8
    var group_g: UInt8
    var id_bits: UInt8
    var reserved0: UInt8
    var length: UInt32
    var capacity: UInt32
    var ids_offset: UInt64
    var codes_offset: UInt64
    var vecs_offset: UInt64
    var ids_stride: UInt32
    var codes_stride: UInt32
    var vecs_stride: UInt32
    var reserved1: UInt32

    @inline(__always) func lengthHost(_ e: Endian) -> Int { Int(toHost(length, fileEndian: e)) }
    @inline(__always) func capacityHost(_ e: Endian) -> Int { Int(toHost(capacity, fileEndian: e)) }
    @inline(__always) func idsOffsetHost(_ e: Endian) -> UInt64 { toHost(ids_offset, fileEndian: e) }
    @inline(__always) func codesOffsetHost(_ e: Endian) -> UInt64 { toHost(codes_offset, fileEndian: e) }
    @inline(__always) func vecsOffsetHost(_ e: Endian) -> UInt64 { toHost(vecs_offset, fileEndian: e) }
    @inline(__always) func idsStrideHost(_ e: Endian) -> Int { Int(toHost(ids_stride, fileEndian: e)) }
    @inline(__always) func codesStrideHost(_ e: Endian) -> Int { Int(toHost(codes_stride, fileEndian: e)) }
    @inline(__always) func vecsStrideHost(_ e: Endian) -> Int { Int(toHost(vecs_stride, fileEndian: e)) }
}

private struct TOCEntry {
    var type: UInt32
    var offset: UInt64
    var size: UInt64
    var align: UInt32
    var flags: UInt32
    var crc32: UInt32
    var reserved: UInt32
    @inline(__always) func typeHost(_ e: Endian) -> SectionType? { SectionType(rawValue: toHost(type, fileEndian: e)) }
    @inline(__always) func offsetHost(_ e: Endian) -> UInt64 { toHost(offset, fileEndian: e) }
    @inline(__always) func sizeHost(_ e: Endian) -> UInt64 { toHost(size, fileEndian: e) }
    @inline(__always) func alignHost(_ e: Endian) -> UInt32 { toHost(align, fileEndian: e) }
    @inline(__always) func flagsHost(_ e: Endian) -> UInt32 { toHost(flags, fileEndian: e) }
    @inline(__always) func crcHost(_ e: Endian) -> UInt32 { toHost(crc32, fileEndian: e) }
}

private struct VIndexHeader {
    var magic: UInt64
    var version_major: UInt16
    var version_minor: UInt16
    var endianness: UInt8
    var arch: UInt8
    var flags: UInt32
    var d: UInt32
    var m: UInt16
    var ks: UInt16
    var kc: UInt32
    var id_bits: UInt8
    var code_group_g: UInt8
    var reservedA: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
    var N_total: UInt64
    var generation: UInt64
    var toc_offset: UInt64
    var toc_entries: UInt32
    var header_crc32: UInt32
    var reservedRest: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64)

    func fileEndian() -> Endian { Endian(rawValue: endianness) ?? .little }
    func magicOK(_ e: Endian) -> Bool { toHost(magic, fileEndian: e) == VINDEX_MAGIC }
}

@inline(__always) private func computeHeaderCRC(_ raw: UnsafeRawPointer) -> UInt32 {
    // Copy header and zero the CRC field using struct field access (same as builder)
    var buf = [UInt8](repeating: 0, count: 256)
    memcpy(&buf, raw, 256)
    // Zero the CRC field at its actual offset (68-71) via struct overlay
    return buf.withUnsafeMutableBytes { bufPtr in
        let hdrPtr = bufPtr.baseAddress!.assumingMemoryBound(to: VIndexHeader.self)
        hdrPtr.pointee.header_crc32 = 0
        return CRC32.hash(bufPtr.baseAddress!, 256)
    }
}

internal struct MmapOpts {
    public var readOnly: Bool = true
    public var verifyCRCs: Bool = true
    public var adviseSequential: Bool = true
    public var adviseWillNeed: Bool = false
    public init() {}
}

internal struct AppendReservation {
    public let listID: Int
    public let oldLen: Int
    public let addLen: Int
    public let idsFileOffset: UInt64
    public let codesFileOffset: UInt64
    public let vecsFileOffset: UInt64
    let idsStride: Int
    let codesStride: Int
    let vecsStride: Int
}

internal enum VIndexError: Error {
    case openFailed(errno: Int32)
    case statFailed(errno: Int32)
    case mmapFailed(errno: Int32)
    case badHeader
    case badCRC
    case unknownSection
    case unsupportedEndianness
    case misalignedSection(expected: UInt64, got: UInt64)
    case cannotGrowSection(SectionType)
    case badListID
    case insufficientCapacity
    case walIOFailed(errno: Int32)
}

internal final class IndexMmap {
    public let path: String
    public let fd: Int32
    public private(set) var fileSize: UInt64
    public let opts: MmapOpts

    private var base: UnsafeMutableRawPointer
    private let prot: Int32
    private let mapFlags: Int32

    private let header: VIndexHeader
    private let fileEndian: Endian
    private var toc: UnsafePointer<TOCEntry>
    private let tocCount: Int

    private(set) public var kc: Int = 0
    private(set) public var d: Int = 0
    private(set) public var m: Int = 0
    private(set) public var ks: Int = 0
    private(set) public var N_total: Int = 0
    private(set) public var codeGroupG: Int = 0
    private(set) public var idBits: Int = 64

    private var secCentroids: UnsafePointer<Float>?
    private var secCodebooks: UnsafePointer<Float>?
    private var secCentroidNorms: UnsafePointer<Float>?
    private var secListsDesc: UnsafeMutablePointer<ListDesc>?
    private var secIDs: UnsafeMutableRawPointer?
    private var secCodes: UnsafeMutableRawPointer?
    private var secVecs: UnsafeMutableRawPointer?
    private var secNormsInv: UnsafeRawPointer?
    private var secNormsSq: UnsafeRawPointer?
    private var secIDMap: UnsafeRawPointer?
    private var secTombstones: UnsafeRawPointer?

    private var tocByType: [SectionType: TOCEntry] = [:]
    private var tailIDs: UInt64 = 0
    private var tailCodes: UInt64 = 0
    private var tailVecs: UInt64 = 0

    private var walFD: Int32 = -1
    private var walPath: String

    public static func open(path: String, opts: MmapOpts = .init()) throws -> IndexMmap {
        let flags = opts.readOnly ? O_RDONLY : O_RDWR
        let fd = Darwin.open(path, flags | O_CLOEXEC)
        guard fd >= 0 else { throw VIndexError.openFailed(errno: errno) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { let err = errno; Darwin.close(fd); throw VIndexError.statFailed(errno: err) }
        let fileSize = UInt64(st.st_size)
        guard fileSize >= 4096 else { let err = errno; Darwin.close(fd); throw VIndexError.badHeader }
        let prot: Int32 = opts.readOnly ? PROT_READ : (PROT_READ | PROT_WRITE)
        let mapFlags: Int32 = MAP_FILE | MAP_SHARED
        let base = mmap(nil, Int(fileSize), prot, mapFlags, fd, 0)
        guard base != MAP_FAILED else { let err = errno; Darwin.close(fd); throw VIndexError.mmapFailed(errno: err) }

        let hdrPtr = base!.bindMemory(to: VIndexHeader.self, capacity: 1)
        let hdr = hdrPtr.pointee
        let fileEndian = hdr.fileEndian()
        guard fileEndian == .little || fileEndian == .big else {
            munmap(base, Int(fileSize)); Darwin.close(fd); throw VIndexError.unsupportedEndianness
        }
        guard hdr.magicOK(fileEndian) else { munmap(base, Int(fileSize)); Darwin.close(fd); throw VIndexError.badHeader }
        if opts.verifyCRCs {
            let calc = computeHeaderCRC(UnsafeRawPointer(hdrPtr))
            let stored = toHost(hdr.header_crc32, fileEndian: fileEndian)
            guard calc == stored else { munmap(base, Int(fileSize)); Darwin.close(fd); throw VIndexError.badCRC }
        }
        let tocOffset = toHost(hdr.toc_offset, fileEndian: fileEndian)
        let tocEntries = Int(toHost(hdr.toc_entries, fileEndian: fileEndian))
        let tocPtr = UnsafeRawPointer(base!).advanced(by: Int(tocOffset)).assumingMemoryBound(to: TOCEntry.self)

        let idx = IndexMmap(path: path, fd: fd, fileSize: fileSize, opts: opts, base: base!, prot: prot, mapFlags: mapFlags, header: hdr, fileEndian: fileEndian, toc: tocPtr, tocCount: tocEntries)
        try idx.indexInit()
        return idx
    }

    private init(path: String, fd: Int32, fileSize: UInt64, opts: MmapOpts, base: UnsafeMutableRawPointer, prot: Int32, mapFlags: Int32, header: VIndexHeader, fileEndian: Endian, toc: UnsafePointer<TOCEntry>, tocCount: Int) {
        self.path = path
        self.fd = fd
        self.fileSize = fileSize
        self.opts = opts
        self.base = base
        self.prot = prot
        self.mapFlags = mapFlags
        self.header = header
        self.fileEndian = fileEndian
        self.toc = toc
        self.tocCount = tocCount
        self.kc = Int(toHost(header.kc, fileEndian: fileEndian))
        self.d  = Int(toHost(header.d,  fileEndian: fileEndian))
        self.m  = Int(toHost(header.m,  fileEndian: fileEndian))
        self.ks = Int(toHost(header.ks, fileEndian: fileEndian))
        self.N_total = Int(toHost(header.N_total, fileEndian: fileEndian))
        self.codeGroupG = Int(header.code_group_g)
        self.idBits = Int(header.id_bits)
        self.walPath = path + ".wal"
    }

    deinit { try? close() }

    // Expose file endianness to callers that need to decode descriptor fields
    public var fileEndianness: Endian { fileEndian }

    public func close() throws {
        if walFD >= 0 { _ = Darwin.close(walFD); walFD = -1 }
        _ = msync(base, Int(fileSize), MS_SYNC)
        _ = munmap(base, Int(fileSize))
        _ = Darwin.close(fd)
    }

    private func slice(_ e: TOCEntry) -> UnsafeMutableRawPointer {
        let off = Int(e.offsetHost(fileEndian))
        return UnsafeMutableRawPointer(base).advanced(by: off)
    }

    private func mapSection(_ ty: SectionType) -> TOCEntry? { tocByType[ty] }

    private func indexInit() throws {
        for i in 0..<tocCount {
            let te = toc.advanced(by: i).pointee
            guard let ty = te.typeHost(fileEndian) else { throw VIndexError.unknownSection }
            tocByType[ty] = te
            let off = te.offsetHost(fileEndian)
            let al  = UInt64(te.alignHost(fileEndian))
            if al != 0 && (off % UInt64(al)) != 0 { throw VIndexError.misalignedSection(expected: UInt64(al), got: off % UInt64(al)) }
            if opts.verifyCRCs && te.sizeHost(fileEndian) > 0 {
                let p = slice(te)
                let crc = CRC32.hash(p, Int(te.sizeHost(fileEndian)))
                let stored = te.crcHost(fileEndian)
                guard crc == stored else { throw VIndexError.badCRC }
            }
        }
        if let e = mapSection(.centroids) { secCentroids = UnsafePointer(slice(e).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.codebooks) { secCodebooks = UnsafePointer(slice(e).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.centroidNorms) { secCentroidNorms = UnsafePointer(slice(e).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.listsDesc) { secListsDesc = slice(e).assumingMemoryBound(to: ListDesc.self) }
        if let e = mapSection(.ids) { secIDs = slice(e) }
        if let e = mapSection(.codes) { secCodes = slice(e) }
        if let e = mapSection(.vecs) { secVecs = slice(e) }
        if let e = mapSection(.normsInv) { secNormsInv = UnsafeRawPointer(slice(e)) }
        if let e = mapSection(.normsSq) { secNormsSq = UnsafeRawPointer(slice(e)) }
        if let e = mapSection(.idMap) { secIDMap = UnsafeRawPointer(slice(e)) }
        if let e = mapSection(.tombstones) { secTombstones = UnsafeRawPointer(slice(e)) }

        if let descs = secListsDesc {
            var idsMax: UInt64 = 0, codesMax: UInt64 = 0, vecsMax: UInt64 = 0
            for i in 0..<kc {
                let dsc = descs.advanced(by: i).pointee
                let cap  = UInt64(dsc.capacityHost(fileEndian))
                let idsOff   = dsc.idsOffsetHost(fileEndian)
                let codesOff = dsc.codesOffsetHost(fileEndian)
                let vecsOff  = dsc.vecsOffsetHost(fileEndian)
                let idsStride = UInt64(dsc.idsStrideHost(fileEndian))
                let codesStride = UInt64(dsc.codesStrideHost(fileEndian))
                let vecsStride = UInt64(dsc.vecsStrideHost(fileEndian))
                idsMax   = max(idsMax,   idsOff   &+ cap &* idsStride)
                codesMax = max(codesMax, codesOff &+ cap &* codesStride)
                vecsMax  = max(vecsMax,  vecsOff  &+ cap &* vecsStride)
            }
            tailIDs = idsMax; tailCodes = codesMax; tailVecs = vecsMax
        }

        if opts.adviseSequential { _ = posix_madvise(base, Int(fileSize), POSIX_MADV_SEQUENTIAL) }
        if opts.adviseWillNeed { _ = posix_madvise(base, Int(fileSize), POSIX_MADV_WILLNEED) }

        if !opts.readOnly {
            walFD = Darwin.open(walPath, O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR)
            if walFD < 0 { throw VIndexError.walIOFailed(errno: errno) }
        }
    }

    public func mmapCentroids() -> (ptr: UnsafePointer<Float>, kc: Int, d: Int)? {
        guard let p = secCentroids else { return nil }
        return (p, kc, d)
    }
    public func mmapCodebooks() -> (ptr: UnsafePointer<Float>, m: Int, ks: Int, dsub: Int)? {
        guard let p = secCodebooks else { return nil }
        let dsub = d / max(m, 1)
        return (p, m, ks, dsub)
    }
    public func mmapLists() -> (ptr: UnsafeMutablePointer<ListDesc>, kc: Int)? {
        guard let p = secListsDesc else { return nil }
        return (p, kc)
    }
    public func idsBase() -> UnsafeMutableRawPointer? { secIDs }
    public func codesBase() -> UnsafeMutableRawPointer? { secCodes }
    public func vecsBase() -> UnsafeMutableRawPointer? { secVecs }

    // Return a raw pointer and size for a section if present
    public func sectionSlice(_ ty: SectionType) -> (ptr: UnsafeRawPointer, size: Int)? {
        guard let e = tocByType[ty] else { return nil }
        let size = Int(e.sizeHost(fileEndian))
        let p: UnsafeRawPointer
        switch ty {
        case .centroids: guard let s = secCentroids else { return nil }; p = UnsafeRawPointer(s)
        case .codebooks: guard let s = secCodebooks else { return nil }; p = UnsafeRawPointer(s)
        case .centroidNorms: guard let s = secCentroidNorms else { return nil }; p = UnsafeRawPointer(s)
        case .listsDesc: guard let s = secListsDesc else { return nil }; p = UnsafeRawPointer(s)
        case .ids: guard let s = secIDs else { return nil }; p = UnsafeRawPointer(s)
        case .codes: guard let s = secCodes else { return nil }; p = UnsafeRawPointer(s)
        case .vecs: guard let s = secVecs else { return nil }; p = UnsafeRawPointer(s)
        case .normsInv: guard let s = secNormsInv else { return nil }; p = s
        case .normsSq: guard let s = secNormsSq else { return nil }; p = s
        case .idMap: guard let s = secIDMap else { return nil }; p = s
        case .tombstones: guard let s = secTombstones else { return nil }; p = s
        default:
            return nil
        }
        return (p, size)
    }

    // Read IDMap blob (serialized) if present
    public func readIDMapBlob() -> Data? {
        guard let (p, sz) = sectionSlice(.idMap), sz > 0 else { return nil }
        return Data(bytes: p, count: sz)
    }

    // Write IDMap blob into existing section (size must fit). Updates CRC in TOC.
    public func writeIDMapBlob(_ blob: Data) throws {
        guard !opts.readOnly else { throw VIndexError.walIOFailed(errno: EROFS) }
        guard let e = tocByType[.idMap] else { throw VIndexError.unknownSection }
        let maxSize = Int(e.sizeHost(fileEndian))
        guard blob.count <= maxSize else { throw VIndexError.cannotGrowSection(.idMap) }
        guard let basePtr = secIDMap else { throw VIndexError.unknownSection }
        blob.withUnsafeBytes { src in
            memcpy(UnsafeMutableRawPointer(mutating: basePtr), src.baseAddress!, blob.count)
            if maxSize > blob.count {
                memset(UnsafeMutableRawPointer(mutating: basePtr).advanced(by: blob.count), 0, maxSize - blob.count)
            }
        }
        _ = msync(UnsafeMutableRawPointer(mutating: basePtr), maxSize, MS_SYNC)
        // Update CRC in TOC entry in-place
        try updateSectionCRC(.idMap)
    }

    private func updateSectionCRC(_ ty: SectionType) throws {
        // Find toc entry index
        var idxFound: Int? = nil
        for i in 0..<tocCount {
            let te = toc.advanced(by: i).pointee
            if let t = te.typeHost(fileEndian), t == ty {
                idxFound = i; break
            }
        }
        guard let i = idxFound else { throw VIndexError.unknownSection }
        var te = toc.advanced(by: i).pointee
        let p = slice(te)
        let sz = Int(te.sizeHost(fileEndian))
        let newCRC = CRC32.hash(p, sz)
        // Store CRC directly without endian conversion (same as builder)
        // Builder writes CRC in native format, reader applies toHost() during validation
        te.crc32 = newCRC
        let mutToc = UnsafeMutablePointer<TOCEntry>(mutating: toc)
        mutToc.advanced(by: i).pointee = te
        // Sync the TOC entry cache
        tocByType[ty] = te
        // msync just the TOC entry
        let entryPtr = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(mutToc)).advanced(by: i * MemoryLayout<TOCEntry>.stride)
        _ = msync(entryPtr, MemoryLayout<TOCEntry>.stride, MS_SYNC)
    }

    @inline(__always) public func snapshotListLength(listID: Int) -> Int {
        guard let descs = secListsDesc, listID >= 0, listID < kc else { return 0 }
        let p = withUnsafePointer(to: &descs[listID].length) { UnsafeRawPointer($0) }
        let v = p.bindMemory(to: UInt32.self, capacity: 1)
        let len = atomic_load_u32_acquire(v)
        return Int(len)
    }

    private struct WalAppend { var tag: UInt32; var listID: UInt32; var oldLen: UInt32; var delta: UInt32; var idsOff: UInt64; var codesOff: UInt64; var vecsOff: UInt64; var crc32: UInt32 }
    private struct WalCommit { var tag: UInt32; var listID: UInt32; var newLen: UInt32; var crc32: UInt32 }
    private let WAL_APPEND_TAG: UInt32 = 0xAABBCCDD
    private let WAL_COMMIT_TAG: UInt32 = 0xDDCCBBAA

    public func mmap_append_begin(listID: Int, addLen: Int) throws -> AppendReservation {
        guard !opts.readOnly, let descs = secListsDesc, listID >= 0, listID < kc else { throw VIndexError.badListID }
        var dsc = descs[listID]
        let oldLen = dsc.lengthHost(fileEndian)
        let cap    = dsc.capacityHost(fileEndian)
        let need   = oldLen + addLen
        let idsStride = dsc.idsStrideHost(fileEndian)
        let codesStride = dsc.codesStrideHost(fileEndian)
        let vecsStride = dsc.vecsStrideHost(fileEndian)
        if addLen <= 0 { return AppendReservation(listID: listID, oldLen: oldLen, addLen: 0, idsFileOffset: dsc.idsOffsetHost(fileEndian), codesFileOffset: dsc.codesOffsetHost(fileEndian), vecsFileOffset: dsc.vecsOffsetHost(fileEndian), idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride) }

        if need > cap {
            guard mapSection(.ids) != nil else { throw VIndexError.cannotGrowSection(.ids) }
            guard mapSection(.codes) != nil else { throw VIndexError.cannotGrowSection(.codes) }
            var _: TOCEntry? = nil; var vecsBasePtr: UnsafeMutableRawPointer? = nil
            if vecsStride > 0 { guard mapSection(.vecs) != nil else { throw VIndexError.cannotGrowSection(.vecs) }; vecsBasePtr = secVecs }
            let newCap = max(need, max(cap * 2, 256))
            let newIDsOff = alignUp(tailIDs, 64)
            let newCodesOff = alignUp(tailCodes, 64)
            let newVecsOff = alignUp(tailVecs, 64)
            let idsBytes = UInt64(newCap * idsStride)
            let codesBytes = UInt64(newCap * codesStride)
            let vecsBytes = UInt64(newCap * vecsStride)
            try ensureFileCapacity(for: .ids, tail: newIDsOff &+ idsBytes)
            try ensureFileCapacity(for: .codes, tail: newCodesOff &+ codesBytes)
            if vecsStride > 0 { try ensureFileCapacity(for: .vecs, tail: newVecsOff &+ vecsBytes) }
            // After possible remap, refresh bases
            guard let idsBase2 = secIDs, let codesBase2 = secCodes else { throw VIndexError.mmapFailed(errno: EFAULT) }
            let oldIDs = idsBase2.advanced(by: Int(dsc.idsOffsetHost(fileEndian)))
            let newIDs = idsBase2.advanced(by: Int(newIDsOff))
            if oldLen > 0 { memcpy(newIDs, oldIDs, oldLen * idsStride) }
            let oldCodes = codesBase2.advanced(by: Int(dsc.codesOffsetHost(fileEndian)))
            let newCodes = codesBase2.advanced(by: Int(newCodesOff))
            if oldLen > 0 { memcpy(newCodes, oldCodes, oldLen * codesStride) }
            var newVecsOffUsed: UInt64 = 0
            if vecsStride > 0 {
                guard let vecsBase = secVecs else { throw VIndexError.mmapFailed(errno: EFAULT) }
                let oldVecs = vecsBase.advanced(by: Int(dsc.vecsOffsetHost(fileEndian)))
                let newVecs = vecsBase.advanced(by: Int(newVecsOff))
                if oldLen > 0 { memcpy(newVecs, oldVecs, oldLen * vecsStride) }
                newVecsOffUsed = newVecsOff
                tailVecs = alignUp(newVecsOff &+ vecsBytes, 64)
            }
            writeListDescOffsets(listID: listID, idsOff: newIDsOff, codesOff: newCodesOff, vecsOff: newVecsOffUsed, newCapacity: newCap, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride)
            tailIDs = alignUp(newIDsOff &+ idsBytes, 64); tailCodes = alignUp(newCodesOff &+ codesBytes, 64)
        }
        dsc = secListsDesc![listID]
        return AppendReservation(listID: listID, oldLen: oldLen, addLen: addLen, idsFileOffset: dsc.idsOffsetHost(fileEndian) &+ UInt64(oldLen * idsStride), codesFileOffset: dsc.codesOffsetHost(fileEndian) &+ UInt64(oldLen * codesStride), vecsFileOffset: dsc.vecsOffsetHost(fileEndian) &+ UInt64((vecsStride > 0 ? oldLen * vecsStride : 0)), idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride)
    }

    public func mmap_append_commit(_ res: AppendReservation, idsSrc: UnsafeRawPointer?, codesSrc: UnsafeRawPointer?, vecsSrc: UnsafeRawPointer?) throws {
        guard !opts.readOnly, res.addLen > 0, let descs = secListsDesc else { return }
        try writeWalAppend(listID: res.listID, oldLen: res.oldLen, delta: res.addLen, idsOff: res.idsFileOffset, codesOff: res.codesFileOffset, vecsOff: res.vecsFileOffset)
        if let ids = idsSrc, let idsBase = secIDs {
            memcpy(idsBase.advanced(by: Int(res.idsFileOffset)), ids, res.addLen * res.idsStride)
            _ = msync(idsBase.advanced(by: Int(res.idsFileOffset)), res.addLen * res.idsStride, MS_SYNC)
        }
        if let codes = codesSrc, let codesBase = secCodes {
            memcpy(codesBase.advanced(by: Int(res.codesFileOffset)), codes, res.addLen * res.codesStride)
            _ = msync(codesBase.advanced(by: Int(res.codesFileOffset)), res.addLen * res.codesStride, MS_SYNC)
        }
        if res.vecsStride > 0, let vecs = vecsSrc, let vecsBase = secVecs {
            memcpy(vecsBase.advanced(by: Int(res.vecsFileOffset)), vecs, res.addLen * res.vecsStride)
            _ = msync(vecsBase.advanced(by: Int(res.vecsFileOffset)), res.addLen * res.vecsStride, MS_SYNC)
        }
        let newLen = res.oldLen + res.addLen
        // release-store length
        var dsc = descs[res.listID]
        withUnsafeMutablePointer(to: &dsc.length) { p in atomic_store_u32_release(p, UInt32(truncatingIfNeeded: newLen)) }
        descs[res.listID] = dsc
        let dptr = UnsafeMutableRawPointer(descs).advanced(by: res.listID * MemoryLayout<ListDesc>.stride)
        _ = msync(dptr, MemoryLayout<ListDesc>.stride, MS_SYNC)
        try writeWalCommit(listID: res.listID, newLen: newLen)
    }

    public func mmap_wal_replay() throws {
        guard !opts.readOnly else { return }
        let fd = Darwin.open(walPath, O_RDONLY | O_CLOEXEC)
        if fd < 0 { return }
        defer { _ = Darwin.close(fd) }
        var lastCommitForList: [UInt32: UInt32] = [:]
        let recordSizeAppend = MemoryLayout<WalAppend>.size
        let recordSizeCommit = MemoryLayout<WalCommit>.size
        var offset: off_t = 0
        func readExact(_ n: Int) -> [UInt8]? {
            var buf = [UInt8](repeating: 0, count: n)
            var got = 0
            while got < n {
                let r = buf.withUnsafeMutableBytes { pread(fd, $0.baseAddress!.advanced(by: got), n - got, offset) }
                if r <= 0 { return nil }
                got += r; offset += off_t(r)
            }
            return buf
        }
        while true {
            guard let tagBytes = readExact(4) else { break }
            let tag = tagBytes.withUnsafeBytes { UInt32(littleEndian: $0.load(as: UInt32.self)) }
            if tag == WAL_APPEND_TAG {
                _ = readExact(recordSizeAppend - 4)
            } else if tag == WAL_COMMIT_TAG {
                guard let rest = readExact(recordSizeCommit - 4) else { break }
                let listID = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 0, as: UInt32.self)) }
                let newLen = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 4, as: UInt32.self)) }
                let crc32 = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 8, as: UInt32.self)) }
                var tmp = [UInt8](repeating: 0, count: 8)
                withUnsafeBytes(of: listID.littleEndian) { tmp.replaceSubrange(0..<4, with: $0) }
                withUnsafeBytes(of: newLen.littleEndian) { tmp.replaceSubrange(4..<8, with: $0) }
                let calc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, 8) }
                if calc == crc32 { lastCommitForList[listID] = newLen } else { break }
            } else { break }
        }
        if let descs = secListsDesc { for (lid, nl) in lastCommitForList { let i = Int(lid); if i >= 0 && i < kc { var dsc = descs[i]; withUnsafeMutablePointer(to: &dsc.length) { p in atomic_store_u32_release(p, nl) }; descs[i] = dsc; let p = UnsafeMutableRawPointer(descs).advanced(by: i * MemoryLayout<ListDesc>.stride); _ = msync(p, MemoryLayout<ListDesc>.stride, MS_SYNC) } } }
    }

    private func writeListDescOffsets(listID: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64, newCapacity: Int, idsStride: Int, codesStride: Int, vecsStride: Int) {
        guard let descs = secListsDesc else { return }
        var dsc = descs[listID]
        dsc.ids_offset   = fromHost(idsOff,   fileEndian: fileEndian)
        dsc.codes_offset = fromHost(codesOff, fileEndian: fileEndian)
        dsc.vecs_offset  = fromHost(vecsOff,  fileEndian: fileEndian)
        dsc.capacity     = fromHost(UInt32(truncatingIfNeeded: newCapacity), fileEndian: fileEndian)
        dsc.ids_stride   = fromHost(UInt32(truncatingIfNeeded: idsStride),   fileEndian: fileEndian)
        dsc.codes_stride = fromHost(UInt32(truncatingIfNeeded: codesStride), fileEndian: fileEndian)
        dsc.vecs_stride  = fromHost(UInt32(truncatingIfNeeded: vecsStride),  fileEndian: fileEndian)
        descs[listID] = dsc
    }

    private func ensureFileCapacity(for ty: SectionType, tail: UInt64) throws {
        guard let e = mapSection(ty) else { throw VIndexError.cannotGrowSection(ty) }
        let off = e.offsetHost(fileEndian)
        let needEnd = off + tail
        if needEnd > fileSize {
            // Ensure this section is the last by offset
            for (t, other) in tocByType {
                if t == ty { continue }
                let oOff = other.offsetHost(fileEndian)
                if oOff > off { throw VIndexError.cannotGrowSection(ty) }
            }
            // Extend file size
            let newSize = alignUp(needEnd, UInt64(getpagesize()))
            if ftruncate(fd, off_t(newSize)) != 0 { throw VIndexError.statFailed(errno: errno) }
            // Remap
            _ = msync(base, Int(fileSize), MS_SYNC)
            _ = munmap(base, Int(fileSize))
            let newMap = mmap(nil, Int(newSize), prot, mapFlags, fd, 0)
            if newMap == MAP_FAILED { throw VIndexError.mmapFailed(errno: errno) }
            base = newMap!
            fileSize = newSize
            // Rebuild pointers and section maps
            // Recompute TOC pointer
            let tocOffset = toHost(header.toc_offset, fileEndian: fileEndian)
            toc = UnsafeRawPointer(base).advanced(by: Int(tocOffset)).assumingMemoryBound(to: TOCEntry.self)
            tocByType.removeAll(keepingCapacity: true)
            // Rebuild toc map and section pointers, recompute tails
            // Note: skip CRC revalidation on remap; data not modified here.
            for i in 0..<tocCount {
                let te = toc.advanced(by: i).pointee
                guard let ty2 = te.typeHost(fileEndian) else { throw VIndexError.unknownSection }
                tocByType[ty2] = te
            }
            if let e = mapSection(.centroids) { secCentroids = UnsafePointer(slice(e).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.codebooks) { secCodebooks = UnsafePointer(slice(e).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.centroidNorms) { secCentroidNorms = UnsafePointer(slice(e).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.listsDesc) { secListsDesc = slice(e).assumingMemoryBound(to: ListDesc.self) }
            if let e = mapSection(.ids) { secIDs = slice(e) }
            if let e = mapSection(.codes) { secCodes = slice(e) }
            if let e = mapSection(.vecs) { secVecs = slice(e) }
            if let e = mapSection(.normsInv) { secNormsInv = UnsafeRawPointer(slice(e)) }
            if let e = mapSection(.normsSq) { secNormsSq = UnsafeRawPointer(slice(e)) }
            if let e = mapSection(.idMap) { secIDMap = UnsafeRawPointer(slice(e)) }
            if let e = mapSection(.tombstones) { secTombstones = UnsafeRawPointer(slice(e)) }
            if let descs = secListsDesc {
                var idsMax: UInt64 = 0, codesMax: UInt64 = 0, vecsMax: UInt64 = 0
                for i in 0..<kc {
                    let dsc = descs.advanced(by: i).pointee
                    let cap  = UInt64(dsc.capacityHost(fileEndian))
                    let idsOff   = dsc.idsOffsetHost(fileEndian)
                    let codesOff = dsc.codesOffsetHost(fileEndian)
                    let vecsOff  = dsc.vecsOffsetHost(fileEndian)
                    let idsStride = UInt64(dsc.idsStrideHost(fileEndian))
                    let codesStride = UInt64(dsc.codesStrideHost(fileEndian))
                    let vecsStride = UInt64(dsc.vecsStrideHost(fileEndian))
                    idsMax   = max(idsMax,   idsOff   &+ cap &* idsStride)
                    codesMax = max(codesMax, codesOff &+ cap &* codesStride)
                    vecsMax  = max(vecsMax,  vecsOff  &+ cap &* vecsStride)
                }
                tailIDs = idsMax; tailCodes = codesMax; tailVecs = vecsMax
            }
        }
    }

    private func writeWalAppend(listID: Int, oldLen: Int, delta: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64) throws {
        var rec = WalAppend(tag: WAL_APPEND_TAG.littleEndian, listID: UInt32(listID).littleEndian, oldLen: UInt32(oldLen).littleEndian, delta: UInt32(delta).littleEndian, idsOff: idsOff.littleEndian, codesOff: codesOff.littleEndian, vecsOff: vecsOff.littleEndian, crc32: 0)
        var tmp = [UInt8](repeating: 0, count: MemoryLayout<WalAppend>.size - 4)
        withUnsafeBytes(of: rec.tag) { tmp.replaceSubrange(0..<4, with: $0) }
        withUnsafeBytes(of: rec.listID) { tmp.replaceSubrange(4..<8, with: $0) }
        withUnsafeBytes(of: rec.oldLen) { tmp.replaceSubrange(8..<12, with: $0) }
        withUnsafeBytes(of: rec.delta)  { tmp.replaceSubrange(12..<16, with: $0) }
        withUnsafeBytes(of: rec.idsOff) { tmp.replaceSubrange(16..<24, with: $0) }
        withUnsafeBytes(of: rec.codesOff) { tmp.replaceSubrange(24..<32, with: $0) }
        withUnsafeBytes(of: rec.vecsOff) { tmp.replaceSubrange(32..<40, with: $0) }
        let crc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, tmp.count) }
        rec.crc32 = crc.littleEndian
        var r = rec
        let wrote = withUnsafeBytes(of: &r) { write(walFD, $0.baseAddress!, $0.count) }
        if wrote != MemoryLayout<WalAppend>.size { throw VIndexError.walIOFailed(errno: errno) }
        if fsync(walFD) != 0 { throw VIndexError.walIOFailed(errno: errno) }
    }

    private func writeWalCommit(listID: Int, newLen: Int) throws {
        var rec = WalCommit(tag: WAL_COMMIT_TAG.littleEndian, listID: UInt32(listID).littleEndian, newLen: UInt32(newLen).littleEndian, crc32: 0)
        var tmp = [UInt8](repeating: 0, count: 8)
        withUnsafeBytes(of: rec.listID) { tmp.replaceSubrange(0..<4, with: $0) }
        withUnsafeBytes(of: rec.newLen) { tmp.replaceSubrange(4..<8, with: $0) }
        let crc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, 8) }
        rec.crc32 = crc.littleEndian
        var r = rec
        let wrote = withUnsafeBytes(of: &r) { write(walFD, $0.baseAddress!, $0.count) }
        if wrote != MemoryLayout<WalCommit>.size { throw VIndexError.walIOFailed(errno: errno) }
        if fsync(walFD) != 0 { throw VIndexError.walIOFailed(errno: errno) }
    }
}
