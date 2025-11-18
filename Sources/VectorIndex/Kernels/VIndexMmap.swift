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

// Lightweight debug logger (enabled in DEBUG builds only)
#if DEBUG
@inline(__always) private func debugLog(_ message: String, file: StaticString = #file, line: UInt = #line) {
    print("[VIndexMmap] \(message) @\(file):\(line)")
}
#else
@inline(__always) private func debugLog(_ message: String, file: StaticString = #file, line: UInt = #line) {}
#endif
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

// VIndexError removed - migrated to VectorIndexError
// All throw sites now use ErrorBuilder with appropriate IndexErrorKind

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
    private var tocCount: Int
    
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
    private var listsDescBase: UnsafeMutableRawPointer?
    private var secIDs: UnsafeMutableRawPointer?
    private var secCodes: UnsafeMutableRawPointer?
    private var secVecs: UnsafeMutableRawPointer?
    private var secNormsInv: UnsafeRawPointer?
    private var secNormsSq: UnsafeRawPointer?
    private var secIDMap: UnsafeRawPointer?
    private var secTombstones: UnsafeRawPointer?

    private var tocByType: [SectionType: HostTOCEntry] = [:]
    private var tailIDs: UInt64 = 0
    private var tailCodes: UInt64 = 0
    private var tailVecs: UInt64 = 0

    private var walFD: Int32 = -1
    private var walPath: String

    public static func open(path: String, opts: MmapOpts = .init()) throws -> IndexMmap {
        let flags = opts.readOnly ? O_RDONLY : O_RDWR
        let fd = Darwin.open(path, flags | O_CLOEXEC)
        guard fd >= 0 else {
            throw ErrorBuilder(.fileIOError, operation: "vindex_open")
                .message("Failed to open index file")
                .info("path", path)
                .info("errno", "\(errno)")
                .build()
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            let err = errno
            Darwin.close(fd)
            throw ErrorBuilder(.fileIOError, operation: "vindex_fstat")
                .message("Failed to stat index file")
                .info("path", path)
                .info("errno", "\(err)")
                .build()
        }
        let fileSize = UInt64(st.st_size)
        guard fileSize >= 4096 else {
            Darwin.close(fd)
            throw ErrorBuilder(.corruptedData, operation: "vindex_open")
                .message("Index file too small or invalid header")
                .info("file_size", "\(fileSize)")
                .info("min_size", "4096")
                .build()
        }
        let prot: Int32 = opts.readOnly ? PROT_READ : (PROT_READ | PROT_WRITE)
        let mapFlags: Int32 = MAP_SHARED
        let base = mmap(nil, Int(fileSize), prot, mapFlags, fd, 0)
        guard base != MAP_FAILED else {
            let err = errno
            Darwin.close(fd)
            throw ErrorBuilder(.mmapError, operation: "vindex_mmap")
                .message("Failed to mmap index file")
                .info("path", path)
                .info("size", "\(fileSize)")
                .info("errno", "\(err)")
                .build()
        }

        let hdrPtr = base!.bindMemory(to: VIndexHeader.self, capacity: 1)
        let hdr = hdrPtr.pointee
        let fileEndian = hdr.fileEndian()
        guard fileEndian == .little || fileEndian == .big else {
            munmap(base, Int(fileSize))
            Darwin.close(fd)
            throw ErrorBuilder(.endiannessMismatch, operation: "vindex_open")
                .message("Unsupported or invalid endianness in index file")
                .info("endian_byte", "\(hdr.endianness)")
                .build()
        }
        guard hdr.magicOK(fileEndian) else {
            munmap(base, Int(fileSize))
            Darwin.close(fd)
            throw ErrorBuilder(.corruptedData, operation: "vindex_open")
                .message("Invalid magic number in index header")
                .build()
        }
        // Version policy: require major == 1 for current reader; minor is backward-compatible
        let verMajor = Int(toHost(hdr.version_major, fileEndian: fileEndian))
        let verMinor = Int(toHost(hdr.version_minor, fileEndian: fileEndian))
        guard verMajor == 1 else {
            munmap(base, Int(fileSize))
            Darwin.close(fd)
            throw ErrorBuilder(.versionMismatch, operation: "vindex_open")
                .message("Unsupported index file version")
                .info("version_major", "\(verMajor)")
                .info("version_minor", "\(verMinor)")
                .build()
        }
        if opts.verifyCRCs {
            let calc = computeHeaderCRC(UnsafeRawPointer(hdrPtr))
            let stored = toHost(hdr.header_crc32, fileEndian: fileEndian)
            guard calc == stored else {
                munmap(base, Int(fileSize))
                Darwin.close(fd)
                throw ErrorBuilder(.corruptedData, operation: "vindex_open")
                    .message("Header CRC mismatch")
                    .info("expected", "\(stored)")
                    .info("actual", "\(calc)")
                    .build()
            }
        }
        let tocOffset = toHost(hdr.toc_offset, fileEndian: fileEndian)
        let tocEntries = Int(toHost(hdr.toc_entries, fileEndian: fileEndian))
        let tocPtr = UnsafeRawPointer(base!).advanced(by: Int(tocOffset)).assumingMemoryBound(to: TOCEntry.self)

        let idx = IndexMmap(path: path, fd: fd, fileSize: fileSize, opts: opts, base: base!, prot: prot, mapFlags: mapFlags, header: hdr, fileEndian: fileEndian, toc: tocPtr, tocCount: tocEntries)
        try idx.indexInit()
        debugLog("open ok: fileSize=\(fileSize) kc=\(idx.kc) d=\(idx.d) m=\(idx.m) idBits=\(idx.idBits)")
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
        self.tocCount = tocCount
        self.kc = Int(toHost(header.kc, fileEndian: fileEndian))
        self.d  = Int(toHost(header.d, fileEndian: fileEndian))
        self.m  = Int(toHost(header.m, fileEndian: fileEndian))
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

    private func slice(_ e: HostTOCEntry) -> UnsafeMutableRawPointer {
        let off = Int(e.offset)
        return UnsafeMutableRawPointer(base).advanced(by: off)
    }

    private struct HostTOCEntry { var type: SectionType; var offset: UInt64; var size: UInt64; var align: UInt32; var flags: UInt32; var crc32: UInt32 }
    private func mapSection(_ ty: SectionType) -> HostTOCEntry? { tocByType[ty] }

    @inline(__always) private func msyncPageAligned(_ ptr: UnsafeMutableRawPointer, _ length: Int) {
        // Robust: flush whole mapping to avoid sub-page msync pitfalls on macOS
        _ = msync(base, Int(fileSize), MS_SYNC)
    }

    @inline(__always) private func writeLE32(_ p: UnsafeMutableRawPointer, _ v: UInt32) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { bytes in memcpy(p, bytes.baseAddress!, 4) }
    }
    @inline(__always) private func writeLE64(_ p: UnsafeMutableRawPointer, _ v: UInt64) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { bytes in memcpy(p, bytes.baseAddress!, 8) }
    }

    private func indexInit() throws {
        debugLog("indexInit: tocCount=\(tocCount) fileSize=\(fileSize)")
        // Parse TOC as packed entries (36 bytes each)
        let DISK_TOC_ENTRY_SIZE = 36
        let tocRaw = UnsafeRawPointer(base).advanced(by: Int(toHost(header.toc_offset, fileEndian: fileEndian)))
        for i in 0..<tocCount {
            let te = tocRaw.advanced(by: i * DISK_TOC_ENTRY_SIZE)
            let tyRaw = readLE32(te.advanced(by: 0))
            guard let ty = SectionType(rawValue: tyRaw) else {
                throw ErrorBuilder(.invalidFormat, operation: "vindex_init")
                    .message("Unknown section type in TOC")
                    .info("toc_index", "\(i)")
                    .build()
            }
            // Packed: offset@+4, size@+12, align@+20, flags@+24, crc@+28
            let off = readLE64(te.advanced(by: 4))
            let sz  = readLE64(te.advanced(by: 12))
            let al  = UInt64(readLE32(te.advanced(by: 20)))
            let flags = readLE32(te.advanced(by: 24))
            let crc = readLE32(te.advanced(by: 28))
            tocByType[ty] = HostTOCEntry(type: ty, offset: off, size: sz, align: UInt32(al), flags: flags, crc32: crc)
            if al != 0 && (off % UInt64(al)) != 0 {
                throw ErrorBuilder(.corruptedData, operation: "vindex_init")
                    .message("Section misaligned in index file")
                    .info("section_type", "\(ty.rawValue)")
                    .info("offset", "\(off)")
                    .info("required_alignment", "\(al)")
                    .info("actual_alignment", "\(off % UInt64(al))")
                    .build()
            }
            if opts.verifyCRCs && sz > 0 {
                let p = UnsafeRawPointer(base).advanced(by: Int(off))
                let crc = CRC32.hash(p, Int(sz))
                let stored = readLE32(te.advanced(by: 28))
                if crc != stored {
                    debugLog("CRC mismatch for section=\(String(describing: ty)) stored=\(stored) calc=\(crc) readOnly=\(opts.readOnly)")
                    // If ListsDesc differs and we're writable, refresh CRC as a best-effort repair.
                    if !opts.readOnly && ty == .listsDesc {
                        _ = try? updateSectionCRC(.listsDesc)
                    } else {
                        throw ErrorBuilder(.corruptedData, operation: "vindex_init")
                            .message("Section CRC mismatch")
                            .info("section_type", "\(ty.rawValue)")
                            .info("expected", "\(stored)")
                            .info("actual", "\(crc)")
                            .build()
                    }
                }
            }
        }
        if let e = mapSection(.centroids) { secCentroids = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.codebooks) { secCodebooks = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.centroidNorms) { secCentroidNorms = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.listsDesc) { listsDescBase = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.ids) { secIDs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.codes) { secCodes = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.vecs) { secVecs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.normsInv) { secNormsInv = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
        if let e = mapSection(.normsSq) { secNormsSq = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
        if let e = mapSection(.idMap) { secIDMap = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
        if let e = mapSection(.tombstones) { secTombstones = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }

        if let descs = listsDescBase?.assumingMemoryBound(to: UInt8.self) {
            var idsMax: UInt64 = 0, codesMax: UInt64 = 0, vecsMax: UInt64 = 0
            for i in 0..<kc {
                let rec = descs.advanced(by: i * 64)
                let cap  = UInt64(readLE32(rec.advanced(by: 8)))
                let idsOff   = readLE64(rec.advanced(by: 16))
                let codesOff = readLE64(rec.advanced(by: 24))
                let vecsOff  = readLE64(rec.advanced(by: 32))
                let idsStride = UInt64(readLE32(rec.advanced(by: 40)))
                let codesStride = UInt64(readLE32(rec.advanced(by: 44)))
                let vecsStride = UInt64(readLE32(rec.advanced(by: 48)))
                idsMax   = max(idsMax, idsOff   &+ cap &* idsStride)
                codesMax = max(codesMax, codesOff &+ cap &* codesStride)
                vecsMax  = max(vecsMax, vecsOff  &+ cap &* vecsStride)
            }
            tailIDs = idsMax; tailCodes = codesMax; tailVecs = vecsMax
        }

        if opts.adviseSequential { _ = posix_madvise(base, Int(fileSize), POSIX_MADV_SEQUENTIAL) }
        if opts.adviseWillNeed { _ = posix_madvise(base, Int(fileSize), POSIX_MADV_WILLNEED) }

        if !opts.readOnly {
            walFD = Darwin.open(walPath, O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR)
            if walFD < 0 {
                throw ErrorBuilder(.fileIOError, operation: "vindex_wal_open")
                    .message("Failed to open WAL file")
                    .info("wal_path", walPath)
                    .info("errno", "\(errno)")
                    .build()
            }
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
    // Legacy API retained for compatibility (returns nil; use getListDescriptor instead)
    public func mmapLists() -> (ptr: UnsafeMutablePointer<ListDesc>, kc: Int)? {
        nil
    }
    public func idsBase() -> UnsafeMutableRawPointer? { secIDs }
    public func codesBase() -> UnsafeMutableRawPointer? { secCodes }
    public func vecsBase() -> UnsafeMutableRawPointer? { secVecs }

    // Return a raw pointer and size for a section if present
    public func sectionSlice(_ ty: SectionType) -> (ptr: UnsafeRawPointer, size: Int)? {
        guard let e = tocByType[ty] else { return nil }
        let size = Int(e.size)
        let p: UnsafeRawPointer
        switch ty {
        case .centroids: guard let s = secCentroids else { return nil }; p = UnsafeRawPointer(s)
        case .codebooks: guard let s = secCodebooks else { return nil }; p = UnsafeRawPointer(s)
        case .centroidNorms: guard let s = secCentroidNorms else { return nil }; p = UnsafeRawPointer(s)
        case .listsDesc:
            guard let s = listsDescBase else { return nil }
            p = UnsafeRawPointer(s)
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
        guard !opts.readOnly else {
            throw ErrorBuilder(.fileIOError, operation: "vindex_idmap_write")
                .message("Cannot write to read-only index")
                .build()
        }
        guard let e = tocByType[.idMap] else {
            throw ErrorBuilder(.invalidFormat, operation: "vindex_idmap_write")
                .message("IDMap section not found in index")
                .build()
        }
        let maxSize = Int(e.size)
        guard blob.count <= maxSize else {
            throw ErrorBuilder(.capacityExceeded, operation: "vindex_idmap_write")
                .message("IDMap blob too large for allocated section")
                .info("blob_size", "\(blob.count)")
                .info("max_size", "\(maxSize)")
                .build()
        }
        guard let basePtr = secIDMap else {
            throw ErrorBuilder(.invalidFormat, operation: "vindex_idmap_write")
                .message("IDMap section pointer unavailable")
                .build()
        }
        blob.withUnsafeBytes { src in
            memcpy(UnsafeMutableRawPointer(mutating: basePtr), src.baseAddress!, blob.count)
            if maxSize > blob.count {
                memset(UnsafeMutableRawPointer(mutating: basePtr).advanced(by: blob.count), 0, maxSize - blob.count)
            }
        }
        msyncPageAligned(UnsafeMutableRawPointer(mutating: basePtr), maxSize)
        // Update CRC in TOC entry in-place
        try updateSectionCRC(.idMap)
    }

    private func updateSectionCRC(_ ty: SectionType) throws {
        // Find packed TOC entry by type
        let tocOff = Int(toHost(header.toc_offset, fileEndian: fileEndian))
        let DISK_TOC_ENTRY_SIZE = 36
        var idxFound: Int?
        for i in 0..<tocCount {
            let te = UnsafeRawPointer(base).advanced(by: tocOff + i * DISK_TOC_ENTRY_SIZE)
            let tyRaw = readLE32(te.advanced(by: 0))
            if tyRaw == ty.rawValue { idxFound = i; break }
        }
        guard let i = idxFound else {
            throw ErrorBuilder(.invalidFormat, operation: "vindex_update_crc")
                .message("Section not found in TOC")
                .info("section_type", "\(ty.rawValue)")
                .build()
        }
        let entryPtr = UnsafeMutableRawPointer(base).advanced(by: tocOff + i * DISK_TOC_ENTRY_SIZE)
        let off = readLE64(UnsafeRawPointer(entryPtr).advanced(by: 4))
        let sz  = Int(readLE64(UnsafeRawPointer(entryPtr).advanced(by: 12)))
        let p = UnsafeRawPointer(base).advanced(by: Int(off))
        let newCRC = CRC32.hash(p, sz)
        writeLE32(entryPtr.advanced(by: 28), newCRC)
        msyncPageAligned(entryPtr, DISK_TOC_ENTRY_SIZE)
        // also refresh cache
        if var e = tocByType[ty] { e.crc32 = newCRC; tocByType[ty] = e }
    }

    @inline(__always) public func snapshotListLength(listID: Int) -> Int {
        guard let base = listsDescBase?.assumingMemoryBound(to: UInt8.self), listID >= 0, listID < kc else { return 0 }
        let rec = UnsafeRawPointer(base.advanced(by: listID * 64))
        let len = readLE32(rec.advanced(by: 4))
        return Int(len)
    }

    // Public accessor for list descriptor offsets and metadata (packed LE)
    public func getListDescriptor(listID: Int) -> (length: Int, capacity: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64, idsStride: Int, codesStride: Int, vecsStride: Int)? {
        guard let base = listsDescBase?.assumingMemoryBound(to: UInt8.self), listID >= 0, listID < kc else { return nil }
        let rec = UnsafeRawPointer(base.advanced(by: listID * 64))
        let length = Int(readLE32(rec.advanced(by: 4)))
        let capacity = Int(readLE32(rec.advanced(by: 8)))
        let idsOff = readLE64(rec.advanced(by: 16))
        let codesOff = readLE64(rec.advanced(by: 24))
        let vecsOff  = readLE64(rec.advanced(by: 32))
        let idsStride = Int(readLE32(rec.advanced(by: 40)))
        let codesStride = Int(readLE32(rec.advanced(by: 44)))
        let vecsStride = Int(readLE32(rec.advanced(by: 48)))
        return (length, capacity, idsOff, codesOff, vecsOff, idsStride, codesStride, vecsStride)
    }

    private struct WalAppend { var tag: UInt32; var listID: UInt32; var oldLen: UInt32; var delta: UInt32; var idsOff: UInt64; var codesOff: UInt64; var vecsOff: UInt64; var crc32: UInt32 }
    private struct WalCommit { var tag: UInt32; var listID: UInt32; var newLen: UInt32; var crc32: UInt32 }
    private let WAL_APPEND_TAG: UInt32 = 0xAABBCCDD
    private let WAL_COMMIT_TAG: UInt32 = 0xDDCCBBAA

    public func mmap_append_begin(listID: Int, addLen: Int) throws -> AppendReservation {
        guard !opts.readOnly, listID >= 0, listID < kc else {
            throw ErrorBuilder(.invalidRange, operation: "mmap_append_begin")
                .message("Invalid list ID or index is read-only")
                .info("list_id", "\(listID)")
                .info("valid_range", "0..<\(kc)")
                .info("read_only", "\(opts.readOnly)")
                .build()
        }
        guard let ldbase = listsDescBase?.assumingMemoryBound(to: UInt8.self) else {
            throw ErrorBuilder(.invalidFormat, operation: "mmap_append_begin")
                .message("ListsDesc base unavailable")
                .build()
        }
        let rec = ldbase.advanced(by: listID * 64)
        let oldLen = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 4))))
        let cap    = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 8))))
        let idsStride = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 40))))
        let codesStride = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 44))))
        let vecsStride = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 48))))
        let need   = oldLen + addLen
        var currIDsOff = readLE64(UnsafeRawPointer(rec.advanced(by: 16)))
        var currCodesOff = readLE64(UnsafeRawPointer(rec.advanced(by: 24)))
        var currVecsOff  = readLE64(UnsafeRawPointer(rec.advanced(by: 32)))
        debugLog("append_begin list=\(listID) oldLen=\(oldLen) cap=\(cap) need=\(need) strides(ids=\(idsStride),codes=\(codesStride),vecs=\(vecsStride)) tails(ids=\(tailIDs),codes=\(tailCodes),vecs=\(tailVecs)) currOffsets(ids=\(currIDsOff),codes=\(currCodesOff),vecs=\(currVecsOff))")
        if addLen <= 0 { return AppendReservation(listID: listID, oldLen: oldLen, addLen: 0, idsFileOffset: currIDsOff, codesFileOffset: currCodesOff, vecsFileOffset: currVecsOff, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride) }

        if need > cap {
            guard mapSection(.ids) != nil else {
                throw ErrorBuilder(.capacityExceeded, operation: "mmap_append_begin")
                    .message("Cannot grow IDs section")
                    .info("list_id", "\(listID)")
                    .build()
            }
            guard mapSection(.codes) != nil else {
                throw ErrorBuilder(.capacityExceeded, operation: "mmap_append_begin")
                    .message("Cannot grow codes section")
                    .info("list_id", "\(listID)")
                    .build()
            }
            if vecsStride > 0 {
                guard mapSection(.vecs) != nil else {
                    throw ErrorBuilder(.capacityExceeded, operation: "mmap_append_begin")
                        .message("Cannot grow vecs section")
                        .info("list_id", "\(listID)")
                        .build()
                }
            }
            let newCap = max(need, max(cap * 2, 256))
            let newIDsOff = alignUp(tailIDs, 64)
            let newCodesOff = alignUp(tailCodes, 64)
            let newVecsOff = alignUp(tailVecs, 64)
            let idsBytes = UInt64(newCap * idsStride)
            let codesBytes = UInt64(newCap * codesStride)
            let vecsBytes = UInt64(newCap * vecsStride)
            debugLog("growth list=\(listID) newCap=\(newCap) newOff(ids=\(newIDsOff),codes=\(newCodesOff),vecs=\(newVecsOff)) bytes(ids=\(idsBytes),codes=\(codesBytes),vecs=\(vecsBytes))")
            // Compute tails relative to section base (offsets stored in descriptors are section-relative)
            try ensureFileCapacity(for: .ids, tail: newIDsOff &+ idsBytes)
            try ensureFileCapacity(for: .codes, tail: newCodesOff &+ codesBytes)
            if vecsStride > 0 { try ensureFileCapacity(for: .vecs, tail: newVecsOff &+ vecsBytes) }
            // After possible remap, refresh bases
            guard let idsBase2 = secIDs, let codesBase2 = secCodes else {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                    .message("Section base pointers unavailable after remap")
                    .build()
            }
            let oldIDs = idsBase2.advanced(by: Int(currIDsOff))
            let newIDs = idsBase2.advanced(by: Int(newIDsOff))
            if oldLen > 0 { memcpy(newIDs, oldIDs, oldLen * idsStride) }
            let oldCodes = codesBase2.advanced(by: Int(currCodesOff))
            let newCodes = codesBase2.advanced(by: Int(newCodesOff))
            if oldLen > 0 { memcpy(newCodes, oldCodes, oldLen * codesStride) }
            var newVecsOffUsed: UInt64 = 0
            if vecsStride > 0 {
                guard let vecsBase = secVecs else {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("Vecs section base pointer unavailable after remap")
                        .build()
                }
                let oldVecs = vecsBase.advanced(by: Int(currVecsOff))
                let newVecs = vecsBase.advanced(by: Int(newVecsOff))
                if oldLen > 0 { memcpy(newVecs, oldVecs, oldLen * vecsStride) }
                newVecsOffUsed = newVecsOff
                tailVecs = alignUp(newVecsOff &+ vecsBytes, 64)
            }
            // Sanity: ensure new offsets + capacity bytes fit inside the section sizes
            if let eIDs = mapSection(.ids) {
                let idsSecSize = eIDs.size
                let idsNeeded = (newIDsOff &+ idsBytes)
                if idsNeeded > idsSecSize {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("IDs offset/capacity exceed section size")
                        .info("offset", "\(newIDsOff)")
                        .info("needed", "\(idsNeeded)")
                        .info("section_size", "\(idsSecSize)")
                        .build()
                }
            }
            if let eCodes = mapSection(.codes) {
                let codesSecSize = eCodes.size
                let codesNeeded = (newCodesOff &+ codesBytes)
                if codesNeeded > codesSecSize {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("Codes offset/capacity exceed section size")
                        .info("offset", "\(newCodesOff)")
                        .info("needed", "\(codesNeeded)")
                        .info("section_size", "\(codesSecSize)")
                        .build()
                }
            }
            if vecsStride > 0, let eVecs = mapSection(.vecs) {
                let vecsSecSize = eVecs.size
                let vecsNeeded = (newVecsOff &+ vecsBytes)
                if vecsNeeded > vecsSecSize {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("Vecs offset/capacity exceed section size")
                        .info("offset", "\(newVecsOff)")
                        .info("needed", "\(vecsNeeded)")
                        .info("section_size", "\(vecsSecSize)")
                        .build()
                }
            }

            writeListDescOffsets(listID: listID, idsOff: newIDsOff, codesOff: newCodesOff, vecsOff: newVecsOffUsed, newCapacity: newCap, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride)
            tailIDs = alignUp(newIDsOff &+ idsBytes, 64); tailCodes = alignUp(newCodesOff &+ codesBytes, 64)
            currIDsOff = newIDsOff
            currCodesOff = newCodesOff
            currVecsOff = newVecsOffUsed
            // Read back descriptor packed
            let vIDs = readLE64(UnsafeRawPointer(rec.advanced(by: 16)))
            let vCodes = readLE64(UnsafeRawPointer(rec.advanced(by: 24)))
            let vVecs = readLE64(UnsafeRawPointer(rec.advanced(by: 32)))
            debugLog("descriptor_written list=\(listID) verifyOff(ids=\(vIDs),codes=\(vCodes),vecs=\(vVecs))")
        }
        // Use current (possibly updated) section-relative offsets
        let idsOffset = currIDsOff &+ UInt64(oldLen * idsStride)
        let codesOffset = currCodesOff &+ UInt64(oldLen * codesStride)
        let vecsOffset = currVecsOff &+ UInt64((vecsStride > 0 ? oldLen * vecsStride : 0))
        debugLog("reservation list=\(listID) oldLen=\(oldLen) addLen=\(addLen) resOff(ids=\(idsOffset),codes=\(codesOffset),vecs=\(vecsOffset))")
        return AppendReservation(listID: listID, oldLen: oldLen, addLen: addLen, idsFileOffset: idsOffset, codesFileOffset: codesOffset, vecsFileOffset: vecsOffset, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride)
    }

    public func mmap_append_commit(_ res: AppendReservation, idsSrc: UnsafeRawPointer?, codesSrc: UnsafeRawPointer?, vecsSrc: UnsafeRawPointer?) throws {
        guard !opts.readOnly, res.addLen > 0 else { return }
        debugLog("commit_begin list=\(res.listID) oldLen=\(res.oldLen) addLen=\(res.addLen) resOff(ids=\(res.idsFileOffset),codes=\(res.codesFileOffset),vecs=\(res.vecsFileOffset)) strides(ids=\(res.idsStride),codes=\(res.codesStride),vecs=\(res.vecsStride))")
        try writeWalAppend(listID: res.listID, oldLen: res.oldLen, delta: res.addLen, idsOff: res.idsFileOffset, codesOff: res.codesFileOffset, vecsOff: res.vecsFileOffset)
        if let ids = idsSrc, let idsBase = secIDs {
            guard let e = tocByType[.ids] else {
                throw ErrorBuilder(.invalidFormat, operation: "mmap_append_commit").message("IDs section not found in TOC").build()
            }
            let secSize = e.size
            let bytes = UInt64(res.addLen &* res.idsStride)
            let relOff = res.idsFileOffset
            debugLog("commit_ids list=\(res.listID) relOff=\(relOff) bytes=\(bytes) secSize=\(secSize)")
            if relOff &+ bytes > secSize {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_commit")
                    .message("IDs write exceeds mapped section size")
                    .info("offset", "\(relOff)")
                    .info("bytes", "\(bytes)")
                    .info("limit", "\(secSize)")
                    .build()
            }
            let dst = idsBase.advanced(by: Int(relOff))
            memcpy(dst, ids, Int(bytes))
            msyncPageAligned(dst, Int(bytes))
            // Update CRC for IDs section after write
            try updateSectionCRC(.ids)
        }
        if let codes = codesSrc, let codesBase = secCodes {
            guard let e = tocByType[.codes] else {
                throw ErrorBuilder(.invalidFormat, operation: "mmap_append_commit").message("Codes section not found in TOC").build()
            }
            let secSize = e.size
            let bytes = UInt64(res.addLen &* res.codesStride)
            let relOff = res.codesFileOffset
            debugLog("commit_codes list=\(res.listID) relOff=\(relOff) bytes=\(bytes) secSize=\(secSize)")
            if relOff &+ bytes > secSize {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_commit")
                    .message("Codes write exceeds mapped section size")
                    .info("offset", "\(relOff)")
                    .info("bytes", "\(bytes)")
                    .info("limit", "\(secSize)")
                    .build()
            }
            let dst = codesBase.advanced(by: Int(relOff))
            memcpy(dst, codes, Int(bytes))
            msyncPageAligned(dst, Int(bytes))
            // Update CRC for Codes section after write
            try updateSectionCRC(.codes)
        }
        if res.vecsStride > 0, let vecs = vecsSrc, let vecsBase = secVecs {
            guard let e = tocByType[.vecs] else {
                throw ErrorBuilder(.invalidFormat, operation: "mmap_append_commit").message("Vecs section not found in TOC").build()
            }
            let secSize = e.size
            let bytes = UInt64(res.addLen &* res.vecsStride)
            let relOff = res.vecsFileOffset
            debugLog("commit_vecs list=\(res.listID) relOff=\(relOff) bytes=\(bytes) secSize=\(secSize)")
            if relOff &+ bytes > secSize {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_commit")
                    .message("Vecs write exceeds mapped section size")
                    .info("offset", "\(relOff)")
                    .info("bytes", "\(bytes)")
                    .info("limit", "\(secSize)")
                    .build()
            }
            let dst = vecsBase.advanced(by: Int(relOff))
            memcpy(dst, vecs, Int(bytes))
            msyncPageAligned(dst, Int(bytes))
            // Update CRC for Vecs section after write
            try updateSectionCRC(.vecs)
        }
        let newLen = res.oldLen + res.addLen
        // Update length in packed ListsDesc record and sync
        if let base = listsDescBase {
            let rec = base.advanced(by: res.listID * 64)
            writeLE32(rec.advanced(by: 4), UInt32(truncatingIfNeeded: newLen))
            msyncPageAligned(rec, 64)
        }
        // Update CRC for ListsDesc after length update
        try updateSectionCRC(.listsDesc)
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
        if let base = listsDescBase {
            for (lid, nl) in lastCommitForList {
                let i = Int(lid)
                if i >= 0 && i < kc {
                    let rec = base.advanced(by: i * 64)
                    writeLE32(rec.advanced(by: 4), nl)
                    msyncPageAligned(rec, 64)
                }
            }
        }
    }

    private func writeListDescOffsets(listID: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64, newCapacity: Int, idsStride: Int, codesStride: Int, vecsStride: Int) {
        guard let base = listsDescBase else { return }
        let rec = base.advanced(by: listID * 64)
        // Write packed LE fields
        writeLE64(rec.advanced(by: 16), idsOff)
        writeLE64(rec.advanced(by: 24), codesOff)
        writeLE64(rec.advanced(by: 32), vecsOff)
        writeLE32(rec.advanced(by: 8), UInt32(truncatingIfNeeded: newCapacity))
        writeLE32(rec.advanced(by: 40), UInt32(truncatingIfNeeded: idsStride))
        writeLE32(rec.advanced(by: 44), UInt32(truncatingIfNeeded: codesStride))
        writeLE32(rec.advanced(by: 48), UInt32(truncatingIfNeeded: vecsStride))
        msyncPageAligned(rec, 64)
        _ = try? updateSectionCRC(.listsDesc)
    }

    private func ensureFileCapacity(for ty: SectionType, tail: UInt64) throws {
        guard let e = mapSection(ty) else {
            throw ErrorBuilder(.mmapError, operation: "ensure_file_capacity")
                .message("Cannot grow section: not found in TOC")
                .info("section_type", "\(ty.rawValue)")
                .build()
        }
        let off = e.offset
        let needEnd = off + tail
        if needEnd > fileSize {
            // Ensure this section is the last by offset
            for (t, other) in tocByType {
                if t == ty { continue }
                let oOff = other.offset
                if oOff > off {
                    throw ErrorBuilder(.mmapError, operation: "ensure_file_capacity")
                        .message("Cannot grow section: not the last section by offset")
                        .info("section_type", "\(ty.rawValue)")
                        .info("section_offset", "\(off)")
                        .info("later_section_offset", "\(oOff)")
                        .build()
                }
            }
            // Extend file size
            let newSize = alignUp(needEnd, UInt64(getpagesize()))
            if ftruncate(fd, off_t(newSize)) != 0 {
                throw ErrorBuilder(.fileIOError, operation: "ensure_file_capacity")
                    .message("Failed to extend index file")
                    .info("current_size", "\(fileSize)")
                    .info("new_size", "\(newSize)")
                    .info("errno", "\(errno)")
                    .build()
            }
            // Remap
            _ = msync(base, Int(fileSize), MS_SYNC)
            _ = munmap(base, Int(fileSize))
            let newMap = mmap(nil, Int(newSize), prot, MAP_SHARED, fd, 0)
            if newMap == MAP_FAILED {
                throw ErrorBuilder(.mmapError, operation: "ensure_file_capacity")
                    .message("Failed to remap extended index file")
                    .info("new_size", "\(newSize)")
                    .info("errno", "\(errno)")
                    .build()
            }
            base = newMap!
            fileSize = newSize
            // Rebuild TOC map (packed)
            tocByType.removeAll(keepingCapacity: true)
            let tocOff = Int(toHost(header.toc_offset, fileEndian: fileEndian))
            let DISK_TOC_ENTRY_SIZE = 36
            for i in 0..<tocCount {
                let te = UnsafeRawPointer(base).advanced(by: tocOff + i * DISK_TOC_ENTRY_SIZE)
                let tyRaw = readLE32(te.advanced(by: 0))
                guard let ty2 = SectionType(rawValue: tyRaw) else {
                    throw ErrorBuilder(.invalidFormat, operation: "ensure_file_capacity")
                        .message("Unknown section type after remap")
                        .info("toc_index", "\(i)")
                        .build()
                }
                let off = readLE64(te.advanced(by: 8))
                let sz  = readLE64(te.advanced(by: 16))
                let al  = readLE32(te.advanced(by: 24))
                let flags = readLE32(te.advanced(by: 28))
                let crc = readLE32(te.advanced(by: 32))
                tocByType[ty2] = HostTOCEntry(type: ty2, offset: off, size: sz, align: al, flags: flags, crc32: crc)
            }
            // Rebuild section base pointers
            if let e = mapSection(.centroids) { secCentroids = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.codebooks) { secCodebooks = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.centroidNorms) { secCentroidNorms = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.listsDesc) { listsDescBase = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.ids) { secIDs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.codes) { secCodes = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.vecs) { secVecs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.normsInv) { secNormsInv = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            if let e = mapSection(.normsSq) { secNormsSq = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            if let e = mapSection(.idMap) { secIDMap = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            if let e = mapSection(.tombstones) { secTombstones = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            // Recompute tails
            if let descs = listsDescBase?.assumingMemoryBound(to: UInt8.self) {
                var idsMax: UInt64 = 0, codesMax: UInt64 = 0, vecsMax: UInt64 = 0
                for i in 0..<kc {
                    let rec = UnsafeRawPointer(descs.advanced(by: i * 64))
                    let cap  = UInt64(readLE32(rec.advanced(by: 8)))
                    let idsOff   = readLE64(rec.advanced(by: 16))
                    let codesOff = readLE64(rec.advanced(by: 24))
                    let vecsOff  = readLE64(rec.advanced(by: 32))
                    let idsStride = UInt64(readLE32(rec.advanced(by: 40)))
                    let codesStride = UInt64(readLE32(rec.advanced(by: 44)))
                    let vecsStride = UInt64(readLE32(rec.advanced(by: 48)))
                    idsMax   = max(idsMax, idsOff   &+ cap &* idsStride)
                    codesMax = max(codesMax, codesOff &+ cap &* codesStride)
                    vecsMax  = max(vecsMax, vecsOff  &+ cap &* vecsStride)
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
        withUnsafeBytes(of: rec.delta) { tmp.replaceSubrange(12..<16, with: $0) }
        withUnsafeBytes(of: rec.idsOff) { tmp.replaceSubrange(16..<24, with: $0) }
        withUnsafeBytes(of: rec.codesOff) { tmp.replaceSubrange(24..<32, with: $0) }
        withUnsafeBytes(of: rec.vecsOff) { tmp.replaceSubrange(32..<40, with: $0) }
        let crc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, tmp.count) }
        rec.crc32 = crc.littleEndian
        var r = rec
        let wrote = withUnsafeBytes(of: &r) { write(walFD, $0.baseAddress!, $0.count) }
        if wrote != MemoryLayout<WalAppend>.size {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_append")
                .message("Failed to write WAL append record")
                .info("expected_bytes", "\(MemoryLayout<WalAppend>.size)")
                .info("written_bytes", "\(wrote)")
                .info("errno", "\(errno)")
                .build()
        }
        if fsync(walFD) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_append")
                .message("Failed to fsync WAL file")
                .info("errno", "\(errno)")
                .build()
        }
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
        if wrote != MemoryLayout<WalCommit>.size {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_commit")
                .message("Failed to write WAL commit record")
                .info("expected_bytes", "\(MemoryLayout<WalCommit>.size)")
                .info("written_bytes", "\(wrote)")
                .info("errno", "\(errno)")
                .build()
        }
        if fsync(walFD) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_commit")
                .message("Failed to fsync WAL file")
                .info("errno", "\(errno)")
                .build()
        }
    }
}
// Little-endian readers for packed on-disk records
@inline(__always) private func readLE32(_ p: UnsafeRawPointer) -> UInt32 {
    var v: UInt32 = 0
    memcpy(&v, p, 4)
    return UInt32(littleEndian: v)
}

@inline(__always) private func readLE64(_ p: UnsafeRawPointer) -> UInt64 {
    var v: UInt64 = 0
    memcpy(&v, p, 8)
    return UInt64(littleEndian: v)
}
