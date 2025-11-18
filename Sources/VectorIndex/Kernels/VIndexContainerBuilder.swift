//
//  VIndexContainerBuilder.swift
//  Minimal S1 container builder for tests and durable ingestion
//

import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

@inline(__always) private func alignUpU64(_ x: UInt64, _ a: UInt64) -> UInt64 { let m = a &- 1; return (x &+ m) & ~m }

// Local CRC32 for builder (duplicate of VIndexMmap.swift but self-contained)
private struct _CRC32 {
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
            c = _CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
        }
        return c ^ 0xFFFF_FFFF
    }
}

// These mirror VIndexMmap.swift disk structs
private struct _TOCEntry { var type: UInt32; var offset: UInt64; var size: UInt64; var align: UInt32; var flags: UInt32; var crc32: UInt32; var reserved: UInt32 }
private struct _Header {
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
}

// Public builder API
internal enum VIndexContainerBuilder {
    /// Create a minimal container with ListsDesc + IDs + Codes (or Vecs) sections.
    ///
    /// - Parameters:
    ///   - path: file path to create
    ///   - format: .pq8 / .pq4 / .flat
    ///   - k_c: number of lists
    ///   - m: PQ subspaces (0 for flat)
    ///   - d: dimension (for flat)
    ///   - idBits: 32 or 64
    ///   - group: interleave group (4 or 8)
    ///   - idCap: initial per-list capacity for IDs (pre-sized)
    ///   - payloadCap: initial per-list capacity for Codes (PQ) or Vecs (flat)
    ///
    /// The builder places ListsDesc, then IDs, then Codes/Vecs (last), so that Codes/Vecs can grow.
    @discardableResult
    public static func createMinimalContainer(path: String,
                                              format: IVFFormat,
                                              k_c: Int,
                                              m: Int,
                                              d: Int,
                                              idBits: Int = 64,
                                              group: Int = 4,
                                              idCap: Int = 256,
                                              payloadCap: Int = 64,
                                              includeIDMap: Bool = true) throws -> IndexMmap {
        precondition(k_c > 0)
        precondition(idBits == 32 || idBits == 64)
        precondition(group == 4 || group == 8)
        let page = UInt64(getpagesize())
        let headerSize: UInt64 = 256
        var tocCount: Int = 3 // ListsDesc, IDs, Codes/Vecs (+ optional IDMap)
        let tocSize = UInt64(tocCount * MemoryLayout<_TOCEntry>.stride)

        // Compute strides
        let idStride = (idBits == 32) ? MemoryLayout<UInt32>.stride : MemoryLayout<UInt64>.stride
        let codesStride: Int
        let vecsStride: Int
        switch format {
        case .pq8: codesStride = m; vecsStride = 0
        case .pq4: precondition(m % 2 == 0); codesStride = m >> 1; vecsStride = 0
        case .flat: codesStride = 0; vecsStride = d * MemoryLayout<Float>.stride
        }

        // Layout offsets (file-relative)
        var off: UInt64 = alignUpU64(headerSize, 64)
        let tocOffset = off
        off = alignUpU64(tocOffset &+ tocSize, 64)

        // ListsDesc section (packed 64-byte records)
        let DISK_LISTDESC_SIZE: UInt64 = 64
        let listsDescOffset = off
        let listsDescSize = UInt64(k_c) &* DISK_LISTDESC_SIZE
        off = alignUpU64(listsDescOffset &+ listsDescSize, 64)

        // IDs section (pre-sized for idCap)
        let idsOffset = off
        let perListIDsBytes = UInt64(idCap * idStride)
        // per-list inner offsets inside IDs section, aligned to 64B
        var perListIDsOff: [UInt64] = []
        perListIDsOff.reserveCapacity(k_c)
        var idsTail: UInt64 = 0
        for _ in 0..<k_c { let start = alignUpU64(idsTail, 64); perListIDsOff.append(start); idsTail = start &+ perListIDsBytes }
        let idsSize = alignUpU64(idsTail, 64)
        off = alignUpU64(idsOffset &+ idsSize, page)

        // Codes or Vecs section (placed last so it can grow)
        let payloadOffset = off
        let perListPayloadStride = (format == .flat) ? vecsStride : codesStride
        let perListPayloadBytes = UInt64(payloadCap * max(perListPayloadStride, 1))
        var perListPayloadOff: [UInt64] = []
        perListPayloadOff.reserveCapacity(k_c)
        var payloadTail: UInt64 = 0
        for _ in 0..<k_c { let start = alignUpU64(payloadTail, 64); perListPayloadOff.append(start); payloadTail = start &+ perListPayloadBytes }
        let payloadSize = alignUpU64(payloadTail, page)
        off = alignUpU64(payloadOffset &+ payloadSize, page)

        // Optional: IDMap section (fixed-size blob, can be grown by rebuilding container)
        var idMapOffset: UInt64 = 0
        var idMapSize: UInt64 = 0
        if includeIDMap {
            tocCount += 1
            idMapOffset = off
            idMapSize = 2048 // Sufficient for typical IDMap snapshots (~1000 IDs)
            off = alignUpU64(idMapOffset &+ idMapSize, 64)
        }

        // Total file size
        let fileSize = off

        // Create and size the file
        let fd = Darwin.open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR)
        guard fd >= 0 else {
            throw ErrorBuilder(.fileIOError, operation: "container_create")
                .message("Failed to open container file")
                .path(path)
                .errno(errno)
                .build()
        }
        defer { _ = Darwin.close(fd) }
        if ftruncate(fd, off_t(fileSize)) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "container_resize")
                .message("Failed to resize container file")
                .path(path)
                .info("size", "\(fileSize)")
                .errno(errno)
                .build()
        }

        // Map and write
        guard let base = mmap(nil, Int(fileSize), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0), base != MAP_FAILED else {
            throw ErrorBuilder(.mmapError, operation: "container_mmap")
                .message("Failed to mmap container file")
                .path(path)
                .info("size", "\(fileSize)")
                .errno(errno)
                .build()
        }
        defer { _ = msync(base, Int(fileSize), MS_SYNC); _ = munmap(base, Int(fileSize)) }

        // Zero sections
        func zero(_ off: UInt64, _ size: UInt64) { memset(UnsafeMutableRawPointer(base).advanced(by: Int(off)), 0, Int(size)) }
        zero(listsDescOffset, listsDescSize)
        zero(idsOffset, idsSize)
        zero(payloadOffset, payloadSize)
        if includeIDMap {
            zero(idMapOffset, idMapSize)
        }

        // Helpers to store little-endian values
        @inline(__always) func storeLE32(_ ptr: UnsafeMutableRawPointer, _ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { bytes in
                memcpy(ptr, bytes.baseAddress!, 4)
            }
        }
        @inline(__always) func storeLE64(_ ptr: UnsafeMutableRawPointer, _ v: UInt64) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { bytes in
                memcpy(ptr, bytes.baseAddress!, 8)
            }
        }

        // Write ListsDesc records (packed 64 bytes)
        for i in 0..<k_c {
            let recBase = UnsafeMutableRawPointer(base).advanced(by: Int(listsDescOffset &+ UInt64(i) &* DISK_LISTDESC_SIZE))
            // Zero record
            memset(recBase, 0, Int(DISK_LISTDESC_SIZE))
            // Byte fields
            recBase.storeBytes(of: UInt8((format == .pq8) ? 2 : (format == .pq4 ? 3 : 1)), as: UInt8.self) // format at +0
            recBase.advanced(by: 1).storeBytes(of: UInt8(group), as: UInt8.self)                           // group at +1
            recBase.advanced(by: 2).storeBytes(of: UInt8(idBits), as: UInt8.self)                          // id_bits at +2
            recBase.advanced(by: 3).storeBytes(of: UInt8(0), as: UInt8.self)                               // reserved at +3
            // u32 fields
            storeLE32(recBase.advanced(by: 4), 0)                                                          // length
            storeLE32(recBase.advanced(by: 8), UInt32(idCap))                                              // capacity
            // u64 offsets (section-relative)
            storeLE64(recBase.advanced(by: 16), perListIDsOff[i])                                          // ids_offset
            if format == .flat {
                storeLE64(recBase.advanced(by: 32), perListPayloadOff[i])                                  // vecs_offset at +32
            } else {
                storeLE64(recBase.advanced(by: 24), perListPayloadOff[i])                                  // codes_offset at +24
            }
            // strides
            storeLE32(recBase.advanced(by: 40), UInt32(idStride))
            storeLE32(recBase.advanced(by: 44), UInt32(codesStride))
            storeLE32(recBase.advanced(by: 48), UInt32(vecsStride))
            storeLE32(recBase.advanced(by: 52), 0)                                                         // reserved1
        }

        // Build TOC entries (packed 36 bytes per entry)
        let DISK_TOC_ENTRY_SIZE: UInt64 = 36
        @inline(__always) func writeTOCEntry(_ idx: Int, type: UInt32, offset: UInt64, size: UInt64, align: UInt32, flags: UInt32, crc32: UInt32, reserved: UInt32) {
            let basePtr = UnsafeMutableRawPointer(base).advanced(by: Int(tocOffset &+ UInt64(idx) &* DISK_TOC_ENTRY_SIZE))
            // Packed layout (36 bytes): type@0 (u32), offset@4 (u64), size@12 (u64), align@20 (u32), flags@24 (u32), crc@28 (u32), reserved@32 (u32)
            storeLE32(basePtr.advanced(by: 0), type)
            storeLE64(basePtr.advanced(by: 4), offset)
            storeLE64(basePtr.advanced(by: 12), size)
            storeLE32(basePtr.advanced(by: 20), align)
            storeLE32(basePtr.advanced(by: 24), flags)
            storeLE32(basePtr.advanced(by: 28), crc32)
            storeLE32(basePtr.advanced(by: 32), reserved)
        }
        // ListsDesc entry
        writeTOCEntry(0, type: SectionType.listsDesc.rawValue, offset: listsDescOffset, size: listsDescSize, align: 64, flags: 0, crc32: 0, reserved: 0)
        // IDs entry
        writeTOCEntry(1, type: SectionType.ids.rawValue, offset: idsOffset, size: idsSize, align: 64, flags: 0, crc32: 0, reserved: 0)
        // Payload entry
        let payloadType: SectionType = (format == .flat) ? .vecs : .codes
        writeTOCEntry(2, type: payloadType.rawValue, offset: payloadOffset, size: payloadSize, align: UInt32(getpagesize()), flags: 0, crc32: 0, reserved: 0)
        if includeIDMap {
            writeTOCEntry(3, type: SectionType.idMap.rawValue, offset: idMapOffset, size: idMapSize, align: 64, flags: 0, crc32: 0, reserved: 0)
        }

        // Compute CRCs over sections and write back into TOC entries
        func writeCRC(at index: Int, offset: UInt64, size: UInt64) {
            let p = UnsafeRawPointer(base).advanced(by: Int(offset))
            let c = _CRC32.hash(p, Int(size))
            let crcPtr = UnsafeMutableRawPointer(base).advanced(by: Int(tocOffset &+ UInt64(index) &* DISK_TOC_ENTRY_SIZE + 28))
            storeLE32(crcPtr, c)
        }
        writeCRC(at: 0, offset: listsDescOffset, size: listsDescSize)
        writeCRC(at: 1, offset: idsOffset, size: idsSize)
        writeCRC(at: 2, offset: payloadOffset, size: payloadSize)
        if includeIDMap { writeCRC(at: 3, offset: idMapOffset, size: idMapSize) }

        // Write header with CRC
        let hdr = _Header(
            magic: UInt64(0x00585845444E4956),
            version_major: 1,
            version_minor: 0,
            endianness: 1,
            arch: 0,
            flags: 0,
            d: UInt32(d),
            m: UInt16(m),
            ks: 0,
            kc: UInt32(k_c),
            id_bits: UInt8(idBits),
            code_group_g: UInt8(group),
            reservedA: (0, 0, 0, 0, 0, 0),
            N_total: 0,
            generation: 0,
            toc_offset: tocOffset,
            toc_entries: UInt32(tocCount),
            header_crc32: 0,
            reservedRest: (0, 0, 0, 0, 0, 0, 0)
        )
        // Compute header CRC over 256 bytes with crc field zeroed
        let hdrPtr = UnsafeMutableRawPointer(base).assumingMemoryBound(to: _Header.self)
        hdrPtr.pointee = hdr
        // Zero CRC field in place then compute
        hdrPtr.pointee.header_crc32 = 0
        let crc = _CRC32.hash(UnsafeRawPointer(hdrPtr), 256)
        hdrPtr.pointee.header_crc32 = crc

        // Sync to disk
        _ = msync(base, Int(fileSize), MS_SYNC)

        // Return opened IndexMmap
        var o = MmapOpts()
        o.readOnly = false
        return try IndexMmap.open(path: path, opts: o)
    }
}
