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
fileprivate struct _TOCEntry { var type: UInt32; var offset: UInt64; var size: UInt64; var align: UInt32; var flags: UInt32; var crc32: UInt32; var reserved: UInt32 }
fileprivate struct _Header {
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

        // ListsDesc section
        let listsDescOffset = off
        let listsDescSize = UInt64(k_c * MemoryLayout<ListDesc>.stride)
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
        guard fd >= 0 else { throw VIndexError.openFailed(errno: errno) }
        defer { _ = Darwin.close(fd) }
        if ftruncate(fd, off_t(fileSize)) != 0 { throw VIndexError.statFailed(errno: errno) }

        // Map and write
        guard let base = mmap(nil, Int(fileSize), PROT_READ | PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0), base != MAP_FAILED else {
            throw VIndexError.mmapFailed(errno: errno)
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

        // Write ListsDesc array
        let descsPtr = UnsafeMutableRawPointer(base).advanced(by: Int(listsDescOffset)).assumingMemoryBound(to: ListDesc.self)
        for i in 0..<k_c {
            var dsc = ListDesc(format: 0, group_g: UInt8(group), id_bits: UInt8(idBits), reserved0: 0, length: 0, capacity: UInt32(idCap), ids_offset: 0, codes_offset: 0, vecs_offset: 0, ids_stride: UInt32(idStride), codes_stride: UInt32(codesStride), vecs_stride: UInt32(vecsStride), reserved1: 0)
            switch format {
            case .pq8: dsc.format = 2
            case .pq4: dsc.format = 3
            case .flat: dsc.format = 1
            }
            dsc.length = 0
            dsc.capacity = UInt32(idCap)
            dsc.ids_offset = perListIDsOff[i]
            if format == .flat { dsc.vecs_offset = perListPayloadOff[i] } else { dsc.codes_offset = perListPayloadOff[i] }
            descsPtr.advanced(by: i).pointee = dsc
        }

        // Build TOC entries
        // All TOC fields are stored in little-endian (file) format
        let tocPtr = UnsafeMutableRawPointer(base).advanced(by: Int(tocOffset)).assumingMemoryBound(to: _TOCEntry.self)
        // ListsDesc entry
        tocPtr[0] = _TOCEntry(type: SectionType.listsDesc.rawValue, offset: listsDescOffset, size: listsDescSize, align: 64, flags: 0, crc32: 0, reserved: 0)
        tocPtr[1] = _TOCEntry(type: SectionType.ids.rawValue, offset: idsOffset, size: idsSize, align: 64, flags: 0, crc32: 0, reserved: 0)
        let payloadType: SectionType = (format == .flat) ? .vecs : .codes
        tocPtr[2] = _TOCEntry(type: payloadType.rawValue, offset: payloadOffset, size: payloadSize, align: UInt32(getpagesize()), flags: 0, crc32: 0, reserved: 0)
        if includeIDMap {
            tocPtr[3] = _TOCEntry(type: SectionType.idMap.rawValue, offset: idMapOffset, size: idMapSize, align: 64, flags: 0, crc32: 0, reserved: 0)
        }

        // Compute CRCs over sections and store directly (no endian conversion needed on LE)
        for i in 0..<tocCount {
            var te = tocPtr[i]
            let p = UnsafeRawPointer(base).advanced(by: Int(te.offset))
            te.crc32 = _CRC32.hash(p, Int(te.size))
            tocPtr[i] = te
        }

        // Write header with CRC
        var hdr = _Header(
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
            reservedA: (0,0,0,0,0,0),
            N_total: 0,
            generation: 0,
            toc_offset: tocOffset,
            toc_entries: UInt32(tocCount),
            header_crc32: 0,
            reservedRest: (0,0,0,0,0,0,0)
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
