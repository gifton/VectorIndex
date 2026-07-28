import XCTest
@testable import VectorIndex
import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

final class VIndexMmapErrorTests: XCTestCase {
    @inline(__always) private func readUnalignedLE32(_ p: UnsafeRawPointer) -> UInt32 {
        var v: UInt32 = 0
        memcpy(&v, p, 4)
        return UInt32(littleEndian: v)
    }
    @inline(__always) private func readUnalignedLE64(_ p: UnsafeRawPointer) -> UInt64 {
        var v: UInt64 = 0
        memcpy(&v, p, 8)
        return UInt64(littleEndian: v)
    }
    private func tempPath(_ suffix: String = ".vindex") -> String {
        URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("vindex_test_\(UUID().uuidString)\(suffix)").path
    }

    func testHeaderCRCMismatchThrows() throws {
        let path = tempPath()
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: 8, d: 0, includeIDMap: false)
        try mmap.close()
        defer { _ = try? FileManager.default.removeItem(atPath: path) }

        // Flip a reserved header byte (keep version/magic intact) without updating header CRC
        let fh = try XCTUnwrap(FileHandle(forUpdatingAtPath: path))
        defer { try? fh.close() }
        fh.seek(toFileOffset: 80) // reserved region beyond CRC field
        let b: UInt8 = 0
        fh.write(Data([b ^ 0xFF]))

        var opts = MmapOpts()
        opts.verifyCRCs = true
        do {
            _ = try IndexMmap.open(path: path, opts: opts)
            XCTFail("Expected header CRC mismatch to throw")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .corruptedData)
            XCTAssertTrue(e.message.lowercased().contains("header"))
        }
    }

    func testVersionMismatchThrows() throws {
        let path = tempPath()
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: 8, d: 0, includeIDMap: false)
        try mmap.close()
        defer { _ = try? FileManager.default.removeItem(atPath: path) }

        // Read and modify header.version_major to 2, then recompute header CRC
        let fd = Darwin.open(path, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { _ = Darwin.close(fd) }
        let hdrSize = 256
        guard let base = mmapFile(fd: fd, size: hdrSize) else { XCTFail("mmap header failed"); return }
        defer { _ = munmap(base, hdrSize) }
        // Update version_major at offset 8..9 (LE = 2)
        let majorOffset = 8
        base.advanced(by: majorOffset).storeBytes(of: UInt16(2), as: UInt16.self)
        // Zero header_crc32 at offset 68..71, recompute over 256 bytes, store back
        let crcOffset = 68
        base.advanced(by: crcOffset).storeBytes(of: UInt32(0), as: UInt32.self)
        let newCRC = CRC32.hash(UnsafeRawPointer(base), hdrSize)
        base.advanced(by: crcOffset).storeBytes(of: newCRC, as: UInt32.self)
        _ = msync(base, hdrSize, MS_SYNC)

        do {
            _ = try IndexMmap.open(path: path)
            XCTFail("Expected version mismatch to throw")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .versionMismatch)
            XCTAssertTrue(e.context.additionalInfo["version_major"] != nil)
        }
    }

    func testSectionCRCMismatchThrows() throws {
        let path = tempPath()
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: 8, d: 0, includeIDMap: false)
        try mmap.close()
        defer { _ = try? FileManager.default.removeItem(atPath: path) }

        // Corrupt one byte in IDs section (TOC[1] per builder layout)
        let fd = Darwin.open(path, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { _ = Darwin.close(fd) }
        // Map just header to get toc_offset and toc_entries (header is LE per builder)
        let hdrSize = 256
        var hdrBuf = [UInt8](repeating: 0, count: hdrSize)
        _ = hdrBuf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, hdrSize, 0) }
        // Use unaligned-safe little-endian loads; header layout puts toc_offset at +56 and toc_entries at +64
        let tocOffset = hdrBuf.withUnsafeBytes { raw -> UInt64 in
            readUnalignedLE64(raw.baseAddress!.advanced(by: 56))
        }
        let tocEntries = Int(hdrBuf.withUnsafeBytes { raw -> UInt32 in
            readUnalignedLE32(raw.baseAddress!.advanced(by: 64))
        })
        XCTAssertGreaterThanOrEqual(tocEntries, 2)
        // Load entire TOC and locate IDs entry by type for robustness
        let DISK_TOC_ENTRY_SIZE = 36
        var tocAll = [UInt8](repeating: 0, count: Int(tocEntries) * DISK_TOC_ENTRY_SIZE)
        let tocBytes = tocAll.count
        let gotTOC = tocAll.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }
        XCTAssertEqual(gotTOC, tocBytes)
        var idsOffset: UInt64 = 0
        var foundIDs = false
        tocAll.withUnsafeBytes { raw in
            for i in 0..<tocEntries {
                let base = raw.baseAddress!.advanced(by: i * DISK_TOC_ENTRY_SIZE)
                let ty = readUnalignedLE32(base)
                if ty == SectionType.ids.rawValue {
                    idsOffset = readUnalignedLE64(base.advanced(by: 4))
                    foundIDs = true
                    break
                }
            }
        }
        XCTAssertTrue(foundIDs, "IDs TOC entry not found")
        // Flip first byte of IDs section
        var one = [UInt8](repeating: 0, count: 1)
        _ = one.withUnsafeMutableBytes { pread(fd, $0.baseAddress, 1, off_t(idsOffset)) }
        one[0] ^= 0xFF
        _ = one.withUnsafeBytes { pwrite(fd, $0.baseAddress, 1, off_t(idsOffset)) }
        // Open with CRC verification
        var opts = MmapOpts(); opts.verifyCRCs = true
        do {
            _ = try IndexMmap.open(path: path, opts: opts)
            XCTFail("Expected section CRC mismatch to throw")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .corruptedData)
            XCTAssertTrue(e.message.lowercased().contains("section"))
        }
    }

    func testOpenMissingFileThrows() {
        let path = tempPath()
        do {
            _ = try IndexMmap.open(path: path)
            XCTFail("Expected missing file to throw")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .fileIOError)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testEnsureCapacityGrowOrRemapFailure() throws {
        // This test attempts a large growth to trigger either fileIOError (ftruncate) or mmapError (remap). If environment permits growth, skip.
        let path = tempPath()
        let k_c = 1, m = 8
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: k_c, m: m, d: 0, idBits: 64, group: 4, idCap: 1, payloadCap: 1, includeIDMap: false)
        defer { try? mmap.close(); _ = try? FileManager.default.removeItem(atPath: path) }
        // Try to reserve a very large number of codes to force growth
        do {
            _ = try mmap.mmap_append_begin(listID: 0, addLen: 200_000_000) // ~200M entries
            // If we got here, environment allowed sparse growth; skip assertion
            throw XCTSkip("Environment allowed huge growth; cannot reliably trigger remap/grow failure here.")
        } catch let e as VectorIndexError {
            // Either .fileIOError or .mmapError is acceptable depending on where it failed
            XCTAssert([IndexErrorKind.fileIOError, .mmapError].contains(e.kind))
        }
    }

    // MARK: - Helpers
    private func mmapFile(fd: Int32, size: Int) -> UnsafeMutableRawPointer? {
        let p = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0)
        return (p == MAP_FAILED) ? nil : p
    }

    // MARK: - WAL replay (B13) — no prior coverage existed for mmap_wal_replay at all.

    /// Locates the ListsDesc section's file offset by parsing the header + TOC directly
    /// (same technique as testSectionCRCMismatchThrows above), then overwrites list
    /// `listID`'s packed `length` field (record-relative offset +4) in place.
    private func setListLength(path: String, listID: Int, newLength: UInt32) throws {
        let fd = Darwin.open(path, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { _ = Darwin.close(fd) }
        var hdrBuf = [UInt8](repeating: 0, count: 256)
        _ = hdrBuf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, 256, 0) }
        let tocOffset = hdrBuf.withUnsafeBytes { readUnalignedLE64($0.baseAddress!.advanced(by: 56)) }
        let tocEntries = Int(hdrBuf.withUnsafeBytes { readUnalignedLE32($0.baseAddress!.advanced(by: 64)) })
        let DISK_TOC_ENTRY_SIZE = 36
        var tocAll = [UInt8](repeating: 0, count: tocEntries * DISK_TOC_ENTRY_SIZE)
        let tocBytes = tocAll.count
        _ = tocAll.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }
        var listsDescOffset: UInt64 = 0
        var found = false
        tocAll.withUnsafeBytes { raw in
            for i in 0..<tocEntries {
                let base = raw.baseAddress!.advanced(by: i * DISK_TOC_ENTRY_SIZE)
                if readUnalignedLE32(base) == SectionType.listsDesc.rawValue {
                    listsDescOffset = readUnalignedLE64(base.advanced(by: 4))
                    found = true
                    break
                }
            }
        }
        XCTAssertTrue(found, "ListsDesc TOC entry not found")
        var v = newLength.littleEndian
        let fieldOffset = off_t(listsDescOffset) + off_t(listID * 64 + 4)
        _ = withUnsafeBytes(of: &v) { pwrite(fd, $0.baseAddress, 4, fieldOffset) }
    }

    func testWalReplayAppliesLengthFromValidCommitRecord() throws {
        let path = tempPath()
        let m = 4
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 16, payloadCap: 16, includeIDMap: false)
        defer {
            try? mmap.close()
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }

        let n = 3
        let ids: [UInt64] = [10, 11, 12]
        let codes = [UInt8](repeating: 7, count: n * m)
        let res = try mmap.mmap_append_begin(listID: 0, addLen: n)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap.mmap_append_commit(res, idsSrc: UnsafeRawPointer(idBuf.baseAddress!), codesSrc: UnsafeRawPointer(codeBuf.baseAddress!), vecsSrc: nil)
            }
        }
        XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, n)

        // Simulate a crash where the fsync'd WAL made it to disk but the separate,
        // synchronous listsDesc-length write did not: roll the on-disk length back to 0.
        try setListLength(path: path, listID: 0, newLength: 0)
        XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, 0, "sanity: rollback landed")

        try mmap.mmap_wal_replay()
        XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, n,
                       "replay must restore length from the validated WAL commit record")
    }

    func testWalReplayStopsAtCorruptAppendRecordCRC() throws {
        let path = tempPath()
        let m = 4
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 16, payloadCap: 16, includeIDMap: false)
        defer {
            try? mmap.close()
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }

        let n = 3
        let ids: [UInt64] = [10, 11, 12]
        let codes = [UInt8](repeating: 7, count: n * m)
        let res = try mmap.mmap_append_begin(listID: 0, addLen: n)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap.mmap_append_commit(res, idsSrc: UnsafeRawPointer(idBuf.baseAddress!), codesSrc: UnsafeRawPointer(codeBuf.baseAddress!), vecsSrc: nil)
            }
        }
        try setListLength(path: path, listID: 0, newLength: 0)

        // Flip a byte inside the WAL append record's CRC field. writeWalAppend runs before
        // writeWalCommit, so this is the very first record in the .wal file: WalAppend is
        // 44 bytes (tag4+listID4+oldLen4+delta4+idsOff8+codesOff8+vecsOff8+crc32(4)), so its
        // crc32 field is at absolute file offset 40..43.
        let walPath = path + ".wal"
        let walFD = Darwin.open(walPath, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(walFD, 0)
        defer { _ = Darwin.close(walFD) }
        var crcByte: UInt8 = 0
        _ = withUnsafeMutableBytes(of: &crcByte) { pread(walFD, $0.baseAddress, 1, 40) }
        crcByte ^= 0xFF
        _ = withUnsafeBytes(of: &crcByte) { pwrite(walFD, $0.baseAddress, 1, 40) }

        try mmap.mmap_wal_replay()

        XCTAssertEqual(mmap.getListDescriptor(listID: 0)?.length, 0,
                       "corrupt append record must halt replay before the commit record is ever applied")
    }
}
