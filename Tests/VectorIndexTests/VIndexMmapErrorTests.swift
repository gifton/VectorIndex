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

    /// Shared fixture for the P4 deferred-CRC tests below: a small, writable, pre-sized PQ8
    /// container (mirrors the pattern in Kernel30AppendTests.testDurablePQ8AppendWithRemap /
    /// RegressionA2 — `VIndexContainerBuilder.createMinimalContainer` then use directly).
    /// `idCap`/`payloadCap` are sized generously relative to the tests' commit counts so the
    /// (pre-existing, out-of-scope) growth-path limitations documented in MmapAppendBenchmark
    /// never engage.
    private func makeFixtureContainer(idCap: Int = 4096, payloadCap: Int = 4096, m: Int = 8) throws -> (IndexMmap, String) {
        let path = tempPath()
        let mmap = try VIndexContainerBuilder.createMinimalContainer(
            path: path, format: .pq8, k_c: 1, m: m, d: 0,
            idBits: 64, group: 4, idCap: idCap, payloadCap: payloadCap, includeIDMap: false)
        return (mmap, path)
    }

    // MARK: - P4: deferred section CRCs (quadratic -> linear ingestion)

    /// Failing-first test for the P4 design: `mmap_append_commit` must not hash any section
    /// bytes (that was the O(N^2) driver — each commit re-hashed the *entire* IDs/Codes/Vecs/
    /// ListsDesc sections). CRC freshness moves to `flush()` (and, transitively, `close()`).
    func testCommitPathDefersSectionCRCs() throws {
        let (mmap, path) = try makeFixtureContainer()
        defer {
            try? mmap.close()
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }
        let m = 8
        let addLen = 4
        let before = mmap.crcBytesHashed
        for c in 0..<50 {
            let ids: [UInt64] = (0..<addLen).map { UInt64(c * addLen + $0) }
            let codes = [UInt8](repeating: UInt8(truncatingIfNeeded: c), count: addLen * m)
            let res = try mmap.mmap_append_begin(listID: 0, addLen: addLen)
            try ids.withUnsafeBufferPointer { idBuf in
                try codes.withUnsafeBufferPointer { codeBuf in
                    try mmap.mmap_append_commit(res,
                        idsSrc: UnsafeRawPointer(idBuf.baseAddress!),
                        codesSrc: UnsafeRawPointer(codeBuf.baseAddress!),
                        vecsSrc: nil)
                }
            }
        }
        XCTAssertEqual(mmap.crcBytesHashed - before, 0,
            "commits must not hash section bytes; CRCs are deferred to flush/close")
        try mmap.flush()
        XCTAssertGreaterThan(mmap.crcBytesHashed - before, 0,
            "flush recomputes and persists section CRCs")
    }

    /// Crash-window test: an unclean shutdown (handle dropped without `close()`, so section CRCs
    /// are stale and the WAL is non-empty) must not make a reopen throw a spurious section-CRC
    /// mismatch. Reopening applies WAL replay (unchanged algorithm — see
    /// testWalReplayAppliesLengthFromValidCommitRecord), recomputes+persists CRCs, and truncates
    /// the WAL; a *second* reopen (now clean) verifies CRCs strictly, exactly as before P4.
    func testUncleanCloseThenReopenRecomputesCRCsViaWAL() throws {
        let path = tempPath()
        let m = 4
        var mmap: IndexMmap? = try VIndexContainerBuilder.createMinimalContainer(
            path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 16, payloadCap: 16, includeIDMap: false)
        defer {
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }

        let n = 3
        let ids: [UInt64] = [10, 11, 12]
        let codes = [UInt8](repeating: 7, count: n * m)
        let res = try mmap!.mmap_append_begin(listID: 0, addLen: n)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap!.mmap_append_commit(res,
                    idsSrc: UnsafeRawPointer(idBuf.baseAddress!),
                    codesSrc: UnsafeRawPointer(codeBuf.baseAddress!),
                    vecsSrc: nil)
            }
        }
        XCTAssertEqual(mmap!.getListDescriptor(listID: 0)?.length, n, "sanity: commit landed")
        XCTAssertEqual(mmap!.crcBytesHashed, 0,
            "sanity: the commit above must not have hashed any section bytes")

        // Simulate a crash: drop the handle WITHOUT close(), so flush() never runs. Section CRCs
        // on disk are now stale relative to the (already-durable, per-commit-msync'd) payload
        // bytes and ListsDesc length; the WAL still holds the append+commit records.
        mmap!._abandonWithoutClose()
        mmap = nil

        // Reopen (writable, so the repair branch can run): must NOT throw a section-CRC
        // mismatch despite the stale on-disk CRCs, because the non-empty WAL marks the
        // container unclean and strict verification is skipped in favor of replay + repair.
        var reopenOpts = MmapOpts(); reopenOpts.readOnly = false
        let reopened = try IndexMmap.open(path: path, opts: reopenOpts)
        XCTAssertEqual(reopened.getListDescriptor(listID: 0)?.length, n,
            "replay must restore the committed length")

        // Data reads back correctly through the repaired handle.
        let desc = try XCTUnwrap(reopened.getListDescriptor(listID: 0))
        let idsBase = try XCTUnwrap(reopened.idsBase())
        var gotIDs: [UInt64] = []
        for i in 0..<n {
            var v: UInt64 = 0
            memcpy(&v, idsBase.advanced(by: Int(desc.idsOff) + i * desc.idsStride), 8)
            gotIDs.append(v)
        }
        XCTAssertEqual(gotIDs, ids)
        try reopened.close()

        // Second reopen: the repair above truncated the WAL, so the container is now clean and
        // strict section-CRC verification must pass (exactly as it did pre-P4).
        var reopenOpts2 = MmapOpts(); reopenOpts2.readOnly = true
        let reopened2 = try IndexMmap.open(path: path, opts: reopenOpts2)
        XCTAssertEqual(reopened2.getListDescriptor(listID: 0)?.length, n)
        try reopened2.close()
    }

    /// Task 16a (audit finding): a READ-ONLY open of a crash-dirty container with
    /// `verifyCRCs = true` must fail loud, not silently skip verification. Before this fix,
    /// `indexInit`'s strict-CRC gate was `opts.verifyCRCs && sz > 0 && !walDirty` for ALL opens,
    /// and the replay+recompute repair only ran `if !opts.readOnly` — so a dirty container opened
    /// read-only got neither verification nor repair, and `opts.verifyCRCs = true` became a
    /// silent no-op. Reuses `testUncleanCloseThenReopenRecomputesCRCsViaWAL`'s crash simulation
    /// (`_abandonWithoutClose()` after a commit, leaving the WAL non-empty i.e. "dirty") but,
    /// unlike that test, goes straight to a READ-ONLY reopen (no interim writable repair) to
    /// exercise exactly the gap the audit flagged.
    func testReadOnlyOpenOfDirtyContainerWithVerifyCRCsThrows() throws {
        let path = tempPath()
        let m = 4
        var mmap: IndexMmap? = try VIndexContainerBuilder.createMinimalContainer(
            path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 16, payloadCap: 16, includeIDMap: false)
        defer {
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }

        let n = 3
        let ids: [UInt64] = [10, 11, 12]
        let codes = [UInt8](repeating: 7, count: n * m)
        let res = try mmap!.mmap_append_begin(listID: 0, addLen: n)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap!.mmap_append_commit(res,
                    idsSrc: UnsafeRawPointer(idBuf.baseAddress!),
                    codesSrc: UnsafeRawPointer(codeBuf.baseAddress!),
                    vecsSrc: nil)
            }
        }

        // Simulate a crash: drop the handle WITHOUT close(), so flush() never runs and the WAL
        // is left non-empty (dirty).
        mmap!._abandonWithoutClose()
        mmap = nil

        var reopenOpts = MmapOpts(); reopenOpts.readOnly = true; reopenOpts.verifyCRCs = true
        do {
            _ = try IndexMmap.open(path: path, opts: reopenOpts)
            XCTFail("Expected a read-only open of a dirty container with verifyCRCs=true to throw")
        } catch let e as VectorIndexError {
            // Reuses .corruptedData (Data Integrity family): the same kind this file already
            // throws for "read-only handle, CRC verification can't be trusted" (see the
            // ListsDesc-mismatch branch in indexInit()) rather than adding a new IndexErrorKind
            // case, since this task's change surface is scoped to VIndexMmap.swift +
            // VIndexMmapErrorTests.swift only.
            XCTAssertEqual(e.kind, .corruptedData)
            XCTAssertTrue(e.message.lowercased().contains("writable") || e.message.lowercased().contains("repair"),
                "error should direct the caller toward a writable open to repair the container")
        }
    }

    // MARK: - P5: ranged page-aligned msync + per-commit flush accounting

    /// Failing-first test for P5: `msyncPageAligned` must honor its `ptr`/`length` parameters
    /// (page-align the start down, the end up, clamp to the mapping) instead of always flushing
    /// the whole mapping via `msync(base, fileSize, MS_SYNC)` regardless of what was asked for.
    /// Task 3 (P4) already dropped the per-commit `updateSectionCRC` calls, so today's commit
    /// path has exactly 3 msync call sites for a PQ-format commit against this (codes-only)
    /// fixture: the IDs memcpy, the Codes memcpy, and the ListsDesc length write. (A flat/vecs
    /// format commit would add a 4th for the vecs memcpy.)
    func testCommitFlushesOnlyTouchedPages() throws {
        let (mmap, path) = try makeFixtureContainer()
        defer {
            try? mmap.close()
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }
        let m = 8
        let addLen = 4
        let ids: [UInt64] = (0..<addLen).map { UInt64($0) }
        let codes = [UInt8](repeating: 1, count: addLen * m)

        let pageSize = Int(getpagesize())
        let callsBefore = mmap.msyncCallCount
        let bytesBefore = mmap.msyncBytesFlushed
        let res = try mmap.mmap_append_begin(listID: 0, addLen: addLen)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap.mmap_append_commit(res,
                    idsSrc: UnsafeRawPointer(idBuf.baseAddress!),
                    codesSrc: UnsafeRawPointer(codeBuf.baseAddress!),
                    vecsSrc: nil)
            }
        }
        let calls = mmap.msyncCallCount - callsBefore
        let bytes = mmap.msyncBytesFlushed - bytesBefore
        XCTAssertEqual(calls, 3, "PQ commit = ids + codes + listsDesc flushes only")
        XCTAssertLessThanOrEqual(bytes, 3 * 2 * pageSize,
            "each flush covers only the touched range rounded to page boundaries, not fileSize")
    }

    /// Fix round 1 (review of commit `2b4b823`): `msyncPageAligned` now `throws` on a failed
    /// `msync(2)` instead of silently discarding the return value, and increments the new
    /// `msyncFailureCount` before throwing. Genuinely forcing `msync(2)` to fail is not portable
    /// to do safely in a unit test -- unlike `testEnsureCapacityGrowOrRemapFailure`'s `XCTSkip` for
    /// a real remap failure, or `testCloseAfterDanglingMappingLeavesOnDiskTOCUntouched`'s use of
    /// the `_simulateDanglingMapping()` test hook for a real dangling-pointer scenario, there is no
    /// safe way to simulate an EIO/ENOMEM/disk-full `msync` without either actually exhausting
    /// resources (unreliable, environment-dependent, and potentially harmful to the test host) or
    /// corrupting the mapping in a way that would crash the process rather than return an error
    /// code (the reviewer's own instruction explicitly rules out "msync on a deliberately-invalid-
    /// but-safe range" for exactly this reason). So this test pins the two things that ARE safely
    /// verifiable: (1) `msyncFailureCount` stays at 0 across a normal commit + `flush()` cycle --
    /// telemetry does not spuriously report failures when nothing failed; (2) the non-discardable
    /// shape is enforced at compile time, not by convention -- `msyncPageAligned` is `throws` (not
    /// `@discardableResult` on a `Bool`), so every one of its 10 call sites in `VIndexMmap.swift`
    /// requires an explicit `try` to even compile (verified: this file's callers -- and this test
    /// module's `try mmap.mmap_append_commit(...)` / `try mmap.flush()` below -- would fail to
    /// build if any call site dropped the `try` and let a thrown flush failure vanish silently).
    func testMsyncFailureCountStaysZeroOnNormalCommitAndFlush() throws {
        let (mmap, path) = try makeFixtureContainer()
        defer {
            try? mmap.close()
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }
        let m = 8
        let addLen = 4
        let ids: [UInt64] = (0..<addLen).map { UInt64($0) }
        let codes = [UInt8](repeating: 1, count: addLen * m)

        XCTAssertEqual(mmap.msyncFailureCount, 0, "sanity: no failures before any I/O")
        let res = try mmap.mmap_append_begin(listID: 0, addLen: addLen)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap.mmap_append_commit(res,
                    idsSrc: UnsafeRawPointer(idBuf.baseAddress!),
                    codesSrc: UnsafeRawPointer(codeBuf.baseAddress!),
                    vecsSrc: nil)
            }
        }
        XCTAssertEqual(mmap.msyncFailureCount, 0, "a normal commit must not record any msync failures")
        try mmap.flush()
        XCTAssertEqual(mmap.msyncFailureCount, 0, "a normal flush must not record any msync failures")
    }

    /// Fix round 2 (review of commit `245a932`): `writeListDescOffsets` (called only from
    /// `mmap_append_begin`'s growth branch) writes the new idsOff/codesOff/vecsOff/capacity into
    /// the mapped ListsDesc record BEFORE its `msyncPageAligned` call. If that msync throws (fix
    /// round 1), the growth branch's own `tailIDs`/`tailCodes`/`tailVecs` watermark update never
    /// runs (it's a few lines further down, reached only on success) -- so those in-memory
    /// watermarks go stale relative to the on-disk descriptor, which already reflects the new
    /// claim. A later growth on the SAME live handle would then compute its next offset from the
    /// stale watermark, silently overlapping the region the descriptor already claims. The fix:
    /// `growthPoisoned` is set in that failure path and checked at the top of every subsequent
    /// `mmap_append_begin` call, refusing further growth until the handle is reopened (the
    /// tested recovery path -- see `testGrowthWritesWalSentinelBeforeMutatingPayloadSections`).
    ///
    /// Forcing a real growth-path `msync` failure is the same non-portable injection problem as
    /// `testMsyncFailureCountStaysZeroOnNormalCommitAndFlush` above (and `XCTSkip`'d for the same
    /// reason in `testEnsureCapacityGrowOrRemapFailure`), so this pins the mechanism directly via
    /// the `_simulateGrowthPoisoned()` test hook (same pattern as `_simulateDanglingMapping()`
    /// for `mappingValid` in fix round 1, I3): set the flag exactly as the real failure path
    /// would, then assert `mmap_append_begin` throws the poison error instead of proceeding.
    func testGrowthPoisonedHandleRefusesFurtherAppendBegin() throws {
        let (mmap, path) = try makeFixtureContainer()
        defer {
            try? mmap.close()
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }
        // Sanity: an unpoisoned handle can begin an append normally.
        _ = try mmap.mmap_append_begin(listID: 0, addLen: 1)

        mmap._simulateGrowthPoisoned()
        do {
            _ = try mmap.mmap_append_begin(listID: 0, addLen: 1)
            XCTFail("Expected mmap_append_begin to throw once the handle is growth-poisoned")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .mmapError)
            XCTAssertTrue(e.message.lowercased().contains("reopen"),
                "error should direct the caller to reopen the container")
        }
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

    // MARK: - Fix round 1 (review of commit bf9566e): C1, I2, I3

    /// C1 (Critical), REQUIRED covering test per the reviewer's finding: a failed *writable*
    /// `open()` used to have deinit's `close()` call `flush()`, which recomputed CRCs from
    /// whatever (corrupt) bytes were on disk and persisted them — silently "repairing" the
    /// checksum over the corruption and destroying detectability on every future open. Reuses
    /// `testSectionCRCMismatchThrows`'s corruption technique but (a) opens `readOnly = false`
    /// (the vulnerable path — `testSectionCRCMismatchThrows` itself only exercises the default
    /// `readOnly = true`, which never called `flush()` even pre-fix) and (b) pins the on-disk TOC
    /// bytes byte-for-byte across the failed open, not just the thrown error.
    func testFailedWritableOpenDoesNotRewriteSectionCRCs() throws {
        let path = tempPath()
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: path, format: .pq8, k_c: 1, m: 8, d: 0, includeIDMap: false)
        try mmap.close()
        defer { _ = try? FileManager.default.removeItem(atPath: path); _ = try? FileManager.default.removeItem(atPath: path + ".wal") }

        let fd = Darwin.open(path, O_RDWR | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { _ = Darwin.close(fd) }
        let hdrSize = 256
        var hdrBuf = [UInt8](repeating: 0, count: hdrSize)
        _ = hdrBuf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, hdrSize, 0) }
        let tocOffset = hdrBuf.withUnsafeBytes { readUnalignedLE64($0.baseAddress!.advanced(by: 56)) }
        let tocEntries = Int(hdrBuf.withUnsafeBytes { readUnalignedLE32($0.baseAddress!.advanced(by: 64)) })
        XCTAssertGreaterThanOrEqual(tocEntries, 2)
        let DISK_TOC_ENTRY_SIZE = 36
        var tocAll = [UInt8](repeating: 0, count: tocEntries * DISK_TOC_ENTRY_SIZE)
        let tocBytes = tocAll.count
        XCTAssertEqual(tocAll.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }, tocBytes)
        var idsOffset: UInt64 = 0
        var foundIDs = false
        tocAll.withUnsafeBytes { raw in
            for i in 0..<tocEntries {
                let base = raw.baseAddress!.advanced(by: i * DISK_TOC_ENTRY_SIZE)
                if readUnalignedLE32(base) == SectionType.ids.rawValue {
                    idsOffset = readUnalignedLE64(base.advanced(by: 4))
                    foundIDs = true
                    break
                }
            }
        }
        XCTAssertTrue(foundIDs, "IDs TOC entry not found")
        var one = [UInt8](repeating: 0, count: 1)
        _ = one.withUnsafeMutableBytes { pread(fd, $0.baseAddress, 1, off_t(idsOffset)) }
        one[0] ^= 0xFF
        _ = one.withUnsafeBytes { pwrite(fd, $0.baseAddress, 1, off_t(idsOffset)) }

        // Snapshot the ENTIRE on-disk TOC (every entry's offset/size/align/flags/crc32) right
        // after corrupting, before the vulnerable open.
        var tocBefore = [UInt8](repeating: 0, count: tocBytes)
        XCTAssertEqual(tocBefore.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }, tocBytes)

        var opts = MmapOpts(); opts.readOnly = false; opts.verifyCRCs = true
        do {
            _ = try IndexMmap.open(path: path, opts: opts)
            XCTFail("Expected section CRC mismatch to throw on a writable open")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .corruptedData)
            XCTAssertTrue(e.message.lowercased().contains("section"))
        }

        var tocAfter = [UInt8](repeating: 0, count: tocBytes)
        XCTAssertEqual(tocAfter.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }, tocBytes)
        XCTAssertEqual(tocBefore, tocAfter,
            "a failed writable open must not rewrite ANY on-disk TOC bytes (CRCs included) -- " +
            "doing so would self-certify the corrupt section and permanently destroy detectability")

        // Re-open readOnly (deinit's close() never calls flush() for readOnly handles either
        // way) to confirm the corruption is still detected -- i.e. genuinely not repaired.
        var reopenOpts = MmapOpts(); reopenOpts.readOnly = true; reopenOpts.verifyCRCs = true
        do {
            _ = try IndexMmap.open(path: path, opts: reopenOpts)
            XCTFail("Expected the corruption to still be detected on a subsequent open")
        } catch let e as VectorIndexError {
            XCTAssertEqual(e.kind, .corruptedData)
        }
    }

    /// I2 (Important): the growth branch of `mmap_append_begin` mutates IDs/Codes sections
    /// directly (relocation memcpys) and refreshes only ListsDesc's own CRC, entirely outside
    /// the commit/WAL protocol. Without a pessimistic WAL write *before* that mutation, a crash
    /// between `mmap_append_begin` (growth done) and the matching `mmap_append_commit` would
    /// leave an empty WAL (the clean marker) next to stale IDs/Codes CRCs, and the next open
    /// would strict-verify and throw a false-positive corruption error.
    ///
    /// The growth branch always attempts IDs first, and — per the pre-existing, documented,
    /// out-of-scope defect ("only the last-by-offset section can grow, so IDs can never grow";
    /// see MmapAppendBenchmark's fixture comment) — that attempt always eventually fails its own
    /// "offset/capacity exceed section size" sanity check, because the IDs section's on-disk size
    /// is frozen at build time while growth always requests at least double the current capacity.
    /// That failure happens *after* the relocation memcpys, so it is actually the sharper version
    /// of the scenario I2 describes: real payload mutation happened, then the call unwound via a
    /// thrown error instead of continuing on into a matching `mmap_append_commit` at all. This
    /// test therefore expects `mmap_append_begin` to throw, and checks the WAL/reopen behavior
    /// around that.
    func testGrowthWritesWalSentinelBeforeMutatingPayloadSections() throws {
        let path = tempPath()
        let m = 4
        var mmap: IndexMmap? = try VIndexContainerBuilder.createMinimalContainer(
            path: path, format: .pq8, k_c: 1, m: m, d: 0, idCap: 2, payloadCap: 2, includeIDMap: false)
        defer {
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }
        let walPath = path + ".wal"

        var st = stat()
        XCTAssertEqual(stat(walPath, &st), 0)
        XCTAssertEqual(st.st_size, 0, "sanity: WAL starts empty on a freshly built container")

        // addLen=5 > idCap/payloadCap(=2) forces the growth branch, entirely inside begin -- no
        // commit has happened yet. Expected to throw once it reaches the IDs sanity check (see
        // doc comment above); the sentinel write and the relocation memcpys both happen first.
        do {
            _ = try mmap!.mmap_append_begin(listID: 0, addLen: 5)
            XCTFail("Expected the pre-existing IDs-cannot-grow limitation to throw here")
        } catch is VectorIndexError {
            // Expected -- see doc comment.
        }

        XCTAssertEqual(stat(walPath, &st), 0)
        XCTAssertGreaterThan(st.st_size, 0,
            "growth must pessimistically mark the WAL non-empty before mutating payload sections, " +
            "even when the growth attempt itself later fails")

        // Simulate a crash right after the failed growth attempt.
        mmap!._abandonWithoutClose()
        mmap = nil

        // Reopen (writable, so the unclean-repair branch can run) must not throw a spurious
        // section-CRC mismatch despite the growth branch having already mutated section bytes
        // before failing. List length must still be 0 -- no commit ever completed, so WAL replay
        // has nothing to apply.
        var reopenOpts = MmapOpts(); reopenOpts.readOnly = false
        let reopened = try IndexMmap.open(path: path, opts: reopenOpts)
        XCTAssertEqual(reopened.getListDescriptor(listID: 0)?.length, 0)
        try reopened.close()
    }

    /// I3 (Important): `ensureFileCapacity`'s remap `munmap()`s the old mapping before attempting
    /// the replacement `mmap()`; if that `mmap()` fails, `base` is left dangling for the rest of
    /// the handle's life. `close()`/`flush()` must detect that (via `mappingValid`) and skip all
    /// `base`-touching work instead of reading/writing through a dangling pointer (which would at
    /// best crash the process and at worst corrupt unrelated memory if the address range was
    /// since reused). A real remap failure is OS/environment-dependent and not reliably
    /// reproducible (see `testEnsureCapacityGrowOrRemapFailure`'s own `XCTSkip` for the same
    /// reason), so this uses the `_simulateDanglingMapping()` test hook to set the exact
    /// post-failed-remap flag state deterministically -- it flips `mappingValid` WITHOUT actually
    /// unmapping anything, so "close() doesn't crash" alone can't distinguish the guard being
    /// present from absent (the underlying memory stays perfectly valid either way in this
    /// simulation). The meaningful, guard-sensitive assertion instead: P4 already means the
    /// on-disk TOC CRCs are stale relative to the just-committed payload the moment this test
    /// simulates the dangle, so if `close()`'s flush() actually ran, it would rewrite those CRC
    /// bytes to match; if the `mappingValid` guard correctly skips it, they stay exactly as they
    /// were.
    func testCloseAfterDanglingMappingLeavesOnDiskTOCUntouched() throws {
        let (mmap, path) = try makeFixtureContainer()
        defer {
            _ = try? FileManager.default.removeItem(atPath: path)
            _ = try? FileManager.default.removeItem(atPath: path + ".wal")
        }
        let m = 8, addLen = 4
        let ids: [UInt64] = (0..<addLen).map { UInt64($0) }
        let codes = [UInt8](repeating: 1, count: addLen * m)
        let res = try mmap.mmap_append_begin(listID: 0, addLen: addLen)
        try ids.withUnsafeBufferPointer { idBuf in
            try codes.withUnsafeBufferPointer { codeBuf in
                try mmap.mmap_append_commit(res,
                    idsSrc: UnsafeRawPointer(idBuf.baseAddress!),
                    codesSrc: UnsafeRawPointer(codeBuf.baseAddress!),
                    vecsSrc: nil)
            }
        }
        XCTAssertEqual(mmap.crcBytesHashed, 0, "sanity: the commit above deferred all CRC hashing")

        // Snapshot the on-disk TOC right after the commit -- these CRC bytes are already stale
        // relative to the just-written payload (that staleness is the whole point of P4), so a
        // real flush() here would visibly change them.
        let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { _ = Darwin.close(fd) }
        var hdrBuf = [UInt8](repeating: 0, count: 256)
        _ = hdrBuf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, 256, 0) }
        let tocOffset = hdrBuf.withUnsafeBytes { readUnalignedLE64($0.baseAddress!.advanced(by: 56)) }
        let tocEntries = Int(hdrBuf.withUnsafeBytes { readUnalignedLE32($0.baseAddress!.advanced(by: 64)) })
        let DISK_TOC_ENTRY_SIZE = 36
        let tocBytes = tocEntries * DISK_TOC_ENTRY_SIZE
        var tocBefore = [UInt8](repeating: 0, count: tocBytes)
        XCTAssertEqual(tocBefore.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }, tocBytes)

        mmap._simulateDanglingMapping()
        XCTAssertNoThrow(try mmap.close(),
            "close() must not crash or throw when the mapping is marked invalid")

        var tocAfter = [UInt8](repeating: 0, count: tocBytes)
        XCTAssertEqual(tocAfter.withUnsafeMutableBytes { pread(fd, $0.baseAddress, tocBytes, off_t(tocOffset)) }, tocBytes)
        XCTAssertEqual(tocBefore, tocAfter,
            "close() must skip flush() entirely once mappingValid is false -- it must not " +
            "recompute/persist CRCs (or touch the mapping in any other way)")
    }
}
