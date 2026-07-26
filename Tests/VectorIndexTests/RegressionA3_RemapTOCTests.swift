import XCTest
@testable import VectorIndex

final class RegressionA3_RemapTOCTests: XCTestCase {
    /// Guard test for spec bug A3: ensureFileCapacity's TOC re-parse on the mmap grow/remap
    /// path must use the same packed field offsets (4/12/20/24/28) as the writer/indexInit,
    /// not the previously-mismatched 8/16/24/28/32. If a growth event ever forces a real
    /// ftruncate+munmap+mmap remap, every tocByType entry gets rebuilt from disk; with the
    /// old (buggy) offsets every section's offset/size decodes to garbage
    /// (offset = size<<32, size = align<<32) and secCodes/secIDs/etc. become wild pointers.
    ///
    /// This exact scenario (idCap=32, payloadCap=4, n=10 -- mirroring
    /// testDurablePQ8AppendWithRemap in Kernel30AppendTests.swift) does NOT actually drive
    /// ensureFileCapacity's remap branch: VIndexContainerBuilder writes the list descriptor's
    /// single shared "capacity" field from idCap (not payloadCap), so mmap_append_begin's
    /// `need > cap` growth gate never fires (10 <= 32) and codes are written into the
    /// page-rounded slack that already exists past the declared payload size. So a fresh
    /// reopen-and-reread passes today regardless of the TOC-offset bug (see task-4-report.md
    /// for the manual investigation that forced a genuine remap and observed a crash with the
    /// pre-fix offsets). Kept here as a regression guard and as documentation of the
    /// consistency correction; not an independently-reproducing red test.
    func testRemapThenReopenPreservesSections() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vindex_a3_\(UUID().uuidString).vindex").path
        let k_c = 1, m = 8
        let mmap = try VIndexContainerBuilder.createMinimalContainer(
            path: tmp, format: .pq8, k_c: k_c, m: m, d: 0,
            idBits: 64, group: 4, idCap: 32, payloadCap: 4)  // payloadCap=4 forces remap (see note above: it doesn't)
        var opts = IVFAppendOpts.default; opts.format = .pq8; opts.durable = true
        let h = try ivf_create_mmap(k_c: k_c, m: m, d: 0, mmap: mmap, opts: opts)

        let n = 10   // > payloadCap => intended to trigger ensureFileCapacity remap
        let listIDs = [Int32](repeating: 0, count: n)
        let extIDs = (0..<n).map { UInt64($0 + 100) }
        var codes = [UInt8](repeating: 0, count: n * m)
        for i in 0..<n { for j in 0..<m { codes[i*m + j] = UInt8(1 + j) } }
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: codes,
                       n: n, m: m, index: h, opts: opts, internalIDsOut: nil)
        try mmap.close()

        // Reopen fresh: indexInit parses the TOC. If the remap (were it to fire) wrote a
        // self-consistent file, a fresh read of list 0 must return the appended codes.
        var reopenOpts = MmapOpts(); reopenOpts.readOnly = true
        let reopened = try IndexMmap.open(path: tmp, opts: reopenOpts)
        defer { try? reopened.close(); _ = try? FileManager.default.removeItem(atPath: tmp) }
        let h2 = try ivf_create_mmap(k_c: k_c, m: m, d: 0, mmap: reopened, opts: opts)
        let (len, _, _, codesPtr, _) = try h2.readList(listID: 0)
        XCTAssertEqual(len, n)
        let first = Array(UnsafeBufferPointer<UInt8>(start: codesPtr!, count: m))
        XCTAssertEqual(first, (0..<m).map { j in UInt8(1 + j) })
    }
}
