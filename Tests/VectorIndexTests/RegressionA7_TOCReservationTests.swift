import XCTest
@testable import VectorIndex

final class RegressionA7_TOCReservationTests: XCTestCase {
    /// Guard test for spec bug A7, scope-corrected during Task 4/5 verification
    /// (see task-4-report.md §3a-2/§6 and task-5-report.md).
    ///
    /// `VIndexContainerBuilder.createMinimalContainer` computed `tocSize` from `tocCount = 3`
    /// (ListsDesc, IDs, Codes/Vecs) *before* `tocCount` was incremented to 4 for the optional
    /// IDMap TOC entry (`includeIDMap: true` is the default). The 4th 36-byte TOC entry
    /// (`writeTOCEntry(3, ...)` and especially `writeCRC(at: 3, ...)`) was then written into
    /// space that was never reserved, physically overlapping the start of the ListsDesc region
    /// and clobbering list 0's descriptor `capacity` field (record-relative offset +8) with the
    /// CRC32 of the (zeroed) IDMap section.
    ///
    /// This is a real, silent corruption of durable on-disk state (not "harmless slack" as the
    /// original A7 brief assumed for the stride-vs-packed-constant mismatch alone) that occurs
    /// on every `createMinimalContainer` call with the default `includeIDMap: true`.
    func testListZeroDescriptorCapacityNotClobberedByIDMapTOCEntry() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vindex_a7_\(UUID().uuidString).vindex").path
        defer { _ = try? FileManager.default.removeItem(atPath: tmp) }

        let k_c = 1, m = 8
        let idCap = 32
        let mmap = try VIndexContainerBuilder.createMinimalContainer(
            path: tmp, format: .pq8, k_c: k_c, m: m, d: 0,
            idBits: 64, group: 4, idCap: idCap, payloadCap: 16, includeIDMap: true)

        let desc = mmap.getListDescriptor(listID: 0)
        XCTAssertNotNil(desc, "expected list 0 descriptor to be readable")
        XCTAssertEqual(desc?.capacity, idCap,
                        "list 0 descriptor capacity was clobbered by the (unreserved) IDMap TOC "
                        + "entry's CRC write (got \(String(describing: desc?.capacity)), expected \(idCap))")

        try mmap.close()

        // Sanity: a fresh reopen must still report the same (correct, post-fix) capacity.
        var reopenOpts = MmapOpts()
        reopenOpts.readOnly = true
        let reopened = try IndexMmap.open(path: tmp, opts: reopenOpts)
        defer { try? reopened.close() }
        let desc2 = reopened.getListDescriptor(listID: 0)
        XCTAssertEqual(desc2?.capacity, idCap)
    }
}
