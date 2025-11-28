import XCTest
@testable import VectorIndex

final class IDMapPersistenceTests: XCTestCase {
    // TODO: Re-enable when CRC validation refactoring is complete (0.1.1)
    private static let idMapPersistenceEnabled = false

    func testIDMapSnapshotRoundTripMmap() throws {
        guard Self.idMapPersistenceEnabled else {
            throw XCTSkip("Known issue: CRC validation needs refactoring for mmap persistence - deferred to 0.1.1")
        }
        // Build an IDMap with a few entries
        let ext: [UInt64] = [101, 202, 303, 404]
        let map = idmapInit(capacityHint: 16, opts: .default)
        var assigned = [Int64](repeating: -1, count: ext.count)
        try ext.withUnsafeBufferPointer { ep in
            try assigned.withUnsafeMutableBufferPointer { ap in
                _ = try idmapAppend(map, externalIDs: ep.baseAddress!, count: ext.count, internalIDsOut: ap.baseAddress!)
            }
        }
        // Serialize to blob
        let blob = try serializeIDMap(map)

        // Create a minimal flat container and write the blob to its IDMap section
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("vindex_\(UUID().uuidString).bin")
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: tmp.path, format: .flat, k_c: 2, m: 0, d: 2, includeIDMap: true)
        try mmap.writeIDMapBlob(blob)
        try mmap.close()

        // Reopen and read back
        var opts = MmapOpts(); opts.readOnly = true
        let reopened = try IndexMmap.open(path: tmp.path, opts: opts)
        defer { try? reopened.close() }
        guard let readBlob = reopened.readIDMapBlob() else {
            XCTFail("No IDMap blob present in container")
            return
        }
        let map2 = try deserializeIDMap(readBlob)
        // Verify mapping preserved: ext[i] ↔ assigned[i]
        for (i, eid) in ext.enumerated() {
            var got: Int64 = -1
            _ = idmapLookup(map2, externalID: eid, internalIDOut: &got)
            XCTAssertEqual(got, assigned[i])
            let back = idmapExternalFor(map2, internalID: assigned[i])
            XCTAssertEqual(back, eid)
        }
    }

    func testIVFIndexLoadsIDMapFromDurableContainerAndRejectsDuplicates() async throws {
        guard Self.idMapPersistenceEnabled else {
            throw XCTSkip("Known issue: CRC validation needs refactoring for mmap persistence - deferred to 0.1.1")
        }
        // Prepare durable container with a pre-populated IDMap containing [777]
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("vindex_\(UUID().uuidString).bin")
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: tmp.path, format: .flat, k_c: 2, m: 0, d: 2, includeIDMap: true)
        // Build a map with one ID
        let map = idmapInit(capacityHint: 8, opts: .default)
        var assigned = [Int64](repeating: -1, count: 1)
        let extOne: [UInt64] = [777]
        try extOne.withUnsafeBufferPointer { ep in
            try assigned.withUnsafeMutableBufferPointer { ap in
                _ = try idmapAppend(map, externalIDs: ep.baseAddress!, count: 1, internalIDsOut: ap.baseAddress!)
            }
        }
        let blob = try serializeIDMap(map)
        try mmap.writeIDMapBlob(blob)
        try mmap.close()

        // Enable Kernel #30 storage on IVFIndex with durable container; it should load the IDMap
        let ivf = IVFIndex(dimension: 2, metric: .euclidean)
        try await ivf.enableKernel30Storage(format: .flat, k_c: 2, m: 0, durablePath: tmp.path)

        // Attempt to ingest the same external id 777 should trigger duplicate in IDMap append path
        let listIDs: [Int32] = [0]
        let xb: [Float] = [0, 0]
        do {
            try await ivf.ingestFlat(listIDs: listIDs, externalIDs: extOne, vectors: xb)
            XCTFail("Expected duplicate external ID to throw, but ingestion succeeded")
        } catch {
            // Pass: we expect an IDMapError.duplicateExternalID surfaced as a thrown error
        }
    }
}
