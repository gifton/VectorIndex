import XCTest
@testable import VectorIndex

final class Kernel30AppendTests: XCTestCase {
    func testPQ8AppendAndGrowth() throws {
        var opts = IVFAppendOpts.default
        opts.format = .pq8
        opts.reserve_min = 2
        let m = 8
        let k_c = 2
        let h = try IVFListHandle(k_c: k_c, m: m, d: 0, opts: opts)

        // Two vectors to list 0, one to list 1
        let listIDs: [Int32] = [0, 0, 1]
        let extIDs: [UInt64] = [10, 11, 20]
        var codes = [UInt8](repeating: 0, count: listIDs.count * m)
        for i in 0..<listIDs.count { for j in 0..<m { codes[i*m + j] = UInt8(i*10 + j) } }

        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: codes, n: listIDs.count, m: m, index: h, opts: h.opts, internalIDsOut: nil)

        let stats0 = try h.getListStats(listID: 0)
        let stats1 = try h.getListStats(listID: 1)
        XCTAssertEqual(stats0.length, 2)
        XCTAssertEqual(stats1.length, 1)

        // Verify codes on list 0
        let (len0, _, _, codesPtr0, _) = try h.readList(listID: 0)
        XCTAssertEqual(len0, 2)
        let got0 = Array(UnsafeBufferPointer<UInt8>(start: codesPtr0, count: 2*m))
        XCTAssertEqual(got0[0..<m], codes[0..<m])
        XCTAssertEqual(got0[m..<(2*m)], codes[m..<(2*m)])

        // Growth: append 10 more to list 0 to force capacity increase
        let add = 10
        let ids2 = [UInt64](repeating: 100, count: add)
        let list2 = [Int32](repeating: 0, count: add)
        let codes2 = [UInt8](repeating: 7, count: add * m)
        try ivf_append(list_ids: list2, external_ids: ids2, codes: codes2, n: add, m: m, index: h, opts: h.opts, internalIDsOut: nil)
        let stats0b = try h.getListStats(listID: 0)
        XCTAssertEqual(stats0b.length, 2 + add)
        XCTAssertGreaterThanOrEqual(stats0b.capacity, stats0b.length)
    }

    func testPQ4PackedAndUnpacked() throws {
        var opts = IVFAppendOpts.default
        opts.format = .pq4
        opts.reserve_min = 2
        let m = 8 // even
        let h = try IVFListHandle(k_c: 1, m: m, d: 0, opts: opts)
        // Prepare two vectors of half-byte codes
        var unpacked = [UInt8](repeating: 0, count: 2*m)
        for i in 0..<(2*m) { unpacked[i] = UInt8(i & 0x0F) }
        // Append unpacked
        var listIDs: [Int32] = [0, 0]
        var extIDs: [UInt64] = [1, 2]
        var localOpts = h.opts; localOpts.pack4_unpacked = true
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: unpacked, n: 2, m: m, index: h, opts: localOpts, internalIDsOut: nil)
        // Repack locally to compare
        func pack(_ src: [UInt8]) -> [UInt8] {
            var out = [UInt8](repeating: 0, count: (src.count/2))
            var j=0; while j < src.count { out[j>>1] = (src[j]&0xF) | ((src[j+1]&0xF)<<4); j+=2 }; return out
        }
        let expected = pack(Array(unpacked[0..<m])) + pack(Array(unpacked[m..<(2*m)]))
        let (_, _, _, codesPtr, _) = try h.readList(listID: 0)
        let got = Array(UnsafeBufferPointer<UInt8>(start: codesPtr, count: expected.count))
        XCTAssertEqual(got, expected)
        // Now append packed path
        let packed = expected
        listIDs = [0]
        extIDs = [3]
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: packed, n: 1, m: m, index: h, opts: h.opts, internalIDsOut: nil)
        let stats = try h.getListStats(listID: 0)
        XCTAssertEqual(stats.length, 3)
    }

    func testFlatAppendAndInsert() throws {
        var opts = IVFAppendOpts.default
        opts.format = .flat
        opts.reserve_min = 2
        let d = 4
        let h = try IVFListHandle(k_c: 1, m: 0, d: d, opts: opts)
        let listIDs: [Int32] = [0, 0]
        let extIDs: [UInt64] = [10, 11]
        let xb: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        try ivf_append_flat(list_ids: listIDs, external_ids: extIDs, xb: xb, n: 2, d: d, index: h, opts: h.opts, internalIDsOut: nil)
        let stats = try h.getListStats(listID: 0)
        XCTAssertEqual(stats.length, 2)
        // Insert at position 1
        let insIDs: [UInt64] = [99]
        let insXB: [Float] = [9, 9, 9, 9]
        try ivf_insert_at_flat(list_id: 0, pos: 1, external_ids: insIDs, xb: insXB, n: 1, index: h)
        let stats2 = try h.getListStats(listID: 0)
        XCTAssertEqual(stats2.length, 3)
        // Check ordering: [1..4], [9..9], [5..8]
        let (_, _, _, _, xbPtr) = try h.readList(listID: 0)
        let values = Array(UnsafeBufferPointer<Float>(start: xbPtr, count: 3*d))
        XCTAssertEqual(Array(values[0..<4]), [1, 2, 3, 4])
        XCTAssertEqual(Array(values[4..<8]), [9, 9, 9, 9])
        XCTAssertEqual(Array(values[8..<12]), [5, 6, 7, 8])
    }

    func testDurablePQ8AppendWithRemap() throws {
        // Build a minimal container with small codes capacity to force remap; pre-size IDs larger
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("vindex_test_\(UUID().uuidString).vindex").path
        let k_c = 1
        let m = 8
        // IDs pre-sized to 32, codes small (4) so we extend only codes (placed last)
        let mmap = try VIndexContainerBuilder.createMinimalContainer(path: tmp, format: .pq8, k_c: k_c, m: m, d: 0, idBits: 64, group: 4, idCap: 32, payloadCap: 4)
        defer { try? mmap.close(); _ = try? FileManager.default.removeItem(atPath: tmp) }

        var opts = IVFAppendOpts.default
        opts.format = .pq8
        opts.durable = true
        let h = try ivf_create_mmap(k_c: k_c, m: m, d: 0, mmap: mmap, opts: opts)

        // Append 10 vectors to trigger growth beyond payloadCap=4
        let n = 10
        let listIDs = [Int32](repeating: 0, count: n)
        let extIDs = (0..<n).map { UInt64($0 + 100) }
        var codes = [UInt8](repeating: 0, count: n * m)
        for i in 0..<n { for j in 0..<m { codes[i*m + j] = UInt8(1 + j) } }
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: codes, n: n, m: m, index: h, opts: opts, internalIDsOut: nil)

        // Read back via durable path (mapped pointers)
        let (len, idsU64, _, codesPtr, _) = try h.readList(listID: 0)
        XCTAssertEqual(len, n)
        XCTAssertNotNil(idsU64)
        XCTAssertNotNil(codesPtr)
        // Spot check first row codes
        let first = Array(UnsafeBufferPointer<UInt8>(start: codesPtr, count: m))
        XCTAssertEqual(first, (0..<m).map { j in UInt8(1 + j) })
    }
}
