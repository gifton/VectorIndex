import XCTest
@testable import VectorIndex

final class RegressionA2_DurableListStatsTests: XCTestCase {
    func testDurableGetListStatsReturnsCapacity() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vindex_a2_\(UUID().uuidString).vindex").path
        let k_c = 1, m = 8
        let mmap = try VIndexContainerBuilder.createMinimalContainer(
            path: tmp, format: .pq8, k_c: k_c, m: m, d: 0,
            idBits: 64, group: 4, idCap: 32, payloadCap: 16)
        defer { try? mmap.close(); _ = try? FileManager.default.removeItem(atPath: tmp) }

        var opts = IVFAppendOpts.default
        opts.format = .pq8
        opts.durable = true
        let h = try ivf_create_mmap(k_c: k_c, m: m, d: 0, mmap: mmap, opts: opts)

        let n = 5
        let listIDs = [Int32](repeating: 0, count: n)
        let extIDs = (0..<n).map { UInt64($0 + 100) }
        var codes = [UInt8](repeating: 0, count: n * m)
        for i in 0..<n { for j in 0..<m { codes[i*m + j] = UInt8(1 + j) } }
        try ivf_append(list_ids: listIDs, external_ids: extIDs, codes: codes,
                       n: n, m: m, index: h, opts: opts, internalIDsOut: nil)

        // Before the fix this throws .contractViolation ("mmap list descriptors unavailable").
        let stats = try h.getListStats(listID: 0, durable: true)
        XCTAssertEqual(stats.length, n)
        XCTAssertGreaterThanOrEqual(stats.capacity, n)
        XCTAssertGreaterThan(stats.bytesIDs, 0)
    }
}
