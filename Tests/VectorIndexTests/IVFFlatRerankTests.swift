import XCTest
@testable import VectorIndex

final class IVFFlatRerankTests: XCTestCase {
    func testKernel30FlatSearchRerank_Euclidean() async throws {
        let d = 2
        let idx = IVFIndex(dimension: d, metric: .euclidean)
        // Enable Kernel #30 storage (heap mode)
        try await idx.enableKernel30Storage(format: .flat, k_c: 2, m: 0, durablePath: nil)

        // Ingest a few IVF-Flat vectors across lists
        // list 0: [1,0] (id 100)
        // list 1: [0,1] (id 200), [0,0.9] (id 201)
        do {
            let listIDs: [Int32] = [0, 1, 1]
            let extIDs: [UInt64] = [100, 200, 201]
            let xb: [Float] = [
                1, 0,
                0, 1,
                0, 0.9
            ]
            try await idx.ingestFlat(listIDs: listIDs, externalIDs: extIDs, vectors: xb)
        }

        // Query near [1,0] should return 100 first
        do {
            let res = try await idx.search(query: [0.98, 0.02], k: 1, filter: nil)
            XCTAssertEqual(res.count, 1)
            XCTAssertEqual(res.first?.id, "100")
        }

        // Query near [0,1] should return 200
        do {
            let res = try await idx.search(query: [0.02, 0.99], k: 1, filter: nil)
            XCTAssertEqual(res.count, 1)
            XCTAssertEqual(res.first?.id, "200")
        }

        // Query near [0,0.9] should return 201 when k=1
        do {
            let res = try await idx.search(query: [0.01, 0.91], k: 1, filter: nil)
            XCTAssertEqual(res.count, 1)
            XCTAssertEqual(res.first?.id, "201")
        }
    }
}

