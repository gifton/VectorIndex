import XCTest
@testable import VectorIndex

final class MicrokernelIntegrationTests: XCTestCase {
    func testFlatOptimizedUsesKernelsWhenContiguous_Euclidean() async throws {
        let d = 64, n = 128
        let flat = FlatIndex(dimension: d, metric: .euclidean)
        let opt = FlatIndexOptimized(dimension: d, metric: .euclidean)

        for i in 0..<n {
            let v = (0..<d).map { _ in Float.random(in: -1...1) }
            try await flat.insert(id: "id_\(i)", vector: v, metadata: nil)
            try await opt.insert(id: "id_\(i)", vector: v, metadata: nil)
        }

        let q = (0..<d).map { _ in Float.random(in: -1...1) }
        let k = 10
        let r1 = try await flat.search(query: q, k: k, filter: nil)
        let r2 = try await opt.search(query: q, k: k, filter: nil)

        XCTAssertEqual(r1.map { $0.id }, r2.map { $0.id })
        // Distances should be very close (sqrt of L2^2 vs scalar)
        for (a, b) in zip(r1, r2) {
            XCTAssertLessThan(abs(a.score - b.score), 1e-4)
        }
    }

    func testFlatOptimizedUsesKernelsWhenContiguous_Dot() async throws {
        let d = 64, n = 128
        let flat = FlatIndex(dimension: d, metric: .dotProduct)
        let opt = FlatIndexOptimized(dimension: d, metric: .dotProduct)

        for i in 0..<n {
            let v = (0..<d).map { _ in Float.random(in: -1...1) }
            try await flat.insert(id: "id_\(i)", vector: v, metadata: nil)
            try await opt.insert(id: "id_\(i)", vector: v, metadata: nil)
        }

        let q = (0..<d).map { _ in Float.random(in: -1...1) }
        let k = 10
        let r1 = try await flat.search(query: q, k: k, filter: nil)
        let r2 = try await opt.search(query: q, k: k, filter: nil)

        XCTAssertEqual(r1.map { $0.id }, r2.map { $0.id })
        for (a, b) in zip(r1, r2) { XCTAssertLessThan(abs(a.score - b.score), 1e-4) }
    }
}
