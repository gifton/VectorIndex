import XCTest
import VectorCore
@testable import VectorIndex

final class TypedOverloadsTests: XCTestCase {
    func testFlatTypedInsertSearch() async throws {
        let idx = FlatIndex(dimension: 2)
        let v: DynamicVector = DynamicVector([1, 0])
        try await idx.insert(id: "a", vector: v, metadata: nil)
        let q: DynamicVector = DynamicVector([0.9, 0])
        let res = try await idx.search(query: q, k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
    }

    func testHNSWTypedBatch() async throws {
        let idx = HNSWIndex(dimension: 2)
        let items: [(String, DynamicVector, [String: String]?)] = [
            ("x", DynamicVector([0, 0]), nil), ("y", DynamicVector([1, 0]), nil)
        ]
        try await idx.batchInsert(items)
        let queries = [DynamicVector([1, 0])]
        let res = try await idx.batchSearch(queries: queries, k: 1, filter: nil)
        XCTAssertEqual(res.first?.first?.id, "y")
    }

    func testIVFTyped() async throws {
        let ivf = IVFIndex(dimension: 2, metric: .euclidean, config: .init(nlist: 2, nprobe: 1))
        try await ivf.batchInsert([("p", DynamicVector([0, 0]), nil), ("q", DynamicVector([1, 0]), nil)])
        try await ivf.optimize()
        let res = try await ivf.search(query: DynamicVector([0.99, 0]), k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "q")
    }

    // MARK: - FlatIndexOptimized typed overloads

    func testFlatOptimizedTypedInsertSearch() async throws {
        let idx = FlatIndexOptimized(dimension: 2)
        let v = DynamicVector([1, 0])
        try await idx.insert(id: "a", vector: v, metadata: nil)
        let res = try await idx.search(query: DynamicVector([0.9, 0]), k: 1, filter: nil)
        XCTAssertEqual(res.first?.id, "a")
    }

    // MARK: - NormalizationHint cosine fast-path tests

    /// Helper: generate a deterministic unit-normalized vector.
    private func makeNormalizedVector(dim: Int, seed: UInt64) -> [Float] {
        var state = seed
        var raw = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            raw[i] = Float(Int64(bitPattern: state >> 33)) / Float(Int64.max >> 33)
        }
        var mag: Float = 0
        for v in raw { mag += v * v }
        mag = sqrt(mag)
        guard mag > 1e-12 else { return raw }
        for i in 0..<dim { raw[i] /= mag }
        return raw
    }

    func testFlatCosineNormalizationHint() async throws {
        let dim = 16
        let idx = FlatIndex(dimension: dim, metric: .cosine)
        let raw = FlatIndex(dimension: dim, metric: .cosine)

        for i in 0..<20 {
            let v = makeNormalizedVector(dim: dim, seed: UInt64(i) &+ 100)
            try await idx.insert(id: "v\(i)", vector: v, metadata: nil)
            try await raw.insert(id: "v\(i)", vector: v, metadata: nil)
        }

        // Search with NormalizationHint-wrapped query
        let q = makeNormalizedVector(dim: dim, seed: 999)
        let hint = NormalizationHint(vector: DynamicVector(q), isNormalized: true)

        let rA = try await idx.search(query: hint, k: 5, filter: nil)
        let rB = try await raw.search(query: q, k: 5, filter: nil)

        XCTAssertEqual(rA.map(\.id), rB.map(\.id))
        for (a, b) in zip(rA, rB) {
            XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
        }
    }

    func testFlatOptimizedCosineNormalizationHint() async throws {
        let dim = 16
        let idx = FlatIndexOptimized(dimension: dim, metric: .cosine)
        let raw = FlatIndexOptimized(dimension: dim, metric: .cosine)

        for i in 0..<20 {
            let v = makeNormalizedVector(dim: dim, seed: UInt64(i) &+ 200)
            try await idx.insert(id: "v\(i)", vector: v, metadata: nil)
            try await raw.insert(id: "v\(i)", vector: v, metadata: nil)
        }

        let q = makeNormalizedVector(dim: dim, seed: 888)
        let hint = NormalizationHint(vector: DynamicVector(q), isNormalized: true)

        let rA = try await idx.search(query: hint, k: 5, filter: nil)
        let rB = try await raw.search(query: q, k: 5, filter: nil)

        XCTAssertEqual(rA.map(\.id), rB.map(\.id))
        for (a, b) in zip(rA, rB) {
            XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
        }
    }

    func testHNSWCosineNormalizationHint() async throws {
        let dim = 16
        let idx = HNSWIndex(dimension: dim, metric: .cosine)
        let raw = HNSWIndex(dimension: dim, metric: .cosine)

        for i in 0..<30 {
            let v = makeNormalizedVector(dim: dim, seed: UInt64(i) &+ 300)
            try await idx.insert(id: "v\(i)", vector: v, metadata: nil)
            try await raw.insert(id: "v\(i)", vector: v, metadata: nil)
        }

        let q = makeNormalizedVector(dim: dim, seed: 777)
        let hint = NormalizationHint(vector: DynamicVector(q), isNormalized: true)

        let rA = try await idx.search(query: hint, k: 5, filter: nil)
        let rB = try await raw.search(query: q, k: 5, filter: nil)

        XCTAssertEqual(rA.map(\.id), rB.map(\.id))
        for (a, b) in zip(rA, rB) {
            XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
        }
    }

    func testIVFCosineNormalizationHint() async throws {
        let dim = 16
        let idx = IVFIndex(dimension: dim, metric: .cosine, config: .init(nlist: 4, nprobe: 4))
        let raw = IVFIndex(dimension: dim, metric: .cosine, config: .init(nlist: 4, nprobe: 4))

        for i in 0..<30 {
            let v = makeNormalizedVector(dim: dim, seed: UInt64(i) &+ 400)
            try await idx.insert(id: "v\(i)", vector: v, metadata: nil)
            try await raw.insert(id: "v\(i)", vector: v, metadata: nil)
        }

        let q = makeNormalizedVector(dim: dim, seed: 666)
        let hint = NormalizationHint(vector: DynamicVector(q), isNormalized: true)

        let rA = try await idx.search(query: hint, k: 5, filter: nil)
        let rB = try await raw.search(query: q, k: 5, filter: nil)

        XCTAssertEqual(rA.map(\.id), rB.map(\.id))
        for (a, b) in zip(rA, rB) {
            XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
        }
    }
}
