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
            XCTAssertLessThan(abs(a.distance - b.distance), 1e-4)
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
        for (a, b) in zip(r1, r2) { XCTAssertLessThan(abs(a.distance - b.distance), 1e-4) }
    }

    func testL2SqrDotTrickNormSquaredParity_NonMultipleOf16() {
        // d=257 forces both (a) auto dot-trick selection (d >= 256, per
        // L2SqrKernel.swift's useDotTrick heuristic) and (b) a non-multiple-
        // of-16 remainder in the on-the-fly ‖·‖² computation inside
        // _normSquared, which now delegates to Norms.l2NormSquared. Guards
        // against accumulation-order drift introduced by that delegation
        // (Task 6 / B8) — this exact path had no prior test coverage.
        let d = 257, n = 5
        var q = [Float](repeating: 0, count: d)
        var xb = [Float](repeating: 0, count: n * d)
        for i in 0..<d { q[i] = Float.random(in: -1...1) }
        for i in 0..<(n * d) { xb[i] = Float.random(in: -1...1) }

        var dotTrickOut = [Float](repeating: 0, count: n)
        var scalarOut = [Float](repeating: 0, count: n)

        q.withUnsafeBufferPointer { qb in
            xb.withUnsafeBufferPointer { xbb in
                var opts = L2SqrOpts(algo: .dotTrick, useDotTrick: true, prefetchDistance: 8, strictFP: false, numThreads: 1)
                dotTrickOut.withUnsafeMutableBufferPointer { out in
                    withUnsafePointer(to: &opts) { optsPtr in
                        l2sqr_f32_block(qb.baseAddress!, xbb.baseAddress!, n, d, out.baseAddress!, nil, .nan, optsPtr)
                    }
                }
                scalarOut.withUnsafeMutableBufferPointer { out in
                    IndexOps.Scoring.L2Sqr.runScalarRef(q: qb.baseAddress!, xb: xbb.baseAddress!, n: n, d: d, out: out.baseAddress!)
                }
            }
        }

        for i in 0..<n {
            XCTAssertEqual(dotTrickOut[i], scalarOut[i], accuracy: 1e-2,
                           "row \(i): dot-trick vs scalar mismatch at d=\(d) (exercises _normSquared's non-16-aligned remainder path)")
        }
    }
}
