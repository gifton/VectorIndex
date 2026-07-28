import XCTest
@testable import VectorIndex

final class PQEncodeParity_SwiftOnly_Tests: XCTestCase {
    private func makeData(n: Int, d: Int, m: Int, ks: Int) -> (x: [Float], codebooks: [Float]) {
        precondition(d % m == 0)
        let dsub = d / m
        var x = [Float](repeating: 0, count: n*d)
        var cb = [Float](repeating: 0, count: m*ks*dsub)
        for i in 0..<(n*d) {
            x[i] = Float(sin(Double(i * 131 % 1024)) * 0.25 + cos(Double(i * 17 % 997)) * 0.125)
        }
        for j in 0..<(m*ks*dsub) {
            cb[j] = Float(sin(Double(j * 313 % 2048)) * 0.2 + cos(Double(j * 23 % 1237)) * 0.15)
        }
        return (x, cb)
    }

    private func computeCentroidSq(codebooks: UnsafePointer<Float>, m: Int, ks: Int, dsub: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m * ks)
        for j in 0..<m {
            let base = codebooks + j*ks*dsub
            for k in 0..<ks {
                var s: Float = 0
                let cptr = base + k*dsub
                for t in 0..<dsub { let v = cptr[t]; s += v*v }
                out[j*ks + k] = s
            }
        }
        return out
    }

    private func forceSwiftPath(_ body: () -> Void) {
        let key = "VECTORINDEX_DISABLE_C_PQ"
        let old = getenv(key)
        _ = "1".withCString { setenv(key, $0, 1) }
        body()
        if let o = old { setenv(key, o, 1) } else { unsetenv(key) }
    }

    func testU8_SoA_WithCSQ_vs_Default() {
        let n = 12, d = 24, m = 6, ks = 256, dsub = d/m
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        let csq = computeCentroidSq(codebooks: cb, m: m, ks: ks, dsub: dsub)

        var codes1 = [UInt8](repeating: 0, count: n*m)
        var codes2 = [UInt8](repeating: 0, count: n*m)
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: csq)

        forceSwiftPath {
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    csq.withUnsafeBufferPointer { csqb in
                        codes1.withUnsafeMutableBufferPointer { out1 in
                            pq_encode_u8_f32_withCSQ(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, csqb.baseAddress!, out1.baseAddress!, &opts)
                        }
                        codes2.withUnsafeMutableBufferPointer { out2 in
                            pq_encode_u8_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, out2.baseAddress!, &opts)
                        }
                    }
                }
            }
        }
        XCTAssertEqual(codes1, codes2)
    }

    func testResidualU8_SoA_WithCSQ_vs_Default() {
        let n = 8, d = 32, m = 8, ks = 256, kc = 4, dsub = d/m
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var coarse = [Float](repeating: 0, count: kc * d)
        for i in 0..<(kc*d) { coarse[i] = Float(cos(Double(i * 19 % 4096)) * 0.33) }
        var assignments = [Int32](repeating: 0, count: n)
        for i in 0..<n { assignments[i] = Int32(i % kc) }
        let csq = computeCentroidSq(codebooks: cb, m: m, ks: ks, dsub: dsub)

        var codes1 = [UInt8](repeating: 0, count: n*m)
        var codes2 = [UInt8](repeating: 0, count: n*m)
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: csq)

        forceSwiftPath {
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    csq.withUnsafeBufferPointer { csqb in
                        coarse.withUnsafeBufferPointer { gb in
                            assignments.withUnsafeBufferPointer { asg in
                                codes1.withUnsafeMutableBufferPointer { out1 in
                                    pq_encode_residual_u8_f32_withCSQ(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, csqb.baseAddress!, gb.baseAddress!, asg.baseAddress!, out1.baseAddress!, &opts)
                                }
                                codes2.withUnsafeMutableBufferPointer { out2 in
                                    pq_encode_residual_u8_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, gb.baseAddress!, asg.baseAddress!, out2.baseAddress!, &opts)
                                }
                            }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(codes1, codes2)
    }

    /// Guards Task 7 (spec A6): `ensureCentroidSqNorms` allocates a fresh
    /// [m*ks] buffer whenever `opts.centroidSqNorms == nil`. Before the fix,
    /// every call on this path leaked that buffer. This test drives the
    /// allocate-then-free path 201 times (1 baseline + 200 repeats) under the
    /// Swift fallback and asserts the output stays byte-identical each time —
    /// a misplaced `defer` (freeing before use, or not at all under repeated
    /// pressure) would surface here as a crash, double-free, or code drift.
    func testRepeatedEncodeWithoutPrecomputedNormsIsStable() throws {
        let n = 12, d = 24, m = 6, ks = 256
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .aOS, centroidSqNorms: nil)

        forceSwiftPath {
            var first = [UInt8](repeating: 0, count: n*m)
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    first.withUnsafeMutableBufferPointer { out in
                        pq_encode_u8_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, out.baseAddress!, &opts)
                    }
                }
            }

            for _ in 0..<200 {
                var codes = [UInt8](repeating: 0, count: n*m)
                x.withUnsafeBufferPointer { xb in
                    cb.withUnsafeBufferPointer { cbb in
                        codes.withUnsafeMutableBufferPointer { out in
                            pq_encode_u8_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, out.baseAddress!, &opts)
                        }
                    }
                }
                XCTAssertEqual(codes, first)
            }
        }
    }

    /// Guards Task 9 (spec A6, site 3): `pq_encode_u4_f32`'s only
    /// `ensureCentroidSqNorms` call site (PQEncode.swift:213) is reached whenever the
    /// C fast path is skipped (`.sOA` layout here, since the C branch only ever
    /// triggers for `.aOS`) -- previously called from zero test files. Drives the
    /// allocate-then-free path 51 times (1 baseline + 50 repeats) and asserts
    /// byte-identical packed output each time, the same idiom
    /// `testRepeatedEncodeWithoutPrecomputedNormsIsStable` uses for the u8 path.
    func testU4RepeatedEncodeWithoutPrecomputedNormsIsStable() {
        let n = 12, d = 24, m = 6, ks = 16
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: nil)

        func encodeOnce() -> [UInt8] {
            var codes = [UInt8](repeating: 0, count: n * (m / 2))
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    codes.withUnsafeMutableBufferPointer { out in
                        pq_encode_u4_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, out.baseAddress!, &opts)
                    }
                }
            }
            return codes
        }

        let first = encodeOnce()
        for _ in 0..<50 {
            XCTAssertEqual(encodeOnce(), first)
        }
    }

    /// Guards Task 9 (spec A6, site 5): `pq_encode_residual_u8_f32`'s Swift-fallback
    /// `ensureCentroidSqNorms` call site (PQEncode.swift:290), reached when the
    /// C-fast-path condition (`_useCPQEncode && layout == .aOS`) is false. The only
    /// existing residual-u8 test (`testResidualU8_SoA_WithCSQ_vs_Default`) always
    /// passes a precomputed `csq`, so it never reaches this site -- forcing
    /// `centroidSqNorms: nil` here does. Drives 50 repeats, asserts stability.
    func testResidualU8RepeatedEncodeWithoutPrecomputedNormsIsStable() {
        let n = 8, d = 32, m = 8, ks = 256, kc = 4
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var coarse = [Float](repeating: 0, count: kc * d)
        for i in 0..<(kc * d) { coarse[i] = Float(cos(Double(i * 19 % 4096)) * 0.33) }
        var assignments = [Int32](repeating: 0, count: n)
        for i in 0..<n { assignments[i] = Int32(i % kc) }
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: nil)

        func encodeOnce() -> [UInt8] {
            var codes = [UInt8](repeating: 0, count: n * m)
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    coarse.withUnsafeBufferPointer { gb in
                        assignments.withUnsafeBufferPointer { asg in
                            codes.withUnsafeMutableBufferPointer { out in
                                pq_encode_residual_u8_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, gb.baseAddress!, asg.baseAddress!, out.baseAddress!, &opts)
                            }
                        }
                    }
                }
            }
            return codes
        }

        let first = encodeOnce()
        for _ in 0..<50 {
            XCTAssertEqual(encodeOnce(), first)
        }
    }

    /// Guards Task 9 (spec A6, site 6): `pq_encode_residual_u4_f32`'s only
    /// `ensureCentroidSqNorms` call site (PQEncode.swift:413) -- same shape as site 3,
    /// the C path here never branches on `useDotTrick`/csq internally either.
    /// Previously called from zero test files. Drives 50 repeats, asserts stability.
    func testResidualU4RepeatedEncodeWithoutPrecomputedNormsIsStable() {
        let n = 8, d = 32, m = 8, ks = 16, kc = 4
        let (x, cb) = makeData(n: n, d: d, m: m, ks: ks)
        var coarse = [Float](repeating: 0, count: kc * d)
        for i in 0..<(kc * d) { coarse[i] = Float(cos(Double(i * 19 % 4096)) * 0.33) }
        var assignments = [Int32](repeating: 0, count: n)
        for i in 0..<n { assignments[i] = Int32(i % kc) }
        var opts = PQEncodeOpts(useDotTrick: true, outputLayout: .sOA, centroidSqNorms: nil)

        func encodeOnce() -> [UInt8] {
            var codes = [UInt8](repeating: 0, count: n * (m / 2))
            x.withUnsafeBufferPointer { xb in
                cb.withUnsafeBufferPointer { cbb in
                    coarse.withUnsafeBufferPointer { gb in
                        assignments.withUnsafeBufferPointer { asg in
                            codes.withUnsafeMutableBufferPointer { out in
                                pq_encode_residual_u4_f32(xb.baseAddress!, Int64(n), Int32(d), Int32(m), Int32(ks), cbb.baseAddress!, gb.baseAddress!, asg.baseAddress!, out.baseAddress!, &opts)
                            }
                        }
                    }
                }
            }
            return codes
        }

        let first = encodeOnce()
        for _ in 0..<50 {
            XCTAssertEqual(encodeOnce(), first)
        }
    }
}
