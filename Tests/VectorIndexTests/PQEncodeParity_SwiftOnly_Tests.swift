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
}
