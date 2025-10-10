import XCTest
@testable import VectorIndex

final class CosineKernelTests: XCTestCase {

    // Scalar reference cosine for small sizes
    private func cosineScalar(q: [Float], x: [Float]) -> Float {
        var dot: Float = 0
        var nq: Float = 0
        var nx: Float = 0
        for i in 0..<q.count { dot += q[i]*x[i]; nq += q[i]*q[i]; nx += x[i]*x[i] }
        if nq == 0 || nx == 0 { return 0 }
        let c = dot / (sqrt(nq) * sqrt(nx))
        return max(-1, min(1, c))
    }

    func testCosineTwoPassMatchesScalar_SmallDims() {
        let dims = [1, 2, 3, 7, 15, 16, 31, 32]
        for d in dims {
            let n = 13
            var q = (0..<d).map { _ in Float.random(in: -1...1) }
            var xb = (0..<(n*d)).map { _ in Float.random(in: -1...1) }
            var out = [Float](repeating: .nan, count: n)

            q.withUnsafeBufferPointer { qp in
                xb.withUnsafeMutableBufferPointer { xbp in
                    out.withUnsafeMutableBufferPointer { op in
                        IndexOps.Scoring.Cosine.run(
                            q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d,
                            out: op.baseAddress!, dbInvNorms: nil, queryInvNorm: nil
                        )
                    }
                }
            }

            for i in 0..<n {
                let xi = Array(xb[(i*d)..<((i+1)*d)])
                let ref = cosineScalar(q: q, x: xi)
                XCTAssertLessThan(abs(out[i] - ref), 1e-5, "d=\(d) i=\(i)")
            }
        }
    }

    func testCosineFusedMatchesScalar_FusedDims() {
        let dims = [512, 768, 1024, 1536]
        for d in dims {
            let n = 16
            var q = (0..<d).map { _ in Float.random(in: -1...1) }
            var xb = (0..<(n*d)).map { _ in Float.random(in: -1...1) }
            var out = [Float](repeating: .nan, count: n)
            var inv = [Float](repeating: 0, count: n)

            // Precompute inv norms
            xb.withUnsafeBufferPointer { xbp in
                inv.withUnsafeMutableBufferPointer { ibp in
                    IndexOps.Scoring.Cosine.precomputeInvNorms(xb: xbp.baseAddress!, n: n, d: d, out: ibp.baseAddress!, epsilon: 1e-12)
                }
            }
            let qInv = q.withUnsafeBufferPointer { qp in
                IndexOps.Scoring.Cosine.computeQueryInvNorm(q: qp.baseAddress!, d: d, epsilon: 1e-12)
            }

            q.withUnsafeBufferPointer { qp in
                xb.withUnsafeBufferPointer { xbp in
                    inv.withUnsafeBufferPointer { ibp in
                        out.withUnsafeMutableBufferPointer { op in
                            IndexOps.Scoring.Cosine.run(
                                q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d,
                                out: op.baseAddress!, dbInvNorms: ibp.baseAddress!, queryInvNorm: qInv
                            )
                        }
                    }
                }
            }

            for i in 0..<n {
                let xi = Array(xb[(i*d)..<((i+1)*d)])
                let ref = cosineScalar(q: q, x: xi)
                XCTAssertLessThan(abs(out[i] - ref), 1e-4, "d=\(d) i=\(i)")
            }
        }
    }

    func testCosineF16NormsParity() {
        let d = 64, n = 33
        var q = (0..<d).map { _ in Float.random(in: -1...1) }
        var xb = (0..<(n*d)).map { _ in Float.random(in: -1...1) }
        var invF32 = [Float](repeating: 0, count: n)
        var invF16 = [Float16](repeating: 0, count: n)
        var outF32 = [Float](repeating: .nan, count: n)
        var outF16 = [Float](repeating: .nan, count: n)

        xb.withUnsafeBufferPointer { xbp in
            invF32.withUnsafeMutableBufferPointer { ibp in
                IndexOps.Scoring.Cosine.precomputeInvNorms(xb: xbp.baseAddress!, n: n, d: d, out: ibp.baseAddress!, epsilon: 1e-12)
            }
        }
        xb.withUnsafeBufferPointer { xbp in
            invF16.withUnsafeMutableBufferPointer { ibp in
                IndexOps.Scoring.Cosine.precomputeInvNormsF16(xb: xbp.baseAddress!, n: n, d: d, out: ibp.baseAddress!, epsilon: 1e-12)
            }
        }
        let qInv = q.withUnsafeBufferPointer { qp in
            IndexOps.Scoring.Cosine.computeQueryInvNorm(q: qp.baseAddress!, d: d, epsilon: 1e-12)
        }

        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                invF32.withUnsafeBufferPointer { nbp in
                    outF32.withUnsafeMutableBufferPointer { op in
                        IndexOps.Scoring.Cosine.run(q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d, out: op.baseAddress!, dbInvNorms: nbp.baseAddress!, queryInvNorm: qInv)
                    }
                }
                invF16.withUnsafeBufferPointer { nbp16 in
                    outF16.withUnsafeMutableBufferPointer { op2 in
                        IndexOps.Scoring.Cosine.runF16(q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d, out: op2.baseAddress!, dbInvNormsF16: nbp16.baseAddress!, queryInvNorm: qInv)
                    }
                }
            }
        }
        for i in 0..<n { XCTAssertLessThan(abs(outF32[i] - outF16[i]), 1e-3) }
    }

    func testZeroNormHandling() {
        let d = 16, n = 4
        let q = (0..<d).map { _ in Float.random(in: -1...1) }
        var xb = [Float]()
        // First row zeros
        xb.append(contentsOf: Array(repeating: 0, count: d))
        // Others random
        xb.append(contentsOf: (0..<(3*d)).map { _ in Float.random(in: -1...1) })
        var out = [Float](repeating: .nan, count: n)
        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                out.withUnsafeMutableBufferPointer { op in
                    IndexOps.Scoring.Cosine.run(q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d, out: op.baseAddress!, dbInvNorms: nil, queryInvNorm: nil, options: .init())
                }
            }
        }
        XCTAssertEqual(out[0], 0, accuracy: 1e-6)
    }

    func testClampBehavior() {
        // Construct vectors to yield dot higher than product of norms due to numeric error; clamp should limit to 1
        let d = 8, n = 1
        let q: [Float] = Array(repeating: 1000.0, count: d)
        let xb: [Float] = Array(repeating: 1000.0, count: d)
        var out = [Float](repeating: .nan, count: n)
        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                out.withUnsafeMutableBufferPointer { op in
                    IndexOps.Scoring.Cosine.run(q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d, out: op.baseAddress!)
                }
            }
        }
        XCTAssertLessThanOrEqual(out[0], 1.0 + 1e-6)
    }
}
