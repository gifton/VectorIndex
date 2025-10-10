import Foundation
import simd
import VectorCore

public extension IndexOps.Scoring {
    enum ScoreBlock {
        // Optional norms handle for cosine fused path
        public struct CosineNormsHandle {
            public let dbInvNormsF32: UnsafePointer<Float>?
            public let dbInvNormsF16: UnsafePointer<Float16>?
            public let queryInvNorm: Float?
            public let epsilon: Float
            public init(dbInvNormsF32: UnsafePointer<Float>?, dbInvNormsF16: UnsafePointer<Float16>?, queryInvNorm: Float?, epsilon: Float = 1e-12) {
                self.dbInvNormsF32 = dbInvNormsF32
                self.dbInvNormsF16 = dbInvNormsF16
                self.queryInvNorm = queryInvNorm
                self.epsilon = epsilon
            }
        }
        /// Compute per-row scores for a single query against a block of base vectors.
        /// - Euclidean: writes L2^2 (no sqrt).
        /// - DotProduct: writes inner products.
        /// - Cosine: writes cosine similarity in [-1,1].
        public static func run(
            q: UnsafePointer<Float>,
            xb: UnsafePointer<Float>,
            n: Int,
            d: Int,
            metric: SupportedDistanceMetric,
            out: UnsafeMutablePointer<Float>,
            cosineNorms: CosineNormsHandle? = nil
        ) {
            switch metric {
            case .euclidean:
                IndexOps.Scoring.L2Sqr.run(q: q, xb: xb, n: n, d: d, out: out)
            case .dotProduct:
                IndexOps.Scoring.InnerProduct.run(q: q, xb: xb, n: n, d: d, out: out)
            case .cosine:
                if let h = cosineNorms {
                    if let invF32 = h.dbInvNormsF32 {
                        let qInv = h.queryInvNorm ?? IndexOps.Scoring.Cosine.computeQueryInvNorm(q: q, d: d, epsilon: h.epsilon)
                        IndexOps.Scoring.Cosine.run(q: q, xb: xb, n: n, d: d, out: out, dbInvNorms: invF32, queryInvNorm: qInv)
                    } else if let invF16 = h.dbInvNormsF16 {
                        let qInv = h.queryInvNorm ?? IndexOps.Scoring.Cosine.computeQueryInvNorm(q: q, d: d, epsilon: h.epsilon)
                        IndexOps.Scoring.Cosine.runF16(q: q, xb: xb, n: n, d: d, out: out, dbInvNormsF16: invF16, queryInvNorm: qInv)
                    } else {
                        IndexOps.Scoring.Cosine.run(q: q, xb: xb, n: n, d: d, out: out)
                    }
                } else {
                    // On-the-fly norms (two-pass) when no cache provided
                    IndexOps.Scoring.Cosine.run(q: q, xb: xb, n: n, d: d, out: out)
                }
            default:
                // Fallback: compute scalar distances via existing util into out
                // Note: this is slow and should be rarely hit in practice.
                var idx = 0
                while idx < n {
                    let row = xb.advanced(by: idx * d)
                    var tmp = [Float](repeating: 0, count: d)
                    for j in 0..<d { tmp[j] = row[j] }
                    out[idx] = distance(Array(UnsafeBufferPointer(start: q, count: d)), tmp, metric: metric)
                    idx += 1
                }
            }
        }

        @inline(__always)
        private static func sumSquares(ptr: UnsafePointer<Float>, d: Int) -> Float {
            let d4 = d & ~3
            var acc = SIMD4<Float>.zero
            var j = 0
            while j < d4 {
                let v = load4(ptr.advanced(by: j))
                acc += v * v
                j += 4
            }
            var s = acc[0] + acc[1] + acc[2] + acc[3]
            if d - d4 >= 1 { s += ptr[d4+0] * ptr[d4+0] }
            if d - d4 >= 2 { s += ptr[d4+1] * ptr[d4+1] }
            if d - d4 >= 3 { s += ptr[d4+2] * ptr[d4+2] }
            return s
        }

        @inline(__always)
        private static func load4(_ p: UnsafePointer<Float>) -> SIMD4<Float> {
            if Int(bitPattern: p) & (MemoryLayout<SIMD4<Float>>.alignment - 1) == 0 {
                return p.withMemoryRebound(to: SIMD4<Float>.self, capacity: 1) { $0.pointee }
            } else {
                return SIMD4<Float>(p[0], p[1], p[2], p[3])
            }
        }
    }
}
