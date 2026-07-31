//
//  CentroidBatchScore.swift
//  VectorIndex
//
//  P3b: one `cblas_sgemm` cross-term for all of `batchSearch`'s query x
//  centroid probes. VectorIndex's first direct BLAS use -- `import
//  Accelerate` is already present in 4 kernel files (ResidualKernel,
//  IVFSelect, PQTrain, S2_RNGDtype); this is the 5th site.
//
//  `MatrixDistance` was rejected for this call site (mandatory double copy
//  of one of the two operands); this calls `cblas_sgemm` directly against
//  the already-contiguous `centroidsFlat` cache (Task 6/P3a) and a
//  once-flattened query batch, avoiding both copies.
//

import Accelerate
import VectorCore

internal enum CentroidBatchScore {
    /// One sgemm cross-term for q queries x kc centroids, row-major.
    /// out[qi*kc + ci] is "smaller is better", ordering-equivalent per row to
    /// DistanceUtils.distance:
    ///   euclidean  -> ||c||^2 - 2<q,c>          (||q||^2 omitted: constant per row)
    ///   dotProduct -> -<q,c>
    ///   cosine     -> 1 - <q,c>*qInv*cInv
    /// Returns false (out untouched) for metrics without a GEMM form
    /// (manhattan/chebyshev) -- caller falls back to the per-query path.
    ///
    /// COSINE GUARD PARITY: the scalar per-query path (`centroidScores` in
    /// IVFIndex.swift, mirroring `DistanceUtils.distance`'s cosine case)
    /// short-circuits to the max distance (1) whenever
    /// `sqrt(qNormSq * centroidNormsSq[ci]) <= .ulpOfOne`, guarding against a
    /// degenerate (near-zero-norm) query or centroid producing a meaningless
    /// similarity via an epsilon-inflated inverse norm. This GEMM path
    /// replicates that *exact* guard per (query, centroid) pair -- using the
    /// same raw-norms product form, not a per-factor approximation from the
    /// two inverse norms -- so probe selection cannot diverge from the
    /// single-query path at degenerate centroids.
    static func run(
        queries: UnsafePointer<Float>, q: Int,
        centroids: UnsafePointer<Float>, kc: Int, d: Int,
        metric: SupportedDistanceMetric,
        centroidNormsSq: [Float], centroidInvNorms: [Float],
        queriesAreNormalized: Bool,
        out: inout [Float]
    ) -> Bool {
        switch metric {
        case .euclidean, .dotProduct, .cosine: break
        default: return false
        }
        precondition(out.count >= q * kc)
        guard q > 0, kc > 0, d > 0 else { return true }
        // out <- -2*Q*C^T  (euclidean) / -1*Q*C^T (dot, cosine pre-scale)
        let alpha: Float = (metric == .euclidean) ? -2 : -1
        out.withUnsafeMutableBufferPointer { ob in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(q), Int32(kc), Int32(d),
                        alpha, queries, Int32(d),
                        centroids, Int32(d),
                        0, ob.baseAddress!, Int32(kc))
            switch metric {
            case .euclidean:
                for qi in 0..<q {
                    let row = ob.baseAddress! + qi * kc
                    for ci in 0..<kc { row[ci] += centroidNormsSq[ci] }
                }
            case .dotProduct:
                break   // -<q,c> already
            case .cosine:
                for qi in 0..<q {
                    let qPtr = queries + qi * d
                    let qNormSq: Float = queriesAreNormalized ? 1.0 :
                        IndexOps.Support.Norms.l2NormSquared(vector: qPtr, dimension: d)
                    let qInv: Float = queriesAreNormalized ? 1.0 : 1.0 / (qNormSq.squareRoot() + 1e-12)
                    let row = ob.baseAddress! + qi * kc
                    // row currently holds -<q,c> => 1 - dot*qInv*cInv = 1 + row*qInv*cInv,
                    // EXCEPT where the near-zero-norm guard forces max distance (1) --
                    // see the COSINE GUARD PARITY note above.
                    for ci in 0..<kc {
                        let denom = (qNormSq * centroidNormsSq[ci]).squareRoot()
                        row[ci] = denom > .ulpOfOne ? (1 + row[ci] * qInv * centroidInvNorms[ci]) : 1
                    }
                }
            default: break
            }
        }
        return true
    }
}
