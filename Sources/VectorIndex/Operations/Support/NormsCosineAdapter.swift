import Foundation

public extension IndexOps.Support.Norms.NormCache {
    /// Convenience wrapper to produce a ScoreBlock.CosineNormsHandle from this cache.
    /// Returns nil when no inverse norms are present or dtype is unsupported for fused cosine.
    func toCosineNormsHandle() -> IndexOps.Scoring.ScoreBlock.CosineNormsHandle? {
        guard mode.needsInv else { return nil }
        switch invDType {
        case .float32:
            guard let p = invNorms else { return nil }
            return .init(dbInvNormsF32: UnsafePointer(p), dbInvNormsF16: nil, queryInvNorm: nil, epsilon: epsilon)
        case .float16:
            guard let p16 = invPointer_f16() else { return nil }
            return .init(dbInvNormsF32: nil, dbInvNormsF16: UnsafePointer(p16), queryInvNorm: nil, epsilon: epsilon)
        case .bfloat16:
            // Not supported without widening; use two-pass cosine instead.
            return nil
        }
    }
}

