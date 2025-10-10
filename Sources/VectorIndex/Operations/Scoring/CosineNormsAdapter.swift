import Foundation

public extension IndexOps.Scoring.ScoreBlock.CosineNormsHandle {
    static func from(cache: IndexOps.Support.Norms.NormCache) -> IndexOps.Scoring.ScoreBlock.CosineNormsHandle? {
        guard cache.mode.needsInv else { return nil }
        switch cache.invDType {
        case .float32:
            guard let ptr = cache.invNorms else { return nil }
            return .init(dbInvNormsF32: UnsafePointer(ptr), dbInvNormsF16: nil, queryInvNorm: nil, epsilon: cache.epsilon)
        case .float16:
            guard let ptr = cache.invPointer_f16() else { return nil }
            return .init(dbInvNormsF32: nil, dbInvNormsF16: UnsafePointer(ptr), queryInvNorm: nil, epsilon: cache.epsilon)
        case .bfloat16:
            // Not supported by fused path without widening; fall back to two-pass
            return nil
        }
    }
}
