//
//  DistanceUtils.swift
//  VectorIndex
//
//  Helpers to compute distances using VectorCore SIMD providers.
//

import Foundation
import VectorCore

/// Distance with optional pre-computed query inverse norm (cosine fast path).
/// When `queryIsNormalized` is true, the query's magnitude computation is skipped.
@inlinable
func distance(_ a: [Float], _ b: [Float], metric: SupportedDistanceMetric, queryIsNormalized: Bool = false) -> Float {
    switch metric {
    case .euclidean:
        let d2 = a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                SwiftFloatSIMDProvider.distanceSquared(ab.baseAddress!, bb.baseAddress!, count: ab.count)
            }
        }
        return sqrt(d2)
    case .cosine:
        var dot: Float = 0
        var bmag2: Float = 0
        var amag2: Float = queryIsNormalized ? 1.0 : 0.0
        a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                dot = SwiftFloatSIMDProvider.dot(ab.baseAddress!, bb.baseAddress!, count: ab.count)
                if !queryIsNormalized {
                    amag2 = SwiftFloatSIMDProvider.sumOfSquares(ab.baseAddress!, count: ab.count)
                }
                bmag2 = SwiftFloatSIMDProvider.sumOfSquares(bb.baseAddress!, count: bb.count)
            }
        }
        let denom = sqrt(amag2 * bmag2)
        guard denom > .ulpOfOne else { return 1 }
        let sim = max(-1, min(1, dot / denom))
        return 1 - sim
    case .dotProduct:
        let d = a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                SwiftFloatSIMDProvider.dot(ab.baseAddress!, bb.baseAddress!, count: ab.count)
            }
        }
        return -d
    case .manhattan:
        // SIMD4-vectorized Manhattan: eliminates temporary diff allocation
        return a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                let count = ab.count
                let simdCount = count >> 2
                var acc = SIMD4<Float>.zero
                for i in 0..<simdCount {
                    let off = i &* 4
                    let a4 = SIMD4<Float>(ab[off], ab[off &+ 1], ab[off &+ 2], ab[off &+ 3])
                    let b4 = SIMD4<Float>(bb[off], bb[off &+ 1], bb[off &+ 2], bb[off &+ 3])
                    let diff = a4 - b4
                    acc += diff.replacing(with: -diff, where: diff .< .zero)
                }
                var result = acc[0] + acc[1] + acc[2] + acc[3]
                let tail = simdCount &* 4
                for i in tail..<count { result += abs(ab[i] - bb[i]) }
                return result
            }
        }
    case .chebyshev:
        // Max magnitude of (a-b)
        return a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                var diff = [Float](repeating: 0, count: ab.count)
                return diff.withUnsafeMutableBufferPointer { db in
                    SwiftFloatSIMDProvider.subtract(ab.baseAddress!, bb.baseAddress!, result: db.baseAddress!, count: ab.count)
                    return SwiftFloatSIMDProvider.maximumMagnitude(db.baseAddress!, count: db.count)
                }
            }
        }
    }
}
