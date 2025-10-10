//
//  DistanceUtils.swift
//  VectorIndex
//
//  Helpers to compute distances using VectorCore SIMD providers.
//

import Foundation
import VectorCore

@inlinable
func distance(_ a: [Float], _ b: [Float], metric: SupportedDistanceMetric) -> Float {
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
        var amag2: Float = 0
        var bmag2: Float = 0
        a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                dot = SwiftFloatSIMDProvider.dot(ab.baseAddress!, bb.baseAddress!, count: ab.count)
                amag2 = SwiftFloatSIMDProvider.sumOfSquares(ab.baseAddress!, count: ab.count)
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
        // Sum of magnitudes of (a-b)
        let sumAbs = a.withUnsafeBufferPointer { ab in
            b.withUnsafeBufferPointer { bb in
                var diff = [Float](repeating: 0, count: ab.count)
                return diff.withUnsafeMutableBufferPointer { db in
                    SwiftFloatSIMDProvider.subtract(ab.baseAddress!, bb.baseAddress!, result: db.baseAddress!, count: ab.count)
                    return SwiftFloatSIMDProvider.sumOfMagnitudes(db.baseAddress!, count: db.count)
                }
            }
        }
        return sumAbs
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

