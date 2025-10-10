//
//  RNG.swift
//  VectorIndex
//
//  Deterministic pseudorandom number generator for reproducible k-means seeding
//  and training. Uses a simple linear congruential generator (LCG) suitable for
//  non-cryptographic purposes.
//
//  Mathematical Properties:
//    - Period: 2^64 (full cycle on UInt64)
//    - Multiplier: 2862933555777941757 (prime, passes spectral test)
//    - Increment: 3037000493 (odd, ensures full period)
//
//  Typical usage:
//    var rng = RNGState(seed: 42, stream: 0)
//    let index = Int(rng.next() % UInt64(n))
//    let prob = rng.nextFloat()
//

import Foundation

/// Deterministic pseudorandom number generator state.
///
/// Thread-safe when not shared (value semantics).
/// For parallel RNG: use different stream IDs per thread.
///
/// Algorithm: 64-bit Linear Congruential Generator (LCG)
///   s_next = a * s + c  (mod 2^64)
///
/// Quality: Suitable for sampling, shuffling, and initialization.
/// NOT suitable for cryptography or high-quality Monte Carlo.
@frozen
public struct RNGState: Sendable {
    /// Internal state (64-bit seed)
    @usableFromInline
    internal var s: UInt64

    /// Initialize RNG with seed and optional stream ID.
    ///
    /// - Parameters:
    ///   - seed: Primary seed value (0 will be treated as 1)
    ///   - stream: Stream identifier for parallel RNG (default: 0)
    ///
    /// Stream mixing: seed ^ (stream << 32) ensures different sequences
    /// for different streams even with the same base seed.
    @inlinable
    public init(seed: UInt64, stream: UInt64 = 0) {
        // Ensure non-zero seed (LCG with s=0 produces trivial sequence)
        let baseSeed = (seed == 0) ? 1 : seed
        // Mix stream ID into high bits to generate independent sequences
        self.s = baseSeed ^ (stream << 32)
    }

    /// Generate next 64-bit random integer.
    ///
    /// Updates internal state via LCG recurrence:
    ///   s = a * s + c  (wrapping modulo 2^64)
    ///
    /// - Returns: Uniformly distributed UInt64 in [0, 2^64)
    @inlinable
    public mutating func next() -> UInt64 {
        // Wrapping arithmetic is correct here (modular LCG)
        s = 2862933555777941757 &* s &+ 3037000493
        return s
    }

    /// Generate random float in [0, 1).
    ///
    /// Uses high 53 bits for mantissa precision (IEEE 754 double has 53-bit significand).
    /// Converting to Float32 loses precision but is sufficient for sampling probabilities.
    ///
    /// - Returns: Float uniformly distributed in [0, 1)
    @inlinable
    public mutating func nextFloat() -> Float {
        // Use top 53 bits (better quality than low bits in LCG)
        let u = next() >> 11
        // Normalize to [0, 1): divide by 2^53
        return Float(u) / Float(1 << 53)
    }

    /// Generate random double in [0, 1) with full precision.
    ///
    /// Uses 53 high bits for IEEE 754 double precision.
    ///
    /// - Returns: Double uniformly distributed in [0, 1)
    @inlinable
    public mutating func nextDouble() -> Double {
        let u = next() >> 11
        return Double(u) / Double(1 << 53)
    }

    /// Generate random integer in range [0, bound).
    ///
    /// Uses modulo reduction (slight bias for non-power-of-2 bounds, acceptable for k-means).
    ///
    /// - Parameter bound: Upper bound (exclusive)
    /// - Returns: Random integer in [0, bound)
    @inlinable
    public mutating func nextInt(bound: Int) -> Int {
        precondition(bound > 0, "Bound must be positive")
        return Int(next() % UInt64(bound))
    }
}

// MARK: - Compatibility with existing inline RNG

/// Legacy inline RNG structure (for IVFIndex.swift compatibility).
/// Identical to RNGState, provided for gradual migration.
@usableFromInline
internal struct RNG {
    @usableFromInline
    internal var s: UInt64

    @inlinable
    internal init(s: UInt64) {
        self.s = (s == 0) ? 1 : s
    }

    @inlinable
    internal mutating func next() -> UInt64 {
        s = 2862933555777941757 &* s &+ 3037000493
        return s
    }

    @inlinable
    internal mutating func uniform() -> Float {
        Float(next() >> 11) / Float(1 << 53)
    }
}
