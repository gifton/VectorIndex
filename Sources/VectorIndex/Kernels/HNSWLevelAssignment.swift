//
//  HNSWLevelAssignment.swift
//  Kernel #35 — HNSW Level Assignment (Randomized)
//
//  Implements level sampling:
//    L = floor( -log(u) / log(M) ), u ~ U(0,1) clipped away from {0,1},
//    then clamp to [0, cap].
//
//  Notes:
//   - λ = 1 / log(M) per spec derivation.
//   - Deterministic RNG with seed+stream via xoroshiro128** (SplitMix64 seeding).
//   - C ABI shim `hnsw_sample_level` matches the spec.
//

import Foundation

// MARK: - Portable RNG (xoroshiro128** with SplitMix64 seeding)

/// SplitMix64 for stateless seed expansion (public domain).
@inline(__always)
private func splitmix64(_ x: inout UInt64) -> UInt64 {
    x &+= 0x9E3779B97F4A7C15
    var z = x
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
}

/// xoroshiro128** RNG (public domain).
public struct HNSWXoroRNGState {
    public var s0: UInt64
    public var s1: UInt64

    /// Construct from a (seed, stream) pair deterministically.
    /// Use different `stream` per worker to get split, lock-free streams.
    public static func from(seed: UInt64, stream: UInt64) -> HNSWXoroRNGState {
        var x = seed ^ (stream &* 0x9E3779B97F4A7C15)
        let s0 = splitmix64(&x)
        let s1 = splitmix64(&x)
        var st = HNSWXoroRNGState(s0: s0, s1: s1)
        // Warm-up a couple of steps to decorrelate trivial seeds.
        _ = st.nextU64(); _ = st.nextU64()
        return st
    }

    @inline(__always) private mutating func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 {
        (x << k) | (x >> (64 &- k))
    }

    /// Next 64 random bits.
    @inline(__always) public mutating func nextU64() -> UInt64 {
        let s0p = s0
        var s1p = s1
        let result = rotl(s0p &* 5, 7) &* 9

        s1p ^= s0p
        s0 = rotl(s0p, 24) ^ s1p ^ (s1p << 16) // a, b
        s1 = rotl(s1p, 37)                     // c

        return result
    }

    /// Next uniform in [0,1) as Float, with high-quality mantissa.
    @inline(__always) public mutating func nextFloat01() -> Float {
        // Use 53-bit to double, then cast to Float to match cross-lang determinism reasonably.
        let v = nextU64() >> 11 // keep 53 bits
        let d = Double(v) * (1.0 / 9007199254740992.0) // 2^53
        return Float(d)
    }
}

// MARK: - Sampling

/// Swift API per spec.
/// Returns level in [0, cap]. Behavior is deterministic given `rng`.
@inlinable
public func hnswSampleLevel(M: Int, cap: Int, rng: inout HNSWXoroRNGState) -> Int {
    // Validate inputs per spec; tolerate minor pathologies.
    guard cap >= 0 else { return 0 }
    let Mv = max(M, 2) // enforce M >= 2
    var logm = logf(Float(Mv))
    if logm <= 1e-9 { logm = 1e-9 } // guard λ = 1/log(M)

    // u in (0,1), clipped away from 0 and 1 to avoid log under/overflow.
    let uRaw = rng.nextFloat01()
    let u = min(1.0 - 1e-9, max(1e-9, uRaw))

    // L = floor( -log(u) / log(M) ), capped
    let Lf = floorf(-logf(u) / logm)
    let L = Int(Lf)
    return min(L, cap)
}

// MARK: - C ABI

/// C shim: expects rngState[0..1] to hold xoroshiro128** state.
/// Advances the state and returns the sampled level (>=0), or -1 on bad args.
@_cdecl("hnsw_sample_level")
public func c_hnsw_sample_level(_ M: Int32, _ cap: Int32, _ rngState: UnsafeMutablePointer<UInt64>?) -> Int32 {
    guard let r = rngState else { return -1 }
    // Load state (two u64 words).
    var st = HNSWXoroRNGState(s0: r[0], s1: r[1])
    let level = hnswSampleLevel(M: Int(M), cap: Int(cap), rng: &st)
    // Store back advanced state.
    r[0] = st.s0; r[1] = st.s1
    return Int32(level)
}
