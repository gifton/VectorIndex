//
//  S2_RNGDtype.swift
//  VectorIndex
//
//  Kernel #S2: RNG & Dtype Helpers
//  Fast deterministic RNG and robust dtype conversion/packing utilities.
//
//  Components:
//  - RNG: Xoroshiro128**, Philox4x32-10 (counter-based)
//  - Dtype: f32↔f16/bf16, int8 quantization, u4 pack/unpack
//  - Utilities: Endian helpers, alignment macros
//
//  Performance Targets:
//  - f32↔f16: ≥30 GB/s (NEON-optimized for aligned buffers)
//  - i8 quantize: ≥20 GB/s (vectorized clamp)
//  - u4 pack/unpack: ≥40 GB/s (L1-resident)
//

import Foundation
import Accelerate

// MARK: - RNG: Xoroshiro128**

/// Xoroshiro128** - Fast, high-quality 64-bit RNG for per-thread streams.
///
/// Algorithm: https://prng.di.unimi.it/xoroshiro128starstar.c
/// Period: 2^128 - 1
/// Quality: Passes BigCrush (TestU01 suite)
///
/// Thread Safety: Value semantics (struct), not Sendable (mutable state)
@frozen
public struct S2Xoroshiro128 {
    @usableFromInline
    internal var s0: UInt64
    @usableFromInline
    internal var s1: UInt64

    /// Initialize from seed and stream ID using SplitMix64.
    ///
    /// - Parameters:
    ///   - seed: Base seed value
    ///   - streamID: Stream identifier for parallel RNG
    ///   - taskID: Task identifier for further splitting
    ///
    /// - Note: Uses SplitMix64 to generate high-quality initial state.
    ///         Ensures s0 and s1 are not both zero.
    @inlinable
    public init(seed: UInt64, streamID: UInt64 = 0, taskID: UInt64 = 0) {
        // Combine seed components via mixing
        var state = seed &+ 0x9e3779b97f4a7c15
        state = state ^ (streamID << 32)
        state = state ^ (taskID << 16)

        // SplitMix64 to generate initial state
        func splitmix64(_ z: inout UInt64) -> UInt64 {
            z = z &+ 0x9e3779b97f4a7c15
            var result = z
            result = (result ^ (result >> 30)) &* 0xbf58476d1ce4e5b9
            result = (result ^ (result >> 27)) &* 0x94d049bb133111eb
            return result ^ (result >> 31)
        }

        self.s0 = splitmix64(&state)
        self.s1 = splitmix64(&state)

        // Ensure non-zero state (zero state has period 1)
        if s0 == 0 && s1 == 0 {
            s0 = 0x9e3779b97f4a7c15
            s1 = 0x3c6ef372fe94f82b
        }
    }

    /// Generate next 64-bit random value.
    ///
    /// - Returns: Uniformly distributed UInt64
    @inlinable
    public mutating func nextU64() -> UInt64 {
        let result = rotl(s0 &* 5, 7) &* 9

        let t = s1 ^ s0
        s0 = rotl(s0, 24) ^ t ^ (t << 16)
        s1 = rotl(t, 37)

        return result
    }

    /// Generate next 32-bit random value.
    ///
    /// - Returns: Uniformly distributed UInt32 (upper 32 bits of nextU64)
    @inlinable
    public mutating func nextU32() -> UInt32 {
        UInt32(truncatingIfNeeded: nextU64() >> 32)
    }

    /// Generate random float in [0, 1).
    ///
    /// - Returns: Float uniformly distributed in [0, 1)
    @inlinable
    public mutating func nextUniform() -> Float {
        // Use upper 53 bits for precision
        let u = nextU64() >> 11
        return Float(u) * 0x1.0p-53  // Multiply by 2^-53
    }

    /// Generate random double in [0, 1).
    ///
    /// - Returns: Double uniformly distributed in [0, 1)
    @inlinable
    public mutating func nextUniformF64() -> Double {
        let u = nextU64() >> 11
        return Double(u) * 0x1.0p-53
    }

    /// Rotate left helper.
    @inline(__always)
    @usableFromInline
    internal func rotl(_ x: UInt64, _ k: Int) -> UInt64 {
        (x << k) | (x >> (64 - k))
    }

    /// Create a new independent RNG by jumping ahead 2^64 steps.
    ///
    /// - Returns: New RNG with independent sequence
    public mutating func jump() -> S2Xoroshiro128 {
        // Jump coefficients for 2^64 advancement
        let jumpCoeffs: [UInt64] = [0xdf900294d8f554a5, 0x170865df4b3201fc]

        var s0New: UInt64 = 0
        var s1New: UInt64 = 0

        for coeff in jumpCoeffs {
            for b in 0..<64 {
                if (coeff & (1 << b)) != 0 {
                    s0New ^= s0
                    s1New ^= s1
                }
                _ = nextU64()
            }
        }

        return S2Xoroshiro128(s0: s0New, s1: s1New)
    }

    private init(s0: UInt64, s1: UInt64) {
        self.s0 = s0
        self.s1 = s1
    }
}

// MARK: - RNG: Philox4x32-10

/// Philox4x32-10 - Counter-based RNG for reproducible parallelism.
///
/// Algorithm: https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
/// Quality: Passes SmallCrush, Crush (TestU01 suite)
///
/// Thread Safety: Stateless function, fully Sendable
@frozen
public struct S2Philox4x32 {
    @usableFromInline
    internal let key0: UInt32
    @usableFromInline
    internal let key1: UInt32

    /// Initialize with seed and stream ID.
    ///
    /// - Parameters:
    ///   - seed: 64-bit seed value
    ///   - streamID: Stream identifier for parallel RNG
    @inlinable
    public init(seed: UInt64, streamID: UInt64 = 0) {
        let combined = seed ^ (streamID << 32)
        self.key0 = UInt32(truncatingIfNeeded: combined)
        self.key1 = UInt32(truncatingIfNeeded: combined >> 32)
    }

    /// Generate 4 random 32-bit values from counter.
    ///
    /// - Parameters:
    ///   - counterLo: Low 64 bits of counter
    ///   - counterHi: High 64 bits of counter
    ///
    /// - Returns: Array of 4 UInt32 values
    @inlinable
    public func generate(counterLo: UInt64, counterHi: UInt64) -> [UInt32] {
        var ctr0 = UInt32(truncatingIfNeeded: counterLo)
        var ctr1 = UInt32(truncatingIfNeeded: counterLo >> 32)
        var ctr2 = UInt32(truncatingIfNeeded: counterHi)
        var ctr3 = UInt32(truncatingIfNeeded: counterHi >> 32)

        var k0 = key0
        var k1 = key1

        // 10 rounds of Philox
        for _ in 0..<10 {
            (ctr0, ctr1, ctr2, ctr3) = singleRound(ctr0, ctr1, ctr2, ctr3, k0, k1)
            // Bump keys
            k0 = k0 &+ 0x9E3779B9  // Golden ratio constant
            k1 = k1 &+ 0xBB67AE85
        }

        return [ctr0, ctr1, ctr2, ctr3]
    }

    /// Single Philox round.
    @inline(__always)
    @usableFromInline
    internal func singleRound(
        _ c0: UInt32, _ c1: UInt32, _ c2: UInt32, _ c3: UInt32,
        _ k0: UInt32, _ k1: UInt32
    ) -> (UInt32, UInt32, UInt32, UInt32) {
        // Multiply-add step
        let prod0 = UInt64(c0) * 0xD2511F53
        let prod1 = UInt64(c2) * 0xCD9E8D57

        let hi0 = UInt32(truncatingIfNeeded: prod0 >> 32)
        let lo0 = UInt32(truncatingIfNeeded: prod0)
        let hi1 = UInt32(truncatingIfNeeded: prod1 >> 32)
        let lo1 = UInt32(truncatingIfNeeded: prod1)

        // Shuffle and XOR with keys
        return (
            hi1 ^ c1 ^ k0,
            lo1,
            hi0 ^ c3 ^ k1,
            lo0
        )
    }
}

// MARK: - RNG Utilities

/// Split seed into worker-specific seed and stream ID.
///
/// - Parameters:
///   - seed: Base seed
///   - workerID: Worker identifier
///   - taskID: Task identifier
///
/// - Returns: Tuple of (derived seed, stream ID)
@inlinable
public func rngSplit(seed: UInt64, workerID: Int, taskID: Int) -> (seed: UInt64, streamID: UInt64) {
    let derivedSeed = seed ^ (UInt64(workerID) << 32)
    let streamID = UInt64(taskID)
    return (derivedSeed, streamID)
}

/// Fisher-Yates shuffle in place.
///
/// - Parameters:
///   - array: Array to permute
///   - rng: RNG state (will be modified)
///
/// - Complexity: O(n)
@inlinable
public func randpermInPlace(_ array: inout [UInt32], rng: inout S2Xoroshiro128) {
    let n = array.count
    for i in stride(from: n - 1, to: 0, by: -1) {
        let j = Int(rng.nextU64() % UInt64(i + 1))
        array.swapAt(i, j)
    }
}

/// Sample without replacement using Algorithm S (Vitter's reservoir sampling variant).
///
/// - Parameters:
///   - n: Population size
///   - k: Sample size
///   - rng: RNG state (will be modified)
///
/// - Returns: Array of k sampled indices (0..<n)
///
/// - Complexity: O(n) expected
@inlinable
public func sampleWithoutReplacement(n: Int, k: Int, rng: inout S2Xoroshiro128) -> [UInt32] {
    precondition(k <= n, "Sample size must not exceed population size")

    var result = [UInt32]()
    result.reserveCapacity(k)

    var seen = 0
    for i in 0..<n {
        let remaining = n - i
        let needed = k - seen

        let threshold = Double(needed) / Double(remaining)
        if rng.nextUniformF64() < threshold {
            result.append(UInt32(i))
            seen += 1
            if seen == k {
                break
            }
        }
    }

    return result
}

/// Weighted random selection (for k-means++ seeding).
///
/// - Parameters:
///   - weights: Non-negative weights (need not sum to 1)
///   - rng: RNG state (will be modified)
///
/// - Returns: Selected index in [0, weights.count)
///
/// - Complexity: O(n) linear scan
@inlinable
public func weightedPick(weights: [Float], rng: inout S2Xoroshiro128) -> Int {
    precondition(!weights.isEmpty, "Weights array must not be empty")

    // Compute cumulative sum
    var cumsum: Float = 0
    for w in weights {
        cumsum += max(w, 0)  // Clamp negative weights to 0
    }

    precondition(cumsum > 0, "Total weight must be positive")

    // Sample uniform and find bin
    let u = rng.nextUniform() * cumsum
    var acc: Float = 0
    for (i, w) in weights.enumerated() {
        acc += max(w, 0)
        if u < acc {
            return i
        }
    }

    // Fallback (should not reach here due to floating-point precision)
    return weights.count - 1
}

/// Generate Gaussian random variables using Box-Muller transform.
///
/// - Parameters:
///   - count: Number of samples (if odd, last sample is discarded)
///   - rng: RNG state (will be modified)
///
/// - Returns: Array of Gaussian samples with mean=0, stddev=1
///
/// - Complexity: O(n)
@inlinable
public func gaussianBoxMuller(count: Int, rng: inout S2Xoroshiro128) -> [Float] {
    var result = [Float]()
    result.reserveCapacity(count)

    let pairs = count / 2
    for _ in 0..<pairs {
        let u1 = rng.nextUniform()
        let u2 = rng.nextUniform()

        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2

        result.append(r * cos(theta))
        result.append(r * sin(theta))
    }

    // If count is odd, generate one more but discard second value
    if count % 2 == 1 {
        let u1 = rng.nextUniform()
        let u2 = rng.nextUniform()
        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2
        result.append(r * cos(theta))
    }

    return result
}

// MARK: - Dtype: f32 ↔ f16

/// Convert f32 array to f16 (IEEE 754 binary16) with round-to-nearest-even.
///
/// - Parameters:
///   - src: Source buffer of Float32 values
///   - dst: Destination buffer for UInt16 bit patterns
///   - count: Number of elements to convert
///
/// - Note: Uses Swift's native Float16 for correct IEEE 754 semantics.
///         Preserves NaN payloads, sign of zero, and Inf values.
///
/// - Performance: ~5-8 GB/s (scalar path). SIMD optimization in Phase 2.
@inlinable
public func f32ToF16Batch(
    _ src: UnsafePointer<Float>,
    _ dst: UnsafeMutablePointer<UInt16>,
    _ count: Int
) {
    for i in 0..<count {
        dst[i] = Float16(src[i]).bitPattern
    }
}

/// Convert f16 array to f32.
///
/// - Parameters:
///   - src: Source buffer of UInt16 bit patterns
///   - dst: Destination buffer for Float32 values
///   - count: Number of elements to convert
@inlinable
public func f16ToF32Batch(
    _ src: UnsafePointer<UInt16>,
    _ dst: UnsafeMutablePointer<Float>,
    _ count: Int
) {
    for i in 0..<count {
        dst[i] = Float(Float16(bitPattern: src[i]))
    }
}

// MARK: - Dtype: f32 ↔ bf16

/// Convert f32 to bf16 (Brain Float16) with round-to-nearest-even.
///
/// **FIXED VERSION** - Corrected critical bug from external implementation.
///
/// BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
/// (Same exponent range as f32, truncated mantissa)
///
/// - Parameters:
///   - x: Input Float32 value
///
/// - Returns: UInt16 bit pattern of bf16 value
///
/// - Note: Preserves NaN payloads, sign of zero, handles overflow correctly.
@inline(__always)
@usableFromInline
internal func f32ToBF16BitsNearest(_ x: Float) -> UInt16 {
    let u = x.bitPattern

    // Handle Inf/NaN (exponent = 0xFF)
    if (u & 0x7F800000) == 0x7F800000 {
        var hi = UInt16(truncatingIfNeeded: u >> 16)
        // Preserve NaN payload (ensure at least one mantissa bit set for NaN)
        if (u & 0x007FFFFF) != 0 {
            hi |= 0x0001  // Force NaN (not Inf)
        }
        return hi
    }

    // Round to nearest, ties to even
    // bf16 keeps upper 16 bits of f32, so we look at bit 15 (round bit)
    // and bits [14:0] (sticky bits) to decide rounding
    let lsb = (u >> 16) & 1           // LSB of result
    let roundBit = (u >> 15) & 1      // Bit that determines rounding
    let stickyBits = u & 0x7FFF       // All bits below round bit

    var result = u >> 16

    // Round up if:
    // 1. Round bit is set AND (sticky bits != 0 OR lsb is set for ties-to-even)
    if roundBit != 0 && (stickyBits != 0 || lsb != 0) {
        result += 1
    }

    return UInt16(truncatingIfNeeded: result)
}

/// Convert f32 array to bf16 with round-to-nearest-even.
///
/// - Parameters:
///   - src: Source buffer of Float32 values
///   - dst: Destination buffer for UInt16 bit patterns
///   - count: Number of elements to convert
///
/// - Performance: ~5 GB/s (scalar path). SIMD optimization in Phase 2.
@inlinable
public func f32ToBF16Batch(
    _ src: UnsafePointer<Float>,
    _ dst: UnsafeMutablePointer<UInt16>,
    _ count: Int
) {
    for i in 0..<count {
        dst[i] = f32ToBF16BitsNearest(src[i])
    }
}

/// Convert bf16 to f32 (trivial: just zero-extend mantissa).
///
/// - Parameters:
///   - src: Source buffer of UInt16 bit patterns
///   - dst: Destination buffer for Float32 values
///   - count: Number of elements to convert
@inlinable
public func bf16ToF32Batch(
    _ src: UnsafePointer<UInt16>,
    _ dst: UnsafeMutablePointer<Float>,
    _ count: Int
) {
    for i in 0..<count {
        // BF16 to F32: just shift left 16 bits (zero-extend mantissa)
        let bits = UInt32(src[i]) << 16
        dst[i] = Float(bitPattern: bits)
    }
}

// MARK: - Dtype: int8 Quantization

/// Clamp to int8 range [-128, 127].
@inline(__always)
@usableFromInline
internal func clampI8(_ x: Int32) -> Int8 {
    if x < -128 { return -128 }
    if x > 127 { return 127 }
    return Int8(truncatingIfNeeded: x)
}

/// Symmetric int8 quantization: y = clamp(round(x / scale)).
///
/// - Parameters:
///   - x: Source buffer of Float32 values
///   - scale: Quantization scale (typically max(|x|) / 127)
///   - y: Destination buffer for Int8 values
///
/// - Note: Scale should be precomputed by caller.
///         Uses round-to-nearest-even for consistency.
///
/// - Performance: ~3 GB/s (scalar). SIMD optimization in Phase 2.
@inlinable
public func quantizeSymmetric(
    x: UnsafeBufferPointer<Float>,
    scale: Float,
    y: UnsafeMutableBufferPointer<Int8>
) {
    precondition(x.count == y.count, "Buffer sizes must match")

    let inv = 1.0 / max(scale, 1e-30)  // Avoid division by zero
    for i in 0..<x.count {
        let r = (x[i] * inv).rounded(.toNearestOrEven)
        y[i] = clampI8(Int32(r))
    }
}

/// Dequantize symmetric int8: x = y * scale.
///
/// - Parameters:
///   - y: Source buffer of Int8 values
///   - scale: Quantization scale
///   - x: Destination buffer for Float32 values
@inlinable
public func dequantizeSymmetric(
    y: UnsafeBufferPointer<Int8>,
    scale: Float,
    x: UnsafeMutableBufferPointer<Float>
) {
    precondition(x.count == y.count, "Buffer sizes must match")

    for i in 0..<y.count {
        x[i] = Float(y[i]) * scale
    }
}

/// Affine int8 quantization: y = clamp(round(x / scale) + zeroPoint).
///
/// - Parameters:
///   - x: Source buffer of Float32 values
///   - scale: Quantization scale
///   - zeroPoint: Zero-point offset [-128, 127]
///   - y: Destination buffer for Int8 values
@inlinable
public func quantizeAffine(
    x: UnsafeBufferPointer<Float>,
    scale: Float,
    zeroPoint: Int32,
    y: UnsafeMutableBufferPointer<Int8>
) {
    precondition(x.count == y.count, "Buffer sizes must match")

    let inv = 1.0 / max(scale, 1e-30)
    for i in 0..<x.count {
        let r = (x[i] * inv).rounded(.toNearestOrEven)
        let quantized = Int32(r) + zeroPoint
        y[i] = clampI8(quantized)
    }
}

/// Dequantize affine int8: x = (y - zeroPoint) * scale.
///
/// - Parameters:
///   - y: Source buffer of Int8 values
///   - scale: Quantization scale
///   - zeroPoint: Zero-point offset
///   - x: Destination buffer for Float32 values
@inlinable
public func dequantizeAffine(
    y: UnsafeBufferPointer<Int8>,
    scale: Float,
    zeroPoint: Int32,
    x: UnsafeMutableBufferPointer<Float>
) {
    precondition(x.count == y.count, "Buffer sizes must match")

    for i in 0..<y.count {
        x[i] = (Float(y[i]) - Float(zeroPoint)) * scale
    }
}

// MARK: - Dtype: PQ u4 Pack/Unpack

/// Pack two 4-bit values into one byte.
///
/// Layout: low nibble = first code, high nibble = second code
///
/// - Parameters:
///   - lo: First 4-bit value (low nibble)
///   - hi: Second 4-bit value (high nibble)
///
/// - Returns: Packed byte
@inline(__always)
public func packPair(_ lo: UInt8, _ hi: UInt8) -> UInt8 {
    (lo & 0x0F) | ((hi & 0x0F) << 4)
}

/// Unpack byte into two 4-bit values.
///
/// - Parameter packed: Packed byte
///
/// - Returns: Tuple of (low nibble, high nibble)
@inline(__always)
public func unpackPair(_ packed: UInt8) -> (lo: UInt8, hi: UInt8) {
    (packed & 0x0F, packed >> 4)
}

/// Pack array of 4-bit indices into packed u4 format.
///
/// - Parameters:
///   - indices: Array of 4-bit values (must have even count)
///   - packed: Destination buffer (size = indices.count / 2)
@inlinable
public func packNibblesU4(
    indices: UnsafeBufferPointer<UInt8>,
    packed: UnsafeMutableBufferPointer<UInt8>
) {
    precondition(indices.count % 2 == 0, "Indices count must be even")
    precondition(packed.count == indices.count / 2, "Packed buffer size incorrect")

    for i in 0..<packed.count {
        let lo = indices[2 * i]
        let hi = indices[2 * i + 1]
        packed[i] = packPair(lo, hi)
    }
}

/// Unpack u4 format into array of 4-bit indices.
///
/// - Parameters:
///   - packed: Source buffer of packed bytes
///   - indices: Destination buffer (size = packed.count * 2)
@inlinable
public func unpackNibblesU4(
    packed: UnsafeBufferPointer<UInt8>,
    indices: UnsafeMutableBufferPointer<UInt8>
) {
    precondition(indices.count == packed.count * 2, "Indices buffer size incorrect")

    for i in 0..<packed.count {
        let (lo, hi) = unpackPair(packed[i])
        indices[2 * i] = lo
        indices[2 * i + 1] = hi
    }
}

// MARK: - Endian Helpers

/// Load 16-bit little-endian value from memory.
@inline(__always)
public func load16LE(_ p: UnsafeRawPointer) -> UInt16 {
    p.load(as: UInt16.self).littleEndian
}

/// Load 32-bit little-endian value from memory.
@inline(__always)
public func load32LE(_ p: UnsafeRawPointer) -> UInt32 {
    p.load(as: UInt32.self).littleEndian
}

/// Load 64-bit little-endian value from memory.
@inline(__always)
public func load64LE(_ p: UnsafeRawPointer) -> UInt64 {
    p.load(as: UInt64.self).littleEndian
}

/// Store 16-bit value as little-endian.
@inline(__always)
public func store16LE(_ p: UnsafeMutableRawPointer, _ value: UInt16) {
    p.storeBytes(of: value.littleEndian, as: UInt16.self)
}

/// Store 32-bit value as little-endian.
@inline(__always)
public func store32LE(_ p: UnsafeMutableRawPointer, _ value: UInt32) {
    p.storeBytes(of: value.littleEndian, as: UInt32.self)
}

/// Store 64-bit value as little-endian.
@inline(__always)
public func store64LE(_ p: UnsafeMutableRawPointer, _ value: UInt64) {
    p.storeBytes(of: value.littleEndian, as: UInt64.self)
}

// MARK: - Alignment Helpers

/// Round up to next multiple of alignment.
///
/// - Parameters:
///   - x: Value to align
///   - alignment: Alignment (must be power of 2)
///
/// - Returns: Aligned value
@inline(__always)
public func alignUp(_ x: Int, to alignment: Int) -> Int {
    precondition(alignment > 0 && (alignment & (alignment - 1)) == 0, "Alignment must be power of 2")
    return (x + alignment - 1) & ~(alignment - 1)
}

/// Check if pointer is aligned.
///
/// - Parameters:
///   - ptr: Pointer to check
///   - alignment: Required alignment (must be power of 2)
///
/// - Returns: True if aligned
@inline(__always)
public func isAligned(_ ptr: UnsafeRawPointer, to alignment: Int) -> Bool {
    precondition(alignment > 0 && (alignment & (alignment - 1)) == 0, "Alignment must be power of 2")
    return (Int(bitPattern: ptr) & (alignment - 1)) == 0
}

/// Pad vector length to multiple.
///
/// - Parameters:
///   - length: Vector length
///   - multiple: Multiple to pad to
///
/// - Returns: Padded length
@inline(__always)
public func padTo(_ length: Int, multiple: Int) -> Int {
    ((length + multiple - 1) / multiple) * multiple
}
