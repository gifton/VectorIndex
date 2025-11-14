//
//  LayoutTransforms.swift
//  VectorIndex — Kernel #48: Memory Layout Transformations
//
//  Implements AoS <-> AoSoA (R ∈ {4,8}) for Float32 vectors,
//  PQ code interleaving for UInt8 and packed 4-bit codes,
//  plus utilities, configuration, and optional telemetry.
//
//  Targets Apple Silicon / ARM64 but uses portable Swift where possible.
//

import Foundation

// MARK: - Layout Parameters & Errors

/// Dimension chunk size (NEON-friendly; 64B / 4B per float)
public let V_FLOAT32: Int = 16

/// Row block size for AoSoA transformation
public enum RowBlockSize: Int, Sendable {
    case r4 = 4
    case r8 = 8

    /// Recommended block size based on dimension
    public static func recommended(dimension d: Int) -> RowBlockSize {
        // Larger dimensions benefit from smaller blocks (better L1 fit)
        return d >= 1024 ? .r4 : .r8
    }
}

/// Group size for PQ code interleaving
public enum PQGroupSize: Int, Sendable {
    case g4 = 4   // better cache residency per group
    case g8 = 8   // better vectorization per group

    public static func recommended(subquantizers m: Int) -> PQGroupSize {
        return m >= 16 ? .g8 : .g4
    }
}

// LayoutError removed - migrated to VectorIndexError
// All throw sites now use ErrorBuilder with appropriate IndexErrorKind

// MARK: - Configuration

/// Options for layout transformation
public struct LayoutTransformOpts: Sendable {
    /// Row block size for vector interleaving
    public let rowBlockSize: RowBlockSize
    /// Group size for PQ code interleaving
    public let pqGroupSize: PQGroupSize
    /// Enable parallel transformation for large datasets
    public let enableParallel: Bool
    /// Parallel threshold (number of vectors)
    public let parallelThreshold: Int
    /// Enable telemetry recording
    public let enableTelemetry: Bool

    public init(
        rowBlockSize: RowBlockSize,
        pqGroupSize: PQGroupSize,
        enableParallel: Bool,
        parallelThreshold: Int,
        enableTelemetry: Bool
    ) {
        self.rowBlockSize = rowBlockSize
        self.pqGroupSize = pqGroupSize
        self.enableParallel = enableParallel
        self.parallelThreshold = parallelThreshold
        self.enableTelemetry = enableTelemetry
    }

    public static let `default` = LayoutTransformOpts(
        rowBlockSize: .r8,
        pqGroupSize: .g8,
        enableParallel: true,
        parallelThreshold: 10_000,
        enableTelemetry: false
    )
}

// MARK: - Padding Utilities

/// Calculate padded dimension for AoSoA layout (round up to multiple of V=16)
@inline(__always)
public func paddedDimension(_ d: Int) -> Int {
    return ((d + V_FLOAT32 - 1) / V_FLOAT32) * V_FLOAT32
}

/// Calculate buffer size for n vectors with dimension d in AoSoA layout
/// (Name matches spec; retained as-is.)
@inline(__always)
public func asoaBufferSize(n: Int, d: Int) -> Int {
    return n * paddedDimension(d)
}

/// Optional alias (ergonomic)
@inline(__always)
public func aosoaBufferSize(n: Int, d: Int) -> Int {
    return asoaBufferSize(n: n, d: d)
}

// MARK: - Validation

@inline(__always)
public func validateInterleaveParams(n: Int, d: Int, R: Int) throws {
    guard R == 4 || R == 8 else {
        throw ErrorBuilder(.invalidParameter, operation: "validate_interleave_params")
            .message("Invalid row block size")
            .info("R", "\(R)")
            .info("valid_values", "4, 8")
            .build()
    }
    guard n > 0 && d > 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "validate_interleave_params")
            .message("Invalid dimensions: both n and d must be positive")
            .info("n", "\(n)")
            .info("d", "\(d)")
            .build()
    }

    let dP = paddedDimension(d)
    let required = n &* dP
    // 4 bytes per float; just ensure element count itself doesn't overflow common Int arithmetic.
    guard required < (Int.max >> 2) else {
        throw ErrorBuilder(.capacityExceeded, operation: "validate_interleave_params")
            .message("Buffer size would overflow")
            .info("required_elements", "\(required)")
            .info("max_elements", "\(Int.max >> 2)")
            .build()
    }
}

@inline(__always)
private func isAligned<T>(_ p: UnsafePointer<T>, to alignment: Int) -> Bool {
    return (Int(bitPattern: p) & (alignment - 1)) == 0
}

@inline(__always)
private func isAligned<T>(_ p: UnsafeMutablePointer<T>, to alignment: Int) -> Bool {
    return (Int(bitPattern: p) & (alignment - 1)) == 0
}

// MARK: - Telemetry

public struct LayoutTransformTelemetry {
    public let transformType: String      // "vec_interleave", "vec_deinterleave", "pq_interleave_u8", ...
    public let vectorCount: Int
    public let dimension: Int             // for vectors
    public let subquantizers: Int         // for PQ codes (m)
    public let rowBlockSize: Int          // R (vectors)
    public let groupSize: Int             // g (PQ)
    public let bytesTransformed: Int
    public let executionTimeNanos: UInt64

    public var throughputGBps: Double {
        let secs = Double(executionTimeNanos) / 1e9
        return secs > 0 ? (Double(bytesTransformed) / 1e9) / secs : .infinity
    }

    public var throughputMBps: Double { throughputGBps * 1000 }
}

// Optional hook the host app can implement.
public enum GlobalTelemetryRecorder {
    public nonisolated(unsafe) static var record: ((LayoutTransformTelemetry) -> Void)? = nil
}

// MARK: - Core: Vector Interleaving (AoS -> AoSoA)

/// Transform vectors from AoS [n][d] to AoSoA (R-row interleaved, d padded to 16)
/// Complexity: O(n * d); Memory-bound.
/// Thread-safe with disjoint outputs.
@inlinable
public func vecsInterleave_f32(
    aos: UnsafePointer<Float>,
    n: Int,
    d: Int,
    R: Int,  // 4 or 8
    aosoa: UnsafeMutablePointer<Float>
) {
    // Precondition checks (debug-friendly)
    assert(R == 4 || R == 8, "R must be 4 or 8")
    // Alignment helps but is not strictly required; keep as performance hint.
    // assert(isAligned(aos, to: 64) && isAligned(aosoa, to: 64), "Expected 64B alignment")

    let V = V_FLOAT32
    let dP = paddedDimension(d)
    let numBlocks = (n + R - 1) / R
    let numDimChunks = (dP + V - 1) / V

    // Optional telemetry
    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif

    for blockIdx in 0..<numBlocks {
        let rowStart = blockIdx * R
        let rowEnd = min(rowStart + R, n)
        let blockRows = rowEnd - rowStart

        for dimChunk in 0..<numDimChunks {
            let dimStart = dimChunk * V
            let dimEnd = min(dimStart + V, d)
            let chunkDims = max(0, dimEnd - dimStart)

            // Output offset for this block and dim-chunk, compact (no row padding beyond n)
            // Layout per block contributes `blockRows * V` elements per chunk.
            let outOffset = (rowStart &* dP) &+ (dimChunk &* blockRows &* V)

            // Transpose blockRows × V dims into output
            for row in 0..<blockRows {
                let gRow = rowStart + row
                let inRowOffset = gRow &* d

                // Stride across dims by blockRows (compact last block)
                var outBase = outOffset &+ row
                var inBase = inRowOffset &+ dimStart
                var dim = 0

                if chunkDims == V {
                    for _ in 0..<V {
                        aosoa[outBase] = aos[inBase]
                        outBase &+= blockRows
                        inBase &+= 1
                    }
                } else {
                    // Partial tail: copy real dims
                    while dim < chunkDims {
                        aosoa[outBase] = aos[inBase]
                        outBase &+= blockRows
                        inBase &+= 1
                        dim &+= 1
                    }
                    // Pad remaining dims with zeros for this row
                    while dim < V {
                        aosoa[outBase] = 0.0
                        outBase &+= blockRows
                        dim &+= 1
                    }
                }
            }
        }
    }

    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds
    let bytesRead  = n &* d  &* MemoryLayout<Float>.stride
    let bytesWrite = n &* dP &* MemoryLayout<Float>.stride
    let telemetry = LayoutTransformTelemetry(
        transformType: "vec_interleave",
        vectorCount: n,
        dimension: d,
        subquantizers: 0,
        rowBlockSize: R,
        groupSize: 0,
        bytesTransformed: bytesRead &+ bytesWrite,
        executionTimeNanos: t1 &- t0
    )
    GlobalTelemetryRecorder.record?(telemetry)
    #endif
}

// Optional SIMD-ish variant using Swift's SIMD types for readability.
// (The main function above already writes in the AoSoA-friendly order.)
@inlinable
public func vecsInterleave_f32_SIMD(
    aos: UnsafePointer<Float>,
    n: Int,
    d: Int,
    R: Int,
    aosoa: UnsafeMutablePointer<Float>
) {
    // Fallback to scalar if V misaligned or dimension tails present.
    // This wrapper still calls the scalar implementation; a deeper
    // NEON-optimized kernel can replace it if you decide later.
    vecsInterleave_f32(aos: aos, n: n, d: d, R: R, aosoa: aosoa)
}

// MARK: - Vector Deinterleaving (AoSoA -> AoS)

/// Inverse of `vecsInterleave_f32`. Ignores padded zeros.
@inlinable
public func vecsDeinterleave_f32(
    aosoa: UnsafePointer<Float>,
    n: Int,
    d: Int,
    R: Int,
    aos: UnsafeMutablePointer<Float>
) {
    assert(R == 4 || R == 8, "R must be 4 or 8")

    let V = V_FLOAT32
    let dP = paddedDimension(d)
    let numBlocks = (n + R - 1) / R
    let numDimChunks = (dP + V - 1) / V

    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif

    for blockIdx in 0..<numBlocks {
        let rowStart = blockIdx * R
        let rowEnd   = min(rowStart + R, n)
        let blockRows = rowEnd - rowStart

        for dimChunk in 0..<numDimChunks {
            let dimStart = dimChunk * V
            let dimEnd   = min(dimStart + V, d)
            let realDims = max(0, dimEnd - dimStart)

            // Mirror compact layout used in interleave
            let inOffset = (rowStart &* dP) &+ (dimChunk &* blockRows &* V)

            for row in 0..<blockRows {
                let gRow = rowStart + row
                let outRowOffset = gRow &* d

                var inBase  = inOffset &+ row
                var outBase = outRowOffset &+ dimStart

                // Copy only real dimensions (skip padded tail)
                for _ in 0..<realDims {
                    aos[outBase] = aosoa[inBase]
                    inBase &+= blockRows
                    outBase &+= 1
                }
            }
        }
    }

    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds
    let bytesRead  = n &* dP &* MemoryLayout<Float>.stride
    let bytesWrite = n &* d  &* MemoryLayout<Float>.stride
    let telemetry = LayoutTransformTelemetry(
        transformType: "vec_deinterleave",
        vectorCount: n,
        dimension: d,
        subquantizers: 0,
        rowBlockSize: R,
        groupSize: 0,
        bytesTransformed: bytesRead &+ bytesWrite,
        executionTimeNanos: t1 &- t0
    )
    GlobalTelemetryRecorder.record?(telemetry)
    #endif
}

// MARK: - In-place interleaving using a temporary output (safe & simple)
//
// Note: True in-place AoS->AoSoA without a full-sized temp buffer requires a
// permutation-by-cycles algorithm with scattered loads—complex and slower.
// This implementation guarantees correctness and writes the result back into
// `buf` while using a temporary output equal to the final AoSoA size.

@inlinable
public func vecsInterleaveInPlace_f32(
    buf: UnsafeMutablePointer<Float>,
    n: Int,
    d: Int,
    R: Int,
    scratch: UnsafeMutablePointer<Float> // Kept for API parity; not required here.
) {
    let dP = paddedDimension(d)
    let total = n &* dP
    let tmp = UnsafeMutablePointer<Float>.allocate(capacity: total)
    defer { tmp.deallocate() }

    vecsInterleave_f32(aos: UnsafePointer(buf), n: n, d: d, R: R, aosoa: tmp)

    // Copy back into caller-provided buffer
    // (Callers should ensure `buf` has capacity for n * dP floats.)
    _ = tmp.withMemoryRebound(to: UInt8.self, capacity: total &* MemoryLayout<Float>.stride) { srcB in
        buf.withMemoryRebound(to: UInt8.self, capacity: total &* MemoryLayout<Float>.stride) { dstB in
            memcpy(dstB, srcB, total &* MemoryLayout<Float>.stride)
        }
    }
}

// MARK: - PQ Code Interleaving (UInt8)

/// Transform PQ codes from AoS [n][m] to group-interleaved layout.
///
/// - Throws:
///   - `VectorIndexError(.invalidParameter)`: If m is not divisible by g
@inlinable
public func pqCodesInterleave_u8(
    aos: UnsafePointer<UInt8>,
    n: Int,
    m: Int,
    g: Int,
    out: UnsafeMutablePointer<UInt8>
) throws {
    guard m % g == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_codes_interleave_u8")
            .message("m must be divisible by g")
            .info("m", "\(m)")
            .info("g", "\(g)")
            .build()
    }

    let numGroups = m / g

    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif

    for group in 0..<numGroups {
        let subspaceStart = group &* g
        let outGroupOffset = group &* n &* g

        for vec in 0..<n {
            let inVecOffset  = vec &* m &+ subspaceStart
            let outVecOffset = outGroupOffset &+ vec &* g

            // Copy g bytes for this vector’s group
            // (small loops are often as fast as memcpy for tiny sizes)
            for sub in 0..<g {
                out[outVecOffset &+ sub] = aos[inVecOffset &+ sub]
            }
        }
    }

    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds
    let bytes = n &* m &* MemoryLayout<UInt8>.stride
    let telemetry = LayoutTransformTelemetry(
        transformType: "pq_interleave_u8",
        vectorCount: n,
        dimension: 0,
        subquantizers: m,
        rowBlockSize: 0,
        groupSize: g,
        bytesTransformed: bytes,
        executionTimeNanos: t1 &- t0
    )
    GlobalTelemetryRecorder.record?(telemetry)
    #endif
}

/// Inverse of `pqCodesInterleave_u8`
///
/// - Throws:
///   - `VectorIndexError(.invalidParameter)`: If m is not divisible by g
@inlinable
public func pqCodesDeinterleave_u8(
    interleaved: UnsafePointer<UInt8>,
    n: Int,
    m: Int,
    g: Int,
    aos: UnsafeMutablePointer<UInt8>
) throws {
    guard m % g == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_codes_deinterleave_u8")
            .message("m must be divisible by g")
            .info("m", "\(m)")
            .info("g", "\(g)")
            .build()
    }

    let numGroups = m / g

    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif

    for group in 0..<numGroups {
        let subspaceStart  = group &* g
        let inGroupOffset  = group &* n &* g

        for vec in 0..<n {
            let outVecOffset = vec &* m &+ subspaceStart
            let inVecOffset  = inGroupOffset &+ vec &* g

            for sub in 0..<g {
                aos[outVecOffset &+ sub] = interleaved[inVecOffset &+ sub]
            }
        }
    }

    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds
    let bytes = n &* m &* MemoryLayout<UInt8>.stride
    let telemetry = LayoutTransformTelemetry(
        transformType: "pq_deinterleave_u8",
        vectorCount: n,
        dimension: 0,
        subquantizers: m,
        rowBlockSize: 0,
        groupSize: g,
        bytesTransformed: bytes,
        executionTimeNanos: t1 &- t0
    )
    GlobalTelemetryRecorder.record?(telemetry)
    #endif
}

// MARK: - PQ Code Interleaving (4-bit packed)

/// Transform packed 4-bit PQ codes to group-interleaved layout.
/// Each byte contains two 4-bit codes (low nibble, high nibble).
///
/// - Throws:
///   - `VectorIndexError(.invalidParameter)`: If m is not divisible by g, or if m or g is not even
@inlinable
public func pqCodesInterleave_u4(
    aos_packed: UnsafePointer<UInt8>,
    n: Int,
    m: Int,   // number of subquantizers
    g: Int,   // group size (even)
    out_packed: UnsafeMutablePointer<UInt8>
) throws {
    guard m % g == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_codes_interleave_u4")
            .message("m must be divisible by g")
            .info("m", "\(m)")
            .info("g", "\(g)")
            .build()
    }
    guard m % 2 == 0 && g % 2 == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_codes_interleave_u4")
            .message("For 4-bit PQ, both m and g must be even")
            .info("m", "\(m)")
            .info("g", "\(g)")
            .info("m_even", "\(m % 2 == 0)")
            .info("g_even", "\(g % 2 == 0)")
            .build()
    }

    let numGroups = m / g
    let bytesPerVec = m / 2
    let bytesPerGroupPerVec = g / 2

    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif

    for group in 0..<numGroups {
        let subspaceStart = group &* g
        let outGroupOffset = group &* n &* bytesPerGroupPerVec

        for vec in 0..<n {
            let inVecByteBase  = vec &* bytesPerVec &+ (subspaceStart / 2)
            let outVecByteBase = outGroupOffset &+ vec &* bytesPerGroupPerVec

            // Copy contiguous bytes for this group (each byte packs 2 codes)
            for b in 0..<bytesPerGroupPerVec {
                out_packed[outVecByteBase &+ b] = aos_packed[inVecByteBase &+ b]
            }
        }
    }

    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds
    let bytes = n &* (m / 2) &* MemoryLayout<UInt8>.stride
    let telemetry = LayoutTransformTelemetry(
        transformType: "pq_interleave_u4",
        vectorCount: n,
        dimension: 0,
        subquantizers: m,
        rowBlockSize: 0,
        groupSize: g,
        bytesTransformed: bytes,
        executionTimeNanos: t1 &- t0
    )
    GlobalTelemetryRecorder.record?(telemetry)
    #endif
}

/// Inverse of `pqCodesInterleave_u4`
///
/// - Throws:
///   - `VectorIndexError(.invalidParameter)`: If m is not divisible by g, or if m or g is not even
@inlinable
public func pqCodesDeinterleave_u4(
    interleaved_packed: UnsafePointer<UInt8>,
    n: Int,
    m: Int,
    g: Int,
    aos_packed: UnsafeMutablePointer<UInt8>
) throws {
    guard m % g == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_codes_deinterleave_u4")
            .message("m must be divisible by g")
            .info("m", "\(m)")
            .info("g", "\(g)")
            .build()
    }
    guard m % 2 == 0 && g % 2 == 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_codes_deinterleave_u4")
            .message("For 4-bit PQ, both m and g must be even")
            .info("m", "\(m)")
            .info("g", "\(g)")
            .info("m_even", "\(m % 2 == 0)")
            .info("g_even", "\(g % 2 == 0)")
            .build()
    }

    let numGroups = m / g
    let bytesPerVec = m / 2
    let bytesPerGroupPerVec = g / 2

    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif

    for group in 0..<numGroups {
        let subspaceStart = group &* g
        let inGroupOffset = group &* n &* bytesPerGroupPerVec

        for vec in 0..<n {
            let outVecByteBase = vec &* bytesPerVec &+ (subspaceStart / 2)
            let inVecByteBase  = inGroupOffset &+ vec &* bytesPerGroupPerVec

            for b in 0..<bytesPerGroupPerVec {
                aos_packed[outVecByteBase &+ b] = interleaved_packed[inVecByteBase &+ b]
            }
        }
    }

    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds
    let bytes = n &* (m / 2) &* MemoryLayout<UInt8>.stride
    let telemetry = LayoutTransformTelemetry(
        transformType: "pq_deinterleave_u4",
        vectorCount: n,
        dimension: 0,
        subquantizers: m,
        rowBlockSize: 0,
        groupSize: g,
        bytesTransformed: bytes,
        executionTimeNanos: t1 &- t0
    )
    GlobalTelemetryRecorder.record?(telemetry)
    #endif
}

// MARK: - Convenience API (Swift Arrays)

public enum LayoutTransform { }

public extension LayoutTransform {

    // MARK: Flat-array overloads (AoS flat -> AoSoA flat)

    /// Interleave a flat AoS array [n*d] into AoSoA [n*padded(d)] using R.
    /// Returns a newly allocated interleaved buffer.
    static func interleave(
        aosFlat: [Float],
        n: Int,
        d: Int,
        rowBlockSize R: RowBlockSize = .r8
    ) -> [Float] {
        precondition(n > 0 && d > 0, "n and d must be > 0")
        precondition(aosFlat.count == n * d, "aosFlat.count must be n*d")
        let dP = paddedDimension(d)
        var out = [Float](repeating: 0, count: n * dP)
        aosFlat.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                vecsInterleave_f32(
                    aos: src.baseAddress!,
                    n: n,
                    d: d,
                    R: R.rawValue,
                    aosoa: dst.baseAddress!
                )
            }
        }
        return out
    }

    /// Interleave a flat AoS array [n*d] into AoSoA [n*padded(d)] using opts.
    /// Uses parallel path when enabled and above threshold.
    static func interleave(
        aosFlat: [Float],
        n: Int,
        d: Int,
        opts: LayoutTransformOpts
    ) -> [Float] {
        precondition(n > 0 && d > 0, "n and d must be > 0")
        precondition(aosFlat.count == n * d, "aosFlat.count must be n*d")
        let dP = paddedDimension(d)
        var out = [Float](repeating: 0, count: n * dP)
        aosFlat.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                if opts.enableParallel && n >= opts.parallelThreshold {
                    vecsInterleave_f32_parallel(
                        aos: src.baseAddress!,
                        n: n,
                        d: d,
                        R: opts.rowBlockSize.rawValue,
                        aosoa: dst.baseAddress!,
                        parallelThreshold: opts.parallelThreshold
                    )
                } else {
                    vecsInterleave_f32(
                        aos: src.baseAddress!,
                        n: n,
                        d: d,
                        R: opts.rowBlockSize.rawValue,
                        aosoa: dst.baseAddress!
                    )
                }
            }
        }
        return out
    }

    /// Deinterleave a flat AoSoA array [n*padded(d)] back to flat AoS [n*d].
    /// Named 'deinterleaveFlat' to avoid ambiguity with the 2D [[Float]] variant.
    static func deinterleaveFlat(
        interleaved: [Float],
        n: Int,
        d: Int,
        rowBlockSize R: RowBlockSize
    ) -> [Float] {
        precondition(n > 0 && d > 0, "n and d must be > 0")
        let dP = paddedDimension(d)
        precondition(interleaved.count == n * dP, "interleaved.count must be n * paddedDimension(d)")
        var aosFlat = [Float](repeating: 0, count: n * d)
        interleaved.withUnsafeBufferPointer { src in
            aosFlat.withUnsafeMutableBufferPointer { dst in
                vecsDeinterleave_f32(
                    aosoa: src.baseAddress!,
                    n: n,
                    d: d,
                    R: R.rawValue,
                    aos: dst.baseAddress!
                )
            }
        }
        return aosFlat
    }

    /// Deinterleave a flat AoSoA array [n*padded(d)] back to flat AoS [n*d] using opts.
    /// Named 'deinterleaveFlat' to avoid ambiguity with the 2D [[Float]] variant.
    static func deinterleaveFlat(
        interleaved: [Float],
        n: Int,
        d: Int,
        opts: LayoutTransformOpts
    ) -> [Float] {
        precondition(n > 0 && d > 0, "n and d must be > 0")
        let dP = paddedDimension(d)
        precondition(interleaved.count == n * dP, "interleaved.count must be n * paddedDimension(d)")
        var aosFlat = [Float](repeating: 0, count: n * d)
        interleaved.withUnsafeBufferPointer { src in
            aosFlat.withUnsafeMutableBufferPointer { dst in
                if opts.enableParallel && n >= opts.parallelThreshold {
                    vecsDeinterleave_f32_parallel(
                        aosoa: src.baseAddress!,
                        n: n,
                        d: d,
                        R: opts.rowBlockSize.rawValue,
                        aos: dst.baseAddress!,
                        parallelThreshold: opts.parallelThreshold
                    )
                } else {
                    vecsDeinterleave_f32(
                        aosoa: src.baseAddress!,
                        n: n,
                        d: d,
                        R: opts.rowBlockSize.rawValue,
                        aos: dst.baseAddress!
                    )
                }
            }
        }
        return aosFlat
    }

    /// High-level API: transform Swift [[Float]] (AoS) to interleaved [Float] (AoSoA)
    /// Note: This flattens input and pads per spec.
    static func interleave(
        vectors: [[Float]],
        rowBlockSize R: RowBlockSize = .r8
    ) -> [Float] {
        precondition(!vectors.isEmpty, "vectors must not be empty")
        let n = vectors.count
        let d = vectors[0].count
        for (i, v) in vectors.enumerated() {
            precondition(v.count == d, "All rows must have same dimension. Row \(i) has \(v.count), expected \(d).")
        }

        let dP = paddedDimension(d)
        var out = [Float](repeating: 0, count: n * dP)

        vectors.withUnsafeBufferPointer { outer in
            // Build a temporary AoS-flat buffer
            var aosFlat = [Float](repeating: 0, count: n * d)
            var offset = 0
            for row in outer {
                aosFlat.replaceSubrange(offset..<(offset + d), with: row)
                offset += d
            }
            aosFlat.withUnsafeBufferPointer { aosBuf in
                out.withUnsafeMutableBufferPointer { outBuf in
                    vecsInterleave_f32(
                        aos: aosBuf.baseAddress!,
                        n: n,
                        d: d,
                        R: R.rawValue,
                        aosoa: outBuf.baseAddress!
                    )
                }
            }
        }
        return out
    }

    /// High-level API: transform Swift [[Float]] using options.
    static func interleave(
        vectors: [[Float]],
        opts: LayoutTransformOpts
    ) -> [Float] {
        precondition(!vectors.isEmpty, "vectors must not be empty")
        let n = vectors.count
        let d = vectors[0].count
        for (i, v) in vectors.enumerated() {
            precondition(v.count == d, "All rows must have same dimension. Row \(i) has \(v.count), expected \(d).")
        }
        // Flatten then dispatch to flat-array + opts path
        var aosFlat = [Float](repeating: 0, count: n * d)
        var off = 0
        for row in vectors { aosFlat.replaceSubrange(off..<(off + d), with: row); off += d }
        return interleave(aosFlat: aosFlat, n: n, d: d, opts: opts)
    }

    /// High-level API: transform interleaved [Float] (AoSoA) back to [[Float]] (AoS)
    static func deinterleave(
        interleaved: [Float],
        n: Int,
        d: Int,
        rowBlockSize R: RowBlockSize
    ) -> [[Float]] {
        precondition(n > 0 && d > 0, "n and d must be > 0")
        let dP = paddedDimension(d)
        precondition(interleaved.count == n * dP, "interleaved.count must be n * paddedDimension(d)")

        var aosFlat = [Float](repeating: 0, count: n * d)
        interleaved.withUnsafeBufferPointer { inBuf in
            aosFlat.withUnsafeMutableBufferPointer { outBuf in
                vecsDeinterleave_f32(
                    aosoa: inBuf.baseAddress!,
                    n: n,
                    d: d,
                    R: R.rawValue,
                    aos: outBuf.baseAddress!
                )
            }
        }

        // Rebuild [[Float]]
        var result = Array(repeating: [Float](), count: n)
        var offset = 0
        for i in 0..<n {
            result[i] = Array(aosFlat[offset..<(offset + d)])
            offset += d
        }
        return result
    }

    /// High-level API: deinterleave using options.
    static func deinterleave(
        interleaved: [Float],
        n: Int,
        d: Int,
        opts: LayoutTransformOpts
    ) -> [[Float]] {
        return deinterleave(interleaved: interleaved, n: n, d: d, rowBlockSize: opts.rowBlockSize)
    }
}

// MARK: - (Optional) Parallel wrappers for large datasets

/// Parallel block dispatcher: splits over R-row blocks.
/// Concurrency safety: each block writes to a disjoint AoSoA tile in output,
/// so there is no cross-thread overlap in destination memory.
public func vecsInterleave_f32_parallel(
    aos: UnsafePointer<Float>,
    n: Int,
    d: Int,
    R: Int,
    aosoa: UnsafeMutablePointer<Float>,
    parallelThreshold: Int = 10_000
) {
    let V = V_FLOAT32
    let dP = paddedDimension(d)
    let numBlocks = (n + R - 1) / R
    let numDimChunks = (dP + V - 1) / V

    if n < parallelThreshold {
        vecsInterleave_f32(aos: aos, n: n, d: d, R: R, aosoa: aosoa)
        return
    }

    #if canImport(Darwin)
    DispatchQueue.concurrentPerform(iterations: numBlocks) { blockIdx in
        let rowStart = blockIdx * R
        let rowEnd = min(rowStart + R, n)
        let blockRows = rowEnd - rowStart

        for dimChunk in 0..<numDimChunks {
            let dimStart = dimChunk * V
            let dimEnd = min(dimStart + V, d)
            let chunkDims = max(0, dimEnd - dimStart)
            let outOffset = (rowStart &* dP) &+ (dimChunk &* blockRows &* V)

            for row in 0..<blockRows {
                let gRow = rowStart + row
                let inRowOffset = gRow &* d

                var outBase = outOffset &+ row
                var inBase = inRowOffset &+ dimStart
                var dim = 0

                if chunkDims == V {
                    for _ in 0..<V {
                        aosoa[outBase] = aos[inBase]
                        outBase &+= blockRows
                        inBase &+= 1
                    }
                } else {
                    while dim < chunkDims {
                        aosoa[outBase] = aos[inBase]
                        outBase &+= blockRows
                        inBase &+= 1
                        dim &+= 1
                    }
                    while dim < V {
                        aosoa[outBase] = 0.0
                        outBase &+= blockRows
                        dim &+= 1
                    }
                }
            }
        }
    }
    #else
    // Non-Darwin fallback: serial
    vecsInterleave_f32(aos: aos, n: n, d: d, R: R, aosoa: aosoa)
    #endif
}

// MARK: - Parallel Deinterleave (AoSoA -> AoS)

/// Parallel deinterleave dispatcher: splits over R-row blocks.
/// Concurrency safety: each block writes to a disjoint AoS row range
/// [blockIdx*R .. < min((blockIdx+1)*R, n)), so no cross-thread overlap.
public func vecsDeinterleave_f32_parallel(
    aosoa: UnsafePointer<Float>,
    n: Int,
    d: Int,
    R: Int,
    aos: UnsafeMutablePointer<Float>,
    parallelThreshold: Int = 10_000
) {
    let V = V_FLOAT32
    let dP = paddedDimension(d)
    let numBlocks = (n + R - 1) / R
    let numDimChunks = (dP + V - 1) / V

    if n < parallelThreshold {
        vecsDeinterleave_f32(aosoa: aosoa, n: n, d: d, R: R, aos: aos)
        return
    }

    #if canImport(Darwin)
    DispatchQueue.concurrentPerform(iterations: numBlocks) { blockIdx in
        let rowStart = blockIdx * R
        let rowEnd   = min(rowStart + R, n)
        let blockRows = rowEnd - rowStart

        for dimChunk in 0..<numDimChunks {
            let dimStart = dimChunk * V
            let dimEnd   = min(dimStart + V, d)
            let realDims = max(0, dimEnd - dimStart)

            let inOffset = (rowStart &* dP) &+ (dimChunk &* blockRows &* V)

            for row in 0..<blockRows {
                let gRow = rowStart + row
                let outRowOffset = gRow &* d

                var inBase  = inOffset &+ row
                var outBase = outRowOffset &+ dimStart

                // Copy only real dimensions (skip padded tail)
                for _ in 0..<realDims {
                    aos[outBase] = aosoa[inBase]
                    inBase &+= blockRows
                    outBase &+= 1
                }
            }
        }
    }
    #else
    // Non-Darwin fallback: serial
    vecsDeinterleave_f32(aosoa: aosoa, n: n, d: d, R: R, aos: aos)
    #endif
}

// MARK: - Lightweight self-checks (debug builds)

#if DEBUG
@inline(__always)
private func almostEqual(_ a: Float, _ b: Float, eps: Float = 1e-5) -> Bool {
    return abs(a - b) <= eps
}
#endif
