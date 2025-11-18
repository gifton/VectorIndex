// ===----------------------------------------------------------------------===//
//  PQEncode.swift
//  VectorIndex
//
//  Kernel #20: Product Quantization (PQ) Encoding
//
//  Implements u8 (ks=256) and u4 (ks=16, packed) encoders,
//  residual IVF-PQ variants, and optional dot-product trick.
//
//  Spec references:
//  - API shapes & layouts: 20_pq_encode.md  (u8/u4, residual, AoS/SoA, packing)
//  - Deterministic tie-breaking: (distance, index)
//  - Dot trick: ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x,c>
//  - Inner product & L2^2 microkernels (hooks/mocks below)
// ===----------------------------------------------------------------------===//

import Foundation
#if canImport(CPQEncode)
import CPQEncode
#endif

// MARK: - Public Options & Layout

/// Code layout for output buffer.
/// - aOS: codes[i*m + j]  (vector major, subspace minor)
/// - sOA: codes[j*n + i]  (subspace major, vector minor) — better for ADC scan
public enum PQCodeLayout: UInt8 {
    case aOS = 0
    case sOA = 1
}

/// Optional encoder options. Pass `nil` for defaults.
@frozen
public struct PQEncodeOpts {
    /// Use dot-product trick when ks is large (default: true).
    /// Requires centroid squared-norms; if not provided, they
    /// are precomputed once per call.
    public var useDotTrick: Bool = true

    /// Output code layout (default: .aOS).
    public var outputLayout: PQCodeLayout = .aOS

    /// Optional pointer to precomputed centroid squared norms, layout [m][ks].
    /// If nil, they are computed once per call.
    public var centroidSqNorms: UnsafePointer<Float>?

    /// Reserved for future (prefetch distance, etc.)
    public var reserved0: Int32 = 0
    public var reserved1: Int32 = 0

    @inlinable public init(
        useDotTrick: Bool = true,
        outputLayout: PQCodeLayout = .aOS,
        centroidSqNorms: UnsafePointer<Float>? = nil
    ) {
        self.useDotTrick = useDotTrick
        self.outputLayout = outputLayout
        self.centroidSqNorms = centroidSqNorms
    }
}

// MARK: - Public C-style API (spec-compliant)

// 8-bit encoding (ks = 256). AoS or SoA codes depending on opts.
@inlinable
public func pq_encode_u8_f32(
    _ x: UnsafePointer<Float>,                    // [n × d]
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,
    _ ks32: Int32,                                // must be 256
    _ codebooks: UnsafePointer<Float>,            // [m × ks × dsub]
    _ codes: UnsafeMutablePointer<UInt8>,         // [n × m] or [m × n]
    _ optsPtr: UnsafePointer<PQEncodeOpts>?       // nullable
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 256, "ks must be 256 for u8 encoding")

    let dsub = d / m
    let opts = optsPtr?.pointee ?? PQEncodeOpts()
    let layout = opts.outputLayout

    #if canImport(CPQEncode)
    if _useCPQEncode && layout == .aOS {
        var cOpts = opts._toC(ks: ks, layout: layout)
        if opts.useDotTrick {
            let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
                maybeSq: opts.centroidSqNorms, codebooks: codebooks, m: m, ks: ks, dsub: dsub
            )
            CPQEncode.cpq_encode_u8_f32_with_csq(
                x, n64, CInt(d), CInt(m), CInt(ks), codebooks, centroidSq, codes, &cOpts
            )
        } else {
            CPQEncode.cpq_encode_u8_f32(
                x, n64, CInt(d), CInt(m), CInt(ks), codebooks, codes, &cOpts
            )
        }
        return
    }
    #endif

    // Prepare centroid squared-norms [m*ks] if needed (for dot trick).
    let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
        maybeSq: opts.centroidSqNorms, codebooks: codebooks, m: m, ks: ks, dsub: dsub
    )

    // Encode each vector.
    for i in 0..<n {
        let xRow = x + i*d
        for j in 0..<m {
            let q = argminCode_u8(
                xSub: xRow + j*dsub,
                codebook_j: codebooks + (j*ks*dsub),
                centroidSq_j: centroidSq + (j*ks),
                ks: ks,
                dsub: dsub,
                useDot: opts.useDotTrick
            )
            storeU8(code: UInt8(truncatingIfNeeded: q),
                    into: codes, layout: layout, i: i, j: j, n: n, m: m)
        }
    }
}

// 8-bit encoding (ks = 256) with precomputed centroid squared-norms (csq).
// Dot-trick path only. AoS routes to C fast path; SoA stays in Swift.
@inlinable
public func pq_encode_u8_f32_withCSQ(
    _ x: UnsafePointer<Float>,
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,
    _ ks32: Int32,                                 // must be 256
    _ codebooks: UnsafePointer<Float>,             // [m × ks × dsub]
    _ centroidSq: UnsafePointer<Float>,            // [m × ks]
    _ codes: UnsafeMutablePointer<UInt8>,          // [n × m] or [m × n]
    _ optsPtr: UnsafePointer<PQEncodeOpts>?        // nullable
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 256, "ks must be 256 for u8 encoding")

    let dsub = d / m
    let opts = optsPtr?.pointee ?? PQEncodeOpts()
    let layout = opts.outputLayout

    #if canImport(CPQEncode)
    if _useCPQEncode && layout == .aOS {
        var cOpts = opts._toC(ks: ks, layout: layout)
        CPQEncode.cpq_encode_u8_f32_with_csq(
            x, n64, CInt(d), CInt(m), CInt(ks), codebooks, centroidSq, codes, &cOpts
        )
        return
    }
    #endif

    // Swift path (SoA or when C backend is disabled)
    for i in 0..<n {
        let xRow = x + i*d
        for j in 0..<m {
            let q = argminCode_u8(
                xSub: xRow + j*dsub,
                codebook_j: codebooks + (j*ks*dsub),
                centroidSq_j: centroidSq + (j*ks),
                ks: ks,
                dsub: dsub,
                useDot: true
            )
            storeU8(code: UInt8(truncatingIfNeeded: q),
                    into: codes, layout: layout, i: i, j: j, n: n, m: m)
        }
    }
}

// 4-bit encoding (ks = 16). Packed two nibbles per byte.
@inlinable
public func pq_encode_u4_f32(
    _ x: UnsafePointer<Float>,                    // [n × d]
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,                                 // must be even (packed pairs)
    _ ks32: Int32,                                // must be 16
    _ codebooks: UnsafePointer<Float>,            // [m × ks × dsub]
    _ codesPacked: UnsafeMutablePointer<UInt8>,   // [n × (m/2)] or [(m/2) × n]
    _ optsPtr: UnsafePointer<PQEncodeOpts>?
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 16, "ks must be 16 for u4 encoding")
    precondition(m % 2 == 0, "m must be even for u4 packed encoding")

    let dsub = d / m
    let opts = optsPtr?.pointee ?? PQEncodeOpts()
    let layout = opts.outputLayout

    #if canImport(CPQEncode)
    if _useCPQEncode && layout == .aOS {
        var cOpts = opts._toC(ks: ks, layout: layout)
        CPQEncode.cpq_encode_u4_f32(
            x, n64, CInt(d), CInt(m), CInt(ks), codebooks, codesPacked, &cOpts
        )
        return
    }
    #endif

    // Prepare centroid squared-norms [m*ks] if needed.
    let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
        maybeSq: opts.centroidSqNorms, codebooks: codebooks, m: m, ks: ks, dsub: dsub
    )

    for i in 0..<n {
        let xRow = x + i*d
        for jPair in stride(from: 0, to: m, by: 2) {
            // Encode two neighboring subspaces (j, j+1), then pack.
            let j0 = jPair + 0
            let j1 = jPair + 1

            let q0 = argminCode_u8(
                xSub: xRow + j0*dsub,
                codebook_j: codebooks + (j0*ks*dsub),
                centroidSq_j: centroidSq + (j0*ks),
                ks: ks, dsub: dsub, useDot: opts.useDotTrick)

            let q1 = argminCode_u8(
                xSub: xRow + j1*dsub,
                codebook_j: codebooks + (j1*ks*dsub),
                centroidSq_j: centroidSq + (j1*ks),
                ks: ks, dsub: dsub, useDot: opts.useDotTrick)

            let packed = pq_pack_u4_pair(UInt8(truncatingIfNeeded: q0),
                                         UInt8(truncatingIfNeeded: q1))
            storeU4Packed(byte: packed, into: codesPacked, layout: layout,
                          i: i, jPair: jPair, n: n, m: m)
        }
    }
}

// Residual IVF-PQ (u8): codes of r = x − coarse[assign[i]]
@inlinable
public func pq_encode_residual_u8_f32(
    _ x: UnsafePointer<Float>,                    // [n × d]
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,
    _ ks32: Int32,                                // 256
    _ residualCodebooks: UnsafePointer<Float>,    // [m × ks × dsub]
    _ coarseCentroids: UnsafePointer<Float>,      // [kc × d]
    _ assignments: UnsafePointer<Int32>,          // [n]
    _ codes: UnsafeMutablePointer<UInt8>,         // [n × m] or [m × n]
    _ optsPtr: UnsafePointer<PQEncodeOpts>?
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 256, "ks must be 256 for residual u8 encoding")

    let dsub = d / m
    let opts = optsPtr?.pointee ?? PQEncodeOpts()
    let layout = opts.outputLayout

    #if canImport(CPQEncode)
    if _useCPQEncode && layout == .aOS {
        var cOpts = opts._toC(ks: ks, layout: layout)
        if opts.useDotTrick {
            let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
                maybeSq: opts.centroidSqNorms, codebooks: residualCodebooks, m: m, ks: ks, dsub: dsub
            )
            CPQEncode.cpq_encode_residual_u8_f32_with_csq(
                x, n64, CInt(d), CInt(m), CInt(ks), residualCodebooks, centroidSq, coarseCentroids, assignments, codes, &cOpts
            )
        } else {
            CPQEncode.cpq_encode_residual_u8_f32(
                x, n64, CInt(d), CInt(m), CInt(ks), residualCodebooks, coarseCentroids, assignments, codes, &cOpts
            )
        }
        return
    }
    #endif

    // Centroid squared-norms [m*ks] (for residual codebooks).
    let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
        maybeSq: opts.centroidSqNorms, codebooks: residualCodebooks, m: m, ks: ks, dsub: dsub
    )

    // Per-subspace scratch buffer for residual r_j (no global materialization).
    let rBuf = UnsafeMutablePointer<Float>.allocate(capacity: dsub)
    defer { rBuf.deallocate() }

    for i in 0..<n {
        let xRow = x + i*d
        let coarse = coarseCentroids + Int(assignments[i]) * d

        for j in 0..<m {
            // Build residual sub-vector r_j into scratch (once for this (i,j))
            let xSub = xRow + j*dsub
            let cSub = coarse + j*dsub
            for t in 0..<dsub { rBuf[t] = xSub[t] - cSub[t] }

            let q = argminCodeResidual_u8(
                rSub: rBuf,
                codebook_j: residualCodebooks + (j*ks*dsub),
                centroidSq_j: centroidSq + (j*ks),
                ks: ks, dsub: dsub,
                useDot: opts.useDotTrick
            )
            storeU8(code: UInt8(truncatingIfNeeded: q),
                    into: codes, layout: layout, i: i, j: j, n: n, m: m)
        }
    }
}

// Residual IVF-PQ (u8) with precomputed centroid squared-norms (csq).
// Dot-trick path only. AoS routes to C fast path; SoA stays in Swift.
@inlinable
public func pq_encode_residual_u8_f32_withCSQ(
    _ x: UnsafePointer<Float>,
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,
    _ ks32: Int32,                                // 256
    _ residualCodebooks: UnsafePointer<Float>,    // [m × ks × dsub]
    _ centroidSq: UnsafePointer<Float>,           // [m × ks]
    _ coarseCentroids: UnsafePointer<Float>,      // [kc × d]
    _ assignments: UnsafePointer<Int32>,          // [n]
    _ codes: UnsafeMutablePointer<UInt8>,         // [n × m] or [m × n]
    _ optsPtr: UnsafePointer<PQEncodeOpts>?
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 256, "ks must be 256 for residual u8 encoding")

    let dsub = d / m
    let opts = optsPtr?.pointee ?? PQEncodeOpts()
    let layout = opts.outputLayout

    #if canImport(CPQEncode)
    if _useCPQEncode && layout == .aOS {
        var cOpts = opts._toC(ks: ks, layout: layout)
        CPQEncode.cpq_encode_residual_u8_f32_with_csq(
            x, n64, CInt(d), CInt(m), CInt(ks), residualCodebooks, centroidSq, coarseCentroids, assignments, codes, &cOpts
        )
        return
    }
    #endif

    // Swift path (SoA or when C backend is disabled)
    let rBuf = UnsafeMutablePointer<Float>.allocate(capacity: dsub)
    defer { rBuf.deallocate() }
    for i in 0..<n {
        let xRow = x + i*d
        let coarse = coarseCentroids + Int(assignments[i]) * d
        for j in 0..<m {
            // r_j = x_j - g_j
            let xSub = xRow + j*dsub
            let gSub = coarse + j*dsub
            for t in 0..<dsub { rBuf[t] = xSub[t] - gSub[t] }
            let q = argminCodeResidual_u8(
                rSub: rBuf,
                codebook_j: residualCodebooks + (j*ks*dsub),
                centroidSq_j: centroidSq + (j*ks),
                ks: ks, dsub: dsub, useDot: true
            )
            storeU8(code: UInt8(truncatingIfNeeded: q),
                    into: codes, layout: layout, i: i, j: j, n: n, m: m)
        }
    }
}

// Residual IVF-PQ (u4): packed, two nibbles per byte.
@inlinable
public func pq_encode_residual_u4_f32(
    _ x: UnsafePointer<Float>,                    // [n × d]
    _ n64: Int64,
    _ d32: Int32,
    _ m32: Int32,                                 // even
    _ ks32: Int32,                                // 16
    _ residualCodebooks: UnsafePointer<Float>,    // [m × ks × dsub]
    _ coarseCentroids: UnsafePointer<Float>,      // [kc × d]
    _ assignments: UnsafePointer<Int32>,          // [n]
    _ codesPacked: UnsafeMutablePointer<UInt8>,   // [n × m/2] or [(m/2) × n]
    _ optsPtr: UnsafePointer<PQEncodeOpts>?
) {
    let n = Int(n64), d = Int(d32), m = Int(m32), ks = Int(ks32)
    precondition(d % m == 0, "d must be divisible by m")
    precondition(ks == 16, "ks must be 16 for residual u4 encoding")
    precondition(m % 2 == 0, "m must be even for u4 packed encoding")

    let dsub = d / m
    let opts = optsPtr?.pointee ?? PQEncodeOpts()
    let layout = opts.outputLayout

    #if canImport(CPQEncode)
    if _useCPQEncode && layout == .aOS {
        var cOpts = opts._toC(ks: ks, layout: layout)
        CPQEncode.cpq_encode_residual_u4_f32(
            x, n64, CInt(d), CInt(m), CInt(ks), residualCodebooks, coarseCentroids, assignments, codesPacked, &cOpts
        )
        return
    }
    #endif

    let centroidSq: UnsafePointer<Float> = ensureCentroidSqNorms(
        maybeSq: opts.centroidSqNorms, codebooks: residualCodebooks, m: m, ks: ks, dsub: dsub
    )

    let rBuf = UnsafeMutablePointer<Float>.allocate(capacity: dsub)
    defer { rBuf.deallocate() }

    for i in 0..<n {
        let xRow = x + i*d
        let coarse = coarseCentroids + Int(assignments[i]) * d

        for jPair in stride(from: 0, to: m, by: 2) {
            let j0 = jPair + 0
            let j1 = jPair + 1

            // r_0
            for t in 0..<dsub { rBuf[t] = (xRow + j0*dsub)[t] - (coarse + j0*dsub)[t] }
            let q0 = argminCodeResidual_u8(
                rSub: rBuf,
                codebook_j: residualCodebooks + (j0*ks*dsub),
                centroidSq_j: centroidSq + (j0*ks),
                ks: ks, dsub: dsub, useDot: opts.useDotTrick)

            // r_1
            for t in 0..<dsub { rBuf[t] = (xRow + j1*dsub)[t] - (coarse + j1*dsub)[t] }
            let q1 = argminCodeResidual_u8(
                rSub: rBuf,
                codebook_j: residualCodebooks + (j1*ks*dsub),
                centroidSq_j: centroidSq + (j1*ks),
                ks: ks, dsub: dsub, useDot: opts.useDotTrick)

            let packed = pq_pack_u4_pair(UInt8(truncatingIfNeeded: q0),
                                         UInt8(truncatingIfNeeded: q1))
            storeU4Packed(byte: packed, into: codesPacked, layout: layout,
                          i: i, jPair: jPair, n: n, m: m)
        }
    }
}

// MARK: - Packing helpers (u4)

/// Pack two 4-bit codes into one byte: low nibble = q1, high = q2.
@inlinable
public func pq_pack_u4_pair(_ qLow: UInt8, _ qHigh: UInt8) -> UInt8 {
    (qLow & 0x0F) | ((qHigh & 0x0F) << 4)
}

/// Unpack two 4-bit codes from byte.
@inlinable
public func pq_unpack_u4_pair(_ byte: UInt8) -> (low: UInt8, high: UInt8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

// MARK: - Core argmin (per-subspace)

@inline(__always)
@usableFromInline
internal func argminCode_u8(
    xSub: UnsafePointer<Float>,
    codebook_j: UnsafePointer<Float>,     // [ks × dsub]
    centroidSq_j: UnsafePointer<Float>,   // [ks]
    ks: Int,
    dsub: Int,
    useDot: Bool
) -> Int {
    if useDot {
        // Compute ‖x‖² once; then evaluate dist = ‖x‖² + ‖c‖² − 2⟨x,c⟩ (stable tie-break)
        let x2 = sqnorm(xSub, dsub)
        var bestK = 0
        var bestD = x2 + centroidSq_j[0] - 2.0 * dot(xSub, codebook_j + 0*dsub, dsub)
        for k in 1..<ks {
            let dist = x2 + centroidSq_j[k] - 2.0 * dot(xSub, codebook_j + k*dsub, dsub)
            if (dist < bestD) || (dist == bestD && k < bestK) {  // stable, deterministic
                bestD = dist; bestK = k
            }
        }
        return bestK
    } else {
        // Naive: argmin_k Σ (x − c_k)^2  with deterministic tie-break.
        var bestK = 0
        var bestD = l2sqr(xSub, codebook_j + 0*dsub, dsub)
        for k in 1..<ks {
            let dist = l2sqr(xSub, codebook_j + k*dsub, dsub)
            if (dist < bestD) || (dist == bestD && k < bestK) {
                bestD = dist; bestK = k
            }
        }
        return bestK
    }
}

@inline(__always)
@usableFromInline
internal func argminCodeResidual_u8(
    rSub: UnsafePointer<Float>,           // residual subvector (scratch)
    codebook_j: UnsafePointer<Float>,     // [ks × dsub]
    centroidSq_j: UnsafePointer<Float>,   // [ks]
    ks: Int,
    dsub: Int,
    useDot: Bool
) -> Int {
    if useDot {
        let r2 = sqnorm(rSub, dsub)
        var bestK = 0
        var bestD = r2 + centroidSq_j[0] - 2.0 * dot(rSub, codebook_j + 0*dsub, dsub)
        for k in 1..<ks {
            let dist = r2 + centroidSq_j[k] - 2.0 * dot(rSub, codebook_j + k*dsub, dsub)
            if (dist < bestD) || (dist == bestD && k < bestK) {
                bestD = dist; bestK = k
            }
        }
        return bestK
    } else {
        var bestK = 0
        var bestD = l2sqr(rSub, codebook_j + 0*dsub, dsub)
        for k in 1..<ks {
            let dist = l2sqr(rSub, codebook_j + k*dsub, dsub)
            if (dist < bestD) || (dist == bestD && k < bestK) {
                bestD = dist; bestK = k
            }
        }
        return bestK
    }
}

// MARK: - Centroid squared-norms (once per call unless provided)

@inline(__always)
@usableFromInline
internal func ensureCentroidSqNorms(
    maybeSq: UnsafePointer<Float>?,
    codebooks: UnsafePointer<Float>,
    m: Int, ks: Int, dsub: Int
) -> UnsafePointer<Float> {
    if let p = maybeSq { return p }
    // Allocate & compute [m * ks] once per call.
    let buf = UnsafeMutablePointer<Float>.allocate(capacity: m * ks)
    for j in 0..<m {
        let base = codebooks + j*ks*dsub
        let out  = buf + j*ks
        for k in 0..<ks {
            out[k] = sqnorm(base + k*dsub, dsub)
        }
    }
    // We intentionally leak this buffer by returning an UnsafePointer
    // to keep the API zero-copy and avoid per-subspace deallocs. In practice,
    // pass precomputed norms via opts.centroidSqNorms for long-running pipelines.
    return UnsafePointer(buf)
}

// MARK: - Layouted stores

@inline(__always)
@usableFromInline
internal func storeU8(
    code: UInt8,
    into dst: UnsafeMutablePointer<UInt8>,
    layout: PQCodeLayout,
    i: Int, j: Int, n: Int, m: Int
) {
    switch layout {
    case .aOS: dst[i*m + j] = code        // codes[i*m + j]
    case .sOA: dst[j*n + i] = code        // codes[j*n + i]
    }
}

@inline(__always)
@usableFromInline
internal func storeU4Packed(
    byte: UInt8,
    into dst: UnsafeMutablePointer<UInt8>,
    layout: PQCodeLayout,
    i: Int, jPair: Int, n: Int, m: Int
) {
    let col = jPair >> 1        // pair index
    switch layout {
    case .aOS: dst[i*(m>>1) + col] = byte            // codes[i*m/2 + j/2]
    case .sOA: dst[col*n + i]     = byte             // codes[(j/2)*n + i]
    }
}

// MARK: - Math helpers (SIMD-friendly; replace with VectorCore if available)

@inline(__always)
@usableFromInline
internal func l2sqr(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    var sum: Float = 0
    let dv = d & ~3
    var j = 0
    while j < dv {
        let va = load4(a, j), vb = load4(b, j)
        let diff = va - vb
        sum += sum4(diff * diff)
        j += 4
    }
    while j < d {
        let diff = a[j] - b[j]
        sum += diff * diff
        j += 1
    }
    return sum
}

@inline(__always)
@usableFromInline
internal func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    var acc: Float = 0
    let dv = d & ~3
    var j = 0
    while j < dv {
        let va = load4(a, j), vb = load4(b, j)
        acc += sum4(va * vb)
        j += 4
    }
    while j < d { acc += a[j] * b[j]; j += 1 }
    return acc
}

@inline(__always)
@usableFromInline
internal func sqnorm(_ a: UnsafePointer<Float>, _ d: Int) -> Float {
    var acc: Float = 0
    let dv = d & ~3
    var j = 0
    while j < dv {
        let v = load4(a, j)
        acc += sum4(v * v)
        j += 4
    }
    while j < d { acc += a[j] * a[j]; j += 1 }
    return acc
}

// MARK: - Tiny SIMD4 helpers

@inline(__always)
@usableFromInline
internal func load4(_ base: UnsafePointer<Float>, _ off: Int) -> SIMD4<Float> {
    (base + off).withMemoryRebound(to: SIMD4<Float>.self, capacity: 1) { $0.pointee }
}

@inline(__always)
@usableFromInline
internal func sum4(_ v: SIMD4<Float>) -> Float { v[0] + v[1] + v[2] + v[3] }
