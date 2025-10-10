//===----------------------------------------------------------------------===//
//  PQLUT.swift
//  VectorIndex
//
//  Kernel #21: PQ Lookup Table (LUT) Construction
//
//  Builds per-query lookup tables L[j, k] = ||q_j - C_j[k]||^2 enabling O(m)
//  ADC scans. Supports direct L2 path and dot-product trick using precomputed
//  centroid norms. Also includes fused residual LUTs for IVF-PQ.
//
//  Layouts (row-major):
//    - codebooks: [m × ks × dsub] with C_j[k][i] at codebooks[j*ks*dsub + k*dsub + i]
//    - lut:       [m × ks]        with lut[j*ks + k]
//    - q:         [d], dsub = d/m
//
//  References: Spec #21 (API, options, algorithms, SIMD pattern)
//===----------------------------------------------------------------------===//

import Foundation
import simd

// MARK: - Alignment & Prefetch (mocked for portability)

@inline(__always)
@usableFromInline
internal func _verifyAlignment(_ ptr: UnsafeRawPointer?, _ alignment: Int, _ label: String) {
#if DEBUG
    if let p = ptr {
        let addr = Int(bitPattern: p)
        precondition(addr % alignment == 0, "\(label) must be \(alignment)-byte aligned (got: \(addr))")
    }
#endif
}

@inline(__always)
@usableFromInline
internal func _prefetch(_ ptr: UnsafeRawPointer?) {
    // Advisory only; left as a no-op in portable Swift.
    // On Apple Silicon, compilers often auto-prefetch. Spec includes the
    // knob for tuning, so we keep the API for future inline asm if desired.
    _ = ptr
}

// MARK: - Options

/// Configuration options for PQ LUT construction (mirrors spec).
public struct PQLutOpts: Sendable {
    /// Use dot-product distance optimization (||q||² + ||c||² − 2<q, c>).
    /// If `nil`, we auto-enable when `centroidNorms != nil` and `ks >= 64`.
    public var useDotTrick: Bool?

    /// Include per-subspace query norm in LUT entries.
    /// When false, L[j][k] = ||C_j[k]||² − 2<q_j, C_j[k]> (ADC needs bias).
    public var includeQNorm: Bool

    /// Scalar strict-FP fallback (disable SIMD/reassociation).
    public var strictFP: Bool

    /// Software prefetch lookahead in centroids (advisory).
    public var prefetchDistance: Int

    /// Batch builder thread count (0 or 1 = single-threaded).
    public var numThreads: Int

    public static let `default` = PQLutOpts(
        useDotTrick: nil,               // auto
        includeQNorm: true,
        strictFP: false,
        prefetchDistance: 8,
        numThreads: 0
    )
}

// MARK: - Helpers (SIMD & Scalar kernels)

@inline(__always)
@usableFromInline
internal func _simd_l2sqr(
    _ a: UnsafePointer<Float>,
    _ b: UnsafePointer<Float>,
    _ len: Int
) -> Float {
    // Dual-accumulator, unrolled by 8 (2×SIMD4) as in spec
    // len8 = floor(len/8) * 8
    let len8 = len & ~7
    var acc0 = SIMD4<Float>.zero
    var acc1 = SIMD4<Float>.zero

    var i = 0
    while i < len8 {
        let a0 = UnsafeRawPointer(a + i + 0).load(as: SIMD4<Float>.self)
        let a1 = UnsafeRawPointer(a + i + 4).load(as: SIMD4<Float>.self)
        let b0 = UnsafeRawPointer(b + i + 0).load(as: SIMD4<Float>.self)
        let b1 = UnsafeRawPointer(b + i + 4).load(as: SIMD4<Float>.self)

        let d0 = a0 - b0
        let d1 = a1 - b1

        acc0 += d0 * d0
        acc1 += d1 * d1
        i &+= 8
    }

    var sum = acc0[0] + acc0[1] + acc0[2] + acc0[3]
    sum +=     acc1[0] + acc1[1] + acc1[2] + acc1[3]

    // Scalar tail
    while i < len {
        let diff = a[i] - b[i]
        sum += diff * diff
        i &+= 1
    }
    return sum
}

@inline(__always)
@usableFromInline
internal func _simd_dot(
    _ a: UnsafePointer<Float>,
    _ b: UnsafePointer<Float>,
    _ len: Int
) -> Float {
    // Dual accumulator, unrolled by 8 (2×SIMD4)
    let len8 = len & ~7
    var acc0 = SIMD4<Float>.zero
    var acc1 = SIMD4<Float>.zero

    var i = 0
    while i < len8 {
        let a0 = UnsafeRawPointer(a + i + 0).load(as: SIMD4<Float>.self)
        let a1 = UnsafeRawPointer(a + i + 4).load(as: SIMD4<Float>.self)
        let b0 = UnsafeRawPointer(b + i + 0).load(as: SIMD4<Float>.self)
        let b1 = UnsafeRawPointer(b + i + 4).load(as: SIMD4<Float>.self)

        acc0 += a0 * b0
        acc1 += a1 * b1
        i &+= 8
    }

    var sum = acc0[0] + acc0[1] + acc0[2] + acc0[3]
    sum +=     acc1[0] + acc1[1] + acc1[2] + acc1[3]

    // Scalar tail
    while i < len {
        sum += a[i] * b[i]
        i &+= 1
    }
    return sum
}

@inline(__always)
@usableFromInline
internal func _scalar_l2sqr(
    _ a: UnsafePointer<Float>,
    _ b: UnsafePointer<Float>,
    _ len: Int
) -> Float {
    var s: Float = 0
    for i in 0..<len {
        let d = a[i] - b[i]
        s += d * d
    }
    return s
}

@inline(__always)
@usableFromInline
internal func _scalar_dot(
    _ a: UnsafePointer<Float>,
    _ b: UnsafePointer<Float>,
    _ len: Int
) -> Float {
    var s: Float = 0
    for i in 0..<len { s += a[i] * b[i] }
    return s
}

// MARK: - Public API

/// Compute per-subspace squared norms of the query: q_sub_norms[j] = ||q_j||².
/// Use once and reuse across multiple LUTs (e.g., IVF-PQ nprobe > 1).
@inlinable
public func pq_query_subnorms_f32(
    query q: UnsafePointer<Float>,
    dimension d: Int,
    m: Int,
    out qSubNorms: UnsafeMutablePointer<Float>
) {
    precondition(d > 0 && m > 0 && d % m == 0, "d must be divisible by m")
    let dsub = d / m

    for j in 0..<m {
        let qj = q + j * dsub
        qSubNorms[j] = _simd_dot(qj, qj, dsub)  // fast path
    }
}

/// Primary LUT builder: L[j,k] = ||q_j − C_j[k]||².
@inlinable
public func pq_lut_l2_f32(
    query q: UnsafePointer<Float>,                 // [d]
    dimension d: Int,
    m: Int,
    ks: Int,                                       // 256 (u8) or 16 (u4)
    codebooks: UnsafePointer<Float>,               // [m × ks × dsub]
    out lut: UnsafeMutablePointer<Float>,          // [m × ks]
    centroidNorms: UnsafePointer<Float>? = nil,    // [m × ks] (optional)
    qSubNorms: UnsafePointer<Float>? = nil,        // [m] (optional)
    opts inOpts: PQLutOpts = .default
) {
    precondition(d > 0 && m > 0 && d % m == 0, "d must be divisible by m")
    precondition(ks > 0, "ks > 0")

    // Validate/derive options
    let dsub = d / m
    let useDot: Bool = {
        if let forced = inOpts.useDotTrick { return forced }
        return (centroidNorms != nil) && (ks >= 64)
    }()
    let includeQ = inOpts.includeQNorm
    let strictFP = inOpts.strictFP

    // Debug alignment checks (advisory)
    _verifyAlignment(q, 64, "q")
    _verifyAlignment(codebooks, 64, "codebooks")
    _verifyAlignment(lut, 64, "lut")

    for j in 0..<m {
        let qj = q + j * dsub
        let cbJ = codebooks + j * ks * dsub
        let lutJ = lut + j * ks
        let qjNorm: Float = includeQ
            ? (qSubNorms != nil ? qSubNorms![j] : (strictFP ? _scalar_dot(qj, qj, dsub) : _simd_dot(qj, qj, dsub)))
            : 0.0

        if useDot {
            // Dot-product trick: ||q||² + ||c||² - 2<q, c>
            // If includeQ=false, we omit qjNorm to save m additions (ADC adds a bias).
            let cNormBase = centroidNorms! + j * ks
            if strictFP {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    let dot = _scalar_dot(qj, c, dsub)
                    let dist = (includeQ ? qjNorm : 0.0) + cNormBase[k] - 2.0 * dot
                    lutJ[k] = dist
                    if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                        _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                    }
                }
            } else {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    let dot = _simd_dot(qj, c, dsub)
                    let dist = (includeQ ? qjNorm : 0.0) + cNormBase[k] - 2.0 * dot
                    lutJ[k] = dist
                    if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                        _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                    }
                }
            }
        } else {
            // Direct L2: Σ (q - c)^2  (reference path, SIMD-accelerated).
            if strictFP {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    lutJ[k] = _scalar_l2sqr(qj, c, dsub)
                    if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                        _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                    }
                }
            } else {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    lutJ[k] = _simd_l2sqr(qj, c, dsub)
                    if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                        _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                    }
                }
            }
        }
    }
}

/// Residual LUT for IVF‑PQ: r = q − coarse, fused with LUT construction to
/// avoid materializing residuals (saves d*4 bytes).
@inlinable
public func pq_lut_residual_l2_f32(
    query q: UnsafePointer<Float>,                // [d]
    coarseCentroid coarse: UnsafePointer<Float>,  // [d]
    dimension d: Int,
    m: Int,
    ks: Int,
    codebooks: UnsafePointer<Float>,              // residual codebooks [m × ks × dsub]
    out lut: UnsafeMutablePointer<Float>,         // [m × ks]
    centroidNorms: UnsafePointer<Float>? = nil,   // [m × ks] (optional)
    opts inOpts: PQLutOpts = .default
) {
    precondition(d > 0 && m > 0 && d % m == 0, "d must be divisible by m")
    let dsub = d / m

    let useDot: Bool = {
        if let forced = inOpts.useDotTrick { return forced }
        return (centroidNorms != nil) && (ks >= 64)
    }()
    let includeQ = inOpts.includeQNorm
    let strictFP = inOpts.strictFP

    for j in 0..<m {
        let qj = q + j * dsub
        let cj = coarse + j * dsub
        let cbJ = codebooks + j * ks * dsub
        let lutJ = lut + j * ks

        if useDot {
            // ||(q - coarse)||² + ||c||² - 2 <(q - coarse), c>
            // If includeQ=false, omit the residual norm part (ADC adds bias).
            let cNormBase = centroidNorms! + j * ks

            // Compute residual norm once per subspace if needed
            let rNorm: Float
            if includeQ {
                rNorm = strictFP
                    ? _scalar_l2sqr(qj, cj, dsub)
                    : _simd_l2sqr(qj, cj, dsub)
            } else {
                rNorm = 0
            }

            if strictFP {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    // dot(q - coarse, c)
                    var dot: Float = 0
                    for i in 0..<dsub { dot += (qj[i] - cj[i]) * c[i] }
                    lutJ[k] = rNorm + cNormBase[k] - 2.0 * dot
                }
            } else {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    // SIMD: dot(q - coarse, c)
                    let len8 = dsub & ~7
                    var acc0 = SIMD4<Float>.zero
                    var acc1 = SIMD4<Float>.zero
                    var i = 0
                    while i < len8 {
                        let q0 = UnsafeRawPointer(qj + i + 0).load(as: SIMD4<Float>.self)
                        let q1 = UnsafeRawPointer(qj + i + 4).load(as: SIMD4<Float>.self)
                        let cc0 = UnsafeRawPointer(cj + i + 0).load(as: SIMD4<Float>.self)
                        let cc1 = UnsafeRawPointer(cj + i + 4).load(as: SIMD4<Float>.self)
                        let rc0 = q0 - cc0
                        let rc1 = q1 - cc1
                        let c0  = UnsafeRawPointer(c + i + 0).load(as: SIMD4<Float>.self)
                        let c1  = UnsafeRawPointer(c + i + 4).load(as: SIMD4<Float>.self)
                        acc0 += rc0 * c0
                        acc1 += rc1 * c1
                        i &+= 8
                    }
                    var dot = acc0[0] + acc0[1] + acc0[2] + acc0[3]
                    dot +=     acc1[0] + acc1[1] + acc1[2] + acc1[3]
                    while i < dsub { dot += (qj[i] - cj[i]) * c[i]; i &+= 1 }
                    lutJ[k] = rNorm + cNormBase[k] - 2.0 * dot
                    if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                        _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                    }
                }
            }
        } else {
            // Direct residual L2: Σ ( (q - coarse) - c )², fused without storing r.
            if strictFP {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    var s: Float = 0
                    for i in 0..<dsub {
                        let diff = (qj[i] - cj[i]) - c[i]
                        s += diff * diff
                    }
                    lutJ[k] = s
                }
            } else {
                for k in 0..<ks {
                    let c = cbJ + k * dsub
                    // SIMD: ((q - coarse) - c)^2
                    let len8 = dsub & ~7
                    var acc0 = SIMD4<Float>.zero
                    var acc1 = SIMD4<Float>.zero
                    var i = 0
                    while i < len8 {
                        let q0 = UnsafeRawPointer(qj + i + 0).load(as: SIMD4<Float>.self)
                        let q1 = UnsafeRawPointer(qj + i + 4).load(as: SIMD4<Float>.self)
                        let cc0 = UnsafeRawPointer(cj + i + 0).load(as: SIMD4<Float>.self)
                        let cc1 = UnsafeRawPointer(cj + i + 4).load(as: SIMD4<Float>.self)
                        let r0 = (q0 - cc0) - UnsafeRawPointer(c + i + 0).load(as: SIMD4<Float>.self)
                        let r1 = (q1 - cc1) - UnsafeRawPointer(c + i + 4).load(as: SIMD4<Float>.self)
                        acc0 += r0 * r0
                        acc1 += r1 * r1
                        i &+= 8
                    }
                    var sum = acc0[0] + acc0[1] + acc0[2] + acc0[3]
                    sum    += acc1[0] + acc1[1] + acc1[2] + acc1[3]
                    while i < dsub {
                        let diff = (qj[i] - cj[i]) - c[i]
                        sum += diff * diff
                        i &+= 1
                    }
                    lutJ[k] = sum
                    if inOpts.prefetchDistance > 0 && k + inOpts.prefetchDistance < ks {
                        _prefetch(UnsafeRawPointer(cbJ + (k + inOpts.prefetchDistance) * dsub))
                    }
                }
            }
        }
    }
}

/// Build LUTs for a batch of queries: luts[qi][j*ks + k]. Parallelizes over queries
/// if `opts.numThreads > 1`.
@inlinable
public func pq_lut_batch_l2_f32(
    queries: UnsafePointer<Float>,                 // [nQ × d]
    nQueries: Int,
    dimension d: Int,
    m: Int,
    ks: Int,
    codebooks: UnsafePointer<Float>,               // [m × ks × dsub]
    out luts: UnsafeMutablePointer<Float>,         // [nQ × m × ks]
    centroidNorms: UnsafePointer<Float>? = nil,    // [m × ks] (optional)
    opts inOpts: PQLutOpts = .default
) {
    precondition(nQueries >= 0 && d > 0 && m > 0 && d % m == 0)
    let dsub = d / m
    let perLUT = m * ks

    if inOpts.numThreads <= 1 || nQueries < 2 {
        // Sequential (low latency)
        for qid in 0..<nQueries {
            let qPtr = queries + qid * d
            let lutPtr = luts + qid * perLUT
            pq_lut_l2_f32(
                query: qPtr,
                dimension: d,
                m: m,
                ks: ks,
                codebooks: codebooks,
                out: lutPtr,
                centroidNorms: centroidNorms,
                qSubNorms: nil,
                opts: inOpts
            )
        }
    } else {
        // Coarse-grained parallelism across queries (throughput path)
        let threads = max(1, inOpts.numThreads)
        let group = DispatchGroup()
        let queue = DispatchQueue.global(qos: .userInitiated)

        // Chunk queries evenly
        let chunk = (nQueries + threads - 1) / threads
        for t in 0..<threads {
            let start = t * chunk
            let end = min(nQueries, start + chunk)
            if start >= end { continue }
            group.enter()
            queue.async {
                for qid in start..<end {
                    let qPtr = queries + qid * d
                    let lutPtr = luts + qid * perLUT
                    pq_lut_l2_f32(
                        query: qPtr,
                        dimension: d,
                        m: m,
                        ks: ks,
                        codebooks: codebooks,
                        out: lutPtr,
                        centroidNorms: centroidNorms,
                        qSubNorms: nil,
                        opts: inOpts
                    )
                }
                group.leave()
            }
        }
        group.wait()
    }

    _ = dsub // silence unused var in certain build flags
}

// MARK: - Convenience (Swift Array front-ends)

extension Array where Element == Float {
    /// Build a single LUT into a newly allocated [Float].
    public func pqBuildLUT(
        m: Int,
        ks: Int,
        codebooks: [Float],
        centroidNorms: [Float]? = nil,
        opts: PQLutOpts = .default
    ) -> [Float] {
        precondition(self.count > 0)
        let d = self.count
        precondition(d % m == 0, "d must be divisible by m")
        precondition(codebooks.count == m * ks * (d / m))

        var out = [Float](repeating: 0, count: m * ks)
        self.withUnsafeBufferPointer { qPtr in
            codebooks.withUnsafeBufferPointer { cbPtr in
                out.withUnsafeMutableBufferPointer { lutPtr in
                    if let cn = centroidNorms {
                        cn.withUnsafeBufferPointer { cnPtr in
                            pq_lut_l2_f32(
                                query: qPtr.baseAddress!, dimension: d, m: m, ks: ks,
                                codebooks: cbPtr.baseAddress!, out: lutPtr.baseAddress!,
                                centroidNorms: cnPtr.baseAddress!, qSubNorms: nil, opts: opts
                            )
                        }
                    } else {
                        pq_lut_l2_f32(
                            query: qPtr.baseAddress!, dimension: d, m: m, ks: ks,
                            codebooks: cbPtr.baseAddress!, out: lutPtr.baseAddress!,
                            centroidNorms: nil, qSubNorms: nil, opts: opts
                        )
                    }
                }
            }
        }
        return out
    }
}
