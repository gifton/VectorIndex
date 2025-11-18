import Foundation
import simd

// L2^2 distance microkernel (#01)
// - Direct path: Σ (q - x)^2
// - Dot-trick path: ‖q‖² + ‖x‖² - 2⟨q, x⟩ (fused IP+epilogue)
// - Specialized dims: 512, 768, 1024, 1536
// - Generic SIMD fallback (Swift SIMD4<Float>)
// - CPU-bound only; no background work

// MARK: - Public C-like API & Options

public enum L2SqrAlgo: Int32, Sendable {
    case auto      = 0
    case direct    = 1
    case dotTrick  = 2
}

public struct L2SqrOpts: Sendable {
    public var algo: L2SqrAlgo
    // NOTE: `useDotTrick` is redundant with `algo`; kept for compatibility but ignored when `algo != .auto`.
    public var useDotTrick: Bool
    public var prefetchDistance: Int32
    public var strictFP: Bool
    public var numThreads: Int32 // 0=auto, >0 explicit

    public static let `default` = L2SqrOpts(
        algo: .auto, useDotTrick: false, prefetchDistance: 8, strictFP: false, numThreads: 0
    )

    public init(algo: L2SqrAlgo = .auto,
                useDotTrick: Bool = false,
                prefetchDistance: Int32 = 8,
                strictFP: Bool = false,
                numThreads: Int32 = 0) {
        self.algo = algo
        self.useDotTrick = useDotTrick
        self.prefetchDistance = prefetchDistance
        self.strictFP = strictFP
        self.numThreads = numThreads
    }
}

// MARK: - Telemetry (opt-in sink)

public struct L2SqrTelemetry {
    public let rows: Int
    public let dim: Int
    public let usedDotTrick: Bool
    public let specializedDim: Int?     // 512/768/1024/1536 or nil
    public let bytesRead: Int           // rough accounting
    public let executionTimeNanos: UInt64
}

public enum L2SqrTelemetryRecorder {
    public nonisolated(unsafe) static var sink: ((L2SqrTelemetry) -> Void)?
    @inline(__always) public static func record(_ t: L2SqrTelemetry) { sink?(t) }
}

// MARK: - Helper: Alignment verification (performance hint)
//
// Note: 16-byte alignment enables optimal SIMD performance, but unaligned data
// is handled correctly (just slower). We verify alignment in debug builds for
// performance profiling, but don't enforce it since Swift [Float] arrays are
// not guaranteed to be aligned. The SIMD4<Float> operations handle unaligned
// loads correctly, they're just slightly slower due to extra memory ops.

@inline(__always)
func _verifyAlignment(_ ptr: UnsafeRawPointer?, _ label: String, alignment: Int = 16) {
    // Alignment check removed - Swift arrays are not guaranteed to be aligned,
    // and SIMD4 operations handle unaligned data correctly (with minor perf impact)
    // For optimal performance, users can pre-align data using:
    //   UnsafeMutableRawBufferPointer.allocate(byteCount:alignment:)
}

// MARK: - Helper: Prefetch (hint-only; no-op on Swift)

@inline(__always)
func _prefetchRow(_ base: UnsafeRawPointer, _ byteStride: Int) {
    _ = base; _ = byteStride
}

// MARK: - SIMD helpers

extension SIMD4 where Scalar == Float {
    @inline(__always) init(_ ptr: UnsafePointer<Float>) {
        self.init(ptr[0], ptr[1], ptr[2], ptr[3])
    }
    @inline(__always) func sum() -> Float { self[0] + self[1] + self[2] + self[3] }
}

// MARK: - Public API

@inline(__always)
public func l2sqr_f32_single(
    _ q: UnsafePointer<Float>, _ x: UnsafePointer<Float>, _ d: Int
) -> Float {
    _l2sqr_single_direct(q: q, x: x, d: d, kahan: false)
}

public func l2sqr_f32_block(
    _ q: UnsafePointer<Float>,
    _ xb: UnsafePointer<Float>,
    _ n: Int,
    _ d: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ xb_norm: UnsafePointer<Float>? = nil,
    _ q_norm: Float = .nan,
    _ opts: UnsafePointer<L2SqrOpts>? = nil
) {
    _verifyAlignment(q, "q")
    _verifyAlignment(xb, "xb")
    _verifyAlignment(out, "out")
    if let xbNorm = xb_norm { _verifyAlignment(xbNorm, "xb_norm") }

    guard n > 0, d > 0 else { return }

    let options = opts?.pointee ?? .default
    let wantDot = (options.algo == .auto) ? options.useDotTrick : false
    let canDot  = (xb_norm != nil) || !q_norm.isNaN

    // Decide algorithm
    let useDotTrick: Bool = {
        switch options.algo {
        case .direct: return false
        case .dotTrick: return true // compute norms on-the-fly if needed
        case .auto:
            // Prefer dot-trick when norms are available; otherwise for large dims.
            if canDot { return true }
            // fallback to hint or dimension heuristic
            if wantDot { return true }
            return d >= 256
        }
    }()

    let t0 = DispatchTime.now().uptimeNanoseconds

    if useDotTrick {
        _l2sqr_block_dot_fused(
            q: q, xb: xb, n: n, d: d,
            out: out,
            xbNorm: xb_norm,
            qNorm: q_norm,
            opts: options
        )
    } else {
        _l2sqr_block_direct(
            q: q, xb: xb, n: n, d: d,
            out: out,
            opts: options
        )
    }

    let t1 = DispatchTime.now().uptimeNanoseconds
    let dbBytes = n * d * MemoryLayout<Float>.stride
    let qBytes  = d * MemoryLayout<Float>.stride
    let normBytes = (xb_norm != nil ? n * MemoryLayout<Float>.stride : 0)
    L2SqrTelemetryRecorder.record(L2SqrTelemetry(
        rows: n,
        dim: d,
        usedDotTrick: useDotTrick,
        specializedDim: _isSpecializedDim(d) ? d : nil,
        bytesRead: dbBytes + qBytes + normBytes,
        executionTimeNanos: t1 &- t0
    ))
}

// MARK: - Internal: Direct paths

@inline(__always)
func _isSpecializedDim(_ d: Int) -> Bool {
    d == 512 || d == 768 || d == 1024 || d == 1536
}

@inline(__always)
func _l2sqr_block_direct(
    q: UnsafePointer<Float>, xb: UnsafePointer<Float>, n: Int, d: Int,
    out: UnsafeMutablePointer<Float>, opts: L2SqrOpts
) {
    // Sendable wrappers for pointers used in concurrent closures
    struct UnsafeSendablePtr<T>: @unchecked Sendable { let p: UnsafePointer<T> }
    struct UnsafeSendableMutPtr<T>: @unchecked Sendable { let p: UnsafeMutablePointer<T> }
    let qS = UnsafeSendablePtr(p: q)
    let xbS = UnsafeSendablePtr(p: xb)
    let outS = UnsafeSendableMutPtr(p: out)

    let threads: Int = {
        if opts.numThreads > 0 { return Int(opts.numThreads) }
        if opts.numThreads == 0 { return max(1, ProcessInfo.processInfo.activeProcessorCount) }
        return 1
    }()

    if threads == 1 || n < 1024 {
        switch d {
        case 512:  _l2sqr_direct_d512(q, xb, n, out, opts.strictFP)
        case 768:  _l2sqr_direct_d768(q, xb, n, out, opts.strictFP)
        case 1024: _l2sqr_direct_d1024(q, xb, n, out, opts.strictFP)
        case 1536: _l2sqr_direct_d1536(q, xb, n, out, opts.strictFP)
        default:   _l2sqr_direct_generic(q, xb, n, d, out, opts.strictFP, Int(opts.prefetchDistance))
        }
        return
    }

    let rowsPer = (n + threads - 1) / threads
    DispatchQueue.concurrentPerform(iterations: threads) { tid in
        let start = tid * rowsPer
        let end = min(n, start + rowsPer)
        guard start < end else { return }
        switch d {
        case 512:  _l2sqr_direct_d512(qS.p, xbS.p + start * d, end - start, outS.p + start, opts.strictFP)
        case 768:  _l2sqr_direct_d768(qS.p, xbS.p + start * d, end - start, outS.p + start, opts.strictFP)
        case 1024: _l2sqr_direct_d1024(qS.p, xbS.p + start * d, end - start, outS.p + start, opts.strictFP)
        case 1536: _l2sqr_direct_d1536(qS.p, xbS.p + start * d, end - start, outS.p + start, opts.strictFP)
        default:   _l2sqr_direct_generic(qS.p, xbS.p + start * d, end - start, d, outS.p + start, opts.strictFP, Int(opts.prefetchDistance))
        }
    }
}

@inline(__always)
func _l2sqr_single_direct(
    q: UnsafePointer<Float>,
    x: UnsafePointer<Float>,
    d: Int,
    kahan: Bool
) -> Float {
    var acc0 = SIMD4<Float>.zero
    var acc1 = SIMD4<Float>.zero
    var acc2 = SIMD4<Float>.zero
    var acc3 = SIMD4<Float>.zero

    var j = 0
    while j + 15 < d {
        let q0 = SIMD4<Float>(q + j + 0), x0 = SIMD4<Float>(x + j + 0)
        let q1 = SIMD4<Float>(q + j + 4), x1 = SIMD4<Float>(x + j + 4)
        let q2 = SIMD4<Float>(q + j + 8), x2 = SIMD4<Float>(x + j + 8)
        let q3 = SIMD4<Float>(q + j + 12), x3 = SIMD4<Float>(x + j + 12)
        let d0 = q0 - x0
        let d1 = q1 - x1
        let d2 = q2 - x2
        let d3 = q3 - x3
        acc0 += d0 * d0
        acc1 += d1 * d1
        acc2 += d2 * d2
        acc3 += d3 * d3
        j += 16
    }

    var sum = (acc0 + acc1 + acc2 + acc3).sum()
    if kahan {
        var c: Float = 0
        for t in j..<d {
            let diff = q[t] - x[t]
            let y = (diff * diff) - c
            let tSum = sum + y
            c = (tSum - sum) - y
            sum = tSum
        }
        return sum
    } else {
        for t in j..<d {
            let diff = q[t] - x[t]
            sum += diff * diff
        }
        return sum
    }
}

@inline(__always)
func _l2sqr_direct_generic(
    _ q: UnsafePointer<Float>,
    _ xb: UnsafePointer<Float>,
    _ n: Int, _ d: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ strict: Bool,
    _ prefetchDistance: Int
) {
    let rowBytes = d * MemoryLayout<Float>.stride
    for i in 0..<n {
        // Prefetch next row (hint only)
        let pfRow = i + prefetchDistance
        if pfRow < n { _prefetchRow(UnsafeRawPointer(xb + pfRow * d), rowBytes) }
        let x = xb + i * d
        out[i] = _l2sqr_single_direct(q: q, x: x, d: d, kahan: strict)
    }
}

// MARK: - Specialized direct dims (fully unrolled inner structure)

@inline(__always)
func _l2sqr_direct_d512(
    _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ n: Int,
    _ out: UnsafeMutablePointer<Float>, _ strict: Bool
) {
    let d = 512
    for i in 0..<n {
        let x = xb + i * d
        var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
        for j in stride(from: 0, to: d, by: 16) {
            let q0 = SIMD4<Float>(q + j + 0), x0 = SIMD4<Float>(x + j + 0)
            let q1 = SIMD4<Float>(q + j + 4), x1 = SIMD4<Float>(x + j + 4)
            let q2 = SIMD4<Float>(q + j + 8), x2 = SIMD4<Float>(x + j + 8)
            let q3 = SIMD4<Float>(q + j + 12), x3 = SIMD4<Float>(x + j + 12)
            let d0 = q0 - x0; a0 += d0 * d0
            let d1 = q1 - x1; a1 += d1 * d1
            let d2 = q2 - x2; a2 += d2 * d2
            let d3 = q3 - x3; a3 += d3 * d3
        }
        let sum = (a0 + a1 + a2 + a3).sum()
        out[i] = sum
    }
}

@inline(__always)
func _l2sqr_direct_d768(
    _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ n: Int,
    _ out: UnsafeMutablePointer<Float>, _ strict: Bool
) {
    let d = 768
    for i in 0..<n {
        let x = xb + i * d
        var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
        for j in stride(from: 0, to: d, by: 16) {
            let q0 = SIMD4<Float>(q + j + 0), x0 = SIMD4<Float>(x + j + 0)
            let q1 = SIMD4<Float>(q + j + 4), x1 = SIMD4<Float>(x + j + 4)
            let q2 = SIMD4<Float>(q + j + 8), x2 = SIMD4<Float>(x + j + 8)
            let q3 = SIMD4<Float>(q + j + 12), x3 = SIMD4<Float>(x + j + 12)
            let d0 = q0 - x0; a0 += d0 * d0
            let d1 = q1 - x1; a1 += d1 * d1
            let d2 = q2 - x2; a2 += d2 * d2
            let d3 = q3 - x3; a3 += d3 * d3
        }
        let sum = (a0 + a1 + a2 + a3).sum()
        out[i] = sum
    }
}

@inline(__always)
func _l2sqr_direct_d1024(
    _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ n: Int,
    _ out: UnsafeMutablePointer<Float>, _ strict: Bool
) {
    let d = 1024
    for i in 0..<n {
        let x = xb + i * d
        var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
        for j in stride(from: 0, to: d, by: 16) {
            let q0 = SIMD4<Float>(q + j + 0), x0 = SIMD4<Float>(x + j + 0)
            let q1 = SIMD4<Float>(q + j + 4), x1 = SIMD4<Float>(x + j + 4)
            let q2 = SIMD4<Float>(q + j + 8), x2 = SIMD4<Float>(x + j + 8)
            let q3 = SIMD4<Float>(q + j + 12), x3 = SIMD4<Float>(x + j + 12)
            let d0 = q0 - x0; a0 += d0 * d0
            let d1 = q1 - x1; a1 += d1 * d1
            let d2 = q2 - x2; a2 += d2 * d2
            let d3 = q3 - x3; a3 += d3 * d3
        }
        let sum = (a0 + a1 + a2 + a3).sum()
        out[i] = sum
    }
}

@inline(__always)
func _l2sqr_direct_d1536(
    _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ n: Int,
    _ out: UnsafeMutablePointer<Float>, _ strict: Bool
) {
    let d = 1536
    for i in 0..<n {
        let x = xb + i * d
        var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
        for j in stride(from: 0, to: d, by: 16) {
            let q0 = SIMD4<Float>(q + j + 0), x0 = SIMD4<Float>(x + j + 0)
            let q1 = SIMD4<Float>(q + j + 4), x1 = SIMD4<Float>(x + j + 4)
            let q2 = SIMD4<Float>(q + j + 8), x2 = SIMD4<Float>(x + j + 8)
            let q3 = SIMD4<Float>(q + j + 12), x3 = SIMD4<Float>(x + j + 12)
            let d0 = q0 - x0; a0 += d0 * d0
            let d1 = q1 - x1; a1 += d1 * d1
            let d2 = q2 - x2; a2 += d2 * d2
            let d3 = q3 - x3; a3 += d3 * d3
        }
        let sum = (a0 + a1 + a2 + a3).sum()
        out[i] = sum
    }
}

// MARK: - Internal: Dot-product trick (fused)

@inline(__always)
func _l2sqr_block_dot_fused(
    q: UnsafePointer<Float>,
    xb: UnsafePointer<Float>,
    n: Int,
    d: Int,
    out: UnsafeMutablePointer<Float>,
    xbNorm: UnsafePointer<Float>?,
    qNorm: Float,
    opts: L2SqrOpts
) {
    // Sendable wrappers for pointers used in concurrent closures
    struct UnsafeSendablePtr<T>: @unchecked Sendable { let p: UnsafePointer<T> }
    struct UnsafeSendableMutPtr<T>: @unchecked Sendable { let p: UnsafeMutablePointer<T> }
    let qS = UnsafeSendablePtr(p: q)
    let xbS = UnsafeSendablePtr(p: xb)
    let outS = UnsafeSendableMutPtr(p: out)
    let xbNormS: UnsafeSendablePtr<Float>? = xbNorm.map { .init(p: $0) }

    let computedQNorm: Float = qNorm.isNaN ? _normSquared(q, d) : qNorm
    let haveXbNorm = (xbNorm != nil)
    let vecWidth = 4
    let dBlocked = (d / vecWidth) * vecWidth

    let threads: Int = {
        if opts.numThreads > 0 { return Int(opts.numThreads) }
        if opts.numThreads == 0 { return max(1, ProcessInfo.processInfo.activeProcessorCount) }
        return 1
    }()

    if threads == 1 || n < 1024 {
        _l2sqr_block_dot_fused_serial(
            q: q, xb: xb, n: n, d: d, dBlocked: dBlocked,
            out: out,
            xbNorm: xbNorm,
            qNorm: computedQNorm
        )
        return
    }

    let rowsPer = (n + threads - 1) / threads
    DispatchQueue.concurrentPerform(iterations: threads) { tid in
        let start = tid * rowsPer
        let end = min(n, start + rowsPer)
        guard start < end else { return }
        let xbNormPart = haveXbNorm ? (xbNormS!.p + start) : nil
        _l2sqr_block_dot_fused_serial(
            q: qS.p, xb: xbS.p + start * d, n: end - start, d: d, dBlocked: dBlocked,
            out: outS.p + start,
            xbNorm: xbNormPart,
            qNorm: computedQNorm
        )
    }
}

@inline(__always)
private func _l2sqr_block_dot_fused_serial(
    q: UnsafePointer<Float>,
    xb: UnsafePointer<Float>,
    n: Int,
    d: Int,
    dBlocked: Int,
    out: UnsafeMutablePointer<Float>,
    xbNorm: UnsafePointer<Float>?,
    qNorm: Float
) {
    for i in 0..<n {
        let row = xb + i * d
        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero
        var acc2 = SIMD4<Float>.zero
        var acc3 = SIMD4<Float>.zero

        var j = 0
        while j + 15 < dBlocked {
            let q0 = SIMD4<Float>(q + j + 0), x0 = SIMD4<Float>(row + j + 0)
            let q1 = SIMD4<Float>(q + j + 4), x1 = SIMD4<Float>(row + j + 4)
            let q2 = SIMD4<Float>(q + j + 8), x2 = SIMD4<Float>(row + j + 8)
            let q3 = SIMD4<Float>(q + j + 12), x3 = SIMD4<Float>(row + j + 12)
            acc0 += q0 * x0
            acc1 += q1 * x1
            acc2 += q2 * x2
            acc3 += q3 * x3
            j += 16
        }
        var dot = (acc0 + acc1 + acc2 + acc3).sum()
        for t in j..<d { dot += q[t] * row[t] }

        let xn = (xbNorm != nil) ? xbNorm![i] : _normSquared(row, d)
        var dist = qNorm + xn - 2.0 * dot
        if dist < 0 { dist = 0 }
        out[i] = dist
    }
}

// MARK: - Helpers: norm^2 (SIMD)

@inline(__always)
func _normSquared(_ v: UnsafePointer<Float>, _ d: Int) -> Float {
    var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
    var j = 0
    while j + 15 < d {
        let v0 = SIMD4<Float>(v + j + 0)
        let v1 = SIMD4<Float>(v + j + 4)
        let v2 = SIMD4<Float>(v + j + 8)
        let v3 = SIMD4<Float>(v + j + 12)
        a0 += v0 * v0
        a1 += v1 * v1
        a2 += v2 * v2
        a3 += v3 * v3
        j += 16
    }
    var sum = (a0 + a1 + a2 + a3).sum()
    for t in j..<d { let val = v[t]; sum += val * val }
    return sum
}
