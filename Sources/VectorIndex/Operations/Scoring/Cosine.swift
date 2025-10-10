import Foundation
import simd
import Dispatch

public extension IndexOps.Scoring {
    enum Cosine {
        // MARK: Options
        public struct Options {
            public var epsilon: Float = 1e-12
            public var enableTelemetry: Bool = false
            public var verifyAlignment: Bool = false
            public var clampOutput: Bool = true
            public var useFusedKernel: Bool = true
            public init() {}
        }

        // MARK: Telemetry
        public struct Telemetry {
            public let kernelVariant: String
            public let rowsProcessed: Int
            public let bytesRead: Int
            public let usedFusedPath: Bool
            public let usedF16Norms: Bool
            public let zeroNormCount: Int
            public let clampedCount: Int
            public let executionTimeNanos: UInt64
            public var bandwidthGBps: Double {
                let s = Double(executionTimeNanos) / 1e9
                return s > 0 ? (Double(bytesRead) / 1e9) / s : 0
            }
        }
        public enum TelemetryRecorder {
            nonisolated(unsafe) public static var sink: ((Telemetry) -> Void)?
            @inline(__always) public static func record(_ t: Telemetry) { sink?(t) }
        }

        // MARK: Public API (f32 inv-norms)
        public static func run(
            q: UnsafePointer<Float>,
            xb: UnsafePointer<Float>,
            n: Int,
            d: Int,
            out: UnsafeMutablePointer<Float>,
            dbInvNorms: UnsafePointer<Float>? = nil,
            queryInvNorm: Float? = nil,
            options: Options = .init()
        ) {
            if n == 0 { return }
            if d == 0 { for i in 0..<n { out[i] = 0 }; return }

            if options.verifyAlignment {
                verifyAlignment(q, 64, "q")
                verifyAlignment(xb, 64, "xb")
                verifyAlignment(out, 64, "out")
                if let norms = dbInvNorms { verifyAlignment(norms, 64, "dbInvNorms") }
            }

            let t0 = options.enableTelemetry ? DispatchTime.now().uptimeNanoseconds : 0
            let bytesRead = n * d * MemoryLayout<Float>.stride
                            + d * MemoryLayout<Float>.stride
                            + (dbInvNorms != nil ? n * MemoryLayout<Float>.stride : 0)

            let qInv = queryInvNorm ?? computeQueryInvNorm_impl(q: q, d: d, epsilon: options.epsilon)

            var clampedCount = 0
            var zeroNormCount = 0

            if let norms = dbInvNorms, options.useFusedKernel {
                // Fused path
                switch d {
                case 512:  fused_fixedD(q, xb, n, 512, norms, qInv, out, options.clampOutput, &clampedCount, 1.0 / max(options.epsilon, 1e-38), &zeroNormCount)
                case 768:  fused_fixedD(q, xb, n, 768, norms, qInv, out, options.clampOutput, &clampedCount, 1.0 / max(options.epsilon, 1e-38), &zeroNormCount)
                case 1024: fused_fixedD(q, xb, n, 1024, norms, qInv, out, options.clampOutput, &clampedCount, 1.0 / max(options.epsilon, 1e-38), &zeroNormCount)
                case 1536: fused_fixedD(q, xb, n, 1536, norms, qInv, out, options.clampOutput, &clampedCount, 1.0 / max(options.epsilon, 1e-38), &zeroNormCount)
                default:   fused_generic(q: q, xb: xb, n: n, d: d, norms: norms, qInv: qInv, out: out, clamp: options.clampOutput, clampedCount: &clampedCount, zeroNormSentinel: 1.0 / max(options.epsilon, 1e-38), zeroCount: &zeroNormCount)
                }
                if options.enableTelemetry {
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    TelemetryRecorder.record(Telemetry(
                        kernelVariant: (d == 512 || d == 768 || d == 1024 || d == 1536) ? "d\(d)_fused" : "generic_fused",
                        rowsProcessed: n,
                        bytesRead: bytesRead,
                        usedFusedPath: true,
                        usedF16Norms: false,
                        zeroNormCount: zeroNormCount,
                        clampedCount: clampedCount,
                        executionTimeNanos: t1 &- t0
                    ))
                }
                return
            }

            // Two-pass: reuse inner product kernel, then scale per-row
            IndexOps.Scoring.InnerProduct.run(q: q, xb: xb, n: n, d: d, out: out)
            if let norms = dbInvNorms {
                if options.clampOutput {
                    for i in 0..<n {
                        let v = out[i] * qInv * norms[i]
                        let c = clampUnit(v)
                        if c != v { clampedCount += 1 }
                        if norms[i] >= 1.0 / max(options.epsilon, 1e-38) { zeroNormCount += 1 }
                        out[i] = c
                    }
                } else {
                    for i in 0..<n {
                        if norms[i] >= 1.0 / max(options.epsilon, 1e-38) { zeroNormCount += 1 }
                        out[i] = out[i] * qInv * norms[i]
                    }
                }
            } else {
                // Compute inv-norms on the fly per row (one extra read over xb)
                for i in 0..<n {
                    let row = xb.advanced(by: i * d)
                    let inv = 1.0 / (sqrt(sumSquares(ptr: row, d: d)) + options.epsilon)
                    if inv >= 1.0 / max(options.epsilon, 1e-38) { zeroNormCount += 1 }
                    let v = out[i] * qInv * inv
                    let c = options.clampOutput ? clampUnit(v) : v
                    if options.clampOutput && c != v { clampedCount += 1 }
                    out[i] = c
                }
            }

            if options.enableTelemetry {
                let t1 = DispatchTime.now().uptimeNanoseconds
                TelemetryRecorder.record(Telemetry(
                    kernelVariant: (dbInvNorms != nil ? ((d == 512 || d == 768 || d == 1024 || d == 1536) ? "d\(d)_2pass" : "generic_2pass") : "generic_2pass"),
                    rowsProcessed: n,
                    bytesRead: bytesRead,
                    usedFusedPath: false,
                    usedF16Norms: false,
                    zeroNormCount: zeroNormCount,
                    clampedCount: clampedCount,
                    executionTimeNanos: t1 &- t0
                ))
            }
        }

        // MARK: f16 inv-norms path (generic fused)
        public static func runF16(
            q: UnsafePointer<Float>,
            xb: UnsafePointer<Float>,
            n: Int,
            d: Int,
            out: UnsafeMutablePointer<Float>,
            dbInvNormsF16: UnsafePointer<Float16>,
            queryInvNorm: Float,
            options: Options = .init()
        ) {
            if n == 0 { return }
            if d == 0 { for i in 0..<n { out[i] = 0 }; return }
            let t0 = options.enableTelemetry ? DispatchTime.now().uptimeNanoseconds : 0
            var clamped = 0, zeros = 0
            fused_generic_f16(q: q, xb: xb, n: n, d: d, normsF16: dbInvNormsF16, qInv: queryInvNorm, out: out, clamp: options.clampOutput, clampedCount: &clamped, zeroCount: &zeros)
            if options.enableTelemetry {
                let t1 = DispatchTime.now().uptimeNanoseconds
                let bytes = n * d * MemoryLayout<Float>.stride + d * MemoryLayout<Float>.stride + n * MemoryLayout<Float16>.stride
                TelemetryRecorder.record(Telemetry(
                    kernelVariant: "generic_fused_f16",
                    rowsProcessed: n,
                    bytesRead: bytes,
                    usedFusedPath: true,
                    usedF16Norms: true,
                    zeroNormCount: zeros,
                    clampedCount: clamped,
                    executionTimeNanos: t1 &- t0
                ))
            }
        }

        // MARK: Helpers
        @inline(__always) private static func hsum4(_ v: SIMD4<Float>) -> Float { v[0]+v[1]+v[2]+v[3] }
        @inline(__always) private static func load4(_ p: UnsafePointer<Float>) -> SIMD4<Float> {
            if Int(bitPattern: p) & (MemoryLayout<SIMD4<Float>>.alignment - 1) == 0 {
                return p.withMemoryRebound(to: SIMD4<Float>.self, capacity: 1) { $0.pointee }
            } else { return SIMD4<Float>(p[0], p[1], p[2], p[3]) }
        }
        @inline(__always) private static func clampUnit(_ x: Float) -> Float { min(1, max(-1, x)) }
        @inline(__always) private static func verifyAlignment<T>(_ ptr: UnsafePointer<T>, _ alignment: Int, _ label: String) {
            #if DEBUG
            assert(Int(bitPattern: ptr) % alignment == 0, "\(label) must be \(alignment)-byte aligned")
            #endif
        }
        @inline(__always) private static func sumSquares(ptr: UnsafePointer<Float>, d: Int) -> Float {
            let d4 = d & ~3
            var acc = SIMD4<Float>.zero
            var j = 0
            while j < d4 { let v = load4(ptr.advanced(by: j)); acc += v * v; j += 4 }
            var s = hsum4(acc)
            if d - d4 >= 1 { s += ptr[d4+0] * ptr[d4+0] }
            if d - d4 >= 2 { s += ptr[d4+1] * ptr[d4+1] }
            if d - d4 >= 3 { s += ptr[d4+2] * ptr[d4+2] }
            return s
        }
        @inline(__always) private static func computeQueryInvNorm_impl(q: UnsafePointer<Float>, d: Int, epsilon: Float) -> Float {
            if d == 0 { return 1.0 / epsilon }
            let s = sumSquares(ptr: q, d: d)
            return 1.0 / (sqrt(s) + epsilon)
        }

        // MARK: Fused generic
        private static func fused_generic(
            q: UnsafePointer<Float>, xb: UnsafePointer<Float>, n: Int, d: Int,
            norms: UnsafePointer<Float>, qInv: Float, out: UnsafeMutablePointer<Float>,
            clamp: Bool, clampedCount: inout Int, zeroNormSentinel: Float, zeroCount: inout Int
        ) {
            let D = d, D4 = D & ~3
            var i = 0
            while i + 7 < n {
                var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
                var a4 = SIMD4<Float>.zero, a5 = SIMD4<Float>.zero, a6 = SIMD4<Float>.zero, a7 = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0))
                    let q1 = load4(q.advanced(by: j + 4))
                    let q2 = load4(q.advanced(by: j + 8))
                    let q3 = load4(q.advanced(by: j + 12))
                    let r0 = xb.advanced(by: (i+0) * D + j)
                    let r1 = xb.advanced(by: (i+1) * D + j)
                    let r2 = xb.advanced(by: (i+2) * D + j)
                    let r3 = xb.advanced(by: (i+3) * D + j)
                    let r4 = xb.advanced(by: (i+4) * D + j)
                    let r5 = xb.advanced(by: (i+5) * D + j)
                    let r6 = xb.advanced(by: (i+6) * D + j)
                    let r7 = xb.advanced(by: (i+7) * D + j)
                    a0 += q0 * load4(r0.advanced(by: 0));  a0 += q1 * load4(r0.advanced(by: 4));  a0 += q2 * load4(r0.advanced(by: 8));  a0 += q3 * load4(r0.advanced(by: 12))
                    a1 += q0 * load4(r1.advanced(by: 0));  a1 += q1 * load4(r1.advanced(by: 4));  a1 += q2 * load4(r1.advanced(by: 8));  a1 += q3 * load4(r1.advanced(by: 12))
                    a2 += q0 * load4(r2.advanced(by: 0));  a2 += q1 * load4(r2.advanced(by: 4));  a2 += q2 * load4(r2.advanced(by: 8));  a2 += q3 * load4(r2.advanced(by: 12))
                    a3 += q0 * load4(r3.advanced(by: 0));  a3 += q1 * load4(r3.advanced(by: 4));  a3 += q2 * load4(r3.advanced(by: 8));  a3 += q3 * load4(r3.advanced(by: 12))
                    a4 += q0 * load4(r4.advanced(by: 0));  a4 += q1 * load4(r4.advanced(by: 4));  a4 += q2 * load4(r4.advanced(by: 8));  a4 += q3 * load4(r4.advanced(by: 12))
                    a5 += q0 * load4(r5.advanced(by: 0));  a5 += q1 * load4(r5.advanced(by: 4));  a5 += q2 * load4(r5.advanced(by: 8));  a5 += q3 * load4(r5.advanced(by: 12))
                    a6 += q0 * load4(r6.advanced(by: 0));  a6 += q1 * load4(r6.advanced(by: 4));  a6 += q2 * load4(r6.advanced(by: 8));  a6 += q3 * load4(r6.advanced(by: 12))
                    a7 += q0 * load4(r7.advanced(by: 0));  a7 += q1 * load4(r7.advanced(by: 4));  a7 += q2 * load4(r7.advanced(by: 8));  a7 += q3 * load4(r7.advanced(by: 12))
                    j += 16
                }
                while j < D4 {
                    let qv = load4(q.advanced(by: j))
                    a0 += qv * load4(xb.advanced(by: (i+0) * D + j))
                    a1 += qv * load4(xb.advanced(by: (i+1) * D + j))
                    a2 += qv * load4(xb.advanced(by: (i+2) * D + j))
                    a3 += qv * load4(xb.advanced(by: (i+3) * D + j))
                    a4 += qv * load4(xb.advanced(by: (i+4) * D + j))
                    a5 += qv * load4(xb.advanced(by: (i+5) * D + j))
                    a6 += qv * load4(xb.advanced(by: (i+6) * D + j))
                    a7 += qv * load4(xb.advanced(by: (i+7) * D + j))
                    j += 4
                }
                var s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3)
                var s4 = hsum4(a4), s5 = hsum4(a5), s6 = hsum4(a6), s7 = hsum4(a7)
                while j < D { let qj = q[j]; s0 += qj * xb[(i+0)*D + j]; s1 += qj * xb[(i+1)*D + j]; s2 += qj * xb[(i+2)*D + j]; s3 += qj * xb[(i+3)*D + j]; s4 += qj * xb[(i+4)*D + j]; s5 += qj * xb[(i+5)*D + j]; s6 += qj * xb[(i+6)*D + j]; s7 += qj * xb[(i+7)*D + j]; j += 1 }
                let qS = qInv
                let n0 = norms[i+0], n1 = norms[i+1], n2 = norms[i+2], n3 = norms[i+3]
                let n4 = norms[i+4], n5 = norms[i+5], n6 = norms[i+6], n7 = norms[i+7]
                out[i+0] = optionsClamp(s0 * qS * n0, clamp: clamp, clampedCount: &clampedCount)
                out[i+1] = optionsClamp(s1 * qS * n1, clamp: clamp, clampedCount: &clampedCount)
                out[i+2] = optionsClamp(s2 * qS * n2, clamp: clamp, clampedCount: &clampedCount)
                out[i+3] = optionsClamp(s3 * qS * n3, clamp: clamp, clampedCount: &clampedCount)
                out[i+4] = optionsClamp(s4 * qS * n4, clamp: clamp, clampedCount: &clampedCount)
                out[i+5] = optionsClamp(s5 * qS * n5, clamp: clamp, clampedCount: &clampedCount)
                out[i+6] = optionsClamp(s6 * qS * n6, clamp: clamp, clampedCount: &clampedCount)
                out[i+7] = optionsClamp(s7 * qS * n7, clamp: clamp, clampedCount: &clampedCount)
                zeroNormCountAdd(&zeroCount, n0, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n1, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n2, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n3, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n4, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n5, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n6, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n7, zeroNormSentinel)
                i += 8
            }
            while i + 3 < n {
                var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0)); let q1 = load4(q.advanced(by: j + 4)); let q2 = load4(q.advanced(by: j + 8)); let q3 = load4(q.advanced(by: j + 12))
                    let r0 = xb.advanced(by: (i+0) * D + j); let r1 = xb.advanced(by: (i+1) * D + j); let r2 = xb.advanced(by: (i+2) * D + j); let r3 = xb.advanced(by: (i+3) * D + j)
                    a0 += q0 * load4(r0.advanced(by: 0)); a0 += q1 * load4(r0.advanced(by: 4)); a0 += q2 * load4(r0.advanced(by: 8)); a0 += q3 * load4(r0.advanced(by: 12))
                    a1 += q0 * load4(r1.advanced(by: 0)); a1 += q1 * load4(r1.advanced(by: 4)); a1 += q2 * load4(r1.advanced(by: 8)); a1 += q3 * load4(r1.advanced(by: 12))
                    a2 += q0 * load4(r2.advanced(by: 0)); a2 += q1 * load4(r2.advanced(by: 4)); a2 += q2 * load4(r2.advanced(by: 8)); a2 += q3 * load4(r2.advanced(by: 12))
                    a3 += q0 * load4(r3.advanced(by: 0)); a3 += q1 * load4(r3.advanced(by: 4)); a3 += q2 * load4(r3.advanced(by: 8)); a3 += q3 * load4(r3.advanced(by: 12))
                    j += 16
                }
                while j < D4 {
                    let qv = load4(q.advanced(by: j))
                    a0 += qv * load4(xb.advanced(by: (i+0) * D + j))
                    a1 += qv * load4(xb.advanced(by: (i+1) * D + j))
                    a2 += qv * load4(xb.advanced(by: (i+2) * D + j))
                    a3 += qv * load4(xb.advanced(by: (i+3) * D + j))
                    j += 4
                }
                var s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3)
                while j < D { let qj = q[j]; s0 += qj * xb[(i+0)*D + j]; s1 += qj * xb[(i+1)*D + j]; s2 += qj * xb[(i+2)*D + j]; s3 += qj * xb[(i+3)*D + j]; j += 1 }
                let qS = qInv
                let n0 = norms[i+0], n1 = norms[i+1], n2 = norms[i+2], n3 = norms[i+3]
                out[i+0] = optionsClamp(s0 * qS * n0, clamp: clamp, clampedCount: &clampedCount)
                out[i+1] = optionsClamp(s1 * qS * n1, clamp: clamp, clampedCount: &clampedCount)
                out[i+2] = optionsClamp(s2 * qS * n2, clamp: clamp, clampedCount: &clampedCount)
                out[i+3] = optionsClamp(s3 * qS * n3, clamp: clamp, clampedCount: &clampedCount)
                zeroNormCountAdd(&zeroCount, n0, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n1, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n2, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n3, zeroNormSentinel)
                i += 4
            }
            while i < n {
                var acc = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0)); let q1 = load4(q.advanced(by: j + 4)); let q2 = load4(q.advanced(by: j + 8)); let q3 = load4(q.advanced(by: j + 12))
                    let r = xb.advanced(by: i * D + j)
                    acc += q0 * load4(r.advanced(by: 0)); acc += q1 * load4(r.advanced(by: 4)); acc += q2 * load4(r.advanced(by: 8)); acc += q3 * load4(r.advanced(by: 12))
                    j += 16
                }
                while j < D4 { let qv = load4(q.advanced(by: j)); acc += qv * load4(xb.advanced(by: i * D + j)); j += 4 }
                var s = hsum4(acc); while j < D { let qj = q[j]; s += qj * xb[i*D + j]; j += 1 }
                let nrm = norms[i]
                out[i] = optionsClamp(s * qInv * nrm, clamp: clamp, clampedCount: &clampedCount)
                zeroNormCountAdd(&zeroCount, nrm, zeroNormSentinel)
                i += 1
            }
        }

        private static func fused_generic_f16(
            q: UnsafePointer<Float>, xb: UnsafePointer<Float>, n: Int, d: Int,
            normsF16: UnsafePointer<Float16>, qInv: Float, out: UnsafeMutablePointer<Float>,
            clamp: Bool, clampedCount: inout Int, zeroCount: inout Int
        ) {
            let D = d, D4 = D & ~3
            var i = 0
            while i < n {
                var acc = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0)); let q1 = load4(q.advanced(by: j + 4)); let q2 = load4(q.advanced(by: j + 8)); let q3 = load4(q.advanced(by: j + 12))
                    let r = xb.advanced(by: i * D + j)
                    acc += q0 * load4(r.advanced(by: 0)); acc += q1 * load4(r.advanced(by: 4)); acc += q2 * load4(r.advanced(by: 8)); acc += q3 * load4(r.advanced(by: 12))
                    j += 16
                }
                while j < D4 { let qv = load4(q.advanced(by: j)); acc += qv * load4(xb.advanced(by: i * D + j)); j += 4 }
                var s = hsum4(acc); while j < D { let qj = q[j]; s += qj * xb[i*D + j]; j += 1 }
                let inv = Float(normsF16[i])
                if inv >= Float(Float16.greatestFiniteMagnitude) || inv.isInfinite { zeroCount += 1 }
                out[i] = optionsClamp(s * qInv * inv, clamp: clamp, clampedCount: &clampedCount)
                i += 1
            }
        }

        // MARK: Fixed-D shared fused body
        @inline(__always)
        private static func fused_fixedD(
            _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ n: Int, _ D: Int,
            _ norms: UnsafePointer<Float>, _ qInv: Float, _ out: UnsafeMutablePointer<Float>,
            _ clamp: Bool, _ clampedCount: inout Int, _ zeroNormSentinel: Float, _ zeroCount: inout Int
        ) {
            var i = 0
            while i + 7 < n {
                var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
                var a4 = SIMD4<Float>.zero, a5 = SIMD4<Float>.zero, a6 = SIMD4<Float>.zero, a7 = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0)); let q1 = load4(q.advanced(by: j + 4)); let q2 = load4(q.advanced(by: j + 8)); let q3 = load4(q.advanced(by: j + 12))
                    let r0 = xb.advanced(by: (i+0) * D + j); let r1 = xb.advanced(by: (i+1) * D + j); let r2 = xb.advanced(by: (i+2) * D + j); let r3 = xb.advanced(by: (i+3) * D + j)
                    let r4 = xb.advanced(by: (i+4) * D + j); let r5 = xb.advanced(by: (i+5) * D + j); let r6 = xb.advanced(by: (i+6) * D + j); let r7 = xb.advanced(by: (i+7) * D + j)
                    a0 += q0 * load4(r0.advanced(by: 0)); a0 += q1 * load4(r0.advanced(by: 4)); a0 += q2 * load4(r0.advanced(by: 8)); a0 += q3 * load4(r0.advanced(by: 12))
                    a1 += q0 * load4(r1.advanced(by: 0)); a1 += q1 * load4(r1.advanced(by: 4)); a1 += q2 * load4(r1.advanced(by: 8)); a1 += q3 * load4(r1.advanced(by: 12))
                    a2 += q0 * load4(r2.advanced(by: 0)); a2 += q1 * load4(r2.advanced(by: 4)); a2 += q2 * load4(r2.advanced(by: 8)); a2 += q3 * load4(r2.advanced(by: 12))
                    a3 += q0 * load4(r3.advanced(by: 0)); a3 += q1 * load4(r3.advanced(by: 4)); a3 += q2 * load4(r3.advanced(by: 8)); a3 += q3 * load4(r3.advanced(by: 12))
                    a4 += q0 * load4(r4.advanced(by: 0)); a4 += q1 * load4(r4.advanced(by: 4)); a4 += q2 * load4(r4.advanced(by: 8)); a4 += q3 * load4(r4.advanced(by: 12))
                    a5 += q0 * load4(r5.advanced(by: 0)); a5 += q1 * load4(r5.advanced(by: 4)); a5 += q2 * load4(r5.advanced(by: 8)); a5 += q3 * load4(r5.advanced(by: 12))
                    a6 += q0 * load4(r6.advanced(by: 0)); a6 += q1 * load4(r6.advanced(by: 4)); a6 += q2 * load4(r6.advanced(by: 8)); a6 += q3 * load4(r6.advanced(by: 12))
                    a7 += q0 * load4(r7.advanced(by: 0)); a7 += q1 * load4(r7.advanced(by: 4)); a7 += q2 * load4(r7.advanced(by: 8)); a7 += q3 * load4(r7.advanced(by: 12))
                    j += 16
                }
                while j + 3 < D {
                    let qv = load4(q.advanced(by: j))
                    a0 += qv * load4(xb.advanced(by: (i+0) * D + j))
                    a1 += qv * load4(xb.advanced(by: (i+1) * D + j))
                    a2 += qv * load4(xb.advanced(by: (i+2) * D + j))
                    a3 += qv * load4(xb.advanced(by: (i+3) * D + j))
                    a4 += qv * load4(xb.advanced(by: (i+4) * D + j))
                    a5 += qv * load4(xb.advanced(by: (i+5) * D + j))
                    a6 += qv * load4(xb.advanced(by: (i+6) * D + j))
                    a7 += qv * load4(xb.advanced(by: (i+7) * D + j))
                    j += 4
                }
                var s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3)
                var s4 = hsum4(a4), s5 = hsum4(a5), s6 = hsum4(a6), s7 = hsum4(a7)
                while j < D { let qj = q[j]; s0 += qj * xb[(i+0)*D + j]; s1 += qj * xb[(i+1)*D + j]; s2 += qj * xb[(i+2)*D + j]; s3 += qj * xb[(i+3)*D + j]; s4 += qj * xb[(i+4)*D + j]; s5 += qj * xb[(i+5)*D + j]; s6 += qj * xb[(i+6)*D + j]; s7 += qj * xb[(i+7)*D + j]; j += 1 }
                let qS = qInv
                let n0 = norms[i+0], n1 = norms[i+1], n2 = norms[i+2], n3 = norms[i+3]
                let n4 = norms[i+4], n5 = norms[i+5], n6 = norms[i+6], n7 = norms[i+7]
                out[i+0] = optionsClamp(s0 * qS * n0, clamp: clamp, clampedCount: &clampedCount)
                out[i+1] = optionsClamp(s1 * qS * n1, clamp: clamp, clampedCount: &clampedCount)
                out[i+2] = optionsClamp(s2 * qS * n2, clamp: clamp, clampedCount: &clampedCount)
                out[i+3] = optionsClamp(s3 * qS * n3, clamp: clamp, clampedCount: &clampedCount)
                out[i+4] = optionsClamp(s4 * qS * n4, clamp: clamp, clampedCount: &clampedCount)
                out[i+5] = optionsClamp(s5 * qS * n5, clamp: clamp, clampedCount: &clampedCount)
                out[i+6] = optionsClamp(s6 * qS * n6, clamp: clamp, clampedCount: &clampedCount)
                out[i+7] = optionsClamp(s7 * qS * n7, clamp: clamp, clampedCount: &clampedCount)
                zeroNormCountAdd(&zeroCount, n0, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n1, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n2, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n3, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n4, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n5, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n6, zeroNormSentinel); zeroNormCountAdd(&zeroCount, n7, zeroNormSentinel)
                i += 8
            }
            while i < n {
                var acc = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0)); let q1 = load4(q.advanced(by: j + 4)); let q2 = load4(q.advanced(by: j + 8)); let q3 = load4(q.advanced(by: j + 12))
                    let r = xb.advanced(by: i * D + j)
                    acc += q0 * load4(r.advanced(by: 0)); acc += q1 * load4(r.advanced(by: 4)); acc += q2 * load4(r.advanced(by: 8)); acc += q3 * load4(r.advanced(by: 12))
                    j += 16
                }
                while j + 3 < D { let qv = load4(q.advanced(by: j)); acc += qv * load4(xb.advanced(by: i * D + j)); j += 4 }
                var s = hsum4(acc); while j < D { let qj = q[j]; s += qj * xb[i*D + j]; j += 1 }
                let nrm = norms[i]
                out[i] = optionsClamp(s * qInv * nrm, clamp: clamp, clampedCount: &clampedCount)
                zeroNormCountAdd(&zeroCount, nrm, zeroNormSentinel)
                i += 1
            }
        }

        @inline(__always) private static func optionsClamp(_ v: Float, clamp: Bool, clampedCount: inout Int) -> Float {
            if clamp {
                let c = clampUnit(v)
                if c != v { clampedCount += 1 }
                return c
            } else { return v }
        }

        @inline(__always) private static func zeroNormCountAdd(_ count: inout Int, _ inv: Float, _ sentinel: Float) {
            if inv >= sentinel { count += 1 }
        }

        // MARK: Norm precompute helpers (optional external use)
        public static func computeQueryInvNorm(q: UnsafePointer<Float>, d: Int, epsilon: Float = 1e-12) -> Float {
            computeQueryInvNorm_impl(q: q, d: d, epsilon: epsilon)
        }
        public static func precomputeInvNorms(
            xb: UnsafePointer<Float>, n: Int, d: Int, out: UnsafeMutablePointer<Float>, epsilon: Float = 1e-12
        ) {
            if n == 0 { return }
            if d == 0 { for i in 0..<n { out[i] = 1.0 / epsilon }; return }
            let d4 = d & ~3
            for i in 0..<n {
                let row = xb.advanced(by: i * d)
                var acc = SIMD4<Float>.zero
                var j = 0
                while j < d4 { let v = load4(row.advanced(by: j)); acc += v * v; j += 4 }
                var s = hsum4(acc); while j < d { let v = row[j]; s += v * v; j += 1 }
                out[i] = 1.0 / (sqrt(s) + epsilon)
            }
        }
        public static func precomputeInvNormsF16(
            xb: UnsafePointer<Float>, n: Int, d: Int, out: UnsafeMutablePointer<Float16>, epsilon: Float = 1e-12
        ) {
            if n == 0 { return }
            if d == 0 { for i in 0..<n { out[i] = Float16(1.0 / epsilon) }; return }
            let d4 = d & ~3
            for i in 0..<n {
                let row = xb.advanced(by: i * d)
                var acc = SIMD4<Float>.zero
                var j = 0
                while j < d4 { let v = load4(row.advanced(by: j)); acc += v * v; j += 4 }
                var s = hsum4(acc); while j < d { let v = row[j]; s += v * v; j += 1 }
                let inv = 1.0 / (sqrt(s) + epsilon)
                let clamp = max(-Float(Float16.greatestFiniteMagnitude), min(Float(Float16.greatestFiniteMagnitude), inv))
                out[i] = Float16(clamp)
            }
        }
    }
}
