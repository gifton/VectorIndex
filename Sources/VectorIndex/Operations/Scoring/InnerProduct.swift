import Foundation
import simd
import Dispatch

public extension IndexOps.Scoring {
    enum InnerProduct {
            // MARK: Public API (AoS by default)
            public static func run(
                q: UnsafePointer<Float>,
                xb: UnsafePointer<Float>,
                n: Int,
                d: Int,
                out: UnsafeMutablePointer<Float>,
                stride ldb: Int? = nil,
                options: Options = .init()
            ) {
                if n == 0 { return }
                if d == 0 { for i in 0..<n { out[i] = 0 }; return }

                let stride = ldb ?? d

                // Optional alignment checks
                if options.verifyAlignment {
                    verifyAlignment(UnsafeRawPointer(q), 64, "q")
                    verifyAlignment(UnsafeRawPointer(xb), 64, "xb")
                    verifyAlignment(UnsafeRawPointer(out), 64, "out")
                }

                // Telemetry timing
                let t0 = options.enableTelemetry ? DispatchTime.now().uptimeNanoseconds : 0
                var fastPath = false
                var variant = "generic"

                // Specialized AoS contiguous fast paths
                if !options.forceGeneric && stride == d {
                    switch d {
                    case 512, 768, 1024, 1536:
                        fastPath = true
                        variant = "d\(d)_r"
                        var i = 0
                        // r=8
                        while i + 7 < n {
                            let base = xb.advanced(by: i * d)
                            let dst = out.advanced(by: i)
                            switch d {
                            case 512:  ip_r8_D(q, base, dst, 512)
                            case 768:  ip_r8_D(q, base, dst, 768)
                            case 1024: ip_r8_D(q, base, dst, 1024)
                            default:   ip_r8_D(q, base, dst, 1536)
                            }
                            i += 8
                        }
                        // r=4
                        while i + 3 < n {
                            let base = xb.advanced(by: i * d)
                            let dst = out.advanced(by: i)
                            switch d {
                            case 512:  ip_r4_D(q, base, dst, 512)
                            case 768:  ip_r4_D(q, base, dst, 768)
                            case 1024: ip_r4_D(q, base, dst, 1024)
                            default:   ip_r4_D(q, base, dst, 1536)
                            }
                            i += 4
                        }
                        // r=1
                        while i < n {
                            let base = xb.advanced(by: i * d)
                            let dst = out.advanced(by: i)
                            switch d {
                            case 512:  ip_r1_D(q, base, dst, 512)
                            case 768:  ip_r1_D(q, base, dst, 768)
                            case 1024: ip_r1_D(q, base, dst, 1024)
                            default:   ip_r1_D(q, base, dst, 1536)
                            }
                            i += 1
                        }
                        variant += "8|4|1"
                    default:
                        break
                    }
                }

                if !fastPath {
                    // Generic path supports arbitrary stride
                    generic(q: q, xb: xb, n: n, d: d, out: out, stride: stride)
                }

                if options.enableTelemetry {
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    let bytes = (n * d + d) * MemoryLayout<Float>.stride
                    let t = Telemetry(
                        kernelVariant: fastPath ? variant : "generic",
                        rowsProcessed: n,
                        bytesRead: bytes,
                        fastPathHit: fastPath,
                        vectorWidth: 4,
                        executionTimeNanos: t1 &- t0
                    )
                    TelemetryRecorder.record(t)
                }
            }

            // C ABI wrapper lives at global scope in CABIBridge.swift

            // MARK: Options
            public struct Options {
                public var forceGeneric: Bool = false
                public var enableTelemetry: Bool = false
                public var verifyAlignment: Bool = false
                public init() {}
            }

            // MARK: Generic fallback (AoS with optional stride)
            @usableFromInline
            static func generic(
                q: UnsafePointer<Float>,
                xb: UnsafePointer<Float>,
                n: Int,
                d: Int,
                out: UnsafeMutablePointer<Float>,
                stride ldb: Int
            ) {
                let d4 = d & ~3
                let hasTail = (d4 != d)
                // Prepack q into SIMD4 once
                var qPack = [SIMD4<Float>](repeating: .zero, count: d4 >> 2)
                for s in 0..<(d4 >> 2) { qPack[s] = load4(q.advanced(by: s << 2)) }
                var i = 0
                while i < n {
                    let row = xb.advanced(by: i * ldb)
                    var acc0 = SIMD4<Float>.zero
                    // main
                    var s = 0
                    while s < qPack.count {
                        let qv = qPack[s]
                        let xv = load4(row.advanced(by: s << 2))
                        acc0 += qv * xv
                        s += 1
                    }
                    var sum = hsum4(acc0)
                    // tail
                    if hasTail {
                        if d - d4 >= 1 { sum += q[d4+0] * row[d4+0] }
                        if d - d4 >= 2 { sum += q[d4+1] * row[d4+1] }
                        if d - d4 >= 3 { sum += q[d4+2] * row[d4+2] }
                    }
                    out[i] = sum
                    i += 1
                }
            }

            // MARK: Specialized kernels (AoS contiguous), D in {512,768,1024,1536}
            @inline(__always) static func ip_r1_D(
                _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ D: Int
            ) {
                var a0 = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0))
                    let q1 = load4(q.advanced(by: j + 4))
                    let q2 = load4(q.advanced(by: j + 8))
                    let q3 = load4(q.advanced(by: j + 12))

                    let x0 = load4(xb.advanced(by: j + 0))
                    let x1 = load4(xb.advanced(by: j + 4))
                    let x2 = load4(xb.advanced(by: j + 8))
                    let x3 = load4(xb.advanced(by: j + 12))

                    a0 += q0 * x0
                    a0 += q1 * x1
                    a0 += q2 * x2
                    a0 += q3 * x3
                    j += 16
                }
                while j + 3 < D {
                    let qv = load4(q.advanced(by: j))
                    let xv = load4(xb.advanced(by: j))
                    a0 += qv * xv
                    j += 4
                }
                var s = hsum4(a0)
                while j < D { s += q[j] * xb[j]; j += 1 }
                out[0] = s
            }

            @inline(__always) static func ip_r4_D(
                _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ D: Int
            ) {
                var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0))
                    let q1 = load4(q.advanced(by: j + 4))
                    let q2 = load4(q.advanced(by: j + 8))
                    let q3 = load4(q.advanced(by: j + 12))

                    let x0_0 = load4(xb.advanced(by: 0*D + j + 0))
                    let x0_1 = load4(xb.advanced(by: 0*D + j + 4))
                    let x0_2 = load4(xb.advanced(by: 0*D + j + 8))
                    let x0_3 = load4(xb.advanced(by: 0*D + j + 12))
                    a0 += q0 * x0_0; a0 += q1 * x0_1; a0 += q2 * x0_2; a0 += q3 * x0_3

                    let x1_0 = load4(xb.advanced(by: 1*D + j + 0))
                    let x1_1 = load4(xb.advanced(by: 1*D + j + 4))
                    let x1_2 = load4(xb.advanced(by: 1*D + j + 8))
                    let x1_3 = load4(xb.advanced(by: 1*D + j + 12))
                    a1 += q0 * x1_0; a1 += q1 * x1_1; a1 += q2 * x1_2; a1 += q3 * x1_3

                    let x2_0 = load4(xb.advanced(by: 2*D + j + 0))
                    let x2_1 = load4(xb.advanced(by: 2*D + j + 4))
                    let x2_2 = load4(xb.advanced(by: 2*D + j + 8))
                    let x2_3 = load4(xb.advanced(by: 2*D + j + 12))
                    a2 += q0 * x2_0; a2 += q1 * x2_1; a2 += q2 * x2_2; a2 += q3 * x2_3

                    let x3_0 = load4(xb.advanced(by: 3*D + j + 0))
                    let x3_1 = load4(xb.advanced(by: 3*D + j + 4))
                    let x3_2 = load4(xb.advanced(by: 3*D + j + 8))
                    let x3_3 = load4(xb.advanced(by: 3*D + j + 12))
                    a3 += q0 * x3_0; a3 += q1 * x3_1; a3 += q2 * x3_2; a3 += q3 * x3_3

                    j += 16
                }
                while j + 3 < D {
                    let qv = load4(q.advanced(by: j))
                    a0 += qv * load4(xb.advanced(by: 0*D + j))
                    a1 += qv * load4(xb.advanced(by: 1*D + j))
                    a2 += qv * load4(xb.advanced(by: 2*D + j))
                    a3 += qv * load4(xb.advanced(by: 3*D + j))
                    j += 4
                }
                var s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3)
                while j < D {
                    let qj = q[j]
                    s0 += qj * xb[0*D + j]
                    s1 += qj * xb[1*D + j]
                    s2 += qj * xb[2*D + j]
                    s3 += qj * xb[3*D + j]
                    j += 1
                }
                out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3
            }

            @inline(__always) static func ip_r8_D(
                _ q: UnsafePointer<Float>, _ xb: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ D: Int
            ) {
                var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
                var a4 = SIMD4<Float>.zero, a5 = SIMD4<Float>.zero, a6 = SIMD4<Float>.zero, a7 = SIMD4<Float>.zero
                var j = 0
                while j + 15 < D {
                    let q0 = load4(q.advanced(by: j + 0))
                    let q1 = load4(q.advanced(by: j + 4))
                    let q2 = load4(q.advanced(by: j + 8))
                    let q3 = load4(q.advanced(by: j + 12))

                    let x0_0 = load4(xb.advanced(by: 0*D + j + 0))
                    let x0_1 = load4(xb.advanced(by: 0*D + j + 4))
                    let x0_2 = load4(xb.advanced(by: 0*D + j + 8))
                    let x0_3 = load4(xb.advanced(by: 0*D + j + 12))
                    a0 += q0 * x0_0; a0 += q1 * x0_1; a0 += q2 * x0_2; a0 += q3 * x0_3

                    let x1_0 = load4(xb.advanced(by: 1*D + j + 0))
                    let x1_1 = load4(xb.advanced(by: 1*D + j + 4))
                    let x1_2 = load4(xb.advanced(by: 1*D + j + 8))
                    let x1_3 = load4(xb.advanced(by: 1*D + j + 12))
                    a1 += q0 * x1_0; a1 += q1 * x1_1; a1 += q2 * x1_2; a1 += q3 * x1_3

                    let x2_0 = load4(xb.advanced(by: 2*D + j + 0))
                    let x2_1 = load4(xb.advanced(by: 2*D + j + 4))
                    let x2_2 = load4(xb.advanced(by: 2*D + j + 8))
                    let x2_3 = load4(xb.advanced(by: 2*D + j + 12))
                    a2 += q0 * x2_0; a2 += q1 * x2_1; a2 += q2 * x2_2; a2 += q3 * x2_3

                    let x3_0 = load4(xb.advanced(by: 3*D + j + 0))
                    let x3_1 = load4(xb.advanced(by: 3*D + j + 4))
                    let x3_2 = load4(xb.advanced(by: 3*D + j + 8))
                    let x3_3 = load4(xb.advanced(by: 3*D + j + 12))
                    a3 += q0 * x3_0; a3 += q1 * x3_1; a3 += q2 * x3_2; a3 += q3 * x3_3

                    let x4_0 = load4(xb.advanced(by: 4*D + j + 0))
                    let x4_1 = load4(xb.advanced(by: 4*D + j + 4))
                    let x4_2 = load4(xb.advanced(by: 4*D + j + 8))
                    let x4_3 = load4(xb.advanced(by: 4*D + j + 12))
                    a4 += q0 * x4_0; a4 += q1 * x4_1; a4 += q2 * x4_2; a4 += q3 * x4_3

                    let x5_0 = load4(xb.advanced(by: 5*D + j + 0))
                    let x5_1 = load4(xb.advanced(by: 5*D + j + 4))
                    let x5_2 = load4(xb.advanced(by: 5*D + j + 8))
                    let x5_3 = load4(xb.advanced(by: 5*D + j + 12))
                    a5 += q0 * x5_0; a5 += q1 * x5_1; a5 += q2 * x5_2; a5 += q3 * x5_3

                    let x6_0 = load4(xb.advanced(by: 6*D + j + 0))
                    let x6_1 = load4(xb.advanced(by: 6*D + j + 4))
                    let x6_2 = load4(xb.advanced(by: 6*D + j + 8))
                    let x6_3 = load4(xb.advanced(by: 6*D + j + 12))
                    a6 += q0 * x6_0; a6 += q1 * x6_1; a6 += q2 * x6_2; a6 += q3 * x6_3

                    let x7_0 = load4(xb.advanced(by: 7*D + j + 0))
                    let x7_1 = load4(xb.advanced(by: 7*D + j + 4))
                    let x7_2 = load4(xb.advanced(by: 7*D + j + 8))
                    let x7_3 = load4(xb.advanced(by: 7*D + j + 12))
                    a7 += q0 * x7_0; a7 += q1 * x7_1; a7 += q2 * x7_2; a7 += q3 * x7_3

                    j += 16
                }
                while j + 3 < D {
                    let qv = load4(q.advanced(by: j))
                    a0 += qv * load4(xb.advanced(by: 0*D + j))
                    a1 += qv * load4(xb.advanced(by: 1*D + j))
                    a2 += qv * load4(xb.advanced(by: 2*D + j))
                    a3 += qv * load4(xb.advanced(by: 3*D + j))
                    a4 += qv * load4(xb.advanced(by: 4*D + j))
                    a5 += qv * load4(xb.advanced(by: 5*D + j))
                    a6 += qv * load4(xb.advanced(by: 6*D + j))
                    a7 += qv * load4(xb.advanced(by: 7*D + j))
                    j += 4
                }
                var s0 = hsum4(a0), s1 = hsum4(a1), s2 = hsum4(a2), s3 = hsum4(a3)
                var s4 = hsum4(a4), s5 = hsum4(a5), s6 = hsum4(a6), s7 = hsum4(a7)
                while j < D {
                    let qj = q[j]
                    s0 += qj * xb[0*D + j]
                    s1 += qj * xb[1*D + j]
                    s2 += qj * xb[2*D + j]
                    s3 += qj * xb[3*D + j]
                    s4 += qj * xb[4*D + j]
                    s5 += qj * xb[5*D + j]
                    s6 += qj * xb[6*D + j]
                    s7 += qj * xb[7*D + j]
                    j += 1
                }
                out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3
                out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7
            }

            // MARK: Telemetry
            public struct Telemetry {
                public let kernelVariant: String
                public let rowsProcessed: Int
                public let bytesRead: Int
                public let fastPathHit: Bool
                public let vectorWidth: Int
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

            // MARK: SIMD helpers & low-level utilities
            @inline(__always) static func hsum4(_ v: SIMD4<Float>) -> Float { v[0]+v[1]+v[2]+v[3] }
            @inline(__always) static func load4(_ p: UnsafePointer<Float>) -> SIMD4<Float> {
                if Int(bitPattern: p) & (MemoryLayout<SIMD4<Float>>.alignment - 1) == 0 {
                    return p.withMemoryRebound(to: SIMD4<Float>.self, capacity: 1) { $0.pointee }
                } else {
                    return SIMD4<Float>(p[0], p[1], p[2], p[3])
                }
            }
            @inline(__always) static func verifyAlignment(_ ptr: UnsafeRawPointer, _ alignment: Int, _ label: String) {
                #if DEBUG
                assert(Int(bitPattern: ptr) % alignment == 0, "\(label) must be \(alignment)-byte aligned")
                #endif
            }
    }
}
