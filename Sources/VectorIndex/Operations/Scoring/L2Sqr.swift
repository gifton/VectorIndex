import Foundation

// MARK: - Operations namespace (mirrors VectorCore style)

public extension IndexOps.Scoring {
    enum L2Sqr {
            // MARK: Public API
            /// Swift-friendly entrypoint matching the spec semantics. AoS-only layout.
            public static func run(
                q: UnsafePointer<Float>,
                xb: UnsafePointer<Float>,
                n: Int,
                d: Int,
                out: UnsafeMutablePointer<Float>,
                xb_norm: UnsafePointer<Float>? = nil,
                q_norm: Float? = nil
            ) {
                precondition(n >= 0 && d >= 0, "n and d must be non-negative")
                if n == 0 { return }
                if d == 0 { for i in 0..<n { out[i] = 0 } ; return }

                // Delegate to the microkernel implementation
                let qn = q_norm ?? .nan
                l2sqr_f32_block(q, xb, n, d, out, xb_norm, qn, nil)

                // Minimal compatibility telemetry: we no longer distinguish dot path here.
                let bytes = UInt64(d * MemoryLayout<Float>.stride
                                   + n * d * MemoryLayout<Float>.stride
                                   + (xb_norm != nil ? n * MemoryLayout<Float>.stride : 0))
                Telemetry.record(rows: n, d: d, usedDot: false, bytes: bytes)
            }

            // C ABI wrapper lives at global scope in CABIBridge.swift

            // MARK: Telemetry (unchanged API for callers in this module)
            public struct Telemetry {
                public nonisolated(unsafe) static var enabled: Bool = false
                public nonisolated(unsafe) private(set) static var rowsProcessed: UInt64 = 0
                public nonisolated(unsafe) private(set) static var bytesRead: UInt64 = 0
                public nonisolated(unsafe) private(set) static var usedDotFastPath: UInt64 = 0
                public nonisolated(unsafe) private(set) static var lastD: Int = 0

                @inline(__always)
                public static func record(rows: Int, d: Int, usedDot: Bool, bytes: UInt64) {
                    guard enabled else { return }
                    rowsProcessed &+= UInt64(rows)
                    bytesRead &+= bytes
                    if usedDot { usedDotFastPath &+= 1 }
                    lastD = d
                }

                public static func reset() {
                    rowsProcessed = 0
                    bytesRead = 0
                    usedDotFastPath = 0
                    lastD = 0
                }
            }

            // MARK: - Scalar reference (testing)
            public static func runScalarRef(
                q: UnsafePointer<Float>, xb: UnsafePointer<Float>, n: Int, d: Int, out: UnsafeMutablePointer<Float>
            ) {
                for i in 0..<n {
                    var s: Float = 0
                    let row = xb.advanced(by: i * d)
                    for j in 0..<d {
                        let diff = q[j] - row[j]
                        s += diff * diff
                    }
                    out[i] = s
                }
            }
    }
}
