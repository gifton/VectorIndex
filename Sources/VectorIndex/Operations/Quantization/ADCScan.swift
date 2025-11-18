// Sources/VectorIndex/Operations/Quantization/ADCScan.swift
//
// Kernel #22: ADC (Asymmetric Distance Computation) Scan
// - Computes approximate distances using precomputed LUTs (from Kernel #21)
// - Supports u8 (ks=256) and u4 (ks=16) quantized codes (from Kernel #20)
// - AoS and Interleaved layouts for SIMD/cache optimization
// - Parallel scan with thread pool
//
// Spec reference: Kernel Specification #22 (ADC Scan)
// Dependencies: Kernel #20 (PQ Encode), Kernel #21 (PQ LUT)

import Foundation

// MARK: - Public API types

/// Code memory layout for ADC scanning (Kernel #22)
public enum ADCLayout: Int {
    case aos = 0                    // [n][m] row-major codes
    case interleavedBlock = 1       // [n/g][m][g] interleaved groups
}

/// Options controlling ADC scan behavior.
public struct ADCScanOpts {
    public var layout: ADCLayout                 // code layout
    public var groupSize: Int                    // interleaved group size (g)
    public var stride: Int                       // stride for padded AoS (0 = tight; uses m)
    public var addBias: Float                    // query norm bias
    public var strictFP: Bool                    // Kahan for m >= 64
    public var prefetchDistance: Int             // lookahead in vectors (AoS) or blocks
    public var numThreads: Int                   // 0 = auto (use all cores)

    public init(
        layout: ADCLayout = .aos,
        groupSize: Int = 0,
        stride: Int = 0,
        addBias: Float = 0,
        strictFP: Bool = false,
        prefetchDistance: Int = 8,
        numThreads: Int = 0
    ) {
        self.layout = layout
        self.groupSize = groupSize
        self.stride = stride
        self.addBias = addBias
        self.strictFP = strictFP
        self.prefetchDistance = prefetchDistance
        self.numThreads = numThreads
    }
}

// MARK: - Public API (array conveniences)

@inline(__always)
public func adc_scan_u8(
    codes: [UInt8],
    n: Int,
    m: Int,
    ks: Int,
    lut: [Float],
    out: inout [Float],
    opts: ADCScanOpts? = nil
) {
    out.withUnsafeMutableBufferPointer { outBuf in
        codes.withUnsafeBufferPointer { c in
            lut.withUnsafeBufferPointer { l in
                adc_scan_u8(
                    codes: c, n: n, m: m, ks: ks,
                    lut: l, out: outBuf, opts: opts
                )
            }
        }
    }
}

@inline(__always)
public func adc_scan_u4(
    codes: [UInt8],        // packed nibbles, m/2 bytes per vector
    n: Int,
    m: Int,                // must be even
    ks: Int,               // must be 16
    lut: [Float],
    out: inout [Float],
    opts: ADCScanOpts? = nil
) {
    out.withUnsafeMutableBufferPointer { outBuf in
        codes.withUnsafeBufferPointer { c in
            lut.withUnsafeBufferPointer { l in
                adc_scan_u4(
                    codes: c, n: n, m: m, ks: ks,
                    lut: l, out: outBuf, opts: opts
                )
            }
        }
    }
}

// MARK: - Public API (pointer-oriented)

public func adc_scan_u8(
    codes: UnsafeBufferPointer<UInt8>,
    n: Int,
    m: Int,
    ks: Int,
    lut: UnsafeBufferPointer<Float>,
    out: UnsafeMutableBufferPointer<Float>,
    opts: ADCScanOpts? = nil
) {
    precondition(ks == 256, "adc_scan_u8: ks must be 256")
    precondition(lut.count >= m * ks, "lut must have m*ks floats")
    precondition(out.count >= n, "out must have n floats")

    let options = opts ?? ADCScanOpts()
    switch options.layout {
    case .aos:
        scanU8AoS(codes: codes, n: n, m: m, ks: ks, lut: lut, out: out, opts: options)
    case .interleavedBlock:
        precondition(options.groupSize > 0, "groupSize must be > 0 for INTERLEAVED_BLOCK")
        precondition(codes.count >= n * m, "codes size must be n*m for interleaved u8 (packed by layout)")
        scanU8Interleaved(codes: codes, n: n, m: m, ks: ks, lut: lut, out: out, opts: options)
    }
}

public func adc_scan_u4(
    codes: UnsafeBufferPointer<UInt8>,
    n: Int,
    m: Int,    // even
    ks: Int,   // 16
    lut: UnsafeBufferPointer<Float>,
    out: UnsafeMutableBufferPointer<Float>,
    opts: ADCScanOpts? = nil
) {
    precondition(ks == 16, "adc_scan_u4: ks must be 16")
    precondition(m % 2 == 0, "adc_scan_u4: m must be even")
    precondition(lut.count >= m * ks, "lut must have m*ks floats")
    precondition(out.count >= n, "out must have n floats")

    let options = opts ?? ADCScanOpts()
    switch options.layout {
    case .aos:
        let mBytes = m / 2
        precondition(codes.count >= n * mBytes, "codes size must be n*(m/2) for u4 AoS")
        scanU4AoS(codes: codes, n: n, m: m, ks: ks, lut: lut, out: out, opts: options)
    case .interleavedBlock:
        precondition(options.groupSize > 0, "groupSize must be > 0 for INTERLEAVED_BLOCK")
        // In interleaved u4, bytes are grouped by (pair-of-subspaces) and within that by g vectors
        precondition(codes.count >= (n * (m / 2)), "codes size must be n*(m/2) for u4 interleaved")
        scanU4Interleaved(codes: codes, n: n, m: m, ks: ks, lut: lut, out: out, opts: options)
    }
}

// MARK: - Parallel helpers

@inline(__always)
private func effectiveThreads(_ requested: Int) -> Int {
    if requested > 0 { return requested }
    // Use all active cores by default.
    return max(1, ProcessInfo.processInfo.activeProcessorCount)
}

// MARK: - Sendable wrappers for unsafe pointers

struct UnsafeSendablePtr<T>: @unchecked Sendable {
    let ptr: UnsafePointer<T>
}

struct UnsafeSendableMutPtr<T>: @unchecked Sendable {
    let ptr: UnsafeMutablePointer<T>
}

private func parallelRangeDispatch(
    total: Int,
    numWorkers: Int,
    _ body: @escaping @Sendable (_ start: Int, _ endExclusive: Int) -> Void
) {
    if numWorkers <= 1 || total < 1024 {
        body(0, total)
        return
    }
    let chunk = (total + numWorkers - 1) / numWorkers
    DispatchQueue.concurrentPerform(iterations: numWorkers) { worker in
        let start = worker * chunk
        let end = min(start + chunk, total)
        if start < end { body(start, end) }
    }
}

// MARK: - Core kernels (u8, AoS)

@inline(__always)
private func scanU8AoS(
    codes: UnsafeBufferPointer<UInt8>,
    n: Int,
    m: Int,
    ks: Int,
    lut: UnsafeBufferPointer<Float>,
    out: UnsafeMutableBufferPointer<Float>,
    opts: ADCScanOpts
) {
    let stride = (opts.stride > 0) ? opts.stride : m
    let prefetchD = max(0, opts.prefetchDistance)
    let bias = opts.addBias
    let doKahan = opts.strictFP && m >= 64

    // Create Sendable wrappers for pointers to ensure safe concurrent access
    let codesS = UnsafeSendablePtr(ptr: codes.baseAddress!)
    let lutS = UnsafeSendablePtr(ptr: lut.baseAddress!)
    let outS = UnsafeSendableMutPtr(ptr: out.baseAddress!)

    parallelRangeDispatch(total: n, numWorkers: effectiveThreads(opts.numThreads)) { a, b in
        // Recover base pointers from Sendable wrappers
        let codesBase = codesS.ptr
        let lutBase = lutS.ptr
        let outBase = outS.ptr

        for i in a..<b {
            // Prefetch future code row
            if prefetchD > 0 {
                let fi = i + prefetchD
                if fi < n {
                    let pfPtr = codesBase + fi * stride
                    vi_prefetch_read(pfPtr)
                }
            }

            // Pointers for current vector
            let row = codesBase + i * stride

            if doKahan {
                var sum: Float = 0
                var c: Float = 0
                var j = 0
                while j < m {
                    let code = Int(row[j])
                    let value = lutBase[j * ks + code]
                    let y = value - c
                    let t = sum + y
                    c = (t - sum) - y
                    sum = t
                    j += 1
                }
                outBase[i] = sum + bias
            } else {
                // ILP: 4 accumulators, unrolled by 8
                var s0: Float = 0, s1: Float = 0, s2: Float = 0, s3: Float = 0
                var j = 0

                // Unroll by 16 for better ILP
                while j + 15 < m {
                    let j0 = j
                    s0 += lutBase[(j0 + 0) * ks + Int(row[j0 + 0])]
                    s1 += lutBase[(j0 + 1) * ks + Int(row[j0 + 1])]
                    s2 += lutBase[(j0 + 2) * ks + Int(row[j0 + 2])]
                    s3 += lutBase[(j0 + 3) * ks + Int(row[j0 + 3])]
                    s0 += lutBase[(j0 + 4) * ks + Int(row[j0 + 4])]
                    s1 += lutBase[(j0 + 5) * ks + Int(row[j0 + 5])]
                    s2 += lutBase[(j0 + 6) * ks + Int(row[j0 + 6])]
                    s3 += lutBase[(j0 + 7) * ks + Int(row[j0 + 7])]
                    s0 += lutBase[(j0 + 8) * ks + Int(row[j0 + 8])]
                    s1 += lutBase[(j0 + 9) * ks + Int(row[j0 + 9])]
                    s2 += lutBase[(j0 + 10) * ks + Int(row[j0 + 10])]
                    s3 += lutBase[(j0 + 11) * ks + Int(row[j0 + 11])]
                    s0 += lutBase[(j0 + 12) * ks + Int(row[j0 + 12])]
                    s1 += lutBase[(j0 + 13) * ks + Int(row[j0 + 13])]
                    s2 += lutBase[(j0 + 14) * ks + Int(row[j0 + 14])]
                    s3 += lutBase[(j0 + 15) * ks + Int(row[j0 + 15])]
                    j += 16
                }
                while j + 3 < m {
                    s0 += lutBase[(j + 0) * ks + Int(row[j + 0])]
                    s1 += lutBase[(j + 1) * ks + Int(row[j + 1])]
                    s2 += lutBase[(j + 2) * ks + Int(row[j + 2])]
                    s3 += lutBase[(j + 3) * ks + Int(row[j + 3])]
                    j += 4
                }
                while j < m {
                    s0 += lutBase[j * ks + Int(row[j])]
                    j += 1
                }
                outBase[i] = (s0 + s1 + s2 + s3) + bias
            }
        }
    }
}

// MARK: - Core kernels (u8, Interleaved)

@inline(__always)
private func scanU8Interleaved(
    codes: UnsafeBufferPointer<UInt8>,
    n: Int,
    m: Int,
    ks: Int,
    lut: UnsafeBufferPointer<Float>,
    out: UnsafeMutableBufferPointer<Float>,
    opts: ADCScanOpts
) {
    let g = opts.groupSize
    precondition(g > 0, "groupSize must be > 0")
    let bias = opts.addBias
    let doKahan = opts.strictFP && m >= 64
    let prefetchD = max(0, opts.prefetchDistance)

    // Create Sendable wrappers for pointers to ensure safe concurrent access
    let codesS = UnsafeSendablePtr(ptr: codes.baseAddress!)
    let lutS = UnsafeSendablePtr(ptr: lut.baseAddress!)
    let outS = UnsafeSendableMutPtr(ptr: out.baseAddress!)

    let numBlocks = (n + g - 1) / g
    let blockSpan = m * g // bytes per block for u8

    parallelRangeDispatch(total: numBlocks, numWorkers: effectiveThreads(opts.numThreads)) { a, b in
        // Recover base pointers from Sendable wrappers
        let codesBase = codesS.ptr
        let lutBase = lutS.ptr
        let outBase = outS.ptr
        for block in a..<b {
            let baseIndex = block * g
            let blockSize = min(g, n - baseIndex)

            // Prefetch future block
            if prefetchD > 0 {
                let fb = block + prefetchD
                if fb < numBlocks {
                    let pf = codesBase + fb * blockSpan
                    vi_prefetch_read(pf)
                }
            }

            // Initialize sums for vectors in this block
            // g is typically <= 8; we allocate a small stack-like array
            var sums = [Float](repeating: 0, count: blockSize)

            let blockCodes = codesBase + block * blockSpan

            if doKahan {
                // Kahan per vector
                var sum = [Float](repeating: 0, count: blockSize)
                var comp = [Float](repeating: 0, count: blockSize)

                var j = 0
                while j < m {
                    let lutRow = lutBase + j * ks
                    let subspaceCodes = blockCodes + j * g
                    var v = 0
                    while v < blockSize {
                        let code = Int(subspaceCodes[v])
                        let value = lutRow[code]
                        let y = value - comp[v]
                        let t = sum[v] + y
                        comp[v] = (t - sum[v]) - y
                        sum[v] = t
                        v += 1
                    }
                    j += 1
                }
                for v in 0..<blockSize {
                    outBase[baseIndex + v] = sum[v] + bias
                }
            } else {
                // Standard summation
                var j = 0
                while j < m {
                    let lutRow = lutBase + j * ks
                    let subspaceCodes = blockCodes + j * g
                    var v = 0
                    while v < blockSize {
                        let code = Int(subspaceCodes[v])
                        sums[v] += lutRow[code]
                        v += 1
                    }
                    j += 1
                }
                for v in 0..<blockSize {
                    outBase[baseIndex + v] = sums[v] + bias
                }
            }
        }
    }
}

// MARK: - Core kernels (u4, AoS: packed nibbles)

@inline(__always)
private func scanU4AoS(
    codes: UnsafeBufferPointer<UInt8>, // packed, m/2 bytes per vector
    n: Int,
    m: Int,   // even
    ks: Int,  // 16
    lut: UnsafeBufferPointer<Float>,
    out: UnsafeMutableBufferPointer<Float>,
    opts: ADCScanOpts
) {
    let mBytes = m / 2
    let strideBytes = (opts.stride > 0) ? opts.stride : mBytes
    let prefetchD = max(0, opts.prefetchDistance)
    let bias = opts.addBias
    let doKahan = opts.strictFP && m >= 64

    // Create Sendable wrappers for pointers to ensure safe concurrent access
    let codesS = UnsafeSendablePtr(ptr: codes.baseAddress!)
    let lutS = UnsafeSendablePtr(ptr: lut.baseAddress!)
    let outS = UnsafeSendableMutPtr(ptr: out.baseAddress!)

    parallelRangeDispatch(total: n, numWorkers: effectiveThreads(opts.numThreads)) { a, b in
        // Recover base pointers from Sendable wrappers
        let codesBase = codesS.ptr
        let lutBase = lutS.ptr
        let outBase = outS.ptr

        for i in a..<b {
            if prefetchD > 0 {
                let fi = i + prefetchD
                if fi < n {
                    vi_prefetch_read(codesBase + fi * strideBytes)
                }
            }

            let row = codesBase + i * strideBytes

            if doKahan {
                var sum: Float = 0
                var c: Float = 0
                var j = 0
                while j < m {
                    let byte = row[j >> 1]
                    let c0 = Int(byte & 0xF)          // subspace j
                    let c1 = Int((byte >> 4) & 0xF)   // subspace j+1
                    let v0 = lutBase[j * ks + c0]
                    let y0 = v0 - c
                    var t = sum + y0
                    c = (t - sum) - y0
                    sum = t

                    let v1 = lutBase[(j + 1) * ks + c1]
                    let y1 = v1 - c
                    t = sum + y1
                    c = (t - sum) - y1
                    sum = t

                    j += 2
                }
                outBase[i] = sum + bias
            } else {
                var sum: Float = 0
                var j = 0
                while j < m {
                    let byte = row[j >> 1]
                    sum += lutBase[j * ks + Int(byte & 0xF)]
                    sum += lutBase[(j + 1) * ks + Int((byte >> 4) & 0xF)]
                    j += 2
                }
                outBase[i] = sum + bias
            }
        }
    }
}

// MARK: - Core kernels (u4, Interleaved packed)

/// Interleaved layout for u4 packs bytes per *pair* of subspaces:
/// codes[(i/g)*(m/2)*g + (jPair)*g + (i%g)] holds a byte with:
///   low nibble = code for subspace (2*jPair),
///   high nibble = code for subspace (2*jPair+1)
@inline(__always)
private func scanU4Interleaved(
    codes: UnsafeBufferPointer<UInt8>,
    n: Int,
    m: Int,   // even
    ks: Int,  // 16
    lut: UnsafeBufferPointer<Float>,
    out: UnsafeMutableBufferPointer<Float>,
    opts: ADCScanOpts
) {
    let g = opts.groupSize
    precondition(g > 0)
    let bias = opts.addBias
    let doKahan = opts.strictFP && m >= 64
    let prefetchD = max(0, opts.prefetchDistance)

    // Create Sendable wrappers for pointers to ensure safe concurrent access
    let codesS = UnsafeSendablePtr(ptr: codes.baseAddress!)
    let lutS = UnsafeSendablePtr(ptr: lut.baseAddress!)
    let outS = UnsafeSendableMutPtr(ptr: out.baseAddress!)

    let pairs = m / 2
    let blockSpanBytes = pairs * g   // bytes per block
    let numBlocks = (n + g - 1) / g

    parallelRangeDispatch(total: numBlocks, numWorkers: effectiveThreads(opts.numThreads)) { a, b in
        // Recover base pointers from Sendable wrappers
        let codesBase = codesS.ptr
        let lutBase = lutS.ptr
        let outBase = outS.ptr
        for block in a..<b {
            let baseIndex = block * g
            let blockSize = min(g, n - baseIndex)

            if prefetchD > 0 {
                let fb = block + prefetchD
                if fb < numBlocks {
                    vi_prefetch_read(codesBase + fb * blockSpanBytes)
                }
            }

            let blockCodes = codesBase + block * blockSpanBytes

            if doKahan {
                var sum = [Float](repeating: 0, count: blockSize)
                var comp = [Float](repeating: 0, count: blockSize)

                var jp = 0
                while jp < pairs {
                    let j0 = 2 * jp
                    let lut0 = lutBase + j0 * ks
                    let lut1 = lutBase + (j0 + 1) * ks
                    let pairBytes = blockCodes + jp * g

                    var v = 0
                    while v < blockSize {
                        let byte = pairBytes[v]
                        let c0 = Int(byte & 0xF)
                        let c1 = Int((byte >> 4) & 0xF)

                        let val0 = lut0[c0]
                        let y0 = val0 - comp[v]
                        var t = sum[v] + y0
                        comp[v] = (t - sum[v]) - y0
                        sum[v] = t

                        let val1 = lut1[c1]
                        let y1 = val1 - comp[v]
                        t = sum[v] + y1
                        comp[v] = (t - sum[v]) - y1
                        sum[v] = t

                        v += 1
                    }
                    jp += 1
                }

                for v in 0..<blockSize {
                    outBase[baseIndex + v] = sum[v] + bias
                }
            } else {
                var sums = [Float](repeating: 0, count: blockSize)
                var jp = 0
                while jp < pairs {
                    let j0 = 2 * jp
                    let lut0 = lutBase + j0 * ks
                    let lut1 = lutBase + (j0 + 1) * ks
                    let pairBytes = blockCodes + jp * g

                    var v = 0
                    while v < blockSize {
                        let byte = pairBytes[v]
                        sums[v] += lut0[Int(byte & 0xF)]
                        sums[v] += lut1[Int((byte >> 4) & 0xF)]
                        v += 1
                    }
                    jp += 1
                }

                for v in 0..<blockSize {
                    outBase[baseIndex + v] = sums[v] + bias
                }
            }
        }
    }
}
