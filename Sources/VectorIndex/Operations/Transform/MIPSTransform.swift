// Sources/VectorIndex/Operations/Transform/MIPSTransform.swift
//
// Kernel #10: MIPS-to-L2 Transform
// Converts Maximum Inner Product Search (MIPS) to L2 distance search via
// vector augmentation: x' = [x ; sqrt(R² - ||x||²)], q' = [q ; 0]
//
// Theory: max <q,x> ⟺ min ||q' - x'||² where R² ≥ max_i ||x_i||²
//
// Dependencies: None (pure math transform)
// Thread-safety: Query-local state; safe for concurrent queries

import Foundation
import simd

// MARK: - Transform Mode

/// How MIPS-to-L2 transform is applied
@frozen
public enum MIPSTransformMode: Sendable {
    case explicit       // Materialized augmented dimensions
    case virtual        // On-the-fly computation
    case hybrid         // Auto-select based on R² staleness
}

// MARK: - R² Parameter Management

/// R² parameter for MIPS transform. Must satisfy R² ≥ max_i ‖x_i‖².
/// R² is tracked with a safety margin; staleness flips to true if a new
/// vector exceeds the current bound (so queries can fall back to virtual).
@frozen
public struct R2Parameter {
    public var value: Float
    public var isStale: Bool
    public var maxNormSquared: Float
    public let margin: Float

    @inlinable
    public init(maxNormSquared: Float, margin: Float = 1e-6) {
        self.maxNormSquared = maxNormSquared
        self.margin = margin
        self.value = maxNormSquared * (1.0 + margin)
        self.isStale = false
    }

    /// Observe a new vector's ‖x‖²; flip staleness if bound exceeded.
    @inlinable
    public mutating func observe(normSquared: Float) {
        if normSquared > maxNormSquared {
            maxNormSquared = normSquared
            if normSquared > value {
                isStale = true
            }
        }
    }

    /// Refresh after rematerialization (rebuild explicit storage).
    @inlinable
    public mutating func refresh() {
        value = maxNormSquared * (1.0 + margin)
        isStale = false
    }
}

// MARK: - Augmented Vector Storage (explicit/materialized)

/// Storage for augmented vectors x' = [x ; sqrt(max(0, R² - ‖x‖²))] with
/// paddedDim = roundUp(d+1, 16). Backing memory is 64B-aligned.
public struct AugmentedVectorStorage {
    public let originalDim: Int
    public let paddedDim: Int
    public let count: Int

    /// Augmented vectors [count][paddedDim]. The (d)-th element (0-based) holds
    /// the augmentation value, and [d+1 ..< paddedDim) are zero padding.
    @usableFromInline internal var vectors: UnsafeMutablePointer<Float>?

    /// Raw pointer we own for deallocation.
    @usableFromInline internal var raw: UnsafeMutableRawPointer?

    /// Current R² parameter associated with this storage.
    public var r2: R2Parameter

    @inlinable
    public init(count: Int, originalDim: Int, r2: R2Parameter = .init(maxNormSquared: 0)) {
        self.count = count
        self.originalDim = originalDim
        self.paddedDim = ((originalDim + 1) + 15) & ~15  // round up to multiple of 16
        self.r2 = r2
        self.vectors = nil
        self.raw = nil
    }

    /// Allocate 64B-aligned storage; elements are zero-initialized.
    @inlinable
    public mutating func allocate() {
        precondition(vectors == nil && raw == nil, "Already allocated")
        let byteCount = count * paddedDim * MemoryLayout<Float>.stride
        let alignment = 64
        let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
        rawPtr.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)
        self.raw = rawPtr
        self.vectors = rawPtr.bindMemory(to: Float.self, capacity: count * paddedDim)
    }

    @inlinable
    public func deallocate() {
        guard let raw = raw else { return }
        raw.deallocate()
    }
}

// MARK: - Telemetry (struct only; hook to your recorder if desired)

@frozen
public struct MIPSTransformTelemetry {
    public let mode: MIPSTransformMode
    public let vectorsProcessed: Int
    public let dimension: Int
    public let r2Value: Float
    public let r2Stale: Bool
    public let materialized: Bool
    public let executionTimeNanos: UInt64

    @inlinable
    public var throughputVecsPerSec: Double {
        let seconds = Double(executionTimeNanos) / 1e9
        return Double(vectorsProcessed) / max(seconds, .ulpOfOne)
    }
}

// MARK: - Public API

/// Compute R² = (1 + margin) * max_i ‖x_i‖² over the dataset.
/// - vectors: AoS [n][d]
@inlinable
public func computeR2Parameter(
    vectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    margin: Float = 1e-6
) -> R2Parameter {
    var maxSq: Float = 0
    for i in 0..<n {
        let row = vectors + i * d
        let s = l2NormSquaredSIMD(row, d)
        if s > maxSq { maxSq = s }
    }
    return R2Parameter(maxNormSquared: maxSq, margin: margin)
}

/// Explicit/materialized transform: x → [x ; sqrt(max(0, R² - ‖x‖²))],
/// storing into `augmentedOut` laid out as [n][paddedDim].
/// Padded tail beyond (d+1) is zeroed.
@inlinable
public func mipsMaterializeAugmentation(
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: R2Parameter,
    augmentedOut: UnsafeMutablePointer<Float>,
    paddedDim: Int
) {
    precondition(paddedDim >= d + 1 && (paddedDim % 16 == 0), "paddedDim must be ≥ d+1 and multiple of 16")
    let r2v: Float = r2.value
    let rowBytes = paddedDim * MemoryLayout<Float>.stride

    // Zero all (including padding) up-front for cache-friendly write-combine
    memset(augmentedOut, 0, n * rowBytes)

    for i in 0..<n {
        let x = baseVectors + i * d
        let dst = augmentedOut + i * paddedDim

        // Copy x into first d slots (FIXED: use .update instead of deprecated .assign)
        dst.update(from: x, count: d)

        // Compute sqrt(max(0, R² - ‖x‖²)) for slot d
        let normSq = l2NormSquaredSIMD(x, d)
        let radicand = max(0, r2v - normSq)
        dst[d] = sqrtf(radicand)
        // dst[d+1 ..< paddedDim) remain zero (already memset)
    }
}

/// Augment a query: q' = [q ; 0] and pad to paddedDim with zeros.
@inlinable
public func mipsAugmentQuery(
    query: UnsafePointer<Float>,
    dimension d: Int,
    augmentedOut: UnsafeMutablePointer<Float>,
    paddedDim: Int
) {
    precondition(paddedDim >= d + 1 && (paddedDim % 16 == 0))
    // Zero whole row then copy q; cheaper than partial clears for small dims
    memset(augmentedOut, 0, paddedDim * MemoryLayout<Float>.stride)
    augmentedOut.update(from: query, count: d)
    augmentedOut[d] = 0 // explicit for clarity
}

/// Virtual/on-the-fly transform without materializing x':
/// Computes scores[i] = ‖q‖² + R² − 2·⟨q, x_i⟩ (min L2^2 ≡ max IP).
/// Results can be fed directly to Top‑K with `.min` ordering.
@inlinable
public func mipsVirtualToL2Scores(
    query: UnsafePointer<Float>,
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: R2Parameter,
    scoresOut: UnsafeMutablePointer<Float>
) {
    let qSq = l2NormSquaredSIMD(query, d)
    let r2v = r2.value
    // Compute dot and apply fused epilogue (no temporary buffer)
    for i in 0..<n {
        let x = baseVectors + i * d
        let dot = innerProductSIMD(query, x, d)
        scoresOut[i] = qSq + r2v - 2.0 * dot
    }
}

/// Hybrid mode: use explicit/materialized path when `storage.r2.isStale == false`
/// and storage is available; otherwise fallback to the virtual path.
/// - storage.vectors must hold [n][paddedDim] if used explicitly.
@inlinable
public func mipsHybridScoreBlock(
    query: UnsafePointer<Float>,
    storage: AugmentedVectorStorage,
    baseVectors: UnsafePointer<Float>?,
    scoresOut: UnsafeMutablePointer<Float>
) {
    let n = storage.count
    if !storage.r2.isStale, let augBase = storage.vectors {
        // Explicit: augment query, then use L2^2 block
        // (min L2^2 equals max inner-product ranking; squared is fine)
        let augQRaw: UnsafeMutableRawPointer? = UnsafeMutableRawPointer.allocate(
            byteCount: storage.paddedDim * MemoryLayout<Float>.stride,
            alignment: 64
        )
        defer { augQRaw?.deallocate() }
        let augQ = augQRaw!.bindMemory(to: Float.self, capacity: storage.paddedDim)
        mipsAugmentQuery(query: query, dimension: storage.originalDim, augmentedOut: augQ, paddedDim: storage.paddedDim)

        // Prefer your high-performance L2 microkernel (#01) if available; otherwise fallback.
        l2sqrBlock_dispatch(
            query: augQ,
            database: augBase,
            vectorCount: n,
            dimension: storage.paddedDim,
            output: scoresOut
        )
    } else {
        // Virtual fallback
        precondition(baseVectors != nil, "Virtual mode requires baseVectors")
        mipsVirtualToL2Scores(
            query: query,
            baseVectors: baseVectors!,
            count: n,
            dimension: storage.originalDim,
            r2: storage.r2,
            scoresOut: scoresOut
        )
    }
}

// MARK: - Helper: SIMD L2 norm (sum of squares)

@inlinable
internal func l2NormSquaredSIMD(_ x: UnsafePointer<Float>, _ d: Int) -> Float {
    let w = 4
    let dv = (d / w) * w
    var a0 = SIMD4<Float>(repeating: 0), a1 = SIMD4<Float>(repeating: 0),
        a2 = SIMD4<Float>(repeating: 0), a3 = SIMD4<Float>(repeating: 0)
    var j = 0
    while j + 15 < d {
        let v0 = SIMD4<Float>(x[j + 0], x[j + 1], x[j + 2], x[j + 3])
        let v1 = SIMD4<Float>(x[j + 4], x[j + 5], x[j + 6], x[j + 7])
        let v2 = SIMD4<Float>(x[j + 8], x[j + 9], x[j + 10], x[j + 11])
        let v3 = SIMD4<Float>(x[j + 12], x[j + 13], x[j + 14], x[j + 15])
        // FIXED: Use regular operators, not wrapping operators
        a0 += v0 * v0
        a1 += v1 * v1
        a2 += v2 * v2
        a3 += v3 * v3
        j += 16
    }
    let combined = a0 + a1 + a2 + a3
    var acc = combined[0] + combined[1] + combined[2] + combined[3]
    while j < dv {
        let v = SIMD4<Float>(x[j + 0], x[j + 1], x[j + 2], x[j + 3])
        let vSq = v * v
        acc += vSq[0] + vSq[1] + vSq[2] + vSq[3]
        j += w
    }
    while j < d {
        let v = x[j]
        acc += v * v
        j += 1
    }
    return acc
}

// MARK: - Helper: SIMD inner product (dot)

@inlinable
internal func innerProductSIMD(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    let w = 4
    let dv = (d / w) * w
    var a0 = SIMD4<Float>(repeating: 0), a1 = SIMD4<Float>(repeating: 0),
        a2 = SIMD4<Float>(repeating: 0), a3 = SIMD4<Float>(repeating: 0)
    var j = 0
    while j + 15 < d {
        let q0 = SIMD4<Float>(a[j + 0], a[j + 1], a[j + 2], a[j + 3])
        let x0 = SIMD4<Float>(b[j + 0], b[j + 1], b[j + 2], b[j + 3])
        let q1 = SIMD4<Float>(a[j + 4], a[j + 5], a[j + 6], a[j + 7])
        let x1 = SIMD4<Float>(b[j + 4], b[j + 5], b[j + 6], b[j + 7])
        let q2 = SIMD4<Float>(a[j + 8], a[j + 9], a[j + 10], a[j + 11])
        let x2 = SIMD4<Float>(b[j + 8], b[j + 9], b[j + 10], b[j + 11])
        let q3 = SIMD4<Float>(a[j + 12], a[j + 13], a[j + 14], a[j + 15])
        let x3 = SIMD4<Float>(b[j + 12], b[j + 13], b[j + 14], b[j + 15])
        // FIXED: Use regular operators, not wrapping operators
        a0 += q0 * x0
        a1 += q1 * x1
        a2 += q2 * x2
        a3 += q3 * x3
        j += 16
    }
    let combined = a0 + a1 + a2 + a3
    var acc = combined[0] + combined[1] + combined[2] + combined[3]
    while j < dv {
        let qv = SIMD4<Float>(a[j + 0], a[j + 1], a[j + 2], a[j + 3])
        let xv = SIMD4<Float>(b[j + 0], b[j + 1], b[j + 2], b[j + 3])
        let prod = qv * xv
        acc += prod[0] + prod[1] + prod[2] + prod[3]
        j += w
    }
    while j < d {
        acc += a[j] * b[j]
        j += 1
    }
    return acc
}

// MARK: - L2² block dispatcher (prefers high-perf kernel #01 when present)

/// If your project already includes the L2 microkernel (#01) this shim will
/// naturally inline to it; otherwise a generic fallback is used.
@inlinable
internal func l2sqrBlock_dispatch(
    query: UnsafePointer<Float>,
    database: UnsafePointer<Float>,
    vectorCount n: Int,
    dimension d: Int,
    output: UnsafeMutablePointer<Float>
) {
    // Try to use the real kernel #01 if available
    #if canImport(VectorCore)
    // Wire to your actual entrypoint if available
    generic_l2sqrBlock(query: query, database: database, n: n, d: d, out: output)
    #else
    // Fall back to generic implementation
    generic_l2sqrBlock(query: query, database: database, n: n, d: d, out: output)
    #endif
}

@inlinable
internal func generic_l2sqrBlock(
    query q: UnsafePointer<Float>,
    database xb: UnsafePointer<Float>,
    n: Int,
    d: Int,
    out: UnsafeMutablePointer<Float>
) {
    // Simple, cache-friendly row loop with SIMD inner diff² accumulation
    for i in 0..<n {
        let x = xb + i * d
        var acc0 = SIMD4<Float>(repeating: 0), acc1 = SIMD4<Float>(repeating: 0),
            acc2 = SIMD4<Float>(repeating: 0), acc3 = SIMD4<Float>(repeating: 0)
        var j = 0
        while j + 15 < d {
            let q0 = SIMD4<Float>(q[j + 0], q[j + 1], q[j + 2], q[j + 3])
            let x0 = SIMD4<Float>(x[j + 0], x[j + 1], x[j + 2], x[j + 3])
            let q1 = SIMD4<Float>(q[j + 4], q[j + 5], q[j + 6], q[j + 7])
            let x1 = SIMD4<Float>(x[j + 4], x[j + 5], x[j + 6], x[j + 7])
            let q2 = SIMD4<Float>(q[j + 8], q[j + 9], q[j + 10], q[j + 11])
            let x2 = SIMD4<Float>(x[j + 8], x[j + 9], x[j + 10], x[j + 11])
            let q3 = SIMD4<Float>(q[j + 12], q[j + 13], q[j + 14], q[j + 15])
            let x3 = SIMD4<Float>(x[j + 12], x[j + 13], x[j + 14], x[j + 15])
            // FIXED: Use regular operators, not wrapping operators
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
        let combined = acc0 + acc1 + acc2 + acc3
        var s = combined[0] + combined[1] + combined[2] + combined[3]
        while j < d {
            let diff = q[j] - x[j]
            s += diff * diff
            j += 1
        }
        out[i] = s
    }
}
