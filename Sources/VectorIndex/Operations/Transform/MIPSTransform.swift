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
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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

@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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
        let s = IndexOps.Support.Norms.l2NormSquared(vector: row, dimension: d)
        if s > maxSq { maxSq = s }
    }
    return R2Parameter(maxNormSquared: maxSq, margin: margin)
}

/// Explicit/materialized transform: x → [x ; sqrt(max(0, R² - ‖x‖²))],
/// storing into `augmentedOut` laid out as [n][paddedDim].
/// Padded tail beyond (d+1) is zeroed.
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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
        let normSq = IndexOps.Support.Norms.l2NormSquared(vector: x, dimension: d)
        let radicand = max(0, r2v - normSq)
        dst[d] = sqrtf(radicand)
        // dst[d+1 ..< paddedDim) remain zero (already memset)
    }
}

/// Augment a query: q' = [q ; 0] and pad to paddedDim with zeros.
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
@inlinable
public func mipsVirtualToL2Scores(
    query: UnsafePointer<Float>,
    baseVectors: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    r2: R2Parameter,
    scoresOut: UnsafeMutablePointer<Float>
) {
    let qSq = IndexOps.Support.Norms.l2NormSquared(vector: query, dimension: d)
    let r2v = r2.value
    // Batch-compute dot products via the canonical kernel, then apply the
    // fused epilogue in place (no temporary buffer needed).
    IndexOps.Scoring.InnerProduct.run(q: query, xb: baseVectors, n: n, d: d, out: scoresOut)
    for i in 0..<n {
        scoresOut[i] = qSq + r2v - 2.0 * scoresOut[i]
    }
}

/// Hybrid mode: use explicit/materialized path when `storage.r2.isStale == false`
/// and storage is available; otherwise fallback to the virtual path.
/// - storage.vectors must hold [n][paddedDim] if used explicitly.
@available(*, deprecated, message: "Unused; zero callers/tests found repo-wide. Scheduled for removal in 0.2.0's breaking phase.")
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

        // Canonical L2² microkernel (#01).
        l2sqr_f32_block(augQ, augBase, n, storage.paddedDim, scoresOut)
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
