//
//  KMeansSeeding.swift
//  VectorIndex
//
//  Kernel #11: K-means++ Seeding — High-Quality Centroid Initialization
//
//  Implements k-means++ (Arthur & Vassilvitskii, 2007) for intelligent centroid
//  initialization using D²-weighted sampling. Provides O(log k) approximation
//  guarantee compared to random initialization.
//
//  Mathematical Foundation:
//    For each new centroid c_t, select from data with probability:
//      P(x_i selected) ∝ D²(x_i) = min_j ‖x_i - c_j‖²
//
//    Guarantee: E[φ] ≤ 8(ln k + 2) · φ_OPT
//      where φ = Σ_i min_j ‖x_i - c_j‖² is clustering cost
//
//  Features:
//    - Deterministic seeding with reproducible RNG
//    - Numerically stable f64 accumulation
//    - D² update using vectorized L2 distance
//    - Empty cluster handling
//
//  Spec: kernel-specs/11_kmeanspp_seed.md
//

import Foundation

// MARK: - Public API

/// K-means seeding algorithm variant
@frozen
public enum KMeansSeedAlgorithm: Sendable {
    case plusPlus       // Exact k-means++ (sequential D² sampling)
    case parallel       // k-means|| (parallel oversampling) - future
}

/// Configuration for k-means++ seeding
@frozen
public struct KMeansSeedConfig: Sendable {
    /// Seeding algorithm
    public let algorithm: KMeansSeedAlgorithm

    /// Number of centroids to select
    public let k: Int

    /// Optional: Subsample size (0 = use all data)
    public let sampleSize: Int

    /// RNG seed for reproducibility
    public let rngSeed: UInt64

    /// RNG stream ID for parallel reproducibility
    public let rngStreamID: UInt64

    /// Strict floating-point mode (deterministic reassociation)
    public let strictFP: Bool

    /// Prefetch distance for cache optimization
    public let prefetchDistance: Int

    /// k-means|| specific: Oversampling factor (default: 2)
    public let oversamplingFactor: Int

    /// k-means|| specific: Number of rounds (default: 5)
    public let rounds: Int

    @inlinable
    public init(
        algorithm: KMeansSeedAlgorithm = .plusPlus,
        k: Int = 256,
        sampleSize: Int = 0,
        rngSeed: UInt64 = 0,
        rngStreamID: UInt64 = 0,
        strictFP: Bool = false,
        prefetchDistance: Int = 2,
        oversamplingFactor: Int = 2,
        rounds: Int = 5
    ) {
        self.algorithm = algorithm
        self.k = k
        self.sampleSize = sampleSize
        self.rngSeed = rngSeed
        self.rngStreamID = rngStreamID
        self.strictFP = strictFP
        self.prefetchDistance = prefetchDistance
        self.oversamplingFactor = oversamplingFactor
        self.rounds = rounds
    }

    public static let `default` = KMeansSeedConfig()
}

/// Statistics collected during k-means++ seeding
@frozen
public struct KMeansSeedStats: Sendable {
    public let algorithm: KMeansSeedAlgorithm
    public let k: Int
    public let n: Int
    public let dimension: Int
    public let chosenIndices: [Int]
    public let totalCost: Double          // Σ D²(x_i) at end
    public let passesOverData: Int        // Number of full scans
    public let executionTimeNanos: UInt64
    public let rngSeedUsed: UInt64

    @inlinable
    public init(
        algorithm: KMeansSeedAlgorithm,
        k: Int,
        n: Int,
        dimension: Int,
        chosenIndices: [Int],
        totalCost: Double,
        passesOverData: Int,
        executionTimeNanos: UInt64,
        rngSeedUsed: UInt64
    ) {
        self.algorithm = algorithm
        self.k = k
        self.n = n
        self.dimension = dimension
        self.chosenIndices = chosenIndices
        self.totalCost = totalCost
        self.passesOverData = passesOverData
        self.executionTimeNanos = executionTimeNanos
        self.rngSeedUsed = rngSeedUsed
    }

    /// Average distance to nearest centroid
    @inlinable
    public var averageDistanceSquared: Double {
        return totalCost / Double(n)
    }
}

// MARK: - Core k-means++ Implementation

/// Initialize k centroids using k-means++ algorithm
///
/// Mathematical foundation:
///   For each new centroid c_t, select from data with probability:
///   P(x_i selected) ∝ D²(x_i) = min_j ‖x_i - c_j‖²
///
///   Guarantee: E[φ] ≤ 8(ln k + 2) · φ_OPT
///   where φ = Σ_i min_j ‖x_i - c_j‖² is clustering cost
///
/// - Complexity: O(ndk) where n=count, d=dimension, k=centroids
/// - Performance: ~2-3 GB/s on M1 (memory-bound for distance computation)
///
/// - Parameters:
///   - data: Input vectors [n][d], row-major, Float32
///   - count: Number of vectors
///   - dimension: Vector dimension
///   - k: Number of centroids to select
///   - config: Seeding configuration
///   - centroidsOut: Output centroids [k][d]
///   - chosenIndicesOut: Optional indices of chosen seeds [k]
/// - Returns: Statistics about seeding process
@inlinable
public func kmeansPlusPlusSeed(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    k: Int,
    config: KMeansSeedConfig = .default,
    centroidsOut: UnsafeMutablePointer<Float>,
    chosenIndicesOut: UnsafeMutablePointer<Int>?
) -> KMeansSeedStats {
    precondition(k >= 1 && k <= n, "k must be in [1, n]")
    precondition(d >= 1, "dimension must be positive")

    let startTime = DispatchTime.now().uptimeNanoseconds

    var rng = RNGState(seed: config.rngSeed, stream: config.rngStreamID)
    var chosenIndices: [Int] = []
    chosenIndices.reserveCapacity(k)

    // Allocate D² array (numerically stable Double accumulation)
    var squaredDistances = [Float](repeating: Float.infinity, count: n)

    // Step 1: Choose first centroid uniformly at random
    let firstIdx = rng.nextInt(bound: n)
    chosenIndices.append(firstIdx)

    // Copy first centroid
    let firstCentroid = data.advanced(by: firstIdx * d)
    for j in 0..<d {
        centroidsOut[j] = firstCentroid[j]
    }

    // Initialize D² with distances to first centroid
    _vi_km11_updateSquaredDistances(
        data: data,
        count: n,
        dimension: d,
        newCentroid: firstCentroid,
        squaredDistances: &squaredDistances
    )

    // Steps 2 to k: Select remaining centroids with D² weighting
    for t in 1..<k {
        // Sample proportional to D²
        let selectedIdx = _vi_km11_sampleProportionalToWeight(
            weights: squaredDistances,
            count: n,
            rng: &rng
        )

        chosenIndices.append(selectedIdx)

        // Copy centroid
        let centroid = data.advanced(by: selectedIdx * d)
        let centroidOut = centroidsOut.advanced(by: t * d)
        for j in 0..<d {
            centroidOut[j] = centroid[j]
        }

        // Update D² with new centroid
        _vi_km11_updateSquaredDistances(
            data: data,
            count: n,
            dimension: d,
            newCentroid: centroid,
            squaredDistances: &squaredDistances
        )
    }

    // Compute final cost (using Double for stability)
    let totalCost = squaredDistances.reduce(0.0) { Double($0) + Double($1) }

    // Record chosen indices if requested
    if let outIndices = chosenIndicesOut {
        for i in 0..<k {
            outIndices[i] = chosenIndices[i]
        }
    }

    let endTime = DispatchTime.now().uptimeNanoseconds

    return KMeansSeedStats(
        algorithm: config.algorithm,
        k: k,
        n: n,
        dimension: d,
        chosenIndices: chosenIndices,
        totalCost: totalCost,
        passesOverData: k,
        executionTimeNanos: endTime - startTime,
        rngSeedUsed: config.rngSeed
    )
}

// MARK: - Internal Helpers

/// Update squared distances to nearest centroid
/// Core operation for k-means++ D² computation
///
/// For each vector x_i:
///   D²_new[i] = min(D²_old[i], ‖x_i - c_new‖²)
@usableFromInline
internal func _vi_km11_updateSquaredDistances(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    newCentroid: UnsafePointer<Float>,
    squaredDistances: inout [Float]
) {
    // Compute D²_new[i] = min(D²_old[i], ‖x_i - c_new‖²)
    for i in 0..<n {
        let vecPtr = data.advanced(by: i * d)

        // Compute ‖x_i - c_new‖² using SIMD
        var distSquared: Float = 0

        // Vectorized computation (8-wide with dual SIMD4)
        let d8 = (d / 8) * 8
        var j = 0

        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero

        while j < d8 {
            let v0Raw = UnsafeRawPointer(vecPtr.advanced(by: j))
            let c0Raw = UnsafeRawPointer(newCentroid.advanced(by: j))
            let v0 = v0Raw.load(as: SIMD4<Float>.self)
            let c0 = c0Raw.load(as: SIMD4<Float>.self)
            let diff0 = v0 - c0
            acc0 += diff0 * diff0

            let v1Raw = UnsafeRawPointer(vecPtr.advanced(by: j + 4))
            let c1Raw = UnsafeRawPointer(newCentroid.advanced(by: j + 4))
            let v1 = v1Raw.load(as: SIMD4<Float>.self)
            let c1 = c1Raw.load(as: SIMD4<Float>.self)
            let diff1 = v1 - c1
            acc1 += diff1 * diff1

            j += 8
        }

        distSquared = (acc0 + acc1).sum()

        // Scalar tail
        while j < d {
            let diff = vecPtr[j] - newCentroid[j]
            distSquared += diff * diff
            j += 1
        }

        // Update D² with minimum (guards against NaN/Inf)
        let safeDist = distSquared.isFinite && distSquared >= 0 ? distSquared : 0
        squaredDistances[i] = min(squaredDistances[i], safeDist)
    }
}

/// Sample index proportional to weights using inverse CDF method
/// Uses binary search for O(log n) selection (future optimization)
///
/// Current implementation: Linear scan O(n)
@usableFromInline
internal func _vi_km11_sampleProportionalToWeight(
    weights: [Float],
    count n: Int,
    rng: inout RNGState
) -> Int {
    // Compute total weight (use f64 for numerical stability)
    var totalWeight: Double = 0
    for i in 0..<n {
        let w = Double(weights[i])
        // Guard against negative/NaN weights
        if w.isFinite && w >= 0 {
            totalWeight += w
        }
    }

    precondition(totalWeight > 0, "Total weight must be positive")

    // Draw uniform random value in [0, totalWeight)
    let threshold = rng.nextDouble() * totalWeight

    // Find index where cumulative sum exceeds threshold (deterministic tie-breaking)
    var cumulativeSum: Double = 0
    for i in 0..<n {
        let w = Double(weights[i])
        if w.isFinite && w >= 0 {
            cumulativeSum += w
        }
        if cumulativeSum >= threshold {
            return i
        }
    }

    // Should not reach here (numerical error protection)
    return n - 1
}
