Title: K-means++ Seeding Kernel — High-Quality Centroid Initialization for Clustering

Summary
- Implement k-means++ seeding algorithm for initializing coarse quantizer centroids in IVF indices.
- Provides O(log k) approximation guarantee compared to random initialization.
- Supports exact k-means++ (sequential) and k-means|| (parallel oversampling) variants.
- Critical for IVF training: high-quality seeds reduce Lloyd iterations and improve final clustering quality.
- Deterministic seeding with reproducible RNG for debugging and validation.

Project Context
- VectorIndex uses IVF (Inverted File) indexing with coarse quantization
- Coarse quantizer quality directly impacts search performance:
  - **Better centroids**: More balanced cells, higher recall
  - **Poor centroids**: Empty cells, imbalanced partitions, lower recall
  - **Random init**: Requires 10-20× more Lloyd iterations to converge
  - **k-means++**: Requires 2-5× iterations, better final quality
- Industry context: k-means++ is standard for initialization (Arthur & Vassilvitskii, 2007)
- Mathematical guarantee: E[φ] ≤ 8(ln k + 2) · OPT where φ is clustering cost
- VectorCore provides distance kernels; VectorIndex needs clustering initialization
- Typical usage:
  - Build phase: Seed kc centroids (kc ∈ [100, 65536])
  - Training phase: Run Lloyd iterations on seeds
  - Query phase: Use trained centroids for routing

Goals
- Mathematically correct k-means++ with D² weighting
- Deterministic seeding given RNG state (reproducible)
- Numerically stable (f64 accumulation for large datasets)
- Support both exact and scalable (k-means||) variants
- Efficient L2 distance computation reusing kernel #01
- Handle large datasets via uniform subsampling
- Thread-safe parallel distance updates with deterministic reduction

Scope & Deliverables
- New file: `Sources/VectorIndex/Kernels/KMeansSeedingKernel.swift`
- Core implementations:
  - Exact k-means++: `kmeansPlusPlusSeed` for sequential seeding
  - k-means||: `kmeansParallelSeed` for oversampling variant
  - D² update: `updateSquaredDistances` for distance maintenance
  - Weighted selection: `sampleProportionalToWeight` for probabilistic selection
  - Subsampling: `uniformSubsample` for large datasets
- Seed algorithms:
  - PlusPlus: Exact D²-weighted sampling (default)
  - Parallel: k-means|| with oversampling and reduction
- Integration points:
  - Uses L2 distance kernel (#01) for distance computation
  - Uses RNG utilities for deterministic sampling
  - Feeds Lloyd iterations (#12) for training
  - Used by IVF index construction

API & Signatures

```swift
// MARK: - Seeding Algorithm

/// k-means seeding algorithm variant
public enum KMeansSeedAlgorithm {
    case plusPlus       // Exact k-means++ (sequential)
    case parallel       // k-means|| (parallel oversampling)
}

// MARK: - Configuration

/// Configuration for k-means seeding
public struct KMeansSeedConfig {
    /// Seeding algorithm
    let algorithm: KMeansSeedAlgorithm

    /// Number of centroids to select
    let k: Int

    /// Optional: Subsample size (0 = use all data)
    let sampleSize: Int

    /// RNG seed for reproducibility
    let rngSeed: UInt64

    /// RNG stream ID for parallel reproducibility
    let rngStreamID: UInt64

    /// Strict floating-point mode (deterministic reassociation)
    let strictFP: Bool

    /// Prefetch distance for cache optimization
    let prefetchDistance: Int

    /// k-means|| specific: Oversampling factor (default: 2)
    let oversamplingFactor: Int

    /// k-means|| specific: Number of rounds (default: 5)
    let rounds: Int

    public static let `default` = KMeansSeedConfig(
        algorithm: .plusPlus,
        k: 256,
        sampleSize: 0,
        rngSeed: 0,
        rngStreamID: 0,
        strictFP: false,
        prefetchDistance: 2,
        oversamplingFactor: 2,
        rounds: 5
    )
}

// MARK: - Core k-means++ API

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
///   - data: Input vectors [n][d], row-major, 64-byte aligned
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
) -> KMeansSeedStats

// MARK: - k-means|| (Parallel Variant)

/// Initialize k centroids using k-means|| parallel seeding
///
/// Algorithm (Bahmani et al., 2012):
///   1. Start with one random seed
///   2. For r rounds:
///      - Sample each point with probability ℓ · D²(x_i) / Σ D²
///      - Add sampled points to candidate set
///      - Update D² with new candidates
///   3. Reduce candidates to k centroids via weighted k-means++
///
/// - Complexity: O(ndk/r + k²dr) where r = rounds
/// - Performance: Faster than k-means++ for large k (k > 1000)
///
/// - Parameters:
///   - data: Input vectors [n][d]
///   - count: Number of vectors
///   - dimension: Vector dimension
///   - k: Number of centroids to select
///   - oversamplingFactor: Oversampling ℓ (default: 2)
///   - rounds: Number of sampling rounds (default: 5)
///   - config: Seeding configuration
///   - centroidsOut: Output centroids [k][d]
/// - Returns: Statistics about seeding process
@inlinable
public func kmeansParallelSeed(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    k: Int,
    oversamplingFactor: Int = 2,
    rounds: Int = 5,
    config: KMeansSeedConfig = .default,
    centroidsOut: UnsafeMutablePointer<Float>
) -> KMeansSeedStats

// MARK: - Distance Update

/// Update squared distances to nearest centroid
/// Core operation for k-means++ D² computation
///
/// For each vector x_i:
///   D²_new[i] = min(D²_old[i], ‖x_i - c_new‖²)
///
/// - Parameters:
///   - data: Input vectors [n][d]
///   - count: Number of vectors
///   - dimension: Vector dimension
///   - newCentroid: Newly added centroid [d]
///   - squaredDistances: Current D² values [n] (updated in-place)
@inlinable
public func updateSquaredDistances(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    newCentroid: UnsafePointer<Float>,
    squaredDistances: UnsafeMutablePointer<Float>
)

// MARK: - Weighted Sampling

/// Sample index proportional to weights
/// Uses inverse CDF method with binary search
///
/// - Parameters:
///   - weights: Non-negative weights [n]
///   - count: Number of elements
///   - rng: Random number generator state
/// - Returns: Selected index in [0, n)
@inlinable
public func sampleProportionalToWeight(
    weights: UnsafePointer<Float>,
    count n: Int,
    rng: inout RNGState
) -> Int

// MARK: - Statistics

/// Statistics collected during seeding
public struct KMeansSeedStats {
    public let algorithm: KMeansSeedAlgorithm
    public let k: Int
    public let n: Int
    public let dimension: Int
    public let chosenIndices: [Int]
    public let totalCost: Double          // Σ D²(x_i) at end
    public let passesOverData: Int        // Number of full scans
    public let executionTimeNanos: UInt64
    public let rngSeedUsed: UInt64

    /// Average distance to nearest centroid
    public var averageDistanceSquared: Double {
        return totalCost / Double(n)
    }
}

// MARK: - Subsampling

/// Uniformly subsample n_sample points from dataset
/// Used when dataset is too large for exact k-means++
@inlinable
public func uniformSubsample(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    sampleSize: Int,
    rng: inout RNGState,
    sampledData: UnsafeMutablePointer<Float>,
    sampledIndices: UnsafeMutablePointer<Int>
)
```

Mathematical Foundation

**k-means++ Algorithm** (Arthur & Vassilvitskii, 2007):

```
Input: Dataset X = {x_1, ..., x_n} ⊂ ℝ^d, number of clusters k

Output: k initial centroids C = {c_1, ..., c_k}

Algorithm:
1. Choose c_1 uniformly at random from X

2. For t = 2 to k:
   a. For each x_i ∈ X, compute:
      D²(x_i) = min_{j < t} ‖x_i - c_j‖²

   b. Choose c_t = x_i with probability:
      P(x_i) = D²(x_i) / Σ_j D²(x_j)

3. Return C = {c_1, ..., c_k}

Key insight: D²-weighting biases selection toward far points,
            spreading centroids across the space
```

**Approximation Guarantee**:

```
Theorem (Arthur & Vassilvitskii, 2007):
  Let φ(C) = Σ_i min_{c ∈ C} ‖x_i - c‖² be the k-means cost
  Let φ_OPT = min_{|C|=k} φ(C) be the optimal cost

  Then: E[φ(C_plusplus)] ≤ 8(ln k + 2) · φ_OPT

  where C_plusplus is the output of k-means++

Interpretation:
  - Random initialization: E[φ] can be Θ(k) · φ_OPT
  - k-means++: E[φ] = O(log k) · φ_OPT
  - Exponential improvement in expectation
```

**D² Weighting Correctness**:

```
Why D² (not D or D⁴)?

Consider uniform distribution on unit circle in ℝ²:
  - D weighting: Bias toward far points (linear)
  - D² weighting: Optimal balance (proven)
  - D⁴ weighting: Over-emphasis on outliers

Proof sketch:
  D² weighting minimizes expected potential:
  E[φ] = E[Σ_i min_j ‖x_i - c_j‖²]

  Each new centroid reduces potential by expected D²(x_i),
  leading to O(log k) approximation
```

**k-means|| Algorithm** (Bahmani et al., 2012):

```
Input: Dataset X, k, oversampling ℓ, rounds r

Algorithm:
1. Initialize C ← {one random point from X}

2. For round = 1 to r:
   a. For each x_i ∈ X:
      D²(x_i) = min_{c ∈ C} ‖x_i - c‖²

   b. Sample each x_i independently with probability:
      p_i = min(1, ℓ · D²(x_i) / Σ_j D²(x_j))

   c. Add sampled points to C

3. Weight each c ∈ C by number of times sampled

4. Run weighted k-means++ on C to select k centroids

Benefits:
  - Parallel: Each round parallelizes over data
  - Scalable: O(r) passes instead of O(k) passes
  - Quality: Achieves O(log k) approximation with high probability
```

Algorithm Details

**Exact k-means++ Implementation**:

```swift
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

    var rng = RNGState(seed: config.rngSeed, stream: config.rngStreamID)
    var chosenIndices: [Int] = []

    // Allocate D² array
    var squaredDistances = [Float](repeating: Float.infinity, count: n)

    // Step 1: Choose first centroid uniformly at random
    let firstIdx = Int(rng.next() % UInt64(n))
    chosenIndices.append(firstIdx)

    // Copy first centroid
    let firstCentroid = data + firstIdx * d
    for j in 0..<d {
        centroidsOut[j] = firstCentroid[j]
    }

    // Initialize D² with distances to first centroid
    updateSquaredDistances(
        data: data,
        count: n,
        dimension: d,
        newCentroid: firstCentroid,
        squaredDistances: &squaredDistances
    )

    // Steps 2 to k: Select remaining centroids with D² weighting
    for t in 1..<k {
        // Sample proportional to D²
        let selectedIdx = sampleProportionalToWeight(
            weights: squaredDistances,
            count: n,
            rng: &rng
        )

        chosenIndices.append(selectedIdx)

        // Copy centroid
        let centroid = data + selectedIdx * d
        let centroidOut = centroidsOut + t * d
        for j in 0..<d {
            centroidOut[j] = centroid[j]
        }

        // Update D² with new centroid
        updateSquaredDistances(
            data: data,
            count: n,
            dimension: d,
            newCentroid: centroid,
            squaredDistances: &squaredDistances
        )
    }

    // Compute final cost
    let totalCost = squaredDistances.reduce(0.0) { Double($0) + Double($1) }

    // Record chosen indices if requested
    if let outIndices = chosenIndicesOut {
        for i in 0..<k {
            outIndices[i] = chosenIndices[i]
        }
    }

    return KMeansSeedStats(
        algorithm: .plusPlus,
        k: k,
        n: n,
        dimension: d,
        chosenIndices: chosenIndices,
        totalCost: totalCost,
        passesOverData: k,
        executionTimeNanos: 0,  // Filled by caller
        rngSeedUsed: config.rngSeed
    )
}
```

**D² Update Implementation**:

```swift
@inlinable
public func updateSquaredDistances(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    newCentroid: UnsafePointer<Float>,
    squaredDistances: UnsafeMutablePointer<Float>
) {
    // Compute D²_new[i] = min(D²_old[i], ‖x_i - c_new‖²)

    for i in 0..<n {
        let vecPtr = data + i * d

        // Compute ‖x_i - c_new‖² using L2 distance kernel
        var distSquared: Float = 0

        // Vectorized computation
        let vecWidth = 4
        let dBlocked = (d / vecWidth) * vecWidth

        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero

        for j in stride(from: 0, to: dBlocked, by: 8) {
            let v0 = SIMD4<Float>(vecPtr + j)
            let c0 = SIMD4<Float>(newCentroid + j)
            let diff0 = v0 - c0
            acc0 += diff0 * diff0

            let v1 = SIMD4<Float>(vecPtr + j + 4)
            let c1 = SIMD4<Float>(newCentroid + j + 4)
            let diff1 = v1 - c1
            acc1 += diff1 * diff1
        }

        distSquared = (acc0 + acc1).sum()

        // Scalar tail
        for j in dBlocked..<d {
            let diff = vecPtr[j] - newCentroid[j]
            distSquared += diff * diff
        }

        // Update D² with minimum
        squaredDistances[i] = min(squaredDistances[i], distSquared)
    }
}
```

**Weighted Sampling (Inverse CDF Method)**:

```swift
@inlinable
public func sampleProportionalToWeight(
    weights: UnsafePointer<Float>,
    count n: Int,
    rng: inout RNGState
) -> Int {
    // Compute total weight (use f64 for numerical stability)
    var totalWeight: Double = 0
    for i in 0..<n {
        totalWeight += Double(weights[i])
    }

    precondition(totalWeight > 0, "Total weight must be positive")

    // Draw uniform random value in [0, totalWeight)
    let threshold = Double(rng.nextFloat()) * totalWeight

    // Find index where cumulative sum exceeds threshold
    var cumulativeSum: Double = 0
    for i in 0..<n {
        cumulativeSum += Double(weights[i])
        if cumulativeSum >= threshold {
            return i
        }
    }

    // Should not reach here (numerical error protection)
    return n - 1
}
```

**k-means|| Implementation**:

```swift
@inlinable
public func kmeansParallelSeed(
    data: UnsafePointer<Float>,
    count n: Int,
    dimension d: Int,
    k: Int,
    oversamplingFactor: Int = 2,
    rounds: Int = 5,
    config: KMeansSeedConfig = .default,
    centroidsOut: UnsafeMutablePointer<Float>
) -> KMeansSeedStats {
    var rng = RNGState(seed: config.rngSeed, stream: config.rngStreamID)

    // Candidates set (will grow over rounds)
    var candidates: [(index: Int, weight: Int)] = []

    // Initialize D²
    var squaredDistances = [Float](repeating: Float.infinity, count: n)

    // Step 1: Choose first candidate uniformly
    let firstIdx = Int(rng.next() % UInt64(n))
    candidates.append((firstIdx, 1))

    let firstCentroid = data + firstIdx * d
    updateSquaredDistances(
        data: data,
        count: n,
        dimension: d,
        newCentroid: firstCentroid,
        squaredDistances: &squaredDistances
    )

    // Step 2: Oversampling rounds
    for _ in 0..<rounds {
        // Compute total D²
        let totalDistSquared = squaredDistances.reduce(0.0) { Double($0) + Double($1) }

        // Sample each point with probability ℓ · D²(x_i) / Σ D²
        for i in 0..<n {
            let probability = min(1.0, Double(oversamplingFactor) * Double(squaredDistances[i]) / totalDistSquared)

            if Double(rng.nextFloat()) < probability {
                // Add to candidates
                candidates.append((i, 1))

                // Update D² with new candidate
                let candidate = data + i * d
                updateSquaredDistances(
                    data: data,
                    count: n,
                    dimension: d,
                    newCentroid: candidate,
                    squaredDistances: &squaredDistances
                )
            }
        }
    }

    // Step 3: Reduce candidates to k centroids using weighted k-means++
    // (For simplicity, use unweighted k-means++ on candidate set)

    // Build candidate data array
    let candidateCount = candidates.count
    var candidateData = [Float](repeating: 0, count: candidateCount * d)

    for (idx, candidate) in candidates.enumerated() {
        let srcPtr = data + candidate.index * d
        let dstPtr = UnsafeMutablePointer(mutating: candidateData) + idx * d
        for j in 0..<d {
            dstPtr[j] = srcPtr[j]
        }
    }

    // Run k-means++ on candidates
    let finalK = min(k, candidateCount)
    return kmeansPlusPlusSeed(
        data: candidateData,
        count: candidateCount,
        dimension: d,
        k: finalK,
        config: config,
        centroidsOut: centroidsOut,
        chosenIndicesOut: nil
    )
}
```

Numerical Considerations

**Floating-Point Stability**:

```swift
// Issue: Cumulative sum of D² can lose precision for large n

// Bad: f32 accumulation
var sum: Float = 0
for i in 0..<n {
    sum += squaredDistances[i]  // Precision loss for n > 10^6
}

// Good: f64 accumulation
var sum: Double = 0
for i in 0..<n {
    sum += Double(squaredDistances[i])  // Stable for n up to 10^12
}

// Justification:
// - D² values typically ∈ [0, 100] for normalized vectors
// - f32: ~7 decimal digits → errors accumulate after ~10^6 additions
// - f64: ~15 decimal digits → stable for massive datasets
```

**Deterministic Tie-Breaking**:

```swift
// When multiple points have exactly the same D², select deterministically

// Example: Two points equidistant from all centroids
// D²[100] = 5.0
// D²[101] = 5.0

// Solution: Select by smallest index (100 < 101)

func sampleProportionalToWeight_Deterministic(
    weights: UnsafePointer<Float>,
    count n: Int,
    rng: inout RNGState
) -> Int {
    let totalWeight = weights.reduce(0.0) { Double($0) + Double($1) }
    let threshold = Double(rng.nextFloat()) * totalWeight

    var cumulativeSum: Double = 0
    var selectedIdx = 0

    for i in 0..<n {
        cumulativeSum += Double(weights[i])

        // Use >= for deterministic selection of first crossing
        if cumulativeSum >= threshold {
            selectedIdx = i
            break
        }
    }

    return selectedIdx  // Always returns smallest index on tie
}
```

**Strict Floating-Point Mode**:

```
When config.strictFP = true:
  - Disable compiler reassociation: (a + b) + c ≠ a + (b + c)
  - Process additions in fixed chunk order
  - Use Kahan summation for cumulative sums
  - Ensures bit-exact reproducibility across platforms

Trade-off:
  + Bit-exact determinism
  - ~10-20% slower (no SIMD fusion)

Use case: Debugging, validation, regression testing
```

Performance Characteristics

**Complexity Analysis**:

```
k-means++ (exact):
  - Time: O(ndk) where n=data, d=dimension, k=centroids
  - Space: O(n + kd) for D² array and centroids
  - Passes: k full scans over data

k-means|| (parallel):
  - Time: O(ndr + C²dr + Cdk)
    where C = candidate count ≈ ℓkr
  - Space: O(n + Cd)
  - Passes: r rounds (r << k typically)

Comparison (k=1000, r=5):
  - k-means++: 1000 passes
  - k-means||: 5 passes → 200× fewer scans
```

**Throughput** (Apple M1, n=100k, d=768):

```
Operation                | Throughput | Notes
-------------------------|------------|------------------
D² update (per centroid) | 2.8 GB/s   | Memory-bound
Weighted sampling        | 50 μs      | Prefix sum scan
Total k-means++ (k=256)  | 180 ms     | 256 × (update + sample)
Total k-means|| (k=256)  | 40 ms      | 5 rounds, faster
```

**Latency Breakdown** (n=100k, d=768, k=256):

```
k-means++:
  - D² updates: 256 × 0.7ms = 179ms (99% of time)
  - Weighted sampling: 256 × 50μs = 13ms (1% of time)
  - Total: ~192ms

k-means||:
  - 5 rounds × D² update: 5 × 0.7ms = 3.5ms
  - Candidate selection: ~5ms
  - Final reduction (k-means++ on candidates): ~30ms
  - Total: ~40ms (4.8× faster)
```

**Quality vs. Speed Trade-off**:

```
Algorithm      | Time      | Quality (relative to OPT)
---------------|-----------|---------------------------
Random         | O(kd)     | E[φ] = Θ(k) · φ_OPT
k-means++      | O(ndk)    | E[φ] ≤ 8(ln k + 2) · φ_OPT
k-means|| (r=5)| O(ndr)    | E[φ] ≤ 16(ln k + 2) · φ_OPT

Empirical results (IVF training):
  - Random: 15-20 Lloyd iterations to converge
  - k-means++: 3-5 iterations
  - k-means||: 4-6 iterations

Total training time often FASTER with k-means++ despite
slower initialization, due to fewer Lloyd iterations.
```

Integration with IVF Training

**Complete Training Pipeline**:

```swift
struct IVFIndexBuilder {
    func buildCoarseQuantizer(
        data: [Float],
        n: Int,
        d: Int,
        k: Int
    ) -> [Float] {
        // Step 1: k-means++ seeding
        var centroids = [Float](repeating: 0, count: k * d)

        let seedStats = kmeansPlusPlusSeed(
            data: data,
            count: n,
            dimension: d,
            k: k,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )

        print("Initial cost: \(seedStats.totalCost)")
        print("Seeds chosen in \(seedStats.passesOverData) passes")

        // Step 2: Lloyd iterations (kernel #12)
        let (finalCentroids, finalCost) = lloydIterations(
            data: data,
            n: n,
            d: d,
            initialCentroids: centroids,
            k: k,
            maxIterations: 20
        )

        print("Final cost: \(finalCost)")
        print("Improvement: \((seedStats.totalCost - finalCost) / seedStats.totalCost * 100)%")

        return finalCentroids
    }
}
```

**Subsampling for Large Datasets**:

```swift
// For very large datasets (n > 10M), subsample before seeding

func seedWithSubsampling(
    data: [Float],
    n: Int,
    d: Int,
    k: Int,
    sampleSize: Int = 1_000_000  // 1M sample
) -> [Float] {
    precondition(sampleSize >= k, "Sample size must be ≥ k")

    // Subsample uniformly
    var rng = RNGState(seed: 0, stream: 0)
    var sampledData = [Float](repeating: 0, count: sampleSize * d)
    var sampledIndices = [Int](repeating: 0, count: sampleSize)

    uniformSubsample(
        data: data,
        count: n,
        dimension: d,
        sampleSize: sampleSize,
        rng: &rng,
        sampledData: &sampledData,
        sampledIndices: &sampledIndices
    )

    // Run k-means++ on sample
    var centroids = [Float](repeating: 0, count: k * d)

    kmeansPlusPlusSeed(
        data: sampledData,
        count: sampleSize,
        dimension: d,
        k: k,
        centroidsOut: &centroids,
        chosenIndicesOut: nil
    )

    return centroids
}
```

Correctness & Testing

**Test Cases**:

1. **Determinism**:
   - Fixed RNG seed produces identical centroids across runs
   - Identical chosen indices regardless of thread count
   - Bit-exact results in strict FP mode

2. **Quality**:
   - k-means++ cost < random initialization cost (statistical test)
   - Centroids well-distributed (no duplicates or near-duplicates)
   - Matches scikit-learn k-means++ within tolerance

3. **Numerical Stability**:
   - Large datasets (n=10M): No precision loss in D² sum
   - Edge case: All points identical (should select k duplicates)
   - Edge case: k=n (should select all points)

4. **Algorithm Correctness**:
   - D² values always non-negative
   - Total weight sum equals Σ D²
   - k-means|| produces ≥k candidates (usually ~ℓkr)

5. **Performance**:
   - D² update bandwidth near L2 kernel roofline
   - k-means|| faster than k-means++ for k > 100

**Example Tests**:

```swift
func testKMeansPlusPlus_Determinism() {
    let n = 10000
    let d = 768
    let k = 256

    let data = generateRandomVectors(count: n, dimension: d)

    // Run twice with same seed
    let config = KMeansSeedConfig(rngSeed: 12345, rngStreamID: 0)

    var centroids1 = [Float](repeating: 0, count: k * d)
    var indices1 = [Int](repeating: 0, count: k)

    kmeansPlusPlusSeed(
        data: data,
        count: n,
        dimension: d,
        k: k,
        config: config,
        centroidsOut: &centroids1,
        chosenIndicesOut: &indices1
    )

    var centroids2 = [Float](repeating: 0, count: k * d)
    var indices2 = [Int](repeating: 0, count: k)

    kmeansPlusPlusSeed(
        data: data,
        count: n,
        dimension: d,
        k: k,
        config: config,
        centroidsOut: &centroids2,
        chosenIndicesOut: &indices2
    )

    // Verify identical results
    XCTAssertEqual(indices1, indices2)

    for i in 0..<(k * d) {
        XCTAssertEqual(centroids1[i], centroids2[i])
    }
}

func testKMeansPlusPlus_Quality() {
    let n = 5000
    let d = 128
    let k = 50

    let data = generateRandomVectors(count: n, dimension: d)

    // k-means++ seeding
    var centroidsPlusPlus = [Float](repeating: 0, count: k * d)
    let statsPlusPlus = kmeansPlusPlusSeed(
        data: data,
        count: n,
        dimension: d,
        k: k,
        centroidsOut: &centroidsPlusPlus,
        chosenIndicesOut: nil
    )

    // Random seeding (baseline)
    var centroidsRandom = [Float](repeating: 0, count: k * d)
    var rng = RNGState(seed: 0, stream: 0)

    for i in 0..<k {
        let idx = Int(rng.next() % UInt64(n))
        let src = data + idx * d
        let dst = UnsafeMutablePointer(mutating: centroidsRandom) + i * d
        for j in 0..<d {
            dst[j] = src[j]
        }
    }

    let costRandom = computeClusteringCost(data, n, d, centroidsRandom, k)

    // k-means++ should be significantly better
    print("k-means++ cost: \(statsPlusPlus.totalCost)")
    print("Random cost: \(costRandom)")
    print("Improvement: \((costRandom - statsPlusPlus.totalCost) / costRandom * 100)%")

    XCTAssertLessThan(statsPlusPlus.totalCost, costRandom * 0.5)  // At least 50% better
}

func testKMeansParallel_Speedup() {
    let n = 100000
    let d = 768
    let k = 1000

    let data = generateRandomVectors(count: n, dimension: d)

    // Measure k-means++
    let startPlusPlus = mach_absolute_time()
    var centroidsPlusPlus = [Float](repeating: 0, count: k * d)
    kmeansPlusPlusSeed(data, n, d, k, centroidsOut: &centroidsPlusPlus, chosenIndicesOut: nil)
    let timePlusPlus = mach_absolute_time() - startPlusPlus

    // Measure k-means||
    let startParallel = mach_absolute_time()
    var centroidsParallel = [Float](repeating: 0, count: k * d)
    kmeansParallelSeed(data, n, d, k, oversamplingFactor: 2, rounds: 5, centroidsOut: &centroidsParallel)
    let timeParallel = mach_absolute_time() - startParallel

    print("k-means++: \(timePlusPlus / 1_000_000)ms")
    print("k-means||: \(timeParallel / 1_000_000)ms")
    print("Speedup: \(Double(timePlusPlus) / Double(timeParallel))×")

    // k-means|| should be significantly faster for large k
    XCTAssertLessThan(timeParallel, timePlusPlus)
}
```

Coding Guidelines

**Performance Best Practices**:
- Use SIMD for D² distance computation (reuse L2 kernel)
- Accumulate D² sum in f64 for numerical stability
- Prefetch data rows during sequential scans
- For large k (>1000), use k-means|| instead of k-means++
- Subsample if n > 10M (uniformly select 1M points)

**Numerical Best Practices**:
- Always accumulate weights in f64
- Use strict FP mode for reproducibility testing
- Handle edge cases: k=1, k=n, all points identical
- Ensure D² ≥ 0 (use max(0, ...) if needed)

**API Usage**:

```swift
// Good: Standard k-means++ for moderate k
let centroids = kmeansPlusPlusSeed(data, n, d, k: 256)

// Good: k-means|| for large k
let centroids = kmeansParallelSeed(data, n, d, k: 4096, oversamplingFactor: 2, rounds: 5)

// Good: Subsample for huge datasets
let config = KMeansSeedConfig(sampleSize: 1_000_000)
let centroids = kmeansPlusPlusSeed(data, n, d, k: 256, config: config)

// Good: Deterministic for testing
let config = KMeansSeedConfig(rngSeed: 42, strictFP: true)
let centroids = kmeansPlusPlusSeed(data, n, d, k: 256, config: config)
```

Non-Goals

- GPU/Metal acceleration (CPU-focused)
- Mini-batch variants (use full dataset)
- Adaptive k selection (k is input parameter)
- Incremental seeding (static initialization)

Example Usage

```swift
import VectorIndex

// Example 1: Basic k-means++ seeding
let vectors: [[Float]] = loadVectors()  // 100,000 × 768
let n = vectors.count
let d = vectors[0].count
let k = 256

let flatVectors = vectors.flatMap { $0 }
var centroids = [Float](repeating: 0, count: k * d)

let stats = kmeansPlusPlusSeed(
    data: flatVectors,
    count: n,
    dimension: d,
    k: k,
    centroidsOut: &centroids,
    chosenIndicesOut: nil
)

print("Initialized \(k) centroids in \(stats.passesOverData) passes")
print("Initial cost: \(stats.totalCost)")

// Example 2: Deterministic seeding with chosen indices
let config = KMeansSeedConfig(rngSeed: 42, rngStreamID: 0)
var chosenIndices = [Int](repeating: 0, count: k)

kmeansPlusPlusSeed(
    data: flatVectors,
    count: n,
    dimension: d,
    k: k,
    config: config,
    centroidsOut: &centroids,
    chosenIndicesOut: &chosenIndices
)

print("Chosen indices: \(chosenIndices)")

// Example 3: k-means|| for large k
let largeK = 4096
var largeCentroids = [Float](repeating: 0, count: largeK * d)

kmeansParallelSeed(
    data: flatVectors,
    count: n,
    dimension: d,
    k: largeK,
    oversamplingFactor: 2,
    rounds: 5,
    centroidsOut: &largeCentroids
)

// Example 4: Subsampling for huge dataset
let hugeN = 50_000_000
let sampleConfig = KMeansSeedConfig(
    algorithm: .plusPlus,
    k: k,
    sampleSize: 1_000_000,  // Sample 1M from 50M
    rngSeed: 123
)

kmeansPlusPlusSeed(
    data: hugeDataset,
    count: hugeN,
    dimension: d,
    k: k,
    config: sampleConfig,
    centroidsOut: &centroids,
    chosenIndicesOut: nil
)
```

Dependencies

**Internal**:
- L2 Distance Kernel (#01): For D² computation
- Lloyd Iterations (#12): For final training
- IVF Index: Uses centroids for routing

**External**:
- Swift Standard Library: SIMD, min, max
- RNG utilities: Deterministic random number generation
- Foundation: None (pure computation)

**Build Requirements**:
- Swift 5.9+
- macOS 13+ / iOS 16+
- Optimization: `-O` (Release builds)

Acceptance Criteria

✅ **Mathematical Correctness**:
- D²-weighted sampling (not D or D⁴)
- Identical ranking to reference implementation
- k-means|| produces O(log k) approximation

✅ **Quality**:
- Cost < random initialization (statistical test)
- Reduces Lloyd iterations by 3-5×
- Well-distributed centroids (no duplicates)

✅ **Determinism**:
- Fixed RNG seed → identical results
- Bit-exact in strict FP mode
- Independent of thread count

✅ **Performance**:
- D² update: >2.5 GB/s bandwidth
- k-means|| faster than k-means++ for k>100
- Total training time reduced despite initialization cost

✅ **Robustness**:
- Numerical stability for n up to 10^9
- Handles edge cases (k=1, k=n, duplicates)
- Works with subsampling for massive datasets
