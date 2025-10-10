//
//  KMeansMiniBatchKernel.swift
//  VectorIndex
//
//  Kernel #12: Mini-batch & Streaming K-means (IVF coarse quantizer training / maintenance)
//
//  Features:
//   - Lloyd Mini-batch training (seeded via Kernel #11)
//   - Streaming EWMA updates with optional adaptive decay
//   - Deterministic RNG; per-epoch subsampling + shuffle
//   - Sparse per-batch accumulators (no kc×d temporaries)
//   - Empty-cluster repair (split-largest deterministic strategy)
//   - Inertia tracking + early stopping
//
//  Swift 6 compliance:
//   • All public types conform to Sendable
//   • @unchecked Sendable only for the state class (mutable internal buffers, single-writer)
//   • @inlinable for hot paths; @usableFromInline for helpers accessed by inlinable fn
//   • SIMD loads use UnsafeRawPointer.load(as:) (no internal SIMD initializers)
//   • No deprecated .assign(from:); use .update(from:count:)
//   • Standard arithmetic (+,-,*) for math; wrapping (&+,&-,&*) only for RNG/counters
//
//  Thread safety:
//   • One-shot trainer is pure (no shared state).
//   • KMeansState is single-writer; if sharing across threads, synchronize externally.
//
//  Spec references:
//   - #12: Mini-batch / Streaming K-means (kernel-specs/12_kmeans_minibatch.md)
//   - #11: K-means++ seeding (kernel-specs/11_kmeanspp_seed.md)
//

import Foundation

// MARK: - Public enums / config

@frozen
public enum KMeansAlgorithmMB: Sendable {
    case lloydMiniBatch
    case onlineEWMA
}

@frozen
public enum VectorLayoutMB: Sendable {
    case aos         // [n × d] row-major (fast path)
    case aosoaR      // [n/R][d][R] (correctness path in this kernel)
}

@frozen
public struct KMeansMBConfig: Sendable {
    public let algo: KMeansAlgorithmMB
    public let batchSize: Int
    public let epochs: Int
    public let subsampleN: Int64       // 0 => use full n
    public let tol: Float              // relative inertia improvement
    public let decay: Float            // EWMA (ignored for mini-batch)
    public let seed: UInt64
    public let streamID: UInt64
    public let prefetchDistance: Int   // NOTE: Currently no-op, infrastructure for future optimization
    public let layout: VectorLayoutMB
    public let aosoaRegisterBlock: Int // R (when layout == .aosoaR)
    public let computeAssignments: Bool

    @inlinable
    public init(
        algo: KMeansAlgorithmMB = .lloydMiniBatch,
        batchSize: Int = 1024,
        epochs: Int = 10,
        subsampleN: Int64 = 0,
        tol: Float = 1e-4,
        decay: Float = 0.01,
        seed: UInt64 = 0,
        streamID: UInt64 = 0,
        prefetchDistance: Int = 8,
        layout: VectorLayoutMB = .aos,
        aosoaRegisterBlock: Int = 0,
        computeAssignments: Bool = false
    ) {
        self.algo = algo
        self.batchSize = batchSize
        self.epochs = epochs
        self.subsampleN = subsampleN
        self.tol = tol
        self.decay = decay
        self.seed = seed
        self.streamID = streamID
        self.prefetchDistance = prefetchDistance
        self.layout = layout
        self.aosoaRegisterBlock = aosoaRegisterBlock
        self.computeAssignments = computeAssignments
    }

    public static let `default` = KMeansMBConfig()
}

@frozen
public struct KMeansMBStats: Sendable {
    public let epochsCompleted: Int
    public let batchesProcessed: Int64
    public let rowsSeen: Int64
    public let emptiesRepaired: Int64
    public let inertiaPerEpoch: [Double]
    public let finalInertia: Double
    public let timeInitSec: Double
    public let timeTrainingSec: Double
    public let timeAssignmentSec: Double
    public let bytesRead: Int64

    @inlinable
    public init(
        epochsCompleted: Int,
        batchesProcessed: Int64,
        rowsSeen: Int64,
        emptiesRepaired: Int64,
        inertiaPerEpoch: [Double],
        finalInertia: Double,
        timeInitSec: Double,
        timeTrainingSec: Double,
        timeAssignmentSec: Double,
        bytesRead: Int64
    ) {
        self.epochsCompleted = epochsCompleted
        self.batchesProcessed = batchesProcessed
        self.rowsSeen = rowsSeen
        self.emptiesRepaired = emptiesRepaired
        self.inertiaPerEpoch = inertiaPerEpoch
        self.finalInertia = finalInertia
        self.timeInitSec = timeInitSec
        self.timeTrainingSec = timeTrainingSec
        self.timeAssignmentSec = timeAssignmentSec
        self.bytesRead = bytesRead
    }
}

// Return status (C-like codes for bridging)
@frozen
public enum KMeansMBStatus: Int32, Sendable {
    case success = 0
    case invalidDim = -1
    case invalidK = -2
    case nullPtr = -3
    case invalidLayout = -4
    case noConvergence = 1
}

// MARK: - Streaming state (EWMA)

/// Streaming k-means state for online EWMA updates.
///
/// Single-writer model: If sharing across threads, synchronize externally.
/// Mutable buffers are internal; @unchecked Sendable is safe for single-writer usage.
public final class KMeansState: @unchecked Sendable {
    @usableFromInline internal let d: Int
    @usableFromInline internal let kc: Int
    @usableFromInline internal var centroids: [Float]           // [kc × d]
    @usableFromInline internal var counts: [Int64]              // [kc]
    @usableFromInline internal var decay: Float                 // default η
    @usableFromInline internal var totalUpdates: Int64

    @inlinable
    public init(d: Int, kc: Int, initCentroids: UnsafePointer<Float>, decay: Float) {
        self.d = d
        self.kc = kc
        self.decay = decay
        self.totalUpdates = 0
        self.centroids = [Float](repeating: 0, count: kc * d)
        self.counts = [Int64](repeating: 0, count: kc)
        // Copy initial centroids
        self.centroids.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.update(from: initCentroids, count: kc * d)
        }
    }

    @inlinable
    public func finalize(into out: UnsafeMutablePointer<Float>) {
        // Copy centroids out
        out.update(from: centroids, count: kc * d)
    }
}

// MARK: - Internal helpers

@usableFromInline
internal func _vi_km12_nowNanos() -> UInt64 {
    return DispatchTime.now().uptimeNanoseconds
}

@usableFromInline
@inline(__always)
internal func _vi_km12_load4(_ base: UnsafePointer<Float>, _ offset: Int) -> SIMD4<Float> {
    let raw = UnsafeRawPointer(base.advanced(by: offset))
    return raw.load(as: SIMD4<Float>.self)
}

@usableFromInline
@inline(__always)
internal func _vi_km12_sum4(_ v: SIMD4<Float>) -> Float { v[0] + v[1] + v[2] + v[3] }

/// Prefetch hint (no-op for now; infrastructure for future platform-specific optimization)
///
/// Future: Could use __builtin_prefetch (ARM) or _mm_prefetch (x86) via C bridge.
/// Current: Documentation placeholder to preserve API surface.
@usableFromInline
@inline(__always)
internal func _vi_km12_prefetch(_ ptr: UnsafeRawPointer?) {
    _ = ptr
    // TODO: Implement platform-specific prefetch
    // #if arch(arm64)
    //   __builtin_prefetch(ptr, 0, 0)
    // #elseif arch(x86_64)
    //   _mm_prefetch(ptr, _MM_HINT_T0)
    // #endif
}

/// L2 squared distance: AoS fast path (8-wide unroll using SIMD4)
///
/// Computes ‖a - b‖² = Σ(a[i] - b[i])²
/// Uses dual SIMD4 accumulators for ILP (instruction-level parallelism)
@inlinable
public func _vi_km12_l2sq_aos(
    _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int
) -> Float {
    var acc0 = SIMD4<Float>(repeating: 0)
    var acc1 = SIMD4<Float>(repeating: 0)
    let d8 = (d / 8) * 8
    var j = 0
    while j < d8 {
        let a0 = _vi_km12_load4(a, j)
        let b0 = _vi_km12_load4(b, j)
        let d0 = a0 - b0  // ✓ Standard arithmetic for math
        acc0 = acc0 + d0 * d0

        let a1 = _vi_km12_load4(a, j + 4)
        let b1 = _vi_km12_load4(b, j + 4)
        let d1 = a1 - b1  // ✓ Standard arithmetic
        acc1 = acc1 + d1 * d1

        j += 8
    }
    var sum = _vi_km12_sum4(acc0) + _vi_km12_sum4(acc1)
    while j < d {
        let df = a[j] - b[j]  // ✓ Standard arithmetic
        sum += df * df
        j += 1
    }
    return sum
}

/// Scalar element access for AoSoA_R (correctness path)
///
/// Layout: [n/R][d][R] ⇒ element(i,j) at group = i/R, lane = i%R:
///   base[group*d*R + j*R + lane]
@usableFromInline
@inline(__always)
internal func _vi_km12_aosoa_elem(
    base: UnsafePointer<Float>, d: Int, r: Int, i: Int64, j: Int
) -> Float {
    let group = Int(i / Int64(r))
    let lane = Int(i % Int64(r))
    let offset = group * d * r + j * r + lane
    return base[offset]
}

/// Compute L2 for a single vector (AoSoA): strided scalar loop (correctness > speed)
@usableFromInline
internal func _vi_km12_l2sq_aosoa(
    _ x: UnsafePointer<Float>, _ c: UnsafePointer<Float>, nIndex: Int64, d: Int, R: Int
) -> Float {
    var s: Float = 0
    for j in 0..<d {
        let xv = _vi_km12_aosoa_elem(base: x, d: d, r: R, i: nIndex, j: j)
        let df = xv - c[j]  // ✓ Standard arithmetic
        s += df * df
    }
    return s
}

/// Deterministic Fisher–Yates shuffle
@usableFromInline
internal func _vi_km12_shuffle(_ idx: inout [Int64], rng: inout RNGState) {
    if idx.count <= 1 { return }
    for i in stride(from: idx.count - 1, through: 1, by: -1) {
        let j = rng.nextInt(bound: i + 1)
        if i != j { idx.swapAt(i, j) }
    }
}

/// Reservoir sampling of M indices from [0, n)
///
/// Algorithm R (Vitter, 1985): O(n) single-pass sampling
@usableFromInline
internal func _vi_km12_reservoir(_ n: Int64, _ m: Int64, rng: inout RNGState) -> [Int64] {
    var res = [Int64]()
    res.reserveCapacity(Int(m))
    for i in 0..<m { res.append(i) }
    var i: Int64 = m
    while i < n {
        let j = Int64(rng.next() % UInt64(i + 1))  // ✓ Wrapping ok for RNG
        if j < m { res[Int(j)] = i }
        i += 1
    }
    return res
}

// MARK: - Empty-cluster repair (split-largest: deterministic)

/// Repair empty clusters by splitting the largest cluster
///
/// Strategy: Find centroid with most assignments, split it by selecting
/// the farthest point in the current batch. Deterministic tie-breaking.
@usableFromInline
internal func _vi_km12_repairEmpties_splitLargest(
    xBatchIdx: UnsafePointer<Int64>,           // indices of rows in the current batch
    batchCount: Int,
    X: UnsafePointer<Float>, n: Int64, d: Int,
    C: UnsafeMutablePointer<Float>, kc: Int,
    counts: UnsafePointer<Int>,                // per-centroid counts for this batch
    empties: UnsafePointer<Int>, emptiesCount: Int
) {
    // Find c_max with largest N_c
    var cMax = 0
    var nMax = -1
    for c in 0..<kc {
        let nc = counts[c]
        if nc > nMax { nMax = nc; cMax = c }
    }
    // Safety: if all zero (degenerate tiny batch), pick cMax = 0
    if nMax <= 0 { cMax = 0 }

    // Find farthest point in the batch from C[cMax]
    var farIdx = 0
    var farDist = -Float.infinity
    let cMaxPtr = C.advanced(by: cMax * d)

    for bi in 0..<batchCount {
        let gi = xBatchIdx[bi]
        let vec = X.advanced(by: Int(gi) * d)
        let dist = _vi_km12_l2sq_aos(vec, cMaxPtr, d)
        if dist > farDist {
            farDist = dist
            farIdx = bi
        }
    }

    // Deterministic repair: fill empties in ascending centroid order
    for e in 0..<emptiesCount {
        let ce = empties[e]
        let gi = xBatchIdx[farIdx]
        let vec = X.advanced(by: Int(gi) * d)
        let dst = C.advanced(by: ce * d)
        for j in 0..<d { dst[j] = vec[j] }
    }
}

// MARK: - Assignment (tiling over centroids)

/// Assign vector to nearest centroid using tiled scan
///
/// Cache-friendly: processes centroids in tiles of size `tile` to maintain
/// hot data in L1/L2 cache. Tile size 32 chosen empirically for typical d.
@usableFromInline
internal func _vi_km12_assignAOS_tiled(
    xVec: UnsafePointer<Float>,
    C: UnsafePointer<Float>, kc: Int, d: Int,
    tile: Int
) -> (cBest: Int, distBest: Float) {
    var cBest = 0
    var distBest = _vi_km12_l2sq_aos(xVec, C, d)

    var c = 1
    while c < kc {
        let end = min(c + tile, kc)
        // (Optional) prefetch next tile
        if end < kc {
            _vi_km12_prefetch(UnsafeRawPointer(C.advanced(by: end * d)))
        }
        while c < end {
            let dist = _vi_km12_l2sq_aos(xVec, C.advanced(by: c * d), d)
            // Deterministic tie-breaking: prefer lower centroid index
            if dist < distBest || (dist == distBest && c < cBest) {
                distBest = dist
                cBest = c
            }
            c += 1
        }
    }
    return (cBest, distBest)
}

@usableFromInline
internal func _vi_km12_assignAOSOA(
    X: UnsafePointer<Float>, nIndex: Int64,
    C: UnsafePointer<Float>, kc: Int, d: Int, R: Int
) -> (cBest: Int, distBest: Float) {
    var cBest = 0
    var distBest = _vi_km12_l2sq_aosoa(X, C, nIndex: nIndex, d: d, R: R)
    var c = 1
    while c < kc {
        let dist = _vi_km12_l2sq_aosoa(X, C.advanced(by: c * d), nIndex: nIndex, d: d, R: R)
        if dist < distBest || (dist == distBest && c < cBest) {
            distBest = dist
            cBest = c
        }
        c += 1
    }
    return (cBest, distBest)
}

// MARK: - Public API: One-shot Mini-batch training

/// One-shot mini-batch/online k-means training / maintenance for IVF coarse quantizers.
///
/// Algorithm:
///   - Mini-batch Lloyd's: per-batch mean updates with empty-cluster repair
///   - EWMA streaming: exponentially weighted moving average with adaptive decay
///
/// Initialization:
///   - If `initCentroids == nil`, uses k-means++ (Kernel #11) with `cfg.seed`
///   - Otherwise, warm-starts from provided centroids
///
/// Returns `KMeansMBStatus` and fills `statsOut` if provided.
///
/// NOTE: This kernel is intentionally single-threaded for determinism.
///       For parallelism, shard batches externally and merge deterministically.
///
/// - Complexity: O(epochs × n × kc × d) for full Lloyd's
/// - Memory: O(kc × d + batch_size × d) for centroids + sparse accumulators
@inlinable
@discardableResult
public func kmeans_minibatch_f32(
    x: UnsafePointer<Float>,                // [n × d] (AoS) or [n/R][d][R] (AoSoA)
    n: Int64,
    d: Int,
    kc: Int,
    initCentroids: UnsafePointer<Float>?,   // [kc × d] or nil
    cfg: KMeansMBConfig = .default,
    centroidsOut: UnsafeMutablePointer<Float>,  // [kc × d]
    assignOut: UnsafeMutablePointer<Int32>? = nil, // [n] or nil
    statsOut: UnsafeMutablePointer<KMeansMBStats>? = nil
) -> KMeansMBStatus {
    // --- Validate
    if d < 1 || d > 32768 { return .invalidDim }
    if kc < 1 || Int64(kc) > n { return .invalidK }
    if centroidsOut == UnsafeMutablePointer<Float>(bitPattern: 0) { return .nullPtr }

    // ✓ FIX: AoSoA layout requires valid register block
    if cfg.layout == .aosoaR && cfg.aosoaRegisterBlock < 1 {
        return .invalidLayout
    }

    // --- Initialization (seed or copy)
    let tInit0 = _vi_km12_nowNanos()

    if let seeds = initCentroids {
        // Copy provided seeds
        centroidsOut.update(from: seeds, count: kc * d)
    } else {
        // Seed with k-means++ (Kernel #11)
        let seedCfg = KMeansSeedConfig(
            algorithm: .plusPlus, k: kc,
            sampleSize: 0, rngSeed: cfg.seed, rngStreamID: cfg.streamID,
            strictFP: false, prefetchDistance: 2, oversamplingFactor: 2, rounds: 5
        )
        _ = kmeansPlusPlusSeed(
            data: x, count: Int(n), dimension: d, k: kc,
            config: seedCfg, centroidsOut: centroidsOut, chosenIndicesOut: nil
        )
    }
    let tInit1 = _vi_km12_nowNanos()

    // Early exit for streaming mode (we still support one-shot EWMA on x)
    if cfg.algo == .onlineEWMA {
        // Build state and run a single streaming pass
        let state = KMeansState(d: d, kc: kc, initCentroids: centroidsOut, decay: cfg.decay)
        var opts = KMeansUpdateOpts(decay: cfg.decay, normalizeCentroids: false, prefetchDistance: cfg.prefetchDistance, adaptiveDecay: true)
        _ = kmeans_state_update_chunk(state: state, x_chunk: x, m: n, layout: cfg.layout, aosoaRegisterBlock: cfg.aosoaRegisterBlock, opts: &opts, centroidsOut: centroidsOut)
        state.finalize(into: centroidsOut)

        // Stats: EWMA does not compute inertia (expensive)
        let stats = KMeansMBStats(
            epochsCompleted: 1,
            batchesProcessed: n,
            rowsSeen: n,
            emptiesRepaired: 0,
            inertiaPerEpoch: [],
            finalInertia: 0,  // Not computed for EWMA
            timeInitSec: Double(tInit1 - tInit0) * 1e-9,
            timeTrainingSec: 0,
            timeAssignmentSec: 0,
            bytesRead: Int64(n) * Int64(d) * 4
        )
        if let p = statsOut { p.pointee = stats }
        return .success
    }

    // --- Mini-batch Lloyd training
    var rng = RNGState(seed: cfg.seed, stream: cfg.streamID)

    // Sparse accumulators: only allocate for touched centroids
    // Memory: touched·d instead of kc·d (90%+ savings for large k)
    let maxTouched = min(cfg.batchSize, kc)
    var sums = [Double](repeating: 0, count: maxTouched * d)
    var touchedList = [Int](repeating: 0, count: maxTouched)
    var sumIndex = [Int](repeating: -1, count: kc)
    var batchTag = [UInt32](repeating: 0, count: kc)
    var currentTag: UInt32 = 1
    var batchCounts = [Int](repeating: 0, count: kc)

    // Stats
    var inertiaHistory = [Double]()
    inertiaHistory.reserveCapacity(max(1, cfg.epochs))
    var emptiesFixed: Int64 = 0
    var batchesProcessed: Int64 = 0
    var rowsSeen: Int64 = 0

    let tTrain0 = _vi_km12_nowNanos()

    // Epoch loop
    var prevInertia: Double = .infinity
    var epochsDone = 0

    epochLoop: for epoch in 0..<max(1, cfg.epochs) {
        epochsDone = epoch + 1

        // Build epoch index set (subsample or full)
        let nEpoch: Int64
        var indices: [Int64]
        if cfg.subsampleN > 0 && cfg.subsampleN < n {
            // Sample without replacement via reservoir
            indices = _vi_km12_reservoir(n, cfg.subsampleN, rng: &rng)
            _vi_km12_shuffle(&indices, rng: &rng)
            nEpoch = cfg.subsampleN
        } else {
            // Streaming mini-batches by random-with-replacement
            indices = []
            nEpoch = n
        }

        let tile = 32  // Cache tile size (empirically optimal for d ∈ [128, 2048])

        // Process mini-batches
        var processed: Int64 = 0
        while processed < nEpoch {
            // Determine batch set of indices
            let remaining = Int(nEpoch - processed)
            let batchCount = min(cfg.batchSize, remaining)
            var xBatchIdx = [Int64](repeating: 0, count: batchCount)

            if indices.isEmpty {
                // Sample with replacement
                for bi in 0..<batchCount {
                    xBatchIdx[bi] = Int64(rng.next() % UInt64(n))  // ✓ Wrapping ok for RNG
                }
            } else {
                // Slice from shuffled reservoir
                for bi in 0..<batchCount { xBatchIdx[bi] = indices[Int(processed) + bi] }
            }

            // Reset sparse accumulators (O(1) via tag bump)
            currentTag = currentTag &+ 1  // ✓ Wrapping ok for tag counter
            var touched = 0

            // Assignment + accumulation
            for bi in 0..<batchCount {
                let gi = xBatchIdx[bi]

                // Assign
                let (cBest, _) : (Int, Float) = {
                    switch cfg.layout {
                    case .aos:
                        let vec = x.advanced(by: Int(gi) * d)
                        return _vi_km12_assignAOS_tiled(xVec: vec, C: centroidsOut, kc: kc, d: d, tile: tile)
                    case .aosoaR:
                        let R = cfg.aosoaRegisterBlock  // Already validated > 0
                        return _vi_km12_assignAOSOA(X: x, nIndex: gi, C: centroidsOut, kc: kc, d: d, R: R)
                    }
                }()

                // Get/allocate touched slot for centroid
                if batchTag[cBest] != currentTag {
                    // New touched centroid
                    if touched >= maxTouched {
                        // Defensive: enlarge sums/touched up to batchCount
                        let newCap = min(cfg.batchSize, touched + max(1, cfg.batchSize / 2))
                        if newCap > sums.count / d {
                            var newSums = [Double](repeating: 0, count: newCap * d)
                            for t in 0..<(touched * d) { newSums[t] = sums[t] }
                            sums = newSums
                            touchedList += [Int](repeating: 0, count: newCap - touchedList.count)
                        }
                    }
                    batchTag[cBest] = currentTag
                    sumIndex[cBest] = touched
                    touchedList[touched] = cBest
                    // zero the newly exposed sums slot
                    let base = touched * d
                    for j in 0..<d { sums[base + j] = 0 }
                    touched += 1
                }

                // Accumulate vector into sums[cBest]
                let slot = sumIndex[cBest]
                let base = slot * d
                switch cfg.layout {
                case .aos:
                    let vec = x.advanced(by: Int(gi) * d)
                    for j in 0..<d { sums[base + j] += Double(vec[j]) }
                case .aosoaR:
                    let R = cfg.aosoaRegisterBlock
                    for j in 0..<d {
                        let xv = _vi_km12_aosoa_elem(base: x, d: d, r: R, i: gi, j: j)
                        sums[base + j] += Double(xv)
                    }
                }

                batchCounts[cBest] += 1  // ✓ Standard arithmetic for count
            }

            // Update centroids for touched set
            for t in 0..<touched {
                let c = touchedList[t]
                let nC = batchCounts[c]
                if nC > 0 {
                    let invN = 1.0 / Double(nC)
                    let base = t * d
                    let dst = centroidsOut.advanced(by: c * d)
                    for j in 0..<d {
                        dst[j] = Float(sums[base + j] * invN)
                    }
                    batchCounts[c] = 0  // Reset for next batch
                }
            }

            // Identify empties and repair
            var empties = [Int]()
            empties.reserveCapacity(kc / 16)
            for c in 0..<kc {
                if batchTag[c] != currentTag { empties.append(c) }
            }
            if !empties.isEmpty {
                emptiesFixed = emptiesFixed &+ Int64(empties.count)  // ✓ Wrapping ok for counter
                empties.withUnsafeBufferPointer { eptr in
                    xBatchIdx.withUnsafeBufferPointer { biptr in
                        _vi_km12_repairEmpties_splitLargest(
                            xBatchIdx: biptr.baseAddress!, batchCount: batchCount,
                            X: x, n: n, d: d,
                            C: centroidsOut, kc: kc,
                            counts: batchCounts, empties: eptr.baseAddress!, emptiesCount: empties.count
                        )
                    }
                }
            }

            processed = processed &+ Int64(batchCount)  // ✓ Wrapping ok for counter
            batchesProcessed = batchesProcessed &+ 1     // ✓ Wrapping ok for counter
            rowsSeen = rowsSeen &+ Int64(batchCount)     // ✓ Wrapping ok for counter
        }

        // Inertia (sampled for huge n to keep cost bounded)
        let inertia: Double = {
            let sampleM: Int64
            var idxs: [Int64]
            if cfg.subsampleN > 0 && cfg.subsampleN < n {
                sampleM = cfg.subsampleN
                idxs = indices
            } else {
                sampleM = min(n, 10_000)  // Magic constant: 10K sample for inertia estimation
                idxs = _vi_km12_reservoir(n, sampleM, rng: &rng)
            }
            var sum: Double = 0
            for gi in idxs {
                switch cfg.layout {
                case .aos:
                    let vec = x.advanced(by: Int(gi) * d)
                    var best = _vi_km12_l2sq_aos(vec, centroidsOut, d)
                    var c = 1
                    while c < kc {
                        let dist = _vi_km12_l2sq_aos(vec, centroidsOut.advanced(by: c * d), d)
                        if dist < best { best = dist }
                        c += 1
                    }
                    sum += Double(best)
                case .aosoaR:
                    let R = cfg.aosoaRegisterBlock
                    var best = _vi_km12_l2sq_aosoa(x, centroidsOut, nIndex: gi, d: d, R: R)
                    var c = 1
                    while c < kc {
                        let dist = _vi_km12_l2sq_aosoa(x, centroidsOut.advanced(by: c * d), nIndex: gi, d: d, R: R)
                        if dist < best { best = dist }
                        c += 1
                    }
                    sum += Double(best)
                }
            }
            return sum
        }()

        inertiaHistory.append(inertia)

        // Early stopping (skip epoch 0)
        if epoch > 0 {
            let improvement = (prevInertia - inertia) / max(prevInertia, .leastNonzeroMagnitude)
            if improvement < Double(cfg.tol) {
                break epochLoop
            }
        }
        prevInertia = inertia
    }

    let tTrain1 = _vi_km12_nowNanos()

    // Optional: final assignments
    var tAssignSec: Double = 0
    if cfg.computeAssignments, let out = assignOut {
        let tA0 = _vi_km12_nowNanos()
        for i in 0..<Int(n) {
            let (cBest, _): (Int, Float) = {
                switch cfg.layout {
                case .aos:
                    let vec = x.advanced(by: i * d)
                    return _vi_km12_assignAOS_tiled(xVec: vec, C: centroidsOut, kc: kc, d: d, tile: 32)
                case .aosoaR:
                    let R = cfg.aosoaRegisterBlock
                    return _vi_km12_assignAOSOA(X: x, nIndex: Int64(i), C: centroidsOut, kc: kc, d: d, R: R)
                }
            }()
            out[i] = Int32(cBest)
        }
        let tA1 = _vi_km12_nowNanos()
        tAssignSec = Double(tA1 - tA0) * 1e-9
    }

    // Stats
    let stats = KMeansMBStats(
        epochsCompleted: epochsDone,
        batchesProcessed: batchesProcessed,
        rowsSeen: rowsSeen,
        emptiesRepaired: emptiesFixed,
        inertiaPerEpoch: inertiaHistory,
        finalInertia: inertiaHistory.last ?? 0,
        timeInitSec: Double(tInit1 - tInit0) * 1e-9,
        timeTrainingSec: Double(tTrain1 - tTrain0) * 1e-9,
        timeAssignmentSec: tAssignSec,
        bytesRead: rowsSeen * Int64(d) * 4
    )
    if let p = statsOut { p.pointee = stats }

    return .success
}

// MARK: - Stateful Streaming API

@frozen
public struct KMeansUpdateOpts: Sendable {
    public var decay: Float          // 0 -> use state's default
    public var normalizeCentroids: Bool
    public var prefetchDistance: Int
    public var adaptiveDecay: Bool

    @inlinable
    public init(decay: Float = 0, normalizeCentroids: Bool = false, prefetchDistance: Int = 8, adaptiveDecay: Bool = false) {
        self.decay = decay
        self.normalizeCentroids = normalizeCentroids
        self.prefetchDistance = prefetchDistance
        self.adaptiveDecay = adaptiveDecay
    }
}

/// Initialize streaming state (EWMA). Use seeds from #11 or existing IVF centroids.
/// State is single-writer; synchronize externally if shared.
@inlinable
public func kmeans_state_init(
    d: Int, kc: Int,
    initCentroids: UnsafePointer<Float>,
    decay: Float
) -> KMeansState {
    return KMeansState(d: d, kc: kc, initCentroids: initCentroids, decay: decay)
}

/// Update state with a chunk of vectors (EWMA). Optionally return centroids.
/// Deterministic single-threaded update.
/// - layout: AoS or AoSoA_R.
/// - adaptiveDecay: if true, η_c = η0 / (1 + t_c).
@inlinable
@discardableResult
public func kmeans_state_update_chunk(
    state: KMeansState,
    x_chunk: UnsafePointer<Float>,   // [m × d] or AoSoA_R
    m: Int64,
    layout: VectorLayoutMB = .aos,
    aosoaRegisterBlock: Int = 0,
    opts: inout KMeansUpdateOpts,
    centroidsOut: UnsafeMutablePointer<Float>? = nil
) -> KMeansMBStatus {
    let d = state.d
    let kc = state.kc
    let R = (layout == .aosoaR) ? max(1, aosoaRegisterBlock) : 0

    // Process each vector
    for ii in 0..<m {
        let (cBest, _): (Int, Float) = {
            switch layout {
            case .aos:
                let vec = x_chunk.advanced(by: Int(ii) * d)
                return _vi_km12_assignAOS_tiled(xVec: vec, C: state.centroids.withUnsafeBufferPointer { $0.baseAddress! }, kc: kc, d: d, tile: 32)
            case .aosoaR:
                return _vi_km12_assignAOSOA(X: x_chunk, nIndex: ii, C: state.centroids.withUnsafeBufferPointer { $0.baseAddress! }, kc: kc, d: d, R: R)
            }
        }()

        // Compute learning rate
        let tC = state.counts[cBest]
        let eta0 = opts.decay > 0 ? opts.decay : state.decay
        let eta: Float = opts.adaptiveDecay ? (eta0 / Float(1 + tC)) : eta0

        // Update C[c] ← (1-η)·C[c] + η·x
        let cPtr = state.centroids.withUnsafeMutableBufferPointer { $0.baseAddress! }.advanced(by: cBest * d)
        switch layout {
        case .aos:
            let xPtr = x_chunk.advanced(by: Int(ii) * d)
            let oneMinus = 1 - eta
            let d8 = (d / 8) * 8
            var j = 0
            while j < d8 {
                let c0 = _vi_km12_load4(cPtr, j)
                let x0 = _vi_km12_load4(xPtr, j)
                let u0 = c0 * SIMD4<Float>(repeating: oneMinus) + x0 * SIMD4<Float>(repeating: eta)  // ✓ Standard arithmetic
                let raw0 = UnsafeMutableRawPointer(cPtr.advanced(by: j))
                raw0.storeBytes(of: u0, as: SIMD4<Float>.self)

                let c1 = _vi_km12_load4(cPtr, j + 4)
                let x1 = _vi_km12_load4(xPtr, j + 4)
                let u1 = c1 * SIMD4<Float>(repeating: oneMinus) + x1 * SIMD4<Float>(repeating: eta)  // ✓ Standard arithmetic
                let raw1 = UnsafeMutableRawPointer(cPtr.advanced(by: j + 4))
                raw1.storeBytes(of: u1, as: SIMD4<Float>.self)

                j += 8
            }
            while j < d {
                let delta = eta * (xPtr[j] - cPtr[j])  // ✓ Standard arithmetic
                cPtr[j] += delta
                j += 1
            }
        case .aosoaR:
            for j in 0..<d {
                let xv = _vi_km12_aosoa_elem(base: x_chunk, d: d, r: R, i: ii, j: j)
                let delta = eta * (xv - cPtr[j])  // ✓ Standard arithmetic
                cPtr[j] += delta
            }
        }

        state.counts[cBest] = state.counts[cBest] &+ 1  // ✓ Wrapping ok for counter
        state.totalUpdates = state.totalUpdates &+ 1    // ✓ Wrapping ok for counter
    }

    // Optional re-centering (mean 0)
    if opts.normalizeCentroids {
        var mean = [Double](repeating: 0, count: d)
        for c in 0..<kc {
            let cp = state.centroids.withUnsafeBufferPointer { $0.baseAddress! }.advanced(by: c * d)
            for j in 0..<d { mean[j] += Double(cp[j]) }
        }
        let invK = 1.0 / Double(kc)
        for j in 0..<d { mean[j] *= invK }
        for c in 0..<kc {
            let cp = state.centroids.withUnsafeMutableBufferPointer { $0.baseAddress! }.advanced(by: c * d)
            for j in 0..<d { cp[j] -= Float(mean[j]) }
        }
    }

    if let out = centroidsOut {
        state.finalize(into: out)
    }
    return .success
}

/// Finalize streaming state into output centroids.
@inlinable
public func kmeans_state_finalize(
    state: KMeansState, centroidsOut: UnsafeMutablePointer<Float>
) {
    state.finalize(into: centroidsOut)
}

/// Destroy state (no-op under ARC; included for API symmetry with spec).
@inlinable
public func kmeans_state_destroy(_ state: KMeansState) {
    _ = state // ARC frees
}
