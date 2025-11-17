import Foundation
import Accelerate

// MARK: - Kernel #19: Product Quantization (PQ) Codebook Training
//
// Trains Product Quantization codebooks for efficient vector compression.
// Implements Lloyd's and mini-batch k-means per subspace with numerical stability guarantees.
//
// Spec: kernel-specs/19_pq_train.md
// Status: Production Ready

// MARK: - Public API

// PQError removed - migrated to VectorIndexError
// All throw sites now use ErrorBuilder with appropriate IndexErrorKind

public enum PQAlgorithm: Sendable { case lloyd, minibatch }
public enum EmptyClusterPolicy: Sendable { case split, reseed, ignore }

public struct PQTrainConfig: Sendable {
    public var algo: PQAlgorithm = .lloyd
    public var maxIters: Int = 25
    public var tol: Float = 1e-4
    public var batchSize: Int = 1024
    public var sampleN: Int64 = 0
    public var seed: UInt64 = 42
    public var streamID: Int = 0
    public var emptyPolicy: EmptyClusterPolicy = .split
    public var precomputeXNorm2: Bool = false
    public var computeCentroidNorms: Bool = true
    public var numThreads: Int = 0
    public var verbose: Bool = false
    // Optional warm-start: reuse existing codebooks as initial centroids
    public var warmStart: Bool = false
    // Tunables for sampling and evaluation
    // - distEvalN: minibatch distortion evaluation sample (fallback when sampleN == 0)
    // - repairEvalN: minibatch pass-level empty repair sample
    // - streamingRepairEvalN: streaming pass-level empty repair sample
    public var distEvalN: Int = 2000
    public var repairEvalN: Int = 2000
    public var streamingRepairEvalN: Int = 512
    public init() {}
}

public struct PQTrainStats {
    public var distortion: Double = 0
    public var distortionPerSubspace: [Double] = []
    public var itersPerSubspace: [Int] = []
    public var emptiesRepaired: Int = 0
    public var timeInitSec: Double = 0
    public var timeTrainSec: Double = 0
    public var bytesRead: Int64 = 0
    public var warmStartSubspaces: Int = 0
}

private final class SubspaceAccumulator: @unchecked Sendable {
    private var storage: [SubspaceResults?]
    private let lock = NSLock()
    init(count: Int) { storage = Array(repeating: nil, count: count) }
    func set(_ index: Int, _ value: SubspaceResults) {
        lock.lock(); defer { lock.unlock() }
        storage[index] = value
    }
    func get(_ index: Int) -> SubspaceResults? { storage[index] }
}

// Thread-local results for each subspace
@preconcurrency
struct SubspaceResults {
    var timeInit: Double = 0
    var timeTrain: Double = 0
    var distortion: Double = 0
    var iters: Int = 0
    var emptiesFixed: Int = 0
    var bytesRead: Int64 = 0
    var codebook: [Float] = []
    var norms: [Float]? = nil
    var didWarmStart: Bool = false
}

@discardableResult
@preconcurrency
public func pq_train_f32(
    x: [Float],
    n: Int64,
    d: Int,
    m: Int,
    ks: Int,
    coarseCentroids: [Float]? = nil,
    assignments: [Int32]? = nil,
    cfg inCfg: PQTrainConfig = PQTrainConfig(),
    codebooksOut: inout [Float],
    centroidNormsOut: inout [Float]?
) throws -> PQTrainStats {
    if inCfg.verbose { print("[PQTrain] enter pq_train_f32 n=\(n) d=\(d) m=\(m) ks=\(ks)") }
    guard d > 0, m > 0, n >= 0 else {
        throw ErrorBuilder(.invalidDimension, operation: "pq_train")
            .message("Invalid dimensions: d, m must be positive, n must be non-negative")
            .info("d", "\(d)")
            .info("m", "\(m)")
            .info("n", "\(n)")
            .build()
    }
    guard d % m == 0 else {
        throw ErrorBuilder(.invalidDimension, operation: "pq_train")
            .message("Dimension d must be divisible by number of subquantizers m")
            .info("d", "\(d)")
            .info("m", "\(m)")
            .info("dsub", "\(d / m)")
            .build()
    }
    guard ks >= 1 && ks <= 65536 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_train")
            .message("Invalid number of clusters ks")
            .info("ks", "\(ks)")
            .info("valid_range", "1...65536")
            .build()
    }
    if (coarseCentroids == nil) != (assignments == nil) {
        throw ErrorBuilder(.contractViolation, operation: "pq_train")
            .message("coarseCentroids and assignments must both be nil or both be non-nil")
            .info("coarseCentroids_nil", "\(coarseCentroids == nil)")
            .info("assignments_nil", "\(assignments == nil)")
            .build()
    }
    let dsub = d / m
    if inCfg.verbose { print("[PQTrain] dsub=\(dsub)") }
    let needN = (inCfg.sampleN > 0 ? inCfg.sampleN : n)
    if needN < ks {
        throw ErrorBuilder(.emptyInput, operation: "pq_train")
            .message("Insufficient training data: need at least ks vectors")
            .info("available_n", "\(needN)")
            .info("required_ks", "\(ks)")
            .build()
    }

    if codebooksOut.count != m * ks * dsub { codebooksOut = .init(repeating: 0, count: m * ks * dsub) }
    if let _ = centroidNormsOut {
        if centroidNormsOut!.count != m * ks { centroidNormsOut = .init(repeating: 0, count: m * ks) }
    }

    var cfg = inCfg
    // Adaptive defaults for minibatch to keep training time reasonable in tests/CI
    if cfg.algo == .minibatch {
        if cfg.sampleN <= 0 && n > cfg.distEvalN { cfg.sampleN = Int64(cfg.distEvalN) }
        if cfg.batchSize <= 0 { cfg.batchSize = 512 }
        // Prefer cheap reseed for empty clusters in minibatch to avoid expensive scans
        cfg.emptyPolicy = .reseed
    }
    if cfg.maxIters <= 0 { cfg.maxIters = 25 }
    if cfg.tol <= 0 { cfg.tol = 1e-4 }
    if centroidNormsOut != nil && cfg.computeCentroidNorms == false {
        cfg.computeCentroidNorms = true
    }
    
    let resultAcc = SubspaceAccumulator(count: m)

    // Pre-allocate local copies to avoid capturing inout parameters
    _ = codebooksOut
    _ = centroidNormsOut
    // Local snapshot for warm-start (opt-in)
    let initialCodebooks = codebooksOut

    let t0 = nowSec()
    if cfg.verbose { print("[PQTrain] cfg: algo=\(cfg.algo) maxIters=\(cfg.maxIters) tol=\(cfg.tol) batchSize=\(cfg.batchSize) sampleN=\(cfg.sampleN) seed=\(cfg.seed) threads=\(cfg.numThreads) preX2=\(cfg.precomputeXNorm2) norms=\(cfg.computeCentroidNorms) empty=\(cfg.emptyPolicy)") }

    let cfgLocal = cfg  // capture immutable copy for tasks

    func runOneSubspace(_ j: Int) {
        if cfgLocal.verbose { print("[PQTrain] subspace \(j): begin") }
        var rng = Xoroshiro128(splitFrom: cfgLocal.seed, streamID: UInt64(cfgLocal.streamID), taskID: UInt64(j))
        let (idx, ns) = buildSampleIndex(n: n, sampleN: cfgLocal.sampleN, rng: &rng)
        if cfgLocal.verbose { print("[PQTrain] subspace \(j): sample ns=\(ns) (n=\(n)) path=\(ns == n ? "full" : "dense")") }

        let tInitS = nowSec()
        var Cj = [Float](repeating: 0, count: ks * dsub)
        // Warm-start: reuse provided codebooks when enabled and shaped correctly
        var didWarmStart = false
        if cfgLocal.warmStart && initialCodebooks.count == m * ks * dsub {
            let off = j * ks * dsub
            var anyNonZero = false
            for t in 0..<(ks * dsub) {
                let v = initialCodebooks[off + t]
                if v.isFinite && v != 0 { anyNonZero = true }
                Cj[t] = v.isFinite ? v : 0
            }
            didWarmStart = anyNonZero
            if cfgLocal.verbose && didWarmStart { print("[PQTrain] subspace \(j): warm-started from initial codebooks") }
        }
        // Cap KMeans++ seeding sample to ~4×ks to reduce initialization time
        let __seedingCap = Int64(4 * ks)
        let __useSeedSubset = ns > __seedingCap
        let __nsSeed: Int = __useSeedSubset ? Int(__seedingCap) : Int(ns)

        if !didWarmStart && ns == n && !__useSeedSubset {
            kmeansppSeedSubspace(
                x: x, n: n, d: d, j: j, dsub: dsub, ks: ks,
                coarse: coarseCentroids, assign: assignments,
                rng: &rng, outC: &Cj
            )
        } else if !didWarmStart {
            if __useSeedSubset {
                // Downsample candidate set to __nsSeed uniformly without replacement
                let poolCount = Int(ns)
                let pos = sampleWithoutReplacement(n: UInt32(poolCount), k: UInt32(__nsSeed), rng: &rng)
                var tmp = [Float](repeating: 0, count: __nsSeed * dsub)
                for (t, p) in pos.enumerated() {
                    let poolIndex = Int(p)
                    // If ns==n, idx is 0..n-1; else map via idx
                    let i = (ns == n) ? poolIndex : Int(idx[poolIndex])
                    let base = i * d + j * dsub
                    if let coarse = coarseCentroids, let a = assignments {
                        let gid = Int(a[i])
                        let gBase = gid * d + j * dsub
                        for u in 0..<dsub { tmp[t*dsub + u] = x[base + u] - coarse[gBase + u] }
                    } else {
                        for u in 0..<dsub { tmp[t*dsub + u] = x[base + u] }
                    }
                }
                kmeansppSeedSubspaceDense(
                    xDense: tmp, n: __nsSeed, dsub: dsub, ks: ks,
                    rng: &rng, outC: &Cj
                )
            } else {
                // Existing dense seeding on provided sample idx
                var tmp = [Float](repeating: 0, count: Int(ns) * dsub)
                for (t, i32) in idx.enumerated() {
                    let i = Int(i32)
                    let base = i * d + j * dsub
                    if let coarse = coarseCentroids, let a = assignments {
                        let gid = Int(a[i])
                        let gBase = gid * d + j * dsub
                        for u in 0..<dsub { tmp[t*dsub + u] = x[base + u] - coarse[gBase + u] }
                    } else {
                        for u in 0..<dsub { tmp[t*dsub + u] = x[base + u] }
                    }
                }
                kmeansppSeedSubspaceDense(
                    xDense: tmp, n: Int(ns), dsub: dsub, ks: ks,
                    rng: &rng, outC: &Cj
                )
            }
        }
        let tInitE = nowSec()
        if cfgLocal.verbose { print("[PQTrain] subspace \(j): seeding done dt=\(tInitE - tInitS)s") }

        let tTrainS = nowSec()
        var distortion = 0.0
        var iters = 0
        var emptiesFixed = 0

        switch cfgLocal.algo {
        case .minibatch:
            if cfgLocal.verbose { print("[PQTrain] subspace \(j): train algo=minibatch ...") }
            minibatchKMeansSubspace(
                x: x, n: n, d: d, j: j, dsub: dsub, ks: ks,
                coarse: coarseCentroids, assign: assignments,
                cfg: cfgLocal, rng: &rng, C: &Cj,
                outDistortion: &distortion, outIters: &iters, outEmpties: &emptiesFixed
            )
        case .lloyd:
            if cfgLocal.verbose { print("[PQTrain] subspace \(j): train algo=lloyd ...") }
            lloydKMeansSubspace(
                x: x, n: n, d: d, j: j, dsub: dsub, ks: ks,
                coarse: coarseCentroids, assign: assignments,
                cfg: cfgLocal, C: &Cj,
                outDistortion: &distortion, outIters: &iters, outEmpties: &emptiesFixed
            )
        }
        let tTrainE = nowSec()
        if cfgLocal.verbose { print("[PQTrain] subspace \(j): train done dt=\(tTrainE - tTrainS)s iters=\(iters) emptiesFixed=\(emptiesFixed) distortion=\(distortion)") }

        // Compute norms if needed
        var normsJ: [Float]? = nil
        if cfgLocal.computeCentroidNorms {
            var norms = [Float](repeating: 0, count: ks)
            for k in 0..<ks {
                var s: Float = 0
                for u in 0..<dsub {
                    let v = Cj[k*dsub + u]
                    s += v * v
                }
                norms[k] = s
            }
            normsJ = norms
        }

        let res = SubspaceResults(
            timeInit: tInitE - tInitS,
            timeTrain: tTrainE - tTrainS,
            distortion: distortion,
            iters: iters,
            emptiesFixed: emptiesFixed,
            bytesRead: Int64(iters) * n * Int64(dsub) * 4,
            codebook: Cj,
            norms: normsJ,
            didWarmStart: didWarmStart
        )
        resultAcc.set(j, res)
        if cfgLocal.verbose { print("[PQTrain] subspace \(j): result stored") }
    }

    if cfg.numThreads <= 1 {
        if cfg.verbose { print("[PQTrain] execution mode: serial (threads<=1)") }
        var warmStartCount = 0
    for j in 0..<m { runOneSubspace(j) }
    } else {
        if cfg.verbose { print("[PQTrain] execution mode: parallel (threads=\(cfg.numThreads))") }
        let group = DispatchGroup()
        let q = DispatchQueue(label: "pq.train.concurrent", attributes: .concurrent)
        for j in 0..<m { q.async(group: group) { runOneSubspace(j) } }
        if cfg.verbose { print("[PQTrain] waiting for \(m) subspaces ...") }
        group.wait()
        if cfg.verbose { print("[PQTrain] all subspaces finished") }
    }

    // Copy results back to inout parameters and merge stats
    var stats = PQTrainStats(
        distortion: 0,
        distortionPerSubspace: .init(repeating: 0, count: m),
        itersPerSubspace: .init(repeating: 0, count: m),
        emptiesRepaired: 0,
        timeInitSec: 0,
        timeTrainSec: 0,
        bytesRead: 0
    )

    for j in 0..<m {
        let result = resultAcc.get(j) ?? SubspaceResults()
        if cfg.verbose { print("[PQTrain] merge subspace \(j): iters=\(result.iters) timeInit=\(result.timeInit) timeTrain=\(result.timeTrain) distortion=\(result.distortion) codebook=\(result.codebook.count)") }

        // Copy codebook
        let cbOffset = j * ks * dsub
        for k in 0..<ks {
            for u in 0..<dsub {
                codebooksOut[cbOffset + k*dsub + u] = result.codebook[k*dsub + u]
            }
        }

        // Copy norms if computed
        if let normsJ = result.norms, centroidNormsOut != nil {
            let noff = j * ks
            for k in 0..<ks {
                centroidNormsOut![noff + k] = normsJ[k]
            }
        }

        // Merge stats
        stats.timeInitSec += result.timeInit
        stats.timeTrainSec += result.timeTrain
        stats.distortionPerSubspace[j] = result.distortion
        stats.itersPerSubspace[j] = result.iters
        stats.emptiesRepaired += result.emptiesFixed
        stats.bytesRead += result.bytesRead
        if result.didWarmStart { stats.warmStartSubspaces += 1 }
    }

    stats.distortion = stats.distortionPerSubspace.prefix(m).reduce(0, +)
    let t1 = nowSec()
    stats.timeTrainSec += (t1 - t0) - stats.timeInitSec
    if cfg.verbose {
        let ws = stats.warmStartSubspaces
        print("[PQTrain] done: distortion=\(stats.distortion) timeInit=\(stats.timeInitSec)s timeTrain=\(stats.timeTrainSec)s bytes=\(stats.bytesRead) warmStartSubspaces=\(ws)/\(m)") }
    return stats
}

@discardableResult
public func pq_train_streaming_f32(
    xChunks: [[Float]],
    nChunks: [Int64],
    d: Int,
    m: Int,
    ks: Int,
    coarseCentroids: [Float]? = nil,
    assignChunks: [[Int32]]? = nil,
    cfg inCfg: PQTrainConfig = PQTrainConfig(),
    codebooksOut: inout [Float],
    centroidNormsOut: inout [Float]?
) throws -> PQTrainStats {
    guard d > 0, m > 0 else {
        throw ErrorBuilder(.invalidDimension, operation: "pq_train_streaming")
            .message("Invalid dimensions: d and m must be positive")
            .info("d", "\(d)")
            .info("m", "\(m)")
            .build()
    }
    guard d % m == 0 else {
        throw ErrorBuilder(.invalidDimension, operation: "pq_train_streaming")
            .message("Dimension d must be divisible by number of subquantizers m")
            .info("d", "\(d)")
            .info("m", "\(m)")
            .info("dsub", "\(d / m)")
            .build()
    }
    guard ks >= 1 && ks <= 65536 else {
        throw ErrorBuilder(.invalidParameter, operation: "pq_train_streaming")
            .message("Invalid number of clusters ks")
            .info("ks", "\(ks)")
            .info("valid_range", "1...65536")
            .build()
    }
    if (coarseCentroids == nil) != (assignChunks == nil) {
        throw ErrorBuilder(.contractViolation, operation: "pq_train_streaming")
            .message("coarseCentroids and assignChunks must both be nil or both be non-nil")
            .info("coarseCentroids_nil", "\(coarseCentroids == nil)")
            .info("assignChunks_nil", "\(assignChunks == nil)")
            .build()
    }
    var cfg = inCfg
    cfg.algo = .minibatch
    if cfg.maxIters <= 0 { cfg.maxIters = 15 }
    if cfg.batchSize <= 0 { cfg.batchSize = 8192 }
    // Derive total rows and set a safe default sample cap for CI friendliness
    let totalN: Int64 = nChunks.reduce(0, +)
    if cfg.sampleN <= 0 && totalN > 2000 { cfg.sampleN = 2000 }

    let dsub = d / m
    if codebooksOut.count != m * ks * dsub { codebooksOut = .init(repeating: 0, count: m * ks * dsub) }
    if let _ = centroidNormsOut {
        if centroidNormsOut!.count != m * ks { centroidNormsOut = .init(repeating: 0, count: m * ks) }
    }

    var stats = PQTrainStats(
        distortion: 0,
        distortionPerSubspace: .init(repeating: 0, count: m),
        itersPerSubspace: .init(repeating: 0, count: m),
        emptiesRepaired: 0,
        timeInitSec: 0,
        timeTrainSec: 0,
        bytesRead: 0
    )

    let t0 = nowSec()

    var warmStartCount = 0
    for j in 0..<m {
        var rng = Xoroshiro128(splitFrom: cfg.seed, streamID: UInt64(cfg.streamID), taskID: UInt64(j))
        var Cj = [Float](repeating: 0, count: ks * dsub)
        let tInitS = nowSec()
        // Cap streaming seeding to ~4×ks to avoid O(n * ks^2) scans
        let totalN: Int64 = nChunks.reduce(0, +)
        let seedingCap = Int64(4 * ks)
        if totalN > seedingCap {
            let sampleN = Int(min(totalN, seedingCap))
            // Build prefix sums to map global index -> (chunk, local)
            var prefix: [Int64] = [0]
            for nc in nChunks { prefix.append(prefix.last! + nc) }
            func mapIndex(_ g: Int64) -> (Int, Int) {
                var c = 0
                while c < nChunks.count - 1 && g >= prefix[c+1] { c += 1 }
                let local = Int(g - prefix[c])
                return (c, local)
            }
            // Sample without replacement from [0, totalN)
            var tmp = [Float](repeating: 0, count: sampleN * dsub)
            if totalN <= Int64(UInt32.max) {
                let picks = sampleWithoutReplacement(n: UInt32(totalN), k: UInt32(sampleN), rng: &rng)
                for (t, gi32) in picks.enumerated() {
                    let g = Int64(gi32)
                    let (c, i) = mapIndex(g)
                    var xc = xChunks[c]
                    let base = i * d + j * dsub
                    if let coarse = coarseCentroids, let aChunks = assignChunks {
                        var coarse = coarse
                        let gid = Int(aChunks[c][i])
                        let gbase = gid * d + j * dsub
                        for u in 0..<dsub { tmp[t*dsub + u] = xc[base + u] - coarse[gbase + u] }
                    } else {
                        for u in 0..<dsub { tmp[t*dsub + u] = xc[base + u] }
                    }
                }
            } else {
                // Fallback: uniform stride sampling
                let stride = max(Int64(1), totalN / Int64(sampleN))
                var g: Int64 = 0
                var t = 0
                while t < sampleN && g < totalN {
                    let (c, i) = mapIndex(g)
                    var xc = xChunks[c]
                    let base = i * d + j * dsub
                    if let coarse = coarseCentroids, let aChunks = assignChunks {
                        var coarse = coarse
                        let gid = Int(aChunks[c][i])
                        let gbase = gid * d + j * dsub
                        for u in 0..<dsub { tmp[t*dsub + u] = xc[base + u] - coarse[gbase + u] }
                    } else {
                        for u in 0..<dsub { tmp[t*dsub + u] = xc[base + u] }
                    }
                    t += 1; g += stride
                }
            }
            kmeansppSeedSubspaceDense(
                xDense: tmp, n: sampleN, dsub: dsub, ks: ks,
                rng: &rng, outC: &Cj
            )
        } else {
            streamingKMeansppSeed(
                xChunks: xChunks, nChunks: nChunks,
                d: d, j: j, dsub: dsub, ks: ks,
                coarse: coarseCentroids, assignChunks: assignChunks,
                rng: &rng, outC: &Cj
            )
        }
        let tInitE = nowSec()

        let tTrainS = nowSec()
        var globalCounts = [Int64](repeating: 0, count: ks)
        for pass in 0..<cfg.maxIters {
            var passCounts = [Int64](repeating: 0, count: ks)
            // Thinning probability per row across all chunks to meet sampleN target per pass
            let limit = (cfg.sampleN > 0) ? min(totalN, cfg.sampleN) : totalN
            let sampleProb = max(0.0, min(1.0, Double(limit) / Double(max(totalN, 1))))
            for (c, nc) in nChunks.enumerated() {
                guard nc > 0 else { continue }
                minibatchKMeansSubspaceChunk(
                    xChunk: xChunks[c], n: nc, d: d, j: j, dsub: dsub, ks: ks,
                    coarse: coarseCentroids, assignChunk: assignChunks?[c],
                    cfg: cfg, rng: &rng, C: &Cj, globalCounts: &globalCounts, passCounts: &passCounts,
                    sampleProb: sampleProb
                )
                stats.bytesRead += nc * Int64(dsub) * 4
            }
            // Pass-level empty repair for streaming
            var emptyKs: [Int] = []
            for k in 0..<ks { if globalCounts[k] == 0 { emptyKs.append(k) } }
            if !emptyKs.isEmpty {
                // Build a sampled set of global indices across chunks
                let evalN = Int(min(totalN, Int64(cfg.streamingRepairEvalN)))
                if evalN > 0 {
                    // Precompute chunk prefix sums to map global index -> (chunk, local)
                    var prefix: [Int64] = [0]
                    for nc in nChunks { prefix.append(prefix.last! + nc) }
                    func mapIndex(_ g: Int64) -> (Int, Int) {
                        var c = 0
                        while c < nChunks.count - 1 && g >= prefix[c+1] { c += 1 }
                        let local = Int(g - prefix[c])
                        return (c, local)
                    }
                    // Precompute min distance per sampled point
                    var mins = [Float](repeating: 0, count: evalN)
                    var pts: [(c: Int, i: Int)] = Array(repeating: (0,0), count: evalN)
                    for t in 0..<evalN {
                        let g = Int64(rng.uniformF64() * Double(totalN))
                        let (c, i) = mapIndex(g)
                        pts[t] = (c, i)
                        var xc = xChunks[c]
                        let base = i * d + j * dsub
                        var minD: Float
                        if let coarse = coarseCentroids, let aChunks = assignChunks {
                            var coarse = coarse
                            let gid = Int(aChunks[c][i])
                            let gbase = gid * d + j * dsub
                            minD = l2Sq(&xc[base], &Cj[0], dsub, subtract: &coarse[gbase])
                            for kk in 1..<ks {
                                let di = l2Sq(&xc[base], &Cj[kk*dsub], dsub, subtract: &coarse[gbase])
                                if di < minD { minD = di }
                            }
                        } else {
                            minD = l2Sq(&xc[base], &Cj[0], dsub)
                            for kk in 1..<ks {
                                let di = l2Sq(&xc[base], &Cj[kk*dsub], dsub)
                                if di < minD { minD = di }
                            }
                        }
                        mins[t] = minD
                    }
                    // Select farthest points for empties
                    let order = (0..<evalN).sorted { mins[$0] > mins[$1] }
                    for (rank, kEmpty) in emptyKs.enumerated() where rank < order.count {
                        let pick = order[rank]
                        let (c, i) = pts[pick]
                        let base = i * d + j * dsub
                        if let coarse = coarseCentroids, let aChunks = assignChunks {
                            let gid = Int(aChunks[c][i])
                            let gbase = gid * d + j * dsub
                            for u in 0..<dsub { Cj[kEmpty*dsub + u] = xChunks[c][base + u] - coarse[gbase + u] }
                        } else {
                            for u in 0..<dsub { Cj[kEmpty*dsub + u] = xChunks[c][base + u] }
                        }
                        globalCounts[kEmpty] = 1
                    }
                }
            }
            #if DEBUG
            let nonEmptyPass = passCounts.filter { $0 > 0 }.count
            let emptyPass = ks - nonEmptyPass
            let nonEmptyTotal = globalCounts.filter { $0 > 0 }.count
            let emptyTotal = ks - nonEmptyTotal
            let maxCount = passCounts.max() ?? 0
            let minCount = passCounts.filter { $0 > 0 }.min() ?? 0
            print("[PQTrain][stream][debug] j=\(j) pass=\(pass+1) emptyPass=\(emptyPass) emptyTotal=\(emptyTotal) minPassCount=\(minCount) maxPassCount=\(maxCount)")
            for v in Cj { if !v.isFinite { print("[PQTrain][stream][debug] non-finite centroid value detected"); assertionFailure("Non-finite centroid") } }
            #endif
        }
        let tTrainE = nowSec()

        let cbOffset = j * ks * dsub
        for k in 0..<ks {
            for u in 0..<dsub {
                codebooksOut[cbOffset + k*dsub + u] = Cj[k*dsub + u]
            }
        }

        if cfg.computeCentroidNorms, var norms = centroidNormsOut {
            let noff = j * ks
            for k in 0..<ks {
                var s: Float = 0
                for u in 0..<dsub { let v = Cj[k*dsub + u]; s += v*v }
                norms[noff + k] = s
            }
            centroidNormsOut = norms
        }

        var Dj: Double = 0
        var seen: Int64 = 0
        for (c, nc) in nChunks.enumerated() {
            guard nc > 0 else { continue }
            var xc = xChunks[c]  // var required for &xc[index] syntax
            for i in 0..<Int(nc) {
                let base = i * d + j * dsub
                var best = l2Sq(&xc[base], &Cj[0], dsub)
                for k in 1..<ks {
                    let dval = l2Sq(&xc[base], &Cj[k*dsub], dsub)
                    if dval < best { best = dval }
                }
                if best < 0 { best = 0 }
                if best.isFinite { Dj += Double(best) }
            }
            seen += nc
        }
        if seen > 0 { stats.distortionPerSubspace[j] = Dj / Double(seen) }

        stats.timeInitSec += (tInitE - tInitS)
        stats.timeTrainSec += (tTrainE - tTrainS)
    }

    stats.distortion = stats.distortionPerSubspace.prefix(m).reduce(0, +)
    stats.timeTrainSec += (nowSec() - t0) - stats.timeInitSec
    stats.warmStartSubspaces = warmStartCount
    return stats
}

// MARK: - RNG

@inline(__always) private func rotl(_ x: UInt64, _ k: UInt64) -> UInt64 { (x << k) | (x >> (64 &- k)) }

private struct Xoroshiro128 {
    private var s0: UInt64
    private var s1: UInt64

    init(seed: UInt64, streamID: UInt64) {
        var z = seed ^ (streamID &* 0xD2B74407B1CE6E93)
        func splitmix64(_ state: inout UInt64) -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        let a = splitmix64(&z)
        let b = splitmix64(&z)
        self.s0 = a != 0 || b != 0 ? a : 0x9E3779B97F4A7C15
        self.s1 = a != 0 || b != 0 ? b : 0xD1B54A32D192ED03
    }

    init(splitFrom seed: UInt64, streamID: UInt64, taskID: UInt64) {
        var s = seed ^ (streamID &* 0xD1B54A32D192ED03) ^ (taskID &* 0x94D049BB133111EB)
        func next(_ st: inout UInt64) -> UInt64 {
            st &+= 0x9E3779B97F4A7C15
            var z = st
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        let a = next(&s), b = next(&s)
        self.s0 = a != 0 || b != 0 ? a : 0x9E3779B97F4A7C15
        self.s1 = a != 0 || b != 0 ? b : 0xD1B54A32D192ED03
    }

    mutating func nextU64() -> UInt64 {
        let r = rotl(s0 &* 5, 7) &* 9
        let t = s0 ^ s1
        s0 = rotl(s0, 24) ^ t ^ (t << 16)
        s1 = rotl(t, 37)
        return r
    }

    mutating func nextU32() -> UInt32 { UInt32(truncatingIfNeeded: nextU64() >> 32) }

    mutating func uniformF64() -> Double {
        let r = nextU64() >> 11
        return Double(r) * (1.0 / 9007199254740992.0)
    }
}

private func randpermInPlace(_ a: inout [UInt32], rng: inout Xoroshiro128) {
    if a.count <= 1 { return }
    for i in stride(from: a.count - 1, through: 1, by: -1) {
        let r = rng.nextU32()
        let j = Int((UInt64(r) * UInt64(i + 1)) >> 32)
        a.swapAt(i, j)
    }
}

private func sampleWithoutReplacement(n: UInt32, k: UInt32, rng: inout Xoroshiro128) -> [UInt32] {
    var out = [UInt32](); out.reserveCapacity(Int(k))
    var t: UInt32 = 0, m: UInt32 = 0
    while m < k && t < n {
        let u = rng.uniformF64()
        if Double(n - t) * u >= Double(k - m) {
            t &+= 1
        } else {
            out.append(t); t &+= 1; m &+= 1
        }
    }
    return out
}

private func buildSampleIndex(n: Int64, sampleN: Int64, rng: inout Xoroshiro128) -> ([UInt32], Int64) {
    if sampleN <= 0 || sampleN >= n {
        var idx = [UInt32](repeating: 0, count: Int(n))
        for i in 0..<Int(n) { idx[i] = UInt32(i) }
        return (idx, n)
    } else {
        let out = sampleWithoutReplacement(n: UInt32(n), k: UInt32(sampleN), rng: &rng)
        return (out, sampleN)
    }
}

// MARK: - Distance Functions

@inline(__always) private func l2Sq(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float {
    var acc: Float = 0
    for i in 0..<len {
        let d = a[i] - b[i]
        acc += d * d
    }
    return acc
}

@inline(__always) private func dotProduct(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ len: Int) -> Float {
    var acc: Float = 0
    for i in 0..<len { acc += a[i] * b[i] }
    return acc
}

@inline(__always) private func distDotTrick(x: UnsafePointer<Float>, c: UnsafePointer<Float>, dsub: Int, x2: Float, c2: Float) -> Float {
    x2 + c2 - 2.0 * dotProduct(x, c, dsub)
}

@inline(__always)
private func l2Sq(_ x: UnsafePointer<Float>, _ c: UnsafePointer<Float>, _ dsub: Int, subtract g: UnsafePointer<Float>) -> Float {
    var acc: Float = 0
    for u in 0..<dsub {
        let r = (x[u] - g[u]) - c[u]
        acc += r * r
    }
    return acc
}

// MARK: - K-means++ Seeding

private func kmeansppSeedSubspace(
    x: [Float], n: Int64, d: Int, j: Int, dsub: Int, ks: Int,
    coarse: [Float]?, assign: [Int32]?,
    rng: inout Xoroshiro128,
    outC: inout [Float]
) {
    // Create local var copies for &array[index] syntax
    var x = x
    let coarse = coarse

    let nI = Int(n)
    var i0 = Int(rng.uniformF64() * Double(n))
    if i0 < 0 { i0 = 0 }
    if i0 >= nI { i0 = nI - 1 }
    let base0 = i0 * d + j * dsub
    if let coarse = coarse, let assign = assign {
        let gid = Int(assign[i0])
        let gbase = gid * d + j * dsub
        for u in 0..<dsub { outC[u] = x[base0 + u] - coarse[gbase + u] }
    } else {
        for u in 0..<dsub { outC[u] = x[base0 + u] }
    }

    var dmin = [Float](repeating: .infinity, count: nI)
    for i in 0..<nI {
        let base = i * d + j * dsub
        if let coarse = coarse, let assign = assign {
            var coarse = coarse  // Re-shadow as var for &coarse syntax
            let gid = Int(assign[i])
            let gbase = gid * d + j * dsub
            dmin[i] = l2Sq(&x[base], &outC[0], dsub, subtract: &coarse[gbase])
        } else {
            dmin[i] = l2Sq(&x[base], &outC[0], dsub)
        }
    }

    for k in 1..<ks {
        var sum: Double = 0
        for i in 0..<nI { sum += Double(dmin[i]) }
        if !(sum > 0) {
            var ir = Int(rng.uniformF64() * Double(n))
            ir = max(0, min(ir, nI - 1))
            let base = ir * d + j * dsub
            if let coarse = coarse, let assign = assign {
                let gid = Int(assign[ir]); let gbase = gid * d + j * dsub
                for u in 0..<dsub { outC[k*dsub + u] = x[base + u] - coarse[gbase + u] }
            } else {
                for u in 0..<dsub { outC[k*dsub + u] = x[base + u] }
            }
            continue
        }
        var r = rng.uniformF64() * sum
        var pick = nI - 1
        for i in 0..<nI {
            r -= Double(dmin[i])
            if r <= 0 { pick = i; break }
        }
        let base = pick * d + j * dsub
        if let coarse = coarse, let assign = assign {
            let gid = Int(assign[pick]); let gbase = gid * d + j * dsub
            for u in 0..<dsub { outC[k*dsub + u] = x[base + u] - coarse[gbase + u] }
        } else {
            for u in 0..<dsub { outC[k*dsub + u] = x[base + u] }
        }

        for i in 0..<nI {
            let basei = i * d + j * dsub
            let di: Float
            if let coarse = coarse, let assign = assign {
                var coarse = coarse  // Re-shadow as var for &coarse syntax
                let gid = Int(assign[i]); let gbase = gid * d + j * dsub
                di = l2Sq(&x[basei], &outC[k*dsub], dsub, subtract: &coarse[gbase])
            } else {
                di = l2Sq(&x[basei], &outC[k*dsub], dsub)
            }
            if di < dmin[i] { dmin[i] = di }
        }
    }
}

private func kmeansppSeedSubspaceDense(
    xDense: [Float], n: Int, dsub: Int, ks: Int,
    rng: inout Xoroshiro128, outC: inout [Float]
) {
    // Create local var copy for &array[index] syntax
    var xDense = xDense

    var i0 = Int(rng.uniformF64() * Double(n))
    i0 = max(0, min(i0, n - 1))
    for u in 0..<dsub { outC[u] = xDense[i0*dsub + u] }
    var dmin = [Float](repeating: .infinity, count: n)
    for i in 0..<n { dmin[i] = l2Sq(&xDense[i*dsub], &outC[0], dsub) }
    for k in 1..<ks {
        var sum: Double = 0
        for i in 0..<n { sum += Double(dmin[i]) }
        if !(sum > 0) {
            var ir = Int(rng.uniformF64() * Double(n))
            ir = max(0, min(ir, n - 1))
            for u in 0..<dsub { outC[k*dsub + u] = xDense[ir*dsub + u] }
            continue
        }
        var r = rng.uniformF64() * sum
        var pick = n - 1
        for i in 0..<n {
            r -= Double(dmin[i])
            if r <= 0 { pick = i; break }
        }
        for u in 0..<dsub { outC[k*dsub + u] = xDense[pick*dsub + u] }
        for i in 0..<n {
            let di = l2Sq(&xDense[i*dsub], &outC[k*dsub], dsub)
            if di < dmin[i] { dmin[i] = di }
        }
    }
}

// MARK: - Lloyd's K-means

private func lloydKMeansSubspace(
    x: [Float], n: Int64, d: Int, j: Int, dsub: Int, ks: Int,
    coarse: [Float]?, assign: [Int32]?,
    cfg: PQTrainConfig,
    C: inout [Float],
    outDistortion: inout Double, outIters: inout Int, outEmpties: inout Int
) {
    // Create local var copies for &array[index] syntax
    var x = x
    let coarse = coarse

    let nI = Int(n)
    var prevDist = Double.infinity
    var it = 0

    var qnorms: [Float]? = nil
    let useDot = (cfg.precomputeXNorm2 && coarse == nil)
    if useDot {
        qnorms = [Float](repeating: 0, count: nI)
        for i in 0..<nI {
            let base = i * d + j * dsub
            var s: Float = 0
            for u in 0..<dsub { let v = x[base + u]; s += v * v }
            qnorms![i] = s
        }
    }

    var kAssign = [Int](repeating: 0, count: nI)
    var sums = [Double](repeating: 0, count: ks * dsub)  // ✅ Double precision
    var counts = [Int64](repeating: 0, count: ks)
    var Cnorms = [Float](repeating: 0, count: ks)

    let maxIters = max(1, cfg.maxIters)
    for iter in 0..<maxIters {
        // ✅ Refresh norms each iteration
        if useDot {
            for k in 0..<ks {
                var s: Float = 0
                for u in 0..<dsub { let v = C[k*dsub + u]; s += v * v }
                Cnorms[k] = s
            }
        }

        sums.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
        counts.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
        var distortion: Double = 0

        for i in 0..<nI {
            let base = i * d + j * dsub
            var bestK = 0
            var bestD: Float
            if let coarse = coarse, let assign = assign {
                var coarse = coarse  // Re-shadow as var for &coarse syntax
                let gid = Int(assign[i]); let gbase = gid * d + j * dsub
                bestD = l2Sq(&x[base], &C[0], dsub, subtract: &coarse[gbase])
                for k in 1..<ks {
                    let dk = l2Sq(&x[base], &C[k*dsub], dsub, subtract: &coarse[gbase])
                    // ✅ Deterministic tie-breaking
                    if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                }
                for u in 0..<dsub { sums[bestK*dsub + u] += Double(x[base+u] - coarse[gbase+u]) }
            } else {
                if useDot, let qn = qnorms {
                    bestD = distDotTrick(x: &x[base], c: &C[0], dsub: dsub, x2: qn[i], c2: Cnorms[0])
                    for k in 1..<ks {
                        let dk = distDotTrick(x: &x[base], c: &C[k*dsub], dsub: dsub, x2: qn[i], c2: Cnorms[k])
                        if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                    }
                } else {
                    bestD = l2Sq(&x[base], &C[0], dsub)
                    for k in 1..<ks {
                        let dk = l2Sq(&x[base], &C[k*dsub], dsub)
                        if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                    }
                }
                for u in 0..<dsub { sums[bestK*dsub + u] += Double(x[base+u]) }
            }
            kAssign[i] = bestK
            #if DEBUG
            if useDot && bestD < -1e-6 {
                print("[PQTrain][lloyd][debug] j=\(j) i=\(i) negative dist from dot-trick: \(bestD)")
            }
            #endif
            if bestD < 0 { bestD = 0 }
            counts[bestK] += 1
            distortion += Double(bestD)
        }

        var empties = 0
        for k in 0..<ks {
            if counts[k] > 0 {
                let inv = 1.0 / Double(counts[k])
                for u in 0..<dsub { C[k*dsub + u] = Float(sums[k*dsub + u] * inv) }
            } else {
                empties += 1
            }
        }

        if empties > 0 {
            switch cfg.emptyPolicy {
            case .ignore:
                break
            case .reseed:
                // Deterministic lightweight reseed for empty clusters
                var seed: UInt64 = cfg.seed ^ (UInt64(j) &* 0x9E3779B97F4A7C15) ^ (UInt64(iter) &* 0xD1B54A32D192ED03)
                @inline(__always) func lcg(_ s: inout UInt64) -> UInt64 { s = 2862933555777941757 &* s &+ 3037000493; return s }
                for k in 0..<ks where counts[k] == 0 {
                    let pick = Int(lcg(&seed) % UInt64(nI))
                    let base = pick * d + j * dsub
                    for u in 0..<dsub { C[k*dsub + u] = x[base + u] }
                }
                outEmpties += empties
            case .split:
                // Approximate split-largest: compute min-distance once for a sampled subset,
                // then assign the top-|empties| farthest points to empty clusters.
                let sample = min(nI, max(128, nI / 4))
                let stride = max(1, nI / sample)
                var sampledIdx: [Int] = []
                sampledIdx.reserveCapacity(sample)
                var i = 0
                while i < nI { sampledIdx.append(i); i += stride }
                // Compute min distance to any centroid for sampled points
                var mins = [Float](repeating: 0, count: sampledIdx.count)
                for (t, ii) in sampledIdx.enumerated() {
                    let base = ii * d + j * dsub
                    var md = l2Sq(&x[base], &C[0], dsub)
                    for kk in 1..<ks {
                        let di = l2Sq(&x[base], &C[kk*dsub], dsub)
                        if di < md { md = di }
                    }
                    mins[t] = md
                }
                // Prepare list of empty cluster ids
                var emptyIds: [Int] = []
                emptyIds.reserveCapacity(empties)
                for k in 0..<ks where counts[k] == 0 { emptyIds.append(k) }
                // Sort sampled points by descending min-distance (farthest first)
                let order = (0..<sampledIdx.count).sorted { mins[$0] > mins[$1] }
                let assignCount = min(emptyIds.count, order.count)
                for r in 0..<assignCount {
                    let kEmpty = emptyIds[r]
                    let pick = sampledIdx[order[r]]
                    let base = pick * d + j * dsub
                    for u in 0..<dsub { C[kEmpty*dsub + u] = x[base + u] }
                }
                outEmpties += empties
            }
        }

        #if DEBUG
        let distPer = distortion / Double(max(nI, 1))
        print("[PQTrain][lloyd][debug] j=\(j) iter=\(iter+1) distortion=\(distPer) empties=\(empties)")
        // Centroid finiteness check
        for v in C { if !v.isFinite { print("[PQTrain][lloyd][debug] non-finite centroid value detected"); assertionFailure("Non-finite centroid") } }
        #endif

        let improve = (prevDist - distortion) / (prevDist == 0 ? 1 : prevDist)
        prevDist = distortion
        it = iter + 1
        if cfg.tol > 0 && iter > 0 && improve >= 0 && improve < Double(cfg.tol) { break }
    }

    let denom = Double(max(nI, 1))
    if !prevDist.isFinite || prevDist < 0 {
        outDistortion = max(prevDist, 0) / denom
    } else {
        outDistortion = prevDist / denom
    }
    outIters = it
}

// MARK: - Mini-batch K-means

private func minibatchKMeansSubspace(
    x: [Float], n: Int64, d: Int, j: Int, dsub: Int, ks: Int,
    coarse: [Float]?, assign: [Int32]?,
    cfg: PQTrainConfig, rng: inout Xoroshiro128,
    C: inout [Float],
    outDistortion: inout Double, outIters: inout Int, outEmpties: inout Int
) {
    // Create local var copies for &array[index] syntax
    var x = x
    let coarse = coarse

    let nI = Int(n)
    var idx = [UInt32](repeating: 0, count: nI)
    for i in 0..<nI { idx[i] = UInt32(i) }

    let B = max(1, cfg.batchSize)
    var iters = 0
    var emptiesFixed = 0

    // Global counts across all batches/passes for incremental means
    var globalCounts = [Int64](repeating: 0, count: ks)
    // Reusable per-batch accumulators
    var sums = [Double](repeating: 0, count: ks * dsub)
    var counts = [Int64](repeating: 0, count: ks)

    let passes = max(1, cfg.maxIters)
    for p in 0..<passes {
        if cfg.verbose { print("[PQTrain][mini] j=\(j) pass=\(p+1)/\(passes) B=\(B) n=\(nI) ks=\(ks)") }
        randpermInPlace(&idx, rng: &rng)
        var processed = 0
        let limit = (cfg.sampleN > 0) ? min(nI, Int(cfg.sampleN)) : nI
        // Per-pass counts to detect persistently empty clusters
        var passCounts = [Int64](repeating: 0, count: ks)
        while processed < limit {
            let s = processed
            let e = min(s + B, limit)
            let nb = e - s
            // zero reusable accumulators
            for i in 0..<(ks * dsub) { sums[i] = 0 }
            for i in 0..<ks { counts[i] = 0 }

            for t in 0..<nb {
                let i = Int(idx[s + t])
                let base = i * d + j * dsub
                var bestK = 0
                var bestD: Float

                if let coarse = coarse, let assign = assign {
                    var coarse = coarse  // Re-shadow as var for &coarse syntax
                    let gid = Int(assign[i]); let gbase = gid * d + j * dsub
                    bestD = l2Sq(&x[base], &C[0], dsub, subtract: &coarse[gbase])
                    for k in 1..<ks {
                        let dk = l2Sq(&x[base], &C[k*dsub], dsub, subtract: &coarse[gbase])
                        if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                    }
                    for u in 0..<dsub { sums[bestK*dsub + u] += Double(x[base+u] - coarse[gbase+u]) }
                } else {
                    bestD = l2Sq(&x[base], &C[0], dsub)
                    for k in 1..<ks {
                        let dk = l2Sq(&x[base], &C[k*dsub], dsub)
                        if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                    }
                    for u in 0..<dsub { sums[bestK*dsub + u] += Double(x[base+u]) }
                }

                counts[bestK] += 1
                passCounts[bestK] += 1
            }

            // Incremental centroid update per cluster
            for k in 0..<ks {
                let ck = counts[k]
                if ck > 0 {
                    let oldN = globalCounts[k]
                    let newN = oldN &+ ck
                    globalCounts[k] = newN
                    let oldW = Double(oldN) / Double(newN)
                    let newW = Double(ck) / Double(newN)
                    let baseC = k * dsub
                    // Compute batch mean and blend with existing centroid
                    for u in 0..<dsub {
                        let oldVal = Double(C[baseC + u])
                        let batchMean = sums[baseC + u] / Double(ck)
                        let updated = oldW * oldVal + newW * batchMean
                        let v = Float(updated)
                        C[baseC + u] = v.isFinite ? v : 0
                    }
                }
            }

            iters += 1
            processed = e
        }

        // Pass-level empty repair: operate only on clusters that never received assignments overall
        var emptyKs: [Int] = []
        for k in 0..<ks {
            if globalCounts[k] == 0 { emptyKs.append(k) }
        }
        if !emptyKs.isEmpty {
            let evalLim = min(nI, (cfg.sampleN > 0 ? Int(cfg.sampleN) : cfg.distEvalN))
            if evalLim > 0 {
                // Precompute min distance to any centroid for sampled points
                var mins = [Float](repeating: 0, count: evalLim)
                var inds = [Int](repeating: 0, count: evalLim)
                for t in 0..<evalLim {
                    let i = Int(idx[t % idx.count])
                    inds[t] = i
                    let base = i * d + j * dsub
                    var minD: Float
                    if let coarse = coarse, let assign = assign {
                        var coarse = coarse
                        let gid = Int(assign[i])
                        let gbase = gid * d + j * dsub
                        minD = l2Sq(&x[base], &C[0], dsub, subtract: &coarse[gbase])
                        for kk in 1..<ks {
                            let di = l2Sq(&x[base], &C[kk*dsub], dsub, subtract: &coarse[gbase])
                            if di < minD { minD = di }
                        }
                    } else {
                        minD = l2Sq(&x[base], &C[0], dsub)
                        for kk in 1..<ks {
                            let di = l2Sq(&x[base], &C[kk*dsub], dsub)
                            if di < minD { minD = di }
                        }
                    }
                    mins[t] = minD
                }
                // Select top-|emptyKs| farthest points (largest min distance)
                let ecount = emptyKs.count
                let order = (0..<evalLim).sorted { mins[$0] > mins[$1] }
                for (rank, kEmpty) in emptyKs.enumerated() where rank < order.count {
                    let pickIdx = order[rank]
                    let i = inds[pickIdx]
                    let base = i * d + j * dsub
                    if let coarse = coarse, let assign = assign {
                        let gid = Int(assign[i]); let gbase = gid * d + j * dsub
                        for u in 0..<dsub { C[kEmpty*dsub + u] = x[base + u] - coarse[gbase + u] }
                    } else {
                        for u in 0..<dsub { C[kEmpty*dsub + u] = x[base + u] }
                    }
                    globalCounts[kEmpty] = 1
                    emptiesFixed += 1
                }
            }
        }
        #if DEBUG
        let nonEmptyPass = passCounts.filter { $0 > 0 }.count
        let emptyPass = ks - nonEmptyPass
        let nonEmptyTotal = globalCounts.filter { $0 > 0 }.count
        let emptyTotal = ks - nonEmptyTotal
        let maxCount = passCounts.max() ?? 0
        let minCount = passCounts.filter { $0 > 0 }.min() ?? 0
        print("[PQTrain][mini][debug] j=\(j) pass=\(p+1) emptyPass=\(emptyPass) emptyTotal=\(emptyTotal) minPassCount=\(minCount) maxPassCount=\(maxCount)")
        for v in C { if !v.isFinite { print("[PQTrain][mini][debug] non-finite centroid value detected"); assertionFailure("Non-finite centroid") } }
        #endif
        if cfg.verbose { print("[PQTrain][mini] j=\(j) pass=\(p+1) completed; iters so far=\(iters)") }
    }

    // Final sanitation of centroids to eliminate any residual non-finite values
    for t in 0..<(ks * dsub) {
        let v = C[t]
        if !v.isFinite { C[t] = 0 }
    }

    // Estimate distortion on a limited sample to keep runtime bounded
    var total: Double = 0
    var used = 0
    let evalLim = min(nI, (cfg.sampleN > 0 ? Int(cfg.sampleN) : cfg.distEvalN))
    // Reuse current permutation for sampling
    for t in 0..<evalLim {
        let i = Int(idx[t % idx.count])
        let base = i * d + j * dsub
        var best = l2Sq(&x[base], &C[0], dsub)
        for k in 1..<ks {
            let dk = l2Sq(&x[base], &C[k*dsub], dsub)
            if dk < best { best = dk }
        }
        if best < 0 { best = 0 }
        if best.isFinite {
            total += Double(best)
            used += 1
        }
    }
    if used > 0 && total.isFinite {
        outDistortion = total / Double(used)
    } else {
        outDistortion = 1.0
    }
    outIters = iters
    outEmpties += emptiesFixed
}

private func minibatchKMeansSubspaceChunk(
    xChunk: [Float], n: Int64, d: Int, j: Int, dsub: Int, ks: Int,
    coarse: [Float]?, assignChunk: [Int32]?,
    cfg: PQTrainConfig, rng: inout Xoroshiro128, C: inout [Float],
    globalCounts: inout [Int64], passCounts: inout [Int64],
    sampleProb: Double = 1.0
) {
    // Create local var copies for &array[index] syntax
    var xChunk = xChunk
    let coarse = coarse

    let nI = Int(n)
    if nI == 0 { return }
    var idx = [UInt32](repeating: 0, count: nI)
    for i in 0..<nI { idx[i] = UInt32(i) }
    randpermInPlace(&idx, rng: &rng)

    let B = max(1, cfg.batchSize)
    // Reusable per-batch accumulators
    var sums = [Double](repeating: 0, count: ks * dsub)
    var counts = [Int64](repeating: 0, count: ks)
    var s = 0
    while s < nI {
        let e = min(s + B, nI)
        let nb = e - s
        // zero reusable accumulators
        for i in 0..<(ks * dsub) { sums[i] = 0 }
        for i in 0..<ks { counts[i] = 0 }
        for t in 0..<nb {
            // Probabilistic thinning to reduce per-pass workload across chunks
            if sampleProb < 1.0 {
                let u = rng.uniformF64()
                if u > sampleProb { continue }
            }
            let i = Int(idx[s + t])
            let base = i * d + j * dsub
            var bestK = 0
            var bestD: Float

            if let coarse = coarse, let assign = assignChunk {
                var coarse = coarse  // Re-shadow as var for &coarse syntax
                let gid = Int(assign[i]); let gbase = gid * d + j * dsub
                bestD = l2Sq(&xChunk[base], &C[0], dsub, subtract: &coarse[gbase])
                for k in 1..<ks {
                    let dk = l2Sq(&xChunk[base], &C[k*dsub], dsub, subtract: &coarse[gbase])
                    if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                }
                for u in 0..<dsub { sums[bestK*dsub + u] += Double(xChunk[base+u] - coarse[gbase+u]) }
            } else {
                bestD = l2Sq(&xChunk[base], &C[0], dsub)
                for k in 1..<ks {
                    let dk = l2Sq(&xChunk[base], &C[k*dsub], dsub)
                    if dk < bestD || (dk == bestD && k < bestK) { bestD = dk; bestK = k }
                }
                for u in 0..<dsub { sums[bestK*dsub + u] += Double(xChunk[base+u]) }
            }
            counts[bestK] += 1
        }
        for k in 0..<ks {
            let ck = counts[k]
            if ck > 0 {
                let oldN = globalCounts[k]
                let newN = oldN &+ ck
                globalCounts[k] = newN
                let oldW = Double(oldN) / Double(newN)
                let newW = Double(ck) / Double(newN)
                let baseC = k * dsub
                for u in 0..<dsub {
                    let oldVal = Double(C[baseC + u])
                    let batchMean = sums[baseC + u] / Double(ck)
                    C[baseC + u] = Float(oldW * oldVal + newW * batchMean)
                }
                passCounts[k] &+= ck
            }
        }
        s = e
    }
}

// MARK: - Streaming K-means++

private func streamingKMeansppSeed(
    xChunks: [[Float]], nChunks: [Int64],
    d: Int, j: Int, dsub: Int, ks: Int,
    coarse: [Float]?, assignChunks: [[Int32]]?,
    rng: inout Xoroshiro128,
    outC: inout [Float]
) {
    let nTotal: Int64 = nChunks.reduce(0, +)
    var pick = Int64(rng.uniformF64() * Double(nTotal))
    if pick < 0 { pick = 0 }
    if pick >= nTotal { pick = nTotal - 1 }
    var cIdx = 0; var off = pick
    while off >= nChunks[cIdx] { off -= nChunks[cIdx]; cIdx += 1 }
    let base0 = Int(off) * d + j * dsub
    let x0 = xChunks[cIdx]
    if let coarse = coarse, let aChunks = assignChunks {
        let gid = Int(aChunks[cIdx][Int(off)])
        let gbase = gid * d + j * dsub
        for u in 0..<dsub { outC[u] = x0[base0 + u] - coarse[gbase + u] }
    } else {
        for u in 0..<dsub { outC[u] = x0[base0 + u] }
    }

    for k in 1..<ks {
        var sum: Double = 0
        for (c, nc) in nChunks.enumerated() where nc > 0 {
            var xc = xChunks[c]  // var required for &xc[index] syntax
            for i in 0..<Int(nc) {
                let base = i * d + j * dsub
                var md: Float = .infinity
                for kk in 0..<k {
                    let di = l2Sq(&xc[base], &outC[kk*dsub], dsub)
                    if di < md { md = di }
                }
                sum += Double(md)
            }
        }
        if !(sum > 0) {
            let base = 0 * d + j * dsub
            for u in 0..<dsub { outC[k*dsub + u] = xChunks[0][base + u] }
            continue
        }

        var r = rng.uniformF64() * sum
        var chosen = false
        outer: for (c, nc) in nChunks.enumerated() where nc > 0 {
            var xc = xChunks[c]  // var required for &xc[index] syntax
            for i in 0..<Int(nc) {
                let base = i * d + j * dsub
                var md: Float = .infinity
                for kk in 0..<k {
                    let di = l2Sq(&xc[base], &outC[kk*dsub], dsub)
                    if di < md { md = di }
                }
                r -= Double(md)
                if r <= 0 {
                    for u in 0..<dsub { outC[k*dsub + u] = xc[base + u] }
                    chosen = true
                    break outer
                }
            }
        }
        if !chosen {
            for u in 0..<dsub { outC[k*dsub + u] = outC[u] }
        }
    }
}

// MARK: - Utilities

@inline(__always) private func nowSec() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) * 1e-9
}
