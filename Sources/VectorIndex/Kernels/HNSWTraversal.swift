//
//  HNSWTraversal.swift
//  Kernel #33 — HNSW Traversal (Greedy + efSearch)
//  See spec #33. Provides Swift APIs and C ABI shims.
//

import Foundation
import VectorCore

public enum HNSWMetric: Int32 { case L2 = 0, IP = 1, COSINE = 2 }

public struct HNSWTraversalTelemetry {
    public var edgesVisited: Int = 0
    public var neighborBatches: Int = 0
    public var candidatesPushed: Int = 0
    public var earlyExits: Int = 0
    public var greedy_ns: UInt64 = 0
    public var efsearch_ns: UInt64 = 0
    public var scoring_ns: UInt64 = 0
    public var total_ns: UInt64 = 0
}

public enum HNSWTelemetryRecorder { public nonisolated(unsafe) static var record: ((HNSWTraversalTelemetry) -> Void)? }

@inline(__always) private func bitGet(_ bits: UnsafePointer<UInt64>?, _ idx: Int) -> Bool {
    guard let bits = bits else { return true }
    let w = idx >> 6, b = idx & 63
    let word = bits.advanced(by: w).pointee
    return ((word >> UInt64(b)) & 1) != 0
}
@inline(__always) private func visitedTestAndSet(_ words: UnsafeMutablePointer<UInt64>, _ idx: Int) -> Bool {
    let w = idx >> 6, b = idx & 63
    let mask: UInt64 = 1 << UInt64(b)
    let ptr = words.advanced(by: w)
    let old = ptr.pointee
    if (old & mask) != 0 { return true }
    ptr.pointee = old | mask
    return false
}
@inline(__always) private func clampID(_ v: Int32, N: Int) -> Int? { let x = Int(v); return (x >= 0 && x < N) ? x : nil }

@inline(__always) private func dot_f32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    var acc: Float = 0; var i = 0; let u = d & ~3
    while i < u { acc += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]; i += 4 }
    while i < d { acc += a[i] * b[i]; i += 1 }
    return acc
}
@inline(__always) private func l2sq_f32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    var acc: Float = 0; var i = 0; let u = d & ~3
    while i < u {
        let d0 = a[i]-b[i], d1 = a[i+1]-b[i+1], d2 = a[i+2]-b[i+2], d3 = a[i+3]-b[i+3]
        acc += d0*d0; acc += d1*d1; acc += d2*d2; acc += d3*d3; i += 4
    }
    while i < d { let dv = a[i]-b[i]; acc += dv*dv; i += 1 }
    return acc
}
@inline(__always) private func invnorm_f32(_ a: UnsafePointer<Float>, _ d: Int) -> Float {
    let n = sqrtf(max(1e-12, dot_f32(a, a, d))); return 1.0 / n
}
@inline(__always) private func distance_f32(q: UnsafePointer<Float>, d: Int, x: UnsafePointer<Float>, metric: HNSWMetric, qInv: Float?, xInv: Float?) -> Float {
    switch metric {
    case .L2: return l2sq_f32(q, x, d)
    case .IP: return -dot_f32(q, x, d)
    case .COSINE:
        let qi = qInv ?? invnorm_f32(q, d)
        let xi = xInv ?? invnorm_f32(x, d)
        let sim = dot_f32(q, x, d) * qi * xi
        return 1.0 - sim
    }
}

private struct HeapNode { var dist: Float; var id: Int32 }
private struct MinHeap {
    var dists: [Float] = []; var ids: [Int32] = []
    var isEmpty: Bool { dists.isEmpty }; var count: Int { dists.count }
    @inline(__always) private func less(_ i: Int, _ j: Int) -> Bool {
        let di = dists[i], dj = dists[j]
        if di < dj { return true }; if di > dj { return false }; return ids[i] < ids[j]
    }
    mutating func clear(reserving n: Int) { dists.removeAll(keepingCapacity: true); ids.removeAll(keepingCapacity: true); if dists.capacity < n { dists.reserveCapacity(n) }; if ids.capacity < n { ids.reserveCapacity(n) } }
    mutating func push(_ n: HeapNode) { dists.append(n.dist); ids.append(n.id); siftUp(count - 1) }
    mutating func popMin() -> HeapNode { let last = count - 1; let out = HeapNode(dist: dists[0], id: ids[0]); dists[0] = dists[last]; ids[0] = ids[last]; dists.removeLast(); ids.removeLast(); if !dists.isEmpty { siftDown(0) }; return out }
    @inline(__always) func peekDist() -> Float { dists[0] }
    private mutating func siftUp(_ i0: Int) { var i = i0; while i > 0 { let p = (i - 1) >> 1; if less(i, p) { dists.swapAt(i, p); ids.swapAt(i, p); i = p } else { break } } }
    private mutating func siftDown(_ i0: Int) { var i = i0; while true { let l = (i << 1) + 1; if l >= count { break }; var m = l; let r = l + 1; if r < count, less(r, l) { m = r }; if less(m, i) { dists.swapAt(i, m); ids.swapAt(i, m); i = m } else { break } } }
}

@inline(__always) private func resultInsert(ids: inout [Int32], dists: inout [Float], count: inout Int, capacity ef: Int, id: Int32, dist: Float) {
    if count == ef, dist > dists[count - 1] { return }
    var lo = 0, hi = count
    while lo < hi {
        let mid = (lo + hi) >> 1
        let dm = dists[mid]
        if dist < dm || (dist == dm && id < ids[mid]) { hi = mid } else { lo = mid + 1 }
    }
    if count < ef { ids.insert(id, at: lo); dists.insert(dist, at: lo); count &+= 1 } else { ids.removeLast(); dists.removeLast(); ids.insert(id, at: lo); dists.insert(dist, at: lo) }
}

@inline(__always) private func selectBatchSize(_ d: Int) -> Int { if d <= 256 { return 64 }; if d <= 1024 { return 32 }; return 16 }
private func scoreNeighborsBatch_f32(
    q: UnsafePointer<Float>, d: Int,
    xb: UnsafePointer<Float>, N: Int,
    metric: HNSWMetric,
    qInv: Float?, invNorms: UnsafePointer<Float>?,
    ids: UnsafeBufferPointer<Int32>,
    outDists: UnsafeMutableBufferPointer<Float>
) {
    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif
    let count = ids.count
    if ids.isEmpty { return }
    // Gather neighbors contiguously into a scratch buffer
    var gathered = [Float](repeating: 0, count: count * d)
    gathered.withUnsafeMutableBufferPointer { gbuf in
        var row = 0
        while row < count {
            let id = Int(ids[row])
            if id >= 0 && id < N {
                let src = xb.advanced(by: id * d)
                let dst = gbuf.baseAddress!.advanced(by: row * d)
                dst.update(from: src, count: d)
            }
            row += 1
        }
        // Score via ScoreBlock
        let metricMap: SupportedDistanceMetric = {
            switch metric {
            case .L2: return .euclidean
            case .IP: return .dotProduct
            case .COSINE: return .cosine
            }
        }()
        var scores = [Float](repeating: 0, count: count)
        scores.withUnsafeMutableBufferPointer { sbuf in
            let handle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle? = {
                if metric == .COSINE {
                    let qInvN = qInv ?? IndexOps.Support.Norms.queryInvNorm(query: q, dimension: d)
                    return .init(dbInvNormsF32: invNorms, dbInvNormsF16: nil, queryInvNorm: qInvN)
                } else { return nil }
            }()
            IndexOps.Scoring.ScoreBlock.run(
                q: q,
                xb: gbuf.baseAddress!,
                n: count,
                d: d,
                metric: metricMap,
                out: sbuf.baseAddress!,
                cosineNorms: handle
            )
        }
        // Convert scores to distances if needed
        for i in 0..<count {
            switch metric {
            case .L2:
                outDists[i] = scores[i] // L2^2 distance
            case .IP:
                outDists[i] = -scores[i] // distance = -dot
            case .COSINE:
                outDists[i] = 1.0 - scores[i] // distance = 1 - cos_sim
            }
        }
    }
    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds; scoringAccum_ns &+= (t1 - t0)
    #endif
}
#if ENABLE_TELEMETRY
private var scoringAccum_ns: UInt64 = 0
#endif

@inline(__always) private func layerNeighbors(offsets: UnsafePointer<Int32>, neighbors: UnsafePointer<Int32>, u: Int, N: Int) -> UnsafeBufferPointer<Int32> {
    let beg = Int(offsets[u]); let end = Int(offsets[u &+ 1]); let base = neighbors.advanced(by: beg); return UnsafeBufferPointer(start: base, count: max(0, end - beg))
}

private func greedyDescent_core(q: UnsafePointer<Float>, d: Int, entryPoint: Int32, maxLevel: Int32, offsetsPerLayer: UnsafePointer<UnsafePointer<Int32>?>, neighborsPerLayer: UnsafePointer<UnsafePointer<Int32>?>, xb: UnsafePointer<Float>, N: Int, metric: HNSWMetric, invNorms: UnsafePointer<Float>?) -> Int32 {
    guard N > 0, d > 0, entryPoint >= 0, entryPoint < Int32(N) else { return -1 }
    var current = Int(entryPoint)
    let qInv: Float? = (metric == .COSINE) ? invnorm_f32(q, d) : nil
    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif
    if maxLevel > 0 {
        var lvl = Int(maxLevel)
        while lvl >= 1 {
            guard let offsets = offsetsPerLayer.advanced(by: lvl).pointee, let neigh = neighborsPerLayer.advanced(by: lvl).pointee else { return -1 }
            var bestID = current
            var bestDist = distance_f32(q: q, d: d, x: xb.advanced(by: current * d), metric: metric, qInv: qInv, xInv: (metric == .COSINE ? invNorms?.advanced(by: current).pointee : nil))
            var improved = true
            while improved {
                improved = false
                let nbrs = layerNeighbors(offsets: offsets, neighbors: neigh, u: bestID, N: N)
                if nbrs.isEmpty { break }
                var tmpIDs = [Int32](repeating: 0, count: nbrs.count)
                tmpIDs.withUnsafeMutableBufferPointer { buf in for i in 0..<nbrs.count { buf[i] = nbrs[i] } }
                var tmpD = [Float](repeating: 0, count: nbrs.count)
                tmpIDs.withUnsafeBufferPointer { idb in tmpD.withUnsafeMutableBufferPointer { db in scoreNeighborsBatch_f32(q: q, d: d, xb: xb, N: N, metric: metric, qInv: qInv, invNorms: invNorms, ids: idb, outDists: db) } }
                for i in 0..<nbrs.count {
                    let nid = Int(tmpIDs[i]); guard nid >= 0 && nid < N else { continue }
                    let dist = tmpD[i]
                    if dist < bestDist || (dist == bestDist && nid < bestID) { bestDist = dist; bestID = nid; improved = true }
                }
            }
            current = bestID; lvl -= 1
        }
    }
    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds; greedy_ns_accum &+= (t1 - t0)
    #endif
    return Int32(current)
}
#if ENABLE_TELEMETRY
private var greedy_ns_accum: UInt64 = 0
private var efsearch_ns_accum: UInt64 = 0
private var earlyExitCount: Int = 0
private var edgesVisitedCount: Int = 0
private var neighborBatchesCount: Int = 0
private var candidatesPushedCount: Int = 0
#endif

private func efSearch_core(q: UnsafePointer<Float>, d: Int, enterL0: Int32, offsetsL0: UnsafePointer<Int32>, neighborsL0: UnsafePointer<Int32>, xb: UnsafePointer<Float>, N: Int, ef: Int, metric: HNSWMetric, allowBits: UnsafePointer<UInt64>?, allowN: Int, invNorms: UnsafePointer<Float>?, idsOut: UnsafeMutablePointer<Int32>, distsOut: UnsafeMutablePointer<Float>) -> Int {
    if N <= 0 || d <= 0 || ef <= 0 { return -1 }
    guard enterL0 >= 0 && enterL0 < Int32(N) else { return -1 }
    let qInv: Float? = (metric == .COSINE) ? invnorm_f32(q, d) : nil
    let allowDomain = (allowBits != nil && allowN > 0) ? allowN : 0
    let visitedWords = (N + 63) >> 6
    let visited = UnsafeMutablePointer<UInt64>.allocate(capacity: visitedWords)
    visited.initialize(repeating: 0, count: visitedWords); defer { visited.deallocate() }
    var cand = MinHeap(); cand.clear(reserving: ef * 2)
    var resIDs = [Int32](); resIDs.reserveCapacity(ef)
    var resD = [Float](); resD.reserveCapacity(ef)
    var resCount = 0
    let enterIdx = Int(enterL0)
    let enterX = xb.advanced(by: enterIdx * d)
    let enterDist = distance_f32(q: q, d: d, x: enterX, metric: metric, qInv: qInv, xInv: nil)
    _ = visitedTestAndSet(visited, enterIdx)
    cand.push(HeapNode(dist: enterDist, id: enterL0))
    resultInsert(ids: &resIDs, dists: &resD, count: &resCount, capacity: ef, id: enterL0, dist: enterDist)
    #if ENABLE_TELEMETRY
    let t0 = DispatchTime.now().uptimeNanoseconds
    #endif
    while !cand.isEmpty {
        let best = cand.peekDist()
        let worst = (resCount == ef) ? resD[resCount - 1] : Float.greatestFiniteMagnitude
        if resCount == ef && best > worst { #if ENABLE_TELEMETRY
            earlyExitCount &+= 1
            #endif
            break }
        let node = cand.popMin(); let u = Int(node.id)
        let nbrs = layerNeighbors(offsets: offsetsL0, neighbors: neighborsL0, u: u, N: N)
        if nbrs.isEmpty { continue }
        var toScoreIDs = [Int32](); toScoreIDs.reserveCapacity(nbrs.count)
        for v32 in nbrs {
            guard let v = clampID(v32, N: N) else { continue }
            if allowDomain != 0 && (v < 0 || v >= allowDomain || !bitGet(allowBits, v)) { continue }
            if visitedTestAndSet(visited, v) { continue }
            toScoreIDs.append(Int32(v))
        }
        if !toScoreIDs.isEmpty {
            var dvec = [Float](repeating: 0, count: toScoreIDs.count)
            toScoreIDs.withUnsafeBufferPointer { idb in dvec.withUnsafeMutableBufferPointer { db in scoreNeighborsBatch_f32(q: q, d: d, xb: xb, N: N, metric: metric, qInv: qInv, invNorms: invNorms, ids: idb, outDists: db) } }
            #if ENABLE_TELEMETRY
            neighborBatchesCount &+= 1; edgesVisitedCount &+= toScoreIDs.count
            #endif
            for i in 0..<toScoreIDs.count {
                let vid = toScoreIDs[i]; let dist = dvec[i]
                cand.push(HeapNode(dist: dist, id: vid))
                resultInsert(ids: &resIDs, dists: &resD, count: &resCount, capacity: ef, id: vid, dist: dist)
                #if ENABLE_TELEMETRY
                candidatesPushedCount &+= 1
                #endif
            }
        }
    }
    #if ENABLE_TELEMETRY
    let t1 = DispatchTime.now().uptimeNanoseconds; efsearch_ns_accum &+= (t1 - t0)
    #endif
    for i in 0..<resCount { idsOut.advanced(by: i).pointee = resIDs[i]; distsOut.advanced(by: i).pointee = resD[i] }
    return resCount
}

public enum HNSWTraversal {
    public static func greedyDescent(q: UnsafePointer<Float>, d: Int, entryPoint: Int32, maxLevel: Int32, offsetsPerLayer: UnsafePointer<UnsafePointer<Int32>?>, neighborsPerLayer: UnsafePointer<UnsafePointer<Int32>?>, xb: UnsafePointer<Float>, N: Int, metric: HNSWMetric, invNorms: UnsafePointer<Float>?) -> Int32 {
        #if ENABLE_TELEMETRY
        greedy_ns_accum = 0; scoringAccum_ns = 0
        #endif
        let id = greedyDescent_core(q: q, d: d, entryPoint: entryPoint, maxLevel: maxLevel, offsetsPerLayer: offsetsPerLayer, neighborsPerLayer: neighborsPerLayer, xb: xb, N: N, metric: metric, invNorms: invNorms)
        #if ENABLE_TELEMETRY
        var t = HNSWTraversalTelemetry(); t.greedy_ns = greedy_ns_accum; t.scoring_ns = scoringAccum_ns; t.total_ns = greedy_ns_accum &+ scoringAccum_ns; HNSWTelemetryRecorder.record?(t)
        #endif
        return id
    }
    public static func efSearch(q: UnsafePointer<Float>, d: Int, enterL0: Int32, offsetsL0: UnsafePointer<Int32>, neighborsL0: UnsafePointer<Int32>, xb: UnsafePointer<Float>, N: Int, ef: Int, metric: HNSWMetric, allowBits: UnsafePointer<UInt64>?, allowN: Int, invNorms: UnsafePointer<Float>?, idsOut: UnsafeMutablePointer<Int32>, distsOut: UnsafeMutablePointer<Float>) -> Int {
        #if ENABLE_TELEMETRY
        greedy_ns_accum = 0; efsearch_ns_accum = 0; scoringAccum_ns = 0; earlyExitCount = 0; edgesVisitedCount = 0; neighborBatchesCount = 0; candidatesPushedCount = 0
        #endif
        let c = efSearch_core(q: q, d: d, enterL0: enterL0, offsetsL0: offsetsL0, neighborsL0: neighborsL0, xb: xb, N: N, ef: ef, metric: metric, allowBits: allowBits, allowN: allowN, invNorms: invNorms, idsOut: idsOut, distsOut: distsOut)
        #if ENABLE_TELEMETRY
        var t = HNSWTraversalTelemetry(); t.efsearch_ns = efsearch_ns_accum; t.scoring_ns = scoringAccum_ns; t.total_ns = efsearch_ns_accum &+ scoringAccum_ns; t.earlyExits = earlyExitCount; t.edgesVisited = edgesVisitedCount; t.neighborBatches = neighborBatchesCount; t.candidatesPushed = candidatesPushedCount; HNSWTelemetryRecorder.record?(t)
        #endif
        return c
    }
    public static func traverse(q: UnsafePointer<Float>, d: Int, entryPoint: Int32, maxLevel: Int32, offsetsPerLayer: UnsafePointer<UnsafePointer<Int32>?>, neighborsPerLayer: UnsafePointer<UnsafePointer<Int32>?>, xb: UnsafePointer<Float>, N: Int, ef: Int, metric: HNSWMetric, allowBits: UnsafePointer<UInt64>?, allowN: Int, invNorms: UnsafePointer<Float>?, idsOut: UnsafeMutablePointer<Int32>, distsOut: UnsafeMutablePointer<Float>) -> Int {
        let enterL0 = greedyDescent_core(q: q, d: d, entryPoint: entryPoint, maxLevel: maxLevel, offsetsPerLayer: offsetsPerLayer, neighborsPerLayer: neighborsPerLayer, xb: xb, N: N, metric: metric, invNorms: nil)
        if enterL0 < 0 { return -1 }
        return efSearch_core(q: q, d: d, enterL0: enterL0, offsetsL0: offsetsPerLayer.pointee!, neighborsL0: neighborsPerLayer.pointee!, xb: xb, N: N, ef: ef, metric: metric, allowBits: allowBits, allowN: allowN, invNorms: invNorms, idsOut: idsOut, distsOut: distsOut)
    }
}

@_cdecl("hnsw_greedy_descent_f32")
public func c_hnsw_greedy_descent_f32(_ q: UnsafePointer<Float>?, _ d: Int32, _ entryPoint: Int32, _ maxLevel: Int32, _ offsetsPerLayer: UnsafePointer<UnsafePointer<Int32>?>?, _ neighborsPerLayer: UnsafePointer<UnsafePointer<Int32>?>?, _ xb: UnsafePointer<Float>?, _ N: Int32, _ metric: Int32, _ optionalInvNorms: UnsafePointer<Float>?) -> Int32 {
    guard let q = q, let offsetsPerLayer = offsetsPerLayer, let neighborsPerLayer = neighborsPerLayer, let xb = xb else { return -1 }
    guard d > 0, N > 0, let m = HNSWMetric(rawValue: metric) else { return -1 }
    return HNSWTraversal.greedyDescent(q: q, d: Int(d), entryPoint: entryPoint, maxLevel: maxLevel, offsetsPerLayer: offsetsPerLayer, neighborsPerLayer: neighborsPerLayer, xb: xb, N: Int(N), metric: m, invNorms: optionalInvNorms)
}
@_cdecl("hnsw_efsearch_f32")
public func c_hnsw_efsearch_f32(_ q: UnsafePointer<Float>?, _ d: Int32, _ enterL0: Int32, _ offsetsL0: UnsafePointer<Int32>?, _ neighborsL0: UnsafePointer<Int32>?, _ xb: UnsafePointer<Float>?, _ N: Int32, _ ef: Int32, _ metric: Int32, _ allowBitset: UnsafePointer<UInt64>?, _ allowN: Int32, _ idsOut: UnsafeMutablePointer<Int32>?, _ distsOut: UnsafeMutablePointer<Float>?) -> Int32 {
    guard let q = q, let offsetsL0 = offsetsL0, let neighborsL0 = neighborsL0, let xb = xb, let idsOut = idsOut, let distsOut = distsOut else { return -1 }
    guard d > 0, N > 0, ef > 0, let m = HNSWMetric(rawValue: metric) else { return -1 }
    let c = HNSWTraversal.efSearch(q: q, d: Int(d), enterL0: enterL0, offsetsL0: offsetsL0, neighborsL0: neighborsL0, xb: xb, N: Int(N), ef: Int(ef), metric: m, allowBits: allowBitset, allowN: Int(allowN), invNorms: nil, idsOut: idsOut, distsOut: distsOut)
    return Int32(c)
}
@_cdecl("hnsw_traverse_f32")
public func c_hnsw_traverse_f32(_ q: UnsafePointer<Float>?, _ d: Int32, _ entryPoint: Int32, _ maxLevel: Int32, _ offsetsPerLayer: UnsafePointer<UnsafePointer<Int32>?>?, _ neighborsPerLayer: UnsafePointer<UnsafePointer<Int32>?>?, _ xb: UnsafePointer<Float>?, _ N: Int32, _ ef: Int32, _ metric: Int32, _ allowBitset: UnsafePointer<UInt64>?, _ allowN: Int32, _ idsOut: UnsafeMutablePointer<Int32>?, _ distsOut: UnsafeMutablePointer<Float>?) -> Int32 {
    guard let q = q, let offsetsPerLayer = offsetsPerLayer, let neighborsPerLayer = neighborsPerLayer, let xb = xb, let idsOut = idsOut, let distsOut = distsOut else { return -1 }
    guard d > 0, N > 0, ef > 0, let m = HNSWMetric(rawValue: metric) else { return -1 }
    let c = HNSWTraversal.traverse(q: q, d: Int(d), entryPoint: entryPoint, maxLevel: maxLevel, offsetsPerLayer: offsetsPerLayer, neighborsPerLayer: neighborsPerLayer, xb: xb, N: Int(N), ef: Int(ef), metric: m, allowBits: allowBitset, allowN: Int(allowN), invNorms: nil, idsOut: idsOut, distsOut: distsOut)
    return Int32(c)
}
