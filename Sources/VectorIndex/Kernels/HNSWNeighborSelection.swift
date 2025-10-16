//
//  HNSWNeighborSelection.swift
//  Kernel #34 — HNSW Neighbor Selection (Diversity + Prune)
//
//  Implements:
//   - SelectNeighborsHeuristic (diversity) for x_new vs. candidate ids
//   - PruneNeighbors for an anchor node’s adjacency at a given layer
//   - Deterministic ordering: (dist asc, id asc)
//   - Batch scoring fallback (replaceable by ScoreBlock #04)
//   - Telemetry counters & timers (opt-in)
//
//  Notes:
//   - We assume `candidates` were produced for the same layer `l`.
//   - For cosine, pass an inverse-norm cache (#09) as `optionalInvNorms` to avoid
//     recomputing norms; the query’s inverse norm is computed once.
//

import Foundation
import VectorCore

// MARK: - Telemetry (optional, #46)

public struct HNSWNeighborSelectionTelemetry {
    public var candidates_in: Int = 0
    public var candidates_scored: Int = 0
    public var accepted: Int = 0
    public var pruned_edges: Int = 0
    public var symmetry_links: Int = 0
    public var score_ns: UInt64 = 0
    public var select_ns: UInt64 = 0
    public var prune_ns: UInt64 = 0
}

public enum HNSWNeighborSelectionRecorder {
    public nonisolated(unsafe) static var record: ((HNSWNeighborSelectionTelemetry) -> Void)? = nil
}

// MARK: - Distance kernels (fallback; replaceable by ScoreBlock)

@usableFromInline
@inline(__always) func ns_dot_f32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    var acc: Float = 0
    var i = 0
    let u = d & ~3
    while i < u {
        acc += a[i]   * b[i]
        acc += a[i+1] * b[i+1]
        acc += a[i+2] * b[i+2]
        acc += a[i+3] * b[i+3]
        i += 4
    }
    while i < d { acc += a[i] * b[i]; i += 1 }
    return acc
}

@usableFromInline
@inline(__always) func ns_l2sq_f32(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ d: Int) -> Float {
    var acc: Float = 0
    var i = 0
    let u = d & ~3
    while i < u {
        let d0 = a[i]   - b[i]
        let d1 = a[i+1] - b[i+1]
        let d2 = a[i+2] - b[i+2]
        let d3 = a[i+3] - b[i+3]
        acc += d0 * d0
        acc += d1 * d1
        acc += d2 * d2
        acc += d3 * d3
        i += 4
    }
    while i < d { let dv = a[i] - b[i]; acc += dv * dv; i += 1 }
    return acc
}

@usableFromInline
@inline(__always) func ns_invnorm_f32(_ x: UnsafePointer<Float>, _ d: Int) -> Float {
    let n = sqrtf(max(1e-12, ns_dot_f32(x, x, d)))
    return 1.0 / n
}

@usableFromInline
@inline(__always) func ns_distance_f32(
    a: UnsafePointer<Float>, b: UnsafePointer<Float>, d: Int,
    metric: HNSWMetric, invA: Float?, invB: Float?
) -> Float {
    switch metric {
    case .L2: return ns_l2sq_f32(a, b, d)
    case .IP: return -ns_dot_f32(a, b, d)
    case .COSINE:
        let inva = invA ?? ns_invnorm_f32(a, d)
        let invb = invB ?? ns_invnorm_f32(b, d)
        return 1.0 - (ns_dot_f32(a, b, d) * inva * invb)
    }
}

// MARK: - Helpers

@usableFromInline
@inline(__always) func ns_clampID(_ id: Int32, _ N: Int) -> Int? {
    let u = Int(id); return (u >= 0 && u < N) ? u : nil
}
@usableFromInline struct NSPair { @usableFromInline var id: Int32; @usableFromInline var dist: Float }
@usableFromInline @inline(__always) func ns_byDistIdAsc(_ a: NSPair, _ b: NSPair) -> Bool {
    if a.dist < b.dist { return true }
    if a.dist > b.dist { return false }
    return a.id < b.id
}
@usableFromInline struct NSMaxHeap {
    var dists: [Float] = []
    var ids: [Int32] = []
    var count: Int { dists.count }
    mutating func clear() { dists.removeAll(keepingCapacity: true); ids.removeAll(keepingCapacity: true) }
    @inline(__always) private func greater(_ i: Int, _ j: Int) -> Bool {
        let di = dists[i], dj = dists[j]
        if di > dj { return true }
        if di < dj { return false }
        return ids[i] > ids[j]
    }
    mutating func push(_ p: NSPair) { dists.append(p.dist); ids.append(p.id); siftUp(count - 1) }
    mutating func popMax() -> NSPair { let last = count - 1; let out = NSPair(id: ids[0], dist: dists[0]); dists[0] = dists[last]; ids[0] = ids[last]; dists.removeLast(); ids.removeLast(); if !dists.isEmpty { siftDown(0) }; return out }
    @inline(__always) mutating func maybePushCapped(_ p: NSPair, cap: Int) { if count < cap { push(p); return }; let wd=dists[0], wi=ids[0]; if p.dist < wd || (p.dist == wd && p.id < wi) { dists[0] = p.dist; ids[0] = p.id; siftDown(0) } }
    private mutating func siftUp(_ i0: Int) { var i = i0; while i > 0 { let p = (i - 1) >> 1; if greater(i, p) { dists.swapAt(i, p); ids.swapAt(i, p); i = p } else { break } } }
    private mutating func siftDown(_ i0: Int) { var i = i0; while true { let l = (i << 1) + 1; if l >= count { break }; var m = l; let r = l + 1; if r < count, greater(r, l) { m = r }; if greater(m, i) { dists.swapAt(i, m); ids.swapAt(i, m); i = m } else { break } } }
    mutating func toSortedPairs() -> [NSPair] { var out: [NSPair] = []; out.reserveCapacity(count); while !dists.isEmpty { out.append(popMax()) }; out.reverse(); return out }
}

@inline(__always) private func ns_selectBatchSize(_ d: Int) -> Int { if d <= 256 { return 64 }; if d <= 1024 { return 32 }; return 16 }

// MARK: - Core API (Swift) — selection

public func hnsw_select_neighbors_f32_swift(
    x_new: UnsafePointer<Float>, d: Int,
    candidates: UnsafePointer<Int32>, candCount: Int,
    xb: UnsafePointer<Float>, N: Int,
    M: Int, layer: Int,
    metric: HNSWMetric,
    optionalInvNorms: UnsafePointer<Float>?,
    selectedOut: UnsafeMutablePointer<Int32>
) -> Int {
    if d <= 0 || M <= 0 || candCount <= 0 || N <= 0 { return 0 }
    // Optional fused-cosine handle for ScoreBlock
    let metricMap: SupportedDistanceMetric = {
        switch metric {
        case .L2: return .euclidean
        case .IP: return .dotProduct
        case .COSINE: return .cosine
        }
    }()
    let qInvN: Float? = (metric == .COSINE) ? IndexOps.Support.Norms.queryInvNorm(query: x_new, dimension: d) : nil
    #if ENABLE_TELEMETRY
    var tele = HNSWNeighborSelectionTelemetry(); tele.candidates_in = candCount; let score_t0 = DispatchTime.now().uptimeNanoseconds
    #endif
    var heap = NSMaxHeap(); heap.clear()
    let window = min(candCount, max(M << 1, M))
    let bs = ns_selectBatchSize(d)
    var i = 0
    while i < candCount {
        let end = min(i + bs, candCount)
        // Gather valid ids for this chunk
        var chunkIDs: [Int32] = []
        chunkIDs.reserveCapacity(end - i)
        var j = i
        while j < end {
            let cid = candidates.advanced(by: j).pointee
            if let u = ns_clampID(cid, N) { chunkIDs.append(Int32(u)) }
            j += 1
        }
        if !chunkIDs.isEmpty {
            // Gather vectors contiguously
            var gbuf = [Float](repeating: 0, count: chunkIDs.count * d)
            gbuf.withUnsafeMutableBufferPointer { dst in
                var row = 0
                for id in chunkIDs {
                    let u = Int(id)
                    let src = xb.advanced(by: u * d)
                    let out = dst.baseAddress!.advanced(by: row * d)
                    out.update(from: src, count: d)
                    row += 1
                }
            }
            // Score via ScoreBlock
            var scores = [Float](repeating: 0, count: chunkIDs.count)
            scores.withUnsafeMutableBufferPointer { sb in
                gbuf.withUnsafeBufferPointer { gb in
                    let handle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle? = {
                        if metric == .COSINE {
                            return .init(dbInvNormsF32: optionalInvNorms, dbInvNormsF16: nil, queryInvNorm: qInvN)
                        } else { return nil }
                    }()
                    IndexOps.Scoring.ScoreBlock.run(
                        q: x_new,
                        xb: gb.baseAddress!,
                        n: chunkIDs.count,
                        d: d,
                        metric: metricMap,
                        out: sb.baseAddress!,
                        cosineNorms: handle
                    )
                }
            }
            // Convert scores to distances and push to capped heap
            for (idx, id) in chunkIDs.enumerated() {
                let s = scores[idx]
                let dist: Float = {
                    switch metric {
                    case .L2: return s              // L2^2
                    case .IP: return -s             // distance = -dot
                    case .COSINE: return 1.0 - s    // distance = 1 - cos_sim
                    }
                }()
                heap.maybePushCapped(NSPair(id: id, dist: dist), cap: window)
            }
        }
        i = end
    }
    #if ENABLE_TELEMETRY
    tele.candidates_scored = heap.count; let score_t1 = DispatchTime.now().uptimeNanoseconds; tele.score_ns = score_t1 &- score_t0; let select_t0 = DispatchTime.now().uptimeNanoseconds
    #endif
    var C = heap.toSortedPairs(); C.sort(by: ns_byDistIdAsc)
    var selected: [Int32] = []; selected.reserveCapacity(M)
    for p in C where selected.count < M {
        let cID = Int(p.id)
        let cPtr = xb.advanced(by: cID * d)
        let cInv = (metric == .COSINE) ? optionalInvNorms?.advanced(by: cID).pointee : nil
        let dcq = p.dist
        var accept = true
        for sid in selected {
            let sID = Int(sid)
            let sPtr = xb.advanced(by: sID * d)
            let sInv = (metric == .COSINE) ? optionalInvNorms?.advanced(by: sID).pointee : nil
            let dcs = ns_distance_f32(a: cPtr, b: sPtr, d: d, metric: metric, invA: cInv, invB: sInv)
            if dcq > dcs { accept = false; break }
        }
        if accept { selected.append(Int32(cID)) }
    }
    if selected.count < M {
        var inSel = Set<Int32>(selected)
        for p in C where selected.count < M {
            if !inSel.contains(p.id) { selected.append(p.id); inSel.insert(p.id) }
        }
    }
    let written = min(M, selected.count)
    for k in 0..<written { selectedOut.advanced(by: k).pointee = selected[k] }
    #if ENABLE_TELEMETRY
    let select_t1 = DispatchTime.now().uptimeNanoseconds; tele.select_ns = select_t1 &- select_t0; tele.accepted = written; HNSWNeighborSelectionRecorder.record?(tele)
    #endif
    return written
}

// MARK: - Core API (Swift) — prune

public func hnsw_prune_neighbors_f32_swift(
    u: Int32,
    xb: UnsafePointer<Float>, d: Int,
    offsetsL: UnsafePointer<Int32>, neighborsL: UnsafePointer<Int32>,
    M: Int, metric: HNSWMetric,
    optionalInvNorms: UnsafePointer<Float>?,
    N: Int,
    prunedOut: UnsafeMutablePointer<Int32>
) -> Int {
    if d <= 0 || M <= 0 || N <= 0 { return 0 }
    let uInt = Int(u); guard uInt >= 0 && uInt < N else { return -1 }
    let beg = Int(offsetsL[uInt]); let end = Int(offsetsL[uInt &+ 1]); if end <= beg { return 0 }
    var ids: [Int32] = []; ids.reserveCapacity(end - beg)
    var seen = Set<Int32>(minimumCapacity: end - beg)
    for i in beg..<end { let v = neighborsL.advanced(by: i).pointee; if v == u || seen.contains(v) { continue }; if ns_clampID(v, N) != nil { ids.append(v); seen.insert(v) } }
    if ids.isEmpty { return 0 }
    #if ENABLE_TELEMETRY
    var tele = HNSWNeighborSelectionTelemetry(); let prune_t0 = DispatchTime.now().uptimeNanoseconds
    #endif
    let uPtr = xb.advanced(by: uInt * d)
    let uInv = (metric == .COSINE) ? optionalInvNorms?.advanced(by: uInt).pointee : nil
    // Score via ScoreBlock for prune
    var pairs: [NSPair] = []; pairs.reserveCapacity(ids.count)
    if !ids.isEmpty {
        // Gather neighbor vectors
        var gbuf = [Float](repeating: 0, count: ids.count * d)
        gbuf.withUnsafeMutableBufferPointer { dst in
            var row = 0
            for v32 in ids {
                let v = Int(v32)
                let src = xb.advanced(by: v * d)
                let out = dst.baseAddress!.advanced(by: row * d)
                out.update(from: src, count: d)
                row += 1
            }
        }
        var scores = [Float](repeating: 0, count: ids.count)
        let metricMap: SupportedDistanceMetric = {
            switch metric {
            case .L2: return .euclidean
            case .IP: return .dotProduct
            case .COSINE: return .cosine
            }
        }()
        let handle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle? = {
            if metric == .COSINE {
                let qInv = uInv ?? IndexOps.Support.Norms.queryInvNorm(query: uPtr, dimension: d)
                return .init(dbInvNormsF32: optionalInvNorms, dbInvNormsF16: nil, queryInvNorm: qInv)
            } else { return nil }
        }()
        scores.withUnsafeMutableBufferPointer { sb in
            gbuf.withUnsafeBufferPointer { gb in
                IndexOps.Scoring.ScoreBlock.run(
                    q: uPtr,
                    xb: gb.baseAddress!,
                    n: ids.count,
                    d: d,
                    metric: metricMap,
                    out: sb.baseAddress!,
                    cosineNorms: handle
                )
            }
        }
        // Convert scores to distances and pair with ids
        pairs.reserveCapacity(ids.count)
        for i in 0..<ids.count {
            let s = scores[i]
            let dist: Float = {
                switch metric {
                case .L2: return s
                case .IP: return -s
                case .COSINE: return 1.0 - s
                }
            }()
            pairs.append(NSPair(id: ids[i], dist: dist))
        }
    }
    pairs.sort(by: ns_byDistIdAsc)
    let keep = min(M, pairs.count)
    for i in 0..<keep { prunedOut.advanced(by: i).pointee = pairs[i].id }
    #if ENABLE_TELEMETRY
    let prune_t1 = DispatchTime.now().uptimeNanoseconds; tele.prune_ns = prune_t1 &- prune_t0; tele.pruned_edges = keep; HNSWNeighborSelectionRecorder.record?(tele)
    #endif
    return keep
}

// MARK: - C ABI shims

@_cdecl("hnsw_select_neighbors_f32")
public func c_hnsw_select_neighbors_f32(
    _ x_new: UnsafePointer<Float>?, _ d: Int32,
    _ candidates: UnsafePointer<Int32>?, _ candCount: Int32,
    _ xb: UnsafePointer<Float>?, _ N: Int32,
    _ M: Int32, _ layer: Int32,
    _ metric: Int32,
    _ optionalInvNorms: UnsafePointer<Float>?,
    _ selectedOut: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let x_new = x_new, let candidates = candidates, let xb = xb, let selectedOut = selectedOut else { return -1 }
    guard d > 0, N > 0, M >= 0, candCount >= 0 else { return -1 }
    guard let m = HNSWMetric(rawValue: metric) else { return -1 }
    let written = hnsw_select_neighbors_f32_swift(
        x_new: x_new, d: Int(d),
        candidates: candidates, candCount: Int(candCount),
        xb: xb, N: Int(N),
        M: Int(M), layer: Int(layer),
        metric: m, optionalInvNorms: optionalInvNorms,
        selectedOut: selectedOut
    )
    return Int32(written)
}

@_cdecl("hnsw_prune_neighbors_f32")
public func c_hnsw_prune_neighbors_f32(
    _ u: Int32,
    _ xb: UnsafePointer<Float>?, _ d: Int32,
    _ offsetsL: UnsafePointer<Int32>?, _ neighborsL: UnsafePointer<Int32>?,
    _ M: Int32, _ metric: Int32,
    _ optionalInvNorms: UnsafePointer<Float>?,
    _ N: Int32,
    _ prunedOut: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let xb = xb, let offsetsL = offsetsL, let neighborsL = neighborsL, let prunedOut = prunedOut else { return -1 }
    guard d > 0, N > 0, M >= 0 else { return -1 }
    guard let m = HNSWMetric(rawValue: metric) else { return -1 }
    let kept = hnsw_prune_neighbors_f32_swift(
        u: u, xb: xb, d: Int(d),
        offsetsL: offsetsL, neighborsL: neighborsL,
        M: Int(M), metric: m,
        optionalInvNorms: optionalInvNorms,
        N: Int(N),
        prunedOut: prunedOut
    )
    return Int32(kept)
}
