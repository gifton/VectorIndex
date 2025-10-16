import Foundation
import VectorCore
import Dispatch

// Kernel #40: Exact Re-rank on Top-C (original vectors)
//
// Integration notes:
// - Reuses existing kernels: #04 ScoreBlock, #05 TopK selection
// - Provides backend-agnostic vector readers (DenseArray, IVFListVecs, Callback)
// - Supports metrics: L2 (minimize), IP (maximize), Cosine (maximize)
// - Honors options from kernel-spec (#40) including gather tiling and locality reorder
// - Emits telemetry counters/timers (#46) when compiled with VINDEX_TELEM

public extension IndexOps {
    enum Rerank {}
}

public extension IndexOps.Rerank {

    // MARK: - Options
    struct RerankOpts: Sendable {
        public enum Backend: Sendable { case denseArray, ivfListVecs, callback }

        public var backend: Backend
        public var gatherTile: Int                 // rows staged per iteration (32–256 typical)
        public var reorderBySegment: Bool          // locality grouping
        public var haveInvNorms: Bool              // cosine: base inv-norms [N]
        public var haveSqNorms: Bool               // l2: base ||x||^2 [N]
        public var returnSorted: Bool              // outputs sorted best-first
        public var skipMissing: Bool               // ignore tombstoned/missing rows
        public var prefetchDistance: Int           // advisory
        public var strictFP: Bool                  // disable algebraic transforms when true
        public var enableParallel: Bool            // enable tile-parallel scoring
        public var parallelThreshold: Int          // C threshold to switch to parallel
        public var maxConcurrency: Int             // 0 = auto

        public init(
            backend: Backend,
            gatherTile: Int = 128,
            reorderBySegment: Bool = true,
            haveInvNorms: Bool = false,
            haveSqNorms: Bool = false,
            returnSorted: Bool = true,
            skipMissing: Bool = true,
            prefetchDistance: Int = 8,
            strictFP: Bool = false,
            enableParallel: Bool = true,
            parallelThreshold: Int = 8192,
            maxConcurrency: Int = 0
        ) {
            self.backend = backend
            self.gatherTile = max(1, gatherTile)
            self.reorderBySegment = reorderBySegment
            self.haveInvNorms = haveInvNorms
            self.haveSqNorms = haveSqNorms
            self.returnSorted = returnSorted
            self.skipMissing = skipMissing
            self.prefetchDistance = prefetchDistance
            self.strictFP = strictFP
            self.enableParallel = enableParallel
            self.parallelThreshold = max(0, parallelThreshold)
            self.maxConcurrency = maxConcurrency
        }
    }

    // MARK: - Metric helpers (bridge to project metric)
    @inline(__always)
    private static func _minimizes(_ metric: SupportedDistanceMetric) -> Bool {
        switch metric {
        case .euclidean, .manhattan, .chebyshev: return true
        case .dotProduct, .cosine: return false
        }
    }

    @inline(__always)
    private static func _missingSentinel(_ metric: SupportedDistanceMetric) -> Float {
        _minimizes(metric) ? .infinity : -.infinity
    }

    // MARK: - VectorReader Abstraction (Backends)
    protocol VectorReader {
        var dim: Int { get }                   // row dimensionality
        var countHint: Int { get }             // total rows if known; -1 if unknown
        var invNorms: UnsafePointer<Float>? { get }  // optional 1/||x|| (cosine)
        var sqNorms: UnsafePointer<Float>? { get }   // optional ||x||^2 (L2 dot-trick)

        /// A numeric key that groups IDs by physical locality (stable within group).
        func segmentKey(for id: Int64) -> Int

        /// Gather `count` rows for `ids` into `dst` [count * d].
        /// Returns present count and fills `present` flags (1 present, 0 missing) per requested id in order.
        func fetchMany(ids: UnsafePointer<Int64>,
                       count: Int,
                       dst: UnsafeMutablePointer<Float>,
                       present: UnsafeMutablePointer<UInt8>) -> Int
    }

    // DenseArray backend: contiguous matrix xb[N][d] with id == row
    struct DenseArrayReader: VectorReader {
        public let xb: UnsafePointer<Float>       // base pointer [N * d]
        public let N: Int
        public let dim: Int
        public let invNorms: UnsafePointer<Float>?
        public let sqNorms: UnsafePointer<Float>?
        private let segmentShift: Int

        public var countHint: Int { N }

        public init(xb: UnsafePointer<Float>,
                    N: Int,
                    dim: Int,
                    invNorms: UnsafePointer<Float>? = nil,
                    sqNorms: UnsafePointer<Float>? = nil,
                    segmentShift: Int = 8) { // group by 2^segmentShift rows
            self.xb = xb
            self.N = N
            self.dim = dim
            self.invNorms = invNorms
            self.sqNorms = sqNorms
            self.segmentShift = max(0, segmentShift)
        }

        @inline(__always)
        public func segmentKey(for id: Int64) -> Int {
            let row = Int(id)
            return row >> segmentShift
        }

        public func fetchMany(ids: UnsafePointer<Int64>,
                              count: Int,
                              dst: UnsafeMutablePointer<Float>,
                              present: UnsafeMutablePointer<UInt8>) -> Int {
            let d = dim
            var written = 0
            for i in 0..<count {
                let row = Int(ids[i])
                if row >= 0 && row < N {
                    let src = xb.advanced(by: row * d)
                    let out = dst.advanced(by: i * d)
                    out.update(from: src, count: d)
                    present[i] = 1
                    written += 1
                } else {
                    present[i] = 0
                }
            }
            return written
        }
    }

    // IVFListVecs backend: per-list arrays with id -> (list, offset)
    struct IVFListVecsReader: VectorReader {
        public struct List { public let base: UnsafePointer<Float>; public let len: Int }
        public let lists: [List]
        public let id2List: UnsafePointer<Int32>  // [N]
        public let id2Offset: UnsafePointer<Int32>// [N]
        public let N: Int
        public let dim: Int
        public let invNorms: UnsafePointer<Float>?
        public let sqNorms: UnsafePointer<Float>?
        public var countHint: Int { N }
        public init(lists: [List], id2List: UnsafePointer<Int32>, id2Offset: UnsafePointer<Int32>, N: Int, dim: Int, invNorms: UnsafePointer<Float>? = nil, sqNorms: UnsafePointer<Float>? = nil) {
            self.lists = lists
            self.id2List = id2List
            self.id2Offset = id2Offset
            self.N = N
            self.dim = dim
            self.invNorms = invNorms
            self.sqNorms = sqNorms
        }
        @inline(__always) public func segmentKey(for id: Int64) -> Int {
            let row = Int(id)
            if row < 0 || row >= N { return Int.max }
            return Int(id2List[row])
        }
        public func fetchMany(ids: UnsafePointer<Int64>, count: Int, dst: UnsafeMutablePointer<Float>, present: UnsafeMutablePointer<UInt8>) -> Int {
            let d = dim
            var written = 0
            for i in 0..<count {
                let row = Int(ids[i])
                if row >= 0 && row < N {
                    let li = Int(id2List[row])
                    let off = Int(id2Offset[row])
                    let list = lists[li]
                    if off >= 0 && off < list.len {
                        let src = list.base.advanced(by: off * d)
                        let out = dst.advanced(by: i * d)
                        out.update(from: src, count: d)
                        present[i] = 1
                        written += 1
                        continue
                    }
                }
                present[i] = 0
            }
            return written
        }
    }

    // Callback backend: user-provided gather; must preserve order of IDs
    struct CallbackReader: VectorReader {
        public let dim: Int
        public var countHint: Int { -1 }
        public let invNorms: UnsafePointer<Float>?
        public let sqNorms: UnsafePointer<Float>?
        public typealias FetchFn = (_ ids: UnsafePointer<Int64>, _ count: Int, _ dst: UnsafeMutablePointer<Float>, _ present: UnsafeMutablePointer<UInt8>) -> Int
        private let fetchFn: FetchFn
        public init(dim: Int, invNorms: UnsafePointer<Float>? = nil, sqNorms: UnsafePointer<Float>? = nil, fetch: @escaping FetchFn) {
            self.dim = dim; self.invNorms = invNorms; self.sqNorms = sqNorms; self.fetchFn = fetch
        }
        @inline(__always) public func segmentKey(for id: Int64) -> Int { 0 }
        public func fetchMany(ids: UnsafePointer<Int64>, count: Int, dst: UnsafeMutablePointer<Float>, present: UnsafeMutablePointer<UInt8>) -> Int { fetchFn(ids, count, dst, present) }
    }

    // MARK: - Locality reorder helper
    private struct IndexedID { let idx: Int; let id: Int64; let seg: Int }
    @inline(__always)
    private static func buildReorder(_ ids: UnsafePointer<Int64>, _ C: Int, reader: any VectorReader, enabled: Bool) -> [IndexedID] {
        var v = [IndexedID](); v.reserveCapacity(C)
        for i in 0..<C { let id = ids[i]; let seg = enabled ? reader.segmentKey(for: id) : 0; v.append(IndexedID(idx: i, id: id, seg: seg)) }
        if enabled {
            v.sort { (a, b) -> Bool in
                if a.seg != b.seg { return a.seg < b.seg }
                if a.id  != b.id  { return a.id  < b.id }
                return a.idx < b.idx
            }
        }
        return v
    }

    // MARK: - Low-level helpers
    @inline(__always) private static func norm2(_ x: UnsafePointer<Float>, _ d: Int) -> Float { var s: Float = 0; for i in 0..<d { s += x[i] * x[i] }; return s }

    // MARK: - Core scoring (single query, scores aligned with cand_ids)
    @inline(__always)
    @preconcurrency
    private static func scoreBlock(
        q: UnsafePointer<Float>,
        d: Int,
        metric: SupportedDistanceMetric,
        ids: UnsafePointer<Int64>,
        C: Int,
        reader: any VectorReader,
        opts: RerankOpts,
        scoresOut: UnsafeMutablePointer<Float>,      // [C], aligned with cand_ids order
        presentMaskOut: UnsafeMutablePointer<UInt8>? // optional [C], 1 if present
    ) {
        let tile = max(1, opts.gatherTile)
        let order = buildReorder(ids, C, reader: reader, enabled: opts.reorderBySegment)

        // Precompute query’s scalar norm for L2 dot-trick if helpful
        let qNorm: Float = (metric == .euclidean && opts.haveSqNorms && !opts.strictFP) ? norm2(q, d) : 0

        // Decide single-thread vs parallel
        let parallel = opts.enableParallel && C >= max(opts.parallelThreshold, tile * 2)
        if !parallel {
            // Single-thread path
            let scratch = UnsafeMutablePointer<Float>.allocate(capacity: tile * d)
            defer { scratch.deallocate() }
            let present = UnsafeMutablePointer<UInt8>.allocate(capacity: tile)
            defer { present.deallocate() }
            let tileScores = UnsafeMutablePointer<Float>.allocate(capacity: tile)
            defer { tileScores.deallocate() }

            var i = 0
            while i < C {
                let chunk = min(tile, C - i)
                var tmpIDs = [Int64](repeating: -1, count: chunk)
                var origIdx = [Int](repeating: -1, count: chunk)
                for j in 0..<chunk { tmpIDs[j] = order[i + j].id; origIdx[j] = order[i + j].idx }

                // Prefetch lookahead addresses (best-effort) based on backend
                if opts.prefetchDistance > 0 {
                    prefetchLookahead(order: order, base: i, count: chunk, d: d, reader: reader, distance: opts.prefetchDistance)
                }

                tmpIDs.withUnsafeBufferPointer { idPtr in
                    _ = reader.fetchMany(ids: idPtr.baseAddress!, count: chunk, dst: scratch, present: present)
                }

                scoreTile(q: q, d: d, metric: metric, n: chunk, qNorm: qNorm, xb: scratch, out: tileScores, reader: reader, opts: opts)

                for j in 0..<chunk {
                    let dstIdx = origIdx[j]
                    if present[j] == 0 {
                        scoresOut[dstIdx] = _missingSentinel(metric)
                        presentMaskOut?.advanced(by: dstIdx).pointee = 0
                    } else {
                        scoresOut[dstIdx] = tileScores[j]
                        presentMaskOut?.advanced(by: dstIdx).pointee = 1
                    }
                }

                i += chunk
            }
            return
        }

        // Parallel path: partition by tiles
        let tiles = (C + tile - 1) / tile
        let workers = opts.maxConcurrency > 0 ? opts.maxConcurrency : max(1, min(tiles, ProcessInfo.processInfo.activeProcessorCount))
        DispatchQueue.concurrentPerform(iterations: workers) { w in
            var t = w
            while t < tiles {
                let start = t * tile
                if start >= C { break }
                let chunk = min(tile, C - start)

                // Allocate per-task scratch
                let scratch = UnsafeMutablePointer<Float>.allocate(capacity: chunk * d)
                let present = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
                let tileScores = UnsafeMutablePointer<Float>.allocate(capacity: chunk)
                defer { scratch.deallocate(); present.deallocate(); tileScores.deallocate() }

                var tmpIDs = [Int64](repeating: -1, count: chunk)
                var origIdx = [Int](repeating: -1, count: chunk)
                for j in 0..<chunk { tmpIDs[j] = order[start + j].id; origIdx[j] = order[start + j].idx }

                if opts.prefetchDistance > 0 {
                    prefetchLookahead(order: order, base: start, count: chunk, d: d, reader: reader, distance: opts.prefetchDistance)
                }

                tmpIDs.withUnsafeBufferPointer { idPtr in
                    _ = reader.fetchMany(ids: idPtr.baseAddress!, count: chunk, dst: scratch, present: present)
                }

                scoreTile(q: q, d: d, metric: metric, n: chunk, qNorm: qNorm, xb: scratch, out: tileScores, reader: reader, opts: opts)

                // Scatter back to global outputs (disjoint indices across tiles)
                for j in 0..<chunk {
                    let dstIdx = origIdx[j]
                    if present[j] == 0 {
                        scoresOut[dstIdx] = _missingSentinel(metric)
                        presentMaskOut?.advanced(by: dstIdx).pointee = 0
                    } else {
                        scoresOut[dstIdx] = tileScores[j]
                        presentMaskOut?.advanced(by: dstIdx).pointee = 1
                    }
                }

                t += workers
            }
        }
    }

    // Score a tile with #04 kernels
    @inline(__always)
    private static func scoreTile(q: UnsafePointer<Float>, d: Int, metric: SupportedDistanceMetric, n: Int, qNorm: Float, xb: UnsafePointer<Float>, out: UnsafeMutablePointer<Float>, reader: any VectorReader, opts: RerankOpts) {
        switch metric {
        case .euclidean:
            IndexOps.Scoring.L2Sqr.run(
                q: q,
                xb: xb,
                n: n,
                d: d,
                out: out,
                xb_norm: (opts.haveSqNorms && !opts.strictFP) ? reader.sqNorms : nil,
                q_norm: (opts.haveSqNorms && !opts.strictFP) ? qNorm : nil
            )
        case .dotProduct:
            IndexOps.Scoring.InnerProduct.run(q: q, xb: xb, n: n, d: d, out: out)
        case .cosine:
            let handle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle? = {
                guard opts.haveInvNorms, let inv = reader.invNorms else { return nil }
                return .init(dbInvNormsF32: inv, dbInvNormsF16: nil, queryInvNorm: nil)
            }()
            IndexOps.Scoring.ScoreBlock.run(q: q, xb: xb, n: n, d: d, metric: .cosine, out: out, cosineNorms: handle)
        default:
            fatalError("Unsupported metric for rerank")
        }
    }

    // Best-effort prefetch for lookahead rows
    @inline(__always)
    private static func prefetchLookahead(order: [IndexedID], base: Int, count: Int, d: Int, reader: any VectorReader, distance: Int) {
        guard distance > 0 else { return }
        // Only support known backends for exact row address; else skip
        if let dr = reader as? DenseArrayReader {
            let N = dr.N
            let xb = dr.xb
            for j in 0..<count {
                let look = j + distance
                if look < count {
                    let row = Int(order[base + look].id)
                    if row >= 0 && row < N {
                        let p = xb.advanced(by: row * d)
                        vi_prefetch_read(p)
                    }
                }
            }
            return
        }
        if let ir = reader as? IVFListVecsReader {
            let N = ir.N
            for j in 0..<count {
                let look = j + distance
                if look < count {
                    let row = Int(order[base + look].id)
                    if row >= 0 && row < N {
                        let li = Int(ir.id2List[row])
                        let off = Int(ir.id2Offset[row])
                        if li >= 0 && li < ir.lists.count {
                            let lst = ir.lists[li]
                            if off >= 0 && off < lst.len {
                                let p = lst.base.advanced(by: off * d)
                                vi_prefetch_read(p)
                            }
                        }
                    }
                }
            }
            return
        }
        // CallbackReader: cannot infer address → no-op
    }

    // MARK: - Convenience wrappers (Swift arrays → VectorReader inference)
    static func topKDense(
        q: [Float],
        d: Int,
        metric: SupportedDistanceMetric,
        candIDs: [Int64],
        xb: [Float],
        invNorms: [Float]? = nil,
        sqNorms: [Float]? = nil,
        K: Int,
        opts: RerankOpts
    ) -> (scores: [Float], ids: [Int64]) {
        precondition(d > 0)
        precondition(q.count == d)
        let N = xb.count / d
        var outScores = [Float](repeating: 0, count: K)
        var outIDs = [Int64](repeating: -1, count: K)
        q.withUnsafeBufferPointer { qbp in
            xb.withUnsafeBufferPointer { xbbp in
                candIDs.withUnsafeBufferPointer { idbp in
                    var localOpts = opts
                    localOpts.backend = .denseArray
                    let invPtr: UnsafePointer<Float>? = invNorms?.withUnsafeBufferPointer { $0.baseAddress }
                    let sqPtr: UnsafePointer<Float>? = sqNorms?.withUnsafeBufferPointer { $0.baseAddress }
                    let reader = DenseArrayReader(xb: xbbp.baseAddress!, N: N, dim: d, invNorms: invPtr, sqNorms: sqPtr)
                    rerank_exact_topk(q: qbp.baseAddress!, d: d, metric: metric, candIDs: idbp.baseAddress!, C: candIDs.count, K: K, reader: reader, opts: localOpts, topScores: &outScores, topIDs: &outIDs)
                }
            }
        }
        return (outScores, outIDs)
    }

    static func scoresDense(
        q: [Float],
        d: Int,
        metric: SupportedDistanceMetric,
        candIDs: [Int64],
        xb: [Float],
        invNorms: [Float]? = nil,
        sqNorms: [Float]? = nil,
        opts: RerankOpts
    ) -> [Float] {
        precondition(d > 0)
        precondition(q.count == d)
        let N = xb.count / d
        var out = [Float](repeating: _missingSentinel(metric), count: candIDs.count)
        q.withUnsafeBufferPointer { qbp in
            xb.withUnsafeBufferPointer { xbbp in
                candIDs.withUnsafeBufferPointer { idbp in
                    var localOpts = opts
                    localOpts.backend = .denseArray
                    let invPtr: UnsafePointer<Float>? = invNorms?.withUnsafeBufferPointer { $0.baseAddress }
                    let sqPtr: UnsafePointer<Float>? = sqNorms?.withUnsafeBufferPointer { $0.baseAddress }
                    let reader = DenseArrayReader(xb: xbbp.baseAddress!, N: N, dim: d, invNorms: invPtr, sqNorms: sqPtr)
                    rerank_exact_scores(q: qbp.baseAddress!, d: d, metric: metric, candIDs: idbp.baseAddress!, C: candIDs.count, reader: reader, opts: localOpts, scoresOut: &out)
                }
            }
        }
        return out
    }

    // MARK: - IVF list-based conveniences (IVF-Flat)
    // Pointer-based lists version
    static func topKIVF(
        q: [Float],
        d: Int,
        metric: SupportedDistanceMetric,
        candInternalIDs: [Int64],
        id2List: [Int32],
        id2Offset: [Int32],
        listBases: [UnsafePointer<Float>],
        listLengths: [Int],
        K: Int,
        opts: RerankOpts
    ) -> (scores: [Float], ids: [Int64]) {
        precondition(d > 0)
        precondition(q.count == d)
        precondition(id2List.count == id2Offset.count)
        precondition(listBases.count == listLengths.count)
        var outScores = [Float](repeating: 0, count: K)
        var outIDs = [Int64](repeating: -1, count: K)
        q.withUnsafeBufferPointer { qbp in
            id2List.withUnsafeBufferPointer { lmap in
                id2Offset.withUnsafeBufferPointer { omap in
                    // Build List descriptors
                    var lists: [IVFListVecsReader.List] = []
                    lists.reserveCapacity(listBases.count)
                    for i in 0..<listBases.count { lists.append(.init(base: listBases[i], len: listLengths[i])) }
                    let reader = IVFListVecsReader(lists: lists, id2List: lmap.baseAddress!, id2Offset: omap.baseAddress!, N: id2List.count, dim: d, invNorms: nil, sqNorms: nil)
                    candInternalIDs.withUnsafeBufferPointer { ids in
                        var localOpts = opts
                        localOpts.backend = .ivfListVecs
                        rerank_exact_topk(
                            q: qbp.baseAddress!, d: d, metric: metric,
                            candIDs: ids.baseAddress!, C: ids.count, K: K,
                            reader: reader, opts: localOpts,
                            topScores: &outScores, topIDs: &outIDs
                        )
                    }
                }
            }
        }
        return (outScores, outIDs)
    }

    // Array-based lists version (convenience)
    static func topKIVF(
        q: [Float],
        d: Int,
        metric: SupportedDistanceMetric,
        candInternalIDs: [Int64],
        id2List: [Int32],
        id2Offset: [Int32],
        lists: [[Float]],
        K: Int,
        opts: RerankOpts
    ) -> (scores: [Float], ids: [Int64]) {
        precondition(d > 0)
        precondition(q.count == d)
        precondition(id2List.count == id2Offset.count)
        var outScores = [Float](repeating: 0, count: K)
        var outIDs = [Int64](repeating: -1, count: K)
        q.withUnsafeBufferPointer { qbp in
            id2List.withUnsafeBufferPointer { lmap in
                id2Offset.withUnsafeBufferPointer { omap in
                    // Build List descriptors with temporary base pointers
                    var listDescs: [IVFListVecsReader.List] = []
                    listDescs.reserveCapacity(lists.count)
                    // Use nested withUnsafeBufferPointer to ensure lifetimes cover call
                    func buildAndRun(_ idx: Int, _ acc: inout [IVFListVecsReader.List]) {
                        if idx == lists.count {
                            candInternalIDs.withUnsafeBufferPointer { ids in
                                var localOpts = opts
                                localOpts.backend = .ivfListVecs
                                let reader = IVFListVecsReader(lists: acc, id2List: lmap.baseAddress!, id2Offset: omap.baseAddress!, N: id2List.count, dim: d, invNorms: nil, sqNorms: nil)
                                rerank_exact_topk(
                                    q: qbp.baseAddress!, d: d, metric: metric,
                                    candIDs: ids.baseAddress!, C: ids.count, K: K,
                                    reader: reader, opts: localOpts,
                                    topScores: &outScores, topIDs: &outIDs
                                )
                            }
                            return
                        }
                        lists[idx].withUnsafeBufferPointer { lbp in
                            precondition(lbp.count % d == 0, "list vector count must be multiple of d")
                            acc.append(.init(base: lbp.baseAddress!, len: lbp.count / d))
                            buildAndRun(idx + 1, &acc)
                            _ = acc.popLast()
                        }
                    }
                    buildAndRun(0, &listDescs)
                }
            }
        }
        return (outScores, outIDs)
    }

    static func scoresIVF(
        q: [Float],
        d: Int,
        metric: SupportedDistanceMetric,
        candInternalIDs: [Int64],
        id2List: [Int32],
        id2Offset: [Int32],
        lists: [[Float]],
        opts: RerankOpts
    ) -> [Float] {
        precondition(d > 0)
        precondition(q.count == d)
        precondition(id2List.count == id2Offset.count)
        var out = [Float](repeating: _missingSentinel(metric), count: candInternalIDs.count)
        q.withUnsafeBufferPointer { qbp in
            id2List.withUnsafeBufferPointer { lmap in
                id2Offset.withUnsafeBufferPointer { omap in
                    var listDescs: [IVFListVecsReader.List] = []
                    listDescs.reserveCapacity(lists.count)
                    func buildAndRun(_ idx: Int, _ acc: inout [IVFListVecsReader.List]) {
                        if idx == lists.count {
                            candInternalIDs.withUnsafeBufferPointer { ids in
                                var localOpts = opts
                                localOpts.backend = .ivfListVecs
                                let reader = IVFListVecsReader(lists: acc, id2List: lmap.baseAddress!, id2Offset: omap.baseAddress!, N: id2List.count, dim: d, invNorms: nil, sqNorms: nil)
                                rerank_exact_scores(
                                    q: qbp.baseAddress!, d: d, metric: metric,
                                    candIDs: ids.baseAddress!, C: ids.count,
                                    reader: reader, opts: localOpts,
                                    scoresOut: &out
                                )
                            }
                            return
                        }
                        lists[idx].withUnsafeBufferPointer { lbp in
                            precondition(lbp.count % d == 0, "list vector count must be multiple of d")
                            acc.append(.init(base: lbp.baseAddress!, len: lbp.count / d))
                            buildAndRun(idx + 1, &acc)
                            _ = acc.popLast()
                        }
                    }
                    buildAndRun(0, &listDescs)
                }
            }
        }
        return out
    }

    // MARK: - Public API (single query)
    /// Scores-only (aligned with cand_ids)
    static func rerank_exact_scores(
        q: UnsafePointer<Float>, d: Int, metric: SupportedDistanceMetric,
        candIDs: UnsafePointer<Int64>, C: Int,
        reader: any VectorReader, opts: RerankOpts,
        scoresOut: UnsafeMutablePointer<Float>
    ) {
        #if VINDEX_TELEM
        let _ = TELEM_TIMER_GUARD(.t_rerank)
        TELEM_FLAG(.used_prefetch) // advisory; locality reorder acts as prefetch surrogate
        if metric == .cosine { TELEM_FLAG(.used_cosine) }
        TELEM_INC(.vecs_scored, UInt64(C))
        TELEM_ADD_BYTES(.vecs, UInt64(C * d * MemoryLayout<Float>.stride))
        #endif
        scoreBlock(q: q, d: d, metric: metric, ids: candIDs, C: C, reader: reader, opts: opts, scoresOut: scoresOut, presentMaskOut: nil)
    }

    /// Top-K (single query)
    static func rerank_exact_topk(
        q: UnsafePointer<Float>, d: Int, metric: SupportedDistanceMetric,
        candIDs: UnsafePointer<Int64>, C: Int, K: Int,
        reader: any VectorReader, opts: RerankOpts,
        topScores: UnsafeMutablePointer<Float>, topIDs: UnsafeMutablePointer<Int64>
    ) {
        precondition(K <= C && K >= 0)

        // First compute all scores aligned to candIDs and capture presence
        var scores = [Float](repeating: _missingSentinel(metric), count: C)
        var presentMask = [UInt8](repeating: 0, count: C)
        scores.withUnsafeMutableBufferPointer { sb in
            presentMask.withUnsafeMutableBufferPointer { pb in
                scoreBlock(q: q, d: d, metric: metric, ids: candIDs, C: C, reader: reader, opts: opts, scoresOut: sb.baseAddress!, presentMaskOut: pb.baseAddress!)
            }
        }

        // Optionally elide missing candidates
        let useFiltered = opts.skipMissing
        var filteredScores: [Float] = []
        var filteredIDs32: [Int32] = []
        var ids32All: [Int32] = []
        if useFiltered {
            filteredScores.reserveCapacity(C)
            filteredIDs32.reserveCapacity(C)
            for i in 0..<C where presentMask[i] != 0 {
                filteredScores.append(scores[i])
                filteredIDs32.append(Int32(truncatingIfNeeded: candIDs.advanced(by: i).pointee))
            }
        } else {
            ids32All.reserveCapacity(C)
            for i in 0..<C { ids32All.append(Int32(truncatingIfNeeded: candIDs.advanced(by: i).pointee)) }
        }

        // Select top-K using #05 TopK (deterministic tie-break by id)
        let ordering = IndexOps.Selection.ordering(for: metric)
        let selHeap: IndexOps.Selection.TopKHeap = {
            if useFiltered {
                guard !filteredScores.isEmpty else { return IndexOps.Selection.TopKHeap(capacity: K, ordering: ordering) }
                return IndexOps.Selection.selectTopK_streaming(
                    scores: filteredScores,
                    ids: filteredIDs32,
                    count: filteredScores.count,
                    k: K,
                    ordering: ordering
                )
            } else {
                return IndexOps.Selection.selectTopK_streaming(
                    scores: scores,
                    ids: ids32All,
                    count: C,
                    k: K,
                    ordering: ordering
                )
            }
        }()

        let pairs = selHeap.extractSorted()
        let actual = min(K, pairs.count)
        // Emit results; pad if fewer than K present
        for i in 0..<actual {
            topScores[i] = pairs[i].score
            topIDs[i]    = Int64(pairs[i].id)
        }
        if actual < K {
            let sentinel = _missingSentinel(metric)
            for i in actual..<K { topScores[i] = sentinel; topIDs[i] = -1 }
        }

        #if VINDEX_TELEM
        TELEM_SET64(.candidates_kept, UInt64(actual))
        TELEM_INC(.topk_selected, UInt64(K))
        #endif
    }

    // MARK: - Batched Top-K
    static func rerank_exact_topk_batch(
        Q: UnsafePointer<Float>, b: Int, d: Int, metric: SupportedDistanceMetric,
        candOffsets: UnsafePointer<Int64>,  // [b+1]
        candIDs: UnsafePointer<Int64>,      // [candOffsets[b]]
        K: Int,
        reader: any VectorReader, opts: RerankOpts,
        topScores: UnsafeMutablePointer<Float>, // [b*K]
        topIDs: UnsafeMutablePointer<Int64>     // [b*K]
    ) {
        for qi in 0..<b {
            let qPtr = Q.advanced(by: qi * d)
            let start = Int(candOffsets[qi])
            let end   = Int(candOffsets[qi + 1])
            let C     = end - start
            let ids   = candIDs.advanced(by: start)
            rerank_exact_topk(q: qPtr, d: d, metric: metric, candIDs: ids, C: C, K: K, reader: reader, opts: opts, topScores: topScores.advanced(by: qi * K), topIDs: topIDs.advanced(by: qi * K))
        }
    }
}
