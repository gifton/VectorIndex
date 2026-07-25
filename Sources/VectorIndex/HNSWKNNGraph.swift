//
//  HNSWKNNGraph.swift
//  VectorIndex
//
//  kNN-graph producer for the VectorCore.KNNGraph CSR interchange (gap report §3.2).
//  Emits the raw directed graph with Euclidean distances; VectorCore's UMAP stage
//  performs its own t-conorm symmetrization downstream.
//

import Foundation
import VectorCore

extension HNSWIndex {

    /// One TaskGroup chunk of CSR rows.
    private struct KNNChunk: Sendable {
        var degrees: [Int32] = []
        var neighbors: [Int32] = []
        var distances: [Float] = []
    }

    /// Builds the raw directed k-nearest-neighbor graph over all live points.
    ///
    /// Each live point queries the index (approximate neighbors — standard for
    /// UMAP at scale); self-matches are stripped; per-row degree may be < k.
    /// Distances are Euclidean:
    /// - `.euclidean`: √(L2²) straight from the traversal kernel.
    /// - `.cosine`: chord distance √(2·(1−cos θ)) between unit-normalized directions.
    /// - `.dotProduct`: throws — inner product has no metric interpretation.
    ///
    /// Row `i` corresponds to `ids[i]` (live points in insertion order).
    /// Snapshot semantics: mutations during the build are not observed.
    /// Deterministic: a fixed index + fixed (k, ef) yields an identical graph,
    /// independent of TaskGroup scheduling.
    ///
    /// - Parameters:
    ///   - k: neighbors per row (UMAP default 15; useful range 10–50).
    ///   - ef: traversal beam width; default `max(config.efSearch, 2*k)`,
    ///     floored at `k + 1` to leave room for the self-match.
    public func buildKNNGraph(k: Int = 15, ef: Int? = nil) async throws -> (graph: KNNGraph, ids: [VectorID]) {
        guard metric != .dotProduct else {
            throw ErrorBuilder(.invalidParameter, operation: "hnsw_build_knn_graph")
                .message("dotProduct has no Euclidean distance interpretation; use a euclidean or cosine index")
                .build()
        }
        let live = count
        guard live >= 2 else {
            throw ErrorBuilder(.emptyInput, operation: "hnsw_build_knn_graph")
                .message("kNN graph requires at least 2 live points")
                .info("live_count", "\(live)")
                .build()
        }
        guard k >= 1, k <= live - 1 else {
            throw ErrorBuilder(.invalidRange, operation: "hnsw_build_knn_graph")
                .message("k must be in 1...(liveCount - 1)")
                .info("k", "\(k)")
                .info("live_count", "\(live)")
                .build()
        }
        let efEff = max(ef ?? max(config.efSearch, 2 * k), k + 1)
        guard let (ctx, ids) = makeKNNBuildContext(k: k, ef: efEff) else {
            throw ErrorBuilder(.internalInconsistency, operation: "hnsw_build_knn_graph")
                .message("index reports live points but has no entry point")
                .build()
        }

        let nLive = ctx.rowToNode.count
        let chunkSize = 1024
        let chunkCount = (nLive + chunkSize - 1) / chunkSize
        var chunks = [KNNChunk?](repeating: nil, count: chunkCount)
        try await withThrowingTaskGroup(of: (Int, KNNChunk).self) { group in
            for c in 0..<chunkCount {
                let lo = c * chunkSize
                let hi = min(nLive, lo + chunkSize)
                group.addTask {
                    (c, try Self.buildKNNRows(rows: lo..<hi, ctx: ctx))
                }
            }
            for try await (c, out) in group { chunks[c] = out }
        }

        // Stitch in fixed chunk order — output independent of task scheduling.
        var rowOffsets = [Int](repeating: 0, count: nLive + 1)
        var neighborIndices = [Int32]()
        neighborIndices.reserveCapacity(nLive * k)
        var distances = [Float]()
        distances.reserveCapacity(nLive * k)
        var row = 0
        for c in 0..<chunkCount {
            guard let chunk = chunks[c] else {
                throw ErrorBuilder(.internalInconsistency, operation: "hnsw_build_knn_graph")
                    .message("missing chunk result")
                    .info("chunk", "\(c)")
                    .build()
            }
            for deg in chunk.degrees {
                rowOffsets[row + 1] = rowOffsets[row] + Int(deg)
                row += 1
            }
            neighborIndices.append(contentsOf: chunk.neighbors)
            distances.append(contentsOf: chunk.distances)
        }
        // Core's throwing init is the single contract gatekeeper.
        let graph = try KNNGraph(
            pointCount: nLive,
            rowOffsets: rowOffsets,
            neighborIndices: neighborIndices,
            distances: distances
        )
        return (graph, ids)
    }

    /// TaskGroup worker: kNN rows for `rows`, querying with each stored vector.
    private static func buildKNNRows(rows: Range<Int>, ctx: KNNBuildContext) throws -> KNNChunk {
        try Task.checkCancellation()
        var out = KNNChunk()
        out.degrees.reserveCapacity(rows.count)
        out.neighbors.reserveCapacity(rows.count * ctx.k)
        out.distances.reserveCapacity(rows.count * ctx.k)
        var idsOut = [Int32](repeating: -1, count: ctx.ef)
        var distsOut = [Float](repeating: .infinity, count: ctx.ef)

        ctx.vectorStorage.withUnsafeBufferPointer { xbbp in
            ctx.allowBits.withUnsafeBufferPointer { allowBP in
              withExtendedLifetime(ctx.csrOffsets) { withExtendedLifetime(ctx.csrNeighbors) {
                // Pin per-layer CSR pointers (same pattern as performSingleSearch).
                var offPtrs = [UnsafePointer<Int32>?]()
                var nbrPtrs = [UnsafePointer<Int32>?]()
                offPtrs.reserveCapacity(ctx.csrOffsets.count)
                nbrPtrs.reserveCapacity(ctx.csrNeighbors.count)
                for arr in ctx.csrOffsets {
                    arr.withUnsafeBufferPointer { offPtrs.append($0.baseAddress) }
                }
                for arr in ctx.csrNeighbors {
                    arr.withUnsafeBufferPointer { nbrPtrs.append($0.baseAddress) }
                }
                offPtrs.withUnsafeBufferPointer { offArr in
                    nbrPtrs.withUnsafeBufferPointer { nbrArr in
                        Self.withOptionalFloats(ctx.invNorms) { invNormsPtr in
                            let xb = xbbp.baseAddress!
                            for row in rows {
                                let nodeIdx = Int(ctx.rowToNode[row])
                                let q = xb + nodeIdx * ctx.dim
                                let qInv: Float? = ctx.invNorms.map { $0[nodeIdx] }
                                let written = HNSWTraversal.traverse(
                                    q: q, d: ctx.dim,
                                    entryPoint: Int32(ctx.entryPoint), maxLevel: Int32(ctx.maxLevel),
                                    offsetsPerLayer: offArr.baseAddress!,
                                    neighborsPerLayer: nbrArr.baseAddress!,
                                    xb: xb, N: ctx.N, ef: ctx.ef, metric: ctx.metric,
                                    allowBits: allowBP.baseAddress!, allowN: ctx.N,
                                    invNorms: invNormsPtr, qInvNorm: qInv,
                                    idsOut: &idsOut, distsOut: &distsOut
                                )
                                var deg: Int32 = 0
                                if written > 0 {
                                    for i in 0..<written {
                                        if deg == Int32(ctx.k) { break }
                                        let v = Int(idsOut[i])
                                        if v < 0 || v >= ctx.N || v == nodeIdx { continue }
                                        let r = ctx.nodeToRow[v]
                                        if r < 0 { continue } // defensive; allowBits already filters tombstones
                                        let raw = distsOut[i]
                                        // .L2 kernel distance is squared L2; .COSINE is 1 - cos(theta).
                                        // max(0,·) guards FP cancellation; both forms are finite and >= 0.
                                        let dEuc: Float = (ctx.metric == .L2)
                                            ? max(0, raw).squareRoot()
                                            : (2 * max(0, raw)).squareRoot()
                                        out.neighbors.append(r)
                                        out.distances.append(dEuc)
                                        deg &+= 1
                                    }
                                }
                                out.degrees.append(deg)
                            }
                        }
                    }
                }
              } }
            }
        }
        return out
    }

    private static func withOptionalFloats<R>(_ array: [Float]?, _ body: (UnsafePointer<Float>?) -> R) -> R {
        if let array { return array.withUnsafeBufferPointer { body($0.baseAddress) } }
        return body(nil)
    }
}
