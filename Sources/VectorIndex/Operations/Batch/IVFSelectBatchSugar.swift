import Foundation

public extension IndexOps.Batch {
    /// Convenience wrapper around `ivf_select_nprobe_batch_f32` that returns
    /// per-query arrays of IDs and scores instead of flat `[b*nprobe]` buffers.
    ///
    /// - Parameters:
    ///   - Q: Queries [b × d]
    ///   - b: Batch size
    ///   - d: Dimension
    ///   - centroids: Coarse centroids [kc × d]
    ///   - kc: Number of centroids
    ///   - metric: IVF metric
    ///   - nprobe: Lists to probe per query
    ///   - opts: Selection options
    ///   - gatherScores: If true, also returns scores per query
    /// - Returns: Tuple of ([[Int32]], [[Float]]?) sized [b][nprobe]
    static func ivfSelectNprobe(
        Q: [Float],
        b: Int,
        d: Int,
        centroids: [Float],
        kc: Int,
        metric: IVFMetric,
        nprobe: Int,
        opts: IVFSelectOpts = IVFSelectOpts(),
        gatherScores: Bool = true
    ) -> (ids: [[Int32]], scores: [[Float]]?) {
        precondition(Q.count == b * d, "Q must be [b×d]")
        precondition(centroids.count == kc * d, "centroids must be [kc×d]")
        var flatIDs = [Int32](repeating: -1, count: b * nprobe)
        var flatScores: [Float]? = gatherScores ? [Float](repeating: .nan, count: b * nprobe) : nil

        ivf_select_nprobe_batch_f32(
            Q: Q, b: b, d: d,
            centroids: centroids, kc: kc,
            metric: metric, nprobe: nprobe,
            opts: opts,
            listIDsOut: &flatIDs,
            listScoresOut: &flatScores
        )

        // Reshape into [b][nprobe]
        var ids2D: [[Int32]] = Array(repeating: Array(repeating: -1, count: nprobe), count: b)
        var scores2D: [[Float]]? = gatherScores ? Array(repeating: Array(repeating: .nan, count: nprobe), count: b) : nil
        for i in 0..<b {
            let off = i * nprobe
            for j in 0..<nprobe {
                ids2D[i][j] = flatIDs[off + j]
                if gatherScores, let sc = flatScores { scores2D![i][j] = sc[off + j] }
            }
        }
        return (ids2D, scores2D)
    }
}

