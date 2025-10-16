import Foundation
import VectorCore

// IVF Post-ADC orchestration helpers
// Connects Kernel #22 (ADC scan candidates) to Kernel #40 (Exact rerank)

public enum IVFPostADC {
    /// Rerank exact on IVF-Flat lists after ADC scan produced candidate internal IDs.
    /// - Parameters:
    ///   - q: query vector [d]
    ///   - d: dimension
    ///   - metric: SupportedDistanceMetric (.euclidean/.dotProduct/.cosine)
    ///   - candInternalIDs: candidate internal IDs (dense [0,N)) to rerank
    ///   - id2List: mapping internal ID -> list id
    ///   - id2Offset: mapping internal ID -> offset within list
    ///   - lists: per-list AoS vectors (each list is [len*d] floats)
    ///   - K: number of results
    ///   - opts: rerank options (tiling/parallel/norm caches)
    /// - Returns: best-first (scores, internal IDs)
    public static func rerankTopKFlat(
        q: [Float],
        d: Int,
        metric: SupportedDistanceMetric,
        candInternalIDs: [Int64],
        id2List: [Int32],
        id2Offset: [Int32],
        lists: [[Float]],
        K: Int,
        opts: IndexOps.Rerank.RerankOpts
    ) -> (scores: [Float], ids: [Int64]) {
        return IndexOps.Rerank.topKIVF(
            q: q, d: d, metric: metric,
            candInternalIDs: candInternalIDs,
            id2List: id2List, id2Offset: id2Offset,
            lists: lists, K: K, opts: opts
        )
    }
}

