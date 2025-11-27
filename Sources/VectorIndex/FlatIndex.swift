//
//  FlatIndex.swift
//  VectorIndex
//
//  A minimal, CPU-only exact index useful as a baseline and for testing.
//  Uses VectorCore distance metrics on CPU.
//

import Foundation
import VectorCore

public actor FlatIndex: VectorIndexProtocol, AccelerableIndex {
    public let dimension: Int
    public let metric: SupportedDistanceMetric
    private var vectors: [VectorID: ([Float], [String: String]?)] = [:]

    public var count: Int { vectors.count }

    public init(dimension: Int, metric: SupportedDistanceMetric = .euclidean) {
        self.dimension = dimension
        self.metric = metric
    }

    public func insert(id: VectorID, vector: [Float], metadata: [String: String]?) async throws {
        guard vector.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.count)
        }
        vectors[id] = (vector, metadata)
    }

    public func remove(id: VectorID) async throws {
        vectors.removeValue(forKey: id)
    }

    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?)]) async throws {
        for item in items {
            guard item.vector.count == dimension else {
                throw VectorError.dimensionMismatch(expected: dimension, actual: item.vector.count)
            }
            vectors[item.id] = (item.vector, item.metadata)
        }
    }

    public func optimize() async throws {
        // No-op for flat index
    }

    public func search(query: [Float], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [SearchResult] {
        guard k > 0 else { return [] }
        var results: [SearchResult] = []
        results.reserveCapacity(min(k, vectors.count))

        for (id, (vec, meta)) in vectors {
            if let filter = filter, !filter(meta) { continue }
            guard vec.count == query.count else {
                throw VectorError.dimensionMismatch(expected: query.count, actual: vec.count)
            }
            let d = distance(query, vec, metric: metric)
            results.append(SearchResult(id: id, score: d))
        }

        // Keep top-k smallest distances
        results.sort { $0.score < $1.score }
        if results.count > k { results.removeLast(results.count - k) }
        return results
    }

    /// Context for parallel batch search
    private struct FlatBatchSearchContext: @unchecked Sendable {
        let vectors: [VectorID: ([Float], [String: String]?)]
        let dimension: Int
        let metric: SupportedDistanceMetric
        let k: Int
    }

    public func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [[SearchResult]] {
        guard k > 0 else { return queries.map { _ in [] } }
        if queries.isEmpty { return [] }

        // Snapshot data for parallel access
        let ctx = FlatBatchSearchContext(
            vectors: vectors,
            dimension: dimension,
            metric: metric,
            k: k
        )

        return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
            for (queryIndex, query) in queries.enumerated() {
                group.addTask {
                    try Self.performFlatSearch(query: query, queryIndex: queryIndex, ctx: ctx, filter: filter)
                }
            }

            var results = [[SearchResult]](repeating: [], count: queries.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results
        }
    }

    /// Static helper for parallel flat search
    private static func performFlatSearch(
        query: [Float],
        queryIndex: Int,
        ctx: FlatBatchSearchContext,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) throws -> (Int, [SearchResult]) {
        var results: [SearchResult] = []
        results.reserveCapacity(min(ctx.k, ctx.vectors.count))

        for (id, (vec, meta)) in ctx.vectors {
            if let filter = filter, !filter(meta) { continue }
            guard vec.count == query.count else {
                throw VectorError.dimensionMismatch(expected: query.count, actual: vec.count)
            }
            let d = distance(query, vec, metric: ctx.metric)
            results.append(SearchResult(id: id, score: d))
        }

        results.sort { $0.score < $1.score }
        if results.count > ctx.k { results.removeLast(results.count - ctx.k) }
        return (queryIndex, results)
    }

    public func clear() async {
        vectors.removeAll(keepingCapacity: false)
    }

    public func statistics() async -> IndexStats {
        IndexStats(
            indexType: "Flat",
            vectorCount: vectors.count,
            dimension: dimension,
            metric: metric,
            details: [:]
        )
    }

    public func save(to url: URL) async throws {
        let recs: [PersistedRecord] = vectors.map { key, value in
            PersistedRecord(id: key, vector: value.0, metadata: value.1)
        }
        let payload = PersistedIndex(
            type: "Flat",
            version: 1,
            dimension: dimension,
            metric: metric.rawValue,
            records: recs
        )
        let data = try JSONEncoder().encode(payload)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) async throws -> FlatIndex {
        let data = try Data(contentsOf: url)
        let payload = try JSONDecoder().decode(PersistedIndex.self, from: data)
        guard payload.type == "Flat" else { throw VectorError(.invalidData) }
        let idx = FlatIndex(dimension: payload.dimension, metric: .from(raw: payload.metric))
        try await idx.batchInsert(payload.records.map { ($0.id, $0.vector, $0.metadata) })
        return idx
    }

    public func compact() async throws {
        // No-op for flat
    }

    public func contains(id: VectorID) async -> Bool {
        vectors[id] != nil
    }

    public func update(id: VectorID, vector: [Float]?, metadata: [String: String]?) async throws -> Bool {
        guard var entry = vectors[id] else { return false }
        if let v = vector {
            guard v.count == dimension else { throw VectorError.dimensionMismatch(expected: dimension, actual: v.count) }
            entry.0 = v
        }
        if let m = metadata { entry.1 = m }
        vectors[id] = entry
        return true
    }

    public func batchRemove(_ ids: [VectorID]) async throws {
        for id in ids { try await remove(id: id) }
    }
}

// MARK: - AccelerableIndex Implementation
extension FlatIndex {
    public func getCandidates(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> AccelerationCandidates {
        guard k > 0 else { 
            return AccelerationCandidates(
                ids: [],
                vectorStorage: ContiguousArray<Float>(),
                vectorCount: 0,
                dimension: dimension,
                metadata: []
            )
        }
        
        // Pre-allocate arrays with exact capacity for better performance
        let candidateCount = vectors.count
        var ids: [VectorID] = []
        ids.reserveCapacity(candidateCount)
        var metadata: [[String: String]?] = []
        metadata.reserveCapacity(candidateCount)
        
        // Pre-allocate contiguous storage for all vectors
        var vectorStorage = ContiguousArray<Float>()
        vectorStorage.reserveCapacity(candidateCount * dimension)
        
        // Single pass through vectors, minimizing allocations
        for (id, (vec, meta)) in vectors {
            ids.append(id)
            metadata.append(meta)
            // Append directly to contiguous storage (avoids intermediate array)
            vectorStorage.append(contentsOf: vec)
        }
        
        return AccelerationCandidates(
            ids: ids,
            vectorStorage: vectorStorage,
            vectorCount: candidateCount,
            dimension: dimension,
            metadata: metadata
        )
    }
    
    public func getBatchCandidates(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [AccelerationCandidates] {
        // For flat index, return the same candidates for all queries
        let candidates = try await getCandidates(query: queries.first ?? [], k: k, filter: filter)
        return Array(repeating: candidates, count: queries.count)
    }
    
    public func getIndexStructure() async -> IndexStructure {
        .flat
    }
    
    public func finalizeResults(
        candidates: AccelerationCandidates,
        results: AcceleratedResults,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async -> [SearchResult] {
        var finalResults: [SearchResult] = []
        
        for (idx, distance) in zip(results.indices, results.distances) {
            guard idx < candidates.ids.count else { continue }
            
            let metadata = candidates.metadata[idx]
            if let filter = filter, !filter(metadata) { continue }
            
            finalResults.append(SearchResult(
                id: candidates.ids[idx],
                score: distance
            ))
        }
        
        return finalResults
    }
}
