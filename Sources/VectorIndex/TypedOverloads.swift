import Foundation
import VectorCore

// MARK: - VectorProtocol overloads (existing)
// Convenience typed overloads that accept VectorCore vectors directly.

public extension FlatIndex {
    func insert<V: VectorProtocol>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws where V.Scalar == Float {
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
    }

    func batchInsert<V: VectorProtocol>(_ items: [(id: VectorID, vector: V, metadata: [String: String]?)]) async throws where V.Scalar == Float {
        let converted = items.map { ($0.id, $0.vector.toArray(), $0.metadata) }
        try await batchInsert(converted)
    }

    func search<V: VectorProtocol>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [StringSearchResult] where V.Scalar == Float {
        try await search(query: query.toArray(), k: k, filter: filter)
    }

    func batchSearch<V: VectorProtocol>(queries: [V], k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [[StringSearchResult]] where V.Scalar == Float {
        let q = queries.map { $0.toArray() }
        return try await batchSearch(queries: q, k: k, filter: filter)
    }
}

public extension HNSWIndex {
    func insert<V: VectorProtocol>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws where V.Scalar == Float {
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
    }

    func batchInsert<V: VectorProtocol>(_ items: [(id: VectorID, vector: V, metadata: [String: String]?)]) async throws where V.Scalar == Float {
        let converted = items.map { ($0.id, $0.vector.toArray(), $0.metadata) }
        try await batchInsert(converted)
    }

    func search<V: VectorProtocol>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [StringSearchResult] where V.Scalar == Float {
        try await search(query: query.toArray(), k: k, filter: filter)
    }

    func batchSearch<V: VectorProtocol>(queries: [V], k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [[StringSearchResult]] where V.Scalar == Float {
        let q = queries.map { $0.toArray() }
        return try await batchSearch(queries: q, k: k, filter: filter)
    }
}

public extension IVFIndex {
    func insert<V: VectorProtocol>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws where V.Scalar == Float {
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
    }

    func batchInsert<V: VectorProtocol>(_ items: [(id: VectorID, vector: V, metadata: [String: String]?)]) async throws where V.Scalar == Float {
        let converted = items.map { ($0.id, $0.vector.toArray(), $0.metadata) }
        try await batchInsert(converted)
    }

    func search<V: VectorProtocol>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [StringSearchResult] where V.Scalar == Float {
        try await search(query: query.toArray(), k: k, filter: filter)
    }

    func batchSearch<V: VectorProtocol>(queries: [V], k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [[StringSearchResult]] where V.Scalar == Float {
        let q = queries.map { $0.toArray() }
        return try await batchSearch(queries: q, k: k, filter: filter)
    }
}

// MARK: - IndexableVector overloads
// Accept VectorCore 0.2.0 IndexableVector types and propagate optimization hints
// (isNormalized, cachedMagnitude) for cosine fast paths.

public extension FlatIndex {
    func insert<V: IndexableVector>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws {
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
    }

    func search<V: IndexableVector>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [StringSearchResult] {
        try await search(query: query.toArray(), k: k, filter: filter)
    }
}

public extension HNSWIndex {
    func insert<V: IndexableVector>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws {
        // For cosine, propagate stored-side hints so the invNormsCache is
        // populated incrementally instead of marked dirty on every insert.
        let knownInv: Float? = {
            guard metric == .cosine else { return nil }
            if vector.isNormalized { return 1.0 }
            if let m = vector.cachedMagnitude, m.isFinite, m > 1e-12 {
                return 1.0 / m
            }
            return nil
        }()
        try await insert(id: id, vector: vector.toArray(), metadata: metadata, knownInvNorm: knownInv)
    }

    func batchInsert<V: IndexableVector>(_ items: [(id: VectorID, vector: V, metadata: [String: String]?)]) async throws {
        // Convert each item once: extract the cosine hint up-front so the
        // single-frame WAL path can capture it (and so the in-memory cache
        // is populated incrementally during the batch).
        let hinted: [(VectorID, [Float], [String: String]?, Float?)] = items.map { item in
            let knownInv: Float?
            if metric == .cosine {
                if item.vector.isNormalized {
                    knownInv = 1.0
                } else if let m = item.vector.cachedMagnitude, m.isFinite, m > 1e-12 {
                    knownInv = 1.0 / m
                } else {
                    knownInv = nil
                }
            } else {
                knownInv = nil
            }
            return (item.id, item.vector.toArray(), item.metadata, knownInv)
        }
        try await batchInsertWithHints(hinted)
    }

    func search<V: IndexableVector>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [StringSearchResult] {
        // When query is known-normalized for cosine, skip norm computation in traversal kernel
        let qInv: Float? = (query.isNormalized && metric == .cosine) ? 1.0 : nil
        return try await search(query: query.toArray(), k: k, filter: filter, qInvNorm: qInv)
    }
}

public extension IVFIndex {
    func insert<V: IndexableVector>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws {
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
    }

    func search<V: IndexableVector>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [StringSearchResult] {
        try await search(query: query.toArray(), k: k, filter: filter)
    }
}
