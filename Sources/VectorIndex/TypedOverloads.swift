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
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
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
