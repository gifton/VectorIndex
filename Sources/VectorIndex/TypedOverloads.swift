import Foundation
import VectorCore

// Convenience typed overloads that accept VectorCore vectors directly.

public extension FlatIndex {
    func insert<V: VectorProtocol>(id: VectorID, vector: V, metadata: [String: String]? = nil) async throws where V.Scalar == Float {
        try await insert(id: id, vector: vector.toArray(), metadata: metadata)
    }

    func batchInsert<V: VectorProtocol>(_ items: [(id: VectorID, vector: V, metadata: [String: String]?)]) async throws where V.Scalar == Float {
        let converted = items.map { ($0.id, $0.vector.toArray(), $0.metadata) }
        try await batchInsert(converted)
    }

    func search<V: VectorProtocol>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [SearchResult] where V.Scalar == Float {
        try await search(query: query.toArray(), k: k, filter: filter)
    }

    func batchSearch<V: VectorProtocol>(queries: [V], k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [[SearchResult]] where V.Scalar == Float {
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

    func search<V: VectorProtocol>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [SearchResult] where V.Scalar == Float {
        try await search(query: query.toArray(), k: k, filter: filter)
    }

    func batchSearch<V: VectorProtocol>(queries: [V], k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [[SearchResult]] where V.Scalar == Float {
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

    func search<V: VectorProtocol>(query: V, k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [SearchResult] where V.Scalar == Float {
        try await search(query: query.toArray(), k: k, filter: filter)
    }

    func batchSearch<V: VectorProtocol>(queries: [V], k: Int, filter: (@Sendable ([String: String]?) -> Bool)? = nil) async throws -> [[SearchResult]] where V.Scalar == Float {
        let q = queries.map { $0.toArray() }
        return try await batchSearch(queries: q, k: k, filter: filter)
    }
}
