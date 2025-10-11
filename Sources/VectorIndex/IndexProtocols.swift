//
//  IndexProtocols.swift
//  VectorIndex
//
//  CPU-first index protocol surfaces. Keep GPU acceleration out of this
//  package; provide optional bridges elsewhere.
//

import Foundation
import VectorCore

// MARK: - VectorCore Dependency Policy
//
// We import VectorCore for type compatibility:
// - VectorID (String typealias) - shared identifier type from VectorCore
// - SupportedDistanceMetric (enum) - API compatibility
//
// We do NOT use VectorCore's implementations:
// - Distance kernels: VectorIndex has 2× faster unsafe pointer versions
// - Vector types: We work on raw [Float] and UnsafePointer<Float>
// - Batch operations: We provide specialized kernel implementations
//
// This separation maintains VectorIndex as a performance-focused layer
// while VectorCore remains a high-level, type-safe library.

// Note: VectorID is imported directly from VectorCore (no redefinition needed)

/// Basic search result representation.
public struct SearchResult: Sendable, Equatable {
    public let id: VectorID
    public let score: Float
    public init(id: VectorID, score: Float) {
        self.id = id
        self.score = score
    }
}

/// Basic statistics for an index instance.
public struct IndexStats: Sendable, Equatable {
    public let indexType: String
    public let vectorCount: Int
    public let dimension: Int
    public let metric: SupportedDistanceMetric
    /// Implementation-specific details (e.g., nlist, nprobe, layers)
    public let details: [String: String]
    public init(indexType: String, vectorCount: Int, dimension: Int, metric: SupportedDistanceMetric, details: [String: String] = [:]) {
        self.indexType = indexType
        self.vectorCount = vectorCount
        self.dimension = dimension
        self.metric = metric
        self.details = details
    }
}

/// Minimal index protocol: CPU-first and VectorCore-only.
public protocol VectorIndexProtocol: Actor {
    /// Fixed dimension for all vectors in this index
    var dimension: Int { get }

    /// Number of vectors stored
    var count: Int { get }

    /// Default distance metric for this index
    var metric: SupportedDistanceMetric { get }

    /// Initialize an empty index for given dimension and metric
    init(dimension: Int, metric: SupportedDistanceMetric)
    /// Insert or upsert a single vector.
    func insert(id: VectorID, vector: [Float], metadata: [String: String]?) async throws

    /// Remove a vector by ID (no-op if not present).
    func remove(id: VectorID) async throws

    /// k-NN search with optional filter (implementation-defined semantics).
    func search(query: [Float], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [SearchResult]

    /// Batch k-NN search
    func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [[SearchResult]]

    /// Batch insert convenience.
    func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?)]) async throws

    /// Optional optimization hook (e.g., rebuild graphs, compact structures).
    func optimize() async throws

    /// Remove all vectors
    func clear() async

    /// Return basic statistics about the index
    func statistics() async -> IndexStats

    /// Persist the index to a file URL (JSON format, versioned)
    func save(to url: URL) async throws

    /// Load an index instance from a file URL
    static func load(from url: URL) async throws -> Self

    /// Reclaim space and rebuild internal structures if needed
    func compact() async throws

    /// Check whether an id exists in the index
    func contains(id: VectorID) async -> Bool

    /// Update an existing entry. Pass nil to leave field unchanged.
    /// Returns true if the id existed and was updated.
    func update(id: VectorID, vector: [Float]?, metadata: [String:String]?) async throws -> Bool

    /// Remove a batch of ids (best effort). Nonexistent ids are ignored.
    func batchRemove(_ ids: [VectorID]) async throws
}
