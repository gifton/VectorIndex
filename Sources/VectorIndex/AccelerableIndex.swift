//
//  AccelerableIndex.swift
//  VectorIndex
//
//  Protocol for indices that can expose internal data structures for
//  external acceleration (e.g., GPU processing). Allows VectorIndexAccelerated
//  to compose CPU implementations rather than duplicate them.
//

import Foundation
import VectorCore

/// Structure representing candidates for accelerated distance computation
/// Uses contiguous storage to minimize memory allocations and improve cache locality
public struct AccelerationCandidates: Sendable {
    /// The candidate vector IDs
    public let ids: [VectorID]
    /// Contiguous storage for all vectors (row-major: candidates × dimension)
    public let vectorStorage: ContiguousArray<Float>
    /// Number of vectors stored
    public let vectorCount: Int
    /// Dimension of each vector
    public let dimension: Int
    /// Optional metadata for filtering
    public let metadata: [[String: String]?]
    
    /// Initialize with pre-allocated contiguous storage
    public init(ids: [VectorID], vectorStorage: ContiguousArray<Float>, vectorCount: Int, dimension: Int, metadata: [[String: String]?]) {
        self.ids = ids
        self.vectorStorage = vectorStorage
        self.vectorCount = vectorCount
        self.dimension = dimension
        self.metadata = metadata
    }
    
    /// Access a specific vector without copying
    @inlinable
    public func vector(at index: Int) -> ArraySlice<Float> {
        guard index < vectorCount else { return ArraySlice() }
        let startIdx = index * dimension
        let endIdx = startIdx + dimension
        return ArraySlice(vectorStorage[startIdx..<endIdx])
    }
    
    /// Access vectors as a contiguous buffer for zero-copy operations
    @inlinable
    public func withUnsafeVectorBuffer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try vectorStorage.withUnsafeBufferPointer(body)
    }
    
    /// Create from existing storage without copying (zero-copy constructor)
    public static func createWithoutCopying(
        ids: [VectorID],
        existingStorage: ContiguousArray<Float>,
        vectorCount: Int,
        dimension: Int,
        metadata: [[String: String]?]
    ) -> AccelerationCandidates {
        // This constructor assumes the storage is already properly formatted
        // and will be retained for the lifetime of the candidates
        return AccelerationCandidates(
            ids: ids,
            vectorStorage: existingStorage,
            vectorCount: vectorCount,
            dimension: dimension,
            metadata: metadata
        )
    }
}

/// Results from accelerated processing to be integrated back
public struct AcceleratedResults: Sendable {
    /// Sorted indices into the original candidates array
    public let indices: [Int]
    /// Corresponding distances/scores
    public let distances: [Float]
    
    public init(indices: [Int], distances: [Float]) {
        self.indices = indices
        self.distances = distances
    }
}

/// Index-specific structures for acceleration
public enum IndexStructure: Sendable {
    /// HNSW graph structure
    case hnsw(HNSWStructure)
    /// IVF inverted file structure
    case ivf(IVFStructure)
    /// Flat index (no special structure)
    case flat
}

/// HNSW-specific structure for GPU traversal
public struct HNSWStructure: Sendable {
    /// Entry point node index
    public let entryPoint: Int?
    /// Maximum layer in the graph
    public let maxLevel: Int
    /// Adjacency lists per layer (node_index -> [neighbor_indices])
    public let layerGraphs: [[Set<Int>]]
    /// Node levels
    public let nodeLevels: [Int]
    
    public init(entryPoint: Int?, maxLevel: Int, layerGraphs: [[Set<Int>]], nodeLevels: [Int]) {
        self.entryPoint = entryPoint
        self.maxLevel = maxLevel
        self.layerGraphs = layerGraphs
        self.nodeLevels = nodeLevels
    }
}

/// IVF-specific structure for GPU search
public struct IVFStructure: Sendable {
    /// Cluster centroids (nlist × dimension)
    public let centroids: [[Float]]
    /// Inverted lists mapping cluster to vector IDs
    public let invertedLists: [[VectorID]]
    /// Number of clusters to probe
    public let nprobe: Int
    
    public init(centroids: [[Float]], invertedLists: [[VectorID]], nprobe: Int) {
        self.centroids = centroids
        self.invertedLists = invertedLists
        self.nprobe = nprobe
    }
}

/// Protocol for indices that can expose internal structures for acceleration
public protocol AccelerableIndex: VectorIndexProtocol {
    /// Get candidate vectors for a search query
    /// - Parameters:
    ///   - query: The query vector
    ///   - k: Number of results desired
    ///   - filter: Optional metadata filter
    /// - Returns: Candidates that need distance computation
    func getCandidates(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> AccelerationCandidates
    
    /// Get candidates for batch search
    /// - Parameters:
    ///   - queries: The query vectors
    ///   - k: Number of results per query
    ///   - filter: Optional metadata filter
    /// - Returns: Candidates for each query
    func getBatchCandidates(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [AccelerationCandidates]
    
    /// Get the index structure for specialized acceleration
    /// - Returns: Index-specific structure information
    func getIndexStructure() async -> IndexStructure
    
    /// Finalize results from accelerated computation
    /// - Parameters:
    ///   - candidates: The original candidates
    ///   - results: Results from accelerated processing
    ///   - filter: Optional metadata filter
    /// - Returns: Final search results
    func finalizeResults(
        candidates: AccelerationCandidates,
        results: AcceleratedResults,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async -> [SearchResult]
    
    /// Finalize batch results from accelerated computation
    /// - Parameters:
    ///   - batchCandidates: The original candidates for each query
    ///   - batchResults: Results from accelerated processing for each query
    ///   - filter: Optional metadata filter
    /// - Returns: Final search results for each query
    func finalizeBatchResults(
        batchCandidates: [AccelerationCandidates],
        batchResults: [AcceleratedResults],
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async -> [[SearchResult]]
    
    /// Check if acceleration would be beneficial for given parameters
    /// - Parameters:
    ///   - queryCount: Number of queries
    ///   - candidateCount: Estimated number of candidates
    ///   - k: Number of results per query
    /// - Returns: Whether acceleration is recommended
    func shouldAccelerate(
        queryCount: Int,
        candidateCount: Int,
        k: Int
    ) async -> Bool
}

/// Default implementations for common functionality
public extension AccelerableIndex {
    func shouldAccelerate(queryCount: Int, candidateCount: Int, k: Int) async -> Bool {
        // Default heuristics: accelerate if sufficient work
        let totalOperations = queryCount * candidateCount * k
        return totalOperations >= 50_000 && candidateCount >= 500
    }
    
    func finalizeBatchResults(
        batchCandidates: [AccelerationCandidates],
        batchResults: [AcceleratedResults],
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async -> [[SearchResult]] {
        var finalResults: [[SearchResult]] = []
        for (candidates, results) in zip(batchCandidates, batchResults) {
            let queryResults = await finalizeResults(
                candidates: candidates,
                results: results,
                filter: filter
            )
            finalResults.append(queryResults)
        }
        return finalResults
    }
}