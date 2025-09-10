//
//  AccelerableIndexEnhanced.swift
//  VectorIndex
//
//  Enhanced protocol for truly zero-copy acceleration with safe pointer access
//

import Foundation
import VectorCore

/// Enhanced protocol that provides zero-copy access with safe pointer lifetime management
public protocol AccelerableIndexEnhanced: AccelerableIndex {
    /// Execute a closure with zero-copy access to all candidate vectors
    /// The pointers are only valid within the closure scope
    func withCandidateReferences<R>(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?,
        body: (UnsafeBufferPointer<Float>, [VectorID], [[String: String]?], Int, Int) throws -> R
    ) async throws -> R
}

/// Zero-copy acceleration candidates with safe access patterns
public struct SafeAccelerationCandidates: Sendable {
    /// The candidate vector IDs
    public let ids: [VectorID]
    
    /// Storage reference for lifetime management
    @usableFromInline
    internal let storage: ContiguousArray<Float>
    
    /// Offsets or layout information
    @usableFromInline
    internal let offsets: [Int]
    
    /// Number of vectors
    public let vectorCount: Int
    
    /// Dimension of each vector
    public let dimension: Int
    
    /// Optional metadata
    public let metadata: [[String: String]?]
    
    public init(
        ids: [VectorID],
        storage: ContiguousArray<Float>,
        offsets: [Int],
        dimension: Int,
        metadata: [[String: String]?]
    ) {
        self.ids = ids
        self.storage = storage
        self.offsets = offsets
        self.vectorCount = ids.count
        self.dimension = dimension
        self.metadata = metadata
    }
    
    /// Safe access to a specific vector
    @inlinable
    public func withVector<R>(
        at index: Int,
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        guard index < vectorCount else {
            return try body(UnsafeBufferPointer(start: nil, count: 0))
        }
        
        return try storage.withUnsafeBufferPointer { buffer in
            let offset = offsets[index]
            let vectorPtr = UnsafeBufferPointer(
                start: buffer.baseAddress?.advanced(by: offset),
                count: dimension
            )
            return try body(vectorPtr)
        }
    }
    
    /// Safe access to all vectors for batch operations
    @inlinable
    public func withAllVectors<R>(
        _ body: (UnsafeBufferPointer<Float>, [Int]) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeBufferPointer { buffer in
            try body(buffer, offsets)
        }
    }
    
    /// Compute distance to query vector without copying
    @inlinable
    public func distanceToQuery(
        at index: Int,
        query: [Float],
        metric: SupportedDistanceMetric
    ) -> Float {
        withVector(at: index) { vectorPtr in
            guard let baseAddress = vectorPtr.baseAddress else { return Float.infinity }
            
            return query.withUnsafeBufferPointer { queryPtr in
                guard let queryBase = queryPtr.baseAddress else { return Float.infinity }
                
                switch metric {
                case .euclidean:
                    var sum: Float = 0
                    for i in 0..<dimension {
                        let diff = queryBase[i] - baseAddress[i]
                        sum += diff * diff
                    }
                    return sqrt(sum)
                    
                case .cosine:
                    var dot: Float = 0
                    var normA: Float = 0
                    var normB: Float = 0
                    for i in 0..<dimension {
                        let a = queryBase[i]
                        let b = baseAddress[i]
                        dot += a * b
                        normA += a * a
                        normB += b * b
                    }
                    return 1.0 - (dot / (sqrt(normA) * sqrt(normB)))
                    
                case .dotProduct:
                    var dot: Float = 0
                    for i in 0..<dimension {
                        dot += queryBase[i] * baseAddress[i]
                    }
                    return -dot
                    
                default:
                    // For other metrics, fall back to VectorCore
                    let vector = Array(vectorPtr)
                    return distance(query, vector, metric: metric)
                }
            }
        }
    }
}