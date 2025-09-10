//
//  VectorReferenceCollection.swift
//  VectorIndex
//
//  Zero-copy vector reference collection for efficient memory access
//

import Foundation

/// Protocol for collections that provide zero-copy vector access
public protocol VectorReferenceCollection: Sendable {
    /// Access vectors without copying via unsafe buffer pointer
    func withVectorReferences<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
    
    /// Get vector count
    var vectorCount: Int { get }
    
    /// Get vector dimension
    var dimension: Int { get }
}

/// Reference-based acceleration candidates that avoid all copying
public struct ReferenceAccelerationCandidates: Sendable {
    /// The candidate vector IDs
    public let ids: [VectorID]
    
    /// Reference to the underlying storage (retained for lifetime management)
    @usableFromInline
    internal let storageReference: ContiguousArray<Float>
    
    /// Offsets into storage for each vector (start index)
    public let vectorOffsets: [Int]
    
    /// Number of vectors
    public let vectorCount: Int
    
    /// Dimension of each vector
    public let dimension: Int
    
    /// Optional metadata for filtering
    public let metadata: [[String: String]?]
    
    /// Initialize with storage reference and offsets
    public init(
        ids: [VectorID],
        storageReference: ContiguousArray<Float>,
        vectorOffsets: [Int],
        dimension: Int,
        metadata: [[String: String]?]
    ) {
        self.ids = ids
        self.storageReference = storageReference
        self.vectorOffsets = vectorOffsets
        self.vectorCount = ids.count
        self.dimension = dimension
        self.metadata = metadata
    }
    
    /// Access a specific vector safely without copying
    /// The pointer is only valid within the closure scope
    @inlinable
    public func withVectorReference<R>(
        at index: Int,
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        guard index < vectorCount else {
            let empty = UnsafeBufferPointer<Float>(start: nil, count: 0)
            return try body(empty)
        }
        
        return try storageReference.withUnsafeBufferPointer { buffer in
            let offset = vectorOffsets[index]
            let vectorPtr = UnsafeBufferPointer(
                start: buffer.baseAddress?.advanced(by: offset),
                count: dimension
            )
            return try body(vectorPtr)
        }
    }
    
    /// Get vector as ArraySlice (safe but involves bounds checking)
    @inlinable
    public func vectorSlice(at index: Int) -> ArraySlice<Float> {
        guard index < vectorCount else { return ArraySlice() }
        let offset = vectorOffsets[index]
        return ArraySlice(storageReference[offset..<(offset + dimension)])
    }
    
    /// Access all vectors via unsafe buffer with custom strides
    @inlinable
    public func withAllVectorReferences<R>(
        _ body: (UnsafeBufferPointer<Float>, [Int]) throws -> R
    ) rethrows -> R {
        try storageReference.withUnsafeBufferPointer { buffer in
            try body(buffer, vectorOffsets)
        }
    }
    
    /// Convert to standard AccelerationCandidates (for compatibility)
    public func toStandardCandidates() -> AccelerationCandidates {
        // Only copy when absolutely necessary for compatibility
        var vectorStorage = ContiguousArray<Float>()
        vectorStorage.reserveCapacity(vectorCount * dimension)
        
        storageReference.withUnsafeBufferPointer { buffer in
            for offset in vectorOffsets {
                let vectorPtr = buffer.baseAddress?.advanced(by: offset)
                let vectorBuffer = UnsafeBufferPointer(start: vectorPtr, count: dimension)
                vectorStorage.append(contentsOf: vectorBuffer)
            }
        }
        
        return AccelerationCandidates(
            ids: ids,
            vectorStorage: vectorStorage,
            vectorCount: vectorCount,
            dimension: dimension,
            metadata: metadata
        )
    }
}

/// Unified contiguous storage for zero-copy vector access
public final class UnifiedVectorStorage {
    private var storage: ContiguousArray<Float>
    private let dimension: Int
    private var freeList: [Int] = []  // Indices of deleted vectors
    private let lock = NSLock()
    
    public init(dimension: Int, initialCapacity: Int = 1000) {
        self.dimension = dimension
        self.storage = ContiguousArray<Float>(repeating: 0, count: initialCapacity * dimension)
    }
    
    /// Allocate space for a new vector, returns offset
    public func allocateVector(_ vector: [Float]) -> Int {
        lock.lock()
        defer { lock.unlock() }
        
        let offset: Int
        if let reusedOffset = freeList.popLast() {
            offset = reusedOffset
        } else {
            offset = storage.count
            storage.append(contentsOf: Array(repeating: Float(0), count: dimension))
        }
        
        // Copy vector data to allocated space
        storage.withUnsafeMutableBufferPointer { buffer in
            let destPtr = buffer.baseAddress?.advanced(by: offset)
            vector.withUnsafeBufferPointer { srcBuffer in
                destPtr?.update(from: srcBuffer.baseAddress!, count: dimension)
            }
        }
        
        return offset
    }
    
    /// Deallocate a vector at given offset
    public func deallocateVector(at offset: Int) {
        lock.lock()
        defer { lock.unlock() }
        freeList.append(offset)
    }
    
    /// Access storage without copying
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    
    /// Get a reference to the storage (for creating candidates)
    public var storageReference: ContiguousArray<Float> {
        storage
    }
}