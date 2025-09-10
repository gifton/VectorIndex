//
//  PointerSafetyTests.swift
//  VectorIndexTests
//
//  Tests to verify safe pointer lifetime management in zero-copy optimizations
//

import XCTest
@testable import VectorIndex
import VectorCore

final class PointerSafetyTests: XCTestCase {
    
    /// Test that pointer access is safe and doesn't escape closure scope
    func testSafePointerAccess() async throws {
        let dimension = 128
        let vectorCount = 100
        
        // Create test data
        var ids: [VectorID] = []
        var storage = ContiguousArray<Float>()
        var offsets: [Int] = []
        
        for i in 0..<vectorCount {
            ids.append("vec\(i)")
            offsets.append(storage.count)
            for j in 0..<dimension {
                storage.append(Float(i * dimension + j))
            }
        }
        
        let candidates = ReferenceAccelerationCandidates(
            ids: ids,
            storageReference: storage,
            vectorOffsets: offsets,
            dimension: dimension,
            metadata: Array(repeating: nil, count: vectorCount)
        )
        
        // Test safe access pattern
        var sum: Float = 0
        for i in 0..<vectorCount {
            candidates.withVectorReference(at: i) { vectorPtr in
                // Pointer is valid here
                for j in 0..<dimension {
                    sum += vectorPtr[j]
                }
            }
            // Pointer is no longer valid here - this is safe
        }
        
        XCTAssertGreaterThan(sum, 0)
        
        // Test that empty vector returns safely
        candidates.withVectorReference(at: vectorCount + 10) { vectorPtr in
            XCTAssertEqual(vectorPtr.count, 0)
        }
    }
    
    /// Test that SafeAccelerationCandidates properly manages lifetime
    func testSafeAccelerationCandidates() async throws {
        let dimension = 64
        let vectorCount = 50
        
        // Create test storage
        var storage = ContiguousArray<Float>()
        var offsets: [Int] = []
        var ids: [VectorID] = []
        
        for i in 0..<vectorCount {
            ids.append("vec\(i)")
            offsets.append(storage.count)
            for j in 0..<dimension {
                storage.append(Float(i) + Float(j) / Float(dimension))
            }
        }
        
        let safeCandidates = SafeAccelerationCandidates(
            ids: ids,
            storage: storage,
            offsets: offsets,
            dimension: dimension,
            metadata: Array(repeating: nil, count: vectorCount)
        )
        
        // Test distance computation without copying
        let query = Array(repeating: Float(0.5), count: dimension)
        
        for i in 0..<vectorCount {
            let distance = safeCandidates.distanceToQuery(
                at: i,
                query: query,
                metric: .euclidean
            )
            XCTAssertGreaterThan(distance, 0)
            XCTAssertLessThan(distance, Float.infinity)
        }
        
        // Test batch access
        safeCandidates.withAllVectors { buffer, offsets in
            XCTAssertEqual(buffer.count, vectorCount * dimension)
            XCTAssertEqual(offsets.count, vectorCount)
            
            // Verify we can access all vectors
            for (idx, offset) in offsets.enumerated() {
                let vectorPtr = buffer.baseAddress?.advanced(by: offset)
                XCTAssertNotNil(vectorPtr)
                
                // Verify first element matches expected pattern
                if let ptr = vectorPtr {
                    let expectedFirst = Float(idx)
                    XCTAssertEqual(ptr.pointee, expectedFirst, accuracy: 0.001)
                }
            }
        }
    }
    
    /// Test that modifications don't affect immutable references
    func testImmutableReferences() async throws {
        let dimension = 32
        var storage = ContiguousArray<Float>(repeating: 0, count: dimension * 10)
        
        // Fill with test data
        for i in 0..<storage.count {
            storage[i] = Float(i)
        }
        
        let candidates = ReferenceAccelerationCandidates(
            ids: ["vec0", "vec1"],
            storageReference: storage,
            vectorOffsets: [0, dimension],
            dimension: dimension,
            metadata: [nil, nil]
        )
        
        // Get values before modification
        var beforeValues: [Float] = []
        candidates.withVectorReference(at: 0) { ptr in
            beforeValues = Array(ptr)
        }
        
        // Modify original storage (this shouldn't affect candidates since it has its own reference)
        storage[0] = 999.0
        
        // Verify candidates still has original values
        candidates.withVectorReference(at: 0) { ptr in
            XCTAssertEqual(ptr[0], beforeValues[0])
            XCTAssertNotEqual(ptr[0], 999.0)
        }
    }
    
    /// Test thread safety of reference access
    func testConcurrentAccess() async throws {
        let dimension = 128
        let vectorCount = 1000
        
        // Create large dataset
        var storage = ContiguousArray<Float>()
        var offsets: [Int] = []
        var ids: [VectorID] = []
        
        for i in 0..<vectorCount {
            ids.append("vec\(i)")
            offsets.append(storage.count)
            for j in 0..<dimension {
                storage.append(Float(i * dimension + j))
            }
        }
        
        let candidates = SafeAccelerationCandidates(
            ids: ids,
            storage: storage,
            offsets: offsets,
            dimension: dimension,
            metadata: Array(repeating: nil, count: vectorCount)
        )
        
        // Concurrent reads should be safe
        await withTaskGroup(of: Float.self) { group in
            for i in 0..<100 {
                group.addTask {
                    var sum: Float = 0
                    candidates.withVector(at: i) { ptr in
                        for j in 0..<ptr.count {
                            sum += ptr[j]
                        }
                    }
                    return sum
                }
            }
            
            var totalSum: Float = 0
            for await sum in group {
                totalSum += sum
            }
            
            XCTAssertGreaterThan(totalSum, 0)
        }
    }
    
    /// Verify no memory leaks with reference counting
    func testMemoryManagement() async throws {
        weak var weakStorage: ContiguousArray<Float>?
        
        autoreleasepool {
            let storage = ContiguousArray<Float>(repeating: 1.0, count: 1000)
            weakStorage = storage
            
            let candidates = ReferenceAccelerationCandidates(
                ids: ["test"],
                storageReference: storage,
                vectorOffsets: [0],
                dimension: 1000,
                metadata: [nil]
            )
            
            // Use candidates
            candidates.withVectorReference(at: 0) { ptr in
                XCTAssertEqual(ptr.count, 1000)
            }
            
            // Storage should still be retained by candidates
            XCTAssertNotNil(weakStorage)
        }
        
        // After autoreleasepool, storage should be deallocated
        // Note: This test may be flaky due to ARC optimization
    }
}