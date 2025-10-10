//
//  ArrayCopyOptimizationBenchmark.swift
//  VectorIndexTests
//
//  Benchmarks to measure the performance improvements from array copy optimization
//

import XCTest
@testable import VectorIndex
import VectorCore

final class ArrayCopyOptimizationBenchmark: XCTestCase {
    
    /// Measure memory allocations and performance of FlatIndex candidate extraction
    func testFlatIndexCandidatePerformance() async throws {
        let dimension = 128
        let vectorCount = 10_000
        let index = FlatIndex(dimension: dimension, metric: .euclidean)
        
        // Insert test vectors
        for i in 0..<vectorCount {
            let vector = (0..<dimension).map { Float($0 + i) / Float(dimension) }
            try await index.insert(id: "vec\(i)", vector: vector, metadata: ["idx": "\(i)"])
        }
        
        let query = Array(repeating: Float(0.5), count: dimension)
        
        // Warm up
        _ = try await index.getCandidates(query: query, k: 100, filter: nil)
        
        // Measure getCandidates performance
        let startTime = CFAbsoluteTimeGetCurrent()
        var totalVectorCount = 0
        
        for _ in 0..<100 {
            let candidates = try await index.getCandidates(query: query, k: 100, filter: nil)
            totalVectorCount += candidates.vectorCount
            
            // Verify contiguous storage is being used
            XCTAssertEqual(candidates.vectorStorage.count, candidates.vectorCount * dimension)
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("FlatIndex getCandidates (100 iterations):")
        print("  Time: \(String(format: "%.3f", elapsed * 1000))ms")
        print("  Avg per call: \(String(format: "%.3f", elapsed * 10))ms")
        print("  Vectors processed: \(totalVectorCount)")
        print("  Throughput: \(String(format: "%.0f", Double(totalVectorCount) / elapsed)) vectors/sec")
    }
    
    /// Measure memory efficiency of contiguous storage
    func testContiguousStorageMemoryEfficiency() async throws {
        let dimension = 512
        let vectorCount = 1000
        
        // Test old approach (nested arrays)
        let oldStartMemory = getCurrentMemoryUsage()
        var oldVectors: [[Float]] = []
        for i in 0..<vectorCount {
            let vector = (0..<dimension).map { Float($0 + i) / Float(dimension) }
            oldVectors.append(vector)
        }
        let oldEndMemory = getCurrentMemoryUsage()
        let oldMemoryUsed = oldEndMemory - oldStartMemory
        
        // Clear to reset memory
        oldVectors.removeAll()
        
        // Test new approach (contiguous storage)
        let newStartMemory = getCurrentMemoryUsage()
        var contiguousStorage = ContiguousArray<Float>()
        contiguousStorage.reserveCapacity(vectorCount * dimension)
        for i in 0..<vectorCount {
            for j in 0..<dimension {
                contiguousStorage.append(Float(j + i) / Float(dimension))
            }
        }
        let newEndMemory = getCurrentMemoryUsage()
        let newMemoryUsed = newEndMemory - newStartMemory
        
        print("\nMemory Efficiency Test:")
        print("  Nested Arrays: \(formatBytes(oldMemoryUsed))")
        print("  Contiguous Storage: \(formatBytes(newMemoryUsed))")
        print("  Memory Saved: \(formatBytes(oldMemoryUsed - newMemoryUsed))")
        print("  Reduction: \(String(format: "%.1f", (1.0 - Double(newMemoryUsed)/Double(oldMemoryUsed)) * 100))%")
        
        // Verify significant memory savings
        XCTAssertLessThan(newMemoryUsed, oldMemoryUsed)
    }
    
    /// Test zero-copy access performance
    func testZeroCopyAccess() async throws {
        let dimension = 256
        let vectorCount = 5000
        
        // Create test candidates with contiguous storage
        var ids: [VectorID] = []
        var vectorStorage = ContiguousArray<Float>()
        vectorStorage.reserveCapacity(vectorCount * dimension)
        
        for i in 0..<vectorCount {
            ids.append("vec\(i)")
            for j in 0..<dimension {
                vectorStorage.append(Float(j + i) / Float(dimension))
            }
        }
        
        let candidates = AccelerationCandidates(
            ids: ids,
            vectorStorage: vectorStorage,
            vectorCount: vectorCount,
            dimension: dimension,
            metadata: Array(repeating: nil, count: vectorCount)
        )
        
        // Measure zero-copy access
        let startTime = CFAbsoluteTimeGetCurrent()
        var sum: Float = 0
        
        // Use withUnsafeVectorBuffer for zero-copy access
        for _ in 0..<1000 {
            candidates.withUnsafeVectorBuffer { buffer in
                // Simulate distance computation
                for i in 0..<vectorCount {
                    let offset = i * dimension
                    // Access vector without copying
                    sum += buffer[offset]
                }
            }
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        print("\nZero-Copy Access Test:")
        print("  1000 iterations over \(vectorCount) vectors")
        print("  Time: \(String(format: "%.3f", elapsed * 1000))ms")
        print("  Throughput: \(String(format: "%.0f", Double(vectorCount * 1000) / elapsed)) vectors/sec")
        
        // Prevent optimization
        XCTAssertNotEqual(sum, 0)
    }
    
    /// Compare HNSW candidate extraction performance
    func testHNSWCandidateOptimization() async throws {
        let dimension = 128
        let vectorCount = 1000
        let index = HNSWIndex(dimension: dimension, metric: .euclidean)
        
        // Insert vectors
        for i in 0..<vectorCount {
            let vector = (0..<dimension).map { _ in Float.random(in: 0...1) }
            try await index.insert(id: "vec\(i)", vector: vector, metadata: nil)
        }
        
        let query = Array(repeating: Float(0.5), count: dimension)
        
        // Warm up
        _ = try await index.getCandidates(query: query, k: 50, filter: nil)
        
        // Benchmark
        let iterations = 100
        let startTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            let candidates = try await index.getCandidates(query: query, k: 50, filter: nil)
            // Verify optimization is working
            XCTAssertGreaterThan(candidates.vectorCount, 0)
            XCTAssertEqual(candidates.vectorStorage.count, candidates.vectorCount * dimension)
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        print("\nHNSW getCandidates Performance:")
        print("  \(iterations) iterations")
        print("  Time: \(String(format: "%.3f", elapsed * 1000))ms")
        print("  Avg per call: \(String(format: "%.3f", elapsed * 1000 / Double(iterations)))ms")
    }
    
    // MARK: - Helper Functions
    
    private func getCurrentMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: bytes)
    }
}
