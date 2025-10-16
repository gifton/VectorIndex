import XCTest
import VectorCore
@testable import VectorIndex

final class PerformanceBenchmarks: XCTestCase {
    private let enableBenchmarks: Bool = ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"

    override func setUpWithError() throws {
        if !enableBenchmarks {
            throw XCTSkip("Benchmarks disabled by default. Set RUN_BENCHMARKS=1 to enable.")
        }
    }
    
    // Generate random normalized vectors
    func generateVectors(_ count: Int, dimension: Int) -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(count)
        for _ in 0..<count {
            var vec = (0..<dimension).map { _ in Float.random(in: -1...1) }
            let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
            if norm > 0 {
                vec = vec.map { $0 / norm }
            }
            vectors.append(vec)
        }
        return vectors
    }
    
    // MARK: - HNSW Performance Tests
    
    func testHNSWBuildTime1KVectors() async throws {
        let vectors = generateVectors(1000, dimension: 128)
        let index = HNSWIndex(dimension: 128, metric: .euclidean)
        
        let start = CFAbsoluteTimeGetCurrent()
        for (i, vec) in vectors.enumerated() {
            try await index.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        print("HNSW Build time for 1K vectors: \(elapsed * 1000)ms")
        XCTAssertLessThan(elapsed, 1.0, "Should build 1K vectors in < 1 second")
    }
    
    func testHNSWQueryLatency1KVectors() async throws {
        let vectors = generateVectors(1000, dimension: 128)
        let queries = generateVectors(100, dimension: 128)
        let index = HNSWIndex(dimension: 128, metric: .euclidean)
        
        for (i, vec) in vectors.enumerated() {
            try await index.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        
        let start = CFAbsoluteTimeGetCurrent()
        for query in queries {
            _ = try await index.search(query: query, k: 10, filter: nil)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgLatency = (elapsed * 1000) / Double(queries.count)
        
        print("HNSW Average query latency (1K vectors): \(avgLatency)ms")
        XCTAssertLessThan(avgLatency, 1.0, "Query latency should be < 1ms for 1K vectors")
    }
    
    func testHNSWMemoryUsage() async throws {
        let vectorCount = 10000
        let dimension = 128
        let vectors = generateVectors(vectorCount, dimension: dimension)
        let index = HNSWIndex(dimension: dimension, metric: .euclidean)
        
        let rawSize = vectorCount * dimension * MemoryLayout<Float>.size
        
        for (i, vec) in vectors.enumerated() {
            try await index.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        
        let stats = await index.statistics()
        print("HNSW Stats for 10K vectors: \(stats)")
        
        // Note: Can't directly measure actor memory usage, but we can estimate
        // HNSW typically uses 1.5-2x raw size due to graph structure
        let estimatedMemory = rawSize * 2  // Conservative estimate
        print("Estimated memory usage: \(estimatedMemory / (1024*1024))MB for raw size: \(rawSize / (1024*1024))MB")
    }
    
    // MARK: - IVF Performance Tests
    
    func testIVFBuildTimeWithOptimize() async throws {
        let vectors = generateVectors(5000, dimension: 128)
        let index = IVFIndex(dimension: 128, metric: .euclidean, config: .init(nlist: 100, nprobe: 10))
        
        let insertStart = CFAbsoluteTimeGetCurrent()
        for (i, vec) in vectors.enumerated() {
            try await index.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        let insertTime = CFAbsoluteTimeGetCurrent() - insertStart
        
        let optimizeStart = CFAbsoluteTimeGetCurrent()
        try await index.optimize()
        let optimizeTime = CFAbsoluteTimeGetCurrent() - optimizeStart
        
        print("IVF Insert time for 5K vectors: \(insertTime * 1000)ms")
        print("IVF Optimize (kmeans) time: \(optimizeTime * 1000)ms")
        print("IVF Total build time: \((insertTime + optimizeTime) * 1000)ms")
        
        XCTAssertLessThan(insertTime + optimizeTime, 10.0, "Should build and optimize 5K vectors in < 10 seconds")
    }
    
    func testIVFQueryLatency() async throws {
        let vectors = generateVectors(5000, dimension: 128)
        let queries = generateVectors(100, dimension: 128)
        let index = IVFIndex(dimension: 128, metric: .euclidean, config: .init(nlist: 100, nprobe: 10))
        
        for (i, vec) in vectors.enumerated() {
            try await index.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        try await index.optimize()
        
        let start = CFAbsoluteTimeGetCurrent()
        for query in queries {
            _ = try await index.search(query: query, k: 10, filter: nil)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgLatency = (elapsed * 1000) / Double(queries.count)
        
        print("IVF Average query latency (5K vectors, nprobe=10): \(avgLatency)ms")
        XCTAssertLessThan(avgLatency, 2.0, "Query latency should be < 2ms for 5K vectors")
    }
    
    // MARK: - Comparison Tests
    
    func testSpeedupVsBruteForce() async throws {
        let vectors = generateVectors(1000, dimension: 128)
        let queries = generateVectors(10, dimension: 128)
        
        // Brute force (FlatIndex)
        let flat = FlatIndex(dimension: 128, metric: .euclidean)
        for (i, vec) in vectors.enumerated() {
            try await flat.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        
        let flatStart = CFAbsoluteTimeGetCurrent()
        var flatResults: [[SearchResult]] = []
        for query in queries {
            flatResults.append(try await flat.search(query: query, k: 10, filter: nil))
        }
        let flatTime = CFAbsoluteTimeGetCurrent() - flatStart
        
        // HNSW
        let hnsw = HNSWIndex(dimension: 128, metric: .euclidean)
        for (i, vec) in vectors.enumerated() {
            try await hnsw.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        
        let hnswStart = CFAbsoluteTimeGetCurrent()
        var hnswResults: [[SearchResult]] = []
        for query in queries {
            hnswResults.append(try await hnsw.search(query: query, k: 10, filter: nil))
        }
        let hnswTime = CFAbsoluteTimeGetCurrent() - hnswStart
        
        let speedup = flatTime / hnswTime
        print("HNSW speedup vs brute force: \(speedup)x")
        print("Flat time: \(flatTime * 1000)ms, HNSW time: \(hnswTime * 1000)ms")
        
        // Calculate recall
        var totalRecall = 0.0
        for i in 0..<queries.count {
            let flatSet = Set(flatResults[i].prefix(10).map { $0.id })
            let hnswSet = Set(hnswResults[i].prefix(10).map { $0.id })
            let recall = Double(flatSet.intersection(hnswSet).count) / 10.0
            totalRecall += recall
        }
        let avgRecall = totalRecall / Double(queries.count)
        print("HNSW recall@10: \(avgRecall * 100)%")
        
        XCTAssertGreaterThan(speedup, 2.0, "HNSW should be at least 2x faster than brute force")
        XCTAssertGreaterThan(avgRecall, 0.9, "HNSW should have > 90% recall")
    }
    
    // MARK: - Scale Tests (commented out due to runtime)
    
    /*
    func testHNSWLargeScale() async throws {
        // This would test with 100K-1M vectors but takes too long for regular tests
        let vectors = generateVectors(100000, dimension: 128)
        let index = HNSWIndex(dimension: 128, metric: .euclidean)
        
        let start = CFAbsoluteTimeGetCurrent()
        for (i, vec) in vectors.enumerated() {
            if i % 10000 == 0 {
                print("Inserted \(i) vectors...")
            }
            try await index.insert(id: "id\(i)", vector: vec, metadata: nil)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        print("HNSW Build time for 100K vectors: \(elapsed)s")
        
        // Target: < 100ms for 1M vectors means < 10ms for 100K
        XCTAssertLessThan(elapsed, 10.0, "Should build 100K vectors in < 10 seconds")
    }
    */
}
