// import XCTest
// import VectorCore
// @testable import VectorIndex
//
// final class SimplePerformanceTest: XCTestCase {
//    
//    func testHNSWvsFlat() async throws {
//        print("\n=== VectorIndex Performance Evaluation ===\n")
//        
//        let dimension = 128
//        let vectorCount = 1000
//        let queryCount = 100
//        let k = 10
//        
//        // Generate test data
//        var vectors: [[Float]] = []
//        for _ in 0..<vectorCount {
//            var vec = (0..<dimension).map { _ in Float.random(in: -1...1) }
//            let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
//            if norm > 0 { vec = vec.map { $0 / norm } }
//            vectors.append(vec)
//        }
//        
//        var queries: [[Float]] = []
//        for _ in 0..<queryCount {
//            var vec = (0..<dimension).map { _ in Float.random(in: -1...1) }
//            let norm = sqrt(vec.reduce(0) { $0 + $1 * $1 })
//            if norm > 0 { vec = vec.map { $0 / norm } }
//            queries.append(vec)
//        }
//        
//        // Test FlatIndex (brute force baseline)
//        print("1. FlatIndex (Brute Force)")
//        let flat = FlatIndex(dimension: dimension, metric: .euclidean)
//        
//        let flatBuildStart = CFAbsoluteTimeGetCurrent()
//        for (i, vec) in vectors.enumerated() {
//            try await flat.insert(id: "id\(i)", vector: vec, metadata: nil)
//        }
//        let flatBuildTime = CFAbsoluteTimeGetCurrent() - flatBuildStart
//        print("   Build time: \(flatBuildTime * 1000)ms")
//        
//        let flatQueryStart = CFAbsoluteTimeGetCurrent()
//        var flatResults: [[SearchResult]] = []
//        for query in queries {
//            flatResults.append(try await flat.search(query: query, k: k, filter: nil))
//        }
//        let flatQueryTime = CFAbsoluteTimeGetCurrent() - flatQueryStart
//        print("   Query time: \(flatQueryTime * 1000)ms total, \(flatQueryTime * 1000 / Double(queryCount))ms avg")
//        
//        // Test HNSWIndex
//        print("\n2. HNSWIndex")
//        let hnsw = HNSWIndex(dimension: dimension, metric: .euclidean)
//        
//        let hnswBuildStart = CFAbsoluteTimeGetCurrent()
//        for (i, vec) in vectors.enumerated() {
//            try await hnsw.insert(id: "id\(i)", vector: vec, metadata: nil)
//        }
//        let hnswBuildTime = CFAbsoluteTimeGetCurrent() - hnswBuildStart
//        print("   Build time: \(hnswBuildTime * 1000)ms")
//        
//        let hnswQueryStart = CFAbsoluteTimeGetCurrent()
//        var hnswResults: [[SearchResult]] = []
//        for query in queries {
//            hnswResults.append(try await hnsw.search(query: query, k: k, filter: nil))
//        }
//        let hnswQueryTime = CFAbsoluteTimeGetCurrent() - hnswQueryStart
//        print("   Query time: \(hnswQueryTime * 1000)ms total, \(hnswQueryTime * 1000 / Double(queryCount))ms avg")
//        
//        // Calculate recall
//        var totalRecall = 0.0
//        for i in 0..<queries.count {
//            let flatSet = Set(flatResults[i].map { $0.id })
//            let hnswSet = Set(hnswResults[i].map { $0.id })
//            let recall = Double(flatSet.intersection(hnswSet).count) / Double(k)
//            totalRecall += recall
//        }
//        let avgRecall = totalRecall / Double(queries.count)
//        print("   Recall@\(k): \(avgRecall * 100)%")
//        
//        // Test IVFIndex
//        print("\n3. IVFIndex")
//        let ivf = IVFIndex(dimension: dimension, metric: .euclidean, config: .init(nlist: 100, nprobe: 10))
//        
//        let ivfBuildStart = CFAbsoluteTimeGetCurrent()
//        for (i, vec) in vectors.enumerated() {
//            try await ivf.insert(id: "id\(i)", vector: vec, metadata: nil)
//        }
//        let ivfInsertTime = CFAbsoluteTimeGetCurrent() - ivfBuildStart
//        
//        let ivfOptimizeStart = CFAbsoluteTimeGetCurrent()
//        try await ivf.optimize()
//        let ivfOptimizeTime = CFAbsoluteTimeGetCurrent() - ivfOptimizeStart
//        print("   Insert time: \(ivfInsertTime * 1000)ms")
//        print("   Optimize time: \(ivfOptimizeTime * 1000)ms")
//        print("   Total build time: \((ivfInsertTime + ivfOptimizeTime) * 1000)ms")
//        
//        let ivfQueryStart = CFAbsoluteTimeGetCurrent()
//        var ivfResults: [[SearchResult]] = []
//        for query in queries {
//            ivfResults.append(try await ivf.search(query: query, k: k, filter: nil))
//        }
//        let ivfQueryTime = CFAbsoluteTimeGetCurrent() - ivfQueryStart
//        print("   Query time: \(ivfQueryTime * 1000)ms total, \(ivfQueryTime * 1000 / Double(queryCount))ms avg")
//        
//        // Calculate IVF recall
//        totalRecall = 0.0
//        for i in 0..<queries.count {
//            let flatSet = Set(flatResults[i].map { $0.id })
//            let ivfSet = Set(ivfResults[i].map { $0.id })
//            let recall = Double(flatSet.intersection(ivfSet).count) / Double(k)
//            totalRecall += recall
//        }
//        let ivfRecall = totalRecall / Double(queries.count)
//        print("   Recall@\(k): \(ivfRecall * 100)%")
//        
//        // Performance comparison
//        print("\n=== Performance Summary ===")
//        print("Dataset: \(vectorCount) vectors, dimension=\(dimension), \(queryCount) queries")
//        print("\nSpeedup vs Flat:")
//        print("  HNSW: \(flatQueryTime / hnswQueryTime)x faster")
//        print("  IVF: \(flatQueryTime / ivfQueryTime)x faster")
//        
//        print("\nStats:")
//        let hnswStats = await hnsw.statistics()
//        print("  HNSW: \(hnswStats)")
//        let ivfStats = await ivf.statistics()
//        print("  IVF: \(ivfStats)")
//        
//        // Success metrics evaluation
//        print("\n=== Success Metrics Evaluation ===")
//        print("Target: 1M vector HNSW index < 100ms build time")
//        print("Actual: \(vectorCount) vectors in \(hnswBuildTime * 1000)ms")
//        print("Extrapolated for 1M: ~\(hnswBuildTime * 1000000 / Double(vectorCount))ms")
//        
//        print("\nTarget: 95% recall @ 100x speedup")
//        print("Actual: \(avgRecall * 100)% recall @ \(flatQueryTime / hnswQueryTime)x speedup")
//        
//        print("\nTarget: < 10ms query latency for 1M vectors")
//        print("Actual: \(hnswQueryTime * 1000 / Double(queryCount))ms avg latency for \(vectorCount) vectors")
//        
//        print("\nGPU Acceleration: NOT IMPLEMENTED")
//        print("Current implementation is CPU-only")
//    }
// }
