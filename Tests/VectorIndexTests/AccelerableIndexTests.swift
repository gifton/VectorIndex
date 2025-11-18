//
//  AccelerableIndexTests.swift
//  VectorIndexTests
//
//  Tests for the AccelerableIndex protocol implementation
//

import XCTest
@testable import VectorIndex
import VectorCore

final class AccelerableIndexTests: XCTestCase {
    
    func testFlatIndexAcceleration() async throws {
        let index = FlatIndex(dimension: 3, metric: .euclidean)
        
        // Insert test vectors
        try await index.insert(id: "vec1", vector: [1.0, 2.0, 3.0], metadata: ["type": "A"])
        try await index.insert(id: "vec2", vector: [4.0, 5.0, 6.0], metadata: ["type": "B"])
        try await index.insert(id: "vec3", vector: [7.0, 8.0, 9.0], metadata: ["type": "A"])
        
        // Test getCandidates
        let query: [Float] = [2.0, 3.0, 4.0]
        let candidates = try await index.getCandidates(query: query, k: 2, filter: nil)
        
        XCTAssertEqual(candidates.ids.count, 3)
        XCTAssertEqual(candidates.vectorCount, 3)
        XCTAssertEqual(candidates.metadata.count, 3)
        XCTAssertTrue(candidates.ids.contains("vec1"))
        XCTAssertTrue(candidates.ids.contains("vec2"))
        XCTAssertTrue(candidates.ids.contains("vec3"))
        
        // Test getIndexStructure
        let structure = await index.getIndexStructure()
        if case .flat = structure {
            // Expected for flat index
        } else {
            XCTFail("Expected flat structure")
        }
        
        // Test finalizeResults
        let acceleratedResults = AcceleratedResults(
            indices: [0, 1],  // First two candidates
            distances: [1.732, 5.196]  // Example distances
        )
        
        let finalResults = await index.finalizeResults(
            candidates: candidates,
            results: acceleratedResults,
            filter: nil
        )
        
        XCTAssertEqual(finalResults.count, 2)
        XCTAssertEqual(finalResults[0].score, 1.732, accuracy: 0.001)
        XCTAssertEqual(finalResults[1].score, 5.196, accuracy: 0.001)
        
        // Test with filter
        let filteredResults = await index.finalizeResults(
            candidates: candidates,
            results: acceleratedResults,
            filter: { meta in meta?["type"] == "A" }
        )
        
        // Should filter out results based on metadata
        XCTAssertLessThanOrEqual(filteredResults.count, 2)
    }
    
    func testHNSWIndexAcceleration() async throws {
        let index = HNSWIndex(dimension: 3, metric: .euclidean)
        
        // Insert test vectors
        try await index.insert(id: "vec1", vector: [1.0, 2.0, 3.0], metadata: ["type": "A"])
        try await index.insert(id: "vec2", vector: [4.0, 5.0, 6.0], metadata: ["type": "B"])
        try await index.insert(id: "vec3", vector: [7.0, 8.0, 9.0], metadata: ["type": "A"])
        try await index.insert(id: "vec4", vector: [2.0, 3.0, 4.0], metadata: ["type": "C"])
        
        // Test getCandidates
        let query: [Float] = [3.0, 4.0, 5.0]
        let candidates = try await index.getCandidates(query: query, k: 2, filter: nil)
        
        XCTAssertGreaterThan(candidates.ids.count, 0)
        XCTAssertEqual(candidates.ids.count, candidates.vectorCount)
        XCTAssertEqual(candidates.ids.count, candidates.metadata.count)
        
        // Test getIndexStructure
        let structure = await index.getIndexStructure()
        if case .hnsw(let hnswStructure) = structure {
            XCTAssertNotNil(hnswStructure.entryPoint)
            XCTAssertGreaterThanOrEqual(hnswStructure.maxLevel, 0)
            XCTAssertGreaterThan(hnswStructure.layerGraphs.count, 0)
        } else {
            XCTFail("Expected HNSW structure")
        }
        
        // Test batch candidates
        let queries: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        let batchCandidates = try await index.getBatchCandidates(queries: queries, k: 2, filter: nil)
        
        XCTAssertEqual(batchCandidates.count, 2)
        for candidates in batchCandidates {
            XCTAssertGreaterThan(candidates.ids.count, 0)
        }
    }
    
    func testIVFIndexAcceleration() async throws {
        let config = IVFIndex.Configuration(nlist: 2, nprobe: 1)
        let index = IVFIndex(dimension: 3, metric: .euclidean, config: config)
        
        // Insert test vectors
        try await index.insert(id: "vec1", vector: [1.0, 2.0, 3.0], metadata: ["type": "A"])
        try await index.insert(id: "vec2", vector: [4.0, 5.0, 6.0], metadata: ["type": "B"])
        try await index.insert(id: "vec3", vector: [7.0, 8.0, 9.0], metadata: ["type": "A"])
        try await index.insert(id: "vec4", vector: [2.0, 3.0, 4.0], metadata: ["type": "C"])
        
        // Optimize to create centroids
        try await index.optimize()
        
        // Test getCandidates after optimization
        let query: [Float] = [3.0, 4.0, 5.0]
        let candidates = try await index.getCandidates(query: query, k: 2, filter: nil)
        
        XCTAssertGreaterThan(candidates.ids.count, 0)
        XCTAssertEqual(candidates.ids.count, candidates.vectorCount)
        XCTAssertEqual(candidates.ids.count, candidates.metadata.count)
        
        // Test getIndexStructure
        let structure = await index.getIndexStructure()
        if case .ivf(let ivfStructure) = structure {
            XCTAssertGreaterThan(ivfStructure.centroids.count, 0)
            XCTAssertEqual(ivfStructure.centroids.count, ivfStructure.invertedLists.count)
            XCTAssertEqual(ivfStructure.nprobe, 1)
        } else {
            XCTFail("Expected IVF structure")
        }
        
        // Test shouldAccelerate
        let shouldUseGPU = await index.shouldAccelerate(
            queryCount: 100,
            candidateCount: 1000,
            k: 10
        )
        XCTAssertTrue(shouldUseGPU)  // 100 * 1000 * 10 = 1,000,000 > 50,000 threshold
        
        let shouldNotUseGPU = await index.shouldAccelerate(
            queryCount: 1,
            candidateCount: 100,
            k: 10
        )
        XCTAssertFalse(shouldNotUseGPU)  // 1 * 100 * 10 = 1,000 < 50,000 threshold
    }
    
    func testBatchFinalization() async throws {
        let index = FlatIndex(dimension: 3, metric: .euclidean)
        
        // Insert test vectors
        try await index.insert(id: "vec1", vector: [1.0, 2.0, 3.0], metadata: ["type": "A"])
        try await index.insert(id: "vec2", vector: [4.0, 5.0, 6.0], metadata: ["type": "B"])
        
        let candidates = try await index.getCandidates(query: [0, 0, 0], k: 2, filter: nil)
        
        let batchCandidates = [candidates, candidates]
        let batchResults = [
            AcceleratedResults(indices: [0, 1], distances: [1.0, 2.0]),
            AcceleratedResults(indices: [1, 0], distances: [1.5, 2.5])
        ]
        
        let finalBatchResults = await index.finalizeBatchResults(
            batchCandidates: batchCandidates,
            batchResults: batchResults,
            filter: nil
        )
        
        XCTAssertEqual(finalBatchResults.count, 2)
        XCTAssertEqual(finalBatchResults[0].count, 2)
        XCTAssertEqual(finalBatchResults[1].count, 2)
        
        // First batch should have results in order of indices [0, 1]
        XCTAssertEqual(finalBatchResults[0][0].score, 1.0)
        XCTAssertEqual(finalBatchResults[0][1].score, 2.0)
        
        // Second batch should have results in order of indices [1, 0]
        XCTAssertEqual(finalBatchResults[1][0].score, 1.5)
        XCTAssertEqual(finalBatchResults[1][1].score, 2.5)
    }
}
