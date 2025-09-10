//
//  HNSWAlignmentTest.swift
//  VectorIndexTests
//
//  Tests for HNSW index alignment bug fix in getIndexStructure
//

import XCTest
@testable import VectorIndex
import VectorCore

final class HNSWAlignmentTest: XCTestCase {
    
    func testIndexStructureWithDeletedNodes() async throws {
        let index = HNSWIndex(dimension: 3, metric: .euclidean)
        
        // Insert multiple vectors
        try await index.insert(id: "vec1", vector: [1.0, 2.0, 3.0], metadata: nil)
        try await index.insert(id: "vec2", vector: [4.0, 5.0, 6.0], metadata: nil)
        try await index.insert(id: "vec3", vector: [7.0, 8.0, 9.0], metadata: nil)
        try await index.insert(id: "vec4", vector: [2.0, 3.0, 4.0], metadata: nil)
        try await index.insert(id: "vec5", vector: [5.0, 6.0, 7.0], metadata: nil)
        
        // Delete some nodes to create gaps
        try await index.remove(id: "vec2")
        try await index.remove(id: "vec4")
        
        // Get index structure - this should not crash or have alignment issues
        let structure = await index.getIndexStructure()
        
        if case .hnsw(let hnswStructure) = structure {
            // Should have 3 active nodes (vec1, vec3, vec5)
            XCTAssertEqual(hnswStructure.nodeLevels.count, 3, "Should have 3 active nodes")
            
            // Check that all layer graphs have consistent sizing
            for (level, layerGraph) in hnswStructure.layerGraphs.enumerated() {
                XCTAssertEqual(layerGraph.count, 3, "Layer \(level) should have 3 nodes")
                
                // Verify all neighbor indices are within bounds
                for (nodeIdx, neighbors) in layerGraph.enumerated() {
                    for neighborIdx in neighbors {
                        XCTAssertLessThan(neighborIdx, 3, 
                            "Neighbor index \(neighborIdx) out of bounds at node \(nodeIdx) level \(level)")
                        XCTAssertGreaterThanOrEqual(neighborIdx, 0,
                            "Neighbor index \(neighborIdx) is negative at node \(nodeIdx) level \(level)")
                    }
                }
            }
            
            // Entry point should be remapped to compacted index
            if let entryPoint = hnswStructure.entryPoint {
                XCTAssertLessThan(entryPoint, 3, "Entry point should be within compacted range")
                XCTAssertGreaterThanOrEqual(entryPoint, 0, "Entry point should be non-negative")
            }
        } else {
            XCTFail("Expected HNSW structure")
        }
    }
    
    func testCandidatesAlignmentWithDeletedNodes() async throws {
        let index = HNSWIndex(dimension: 3, metric: .euclidean)
        
        // Insert vectors
        try await index.insert(id: "vec1", vector: [1.0, 0.0, 0.0], metadata: ["idx": "0"])
        try await index.insert(id: "vec2", vector: [0.0, 1.0, 0.0], metadata: ["idx": "1"])
        try await index.insert(id: "vec3", vector: [0.0, 0.0, 1.0], metadata: ["idx": "2"])
        try await index.insert(id: "vec4", vector: [1.0, 1.0, 0.0], metadata: ["idx": "3"])
        try await index.insert(id: "vec5", vector: [0.0, 1.0, 1.0], metadata: ["idx": "4"])
        
        // Delete middle nodes
        try await index.remove(id: "vec2")
        try await index.remove(id: "vec3")
        
        // Get candidates - should only return active nodes
        let query: [Float] = [0.5, 0.5, 0.5]
        let candidates = try await index.getCandidates(query: query, k: 5, filter: nil)
        
        // Should have exactly 3 candidates (active nodes only)
        XCTAssertEqual(candidates.ids.count, 3, "Should have 3 active candidates")
        XCTAssertEqual(candidates.vectors.count, 3, "Should have 3 candidate vectors")
        XCTAssertEqual(candidates.metadata.count, 3, "Should have 3 metadata entries")
        
        // Verify returned candidates are the active ones
        XCTAssertTrue(candidates.ids.contains("vec1"))
        XCTAssertTrue(candidates.ids.contains("vec4"))
        XCTAssertTrue(candidates.ids.contains("vec5"))
        XCTAssertFalse(candidates.ids.contains("vec2"))
        XCTAssertFalse(candidates.ids.contains("vec3"))
    }
    
    func testStructureConsistencyAfterCompaction() async throws {
        let index = HNSWIndex(dimension: 3, metric: .euclidean)
        
        // Create a more complex graph
        for i in 0..<20 {
            let vec: [Float] = [Float(i), Float(i*2), Float(i*3)]
            try await index.insert(id: "vec\(i)", vector: vec, metadata: nil)
        }
        
        // Delete every other node
        for i in stride(from: 1, to: 20, by: 2) {
            try await index.remove(id: "vec\(i)")
        }
        
        // Get structure before compaction
        let structureBefore = await index.getIndexStructure()
        
        // Compact the index
        try await index.compact()
        
        // Get structure after compaction
        let structureAfter = await index.getIndexStructure()
        
        // Both should be valid HNSW structures
        guard case .hnsw(let before) = structureBefore,
              case .hnsw(let after) = structureAfter else {
            XCTFail("Expected HNSW structures")
            return
        }
        
        // After compaction, structure should be clean with no gaps
        XCTAssertEqual(after.nodeLevels.count, 10, "Should have 10 nodes after compaction")
        
        // All indices in neighbor lists should be valid
        for layerGraph in after.layerGraphs {
            for neighbors in layerGraph {
                for neighbor in neighbors {
                    XCTAssertLessThan(neighbor, 10, "Neighbor index should be within bounds")
                }
            }
        }
    }
}