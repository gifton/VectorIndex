import XCTest
import VectorCore
@testable import VectorIndex

final class HNSWKNNGraphTests: XCTestCase {
    // Deterministic PRNG (same convention as HNSWRecallTests)
    struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = 2862933555777941757 &* state &+ 3037000493
            return state
        }
        mutating func nextFloat() -> Float {
            Float(next() >> 11) / Float(1 << 53)
        }
        mutating func nextInRange(_ range: ClosedRange<Float>) -> Float {
            range.lowerBound + (range.upperBound - range.lowerBound) * nextFloat()
        }
    }

    /// Seeded Box-Muller over LCG (mirrors VectorCore's UMAP test fixture shape).
    struct Gaussian {
        var rng: LCG
        var spare: Float?
        init(seed: UInt64) { rng = LCG(state: seed) }
        mutating func next() -> Float {
            if let s = spare { spare = nil; return s }
            var u1 = rng.nextFloat()
            if u1 < 1e-12 { u1 = 1e-12 }
            let u2 = rng.nextFloat()
            let r = (-2 * Foundation.log(u1)).squareRoot()
            spare = r * sin(2 * .pi * u2)
            return r * cos(2 * .pi * u2)
        }
    }

    func generateDataset(count: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var rng = LCG(state: seed)
        var data: [[Float]] = []
        data.reserveCapacity(count)
        for _ in 0..<count {
            var v = (0..<dim).map { _ in rng.nextInRange(-1...1) }
            let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
            if norm > 0 { v = v.map { $0 / norm } }
            data.append(v)
        }
        return data
    }

    /// Two isotropic Gaussian blobs; points [half, 2*half) offset by `separation` on axis 0.
    func twoClusters(seed: UInt64, half: Int, dim: Int, separation: Float, scale: Float) -> [[Float]] {
        var g = Gaussian(seed: seed)
        return (0..<(2 * half)).map { i in
            var v = (0..<dim).map { _ in scale * g.next() }
            if i >= half { v[0] += separation }
            return v
        }
    }

    func buildIndex(
        data: [[Float]],
        metric: SupportedDistanceMetric = .euclidean,
        config: HNSWIndex.Configuration = .init()
    ) async throws -> HNSWIndex {
        let index = HNSWIndex(dimension: data[0].count, metric: metric, config: config)
        try await index.batchInsert(data.enumerated().map { (id: "id\($0.0)", vector: $0.1, metadata: nil) })
        return index
    }

    func assertContract(_ graph: KNNGraph, n: Int, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(graph.pointCount, n, file: file, line: line)
        XCTAssertEqual(graph.rowOffsets.count, n + 1, file: file, line: line)
        XCTAssertEqual(graph.rowOffsets[0], 0, file: file, line: line)
        XCTAssertEqual(graph.rowOffsets[n], graph.edgeCount, file: file, line: line)
        for i in 0..<n {
            XCTAssertLessThanOrEqual(graph.rowOffsets[i], graph.rowOffsets[i + 1], file: file, line: line)
            for e in graph.neighborRange(of: i) {
                let j = Int(graph.neighborIndices[e])
                XCTAssertTrue(j >= 0 && j < n, file: file, line: line)
                XCTAssertNotEqual(j, i, "self-loop at row \(i)", file: file, line: line)
                XCTAssertTrue(graph.distances[e].isFinite && graph.distances[e] >= 0, file: file, line: line)
            }
        }
    }

    func neighborSets(_ graph: KNNGraph) -> [Set<Int32>] {
        (0..<graph.pointCount).map { i in
            Set(graph.neighborRange(of: i).map { graph.neighborIndices[$0] })
        }
    }

    // (b) Contract compliance
    func testContractCompliance() async throws {
        let n = 500, k = 10
        let data = generateDataset(count: n, dim: 16, seed: 7)
        let index = try await buildIndex(data: data)
        let (graph, ids) = try await index.buildKNNGraph(k: k)
        assertContract(graph, n: n)
        XCTAssertEqual(ids, (0..<n).map { "id\($0)" }) // live points in insertion order
        for i in 0..<n {
            XCTAssertLessThanOrEqual(graph.neighborRange(of: i).count, k)
            XCTAssertGreaterThan(graph.neighborRange(of: i).count, 0, "empty row \(i)")
        }
    }

    // (a) Recall parity vs Core's exact brute-force reference
    func testRecallVsBruteForce() async throws {
        let n = 2000, k = 15
        let data = generateDataset(count: n, dim: 32, seed: 42)
        let index = try await buildIndex(data: data)
        let (graph, _) = try await index.buildKNNGraph(k: k)
        let reference = try KNNGraph.bruteForce(data.map { DynamicVector($0) }, neighbors: k)
        let approx = neighborSets(graph)
        let truth = neighborSets(reference)
        var totalRecall: Float = 0
        var maxDistErr: Float = 0
        for i in 0..<n {
            totalRecall += Float(approx[i].intersection(truth[i]).count) / Float(truth[i].count)
            // Distance parity on edges both builders found
            var refDist: [Int32: Float] = [:]
            for e in reference.neighborRange(of: i) { refDist[reference.neighborIndices[e]] = reference.distances[e] }
            for e in graph.neighborRange(of: i) {
                if let rd = refDist[graph.neighborIndices[e]] {
                    maxDistErr = max(maxDistErr, abs(rd - graph.distances[e]))
                }
            }
        }
        let avgRecall = totalRecall / Float(n)
        XCTAssertGreaterThanOrEqual(avgRecall, 0.9, "recall@\(k) = \(avgRecall)")
        XCTAssertLessThanOrEqual(maxDistErr, 1e-3, "distance mismatch vs brute force: \(maxDistErr)")
    }

    // (c) Cosine chord conversion ≡ euclidean on unit vectors
    func testCosineMatchesEuclideanOnUnitVectors() async throws {
        let n = 1000, k = 10
        let data = generateDataset(count: n, dim: 24, seed: 99) // unit-normalized
        let euc = try await buildIndex(data: data, metric: .euclidean)
        let cos = try await buildIndex(data: data, metric: .cosine)
        let (gE, _) = try await euc.buildKNNGraph(k: k)
        let (gC, _) = try await cos.buildKNNGraph(k: k)
        assertContract(gC, n: n)
        let setsE = neighborSets(gE)
        let setsC = neighborSets(gC)
        var jaccard: Float = 0
        var maxErr: Float = 0
        for i in 0..<n {
            let inter = setsE[i].intersection(setsC[i]).count
            let union = setsE[i].union(setsC[i]).count
            jaccard += union > 0 ? Float(inter) / Float(union) : 1
            var eDist: [Int32: Float] = [:]
            for e in gE.neighborRange(of: i) { eDist[gE.neighborIndices[e]] = gE.distances[e] }
            for e in gC.neighborRange(of: i) {
                if let ed = eDist[gC.neighborIndices[e]] {
                    maxErr = max(maxErr, abs(ed - gC.distances[e]))
                }
            }
        }
        XCTAssertGreaterThanOrEqual(jaccard / Float(n), 0.9)
        XCTAssertLessThanOrEqual(maxErr, 1e-3, "cosine chord vs euclidean mismatch: \(maxErr)")
    }

    // (d) Determinism: identical arrays across two builds
    func testDeterminism() async throws {
        let data = generateDataset(count: 800, dim: 16, seed: 123)
        let index = try await buildIndex(data: data)
        let (g1, ids1) = try await index.buildKNNGraph(k: 12)
        let (g2, ids2) = try await index.buildKNNGraph(k: 12)
        XCTAssertEqual(g1.rowOffsets, g2.rowOffsets)
        XCTAssertEqual(g1.neighborIndices, g2.neighborIndices)
        XCTAssertEqual(g1.distances, g2.distances)
        XCTAssertEqual(ids1, ids2)
    }

    // (e) dotProduct throws
    func testDotProductThrows() async throws {
        let data = generateDataset(count: 50, dim: 8, seed: 5)
        let index = try await buildIndex(data: data, metric: .dotProduct)
        do {
            _ = try await index.buildKNNGraph(k: 5)
            XCTFail("expected throw for dotProduct")
        } catch let error as VectorIndexError {
            XCTAssertEqual(error.kind, .invalidParameter)
        }
    }

    // (h) Parameter validation
    func testParameterValidation() async throws {
        let data = generateDataset(count: 20, dim: 8, seed: 6)
        let index = try await buildIndex(data: data)
        for badK in [0, 20, 100] {
            do {
                _ = try await index.buildKNNGraph(k: badK)
                XCTFail("expected throw for k=\(badK)")
            } catch let error as VectorIndexError {
                XCTAssertEqual(error.kind, .invalidRange)
            }
        }
        let empty = HNSWIndex(dimension: 8, metric: .euclidean)
        do {
            _ = try await empty.buildKNNGraph(k: 5)
            XCTFail("expected throw for empty index")
        } catch let error as VectorIndexError {
            XCTAssertEqual(error.kind, .emptyInput)
        }
    }

    // (g) Deletions: compacted rows, removed ids absent, neighbors still accurate.
    // NOTE: uses per-id remove(id:) — batchRemove has a pre-existing state-reset bug
    // (HNSWIndex.swift batchRemove), flagged in the PR as a separate ticket.
    func testDeletionsCompaction() async throws {
        let n = 500, k = 10
        let data = generateDataset(count: n, dim: 16, seed: 31)
        let index = try await buildIndex(data: data)
        var removed = Set<Int>()
        for i in stride(from: 0, to: n, by: 7) {
            try await index.remove(id: "id\(i)")
            removed.insert(i)
        }
        let liveOriginal = (0..<n).filter { !removed.contains($0) }
        let (graph, ids) = try await index.buildKNNGraph(k: k)
        assertContract(graph, n: liveOriginal.count)
        XCTAssertEqual(ids, liveOriginal.map { "id\($0)" })
        // Spot-check NN accuracy over the live subset
        let liveData = liveOriginal.map { data[$0] }
        let reference = try KNNGraph.bruteForce(liveData.map { DynamicVector($0) }, neighbors: k)
        let approx = neighborSets(graph)
        let truth = neighborSets(reference)
        var totalRecall: Float = 0
        for i in 0..<liveOriginal.count {
            totalRecall += Float(approx[i].intersection(truth[i]).count) / Float(truth[i].count)
        }
        XCTAssertGreaterThanOrEqual(totalRecall / Float(liveOriginal.count), 0.8)
    }

    // (f) End-to-end: our graph through Core's UMAP separates two clusters.
    // Mirrors VectorCore Tests/ComprehensiveTests/UMAPTests.swift (separationRatio > 2).
    //
    // Points are inserted in seeded-shuffled order: cluster-sequential insertion
    // triggers a pre-existing HNSW construction bug (naive closest-M reverse-edge
    // pruning disconnects well-separated clusters — see
    // testKnownIssue_SequentialClusterInsertDisconnectsGraph). Shuffled order is
    // also the representative ingestion pattern for real corpora.
    func testUMAPIntegrationTwoClusters() async throws {
        let half = 60
        let data = twoClusters(seed: 73, half: half, dim: 10, separation: 12, scale: 0.5)
        var rng = LCG(state: 2024)
        var order = Array(0..<data.count)
        for i in stride(from: order.count - 1, to: 0, by: -1) {
            order.swapAt(i, Int(rng.next() % UInt64(i + 1)))
        }
        let index = HNSWIndex(dimension: 10, metric: .euclidean)
        try await index.batchInsert(order.map { (id: "id\($0)", vector: data[$0], metadata: nil) })
        let (graph, ids) = try await index.buildKNNGraph(k: 10)
        XCTAssertEqual(ids, order.map { "id\($0)" })

        // Graph quality: the clusters are far apart relative to their radii, so the
        // produced kNN graph must contain no cross-cluster edges.
        let isA: [Bool] = order.map { $0 < half } // per graph row
        var crossEdges = 0
        for i in 0..<graph.pointCount {
            for e in graph.neighborRange(of: i) where isA[i] != isA[Int(graph.neighborIndices[e])] {
                crossEdges += 1
            }
        }
        XCTAssertEqual(crossEdges, 0, "cross-cluster edges in kNN graph")

        let result = try Operations.umap(graph: graph, dimensions: 2, config: UMAPConfig(neighbors: 10))
        XCTAssertEqual(result.pointCount, 2 * half)
        XCTAssertEqual(result.dimension, 2)
        XCTAssertTrue(result.coordinates.allSatisfy { $0.isFinite })

        // Inter-centroid distance over max mean within-cluster radius, with
        // cluster membership tracked per graph row (insertion order ≠ data order).
        let m = result.dimension
        var cA = [Float](repeating: 0, count: m)
        var cB = [Float](repeating: 0, count: m)
        var nA = 0, nB = 0
        for i in 0..<result.pointCount {
            for c in 0..<m {
                if isA[i] { cA[c] += result.coordinates[i * m + c] } else { cB[c] += result.coordinates[i * m + c] }
            }
            if isA[i] { nA += 1 } else { nB += 1 }
        }
        XCTAssertEqual(nA, half)
        XCTAssertEqual(nB, half)
        for c in 0..<m { cA[c] /= Float(nA); cB[c] /= Float(nB) }
        func meanRadius(_ member: Bool, _ centroid: [Float]) -> Float {
            var total: Float = 0
            var count = 0
            for i in 0..<result.pointCount where isA[i] == member {
                var sq: Float = 0
                for c in 0..<m {
                    let d = result.coordinates[i * m + c] - centroid[c]
                    sq += d * d
                }
                total += sq.squareRoot()
                count += 1
            }
            return total / Float(count)
        }
        var between: Float = 0
        for c in 0..<m {
            let d = cA[c] - cB[c]
            between += d * d
        }
        let ratio = between.squareRoot() / max(meanRadius(true, cA), meanRadius(false, cB), .leastNormalMagnitude)
        XCTAssertGreaterThan(ratio, 2, "separation ratio \(ratio)")
    }

    // Known issue (pre-existing, NOT a buildKNNGraph bug — flagged for a separate fix):
    // inserting two well-separated clusters sequentially disconnects the navigable
    // graph. `pruneNeighbors` (HNSWIndex.swift) shrinks overflowed reverse-edge lists
    // via hnsw_prune_neighbors_f32_swift, which keeps the M closest with NO diversity
    // heuristic (HNSWNeighborSelection.swift), so the long A↔B bridge edges are
    // discarded from both sides and the first-inserted cluster becomes unreachable
    // from the entry point — public search() can't even find an indexed point at
    // distance 0, regardless of ef. When the prune kernel adopts the same diversity
    // heuristic as hnsw_select_neighbors_f32_swift, this XCTExpectFailure flips to a
    // hard failure: delete this test and re-enable sequential fixtures.
    func testKnownIssue_SequentialClusterInsertDisconnectsGraph() async throws {
        let half = 60
        let data = twoClusters(seed: 73, half: half, dim: 10, separation: 12, scale: 0.5)
        let index = try await buildIndex(data: data) // sequential: all A, then all B
        let hits = try await index.search(query: data[5], k: 1, filter: nil)
        XCTExpectFailure("pre-existing HNSW prune bug: first-inserted cluster unreachable after cluster-sequential insertion") {
            XCTAssertEqual(hits.first?.id, "id5", "indexed point should be its own nearest neighbor")
        }
    }
}
