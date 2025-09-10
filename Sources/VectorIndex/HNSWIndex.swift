//
//  HNSWIndex.swift
//  VectorIndex
//
//  Minimal CPU-only HNSW implementation. Keeps data structures simple and
//  focuses on correctness and a clean seam for later optimization.
//

import Foundation
import VectorCore

public actor HNSWIndex: VectorIndexProtocol, AccelerableIndex {
    
    public struct Configuration: Sendable {
        public let m: Int           // max connections per node (per layer)
        public let efConstruction: Int
        public let efSearch: Int
        public init(m: Int = 16, efConstruction: Int = 200, efSearch: Int = 64) {
            self.m = m
            self.efConstruction = efConstruction
            self.efSearch = efSearch
        }
    }

    // MARK: - Public API
    public let dimension: Int
    public let metric: SupportedDistanceMetric
    public let config: Configuration
    public var count: Int { activeCount }

    public init(dimension: Int, metric: SupportedDistanceMetric = .euclidean, config: Configuration = .init()) {
        self.dimension = dimension
        self.metric = metric
        self.config = config
    }

    // Protocol-required initializer (delegates to designated one)
    public init(dimension: Int, metric: SupportedDistanceMetric) {
        self.dimension = dimension
        self.metric = metric
        self.config = .init()
    }

    public func insert(id: VectorID, vector: [Float], metadata: [String : String]?) async throws {
        try checkVector(vector)
        try await internalInsert(id: id, vector: vector, metadata: metadata)
    }

    public func remove(id: VectorID) async throws {
        if let idx = idToIndex[id] {
            nodes[idx].isDeleted = true
            idToIndex.removeValue(forKey: id)
            // Detach from neighbors
            let lvl = nodes[idx].level
            for l in 0...lvl {
                for n in nodes[idx].neighbors[l] {
                    removeNeighbor(n, idx, level: l)
                }
                nodes[idx].neighbors[l].removeAll(keepingCapacity: false)
            }
            if entryPoint == idx { entryPoint = findAnyActiveIndex() }
            activeCount -= 1
        }
    }

    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String : String]?)]) async throws {
        for item in items { try await insert(id: item.id, vector: item.vector, metadata: item.metadata) }
    }

    public func optimize() async throws {
        // Minimal: no-op; future: rebuild/prune
    }

    public func search(query: [Float], k: Int, filter: (@Sendable ([String : String]?) -> Bool)?) async throws -> [SearchResult] {
        guard k > 0 else { return [] }
        try checkVector(query)
        guard let ep = entryPoint else { return [] }

        // Greedy descent from top layer
        var cur = ep
        if maxLevel > 0 {
            for l in stride(from: maxLevel, through: 1, by: -1) {
                cur = greedySearchLayer(query, enter: cur, level: l)
            }
        }

        // efSearch exploration in layer 0
        let ef = max(config.efSearch, k)
        let resultIdxs = searchLayer(query, enter: cur, ef: ef, level: 0)

        var results: [SearchResult] = []
        results.reserveCapacity(min(k, resultIdxs.count))
        for idx in resultIdxs.prefix(k) {
            let node = nodes[idx]
            if let filter = filter, !filter(node.metadata) { continue }
            let d = distance(query, node.vector, metric: metric)
            results.append(SearchResult(id: node.id, score: d))
        }
        return results
    }

    public func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String : String]?) -> Bool)?) async throws -> [[SearchResult]] {
        var out: [[SearchResult]] = []
        out.reserveCapacity(queries.count)
        for q in queries { out.append(try await search(query: q, k: k, filter: filter)) }
        return out
    }

    public func clear() async {
        nodes.removeAll(keepingCapacity: false)
        idToIndex.removeAll(keepingCapacity: false)
        entryPoint = nil
        maxLevel = 0
        activeCount = 0
    }

    public func contains(id: VectorID) async -> Bool {
        if let i = idToIndex[id] { return i >= 0 && i < nodes.count && !nodes[i].isDeleted }
        return false
    }

    public func update(id: VectorID, vector: [Float]?, metadata: [String : String]?) async throws -> Bool {
        guard let idx = idToIndex[id] else { return false }
        var newVec = nodes[idx].vector
        var newMeta = nodes[idx].metadata
        if let v = vector {
            try checkVector(v)
            newVec = v
        }
        if let m = metadata { newMeta = m }
        try await remove(id: id)
        try await internalInsert(id: id, vector: newVec, metadata: newMeta)
        return true
    }

    public func batchRemove(_ ids: [VectorID]) async throws {
        for id in ids { try await remove(id: id) }
        entryPoint = nil
        maxLevel = 0
        activeCount = 0
    }

    public func statistics() async -> IndexStats {
        // Average degree at level 0 as a quick proxy
        var totalDeg = 0
        var nodeCount = 0
        for n in nodes where !n.isDeleted {
            if !n.neighbors.isEmpty {
                totalDeg += n.neighbors[0].count
                nodeCount += 1
            }
        }
        let avgDeg = nodeCount > 0 ? Float(totalDeg) / Float(nodeCount) : 0
        return IndexStats(
            indexType: "HNSW",
            vectorCount: activeCount,
            dimension: dimension,
            metric: metric,
            details: [
                "maxLevel": String(maxLevel),
                "avgDegreeL0": String(format: "%.2f", avgDeg)
            ]
        )
    }

    // MARK: - Internal Structures
    private struct Node {
        let id: VectorID
        var vector: [Float]
        var metadata: [String:String]?
        let level: Int
        var neighbors: [[Int]] // per level neighbor indices
        var isDeleted: Bool
    }

    private var nodes: [Node] = []
    private var idToIndex: [VectorID:Int] = [:]
    private var entryPoint: Int?
    private var maxLevel: Int = 0
    private var activeCount: Int = 0

    // MARK: - Insertion
    private func internalInsert(id: VectorID, vector: [Float], metadata: [String:String]?) async throws {
        if idToIndex[id] != nil { // replace existing vector: simple remove then reinsert
            try await remove(id: id)
        }
        let level = randomLevel()
        let newIndex = nodes.count
        let node = Node(id: id, vector: vector, metadata: metadata, level: level, neighbors: Array(repeating: [], count: level+1), isDeleted: false)
        nodes.append(node)
        idToIndex[id] = newIndex
        activeCount += 1

        if let oldEP = entryPoint {
            var cur = oldEP
            // Descend from current max layer down to the new node's top layer + 1
            if maxLevel > level {
                for l in stride(from: maxLevel, to: level, by: -1) {
                    cur = greedySearchLayer(vector, enter: cur, level: l)
                }
            }

            // For each level down to 0: connect to neighbors in that layer
            let ef = max(config.efConstruction, config.m)
            for l in stride(from: min(level, maxLevel), through: 0, by: -1) {
                let candidates = searchLayer(vector, enter: cur, ef: ef, level: l)
                // Only neighbors that exist at this layer
                let filtered = candidates.filter { nodes[$0].level >= l }
                let selected = selectNeighbors(for: vector, among: filtered, level: l, maxM: config.m)
                connect(newIndex, with: selected, level: l)
                // Update cur to closest among selected for next lower layer
                if let best = selected.min(by: { distance(vector, nodes[$0].vector, metric: metric) < distance(vector, nodes[$1].vector, metric: metric) }) {
                    cur = best
                }
            }

            // If new node has the highest level, make it the new entry point
            if level > maxLevel {
                maxLevel = level
                entryPoint = newIndex
            }
        } else {
            // First node
            entryPoint = newIndex
            maxLevel = level
        }
    }

    // Connect bidirectionally and enforce M cap on neighbor lists
    private func connect(_ a: Int, with neighbors: [Int], level: Int) {
        // Add neighbors to 'a' (deduplicated)
        var newList = nodes[a].neighbors[level]
        for n in neighbors where n != a {
            if !newList.contains(n) { newList.append(n) }
        }
        nodes[a].neighbors[level] = newList
        pruneNeighbors(of: a, level: level)
        // Add 'a' to each neighbor
        for n in neighbors {
            guard nodes[n].level >= level else { continue }
            if !nodes[n].neighbors[level].contains(a) {
                nodes[n].neighbors[level].append(a)
                pruneNeighbors(of: n, level: level)
            }
        }
    }

    private func pruneNeighbors(of idx: Int, level: Int) {
        var list = nodes[idx].neighbors[level]
        // Dedupe first
        var seen = Set<Int>()
        list = list.filter { seen.insert($0).inserted }
        if list.count <= config.m { nodes[idx].neighbors[level] = list; return }
        // Keep closest M by distance to node idx
        list.sort(by: { distance(nodes[idx].vector, nodes[$0].vector, metric: metric) < distance(nodes[idx].vector, nodes[$1].vector, metric: metric) })
        list = Array(list.prefix(config.m))
        nodes[idx].neighbors[level] = list
    }

    private func removeNeighbor(_ idx: Int, _ neighbor: Int, level: Int) {
        var list = nodes[idx].neighbors[level]
        if let pos = list.firstIndex(of: neighbor) {
            list.remove(at: pos)
            nodes[idx].neighbors[level] = list
        }
    }

    // MARK: - Layer Search
    private func greedySearchLayer(_ query: [Float], enter: Int, level: Int) -> Int {
        var cur = enter
        var curDist = distance(query, nodes[cur].vector, metric: metric)
        var changed = true
        while changed {
            changed = false
            for n in nodes[cur].neighbors[safe: level] ?? [] {
                let d = distance(query, nodes[n].vector, metric: metric)
                if d < curDist {
                    curDist = d
                    cur = n
                    changed = true
                }
            }
        }
        return cur
    }

    private func searchLayer(_ query: [Float], enter: Int, ef: Int, level: Int) -> [Int] {
        // Candidate list (min by distance)
        var candidates: [(Int, Float)] = []
        // Result set (kept sorted ascending by distance)
        var result: [(Int, Float)] = []
        var visited = Set<Int>()

        let enterDist = distance(query, nodes[enter].vector, metric: metric)
        candidates.append((enter, enterDist))
        result.append((enter, enterDist))
        visited.insert(enter)

        while !candidates.isEmpty {
            // Pop best candidate
            var bestIdx = 0
            for i in 1..<candidates.count { if candidates[i].1 < candidates[bestIdx].1 { bestIdx = i } }
            let (cand, candDist) = candidates.remove(at: bestIdx)

            // If this candidate is worse than the worst in result and result is full, we can stop
            if result.count >= ef, let worst = result.last, candDist > worst.1 { break }

            for n in nodes[cand].neighbors[safe: level] ?? [] {
                if visited.insert(n).inserted, !nodes[n].isDeleted {
                    let d = distance(query, nodes[n].vector, metric: metric)
                    // Maintain result set up to ef
                    if result.count < ef || d < (result.last?.1 ?? .infinity) {
                        // Insert into candidates
                        candidates.append((n, d))
                        // Insert sorted into result
                        insertSorted(&result, (n, d))
                        if result.count > ef { _ = result.popLast() }
                    }
                }
            }
        }
        return result.map { $0.0 }
    }

    private func insertSorted(_ array: inout [(Int, Float)], _ element: (Int, Float)) {
        // insertion into ascending array by distance
        let pos = array.firstIndex(where: { $0.1 > element.1 }) ?? array.count
        array.insert(element, at: pos)
    }

    // MARK: - Utilities
    private func randomLevel() -> Int {
        // Geometric distribution with tail decay ~ 1/log(M)
        let m = max(2, config.m)
        let logm = log(Float(m))
        let r = Float.random(in: 0..<1)
        let lvl = Int(-log(r) * (1.0 / logm))
        return min(lvl, 16) // cap to avoid excessive levels
    }

    private func checkVector(_ v: [Float]) throws {
        if v.count != dimension { throw VectorError.dimensionMismatch(expected: dimension, actual: v.count) }
    }
    
    private func findAnyActiveIndex() -> Int? {
        for (i, n) in nodes.enumerated() {
            if !n.isDeleted { return i }
        }
        
        return nil
    }
}

// MARK: - Safe index access helper
private extension Array where Element == [Int] {
    subscript(safe idx: Int) -> [Int]? {
        guard idx >= 0 && idx < count else { return nil }
        return self[idx]
    }
}

// MARK: - Persistence & Compaction
extension HNSWIndex {
    public func save(to url: URL) async throws {
        // Persist as flat records; graph is rebuilt on load
        var recs: [PersistedRecord] = []
        recs.reserveCapacity(activeCount)
        for n in nodes where !n.isDeleted {
            recs.append(PersistedRecord(id: n.id, vector: n.vector, metadata: n.metadata))
        }
        let payload = PersistedIndex(
            type: "HNSW",
            version: 1,
            dimension: dimension,
            metric: metric.rawValue,
            records: recs
        )
        let data = try JSONEncoder().encode(payload)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) async throws -> HNSWIndex {
        let data = try Data(contentsOf: url)
        let payload = try JSONDecoder().decode(PersistedIndex.self, from: data)
        guard payload.type == "HNSW" else { throw VectorError(.invalidData) }
        let idx = HNSWIndex(dimension: payload.dimension, metric: .from(raw: payload.metric))
        try await idx.batchInsert(payload.records.map { ($0.id, $0.vector, $0.metadata) })
        return idx
    }

    public func compact() async throws {
        // Remove deleted nodes and reindex neighbor lists by remapping indices
        var newNodes: [Node] = []
        newNodes.reserveCapacity(activeCount)
        var remap: [Int:Int] = [:] // old -> new
        for (oldIdx, n) in nodes.enumerated() {
            if !n.isDeleted { remap[oldIdx] = newNodes.count; newNodes.append(n) }
        }
        // Remap neighbors per level
        for i in newNodes.indices {
            let lvl = newNodes[i].level
            var lvlLists: [[Int]] = []
            lvlLists.reserveCapacity(lvl+1)
            for l in 0...lvl {
                var mapped: [Int] = []
                for oldN in newNodes[i].neighbors[safe: l] ?? [] {
                    if let nn = remap[oldN] { mapped.append(nn) }
                }
                // prune to M
                if mapped.count > config.m {
                    mapped.sort { distance(newNodes[i].vector, newNodes[$0].vector, metric: metric) < distance(newNodes[i].vector, newNodes[$1].vector, metric: metric) }
                    mapped = Array(mapped.prefix(config.m))
                }
                lvlLists.append(mapped)
            }
            newNodes[i].neighbors = lvlLists
        }
        nodes = newNodes
        // Rebuild idToIndex
        idToIndex.removeAll(keepingCapacity: false)
        activeCount = 0
        var newMax = 0
        for (i, n) in nodes.enumerated() {
            idToIndex[n.id] = i
            if !n.isDeleted { activeCount += 1 }
            if n.level > newMax { newMax = n.level }
        }
        maxLevel = newMax
        // Recompute entry point as any node with max level or first node
        if let idx = nodes.firstIndex(where: { $0.level == maxLevel && !$0.isDeleted }) {
            entryPoint = idx
        } else {
            entryPoint = nodes.isEmpty ? nil : 0
        }
    }
}

// MARK: - Neighbor selection heuristic
private extension HNSWIndex {
    // Select up to maxM diverse neighbors among candidate node indices for a given vector at level.
    // Implements a simple diversity heuristic from HNSW: candidates are considered in increasing
    // distance order; a candidate is selected if it is closer to the new point than to any already
    // selected neighbor, promoting angular diversity.
    func selectNeighbors(for vec: [Float], among candidates: [Int], level: Int, maxM: Int) -> [Int] {
        // Sort candidates by distance to vec
        var sorted: [(Int, Float)] = candidates.map { ($0, distance(vec, nodes[$0].vector, metric: metric)) }
        sorted.sort { $0.1 < $1.1 }
        var selected: [Int] = []
        selected.reserveCapacity(min(maxM, sorted.count))
        for (cand, _) in sorted {
            var good = true
            for s in selected {
                // If candidate is much closer to an already selected neighbor than to the new point,
                // skip it (encourage spread). Criterion: d(cand, s) < d(cand, new)
                let d_cs = distance(nodes[cand].vector, nodes[s].vector, metric: metric)
                let d_cx = distance(nodes[cand].vector, vec, metric: metric)
                if d_cs < d_cx { good = false; break }
            }
            if good { selected.append(cand) }
            if selected.count >= maxM { break }
        }
        // Fallback: if too few selected, fill with nearest remaining
        if selected.count < maxM {
            for (cand, _) in sorted where !selected.contains(cand) {
                selected.append(cand)
                if selected.count >= maxM { break }
            }
        }
        return selected
    }
}

// MARK: - AccelerableIndex Implementation
extension HNSWIndex {
    public func getCandidates(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> AccelerationCandidates {
        guard k > 0 else { 
            return AccelerationCandidates(
                ids: [],
                vectorStorage: ContiguousArray<Float>(),
                vectorCount: 0,
                dimension: dimension,
                metadata: []
            )
        }
        try checkVector(query)
        guard let ep = entryPoint else { 
            return AccelerationCandidates(
                ids: [],
                vectorStorage: ContiguousArray<Float>(),
                vectorCount: 0,
                dimension: dimension,
                metadata: []
            )
        }
        
        // Greedy descent from top layer
        var cur = ep
        if maxLevel > 0 {
            for l in stride(from: maxLevel, through: 1, by: -1) {
                cur = greedySearchLayer(query, enter: cur, level: l)
            }
        }
        
        // efSearch exploration in layer 0
        let ef = max(config.efSearch, k)
        let candidateIndices = searchLayer(query, enter: cur, ef: ef, level: 0)
        
        // Pre-allocate arrays based on actual candidate count
        var ids: [VectorID] = []
        ids.reserveCapacity(candidateIndices.count)
        var metadata: [[String: String]?] = []
        metadata.reserveCapacity(candidateIndices.count)
        var vectorStorage = ContiguousArray<Float>()
        vectorStorage.reserveCapacity(candidateIndices.count * dimension)
        
        // Collect candidate data in single pass
        for idx in candidateIndices {
            let node = nodes[idx]
            if !node.isDeleted {
                ids.append(node.id)
                metadata.append(node.metadata)
                // Append directly to contiguous storage (avoids intermediate array)
                vectorStorage.append(contentsOf: node.vector)
            }
        }
        
        return AccelerationCandidates(
            ids: ids,
            vectorStorage: vectorStorage,
            vectorCount: ids.count,
            dimension: dimension,
            metadata: metadata
        )
    }
    
    public func getBatchCandidates(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [AccelerationCandidates] {
        var results: [AccelerationCandidates] = []
        for query in queries {
            let candidates = try await getCandidates(query: query, k: k, filter: filter)
            results.append(candidates)
        }
        return results
    }
    
    public func getIndexStructure() async -> IndexStructure {
        // Create a mapping from original indices to compacted indices
        var oldToNewIndex: [Int: Int] = [:]
        var newToOldIndex: [Int] = []
        var compactedNodeLevels: [Int] = []
        
        // Build index mappings for active nodes only
        for (oldIdx, node) in nodes.enumerated() {
            if !node.isDeleted {
                let newIdx = newToOldIndex.count
                oldToNewIndex[oldIdx] = newIdx
                newToOldIndex.append(oldIdx)
                compactedNodeLevels.append(node.level)
            }
        }
        
        // Build adjacency lists per layer using compacted indices
        var layerGraphs: [[Set<Int>]] = Array(repeating: [], count: maxLevel + 1)
        for level in 0...maxLevel {
            layerGraphs[level] = Array(repeating: Set<Int>(), count: newToOldIndex.count)
        }
        
        for (newIdx, oldIdx) in newToOldIndex.enumerated() {
            let node = nodes[oldIdx]
            for level in 0...node.level {
                if level < layerGraphs.count {
                    // Convert neighbor indices from old to new
                    var compactedNeighbors = Set<Int>()
                    for oldNeighborIdx in node.neighbors[safe: level] ?? [] {
                        if let newNeighborIdx = oldToNewIndex[oldNeighborIdx] {
                            compactedNeighbors.insert(newNeighborIdx)
                        }
                    }
                    layerGraphs[level][newIdx] = compactedNeighbors
                }
            }
        }
        
        // Convert entry point to compacted index
        let compactedEntryPoint = entryPoint.flatMap { oldToNewIndex[$0] }
        
        let structure = HNSWStructure(
            entryPoint: compactedEntryPoint,
            maxLevel: maxLevel,
            layerGraphs: layerGraphs,
            nodeLevels: compactedNodeLevels
        )
        
        return .hnsw(structure)
    }
    
    public func finalizeResults(
        candidates: AccelerationCandidates,
        results: AcceleratedResults,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async -> [SearchResult] {
        var finalResults: [SearchResult] = []
        
        for (idx, distance) in zip(results.indices, results.distances) {
            guard idx < candidates.ids.count else { continue }
            
            let metadata = candidates.metadata[idx]
            if let filter = filter, !filter(metadata) { continue }
            
            finalResults.append(SearchResult(
                id: candidates.ids[idx],
                score: distance
            ))
        }
        
        return finalResults
    }
}
