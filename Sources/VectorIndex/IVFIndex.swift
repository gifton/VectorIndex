//
//  IVFIndex.swift
//  VectorIndex
//
//  CPU-first skeleton for IVF. For now, this delegates to FlatIndex while
//  keeping configuration/state. Later, optimize() will build coarse
//  centroids and assign lists, and search() will probe selected lists.
//

import Foundation
import VectorCore

public actor IVFIndex: VectorIndexProtocol, AccelerableIndex {
    
    public struct Configuration: Sendable {
        public let nlist: Int          // number of coarse centroids
        public let nprobe: Int         // number of lists to probe at search
        public init(nlist: Int = 256, nprobe: Int = 8) {
            self.nlist = nlist
            self.nprobe = nprobe
        }
    }

    public let dimension: Int
    public let metric: SupportedDistanceMetric
    public let config: Configuration

    // Placeholder for centroids and inverted lists
    private var centroids: [[Float]] = []
    private var lists: [[VectorID]] = []
    private var idToListIndex: [VectorID: Int] = [:]

    // Local storage for now (CPU baseline)
    private var store: [VectorID: ([Float], [String:String]?)] = [:]

    public var count: Int { store.count }

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
        guard vector.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.count)
        }
        // If replacing existing, detach from previous list mapping
        if let _ = store[id], let oldList = idToListIndex[id] {
            _ = removeID(id, fromList: oldList)
            idToListIndex.removeValue(forKey: id)
        }
        store[id] = (vector, metadata)
        // If centroids exist, assign to nearest list
        if let ci = nearestCentroidIndex(for: vector), lists.indices.contains(ci) {
            lists[ci].append(id)
            idToListIndex[id] = ci
        }
    }

    public func remove(id: VectorID) async throws {
        store.removeValue(forKey: id)
        if let li = idToListIndex.removeValue(forKey: id) {
            _ = removeID(id, fromList: li)
        } else {
            // best-effort removal in case mapping missing
            for i in lists.indices { if removeID(id, fromList: i) { break } }
        }
    }

    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String : String]?)]) async throws {
        for item in items {
            try await insert(id: item.id, vector: item.vector, metadata: item.metadata)
        }
        // TODO: bulk assignment
    }

    public func optimize() async throws {
        // Build centroids with CPU Lloyd's KMeans and assign points to lists
        // Use k = min(nlist, store.count)
        guard !store.isEmpty else {
            centroids.removeAll(); lists.removeAll(); return
        }
        let k = max(1, min(config.nlist, store.count))
        // Initialize centroids using deterministic k‑means++ (farthest‑point) seeding
        let initialCentroids = kmeansPlusPlusInitRandom(k: k, seed: 42)
        centroids = try await kmeans(centroids: initialCentroids, maxIterations: 20)
        // Build inverted lists
        lists = Array(repeating: [], count: centroids.count)
        idToListIndex.removeAll(keepingCapacity: false)
        for (id, (vec, _)) in store {
            if let ci = nearestCentroidIndex(for: vec), lists.indices.contains(ci) {
                lists[ci].append(id)
                idToListIndex[id] = ci
            }
        }
    }

    // MARK: - KMeans scaffolding (to be implemented)
    public func optimizeKMeans(maxIterations: Int = 15) async throws {
        guard !store.isEmpty else { centroids.removeAll(); lists.removeAll(); return }
        let k = max(1, min(config.nlist, store.count))
        let initC = kmeansPlusPlusInitRandom(k: k, seed: 42)
        centroids = try await kmeans(centroids: initC, maxIterations: maxIterations)
        lists = Array(repeating: [], count: centroids.count)
        for (id, (vec, _)) in store {
            if let ci = nearestCentroidIndex(for: vec) { lists[ci].append(id) }
        }
    }

    // Assign a single vector to nearest centroid (to be implemented)
    private func nearestCentroidIndex(for vector: [Float]) -> Int? {
        guard !centroids.isEmpty else { return nil }
        var best = -1
        var bestD = Float.infinity
        for (i, c) in centroids.enumerated() {
            let d = distance(vector, c, metric: metric)
            if d < bestD { bestD = d; best = i }
        }
        return best >= 0 ? best : nil
    }

    // MARK: - Lloyd's KMeans (CPU)
    private func kmeans(centroids initial: [[Float]], maxIterations: Int) async throws -> [[Float]] {
        precondition(!initial.isEmpty)
        var cents = initial
        let k = cents.count
        let d = dimension
        // Prepare arrays for sums and counts
        var changed = true
        var iter = 0
        var assignments: [String:Int] = [:] // id -> centroid idx
        while changed && iter < maxIterations {
            iter += 1
            changed = false
            var sums = Array(repeating: Array(repeating: Float(0), count: d), count: k)
            var counts = Array(repeating: 0, count: k)

            // Assignment step
            for (id, (vec, _)) in store {
                var best = 0
                var bestD = Float.infinity
                for (i, c) in cents.enumerated() {
                    let dist = distance(vec, c, metric: metric)
                    if dist < bestD { bestD = dist; best = i }
                }
                if assignments[id] != best { assignments[id] = best; changed = true }
                // Accumulate
                var sum = sums[best]
                for j in 0..<d { sum[j] += vec[j] }
                sums[best] = sum
                counts[best] += 1
            }

            // Update step
            for i in 0..<k {
                if counts[i] > 0 {
                    var newc = cents[i]
                    let inv = 1.0 / Float(counts[i])
                    for j in 0..<d { newc[j] = sums[i][j] * inv }
                    cents[i] = newc
                } else {
                    // Reseed empty centroid with the point having max error (farthest from its nearest centroid)
                    var bestVec: [Float]? = nil
                    var bestErr: Float = -Float.infinity
                    for (_, (vec, _)) in store {
                        var minD = Float.infinity
                        for c in cents { let dd = distance(vec, c, metric: metric); if dd < minD { minD = dd } }
                        if minD > bestErr { bestErr = minD; bestVec = vec }
                    }
                    if let v = bestVec { cents[i] = v }
                }
            }
        }
        return cents
    }

    // Seeded k‑means++ (D^2) sampling seeding
    private func kmeansPlusPlusInitRandom(k: Int, seed: UInt64) -> [[Float]] {
        precondition(!store.isEmpty)
        let d = dimension
        var cents: [[Float]] = []
        cents.reserveCapacity(k)
        let items: [[Float]] = store.map { $0.value.0 }
        struct RNG { var s: UInt64; mutating func next() -> UInt64 { s = 2862933555777941757 &* s &+ 3037000493; return s }; mutating func uniform() -> Float { Float(next() >> 11) / Float(1 << 53) } }
        var rng = RNG(s: seed == 0 ? 1 : seed)
        let firstIdx = min(Int(rng.uniform() * Float(items.count)), max(0, items.count-1))
        cents.append(items[firstIdx])
        while cents.count < k && cents.count < items.count {
            var d2 = Array(repeating: Float(0), count: items.count)
            var total: Float = 0
            for i in 0..<items.count {
                var minD = Float.infinity
                for c in cents { let dist = distance(items[i], c, metric: metric); if dist < minD { minD = dist } }
                let val = minD * minD
                d2[i] = val
                total += val
            }
            if total == 0 { break }
            let r = rng.uniform() * total
            var csum: Float = 0
            var chosen = 0
            for i in 0..<d2.count { csum += d2[i]; if r <= csum { chosen = i; break } }
            if !cents.contains(where: { $0.elementsEqual(items[chosen]) }) {
                cents.append(items[chosen])
            } else {
                var bestI = 0; var bestV = -Float.infinity
                for i in 0..<d2.count { if d2[i] > bestV && !cents.contains(where: { $0.elementsEqual(items[i]) }) { bestV = d2[i]; bestI = i } }
                cents.append(items[bestI])
            }
        }
        while cents.count < k { cents.append(cents.last ?? items.first!) }
        for i in 0..<cents.count { if cents[i].count != d { cents[i] = Array(repeating: 0, count: d) } }
        return cents
    }

    public func search(query: [Float], k: Int, filter: (@Sendable ([String : String]?) -> Bool)?) async throws -> [SearchResult] {
        guard k > 0 else { return [] }
        guard query.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: query.count)
        }
        // If we have centroids/lists, probe; else linear scan
        if !centroids.isEmpty && !lists.isEmpty {
            // Find nprobe nearest centroids
            let probe = min(config.nprobe, centroids.count)
            var centroidDists: [(Int, Float)] = []
            centroidDists.reserveCapacity(centroids.count)
            for (i, c) in centroids.enumerated() {
                centroidDists.append((i, distance(query, c, metric: metric)))
            }
            centroidDists.sort { $0.1 < $1.1 }
            var candidates = Set<VectorID>()
            for (ci, _) in centroidDists.prefix(probe) {
                for id in lists[ci] { candidates.insert(id) }
            }
            // Score candidates
            var results: [SearchResult] = []
            results.reserveCapacity(min(k, candidates.count))
            for id in candidates {
                guard let (vec, meta) = store[id] else { continue }
                if let filter = filter, !filter(meta) { continue }
                let d = distance(query, vec, metric: metric)
                results.append(SearchResult(id: id, score: d))
            }
            results.sort { $0.score < $1.score }
            if results.count > k { results.removeLast(results.count - k) }
            return results
        } else {
            // Linear scan fallback
            var results: [SearchResult] = []
            results.reserveCapacity(min(k, store.count))
            for (id, (vec, meta)) in store {
                if let filter = filter, !filter(meta) { continue }
                let d = distance(query, vec, metric: metric)
                results.append(SearchResult(id: id, score: d))
            }
            results.sort { $0.score < $1.score }
            if results.count > k { results.removeLast(results.count - k) }
            return results
        }
    }

    public func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String : String]?) -> Bool)?) async throws -> [[SearchResult]] {
        var out: [[SearchResult]] = []
        out.reserveCapacity(queries.count)
        for q in queries {
            out.append(try await search(query: q, k: k, filter: filter))
        }
        return out
    }

    public func clear() async {
        centroids.removeAll()
        lists.removeAll()
        idToListIndex.removeAll()
        store.removeAll()
    }

    public func statistics() async -> IndexStats {
        IndexStats(
            indexType: "IVF",
            vectorCount: store.count,
            dimension: dimension,
            metric: metric,
            details: [
                "nlist": String(centroids.count),
                "nprobe": String(config.nprobe),
                "assigned": String(idToListIndex.count)
            ]
        )
    }

    @discardableResult
    private func removeID(_ id: VectorID, fromList i: Int) -> Bool {
        guard lists.indices.contains(i) else { return false }
        var list = lists[i]
        if let pos = list.firstIndex(of: id) {
            list.remove(at: pos)
            lists[i] = list
            return true
        }
        return false
    }

    public func save(to url: URL) async throws {
        let recs: [PersistedRecord] = store.map { (key, value) in
            PersistedRecord(id: key, vector: value.0, metadata: value.1)
        }
        let payload = PersistedIndex(
            type: "IVF",
            version: 1,
            dimension: dimension,
            metric: metric.rawValue,
            records: recs
        )
        let data = try JSONEncoder().encode(payload)
        try data.write(to: url, options: .atomic)
    }

    public static func load(from url: URL) async throws -> IVFIndex {
        let data = try Data(contentsOf: url)
        let payload = try JSONDecoder().decode(PersistedIndex.self, from: data)
        guard payload.type == "IVF" else { throw VectorError(.invalidData) }
        let idx = IVFIndex(dimension: payload.dimension, metric: .from(raw: payload.metric))
        try await idx.batchInsert(payload.records.map { ($0.id, $0.vector, $0.metadata) })
        try await idx.optimize()
        return idx
    }

    public func compact() async throws {
        if centroids.isEmpty { return }
        lists = Array(repeating: [], count: centroids.count)
        idToListIndex.removeAll(keepingCapacity: false)
        for (id, (vec, _)) in store {
            if let ci = nearestCentroidIndex(for: vec), lists.indices.contains(ci) {
                lists[ci].append(id)
                idToListIndex[id] = ci
            }
        }
    }


    public func contains(id: VectorID) async -> Bool {
        store[id] != nil
    }

    public func update(id: VectorID, vector: [Float]?, metadata: [String : String]?) async throws -> Bool {
        guard var entry = store[id] else { return false }
        if let v = vector {
            guard v.count == dimension else { throw VectorError.dimensionMismatch(expected: dimension, actual: v.count) }
            if let li = idToListIndex[id] { _ = removeID(id, fromList: li); idToListIndex.removeValue(forKey: id) }
            entry.0 = v
            if let ci = nearestCentroidIndex(for: v), lists.indices.contains(ci) { lists[ci].append(id); idToListIndex[id] = ci }
        }
        if let m = metadata { entry.1 = m }
        store[id] = entry
        return true
    }

    public func batchRemove(_ ids: [VectorID]) async throws {
        for id in ids { try await remove(id: id) }
    }
}

// MARK: - AccelerableIndex Implementation
extension IVFIndex {
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
        guard query.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: query.count)
        }
        
        // Estimate candidate count for pre-allocation
        let estimatedCandidates: Int
        if !centroids.isEmpty && !lists.isEmpty {
            // Estimate based on average list size
            let avgListSize = store.count / max(1, lists.count)
            estimatedCandidates = min(store.count, config.nprobe * avgListSize)
        } else {
            estimatedCandidates = store.count
        }
        
        // Pre-allocate with estimated capacity
        var ids: [VectorID] = []
        ids.reserveCapacity(estimatedCandidates)
        var metadata: [[String: String]?] = []
        metadata.reserveCapacity(estimatedCandidates)
        var vectorStorage = ContiguousArray<Float>()
        vectorStorage.reserveCapacity(estimatedCandidates * dimension)
        
        if !centroids.isEmpty && !lists.isEmpty {
            // Find nprobe nearest centroids
            let probe = min(config.nprobe, centroids.count)
            var centroidDists: [(Int, Float)] = []
            centroidDists.reserveCapacity(centroids.count)
            for (i, c) in centroids.enumerated() {
                centroidDists.append((i, distance(query, c, metric: metric)))
            }
            centroidDists.sort { $0.1 < $1.1 }
            
            // Collect candidates from probed lists
            var candidateSet = Set<VectorID>()
            candidateSet.reserveCapacity(estimatedCandidates)
            for (ci, _) in centroidDists.prefix(probe) {
                for id in lists[ci] { candidateSet.insert(id) }
            }
            
            // Gather candidate data in single pass
            for id in candidateSet {
                guard let (vec, meta) = store[id] else { continue }
                ids.append(id)
                metadata.append(meta)
                // Append directly to contiguous storage
                vectorStorage.append(contentsOf: vec)
            }
        } else {
            // Linear scan: all vectors are candidates
            for (id, (vec, meta)) in store {
                ids.append(id)
                metadata.append(meta)
                // Append directly to contiguous storage
                vectorStorage.append(contentsOf: vec)
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
        if centroids.isEmpty || lists.isEmpty {
            return .flat
        }
        
        let structure = IVFStructure(
            centroids: centroids,
            invertedLists: lists,
            nprobe: config.nprobe
        )
        
        return .ivf(structure)
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
