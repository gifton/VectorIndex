//
//  FlatIndexOptimized.swift
//  VectorIndex
//
//  Zero-copy implementation of FlatIndex using unified contiguous storage
//

import Foundation
import Darwin
import VectorCore

public actor FlatIndexOptimized: VectorIndexProtocol, AccelerableIndex {
    public let dimension: Int
    public let metric: SupportedDistanceMetric
    
    // Unified contiguous storage for all vectors
    private var vectorStorage: ContiguousArray<Float>
    
    // Mapping from ID to storage offset
    private var idToOffset: [VectorID: Int] = [:]
    
    // Metadata storage
    private var idToMetadata: [VectorID: [String: String]?] = [:]
    
    // Track deleted offsets for reuse
    private var freeOffsets: Set<Int> = []

    // Optional fused-cosine norms cache (lifetime-bound to this index)
    private var cosineNormCache: IndexOps.Support.Norms.NormCache? = nil
    private var cosineNormsHandle: IndexOps.Scoring.ScoreBlock.CosineNormsHandle? = nil
    
    public var count: Int { idToOffset.count }
    
    public init(dimension: Int, metric: SupportedDistanceMetric = .euclidean) {
        self.dimension = dimension
        self.metric = metric
        self.vectorStorage = ContiguousArray<Float>()
        self.vectorStorage.reserveCapacity(1000 * dimension) // Initial capacity
    }
    
    public func insert(id: VectorID, vector: [Float], metadata: [String : String]?) async throws {
        guard vector.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.count)
        }
        
        // Remove existing if present
        if let existingOffset = idToOffset[id] {
            freeOffsets.insert(existingOffset)
        }
        
        // Find or allocate offset
        let offset: Int
        if let reusedOffset = freeOffsets.popFirst() {
            // Reuse freed space
            offset = reusedOffset
            // Update existing storage in-place
            for i in 0..<dimension {
                vectorStorage[offset + i] = vector[i]
            }
        } else {
            // Allocate new space at end
            offset = vectorStorage.count
            vectorStorage.append(contentsOf: vector)
        }
        
        idToOffset[id] = offset
        idToMetadata[id] = metadata
    }
    
    public func remove(id: VectorID) async throws {
        if let offset = idToOffset.removeValue(forKey: id) {
            freeOffsets.insert(offset)
            idToMetadata.removeValue(forKey: id)
        }
    }
    
    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String : String]?)]) async throws {
        // Pre-allocate space for efficiency
        let newItemCount = items.count - items.compactMap { idToOffset[$0.id] }.count
        vectorStorage.reserveCapacity(vectorStorage.count + newItemCount * dimension)
        
        for item in items {
            try await insert(id: item.id, vector: item.vector, metadata: item.metadata)
        }
    }
    
    public func optimize() async throws {
        // Compact storage by removing gaps from deleted vectors
        var newStorage = ContiguousArray<Float>()
        var newIdToOffset: [VectorID: Int] = [:]
        
        newStorage.reserveCapacity(count * dimension)
        
        for (id, oldOffset) in idToOffset.sorted(by: { $0.value < $1.value }) {
            let newOffset = newStorage.count
            // Copy vector to new compacted location
            let vectorStart = oldOffset
            let vectorEnd = oldOffset + dimension
            newStorage.append(contentsOf: vectorStorage[vectorStart..<vectorEnd])
            newIdToOffset[id] = newOffset
        }
        
        vectorStorage = newStorage
        idToOffset = newIdToOffset
        freeOffsets.removeAll()

        // Invalidate cosine norm cache after compaction (layout changed)
        cosineNormCache = nil
        cosineNormsHandle = nil
    }
    
    public func search(query: [Float], k: Int, filter: (@Sendable ([String : String]?) -> Bool)?) async throws -> [SearchResult] {
        guard k > 0 else { return [] }
        guard query.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: query.count)
        }

        // Fast path: if storage is compact/contiguous AoS and no filter, use microkernels
        if filter == nil, let fast = fastSearchWithMicrokernels(query: query, k: k) {
            return fast
        }
        
        var results: [(VectorID, Float)] = []
        results.reserveCapacity(min(k, count))
        
        // Process vectors without copying using direct storage access
        for (id, offset) in idToOffset {
            let metadata = idToMetadata[id] ?? nil
            if let filter = filter, !filter(metadata) { continue }
            
            // Compute distance directly from storage without copying
            var distance: Float = 0
            switch metric {
            case .euclidean:
                for i in 0..<dimension {
                    let diff = query[i] - vectorStorage[offset + i]
                    distance += diff * diff
                }
                distance = sqrt(distance)
            case .cosine:
                var dot: Float = 0
                var normA: Float = 0
                var normB: Float = 0
                for i in 0..<dimension {
                    let a = query[i]
                    let b = vectorStorage[offset + i]
                    dot += a * b
                    normA += a * a
                    normB += b * b
                }
                distance = 1.0 - (dot / (sqrt(normA) * sqrt(normB)))
            case .dotProduct:
                for i in 0..<dimension {
                    distance -= query[i] * vectorStorage[offset + i]
                }
            default:
                // Fallback to global distance function for other metrics
                let vec = Array(vectorStorage[offset..<(offset + dimension)])
                distance = VectorIndex.distance(query, vec, metric: metric)
            }
            
            results.append((id, distance))
        }
        
        // Sort and return top-k
        results.sort { $0.1 < $1.1 }
        return results.prefix(k).map { SearchResult(id: $0.0, score: $0.1) }
    }
    
    public func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String : String]?) -> Bool)?) async throws -> [[SearchResult]] {
        var output: [[SearchResult]] = []
        output.reserveCapacity(queries.count)
        for query in queries {
            output.append(try await search(query: query, k: k, filter: filter))
        }
        return output
    }
    
    public func clear() async {
        vectorStorage.removeAll(keepingCapacity: true)
        idToOffset.removeAll(keepingCapacity: true)
        idToMetadata.removeAll(keepingCapacity: true)
        freeOffsets.removeAll()
        cosineNormCache = nil
        cosineNormsHandle = nil
    }
    
    public func statistics() async -> IndexStats {
        let fragmentation = Float(freeOffsets.count * dimension) / Float(vectorStorage.count)
        return IndexStats(
            indexType: "FlatOptimized",
            vectorCount: count,
            dimension: dimension,
            metric: metric,
            details: [
                "storageSize": "\(vectorStorage.count)",
                "fragmentation": String(format: "%.1f%%", fragmentation * 100)
            ]
        )
    }
    
    public func contains(id: VectorID) async -> Bool {
        idToOffset[id] != nil
    }
    
    public func update(id: VectorID, vector: [Float]?, metadata: [String : String]?) async throws -> Bool {
        guard let offset = idToOffset[id] else { return false }
        
        if let vector = vector {
            guard vector.count == dimension else {
                throw VectorError.dimensionMismatch(expected: dimension, actual: vector.count)
            }
            // Update vector in-place without reallocation
            for i in 0..<dimension {
                vectorStorage[offset + i] = vector[i]
            }
        }
        
        if let metadata = metadata {
            idToMetadata[id] = metadata
        }

        // Invalidate cosine cache on vector updates
        cosineNormCache = nil
        cosineNormsHandle = nil

        return true
    }
    
    public func batchRemove(_ ids: [VectorID]) async throws {
        for id in ids {
            try await remove(id: id)
        }
        // Invalidate cosine cache on removals
        cosineNormCache = nil
        cosineNormsHandle = nil
    }
    
    // MARK: - Persistence
    
    public func save(to url: URL) async throws {
        // Compact before saving to minimize file size
        try await optimize()
        
        var records: [PersistedRecord] = []
        for (id, offset) in idToOffset {
            let vector = Array(vectorStorage[offset..<(offset + dimension)])
            records.append(PersistedRecord(id: id, vector: vector, metadata: idToMetadata[id] ?? nil))
        }
        
        let payload = PersistedIndex(
            type: "FlatOptimized",
            version: 1,
            dimension: dimension,
            metric: metric.rawValue,
            records: records
        )
        let data = try JSONEncoder().encode(payload)
        try data.write(to: url, options: .atomic)
    }
    
    public static func load(from url: URL) async throws -> FlatIndexOptimized {
        let data = try Data(contentsOf: url)
        let payload = try JSONDecoder().decode(PersistedIndex.self, from: data)
        guard payload.type == "FlatOptimized" || payload.type == "Flat" else {
            throw VectorError(.invalidData)
        }
        
        let index = FlatIndexOptimized(dimension: payload.dimension, metric: .from(raw: payload.metric))
        try await index.batchInsert(payload.records.map { ($0.id, $0.vector, $0.metadata) })
        return index
    }
    
    public func compact() async throws {
        try await optimize()
    }
}

// MARK: - Microkernel fast path integration
extension FlatIndexOptimized {
    /// Attempt a fast search using microkernels when storage is compact and contiguous.
    /// Falls back by throwing if layout isn't suitable.
    fileprivate func fastSearchWithMicrokernels(query: [Float], k: Int) -> [SearchResult]? {
        // Storage must be fully compact: no holes and contiguous offsets
        guard freeOffsets.isEmpty, idToOffset.count > 0 else { return nil }
        guard vectorStorage.count == idToOffset.count * dimension else { return nil }

        // Only support metrics that map directly to kernels for now
        guard metric == .euclidean || metric == .dotProduct || metric == .cosine else { return nil }

        // IDs sorted by increasing storage offset so row index matches output order
        let sorted = idToOffset.sorted { $0.value < $1.value }
        // Validate contiguity
        if let first = sorted.first?.value, first != 0 { return nil }
        for i in 1..<sorted.count {
            if sorted[i].value != sorted[i-1].value + dimension { return nil }
        }

        var distances = [Float](repeating: 0, count: sorted.count)
        query.withUnsafeBufferPointer { qp in
            vectorStorage.withUnsafeBufferPointer { xbp in
                guard let qptr = qp.baseAddress, let xbptr = xbp.baseAddress else { return }
                let norms = (metric == .cosine) ? cosineNormsHandle : nil
                IndexOps.Scoring.ScoreBlock.run(q: qptr, xb: xbptr, n: sorted.count, d: dimension, metric: metric, out: &distances, cosineNorms: norms)
                // Convert scores to distances as expected by API
                switch metric {
                case .euclidean:
                    // distances currently hold L2^2, sqrt after top-K build
                    break
                case .dotProduct:
                    for i in 0..<distances.count { distances[i] = -distances[i] }
                case .cosine:
                    for i in 0..<distances.count { distances[i] = 1.0 - distances[i] }
                default:
                    break
                }
            }
        }

        // Use Top-K selection instead of full sort
        var idsIdx = [Int32](repeating: 0, count: sorted.count)
        for i in 0..<sorted.count { idsIdx[i] = Int32(i) }

        // Select ordering on raw scores produced by ScoreBlock:
        // - Euclidean: L2^2 (smaller is better) => .min
        // - DotProduct: dot (larger is better)   => .max (do NOT negate here)
        // - Cosine: similarity in [-1,1] (larger is better) => .max (convert to distance on output)
        let ordering = IndexOps.Selection.ordering(for: metric)
        let heap = idsIdx.withUnsafeBufferPointer { ip -> IndexOps.Selection.TopKHeap in
            distances.withUnsafeBufferPointer { sp in
                return IndexOps.Selection.selectTopK(
                    scores: sp.baseAddress!,
                    ids: ip.baseAddress!,
                    count: sorted.count,
                    k: k,
                    ordering: ordering
                )
            }
        }

        var h = heap
        let merged = h.extractSorted() // best→worst
        h.deallocate()

        // Map indices back to external IDs and convert to API distance semantics
        var results: [SearchResult] = []
        results.reserveCapacity(merged.count)
        for (score, idx32) in merged {
            let idx = Int(idx32)
            guard idx >= 0 && idx < sorted.count else { continue }
            let id = sorted[idx].key
            switch metric {
            case .euclidean:
                results.append(SearchResult(id: id, score: sqrt(score)))
            case .dotProduct:
                results.append(SearchResult(id: id, score: -score))
            case .cosine:
                results.append(SearchResult(id: id, score: 1.0 - score))
            default:
                results.append(SearchResult(id: id, score: score))
            }
            if results.count == k { break }
        }
        return results
    }
}

// MARK: - Cosine fused norms adapter (example wiring)
extension FlatIndexOptimized {
    /// Build and enable a fused-cosine inverse-norm cache for the current storage layout.
    /// Requires compact, contiguous AoS layout (same preconditions as the microkernel fast path).
    /// On success, cosine fast path will use a fused single-pass kernel.
    public func enableCosineFusedNormCache(dtype: IndexOps.Support.Norms.NormDType = .float16) async throws {
        guard metric == .cosine else { return }
        // Require compact contiguous storage
        guard freeOffsets.isEmpty, idToOffset.count > 0 else { return }
        guard vectorStorage.count == idToOffset.count * dimension else { return }

        var nc = IndexOps.Support.Norms.NormCache(count: idToOffset.count, dimension: dimension, mode: .inv, invDType: dtype, epsilon: 1e-12)
        nc.allocate()
        let count = idToOffset.count
        let dim = dimension
        let eps = nc.epsilon
        let invOut = nc.invRaw
        let dt = dtype
        vectorStorage.withUnsafeBufferPointer { xbp in
            guard let src = xbp.baseAddress else { return }
            // Ensure 64-byte alignment for the source pointer by copying into an aligned scratch buffer.
            let total = count * dim * MemoryLayout<Float>.stride
            var raw: UnsafeMutableRawPointer?
            let rc = posix_memalign(&raw, 64, total)
            guard rc == 0, let raw = raw else { return }
            defer { free(raw) }
            let dst = raw.assumingMemoryBound(to: Float.self)
            dst.assign(from: src, count: count * dim)

            IndexOps.Support.Norms.normsBuild(
                vectors: UnsafePointer(dst),
                count: count,
                dimension: dim,
                mode: IndexOps.Support.Norms.NormMode.inv,
                epsilon: eps,
                invOut: invOut,
                sqOut: UnsafeMutablePointer<Float>? (nil),
                invDType: dt
            )
        }
        cosineNormsHandle = nc.toCosineNormsHandle()
        cosineNormCache = nc
    }

    /// Disable fused-cosine norm cache and fall back to two-pass cosine.
    public func disableCosineFusedNormCache() async {
        cosineNormCache = nil
        cosineNormsHandle = nil
    }
}

// MARK: - AccelerableIndex Implementation (Zero-Copy)
extension FlatIndexOptimized {
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
        
        // Return reference-based candidates without any copying
        var ids: [VectorID] = []
        var offsets: [Int] = []
        var metadata: [[String: String]?] = []
        
        ids.reserveCapacity(count)
        offsets.reserveCapacity(count)
        metadata.reserveCapacity(count)
        
        for (id, offset) in idToOffset {
            ids.append(id)
            offsets.append(offset)
            metadata.append(idToMetadata[id] ?? nil)
        }
        
        // Create reference-based candidates that point to existing storage
        let refCandidates = ReferenceAccelerationCandidates(
            ids: ids,
            storageReference: vectorStorage,
            vectorOffsets: offsets,
            dimension: dimension,
            metadata: metadata
        )
        
        // Convert to standard format (this is the ONLY place we copy, and only for compatibility)
        return refCandidates.toStandardCandidates()
    }
    
    public func getBatchCandidates(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [AccelerationCandidates] {
        // For flat index, all queries get the same candidates
        let candidates = try await getCandidates(query: queries.first ?? [], k: k, filter: filter)
        return Array(repeating: candidates, count: queries.count)
    }
    
    public func getIndexStructure() async -> IndexStructure {
        return .flat
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
