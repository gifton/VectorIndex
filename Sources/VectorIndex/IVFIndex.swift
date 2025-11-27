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
    private var store: [VectorID: ([Float], [String: String]?)] = [:]

    // Kernel #30 integration (optional)
    private var kernel30: IVFListHandle?
    private var kernel30Mmap: IndexMmap?
    // Kernel #50 integration (ID remap + registry)
    private var idMap50: IDMap?
    private var idRegistry: ExternalIDRegistry?
    // Kernel #30 helpers for IVF-Flat rerank (#40):
    // Mapping arrays (internalID -> list, offset). Kept in sync during ingestion.
    private var id2List30: [Int32] = []
    private var id2Offset30: [Int32] = []
    // Per-list internalID vectors to enumerate candidates efficiently (list offset -> internalID).
    private var internalIDsByList30: [[Int64]] = []
    // If true, mapping arrays cover all existing vectors; false when opening durable containers with preexisting data.
    private var mappingComplete30: Bool = true

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

    public func insert(id: VectorID, vector: [Float], metadata: [String: String]?) async throws {
        guard vector.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.count)
        }
        // If replacing existing, detach from previous list mapping
        if store[id] != nil, let oldList = idToListIndex[id] {
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

    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?)]) async throws {
        for item in items {
            try await insert(id: item.id, vector: item.vector, metadata: item.metadata)
        }
        // TODO: bulk assignment
    }

    // MARK: - Kernel #30 Ingestion (Optional)

    /// Configure Kernel #30 storage. If `durablePath` is provided, opens mmap container and enables durable writes.
    public func enableKernel30Storage(format: IVFFormat, k_c: Int, m: Int = 0, durablePath: String? = nil, opts inOpts: IVFAppendOpts? = nil) async throws {
        var opts = inOpts ?? .default
        opts.format = format
        if let durablePath {
            opts.durable = true
            var mo = MmapOpts(); mo.readOnly = false
            let mmap = try IndexMmap.open(path: durablePath, opts: mo)
            self.kernel30Mmap = mmap
            self.kernel30 = try ivf_create_mmap(k_c: k_c, m: m, d: (format == .flat ? dimension : 0), mmap: mmap, opts: opts)
            // Attempt to load IDMap from durable container if present
            if let blob = mmap.readIDMapBlob(), !blob.isEmpty {
                if let loaded = try? deserializeIDMap(blob) {
                    self.idMap50 = loaded
                }
            }
        } else {
            opts.durable = false
            self.kernel30 = try IVFListHandle(k_c: k_c, m: m, d: (format == .flat ? dimension : 0), opts: opts)
        }
        // Initialize ID map and registry for #50 if not present
        if self.idRegistry == nil { self.idRegistry = ExternalIDRegistry() }
        if self.idMap50 == nil {
            // Capacity hint: reserve across lists; use a conservative minimum
            let capHint = max(1024, opts.reserve_min * max(1, k_c))
            self.idMap50 = idmapInit(capacityHint: capHint, opts: .default)
        }
        // Initialize per-list tracking for IVF-Flat rerank path
        self.internalIDsByList30 = Array(repeating: [], count: max(k_c, 0))
        self.id2List30.removeAll(keepingCapacity: false)
        self.id2Offset30.removeAll(keepingCapacity: false)
        self.mappingComplete30 = true
        // If durable with pre-existing data, mark mapping incomplete
        if let h = self.kernel30 {
            var anyPreexisting = false
            for lid in 0..<h.k_c {
                if let stats = try? h.getListStats(listID: Int32(lid)) {
                    if stats.length > 0 { anyPreexisting = true }
                }
            }
            if anyPreexisting { self.mappingComplete30 = false }
        }
    }

    /// Ingest encoded PQ vectors via Kernel #30.
    public func ingestEncodedPQ(listIDs: [Int32], externalIDs: [UInt64], codes: [UInt8], m: Int, opts inOpts: IVFAppendOpts? = nil) async throws {
        guard let h = kernel30 else { throw VectorError(.operationFailed) }
        var localOpts = inOpts ?? h.opts
        localOpts.format = .pq8 // caller may pass .pq4 via codes packing/opts
        let n = listIDs.count
        precondition(externalIDs.count == n)
        // Map external IDs -> internal dense IDs via #50
        if let idMap = idMap50 {
            var assigned = [Int64](repeating: 0, count: n)
            _ = try externalIDs.withUnsafeBufferPointer { extPtr in
                try assigned.withUnsafeMutableBufferPointer { dst in
                    try idmapAppend(idMap, externalIDs: extPtr.baseAddress!, count: n, internalIDsOut: dst.baseAddress)
                }
            }
            // Ingest into kernel #30 and verify internal IDs remain aligned
            var returned = [Int64](repeating: -1, count: n)
            try ivf_append(list_ids: listIDs, external_ids: externalIDs, codes: codes, n: n, m: m, index: h, opts: localOpts, internalIDsOut: &returned)
            // Best-effort sanity check (should match 1:1)
            #if DEBUG
            if assigned != returned {
                // If mismatch, this indicates counter drift between kernels
                throw VectorError(.operationFailed)
            }
            #endif
            // Persist IDMap snapshot if durable
            persistKernel30IDMapSnapshot()
        } else {
            // Fallback without ID map
            try ivf_append(list_ids: listIDs, external_ids: externalIDs, codes: codes, n: n, m: m, index: h, opts: localOpts, internalIDsOut: nil)
        }
    }

    /// Ingest IVF-Flat vectors via Kernel #30.
    public func ingestFlat(listIDs: [Int32], externalIDs: [UInt64], vectors: [Float], opts inOpts: IVFAppendOpts? = nil) async throws {
        guard let h = kernel30, h.format == .flat else { throw VectorError(.operationFailed) }
        let n = listIDs.count
        precondition(vectors.count == n * dimension)
        let localOpts = inOpts ?? h.opts
        precondition(externalIDs.count == n)
        if let idMap = idMap50 {
            // Snapshot pre-append lengths per list to compute offsets
            var oldLen: [Int32: Int] = [:]
            var perListCounts: [Int32: Int] = [:]
            for lid in listIDs {
                if oldLen[lid] == nil {
                    if let st = try? h.getListStats(listID: lid) { oldLen[lid] = st.length } else { oldLen[lid] = 0 }
                }
                perListCounts[lid] = 0
            }
            var assigned = [Int64](repeating: 0, count: n)
            _ = try externalIDs.withUnsafeBufferPointer { extPtr in
                try assigned.withUnsafeMutableBufferPointer { dst in
                    try idmapAppend(idMap, externalIDs: extPtr.baseAddress!, count: n, internalIDsOut: dst.baseAddress)
                }
            }
            var returned = [Int64](repeating: -1, count: n)
            try ivf_append_flat(list_ids: listIDs, external_ids: externalIDs, xb: vectors, n: n, d: dimension, index: h, opts: localOpts, internalIDsOut: &returned)
            #if DEBUG
            if assigned != returned { throw VectorError(.operationFailed) }
            #endif
            // Update mapping arrays incrementally
            // Ensure id2 arrays large enough
            let maxID = Int((returned.max() ?? -1))
            if maxID >= 0 {
                if id2List30.count <= maxID { id2List30.append(contentsOf: repeatElement(-1, count: maxID + 1 - id2List30.count)) }
                if id2Offset30.count <= maxID { id2Offset30.append(contentsOf: repeatElement(-1, count: maxID + 1 - id2Offset30.count)) }
            }
            // Prepare per-list internalIDs storage if we are in a fresh mapping session
            if mappingComplete30 {
                // Ensure per-list vectors are at least oldLen
                for (lid, len) in oldLen {
                    let i = Int(lid)
                    if internalIDsByList30.indices.contains(i) {
                        if internalIDsByList30[i].count < len {
                            internalIDsByList30[i].reserveCapacity(len)
                            // Append placeholders for legacy region; we cannot reconstruct past internalIDs.
                            // Since mappingComplete30 implies no preexisting data, this block should rarely run.
                            while internalIDsByList30[i].count < len { internalIDsByList30[i].append(-1) }
                        }
                    }
                }
            }
            // Assign per element
            for idx in 0..<n {
                let lid = listIDs[idx]
                let base = oldLen[lid] ?? 0
                let localIdx = perListCounts[lid, default: 0]
                perListCounts[lid] = localIdx + 1
                let off = base + localIdx
                let iid = returned[idx]
                let ii = Int(iid)
                if ii >= 0 {
                    id2List30[ii] = lid
                    id2Offset30[ii] = Int32(off)
                    if mappingComplete30 {
                        let l = Int(lid)
                        if internalIDsByList30.indices.contains(l) {
                            if internalIDsByList30[l].count == off {
                                internalIDsByList30[l].append(iid)
                            } else if internalIDsByList30[l].count < off {
                                internalIDsByList30[l].reserveCapacity(off + 1)
                                while internalIDsByList30[l].count < off { internalIDsByList30[l].append(-1) }
                                internalIDsByList30[l].append(iid)
                            } else {
                                // Should not happen; keep safe
                                internalIDsByList30[l][off] = iid
                            }
                        }
                    }
                }
            }
            // Persist IDMap snapshot if durable
            persistKernel30IDMapSnapshot()
        } else {
            try ivf_append_flat(list_ids: listIDs, external_ids: externalIDs, xb: vectors, n: n, d: dimension, index: h, opts: localOpts, internalIDsOut: nil)
        }
    }

    public func optimize() async throws {
        // Build centroids with CPU Lloyd's KMeans and assign points to lists
        // Use k = min(nlist, store.count)
        guard !store.isEmpty else {
            centroids.removeAll(); lists.removeAll(); return
        }
        let k = max(1, min(config.nlist, store.count))
        // Initialize centroids using deterministic k‑means++ (farthest‑point) seeding
        let initialCentroids = try kmeansPlusPlusInitRandom(k: k, seed: 42)
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
        let initC = try kmeansPlusPlusInitRandom(k: k, seed: 42)
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

    // MARK: - Lloyd's KMeans using Kernel #12
    private func kmeans(centroids initial: [[Float]], maxIterations: Int) async throws -> [[Float]] {
        precondition(!initial.isEmpty)
        let k = initial.count
        let d = dimension
        let items: [[Float]] = store.map { $0.value.0 }

        // Flatten data for kernel
        let flatData = items.flatMap { $0 }
        var flatCentroids = initial.flatMap { $0 }

        // Configure mini-batch k-means
        let cfg = KMeansMBConfig(
            algo: .lloydMiniBatch,
            batchSize: min(1024, items.count),  // Adaptive batch size
            epochs: maxIterations,
            subsampleN: 0,  // Use all data
            tol: 1e-4,
            decay: 0.01,
            seed: 42,
            streamID: 0,
            prefetchDistance: 8,
            layout: .aos,
            aosoaRegisterBlock: 0,
            computeAssignments: false
        )

        // Run kernel
        let status = kmeans_minibatch_f32(
            x: flatData,
            n: Int64(items.count),
            d: d,
            kc: k,
            initCentroids: flatCentroids,
            cfg: cfg,
            centroidsOut: &flatCentroids,
            assignOut: nil,
            statsOut: nil
        )

        guard status == .success else {
            throw VectorError(.operationFailed)
        }

        // Reshape flat centroids to [[Float]]
        var cents: [[Float]] = []
        cents.reserveCapacity(k)
        for i in 0..<k {
            let start = i * d
            let end = start + d
            cents.append(Array(flatCentroids[start..<end]))
        }

        return cents
    }

    // Seeded k-means++ (D²) sampling using Kernel #11
    private func kmeansPlusPlusInitRandom(k: Int, seed: UInt64) throws -> [[Float]] {
        precondition(!store.isEmpty)
        let d = dimension
        let items: [[Float]] = store.map { $0.value.0 }
        let flatData = items.flatMap { $0 }

        // Allocate output buffer
        var flatCentroids = [Float](repeating: 0, count: k * d)

        // Use Kernel #11 for seeding
        let cfg = KMeansSeedConfig(
            algorithm: .plusPlus,
            k: k,
            sampleSize: 0,
            rngSeed: seed,
            rngStreamID: 0,
            strictFP: false,
            prefetchDistance: 2,
            oversamplingFactor: 2,
            rounds: 5
        )

        // store.isEmpty precondition ensures n >= 1,
        // and k is validated by caller to be reasonable
        _ = try kmeansPlusPlusSeed(
            data: flatData,
            count: items.count,
            dimension: d,
            k: k,
            config: cfg,
            centroidsOut: &flatCentroids,
            chosenIndicesOut: nil
        )

        // Reshape flat centroids to [[Float]]
        var cents: [[Float]] = []
        cents.reserveCapacity(k)
        for i in 0..<k {
            let start = i * d
            let end = start + d
            cents.append(Array(flatCentroids[start..<end]))
        }

        return cents
    }

    public func search(query: [Float], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [SearchResult] {
        guard k > 0 else { return [] }
        guard query.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: query.count)
        }
        // Kernel #30 IVF-Flat accelerated path with exact rerank (#40).
        if let h = kernel30, h.format == .flat, mappingComplete30,
           metric == .euclidean || metric == .dotProduct || metric == .cosine {
            return try await searchKernel30Flat(query: query, k: k, filter: filter)
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

    /// Context for parallel batch search - bundles immutable data for worker tasks
    private struct IVFBatchSearchContext: @unchecked Sendable {
        let centroids: [[Float]]
        let lists: [[VectorID]]
        let store: [VectorID: ([Float], [String: String]?)]
        let dimension: Int
        let metric: SupportedDistanceMetric
        let nprobe: Int
        let k: Int
    }

    public func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [[SearchResult]] {
        guard k > 0 else { return queries.map { _ in [] } }
        if queries.isEmpty { return [] }

        // Validate all queries upfront
        for q in queries {
            guard q.count == dimension else {
                throw VectorError.dimensionMismatch(expected: dimension, actual: q.count)
            }
        }

        // If kernel30 is active, fall back to sequential (kernel might not be thread-safe)
        if let h = kernel30, h.format == .flat, mappingComplete30,
           metric == .euclidean || metric == .dotProduct || metric == .cosine {
            var out: [[SearchResult]] = []
            out.reserveCapacity(queries.count)
            for q in queries {
                out.append(try await searchKernel30Flat(query: q, k: k, filter: filter))
            }
            return out
        }

        // For standard path, use parallel execution
        if !centroids.isEmpty && !lists.isEmpty {
            // Snapshot data for parallel access
            let ctx = IVFBatchSearchContext(
                centroids: centroids,
                lists: lists,
                store: store,
                dimension: dimension,
                metric: metric,
                nprobe: min(config.nprobe, centroids.count),
                k: k
            )

            return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
                for (queryIndex, query) in queries.enumerated() {
                    group.addTask {
                        Self.performIVFSearch(query: query, queryIndex: queryIndex, ctx: ctx, filter: filter)
                    }
                }

                var results = [[SearchResult]](repeating: [], count: queries.count)
                for try await (index, result) in group {
                    results[index] = result
                }
                return results
            }
        } else {
            // Linear scan fallback - also parallelize
            let storeCopy = store
            let dim = dimension
            let met = metric
            let kk = k

            return try await withThrowingTaskGroup(of: (Int, [SearchResult]).self) { group in
                for (queryIndex, query) in queries.enumerated() {
                    group.addTask {
                        Self.performLinearSearch(query: query, queryIndex: queryIndex, store: storeCopy, dimension: dim, metric: met, k: kk, filter: filter)
                    }
                }

                var results = [[SearchResult]](repeating: [], count: queries.count)
                for try await (index, result) in group {
                    results[index] = result
                }
                return results
            }
        }
    }

    /// Static helper for parallel IVF search with centroids
    private static func performIVFSearch(
        query: [Float],
        queryIndex: Int,
        ctx: IVFBatchSearchContext,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) -> (Int, [SearchResult]) {
        // Find nprobe nearest centroids
        var centroidDists: [(Int, Float)] = []
        centroidDists.reserveCapacity(ctx.centroids.count)
        for (i, c) in ctx.centroids.enumerated() {
            centroidDists.append((i, distance(query, c, metric: ctx.metric)))
        }
        centroidDists.sort { $0.1 < $1.1 }

        // Gather candidates from top nprobe lists
        var candidates = Set<VectorID>()
        for (ci, _) in centroidDists.prefix(ctx.nprobe) {
            for id in ctx.lists[ci] { candidates.insert(id) }
        }

        // Score candidates
        var results: [SearchResult] = []
        results.reserveCapacity(min(ctx.k, candidates.count))
        for id in candidates {
            guard let (vec, meta) = ctx.store[id] else { continue }
            if let filter = filter, !filter(meta) { continue }
            let d = distance(query, vec, metric: ctx.metric)
            results.append(SearchResult(id: id, score: d))
        }
        results.sort { $0.score < $1.score }
        if results.count > ctx.k { results.removeLast(results.count - ctx.k) }

        return (queryIndex, results)
    }

    /// Static helper for parallel linear scan search
    private static func performLinearSearch(
        query: [Float],
        queryIndex: Int,
        store: [VectorID: ([Float], [String: String]?)],
        dimension: Int,
        metric: SupportedDistanceMetric,
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) -> (Int, [SearchResult]) {
        var results: [SearchResult] = []
        results.reserveCapacity(min(k, store.count))
        for (id, (vec, meta)) in store {
            if let filter = filter, !filter(meta) { continue }
            let d = distance(query, vec, metric: metric)
            results.append(SearchResult(id: id, score: d))
        }
        results.sort { $0.score < $1.score }
        if results.count > k { results.removeLast(results.count - k) }

        return (queryIndex, results)
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
                // Report configured nlist for consistency, even if training
                // used fewer centroids due to small dataset size.
                "nlist": String(config.nlist),
                // Also report how many centroids are currently trained/built.
                "trained_nlist": String(centroids.count),
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

    public func update(id: VectorID, vector: [Float]?, metadata: [String: String]?) async throws -> Bool {
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

    // MARK: - Utilities (#50) — Map internal IDs to public VectorIDs
    public func mapInternalToPublicIDs(_ internalIDs: [Int64]) -> [VectorID] {
        guard let idMap = idMap50 else { return [] }
        var outs: [VectorID] = []
        outs.reserveCapacity(internalIDs.count)
        for iid in internalIDs {
            let ext = idmapExternalFor(idMap, internalID: iid)
            if let s = idRegistry?.getString(ext) {
                outs.append(s)
            } else {
                // If registry unused for this ingest path, represent as numeric string
                outs.append(String(ext))
            }
        }
        return outs
    }
}

// MARK: - Kernel #30 IVF-Flat exact rerank integration (#40)
extension IVFIndex {
    // Synthetic candidate ID encoder: high 32 bits = listID, low 32 bits = offset
    @inline(__always) private func packCandID(list: Int32, offset: Int32) -> Int64 {
        (Int64(list) << 32) | (Int64(UInt32(bitPattern: offset)))
    }
    @inline(__always) private func unpackCandID(_ id: Int64) -> (Int32, Int32) {
        let list = Int32(truncatingIfNeeded: id >> 32)
        let off = Int32(truncatingIfNeeded: id & 0xFFFF_FFFF)
        return (list, off)
    }

    private func searchKernel30Flat(query: [Float], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [SearchResult] {
        guard let h = kernel30, h.format == .flat else { return [] }
        // 1) Select lists to probe
        let kc = h.k_c
        var probeLists: [Int32] = []
        if !centroids.isEmpty && (centroids.count == kc) && (centroids.first?.count == dimension) {
            var ids: [Int32] = []
            var scores: [Float]?
            let metricSel: IVFMetric = (metric == .euclidean) ? .l2 : (metric == .dotProduct ? .ip : .cosine)
            // Flatten centroids [[Float]] -> [Float]
            let flatC = centroids.flatMap { $0 }
            ivf_select_nprobe_f32(q: query, d: dimension, centroids: flatC, kc: kc, metric: metricSel, nprobe: min(config.nprobe, kc), listIDsOut: &ids, listScoresOut: &scores)
            probeLists = ids
        } else {
            // Fallback: probe all lists
            probeLists = (0..<kc).map { Int32($0) }
        }

        // 2) Build per-list info and candidate internal IDs
        var listBases: [Int32: UnsafePointer<Float>] = [:]
        var listLengths: [Int32: Int] = [:]
        var idPtrsU64: [Int32: UnsafePointer<UInt64>] = [:]
        var idPtrsU32: [Int32: UnsafePointer<UInt32>] = [:]
        var candInternalIDs: [Int64] = []
        candInternalIDs.reserveCapacity(1024)
        for lid in probeLists {
            let (len, idsU64, idsU32, _, xbPtr) = try h.readList(listID: lid)
            guard let xb = xbPtr, len > 0 else { continue }
            listBases[lid] = xb
            listLengths[lid] = len
            if let p64 = idsU64 { idPtrsU64[lid] = p64 }
            if let p32 = idsU32 { idPtrsU32[lid] = p32 }
            // Map list offsets -> internal dense IDs via internalIDsByList30
            let li = Int(lid)
            guard internalIDsByList30.indices.contains(li) else { continue }
            let listVec = internalIDsByList30[li]
            candInternalIDs.reserveCapacity(candInternalIDs.count + len)
            for off in 0..<len {
                let iid = listVec[off]
                if iid >= 0 { candInternalIDs.append(iid) }
            }
        }
        guard !candInternalIDs.isEmpty else { return [] }

        // 3) Build IVFListVecs reader and run exact rerank without materializing all vectors
        let Nint = max(id2List30.count, id2Offset30.count)
        // Build lists array of length kc; unused lists get len=0 and a safe dummy base
        var lists: [IndexOps.Rerank.IVFListVecsReader.List] = []
        lists.reserveCapacity(kc)
        var dummy: [Float] = [0]
        let d = dimension
        for li in 0..<kc {
            let lid = Int32(li)
            if let base = listBases[lid], let len = listLengths[lid] {
                lists.append(.init(base: base, len: len))
            } else {
                let base = dummy.withUnsafeMutableBufferPointer { $0.baseAddress! }
                lists.append(.init(base: UnsafePointer(base), len: 0))
            }
        }
        var topScores = [Float](repeating: 0, count: k)
        var topIDs = [Int64](repeating: -1, count: k)
        query.withUnsafeBufferPointer { qbp in
            id2List30.withUnsafeBufferPointer { lptr in
                id2Offset30.withUnsafeBufferPointer { optr in
                    lists.withUnsafeBufferPointer { lsbp in
                        candInternalIDs.withUnsafeBufferPointer { cidbp in
                            let reader = IndexOps.Rerank.IVFListVecsReader(
                                lists: Array(lsbp),
                                id2List: lptr.baseAddress!,
                                id2Offset: optr.baseAddress!,
                                N: Nint,
                                dim: d,
                                invNorms: nil,
                                sqNorms: nil
                            )
                            let opts = IndexOps.Rerank.RerankOpts(
                                backend: .ivfListVecs,
                                gatherTile: 128,
                                reorderBySegment: true,
                                haveInvNorms: false,
                                haveSqNorms: false,
                                returnSorted: true,
                                skipMissing: true,
                                prefetchDistance: 8,
                                strictFP: false,
                                enableParallel: true,
                                parallelThreshold: 2048,
                                maxConcurrency: 0
                            )
                            topScores.withUnsafeMutableBufferPointer { sb in
                                topIDs.withUnsafeMutableBufferPointer { ib in
                                    IndexOps.Rerank.rerank_exact_topk(
                                        q: qbp.baseAddress!, d: d, metric: metric,
                                        candIDs: cidbp.baseAddress!, C: candInternalIDs.count, K: k,
                                        reader: reader, opts: opts,
                                        topScores: sb.baseAddress!, topIDs: ib.baseAddress!
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4) Map internal IDs to external VectorIDs and apply optional filter
        var results: [SearchResult] = []
        results.reserveCapacity(k)
        outer: for i in 0..<topIDs.count {
            let iid = topIDs[i]
            if iid < 0 { break }
            var ext: UInt64 = 0
            if let idMap = self.idMap50 { ext = idmapExternalFor(idMap, internalID: iid) } else { ext = 0 }
            let vid: VectorID = idRegistry?.getString(ext) ?? String(ext)
            if let filter = filter {
                let meta = store[vid]?.1
                if !filter(meta) { continue }
            }
            results.append(SearchResult(id: vid, score: topScores[i]))
            if results.count == k { break outer }
        }
        return results
    }
}

// MARK: - Durable IDMap snapshot helpers
extension IVFIndex {
    private func persistKernel30IDMapSnapshot() {
        guard let mmap = self.kernel30Mmap, let map = self.idMap50 else { return }
        do {
            let blob = try serializeIDMap(map)
            try mmap.writeIDMapBlob(blob)
        } catch {
            // Best-effort only; ignore failures
        }
    }
}
