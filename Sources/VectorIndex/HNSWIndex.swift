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
        public let rngSeed: UInt64   // deterministic seeding for #35
        public let rngStream: UInt64 // stream id for sharding
        public init(m: Int = 16, efConstruction: Int = 200, efSearch: Int = 64, rngSeed: UInt64 = 0xDEADBEEFCAFEBABE, rngStream: UInt64 = 0) {
            self.m = m
            self.efConstruction = efConstruction
            self.efSearch = efSearch
            self.rngSeed = rngSeed
            self.rngStream = rngStream
        }
    }

    // MARK: - Public API
    public let dimension: Int
    public let metric: SupportedDistanceMetric
    public let config: Configuration
    public var count: Int { activeCount }

    /// `toHNSWMetric(metric)` computed once at init and cached here -- `metric` is
    /// an invariant `let` validated at init, so recomputing this pure mapping on
    /// every search/batchSearch/makeKNNBuildContext/insert/prune call (B19) was
    /// wasted work on the query hot path.
    private let hnswMetric: HNSWMetric

    /// Supported metrics for HNSW traversal kernel
    private static let supportedMetrics: Set<SupportedDistanceMetric> = [.euclidean, .dotProduct, .cosine]

    /// Convert SupportedDistanceMetric to HNSWMetric (validated at init).
    /// Only called once per instance now, from the designated initializer below.
    @inline(__always)
    private static func toHNSWMetric(_ metric: SupportedDistanceMetric) -> HNSWMetric {
        switch metric {
        case .euclidean: return .L2
        case .dotProduct: return .IP
        case .cosine: return .COSINE
        case .manhattan, .chebyshev:
            // Should never reach here due to init validation
            preconditionFailure("Unsupported metric '\(metric)' in HNSWIndex")
        }
    }

    public init(dimension: Int, metric: SupportedDistanceMetric = .euclidean, config: Configuration = .init()) {
        guard Self.supportedMetrics.contains(metric) else {
            preconditionFailure("HNSWIndex does not support metric '\(metric)'. Supported metrics: euclidean, dotProduct, cosine. Use FlatIndex for manhattan/chebyshev.")
        }
        self.dimension = dimension
        self.metric = metric
        self.config = config
        self.hnswMetric = Self.toHNSWMetric(metric)
        self.rng35 = HNSWXoroRNGState.from(seed: config.rngSeed, stream: config.rngStream)
    }

    // Protocol-required initializer. B19: this comment already claimed "delegates
    // to the designated one" but the body actually duplicated the precondition
    // check and RNG init instead -- a real bug (comment vs. code), not just style,
    // since this exact two-labeled-argument call shape (HNSWIndex(dimension:
    // metric:), no config:) is what resolves to *this* initializer rather than the
    // designated one with a defaulted config:, and it's the shape ~15+ call sites
    // across HNSWWALTests, TypedOverloadsTests, AccelerableIndexTests,
    // HNSWKNNGraphTests, and ArrayCopyOptimizationBenchmark use. Now it actually
    // delegates.
    public init(dimension: Int, metric: SupportedDistanceMetric) {
        self.init(dimension: dimension, metric: metric, config: .init())
    }

    public func insert(id: VectorID, vector: [Float], metadata: [String: String]?) async throws {
        try checkVector(vector)
        try await internalInsert(id: id, vector: vector, metadata: metadata, knownInvNorm: nil)
    }

    /// Internal entry point used by the `IndexableVector` typed overload to
    /// carry a pre-computed inverse norm (1/‖v‖) for cosine. When provided,
    /// the invNormsCache is populated incrementally on the insert side so the
    /// next search does not trigger a full O(N·d) rebuild.
    func insert(id: VectorID, vector: [Float], metadata: [String: String]?, knownInvNorm: Float?) async throws {
        try checkVector(vector)
        try await internalInsert(id: id, vector: vector, metadata: metadata, knownInvNorm: knownInvNorm)
    }

    public func remove(id: VectorID) async throws {
        if let wal = walWriter, idToIndex[id] != nil {
            try wal.append(.remove(id: id))
        }
        internalRemove(id: id)
    }

    /// Soft-delete path with no WAL logging. Called by `remove()` after it
    /// has appended a remove record, and by the WAL replay loop.
    private func internalRemove(id: VectorID) {
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

    public func batchInsert(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?)]) async throws {
        try await batchInsertWithHints(items.map { ($0.id, $0.vector, $0.metadata, nil) })
    }

    /// Internal batch entry point used by both the plain `batchInsert` path
    /// and the typed `IndexableVector` overload. Draws every level up-front,
    /// serializes them into a single WAL frame (so batch semantics are atomic
    /// across a crash), then applies all items via `internalInsertAtLevel`
    /// without firing per-item WAL writes.
    internal func batchInsertWithHints(_ items: [(id: VectorID, vector: [Float], metadata: [String: String]?, knownInvNorm: Float?)]) async throws {
        guard !items.isEmpty else { return }
        for item in items { try checkVector(item.vector) }

        var walItems: [HNSWWALInsertItem] = []
        walItems.reserveCapacity(items.count)
        for item in items {
            let level = randomLevel()
            walItems.append(HNSWWALInsertItem(
                id: item.id, level: Int32(level),
                vector: item.vector, metadata: item.metadata,
                knownInvNorm: item.knownInvNorm
            ))
        }
        if let wal = walWriter {
            try wal.append(.batchInsert(walItems))
        }
        for wi in walItems {
            try await internalInsertAtLevel(
                id: wi.id, level: Int(wi.level),
                vector: wi.vector, metadata: wi.metadata,
                knownInvNorm: wi.knownInvNorm
            )
        }
    }

    public func optimize() async throws {
        // Minimal: no-op; future: rebuild/prune
    }

    public func search(query: [Float], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [StringSearchResult] {
        try await search(query: query, k: k, filter: filter, qInvNorm: nil)
    }

    /// Internal search accepting an optional pre-computed query inverse norm
    /// for cosine metric. When the query is known to be unit-normalized, pass 1.0
    /// to skip the norm computation in the traversal kernel.
    func search(query: [Float], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?, qInvNorm: Float?) async throws -> [StringSearchResult] {
        guard k > 0 else { return [] }
        try checkVector(query)
        guard let ep = entryPoint else { return [] }

        // Ensure cached traversal data is up-to-date
        rebuildCSRIfNeeded()
        let N = nodes.count
        if N == 0 { return [] }
        // Allow bitset (exclude deleted nodes)
        let words = (N + 63) >> 6
        var allowBits = [UInt64](repeating: 0, count: words)
        for i in 0..<N {
            if !nodes[i].isDeleted {
                let w = i >> 6, b = i & 63
                allowBits[w] |= (1 &<< UInt64(b))
            }
        }

        // Map metric (validated at init, cached in hnswMetric)
        let m33 = hnswMetric
        let ef = max(config.efSearch, k)

        // Prepare output buffers
        var idsOut = [Int32](repeating: -1, count: ef)
        var distsOut = [Float](repeating: .infinity, count: ef)

        // Bridge pointers - vectorStorage is the primary contiguous buffer
        return query.withUnsafeBufferPointer { qbp in
            vectorStorage.withUnsafeBufferPointer { xbbp in
                allowBits.withUnsafeBufferPointer { allowBP in
                  withExtendedLifetime(csrOffsetsCache) { withExtendedLifetime(csrNeighborsCache) {
                    // Build pointer arrays for layers from cached CSR
                    let offPtrsOpt = csrOffsetsCache.map { arr in arr.withUnsafeBufferPointer { Optional($0.baseAddress!) } }
                    let nbrPtrsOpt = csrNeighborsCache.map { arr in arr.withUnsafeBufferPointer { Optional($0.baseAddress!) } }
                    return offPtrsOpt.withUnsafeBufferPointer { offArr in
                        nbrPtrsOpt.withUnsafeBufferPointer { nbrArr in
                            // Compute inverse norms for cosine if needed (cached)
                            let invNormsPtr = rebuildInvNormsIfNeededForCosine()
                            let written = HNSWTraversal.traverse(
                                q: qbp.baseAddress!, d: dimension,
                                entryPoint: Int32(ep), maxLevel: Int32(maxLevel),
                                offsetsPerLayer: offArr.baseAddress!,
                                neighborsPerLayer: nbrArr.baseAddress!,
                                xb: xbbp.baseAddress!, N: N, ef: ef, metric: m33,
                                allowBits: allowBP.baseAddress!, allowN: N, invNorms: invNormsPtr,
                                qInvNorm: qInvNorm,
                                idsOut: &idsOut, distsOut: &distsOut
                            )
                            var results: [StringSearchResult] = []
                            results.reserveCapacity(k)
                            if written > 0 {
                                var picked = 0
                                for i in 0..<written {
                                    let idx = Int(idsOut[i]); if idx < 0 || idx >= N { continue }
                                    let node = nodes[idx]
                                    if let filter = filter, !filter(node.metadata) { continue }
                                    var score = distsOut[i]
                                    if metric == .euclidean { score = sqrt(score) }
                                    results.append(StringSearchResult(id: node.id, distance: score))
                                    picked += 1
                                    if picked == k { break }
                                }
                            }
                            return results
                        }
                    }
                  } }
                }
            }
        }
    }

    /// Context for parallel batch search - bundles all data needed by worker tasks.
    /// vectorStorage is ContiguousArray<Float> (B19), matching KNNBuildContext and
    /// the actor's own `private var vectorStorage: ContiguousArray<Float>` storage
    /// -- a plain value assignment is COW-shared (zero bytes copied unless the
    /// actor later mutates mid-search), whereas the previous [Float] forced an
    /// eager Array(vectorStorage) buffer copy of the entire vector store on every
    /// batchSearch() call, even though the TaskGroup workers only ever read it.
    private struct BatchSearchContext: @unchecked Sendable {
        let vectorStorage: ContiguousArray<Float>
        let csrOffsets: [[Int32]]
        let csrNeighbors: [[Int32]]
        let invNorms: [Float]?
        let allowBits: [UInt64]
        let nodeInfo: [(id: VectorID, metadata: [String: String]?)]
        let dim: Int
        let maxLevel: Int
        let entryPoint: Int
        let N: Int
        let ef: Int
        let k: Int
        let metric: HNSWMetric
        let isEuclidean: Bool
    }

    public func batchSearch(queries: [[Float]], k: Int, filter: (@Sendable ([String: String]?) -> Bool)?) async throws -> [[StringSearchResult]] {
        guard k > 0 else { return queries.map { _ in [] } }
        if queries.isEmpty { return [] }

        // Validate all queries upfront
        for q in queries {
            try checkVector(q)
        }

        guard let ep = entryPoint else { return queries.map { _ in [] } }

        // Rebuild caches once for entire batch
        rebuildCSRIfNeeded()
        let N = nodes.count
        if N == 0 { return queries.map { _ in [] } }

        // Map metric (validated at init, cached in hnswMetric)
        let m33 = hnswMetric

        // Build context with all data needed for parallel search
        let ctx = BatchSearchContext(
            vectorStorage: vectorStorage,
            csrOffsets: csrOffsetsCache,
            csrNeighbors: csrNeighborsCache,
            invNorms: rebuildInvNormsIfNeededForCosine().map { Array(UnsafeBufferPointer(start: $0, count: N)) },
            allowBits: {
                let words = (N + 63) >> 6
                var bits = [UInt64](repeating: 0, count: words)
                for i in 0..<N {
                    if !nodes[i].isDeleted {
                        let w = i >> 6, b = i & 63
                        bits[w] |= (1 &<< UInt64(b))
                    }
                }
                return bits
            }(),
            nodeInfo: nodes.map { ($0.id, $0.metadata) },
            dim: dimension,
            maxLevel: maxLevel,
            entryPoint: ep,
            N: N,
            ef: max(config.efSearch, k),
            k: k,
            metric: m33,
            isEuclidean: (metric == .euclidean)
        )

        // Perform parallel searches using TaskGroup
        return try await withThrowingTaskGroup(of: (Int, [StringSearchResult]).self) { group in
            for (queryIndex, query) in queries.enumerated() {
                group.addTask {
                    Self.performSingleSearch(query: query, queryIndex: queryIndex, ctx: ctx, filter: filter)
                }
            }

            // Collect results in original query order
            var results = [[StringSearchResult]](repeating: [], count: queries.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results
        }
    }

    /// Static helper for parallel search - runs in TaskGroup
    private static func performSingleSearch(
        query: [Float],
        queryIndex: Int,
        ctx: BatchSearchContext,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) -> (Int, [StringSearchResult]) {
        var idsOut = [Int32](repeating: -1, count: ctx.ef)
        var distsOut = [Float](repeating: .infinity, count: ctx.ef)

        let written = query.withUnsafeBufferPointer { qbp in
            ctx.vectorStorage.withUnsafeBufferPointer { xbbp in
                ctx.allowBits.withUnsafeBufferPointer { allowBP in
                  withExtendedLifetime(ctx.csrOffsets) { withExtendedLifetime(ctx.csrNeighbors) {
                    // Build pointer arrays for CSR layers
                    var offPtrs = [UnsafePointer<Int32>?]()
                    var nbrPtrs = [UnsafePointer<Int32>?]()
                    offPtrs.reserveCapacity(ctx.csrOffsets.count)
                    nbrPtrs.reserveCapacity(ctx.csrNeighbors.count)

                    for arr in ctx.csrOffsets {
                        arr.withUnsafeBufferPointer { offPtrs.append($0.baseAddress) }
                    }
                    for arr in ctx.csrNeighbors {
                        arr.withUnsafeBufferPointer { nbrPtrs.append($0.baseAddress) }
                    }

                    return offPtrs.withUnsafeBufferPointer { offArr in
                        nbrPtrs.withUnsafeBufferPointer { nbrArr in
                            if let invNorms = ctx.invNorms {
                                return invNorms.withUnsafeBufferPointer { invNormsBP in
                                    HNSWTraversal.traverse(
                                        q: qbp.baseAddress!, d: ctx.dim,
                                        entryPoint: Int32(ctx.entryPoint), maxLevel: Int32(ctx.maxLevel),
                                        offsetsPerLayer: offArr.baseAddress!,
                                        neighborsPerLayer: nbrArr.baseAddress!,
                                        xb: xbbp.baseAddress!, N: ctx.N, ef: ctx.ef, metric: ctx.metric,
                                        allowBits: allowBP.baseAddress!, allowN: ctx.N,
                                        invNorms: invNormsBP.baseAddress,
                                        idsOut: &idsOut, distsOut: &distsOut
                                    )
                                }
                            } else {
                                return HNSWTraversal.traverse(
                                    q: qbp.baseAddress!, d: ctx.dim,
                                    entryPoint: Int32(ctx.entryPoint), maxLevel: Int32(ctx.maxLevel),
                                    offsetsPerLayer: offArr.baseAddress!,
                                    neighborsPerLayer: nbrArr.baseAddress!,
                                    xb: xbbp.baseAddress!, N: ctx.N, ef: ctx.ef, metric: ctx.metric,
                                    allowBits: allowBP.baseAddress!, allowN: ctx.N,
                                    invNorms: nil,
                                    idsOut: &idsOut, distsOut: &distsOut
                                )
                            }
                        }
                    }
                  } }
                }
            }
        }

        // Build results from traversal output
        var results: [StringSearchResult] = []
        results.reserveCapacity(ctx.k)
        if written > 0 {
            var picked = 0
            for i in 0..<written {
                let idx = Int(idsOut[i])
                if idx < 0 || idx >= ctx.N { continue }
                let (nodeId, metadata) = ctx.nodeInfo[idx]
                if let filter = filter, !filter(metadata) { continue }
                var score = distsOut[i]
                if ctx.isEuclidean { score = sqrt(score) }
                results.append(StringSearchResult(id: nodeId, distance: score))
                picked += 1
                if picked == ctx.k { break }
            }
        }
        return (queryIndex, results)
    }

    /// Read-only snapshot for parallel kNN-graph construction (see HNSWKNNGraph.swift).
    /// Immutable value snapshot; vectorStorage is COW-shared (no element copy unless
    /// the actor mutates mid-build, which yields snapshot semantics).
    struct KNNBuildContext: @unchecked Sendable {
        let vectorStorage: ContiguousArray<Float>
        let csrOffsets: [[Int32]]
        let csrNeighbors: [[Int32]]
        let invNorms: [Float]?          // cosine only
        let allowBits: [UInt64]
        let rowToNode: [Int32]          // graph row -> internal node index (insertion order)
        let nodeToRow: [Int32]          // internal node index -> graph row, -1 for tombstones
        let dim: Int
        let maxLevel: Int
        let entryPoint: Int
        let N: Int
        let ef: Int
        let k: Int
        let metric: HNSWMetric
    }

    /// Snapshot factory for `buildKNNGraph`. Returns nil when the index has no entry point.
    func makeKNNBuildContext(k: Int, ef: Int) -> (ctx: KNNBuildContext, ids: [VectorID])? {
        guard let ep = entryPoint else { return nil }
        rebuildCSRIfNeeded()
        let N = nodes.count
        let words = (N + 63) >> 6
        var allowBits = [UInt64](repeating: 0, count: words)
        var rowToNode = [Int32]()
        rowToNode.reserveCapacity(activeCount)
        var nodeToRow = [Int32](repeating: -1, count: N)
        var ids: [VectorID] = []
        ids.reserveCapacity(activeCount)
        for i in 0..<N where !nodes[i].isDeleted {
            allowBits[i >> 6] |= (1 &<< UInt64(i & 63))
            nodeToRow[i] = Int32(rowToNode.count)
            rowToNode.append(Int32(i))
            ids.append(nodes[i].id)
        }
        let invNorms = rebuildInvNormsIfNeededForCosine().map { Array(UnsafeBufferPointer(start: $0, count: N)) }
        let ctx = KNNBuildContext(
            vectorStorage: vectorStorage,
            csrOffsets: csrOffsetsCache,
            csrNeighbors: csrNeighborsCache,
            invNorms: invNorms,
            allowBits: allowBits,
            rowToNode: rowToNode,
            nodeToRow: nodeToRow,
            dim: dimension,
            maxLevel: maxLevel,
            entryPoint: ep,
            N: N,
            ef: ef,
            k: k,
            metric: hnswMetric
        )
        return (ctx, ids)
    }

    public func clear() async {
        nodes.removeAll(keepingCapacity: false)
        idToIndex.removeAll(keepingCapacity: false)
        vectorStorage.removeAll(keepingCapacity: false)
        entryPoint = nil
        maxLevel = 0
        activeCount = 0
        // Invalidate caches
        csrOffsetsCache.removeAll(keepingCapacity: false)
        csrNeighborsCache.removeAll(keepingCapacity: false)
        invNormsCache = nil
        csrDirty = true; invNormsDirty = true
    }

    public func contains(id: VectorID) async -> Bool {
        if let i = idToIndex[id] { return i >= 0 && i < nodes.count && !nodes[i].isDeleted }
        return false
    }

    public func update(id: VectorID, vector: [Float]?, metadata: [String: String]?) async throws -> Bool {
        guard let idx = idToIndex[id] else { return false }
        var newVec = vectorArray(at: idx)
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
        // `remove()` -> `internalRemove()` already maintains entryPoint/activeCount
        // per id; only the CSR/invNorms caches need invalidating for the batch.
        markCSRDirty(); markInvNormsDirty()
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
        let vectorOffset: Int  // Offset into vectorStorage (in floats, not bytes)
        var metadata: [String: String]?
        let level: Int
        var neighbors: [[Int]] // per level neighbor indices
        var isDeleted: Bool
    }

    private var nodes: [Node] = []
    private var idToIndex: [VectorID: Int] = [:]
    private var entryPoint: Int?
    private var maxLevel: Int = 0
    private var activeCount: Int = 0
    private var rng35: HNSWXoroRNGState = HNSWXoroRNGState.from(seed: 0xDEADBEEFCAFEBABE, stream: 0)

    // MARK: - Contiguous Vector Storage
    // Primary storage for all vectors in row-major layout: [v0[0..d], v1[0..d], ...]
    // This eliminates per-node heap allocations and enables cache-friendly access.
    private var vectorStorage: ContiguousArray<Float> = []

    // MARK: - Cached traversal buffers (CSR + optional inv norms)
    // These accelerate Kernel #33 traversal. Rebuilt lazily when marked dirty.
    private var csrOffsetsCache: [[Int32]] = []      // per-layer CSR offsets [N+1]
    private var csrNeighborsCache: [[Int32]] = []    // per-layer neighbor lists (flat)
    private var invNormsCache: [Float]?              // optional: inverse L2 norms for cosine

    private var csrDirty: Bool = true
    private var invNormsDirty: Bool = true

    // MARK: - Durability (optional sidecar WAL)
    // When non-nil, every protocol-level mutation is appended to the WAL
    // before the in-memory state is touched (write-ahead discipline).
    private var walWriter: HNSWWALWriter?

    // MARK: - Vector Access Helpers
    /// Returns a copy of the vector at the given node index as [Float] array.
    /// Use this for distance computations during construction that need [Float].
    @inline(__always)
    private func vectorArray(at nodeIndex: Int) -> [Float] {
        let offset = nodes[nodeIndex].vectorOffset
        return Array(vectorStorage[offset..<(offset + dimension)])
    }

    // MARK: - Insertion

    /// Live insert path — draws a random level from `rng35`, records the
    /// operation in the WAL (if enabled) before any in-memory mutation, then
    /// delegates to `internalInsertAtLevel`. The split isolates RNG consumption
    /// from graph mutation so that WAL replay can reuse the exact level that
    /// was sampled during the original insert (without re-advancing the RNG).
    private func internalInsert(id: VectorID, vector: [Float], metadata: [String: String]?, knownInvNorm: Float? = nil) async throws {
        let level = randomLevel()
        if let wal = walWriter {
            try wal.append(.insert(HNSWWALInsertItem(
                id: id, level: Int32(level),
                vector: vector, metadata: metadata,
                knownInvNorm: knownInvNorm
            )))
        }
        try await internalInsertAtLevel(id: id, level: level, vector: vector, metadata: metadata, knownInvNorm: knownInvNorm)
    }

    /// Replay-friendly insert path — takes a pre-sampled `level` and performs
    /// graph mutation without consulting `rng35`. Called by both the live
    /// wrapper above and (later) by the WAL replay loop with a level read
    /// directly from a log record.
    private func internalInsertAtLevel(id: VectorID, level: Int, vector: [Float], metadata: [String: String]?, knownInvNorm: Float? = nil) async throws {
        if idToIndex[id] != nil { // replace existing vector: simple remove then reinsert
            try await remove(id: id)
        }
        let newIndex = nodes.count

        // Store vector in contiguous storage and record offset
        let vectorOffset = vectorStorage.count
        vectorStorage.append(contentsOf: vector)

        let node = Node(id: id, vectorOffset: vectorOffset, metadata: metadata, level: level, neighbors: Array(repeating: [], count: level+1), isDeleted: false)
        nodes.append(node)
        idToIndex[id] = newIndex
        activeCount += 1
        appendInvNormIncremental(knownInvNorm: knownInvNorm); markCSRDirty()

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
                var filtered = candidates.filter { nodes[$0].level >= l }
                // Deduplicate candidates
                var seen = Set<Int>()
                filtered = filtered.filter { seen.insert($0).inserted }

                // Select neighbors using #34 kernel with contiguous vectorStorage
                let selected: [Int] = vector.withUnsafeBufferPointer { vbp in
                    vectorStorage.withUnsafeBufferPointer { xbbp in
                        let ids32 = filtered.map { Int32($0) }
                        var out = [Int32](repeating: -1, count: config.m)
                        let metric34 = hnswMetric
                        let invPtr = (metric == .cosine) ? rebuildInvNormsIfNeededForCosine() : nil
                        let written = ids32.withUnsafeBufferPointer { cbp in
                            out.withUnsafeMutableBufferPointer { obp in
                                hnsw_select_neighbors_f32_swift(
                                    x_new: vbp.baseAddress!, d: dimension,
                                    candidates: cbp.baseAddress!, candCount: ids32.count,
                                    xb: xbbp.baseAddress!, N: nodes.count,
                                    M: config.m, layer: l,
                                    metric: metric34,
                                    optionalInvNorms: invPtr,
                                    selectedOut: obp.baseAddress!
                                )
                            }
                        }
                        if written <= 0 { return [] }
                        return Array(out.prefix(written)).map { Int($0) }
                    }
                }

                connect(newIndex, with: selected, level: l)
                // Update cur to closest among selected for next lower layer.
                // Precompute each candidate's vector + distance once (B17c): the
                // previous min(by:) comparator called vectorArray(at:)/distance(...)
                // twice per comparison for both operands, with zero caching across
                // comparisons -- up to 2(M-1) redundant allocations and full-d
                // distance recomputations for an M-candidate selection.
                if !selected.isEmpty {
                    let dists = selected.map { distance(vector, vectorArray(at: $0), metric: metric) }
                    var bestIdx = 0
                    for i in 1..<dists.count where dists[i] < dists[bestIdx] { bestIdx = i }
                    cur = selected[bestIdx]
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
        markCSRDirty()
        // Add 'a' to each neighbor
        for n in neighbors {
            guard nodes[n].level >= level else { continue }
            if !nodes[n].neighbors[level].contains(a) {
                nodes[n].neighbors[level].append(a)
                pruneNeighbors(of: n, level: level)
                markCSRDirty()
            }
        }
    }

    // Reverse-edge shrink for `idx`'s neighbor list at `level`. Reuses the exact
    // diversity-heuristic kernel that insertion calls (`hnsw_select_neighbors_f32_swift`,
    // Kernels/HNSWNeighborSelection.swift:132), treating idx's own vector as the
    // reference point ("x_new") and its current neighbor list as the candidate set.
    // This mirrors insertion's neighbor selection exactly and prevents the pathology
    // where naive closest-M pruning discards every long bridge edge between
    // well-separated clusters, disconnecting the navigable graph (A9,
    // docs/verification-gap-analysis-p0.md §6.1; hnswlib applies the heuristic on
    // both the insert and prune paths for the same reason).
    private func pruneNeighbors(of idx: Int, level: Int) {
        guard idx >= 0 && idx < nodes.count else { return }
        guard level >= 0 && level < nodes[idx].neighbors.count else { return }
        let current = nodes[idx].neighbors[level]
        if current.isEmpty { nodes[idx].neighbors[level].removeAll(keepingCapacity: false); return }

        let N = nodes.count
        let candidateIDs = current.map { Int32($0) }
        let metric34 = hnswMetric
        let invPtr = (metric == .cosine) ? rebuildInvNormsIfNeededForCosine() : nil
        let anchorOffset = nodes[idx].vectorOffset

        var selected = [Int32](repeating: -1, count: config.m)
        let written: Int = vectorStorage.withUnsafeBufferPointer { xbbp in
            let xNewPtr = xbbp.baseAddress! + anchorOffset
            return candidateIDs.withUnsafeBufferPointer { cbp in
                selected.withUnsafeMutableBufferPointer { sbp in
                    hnsw_select_neighbors_f32_swift(
                        x_new: xNewPtr, d: dimension,
                        candidates: cbp.baseAddress!, candCount: candidateIDs.count,
                        xb: xbbp.baseAddress!, N: N,
                        M: config.m, layer: level,
                        metric: metric34,
                        optionalInvNorms: invPtr,
                        selectedOut: sbp.baseAddress!
                    )
                }
            }
        }
        if written > 0 {
            nodes[idx].neighbors[level] = Array(selected.prefix(written)).map { Int($0) }
        } else {
            nodes[idx].neighbors[level].removeAll(keepingCapacity: false)
        }
        markCSRDirty()
    }

    private func removeNeighbor(_ idx: Int, _ neighbor: Int, level: Int) {
        var list = nodes[idx].neighbors[level]
        if let pos = list.firstIndex(of: neighbor) {
            list.remove(at: pos)
            nodes[idx].neighbors[level] = list
            markCSRDirty()
        }
    }

    // MARK: - Layer Search
    private func greedySearchLayer(_ query: [Float], enter: Int, level: Int) -> Int {
        var cur = enter
        var curDist = distance(query, vectorArray(at: cur), metric: metric)
        var changed = true
        while changed {
            changed = false
            for n in nodes[cur].neighbors[safe: level] ?? [] {
                let d = distance(query, vectorArray(at: n), metric: metric)
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

        let enterDist = distance(query, vectorArray(at: enter), metric: metric)
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
                    let d = distance(query, vectorArray(at: n), metric: metric)
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
        // Kernel #35: Sample level with deterministic RNG per configuration
        // Cap retained at 16 to match prior behavior.
        let cap = 16
        return hnswSampleLevel(M: config.m, cap: cap, rng: &rng35)
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

// MARK: - Test observability hooks
extension HNSWIndex {
    /// Number of entries currently in the inverse-norm cache, or 0 if nil.
    /// Used by tests to verify incremental cache population across inserts.
    internal var _testInvNormsCacheCount: Int { invNormsCache?.count ?? 0 }

    /// Whether the inverse-norm cache is currently marked dirty.
    internal var _testInvNormsCacheIsDirty: Bool { invNormsDirty }
}

// MARK: - Kernel #33: Traversal caches and builders
private extension HNSWIndex {
    @inline(__always) func markInvNormsDirty() { invNormsDirty = true }
    @inline(__always) func markCSRDirty() { csrDirty = true }

    /// Append the inverse norm for the most recently inserted node to the
    /// existing invNormsCache, keeping it coherent across inserts so the next
    /// search does not trigger a full O(N·d) rebuild.
    ///
    /// If `knownInvNorm` is provided (e.g. from an `IndexableVector` with
    /// `isNormalized` or `cachedMagnitude`), the append is O(1). Otherwise
    /// the inverse norm is computed for just the single new vector in O(d).
    ///
    /// Falls back to marking the cache dirty when the cache is absent or
    /// already out of sync — the existing lazy rebuild path remains correct.
    func appendInvNormIncremental(knownInvNorm: Float?) {
        guard metric == .cosine else { return }
        // Must be called immediately after nodes.append, so the new node lives
        // at nodes.count - 1 and the cache should be exactly one entry short.
        guard let existing = invNormsCache,
              !invNormsDirty,
              existing.count + 1 == nodes.count else {
            markInvNormsDirty()
            return
        }
        let newIdx = nodes.count - 1
        let inv: Float
        if let k = knownInvNorm, k.isFinite, k > 0 {
            inv = k
        } else {
            let offset = nodes[newIdx].vectorOffset
            let eps: Float = 1e-12
            var s: Float = 0
            var j = 0
            while j < dimension {
                let v = vectorStorage[offset + j]
                s += v * v
                j &+= 1
            }
            inv = 1.0 / sqrtf(max(s, eps))
        }
        invNormsCache!.append(inv)
    }

    func rebuildCSRIfNeeded() {
        guard csrDirty else { return }
        let N = nodes.count
        let L = max(0, maxLevel)
        csrOffsetsCache.removeAll(keepingCapacity: true)
        csrNeighborsCache.removeAll(keepingCapacity: true)
        if N == 0 {
            csrDirty = false
            return
        }
        csrOffsetsCache.reserveCapacity(L + 1)
        csrNeighborsCache.reserveCapacity(L + 1)
        for level in 0...L {
            var offsets = [Int32](repeating: 0, count: N + 1)
            var flat: [Int32] = []
            var pos = 0
            for u in 0..<N {
                if nodes[u].level >= level {
                    let nbrs = nodes[u].neighbors[safe: level] ?? []
                    if !nbrs.isEmpty {
                        flat.reserveCapacity(pos + nbrs.count)
                        for v in nbrs {
                            if v >= 0 && v < N {
                                flat.append(Int32(v))
                                pos &+= 1
                            }
                        }
                    }
                }
                offsets[u + 1] = Int32(pos)
            }
            csrOffsetsCache.append(offsets)
            csrNeighborsCache.append(flat)
        }
        csrDirty = false
    }

    func rebuildInvNormsIfNeededForCosine() -> UnsafePointer<Float>? {
        guard metric == .cosine else { return nil }
        let N = nodes.count
        if N == 0 { invNormsCache = nil; invNormsDirty = false; return nil }
        if invNormsCache == nil || invNormsDirty || invNormsCache!.count != N {
            invNormsCache = [Float](repeating: 0, count: N)
            let eps: Float = 1e-12
            for i in 0..<N {
                let offset = nodes[i].vectorOffset
                var s: Float = 0
                var j = 0
                while j < dimension {
                    let v = vectorStorage[offset + j]
                    s += v * v
                    j &+= 1
                }
                let inv = 1.0 / sqrtf(max(s, eps))
                invNormsCache![i] = inv
            }
            invNormsDirty = false
        }
        return invNormsCache!.withUnsafeBufferPointer { $0.baseAddress }
    }
}

// MARK: - Safe index access helper
private extension Array where Element == [Int] {
    subscript(safe idx: Int) -> [Int]? {
        guard idx >= 0 && idx < count else { return nil }
        return self[idx]
    }
}

// MARK: - Durability (sidecar WAL)
extension HNSWIndex {
    /// Open (or create) a write-ahead log sidecar inside `directory`. On open,
    /// any existing WAL file is replayed against the current in-memory state
    /// to reconstruct mutations that landed after the last snapshot. After
    /// this call returns, every subsequent `insert` / `remove` / `batchInsert`
    /// is durably recorded before the in-memory state is updated.
    ///
    /// Note: the WAL captures protocol-level mutations only. A full snapshot
    /// (produced by `save(to:)` or `checkpointWAL(to:)`) is still required for
    /// baseline durability — the WAL is an operation log on top of that.
    public func enableWAL(directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let walPath = directory.appendingPathComponent("hnsw.wal").path

        // Replay any existing log BEFORE opening the writer. Records are
        // applied in order via `internalInsertAtLevel` / `internalRemove`,
        // which bypass the WAL append path so replay does not re-log.
        let (records, _) = try HNSWWALReader.replay(path: walPath)
        for record in records {
            try await applyReplayRecord(record)
        }

        self.walWriter = try HNSWWALWriter(path: walPath)
    }

    /// Write a fresh snapshot at `snapshotURL` and truncate the WAL.
    /// Strict ordering: snapshot write completes (atomic rename) before the
    /// WAL truncation. A crash between the two leaves a valid snapshot plus a
    /// redundant WAL; replay is safe because duplicate-id inserts collapse
    /// into a remove-then-reinsert with the same level.
    public func checkpointWAL(to snapshotURL: URL) async throws {
        try await self.save(to: snapshotURL)
        if let wal = walWriter {
            try wal.truncate()
        }
    }

    /// Close the WAL. Writes a checkpoint first when `autoCheckpoint` is true
    /// and a snapshot URL is supplied (otherwise the WAL is closed with all
    /// its records intact — the next `enableWAL` call will replay them).
    public func disableWAL(checkpointTo snapshotURL: URL? = nil) async throws {
        if let snapshot = snapshotURL {
            try await checkpointWAL(to: snapshot)
        }
        walWriter?.close()
        walWriter = nil
    }

    /// Factory: load an index from `snapshotURL` if present (or create an
    /// empty one), then enable the WAL sidecar under `walDirectory` — which
    /// includes replaying any pending WAL records.
    public static func openDurable(
        snapshotURL: URL?,
        walDirectory: URL,
        dimension: Int,
        metric: SupportedDistanceMetric = .euclidean,
        config: Configuration = .init()
    ) async throws -> HNSWIndex {
        let idx: HNSWIndex
        if let snap = snapshotURL, FileManager.default.fileExists(atPath: snap.path) {
            idx = try await HNSWIndex.load(from: snap)
        } else {
            idx = HNSWIndex(dimension: dimension, metric: metric, config: config)
        }
        try await idx.enableWAL(directory: walDirectory)
        return idx
    }

    /// Dispatch a decoded WAL record to the in-memory state via the internal
    /// (non-WAL-writing) mutation methods. Called only by the replay loop.
    private func applyReplayRecord(_ record: HNSWWALRecord) async throws {
        switch record {
        case .insert(let item):
            try checkVector(item.vector)
            try await internalInsertAtLevel(
                id: item.id, level: Int(item.level),
                vector: item.vector, metadata: item.metadata,
                knownInvNorm: item.knownInvNorm
            )
        case .remove(let id):
            internalRemove(id: id)
        case .update:
            // MVP: update() emits a remove + insert pair, so standalone
            // .update records are not produced. Ignore if encountered for
            // forward compatibility with future single-frame updates.
            break
        case .clear:
            // MVP: clear() does not currently log to the WAL, so .clear
            // records are not produced. Accept them as a forward-compat
            // signal to reset in-memory state without cascading.
            nodes.removeAll(keepingCapacity: false)
            idToIndex.removeAll(keepingCapacity: false)
            vectorStorage.removeAll(keepingCapacity: false)
            entryPoint = nil
            maxLevel = 0
            activeCount = 0
            csrOffsetsCache.removeAll(keepingCapacity: false)
            csrNeighborsCache.removeAll(keepingCapacity: false)
            invNormsCache = nil
            csrDirty = true
            invNormsDirty = true
        case .batchInsert(let items):
            for item in items {
                try checkVector(item.vector)
                try await internalInsertAtLevel(
                    id: item.id, level: Int(item.level),
                    vector: item.vector, metadata: item.metadata,
                    knownInvNorm: item.knownInvNorm
                )
            }
        }
    }
}

// MARK: - Persistence & Compaction
extension HNSWIndex {
    public func save(to url: URL) async throws {
        // Persist as flat records; graph is rebuilt on load
        var recs: [PersistedRecord] = []
        recs.reserveCapacity(activeCount)
        for (i, n) in nodes.enumerated() where !n.isDeleted {
            let vector = vectorArray(at: i)
            recs.append(PersistedRecord(id: n.id, vector: vector, metadata: n.metadata))
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
        // Also rebuild vectorStorage to eliminate gaps from deleted vectors
        var newNodes: [Node] = []
        newNodes.reserveCapacity(activeCount)
        var newVectorStorage = ContiguousArray<Float>()
        newVectorStorage.reserveCapacity(activeCount * dimension)
        var remap: [Int: Int] = [:] // old -> new

        for (oldIdx, n) in nodes.enumerated() {
            if !n.isDeleted {
                let newIdx = newNodes.count
                remap[oldIdx] = newIdx
                // Copy vector to new contiguous storage
                let newOffset = newVectorStorage.count
                let oldOffset = n.vectorOffset
                newVectorStorage.append(contentsOf: vectorStorage[oldOffset..<(oldOffset + dimension)])
                // Create new node with updated offset
                let newNode = Node(id: n.id, vectorOffset: newOffset, metadata: n.metadata,
                                   level: n.level, neighbors: n.neighbors, isDeleted: false)
                newNodes.append(newNode)
            }
        }

        // Remap neighbors per level
        let dim = dimension
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
                    let nodeOffset = newNodes[i].vectorOffset
                    let nodeVec = Array(newVectorStorage[nodeOffset..<(nodeOffset + dim)])
                    // Precompute each candidate's distance once (B17c): the previous
                    // sort comparator recomputed vec0/vec1 (fresh Array allocations)
                    // on every pairwise comparison, O(mapped.count log mapped.count)
                    // times, re-copying the same handful of candidates repeatedly.
                    let withDist: [(id: Int, dist: Float)] = mapped.map { cand in
                        let off = newNodes[cand].vectorOffset
                        let vec = Array(newVectorStorage[off..<(off + dim)])
                        return (cand, distance(nodeVec, vec, metric: metric))
                    }
                    mapped = withDist.sorted { $0.dist < $1.dist }.prefix(config.m).map { $0.id }
                }
                lvlLists.append(mapped)
            }
            newNodes[i].neighbors = lvlLists
        }

        nodes = newNodes
        vectorStorage = newVectorStorage

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
        // Mark caches dirty (structure has changed)
        markCSRDirty(); markInvNormsDirty()
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
        var candidateVectorStorage = ContiguousArray<Float>()
        candidateVectorStorage.reserveCapacity(candidateIndices.count * dimension)

        // Collect candidate data in single pass
        for idx in candidateIndices {
            let node = nodes[idx]
            if !node.isDeleted {
                ids.append(node.id)
                metadata.append(node.metadata)
                // Copy vector from our vectorStorage to the output contiguous storage
                let offset = node.vectorOffset
                candidateVectorStorage.append(contentsOf: self.vectorStorage[offset..<(offset + dimension)])
            }
        }

        return AccelerationCandidates(
            ids: ids,
            vectorStorage: candidateVectorStorage,
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
    ) async -> [StringSearchResult] {
        var finalResults: [StringSearchResult] = []
        
        for (idx, distance) in zip(results.indices, results.distances) {
            guard idx < candidates.ids.count else { continue }
            
            let metadata = candidates.metadata[idx]
            if let filter = filter, !filter(metadata) { continue }
            
            finalResults.append(StringSearchResult(
                id: candidates.ids[idx],
                distance: distance
            ))
        }
        
        return finalResults
    }
}
