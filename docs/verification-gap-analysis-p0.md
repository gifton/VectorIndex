# P0 Verification Report — Gap-Analysis Claims & kNN-Graph Contract

Date: 2026-06-11. Scope: re-verification of every `[VI-unverified-here]` claim in
VectorCore's `Docs/gap-analysis-hn-semantic-search.md` (§3.2/§3.4/§3.5/§3.6/§6), plus the
ratified cross-package CSR interchange contract that `HNSWIndex.buildKNNGraph` implements.

## 1. Claim verdicts (all four confirmed — the gap report stands as written)

### (a) Public save/load is JSON-only; HNSW `load` rebuilds via `batchInsert` with no topology serialization — TRUE

- `Sources/VectorIndex/Persistence.swift:4-24` — `PersistedIndex`/`PersistedRecord` carry
  type/version/dimension/metric plus per-record `(id, vector, metadata)`; vectors are text
  arrays under `JSONEncoder`.
- `Sources/VectorIndex/HNSWIndex.swift:1002-1028` — `save(to:)` is `JSONEncoder().encode(payload)`;
  `load(from:)` decodes JSON, creates a fresh index, and calls `batchInsert` — graph topology
  (levels, neighbor lists) is never serialized and is re-derived at O(n log n) insert cost.
- Same pattern: `IVFIndex.swift:656-679` (load → `batchInsert` → `optimize()` re-clusters),
  `FlatIndex.swift:154-176`.

### (b) A binary/mmap format exists but is not wired to public persistence — TRUE

- `Sources/VectorIndex/Kernels/VIndexMmap.swift:1-1059` — full container format: magic
  `VINDEX\0\0` (`0x00585845444E4956` LE, line 71), 256-byte header with version/endianness/arch,
  packed TOC (centroids, codebooks, lists, ids, codes, vecs, norms, idMap, tombstones, walAnchor),
  CRC32 section validation, mmap open/close with alignment checks, and a durable WAL append protocol.
- Consumed internally by IVF kernel #30 (`IVFIndex.swift:37-50`, `kernel30Mmap: IndexMmap?`),
  built by `VIndexContainerBuilder.swift`, appended by `IVFAppend.swift`.
- No public `save`/`load` path reaches it (`IndexProtocols.swift:87-90` route to JSON only).
  This is the P1 work item.

### (c) Metadata/temporal filtering is post-search in all three indexes — TRUE

- `HNSWIndex.swift:213` — filter applied to results *after* `HNSWTraversal.traverse` completes;
  distances already computed.
- `IVFIndex.swift:448-449` — candidates gathered from probed lists, distance computed, *then*
  filter check.
- `FlatIndex.swift:61-65` — filter checked before the distance op but inside an exhaustive scan;
  no candidate-generation pruning anywhere. Pushdown is the P2 work item.

### (d) PQ kernel inventory: u8, u4, and residual/IVF-PQ variants — TRUE

- `Sources/CPQEncode/include/cpq_encode.h:41-118` — `cpq_encode_u8_f32` (+`_with_csq`),
  `cpq_encode_u4_f32` (ks=16, even m), residual variants `cpq_encode_residual_{u8,u4}_f32`
  taking coarse centroids + assignments; u4 pack/unpack helpers; AoS/SoA-blocked/interleaved layouts.
- Training: `Kernels/PQTrain.swift:83-150` — Lloyd + mini-batch with warm start and
  empty-cluster policies.

## 2. EmbedKit dimension — CONFIRMED 384, unit-normalized

- `EmbedKit/Sources/EmbedKitONNX/LocalONNXModel.swift:68` — `dimensions: Int = 384` (default,
  matches all-MiniLM-L6-v2).
- `EmbedKit/Sources/EmbedKit/Core/Types.swift:241` — `normalizeOutput: Bool = true` by default
  and in every factory preset; `EmbeddingMetadata.normalized` tracks it per embedding.
- Consequence: cosine and Euclidean kNN agree on this corpus, and the chord conversion
  d = √(2·(1−cos θ)) used by `buildKNNGraph` on cosine indexes is exact for these vectors.

## 3. Ratified cross-package contract (FINAL)

`VectorCore.KNNGraph` (CSR) is the Core-owned interchange type
(`Sources/VectorCore/ManifoldLearning/KNNGraph.swift`, released in **VectorCore 0.3.1**).
VectorIndex PRODUCES graphs; data flows Index → Core, code never does. Producer semantics:

- **Raw directed** kNN graph — no symmetrization, no similarity weights. Core's
  fuzzy-simplicial-set stage performs its own t-conorm symmetrization and expects distances.
- **Euclidean distances**, finite, ≥ 0. Cosine indexes convert via chord distance
  √(2·(1−cos θ)) (distance between unit-normalized directions). `dotProduct` indexes are
  rejected (no metric interpretation).
- **No self-loops** (self-matches stripped by internal node index). Zero distances to true
  duplicates are legal. Variable per-row degree is legal; rows need not be distance-sorted.
- All construction goes through Core's throwing `KNNGraph.init` — Core validation is the
  single contract gatekeeper.

### Proposed Package_Boundaries.md entry — for the user to land in the VectorCore repo

> KNNGraph (CSR kNN interchange) is Core-owned (`Sources/VectorCore/ManifoldLearning/KNNGraph.swift`).
> VectorIndex is a producer: `HNSWIndex.buildKNNGraph` emits a raw directed Euclidean kNN graph
> validated solely by Core's throwing `KNNGraph` init. Data flows Index→Core (graphs); code flows
> Core→Index (the type). `KNNGraph.bruteForce` remains the exact reference builder at sample scale.
> The gap report §3.2 sketch's `symmetrize:`/weights wording is superseded: producers emit raw
> directed distances; Core symmetrizes.

## 4. Dependency status

VectorCore **0.3.1** (released 2026-06-11) contains `KNNGraph`, `Operations.umap(graph:...)`,
`Operations.umap(_:graph:config:)`, `UMAPConfig`, `UMAPResult`, and `FuzzySimplicialSet` —
verified byte-identical to the `feature/lapack-linkage` worktree the contract was authored
against. This repo's pin is bumped to `from: "0.3.1"` in the same PR; no local dependency
override is needed and no merge gate remains.

## 5. Benchmark results

(filled in by Task 5/6 — Release build, hardware documented)

## 6. Pre-existing issues found (separate tickets, not fixed here)

### 6.1 HNSW reverse-edge pruning disconnects well-separated clusters (HIGH priority)

Discovered while validating the UMAP integration fixture; affects public `search()`,
not just graph building. Evidence (two Gaussian clusters, separation 12, σ=0.5, d=10,
n=120, all of cluster A inserted before cluster B, default config):

- `search(query: A-point, k: 11)` returns ONLY cluster-B points at distance ≈ 11 and does
  not return the queried point's own indexed node (distance 0). `efSearch = 512 > n` does
  not help — cluster A is unreachable from the entry point, i.e. the layer-0 navigable
  graph is disconnected.
- Interleaving the insertion order fully fixes it (`id5` @ 0.0 plus exact neighbors).
- `buildKNNGraph` output matches `search()` edge-for-edge, confirming the producer
  faithfully reflects index behavior.

Mechanism: new-node wiring uses the HNSW diversity heuristic
(`hnsw_select_neighbors_f32_swift`, `Kernels/HNSWNeighborSelection.swift:222-242`), but
reverse-edge shrink on existing nodes (`pruneNeighbors`, `HNSWIndex.swift:667,674` →
`hnsw_prune_neighbors_f32_swift`, `HNSWNeighborSelection.swift:330-332`) keeps the M
closest with no diversity heuristic. When a second distant cluster streams in, the
boundary nodes' overflowed lists are pruned to same-cluster-only on both sides, deleting
every A↔B bridge (hnswlib applies the heuristic in both places to prevent exactly this).
Suggested fix: route prune through the same diversity-heuristic selection.

Relevance at scale: HN ingestion is temporal, and topic drift over years could produce a
milder version of this pathology in the 5M index. Pinned by
`HNSWKNNGraphTests.testKnownIssue_SequentialClusterInsertDisconnectsGraph` (strict
`XCTExpectFailure` — flips to a hard failure when the prune kernel is fixed).

### 6.2 `HNSWIndex.batchRemove` state reset

`HNSWIndex.batchRemove` unconditionally resets `entryPoint`/`maxLevel`/`activeCount` even
on partial removal, leaving the index unsearchable. Per-id `remove(id:)` behaves correctly
and is what the kNN-graph deletion tests use.
