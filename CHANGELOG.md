## 0.1.0-alpha (Unreleased)

Initial alpha release to unblock downstream packages.

Highlights:
- Public actors: `FlatIndex`, `FlatIndexOptimized`, `IVFIndex`, `HNSWIndex` (all conform to `VectorIndexProtocol` and `AccelerableIndex`).
- Shared types: `SearchResult`, `IndexStats`, `IndexStructure`, `AccelerationCandidates`, `AcceleratedResults`.
- Kernels under `IndexOps` namespace: Scoring (L2, Cosine, InnerProduct, ScoreBlock), Selection (TopK), RangeQuery, Rerank (Exact), Reservoir, Dedup, Filtering, Quantization (PQ, ADC/LUT, Post‑ADC), Support (Norms, LayoutTransforms, Prefetch), Transforms (MIPS), Telemetry.
- JSON persistence for indices.
- Strict Swift 6 concurrency (actors, `@Sendable`).
- C ABI shims for selected kernels (e.g., HNSW traversal, scoring blocks).

Notes:
- API is alpha and may evolve. We will keep syntactic and structural parity with VectorCore where applicable (e.g., `SupportedDistanceMetric`, typed overloads, provider seams).
- Linux is currently not a declared platform. Conditional imports have been added in several files to ease future portability.
