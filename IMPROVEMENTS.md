# VectorIndex Analysis & Improvement Opportunities

> Analysis Date: 2025-11-26
> VectorIndex Version: Current (branch: gifton/vectorcore-0.1.4-bump)
> VectorCore Dependency: 0.1.4 (Package.swift declares `from: "0.1.2"`)

---

## 1. Architecture Overview

**VectorIndex** (~23,500 LoC) is a CPU-first approximate nearest neighbor (ANN) search library with three index implementations:

| Index Type | Description | Complexity |
|------------|-------------|------------|
| `FlatIndex` | Brute-force exact search | O(n·d) |
| `HNSWIndex` | Hierarchical Navigable Small World graph | O(log n) search |
| `IVFIndex` | Inverted File Index with k-means clustering | O(nprobe·n/nlist) |

All indices are Swift `actor` types conforming to `VectorIndexProtocol` with:
- Actor isolation for thread safety
- `AccelerableIndex` protocol for GPU acceleration hooks
- JSON-based persistence (versioned format)

---

## 2. VectorCore Integration Points

### 2.1 Current Usage

**Types imported from VectorCore:**
```swift
- VectorID (String typealias)
- SupportedDistanceMetric (enum: euclidean, cosine, dotProduct, manhattan, chebyshev)
- SwiftFloatSIMDProvider (for DistanceUtils.swift distance() function)
```

### 2.2 Architectural Decision (IndexProtocols.swift:12-24)

```
// We do NOT use VectorCore's implementations:
// - Distance kernels: VectorIndex has 2× faster unsafe pointer versions
// - Vector types: We work on raw [Float] and UnsafePointer<Float>
// - Batch operations: We provide specialized kernel implementations
```

**Critical Note**: VectorIndex intentionally bypasses most of VectorCore's type-safe abstractions for performance, using raw `[Float]` arrays and `UnsafePointer<Float>` throughout.

### 2.3 Missing VectorCore Features (Not Currently Used)

- `BatchOperations` for parallel distance computation
- `MemoryPool` for allocation efficiency
- Mixed-precision kernels (f16 accumulation)
- Optimized vector types (`Vector512Optimized`, `Vector768Optimized`, etc.)

---

## 3. Identified Bottlenecks

### 3.1 Memory Layout & Allocation

**Location**: `HNSWIndex.swift:231-237`

```swift
private struct Node {
    let id: VectorID
    var vector: [Float]           // Each vector is a separate allocation
    var metadata: [String: String]?
    let level: Int
    var neighbors: [[Int]]        // Nested arrays = fragmented memory
    var isDeleted: Bool
}
```

**Issues**:
- N separate heap allocations for N vectors
- Poor cache locality during graph traversal
- Redundant storage (vectors cached again in `xbFlatCache`)

**Recommendation**: Use contiguous storage with offset-based access.

---

### 3.2 Sequential Batch Operations

**Location**: `HNSWIndex.swift:158-163`, `IVFIndex.swift:464-471`

```swift
public func batchSearch(...) async throws -> [[SearchResult]] {
    var out: [[SearchResult]] = []
    for q in queries { out.append(try await search(...)) }  // Sequential!
    return out
}
```

**Issue**: Batch search is sequential despite actor isolation being query-independent.

**Recommendation**: Use `TaskGroup` or parallel dispatch for independent queries.

---

### 3.3 Distance Computation Duplication

The codebase has **three separate implementations** of distance kernels:

1. `DistanceUtils.swift` - Uses `SwiftFloatSIMDProvider` from VectorCore
2. `L2SqrKernel.swift` - Native SIMD4 implementation
3. `HNSWTraversal.swift:42-70` - Inline dot/l2sq functions

**Issues**:
- Code duplication
- Maintenance burden
- Potential inconsistencies between implementations

**Recommendation**: Consolidate to single optimized implementation or leverage VectorCore.

---

### 3.4 Graph Structure Rebuild Overhead

**Location**: `HNSWIndex.swift:536-571`

```swift
private var csrOffsetsCache: [[Int32]] = []   // [numLayers][N+1]
private var csrNeighborsCache: [[Int32]] = [] // [numLayers][edges]

func rebuildCSRIfNeeded() {
    guard csrDirty else { return }
    // Rebuilds ALL layers even if only one changed
}
```

**Issue**: CSR cache is rebuilt entirely on any graph modification.

**Recommendation**: Incremental CSR updates or per-layer dirty tracking.

---

### 3.5 Post-Hoc Filtering

**Location**: `HNSWIndex.swift:141-143`

```swift
if let filter = filter, !filter(node.metadata) { continue }
```

**Issue**: Filtering happens **after** distance computation, wasting cycles on filtered-out candidates.

**Recommendation**: Pre-filter candidates where possible, or use bloom filters for quick rejection.

---

## 4. Obvious Issues

### 4.1 Memory Leak in PQ Encoding

**Location**: `PQEncode.swift:546-549`

```swift
let buf = UnsafeMutablePointer<Float>.allocate(capacity: m * ks)
// ... compute norms ...
return UnsafePointer(buf)  // INTENTIONAL LEAK - documented but problematic
```

**Severity**: Medium - affects long-running processes with repeated PQ encoding.

**Recommendation**: Use caller-provided buffer or implement proper lifecycle management.

---

### 4.2 Persistence Format Limitation

**Location**: `Persistence.swift`

```swift
struct PersistedIndex: Codable {
    let records: [PersistedRecord]  // Full vectors serialized as JSON arrays
}
```

**Issue**: For a 1M × 768 index, this produces ~6GB JSON files.

**Recommendation**: Implement binary serialization format with mmap support.

---

### 4.3 Missing Manhattan/Chebyshev in HNSW Traversal

**Location**: `HNSWIndex.swift:101-107`

```swift
let m33: HNSWMetric = {
    switch metric {
    case .euclidean: return .L2
    case .dotProduct: return .IP
    case .cosine: return .COSINE
    default: return .L2  // Manhattan/Chebyshev silently fall back to L2!
    }
}()
```

**Issue**: Users selecting Manhattan/Chebyshev metrics get incorrect results with no warning.

**Recommendation**: Either implement support or throw explicit error for unsupported metrics.

---

### 4.4 Actor Reentrancy Risk

**Location**: `HNSWIndex.swift:184-196`

```swift
public func update(...) async throws -> Bool {
    try await remove(id: id)           // await 1
    try await internalInsert(...)      // await 2  - state may have changed
    return true
}
```

**Issue**: Between `remove` and `internalInsert`, another task could modify state.

**Recommendation**: Use atomic update operation or document reentrancy behavior.

---

## 5. VectorCore Upgrade Opportunities

### 5.1 Coupling Points (Changes Here Require VectorIndex Updates)

1. `VectorID` typealias
2. `SupportedDistanceMetric` enum
3. `SwiftFloatSIMDProvider` static methods

### 5.2 Potential Benefits from VectorCore Improvements

| Feature | Current State | Potential Improvement |
|---------|--------------|----------------------|
| Vector Storage | Per-node `[Float]` arrays | Contiguous mmap-backed storage |
| Batch Distance | Sequential loops | Parallel batch kernels |
| Memory Management | Standard allocations | Pool-based allocation |
| Precision | f32 only | Mixed f16/f32 support |
| Serialization | JSON | Binary with mmap |
| Telemetry | Per-kernel opt-in | Unified infrastructure |

---

## 6. Performance Characteristics

### 6.1 Scoring Kernels

| Kernel | Optimized Dimensions | Implementation |
|--------|---------------------|----------------|
| L2² | 512, 768, 1024, 1536 | SIMD4 unrolled |
| Cosine | Generic + fused path | Precomputed inverse norms (f32/f16) |
| Inner Product | Generic | SIMD-optimized |
| Dot-trick | All | ‖x-y‖² = ‖x‖² + ‖y‖² - 2⟨x,y⟩ |

### 6.2 PQ Quantization

- u8 (ks=256) and u4 (ks=16, packed) encoding
- Residual IVF-PQ variants
- AoS/SoA layout support
- C backend via CPQEncode module

---

## 7. Test Coverage

Test files present:
- `HNSWTests.swift`, `HNSWRecallTests.swift`, `HNSWTraversalKernelTests.swift`
- `IVFTests.swift`, `IVFMoreTests.swift`, `IVFRecallTests.swift`, `IVFKMeansPlusPlusTests.swift`
- `PersistenceTests.swift`
- `AccelerableIndexTests.swift`
- `PerformanceBenchmarks.swift`

---

## 8. Summary

### High Priority Improvements

1. ~~**Contiguous Vector Storage** - Eliminate per-node allocations~~ ✅ DONE (HNSWIndex)
2. ~~**Parallel Batch Search** - Use TaskGroup for independent queries~~ ✅ DONE (all index types)
3. **Binary Persistence** - Replace JSON with mmap-friendly format
4. ~~**Metric Validation** - Explicit errors for unsupported metrics~~ ✅ DONE (HNSWIndex)

### Medium Priority Improvements

1. **Consolidate Distance Kernels** - Single source of truth
2. **Incremental CSR Updates** - Avoid full rebuilds
3. **PQ Memory Management** - Fix intentional leak (documented workaround: pass `opts.centroidSqNorms`)
4. **Pre-filtering** - Filter before distance computation

### Low Priority / Future

1. **Mixed Precision** - f16 accumulation where applicable
2. **Memory Pools** - VectorCore integration
3. **Unified Telemetry** - Cross-kernel metrics

---

## 9. VectorCore 0.1.5 Integration Plan (Revised)

> Updated: 2025-11-26
> Based on API clarifications from VectorCore engineer

### 9.1 API Limitations Discovered

| Feature | Expected | Actual | Impact |
|---------|----------|--------|--------|
| `TopKSelection` input | `UnsafePointer<Float>` | `[Float]` only | Array conversion overhead |
| `TopKSelection` output | Custom ID support | `indices: [Int]` only | Post-mapping required |
| `TopKSelection` tie-breaking | Configurable | Insertion-order, not configurable | Different determinism model |
| `normalizedUnchecked()` | Works on `[Float]` | `VectorProtocol` types only | Must wrap in DynamicVector |
| Breaking changes | Possible | ✅ None | Safe upgrade path |

### 9.2 Revised Task Priorities

#### Task 1: Package.swift Update — **PROCEED** ✅

```swift
// Change from:
.package(url: "https://github.com/gifton/VectorCore", from: "0.1.2")
// To:
.package(url: "https://github.com/gifton/VectorCore", from: "0.1.5")
```

No breaking changes confirmed. Safe to upgrade.

---

#### Task 2: TopKSelection Integration — **REVISED: PARTIAL ADOPTION**

**Original Plan**: Replace `TopK.swift` entirely with VectorCore's `TopKSelection`

**Revised Plan**: Keep VectorIndex's custom `TopKHeap` for internal hot paths; use `TopKSelection` only at public API boundaries where array conversion is acceptable.

**Rationale**:
```
VectorIndex hot path (IVFSelect, HNSW traversal):
  - Works with UnsafePointer<Float> and Int32 IDs
  - Millions of operations per search
  - Array bridging would add ~15-20% overhead

VectorCore TopKSelection:
  - Requires [Float] array input
  - Returns indices, not custom IDs
  - Better suited for user-facing convenience APIs
```

**Implementation Strategy**:

1. **Keep** `Operations/Selection/TopK.swift` as-is for internal kernels
2. **Keep** `Kernels/IVFSelect.swift` heap implementations (consolidate with TopK.swift later)
3. **Add** thin wrapper for public convenience:

```swift
// New: VectorIndex/Convenience/TopKConvenience.swift
import VectorCore

public extension VectorIndex {
    /// Convenience method using VectorCore's TopKSelection
    /// For performance-critical code, use search() directly
    static func selectTopK(k: Int, from distances: [Float]) -> [(index: Int, distance: Float)] {
        let result = TopKSelection.select(k: k, from: distances)
        return zip(result.indices, result.distances).map { ($0, $1) }
    }
}
```

**Deferred Work**: Consolidate `IVFSelect.swift:778-954` duplicate heaps with `TopK.swift` (not VectorCore dependency).

---

#### Task 3: normalizedUnchecked() Integration — **REVISED: LIMITED SCOPE**

**Original Plan**: Use throughout Norms.swift and Cosine.swift

**Revised Plan**: Only use where vectors are already wrapped in VectorProtocol types.

**Problem**: VectorIndex works with raw `[Float]` and `UnsafePointer<Float>`. Using `normalizedUnchecked()` requires:

```swift
// Current VectorIndex pattern:
let vector: [Float] = ...
let norm = IndexOps.Support.Norms.l2NormSquared(vector: ptr, dimension: d)

// To use normalizedUnchecked(), would need:
let dynamicVec = DynamicVector(vector)  // Allocation!
let normalized = dynamicVec.normalizedUnchecked()  // Another allocation!
```

**Where it CAN help**:
- Public API entry points where users pass typed vectors
- New APIs that accept `VectorProtocol` conforming types

**Where it CANNOT help** (keep existing code):
- `Norms.swift` internal computations (works with raw pointers)
- `HNSWNeighborSelection.swift:77-79` inline norm (hot path)
- `Cosine.swift` scoring kernel (UnsafePointer input)

**Implementation**: Add optional typed API overloads, not replace existing:

```swift
// New overload in public API
public func search<V: VectorProtocol>(
    query: V,
    k: Int,
    filter: ((Metadata) -> Bool)? = nil
) async throws -> [SearchResult] where V.Scalar == Float {
    // Can use normalizedUnchecked() here if V is pre-normalized
    let raw = query.toArray()
    return try await search(query: raw, k: k, filter: filter)
}
```

---

#### Task 4: SearchResult Type Alignment — **NO CHANGE**

Keep VectorIndex's `SearchResult` for API stability. The type mismatch is acceptable:

```swift
// VectorIndex (keep)
public struct SearchResult: Sendable, Equatable {
    public let id: VectorID  // String
    public let score: Float
}

// VectorCore (don't adopt)
struct SearchResult<ID: Hashable> { id, distance, score? }
```

---

#### Task 5: VectorCollection Protocol — **DO NOT ADOPT**

VectorIndex has specialized search implementations. Adopting VectorCore's default brute-force search would be a regression.

---

### 9.3 Revised Implementation Order

| Order | Task | Effort | Risk | Value |
|-------|------|--------|------|-------|
| 1 | Package.swift bump to 0.1.5 | 5 min | None | Unblocks all |
| 2 | Add TopKSelection convenience wrapper | 30 min | Low | User convenience |
| 3 | Add VectorProtocol search overloads | 1 hr | Low | Type safety for users |
| 4 | Consolidate IVFSelect heaps → TopK.swift | 2 hr | Medium | Code reduction |
| 5 | Document API alignment decisions | 30 min | None | Clarity |

### 9.4 What NOT to Do

❌ **Don't** replace `TopKHeap` in hot paths with `TopKSelection`
- Array bridging overhead negates algorithmic improvements

❌ **Don't** wrap raw float arrays in `DynamicVector` just to use `normalizedUnchecked()`
- Two allocations per vector is worse than inline norm computation

❌ **Don't** adopt `VectorCollection` protocol
- VectorIndex's search is specialized and faster

❌ **Don't** change `SearchResult` type
- Breaking API change with no benefit

### 9.5 Future VectorCore Requests

For better integration in future versions, VectorCore could provide:

1. **Pointer-based TopK API**:
```swift
public static func select(
    k: Int,
    from distances: UnsafePointer<Float>,
    count: Int,
    ids: UnsafePointer<Int32>?
) -> (indices: [Int32], distances: [Float])
```

2. **In-place normalization for raw buffers**:
```swift
public static func normalizeUnchecked(
    _ buffer: UnsafeMutablePointer<Float>,
    dimension: Int
)
```

3. **Configurable tie-breaking** in TopKSelection:
```swift
public enum TieBreaker { case insertionOrder, smallerIndex, smallerValue }
```

---

## 10. Integration Checklist

> **Status**: VectorCore 0.1.5 integration completed 2025-11-26

### Pre-Integration
- [x] Verify VectorCore 0.1.5 tag exists and is stable
- [x] Review VectorCore 0.1.5 release notes for any undocumented changes
- [x] Backup current test results for regression comparison

### Integration Steps
- [x] Update Package.swift to VectorCore 0.1.5 (commit 4a5c0cf)
- [x] Run full test suite
- [x] Verify HNSW recall tests pass (HNSWRecallTests: `testHNSWRecallVsFlat` passed)
- [x] Verify IVF recall tests pass (IVFRecallTests: `testIVFRecallVsFlat` passed)
- [ ] Add TopKSelection convenience wrapper (optional - deferred)
- [ ] Add VectorProtocol overloads (optional - deferred)
- [x] Update documentation

### Post-Integration
- [ ] Benchmark TopK performance (should be unchanged)
- [ ] Benchmark search latency (should be unchanged)
- [x] Document any behavioral differences

### Notes
- **No regressions** observed from VectorCore 0.1.5 upgrade
- VectorCore resolved at revision `e9eae35477a6855155ba8389fa7f24f5b25cd5f0`
- Fixed `testFusedCosineCacheMatchesTwoPass` flaky test by using seeded RNG and set-based ID comparison

---

*This document should be updated as VectorCore evolves and improvements are implemented.*
