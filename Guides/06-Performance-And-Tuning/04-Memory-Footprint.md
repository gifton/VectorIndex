# Memory Footprint

> **Reading time:** 8 minutes
> **Prerequisites:** [Index Selection Guide](./03-Index-Selection-Guide.md)

---

## The Concept

Understanding memory usage is crucial for capacity planning. Each index type has different overhead characteristics.

---

## Memory Components

```
Total Index Memory = Vector Storage + Index Structure + Metadata + Runtime

Vector Storage:    n × d × sizeof(Float)
Index Structure:   Varies by index type
Metadata:          Dictionary overhead, IDs, user metadata
Runtime:           Query buffers, caches, temporary allocations
```

---

## By Index Type

### FlatIndex

```
Memory = Vectors + ID Mapping + Metadata

Vectors:     n × d × 4 bytes
ID Mapping:  n × ~64 bytes (String ID + pointer)
Metadata:    Variable (user-defined)

Example (1M vectors, 512D):
  Vectors:    1M × 512 × 4 = 2 GB
  ID Mapping: 1M × 64 = 64 MB
  Total:      ~2.1 GB
```

### IVF (IVFIndex)

```
Memory = Vectors + Centroids + Lists + ID Mapping

Vectors:     n × d × 4 bytes
Centroids:   nlist × d × 4 bytes
Lists:       n × ~8 bytes (ID pointers)
ID Mapping:  n × ~64 bytes

Example (1M vectors, 512D, nlist=1024):
  Vectors:    2 GB
  Centroids:  1024 × 512 × 4 = 2 MB
  Lists:      1M × 8 = 8 MB
  ID Mapping: 64 MB
  Total:      ~2.1 GB (minimal overhead)
```

### HNSW (HNSWIndex)

```
Memory = Vectors + Graph + CSR Cache + ID Mapping

Vectors:     n × d × 4 bytes
Graph:       n × avg_connections × 4 bytes
CSR Cache:   ~same as graph
ID Mapping:  n × ~64 bytes

avg_connections ≈ M × 1.5 (Layer 0: 2M, upper: M)

Example (1M vectors, 512D, M=16):
  Vectors:    2 GB
  Graph:      1M × 24 × 4 = 96 MB
  CSR Cache:  ~100 MB
  ID Mapping: 64 MB
  Total:      ~2.3 GB (+15% overhead)

For M=32:
  Graph:      1M × 48 × 4 = 192 MB
  Total:      ~2.4 GB (+20% overhead)
```

### IVF-PQ

```
Memory = PQ Codes + Codebooks + Centroids + ID Mapping

PQ Codes:    n × m bytes (m = number of subspaces)
Codebooks:   m × 256 × dsub × 4 bytes
Centroids:   nlist × d × 4 bytes
ID Mapping:  n × ~8 bytes (just internal ID)

Example (1M vectors, 512D, m=64, nlist=4096):
  PQ Codes:   1M × 64 = 64 MB
  Codebooks:  64 × 256 × 8 × 4 = 512 KB
  Centroids:  4096 × 512 × 4 = 8 MB
  ID Mapping: 8 MB
  Total:      ~80 MB (96% reduction!)
```

---

## Comparison Table

| Index | 1M × 512D | 10M × 512D | 100M × 512D |
|-------|-----------|------------|-------------|
| **FlatIndex** | 2.1 GB | 21 GB | 210 GB |
| **IVF** | 2.1 GB | 21 GB | 210 GB |
| **HNSW (M=16)** | 2.3 GB | 23 GB | 230 GB |
| **HNSW (M=32)** | 2.4 GB | 24 GB | 240 GB |
| **IVF-PQ (m=64)** | 80 MB | 800 MB | 8 GB |

---

## Measuring Actual Memory

### Swift Memory API

```swift
import Darwin

func getMemoryUsage() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.resident_size : 0
}

// Usage:
let before = getMemoryUsage()
// ... build index
let after = getMemoryUsage()
let indexMemory = after - before
print("Index memory: \(indexMemory / 1_000_000) MB")
```

### Profiling with Instruments

```
Xcode Instruments → Allocations
  - Track all allocations during index build
  - Identify memory-heavy operations
  - Find potential leaks
```

---

## Reducing Memory Usage

### Strategy 1: Use PQ Compression

```
Full vectors: 2 GB
PQ (m=64):    64 MB
Savings:      97%
```

### Strategy 2: Reduce HNSW M

```
M=32: 2.4 GB, 98% recall
M=16: 2.3 GB, 95% recall
M=8:  2.2 GB, 90% recall

Trade recall for memory
```

### Strategy 3: Reduce Dimension

```
Before: 1536D embeddings = 6 KB/vector
After:  512D (via PCA) = 2 KB/vector
Savings: 67%

May lose some semantic quality
```

### Strategy 4: External Storage

```
Store vectors on disk, load on demand:
  - Memory-mapped files
  - Lazy loading
  - LRU cache for hot vectors

VectorIndex supports mmap via Kernel #30
```

---

## 🔗 VectorCore Connection

VectorCore's storage choices affect memory:

```swift
// 🔗 VectorCore: ContiguousArray is memory-efficient

// VectorIndex uses ContiguousArray for vector storage
private var vectorStorage: ContiguousArray<Float> = []

// This avoids per-element overhead of [[Float]]
// Savings: ~8 bytes per vector (no array header per vector)
```

---

## Key Takeaways

1. **Vectors dominate memory.** n × d × 4 bytes is the baseline.

2. **HNSW adds ~10-20%.** Graph structure overhead.

3. **IVF-PQ saves ~95%.** Massive compression for large scale.

4. **Measure actual usage.** Estimates can be off.

5. **Trade-offs exist.** Less memory often means less recall.

---

## Next Up

Let's put everything together in a complete system walkthrough:

**[→ Chapter 7: Capstone](../07-Capstone/README.md)**
