# Swift 6 Compliance & Sendable Conformance Plan for VectorIndex

## Executive Summary

This document outlines a comprehensive plan to fix compilation issues and ensure proper Swift 6 compliance for the VectorIndex package, based on analysis of both VectorIndex and EmbedKit codebases.

## Current Status

- **Swift Version**: 6.1.2 with Swift 6.0 tools
- **Concurrency Model**: `StrictConcurrency` enabled in Package.swift
- **Build Status**: Main library builds successfully, tests have compilation errors
- **Key Issues**: Type mismatches in tests, some actor isolation concerns in storage layer

## Key Patterns from EmbedKit to Adopt

### 1. `@preconcurrency` Import for Metal
```swift
@preconcurrency import Metal
```

### 2. `nonisolated` Properties for Thread-Safe Resources
```swift
nonisolated public let device: MTLDevice
nonisolated public let commandQueue: MTLCommandQueue
```

### 3. Actor-Based Resource Management
- Separate actors for different concerns (like `MetalResourceManager`)
- Clear isolation boundaries
- Proper async/await patterns without blocking

---

## Phase 1: Fix Immediate Compilation Errors

### 1.1 Test Type Conversions
**Location**: `Tests/VectorIndexTests/FlatIndexTests.swift` (lines 68-97)

```swift
// Before: 
let vectors: [[Double]] = [[1.0, 2.0], [3.0, 4.0]]

// After:
let vectors: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
```

### 1.2 Async Test Assertions
**Location**: `PersistenceEdgeTests.swift`, `FlatIndexEdgeCasesTests.swift`

```swift
// Before:
XCTAssertThrowsErrorAsync({ await someFunc() })

// After:
XCTAssertThrowsErrorAsync(try await someFunc())
```

### 1.3 API Updates
**Location**: `HNSWAlignmentTest.swift` (line 80)

```swift
// Before:
candidates.vectors

// After:
candidates.vectorStorage.withUnsafeBufferPointer { buffer in
    // Use buffer
}
```

### 1.4 Closure Signatures
**Location**: `ArrayCopyOptimizationBenchmark.swift` (line 151)

```swift
// Before:
.map { /* ignoring parameter */ }

// After:
.map { _ in /* explicitly ignore */ }
```

---

## Phase 2: Metal Integration Sendable Compliance

### 2.1 Update Metal Imports
All Metal-related files should use:
```swift
@preconcurrency import Metal
```

### 2.2 Mark Metal Resources as `nonisolated`
```swift
public actor MetalDevice {
    nonisolated public let device: MTLDevice
    nonisolated public let commandQueue: MTLCommandQueue
    nonisolated public let library: MTLLibrary
    
    // Actor-isolated state
    private var bufferCache: [String: MTLBuffer] = [:]
    private var statistics: DeviceStatistics = DeviceStatistics()
}
```

### 2.3 Buffer Management Pattern
```swift
public struct MetalBuffer: Sendable {
    let buffer: MTLBuffer
    let length: Int
    
    // Safe access pattern
    func withContents<T>(_ body: (UnsafeBufferPointer<Float>) throws -> T) rethrows -> T {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: length)
        let buffer = UnsafeBufferPointer(start: ptr, count: length)
        return try body(buffer)
    }
}
```

---

## Phase 3: Storage Layer Refactoring

### 3.1 Replace `UnifiedVectorStorage` with Actor
Current implementation uses `NSLock` which can cause actor isolation issues.

```swift
// Instead of NSLock-based class:
public actor UnifiedVectorStorage {
    private var storage: ContiguousArray<Float>
    private let dimension: Int
    private let capacity: Int
    
    public func withUnsafeBufferPointer<T>(
        _ body: (UnsafeBufferPointer<Float>) throws -> T
    ) async rethrows -> T {
        try storage.withUnsafeBufferPointer(body)
    }
    
    public func getVector(at index: Int) async -> [Float]? {
        guard index >= 0 && index < capacity else { return nil }
        let start = index * dimension
        let end = start + dimension
        return Array(storage[start..<end])
    }
}
```

### 3.2 Sendable Wrappers for Unsafe Operations
```swift
public struct SafeVectorReference: Sendable {
    private let data: Data // Instead of raw pointers
    public let dimension: Int
    public let count: Int
    
    init(copying buffer: UnsafeBufferPointer<Float>) {
        self.data = Data(bytes: buffer.baseAddress!, 
                        count: buffer.count * MemoryLayout<Float>.size)
        self.dimension = /* calculate */
        self.count = /* calculate */
    }
    
    func withUnsafeBufferPointer<T>(
        _ body: (UnsafeBufferPointer<Float>) throws -> T
    ) rethrows -> T {
        try data.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: Float.self)
            return try body(buffer)
        }
    }
}
```

---

## Phase 4: Kernel Wrapper Updates

### 4.1 Proper Actor Isolation
```swift
public actor HNSWSearchLayerKernel {
    private let device: MetalDevice
    nonisolated private let pipeline: MTLComputePipelineState
    
    // Configuration (immutable, thus safe)
    nonisolated private let configuration: SearchConfiguration
    
    // Actor-isolated execution
    public func execute(
        query: [Float],
        graph: GraphStructure,
        visited: VisitedSet
    ) async throws -> SearchResults {
        // All mutable state access here
        // Create command buffer
        // Execute kernel
        // Wait for completion
        // Return results
    }
}
```

### 4.2 Result Types Must Be Sendable
```swift
public struct SearchLayerResult: Sendable {
    public let candidateIDs: [UInt32]
    public let distances: [Float]
    public let visitedCount: Int
    // No raw pointers or non-Sendable types
}

public struct KernelExecutionMetrics: Sendable {
    public let executionTime: TimeInterval
    public let memoryUsed: Int
    public let threadsDispatched: Int
}
```

---

## Phase 5: Index Protocol Updates

### 5.1 Ensure All Protocol Requirements Are Sendable
```swift
public protocol VectorIndexProtocol: Actor {
    // Filter must be Sendable
    func search(
        query: [Float], 
        k: Int, 
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [SearchResult]
    
    func add(
        id: VectorID, 
        vector: [Float], 
        metadata: [String: String]?
    ) async throws
    
    func delete(id: VectorID) async throws
}
```

### 5.2 AccelerableIndex Compliance
```swift
public protocol AccelerableIndex: VectorIndexProtocol {
    func getCandidates(limit: Int) async -> AccelerationCandidates
}

// Ensure AccelerationCandidates is fully Sendable
public struct AccelerationCandidates: Sendable {
    public let vectorStorage: SafeVectorReference // Not raw pointers
    public let metadata: [String: SendableValue]
    public let indexStructure: SendableIndexStructure
}

public enum SendableValue: Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case bool(Bool)
    case data(Data)
}
```

---

## Phase 6: Configuration & Build Settings

### 6.1 Package.swift Updates
```swift
// Already have:
.enableExperimentalFeature("StrictConcurrency")

// Consider adding for gradual migration:
swiftSettings: [
    .unsafeFlags([
        "-Xfrontend", "-warn-concurrency",
        "-Xfrontend", "-enable-actor-data-race-checks"
    ], .when(configuration: .debug))
]
```

### 6.2 Conditional Compilation for Migration
```swift
#if compiler(>=6.0)
    // Swift 6 compliant code
    public actor VectorStorage { }
#else
    // Legacy code with warnings
    @available(*, deprecated, message: "Will be replaced with actor in Swift 6")
    public class VectorStorage { }
#endif
```

---

## Phase 7: Documentation & Guidelines

### 7.1 Add Concurrency Documentation
```swift
/// Thread-safe vector index using actor isolation.
/// 
/// ## Concurrency
/// This index is fully thread-safe and can be accessed from multiple
/// concurrent contexts. All operations are isolated to the actor's
/// serial executor.
///
/// ## Example
/// ```swift
/// let index = HNSWIndex()
/// 
/// // Safe concurrent access
/// await withTaskGroup(of: Void.self) { group in
///     group.addTask { await index.add(...) }
///     group.addTask { await index.search(...) }
/// }
/// ```
public actor HNSWIndex { }
```

### 7.2 Migration Guide
Create `MIGRATION_GUIDE.md`:
- Step-by-step instructions for users
- Breaking changes documentation
- Performance implications
- Code examples for common patterns

---

## Implementation Priority

### Critical (Blocking Compilation)
1. Fix test type conversions (Float vs Double)
2. Fix async test patterns
3. Update API usage for changed properties
4. Fix closure signatures

### High (Core Functionality)
1. Add `@preconcurrency` to Metal imports
2. Mark Metal resources as `nonisolated`
3. Fix `UnifiedVectorStorage` actor isolation
4. Ensure all result types are Sendable

### Medium (Clean Architecture)
1. Refactor storage layer to actors
2. Update kernel wrappers with proper isolation
3. Add SafeVectorReference wrapper
4. Document concurrency boundaries

### Low (Polish)
1. Add comprehensive documentation
2. Create migration guides
3. Add performance monitoring
4. Create example applications

---

## Testing Strategy

### 1. Enable Strict Concurrency Checking
```swift
// In test targets
.testTarget(
    name: "VectorIndexTests",
    dependencies: ["VectorIndex"],
    swiftSettings: [
        .enableExperimentalFeature("StrictConcurrency")
    ]
)
```

### 2. Add Actor Isolation Tests
```swift
func testConcurrentAccess() async throws {
    let index = HNSWIndex()
    
    // Verify no data races
    await withTaskGroup(of: Void.self) { group in
        for i in 0..<100 {
            group.addTask {
                await index.add(id: "\(i)", vector: [...])
            }
        }
    }
}
```

### 3. Thread Sanitizer
```bash
# Run tests with Thread Sanitizer
swift test --sanitize=thread
```

### 4. Benchmark Actor Contention
```swift
func benchmarkActorContention() async {
    // Measure performance under concurrent load
    // Compare with NSLock-based implementation
}
```

---

## Success Metrics

1. **Zero Compilation Warnings**: No concurrency-related warnings
2. **All Tests Pass**: Including with Thread Sanitizer
3. **No Performance Regression**: Actor-based implementation maintains performance
4. **Clear Documentation**: Users understand concurrency model
5. **Smooth Migration**: Existing users can upgrade without major refactoring

---

## Timeline Estimate

- **Phase 1**: 1-2 days (immediate fixes)
- **Phase 2-3**: 3-4 days (Metal and storage refactoring)
- **Phase 4-5**: 2-3 days (kernel and protocol updates)
- **Phase 6-7**: 2-3 days (configuration and documentation)
- **Testing**: 2-3 days

**Total**: ~2-3 weeks for complete implementation

---

## Notes

- EmbedKit's patterns have been battle-tested and should be adopted
- The `@preconcurrency` import is crucial for Metal framework compatibility
- Actor isolation provides better guarantees than lock-based approaches
- Performance impact should be minimal with proper design
- Consider gradual migration path for existing users
<!-- moved to docs/migration-docs/ -->
