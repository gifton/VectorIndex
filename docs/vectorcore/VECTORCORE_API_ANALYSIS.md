# VectorCore API Patterns & Architecture Analysis
## Comprehensive Technical Report for VectorIndex Synchronization

---

## EXECUTIVE SUMMARY

VectorCore is a high-performance, type-safe Swift framework for vector operations with the following core characteristics:

- **Protocol-First Design**: Heavy use of composition and protocol extensions for extensibility
- **Generic Dimension System**: Compile-time dimension checking (Dim2...Dim3072) + runtime DynamicDimension
- **Error Handling First-Class**: Rich error context, chaining, and categorization with ErrorBuilder pattern
- **Async-First Operations**: Modern Swift Concurrency with actor-based configuration
- **Sendable-Everywhere**: Thread-safe by default with strict concurrency support
- **No GPU in Core**: CPU-only focus with hooks for hardware acceleration in separate packages

---

## 1. API PATTERNS & CONVENTIONS

### 1.1 Namespace Organization

VectorCore uses a **namespace enum pattern** for the main entry point:

```swift
public enum VectorCore {
    public struct Configuration: Sendable { ... }
    public static func createVector(dimension: Int, data: [Float]? = nil) -> any VectorType
    public static func createBatch(dimension: Int, from data: [[Float]]) throws -> [any VectorType]
    public static var configuration: Configuration
    public static var version: String
}
```

**Pattern to adopt in VectorIndex:**
- Use `VectorIndex` enum as main namespace (if not already)
- Group static factory methods and configuration here
- Expose global configuration through this entry point

### 1.2 Protocol Composition Hierarchy

VectorCore establishes clear protocol layers:

```
VectorProtocol (core interface)
    ↓
VectorType (factory protocol, extends VectorProtocol)
    ↓
VectorCoreOperations (additional operations)
    ↓
OptimizedVector (SIMD-specific optimizations)
```

**Key Traits of VectorProtocol:**
- Minimal required implementation (storage, scalarCount, toArray(), init variants)
- Rich functionality through protocol extensions
- Hashable, Codable, Collection conformance
- Associated types: Scalar, Storage

### 1.3 Error Handling Pattern

VectorCore uses a **hierarchical error system**:

```swift
// Error Kind (categorical)
public enum ErrorKind: String, CaseIterable, Sendable {
    case dimensionMismatch, invalidDimension, invalidData, ...
}

// Error Context (rich debugging info)
public struct ErrorContext: Sendable {
    public let file: StaticString        // Debug builds only
    public let line: UInt
    public let function: StaticString
    public let timestamp: Date
    public let additionalInfo: [String: String]
}

// Rich Error Type
public struct VectorError: Error, Sendable {
    public let kind: ErrorKind
    public let context: ErrorContext
    public let underlyingError: (any Error)?
    public var errorChain: [VectorError]  // Root cause analysis
}
```

**Builder Pattern for Errors:**

```swift
throw ErrorBuilder(.dimensionMismatch)
    .message("Cannot add vectors of different sizes")
    .dimension(expected: 128, actual: 256)
    .build()
```

**VectorIndex Adoption:**
- Consider using VectorError types for consistency
- Implement similar error categorization (dimension, bounds, data, operation, resource, config, system)
- Add error chaining for kernel debugging

### 1.4 Configuration Management Pattern

**Thread-Safe Configuration via Actor:**

```swift
public actor ThreadSafeConfiguration<T: Sendable> {
    public func get() -> T
    public func update(_ newValue: T)
    public func update<V>(_ keyPath: WritableKeyPath<T, V>, to newValue: V)
}

// Usage in BatchOperations
private static let _configuration = ThreadSafeConfiguration(Configuration())

public static func configuration() async -> Configuration {
    await _configuration.get()
}
```

**Pattern Adoption:** VectorIndex kernels and index implementations should expose configuration through similar actor-based interfaces.

---

## 2. TYPE SYSTEM & DIMENSION HANDLING

### 2.1 Dimension Specification Types

**Static Dimensions (Compile-time checked):**

```swift
public protocol StaticDimension: Sendable {
    static var value: Int { get }
    associatedtype Storage: VectorStorage
}

// Predefined: Dim2, Dim3, Dim4, Dim8, Dim16, Dim32, Dim64, Dim128, Dim256, Dim512, Dim768, Dim1024, Dim1536, Dim2048, Dim3072
public struct Dim128: StaticDimension {
    public static let value = 128
    public typealias Storage = DimensionStorage<Dim128, Float>
}
```

**Dynamic Dimensions (Runtime determined):**

```swift
public struct DynamicDimension {
    public let size: Int
    public init(_ size: Int)
    public init(validating size: Int) throws
    public static func make(_ size: Int) throws -> DynamicDimension
}
```

**Factory Pattern for Type Selection:**

```swift
public enum VectorTypeFactory {
    public static func create<D: StaticDimension>(_ type: D.Type, from values: [Float]) throws -> Vector<D>
    
    public static func vector(of dimension: Int, from values: [Float]) throws -> any VectorType
    // Returns Vector<Dim128>, <Dim256>, etc. or DynamicVector based on dimension
    
    public static func zeros(dimension: Int) -> any VectorType
    public static func isSupported(dimension: Int) -> Bool
    public static func optimalDimension(for size: Int) -> Int
}
```

**Key Insight:** VectorCore prioritizes compile-time safety when possible, falls back to runtime-safe alternatives.

### 2.2 Vector Type Hierarchy

```
Vector<D: StaticDimension>         // Fixed-dimension, optimized
    ↓
DynamicVector                       // Runtime-determined dimension
    ↓
any VectorType (protocol)           // Unified interface
    ↓
any VectorProtocol (protocol)       // Core operations
```

---

## 3. COMMON COMPONENTS

### 3.1 Distance Metrics

**Protocol Definition:**

```swift
public protocol DistanceMetric<Scalar>: Sendable {
    associatedtype Scalar: BinaryFloatingPoint
    
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Scalar where V.Scalar == Scalar
    var name: String { get }
    var identifier: String { get }
}
```

**Built-in Implementations:**

```swift
public struct EuclideanDistance: DistanceMetric {
    public typealias Scalar = Float
    public var name: String { "euclidean" }
    @inlinable
    public func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float { ... }
    public func batchDistance<Vector: VectorProtocol>(query: Vector, candidates: [Vector]) -> [DistanceScore]
}

public struct CosineDistance: DistanceMetric { ... }
// Additional: DotProduct, Manhattan, Chebyshev
```

**Supported Metrics Enum:**

```swift
public enum SupportedDistanceMetric: String, CaseIterable, Sendable {
    case euclidean, cosine, dotProduct, manhattan, chebyshev
    
    public var displayName: String { ... }
}
```

**VectorIndex Note:** Uses SupportedDistanceMetric for API compatibility but implements faster unsafe pointer versions internally.

### 3.2 Storage Abstraction

**Core Protocol:**

```swift
public protocol VectorStorage: Sendable {
    associatedtype Element: BinaryFloatingPoint & Hashable & Codable
    
    var count: Int { get }
    subscript(index: Int) -> Element { get set }
}
```

**Implementations:**
- `DimensionStorage<D: StaticDimension, T>` - Fixed-size, optimized
- `ArrayStorage<D>` - Flexible, dynamic size
- `DimensionSpecificStorage` - Specialized implementations
- `AlignedValueStorage`, `AlignedDynamicArrayStorage` - SIMD-aligned memory
- `SoA` (Structure of Arrays) - Cache-optimized layout

### 3.3 Serialization

**Protocol:**

```swift
public protocol VectorSerializable {
    associatedtype SerializedForm
    func serialize() throws -> SerializedForm
    static func deserialize(from: SerializedForm) throws -> Self
}
```

**Binary Format:**

```swift
public struct BinaryHeader {
    let magic: UInt32
    let version: UInt16
    let dimension: UInt32
    let flags: UInt16
}
```

**Key Pattern:** Versioning + checksums (CRC32) for data integrity.

---

## 4. MODULE STRUCTURE

### 4.1 Directory Organization

```
Sources/VectorCore/
├── Core/                       # Core concepts
│   ├── Dimension.swift         # Dim2...Dim3072, DynamicDimension
│   └── Operators.swift         # .*, ./  operators
├── Protocols/                  # All public protocols
│   ├── VectorProtocol.swift    # Core vector interface
│   ├── DistanceMetric.swift    # Distance metric protocol
│   ├── CoreProtocols.swift     # AccelerationProvider, VectorSerializable
│   ├── ProviderProtocols.swift # SupportedDistanceMetric enum
│   ├── ComputeProvider.swift   # Hardware acceleration providers
│   └── ...
├── Vectors/                    # Vector implementations
│   ├── Vector.swift            # Vector<D> generic
│   ├── DynamicVector.swift     # Runtime-sized vector
│   ├── Vector512Optimized.swift # Specialized implementations
│   └── VectorSerialization.swift
├── Operations/                 # Vector operations
│   ├── BatchOperations.swift   # Async-first batch processing
│   ├── DistanceMetrics.swift   # Concrete distance metric implementations
│   ├── Operations.swift        # Core operations
│   ├── Kernels/                # Specialized kernels
│   │   ├── EuclideanKernels.swift
│   │   ├── CosineKernels.swift
│   │   └── ...
│   └── ...
├── Storage/                    # Storage implementations
├── Platform/                   # Platform-specific optimizations
│   ├── SIMDProvider.swift      # SIMD abstraction
│   ├── SIMDOperations.swift    # Platform-specific SIMD ops
│   └── AccelerateSIMDProvider.swift
├── Configuration/              # Configuration management
├── Execution/                  # ComputeDevice enum
├── Errors/                     # VectorError system
├── Logging/                    # Logger, LogLevel
├── Utilities/                  # MemoryPool, etc.
└── VectorCore.swift            # Main namespace enum
```

### 4.2 Public vs Internal Boundaries

**Hard Rules:**
- All protocols: public
- All error types: public
- All dimension types: public
- All distance metrics: public
- Configuration and logging: public
- Platform-specific implementations: internal (protocols public, implementations internal)
- Kernels: internal (organized but not directly exposed)

---

## 5. ERROR HANDLING PATTERNS

### 5.1 Error Categories (for analytics)

```swift
public enum ErrorKind: String, CaseIterable, Sendable {
    // Dimension errors
    case dimensionMismatch, invalidDimension, unsupportedDimension
    
    // Index errors
    case indexOutOfBounds, invalidRange
    
    // Data errors
    case invalidData, dataCorruption, insufficientData
    
    // Operation errors
    case invalidOperation, unsupportedOperation, operationFailed
    
    // Resource errors
    case allocationFailed, resourceExhausted, resourceUnavailable
    
    // Configuration errors
    case invalidConfiguration, missingConfiguration
    
    // System errors
    case systemError, unknown
}

// Severity Mapping
var severity: ErrorSeverity { ... }  // critical, high, medium, low, info

// Category Mapping
var category: ErrorCategory { ... }  // dimension, bounds, data, operation, resource, config, system
```

### 5.2 Convenience Factory Methods

```swift
// These are static methods on VectorError:
static func dimensionMismatch(expected: Int, actual: Int, ...) -> VectorError
static func indexOutOfBounds(index: Int, dimension: Int, ...) -> VectorError
static func invalidOperation(_ operation: String, reason: String, ...) -> VectorError
static func invalidData(_ description: String, ...) -> VectorError
static func allocationFailed(size: Int, reason: String?, ...) -> VectorError
static func invalidDimension(_ dimension: Int, reason: String, ...) -> VectorError
static func divisionByZero(operation: String, ...) -> VectorError
static func zeroVectorError(operation: String, ...) -> VectorError
// ... many more
```

### 5.3 VectorIndex Adoption Recommendations

Create parallel error system in VectorIndex:

```swift
public enum VectorIndexError: Error, Sendable {
    case indexOperationFailed(reason: String)
    case invalidQuery(reason: String)
    case vectorNotFound(id: VectorID)
    case persistenceError(reason: String)
    // ... additional index-specific errors
}
```

---

## 6. TESTING PATTERNS

### 6.1 Test Organization in VectorCore

```
Tests/
├── MinimalTests/               # Fast, essential tests
│   ├── BasicVectorTests.swift
│   ├── ErrorHandlingTests.swift
│   └── ...
├── ComprehensiveTests/         # Extended test coverage
│   ├── VectorTypeFactoryTests.swift
│   ├── VectorSerializationTests.swift
│   ├── OperationsValidationTests.swift
│   ├── MixedPrecisionTests.swift (multiple phases)
│   ├── KernelAutoTunerTests.swift
│   ├── MemoryPoolTests.swift
│   └── TestSupport.swift       # Common utilities
└── Benchmarks/
    └── VectorCoreBench/
```

### 6.2 Key Test Patterns

1. **Test Helpers:** Shared utilities in TestSupport.swift
2. **Mock Patterns:** ErrorHandlingMocks for error testing
3. **Validation Tests:** Separate test suite for correctness verification
4. **Performance Benchmarks:** Separate benchmark targets
5. **Edge Case Tests:** Specific test files for boundary conditions

### 6.3 Test Naming Conventions

- `{Component}Tests.swift` - Standard tests
- `{Component}BenchmarkTests.swift` - Performance tests
- `{Component}ValidationTests.swift` - Correctness validation
- `{Component}EdgeCaseTests.swift` - Boundary conditions

---

## 7. LOGGING PATTERNS

### 7.1 Logger Architecture

```swift
public enum LogLevel: Int, Comparable, Sendable {
    case debug = 0
    case info = 1
    case warning = 2
    case error = 3
    case critical = 4
}

public struct Logger: Sendable {
    public init(subsystem: String = "com.vectorcore", category: String)
    
    public func debug(_ message: @autoclosure () -> String, ...)
    public func info(_ message: @autoclosure () -> String, ...)
    public func warning(_ message: @autoclosure () -> String, ...)
    public func error(_ message: @autoclosure () -> String, ...)
    public func critical(_ message: @autoclosure () -> String, ...)
    public func error(_ error: VectorError, message: String?, ...)
}

public final class LogConfiguration: @unchecked Sendable {
    public var minimumLevel: LogLevel { get set }
}
```

### 7.2 Global Loggers

```swift
public let coreLogger = Logger(category: "Core")
public let storageLogger = Logger(category: "Storage")
public let batchLogger = Logger(category: "Batch")
public let metricsLogger = Logger(category: "Metrics")
public let performanceLogger = Logger(category: "Performance")
```

### 7.3 Performance Timer

```swift
public struct PerformanceTimer {
    public init(operation: String, logger: Logger = performanceLogger)
    public func log(threshold: TimeInterval = 0.001)
}
```

---

## 8. CONCURRENCY MODEL

### 8.1 Async-First Operations

```swift
public enum BatchOperations {
    public struct Configuration: Sendable {
        public var parallelThreshold: Int = 1000
        public var oversubscription: Double = 2.0
        public var minimumChunkSize: Int = 256
        public var defaultBatchSize: Int = 1024
    }
    
    public static func findNearest<V: VectorProtocol & Sendable, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M = EuclideanDistance()
    ) async -> [(index: Int, distance: Float)] where V.Scalar == M.Scalar, M.Scalar == Float
    
    public static func pairwiseDistances<M: DistanceMetric>(
        _ vectors: [Vector],
        metric: M = EuclideanDistance()
    ) async -> [[Float]]
}
```

### 8.2 Actor-Based Configuration

All mutable configuration exposed through actor interface for thread safety:

```swift
// Get configuration
let config = await BatchOperations.configuration()

// Update configuration
await BatchOperations.updateConfiguration { config in
    config.parallelThreshold = 500
}
```

### 8.3 Auto-Parallelization Heuristics

- Thresholds configurable per operation
- Automatic choice between serial and parallel execution
- Chunk sizing based on CPU cores and cache
- `ParallelHeuristic` system for intelligent decisions

---

## 9. NAMING CONVENTIONS

### 9.1 Type Naming

```swift
// Protocols: PascalCase + "Protocol" suffix (optional, often omitted)
VectorProtocol, DistanceMetric, VectorStorage

// Structs: PascalCase
Vector<D>, DynamicVector, SearchResult, IndexStats

// Enums: PascalCase
VectorCore, LogLevel, ComputeDevice, SupportedDistanceMetric

// Type Aliases: lowercase
typealias VectorID = String
typealias Timestamp = UInt64
typealias DistanceScore = Float
```

### 9.2 Method Naming

```swift
// Factory methods: create, make, or direct functions
VectorTypeFactory.vector(of:from:)
VectorCore.createVector(dimension:data:)
DynamicDimension.make(_:)

// Query methods: get, is, has
func configuration() async -> Configuration
var isFinite: Bool
func hasOptimizedSupport(for dimension: Int) -> Bool

// Operations: verb-based
func distance(to:metric:) -> Float
func normalized() -> Result<Self, VectorError>
func dotProduct(_ other: Self) -> Scalar
```

### 9.3 Batch Operation Naming

```swift
// Batch methods often follow: operation + Target pattern
BatchOperations.findNearest(to:in:k:metric:)
BatchOperations.pairwiseDistances(_:metric:)
metric.batchDistance(query:candidates:)
```

---

## 10. KEY DESIGN PRINCIPLES

### 10.1 Protocol Composition Over Inheritance

- Use generic protocols with associated types
- Provide protocol extensions for shared implementation
- Keep required implementations minimal

### 10.2 Safety and Type-Safety

- Sendable everywhere (thread-safe by default)
- Codable support for serialization
- Hashable for use in collections
- Result types or throwing functions, never silent failures

### 10.3 Error Context Richness

- Every error includes source location (debug builds)
- Timestamp for debugging
- Rich additionalInfo dictionary
- Error chaining for root cause analysis

### 10.4 Performance by Default

- SIMD operations abstracted through platform-specific providers
- Inline hints for hot paths
- Unsafe buffer access for performance-critical code
- But always safe by default (require opt-in to unsafe)

### 10.5 Configuration Flexibility

- Global configuration through actor-based system
- Auto-tuning capabilities for parallelization decisions
- Compile-time optimization flags

---

## 11. VECTORINDEX SYNCHRONIZATION CHECKLIST

### 11.1 High Priority (Immediate Adoption)

- [ ] **Error Handling:** Implement error categorization (dimensionMismatch, bounds, data, operation, resource, config, system)
- [ ] **Logging:** Use similar Logger architecture with categories (IndexCore, IndexOperations, IndexPersistence)
- [ ] **Configuration:** Expose index-specific settings through actor-based ThreadSafeConfiguration
- [ ] **Distance Metrics:** Use VectorCore's DistanceMetric protocol for consistency
- [ ] **Type Aliases:** Define VectorID consistently (already using VectorCore's)

### 11.2 Medium Priority (Next Phase)

- [ ] **Dimension Handling:** Leverage VectorCore's dimension types for metadata
- [ ] **Batch Operations:** Match async-first patterns with TaskGroup-based parallelization
- [ ] **Sendable Types:** Ensure all public types conform to Sendable
- [ ] **Result Types:** Use Result<Success, VectorIndexError> instead of throws where appropriate
- [ ] **Testing:** Reorganize tests into MinimalTests and ComprehensiveTests categories

### 11.3 Lower Priority (Nice-to-Have)

- [ ] **Protocol Composition:** Define IndexProtocol with rich extensions
- [ ] **Storage Abstraction:** Consider storage protocols for index backends
- [ ] **Serialization:** Implement VectorSerializable for index structures
- [ ] **Performance Metrics:** Add performance logging similar to PerformanceTimer
- [ ] **Auto-Tuning:** Implement heuristics for index operation parallelization

---

## 12. SPECIFIC CODE PATTERNS TO ADOPT

### 12.1 Factory Methods Pattern

```swift
// From: VectorTypeFactory.swift
public static func create<D: StaticDimension>(_ type: D.Type, from values: [Float]) throws -> Vector<D> {
    guard values.count == D.value else {
        throw VectorError.dimensionMismatch(expected: D.value, actual: values.count)
    }
    return try Vector<D>(values)
}

// Apply to VectorIndex:
public static func create(type: IndexType, dimension: Int, metric: SupportedDistanceMetric) throws -> any VectorIndexProtocol {
    guard dimension > 0 else {
        throw VectorIndexError.invalidConfiguration(reason: "Dimension must be positive")
    }
    // ... implementation
}
```

### 12.2 ErrorBuilder Pattern

```swift
// From: VectorError.swift
throw ErrorBuilder(.dimensionMismatch)
    .message("Cannot add vectors of different sizes")
    .dimension(expected: 128, actual: 256)
    .build()

// Apply to VectorIndex:
throw IndexErrorBuilder(.operationFailed)
    .message("Search failed due to corruption")
    .parameter("nlist", value: String(nlist))
    .parameter("nprobe", value: String(nprobe))
    .build()
```

### 12.3 Protocol Extension Pattern

```swift
// From: VectorProtocol extensions (multiple files)
public extension VectorProtocol {
    var magnitude: Scalar { Foundation.sqrt(dotProduct(self)) }
    var magnitudeSquared: Scalar { dotProduct(self) }
    var sum: Scalar { ... }
    var mean: Scalar { ... }
}

// Apply to VectorIndex:
public extension VectorIndexProtocol {
    func searchStats() async -> SearchStatistics { ... }
    func validate() async throws -> ValidationResult { ... }
}
```

### 12.4 Async Batch Processing Pattern

```swift
// From: BatchOperations.swift
public static func findNearest<V: VectorProtocol & Sendable, M: DistanceMetric>(
    to query: V,
    in vectors: [V],
    k: Int,
    metric: M = EuclideanDistance()
) async -> [(index: Int, distance: Float)] {
    // Smart parallelization
    let shouldParallelize = vectors.count >= config.parallelThreshold
    if shouldParallelize {
        // TaskGroup-based parallel implementation
    } else {
        // Serial implementation
    }
}

// Apply to VectorIndex:
public func batchSearch(
    queries: [[Float]],
    k: Int,
    filter: (@Sendable ([String: String]?) -> Bool)?
) async throws -> [[SearchResult]] {
    let shouldParallelize = queries.count >= self.batchParallelThreshold
    // Implement accordingly
}
```

---

## 13. CONSISTENCY CHECKPOINTS

### 13.1 Public API Alignment

Compare VectorIndex public types with VectorCore:

```swift
// VectorCore provides:
- VectorID (typealias)
- SupportedDistanceMetric (enum)
- VectorError (struct with rich context)
- DistanceMetric<Scalar> (protocol)

// VectorIndex should leverage or mirror:
- VectorIndexProtocol (actor-based, like VectorCore's protocols)
- SearchResult, IndexStats (already defined)
- Potential: VectorIndexError (similar hierarchy to VectorError)
```

### 13.2 Import Dependencies

```swift
// VectorIndex already correctly imports VectorCore
import VectorCore

// Uses:
- VectorID (from VectorCore)
- SupportedDistanceMetric (from VectorCore)
- VectorProtocol (where needed in generics)
```

### 13.3 Extension Points for Consistency

```swift
// VectorCore-compatible extensions in VectorIndex

public extension VectorIndexProtocol {
    // Add convenience methods
    func searchAsync(query: [Float], k: Int) async throws -> [SearchResult] {
        try await search(query: query, k: k, filter: nil)
    }
}

public extension SearchResult {
    // Add utility methods
    var isValid: Bool { score >= 0 }
}
```

---

## 14. RECOMMENDATIONS SUMMARY

### Highest Impact Changes

1. **Error Categorization**: Adopt VectorCore's error kind + severity + category system
2. **Logging Infrastructure**: Implement similar Logger architecture with categories
3. **Thread-Safe Configuration**: Use actor-based ThreadSafeConfiguration for mutable settings
4. **Sendable Compliance**: Ensure all public types conform to Sendable
5. **Async Batch Operations**: Restructure batch operations to match VectorCore's async-first pattern

### Architectural Consistency

1. **Protocol-First Design**: Use protocol composition for index backends (FlatIndex, IVFIndex, HNSWIndex as implementations of protocol)
2. **Factory Methods**: Provide centralized factory for index creation with dimension validation
3. **Result Types**: Prefer Result<T, VectorIndexError> with proper error chaining
4. **Type Aliases**: Continue using VectorID, SupportedDistanceMetric from VectorCore

### Code Organization

1. **Keep Current Structure**: VectorIndex's directory organization is already sound
2. **Enhance Documentation**: Add rich comments similar to VectorCore's style
3. **Test Organization**: Consider splitting tests into MinimalTests and ComprehensiveTests
4. **Public API Boundaries**: Ensure kernels remain internal, only expose protocols

---

## REFERENCES

**VectorCore Files Examined:**
- VectorCore.swift (main namespace)
- Protocols: VectorProtocol.swift, DistanceMetric.swift, CoreProtocols.swift, ProviderProtocols.swift
- Errors: Errors/VectorError.swift
- Logging: Logging/Logger.swift
- Configuration: Configuration/ThreadSafeConfiguration.swift, Execution/ComputeDevice.swift
- Operations: Operations/BatchOperations.swift, Operations/DistanceMetrics.swift
- Factory: Factory/VectorTypeFactory.swift
- Dimensions: Core/Dimension.swift
- Vectors: Vectors/Vector.swift, Vectors/DynamicVector.swift

**VectorIndex Files Cross-Referenced:**
- IndexProtocols.swift
- Package.swift (dependency declaration)
- FlatIndex.swift, HNSWIndex.swift, IVFIndex.swift
<!-- moved to docs/vectorcore/ -->
