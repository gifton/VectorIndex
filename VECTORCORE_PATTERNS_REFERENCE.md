# VectorCore Patterns Quick Reference
## Implementation Patterns for VectorIndex

---

## PATTERN 1: Namespace Enum Entry Point

**VectorCore Pattern:**
```swift
public enum VectorCore {
    public struct Configuration: Sendable { ... }
    public static func createVector(dimension: Int, data: [Float]? = nil) -> any VectorType
    public static var version: String
}

// Usage
let vector = VectorCore.createVector(dimension: 128)
let config = VectorCore.configuration
```

**VectorIndex Application:**
```swift
// In a new file: VectorIndexNamespace.swift
public enum VectorIndex {
    /// Get the version of VectorIndex
    public static var version: String { "0.1.0" }
    
    /// Global configuration for all indices
    public static let configuration = IndexConfiguration()
    
    /// Create an index of the specified type
    public static func createIndex(
        type: IndexType,
        dimension: Int,
        metric: SupportedDistanceMetric
    ) throws -> any VectorIndexProtocol {
        // Delegate to factory
        try IndexFactory.create(type: type, dimension: dimension, metric: metric)
    }
}
```

---

## PATTERN 2: Error Hierarchy with Builder

**VectorCore Pattern:**
```swift
public enum ErrorKind: String, CaseIterable, Sendable {
    case dimensionMismatch, indexOutOfBounds, invalidData, ...
}

public struct VectorError: Error, Sendable {
    public let kind: ErrorKind
    public let context: ErrorContext
    public let underlyingError: (any Error)?
    public var errorChain: [VectorError]
}

public struct ErrorBuilder {
    public func message(_ msg: String) -> ErrorBuilder
    public func dimension(expected: Int, actual: Int) -> ErrorBuilder
    public func parameter(_ name: String, value: String) -> ErrorBuilder
    public func build() -> VectorError
}

// Usage
throw ErrorBuilder(.dimensionMismatch)
    .message("Incompatible dimensions")
    .dimension(expected: 128, actual: 256)
    .build()
```

**VectorIndex Application:**
```swift
// In new file: Sources/VectorIndex/Errors/IndexError.swift
public enum IndexErrorKind: String, CaseIterable, Sendable {
    // Index-specific errors
    case indexCorrupted
    case invalidConfiguration
    case persistenceFailed
    case searchFailed
    case graphBuildFailed
    
    var severity: ErrorSeverity {
        switch self {
        case .indexCorrupted, .graphBuildFailed: return .critical
        case .persistenceFailed: return .high
        case .searchFailed: return .medium
        case .invalidConfiguration: return .low
        }
    }
    
    var category: ErrorCategory {
        switch self {
        case .indexCorrupted: return .data
        case .invalidConfiguration: return .configuration
        case .persistenceFailed: return .operation
        case .searchFailed: return .operation
        case .graphBuildFailed: return .operation
        }
    }
}

public struct IndexError: Error, Sendable, CustomStringConvertible {
    public let kind: IndexErrorKind
    public let context: ErrorContext
    public let underlyingError: (any Error)?
    public var errorChain: [IndexError] = []
    
    public var description: String {
        var result = "IndexError.\(kind.rawValue)"
        if let msg = context.additionalInfo["message"] {
            result += ": \(msg)"
        }
        return result
    }
    
    public static func corrupted(reason: String) -> IndexError {
        IndexError(
            kind: .indexCorrupted,
            context: ErrorContext(additionalInfo: ["message": reason])
        )
    }
}

public struct IndexErrorBuilder {
    private var kind: IndexErrorKind
    private var info: [String: String] = [:]
    
    public init(_ kind: IndexErrorKind) {
        self.kind = kind
    }
    
    @discardableResult
    public func message(_ msg: String) -> IndexErrorBuilder {
        var copy = self
        copy.info["message"] = msg
        return copy
    }
    
    @discardableResult
    public func parameter(_ name: String, value: String) -> IndexErrorBuilder {
        var copy = self
        copy.info[name] = value
        return copy
    }
    
    public func build() -> IndexError {
        IndexError(
            kind: kind,
            context: ErrorContext(additionalInfo: info)
        )
    }
}

// Usage
throw IndexErrorBuilder(.graphBuildFailed)
    .message("Failed to build HNSW graph")
    .parameter("nlist", value: "1000")
    .build()
```

---

## PATTERN 3: Thread-Safe Configuration via Actor

**VectorCore Pattern:**
```swift
public actor ThreadSafeConfiguration<T: Sendable> {
    private var value: T
    
    public func get() -> T { value }
    public func update(_ newValue: T) { value = newValue }
}

// Usage in BatchOperations
private static let _configuration = ThreadSafeConfiguration(Configuration())

public static func configuration() async -> Configuration {
    await _configuration.get()
}

public static func updateConfiguration(_ update: (inout Configuration) -> Void) async {
    var config = await _configuration.get()
    update(&config)
    await _configuration.update(config)
}
```

**VectorIndex Application:**
```swift
// In new file: Sources/VectorIndex/Configuration/IndexConfiguration.swift
public struct IndexConfiguration: Sendable {
    /// Enable parallel search operations
    public var enableParallelSearch: Bool = true
    
    /// Threshold for enabling parallelization
    public var parallelThreshold: Int = 1000
    
    /// Enable logging
    public var enableLogging: Bool = false
    
    /// Minimum log level
    public var minimumLogLevel: LogLevel = .warning
    
    public init() {}
}

public actor IndexConfigurationManager {
    private static let shared = IndexConfigurationManager()
    private var _configuration: IndexConfiguration
    
    private init() {
        self._configuration = IndexConfiguration()
    }
    
    public static func current() -> IndexConfigurationManager {
        shared
    }
    
    public func get() -> IndexConfiguration {
        _configuration
    }
    
    public func update(_ newValue: IndexConfiguration) {
        _configuration = newValue
    }
    
    public func update<V>(_ keyPath: WritableKeyPath<IndexConfiguration, V>, to value: V) {
        _configuration[keyPath: keyPath] = value
    }
}

// Usage
let config = await IndexConfigurationManager.current().get()
await IndexConfigurationManager.current().update { config in
    config.enableParallelSearch = true
}
```

---

## PATTERN 4: Protocol Extension for Rich Operations

**VectorCore Pattern:**
```swift
public protocol VectorProtocol: Sendable, Hashable, Codable, Collection {
    associatedtype Scalar: BinaryFloatingPoint & Hashable & Codable
    associatedtype Storage: Sendable
    
    var storage: Storage { get set }
    var scalarCount: Int { get }
    
    init()
    init(_ array: [Scalar]) throws
    init(repeating value: Scalar)
    func toArray() -> [Scalar]
    // ... unsafe buffer access
}

// Rich extensions on the protocol
public extension VectorProtocol {
    var magnitude: Scalar { sqrt(dotProduct(self)) }
    var isZero: Bool { withUnsafeBufferPointer { buffer in buffer.allSatisfy { $0 == 0 } } }
    func dotProduct(_ other: Self) -> Scalar { ... }
    func normalized() -> Result<Self, VectorError> { ... }
}
```

**VectorIndex Application:**
```swift
// In: Sources/VectorIndex/IndexProtocols.swift (extend existing)
public extension VectorIndexProtocol {
    /// Convenience method for single vector search
    func searchAsync(
        query: [Float],
        k: Int,
        timeout: TimeInterval? = nil
    ) async throws -> [SearchResult] {
        try await search(query: query, k: k, filter: nil)
    }
    
    /// Get count as Int (convenience)
    var size: Int { count }
    
    /// Check if index is empty
    var isEmpty: Bool { count == 0 }
    
    /// Get current memory usage estimate (optional)
    func estimatedMemoryUsage() async -> Int {
        // Default implementation: return 0
        // Subclasses can override
        0
    }
}
```

---

## PATTERN 5: Async-First Batch Operations

**VectorCore Pattern:**
```swift
public enum BatchOperations {
    public struct Configuration: Sendable {
        public var parallelThreshold: Int = 1000
        public var minimumChunkSize: Int = 256
    }
    
    public static func findNearest<V: VectorProtocol & Sendable, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M = EuclideanDistance()
    ) async -> [(index: Int, distance: Float)] {
        let shouldParallelize = vectors.count >= config.parallelThreshold
        
        if shouldParallelize {
            // Use TaskGroup for parallel execution
        } else {
            // Serial implementation
        }
    }
}
```

**VectorIndex Application:**
```swift
// In: Sources/VectorIndex/Operations/BatchSearch.swift
public actor BatchSearchOperations {
    private let configuration: IndexConfiguration
    
    public init(configuration: IndexConfiguration) {
        self.configuration = configuration
    }
    
    /// Batch search with smart parallelization
    public func batchSearch(
        index: any VectorIndexProtocol,
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[SearchResult]] {
        let shouldParallelize = queries.count >= configuration.parallelThreshold
            && configuration.enableParallelSearch
        
        if shouldParallelize {
            return try await withThrowingTaskGroup(
                of: (Int, [SearchResult]).self
            ) { group in
                for (idx, query) in queries.enumerated() {
                    group.addTask {
                        (idx, try await index.search(query: query, k: k, filter: filter))
                    }
                }
                
                var results = Array(repeating: [SearchResult](), count: queries.count)
                for try await (idx, searchResults) in group {
                    results[idx] = searchResults
                }
                return results
            }
        } else {
            return try await queries.asyncMap { query in
                try await index.search(query: query, k: k, filter: filter)
            }
        }
    }
}
```

---

## PATTERN 6: Logging with Categories

**VectorCore Pattern:**
```swift
public enum LogLevel: Int, Comparable, Sendable {
    case debug, info, warning, error, critical
}

public struct Logger: Sendable {
    public init(subsystem: String = "com.vectorcore", category: String)
    public func debug(_ message: @autoclosure () -> String, ...)
    public func info(_ message: @autoclosure () -> String, ...)
    public func error(_ message: @autoclosure () -> String, ...)
}

public let coreLogger = Logger(category: "Core")
public let performanceLogger = Logger(category: "Performance")
```

**VectorIndex Application:**
```swift
// In: Sources/VectorIndex/Logging/IndexLoggers.swift
import VectorCore

public let indexCoreLogger = Logger(subsystem: "com.vectorindex", category: "IndexCore")
public let indexSearchLogger = Logger(subsystem: "com.vectorindex", category: "Search")
public let indexPersistenceLogger = Logger(subsystem: "com.vectorindex", category: "Persistence")
public let indexBuildLogger = Logger(subsystem: "com.vectorindex", category: "Build")
public let indexPerformanceLogger = Logger(subsystem: "com.vectorindex", category: "Performance")

// Usage
indexSearchLogger.debug("Starting k-NN search with k=\(k)")
indexPerformanceLogger.info("Search completed in \(elapsed)ms")
```

---

## PATTERN 7: Factory Pattern for Type Selection

**VectorCore Pattern:**
```swift
public enum VectorTypeFactory {
    public static func create<D: StaticDimension>(_ type: D.Type, from values: [Float]) throws -> Vector<D>
    
    public static func vector(of dimension: Int, from values: [Float]) throws -> any VectorType {
        switch dimension {
        case 128: return try Vector<Dim128>(values)
        case 256: return try Vector<Dim256>(values)
        // ... etc
        default: return try DynamicVector(dimension: dimension, from: values)
        }
    }
    
    public static func zeros(dimension: Int) -> any VectorType
    public static func isSupported(dimension: Int) -> Bool
}
```

**VectorIndex Application:**
```swift
// In: Sources/VectorIndex/Factory/IndexFactory.swift
public enum IndexType: String, Sendable {
    case flat
    case ivf
    case hnsw
}

public enum IndexFactory {
    public static func create(
        type: IndexType,
        dimension: Int,
        metric: SupportedDistanceMetric
    ) throws -> any VectorIndexProtocol {
        guard dimension > 0 else {
            throw IndexErrorBuilder(.invalidConfiguration)
                .message("Dimension must be positive")
                .parameter("dimension", value: String(dimension))
                .build()
        }
        
        switch type {
        case .flat:
            return FlatIndex(dimension: dimension, metric: metric)
        case .ivf:
            return IVFIndex(dimension: dimension, metric: metric)
        case .hnsw:
            return HNSWIndex(dimension: dimension, metric: metric)
        }
    }
    
    public static func isSupported(_ type: IndexType) -> Bool {
        // Always supported for now
        true
    }
}
```

---

## PATTERN 8: Sendable Type Conformance

**VectorCore Pattern:**
```swift
public protocol VectorProtocol: Sendable, Hashable, Codable, Collection { ... }

public struct Vector<D: StaticDimension>: Sendable { ... }

public enum LogLevel: Int, Comparable, Sendable { ... }

public struct VectorError: Error, Sendable { ... }
```

**VectorIndex Application:**
```swift
// Ensure all public types are Sendable
public struct SearchResult: Sendable, Equatable {
    public let id: VectorID
    public let score: Float
}

public struct IndexStats: Sendable, Equatable {
    public let indexType: String
    public let vectorCount: Int
    public let dimension: Int
    public let metric: SupportedDistanceMetric
    public let details: [String: String]
}

public enum IndexType: String, Sendable { ... }

// Protocols inherit Sendable
public protocol VectorIndexProtocol: Actor { ... }
```

---

## PATTERN 9: Result Types with Proper Error Chain

**VectorCore Pattern:**
```swift
public extension Result where Failure == VectorError {
    func mapErrorContext(_ transform: (VectorError) -> VectorError) -> Result<Success, VectorError> {
        mapError(transform)
    }
    
    func chainError(_ error: VectorError) -> Result<Success, VectorError> {
        mapError { $0.chain(with: error) }
    }
}

// Usage
try operation()
    .mapError { IndexError.wrapVectorError($0) }
    .chainError(contextError)
```

**VectorIndex Application:**
```swift
public extension Result where Failure == IndexError {
    func mapErrorContext(_ transform: (IndexError) -> IndexError) -> Result<Success, IndexError> {
        mapError(transform)
    }
    
    func chainError(_ error: IndexError) -> Result<Success, IndexError> {
        mapError { $0.chain(with: error) }
    }
}

// Usage example
do {
    let results = try await index.search(query: query, k: k, filter: nil)
    return results
} catch let error as IndexError {
    // Already an IndexError
    throw error
} catch let error as VectorError {
    // Wrap VectorError
    throw IndexErrorBuilder(.searchFailed)
        .message("Search operation failed: \(error)")
        .build()
} catch {
    throw IndexError(kind: .searchFailed, context: ErrorContext())
}
```

---

## PATTERN 10: Documentation with Type Signatures

**VectorCore Pattern:**
```swift
/// Euclidean (L2) distance metric with SIMD acceleration
///
/// Computes the L2 norm (Euclidean distance) between two vectors:
/// `distance = sqrt(sum((a_i - b_i)^2))`
///
/// ## Performance Characteristics
/// - Time Complexity: O(n) where n is vector dimension
/// - Space Complexity: O(1)
/// - Uses SIMD acceleration for 4-8x speedup
///
/// ## Example Usage
/// ```swift
/// let metric = EuclideanDistance()
/// let distance = metric.distance(vectorA, vectorB)
/// ```
///
/// - Precondition: Both vectors must have the same dimension
public struct EuclideanDistance: DistanceMetric {
    public typealias Scalar = Float
    
    @inlinable
    public func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        // Implementation
    }
}
```

**VectorIndex Application:**
```swift
/// FlatIndex: Exhaustive search index with no structure
///
/// Performs exhaustive linear search through all vectors. No preprocessing or structure.
/// Guarantees exact results but has O(n*d) search complexity.
///
/// ## When to Use
/// - Small datasets (< 10K vectors)
/// - When exact results are critical
/// - As a baseline for performance comparison
///
/// ## Performance Characteristics
/// - Insert: O(1)
/// - Search: O(n*d) where n=vectors, d=dimension
/// - Memory: O(n*d)
///
/// - Parameters:
///   - dimension: Vector dimension (must be consistent)
///   - metric: Distance metric for similarity computation
///
/// ## Example
/// ```swift
/// let index = FlatIndex(dimension: 128, metric: .euclidean)
/// try await index.insert(id: "vec1", vector: embedding, metadata: nil)
/// let results = try await index.search(query: query, k: 10, filter: nil)
/// ```
actor FlatIndex: VectorIndexProtocol {
    public let dimension: Int
    public var count: Int { vectors.count }
    public let metric: SupportedDistanceMetric
    
    private var vectors: [VectorID: [Float]] = [:]
    private var metadata: [VectorID: [String: String]] = [:]
}
```

---

## CHECKLIST: Adopting VectorCore Patterns

- [ ] Define namespace enum with static factory methods and configuration
- [ ] Implement error hierarchy with ErrorKind enum and ErrorBuilder
- [ ] Use actor-based ThreadSafeConfiguration for mutable settings
- [ ] Add protocol extensions for convenience methods
- [ ] Implement async-first batch operations with smart parallelization
- [ ] Define logging categories matching component organization
- [ ] Create factory pattern for index type selection
- [ ] Ensure all public types conform to Sendable
- [ ] Use Result<T, IndexError> with proper error chaining
- [ ] Document with rich comments including complexity analysis and examples

