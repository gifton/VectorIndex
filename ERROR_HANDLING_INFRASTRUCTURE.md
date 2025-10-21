# Error Handling Infrastructure Plan
**VectorIndex v0.1.0+ Error Handling System Design**

**Status:** Planning Document
**Created:** 2025-10-20
**Target:** v0.1.1 - v0.2.0 implementation

---

## Executive Summary

This document outlines a comprehensive error handling infrastructure for VectorIndex that:
- **Aligns with VectorCore** error patterns for consistency across the GSuite vector ecosystem
- **Preserves performance** in hot paths while providing rich debugging information
- **Enables recovery** through clear error categorization and guidance
- **Maintains Swift 6 compliance** with proper Sendable conformance
- **Provides migration path** from current precondition/fatalError patterns

**Key Principle:** *Fail gracefully with actionable information, fail fast with clear diagnostics.*

---

## 1. Current State Analysis

### 1.1 Existing Error Types (Inconsistent)

```swift
// Internal errors (simple enums)
internal enum VIndexError: Error { ... }        // 12 cases, some with associated values
internal enum IDMapError: Error { ... }          // 5 cases with associated values

// Public errors (mixed patterns)
public enum IVFError: Error { ... }              // 10 cases, no context
public enum PQError: Int32, Error { ... }        // C-style codes
public enum ResidualError: Int32, Error { ... }  // C-style codes
public enum LayoutError: Error, CustomStringConvertible { ... }

// Status codes (not Error-conforming)
public enum KMeansMBStatus: Int32 { ... }        // Should conform to Error
```

### 1.2 Error-Prone Patterns (163 occurrences)

**Current Distribution:**
- `precondition()`: ~60 occurrences → Should be throws for recoverable conditions
- `fatalError()`: ~15 occurrences → Appropriate for programmer errors only
- `assert()`: ~88 occurrences → Debug-only checks, appropriate usage

**Problem Areas:**
```swift
// Example 1: Non-recoverable precondition (should throw)
precondition(k >= 1 && k <= n, "k must be in [1, n]")
// Better: throw VectorIndexError(.invalidParameter, message: "k=\(k) must be in [1, \(n)]")

// Example 2: Fatal error for storage mismatch (could throw)
fatalError("ID storage kind mismatch")
// Better: throw VectorIndexError(.internalInconsistency, ...)

// Example 3: Good usage (programmer error, should crash)
fatalError("Unreachable: all layout cases handled")
```

### 1.3 VectorCore Alignment Gap

VectorCore provides:
- ✅ Hierarchical error system (ErrorKind + ErrorContext + VectorError)
- ✅ Builder pattern for error construction
- ✅ Error chaining for root cause analysis
- ✅ Performance-aware context collection (debug vs release)

VectorIndex currently has:
- ❌ No unified error taxonomy
- ❌ No structured error context
- ❌ No error recovery guidance
- ❌ Inconsistent error types across subsystems

---

## 2. Proposed Error Architecture

### 2.1 Error Taxonomy (VectorCore-Aligned)

```swift
/// Categorical error classification for VectorIndex operations
///
/// Aligned with VectorCore.ErrorKind for ecosystem consistency.
/// Each category maps to specific recovery strategies and user guidance.
public enum IndexErrorKind: String, CaseIterable, Sendable {
    // MARK: - Input Validation (User Error)
    case dimensionMismatch      // Vector dimensions don't match index/config
    case invalidDimension       // Dimension out of valid range (0 or too large)
    case invalidParameter       // Parameter value invalid (k, batchSize, etc.)
    case invalidRange           // Index/range out of bounds
    case emptyInput             // Required input is empty

    // MARK: - Data Integrity (Corruption/Format)
    case corruptedData          // Data fails integrity checks (CRC, magic bytes)
    case invalidFormat          // Data format unrecognized/unsupported
    case versionMismatch        // Serialization version incompatible
    case endiannessMismatch     // Byte order incompatible

    // MARK: - Resource Constraints (System Limits)
    case memoryExhausted        // Allocation failed, out of memory
    case capacityExceeded       // Operation exceeds configured capacity
    case fileIOError            // File system operation failed
    case mmapError              // Memory mapping failed

    // MARK: - Operation Failures (Runtime)
    case convergenceFailure     // Algorithm failed to converge (k-means, etc.)
    case numericInstability     // Floating-point error, overflow, NaN
    case emptyCluster           // Clustering produced empty partition
    case duplicateID            // Duplicate external ID in IDMap

    // MARK: - Configuration Issues (Setup Error)
    case unsupportedLayout      // Data layout not supported for operation
    case incompatibleConfig     // Config options conflict
    case missingDependency      // Required component not initialized

    // MARK: - Internal Errors (Bugs)
    case internalInconsistency  // Internal state corruption (should never happen)
    case notImplemented         // Feature not yet implemented
    case contractViolation      // API contract violated by caller
}

extension IndexErrorKind {
    /// User-facing description of error category
    public var description: String {
        switch self {
        case .dimensionMismatch: return "Vector dimension mismatch"
        case .invalidDimension: return "Invalid vector dimension"
        case .invalidParameter: return "Invalid parameter value"
        case .invalidRange: return "Index or range out of bounds"
        case .emptyInput: return "Empty or missing required input"
        case .corruptedData: return "Data corruption detected"
        case .invalidFormat: return "Invalid data format"
        case .versionMismatch: return "Incompatible version"
        case .endiannessMismatch: return "Byte order mismatch"
        case .memoryExhausted: return "Memory allocation failed"
        case .capacityExceeded: return "Capacity limit exceeded"
        case .fileIOError: return "File I/O error"
        case .mmapError: return "Memory mapping failed"
        case .convergenceFailure: return "Algorithm convergence failed"
        case .numericInstability: return "Numerical instability detected"
        case .emptyCluster: return "Empty cluster detected"
        case .duplicateID: return "Duplicate ID"
        case .unsupportedLayout: return "Unsupported data layout"
        case .incompatibleConfig: return "Incompatible configuration"
        case .missingDependency: return "Missing required component"
        case .internalInconsistency: return "Internal consistency error"
        case .notImplemented: return "Feature not implemented"
        case .contractViolation: return "API contract violation"
        }
    }

    /// Is this error recoverable by the caller?
    public var isRecoverable: Bool {
        switch self {
        // User can fix these
        case .dimensionMismatch, .invalidDimension, .invalidParameter,
             .invalidRange, .emptyInput, .incompatibleConfig:
            return true
        // Retry might work
        case .memoryExhausted, .fileIOError, .convergenceFailure:
            return true
        // Likely fatal or requires major intervention
        case .corruptedData, .versionMismatch, .internalInconsistency,
             .contractViolation, .mmapError:
            return false
        // Context-dependent
        default:
            return false
        }
    }
}
```

### 2.2 Error Context (Debug Information)

```swift
/// Rich debugging context for error diagnosis
///
/// Performance-aware: Full context in debug builds, minimal in release.
/// All fields are Sendable for Swift 6 concurrency compliance.
public struct IndexErrorContext: Sendable {
    // MARK: - Source Location (Debug Only)
    #if DEBUG
    public let file: StaticString
    public let line: UInt
    public let function: StaticString
    #endif

    // MARK: - Temporal Context
    public let timestamp: Date

    // MARK: - Operation Context
    public let operation: String              // "kmeans_minibatch", "ivf_search", etc.
    public let additionalInfo: [String: String]  // Structured key-value pairs

    // MARK: - System Context (Optional)
    public let threadID: UInt64?
    public let memoryPressure: Bool?          // Was system under memory pressure?

    @inlinable
    public init(
        file: StaticString = #file,
        line: UInt = #line,
        function: StaticString = #function,
        operation: String,
        additionalInfo: [String: String] = [:],
        threadID: UInt64? = nil,
        memoryPressure: Bool? = nil
    ) {
        #if DEBUG
        self.file = file
        self.line = line
        self.function = function
        #endif
        self.timestamp = Date()
        self.operation = operation
        self.additionalInfo = additionalInfo
        self.threadID = threadID
        self.memoryPressure = memoryPressure
    }
}
```

### 2.3 Unified Error Type

```swift
/// Primary error type for VectorIndex operations
///
/// Provides rich context for debugging while maintaining Sendable conformance.
/// Supports error chaining for multi-layer operations.
public struct VectorIndexError: Error, Sendable, CustomStringConvertible {
    // MARK: - Core Fields
    public let kind: IndexErrorKind
    public let message: String
    public let context: IndexErrorContext

    // MARK: - Error Chaining
    /// Underlying error that caused this error (if any)
    /// Note: Type-erased to maintain Sendable
    public let underlyingError: (any Error)?

    /// Chain of errors from root cause to this error
    public var errorChain: [VectorIndexError] {
        var chain: [VectorIndexError] = [self]
        if let underlying = underlyingError as? VectorIndexError {
            chain.append(contentsOf: underlying.errorChain)
        }
        return chain
    }

    // MARK: - Initialization
    public init(
        kind: IndexErrorKind,
        message: String,
        context: IndexErrorContext,
        underlyingError: (any Error)? = nil
    ) {
        self.kind = kind
        self.message = message
        self.context = context
        self.underlyingError = underlyingError
    }

    // MARK: - CustomStringConvertible
    public var description: String {
        var desc = "\(kind.description): \(message)"
        #if DEBUG
        desc += " [\(context.operation) at \(context.file):\(context.line)]"
        #endif
        if !context.additionalInfo.isEmpty {
            desc += " {\(context.additionalInfo.map { "\($0)=\($1)" }.joined(separator: ", "))}"
        }
        if let underlying = underlyingError {
            desc += "\n  → Caused by: \(underlying)"
        }
        return desc
    }

    // MARK: - Recovery Guidance
    public var recoveryMessage: String {
        switch kind {
        case .dimensionMismatch:
            return "Ensure all vectors have the same dimension as the index."
        case .invalidParameter:
            return "Check parameter constraints in documentation."
        case .memoryExhausted:
            return "Reduce batch size, dataset size, or increase available memory."
        case .convergenceFailure:
            return "Try adjusting convergence tolerance, increasing epochs, or changing initialization."
        case .corruptedData:
            return "Data may be corrupted. Re-create the index from source data."
        case .internalInconsistency:
            return "This is a bug. Please file an issue at https://github.com/anthropics/vectorindex/issues"
        default:
            return "See error details for specific guidance."
        }
    }
}
```

### 2.4 Error Builder (Ergonomic Construction)

```swift
/// Builder pattern for ergonomic error construction with fluent API
///
/// Example:
/// ```swift
/// throw ErrorBuilder(.dimensionMismatch, operation: "ivf_search")
///     .message("Query dimension \(qd) != index dimension \(d)")
///     .info("query_dim", "\(qd)")
///     .info("index_dim", "\(d)")
///     .build()
/// ```
public struct ErrorBuilder {
    private var kind: IndexErrorKind
    private var message: String = ""
    private var operation: String
    private var additionalInfo: [String: String] = [:]
    private var underlying: (any Error)? = nil

    #if DEBUG
    private let file: StaticString
    private let line: UInt
    private let function: StaticString
    #endif

    @inlinable
    public init(
        _ kind: IndexErrorKind,
        operation: String,
        file: StaticString = #file,
        line: UInt = #line,
        function: StaticString = #function
    ) {
        self.kind = kind
        self.operation = operation
        #if DEBUG
        self.file = file
        self.line = line
        self.function = function
        #endif
    }

    /// Set error message
    @inlinable
    public func message(_ msg: String) -> Self {
        var copy = self
        copy.message = msg
        return copy
    }

    /// Add structured info (key-value)
    @inlinable
    public func info(_ key: String, _ value: String) -> Self {
        var copy = self
        copy.additionalInfo[key] = value
        return copy
    }

    /// Add dimension info (expected vs actual)
    @inlinable
    public func dimension(expected: Int, actual: Int) -> Self {
        var copy = self
        copy.additionalInfo["expected_dim"] = "\(expected)"
        copy.additionalInfo["actual_dim"] = "\(actual)"
        return copy
    }

    /// Add range info (for bounds checking)
    @inlinable
    public func range(index: Int, count: Int) -> Self {
        var copy = self
        copy.additionalInfo["index"] = "\(index)"
        copy.additionalInfo["count"] = "\(count)"
        return copy
    }

    /// Add underlying error (for chaining)
    @inlinable
    public func underlying(_ error: any Error) -> Self {
        var copy = self
        copy.underlying = error
        return copy
    }

    /// Build final error
    @inlinable
    public func build() -> VectorIndexError {
        let context = IndexErrorContext(
            #if DEBUG
            file: file,
            line: line,
            function: function,
            #endif
            operation: operation,
            additionalInfo: additionalInfo
        )
        return VectorIndexError(
            kind: kind,
            message: message.isEmpty ? kind.description : message,
            context: context,
            underlyingError: underlying
        )
    }
}
```

---

## 3. Performance Considerations

### 3.1 Hot Path Optimization

**Problem:** Error construction can add overhead in tight loops.

**Strategy: Conditional Context Collection**

```swift
// Hot path: Minimal overhead in release builds
@inline(__always)
func validateDimension(_ d: Int, expected: Int, operation: String) throws {
    guard d == expected else {
        #if DEBUG
        // Full context in debug
        throw ErrorBuilder(.dimensionMismatch, operation: operation)
            .dimension(expected: expected, actual: d)
            .build()
        #else
        // Minimal allocation in release
        throw VectorIndexError(
            kind: .dimensionMismatch,
            message: "dim \(d) != \(expected)",
            context: IndexErrorContext(operation: operation)
        )
        #endif
    }
}
```

### 3.2 Fast Path Assertions

**Preserve current assert() for invariants:**

```swift
// Invariant checks (compiled out in release)
assert(idx >= 0 && idx < count, "Internal: index out of bounds")

// vs User-facing validation (always checked)
guard idx >= 0 && idx < count else {
    throw ErrorBuilder(.invalidRange, operation: "access")
        .range(index: idx, count: count)
        .build()
}
```

### 3.3 Result Type for Batch Operations

**Avoid throwing in high-frequency operations:**

```swift
/// Fast path: Return Result instead of throwing
/// (Avoids exception overhead for expected failures)
@inlinable
public func tryInsert(_ id: UInt64, _ internalID: Int64) -> Result<Void, VectorIndexError> {
    guard internalID >= 0 else {
        return .failure(
            VectorIndexError(
                kind: .invalidParameter,
                message: "Internal ID must be non-negative",
                context: IndexErrorContext(operation: "idmap_insert")
            )
        )
    }

    // ... actual insertion logic ...
    return .success(())
}
```

---

## 4. Migration Strategy

### 4.1 Phase 1: Foundation (v0.1.1)

**Scope:** Add error infrastructure, no breaking changes

**Tasks:**
1. ✅ Define `IndexErrorKind` enum (all 23 categories)
2. ✅ Define `IndexErrorContext` struct
3. ✅ Define `VectorIndexError` struct with error chaining
4. ✅ Define `ErrorBuilder` for ergonomic construction
5. ✅ Add unit tests for error types
6. ✅ Document error handling patterns in CONTRIBUTING.md

**Non-Breaking:** Existing error types remain, new types added alongside.

### 4.2 Phase 2: Kernel Migration (v0.1.2)

**Scope:** Migrate kernel preconditions → throws

**Priority Order:**
1. **IVFAppend** (10 preconditions) → Public API, user-facing
2. **KMeansMiniBatch** (15 preconditions) → Training ops
3. **PQTrain** (9 preconditions) → Quantization
4. **VIndexMmap** (4 fatalErrors) → File I/O

**Example Migration:**

```swift
// Before:
precondition(d > 0, "dimension must be positive")

// After:
guard d > 0 else {
    throw ErrorBuilder(.invalidDimension, operation: "ivf_create")
        .message("Dimension must be positive")
        .info("dimension", "\(d)")
        .build()
}
```

**Testing:** Add error path tests for each migrated precondition.

### 4.3 Phase 3: Error Type Consolidation (v0.2.0)

**Scope:** Deprecate old error types, migrate to `VectorIndexError`

**Deprecation Process:**
```swift
@available(*, deprecated, renamed: "VectorIndexError")
public enum IVFError: Error {
    // ... existing cases ...

    /// Convert to new unified error type
    public func toVectorIndexError() -> VectorIndexError {
        switch self {
        case .invalidDimensions:
            return VectorIndexError(
                kind: .dimensionMismatch,
                message: "Invalid dimensions for IVF operation",
                context: IndexErrorContext(operation: "ivf")
            )
        // ... other cases ...
        }
    }
}
```

**Migration Guide:**
- Provide conversion utilities
- Update all internal call sites
- Deprecate but don't remove old types (one major version grace period)

### 4.4 Phase 4: Error Recovery Patterns (v0.2.1)

**Scope:** Add recovery guidance and retry logic

**Features:**
1. **Automatic Retry for Transient Errors:**
```swift
public func withRetry<T>(
    maxAttempts: Int = 3,
    operation: () throws -> T
) throws -> T {
    var lastError: Error?
    for attempt in 1...maxAttempts {
        do {
            return try operation()
        } catch let error as VectorIndexError where error.kind.isRecoverable {
            lastError = error
            Thread.sleep(forTimeInterval: Double(attempt) * 0.1)
            continue
        } catch {
            throw error  // Non-recoverable, fail immediately
        }
    }
    throw lastError!
}
```

2. **Error Telemetry Integration:**
```swift
extension VectorIndexError {
    /// Log error to telemetry (if enabled)
    public func logToTelemetry() {
        #if ENABLE_TELEMETRY
        TelemetryRecorder.recordError(
            kind: self.kind.rawValue,
            operation: self.context.operation,
            recoverable: self.kind.isRecoverable
        )
        #endif
    }
}
```

---

## 5. Testing Strategy

### 5.1 Error Construction Tests

```swift
final class ErrorInfrastructureTests: XCTestCase {
    func testErrorBuilder() {
        let error = ErrorBuilder(.dimensionMismatch, operation: "test")
            .message("Test error")
            .dimension(expected: 128, actual: 256)
            .build()

        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertEqual(error.message, "Test error")
        XCTAssertEqual(error.context.additionalInfo["expected_dim"], "128")
        XCTAssertEqual(error.context.additionalInfo["actual_dim"], "256")
    }

    func testErrorChaining() {
        let rootError = VectorIndexError(
            kind: .fileIOError,
            message: "Failed to open file",
            context: IndexErrorContext(operation: "open")
        )

        let chainedError = ErrorBuilder(.corruptedData, operation: "load")
            .message("Cannot load corrupted index")
            .underlying(rootError)
            .build()

        XCTAssertEqual(chainedError.errorChain.count, 2)
        XCTAssertEqual(chainedError.errorChain[1].kind, .fileIOError)
    }
}
```

### 5.2 Error Path Coverage

**Requirement:** Every throw statement must have corresponding test.

```swift
final class IVFErrorPathTests: XCTestCase {
    func testThrowsOnInvalidDimension() {
        let handle = try! ivf_create(k_c: 10, m: 16, d: 128)

        // Intentionally wrong dimension
        var badData = [Float](repeating: 0, count: 10 * 64)  // d=64, not 128

        XCTAssertThrowsError(try ivf_append(handle, &badData, count: 10)) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .dimensionMismatch)
            XCTAssertTrue(indexError.kind.isRecoverable)
        }
    }
}
```

### 5.3 Performance Regression Tests

**Ensure error infrastructure doesn't slow happy path:**

```swift
func testErrorInfrastructureOverhead() {
    let n = 10_000
    var data = [Float](repeating: 1.0, count: n * 128)

    measure {
        // This shouldn't throw, so error infrastructure is cold path
        let handle = try! ivf_create(k_c: 100, m: 16, d: 128)
        try! ivf_append(handle, &data, count: n)
    }

    // Ensure <5% overhead vs baseline (measured without error infrastructure)
}
```

---

## 6. Documentation Requirements

### 6.1 Error Handling Guide (ERRORS.md)

**Contents:**
1. Overview of error taxonomy
2. When to use throws vs Result vs precondition
3. Error construction patterns
4. Error recovery examples
5. Debugging with error context

### 6.2 API Documentation

**All throwing functions must document:**
```swift
/// Appends vectors to IVF index
///
/// - Throws:
///   - `VectorIndexError(.dimensionMismatch)`: If vector dimension doesn't match index
///   - `VectorIndexError(.capacityExceeded)`: If list capacity would be exceeded
///   - `VectorIndexError(.invalidParameter)`: If count ≤ 0 or list_id invalid
///
/// - Parameters:
///   - handle: IVF list handle from `ivf_create`
///   - data: Vector data in row-major layout [count × d]
///   - count: Number of vectors to append
///   - list_id: Target partition (0..<k_c)
public func ivf_append(
    _ handle: IVFListHandle,
    _ data: UnsafePointer<Float>,
    count: Int,
    list_id: Int
) throws
```

---

## 7. Integration with Logging

**Deferred to Phase 4, but designed for:**

```swift
extension VectorIndexError {
    /// Structured log representation
    public var logMetadata: [String: String] {
        var metadata = [
            "error_kind": kind.rawValue,
            "operation": context.operation,
            "recoverable": "\(kind.isRecoverable)"
        ]
        metadata.merge(context.additionalInfo) { _, new in new }
        return metadata
    }
}

// Usage with future logging infrastructure:
// logger.error("Operation failed", metadata: error.logMetadata)
```

---

## 8. Success Metrics

### 8.1 Code Quality Metrics

- **Precondition Reduction:** 163 → <50 (migrate 70% to throws)
- **Error Test Coverage:** >90% of error paths tested
- **Error Documentation:** 100% of throwing functions documented

### 8.2 Developer Experience Metrics

- **Time to Diagnose:** Rich context reduces debug time by ~40%
- **Error Recovery Rate:** 60%+ of errors provide actionable recovery
- **API Clarity:** All error conditions documented inline

### 8.3 Performance Metrics

- **Happy Path Overhead:** <2% vs baseline (measured on IVF search)
- **Error Construction Cost:** <100ns per error (measured in release)
- **Memory Overhead:** <1KB per error instance

---

## 9. Open Questions & Risks

### 9.1 Open Questions

1. **Q:** Should we support error localization (i18n)?
   **A:** Defer to v0.3.0+, focus on developer-facing errors first

2. **Q:** How to handle errors in @inlinable functions?
   **A:** Error types must be public; use conditional compilation for debug context

3. **Q:** Should we log all errors automatically?
   **A:** No, let caller decide logging policy (avoid hidden side effects)

### 9.2 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Performance regression in hot paths | High | Benchmark before/after, use Result<> for batch ops |
| Breaking API changes during migration | Medium | Phased deprecation, conversion utilities |
| Error message inconsistency | Low | Centralized message templates, linter rules |
| Incomplete error coverage | Medium | Mandate tests for all error paths, coverage reports |

---

## 10. Implementation Checklist

### Phase 1 (v0.1.1) - Foundation
- [ ] Define `IndexErrorKind` enum with all 23 categories
- [ ] Define `IndexErrorContext` struct
- [ ] Define `VectorIndexError` struct
- [ ] Define `ErrorBuilder` with fluent API
- [ ] Add unit tests for error types (construction, chaining, description)
- [ ] Document in ERRORS.md (new file)
- [ ] Add error handling section to CONTRIBUTING.md

### Phase 2 (v0.1.2) - Kernel Migration
- [ ] Audit all 163 precondition/fatalError calls
- [ ] Migrate IVFAppend preconditions → throws (10 sites)
- [ ] Migrate KMeansMiniBatch preconditions → throws (15 sites)
- [ ] Migrate PQTrain preconditions → throws (9 sites)
- [ ] Migrate VIndexMmap fatalErrors → throws (4 sites)
- [ ] Add error path tests for migrated functions
- [ ] Update API documentation with throws clauses

### Phase 3 (v0.2.0) - Consolidation
- [ ] Deprecate `IVFError`, add `.toVectorIndexError()` converter
- [ ] Deprecate `PQError`, add converter
- [ ] Deprecate `ResidualError`, add converter
- [ ] Deprecate `VIndexError`, add converter
- [ ] Deprecate `IDMapError`, add converter
- [ ] Deprecate `LayoutError`, add converter
- [ ] Update all internal call sites to use `VectorIndexError`
- [ ] Create migration guide for users

### Phase 4 (v0.2.1) - Recovery & Telemetry
- [ ] Implement `withRetry()` utility for transient errors
- [ ] Add error telemetry integration points
- [ ] Add error aggregation for batch operations
- [ ] Document recovery patterns for each error kind
- [ ] Add integration tests for retry logic

---

## 11. References

### 11.1 VectorCore Alignment
- See: `VECTORCORE_API_ANALYSIS.md` (Section 1.3: Error Handling Pattern)
- VectorCore error types: `ErrorKind`, `ErrorContext`, `VectorError`
- Builder pattern examples: `ErrorBuilder` usage

### 11.2 Swift Error Handling Best Practices
- Swift Evolution: SE-0413 (Typed throws)
- Apple Documentation: "Error Handling in Swift"
- Performance: "Zero-cost error handling" (Result<> vs throws)

### 11.3 Industry Patterns
- Rust: `Result<T, E>` and error chaining
- Go: Explicit error returns with context
- Java: Checked exceptions with recovery guidance

---

## Appendix A: Error Kind Reference

### Input Validation Errors (Recoverable)
- `dimensionMismatch` - Vector dimensions don't align
- `invalidDimension` - Dimension value invalid (≤0 or too large)
- `invalidParameter` - Parameter constraint violation (k, batchSize, etc.)
- `invalidRange` - Index out of bounds
- `emptyInput` - Required data missing

**Recovery:** Validate inputs before calling API, adjust parameters.

### Data Integrity Errors (Usually Fatal)
- `corruptedData` - CRC/integrity check failed
- `invalidFormat` - Unrecognized data format
- `versionMismatch` - Serialization version incompatible
- `endiannessMismatch` - Byte order incompatible

**Recovery:** Re-create index from source data, verify serialization version.

### Resource Errors (Retry-able)
- `memoryExhausted` - Allocation failed
- `capacityExceeded` - Operation exceeds limits
- `fileIOError` - File system failure
- `mmapError` - Memory mapping failed

**Recovery:** Reduce batch size, free memory, check file permissions, retry.

### Runtime Errors (Context-Dependent)
- `convergenceFailure` - Algorithm didn't converge
- `numericInstability` - Floating-point error (NaN, overflow)
- `emptyCluster` - Clustering produced empty partition
- `duplicateID` - Duplicate external ID

**Recovery:** Adjust algorithm parameters, validate input data quality.

### Configuration Errors (Setup Phase)
- `unsupportedLayout` - Data layout incompatible
- `incompatibleConfig` - Config options conflict
- `missingDependency` - Required component not initialized

**Recovery:** Fix configuration, initialize dependencies in correct order.

### Internal Errors (Bugs - Report Issues)
- `internalInconsistency` - Internal state corrupted
- `notImplemented` - Feature stub
- `contractViolation` - API contract violated

**Recovery:** File bug report, check API usage correctness.

---

## Appendix B: Migration Examples

### Example 1: Dimension Validation

```swift
// Before (v0.1.0):
public func ivf_append(_ handle: IVFListHandle, _ data: UnsafePointer<Float>, count: Int) {
    precondition(count > 0, "count must be positive")
    precondition(handle.dimension > 0, "invalid handle")
    // ... implementation
}

// After (v0.1.1+):
public func ivf_append(_ handle: IVFListHandle, _ data: UnsafePointer<Float>, count: Int) throws {
    guard count > 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "ivf_append")
            .message("Count must be positive")
            .info("count", "\(count)")
            .build()
    }

    guard handle.dimension > 0 else {
        throw ErrorBuilder(.internalInconsistency, operation: "ivf_append")
            .message("Invalid handle: dimension must be positive")
            .info("dimension", "\(handle.dimension)")
            .build()
    }

    // ... implementation (can now throw)
}
```

### Example 2: Error Chaining

```swift
// Low-level function
func mmapFile(_ path: String) throws -> UnsafeMutableRawPointer {
    // ... open file ...
    guard fd >= 0 else {
        throw ErrorBuilder(.fileIOError, operation: "mmap_open")
            .message("Failed to open file")
            .info("path", path)
            .info("errno", "\(errno)")
            .build()
    }

    // ... mmap call ...
    guard ptr != MAP_FAILED else {
        throw ErrorBuilder(.mmapError, operation: "mmap_call")
            .message("mmap failed")
            .info("errno", "\(errno)")
            .build()
    }

    return ptr
}

// High-level function (chains errors)
func loadIndex(_ path: String) throws -> IVFIndex {
    do {
        let ptr = try mmapFile(path)
        // ... rest of loading ...
    } catch let lowLevelError {
        throw ErrorBuilder(.corruptedData, operation: "load_index")
            .message("Failed to load index from file")
            .info("path", path)
            .underlying(lowLevelError)  // Chain the low-level error
            .build()
    }
}

// User sees full error chain:
// VectorIndexError(.corruptedData): Failed to load index from file {path=/data/index.bin}
//   → Caused by: VectorIndexError(.mmapError): mmap failed {errno=12}
```

---

**End of Document**

*For questions or feedback on this plan, contact the VectorIndex team or file an issue.*
