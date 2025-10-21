# Error Handling Guide
**VectorIndex Error Infrastructure**

This document describes how to use the error handling infrastructure in VectorIndex.

---

## Quick Start

### Throwing Errors

```swift
// Basic error
throw ErrorBuilder(.dimensionMismatch, operation: "search")
    .message("Query dimension doesn't match index")
    .build()

// With metadata
throw ErrorBuilder(.invalidParameter, operation: "kmeans")
    .message("k must be positive")
    .info("k", "\(k)")
    .build()

// With convenience helpers
throw ErrorBuilder(.dimensionMismatch, operation: "search")
    .dimension(expected: 128, actual: 256)
    .build()
```

### Catching Errors

```swift
do {
    try someOperation()
} catch let error as VectorIndexError {
    print("Error: \(error.shortDescription)")
    print("Recovery: \(error.recoveryMessage)")

    if error.kind.isRecoverable {
        // Attempt recovery
    }
}
```

---

## Error Types

### VectorIndexError

The primary error type for all VectorIndex operations.

```swift
public struct VectorIndexError: Error, Sendable {
    let kind: IndexErrorKind        // Error category
    let message: String              // Human-readable message
    let context: IndexErrorContext   // Debugging context
    let underlyingError: (any Error)? // Optional underlying cause

    var errorChain: [VectorIndexError]  // Full error chain
    var rootCause: VectorIndexError     // Deepest error
    var recoveryMessage: String         // How to fix
}
```

### IndexErrorKind

23 error categories organized into 6 groups:

#### Input Validation (Recoverable)
- `.dimensionMismatch` - Vector dimensions don't align
- `.invalidDimension` - Dimension ≤ 0 or too large
- `.invalidParameter` - Parameter constraint violated
- `.invalidRange` - Index out of bounds
- `.emptyInput` - Required data missing

#### Data Integrity (Usually Fatal)
- `.corruptedData` - CRC/integrity check failed
- `.invalidFormat` - Unrecognized data format
- `.versionMismatch` - Incompatible serialization version
- `.endiannessMismatch` - Byte order mismatch

#### Resource Constraints (Retry-able)
- `.memoryExhausted` - Allocation failed
- `.capacityExceeded` - Exceeded configured limits
- `.fileIOError` - File system operation failed
- `.mmapError` - Memory mapping failed

#### Operation Failures (Context-Dependent)
- `.convergenceFailure` - Algorithm didn't converge
- `.numericInstability` - NaN, overflow, etc.
- `.emptyCluster` - Empty partition in clustering
- `.duplicateID` - Duplicate external ID

#### Configuration (Setup Phase)
- `.unsupportedLayout` - Incompatible data layout
- `.incompatibleConfig` - Conflicting config options
- `.missingDependency` - Required component not initialized

#### Internal Errors (Bugs)
- `.internalInconsistency` - Internal state corrupted
- `.notImplemented` - Feature stub
- `.contractViolation` - API precondition violated

---

## Usage Patterns

### Pattern 1: Basic Validation

```swift
func search(query: [Float], k: Int) throws -> [SearchResult] {
    // Validate parameters
    guard k > 0 else {
        throw ErrorBuilder(.invalidParameter, operation: "search")
            .message("k must be positive")
            .parameter("k", value: "\(k)")
            .build()
    }

    guard query.count == dimension else {
        throw ErrorBuilder.dimensionMismatch(
            operation: "search",
            expected: dimension,
            actual: query.count
        )
    }

    // ... implementation
}
```

### Pattern 2: Error Chaining

```swift
func loadIndex(path: String) throws -> IVFIndex {
    do {
        let data = try loadFile(path)
        return try deserialize(data)
    } catch let lowError {
        throw ErrorBuilder(.corruptedData, operation: "load_index")
            .message("Failed to load index from file")
            .path(path)
            .underlying(lowError)  // Chain the low-level error
            .build()
    }
}

// Error chain example:
// VectorIndexError(.corruptedData): Failed to load index
//   → Caused by: VectorIndexError(.fileIOError): open failed
```

### Pattern 3: Recovery Logic

```swift
func robustOperation() throws {
    var attempts = 0
    let maxAttempts = 3

    while attempts < maxAttempts {
        do {
            try potentiallyFailingOperation()
            return  // Success!
        } catch let error as VectorIndexError {
            if error.isTransient && attempts < maxAttempts - 1 {
                attempts += 1
                Thread.sleep(forTimeInterval: 0.1 * Double(attempts))
                continue  // Retry
            }
            throw error  // Give up
        }
    }
}
```

### Pattern 4: Structured Metadata

```swift
func processVectors(vectors: [[Float]], listID: Int) throws {
    guard listID >= 0 && listID < numLists else {
        throw ErrorBuilder(.invalidRange, operation: "ivf_append")
            .message("List ID out of range")
            .range(index: listID, count: numLists)
            .info("num_vectors", "\(vectors.count)")
            .info("list_capacity", "\(capacity[listID])")
            .build()
    }

    // ... implementation
}
```

---

## Error Builder API

### Basic Methods

```swift
ErrorBuilder(.kind, operation: "op_name")
    .message("Human-readable description")
    .build()
```

### Metadata Methods

```swift
.info("key", "value")                    // Custom key-value
.dimension(expected: 128, actual: 256)    // Dimension mismatch
.range(index: 100, count: 50)             // Bounds checking
.capacity(current: 1000, maximum: 800)    // Capacity limits
.parameter("name", value: "value")        // Parameter info
.path("/path/to/file")                    // File path
.errno(errno)                             // POSIX errno
```

### Error Chaining

```swift
.underlying(someError)  // Link to underlying cause
```

### Convenience Builders

```swift
// Dimension mismatch
ErrorBuilder.dimensionMismatch(
    operation: "search",
    expected: 128,
    actual: 256
)

// Invalid range
ErrorBuilder.invalidRange(
    operation: "access",
    index: 100,
    count: 50
)

// Invalid parameter
ErrorBuilder.invalidParameter(
    operation: "kmeans",
    name: "k",
    value: "0",
    constraint: "must be > 0"
)
```

---

## Error Properties

### Recoverability

```swift
error.kind.isRecoverable  // Can this be fixed by caller?
error.isTransient         // Should retry help?
error.shouldReport        // Should this be logged automatically?
```

### Error Chain

```swift
error.errorChain  // [VectorIndexError] from top to root
error.rootCause   // Deepest error in chain
```

### Descriptions

```swift
error.description       // Full description with context
error.shortDescription  // Compact, user-facing description
error.recoveryMessage   // Actionable recovery guidance
```

### Metadata

```swift
error.context.operation         // Operation name
error.context.additionalInfo    // [String: String] metadata
error.logMetadata               // Structured logging dictionary
```

---

## Migration from Preconditions

### Before (v0.1.0)

```swift
func append(vectors: [Float], count: Int) {
    precondition(count > 0, "count must be positive")
    precondition(vectors.count == count * dimension, "wrong size")
    // ... implementation
}
```

### After (v0.1.1+)

```swift
func append(vectors: [Float], count: Int) throws {
    guard count > 0 else {
        throw ErrorBuilder.invalidParameter(
            operation: "append",
            name: "count",
            value: "\(count)",
            constraint: "must be positive"
        )
    }

    guard vectors.count == count * dimension else {
        throw ErrorBuilder(.dimensionMismatch, operation: "append")
            .message("Vector array size doesn't match count × dimension")
            .info("array_size", "\(vectors.count)")
            .info("expected_size", "\(count * dimension)")
            .build()
    }

    // ... implementation
}
```

---

## Best Practices

### DO:

✅ Use `ErrorBuilder` for all new error throwing
✅ Provide actionable error messages
✅ Include relevant metadata (dimensions, indices, parameters)
✅ Chain errors when propagating across layers
✅ Check `.isRecoverable` before retry logic
✅ Use convenience builders for common patterns
✅ Let errors propagate up (don't swallow)

### DON'T:

❌ Create `VectorIndexError` directly (use `ErrorBuilder`)
❌ Use generic messages like "Invalid input"
❌ Forget to add operation name
❌ Mix old error types with new infrastructure
❌ Catch errors without re-throwing or handling
❌ Add excessive metadata (keep it relevant)

---

## Debug vs Release

### Debug Builds

- Full source location (file, line, function)
- Rich context in error descriptions
- All metadata included

### Release Builds

- Minimal overhead
- Operation name and metadata only
- No source location
- Performance-optimized

The infrastructure automatically handles this via conditional compilation.

---

## Examples

### Example 1: Dimension Validation

```swift
func validateQueryDimension(_ query: [Float]) throws {
    guard query.count == indexDimension else {
        throw ErrorBuilder.dimensionMismatch(
            operation: "search",
            expected: indexDimension,
            actual: query.count
        )
    }
}
```

### Example 2: Bounds Checking

```swift
func accessList(_ listID: Int) throws -> IVFList {
    guard listID >= 0 && listID < lists.count else {
        throw ErrorBuilder.invalidRange(
            operation: "access_list",
            index: listID,
            count: lists.count
        )
    }
    return lists[listID]
}
```

### Example 3: File I/O with errno

```swift
func openIndexFile(_ path: String) throws -> FileHandle {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else {
        throw ErrorBuilder(.fileIOError, operation: "open_index")
            .message("Failed to open index file")
            .path(path)
            .errno(errno)
            .build()
    }
    return FileHandle(fileDescriptor: fd)
}
```

### Example 4: Algorithm Convergence

```swift
func kmeansCluster(data: [Float], k: Int, maxIters: Int) throws -> [Float] {
    var centroids = initializeCentroids(k: k)

    for iter in 0..<maxIters {
        let (newCentroids, converged) = performIteration(centroids)
        centroids = newCentroids

        if converged {
            return centroids
        }
    }

    // Failed to converge
    throw ErrorBuilder(.convergenceFailure, operation: "kmeans")
        .message("K-means failed to converge")
        .info("max_iterations", "\(maxIters)")
        .info("k", "\(k)")
        .info("num_vectors", "\(data.count / dimension)")
        .build()
}
```

### Example 5: Multi-Layer Error Propagation

```swift
// Low level: File I/O
func readBytes(_ fd: Int32, count: Int) throws -> Data {
    var buffer = Data(count: count)
    let bytesRead = buffer.withUnsafeMutableBytes { ptr in
        read(fd, ptr.baseAddress!, count)
    }

    guard bytesRead == count else {
        throw ErrorBuilder(.fileIOError, operation: "read")
            .message("Incomplete read")
            .info("expected", "\(count)")
            .info("actual", "\(bytesRead)")
            .errno(errno)
            .build()
    }

    return buffer
}

// Mid level: Deserialization
func deserializeHeader(_ fd: Int32) throws -> IndexHeader {
    do {
        let data = try readBytes(fd, count: 256)
        guard let header = try? decodeHeader(data) else {
            throw ErrorBuilder(.invalidFormat, operation: "decode_header")
                .message("Invalid header format")
                .build()
        }
        return header
    } catch let error {
        throw ErrorBuilder(.corruptedData, operation: "deserialize_header")
            .message("Failed to read index header")
            .underlying(error)
            .build()
    }
}

// High level: Index loading
func loadIndex(path: String) throws -> VectorIndex {
    guard let fd = openFile(path) else {
        throw ErrorBuilder(.fileIOError, operation: "load_index")
            .message("Cannot open index file")
            .path(path)
            .build()
    }

    defer { close(fd) }

    do {
        let header = try deserializeHeader(fd)
        // ... load rest of index ...
        return index
    } catch let error {
        throw ErrorBuilder(.corruptedData, operation: "load_index")
            .message("Failed to load index from file")
            .path(path)
            .underlying(error)
            .build()
    }
}

// Resulting error chain when read fails:
// VectorIndexError(.corruptedData): Failed to load index from file {path=/tmp/index.bin}
//   → Caused by: VectorIndexError(.corruptedData): Failed to read index header
//     → Caused by: VectorIndexError(.fileIOError): Incomplete read {expected=256, actual=0, errno=2, errno_desc=No such file or directory}
```

---

## Testing Errors

### Test that errors are thrown

```swift
func testThrowsOnInvalidDimension() {
    let index = createIndex(dimension: 128)
    let wrongQuery = [Float](repeating: 0, count: 256)  // Wrong dimension

    XCTAssertThrowsError(try index.search(query: wrongQuery, k: 10)) { error in
        guard let indexError = error as? VectorIndexError else {
            XCTFail("Expected VectorIndexError")
            return
        }

        XCTAssertEqual(indexError.kind, .dimensionMismatch)
        XCTAssertTrue(indexError.kind.isRecoverable)
        XCTAssertEqual(indexError.context.additionalInfo["expected_dim"], "128")
        XCTAssertEqual(indexError.context.additionalInfo["actual_dim"], "256")
    }
}
```

### Test error metadata

```swift
func testErrorIncludesMetadata() {
    let error = ErrorBuilder(.invalidRange, operation: "test")
        .range(index: 100, count: 50)
        .build()

    XCTAssertEqual(error.context.additionalInfo["index"], "100")
    XCTAssertEqual(error.context.additionalInfo["count"], "50")
}
```

---

## Logging Integration

The error infrastructure provides structured metadata for logging:

```swift
// Future logging integration
logger.error("Operation failed", metadata: error.logMetadata)

// Metadata includes:
// - error_kind: "dimensionMismatch"
// - error_category: "Input Validation"
// - operation: "search"
// - recoverable: "true"
// - transient: "false"
// - All custom metadata from additionalInfo
```

---

## FAQ

**Q: When should I use `precondition` vs `throw`?**

A: Use `precondition` for programmer errors (API contract violations that should never happen in correct code). Use `throw` for runtime errors that users might encounter (invalid input, resource exhaustion, etc.).

**Q: Should I catch and log all errors?**

A: No. Let errors propagate to the caller unless you can meaningfully handle them. Use error chaining to add context as errors bubble up.

**Q: What's the performance impact?**

A: Minimal (<2%) on happy path. Error construction is ~100ns in release builds. Use `Result<T, E>` for ultra-hot paths if needed.

**Q: Can I use the old error types?**

A: Yes, but they'll be deprecated in v0.2.0. New code should use `VectorIndexError`.

**Q: How do I migrate existing `fatalError` calls?**

A: Evaluate each case:
- If it's a programmer error → keep `fatalError`
- If it's a runtime condition → convert to `throw`
- If you're unsure → throw (safer to recover than crash)

---

## See Also

- [ERROR_HANDLING_INFRASTRUCTURE.md](ERROR_HANDLING_INFRASTRUCTURE.md) - Full design document
- [CONTRIBUTING.md](CONTRIBUTING.md) - Error handling guidelines for contributors
- [Phase 1 Implementation](https://github.com/anthropics/vectorindex/issues/XXX) - Tracking issue

---

**Last Updated:** 2025-10-20 (Phase 1 - v0.1.1)
