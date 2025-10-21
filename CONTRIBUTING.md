# Contributing to VectorIndex

Thank you for your interest in contributing to VectorIndex!

---

## Error Handling Guidelines

### When to Use `throw` vs `precondition` vs `fatalError`

VectorIndex uses a structured error handling system introduced in v0.1.1. Follow these guidelines:

#### Use `throw` for Runtime Errors

**Runtime errors** are conditions that can occur during normal operation with valid inputs or system state changes:

```swift
// ✅ User-correctable input errors
guard k > 0 else {
    throw ErrorBuilder.invalidParameter(
        operation: "kmeans",
        name: "k",
        value: "\(k)",
        constraint: "must be > 0"
    )
}

// ✅ Resource exhaustion
guard let buffer = allocateBuffer(size: size) else {
    throw ErrorBuilder(.memoryExhausted, operation: "allocate")
        .info("requested_size", "\(size)")
        .build()
}

// ✅ File I/O failures
guard fileExists(path) else {
    throw ErrorBuilder(.fileIOError, operation: "open")
        .path(path)
        .build()
}

// ✅ Algorithm convergence failures
throw ErrorBuilder(.convergenceFailure, operation: "kmeans")
    .message("Failed to converge after \(maxIters) iterations")
    .build()
```

#### Use `precondition` for API Contract Violations

**Preconditions** enforce API contracts that correct code should never violate:

```swift
// ✅ Structural invariants
precondition(idx >= 0 && idx < count, "index must be in valid range")

// ✅ Required initialization order
precondition(isInitialized, "must call initialize() first")

// ✅ Non-null pointer requirements (internal APIs)
precondition(ptr != nil, "pointer must not be nil")
```

**Key difference:** If a user could trigger the condition through API misuse, use `throw`. If only a programming error would trigger it, use `precondition`.

#### Use `fatalError` for Unreachable Code

**Fatal errors** mark code paths that should be mathematically impossible:

```swift
// ✅ Exhaustive switch with impossible case
switch layout {
case .aos: /* handle */
case .aosoaR: /* handle */
default: fatalError("Unreachable: all layout cases handled")
}

// ✅ Impossible state after validation
guard let value = optional else {
    fatalError("Impossible: value validated non-nil above")
}
```

#### Use `assert` for Debug-Only Checks

**Assertions** verify invariants during development but are compiled out in release builds:

```swift
// ✅ Internal consistency checks
assert(count == vectors.count, "count and vector array size must match")

// ✅ Performance assumptions
assert(isPowerOfTwo(blockSize), "block size should be power of 2 for optimal SIMD")
```

### Creating Errors with ErrorBuilder

Always use `ErrorBuilder` to construct errors:

```swift
// Basic error
throw ErrorBuilder(.dimensionMismatch, operation: "search")
    .message("Query dimension doesn't match index")
    .dimension(expected: indexDim, actual: queryDim)
    .build()

// With metadata
throw ErrorBuilder(.invalidParameter, operation: "ivf_append")
    .parameter("count", value: "\(count)")
    .info("max_capacity", "\(maxCapacity)")
    .build()

// Convenience builders for common cases
throw ErrorBuilder.dimensionMismatch(
    operation: "append",
    expected: 128,
    actual: 256
)
```

### Error Chaining

Chain errors when propagating across layers to preserve context:

```swift
func highLevelOperation() throws {
    do {
        try lowLevelOperation()
    } catch let error {
        throw ErrorBuilder(.corruptedData, operation: "high_level")
            .message("Failed to complete operation")
            .underlying(error)  // Chain the low-level error
            .build()
    }
}
```

### Testing Error Paths

Every `throw` statement must have a corresponding test:

```swift
func testThrowsOnInvalidDimension() {
    XCTAssertThrowsError(try operation(wrongDimension)) { error in
        guard let indexError = error as? VectorIndexError else {
            XCTFail("Expected VectorIndexError")
            return
        }

        XCTAssertEqual(indexError.kind, .dimensionMismatch)
        XCTAssertTrue(indexError.kind.isRecoverable)
        XCTAssertNotNil(indexError.context.additionalInfo["expected_dim"])
    }
}
```

### Checklist for New Throwing Functions

When adding a new throwing function:

- [ ] Document all possible `throws` in function documentation
- [ ] Use `ErrorBuilder` for all errors
- [ ] Add relevant metadata (dimensions, indices, parameters)
- [ ] Include operation name (usually function name or higher-level op)
- [ ] Chain underlying errors when propagating
- [ ] Write tests for all error paths
- [ ] Verify error messages are actionable

Example documentation:

```swift
/// Append vectors to IVF index partition
///
/// - Throws:
///   - `VectorIndexError(.dimensionMismatch)`: If vector dimension doesn't match index
///   - `VectorIndexError(.invalidRange)`: If list_id is out of bounds [0, k_c)
///   - `VectorIndexError(.capacityExceeded)`: If partition would exceed capacity
///
/// - Parameters:
///   - vectors: Vectors in row-major layout [count × d]
///   - count: Number of vectors
///   - listID: Target partition index
public func append(vectors: UnsafePointer<Float>, count: Int, listID: Int) throws {
    // Implementation with proper error handling
}
```

### Migration from Existing Code

When refactoring existing `precondition` calls:

1. **Evaluate**: Is this a programmer error or runtime condition?
2. **User-facing runtime condition** → Convert to `throw`
3. **Programmer error** → Keep as `precondition`
4. **Unsure** → Prefer `throw` (safer to recover than crash)

Example migration:

```swift
// Before
func search(k: Int) {
    precondition(k > 0, "k must be positive")
    // ...
}

// After
func search(k: Int) throws {
    guard k > 0 else {
        throw ErrorBuilder.invalidParameter(
            operation: "search",
            name: "k",
            value: "\(k)",
            constraint: "must be positive"
        )
    }
    // ...
}
```

---

## Code Style

### Swift Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use trailing commas in multi-line arrays/dictionaries
- Place opening braces on same line

### Naming Conventions

- Types: `PascalCase` (e.g., `VectorIndex`, `IVFList`)
- Functions/variables: `camelCase` (e.g., `searchVectors`, `listCount`)
- Constants: `camelCase` (e.g., `maxCapacity`, `defaultBatchSize`)
- Kernel functions: `snake_case` (e.g., `ivf_append`, `kmeans_train`)

### Documentation

- Document all public APIs with doc comments
- Include complexity analysis for algorithms
- Provide usage examples for non-trivial functions
- Document thread-safety guarantees

Example:

```swift
/// Performs k-nearest neighbor search using IVF index
///
/// Searches `nprobe` partitions and returns top-k results.
///
/// ## Complexity
/// - Time: O(nprobe × (m/k) × d) where m = total vectors, k = partitions
/// - Space: O(nprobe × k) for candidate storage
///
/// ## Thread Safety
/// Safe to call concurrently from multiple threads.
///
/// - Parameters:
///   - query: Query vector [d]
///   - k: Number of neighbors to return
///   - nprobe: Number of partitions to search
/// - Returns: k-nearest neighbors sorted by distance
/// - Throws: `VectorIndexError` if query dimension mismatches or k invalid
public func search(query: [Float], k: Int, nprobe: Int) throws -> [SearchResult]
```

---

## Testing

### Test Coverage Requirements

- All public APIs must have tests
- All error paths must be tested
- Performance-critical paths should have benchmarks
- Tests must pass on both macOS and Linux (if supported)

### Test Organization

```
Tests/VectorIndexTests/
├── Core functionality tests (e.g., IVFIndexTests.swift)
├── Error path tests (e.g., ErrorInfrastructureTests.swift)
├── Performance benchmarks (e.g., KMeansKernelBenchmarks.swift)
└── Integration tests (e.g., End2EndTests.swift)
```

### Running Tests

```bash
# All tests
swift test

# Specific test suite
swift test --filter IVFIndexTests

# With verbose output
swift test --verbose
```

---

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Implement your changes** following guidelines above
3. **Add/update tests** for all changes
4. **Run full test suite** and ensure it passes
5. **Update documentation** if needed
6. **Create pull request** with clear description

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Motivation
Why is this change needed?

## Changes
- List of changes made
- Include API additions/modifications
- Note any breaking changes

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Performance benchmarks run (if applicable)

## Documentation
- [ ] Updated API documentation
- [ ] Updated ERRORS.md (if error-related)
- [ ] Added examples if needed
```

---

## Getting Help

- **Issues**: https://github.com/anthropics/vectorindex/issues
- **Discussions**: https://github.com/anthropics/vectorindex/discussions
- **Error Handling**: See [ERRORS.md](ERRORS.md)
- **Architecture**: See [ERROR_HANDLING_INFRASTRUCTURE.md](ERROR_HANDLING_INFRASTRUCTURE.md)

---

## License

By contributing to VectorIndex, you agree that your contributions will be licensed under the same license as the project.
