//
//  ErrorBuilder.swift
//  VectorIndex
//
//  Builder pattern for ergonomic error construction with fluent API
//

import Foundation

/// Builder pattern for ergonomic error construction with fluent API
///
/// Provides a chainable interface for building rich errors with context.
///
/// ## Usage
///
/// **Basic error:**
/// ```swift
/// throw ErrorBuilder(.dimensionMismatch, operation: "search")
///     .message("Query dimension doesn't match index")
///     .build()
/// ```
///
/// **With structured metadata:**
/// ```swift
/// throw ErrorBuilder(.invalidParameter, operation: "kmeans_train")
///     .message("k must be in [1, n]")
///     .info("k", "\(k)")
///     .info("n", "\(n)")
///     .build()
/// ```
///
/// **With dimension info:**
/// ```swift
/// throw ErrorBuilder(.dimensionMismatch, operation: "ivf_append")
///     .message("Vector dimension mismatch")
///     .dimension(expected: indexDim, actual: vectorDim)
///     .build()
/// ```
///
/// **With error chaining:**
/// ```swift
/// do {
///     try openFile(path)
/// } catch let ioError {
///     throw ErrorBuilder(.corruptedData, operation: "load_index")
///         .message("Failed to load index file")
///         .info("path", path)
///         .underlying(ioError)
///         .build()
/// }
/// ```
public struct ErrorBuilder {
    // MARK: - Internal State (accessible to @inlinable)

    @usableFromInline internal var kind: IndexErrorKind
    @usableFromInline internal var message: String = ""
    @usableFromInline internal var operation: String
    @usableFromInline internal var additionalInfo: [String: String] = [:]
    @usableFromInline internal var underlying: (any Error)?

    #if DEBUG
    @usableFromInline internal let file: StaticString
    @usableFromInline internal let line: UInt
    @usableFromInline internal let function: StaticString
    #endif

    // MARK: - Initialization

    /// Create a new error builder
    ///
    /// - Parameters:
    ///   - kind: Error category
    ///   - operation: Name of operation that failed
    ///   - file: Source file (automatically captured)
    ///   - line: Line number (automatically captured)
    ///   - function: Function name (automatically captured)
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

    // MARK: - Fluent API - Message

    /// Set error message
    ///
    /// If not set, defaults to `kind.description`.
    ///
    /// - Parameter msg: Human-readable error message
    /// - Returns: Builder for chaining
    @inlinable
    public func message(_ msg: String) -> Self {
        var copy = self
        copy.message = msg
        return copy
    }

    // MARK: - Fluent API - Metadata

    /// Add structured metadata (key-value)
    ///
    /// Use this for operation-specific context that aids debugging:
    /// - Parameter values: `.info("k", "10")`
    /// - Internal state: `.info("vectors_processed", "1500")`
    /// - IDs: `.info("list_id", "42")`
    ///
    /// - Parameters:
    ///   - key: Metadata key
    ///   - value: Metadata value (will be converted to String)
    /// - Returns: Builder for chaining
    @inlinable
    public func info(_ key: String, _ value: String) -> Self {
        var copy = self
        copy.additionalInfo[key] = value
        return copy
    }

    /// Add dimension information (expected vs actual)
    ///
    /// Convenience method for dimension mismatch errors.
    ///
    /// Equivalent to:
    /// ```swift
    /// .info("expected_dim", "\(expected)")
    /// .info("actual_dim", "\(actual)")
    /// ```
    ///
    /// - Parameters:
    ///   - expected: Expected dimension
    ///   - actual: Actual dimension
    /// - Returns: Builder for chaining
    @inlinable
    public func dimension(expected: Int, actual: Int) -> Self {
        var copy = self
        copy.additionalInfo["expected_dim"] = "\(expected)"
        copy.additionalInfo["actual_dim"] = "\(actual)"
        return copy
    }

    /// Add range information (for bounds checking)
    ///
    /// Convenience method for range/bounds errors.
    ///
    /// Equivalent to:
    /// ```swift
    /// .info("index", "\(index)")
    /// .info("count", "\(count)")
    /// ```
    ///
    /// - Parameters:
    ///   - index: Index that was out of bounds
    ///   - count: Valid range (0 ..< count)
    /// - Returns: Builder for chaining
    @inlinable
    public func range(index: Int, count: Int) -> Self {
        var copy = self
        copy.additionalInfo["index"] = "\(index)"
        copy.additionalInfo["count"] = "\(count)"
        return copy
    }

    /// Add capacity information
    ///
    /// Convenience method for capacity exceeded errors.
    ///
    /// - Parameters:
    ///   - current: Current usage
    ///   - maximum: Maximum capacity
    /// - Returns: Builder for chaining
    @inlinable
    public func capacity(current: Int, maximum: Int) -> Self {
        var copy = self
        copy.additionalInfo["current"] = "\(current)"
        copy.additionalInfo["maximum"] = "\(maximum)"
        return copy
    }

    /// Add parameter information
    ///
    /// Convenience method for invalid parameter errors.
    ///
    /// - Parameters:
    ///   - name: Parameter name
    ///   - value: Parameter value
    /// - Returns: Builder for chaining
    @inlinable
    public func parameter(_ name: String, value: String) -> Self {
        var copy = self
        copy.additionalInfo["param_\(name)"] = value
        return copy
    }

    /// Add file path information
    ///
    /// Convenience method for file I/O errors.
    ///
    /// - Parameter path: File path
    /// - Returns: Builder for chaining
    @inlinable
    public func path(_ path: String) -> Self {
        var copy = self
        copy.additionalInfo["path"] = path
        return copy
    }

    /// Add errno information
    ///
    /// Convenience method for system errors.
    ///
    /// - Parameter errno: POSIX errno value
    /// - Returns: Builder for chaining
    @inlinable
    public func errno(_ errno: Int32) -> Self {
        var copy = self
        copy.additionalInfo["errno"] = "\(errno)"
        copy.additionalInfo["errno_desc"] = String(cString: strerror(errno))
        return copy
    }

    // MARK: - Fluent API - Error Chaining

    /// Add underlying error (for error chaining)
    ///
    /// Use this to chain errors across layers:
    ///
    /// ```swift
    /// do {
    ///     try lowLevelOp()
    /// } catch let lowError {
    ///     throw ErrorBuilder(.corruptedData, operation: "high_level")
    ///         .underlying(lowError)
    ///         .build()
    /// }
    /// ```
    ///
    /// - Parameter error: Underlying error that caused this error
    /// - Returns: Builder for chaining
    @inlinable
    public func underlying(_ error: any Error) -> Self {
        var copy = self
        copy.underlying = error
        return copy
    }

    // MARK: - Build

    /// Build final error
    ///
    /// Constructs the VectorIndexError with all accumulated context.
    ///
    /// - Returns: Fully constructed VectorIndexError
    @inlinable
    public func build() -> VectorIndexError {
        #if DEBUG
        let context = IndexErrorContext(
            file: file,
            line: line,
            function: function,
            operation: operation,
            additionalInfo: additionalInfo
        )
        #else
        let context = IndexErrorContext(
            operation: operation,
            additionalInfo: additionalInfo
        )
        #endif

        return VectorIndexError(
            kind: kind,
            message: message.isEmpty ? kind.description : message,
            context: context,
            underlyingError: underlying
        )
    }
}

// MARK: - Convenience Builders for Common Patterns

extension ErrorBuilder {
    /// Create dimension mismatch error (common pattern)
    ///
    /// Example:
    /// ```swift
    /// throw ErrorBuilder.dimensionMismatch(
    ///     operation: "search",
    ///     expected: 128,
    ///     actual: 256
    /// )
    /// ```
    @inlinable
    public static func dimensionMismatch(
        operation: String,
        expected: Int,
        actual: Int,
        file: StaticString = #file,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorIndexError {
        ErrorBuilder(.dimensionMismatch, operation: operation, file: file, line: line, function: function)
            .message("Vector dimension mismatch")
            .dimension(expected: expected, actual: actual)
            .build()
    }

    /// Create invalid range error (common pattern)
    ///
    /// Example:
    /// ```swift
    /// throw ErrorBuilder.invalidRange(
    ///     operation: "access",
    ///     index: 100,
    ///     count: 50
    /// )
    /// ```
    @inlinable
    public static func invalidRange(
        operation: String,
        index: Int,
        count: Int,
        file: StaticString = #file,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorIndexError {
        ErrorBuilder(.invalidRange, operation: operation, file: file, line: line, function: function)
            .message("Index out of bounds")
            .range(index: index, count: count)
            .build()
    }

    /// Create invalid parameter error (common pattern)
    ///
    /// Example:
    /// ```swift
    /// throw ErrorBuilder.invalidParameter(
    ///     operation: "kmeans",
    ///     name: "k",
    ///     value: "0",
    ///     constraint: "must be > 0"
    /// )
    /// ```
    @inlinable
    public static func invalidParameter(
        operation: String,
        name: String,
        value: String,
        constraint: String,
        file: StaticString = #file,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorIndexError {
        ErrorBuilder(.invalidParameter, operation: operation, file: file, line: line, function: function)
            .message("\(name) \(constraint)")
            .parameter(name, value: value)
            .build()
    }
}
