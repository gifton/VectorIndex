//
//  VectorIndexError.swift
//  VectorIndex
//
//  Primary error type for VectorIndex operations
//

import Foundation

/// Primary error type for VectorIndex operations
///
/// Provides rich context for debugging while maintaining Sendable conformance.
/// Supports error chaining for multi-layer operations.
///
/// ## Usage
///
/// **Creating errors:**
/// ```swift
/// throw VectorIndexError(
///     kind: .dimensionMismatch,
///     message: "Query dimension doesn't match index",
///     context: IndexErrorContext(operation: "search")
/// )
/// ```
///
/// **Error chaining:**
/// ```swift
/// do {
///     try lowLevelOperation()
/// } catch let lowError {
///     throw VectorIndexError(
///         kind: .corruptedData,
///         message: "Failed to load index",
///         context: IndexErrorContext(operation: "load"),
///         underlyingError: lowError
///     )
/// }
/// ```
///
/// **Checking error kind:**
/// ```swift
/// catch let error as VectorIndexError {
///     if error.kind.isRecoverable {
///         // Retry logic
///     }
/// }
/// ```
@frozen
public struct VectorIndexError: Error, Sendable {
    // MARK: - Core Fields

    /// Categorical classification of error
    public let kind: IndexErrorKind

    /// Human-readable error message
    public let message: String

    /// Debugging context (source location, metadata, etc.)
    public let context: IndexErrorContext

    // MARK: - Error Chaining

    /// Underlying error that caused this error (if any)
    ///
    /// Note: Type-erased to maintain Sendable conformance.
    /// Use `errorChain` to traverse the full chain of VectorIndexErrors.
    public let underlyingError: (any Error)?

    /// Chain of errors from root cause to this error
    ///
    /// Returns array starting with this error and including all
    /// underlying VectorIndexErrors in the chain.
    ///
    /// Example:
    /// ```
    /// [
    ///   VectorIndexError(.corruptedData, "Failed to load"),  // This error
    ///   VectorIndexError(.mmapError, "mmap failed"),        // Underlying
    ///   VectorIndexError(.fileIOError, "open failed")       // Root cause
    /// ]
    /// ```
    public var errorChain: [VectorIndexError] {
        var chain: [VectorIndexError] = [self]
        if let underlying = underlyingError as? VectorIndexError {
            chain.append(contentsOf: underlying.errorChain)
        }
        return chain
    }

    /// Root cause of the error (deepest error in chain)
    public var rootCause: VectorIndexError {
        errorChain.last ?? self
    }

    // MARK: - Initialization

    /// Create a new VectorIndexError
    ///
    /// - Parameters:
    ///   - kind: Error category
    ///   - message: Human-readable description
    ///   - context: Debugging context
    ///   - underlyingError: Optional underlying cause (for error chaining)
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
}

// MARK: - CustomStringConvertible

extension VectorIndexError: CustomStringConvertible {
    /// Human-readable description of error
    ///
    /// Format (debug builds):
    /// ```
    /// VectorIndexError(.dimensionMismatch): Query dimension doesn't match
    /// [operation=search at File.swift:42] {expected_dim=128, actual_dim=256}
    /// ```
    ///
    /// Format (release builds):
    /// ```
    /// VectorIndexError(.dimensionMismatch): Query dimension doesn't match
    /// [operation=search] {expected_dim=128, actual_dim=256}
    /// ```
    public var description: String {
        var desc = "VectorIndexError(.\(kind.rawValue)): \(message)"

        // Add context
        desc += " [\(context.description)]"

        // Add underlying error (indented)
        if let underlying = underlyingError {
            desc += "\n  → Caused by: \(underlying)"
        }

        return desc
    }

    /// Compact description without context (useful for error messages to users)
    public var shortDescription: String {
        "\(kind.description): \(message)"
    }
}

// MARK: - LocalizedError

extension VectorIndexError: LocalizedError {
    /// Localized description for error presentation
    public var errorDescription: String? {
        message
    }

    /// Localized failure reason
    public var failureReason: String? {
        kind.description
    }

    /// Localized recovery suggestion
    public var recoverySuggestion: String? {
        recoveryMessage
    }
}

// MARK: - Recovery Guidance

extension VectorIndexError {
    /// User-facing recovery guidance for this error
    ///
    /// Provides actionable advice on how to fix or work around the error.
    public var recoveryMessage: String {
        switch kind {
        // Input Validation
        case .dimensionMismatch:
            return "Ensure all vectors have the same dimension as the index."
        case .invalidDimension:
            return "Check that vector dimension is positive and within supported range."
        case .invalidParameter:
            return "Check parameter constraints in API documentation."
        case .invalidRange:
            return "Verify that indices are within valid bounds (0 ..< count)."
        case .emptyInput:
            return "Provide non-empty input data."

        // Data Integrity
        case .corruptedData:
            return "Data may be corrupted. Re-create the index from source data."
        case .invalidFormat:
            return "Verify data format matches expected serialization version."
        case .versionMismatch:
            return "Rebuild index with current version, or use compatible library version."
        case .endiannessMismatch:
            return "Index was created on system with different byte order. Rebuild on target platform."

        // Resource Constraints
        case .memoryExhausted:
            return "Reduce batch size, dataset size, or increase available memory."
        case .capacityExceeded:
            return "Reduce operation size or increase capacity limits in configuration."
        case .fileIOError:
            return "Check file permissions, disk space, and file system health."
        case .mmapError:
            return "Verify file exists, is readable, and system has available virtual memory."

        // Operation Failures
        case .convergenceFailure:
            return "Try adjusting convergence tolerance, increasing epochs, or changing initialization."
        case .numericInstability:
            return "Check input data for extreme values, NaN, or infinity. Consider normalization."
        case .emptyCluster:
            return "Reduce number of clusters (k) or provide more diverse training data."
        case .duplicateID:
            return "Ensure all external IDs are unique. Check for ID conflicts in input data."

        // Configuration
        case .unsupportedLayout:
            return "Use supported data layout (AoS or AoSoA). See documentation for layout requirements."
        case .incompatibleConfig:
            return "Review configuration options for conflicts or invalid combinations."
        case .missingDependency:
            return "Initialize required components before calling this operation."

        // Internal Errors
        case .internalInconsistency:
            return "This is a bug. Please file an issue at https://github.com/anthropics/vectorindex/issues with error details."
        case .notImplemented:
            return "This feature is not yet implemented. Check documentation for alternatives."
        case .contractViolation:
            return "Review API documentation and ensure all preconditions are met."
        }
    }

    /// Is this error transient (retry might succeed)?
    public var isTransient: Bool {
        switch kind {
        case .memoryExhausted, .fileIOError, .convergenceFailure:
            return true
        default:
            return false
        }
    }

    /// Should this error be logged/reported automatically?
    public var shouldReport: Bool {
        switch kind {
        // Internal errors should always be reported (bugs)
        case .internalInconsistency, .contractViolation:
            return true
        // User errors don't need automatic reporting
        case .dimensionMismatch, .invalidParameter, .invalidRange, .emptyInput:
            return false
        // Everything else is optional
        default:
            return false
        }
    }
}

// MARK: - Metadata for Logging

extension VectorIndexError {
    /// Structured metadata for logging systems
    ///
    /// Returns a flat dictionary suitable for structured logging frameworks.
    ///
    /// Example:
    /// ```swift
    /// logger.error("Operation failed", metadata: error.logMetadata)
    /// ```
    public var logMetadata: [String: String] {
        var metadata = context.toLogMetadata()
        metadata["error_kind"] = kind.rawValue
        metadata["error_category"] = kind.category.rawValue
        metadata["recoverable"] = String(kind.isRecoverable)
        metadata["transient"] = String(isTransient)

        // Add chain depth if chained
        let chainDepth = errorChain.count
        if chainDepth > 1 {
            metadata["error_chain_depth"] = String(chainDepth)
            metadata["root_cause"] = rootCause.kind.rawValue
        }

        return metadata
    }
}
