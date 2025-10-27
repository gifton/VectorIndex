//
//  IndexErrorKind.swift
//  VectorIndex
//
//  Error taxonomy for VectorIndex operations
//  Aligned with VectorCore error patterns for ecosystem consistency
//

import Foundation

/// Categorical error classification for VectorIndex operations
///
/// Aligned with VectorCore.ErrorKind for ecosystem consistency.
/// Each category maps to specific recovery strategies and user guidance.
///
/// ## Categories
///
/// **Input Validation** (User Error)
/// - Errors caused by invalid input parameters or data
/// - Usually recoverable by fixing input
///
/// **Data Integrity** (Corruption/Format)
/// - Errors caused by corrupted or incompatible data
/// - Usually requires re-creating index from source
///
/// **Resource Constraints** (System Limits)
/// - Errors caused by system resource limitations
/// - May be recoverable by reducing load or retrying
///
/// **Operation Failures** (Runtime)
/// - Errors during algorithm execution
/// - May be recoverable by adjusting parameters
///
/// **Configuration Issues** (Setup Error)
/// - Errors in configuration or initialization
/// - Recoverable by fixing configuration
///
/// **Internal Errors** (Bugs)
/// - Programming errors or violated invariants
/// - Should be reported as bugs
@frozen
public enum IndexErrorKind: String, CaseIterable, Sendable {
    // MARK: - Input Validation (User Error)

    /// Vector dimensions don't match index/config
    case dimensionMismatch

    /// Dimension out of valid range (0 or too large)
    case invalidDimension

    /// Parameter value invalid (k, batchSize, etc.)
    case invalidParameter

    /// Index/range out of bounds
    case invalidRange

    /// Required input is empty
    case emptyInput

    // MARK: - Data Integrity (Corruption/Format)

    /// Data fails integrity checks (CRC, magic bytes)
    case corruptedData

    /// Data format unrecognized/unsupported
    case invalidFormat

    /// Serialization version incompatible
    case versionMismatch

    /// Byte order incompatible
    case endiannessMismatch

    // MARK: - Resource Constraints (System Limits)

    /// Allocation failed, out of memory
    case memoryExhausted

    /// Operation exceeds configured capacity
    case capacityExceeded

    /// File system operation failed
    case fileIOError

    /// Memory mapping failed
    case mmapError

    // MARK: - Operation Failures (Runtime)

    /// Algorithm failed to converge (k-means, etc.)
    case convergenceFailure

    /// Floating-point error, overflow, NaN
    case numericInstability

    /// Clustering produced empty partition
    case emptyCluster

    /// Duplicate external ID in IDMap
    case duplicateID

    // MARK: - Configuration Issues (Setup Error)

    /// Data layout not supported for operation
    case unsupportedLayout

    /// Config options conflict
    case incompatibleConfig

    /// Required component not initialized
    case missingDependency

    // MARK: - Internal Errors (Bugs)

    /// Internal state corruption (should never happen)
    case internalInconsistency

    /// Feature not yet implemented
    case notImplemented

    /// API contract violated by caller
    case contractViolation
}

// MARK: - Properties

extension IndexErrorKind {
    /// User-facing description of error category
    public var description: String {
        switch self {
        // Input Validation
        case .dimensionMismatch: return "Vector dimension mismatch"
        case .invalidDimension: return "Invalid vector dimension"
        case .invalidParameter: return "Invalid parameter value"
        case .invalidRange: return "Index or range out of bounds"
        case .emptyInput: return "Empty or missing required input"

        // Data Integrity
        case .corruptedData: return "Data corruption detected"
        case .invalidFormat: return "Invalid data format"
        case .versionMismatch: return "Incompatible version"
        case .endiannessMismatch: return "Byte order mismatch"

        // Resource Constraints
        case .memoryExhausted: return "Memory allocation failed"
        case .capacityExceeded: return "Capacity limit exceeded"
        case .fileIOError: return "File I/O error"
        case .mmapError: return "Memory mapping failed"

        // Operation Failures
        case .convergenceFailure: return "Algorithm convergence failed"
        case .numericInstability: return "Numerical instability detected"
        case .emptyCluster: return "Empty cluster detected"
        case .duplicateID: return "Duplicate ID"

        // Configuration Issues
        case .unsupportedLayout: return "Unsupported data layout"
        case .incompatibleConfig: return "Incompatible configuration"
        case .missingDependency: return "Missing required component"

        // Internal Errors
        case .internalInconsistency: return "Internal consistency error"
        case .notImplemented: return "Feature not implemented"
        case .contractViolation: return "API contract violation"
        }
    }

    /// Is this error recoverable by the caller?
    ///
    /// Recoverable errors typically indicate:
    /// - Invalid user input that can be corrected
    /// - Transient resource issues that may resolve on retry
    /// - Configuration problems that can be fixed
    ///
    /// Non-recoverable errors typically indicate:
    /// - Data corruption requiring re-creation
    /// - Programming errors/bugs
    /// - System-level failures
    public var isRecoverable: Bool {
        switch self {
        // User can fix these
        case .dimensionMismatch, .invalidDimension, .invalidParameter,
             .invalidRange, .emptyInput, .incompatibleConfig, .unsupportedLayout:
            return true

        // Retry might work
        case .memoryExhausted, .fileIOError, .convergenceFailure:
            return true

        // Likely fatal or requires major intervention
        case .corruptedData, .versionMismatch, .internalInconsistency,
             .contractViolation, .mmapError:
            return false

        // Context-dependent (default to non-recoverable for safety)
        default:
            return false
        }
    }

    /// Error category for grouping related errors
    public var category: ErrorCategory {
        switch self {
        case .dimensionMismatch, .invalidDimension, .invalidParameter,
             .invalidRange, .emptyInput:
            return .inputValidation

        case .corruptedData, .invalidFormat, .versionMismatch,
             .endiannessMismatch:
            return .dataIntegrity

        case .memoryExhausted, .capacityExceeded, .fileIOError, .mmapError:
            return .resourceConstraints

        case .convergenceFailure, .numericInstability, .emptyCluster, .duplicateID:
            return .operationFailure

        case .unsupportedLayout, .incompatibleConfig, .missingDependency:
            return .configuration

        case .internalInconsistency, .notImplemented, .contractViolation:
            return .internalError
        }
    }
}

/// High-level error categories for grouping related error kinds
@frozen
public enum ErrorCategory: String, Sendable {
    case inputValidation = "Input Validation"
    case dataIntegrity = "Data Integrity"
    case resourceConstraints = "Resource Constraints"
    case operationFailure = "Operation Failure"
    case configuration = "Configuration"
    case internalError = "Internal Error"
}
