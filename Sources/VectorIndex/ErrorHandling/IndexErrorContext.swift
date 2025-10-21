//
//  IndexErrorContext.swift
//  VectorIndex
//
//  Rich debugging context for error diagnosis
//

import Foundation

/// Rich debugging context for error diagnosis
///
/// Performance-aware: Full context in debug builds, minimal in release.
/// All fields are Sendable for Swift 6 concurrency compliance.
///
/// ## Usage
///
/// ```swift
/// let context = IndexErrorContext(
///     operation: "ivf_search",
///     additionalInfo: ["k": "10", "nprobe": "16"]
/// )
/// ```
///
/// ## Debug vs Release Builds
///
/// **Debug builds** include:
/// - File name, line number, function name
/// - Full stack context
/// - All metadata
///
/// **Release builds** include:
/// - Operation name
/// - Timestamp
/// - Additional info dictionary
/// - Optional thread ID and memory pressure
///
/// This provides rich debugging information during development while
/// minimizing overhead and binary size in production.
@frozen
public struct IndexErrorContext: Sendable {
    // MARK: - Source Location (Debug Only)

    #if DEBUG
    /// Source file where error was created
    public let file: StaticString

    /// Line number where error was created
    public let line: UInt

    /// Function name where error was created
    public let function: StaticString
    #endif

    // MARK: - Temporal Context

    /// When the error occurred
    public let timestamp: Date

    // MARK: - Operation Context

    /// Operation that failed (e.g., "ivf_search", "kmeans_train")
    public let operation: String

    /// Structured key-value metadata about the error
    ///
    /// Use this for operation-specific context:
    /// - Parameter values: `["k": "10", "dimension": "128"]`
    /// - Internal state: `["vectors_processed": "1500", "batch_size": "256"]`
    /// - IDs and identifiers: `["list_id": "42", "vector_id": "abc123"]`
    public let additionalInfo: [String: String]

    // MARK: - System Context (Optional)

    /// Thread ID where error occurred (optional)
    public let threadID: UInt64?

    /// Was system under memory pressure when error occurred? (optional)
    public let memoryPressure: Bool?

    // MARK: - Initialization

    /// Create error context with full debugging information
    ///
    /// - Parameters:
    ///   - file: Source file (automatically captured via #file)
    ///   - line: Line number (automatically captured via #line)
    ///   - function: Function name (automatically captured via #function)
    ///   - operation: Name of operation that failed
    ///   - additionalInfo: Structured metadata (default: empty)
    ///   - threadID: Thread identifier (default: nil)
    ///   - memoryPressure: Memory pressure flag (default: nil)
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

// MARK: - CustomStringConvertible

extension IndexErrorContext: CustomStringConvertible {
    /// Human-readable description of error context
    public var description: String {
        var parts: [String] = []

        // Operation
        parts.append("operation=\(operation)")

        // Source location (debug only)
        #if DEBUG
        let fileName = String(describing: file).split(separator: "/").last ?? ""
        parts.append("at \(fileName):\(line)")
        #endif

        // Additional info
        if !additionalInfo.isEmpty {
            let info = additionalInfo.map { "\($0)=\($1)" }.joined(separator: ", ")
            parts.append("{\(info)}")
        }

        // System context
        if let tid = threadID {
            parts.append("thread=\(tid)")
        }
        if let pressure = memoryPressure, pressure {
            parts.append("memory_pressure=true")
        }

        return parts.joined(separator: " ")
    }
}

// MARK: - Helper Methods

extension IndexErrorContext {
    /// Create a new context with additional information added
    ///
    /// This is useful for building up context as error propagates through layers.
    ///
    /// - Parameter info: Additional key-value pairs to merge
    /// - Returns: New context with merged information
    public func withAdditionalInfo(_ info: [String: String]) -> IndexErrorContext {
        var merged = additionalInfo
        merged.merge(info) { _, new in new }  // New values override

        #if DEBUG
        return IndexErrorContext(
            file: file,
            line: line,
            function: function,
            operation: operation,
            additionalInfo: merged,
            threadID: threadID,
            memoryPressure: memoryPressure
        )
        #else
        return IndexErrorContext(
            operation: operation,
            additionalInfo: merged,
            threadID: threadID,
            memoryPressure: memoryPressure
        )
        #endif
    }

    /// Convert to structured metadata for logging systems
    ///
    /// Returns a flat dictionary suitable for structured logging.
    ///
    /// - Returns: Dictionary of all context fields as strings
    public func toLogMetadata() -> [String: String] {
        var metadata: [String: String] = [
            "operation": operation,
            "timestamp": ISO8601DateFormatter().string(from: timestamp)
        ]

        // Add source location (debug only)
        #if DEBUG
        metadata["file"] = String(describing: file)
        metadata["line"] = String(line)
        metadata["function"] = String(describing: function)
        #endif

        // Merge additional info
        metadata.merge(additionalInfo) { _, new in new }

        // Add system context
        if let tid = threadID {
            metadata["thread_id"] = String(tid)
        }
        if let pressure = memoryPressure {
            metadata["memory_pressure"] = String(pressure)
        }

        return metadata
    }
}
