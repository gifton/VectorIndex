import XCTest
@testable import VectorIndex

/// Comprehensive tests for error handling infrastructure (Phase 1)
final class ErrorInfrastructureTests: XCTestCase {

    // MARK: - IndexErrorKind Tests

    func testErrorKindAllCasesCount() {
        // Verify we have all 23 error kinds as designed
        XCTAssertEqual(IndexErrorKind.allCases.count, 23)
    }

    func testErrorKindDescriptions() {
        // Verify all error kinds have descriptions
        for kind in IndexErrorKind.allCases {
            XCTAssertFalse(kind.description.isEmpty, "\(kind) should have non-empty description")
        }
    }

    func testErrorKindRecoverability() {
        // Input validation errors should be recoverable
        XCTAssertTrue(IndexErrorKind.dimensionMismatch.isRecoverable)
        XCTAssertTrue(IndexErrorKind.invalidParameter.isRecoverable)
        XCTAssertTrue(IndexErrorKind.invalidRange.isRecoverable)

        // Resource errors should be recoverable (retry-able)
        XCTAssertTrue(IndexErrorKind.memoryExhausted.isRecoverable)
        XCTAssertTrue(IndexErrorKind.fileIOError.isRecoverable)

        // Data corruption should not be recoverable
        XCTAssertFalse(IndexErrorKind.corruptedData.isRecoverable)
        XCTAssertFalse(IndexErrorKind.versionMismatch.isRecoverable)

        // Internal errors should not be recoverable
        XCTAssertFalse(IndexErrorKind.internalInconsistency.isRecoverable)
        XCTAssertFalse(IndexErrorKind.contractViolation.isRecoverable)
    }

    func testErrorKindCategories() {
        // Test category grouping
        XCTAssertEqual(IndexErrorKind.dimensionMismatch.category, .inputValidation)
        XCTAssertEqual(IndexErrorKind.invalidParameter.category, .inputValidation)

        XCTAssertEqual(IndexErrorKind.corruptedData.category, .dataIntegrity)
        XCTAssertEqual(IndexErrorKind.versionMismatch.category, .dataIntegrity)

        XCTAssertEqual(IndexErrorKind.memoryExhausted.category, .resourceConstraints)
        XCTAssertEqual(IndexErrorKind.fileIOError.category, .resourceConstraints)

        XCTAssertEqual(IndexErrorKind.convergenceFailure.category, .operationFailure)
        XCTAssertEqual(IndexErrorKind.numericInstability.category, .operationFailure)

        XCTAssertEqual(IndexErrorKind.unsupportedLayout.category, .configuration)
        XCTAssertEqual(IndexErrorKind.incompatibleConfig.category, .configuration)

        XCTAssertEqual(IndexErrorKind.internalInconsistency.category, .internalError)
        XCTAssertEqual(IndexErrorKind.notImplemented.category, .internalError)
    }

    // MARK: - IndexErrorContext Tests

    func testErrorContextCreation() {
        let context = IndexErrorContext(
            operation: "test_op",
            additionalInfo: ["key1": "value1", "key2": "value2"]
        )

        XCTAssertEqual(context.operation, "test_op")
        XCTAssertEqual(context.additionalInfo["key1"], "value1")
        XCTAssertEqual(context.additionalInfo["key2"], "value2")
        XCTAssertNotNil(context.timestamp)
    }

    func testErrorContextDescription() {
        let context = IndexErrorContext(
            operation: "search",
            additionalInfo: ["k": "10", "nprobe": "16"]
        )

        let desc = context.description
        XCTAssertTrue(desc.contains("operation=search"))
        XCTAssertTrue(desc.contains("k=10"))
        XCTAssertTrue(desc.contains("nprobe=16"))
    }

    func testErrorContextWithAdditionalInfo() {
        let original = IndexErrorContext(
            operation: "test",
            additionalInfo: ["key1": "value1"]
        )

        let updated = original.withAdditionalInfo(["key2": "value2", "key1": "overridden"])

        XCTAssertEqual(updated.additionalInfo["key1"], "overridden")
        XCTAssertEqual(updated.additionalInfo["key2"], "value2")
        XCTAssertEqual(updated.operation, "test")
    }

    func testErrorContextLogMetadata() {
        let context = IndexErrorContext(
            operation: "test_op",
            additionalInfo: ["custom_key": "custom_value"],
            threadID: 12345,
            memoryPressure: true
        )

        let metadata = context.toLogMetadata()

        XCTAssertEqual(metadata["operation"], "test_op")
        XCTAssertEqual(metadata["custom_key"], "custom_value")
        XCTAssertEqual(metadata["thread_id"], "12345")
        XCTAssertEqual(metadata["memory_pressure"], "true")
        XCTAssertNotNil(metadata["timestamp"])
    }

    // MARK: - VectorIndexError Tests

    func testErrorCreation() {
        let context = IndexErrorContext(operation: "test")
        let error = VectorIndexError(
            kind: .dimensionMismatch,
            message: "Test error message",
            context: context
        )

        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertEqual(error.message, "Test error message")
        XCTAssertEqual(error.context.operation, "test")
        XCTAssertNil(error.underlyingError)
    }

    func testErrorChaining() {
        let rootError = VectorIndexError(
            kind: .fileIOError,
            message: "Failed to open file",
            context: IndexErrorContext(operation: "open")
        )

        let middleError = VectorIndexError(
            kind: .mmapError,
            message: "mmap failed",
            context: IndexErrorContext(operation: "mmap"),
            underlyingError: rootError
        )

        let topError = VectorIndexError(
            kind: .corruptedData,
            message: "Cannot load corrupted index",
            context: IndexErrorContext(operation: "load"),
            underlyingError: middleError
        )

        // Test error chain
        let chain = topError.errorChain
        XCTAssertEqual(chain.count, 3)
        XCTAssertEqual(chain[0].kind, .corruptedData)
        XCTAssertEqual(chain[1].kind, .mmapError)
        XCTAssertEqual(chain[2].kind, .fileIOError)

        // Test root cause
        XCTAssertEqual(topError.rootCause.kind, .fileIOError)
    }

    func testErrorDescription() {
        let error = VectorIndexError(
            kind: .dimensionMismatch,
            message: "Dimension mismatch",
            context: IndexErrorContext(
                operation: "search",
                additionalInfo: ["expected": "128", "actual": "256"]
            )
        )

        let description = error.description
        XCTAssertTrue(description.contains("VectorIndexError"))
        XCTAssertTrue(description.contains("dimensionMismatch"))
        XCTAssertTrue(description.contains("Dimension mismatch"))
        XCTAssertTrue(description.contains("operation=search"))
    }

    func testErrorShortDescription() {
        let error = VectorIndexError(
            kind: .invalidParameter,
            message: "k must be positive",
            context: IndexErrorContext(operation: "kmeans")
        )

        let shortDesc = error.shortDescription
        XCTAssertTrue(shortDesc.contains("Invalid parameter value"))
        XCTAssertTrue(shortDesc.contains("k must be positive"))
        XCTAssertFalse(shortDesc.contains("operation"))  // Short description excludes context
    }

    func testErrorRecoveryMessages() {
        // Test a few recovery messages
        let dimError = VectorIndexError(
            kind: .dimensionMismatch,
            message: "Dim mismatch",
            context: IndexErrorContext(operation: "test")
        )
        XCTAssertTrue(dimError.recoveryMessage.contains("dimension"))

        let memError = VectorIndexError(
            kind: .memoryExhausted,
            message: "OOM",
            context: IndexErrorContext(operation: "test")
        )
        XCTAssertTrue(memError.recoveryMessage.contains("batch size"))

        let bugError = VectorIndexError(
            kind: .internalInconsistency,
            message: "Bug",
            context: IndexErrorContext(operation: "test")
        )
        XCTAssertTrue(bugError.recoveryMessage.contains("bug"))
        XCTAssertTrue(bugError.recoveryMessage.contains("issue"))
    }

    func testErrorTransientFlag() {
        // Transient errors
        XCTAssertTrue(VectorIndexError(
            kind: .memoryExhausted,
            message: "",
            context: IndexErrorContext(operation: "test")
        ).isTransient)

        XCTAssertTrue(VectorIndexError(
            kind: .fileIOError,
            message: "",
            context: IndexErrorContext(operation: "test")
        ).isTransient)

        // Non-transient errors
        XCTAssertFalse(VectorIndexError(
            kind: .dimensionMismatch,
            message: "",
            context: IndexErrorContext(operation: "test")
        ).isTransient)

        XCTAssertFalse(VectorIndexError(
            kind: .corruptedData,
            message: "",
            context: IndexErrorContext(operation: "test")
        ).isTransient)
    }

    func testErrorLogMetadata() {
        let error = VectorIndexError(
            kind: .dimensionMismatch,
            message: "Test",
            context: IndexErrorContext(
                operation: "search",
                additionalInfo: ["k": "10"]
            )
        )

        let metadata = error.logMetadata
        XCTAssertEqual(metadata["error_kind"], "dimensionMismatch")
        XCTAssertEqual(metadata["error_category"], "Input Validation")
        XCTAssertEqual(metadata["operation"], "search")
        XCTAssertEqual(metadata["k"], "10")
        XCTAssertEqual(metadata["recoverable"], "true")
    }

    // MARK: - ErrorBuilder Tests

    func testErrorBuilderBasic() {
        let error = ErrorBuilder(.dimensionMismatch, operation: "test")
            .message("Test error")
            .build()

        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertEqual(error.message, "Test error")
        XCTAssertEqual(error.context.operation, "test")
    }

    func testErrorBuilderDefaultMessage() {
        let error = ErrorBuilder(.invalidParameter, operation: "test")
            .build()

        // Should use kind.description as default message
        XCTAssertEqual(error.message, IndexErrorKind.invalidParameter.description)
    }

    func testErrorBuilderWithInfo() {
        let error = ErrorBuilder(.invalidParameter, operation: "test")
            .message("Invalid k")
            .info("k", "0")
            .info("constraint", "must be > 0")
            .build()

        XCTAssertEqual(error.context.additionalInfo["k"], "0")
        XCTAssertEqual(error.context.additionalInfo["constraint"], "must be > 0")
    }

    func testErrorBuilderWithDimension() {
        let error = ErrorBuilder(.dimensionMismatch, operation: "search")
            .dimension(expected: 128, actual: 256)
            .build()

        XCTAssertEqual(error.context.additionalInfo["expected_dim"], "128")
        XCTAssertEqual(error.context.additionalInfo["actual_dim"], "256")
    }

    func testErrorBuilderWithRange() {
        let error = ErrorBuilder(.invalidRange, operation: "access")
            .range(index: 100, count: 50)
            .build()

        XCTAssertEqual(error.context.additionalInfo["index"], "100")
        XCTAssertEqual(error.context.additionalInfo["count"], "50")
    }

    func testErrorBuilderWithCapacity() {
        let error = ErrorBuilder(.capacityExceeded, operation: "append")
            .capacity(current: 1000, maximum: 800)
            .build()

        XCTAssertEqual(error.context.additionalInfo["current"], "1000")
        XCTAssertEqual(error.context.additionalInfo["maximum"], "800")
    }

    func testErrorBuilderWithParameter() {
        let error = ErrorBuilder(.invalidParameter, operation: "kmeans")
            .parameter("epochs", value: "0")
            .build()

        XCTAssertEqual(error.context.additionalInfo["param_epochs"], "0")
    }

    func testErrorBuilderWithPath() {
        let error = ErrorBuilder(.fileIOError, operation: "open")
            .path("/tmp/index.bin")
            .build()

        XCTAssertEqual(error.context.additionalInfo["path"], "/tmp/index.bin")
    }

    func testErrorBuilderWithErrno() {
        let error = ErrorBuilder(.fileIOError, operation: "open")
            .errno(2)  // ENOENT
            .build()

        XCTAssertEqual(error.context.additionalInfo["errno"], "2")
        XCTAssertNotNil(error.context.additionalInfo["errno_desc"])
    }

    func testErrorBuilderWithUnderlying() {
        let rootError = VectorIndexError(
            kind: .fileIOError,
            message: "IO failed",
            context: IndexErrorContext(operation: "read")
        )

        let error = ErrorBuilder(.corruptedData, operation: "load")
            .message("Cannot load")
            .underlying(rootError)
            .build()

        XCTAssertNotNil(error.underlyingError)
        XCTAssertEqual(error.errorChain.count, 2)
        XCTAssertEqual(error.rootCause.kind, .fileIOError)
    }

    func testErrorBuilderFluentChaining() {
        let error = ErrorBuilder(.dimensionMismatch, operation: "ivf_search")
            .message("Query dimension doesn't match index")
            .dimension(expected: 128, actual: 256)
            .info("index_name", "my_index")
            .info("query_id", "abc123")
            .build()

        XCTAssertEqual(error.message, "Query dimension doesn't match index")
        XCTAssertEqual(error.context.additionalInfo["expected_dim"], "128")
        XCTAssertEqual(error.context.additionalInfo["actual_dim"], "256")
        XCTAssertEqual(error.context.additionalInfo["index_name"], "my_index")
        XCTAssertEqual(error.context.additionalInfo["query_id"], "abc123")
    }

    func testErrorBuilderConvenienceDimensionMismatch() {
        let error = ErrorBuilder.dimensionMismatch(
            operation: "search",
            expected: 128,
            actual: 256
        )

        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertTrue(error.message.contains("dimension mismatch"))
        XCTAssertEqual(error.context.additionalInfo["expected_dim"], "128")
        XCTAssertEqual(error.context.additionalInfo["actual_dim"], "256")
    }

    func testErrorBuilderConvenienceInvalidRange() {
        let error = ErrorBuilder.invalidRange(
            operation: "access",
            index: 100,
            count: 50
        )

        XCTAssertEqual(error.kind, .invalidRange)
        XCTAssertTrue(error.message.contains("out of bounds"))
        XCTAssertEqual(error.context.additionalInfo["index"], "100")
        XCTAssertEqual(error.context.additionalInfo["count"], "50")
    }

    func testErrorBuilderConvenienceInvalidParameter() {
        let error = ErrorBuilder.invalidParameter(
            operation: "kmeans",
            name: "k",
            value: "0",
            constraint: "must be > 0"
        )

        XCTAssertEqual(error.kind, .invalidParameter)
        XCTAssertTrue(error.message.contains("k"))
        XCTAssertTrue(error.message.contains("must be > 0"))
        XCTAssertEqual(error.context.additionalInfo["param_k"], "0")
    }

    // MARK: - Integration Tests

    func testErrorThrowingAndCatching() throws {
        func throwingFunction() throws {
            throw ErrorBuilder(.dimensionMismatch, operation: "test")
                .message("Test throw")
                .build()
        }

        XCTAssertThrowsError(try throwingFunction()) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .dimensionMismatch)
            XCTAssertEqual(indexError.message, "Test throw")
        }
    }

    func testErrorRecoverabilityPattern() {
        let error = ErrorBuilder(.dimensionMismatch, operation: "test")
            .build()

        // Pattern: Check if error is recoverable before retrying
        if error.kind.isRecoverable {
            // Can attempt recovery
            XCTAssertTrue(true)
        } else {
            XCTFail("Dimension mismatch should be recoverable")
        }
    }

    func testErrorChainingPattern() {
        // Simulate multi-layer error propagation
        func lowLevel() throws {
            throw ErrorBuilder(.fileIOError, operation: "open")
                .path("/tmp/test.bin")
                .errno(2)
                .build()
        }

        func midLevel() throws {
            do {
                try lowLevel()
            } catch let err {
                throw ErrorBuilder(.mmapError, operation: "mmap")
                    .underlying(err)
                    .build()
            }
        }

        func highLevel() throws {
            do {
                try midLevel()
            } catch let err {
                throw ErrorBuilder(.corruptedData, operation: "load_index")
                    .message("Failed to load index")
                    .underlying(err)
                    .build()
            }
        }

        XCTAssertThrowsError(try highLevel()) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            // Verify error chain
            XCTAssertEqual(indexError.errorChain.count, 3)
            XCTAssertEqual(indexError.kind, .corruptedData)
            XCTAssertEqual(indexError.rootCause.kind, .fileIOError)

            // Verify all layers present
            let kinds = indexError.errorChain.map { $0.kind }
            XCTAssertTrue(kinds.contains(.corruptedData))
            XCTAssertTrue(kinds.contains(.mmapError))
            XCTAssertTrue(kinds.contains(.fileIOError))
        }
    }
}
