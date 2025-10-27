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

    // MARK: - Phase 2 Migration Tests: IVFAppend

    func testIVFAppend_ThrowsOnInvalidKc() {
        // Test k_c <= 0
        XCTAssertThrowsError(try IVFListHandle(k_c: 0, m: 0, d: 128, opts: IVFAppendOpts(format: .flat))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError, got \(type(of: error))")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertTrue(indexError.kind.isRecoverable)
            XCTAssertEqual(indexError.context.operation, "ivf_create")
            XCTAssertEqual(indexError.context.additionalInfo["param_k_c"], "0")
            XCTAssertTrue(indexError.message.contains("must be > 0"))
        }

        XCTAssertThrowsError(try IVFListHandle(k_c: -5, m: 0, d: 128, opts: IVFAppendOpts(format: .flat))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["param_k_c"], "-5")
        }
    }

    func testIVFAppend_ThrowsOnInvalidDimensionForFlat() {
        // Test d <= 0 for flat format
        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 0, d: 0, opts: IVFAppendOpts(format: .flat))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.operation, "ivf_create")
            XCTAssertEqual(indexError.context.additionalInfo["param_d"], "0")
            XCTAssertTrue(indexError.message.contains("must be > 0 for flat format"))
        }

        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 0, d: -128, opts: IVFAppendOpts(format: .flat))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["param_d"], "-128")
        }
    }

    func testIVFAppend_ThrowsOnNonZeroMForFlat() {
        // Test m != 0 for flat format
        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 8, d: 128, opts: IVFAppendOpts(format: .flat))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.operation, "ivf_create")
            XCTAssertEqual(indexError.context.additionalInfo["param_m"], "8")
            XCTAssertTrue(indexError.message.contains("must be 0 for flat format"))
        }
    }

    func testIVFAppend_ThrowsOnInvalidMForPQ() {
        // Test m <= 0 for PQ format
        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 0, d: 0, opts: IVFAppendOpts(format: .pq8))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.operation, "ivf_create")
            XCTAssertEqual(indexError.context.additionalInfo["m"], "0")
            XCTAssertEqual(indexError.context.additionalInfo["d"], "0")
            XCTAssertTrue(indexError.message.contains("m must be > 0 and d must be 0"))
        }

        // Test d != 0 for PQ format
        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 8, d: 128, opts: IVFAppendOpts(format: .pq8, group: 4))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["m"], "8")
            XCTAssertEqual(indexError.context.additionalInfo["d"], "128")
        }
    }

    func testIVFAppend_ThrowsOnInvalidGroupSize() {
        // Test group not 4 or 8
        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 16, d: 0, opts: IVFAppendOpts(format: .pq8, group: 2))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.operation, "ivf_create")
            XCTAssertEqual(indexError.context.additionalInfo["param_group"], "2")
            XCTAssertTrue(indexError.message.contains("must be 4 or 8"))
        }

        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 16, d: 0, opts: IVFAppendOpts(format: .pq8, group: 16))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["param_group"], "16")
        }
    }

    func testIVFAppend_ThrowsOnMNotDivisibleByGroup() {
        // Test m % group != 0
        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 15, d: 0, opts: IVFAppendOpts(format: .pq8, group: 4))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.operation, "ivf_create")
            XCTAssertEqual(indexError.context.additionalInfo["m"], "15")
            XCTAssertEqual(indexError.context.additionalInfo["group"], "4")
            XCTAssertTrue(indexError.message.contains("m must be divisible by group size"))
        }

        XCTAssertThrowsError(try IVFListHandle(k_c: 10, m: 17, d: 0, opts: IVFAppendOpts(format: .pq8, group: 8))) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["m"], "17")
            XCTAssertEqual(indexError.context.additionalInfo["group"], "8")
        }
    }

    func testIVFAppend_SucceedsWithValidParameters() {
        // Test that valid parameters succeed
        XCTAssertNoThrow(try IVFListHandle(k_c: 10, m: 0, d: 128, opts: IVFAppendOpts(format: .flat)))
        XCTAssertNoThrow(try IVFListHandle(k_c: 100, m: 8, d: 0, opts: IVFAppendOpts(format: .pq8, group: 4)))
        XCTAssertNoThrow(try IVFListHandle(k_c: 50, m: 16, d: 0, opts: IVFAppendOpts(format: .pq8, group: 8)))
        XCTAssertNoThrow(try IVFListHandle(k_c: 1, m: 0, d: 1, opts: IVFAppendOpts(format: .flat)))
    }

    // MARK: - Phase 2 Migration Tests: KMeansSeeding

    func testKMeansSeeding_ThrowsOnInvalidDimension() {
        // Test d < 1
        var data = [Float](repeating: 0.0, count: 100)
        var centroids = [Float](repeating: 0.0, count: 10)

        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 10,
            dimension: 0,
            k: 2,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError, got \(type(of: error))")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidDimension)
            XCTAssertTrue(indexError.kind.isRecoverable)
            XCTAssertEqual(indexError.context.operation, "kmeans_seed")
            XCTAssertEqual(indexError.context.additionalInfo["dimension"], "0")
            XCTAssertTrue(indexError.message.contains("must be at least 1"))
        }

        // Test negative dimension
        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 10,
            dimension: -5,
            k: 2,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .invalidDimension)
            XCTAssertEqual(indexError.context.additionalInfo["dimension"], "-5")
        }
    }

    func testKMeansSeeding_ThrowsOnInvalidCount() {
        // Test n < 1
        var data = [Float](repeating: 0.0, count: 10)
        var centroids = [Float](repeating: 0.0, count: 10)

        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 0,
            dimension: 10,
            k: 2,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertTrue(indexError.kind.isRecoverable)
            XCTAssertEqual(indexError.context.operation, "kmeans_seed")
            XCTAssertEqual(indexError.context.additionalInfo["param_n"], "0")
            XCTAssertTrue(indexError.message.contains("must be >= 1"))
        }

        // Test negative count
        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: -10,
            dimension: 10,
            k: 2,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["param_n"], "-10")
        }
    }

    func testKMeansSeeding_ThrowsOnKLessThanOne() {
        // Test k < 1
        var data = [Float](repeating: 0.0, count: 100)
        var centroids = [Float](repeating: 0.0, count: 10)

        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 10,
            dimension: 10,
            k: 0,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.operation, "kmeans_seed")
            XCTAssertEqual(indexError.context.additionalInfo["param_k"], "0")
            XCTAssertTrue(indexError.message.contains("must be >= 1"))
        }

        // Test negative k
        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 10,
            dimension: 10,
            k: -5,
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["param_k"], "-5")
        }
    }

    func testKMeansSeeding_ThrowsOnKGreaterThanN() {
        // Test k > n
        var data = [Float](repeating: 0.0, count: 50) // 5 vectors of dim 10
        var centroids = [Float](repeating: 0.0, count: 100) // space for 10 centroids

        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 5,
            dimension: 10,
            k: 10, // k > n (10 > 5)
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }

            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertTrue(indexError.kind.isRecoverable)
            XCTAssertEqual(indexError.context.operation, "kmeans_seed")
            XCTAssertEqual(indexError.context.additionalInfo["k"], "10")
            XCTAssertEqual(indexError.context.additionalInfo["n"], "5")
            XCTAssertTrue(indexError.message.contains("must not exceed number of data points"))
        }

        // Test k = n + 1
        XCTAssertThrowsError(try kmeansPlusPlusSeed(
            data: &data,
            count: 5,
            dimension: 10,
            k: 6, // k = n + 1
            centroidsOut: &centroids,
            chosenIndicesOut: nil
        )) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError")
                return
            }
            XCTAssertEqual(indexError.kind, .invalidParameter)
            XCTAssertEqual(indexError.context.additionalInfo["k"], "6")
            XCTAssertEqual(indexError.context.additionalInfo["n"], "5")
        }
    }

    // Note: Success case tests for kmeansPlusPlusSeed are covered in KMeansPPSeedingTests.swift
    // The tests there use properly aligned data and cover all valid parameter combinations.
    // This test file focuses on error path validation to ensure proper error handling.

    // MARK: - Phase 2 Migration Tests: storeExternalID (IVFAppend - Phase 1)

    func testIVFAppend_32BitIDStorage_NoInternalInconsistency() throws {
        // Regression test: Verify that normal append operations with 32-bit IDs
        // don't trigger the new internal inconsistency error after storeExternalID migration
        let opts = IVFAppendOpts(format: .pq8, id_bits: 32)
        let handle = try IVFListHandle(k_c: 10, m: 8, d: 0, opts: opts)

        // Prepare test data: 100 vectors
        let n = 100
        var listIDs = [Int32](repeating: 0, count: n)
        var externalIDs = [UInt64](repeating: 0, count: n)
        var codes = [UInt8](repeating: 0, count: n * 8)

        // Distribute across lists
        for i in 0..<n {
            listIDs[i] = Int32(i % 10)
            externalIDs[i] = UInt64(i + 1000) // All IDs fit in UInt32
        }

        // Append vectors - should succeed without storage mismatch errors
        XCTAssertNoThrow(
            try ivf_append(
                list_ids: listIDs,
                external_ids: externalIDs,
                codes: codes,
                n: n,
                m: 8,
                index: handle,
                opts: nil,
                internalIDsOut: nil
            )
        )

        // Verify data was stored correctly
        for listID in 0..<10 {
            let stats = try handle.getListStats(listID: Int32(listID))
            XCTAssertTrue(stats.length > 0, "List \(listID) should contain vectors")
        }
    }

    func testIVFAppend_64BitIDStorage_NoInternalInconsistency() throws {
        // Regression test: Verify that normal append operations with 64-bit IDs
        // don't trigger the new internal inconsistency error after storeExternalID migration
        let opts = IVFAppendOpts(format: .pq8, id_bits: 64)
        let handle = try IVFListHandle(k_c: 5, m: 8, d: 0, opts: opts)

        // Prepare test data with large IDs (> UInt32.max)
        let n = 50
        var listIDs = [Int32](repeating: 0, count: n)
        var externalIDs = [UInt64](repeating: 0, count: n)
        var codes = [UInt8](repeating: 0, count: n * 8)

        for i in 0..<n {
            listIDs[i] = Int32(i % 5)
            // Use IDs larger than UInt32.max to ensure we're truly testing 64-bit storage
            externalIDs[i] = UInt64(UInt32.max) + UInt64(i + 1)
        }

        // Append vectors - should succeed without storage mismatch errors
        XCTAssertNoThrow(
            try ivf_append(
                list_ids: listIDs,
                external_ids: externalIDs,
                codes: codes,
                n: n,
                m: 8,
                index: handle,
                opts: nil,
                internalIDsOut: nil
            )
        )

        // Verify data was stored correctly
        for listID in 0..<5 {
            let stats = try handle.getListStats(listID: Int32(listID))
            XCTAssertTrue(stats.length > 0, "List \(listID) should contain vectors")
        }
    }

    func testIVFAppend_32BitIDs_ThrowsOnOversizedID() throws {
        // Verify that the existing idWidthUnsupported error still works after migration
        let opts = IVFAppendOpts(format: .pq8, id_bits: 32)
        let handle = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts)

        // Attempt to store an ID larger than UInt32.max
        var listIDs: [Int32] = [0]
        var externalIDs: [UInt64] = [UInt64(UInt32.max) + 1] // Too large for 32-bit storage
        var codes = [UInt8](repeating: 0, count: 8)

        XCTAssertThrowsError(
            try ivf_append(
                list_ids: listIDs,
                external_ids: externalIDs,
                codes: codes,
                n: 1,
                m: 8,
                index: handle,
                opts: nil,
                internalIDsOut: nil
            )
        ) { error in
            // Should throw VectorIndexError with .invalidParameter, not internal inconsistency
            guard let vecError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError, got \(type(of: error))")
                return
            }
            XCTAssertEqual(vecError.kind, .invalidParameter)
            XCTAssertTrue(vecError.message.contains("32-bit maximum") || vecError.message.contains("ID value exceeds"))
        }
    }

    func testIVFAppendFlat_32And64BitIDs_NoInternalInconsistency() throws {
        // Test flat format with both ID sizes to ensure storeExternalID works in all code paths

        // Test 32-bit IDs
        do {
            let opts32 = IVFAppendOpts(format: .flat, id_bits: 32)
            let handle32 = try IVFListHandle(k_c: 3, m: 0, d: 128, opts: opts32)

            let n = 30
            var listIDs = [Int32](repeating: 0, count: n)
            var externalIDs = [UInt64](repeating: 0, count: n)
            var vectors = [Float](repeating: 1.0, count: n * 128)

            for i in 0..<n {
                listIDs[i] = Int32(i % 3)
                externalIDs[i] = UInt64(i + 5000)
            }

            XCTAssertNoThrow(
                try ivf_append_flat(
                    list_ids: listIDs,
                    external_ids: externalIDs,
                    xb: vectors,
                    n: n,
                    d: 128,
                    index: handle32,
                    opts: nil,
                    internalIDsOut: nil
                )
            )
        }

        // Test 64-bit IDs
        do {
            let opts64 = IVFAppendOpts(format: .flat, id_bits: 64)
            let handle64 = try IVFListHandle(k_c: 3, m: 0, d: 128, opts: opts64)

            let n = 30
            var listIDs = [Int32](repeating: 0, count: n)
            var externalIDs = [UInt64](repeating: 0, count: n)
            var vectors = [Float](repeating: 1.0, count: n * 128)

            for i in 0..<n {
                listIDs[i] = Int32(i % 3)
                externalIDs[i] = UInt64(UInt32.max) + UInt64(i + 1000)
            }

            XCTAssertNoThrow(
                try ivf_append_flat(
                    list_ids: listIDs,
                    external_ids: externalIDs,
                    xb: vectors,
                    n: n,
                    d: 128,
                    index: handle64,
                    opts: nil,
                    internalIDsOut: nil
                )
            )
        }
    }

    func testIVFInsertAt_32And64BitIDs_NoInternalInconsistency() throws {
        // Test ivf_insert_at with both ID sizes to ensure storeExternalID works in insert operations

        // Test 32-bit IDs
        do {
            let opts32 = IVFAppendOpts(format: .pq8, id_bits: 32)
            let handle32 = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts32)

            // First append some vectors
            var listIDs: [Int32] = [0, 0, 0]
            var externalIDs: [UInt64] = [100, 200, 300]
            var codes = [UInt8](repeating: 1, count: 3 * 8)
            try ivf_append(list_ids: listIDs, external_ids: externalIDs, codes: codes, n: 3, m: 8, index: handle32, opts: nil, internalIDsOut: nil)

            // Insert in the middle
            var insertIDs: [UInt64] = [150]
            var insertCodes = [UInt8](repeating: 2, count: 8)
            XCTAssertNoThrow(
                try ivf_insert_at(
                    list_id: 0,
                    pos: 1,
                    external_ids: insertIDs,
                    codes: insertCodes,
                    n: 1,
                    index: handle32
                )
            )

            let stats = try handle32.getListStats(listID: 0)
            XCTAssertEqual(stats.length, 4)
        }

        // Test 64-bit IDs
        do {
            let opts64 = IVFAppendOpts(format: .pq8, id_bits: 64)
            let handle64 = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts64)

            // First append some vectors with large IDs
            var listIDs: [Int32] = [0, 0]
            var externalIDs: [UInt64] = [UInt64(UInt32.max) + 100, UInt64(UInt32.max) + 300]
            var codes = [UInt8](repeating: 1, count: 2 * 8)
            try ivf_append(list_ids: listIDs, external_ids: externalIDs, codes: codes, n: 2, m: 8, index: handle64, opts: nil, internalIDsOut: nil)

            // Insert with large ID
            var insertIDs: [UInt64] = [UInt64(UInt32.max) + 200]
            var insertCodes = [UInt8](repeating: 2, count: 8)
            XCTAssertNoThrow(
                try ivf_insert_at(
                    list_id: 0,
                    pos: 1,
                    external_ids: insertIDs,
                    codes: insertCodes,
                    n: 1,
                    index: handle64
                )
            )

            let stats = try handle64.getListStats(listID: 0)
            XCTAssertEqual(stats.length, 3)
        }
    }

    // Note: Direct testing of the storage mismatch error (VectorIndexError(.internalInconsistency))
    // is not feasible without exposing internal implementation details or using unsafe pointer manipulation.
    // The error represents defensive programming for conditions that should never occur in normal usage:
    // - opts.id_bits modified after storage allocation
    // - Memory corruption of IDStorage enum discriminant
    // - Bug in storage initialization logic
    //
    // The tests above verify that:
    // 1. Normal operations with 32-bit IDs work correctly (no false positives)
    // 2. Normal operations with 64-bit IDs work correctly (no false positives)
    // 3. The existing idWidthUnsupported error still works
    // 4. All code paths that call storeExternalID execute without triggering the error

    // MARK: - Phase 2 Migration Tests: growList (IVFAppend - Phase 2)

    func testIVFAppend_ListGrowth_32BitIDs_NoInternalInconsistency() throws {
        // Test that list growth (capacity increase) works correctly with 32-bit IDs
        // This exercises the growList function's storage type matching logic

        let opts = IVFAppendOpts(
            format: .pq8,
            reserve_factor: 1.5, // Moderate growth factor
            reserve_min: 10,     // Small initial capacity
            id_bits: 32
        )
        let handle = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts)

        // Append enough vectors to trigger multiple growth operations
        let batchSize = 15
        let numBatches = 7  // Will append 7 * 15 = 105 vectors
        let totalVectors = numBatches * batchSize

        for batch in 0..<numBatches {
            var listIDs = [Int32](repeating: 0, count: batchSize)
            var externalIDs = [UInt64](repeating: 0, count: batchSize)
            var codes = [UInt8](repeating: UInt8(batch % 256), count: batchSize * 8)

            for i in 0..<batchSize {
                externalIDs[i] = UInt64(batch * batchSize + i + 1000)
            }

            // Each append may trigger list growth
            XCTAssertNoThrow(
                try ivf_append(
                    list_ids: listIDs,
                    external_ids: externalIDs,
                    codes: codes,
                    n: batchSize,
                    m: 8,
                    index: handle,
                    opts: nil,
                    internalIDsOut: nil
                ),
                "List growth should succeed for 32-bit IDs (batch \(batch))"
            )
        }

        // Verify all vectors were stored
        let stats = try handle.getListStats(listID: 0)
        XCTAssertEqual(stats.length, totalVectors, "All vectors should be stored after growth")
        XCTAssertGreaterThan(stats.capacity, 10, "Capacity should have grown beyond initial reserve_min")
    }

    func testIVFAppend_ListGrowth_64BitIDs_NoInternalInconsistency() throws {
        // Test that list growth works correctly with 64-bit IDs
        // Uses large IDs (> UInt32.max) to ensure we're truly testing 64-bit storage

        let opts = IVFAppendOpts(
            format: .pq8,
            reserve_factor: 2.0, // Aggressive growth
            reserve_min: 8,      // Small initial capacity to force growth
            id_bits: 64
        )
        let handle = try IVFListHandle(k_c: 2, m: 8, d: 0, opts: opts)

        // Append vectors with large IDs across multiple lists
        let totalVectors = 80
        let batchSize = 12

        for batch in 0..<(totalVectors / batchSize) {
            var listIDs = [Int32](repeating: 0, count: batchSize)
            var externalIDs = [UInt64](repeating: 0, count: batchSize)
            var codes = [UInt8](repeating: UInt8(batch % 256), count: batchSize * 8)

            for i in 0..<batchSize {
                listIDs[i] = Int32(i % 2)  // Distribute across 2 lists
                // Use IDs larger than UInt32.max
                externalIDs[i] = UInt64(UInt32.max) + UInt64(batch * batchSize + i + 5000)
            }

            XCTAssertNoThrow(
                try ivf_append(
                    list_ids: listIDs,
                    external_ids: externalIDs,
                    codes: codes,
                    n: batchSize,
                    m: 8,
                    index: handle,
                    opts: nil,
                    internalIDsOut: nil
                ),
                "List growth should succeed for 64-bit IDs (batch \(batch))"
            )
        }

        // Verify both lists received vectors and grew
        for listID in 0..<2 {
            let stats = try handle.getListStats(listID: Int32(listID))
            XCTAssertGreaterThan(stats.length, 0, "List \(listID) should contain vectors")
            XCTAssertGreaterThan(stats.capacity, 8, "List \(listID) capacity should have grown")
        }
    }

    func testIVFAppendFlat_ListGrowth_32And64BitIDs_NoInternalInconsistency() throws {
        // Test that growList works correctly for flat format (vector storage instead of codes)

        // Test 32-bit IDs with flat format
        do {
            let opts32 = IVFAppendOpts(
                format: .flat,
                reserve_min: 5,  // Very small to force multiple growths
                id_bits: 32
            )
            let handle32 = try IVFListHandle(k_c: 1, m: 0, d: 64, opts: opts32)

            // Append in batches to trigger growth
            for batch in 0..<10 {
                var listIDs: [Int32] = [0, 0, 0, 0, 0]
                var externalIDs: [UInt64] = [
                    UInt64(batch * 5 + 0),
                    UInt64(batch * 5 + 1),
                    UInt64(batch * 5 + 2),
                    UInt64(batch * 5 + 3),
                    UInt64(batch * 5 + 4)
                ]
                var vectors = [Float](repeating: Float(batch), count: 5 * 64)

                XCTAssertNoThrow(
                    try ivf_append_flat(
                        list_ids: listIDs,
                        external_ids: externalIDs,
                        xb: vectors,
                        n: 5,
                        d: 64,
                        index: handle32,
                        opts: nil,
                        internalIDsOut: nil
                    ),
                    "Flat format growth should succeed for 32-bit IDs (batch \(batch))"
                )
            }

            let stats32 = try handle32.getListStats(listID: 0)
            XCTAssertEqual(stats32.length, 50)
            XCTAssertGreaterThan(stats32.capacity, 5)
        }

        // Test 64-bit IDs with flat format
        do {
            let opts64 = IVFAppendOpts(
                format: .flat,
                reserve_min: 6,
                id_bits: 64
            )
            let handle64 = try IVFListHandle(k_c: 1, m: 0, d: 32, opts: opts64)

            for batch in 0..<8 {
                var listIDs: [Int32] = [0, 0, 0]
                var externalIDs: [UInt64] = [
                    UInt64(UInt32.max) + UInt64(batch * 3 + 0),
                    UInt64(UInt32.max) + UInt64(batch * 3 + 1),
                    UInt64(UInt32.max) + UInt64(batch * 3 + 2)
                ]
                var vectors = [Float](repeating: Float(batch + 1), count: 3 * 32)

                XCTAssertNoThrow(
                    try ivf_append_flat(
                        list_ids: listIDs,
                        external_ids: externalIDs,
                        xb: vectors,
                        n: 3,
                        d: 32,
                        index: handle64,
                        opts: nil,
                        internalIDsOut: nil
                    ),
                    "Flat format growth should succeed for 64-bit IDs (batch \(batch))"
                )
            }

            let stats64 = try handle64.getListStats(listID: 0)
            XCTAssertEqual(stats64.length, 24)
            XCTAssertGreaterThan(stats64.capacity, 6)
        }
    }

    func testIVFInsertAt_TriggeringGrowth_NoInternalInconsistency() throws {
        // Test that ivf_insert_at can trigger list growth without storage mismatch errors

        let opts = IVFAppendOpts(
            format: .pq8,
            reserve_min: 3,  // Very small capacity to force growth
            id_bits: 64
        )
        let handle = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts)

        // Initial append (fits in initial capacity)
        var initialIDs: [Int32] = [0, 0]
        var initialExtIDs: [UInt64] = [100, 200]
        var initialCodes = [UInt8](repeating: 1, count: 2 * 8)
        try ivf_append(
            list_ids: initialIDs,
            external_ids: initialExtIDs,
            codes: initialCodes,
            n: 2,
            m: 8,
            index: handle,
            opts: nil,
            internalIDsOut: nil
        )

        // Insert operations that will trigger growth
        for i in 0..<5 {
            var insertID: [UInt64] = [UInt64(UInt32.max) + UInt64(i * 100)]
            var insertCodes = [UInt8](repeating: UInt8(i + 2), count: 8)

            XCTAssertNoThrow(
                try ivf_insert_at(
                    list_id: 0,
                    pos: i + 1,  // Insert in middle
                    external_ids: insertID,
                    codes: insertCodes,
                    n: 1,
                    index: handle
                ),
                "Insert triggering growth should succeed (iteration \(i))"
            )
        }

        let stats = try handle.getListStats(listID: 0)
        XCTAssertEqual(stats.length, 7, "Should have 2 initial + 5 inserted vectors")
        XCTAssertGreaterThan(stats.capacity, 3, "Capacity should have grown beyond initial reserve_min")
    }

    // Note: Direct testing of the growList storage mismatch error is not feasible for the same
    // reasons as storeExternalID - it requires internal state corruption that shouldn't occur
    // during normal operations. The tests above verify that:
    // 1. List growth works correctly with 32-bit IDs (no false positives)
    // 2. List growth works correctly with 64-bit IDs (no false positives)
    // 3. Growth works for both PQ and flat formats
    // 4. Growth triggered by both append and insert operations succeeds
    // 5. Multiple consecutive growth operations work correctly

    // MARK: - Phase 3 Migration Tests: ivf_insert_at format validation (IVFAppend - Phase 3)

    func testIVFInsertAt_ThrowsOnFlatFormat() throws {
        // Test that ivf_insert_at properly rejects flat format indices
        // User should use ivf_insert_at_flat instead

        let opts = IVFAppendOpts(format: .flat)
        let handle = try IVFListHandle(k_c: 1, m: 0, d: 128, opts: opts)

        // First append some vectors using the correct function for flat format
        var listIDs: [Int32] = [0, 0, 0]
        var externalIDs: [UInt64] = [100, 200, 300]
        var vectors = [Float](repeating: 1.0, count: 3 * 128)
        try ivf_append_flat(
            list_ids: listIDs,
            external_ids: externalIDs,
            xb: vectors,
            n: 3,
            d: 128,
            index: handle,
            opts: nil,
            internalIDsOut: nil
        )

        // Now attempt to use ivf_insert_at (wrong function for flat format)
        var insertIDs: [UInt64] = [150]
        var dummyCodes = [UInt8](repeating: 0, count: 8)  // Codes parameter is ignored but required

        XCTAssertThrowsError(
            try ivf_insert_at(
                list_id: 0,
                pos: 1,
                external_ids: insertIDs,
                codes: dummyCodes,
                n: 1,
                index: handle
            )
        ) { error in
            guard let indexError = error as? VectorIndexError else {
                XCTFail("Expected VectorIndexError, got \(type(of: error))")
                return
            }

            // Verify error details
            XCTAssertEqual(indexError.kind, .unsupportedLayout)
            XCTAssertTrue(indexError.kind.isRecoverable, "Format error should be recoverable")
            XCTAssertEqual(indexError.context.operation, "ivf_insert_at")
            XCTAssertTrue(
                indexError.message.contains("PQ format"),
                "Error message should mention PQ format requirement"
            )
            XCTAssertTrue(
                indexError.message.contains("ivf_insert_at_flat"),
                "Error message should suggest correct function"
            )
            XCTAssertEqual(indexError.context.additionalInfo["actual_format"], "flat")

            // Verify recovery message is helpful
            let recovery = indexError.recoveryMessage
            XCTAssertFalse(recovery.isEmpty, "Should provide recovery guidance")
        }
    }

    func testIVFInsertAt_SucceedsWithPQ8Format() throws {
        // Regression test: Verify ivf_insert_at still works correctly with PQ8 format

        let opts = IVFAppendOpts(format: .pq8)
        let handle = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts)

        // Append initial vectors
        var listIDs: [Int32] = [0, 0, 0]
        var externalIDs: [UInt64] = [100, 200, 300]
        var codes = [UInt8](repeating: 1, count: 3 * 8)
        try ivf_append(
            list_ids: listIDs,
            external_ids: externalIDs,
            codes: codes,
            n: 3,
            m: 8,
            index: handle,
            opts: nil,
            internalIDsOut: nil
        )

        // Insert should succeed for PQ8 format
        var insertIDs: [UInt64] = [150]
        var insertCodes = [UInt8](repeating: 2, count: 8)

        XCTAssertNoThrow(
            try ivf_insert_at(
                list_id: 0,
                pos: 1,
                external_ids: insertIDs,
                codes: insertCodes,
                n: 1,
                index: handle
            ),
            "ivf_insert_at should succeed with PQ8 format"
        )

        // Verify insertion worked
        let stats = try handle.getListStats(listID: 0)
        XCTAssertEqual(stats.length, 4, "Should have 3 initial + 1 inserted vector")
    }

    func testIVFInsertAt_SucceedsWithPQ4Format() throws {
        // Regression test: Verify ivf_insert_at works correctly with PQ4 format

        let opts = IVFAppendOpts(format: .pq4, group: 4)
        let handle = try IVFListHandle(k_c: 1, m: 8, d: 0, opts: opts)

        // Append initial vectors
        var listIDs: [Int32] = [0, 0]
        var externalIDs: [UInt64] = [100, 200]
        var codes = [UInt8](repeating: 1, count: 2 * 4)  // PQ4: m/2 = 8/2 = 4 bytes per vector
        try ivf_append(
            list_ids: listIDs,
            external_ids: externalIDs,
            codes: codes,
            n: 2,
            m: 8,
            index: handle,
            opts: nil,
            internalIDsOut: nil
        )

        // Insert should succeed for PQ4 format
        var insertIDs: [UInt64] = [150]
        var insertCodes = [UInt8](repeating: 2, count: 4)

        XCTAssertNoThrow(
            try ivf_insert_at(
                list_id: 0,
                pos: 1,
                external_ids: insertIDs,
                codes: insertCodes,
                n: 1,
                index: handle
            ),
            "ivf_insert_at should succeed with PQ4 format"
        )

        // Verify insertion worked
        let stats = try handle.getListStats(listID: 0)
        XCTAssertEqual(stats.length, 3, "Should have 2 initial + 1 inserted vector")
    }

    func testIVFInsertAtFlat_SucceedsWithFlatFormat() throws {
        // Complementary test: Verify ivf_insert_at_flat works correctly with flat format
        // This demonstrates the correct API usage pattern

        let opts = IVFAppendOpts(format: .flat)
        let handle = try IVFListHandle(k_c: 1, m: 0, d: 64, opts: opts)

        // Append initial vectors using ivf_append_flat
        var listIDs: [Int32] = [0, 0]
        var externalIDs: [UInt64] = [100, 300]
        var vectors = [Float](repeating: 1.0, count: 2 * 64)
        try ivf_append_flat(
            list_ids: listIDs,
            external_ids: externalIDs,
            xb: vectors,
            n: 2,
            d: 64,
            index: handle,
            opts: nil,
            internalIDsOut: nil
        )

        // Insert using the correct function for flat format
        var insertIDs: [UInt64] = [200]
        var insertVectors = [Float](repeating: 2.0, count: 64)

        XCTAssertNoThrow(
            try ivf_insert_at_flat(
                list_id: 0,
                pos: 1,
                external_ids: insertIDs,
                xb: insertVectors,
                n: 1,
                index: handle
            ),
            "ivf_insert_at_flat should succeed with flat format"
        )

        // Verify insertion worked
        let stats = try handle.getListStats(listID: 0)
        XCTAssertEqual(stats.length, 3, "Should have 2 initial + 1 inserted vector")
    }

    func testIVFInsertAt_ErrorMessageQuality() throws {
        // Verify that the error message provides clear, actionable guidance

        let opts = IVFAppendOpts(format: .flat)
        let handle = try IVFListHandle(k_c: 1, m: 0, d: 128, opts: opts)

        // Append one vector to ensure list is non-empty
        var listIDs: [Int32] = [0]
        var externalIDs: [UInt64] = [100]
        var vectors = [Float](repeating: 1.0, count: 128)
        try ivf_append_flat(
            list_ids: listIDs,
            external_ids: externalIDs,
            xb: vectors,
            n: 1,
            d: 128,
            index: handle,
            opts: nil,
            internalIDsOut: nil
        )

        // Attempt incorrect function
        var insertIDs: [UInt64] = [200]
        var dummyCodes = [UInt8](repeating: 0, count: 8)

        do {
            try ivf_insert_at(
                list_id: 0,
                pos: 0,
                external_ids: insertIDs,
                codes: dummyCodes,
                n: 1,
                index: handle
            )
            XCTFail("Should have thrown unsupportedLayout error")
        } catch let error as VectorIndexError {
            // Check error message quality
            let fullDescription = error.description
            XCTAssertTrue(
                fullDescription.contains("PQ format"),
                "Full description should explain format requirement"
            )

            let shortDescription = error.shortDescription
            XCTAssertFalse(
                shortDescription.isEmpty,
                "Should have concise user-facing description"
            )

            // Verify metadata is complete
            XCTAssertEqual(error.context.operation, "ivf_insert_at")
            XCTAssertNotNil(error.context.additionalInfo["actual_format"])
            XCTAssertEqual(error.kind.category, .configuration)
        } catch {
            XCTFail("Expected VectorIndexError, got \(type(of: error))")
        }
    }
}
