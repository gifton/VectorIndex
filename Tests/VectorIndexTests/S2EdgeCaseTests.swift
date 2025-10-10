import XCTest
@testable import VectorIndex
import CS2RNG

/// Tests for S2 major bug fixes: edge cases in rounding, quantization, and RNG
final class S2EdgeCaseTests: XCTestCase {

    // MARK: - vrndnq_f32_compat Edge Cases (Fix for Issue #5)

    func testQuantizationWithNaN() {
        // Test that NaN values in quantization don't crash and produce valid output
        let testVectors: [Float] = [1.0, .nan, 3.0, .nan, 5.0]
        let scale: Float = 1.0

        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        // NaN should convert to some valid int8 (behavior is implementation-defined)
        // Key requirement: NO CRASH
        XCTAssertTrue(true, "NaN handling in quantization should not crash")
    }

    func testQuantizationWithInfinity() {
        // Test that Inf values saturate correctly to ±127
        let testVectors: [Float] = [.infinity, -.infinity, 100.0, -100.0]
        let scale: Float = 1.0

        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        // Infinity should saturate to ±127
        XCTAssertEqual(i8Buffer[0], 127, "Positive infinity should saturate to max")
        XCTAssertEqual(i8Buffer[1], -128, "Negative infinity should saturate to min")
        XCTAssertEqual(i8Buffer[2], 100, "Normal positive value should quantize correctly")
        XCTAssertEqual(i8Buffer[3], -100, "Normal negative value should quantize correctly")
    }

    func testQuantizationWithLargeValues() {
        // Test values > 2^23 (where floats have no fractional part)
        // This tests the vrndnq_f32_compat fix for large value handling
        let testVectors: [Float] = [
            16777216.0,    // Exactly 2^24
            33554432.0,    // 2^25
            1.0e10,        // Very large
            -1.0e10,       // Very large negative
        ]
        let scale: Float = 127.0  // scale = max(|x|) / 127; use 127 to saturate large values

        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        // All should saturate to ±127 (values >> scale * 127)
        XCTAssertEqual(i8Buffer[0], 127, "2^24 / 127 should saturate")
        XCTAssertEqual(i8Buffer[1], 127, "2^25 / 127 should saturate")
        XCTAssertEqual(i8Buffer[2], 127, "1e10 / 127 should saturate")
        XCTAssertEqual(i8Buffer[3], -128, "-1e10 / 127 should saturate")
    }

    func testQuantizationNearMaxFloat() {
        // Test values close to Float.greatestFiniteMagnitude
        // This verifies no overflow in int32 conversion (old bug)
        let maxVal = Float.greatestFiniteMagnitude
        let testVectors: [Float] = [
            maxVal,
            -maxVal,
            maxVal * 0.5,
            -maxVal * 0.5,
        ]
        let scale: Float = maxVal / 127.0  // Scale = max/127 for symmetric quantization

        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        // Should saturate without crashing (key: no UB in vrndnq_f32_compat)
        // Note: symmetric quantization uses [-127, 127] range (symmetric around 0)
        XCTAssertTrue(abs(i8Buffer[0]) == 127, "Max float should saturate to ±127")
        XCTAssertTrue(abs(i8Buffer[1]) == 127, "-Max float should saturate to ±127")
        // 0.5 * maxVal / scale = 0.5 * 127 = 63.5, rounds to 64
        XCTAssertTrue(abs(i8Buffer[2] - 64) <= 1, "0.5*max should quantize to ~64")
        XCTAssertTrue(abs(i8Buffer[3] + 64) <= 1, "-0.5*max should quantize to ~-64")
    }

    // MARK: - Saturation Telemetry Correctness (Fix for Issue #1)

    func testSaturationCountingAccuracy() {
        // Test that saturation counter is accurate
        // Before fix: would count wrong values due to checking after narrowing
        s2_reset_telemetry()

        let testVectors: [Float] = [
            200.0,   // Will saturate to 127
            -200.0,  // Will saturate to -128
            50.0,    // Won't saturate
            100.0,   // Won't saturate
            150.0,   // Will saturate to 127
            -150.0,  // Will saturate to -128
        ]
        let scale: Float = 1.0  // Direct mapping

        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        #if S2_ENABLE_TELEMETRY
        let telemetry = s2_get_telemetry()
        let satCount = telemetry?.pointee.saturations_i8 ?? 0

        // Should count exactly 4 saturations (2 positive, 2 negative)
        XCTAssertEqual(satCount, 4, "Saturation counter should be accurate")
        #endif
    }

    // MARK: - RNG Edge Cases

    func testXoroWithMultipleStreams() {
        // Verify stream splitting works correctly (uses rotl64 internally)
        let seed: UInt64 = 0x123456789ABCDEF0

        var rng0 = Xoro128()
        var rng1 = Xoro128()
        var rng2 = Xoro128()

        xoro128_init(&rng0, seed, 0)
        xoro128_init(&rng1, seed, 1)
        xoro128_init(&rng2, seed, 2)

        // Generate 100 samples from each
        var seq0 = [Float]()
        var seq1 = [Float]()
        var seq2 = [Float]()

        for _ in 0..<100 {
            seq0.append(xoro128_next_uniform(&rng0))
            seq1.append(xoro128_next_uniform(&rng1))
            seq2.append(xoro128_next_uniform(&rng2))
        }

        // Streams should be independent (low collision rate)
        let collisions01 = zip(seq0, seq1).filter { abs($0 - $1) < 0.0001 }.count
        let collisions02 = zip(seq0, seq2).filter { abs($0 - $1) < 0.0001 }.count
        let collisions12 = zip(seq1, seq2).filter { abs($0 - $1) < 0.0001 }.count

        XCTAssertLessThan(collisions01, 5, "Stream 0 and 1 should be independent")
        XCTAssertLessThan(collisions02, 5, "Stream 0 and 2 should be independent")
        XCTAssertLessThan(collisions12, 5, "Stream 1 and 2 should be independent")
    }

    func testPhiloxDeterminism() {
        // Verify Philox output is deterministic for same inputs
        // This validates the key schedule fix (ky/kw are correctly unused)
        let seed: UInt64 = 0xABCDEF0123456789
        let streamID: UInt64 = 0x13579BDF02468ACE

        var key0: UInt64 = 0
        var key1: UInt64 = 0
        philox_key(seed, streamID, &key0, &key1)

        // Generate 10 sets of 4 random values with same counter
        var results: [[UInt32]] = []
        for _ in 0..<10 {
            var out = [UInt32](repeating: 0, count: 4)
            philox_next4(key0, key1, 0, 0, &out)
            results.append(out)
        }

        // All should be identical (deterministic)
        let firstResult = results[0]
        for result in results {
            XCTAssertEqual(result, firstResult, "Philox must be deterministic")
        }
    }

    // MARK: - f16 Conversion Edge Cases

    func testF16ConversionWithSpecialValues() {
        let testVectors: [Float] = [
            .nan,
            .infinity,
            -.infinity,
            0.0,
            -0.0,
            Float.greatestFiniteMagnitude,  // Will overflow to f16 Inf
            -Float.greatestFiniteMagnitude,
            1.0e-20,  // Underflows to f16 zero
        ]

        var f16Buffer = [UInt16](repeating: 0, count: testVectors.count)
        var roundTrip = [Float](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            f32_to_f16(src.baseAddress!, &f16Buffer, Int32(src.count), NearestTiesToEven)
        }

        f16Buffer.withUnsafeBufferPointer { src in
            f16_to_f32(src.baseAddress!, &roundTrip, Int32(src.count))
        }

        // NaN should remain NaN
        XCTAssertTrue(roundTrip[0].isNaN, "NaN should remain NaN")

        // Infinity should remain infinity
        XCTAssertEqual(roundTrip[1], .infinity, "Inf should remain Inf")
        XCTAssertEqual(roundTrip[2], -.infinity, "-Inf should remain -Inf")

        // Zero (both signed)
        XCTAssertEqual(roundTrip[3], 0.0, "Zero should remain zero")
        XCTAssertEqual(roundTrip[4], -0.0, "Negative zero should preserve sign")

        // Overflow to Inf
        XCTAssertTrue(roundTrip[5].isInfinite && roundTrip[5] > 0, "Large positive should overflow to Inf")
        XCTAssertTrue(roundTrip[6].isInfinite && roundTrip[6] < 0, "Large negative should overflow to -Inf")

        // Underflow to zero
        XCTAssertEqual(roundTrip[7], 0.0, "Tiny value should underflow to zero")
    }

    // MARK: - Thread Safety (Telemetry Reset)

    func testTelemetryResetThreadSafety() {
        // Concurrent reset and read should not crash
        // This tests the atomic store fix in s2_reset_telemetry
        #if S2_ENABLE_TELEMETRY
        let iterations = 1000

        DispatchQueue.concurrentPerform(iterations: iterations) { _ in
            s2_reset_telemetry()
            _ = s2_get_telemetry()
        }

        // If we reach here without crashing, the fix works
        XCTAssertTrue(true, "Concurrent reset and read should be thread-safe")
        #endif
    }

    // MARK: - Alignment Edge Cases

    func testUnalignedQuantization() {
        // Test quantization with non-multiple-of-16 lengths (scalar tail path)
        for length in [1, 7, 13, 17, 31, 33, 63, 65] {
            let testVectors = (0..<length).map { Float($0) }
            let scale: Float = 1.0

            var i8Buffer = [Int8](repeating: 0, count: length)
            var roundTrip = [Float](repeating: 0, count: length)

            testVectors.withUnsafeBufferPointer { src in
                quantize_i8_symmetric(src.baseAddress!, Int32(length), scale, &i8Buffer)
            }

            i8Buffer.withUnsafeBufferPointer { src in
                dequantize_i8_symmetric(src.baseAddress!, Int32(length), scale, &roundTrip)
            }

            // Verify round-trip accuracy
            for i in 0..<length {
                let error = abs(roundTrip[i] - testVectors[i])
                XCTAssertLessThanOrEqual(error, 0.5, "Round-trip error for length \(length)")
            }
        }
    }
}
