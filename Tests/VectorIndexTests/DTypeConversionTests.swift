import XCTest
@testable import VectorIndex
import CS2RNG

/// Tests for dtype conversion correctness per S2 spec requirements
final class DTypeConversionTests: XCTestCase {

    // MARK: - f32 ↔ f16 Round-trip Tests

    func testF32ToF16RoundTrip() {
        let testVectors: [Float] = [
            0.0, -0.0,                    // Signed zeros
            1.0, -1.0,                     // Simple values
            0.333333, -0.333333,           // Rounding cases
            65504.0,                       // Max f16 normal
            6.10352e-5,                    // Min f16 normal
            5.96046e-8,                    // f16 subnormal
            .infinity, -.infinity,         // Infinities
            .nan,                          // NaN (payload test separate)
            123.456, -789.012             // Typical values
        ]

        var f16Buffer = [UInt16](repeating: 0, count: testVectors.count)
        var roundTrip = [Float](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            f32_to_f16(src.baseAddress!, &f16Buffer, Int32(src.count), NearestTiesToEven)
        }

        f16Buffer.withUnsafeBufferPointer { src in
            f16_to_f32(src.baseAddress!, &roundTrip, Int32(src.count))
        }

        for (i, original) in testVectors.enumerated() {
            let recovered = roundTrip[i]

            if original.isNaN {
                XCTAssertTrue(recovered.isNaN, "NaN should remain NaN")
            } else if original.isInfinite {
                XCTAssertEqual(recovered, original, "Infinity should round-trip exactly")
            } else if original == 0.0 {
                // Check signed zero preservation
                XCTAssertEqual(recovered.sign, original.sign,
                              "Sign of zero must be preserved")
            } else {
                // Normal values: allow 1 ULP difference due to f16 precision loss
                let relativeError = abs(recovered - original) / abs(original)
                XCTAssertLessThan(relativeError, 0.001,
                                 "Round-trip error too large for \(original) → \(recovered)")
            }
        }
    }

    func testF16NaNPayloadPreservation() {
        // Create NaN with specific payload: 0x7E42 (sign=0, exp=all 1s, mantissa=0x042)
        let nanPayload: UInt16 = 0x7E42
        var recovered: Float = 0.0

        withUnsafePointer(to: nanPayload) { ptr in
            f16_to_f32(ptr, &recovered, 1)
        }

        XCTAssertTrue(recovered.isNaN, "NaN should remain NaN")

        // Extract payload from float
        let recoveredBits = recovered.bitPattern
        let recoveredMantissa = recoveredBits & 0x007FFFFF
        XCTAssertNotEqual(recoveredMantissa, 0,
                         "NaN payload should be preserved (non-zero mantissa)")
    }

    func testF32ToF16Saturation() {
        // Values that should saturate to ±Inf in f16
        let overflow: [Float] = [70000.0, -70000.0, Float(1e10), -Float(1e10)]
        var f16Buffer = [UInt16](repeating: 0, count: overflow.count)

        overflow.withUnsafeBufferPointer { src in
            f32_to_f16(src.baseAddress!, &f16Buffer, Int32(src.count), NearestTiesToEven)
        }

        for (i, f16Bits) in f16Buffer.enumerated() {
            let exponent = (f16Bits >> 10) & 0x1F
            XCTAssertEqual(exponent, 0x1F,
                          "Overflow value \(overflow[i]) should saturate to f16 Inf")
        }
    }

    // MARK: - f32 ↔ bf16 Tests

    func testF32ToBF16RoundTrip() {
        let testVectors: [Float] = [
            0.0, -0.0,
            1.0, -1.0,
            3.14159, -2.71828,
            .infinity, -.infinity,
            .nan
        ]

        var bf16Buffer = [UInt16](repeating: 0, count: testVectors.count)
        var roundTrip = [Float](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            f32_to_bf16(src.baseAddress!, &bf16Buffer, Int32(src.count), NearestTiesToEven)
        }

        bf16Buffer.withUnsafeBufferPointer { src in
            bf16_to_f32(src.baseAddress!, &roundTrip, Int32(src.count))
        }

        for (i, original) in testVectors.enumerated() {
            let recovered = roundTrip[i]

            if original.isNaN {
                XCTAssertTrue(recovered.isNaN)
            } else if original.isInfinite {
                XCTAssertEqual(recovered, original)
            } else if original == 0.0 {
                XCTAssertEqual(recovered.sign, original.sign,
                              "BF16: Sign of zero must be preserved")
            } else {
                // BF16 has same exponent range as f32, 7-bit mantissa (vs 23-bit)
                let relativeError = abs(recovered - original) / abs(original)
                XCTAssertLessThan(relativeError, 0.01,  // ~1% due to mantissa loss
                                 "BF16 round-trip error too large")
            }
        }
    }

    // MARK: - int8 Quantization Tests

    func testSymmetricQuantizationRoundTrip() {
        let testVectors: [Float] = [-127.0, -1.0, 0.0, 1.0, 63.5, 127.0]
        let scale: Float = 1.0  // scale=1 → direct mapping

        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)
        var roundTrip = [Float](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        i8Buffer.withUnsafeBufferPointer { src in
            dequantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &roundTrip)
        }

        for (i, original) in testVectors.enumerated() {
            let diff = abs(roundTrip[i] - original)
            XCTAssertLessThanOrEqual(diff, 0.5,
                                    "Quantization error should be ≤0.5 for scale=1.0")
        }
    }

    func testSymmetricQuantizationSaturation() {
        // Test that values outside [-127*scale, 127*scale] saturate correctly
        let scale: Float = 1.0
        let testVectors: [Float] = [-200.0, -128.0, 127.0, 200.0]
        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_symmetric(src.baseAddress!, Int32(src.count), scale, &i8Buffer)
        }

        XCTAssertEqual(i8Buffer[0], -128, "Should saturate to min")
        XCTAssertEqual(i8Buffer[1], -128, "Should saturate to min")
        XCTAssertEqual(i8Buffer[2], 127, "Max value should fit")
        XCTAssertEqual(i8Buffer[3], 127, "Should saturate to max")
    }

    func testAffineQuantizationWithZeroPoint() {
        // Test asymmetric quantization: uint8 range [0,255] mapped via zero_point
        let scale: Float = 2.0
        let zeroPoint: Int32 = -128  // Maps [0, 510] to [-128, 127]

        let testVectors: [Float] = [0.0, 127.0, 254.0, 510.0]
        var i8Buffer = [Int8](repeating: 0, count: testVectors.count)
        var roundTrip = [Float](repeating: 0, count: testVectors.count)

        testVectors.withUnsafeBufferPointer { src in
            quantize_i8_affine(src.baseAddress!, Int32(src.count), scale, zeroPoint, &i8Buffer)
        }

        i8Buffer.withUnsafeBufferPointer { src in
            dequantize_i8_affine(src.baseAddress!, Int32(src.count), scale, zeroPoint, &roundTrip)
        }

        for (i, original) in testVectors.enumerated() {
            let diff = abs(roundTrip[i] - original)
            XCTAssertLessThanOrEqual(diff, scale,
                                    "Affine quantization error should be ≤ scale")
        }
    }

    // MARK: - 4-bit PQ Pack/Unpack Tests

    func testNibblePackUnpackRoundTrip() {
        let testVectors: [UInt8] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        let n = testVectors.count

        var packed = [UInt8](repeating: 0, count: (n + 1) / 2)
        var unpacked = [UInt8](repeating: 0, count: n)

        testVectors.withUnsafeBufferPointer { src in
            pack_nibbles_u4(src.baseAddress!, Int32(n), &packed)
        }

        packed.withUnsafeBufferPointer { src in
            unpack_nibbles_u4(src.baseAddress!, Int32(n), &unpacked)
        }

        XCTAssertEqual(unpacked, testVectors,
                      "4-bit pack/unpack should round-trip exactly")
    }

    func testNibblePackingOrderLowFirst() {
        // Verify "low nibble is first code" spec requirement
        let input: [UInt8] = [0x3, 0xA]  // 3 and 10
        var packed = [UInt8](repeating: 0, count: 1)

        input.withUnsafeBufferPointer { src in
            pack_nibbles_u4(src.baseAddress!, 2, &packed)
        }

        XCTAssertEqual(packed[0], 0xA3,
                      "Low nibble=0x3 (first), high nibble=0xA (second)")
    }

    func testNibblePackOddLength() {
        // Test odd-length packing (last nibble in low position)
        let input: [UInt8] = [0x5, 0x9, 0x2]  // 3 nibbles
        var packed = [UInt8](repeating: 0, count: 2)
        var unpacked = [UInt8](repeating: 0, count: 3)

        input.withUnsafeBufferPointer { src in
            pack_nibbles_u4(src.baseAddress!, 3, &packed)
        }

        packed.withUnsafeBufferPointer { src in
            unpack_nibbles_u4(src.baseAddress!, 3, &unpacked)
        }

        XCTAssertEqual(unpacked, input,
                      "Odd-length pack/unpack should round-trip")
    }

    // MARK: - Endianness Tests

    func testLittleEndianRoundTrip() {
        var buffer = [UInt8](repeating: 0, count: 16)

        buffer.withUnsafeMutableBytes { ptr in
            let base = ptr.baseAddress!

            store_le16(base.advanced(by: 0), 0x1234)
            store_le32(base.advanced(by: 2), 0x56789ABC)
            store_le64(base.advanced(by: 6), 0xFEDCBA9876543210)

            let v16 = le16(base.advanced(by: 0))
            let v32 = le32(base.advanced(by: 2))
            let v64 = le64(base.advanced(by: 6))

            XCTAssertEqual(v16, 0x1234)
            XCTAssertEqual(v32, 0x56789ABC)
            XCTAssertEqual(v64, 0xFEDCBA9876543210)
        }
    }

    // MARK: - SIMD Alignment Tests

    func testAlignedNEONPath() {
        // Test that 16-element batches work correctly (NEON fast path)
        let count = 64  // Multiple of 16 for NEON
        let testVectors = (0..<count).map { Float($0) * 0.1 }

        var f16Buffer = [UInt16](repeating: 0, count: count)
        var roundTrip = [Float](repeating: 0, count: count)

        testVectors.withUnsafeBufferPointer { src in
            f32_to_f16(src.baseAddress!, &f16Buffer, Int32(count), NearestTiesToEven)
        }

        f16Buffer.withUnsafeBufferPointer { src in
            f16_to_f32(src.baseAddress!, &roundTrip, Int32(count))
        }

        for (i, original) in testVectors.enumerated() {
            let relativeError = abs(roundTrip[i] - original) / max(abs(original), 1e-6)
            XCTAssertLessThan(relativeError, 0.01,
                             "NEON path error at index \(i)")
        }
    }

    func testUnalignedScalarPath() {
        // Test odd-length array (forces scalar tail)
        let count = 17  // Not divisible by 16
        let testVectors = (0..<count).map { Float($0) * 0.1 }

        var f16Buffer = [UInt16](repeating: 0, count: count)
        var roundTrip = [Float](repeating: 0, count: count)

        testVectors.withUnsafeBufferPointer { src in
            f32_to_f16(src.baseAddress!, &f16Buffer, Int32(count), NearestTiesToEven)
        }

        f16Buffer.withUnsafeBufferPointer { src in
            f16_to_f32(src.baseAddress!, &roundTrip, Int32(count))
        }

        for (i, original) in testVectors.enumerated() {
            let relativeError = abs(roundTrip[i] - original) / max(abs(original), 1e-6)
            XCTAssertLessThan(relativeError, 0.01,
                             "Scalar tail error at index \(i)")
        }
    }
}
