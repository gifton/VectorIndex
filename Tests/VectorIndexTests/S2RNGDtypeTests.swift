//
//  S2RNGDtypeTests.swift
//  VectorIndexTests
//
//  Comprehensive tests for Kernel #S2: RNG & Dtype Helpers
//
//  Coverage:
//  - RNG reproducibility and uniformity
//  - Dtype conversions (f16, bf16, int8, u4)
//  - Edge cases (NaN, Inf, ±0.0)
//  - Round-trip accuracy
//

import XCTest
@testable import VectorIndex

final class S2RNGDtypeTests: XCTestCase {

    // MARK: - RNG: Xoroshiro128** Tests

    func testXoroshiro128Reproducibility() throws {
        // Same seed should produce same sequence
        let seed: UInt64 = 42
        let streamID: UInt64 = 0

        var rng1 = S2Xoroshiro128(seed: seed, streamID: streamID, taskID: 0)
        var rng2 = S2Xoroshiro128(seed: seed, streamID: streamID, taskID: 0)

        for _ in 0..<100 {
            let v1 = rng1.nextU64()
            let v2 = rng2.nextU64()
            XCTAssertEqual(v1, v2, "Same seed must produce identical sequences")
        }
    }

    func testXoroshiro128StreamIndependence() throws {
        // Different streams should produce different sequences
        let seed: UInt64 = 12345

        var rng1 = S2Xoroshiro128(seed: seed, streamID: 0, taskID: 0)
        var rng2 = S2Xoroshiro128(seed: seed, streamID: 1, taskID: 0)

        var different = false
        for _ in 0..<100 {
            if rng1.nextU64() != rng2.nextU64() {
                different = true
                break
            }
        }

        XCTAssertTrue(different, "Different streams should produce different sequences")
    }

    func testXoroshiro128UniformDistribution() throws {
        // Chi-square test for uniformity in [0, 1)
        var rng = S2Xoroshiro128(seed: 9876, streamID: 0, taskID: 0)

        let numBins = 10
        let numSamples = 10_000
        var bins = [Int](repeating: 0, count: numBins)

        for _ in 0..<numSamples {
            let u = rng.nextUniform()
            XCTAssertGreaterThanOrEqual(u, 0.0)
            XCTAssertLessThan(u, 1.0)

            let bin = min(Int(u * Float(numBins)), numBins - 1)
            bins[bin] += 1
        }

        // Check each bin has roughly numSamples / numBins
        let expected = Float(numSamples) / Float(numBins)
        let tolerance = expected * 0.2  // ±20% tolerance

        for count in bins {
            XCTAssertGreaterThan(Float(count), expected - tolerance)
            XCTAssertLessThan(Float(count), expected + tolerance)
        }
    }

    // MARK: - RNG: Philox4x32-10 Tests

    func testPhilox4x32Reproducibility() throws {
        // Same seed and counter should produce same output
        let seed: UInt64 = 54321
        let philox1 = S2Philox4x32(seed: seed, streamID: 0)
        let philox2 = S2Philox4x32(seed: seed, streamID: 0)

        for counter in 0..<100 {
            let out1 = philox1.generate(counterLo: UInt64(counter), counterHi: 0)
            let out2 = philox2.generate(counterLo: UInt64(counter), counterHi: 0)

            XCTAssertEqual(out1, out2, "Same seed and counter must produce identical output")
        }
    }

    func testPhilox4x32DifferentCounters() throws {
        // Different counters should produce different outputs
        let philox = S2Philox4x32(seed: 11111, streamID: 0)

        let out1 = philox.generate(counterLo: 0, counterHi: 0)
        let out2 = philox.generate(counterLo: 1, counterHi: 0)

        XCTAssertNotEqual(out1, out2, "Different counters should produce different outputs")
    }

    // MARK: - RNG Utilities Tests

    func testWeightedPick() throws {
        var rng = S2Xoroshiro128(seed: 999, streamID: 0, taskID: 0)

        // Test with simple weights
        let weights: [Float] = [1.0, 2.0, 3.0, 4.0]  // Should favor index 3

        var counts = [Int](repeating: 0, count: weights.count)
        let numTrials = 10_000

        for _ in 0..<numTrials {
            let idx = weightedPick(weights: weights, rng: &rng)
            XCTAssertGreaterThanOrEqual(idx, 0)
            XCTAssertLessThan(idx, weights.count)
            counts[idx] += 1
        }

        // Check distribution roughly matches weights
        // Expected proportions: 0.1, 0.2, 0.3, 0.4
        let total = Float(numTrials)
        XCTAssertGreaterThan(Float(counts[0]), total * 0.05)   // ~10%
        XCTAssertLessThan(Float(counts[0]), total * 0.15)
        XCTAssertGreaterThan(Float(counts[3]), total * 0.35)   // ~40%
        XCTAssertLessThan(Float(counts[3]), total * 0.45)
    }

    func testSampleWithoutReplacement() throws {
        var rng = S2Xoroshiro128(seed: 777, streamID: 0, taskID: 0)

        let n = 100
        let k = 20

        let sample = sampleWithoutReplacement(n: n, k: k, rng: &rng)

        // Check sample size
        XCTAssertEqual(sample.count, k)

        // Check uniqueness
        let uniqueCount = Set(sample).count
        XCTAssertEqual(uniqueCount, k, "Sample must contain unique indices")

        // Check range
        for idx in sample {
            XCTAssertLessThan(idx, UInt32(n))
        }
    }

    func testRandpermInPlace() throws {
        var rng = S2Xoroshiro128(seed: 555, streamID: 0, taskID: 0)

        var array: [UInt32] = Array(0..<100)
        let original = array

        randpermInPlace(&array, rng: &rng)

        // Check same elements, different order
        XCTAssertEqual(Set(array), Set(original), "Permutation must contain same elements")
        XCTAssertNotEqual(array, original, "Permutation should change order")
    }

    func testGaussianBoxMuller() throws {
        var rng = S2Xoroshiro128(seed: 333, streamID: 0, taskID: 0)

        let samples = gaussianBoxMuller(count: 10_000, rng: &rng)

        // Compute mean and variance
        let mean = samples.reduce(0.0, +) / Float(samples.count)
        let variance = samples.map { ($0 - mean) * ($0 - mean) }.reduce(0.0, +) / Float(samples.count)

        // Should be approximately N(0, 1)
        XCTAssertLessThan(abs(mean), 0.05, "Mean should be close to 0")
        XCTAssertGreaterThan(variance, 0.9, "Variance should be close to 1")
        XCTAssertLessThan(variance, 1.1)
    }

    // MARK: - Dtype: f16 Tests

    func testF16RoundTrip() throws {
        let testValues: [Float] = [
            0.0, -0.0, 1.0, -1.0,
            0.5, -0.5, 2.5, -2.5,
            100.0, -100.0,
            Float.pi, -Float.pi,
            0.001, -0.001,  // f16 min normal is ~6e-5, use larger values
            65504.0  // Max f16 value
        ]

        var f16bits = [UInt16](repeating: 0, count: testValues.count)
        var roundTrip = [Float](repeating: 0, count: testValues.count)

        // Convert f32 → f16 → f32
        testValues.withUnsafeBufferPointer { srcPtr in
            f16bits.withUnsafeMutableBufferPointer { dstPtr in
                f32ToF16Batch(srcPtr.baseAddress!, dstPtr.baseAddress!, testValues.count)
            }
        }

        f16bits.withUnsafeBufferPointer { srcPtr in
            roundTrip.withUnsafeMutableBufferPointer { dstPtr in
                f16ToF32Batch(srcPtr.baseAddress!, dstPtr.baseAddress!, testValues.count)
            }
        }

        // Check round-trip accuracy
        for i in 0..<testValues.count {
            let original = testValues[i]
            let recovered = roundTrip[i]

            if original.isNaN {
                XCTAssertTrue(recovered.isNaN)
            } else if original.isInfinite {
                XCTAssertEqual(recovered, original)
            } else {
                // Allow small relative error due to precision loss
                let relativeError = abs(recovered - original) / max(abs(original), 1e-6)
                XCTAssertLessThan(relativeError, 0.001, "f16 round-trip error for \(original)")
            }
        }
    }

    func testF16NaNPreservation() throws {
        let nanValue: Float = Float.nan
        var f16bit: UInt16 = 0
        var recovered: Float = 0

        withUnsafePointer(to: nanValue) { srcPtr in
            f32ToF16Batch(srcPtr, &f16bit, 1)
        }

        withUnsafePointer(to: f16bit) { srcPtr in
            f16ToF32Batch(srcPtr, &recovered, 1)
        }

        XCTAssertTrue(recovered.isNaN, "NaN should be preserved through f16 conversion")
    }

    func testF16SignOfZero() throws {
        let positiveZero: Float = 0.0
        let negativeZero: Float = -0.0

        var f16_pos: UInt16 = 0
        var f16_neg: UInt16 = 0
        var recovered_pos: Float = 0
        var recovered_neg: Float = 0

        // Convert +0.0
        withUnsafePointer(to: positiveZero) { srcPtr in
            f32ToF16Batch(srcPtr, &f16_pos, 1)
        }

        // Convert -0.0
        withUnsafePointer(to: negativeZero) { srcPtr in
            f32ToF16Batch(srcPtr, &f16_neg, 1)
        }

        // Recover
        withUnsafePointer(to: f16_pos) { srcPtr in
            f16ToF32Batch(srcPtr, &recovered_pos, 1)
        }

        withUnsafePointer(to: f16_neg) { srcPtr in
            f16ToF32Batch(srcPtr, &recovered_neg, 1)
        }

        // Check sign bit
        XCTAssertEqual(recovered_pos.sign, .plus)
        XCTAssertEqual(recovered_neg.sign, .minus)
    }

    // MARK: - Dtype: bf16 Tests (CRITICAL - Tests the bug fix!)

    func testBF16RoundTrip() throws {
        let testValues: [Float] = [
            0.0, -0.0, 1.0, -1.0,
            0.5, -0.5, 2.5, -2.5,
            100.0, -100.0,
            Float.pi, -Float.pi,
            0.00001, -0.00001,
            3.38e38  // Near max bf16 value (~3.4e38)
        ]

        var bf16bits = [UInt16](repeating: 0, count: testValues.count)
        var roundTrip = [Float](repeating: 0, count: testValues.count)

        // Convert f32 → bf16 → f32
        testValues.withUnsafeBufferPointer { srcPtr in
            bf16bits.withUnsafeMutableBufferPointer { dstPtr in
                f32ToBF16Batch(srcPtr.baseAddress!, dstPtr.baseAddress!, testValues.count)
            }
        }

        bf16bits.withUnsafeBufferPointer { srcPtr in
            roundTrip.withUnsafeMutableBufferPointer { dstPtr in
                bf16ToF32Batch(srcPtr.baseAddress!, dstPtr.baseAddress!, testValues.count)
            }
        }

        // Check round-trip accuracy
        for i in 0..<testValues.count {
            let original = testValues[i]
            let recovered = roundTrip[i]

            if original.isNaN {
                XCTAssertTrue(recovered.isNaN)
            } else if original.isInfinite {
                XCTAssertEqual(recovered, original)
            } else {
                // bf16 has less precision than f16 (7-bit mantissa vs 10-bit)
                // Allow larger relative error
                let relativeError = abs(recovered - original) / max(abs(original), 1e-6)
                XCTAssertLessThan(relativeError, 0.01, "bf16 round-trip error for \(original)")
            }
        }
    }

    func testBF16NaNPreservation() throws {
        let nanValue: Float = Float.nan
        var bf16bit: UInt16 = 0
        var recovered: Float = 0

        withUnsafePointer(to: nanValue) { srcPtr in
            f32ToBF16Batch(srcPtr, &bf16bit, 1)
        }

        withUnsafePointer(to: bf16bit) { srcPtr in
            bf16ToF32Batch(srcPtr, &recovered, 1)
        }

        XCTAssertTrue(recovered.isNaN, "NaN should be preserved through bf16 conversion")
    }

    func testBF16SignOfZero() throws {
        let positiveZero: Float = 0.0
        let negativeZero: Float = -0.0

        var bf16_pos: UInt16 = 0
        var bf16_neg: UInt16 = 0
        var recovered_pos: Float = 0
        var recovered_neg: Float = 0

        // Convert +0.0
        withUnsafePointer(to: positiveZero) { srcPtr in
            f32ToBF16Batch(srcPtr, &bf16_pos, 1)
        }

        // Convert -0.0
        withUnsafePointer(to: negativeZero) { srcPtr in
            f32ToBF16Batch(srcPtr, &bf16_neg, 1)
        }

        // Recover
        withUnsafePointer(to: bf16_pos) { srcPtr in
            bf16ToF32Batch(srcPtr, &recovered_pos, 1)
        }

        withUnsafePointer(to: bf16_neg) { srcPtr in
            bf16ToF32Batch(srcPtr, &recovered_neg, 1)
        }

        // Check sign bit
        XCTAssertEqual(recovered_pos.sign, .plus)
        XCTAssertEqual(recovered_neg.sign, .minus)
    }

    func testBF16RoundingTiesToEven() throws {
        // Test tie-breaking behavior (round to nearest even)
        // This is the CRITICAL test for the bug fix!

        // Value: 1.0 + 2^-8 + 2^-9 (exactly halfway between two bf16 values)
        // In binary: 0x3F800000 + 0x00008000 = 0x3F808000
        // bf16 candidates: 0x3F80 (even) vs 0x3F81 (odd)
        // Should round to 0x3F80 (ties to even)

        let testValue = Float(bitPattern: 0x3F808000)
        var bf16bit: UInt16 = 0

        withUnsafePointer(to: testValue) { srcPtr in
            f32ToBF16Batch(srcPtr, &bf16bit, 1)
        }

        // Should round down to even (0x3F80)
        XCTAssertEqual(bf16bit, 0x3F80, "Tie should round to even")

        // Test with odd LSB
        let testValue2 = Float(bitPattern: 0x3F818000)
        var bf16bit2: UInt16 = 0

        withUnsafePointer(to: testValue2) { srcPtr in
            f32ToBF16Batch(srcPtr, &bf16bit2, 1)
        }

        // Should round up to even (0x3F82)
        XCTAssertEqual(bf16bit2, 0x3F82, "Tie should round to even (up)")
    }

    // MARK: - Dtype: int8 Quantization Tests

    func testInt8QuantizationSymmetric() throws {
        let values: [Float] = [-127.0, -63.5, 0.0, 63.5, 127.0, 200.0, -200.0]
        let scale: Float = 1.0  // 1:1 mapping for simplicity

        var quantized = [Int8](repeating: 0, count: values.count)
        var dequantized = [Float](repeating: 0, count: values.count)

        // Quantize
        values.withUnsafeBufferPointer { srcPtr in
            quantized.withUnsafeMutableBufferPointer { dstPtr in
                quantizeSymmetric(x: srcPtr, scale: scale, y: dstPtr)
            }
        }

        // Check clamping
        XCTAssertEqual(quantized[5], 127, "Should clamp to 127")
        XCTAssertEqual(quantized[6], -128, "Should clamp to -128")

        // Dequantize
        quantized.withUnsafeBufferPointer { srcPtr in
            dequantized.withUnsafeMutableBufferPointer { dstPtr in
                dequantizeSymmetric(y: srcPtr, scale: scale, x: dstPtr)
            }
        }

        // Check round-trip for values in range
        for i in 0..<5 {
            let error = abs(dequantized[i] - values[i])
            XCTAssertLessThan(error, 1.0, "Dequantization error")
        }
    }

    func testInt8QuantizationAffine() throws {
        let values: [Float] = [0.0, 50.0, 100.0, 150.0, 255.0]
        let scale: Float = 1.0
        let zeroPoint: Int32 = -128  // Map [0, 255] to [-128, 127]

        var quantized = [Int8](repeating: 0, count: values.count)
        var dequantized = [Float](repeating: 0, count: values.count)

        // Quantize
        values.withUnsafeBufferPointer { srcPtr in
            quantized.withUnsafeMutableBufferPointer { dstPtr in
                quantizeAffine(x: srcPtr, scale: scale, zeroPoint: zeroPoint, y: dstPtr)
            }
        }

        // Check mapping
        XCTAssertEqual(quantized[0], -128, "0.0 should map to -128")
        XCTAssertEqual(quantized[4], 127, "255.0 should clamp to 127")

        // Dequantize
        quantized.withUnsafeBufferPointer { srcPtr in
            dequantized.withUnsafeMutableBufferPointer { dstPtr in
                dequantizeAffine(y: srcPtr, scale: scale, zeroPoint: zeroPoint, x: dstPtr)
            }
        }

        // Check round-trip
        let error0 = abs(dequantized[0] - values[0])
        XCTAssertLessThan(error0, 1.0)
    }

    // MARK: - Dtype: u4 Pack/Unpack Tests

    func testU4PackUnpack() throws {
        let indices: [UInt8] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        var packed = [UInt8](repeating: 0, count: indices.count / 2)
        var unpacked = [UInt8](repeating: 0, count: indices.count)

        // Pack
        indices.withUnsafeBufferPointer { srcPtr in
            packed.withUnsafeMutableBufferPointer { dstPtr in
                packNibblesU4(indices: srcPtr, packed: dstPtr)
            }
        }

        // Check packing
        XCTAssertEqual(packed[0], packPair(0, 1))
        XCTAssertEqual(packed[1], packPair(2, 3))

        // Unpack
        packed.withUnsafeBufferPointer { srcPtr in
            unpacked.withUnsafeMutableBufferPointer { dstPtr in
                unpackNibblesU4(packed: srcPtr, indices: dstPtr)
            }
        }

        // Check round-trip
        XCTAssertEqual(unpacked, indices, "Pack/unpack should be identity")
    }

    // MARK: - Endian Helpers Tests

    func testEndianHelpers() throws {
        // Use aligned buffer for proper memory access
        var buffer = ContiguousArray<UInt64>(repeating: 0, count: 1)

        buffer.withUnsafeMutableBytes { ptr in
            // Store 16-bit value at offset 0 (aligned)
            store16LE(ptr.baseAddress!, 0x1234)
            // Store 32-bit value at offset 4 (aligned)
            store32LE(ptr.baseAddress!.advanced(by: 4), 0x12345678)
        }

        // Load values
        let loaded16 = buffer.withUnsafeBytes { ptr in
            load16LE(ptr.baseAddress!)
        }

        let loaded32 = buffer.withUnsafeBytes { ptr in
            load32LE(ptr.baseAddress!.advanced(by: 4))
        }

        XCTAssertEqual(loaded16, 0x1234)
        XCTAssertEqual(loaded32, 0x12345678)
    }

    // MARK: - Alignment Helpers Tests

    func testAlignUp() throws {
        XCTAssertEqual(alignUp(0, to: 64), 0)
        XCTAssertEqual(alignUp(1, to: 64), 64)
        XCTAssertEqual(alignUp(64, to: 64), 64)
        XCTAssertEqual(alignUp(65, to: 64), 128)
        XCTAssertEqual(alignUp(100, to: 16), 112)
    }

    func testPadTo() throws {
        XCTAssertEqual(padTo(0, multiple: 16), 0)
        XCTAssertEqual(padTo(1, multiple: 16), 16)
        XCTAssertEqual(padTo(16, multiple: 16), 16)
        XCTAssertEqual(padTo(17, multiple: 16), 32)
    }

    // MARK: - Ported from CS2RNG-dependent tests (deleted with the CS2RNG C target;
    // see Task 2 of the phase-2 cleanup plan for the full per-test triage).

    func testXoroshiro128UniformityChiSquare() throws {
        // Ported from RNGDeterminismTests.testUniformityChiSquare.
        // Chi-square goodness-of-fit test: chi2 = sum((observed-expected)^2 / expected).
        var rng = S2Xoroshiro128(seed: 0xDEADBEEF, streamID: 0, taskID: 0)

        let bins = 100
        var counts = [Int](repeating: 0, count: bins)
        let samples = 100_000

        for _ in 0..<samples {
            let u = rng.nextUniform()
            let bin = min(Int(u * Float(bins)), bins - 1)
            counts[bin] += 1
        }

        let expected = Double(samples) / Double(bins)
        let chiSquare = counts.reduce(0.0) { sum, count in
            let diff = Double(count) - expected
            return sum + (diff * diff) / expected
        }

        // Critical value for 99 degrees of freedom at alpha=0.001 is ~149.
        // Verified observed value on this seed: ~99.06.
        XCTAssertLessThan(chiSquare, 149.0,
                           "Chi-square test failed: S2Xoroshiro128.nextUniform() is not uniform")
    }

    func testXoroshiro128StreamIndependenceQuantitative() throws {
        // Ported from RNGDeterminismTests.testStreamIndependence. Strengthens
        // testXoroshiro128StreamIndependence's boolean check with a quantitative bound.
        let seed: UInt64 = 42

        var rng0 = S2Xoroshiro128(seed: seed, streamID: 0, taskID: 0)
        var rng1 = S2Xoroshiro128(seed: seed, streamID: 1, taskID: 0)

        let seq0 = (0..<1000).map { _ in rng0.nextUniform() }
        let seq1 = (0..<1000).map { _ in rng1.nextUniform() }

        // Verified observed value on this seed: 0 collisions.
        let collisions = zip(seq0, seq1).filter { $0 == $1 }.count
        XCTAssertLessThan(collisions, 10,
                           "Independent streams should have <1% collision rate")
    }

    func testInt8QuantizationExtremeSaturation() throws {
        // Ported from S2EdgeCaseTests.testQuantizationWithLargeValues +
        // testQuantizationNearMaxFloat. Values are chosen to stay within Int32 range
        // after division by scale, so quantizeSymmetric's internal `Int32(r)` conversion
        // does not trap (see this task's finding #1 for the Inf/NaN case that does trap
        // and was deliberately NOT ported).
        let largeValues: [Float] = [
            16_777_216.0,   // 2^24
            33_554_432.0,   // 2^25
            1.0e10,
            -1.0e10
        ]
        var largeQuantized = [Int8](repeating: 0, count: largeValues.count)
        largeValues.withUnsafeBufferPointer { src in
            largeQuantized.withUnsafeMutableBufferPointer { dst in
                quantizeSymmetric(x: src, scale: 127.0, y: dst)
            }
        }
        // Verified observed output: [127, 127, 127, -128]
        XCTAssertEqual(largeQuantized, [127, 127, 127, -128],
                       "Values far beyond scale*127 must saturate to int8 bounds")

        let maxVal = Float.greatestFiniteMagnitude
        let nearMaxValues: [Float] = [maxVal, -maxVal, maxVal * 0.5, -maxVal * 0.5]
        let nearMaxScale = maxVal / 127.0
        var nearMaxQuantized = [Int8](repeating: 0, count: nearMaxValues.count)
        nearMaxValues.withUnsafeBufferPointer { src in
            nearMaxQuantized.withUnsafeMutableBufferPointer { dst in
                quantizeSymmetric(x: src, scale: nearMaxScale, y: dst)
            }
        }
        // Verified observed output: [127, -127, 63, -63]
        XCTAssertEqual(abs(nearMaxQuantized[0]), 127, "Max float should saturate to +-127")
        XCTAssertEqual(abs(nearMaxQuantized[1]), 127, "-Max float should saturate to +-127")
        XCTAssertLessThanOrEqual(abs(Int(nearMaxQuantized[2]) - 64), 1,
                                  "0.5*max should quantize to ~64")
        XCTAssertLessThanOrEqual(abs(Int(nearMaxQuantized[3]) + 64), 1,
                                  "-0.5*max should quantize to ~-64")
    }

    func testNibblePackingOrderLowFirst() throws {
        // Ported from DTypeConversionTests.testNibblePackingOrderLowFirst. Asserts a
        // hand-computed byte value rather than round-tripping through packPair(), so it
        // independently documents the "low nibble = first code" contract (testU4PackUnpack
        // only checks packed[i] == packPair(...), which is tautological w.r.t. packPair
        // itself).
        let input: [UInt8] = [0x3, 0xA]  // codes 3 and 10
        var packed = [UInt8](repeating: 0, count: 1)

        input.withUnsafeBufferPointer { src in
            packed.withUnsafeMutableBufferPointer { dst in
                packNibblesU4(indices: src, packed: dst)
            }
        }

        XCTAssertEqual(packed[0], 0xA3,
                       "Low nibble=0x3 (first code), high nibble=0xA (second code)")
    }

    func testEndianHelpers64Bit() throws {
        // Ported from DTypeConversionTests.testLittleEndianRoundTrip's 64-bit case.
        // testEndianHelpers only covers the 16/32-bit helpers.
        var buffer = ContiguousArray<UInt64>(repeating: 0, count: 1)

        buffer.withUnsafeMutableBytes { ptr in
            store64LE(ptr.baseAddress!, 0xFEDCBA9876543210)
        }

        let loaded64 = buffer.withUnsafeBytes { ptr in
            load64LE(ptr.baseAddress!)
        }

        XCTAssertEqual(loaded64, 0xFEDCBA9876543210)
    }

    func testF16BoundaryValues() throws {
        // Merged from S2EdgeCaseTests.testF16ConversionWithSpecialValues (overflow/
        // underflow) and DTypeConversionTests.testF32ToF16RoundTrip (min-normal/subnormal
        // boundary values). Verified live: Float16(Float.greatestFiniteMagnitude) saturates
        // to +Inf (no trap) -- narrowing float-to-float conversions saturate per IEEE 754,
        // unlike the float-to-Int32 trap in testInt8QuantizationExtremeSaturation's comment.
        let maxVal = Float.greatestFiniteMagnitude
        let testValues: [Float] = [
            maxVal,          // overflows f16 range -> +Inf
            -maxVal,         // overflows f16 range -> -Inf
            1.0e-20,         // underflows f16 range -> 0
            6.10352e-5,      // smallest f16 normal
            5.96046e-8       // smallest f16 subnormal (2^-24)
        ]

        var f16Bits = [UInt16](repeating: 0, count: testValues.count)
        var roundTrip = [Float](repeating: 0, count: testValues.count)

        testValues.withUnsafeBufferPointer { src in
            f16Bits.withUnsafeMutableBufferPointer { dst in
                f32ToF16Batch(src.baseAddress!, dst.baseAddress!, testValues.count)
            }
        }
        f16Bits.withUnsafeBufferPointer { src in
            roundTrip.withUnsafeMutableBufferPointer { dst in
                f16ToF32Batch(src.baseAddress!, dst.baseAddress!, testValues.count)
            }
        }

        XCTAssertTrue(roundTrip[0].isInfinite && roundTrip[0] > 0, "Max f32 should overflow to +Inf in f16")
        XCTAssertTrue(roundTrip[1].isInfinite && roundTrip[1] < 0, "-Max f32 should overflow to -Inf in f16")
        XCTAssertEqual(roundTrip[2], 0.0, "1e-20 should underflow to zero in f16")

        let minNormalError = abs(roundTrip[3] - testValues[3]) / testValues[3]
        XCTAssertLessThan(minNormalError, 0.001, "Smallest f16 normal should round-trip accurately")

        let subnormalError = abs(roundTrip[4] - testValues[4])
        XCTAssertLessThan(subnormalError, 1e-8, "Smallest f16 subnormal should round-trip accurately")
    }
}
