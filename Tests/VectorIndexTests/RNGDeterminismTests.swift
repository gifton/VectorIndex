import XCTest
@testable import VectorIndex
import CS2RNG

/// Tests for RNG determinism guarantees per S2 spec
final class RNGDeterminismTests: XCTestCase {

    /// Verify same seed → same sequence across runs
    func testXoroReproducibility() {
        let seed: UInt64 = 0x123456789ABCDEF0

        var rng1 = Xoro128()
        var rng2 = Xoro128()
        xoro128_init(&rng1, seed, 0)
        xoro128_init(&rng2, seed, 0)

        for _ in 0..<10000 {
            XCTAssertEqual(xoro128_next_uniform(&rng1), xoro128_next_uniform(&rng2), accuracy: 0.0,
                          "Same seed must produce identical sequence")
        }
    }

    /// Verify stream splitting produces independent sequences
    func testStreamIndependence() {
        let seed: UInt64 = 42

        var rng0 = Xoro128()
        var rng1 = Xoro128()
        xoro128_init(&rng0, seed, 0)
        xoro128_init(&rng1, seed, 1)

        let seq0 = (0..<1000).map { _ in xoro128_next_uniform(&rng0) }
        let seq1 = (0..<1000).map { _ in xoro128_next_uniform(&rng1) }

        let collisions = zip(seq0, seq1).filter { $0 == $1 }.count
        XCTAssertLessThan(collisions, 10,
                         "Independent streams should have <1% collision rate")
    }

    /// Verify uniformity via Chi-square test
    func testUniformityChiSquare() {
        var rng = Xoro128()
        xoro128_init(&rng, 0xDEADBEEF, 0)

        let bins = 100
        var counts = [Int](repeating: 0, count: bins)
        let samples = 100_000

        for _ in 0..<samples {
            let u = xoro128_next_uniform(&rng)
            let bin = Int(u * Float(bins))
            counts[min(bin, bins - 1)] += 1
        }

        // Chi-square test: χ² = Σ((observed - expected)² / expected)
        let expected = Double(samples) / Double(bins)
        let chiSquare = counts.reduce(0.0) { sum, count in
            let diff = Double(count) - expected
            return sum + (diff * diff) / expected
        }

        // Critical value for 99 df at α=0.001 is ~149
        XCTAssertLessThan(chiSquare, 149.0,
                         "Chi-square test failed: distribution not uniform")
    }

    /// Verify Philox counter-based determinism
    func testPhiloxReproducibility() {
        let seed: UInt64 = 0x0123456789ABCDEF
        let streamID: UInt64 = 0x13579BDF02468ACE

        var key0: UInt64 = 0
        var key1: UInt64 = 0
        philox_key(seed, streamID, &key0, &key1)

        var out1 = [UInt32](repeating: 0, count: 4)
        var out2 = [UInt32](repeating: 0, count: 4)

        philox_next4(key0, key1, 0, 0, &out1)
        philox_next4(key0, key1, 0, 0, &out2)

        XCTAssertEqual(out1, out2,
                      "Philox with same counter must be deterministic")
    }
}
