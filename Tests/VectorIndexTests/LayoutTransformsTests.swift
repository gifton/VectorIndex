import XCTest
@testable import VectorIndex

final class LayoutTransformsTests: XCTestCase {

    func testRoundTrip_AoS2D_R4_R8_Tails() {
        let ns = [1, 5, 16]
        let ds = [15, 16, 17, 31, 32, 33]
        let Rs: [RowBlockSize] = [.r4, .r8]

        for n in ns {
            for d in ds {
                // Build simple deterministic AoS rows
                var rows: [[Float]] = []
                rows.reserveCapacity(n)
                for i in 0..<n {
                    var row: [Float] = []
                    row.reserveCapacity(d)
                    for j in 0..<d { row.append(Float(i * d + j)) }
                    rows.append(row)
                }
                for R in Rs {
                    let inter = LayoutTransform.interleave(vectors: rows, rowBlockSize: R)
                    let back = LayoutTransform.deinterleave(interleaved: inter, n: n, d: d, rowBlockSize: R)
                    XCTAssertEqual(back.count, n)
                    for i in 0..<n { XCTAssertEqual(back[i].count, d) }
                    // Compare
                    for i in 0..<n {
                        for j in 0..<d {
                            XCTAssertEqual(back[i][j], rows[i][j], accuracy: 1e-6)
                        }
                    }
                }
            }
        }
    }

    func testRoundTrip_Flat_AoS_to_AoSoA_opts_parallel_and_serial() {
        // Force small threshold to exercise parallel path
        let optsPar = LayoutTransformOpts(rowBlockSize: .r8, pqGroupSize: .g8, enableParallel: true, parallelThreshold: 1, enableTelemetry: false)
        let optsSer = LayoutTransformOpts(rowBlockSize: .r4, pqGroupSize: .g4, enableParallel: false, parallelThreshold: 10_000, enableTelemetry: false)

        let n = 13, d = 17
        var aos: [Float] = []
        aos.reserveCapacity(n*d)
        for i in 0..<(n*d) { aos.append(Float(i % 97) * 0.5) }

        // Serial
        let interSer = LayoutTransform.interleave(aosFlat: aos, n: n, d: d, opts: optsSer)
        let backSer = LayoutTransform.deinterleaveFlat(interleaved: interSer, n: n, d: d, opts: optsSer)
        XCTAssertEqual(backSer.count, aos.count)
        for i in 0..<aos.count { XCTAssertEqual(backSer[i], aos[i], accuracy: 1e-6) }

        // Parallel
        let interPar = LayoutTransform.interleave(aosFlat: aos, n: n, d: d, opts: optsPar)
        let backPar = LayoutTransform.deinterleaveFlat(interleaved: interPar, n: n, d: d, opts: optsPar)
        XCTAssertEqual(backPar.count, aos.count)
        for i in 0..<aos.count { XCTAssertEqual(backPar[i], aos[i], accuracy: 1e-6) }
    }

    func testPQ_u8_group_interleave_identity() {
        let n = 11
        let mChoices = [8, 12, 16]
        let gChoices = [4, 8]
        for m in mChoices {
            for g in gChoices where m % g == 0 {
                // Build AoS codes: [n][m]
                var aos = [UInt8](repeating: 0, count: n * m)
                for i in 0..<n {
                    for j in 0..<m {
                        aos[i*m + j] = UInt8((i * 31 + j * 17) & 0xFF)
                    }
                }
                var inter = [UInt8](repeating: 0, count: n * m)
                aos.withUnsafeBufferPointer { src in
                    inter.withUnsafeMutableBufferPointer { dst in
                        XCTAssertNoThrow(try pqCodesInterleave_u8(aos: src.baseAddress!, n: n, m: m, g: g, out: dst.baseAddress!))
                    }
                }
                var back = [UInt8](repeating: 0, count: n * m)
                inter.withUnsafeBufferPointer { src in
                    back.withUnsafeMutableBufferPointer { dst in
                        XCTAssertNoThrow(try pqCodesDeinterleave_u8(interleaved: src.baseAddress!, n: n, m: m, g: g, aos: dst.baseAddress!))
                    }
                }
                XCTAssertEqual(back, aos)
            }
        }
    }

    func testPQ_u4_group_interleave_identity() {
        let n = 9
        let mChoices = [8, 12, 16] // even
        let gChoices = [4, 8]      // even
        for m in mChoices {
            for g in gChoices where (m % g == 0) {
                // Build packed AoS: n * (m/2) bytes
                let bytesPerVec = m / 2
                var aosPacked = [UInt8](repeating: 0, count: n * bytesPerVec)
                for i in 0..<n {
                    for b in 0..<bytesPerVec {
                        // Two 4-bit codes per byte; keep in 0..15
                        let lo: UInt8 = UInt8(((i * 13 + b * 7) & 0x0F))
                        let hi: UInt8 = UInt8(((i * 29 + b * 11) & 0x0F))
                        aosPacked[i*bytesPerVec + b] = lo | (hi << 4)
                    }
                }
                var interPacked = [UInt8](repeating: 0, count: n * bytesPerVec)
                aosPacked.withUnsafeBufferPointer { src in
                    interPacked.withUnsafeMutableBufferPointer { dst in
                        XCTAssertNoThrow(try pqCodesInterleave_u4(aos_packed: src.baseAddress!, n: n, m: m, g: g, out_packed: dst.baseAddress!))
                    }
                }
                var backPacked = [UInt8](repeating: 0, count: n * bytesPerVec)
                interPacked.withUnsafeBufferPointer { src in
                    backPacked.withUnsafeMutableBufferPointer { dst in
                        XCTAssertNoThrow(try pqCodesDeinterleave_u4(interleaved_packed: src.baseAddress!, n: n, m: m, g: g, aos_packed: dst.baseAddress!))
                    }
                }
                XCTAssertEqual(backPacked, aosPacked)
            }
        }
    }
}
