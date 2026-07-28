import XCTest
@testable import VectorIndex

final class IDFilterTests: XCTestCase {
    // Characterization test pinning idFilterCompactN's current (mask-allocating) behavior
    // BEFORE the B14 refactor, so the refactor can be verified byte-for-byte identical.
    //
    // Fixture: 8 ids [0...7], one allowlist bitset (bit=1 keeps ids {0,2,4,6}), one
    // denylist bitset (bit=1 drops ids {2,6}). Composed keep-set = allow AND NOT deny:
    //   id 0: allow=1, deny=0 -> KEEP    id 2: allow=1, deny=1 -> DROP (denied)
    //   id 4: allow=1, deny=0 -> KEEP    id 6: allow=1, deny=1 -> DROP (denied)
    //   id 1,3,5,7: allow=0 -> DROP
    // Expected stable-order result: ids [0, 4], scores [10, 14].
    func testIdFilterCompactNComposedAllowAndDeny() {
        let n = 8
        let capacity = 64
        var allow0Word: UInt64 = 0
        var denyWord: UInt64 = 0
        for id in [0, 2, 4, 6] { allow0Word |= (1 << UInt64(id)) }
        for id in [2, 6] { denyWord |= (1 << UInt64(id)) }

        let ids: [Int64] = (0..<n).map { Int64($0) }
        let scores: [Float] = (0..<n).map { Float(10 + $0) }
        var idsOut = [Int64](repeating: -1, count: n)
        var scoresOut = [Float](repeating: .nan, count: n)

        let kept = withUnsafePointer(to: &allow0Word) { a0 in
            withUnsafePointer(to: &denyWord) { dp in
                ids.withUnsafeBufferPointer { idsBuf in
                    scores.withUnsafeBufferPointer { scoresBuf in
                        idsOut.withUnsafeMutableBufferPointer { idsOutBuf in
                            scoresOut.withUnsafeMutableBufferPointer { scoresOutBuf in
                                idFilterCompactN(
                                    filters: [a0, dp],
                                    modes: [.allowlist, .denylist],
                                    filterCount: 2,
                                    idsIn: idsBuf.baseAddress!,
                                    scoresIn: scoresBuf.baseAddress,
                                    count: n,
                                    capacity: capacity,
                                    idsOut: idsOutBuf.baseAddress!,
                                    scoresOut: scoresOutBuf.baseAddress
                                )
                            }
                        }
                    }
                }
            }
        }

        XCTAssertEqual(kept, 2)
        XCTAssertEqual(Array(idsOut[0..<kept]), [0, 4])
        XCTAssertEqual(Array(scoresOut[0..<kept]), [10, 14])
    }

    // F=0 (no filters at all) must keep every id — idFilterPassN with all-nil allow/deny
    // pointers returns true after the bounds check.
    func testIdFilterCompactNWithZeroFiltersKeepsAll() {
        let n = 4
        let ids: [Int64] = [5, 6, 7, 8]
        let scores: [Float] = [1, 2, 3, 4]
        var idsOut = [Int64](repeating: -1, count: n)
        var scoresOut = [Float](repeating: .nan, count: n)

        let kept = ids.withUnsafeBufferPointer { idsBuf in
            scores.withUnsafeBufferPointer { scoresBuf in
                idsOut.withUnsafeMutableBufferPointer { idsOutBuf in
                    scoresOut.withUnsafeMutableBufferPointer { scoresOutBuf in
                        idFilterCompactN(
                            filters: [], modes: [], filterCount: 0,
                            idsIn: idsBuf.baseAddress!, scoresIn: scoresBuf.baseAddress,
                            count: n, capacity: 64,
                            idsOut: idsOutBuf.baseAddress!, scoresOut: scoresOutBuf.baseAddress
                        )
                    }
                }
            }
        }

        XCTAssertEqual(kept, 4)
        XCTAssertEqual(idsOut, ids)
        XCTAssertEqual(scoresOut, scores)
    }
}
