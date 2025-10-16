import XCTest
@testable import VectorIndex

final class HNSWLevelAssignmentTests: XCTestCase {
    func testDeterministicSampling() {
        var a = HNSWXoroRNGState.from(seed: 0xABCDEF0123456789, stream: 7)
        var b = HNSWXoroRNGState.from(seed: 0xABCDEF0123456789, stream: 7)
        var seqA: [Int] = []
        var seqB: [Int] = []
        let M = 16, cap = 16
        for _ in 0..<200 {
            seqA.append(hnswSampleLevel(M: M, cap: cap, rng: &a))
            seqB.append(hnswSampleLevel(M: M, cap: cap, rng: &b))
        }
        XCTAssertEqual(seqA, seqB, "Level sampling must be deterministic for same seed/stream")
        // Range sanity
        XCTAssertTrue(seqA.allSatisfy { $0 >= 0 && $0 <= cap })
    }
}
