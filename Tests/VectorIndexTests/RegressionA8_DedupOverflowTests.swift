import XCTest
@testable import VectorIndex

final class RegressionA8_DedupOverflowTests: XCTestCase {
    func testFixedBitsetResetClearsPostSaturationBits() {
        // wCount = ceil(idCapacity/64) must exceed 1_000_000 so touchedCapacity caps at 1_000_000,
        // AND wCount must be large enough that wordCount/4 > 1_000_000 (sparse clear threshold).
        // This ensures ring saturates in sparse-clear mode where the bug manifests.
        // For sparse clear to activate: touchedCount < wordCount/4
        // We want: 1_000_000 < wordCount/4, so wordCount > 4_000_000, so idCapacity > 256M
        let idCapacity: Int64 = 260_000_000           // wCount ~= 4_062_500, wordCount/4 = 1_015_625
        let opts = VisitedOpts(mode: .fixedBitset)
        let vs = DefaultVisitedSet(idCapacity: idCapacity, opts: opts)

        // Touch enough distinct words to saturate the ring (1_000_000) plus a few more.
        // Word index = id >> 6; step by 64 to hit distinct words.
        let cap = 1_000_000
        let extra = 5
        for w in 0..<(cap + extra) {
            let id = Int64(w) << 6
            _ = vs.testAndSet(id: id)
        }

        // Confirm ring saturated
        XCTAssertEqual(vs.touchedCount, cap, "touched-word ring must saturate at capacity")

        // Pre-saturation words should be marked as duplicates
        let postSaturationID = Int64(cap + 1) << 6
        XCTAssertFalse(vs.testAndSet(id: postSaturationID), "post-saturation word should be marked before reset")

        vs.resetForNewQuery()

        // Post-saturation word must be cleared on reset: testAndSet returns true (newly inserted).
        // Without the fix, this would return false because the word is stale (not in sparse-clear list).
        XCTAssertTrue(vs.testAndSet(id: postSaturationID),
                      "post-saturation bit must be cleared on reset when ring overflowed")
    }
}
