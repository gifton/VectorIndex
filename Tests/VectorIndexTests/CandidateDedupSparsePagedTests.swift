import XCTest
@testable import VectorIndex

final class CandidateDedupSparsePagedTests: XCTestCase {
    // Characterization test pinning `.sparsePaged` mode's current behavior BEFORE the B16
    // Page value-struct -> reference-type refactor. `.sparsePaged` has zero existing test
    // coverage (grep-confirmed), so this pins: cross-page allocation, first-seen/duplicate
    // semantics, page reuse across queries (only the FIRST-ever touch of a page counts
    // toward pagesAllocatedThisQuery), and per-touched-page clearing on resetForNewQuery().
    func testSparsePagedCrossPageAllocationDuplicateAndReset() {
        let idCapacity: Int64 = 200_000
        let vs = DefaultVisitedSet(idCapacity: idCapacity, opts: VisitedOpts(mode: .sparsePaged))

        // Three ids on three distinct pages (pageBits defaults to 15 -> 32,768 ids/page).
        let idPage0: Int64 = 10
        let idPage1: Int64 = 40_000   // 40_000 >> 15 == 1
        let idPage2: Int64 = 70_000   // 70_000 >> 15 == 2

        XCTAssertTrue(vs.testAndSet(id: idPage0), "first touch must be newly-seen")
        XCTAssertTrue(vs.testAndSet(id: idPage1))
        XCTAssertTrue(vs.testAndSet(id: idPage2))
        XCTAssertEqual(vs.pagesAllocatedThisQuery, 3, "three distinct pages must each allocate once")

        XCTAssertFalse(vs.testAndSet(id: idPage0), "second touch of the same id must be a duplicate")
        XCTAssertTrue(vs.contains(idPage1))
        XCTAssertFalse(vs.contains(Int64(41_000)), "an untouched id on the same page as idPage1 must not read as set")

        vs.resetForNewQuery()
        XCTAssertEqual(vs.pagesClearedThisQuery, 3, "reset must clear exactly the 3 pages touched last query")

        // Page reuse: touching a NEW id on the already-allocated page 0 must not re-count
        // as a fresh page allocation (the Page object persists across queries in pageTable).
        let idPage0Other: Int64 = 11
        XCTAssertTrue(vs.testAndSet(id: idPage0Other), "bits must be cleared after reset")
        XCTAssertEqual(vs.pagesAllocatedThisQuery, 0, "reusing an already-allocated page must not increment pagesAllocatedThisQuery")

        // Old id from the previous query must read as fresh again post-reset (bits cleared).
        XCTAssertTrue(vs.testAndSet(id: idPage0), "id from previous query must be newly-seen after reset")
    }
}
