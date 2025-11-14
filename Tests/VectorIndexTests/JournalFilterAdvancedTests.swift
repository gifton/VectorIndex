import XCTest
@testable import VectorIndex

final class JournalFilterAdvancedTests: XCTestCase {

    // MARK: - Date Parsing & Boundaries

    func testDateFractionalSecondsAndTimezone() {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime]
        let start = fmt.date(from: "2025-01-01T00:00:00Z")!
        let end   = fmt.date(from: "2025-01-02T00:00:00Z")!

        let f = JournalFilter().dateBetween(start, end).build()

        // Fractional seconds
        XCTAssertTrue(f(["date":"2025-01-01T12:34:56.789Z"]))
        // Timezone offset (+05:00), should parse and compare in absolute time
        XCTAssertTrue(f(["date":"2025-01-01T05:00:00+05:00"]))
        // Outside range
        XCTAssertFalse(f(["date":"2024-12-31T23:59:59Z"]))
        XCTAssertFalse(f(["date":"2025-01-02T00:00:01Z"]))
    }

    func testDateBoundaryInclusive() {
        let fmt = ISO8601DateFormatter(); fmt.formatOptions = [.withInternetDateTime]
        let start = fmt.date(from: "2025-01-01T00:00:00Z")!
        let end   = fmt.date(from: "2025-01-31T23:59:59Z")!
        let f = JournalFilter().dateBetween(start, end).build()
        XCTAssertTrue(f(["date":"2025-01-01T00:00:00Z"]))
        XCTAssertTrue(f(["date":"2025-01-31T23:59:59Z"]))
    }

    func testInvalidDateStringHandling() {
        let fmt = ISO8601DateFormatter(); fmt.formatOptions = [.withInternetDateTime]
        let start = fmt.date(from: "2025-01-01T00:00:00Z")!
        let end   = fmt.date(from: "2025-01-31T23:59:59Z")!
        let jf = JournalFilter().dateBetween(start, end)
        // Invalid string → should be treated as not in range (false), even if allowMissingKeys = true
        XCTAssertFalse(jf.build()(["date":"not-a-date"]))
        XCTAssertFalse(jf.allowMissingKeys(true).build()(["date":"not-a-date"]))
    }

    // MARK: - Tags & Delimiters

    func testTagsSemicolonDelimiter() {
        let meta = ["tags":"work; journal ; mood "]
        let jf = JournalFilter().setKeys(tagsKey: "tags", delimiter: ";").includingTags(["journal"]) // any
        XCTAssertTrue(jf.build()(meta))
        XCTAssertFalse(jf.includingTags(["sleep"]).build()(meta))
    }

    func testTagsWhitespaceAndCase() {
        let meta = ["tags":"Work, Journal, Mood "]
        // Case-sensitive match (current behavior):
        XCTAssertTrue(JournalFilter().includingTags(["Work"]).build()(meta))
        XCTAssertFalse(JournalFilter().includingTags(["work"]).build()(meta))
    }

    func testIncludeTagsEmptyArrayBehavior() {
        // Including empty set results in a filter that rejects all (current semantics)
        let meta = ["tags":"x,y"]
        let jf = JournalFilter().includingTags([])
        XCTAssertFalse(jf.build()(meta))
    }

    func testExcludingTagsEmptyArrayBehavior() {
        // Excluding empty set does nothing
        let meta = ["tags":"x,y"]
        let jf = JournalFilter().excludingTags([])
        XCTAssertTrue(jf.build()(meta))
    }

    func testIncludeExcludeCombined() {
        let meta = ["tags":"journal, public"]
        let jf = JournalFilter()
            .includingTags(["journal"]) // must include
            .excludingTags(["private"]) // must not include
        XCTAssertTrue(jf.build()(meta))
        XCTAssertFalse(jf.excludingTags(["public"]).build()(meta))
    }

    // MARK: - Missing Keys Policy

    func testAllowMissingKeysBehavior() {
        // Missing keys allowed → passes despite constraints configured
        var jf = JournalFilter()
            .includingTags(["journal"]) // would normally require tags
            .allowMissingKeys(true)
        XCTAssertTrue(jf.build()([:]))

        // Missing keys not allowed → fails
        jf = JournalFilter().includingTags(["journal"]).allowMissingKeys(false)
        XCTAssertFalse(jf.build()([:]))
    }

    // MARK: - Custom Predicates

    func testCustomPredicateOrderAndRejection() {
        let meta = ["tags":"journal", "title":""]
        let jf = JournalFilter()
            .includingTags(["journal"]) // passes
            .and { m in !(m["title"]?.isEmpty ?? true) } // rejects
        XCTAssertFalse(jf.build()(meta))
    }

    // MARK: - Convenience Builder Equivalence

    func testMakeFilterConvenienceEquivalence() {
        let fmt = ISO8601DateFormatter(); fmt.formatOptions = [.withInternetDateTime]
        let start = fmt.date(from: "2025-01-01T00:00:00Z")!
        let end   = fmt.date(from: "2025-01-31T23:59:59Z")!
        let metaA = ["date":"2025-01-05T00:00:00Z", "tags":"x,y"]
        let metaB = ["date":"2024-12-31T23:59:59Z", "tags":"x,y"]

        let f1 = JournalFilter()
            .dateBetween(start, end)
            .includingTags(["x"]) // any
            .build()
        let f2 = JournalFilter.makeFilter(
            allowedDateRange: start...end,
            includeTags: ["x"]
        )
        XCTAssertEqual(f1(metaA), f2(metaA))
        XCTAssertEqual(f1(metaB), f2(metaB))
    }

    // MARK: - Concurrency Sanity

    func testConcurrentInvoke() async {
        let f = JournalFilter().includingTags(["a"]).build()
        let metas: [[String:String]] = (0..<100).map { i in
            ["tags": (i % 2 == 0) ? "a,b" : "b,c"]
        }
        await withTaskGroup(of: Bool.self) { group in
            for m in metas { group.addTask { f(m) } }
            var trues = 0
            for await v in group { if v { trues += 1 } }
            XCTAssertEqual(trues, 50)
        }
    }
}

