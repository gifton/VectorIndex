import XCTest
@testable import VectorIndex

final class JournalFilterTests: XCTestCase {
    func testDateRangeInclusive_ISO8601() {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime]
        let start = fmt.date(from: "2025-01-01T00:00:00Z")!
        let end   = fmt.date(from: "2025-01-31T23:59:59Z")!

        let jf = JournalFilter().dateBetween(start, end)
        let f = jf.build()

        XCTAssertTrue(f(["date":"2025-01-15T12:00:00Z"]))
        XCTAssertFalse(f(["date":"2024-12-31T23:59:59Z"]))
        XCTAssertFalse(f([:])) // missing key excluded by default
        XCTAssertTrue(jf.allowMissingKeys(true).build()([:])) // allow missing
    }

    func testTagsAnyAndAll() {
        let meta = ["tags":"work, journal, mood"]
        // Any
        var jf = JournalFilter().includingTags(["mood"]) // any by default
        XCTAssertTrue(jf.build()(meta))
        jf = JournalFilter().includingTags(["mood","sleep"]) // any match
        XCTAssertTrue(jf.build()(meta))
        // Require all
        jf = JournalFilter().includingTags(["work","mood"], requireAll: true)
        XCTAssertTrue(jf.build()(meta))
        jf = JournalFilter().includingTags(["work","sleep"], requireAll: true)
        XCTAssertFalse(jf.build()(meta))
    }

    func testTagsExclude() {
        let meta = ["tags":"work, journal, mood"]
        let jf = JournalFilter().excludingTags(["private"]) // not present
        XCTAssertTrue(jf.build()(meta))
        let jf2 = JournalFilter().excludingTags(["journal"]) // present → exclude
        XCTAssertFalse(jf2.build()(meta))
    }

    func testCustomPredicate() {
        let metaOk = ["title":"Day 1", "tags":"journal"]
        let metaNo = ["title":"", "tags":"journal"]
        let jf = JournalFilter()
            .includingTags(["journal"]) // must include
            .and { m in !(m["title"]?.isEmpty ?? true) }
        let f = jf.build()
        XCTAssertTrue(f(metaOk))
        XCTAssertFalse(f(metaNo))
    }
}

