import Foundation

/// Journaling-oriented filter DSL that compiles to the index `filter:` closure.
///
/// - Supports ISO-8601 date range filtering (on a configurable metadata key)
/// - Supports tag inclusion/exclusion from a delimited metadata field
/// - Allows custom predicates to be AND-ed with built-ins
///
/// Metadata schema assumptions (overrideable):
/// - Date is stored under `dateKey` as ISO-8601 string (e.g., "2025-01-02T12:34:56Z")
/// - Tags are stored under `tagsKey` as a delimiter-separated string (default: comma-separated)
public struct JournalFilter: Sendable {
    // MARK: - Configuration
    public var dateKey: String = "date"
    public var tagsKey: String = "tags"
    public var tagsDelimiter: Character = ","
    /// If true, when `includeTags` is set, require all to be present; otherwise any
    public var requireAllIncludedTags: Bool = false
    /// If true, items missing required keys still pass built-in checks (custom preds still apply)
    public var includeIfMissingKeys: Bool = false

    // MARK: - Criteria
    public var allowedDateRange: ClosedRange<Date>?
    public var includeTags: Set<String>?
    public var excludeTags: Set<String>?
    /// Additional custom predicates AND-ed with built-ins
    public var custom: [@Sendable ([String: String]) -> Bool] = []

    public init() {}

    // MARK: - Builder API
    public func dateBetween(_ start: Date, _ end: Date) -> JournalFilter {
        var copy = self
        copy.allowedDateRange = start...end
        return copy
    }

    public func setKeys(dateKey: String? = nil, tagsKey: String? = nil, delimiter: Character? = nil) -> JournalFilter {
        var copy = self
        if let k = dateKey { copy.dateKey = k }
        if let k = tagsKey { copy.tagsKey = k }
        if let d = delimiter { copy.tagsDelimiter = d }
        return copy
    }

    public func includingTags(_ tags: [String], requireAll: Bool = false) -> JournalFilter {
        var copy = self
        copy.includeTags = Set(tags.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })
        copy.requireAllIncludedTags = requireAll
        return copy
    }

    public func excludingTags(_ tags: [String]) -> JournalFilter {
        var copy = self
        copy.excludeTags = Set(tags.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })
        return copy
    }

    public func allowMissingKeys(_ allow: Bool) -> JournalFilter {
        var copy = self
        copy.includeIfMissingKeys = allow
        return copy
    }

    public func and(_ predicate: @escaping @Sendable ([String: String]) -> Bool) -> JournalFilter {
        var copy = self
        copy.custom.append(predicate)
        return copy
    }

    // MARK: - Build Filter Closure
    /// Compile the DSL to a Sendable metadata predicate for index search.
    /// - Returns: `@Sendable` closure suitable for `search(..., filter:)`.
    public func build() -> @Sendable ([String: String]?) -> Bool {
        // Capture immutable config in locals for Sendable closure
        let dateKey = self.dateKey
        let tagsKey = self.tagsKey
        let delimiter = self.tagsDelimiter
        let requireAll = self.requireAllIncludedTags
        let includeIfMissing = self.includeIfMissingKeys
        let dateRange = self.allowedDateRange
        let includeTags = self.includeTags
        let excludeTags = self.excludeTags
        let customs = self.custom

        return { meta in
            guard let meta = meta else { return includeIfMissing }

            // Date check (ISO-8601) – parse if configured
            if let range = dateRange {
                if let dateStr = meta[dateKey] {
                    // Use a fresh formatter per call to avoid thread-safety issues
                    let fmt = ISO8601DateFormatter()
                    fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                    let date = fmt.date(from: dateStr) ?? ISO8601DateFormatter().date(from: dateStr)
                    guard let dd = date, range.contains(dd) else { return false }
                } else if !includeIfMissing {
                    return false
                }
            }

            // Tags inclusion/exclusion
            if includeTags != nil || excludeTags != nil {
                guard let raw = meta[tagsKey] else { if includeIfMissing { return true } else { return false } }
                let parts = raw.split(separator: delimiter).map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
                let tagSet = Set(parts)

                if let inc = includeTags {
                    if requireAll {
                        if !inc.isSubset(of: tagSet) { return false }
                    } else {
                        if inc.isDisjoint(with: tagSet) { return false }
                    }
                }
                if let exc = excludeTags {
                    if !exc.isDisjoint(with: tagSet) { return false }
                }
            }

            // Custom predicates (AND)
            for p in customs { if p(meta) == false { return false } }
            return true
        }
    }

    // MARK: - Convenience
    /// Convenience single-shot constructor.
    public static func makeFilter(
        dateKey: String = "date",
        allowedDateRange: ClosedRange<Date>? = nil,
        tagsKey: String = "tags",
        tagsDelimiter: Character = ",",
        includeTags: [String]? = nil,
        excludeTags: [String]? = nil,
        requireAllIncludedTags: Bool = false,
        includeIfMissingKeys: Bool = false,
        custom: [@Sendable ([String: String]) -> Bool] = []
    ) -> @Sendable ([String: String]?) -> Bool {
        var jf = JournalFilter()
        jf.dateKey = dateKey
        jf.tagsKey = tagsKey
        jf.tagsDelimiter = tagsDelimiter
        jf.allowedDateRange = allowedDateRange
        if let inc = includeTags { jf.includeTags = Set(inc) }
        if let exc = excludeTags { jf.excludeTags = Set(exc) }
        jf.requireAllIncludedTags = requireAllIncludedTags
        jf.includeIfMissingKeys = includeIfMissingKeys
        jf.custom = custom
        return jf.build()
    }
}
