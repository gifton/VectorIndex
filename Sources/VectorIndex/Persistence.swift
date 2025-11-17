import Foundation
import VectorCore

// MARK: - Persistence Model (versioned JSON)

struct PersistedIndex: Codable {
    let type: String
    let version: Int
    let dimension: Int
    let metric: String
    let records: [PersistedRecord]
}

struct PersistedRecord: Codable {
    let id: String
    let vector: [Float]
    let metadata: [String: String]?
}

extension SupportedDistanceMetric {
    static func from(raw: String) -> SupportedDistanceMetric {
        SupportedDistanceMetric(rawValue: raw) ?? .euclidean
    }
}
