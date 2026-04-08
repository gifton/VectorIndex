//
//  SearchResultsAdapter.swift
//  VectorIndex
//
//  Bridges VectorIndexProtocol search results into VectorCore's
//  SearchResults<String> wrapper, surfacing timing and candidate metadata.
//

import Foundation
import VectorCore

public extension VectorIndexProtocol {
    /// Search returning VectorCore's ``StringSearchResults`` with timing and candidate metadata.
    func searchWithMetadata(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> StringSearchResults {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let results = try await search(query: query, k: k, filter: filter)
        let elapsed = DispatchTime.now().uptimeNanoseconds &- t0
        return StringSearchResults(
            results: results,
            candidatesSearched: count,
            searchTimeNanos: elapsed,
            isExhaustive: true // conservative default; overrides below
        )
    }

    /// Batch search returning ``StringSearchResults`` per query.
    func batchSearchWithMetadata(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [StringSearchResults] {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let batchResults = try await batchSearch(queries: queries, k: k, filter: filter)
        let elapsed = DispatchTime.now().uptimeNanoseconds &- t0
        let perQuery = queries.isEmpty ? elapsed : elapsed / UInt64(queries.count)
        return batchResults.map { results in
            StringSearchResults(
                results: results,
                candidatesSearched: count,
                searchTimeNanos: perQuery,
                isExhaustive: true
            )
        }
    }
}

// MARK: - Accurate overrides per index type

public extension FlatIndex {
    func searchWithMetadata(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> StringSearchResults {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let results = try await search(query: query, k: k, filter: filter)
        let elapsed = DispatchTime.now().uptimeNanoseconds &- t0
        return StringSearchResults(
            results: results,
            candidatesSearched: count,
            searchTimeNanos: elapsed,
            isExhaustive: true
        )
    }
}

public extension FlatIndexOptimized {
    func searchWithMetadata(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> StringSearchResults {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let results = try await search(query: query, k: k, filter: filter)
        let elapsed = DispatchTime.now().uptimeNanoseconds &- t0
        return StringSearchResults(
            results: results,
            candidatesSearched: count,
            searchTimeNanos: elapsed,
            isExhaustive: true
        )
    }
}

public extension HNSWIndex {
    func searchWithMetadata(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> StringSearchResults {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let results = try await search(query: query, k: k, filter: filter)
        let elapsed = DispatchTime.now().uptimeNanoseconds &- t0
        return StringSearchResults(
            results: results,
            candidatesSearched: count,
            searchTimeNanos: elapsed,
            isExhaustive: false
        )
    }
}

public extension IVFIndex {
    func searchWithMetadata(
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> StringSearchResults {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let results = try await search(query: query, k: k, filter: filter)
        let elapsed = DispatchTime.now().uptimeNanoseconds &- t0
        return StringSearchResults(
            results: results,
            candidatesSearched: count,
            searchTimeNanos: elapsed,
            isExhaustive: false
        )
    }
}
