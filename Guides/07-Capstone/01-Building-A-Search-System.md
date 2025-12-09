# Building a Search System

> **Reading time:** 25 minutes
> **Prerequisites:** All previous chapters

---

## Overview

This capstone walks through building a complete semantic search system using VectorIndex. We'll cover ingestion, indexing, search, and evaluation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SearchService                                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Ingestion Layer                              │   │
│  │                                                                   │   │
│  │  Documents ──→ Chunking ──→ Embedding ──→ Index                  │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Search Layer                                │   │
│  │                                                                   │   │
│  │  Query ──→ Embedding ──→ Index Search ──→ Rerank ──→ Results    │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Evaluation Layer                              │   │
│  │                                                                   │   │
│  │  Ground Truth ──→ Metrics (Recall, Latency) ──→ Tuning          │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Define the Data Model

```swift
import Foundation
import VectorIndex

/// A document in our search corpus
struct Document: Identifiable, Codable {
    let id: String
    let title: String
    let content: String
    let metadata: [String: String]

    /// Chunk the document for embedding
    func chunks(maxLength: Int = 512) -> [DocumentChunk] {
        // Simple chunking by paragraphs
        let paragraphs = content.components(separatedBy: "\n\n")
        return paragraphs.enumerated().map { index, text in
            DocumentChunk(
                id: "\(id)_chunk_\(index)",
                documentId: id,
                text: String(text.prefix(maxLength)),
                chunkIndex: index
            )
        }
    }
}

/// A chunk of a document with its embedding
struct DocumentChunk: Identifiable {
    let id: String
    let documentId: String
    let text: String
    let chunkIndex: Int
    var embedding: [Float]?
}
```

---

## Step 2: Build the Index

```swift
/// Main search service
actor SearchService {
    private let index: HNSWIndex
    private let dimension: Int
    private var chunks: [String: DocumentChunk] = [:]

    init(dimension: Int = 768, config: HNSWIndex.Configuration? = nil) async {
        self.dimension = dimension
        // HNSWIndex is an actor - create and configure
        self.index = HNSWIndex(
            dimension: dimension,
            metric: .cosine,  // From VectorCore.SupportedDistanceMetric
            config: config ?? .init(m: 16, efConstruction: 200, efSearch: 64)
        )
    }

    /// Ingest documents into the index
    func ingest(documents: [Document], embedder: (String) async throws -> [Float]) async throws {
        for document in documents {
            let documentChunks = document.chunks()

            for var chunk in documentChunks {
                // Generate embedding
                chunk.embedding = try await embedder(chunk.text)

                guard let embedding = chunk.embedding else { continue }

                // Store chunk
                chunks[chunk.id] = chunk

                // Index the embedding
                try await index.insert(
                    id: chunk.id,
                    vector: embedding,
                    metadata: [
                        "documentId": chunk.documentId,
                        "chunkIndex": String(chunk.chunkIndex)
                    ]
                )
            }
        }

        print("Indexed \(chunks.count) chunks from \(documents.count) documents")
    }

    /// Search for relevant chunks
    func search(query: String, k: Int = 10, embedder: (String) async throws -> [Float]) async throws -> [SearchHit] {
        // Embed the query
        let queryEmbedding = try await embedder(query)

        // Search the index
        let results = try await index.search(
            query: queryEmbedding,
            k: k,
            filter: nil
        )

        // Enrich results with chunk data
        return results.compactMap { result in
            guard let chunk = chunks[result.id] else { return nil }
            return SearchHit(
                chunk: chunk,
                score: result.score
            )
        }
    }

    /// Get index statistics
    func statistics() async -> IndexStats {
        await index.statistics()
    }
}

/// A search result with full chunk data
struct SearchHit {
    let chunk: DocumentChunk
    let score: Float
}
```

---

## Step 3: Implement Evaluation

```swift
/// Evaluation utilities
struct Evaluator {
    /// Compute recall@k against ground truth
    static func recallAtK(
        predicted: [SearchHit],
        groundTruth: [String],  // True relevant chunk IDs
        k: Int
    ) -> Float {
        let predictedIDs = Set(predicted.prefix(k).map { $0.chunk.id })
        let trueIDs = Set(groundTruth.prefix(k))
        let intersection = predictedIDs.intersection(trueIDs)
        return Float(intersection.count) / Float(min(k, trueIDs.count))
    }

    /// Measure search latency
    static func measureLatency(
        service: SearchService,
        queries: [String],
        k: Int,
        embedder: (String) async throws -> [Float]
    ) async throws -> LatencyStats {
        var latencies: [Double] = []

        for query in queries {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try await service.search(query: query, k: k, embedder: embedder)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            latencies.append(elapsed)
        }

        latencies.sort()
        let n = latencies.count

        return LatencyStats(
            p50: latencies[n / 2],
            p90: latencies[n * 9 / 10],
            p99: latencies[n * 99 / 100],
            mean: latencies.reduce(0, +) / Double(n)
        )
    }
}

struct LatencyStats {
    let p50: Double
    let p90: Double
    let p99: Double
    let mean: Double
}
```

---

## Step 4: Parameter Tuning

```swift
/// Parameter tuning workflow
struct Tuner {
    /// Sweep efSearch to find optimal value
    static func tuneEfSearch(
        service: SearchService,
        testQueries: [(query: String, groundTruth: [String])],
        embedder: (String) async throws -> [Float],
        targetRecall: Float = 0.95
    ) async throws -> TuningResult {
        var results: [(ef: Int, recall: Float, latency: Double)] = []

        for ef in [10, 20, 50, 100, 200, 500] {
            // Note: In practice, you'd need to reconfigure the index or
            // use a method that accepts efSearch as a parameter
            var totalRecall: Float = 0

            for (query, groundTruth) in testQueries {
                let hits = try await service.search(query: query, k: 10, embedder: embedder)
                let recall = Evaluator.recallAtK(predicted: hits, groundTruth: groundTruth, k: 10)
                totalRecall += recall
            }

            let avgRecall = totalRecall / Float(testQueries.count)
            let latencyStats = try await Evaluator.measureLatency(
                service: service,
                queries: testQueries.map { $0.query },
                k: 10,
                embedder: embedder
            )

            results.append((ef, avgRecall, latencyStats.p50))

            print("efSearch=\(ef): recall=\(avgRecall), p50=\(latencyStats.p50)ms")
        }

        // Find smallest ef that meets target recall
        let optimal = results.first { $0.recall >= targetRecall }
            ?? results.max(by: { $0.recall < $1.recall })!

        return TuningResult(
            optimalEfSearch: optimal.ef,
            recall: optimal.recall,
            latency: optimal.latency,
            allResults: results
        )
    }
}

struct TuningResult {
    let optimalEfSearch: Int
    let recall: Float
    let latency: Double
    let allResults: [(ef: Int, recall: Float, latency: Double)]
}
```

---

## Step 5: Putting It Together

```swift
/// Complete usage example
func runSearchSystem() async throws {
    // 1. Create the search service
    let service = SearchService(
        dimension: 768,
        config: .init(m: 16, efConstruction: 200, efSearch: 64)
    )

    // 2. Mock embedder (replace with real embedding model)
    let embedder: (String) async throws -> [Float] = { text in
        // In production: call embedding API or local model
        // For demo: return random embeddings
        return (0..<768).map { _ in Float.random(in: -1...1) }
    }

    // 3. Ingest documents
    let documents = [
        Document(id: "doc1", title: "Swift Programming", content: "Swift is a powerful language...", metadata: [:]),
        Document(id: "doc2", title: "Vector Search", content: "Vector search enables semantic...", metadata: [:]),
        // ... more documents
    ]

    try await service.ingest(documents: documents, embedder: embedder)

    // 4. Search
    let results = try await service.search(query: "How does HNSW work?", k: 5, embedder: embedder)

    print("\nSearch Results:")
    for (i, hit) in results.enumerated() {
        print("\(i+1). [\(hit.score)] \(hit.chunk.text.prefix(100))...")
    }

    // 5. Check statistics
    let stats = await service.statistics()
    print("\nIndex Stats:")
    print("  Type: \(stats.indexType)")
    print("  Vectors: \(stats.vectorCount)")
    print("  Dimension: \(stats.dimension)")

    // 6. Evaluate (with ground truth)
    // let tuning = try await Tuner.tuneEfSearch(...)
}
```

---

## Production Considerations

### Error Handling

```swift
enum SearchError: Error {
    case embeddingFailed(String)
    case indexError(VectorError)
    case invalidQuery
}

// Wrap operations in proper error handling
func safeSearch(query: String) async -> Result<[SearchHit], SearchError> {
    do {
        let results = try await service.search(query: query, k: 10, embedder: embedder)
        return .success(results)
    } catch let error as VectorError {
        return .failure(.indexError(error))
    } catch {
        return .failure(.embeddingFailed(error.localizedDescription))
    }
}
```

### Persistence

```swift
// Save index to disk
try await service.save(to: URL(fileURLWithPath: "/path/to/index.json"))

// Load index from disk
let loadedService = try await SearchService.load(from: URL(fileURLWithPath: "/path/to/index.json"))
```

### Monitoring

```swift
// Track metrics over time
struct SearchMetrics {
    var totalQueries: Int = 0
    var totalLatencyMs: Double = 0
    var errorCount: Int = 0

    var averageLatency: Double {
        totalQueries > 0 ? totalLatencyMs / Double(totalQueries) : 0
    }

    mutating func record(latencyMs: Double, success: Bool) {
        totalQueries += 1
        totalLatencyMs += latencyMs
        if !success { errorCount += 1 }
    }
}
```

---

## 🔗 VectorCore + VectorIndex Connection

This system leverages both libraries:

```
VectorCore provides:
  ✓ SIMD-accelerated distance computation
  ✓ Numerically stable operations
  ✓ Optimized memory layout

VectorIndex provides:
  ✓ Scalable index structures (HNSW, IVF)
  ✓ Approximate nearest neighbor search
  ✓ Parameter tuning framework

Together:
  ✓ Fast, accurate, scalable semantic search
```

---

## Key Takeaways

1. **Layer your system.** Ingestion, search, evaluation as separate concerns.

2. **Start simple, iterate.** FlatIndex → HNSW → tuned parameters.

3. **Measure everything.** Recall, latency, throughput guide decisions.

4. **Handle errors gracefully.** Production systems need resilience.

5. **Plan for scale.** Choose index type based on future needs.

---

## What's Next?

Congratulations on completing the VectorIndex Learning Guide!

You now understand:
- **Why** indexing is necessary (curse of dimensionality)
- **How** different indices work (Flat, IVF, HNSW, PQ)
- **When** to use each approach (decision framework)
- **How to tune** for your requirements (recall/latency tradeoffs)

### Continue Learning

- **Read VectorIndex source code** — Apply concepts to real implementation
- **Build your own search system** — Practice with real embeddings
- **Experiment with parameters** — Develop intuition through measurement
- **Contribute to VectorIndex** — Help improve the library!

---

**[← Back to Welcome](../00-Welcome.md)**

---

*VectorIndex Learning Guide • Volume 2 of the VSK Educational Series • Dec 2024*
