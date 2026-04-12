import XCTest
import VectorCore
@testable import VectorIndex

/// Gap 1 coverage — verifies that the HNSW WAL sidecar durably captures
/// protocol-level mutations, replays them deterministically through
/// `internalInsertAtLevel` (without advancing the RNG), and remains
/// torn-write safe under truncation.
final class HNSWWALTests: XCTestCase {

    // MARK: - Fixtures

    private struct LCG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let bits = UInt32(truncatingIfNeeded: state >> 32)
            return Float(bits) / Float(UInt32.max)
        }
    }

    private func normalizedVector(dim: Int, rng: inout LCG) -> [Float] {
        var v = [Float](repeating: 0, count: dim)
        var s: Float = 0
        for i in 0..<dim {
            let x = rng.next() * 2 - 1
            v[i] = x
            s += x * x
        }
        let inv = 1.0 / sqrtf(max(s, 1e-12))
        for i in 0..<dim { v[i] *= inv }
        return v
    }

    private func makeCorpus(count: Int, dim: Int, seed: UInt64) -> [(String, [Float])] {
        var rng = LCG(state: seed)
        return (0..<count).map { i in ("v\(i)", normalizedVector(dim: dim, rng: &rng)) }
    }

    private func tmpDir(name: String = "hnsw-wal-test") -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanUp(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Frame codec

    /// Sanity: every record type round-trips through encode/decode.
    func testFrameRoundTripAllRecordTypes() throws {
        let insertItem = HNSWWALInsertItem(
            id: "alpha", level: 3,
            vector: [0.1, -0.2, 0.3, 0.4],
            metadata: ["k": "v"],
            knownInvNorm: 1.0
        )
        let cases: [HNSWWALRecord] = [
            .insert(insertItem),
            .remove(id: "bravo"),
            .clear,
            .batchInsert([
                insertItem,
                HNSWWALInsertItem(id: "beta", level: 0, vector: [1, 0, 0, 0], metadata: nil, knownInvNorm: nil)
            ])
        ]

        for record in cases {
            let frame = try hnswWalEncodeFrame(record)
            // Write to a temp file and replay
            let path = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("hnsw-wal-roundtrip-\(UUID().uuidString)").path
            defer { try? FileManager.default.removeItem(atPath: path) }
            FileManager.default.createFile(atPath: path, contents: Data(frame))
            let (decoded, didTruncate) = try HNSWWALReader.replay(path: path)
            XCTAssertFalse(didTruncate)
            XCTAssertEqual(decoded.count, 1)
            switch (decoded[0], record) {
            case (.insert(let a), .insert(let b)):
                XCTAssertEqual(a.id, b.id)
                XCTAssertEqual(a.level, b.level)
                XCTAssertEqual(a.vector, b.vector)
                XCTAssertEqual(a.metadata, b.metadata)
                XCTAssertEqual(a.knownInvNorm, b.knownInvNorm)
            case (.remove(let a), .remove(let b)):
                XCTAssertEqual(a, b)
            case (.clear, .clear):
                break
            case (.batchInsert(let a), .batchInsert(let b)):
                XCTAssertEqual(a.count, b.count)
                for (x, y) in zip(a, b) {
                    XCTAssertEqual(x.id, y.id)
                    XCTAssertEqual(x.level, y.level)
                    XCTAssertEqual(x.vector, y.vector)
                }
            default:
                XCTFail("record variant mismatch")
            }
        }
    }

    // MARK: - Clean-close replay

    /// Insert a corpus with WAL enabled, disable cleanly (no checkpoint),
    /// reopen with enableWAL — the search results must match the pre-close
    /// in-memory index.
    func testReplayAfterCleanCloseMatches() async throws {
        let dim = 32
        let walDir = tmpDir()
        defer { cleanUp(walDir) }

        let corpus = makeCorpus(count: 80, dim: dim, seed: 0xD011)
        let original = HNSWIndex(dimension: dim, metric: .cosine)
        try await original.enableWAL(directory: walDir)
        for (id, v) in corpus {
            try await original.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }

        // Capture reference search results.
        var qRng = LCG(state: 99)
        var referenceResults: [[StringSearchResult]] = []
        for _ in 0..<6 {
            let q = normalizedVector(dim: dim, rng: &qRng)
            let r = try await original.search(query: q, k: 5, filter: nil)
            referenceResults.append(r)
        }
        try await original.disableWAL()

        // Fresh index, replay WAL.
        let reopened = HNSWIndex(dimension: dim, metric: .cosine)
        try await reopened.enableWAL(directory: walDir)
        let recoveredCount = await reopened.count
        XCTAssertEqual(recoveredCount, corpus.count)

        // Search with the same queries — ids must match.
        var qRng2 = LCG(state: 99)
        for i in 0..<6 {
            let q = normalizedVector(dim: dim, rng: &qRng2)
            let r = try await reopened.search(query: q, k: 5, filter: nil)
            XCTAssertEqual(r.map { $0.id }, referenceResults[i].map { $0.id },
                           "replay result ordering at query \(i)")
        }
        try await reopened.disableWAL()
    }

    // MARK: - Remove persistence

    /// Soft-deletes committed via remove() survive a reopen.
    func testRemoveRecordIsReplayed() async throws {
        let dim = 16
        let walDir = tmpDir()
        defer { cleanUp(walDir) }

        let corpus = makeCorpus(count: 20, dim: dim, seed: 42)
        let first = HNSWIndex(dimension: dim, metric: .cosine)
        try await first.enableWAL(directory: walDir)
        for (id, v) in corpus {
            try await first.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }
        // Remove half.
        for i in 0..<10 {
            try await first.remove(id: "v\(i)")
        }
        try await first.disableWAL()

        let reopened = HNSWIndex(dimension: dim, metric: .cosine)
        try await reopened.enableWAL(directory: walDir)
        let removeCount = await reopened.count
        XCTAssertEqual(removeCount, 10)
        // Removed ids must not resolve.
        for i in 0..<10 {
            let present = await reopened.contains(id: "v\(i)")
            XCTAssertFalse(present, "removed id v\(i) leaked through replay")
        }
        // Surviving ids must still be searchable.
        for i in 10..<20 {
            let present = await reopened.contains(id: "v\(i)")
            XCTAssertTrue(present, "surviving id v\(i) missing after replay")
        }
        try await reopened.disableWAL()
    }

    // MARK: - Torn frame truncation

    /// Manually truncate the WAL file mid-frame and verify that replay stops
    /// cleanly at the last intact frame rather than crashing or yielding
    /// garbage. Matches the SQLite/Redis convention for torn writes.
    func testTornFrameIsTruncatedOnReplay() async throws {
        let dim = 8
        let walDir = tmpDir()
        defer { cleanUp(walDir) }

        let corpus = makeCorpus(count: 10, dim: dim, seed: 7)
        let idx = HNSWIndex(dimension: dim, metric: .cosine)
        try await idx.enableWAL(directory: walDir)
        for (id, v) in corpus {
            try await idx.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }
        try await idx.disableWAL()

        // Remove the last byte (the CRC trailer), guaranteeing the final frame
        // is torn regardless of alignment. The reader should validate, reject
        // the incomplete frame, and stop at the 9th intact record.
        let walPath = walDir.appendingPathComponent("hnsw.wal").path
        let full = try Data(contentsOf: URL(fileURLWithPath: walPath))
        let truncated = full.prefix(full.count - 1)
        try truncated.write(to: URL(fileURLWithPath: walPath))

        // Reader should detect the torn frame and stop.
        let (records, didTruncate) = try HNSWWALReader.replay(path: walPath)
        XCTAssertTrue(didTruncate)
        XCTAssertLessThan(records.count, corpus.count)

        // Reopen the index via enableWAL — should apply only the intact records.
        let reopened = HNSWIndex(dimension: dim, metric: .cosine)
        try await reopened.enableWAL(directory: walDir)
        let tornCount = await reopened.count
        XCTAssertEqual(tornCount, records.count)
        try await reopened.disableWAL()
    }

    // MARK: - Checkpoint race

    /// Simulate a crash between snapshot write and WAL truncate: the snapshot
    /// contains all records, the WAL still holds the same records. Reopening
    /// from snapshot + replaying the WAL must still yield the correct state
    /// (duplicate-id inserts collapse via the replace-existing path).
    func testCheckpointRaceSnapshotPlusRedundantWAL() async throws {
        let dim = 16
        let walDir = tmpDir()
        let snapshotURL = walDir.appendingPathComponent("snapshot.json")
        defer { cleanUp(walDir) }

        let corpus = makeCorpus(count: 30, dim: dim, seed: 0xBAD)
        let idx = HNSWIndex(dimension: dim, metric: .cosine)
        try await idx.enableWAL(directory: walDir)
        for (id, v) in corpus {
            try await idx.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }
        // Write snapshot — but do NOT truncate the WAL, simulating a crash
        // between save() and ftruncate().
        try await idx.save(to: snapshotURL)
        // Capture reference results to compare against the recovered index.
        var qRng = LCG(state: 55)
        var reference: [[StringSearchResult]] = []
        for _ in 0..<5 {
            let q = normalizedVector(dim: dim, rng: &qRng)
            reference.append(try await idx.search(query: q, k: 5, filter: nil))
        }
        try await idx.disableWAL()

        // Reopen: load snapshot, then enableWAL which replays the old WAL
        // records on top of the loaded state. Duplicate inserts must not
        // corrupt the graph or the contains() map.
        let reopened = try await HNSWIndex.openDurable(
            snapshotURL: snapshotURL,
            walDirectory: walDir,
            dimension: dim,
            metric: .cosine
        )
        // Active count should remain exactly equal to the corpus size —
        // duplicate-id inserts go through remove-then-reinsert which
        // preserves activeCount at 1 per unique id.
        let recoveredCount = await reopened.count
        XCTAssertEqual(recoveredCount, corpus.count)

        var qRng2 = LCG(state: 55)
        for i in 0..<5 {
            let q = normalizedVector(dim: dim, rng: &qRng2)
            let r = try await reopened.search(query: q, k: 5, filter: nil)
            XCTAssertEqual(r.map { $0.id }.sorted(), reference[i].map { $0.id }.sorted(),
                           "recovered ids mismatch query \(i)")
        }
        try await reopened.disableWAL()
    }

    // MARK: - Determinism

    /// Two fresh indices replaying the same WAL file must produce identical
    /// search results. This is the core replay-determinism guarantee — if
    /// levels weren't logged, the two RNG streams would diverge and the
    /// graphs would differ.
    func testReplayDeterminismAcrossTwoFreshIndices() async throws {
        let dim = 32
        let walDir = tmpDir()
        defer { cleanUp(walDir) }

        let corpus = makeCorpus(count: 60, dim: dim, seed: 0x1234)
        let original = HNSWIndex(dimension: dim, metric: .cosine)
        try await original.enableWAL(directory: walDir)
        for (id, v) in corpus {
            try await original.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }
        try await original.disableWAL()

        let a = HNSWIndex(dimension: dim, metric: .cosine)
        let b = HNSWIndex(dimension: dim, metric: .cosine)
        try await a.enableWAL(directory: walDir)
        try await b.enableWAL(directory: walDir)

        var qRng = LCG(state: 888)
        for _ in 0..<8 {
            let q = normalizedVector(dim: dim, rng: &qRng)
            let ra = try await a.search(query: q, k: 7, filter: nil)
            let rb = try await b.search(query: q, k: 7, filter: nil)
            XCTAssertEqual(ra.map { $0.id }, rb.map { $0.id })
            for (x, y) in zip(ra, rb) {
                XCTAssertEqual(x.distance, y.distance, accuracy: 1e-6)
            }
        }
        try await a.disableWAL()
        try await b.disableWAL()
    }

    // MARK: - Batch atomicity

    /// A batchInsert is a single WAL frame. If we truncate mid-frame, none
    /// of the batch's items should survive replay — whole-batch atomicity.
    func testBatchInsertIsAtomicAcrossTornWrite() async throws {
        let dim = 8
        let walDir = tmpDir()
        defer { cleanUp(walDir) }

        // First, a few individual inserts that we want to keep.
        let idx = HNSWIndex(dimension: dim, metric: .cosine)
        try await idx.enableWAL(directory: walDir)
        var rng = LCG(state: 3)
        let keep: [(String, [Float])] = (0..<5).map { ("keep\($0)", normalizedVector(dim: dim, rng: &rng)) }
        for (id, v) in keep {
            try await idx.insert(id: id, vector: v, metadata: nil, knownInvNorm: 1.0)
        }

        // Then a large batch — one single WAL frame.
        let batch: [(VectorID, [Float], [String: String]?)] = (0..<50).map {
            ("batch\($0)", normalizedVector(dim: dim, rng: &rng), nil)
        }
        try await idx.batchInsert(batch)
        try await idx.disableWAL()

        // Truncate the WAL inside the batch frame. Find the offset of the
        // batch frame by reading the file and locating the second magic tag.
        let walPath = walDir.appendingPathComponent("hnsw.wal").path
        let full = try Data(contentsOf: URL(fileURLWithPath: walPath))
        let bytes = [UInt8](full)
        // Locate the start of the batch frame: after five insert frames.
        let (prefix, _) = try HNSWWALReader.replay(path: walPath)
        XCTAssertEqual(prefix.count, 6, "expected 5 inserts + 1 batch frame")

        // Compute end-of-5th-insert by re-scanning until we've seen 5 records
        // and stopping before the batch frame begins.
        var cursor = 0
        var framesSeen = 0
        while framesSeen < 5 {
            let magic = bytes.withUnsafeBufferPointer { bp -> UInt32 in
                var le: UInt32 = 0
                _ = withUnsafeMutableBytes(of: &le) { dst in
                    memcpy(dst.baseAddress!, bp.baseAddress!.advanced(by: cursor), 4)
                }
                return UInt32(littleEndian: le)
            }
            XCTAssertEqual(magic, HNSWWAL_MAGIC)
            let bodyLen = bytes.withUnsafeBufferPointer { bp -> UInt32 in
                var le: UInt32 = 0
                _ = withUnsafeMutableBytes(of: &le) { dst in
                    memcpy(dst.baseAddress!, bp.baseAddress!.advanced(by: cursor + 8), 4)
                }
                return UInt32(littleEndian: le)
            }
            cursor += 12 + Int(bodyLen) + 4 + 4
            framesSeen += 1
        }
        // Truncate to a point INSIDE the batch frame body.
        let batchFrameStart = cursor
        XCTAssertLessThan(batchFrameStart, full.count)
        let truncateAt = batchFrameStart + 50  // partway into the batch body
        try Data(full.prefix(truncateAt)).write(to: URL(fileURLWithPath: walPath))

        // Reopen: only the 5 keep inserts should replay, no batch items.
        let reopened = HNSWIndex(dimension: dim, metric: .cosine)
        try await reopened.enableWAL(directory: walDir)
        let atomicCount = await reopened.count
        XCTAssertEqual(atomicCount, 5)
        for i in 0..<5 {
            let present = await reopened.contains(id: "keep\(i)")
            XCTAssertTrue(present)
        }
        for i in 0..<50 {
            let present = await reopened.contains(id: "batch\(i)")
            XCTAssertFalse(present, "partial batch leaked through — atomicity broken")
        }
        try await reopened.disableWAL()
    }
}
