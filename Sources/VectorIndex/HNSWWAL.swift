//
//  HNSWWAL.swift
//  VectorIndex
//
//  Write-ahead log sidecar for HNSWIndex. Each user mutation is serialized as
//  a single framed record with an in-frame commit tag and a CRC32 covering the
//  whole payload. On open, the log is replayed in order against an in-memory
//  index to reconstruct live state.
//
//  Replay determinism
//  ------------------
//  HNSW insertions depend on `randomLevel()`, which advances a per-index RNG
//  stream. Naive "log (id, vector, metadata)" replay would produce a different
//  graph topology than the original session. To sidestep this, every InsertRecord
//  carries the exact level that was sampled during the live insert; replay calls
//  `internalInsertAtLevel(...)` with the recorded level and never touches the RNG.
//
//  Frame layout (all little-endian)
//  --------------------------------
//      offset  bytes  field
//      0       4      magic (0x484E5741 = "HNWA")
//      4       4      record_type
//      8       4      body_len
//      12      body_len  body
//      12+N    4      commit_tag (0xC0DEDAD1)
//      16+N    4      crc32
//
//  The CRC covers `record_type || body_len || body || commit_tag`. A frame
//  whose CRC fails validation is treated as end-of-log and ignored — matching
//  the SQLite/Redis AOF convention for torn writes.
//

import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

// MARK: - Frame constants

internal let HNSWWAL_MAGIC: UInt32 = 0x484E5741      // "HNWA"
internal let HNSWWAL_COMMIT_TAG: UInt32 = 0xC0DEDAD1

// MARK: - Record types

internal enum HNSWWALRecordType: UInt32 {
    case insert = 1
    case remove = 2
    case update = 3
    case clear = 4
    case batchInsert = 5
}

// MARK: - Record payloads

internal struct HNSWWALInsertItem {
    var id: String
    var level: Int32
    var vector: [Float]
    var metadata: [String: String]?
    var knownInvNorm: Float?
}

internal enum HNSWWALRecord {
    case insert(HNSWWALInsertItem)
    case remove(id: String)
    case update(id: String, vector: [Float]?, metadata: [String: String]?)
    case clear
    case batchInsert([HNSWWALInsertItem])

    var recordType: HNSWWALRecordType {
        switch self {
        case .insert: return .insert
        case .remove: return .remove
        case .update: return .update
        case .clear: return .clear
        case .batchInsert: return .batchInsert
        }
    }
}

// MARK: - Encoding helpers

private struct HNSWWALEncoder {
    var bytes: [UInt8] = []

    @inline(__always) mutating func writeU8(_ v: UInt8) {
        bytes.append(v)
    }

    @inline(__always) mutating func writeU32(_ v: UInt32) {
        let le = v.littleEndian
        withUnsafeBytes(of: le) { bytes.append(contentsOf: $0) }
    }

    @inline(__always) mutating func writeI32(_ v: Int32) {
        writeU32(UInt32(bitPattern: v))
    }

    @inline(__always) mutating func writeFloat(_ v: Float) {
        writeU32(v.bitPattern)
    }

    mutating func writeString(_ s: String) {
        let data = Array(s.utf8)
        writeU32(UInt32(data.count))
        bytes.append(contentsOf: data)
    }

    mutating func writeOptionalMetadata(_ m: [String: String]?) throws {
        guard let m = m else {
            writeU32(0xFFFF_FFFF)  // nil marker
            return
        }
        let json = try JSONEncoder().encode(m)
        writeU32(UInt32(json.count))
        bytes.append(contentsOf: json)
    }

    mutating func writeVector(_ v: [Float]) {
        writeU32(UInt32(v.count))
        for f in v { writeFloat(f) }
    }

    mutating func writeItem(_ item: HNSWWALInsertItem) throws {
        writeString(item.id)
        writeI32(item.level)
        writeVector(item.vector)
        try writeOptionalMetadata(item.metadata)
        if let inv = item.knownInvNorm {
            writeU8(1)
            writeFloat(inv)
        } else {
            writeU8(0)
        }
    }
}

private struct HNSWWALDecoder {
    let bytes: [UInt8]
    var offset: Int = 0

    var remaining: Int { bytes.count - offset }

    mutating func readU8() throws -> UInt8 {
        guard offset < bytes.count else { throw HNSWWALError.truncatedRecord }
        let v = bytes[offset]
        offset += 1
        return v
    }

    mutating func readU32() throws -> UInt32 {
        guard offset + 4 <= bytes.count else { throw HNSWWALError.truncatedRecord }
        var le: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &le) { dst in
            bytes.withUnsafeBufferPointer { src in
                memcpy(dst.baseAddress!, src.baseAddress!.advanced(by: offset), 4)
            }
        }
        offset += 4
        return UInt32(littleEndian: le)
    }

    mutating func readI32() throws -> Int32 {
        Int32(bitPattern: try readU32())
    }

    mutating func readFloat() throws -> Float {
        Float(bitPattern: try readU32())
    }

    mutating func readString() throws -> String {
        let n = Int(try readU32())
        guard offset + n <= bytes.count else { throw HNSWWALError.truncatedRecord }
        let slice = bytes[offset..<(offset + n)]
        offset += n
        return String(decoding: slice, as: UTF8.self)
    }

    mutating func readOptionalMetadata() throws -> [String: String]? {
        let n = try readU32()
        if n == 0xFFFF_FFFF { return nil }
        guard offset + Int(n) <= bytes.count else { throw HNSWWALError.truncatedRecord }
        let slice = Data(bytes[offset..<(offset + Int(n))])
        offset += Int(n)
        return try JSONDecoder().decode([String: String].self, from: slice)
    }

    mutating func readVector() throws -> [Float] {
        let n = Int(try readU32())
        guard offset + n * 4 <= bytes.count else { throw HNSWWALError.truncatedRecord }
        var v = [Float](repeating: 0, count: n)
        for i in 0..<n { v[i] = try readFloat() }
        return v
    }

    mutating func readItem() throws -> HNSWWALInsertItem {
        let id = try readString()
        let level = try readI32()
        let vector = try readVector()
        let metadata = try readOptionalMetadata()
        let invFlag = try readU8()
        let inv: Float? = (invFlag == 1) ? try readFloat() : nil
        return HNSWWALInsertItem(id: id, level: level, vector: vector, metadata: metadata, knownInvNorm: inv)
    }
}

// MARK: - Errors

internal enum HNSWWALError: Error, CustomStringConvertible {
    case openFailed(String, Int32)
    case writeFailed(String, Int32)
    case readFailed(String, Int32)
    case fsyncFailed(Int32)
    case truncateFailed(Int32)
    case truncatedRecord
    case unknownRecordType(UInt32)
    case crcMismatch
    case encodingFailed(String)

    var description: String {
        switch self {
        case .openFailed(let path, let err): return "HNSWWAL open failed for '\(path)' (errno=\(err))"
        case .writeFailed(let path, let err): return "HNSWWAL write failed for '\(path)' (errno=\(err))"
        case .readFailed(let path, let err): return "HNSWWAL read failed for '\(path)' (errno=\(err))"
        case .fsyncFailed(let err): return "HNSWWAL fsync failed (errno=\(err))"
        case .truncateFailed(let err): return "HNSWWAL ftruncate failed (errno=\(err))"
        case .truncatedRecord: return "HNSWWAL record body truncated"
        case .unknownRecordType(let t): return "HNSWWAL unknown record type \(t)"
        case .crcMismatch: return "HNSWWAL frame CRC mismatch"
        case .encodingFailed(let msg): return "HNSWWAL encoding failed: \(msg)"
        }
    }
}

// MARK: - Record <-> body codec

extension HNSWWALRecord {
    fileprivate func encodeBody() throws -> [UInt8] {
        var enc = HNSWWALEncoder()
        switch self {
        case .insert(let item):
            try enc.writeItem(item)
        case .remove(let id):
            enc.writeString(id)
        case .update(let id, let vector, let metadata):
            enc.writeString(id)
            if let v = vector {
                enc.writeU8(1)
                enc.writeVector(v)
            } else {
                enc.writeU8(0)
            }
            if let m = metadata {
                enc.writeU8(1)
                let json = try JSONEncoder().encode(m)
                enc.writeU32(UInt32(json.count))
                enc.bytes.append(contentsOf: json)
            } else {
                enc.writeU8(0)
            }
        case .clear:
            break
        case .batchInsert(let items):
            enc.writeU32(UInt32(items.count))
            for item in items { try enc.writeItem(item) }
        }
        return enc.bytes
    }

    fileprivate static func decode(type: HNSWWALRecordType, body: [UInt8]) throws -> HNSWWALRecord {
        var dec = HNSWWALDecoder(bytes: body)
        switch type {
        case .insert:
            return .insert(try dec.readItem())
        case .remove:
            return .remove(id: try dec.readString())
        case .update:
            let id = try dec.readString()
            let hasVec = try dec.readU8()
            let vector: [Float]? = (hasVec == 1) ? try dec.readVector() : nil
            let hasMeta = try dec.readU8()
            var metadata: [String: String]? = nil
            if hasMeta == 1 {
                let n = Int(try dec.readU32())
                guard dec.offset + n <= body.count else { throw HNSWWALError.truncatedRecord }
                let slice = Data(body[dec.offset..<(dec.offset + n)])
                dec.offset += n
                metadata = try JSONDecoder().decode([String: String].self, from: slice)
            }
            return .update(id: id, vector: vector, metadata: metadata)
        case .clear:
            return .clear
        case .batchInsert:
            let n = Int(try dec.readU32())
            var items: [HNSWWALInsertItem] = []
            items.reserveCapacity(n)
            for _ in 0..<n { items.append(try dec.readItem()) }
            return .batchInsert(items)
        }
    }
}

// MARK: - Framing

/// Serialize a record into a single frame (magic|type|len|body|commit_tag|crc32).
internal func hnswWalEncodeFrame(_ record: HNSWWALRecord) throws -> [UInt8] {
    let body = try record.encodeBody()
    // Build the CRC-covered payload: type||len||body||commit_tag
    var covered = [UInt8]()
    covered.reserveCapacity(4 + 4 + body.count + 4)
    let type = record.recordType.rawValue.littleEndian
    withUnsafeBytes(of: type) { covered.append(contentsOf: $0) }
    let bodyLen = UInt32(body.count).littleEndian
    withUnsafeBytes(of: bodyLen) { covered.append(contentsOf: $0) }
    covered.append(contentsOf: body)
    let commit = HNSWWAL_COMMIT_TAG.littleEndian
    withUnsafeBytes(of: commit) { covered.append(contentsOf: $0) }

    let crc = covered.withUnsafeBufferPointer { bp in
        CRC32.hash(UnsafeRawPointer(bp.baseAddress!), covered.count)
    }

    var frame = [UInt8]()
    frame.reserveCapacity(4 + covered.count + 4)
    let magic = HNSWWAL_MAGIC.littleEndian
    withUnsafeBytes(of: magic) { frame.append(contentsOf: $0) }
    frame.append(contentsOf: covered)
    let crcLE = crc.littleEndian
    withUnsafeBytes(of: crcLE) { frame.append(contentsOf: $0) }
    return frame
}

// MARK: - Writer

/// Append-only WAL writer. Not an actor — expected to be owned by HNSWIndex
/// (an actor) so call sites are already serialized on the actor's executor.
internal final class HNSWWALWriter {
    let path: String
    private var fd: Int32 = -1

    init(path: String) throws {
        self.path = path
        let dir = (path as NSString).deletingLastPathComponent
        if !dir.isEmpty {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
        let flags: Int32 = O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC
        self.fd = Darwin.open(path, flags, 0o600)
        if fd < 0 {
            throw HNSWWALError.openFailed(path, errno)
        }
    }

    deinit {
        if fd >= 0 { _ = Darwin.close(fd) }
    }

    func append(_ record: HNSWWALRecord, fsyncAfter: Bool = true) throws {
        let frame = try hnswWalEncodeFrame(record)
        try frame.withUnsafeBufferPointer { bp in
            var remaining = bp.count
            var cursor = bp.baseAddress!
            while remaining > 0 {
                let n = Darwin.write(fd, cursor, remaining)
                if n < 0 {
                    if errno == EINTR { continue }
                    throw HNSWWALError.writeFailed(path, errno)
                }
                if n == 0 { throw HNSWWALError.writeFailed(path, errno) }
                remaining -= n
                cursor = cursor.advanced(by: n)
            }
        }
        if fsyncAfter {
            if Darwin.fsync(fd) != 0 { throw HNSWWALError.fsyncFailed(errno) }
        }
    }

    func flush() throws {
        if Darwin.fsync(fd) != 0 { throw HNSWWALError.fsyncFailed(errno) }
    }

    func truncate() throws {
        if Darwin.ftruncate(fd, 0) != 0 { throw HNSWWALError.truncateFailed(errno) }
        // Rewind the append offset to 0 as well; O_APPEND means subsequent
        // writes go to end-of-file regardless, but lseek keeps fstat sane.
        _ = Darwin.lseek(fd, 0, SEEK_SET)
        if Darwin.fsync(fd) != 0 { throw HNSWWALError.fsyncFailed(errno) }
    }

    func close() {
        if fd >= 0 {
            _ = Darwin.close(fd)
            fd = -1
        }
    }
}

// MARK: - Reader / replay

internal struct HNSWWALReader {
    /// Reads every valid frame in `path` and yields decoded records in order.
    /// On any torn / invalid frame, stops and returns the records read so far.
    /// Callers that need to distinguish "clean EOF" from "torn truncation" can
    /// inspect `didTruncate` in the returned tuple.
    static func replay(path: String) throws -> (records: [HNSWWALRecord], didTruncate: Bool) {
        guard FileManager.default.fileExists(atPath: path) else {
            return ([], false)
        }
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            throw HNSWWALError.readFailed(path, (error as NSError).code == 0 ? -1 : Int32((error as NSError).code))
        }
        let bytes = [UInt8](data)
        var records: [HNSWWALRecord] = []
        var cursor = 0
        var didTruncate = false

        while cursor < bytes.count {
            // Need at least magic(4) + type(4) + len(4) + commit(4) + crc(4) = 20 bytes
            if cursor + 20 > bytes.count {
                didTruncate = true
                break
            }
            let magic = loadU32LE(bytes, cursor)
            if magic != HNSWWAL_MAGIC {
                didTruncate = true
                break
            }
            let typeRaw = loadU32LE(bytes, cursor + 4)
            let bodyLen = Int(loadU32LE(bytes, cursor + 8))
            let frameEnd = cursor + 12 + bodyLen + 4 + 4  // body + commit + crc
            if frameEnd > bytes.count {
                didTruncate = true
                break
            }
            let commit = loadU32LE(bytes, cursor + 12 + bodyLen)
            if commit != HNSWWAL_COMMIT_TAG {
                didTruncate = true
                break
            }
            let crcStored = loadU32LE(bytes, cursor + 12 + bodyLen + 4)

            // Compute CRC over type||len||body||commit
            let coveredStart = cursor + 4
            let coveredLen = 4 + 4 + bodyLen + 4
            let crcCalc: UInt32 = bytes.withUnsafeBufferPointer { bp in
                CRC32.hash(UnsafeRawPointer(bp.baseAddress!.advanced(by: coveredStart)), coveredLen)
            }
            if crcCalc != crcStored {
                didTruncate = true
                break
            }

            guard let rtype = HNSWWALRecordType(rawValue: typeRaw) else {
                // Unknown record type — treat as end-of-log to remain forward-compatible.
                didTruncate = true
                break
            }

            let body = Array(bytes[(cursor + 12)..<(cursor + 12 + bodyLen)])
            let record: HNSWWALRecord
            do {
                record = try HNSWWALRecord.decode(type: rtype, body: body)
            } catch {
                // Malformed body — stop replay.
                didTruncate = true
                break
            }
            records.append(record)
            cursor = frameEnd
        }
        return (records, didTruncate)
    }

    @inline(__always) private static func loadU32LE(_ bytes: [UInt8], _ off: Int) -> UInt32 {
        var le: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &le) { dst in
            bytes.withUnsafeBufferPointer { src in
                memcpy(dst.baseAddress!, src.baseAddress!.advanced(by: off), 4)
            }
        }
        return UInt32(littleEndian: le)
    }
}
