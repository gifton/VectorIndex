//
//  VIndexMmap.swift
//
//  S1 — Serialization & Mmap Layout (Swift)
//
//  Implements:
//    - Header (256 B) + TOC parsing with CRC32
//    - Section table + typed accessors
//    - List descriptors and zero-copy pointers to IDs/Codes/Vecs
//    - Endianness handling for header/TOC/descriptors
//    - mmap open/close with alignment checks
//    - Durable append protocol: WAL (write-ahead log), begin/commit, replay
//    - msync + release/acquire ordering for lock-free readers
//

import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import CAtomicsShim

// Lightweight debug logger (enabled in DEBUG builds only)
#if DEBUG
@inline(__always) private func debugLog(_ message: String, file: StaticString = #file, line: UInt = #line) {
    print("[VIndexMmap] \(message) @\(file):\(line)")
}
#else
@inline(__always) private func debugLog(_ message: String, file: StaticString = #file, line: UInt = #line) {}
#endif
@inline(__always) private func alignUp(_ x: UInt64, _ a: UInt64) -> UInt64 {
    let m = a &- 1
    return (x &+ m) & ~m
}

public enum Endian: UInt8 { case little = 1, big = 2 }

@inline(__always) private func hostIsLittleEndian() -> Bool {
    var x: UInt16 = 0x0102
    return withUnsafeBytes(of: &x) { $0[0] == 0x02 }
}

@inline(__always) private func toHost<T: FixedWidthInteger>(_ v: T, fileEndian: Endian) -> T {
    ((fileEndian == .little) == hostIsLittleEndian()) ? v : v.byteSwapped
}

internal struct CRC32 {
    private static let table: [UInt32] = {
        (0..<256).map { i -> UInt32 in
            var c = UInt32(i)
            for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
            return c
        }
    }()

    /// Resumable CRC32 update: feeds `data`/`len` into a running `state` so callers can
    /// compute one CRC32 over several non-contiguous byte ranges (e.g. "all of a struct
    /// except one 4-byte field") without first copying them into one contiguous buffer.
    /// Start a fresh computation with `state = 0xFFFF_FFFF`; call `finalize(_:)` on the
    /// last state returned to get the standard CRC32 value.
    @inline(__always) static func update(_ state: UInt32, _ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
        var c = state
        let p = data.bindMemory(to: UInt8.self, capacity: len)
        for i in 0..<len {
            c = CRC32.table[Int((c ^ UInt32(p[i])) & 0xFF)] ^ (c >> 8)
        }
        return c
    }

    /// Applies CRC32's final XOR to a running `update(_:_:_:)` state.
    @inline(__always) static func finalize(_ state: UInt32) -> UInt32 { state ^ 0xFFFF_FFFF }

    @inline(__always) static func hash(_ data: UnsafeRawPointer, _ len: Int) -> UInt32 {
        finalize(update(0xFFFF_FFFF, data, len))
    }
}

// magic: "VINDEX\0\0" (LE)
private let VINDEX_MAGIC = UInt64(0x00585845444E4956)

internal enum SectionType: UInt32, Sendable {
    case centroids = 1, codebooks = 2, centroidNorms = 3, listsDesc = 4
    case ids = 5, codes = 6, vecs = 7, normsInv = 8, normsSq = 9
    case idMap = 10, tombstones = 11, telemetry = 12, freeList = 13, walAnchor = 14
}

internal struct ListDesc {
    var format: UInt8
    var group_g: UInt8
    var id_bits: UInt8
    var reserved0: UInt8
    var length: UInt32
    var capacity: UInt32
    var ids_offset: UInt64
    var codes_offset: UInt64
    var vecs_offset: UInt64
    var ids_stride: UInt32
    var codes_stride: UInt32
    var vecs_stride: UInt32
    var reserved1: UInt32

    @inline(__always) func lengthHost(_ e: Endian) -> Int { Int(toHost(length, fileEndian: e)) }
    @inline(__always) func capacityHost(_ e: Endian) -> Int { Int(toHost(capacity, fileEndian: e)) }
    @inline(__always) func idsOffsetHost(_ e: Endian) -> UInt64 { toHost(ids_offset, fileEndian: e) }
    @inline(__always) func codesOffsetHost(_ e: Endian) -> UInt64 { toHost(codes_offset, fileEndian: e) }
    @inline(__always) func vecsOffsetHost(_ e: Endian) -> UInt64 { toHost(vecs_offset, fileEndian: e) }
    @inline(__always) func idsStrideHost(_ e: Endian) -> Int { Int(toHost(ids_stride, fileEndian: e)) }
    @inline(__always) func codesStrideHost(_ e: Endian) -> Int { Int(toHost(codes_stride, fileEndian: e)) }
    @inline(__always) func vecsStrideHost(_ e: Endian) -> Int { Int(toHost(vecs_stride, fileEndian: e)) }
}

internal struct TOCEntry {
    var type: UInt32
    var offset: UInt64
    var size: UInt64
    var align: UInt32
    var flags: UInt32
    var crc32: UInt32
    var reserved: UInt32
    @inline(__always) func typeHost(_ e: Endian) -> SectionType? { SectionType(rawValue: toHost(type, fileEndian: e)) }
    @inline(__always) func offsetHost(_ e: Endian) -> UInt64 { toHost(offset, fileEndian: e) }
    @inline(__always) func sizeHost(_ e: Endian) -> UInt64 { toHost(size, fileEndian: e) }
    @inline(__always) func alignHost(_ e: Endian) -> UInt32 { toHost(align, fileEndian: e) }
    @inline(__always) func flagsHost(_ e: Endian) -> UInt32 { toHost(flags, fileEndian: e) }
    @inline(__always) func crcHost(_ e: Endian) -> UInt32 { toHost(crc32, fileEndian: e) }
}

internal struct VIndexHeader {
    var magic: UInt64
    var version_major: UInt16
    var version_minor: UInt16
    var endianness: UInt8
    var arch: UInt8
    var flags: UInt32
    var d: UInt32
    var m: UInt16
    var ks: UInt16
    var kc: UInt32
    var id_bits: UInt8
    var code_group_g: UInt8
    var reservedA: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
    var N_total: UInt64
    var generation: UInt64
    var toc_offset: UInt64
    var toc_entries: UInt32
    var header_crc32: UInt32
    var reservedRest: (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64)

    func fileEndian() -> Endian { Endian(rawValue: endianness) ?? .little }
    func magicOK(_ e: Endian) -> Bool { toHost(magic, fileEndian: e) == VINDEX_MAGIC }
}

@inline(__always) private func computeHeaderCRC(_ raw: UnsafeRawPointer) -> UInt32 {
    // Two-region hash directly against the mapped header bytes: no 256-byte copy, and no
    // write access needed (the mapping may be PROT_READ-only when opts.readOnly is set, so
    // we must never mutate it). Must stay byte-for-byte equivalent to "zero the CRC field,
    // then CRC32 all 256 bytes" (what the builder computes at write time in
    // VIndexContainerBuilder.swift) — the 4 zeroed bytes still have to be fed into the
    // running CRC, just from a local zero value instead of the live memory.
    let crcOffset = MemoryLayout<VIndexHeader>.offset(of: \.header_crc32)! // 68
    let crcSize = 4
    var state = CRC32.update(0xFFFF_FFFF, raw, crcOffset)
    var zero: UInt32 = 0
    state = withUnsafeBytes(of: &zero) { CRC32.update(state, $0.baseAddress!, crcSize) }
    let tailOffset = crcOffset + crcSize
    state = CRC32.update(state, raw.advanced(by: tailOffset), 256 - tailOffset)
    return CRC32.finalize(state)
}

internal struct MmapOpts {
    public var readOnly: Bool = true
    public var verifyCRCs: Bool = true
    public var adviseSequential: Bool = true
    public var adviseWillNeed: Bool = false
    public init() {}
}

internal struct AppendReservation {
    public let listID: Int
    public let oldLen: Int
    public let addLen: Int
    public let idsFileOffset: UInt64
    public let codesFileOffset: UInt64
    public let vecsFileOffset: UInt64
    let idsStride: Int
    let codesStride: Int
    let vecsStride: Int
}

// VIndexError removed - migrated to VectorIndexError
// All throw sites now use ErrorBuilder with appropriate IndexErrorKind

internal final class IndexMmap {
    public let path: String
    public let fd: Int32
    public private(set) var fileSize: UInt64
    public let opts: MmapOpts

    private var base: UnsafeMutableRawPointer
    private let prot: Int32
    private let mapFlags: Int32

    private let header: VIndexHeader
    private let fileEndian: Endian
    private var tocCount: Int
    
    private(set) public var kc: Int = 0
    private(set) public var d: Int = 0
    private(set) public var m: Int = 0
    private(set) public var ks: Int = 0
    private(set) public var N_total: Int = 0
    private(set) public var codeGroupG: Int = 0
    private(set) public var idBits: Int = 64

    /// Total bytes hashed by `updateSectionCRC`/`recomputeAllSectionCRCs` over this handle's
    /// lifetime. Instrumentation for the P4 deferred-CRC design: per-commit full-section CRC
    /// recompute was the O(N^2) driver for durable ingestion (`mmap_append_commit` used to call
    /// `updateSectionCRC` 3-4x per commit, each hashing the *entire* section). Commits must add
    /// zero to this counter; only `flush()`/`close()`/unclean-reopen WAL repair should.
    internal private(set) var crcBytesHashed: Int = 0

    /// Number of `msync(2)` syscalls issued by `msyncPageAligned` over this handle's lifetime
    /// (P5). Instrumentation for the ranged-flush fix: before P5, `msyncPageAligned` ignored its
    /// `ptr`/`length` parameters and always flushed the entire mapping, so every one of the
    /// (post-P4) 3-4 per-commit call sites paid for an `fileSize`-sized `msync` regardless of how
    /// few bytes actually changed. Now each call flushes only the page-aligned range that was
    /// actually touched.
    internal private(set) var msyncCallCount: Int = 0

    /// Total bytes covered by all `msync(2)` calls issued by `msyncPageAligned` over this
    /// handle's lifetime (P5) -- the page-aligned, mapping-clamped range passed to each syscall,
    /// summed. Paired with `msyncCallCount` as the deterministic gate for "ranged flush, not
    /// whole-mapping flush": a tiny commit must add at most a few pages per call, not `fileSize`.
    /// Incremented only after the underlying `msync(2)` call succeeds (fix round 1), so this
    /// counter -- like `msyncCallCount` -- means "successfully flushed", not merely "attempted".
    internal private(set) var msyncBytesFlushed: Int = 0

    /// Number of `msync(2)` calls issued by `msyncPageAligned` that returned a nonzero result
    /// (fix round 1). `msync` failing (EIO, ENOMEM, disk full, ...) means the page range was NOT
    /// durably written back; every throwing call site surfaces that as a thrown error instead of
    /// silently continuing, so in practice this counter stays 0 except on paths that deliberately
    /// swallow the throw (e.g. `deinit`'s `try? close()`) -- it exists so even those paths leave
    /// an observable trace that a flush failed, instead of the counters simply looking identical
    /// to a fully successful run.
    internal private(set) var msyncFailureCount: Int = 0

    /// Set once resources are released (via `close()` or the test-only `_abandonWithoutClose()`)
    /// so a subsequent `deinit`-triggered `close()` call is a safe no-op instead of a double
    /// `Darwin.close(fd)`/`munmap` (the `fd` value can be reused by an unrelated open by then).
    private var released = false

    /// Set to `true` only at the very end of a fully-successful `indexInit()` (fix round 1, C1).
    /// If `indexInit()` throws partway through (corrupt-CRC, misaligned section, unknown
    /// section type, ...), `open()`'s local `idx` reference count drops to zero and `deinit`
    /// runs `try? close()` on the half-initialized instance. Without this gate, a writable
    /// `close()` would call `flush()` → `recomputeAllSectionCRCs()`, which recomputes CRCs from
    /// whatever bytes are currently on disk — including the *corrupt* bytes that just failed
    /// verification — and persists them, self-certifying the corruption and permanently
    /// destroying detectability on every future open. `close()` only calls `flush()` when this
    /// is `true`, i.e. only for handles that actually completed `indexInit()`.
    private var initialized = false

    /// `false` exactly while `base` is dangling: `ensureFileCapacity`'s remap clears this right
    /// before `munmap`-ing the old mapping, and only sets it back once the replacement `mmap()`
    /// call succeeds (fix round 1, I3). If that `mmap()` fails, `ensureFileCapacity` throws with
    /// this still `false`, and `base` stays dangling for the rest of this handle's life (the
    /// growth path itself is out of scope for that repair — see the task-3 brief). `flush()` and
    /// `close()`'s own msync/munmap both check this before touching `base`, so a later `close()`
    /// (including the one `deinit` runs) degrades to releasing just the file descriptors instead
    /// of dereferencing/rewriting through a pointer into unmapped (possibly since-reused) address
    /// space.
    private var mappingValid = true

    private var secCentroids: UnsafePointer<Float>?
    private var secCodebooks: UnsafePointer<Float>?
    private var secCentroidNorms: UnsafePointer<Float>?
    private var listsDescBase: UnsafeMutableRawPointer?
    private var secIDs: UnsafeMutableRawPointer?
    private var secCodes: UnsafeMutableRawPointer?
    private var secVecs: UnsafeMutableRawPointer?
    private var secNormsInv: UnsafeRawPointer?
    private var secNormsSq: UnsafeRawPointer?
    private var secIDMap: UnsafeRawPointer?
    private var secTombstones: UnsafeRawPointer?

    private var tocByType: [SectionType: HostTOCEntry] = [:]
    private var tailIDs: UInt64 = 0
    private var tailCodes: UInt64 = 0
    private var tailVecs: UInt64 = 0

    private var walFD: Int32 = -1
    private var walPath: String

    /// Reusable scratch buffer for staging WAL record bytes before CRC32 hashing. Sized to
    /// the larger of the two records' CRC'd prefixes (WalAppend's 40 bytes: tag..vecsOff).
    /// Avoids a fresh heap allocation on every append/commit call.
    private var walScratch = [UInt8](repeating: 0, count: 40)

    public static func open(path: String, opts: MmapOpts = .init()) throws -> IndexMmap {
        let flags = opts.readOnly ? O_RDONLY : O_RDWR
        let fd = Darwin.open(path, flags | O_CLOEXEC)
        guard fd >= 0 else {
            throw ErrorBuilder(.fileIOError, operation: "vindex_open")
                .message("Failed to open index file")
                .info("path", path)
                .info("errno", "\(errno)")
                .build()
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            let err = errno
            Darwin.close(fd)
            throw ErrorBuilder(.fileIOError, operation: "vindex_fstat")
                .message("Failed to stat index file")
                .info("path", path)
                .info("errno", "\(err)")
                .build()
        }
        let fileSize = UInt64(st.st_size)
        guard fileSize >= 4096 else {
            Darwin.close(fd)
            throw ErrorBuilder(.corruptedData, operation: "vindex_open")
                .message("Index file too small or invalid header")
                .info("file_size", "\(fileSize)")
                .info("min_size", "4096")
                .build()
        }
        let prot: Int32 = opts.readOnly ? PROT_READ : (PROT_READ | PROT_WRITE)
        let mapFlags: Int32 = MAP_SHARED
        let base = mmap(nil, Int(fileSize), prot, mapFlags, fd, 0)
        guard base != MAP_FAILED else {
            let err = errno
            Darwin.close(fd)
            throw ErrorBuilder(.mmapError, operation: "vindex_mmap")
                .message("Failed to mmap index file")
                .info("path", path)
                .info("size", "\(fileSize)")
                .info("errno", "\(err)")
                .build()
        }

        let hdrPtr = base!.bindMemory(to: VIndexHeader.self, capacity: 1)
        let hdr = hdrPtr.pointee
        let fileEndian = hdr.fileEndian()
        guard fileEndian == .little || fileEndian == .big else {
            munmap(base, Int(fileSize))
            Darwin.close(fd)
            throw ErrorBuilder(.endiannessMismatch, operation: "vindex_open")
                .message("Unsupported or invalid endianness in index file")
                .info("endian_byte", "\(hdr.endianness)")
                .build()
        }
        guard hdr.magicOK(fileEndian) else {
            munmap(base, Int(fileSize))
            Darwin.close(fd)
            throw ErrorBuilder(.corruptedData, operation: "vindex_open")
                .message("Invalid magic number in index header")
                .build()
        }
        // Version policy: require major == 1 for current reader; minor is backward-compatible
        let verMajor = Int(toHost(hdr.version_major, fileEndian: fileEndian))
        // NOTE(Phase-2 B13): verMinor is read and reported in error diagnostics only; it does
        // not gate any behavior under the current single-format-version policy. Left as-is
        // intentionally — not a bug, not wired up. See PHASE4-ROUTING if minor-version gating
        // is ever introduced.
        let verMinor = Int(toHost(hdr.version_minor, fileEndian: fileEndian))
        guard verMajor == 1 else {
            munmap(base, Int(fileSize))
            Darwin.close(fd)
            throw ErrorBuilder(.versionMismatch, operation: "vindex_open")
                .message("Unsupported index file version")
                .info("version_major", "\(verMajor)")
                .info("version_minor", "\(verMinor)")
                .build()
        }
        if opts.verifyCRCs {
            let calc = computeHeaderCRC(UnsafeRawPointer(hdrPtr))
            let stored = toHost(hdr.header_crc32, fileEndian: fileEndian)
            guard calc == stored else {
                munmap(base, Int(fileSize))
                Darwin.close(fd)
                throw ErrorBuilder(.corruptedData, operation: "vindex_open")
                    .message("Header CRC mismatch")
                    .info("expected", "\(stored)")
                    .info("actual", "\(calc)")
                    .build()
            }
        }
        let tocOffset = toHost(hdr.toc_offset, fileEndian: fileEndian)
        let tocEntries = Int(toHost(hdr.toc_entries, fileEndian: fileEndian))
        let tocPtr = UnsafeRawPointer(base!).advanced(by: Int(tocOffset)).assumingMemoryBound(to: TOCEntry.self)

        let idx = IndexMmap(path: path, fd: fd, fileSize: fileSize, opts: opts, base: base!, prot: prot, mapFlags: mapFlags, header: hdr, fileEndian: fileEndian, toc: tocPtr, tocCount: tocEntries)
        try idx.indexInit()
        debugLog("open ok: fileSize=\(fileSize) kc=\(idx.kc) d=\(idx.d) m=\(idx.m) idBits=\(idx.idBits)")
        return idx
    }

    private init(path: String, fd: Int32, fileSize: UInt64, opts: MmapOpts, base: UnsafeMutableRawPointer, prot: Int32, mapFlags: Int32, header: VIndexHeader, fileEndian: Endian, toc: UnsafePointer<TOCEntry>, tocCount: Int) {
        self.path = path
        self.fd = fd
        self.fileSize = fileSize
        self.opts = opts
        self.base = base
        self.prot = prot
        self.mapFlags = mapFlags
        self.header = header
        self.fileEndian = fileEndian
        self.tocCount = tocCount
        self.kc = Int(toHost(header.kc, fileEndian: fileEndian))
        self.d  = Int(toHost(header.d, fileEndian: fileEndian))
        self.m  = Int(toHost(header.m, fileEndian: fileEndian))
        self.ks = Int(toHost(header.ks, fileEndian: fileEndian))
        self.N_total = Int(toHost(header.N_total, fileEndian: fileEndian))
        self.codeGroupG = Int(header.code_group_g)
        self.idBits = Int(header.id_bits)
        self.walPath = path + ".wal"
    }

    deinit { try? close() }

    // Expose file endianness to callers that need to decode descriptor fields
    public var fileEndianness: Endian { fileEndian }

    public func close() throws {
        guard !released else { return }
        released = true
        // Clean-shutdown path (P4): recompute+persist every section CRC and truncate the WAL
        // to 0 bytes *before* unmapping. An empty WAL is this format's clean/unclean marker
        // (see indexInit's walDirty branch) — only writable handles hold a WAL fd, so this is
        // a no-op for read-only handles (flush() itself also guards on opts.readOnly).
        // `initialized` (C1) additionally gates this: a handle whose indexInit() never finished
        // must not self-certify whatever bytes are currently on disk. flush() independently
        // re-checks `mappingValid` (I3) before touching `base`.
        //
        // flush()'s error (if any) is captured rather than propagated immediately: `released`
        // is already latched above, so an early `throw` here would skip the munmap/fd cleanup
        // below and — because `released` blocks any later retry, including the one `deinit`
        // would otherwise attempt — leak the mapping and both file descriptors permanently.
        // Resources are always released; a flush failure is still surfaced to the caller, just
        // after cleanup has run.
        var flushError: Error?
        if !opts.readOnly && initialized {
            do { try flush() } catch { flushError = error }
        }
        if walFD >= 0 { _ = Darwin.close(walFD); walFD = -1 }
        // I3: `base` is dangling whenever a growth-triggered remap's `mmap()` call failed after
        // its `munmap()` of the old mapping already ran (see ensureFileCapacity). Touching `base`
        // in that state — even via msync/munmap — is unsafe (undefined behavior; on some
        // platforms these can also fail loudly rather than silently). Skip both and still release
        // the file descriptors below.
        if mappingValid {
            _ = msync(base, Int(fileSize), MS_SYNC)
            _ = munmap(base, Int(fileSize))
        }
        _ = Darwin.close(fd)
        if let flushError { throw flushError }
    }

    /// Test-only: releases OS resources (WAL fd, mapping, file fd) WITHOUT running `flush()` —
    /// no section-CRC recompute, no WAL truncate. Simulates exactly what a process crash leaves
    /// on disk (payload bytes and the ListsDesc length bump already durable per-commit via their
    /// own msyncs; section CRCs stale; WAL non-empty) so tests can exercise the unclean-reopen
    /// repair path without an actual crash. Marks the handle released so a later `deinit` does
    /// not attempt a second, corrupting close. Not part of the durability contract — production
    /// code must always call `close()`.
    internal func _abandonWithoutClose() {
        guard !released else { return }
        released = true
        if walFD >= 0 { _ = Darwin.close(walFD); walFD = -1 }
        if mappingValid { _ = munmap(base, Int(fileSize)) }
        _ = Darwin.close(fd)
    }

    /// Test-only (fix round 1, I3): marks the mapping invalid WITHOUT actually unmapping it, so
    /// tests can verify `close()`/`flush()` skip all `base`-touching work when this invariant is
    /// violated — exactly what `ensureFileCapacity` leaves behind after `munmap()` succeeds but
    /// the replacement `mmap()` fails — without needing to force a real remap failure (OS/
    /// environment-dependent, not reliably reproducible in a portable unit test). The real
    /// mapping underneath is untouched; if the caller wants a fully clean teardown after this,
    /// it must not rely on the (now `mappingValid`-gated) msync/munmap in `close()`.
    internal func _simulateDanglingMapping() {
        mappingValid = false
    }

    private func slice(_ e: HostTOCEntry) -> UnsafeMutableRawPointer {
        let off = Int(e.offset)
        return UnsafeMutableRawPointer(base).advanced(by: off)
    }

    private struct HostTOCEntry { var type: SectionType; var offset: UInt64; var size: UInt64; var align: UInt32; var flags: UInt32; var crc32: UInt32 }
    private func mapSection(_ ty: SectionType) -> HostTOCEntry? { tocByType[ty] }

    /// Flushes exactly the touched `[ptr, ptr+length)` range to disk (P5), not the whole mapping.
    /// `msync(2)` requires a page-aligned address and rejects a length that runs past the mapping,
    /// so the start is rounded down to a page boundary, the end rounded up, and the result clamped
    /// to `[base, base+fileSize)` before the syscall. Every existing call site already passes the
    /// actual touched `ptr`/`length` (ids/codes/vecs memcpy destinations, the 64-byte ListsDesc
    /// record, the 36-byte TOC entry, ...) -- this function previously discarded both parameters
    /// and flushed `fileSize` bytes unconditionally, making every commit's 3-4 msync calls pay for
    /// a whole-mapping flush regardless of how few bytes changed.
    ///
    /// Throws (fix round 1) when the underlying `msync(2)` call fails (EIO, ENOMEM, disk full,
    /// ...): every call site sits inside a `throws` function (verified by audit -- see the task-4
    /// report's fix-round-1 section), so a failed flush now surfaces as a thrown error instead of
    /// being silently discarded. This matters specifically for `mmap_append_commit`: its contract
    /// is "returning means the payload is durable," so a flush failure on the commit path MUST
    /// prevent that return, not just log/count it. `msyncCallCount`/`msyncBytesFlushed` are
    /// incremented only after the syscall succeeds, so they mean "successfully flushed";
    /// `msyncFailureCount` is incremented on failure, before the throw, so telemetry stays honest
    /// even on the few paths that deliberately swallow the error (`deinit`'s `try? close()`).
    @inline(__always) private func msyncPageAligned(_ ptr: UnsafeMutableRawPointer, _ length: Int) throws {
        guard length > 0 else { return }
        let pageSize = Int(getpagesize())
        let baseAddr = Int(bitPattern: base)
        let start = Int(bitPattern: ptr)
        // Round start down to a page boundary, end up, and clamp to the mapping.
        let alignedStart = max(start & ~(pageSize - 1), baseAddr)
        let alignedEnd = min((start + length + pageSize - 1) & ~(pageSize - 1),
                             baseAddr + Int(fileSize))
        let len = alignedEnd - alignedStart
        guard len > 0 else { return }
        let rc = msync(UnsafeMutableRawPointer(bitPattern: alignedStart)!, len, MS_SYNC)
        if rc != 0 {
            let err = errno
            msyncFailureCount &+= 1
            throw ErrorBuilder(.fileIOError, operation: "vindex_msync")
                .message("Failed to flush mapped pages to disk")
                .info("path", path)
                .info("bytes", "\(len)")
                .info("errno", "\(err)")
                .build()
        }
        msyncCallCount &+= 1
        msyncBytesFlushed &+= len
    }

    @inline(__always) private func writeLE32(_ p: UnsafeMutableRawPointer, _ v: UInt32) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { bytes in memcpy(p, bytes.baseAddress!, 4) }
    }
    @inline(__always) private func writeLE64(_ p: UnsafeMutableRawPointer, _ v: UInt64) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { bytes in memcpy(p, bytes.baseAddress!, 8) }
    }

    private func indexInit() throws {
        debugLog("indexInit: tocCount=\(tocCount) fileSize=\(fileSize)")
        // P4: the WAL file is the clean/unclean marker. A non-empty WAL means the last writer
        // did not reach a clean flush()/close() (crash, or process still holding the handle
        // elsewhere) — section CRCs are therefore untrustworthy until WAL replay + a recompute
        // repairs them (done below, after the WAL fd is opened). Computed once, up front, so
        // both the strict-verification gate below and the later repair branch agree.
        let walDirty = walIsNonEmpty()
        // Parse TOC as packed entries (36 bytes each)
        let DISK_TOC_ENTRY_SIZE = 36
        let tocRaw = UnsafeRawPointer(base).advanced(by: Int(toHost(header.toc_offset, fileEndian: fileEndian)))
        for i in 0..<tocCount {
            let te = tocRaw.advanced(by: i * DISK_TOC_ENTRY_SIZE)
            let tyRaw = readLE32(te.advanced(by: 0))
            guard let ty = SectionType(rawValue: tyRaw) else {
                throw ErrorBuilder(.invalidFormat, operation: "vindex_init")
                    .message("Unknown section type in TOC")
                    .info("toc_index", "\(i)")
                    .build()
            }
            // Packed: offset@+4, size@+12, align@+20, flags@+24, crc@+28
            let off = readLE64(te.advanced(by: 4))
            let sz  = readLE64(te.advanced(by: 12))
            let al  = UInt64(readLE32(te.advanced(by: 20)))
            let flags = readLE32(te.advanced(by: 24))
            let crc = readLE32(te.advanced(by: 28))
            tocByType[ty] = HostTOCEntry(type: ty, offset: off, size: sz, align: UInt32(al), flags: flags, crc32: crc)
            if al != 0 && (off % UInt64(al)) != 0 {
                throw ErrorBuilder(.corruptedData, operation: "vindex_init")
                    .message("Section misaligned in index file")
                    .info("section_type", "\(ty.rawValue)")
                    .info("offset", "\(off)")
                    .info("required_alignment", "\(al)")
                    .info("actual_alignment", "\(off % UInt64(al))")
                    .build()
            }
            if opts.verifyCRCs && sz > 0 && !walDirty {
                let p = UnsafeRawPointer(base).advanced(by: Int(off))
                let crc = CRC32.hash(p, Int(sz))
                let stored = readLE32(te.advanced(by: 28))
                if crc != stored {
                    debugLog("CRC mismatch for section=\(String(describing: ty)) stored=\(stored) calc=\(crc) readOnly=\(opts.readOnly)")
                    // If ListsDesc differs and we're writable, refresh CRC as a best-effort repair.
                    if !opts.readOnly && ty == .listsDesc {
                        _ = try? updateSectionCRC(.listsDesc)
                    } else {
                        throw ErrorBuilder(.corruptedData, operation: "vindex_init")
                            .message("Section CRC mismatch")
                            .info("section_type", "\(ty.rawValue)")
                            .info("expected", "\(stored)")
                            .info("actual", "\(crc)")
                            .build()
                    }
                }
            }
        }
        if let e = mapSection(.centroids) { secCentroids = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.codebooks) { secCodebooks = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.centroidNorms) { secCentroidNorms = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
        if let e = mapSection(.listsDesc) { listsDescBase = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.ids) { secIDs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.codes) { secCodes = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.vecs) { secVecs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
        if let e = mapSection(.normsInv) { secNormsInv = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
        if let e = mapSection(.normsSq) { secNormsSq = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
        if let e = mapSection(.idMap) { secIDMap = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
        if let e = mapSection(.tombstones) { secTombstones = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }

        if let descs = listsDescBase?.assumingMemoryBound(to: UInt8.self) {
            var idsMax: UInt64 = 0, codesMax: UInt64 = 0, vecsMax: UInt64 = 0
            for i in 0..<kc {
                let rec = descs.advanced(by: i * 64)
                let cap  = UInt64(readLE32(rec.advanced(by: 8)))
                let idsOff   = readLE64(rec.advanced(by: 16))
                let codesOff = readLE64(rec.advanced(by: 24))
                let vecsOff  = readLE64(rec.advanced(by: 32))
                let idsStride = UInt64(readLE32(rec.advanced(by: 40)))
                let codesStride = UInt64(readLE32(rec.advanced(by: 44)))
                let vecsStride = UInt64(readLE32(rec.advanced(by: 48)))
                idsMax   = max(idsMax, idsOff   &+ cap &* idsStride)
                codesMax = max(codesMax, codesOff &+ cap &* codesStride)
                vecsMax  = max(vecsMax, vecsOff  &+ cap &* vecsStride)
            }
            tailIDs = idsMax; tailCodes = codesMax; tailVecs = vecsMax
        }

        if opts.adviseSequential { _ = posix_madvise(base, Int(fileSize), POSIX_MADV_SEQUENTIAL) }
        if opts.adviseWillNeed { _ = posix_madvise(base, Int(fileSize), POSIX_MADV_WILLNEED) }

        if !opts.readOnly {
            walFD = Darwin.open(walPath, O_RDWR | O_CREAT | O_CLOEXEC, S_IRUSR | S_IWUSR)
            if walFD < 0 {
                throw ErrorBuilder(.fileIOError, operation: "vindex_wal_open")
                    .message("Failed to open WAL file")
                    .info("wal_path", walPath)
                    .info("errno", "\(errno)")
                    .build()
            }
            if walDirty {
                // Unclean-reopen repair (P4): strict section-CRC verification was skipped above
                // because it cannot be trusted yet. Replay the WAL (its append/commit records are
                // individually CRC'd — see mmap_wal_replay) to reconcile list lengths to the last
                // committed payload, then recompute+persist every section CRC from the
                // now-consistent bytes, and truncate the WAL back to empty (clean) so this repair
                // doesn't repeat on the next open.
                try mmap_wal_replay()
                try recomputeAllSectionCRCs()
                try msyncPageAligned(base, Int(fileSize))
                try truncateWAL()
            }
        }
        // C1: only reached if every step above succeeded. Gates close()'s flush() call so a
        // handle whose indexInit() threw (corrupt section CRC, misaligned section, ...) never
        // has its (corrupt) on-disk bytes "repaired" into self-certified CRCs by the deinit ->
        // close() path that runs when open()'s failed-init `idx` reference is released.
        initialized = true
    }

    public func mmapCentroids() -> (ptr: UnsafePointer<Float>, kc: Int, d: Int)? {
        guard let p = secCentroids else { return nil }
        return (p, kc, d)
    }
    public func mmapCodebooks() -> (ptr: UnsafePointer<Float>, m: Int, ks: Int, dsub: Int)? {
        guard let p = secCodebooks else { return nil }
        let dsub = d / max(m, 1)
        return (p, m, ks, dsub)
    }
    // Legacy API retained for compatibility (returns nil; use getListDescriptor instead)
    public func mmapLists() -> (ptr: UnsafeMutablePointer<ListDesc>, kc: Int)? {
        nil
    }
    public func idsBase() -> UnsafeMutableRawPointer? { secIDs }
    public func codesBase() -> UnsafeMutableRawPointer? { secCodes }
    public func vecsBase() -> UnsafeMutableRawPointer? { secVecs }

    // Return a raw pointer and size for a section if present
    public func sectionSlice(_ ty: SectionType) -> (ptr: UnsafeRawPointer, size: Int)? {
        guard let e = tocByType[ty] else { return nil }
        let size = Int(e.size)
        let p: UnsafeRawPointer
        switch ty {
        case .centroids: guard let s = secCentroids else { return nil }; p = UnsafeRawPointer(s)
        case .codebooks: guard let s = secCodebooks else { return nil }; p = UnsafeRawPointer(s)
        case .centroidNorms: guard let s = secCentroidNorms else { return nil }; p = UnsafeRawPointer(s)
        case .listsDesc:
            guard let s = listsDescBase else { return nil }
            p = UnsafeRawPointer(s)
        case .ids: guard let s = secIDs else { return nil }; p = UnsafeRawPointer(s)
        case .codes: guard let s = secCodes else { return nil }; p = UnsafeRawPointer(s)
        case .vecs: guard let s = secVecs else { return nil }; p = UnsafeRawPointer(s)
        case .normsInv: guard let s = secNormsInv else { return nil }; p = s
        case .normsSq: guard let s = secNormsSq else { return nil }; p = s
        case .idMap: guard let s = secIDMap else { return nil }; p = s
        case .tombstones: guard let s = secTombstones else { return nil }; p = s
        default:
            return nil
        }
        return (p, size)
    }

    // Read IDMap blob (serialized) if present
    public func readIDMapBlob() -> Data? {
        guard let (p, sz) = sectionSlice(.idMap), sz > 0 else { return nil }
        return Data(bytes: p, count: sz)
    }

    // Write IDMap blob into existing section (size must fit). Updates CRC in TOC.
    public func writeIDMapBlob(_ blob: Data) throws {
        guard !opts.readOnly else {
            throw ErrorBuilder(.fileIOError, operation: "vindex_idmap_write")
                .message("Cannot write to read-only index")
                .build()
        }
        guard let e = tocByType[.idMap] else {
            throw ErrorBuilder(.invalidFormat, operation: "vindex_idmap_write")
                .message("IDMap section not found in index")
                .build()
        }
        let maxSize = Int(e.size)
        guard blob.count <= maxSize else {
            throw ErrorBuilder(.capacityExceeded, operation: "vindex_idmap_write")
                .message("IDMap blob too large for allocated section")
                .info("blob_size", "\(blob.count)")
                .info("max_size", "\(maxSize)")
                .build()
        }
        guard let basePtr = secIDMap else {
            throw ErrorBuilder(.invalidFormat, operation: "vindex_idmap_write")
                .message("IDMap section pointer unavailable")
                .build()
        }
        blob.withUnsafeBytes { src in
            memcpy(UnsafeMutableRawPointer(mutating: basePtr), src.baseAddress!, blob.count)
            if maxSize > blob.count {
                memset(UnsafeMutableRawPointer(mutating: basePtr).advanced(by: blob.count), 0, maxSize - blob.count)
            }
        }
        try msyncPageAligned(UnsafeMutableRawPointer(mutating: basePtr), maxSize)
        // Update CRC in TOC entry in-place
        try updateSectionCRC(.idMap)
    }

    private func updateSectionCRC(_ ty: SectionType) throws {
        // Find packed TOC entry by type
        let tocOff = Int(toHost(header.toc_offset, fileEndian: fileEndian))
        let DISK_TOC_ENTRY_SIZE = 36
        var idxFound: Int?
        for i in 0..<tocCount {
            let te = UnsafeRawPointer(base).advanced(by: tocOff + i * DISK_TOC_ENTRY_SIZE)
            let tyRaw = readLE32(te.advanced(by: 0))
            if tyRaw == ty.rawValue { idxFound = i; break }
        }
        guard let i = idxFound else {
            throw ErrorBuilder(.invalidFormat, operation: "vindex_update_crc")
                .message("Section not found in TOC")
                .info("section_type", "\(ty.rawValue)")
                .build()
        }
        let entryPtr = UnsafeMutableRawPointer(base).advanced(by: tocOff + i * DISK_TOC_ENTRY_SIZE)
        let off = readLE64(UnsafeRawPointer(entryPtr).advanced(by: 4))
        let sz  = Int(readLE64(UnsafeRawPointer(entryPtr).advanced(by: 12)))
        let p = UnsafeRawPointer(base).advanced(by: Int(off))
        crcBytesHashed += sz
        let newCRC = CRC32.hash(p, sz)
        writeLE32(entryPtr.advanced(by: 28), newCRC)
        try msyncPageAligned(entryPtr, DISK_TOC_ENTRY_SIZE)
        // also refresh cache
        if var e = tocByType[ty] { e.crc32 = newCRC; tocByType[ty] = e }
    }

    /// Recomputes and persists every section's CRC (P4). O(total container bytes), called only
    /// from `flush()`/`close()` and from the unclean-reopen repair branch of `indexInit` — never
    /// from the per-commit hot path.
    private func recomputeAllSectionCRCs() throws {
        for ty in tocByType.keys { try updateSectionCRC(ty) }
    }

    /// `ftruncate(walFD, 0)` + reset the write offset back to the start + `fsync`. Guards on
    /// `walFD >= 0` so it is a no-op for read-only handles (which never open a WAL fd). Resetting
    /// the offset matters: `writeWalAppend`/`writeWalCommit` write via the fd's current position
    /// (no `O_APPEND`), so without the `lseek` back to 0 the next append after a flush would land
    /// at the old (pre-truncate) offset, leaving a sparse hole of zero bytes that still makes the
    /// file's `st_size` read as non-empty — silently defeating the "empty WAL ⇒ clean" marker.
    private func truncateWAL() throws {
        guard walFD >= 0 else { return }
        if ftruncate(walFD, 0) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "vindex_wal_truncate")
                .message("Failed to truncate WAL file")
                .info("wal_path", walPath)
                .info("errno", "\(errno)")
                .build()
        }
        if lseek(walFD, 0, SEEK_SET) == -1 {
            throw ErrorBuilder(.fileIOError, operation: "vindex_wal_truncate")
                .message("Failed to reset WAL write offset after truncate")
                .info("wal_path", walPath)
                .info("errno", "\(errno)")
                .build()
        }
        if fsync(walFD) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "vindex_wal_truncate")
                .message("Failed to fsync WAL file after truncate")
                .info("wal_path", walPath)
                .info("errno", "\(errno)")
                .build()
        }
    }

    /// True when the WAL file exists and is non-empty — this format's unclean-shutdown marker
    /// (P4). A missing WAL (never opened writable) or a zero-length WAL (freshly created, or
    /// truncated by a prior clean `flush()`/`close()`) both mean "clean". Uses `stat(2)` on the
    /// path rather than `walFD` so the check also works for read-only opens, which never open a
    /// WAL fd at all.
    private func walIsNonEmpty() -> Bool {
        var st = stat()
        guard stat(walPath, &st) == 0 else { return false }
        return st.st_size > 0
    }

    /// Recomputes+persists all section CRCs, syncs the mapping, then truncates the WAL to mark
    /// the container clean (P4). Internal class, so this is zero public-surface-area impact.
    /// No-op for read-only handles (nothing to write back) and — fix round 1, I3 — no-op whenever
    /// `mappingValid` is `false`, since `recomputeAllSectionCRCs()`/`msyncPageAligned` both
    /// dereference `base`, which is left dangling by a failed growth-triggered remap (see
    /// `ensureFileCapacity`). Called by `close()` before unmapping, and by `indexInit()`'s
    /// unclean-reopen repair branch after WAL replay.
    public func flush() throws {
        guard !opts.readOnly, mappingValid else { return }
        try recomputeAllSectionCRCs()
        try msyncPageAligned(base, Int(fileSize))
        try truncateWAL()
    }

    @inline(__always) public func snapshotListLength(listID: Int) -> Int {
        guard let base = listsDescBase?.assumingMemoryBound(to: UInt8.self), listID >= 0, listID < kc else { return 0 }
        let rec = UnsafeRawPointer(base.advanced(by: listID * 64))
        let len = readLE32(rec.advanced(by: 4))
        return Int(len)
    }

    // Public accessor for list descriptor offsets and metadata (packed LE)
    public func getListDescriptor(listID: Int) -> (length: Int, capacity: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64, idsStride: Int, codesStride: Int, vecsStride: Int)? {
        guard let base = listsDescBase?.assumingMemoryBound(to: UInt8.self), listID >= 0, listID < kc else { return nil }
        let rec = UnsafeRawPointer(base.advanced(by: listID * 64))
        let length = Int(readLE32(rec.advanced(by: 4)))
        let capacity = Int(readLE32(rec.advanced(by: 8)))
        let idsOff = readLE64(rec.advanced(by: 16))
        let codesOff = readLE64(rec.advanced(by: 24))
        let vecsOff  = readLE64(rec.advanced(by: 32))
        let idsStride = Int(readLE32(rec.advanced(by: 40)))
        let codesStride = Int(readLE32(rec.advanced(by: 44)))
        let vecsStride = Int(readLE32(rec.advanced(by: 48)))
        return (length, capacity, idsOff, codesOff, vecsOff, idsStride, codesStride, vecsStride)
    }

    private struct WalAppend { var tag: UInt32; var listID: UInt32; var oldLen: UInt32; var delta: UInt32; var idsOff: UInt64; var codesOff: UInt64; var vecsOff: UInt64; var crc32: UInt32 }
    private struct WalCommit { var tag: UInt32; var listID: UInt32; var newLen: UInt32; var crc32: UInt32 }
    private let WAL_APPEND_TAG: UInt32 = 0xAABBCCDD
    private let WAL_COMMIT_TAG: UInt32 = 0xDDCCBBAA
    /// Fix round 1, I2: written (4 raw bytes, no payload) as the very first on-disk mutation of
    /// `mmap_append_begin`'s growth branch, before `ensureFileCapacity`/the relocation memcpys
    /// touch IDs/Codes/Vecs. Growth mutates those sections directly (outside the commit/WAL
    /// protocol — `writeListDescOffsets` only refreshes ListsDesc's own CRC), so without this a
    /// crash between a growth event and its still-pending `mmap_append_commit` would leave an
    /// *empty* WAL (this format's clean marker) next to now-stale IDs/Codes CRCs, and the next
    /// open would strict-verify and throw a false-positive corruption error on an otherwise-fine
    /// container. Writing this sentinel first makes the WAL pessimistically non-empty for the
    /// whole duration of the growth+append+commit sequence, so an unclean reopen always routes
    /// through replay+recompute instead of strict verification. `mmap_wal_replay` recognizes and
    /// skips it (does not `break` — real records may still follow).
    private let WAL_GROWTH_SENTINEL_TAG: UInt32 = 0x53454E54 // ASCII "SENT", arbitrary/distinct from the tags above

    public func mmap_append_begin(listID: Int, addLen: Int) throws -> AppendReservation {
        guard !opts.readOnly, listID >= 0, listID < kc else {
            throw ErrorBuilder(.invalidRange, operation: "mmap_append_begin")
                .message("Invalid list ID or index is read-only")
                .info("list_id", "\(listID)")
                .info("valid_range", "0..<\(kc)")
                .info("read_only", "\(opts.readOnly)")
                .build()
        }
        guard let ldbase = listsDescBase?.assumingMemoryBound(to: UInt8.self) else {
            throw ErrorBuilder(.invalidFormat, operation: "mmap_append_begin")
                .message("ListsDesc base unavailable")
                .build()
        }
        let rec = ldbase.advanced(by: listID * 64)
        let oldLen = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 4))))
        let cap    = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 8))))
        let idsStride = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 40))))
        let codesStride = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 44))))
        let vecsStride = Int(readLE32(UnsafeRawPointer(rec.advanced(by: 48))))
        let need   = oldLen + addLen
        var currIDsOff = readLE64(UnsafeRawPointer(rec.advanced(by: 16)))
        var currCodesOff = readLE64(UnsafeRawPointer(rec.advanced(by: 24)))
        var currVecsOff  = readLE64(UnsafeRawPointer(rec.advanced(by: 32)))
        debugLog("append_begin list=\(listID) oldLen=\(oldLen) cap=\(cap) need=\(need) strides(ids=\(idsStride),codes=\(codesStride),vecs=\(vecsStride)) tails(ids=\(tailIDs),codes=\(tailCodes),vecs=\(tailVecs)) currOffsets(ids=\(currIDsOff),codes=\(currCodesOff),vecs=\(currVecsOff))")
        if addLen <= 0 { return AppendReservation(listID: listID, oldLen: oldLen, addLen: 0, idsFileOffset: currIDsOff, codesFileOffset: currCodesOff, vecsFileOffset: currVecsOff, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride) }

        if need > cap {
            // I2: pessimistically mark the WAL non-empty *before* any on-disk mutation below
            // (ensureFileCapacity's ftruncate/remap, the IDs/Codes/Vecs relocation memcpys, and
            // writeListDescOffsets, which only refreshes ListsDesc's own CRC). None of that is
            // covered by the append/commit WAL protocol below — it happens here, in begin, not
            // commit — so without this a crash before the matching mmap_append_commit call would
            // leave an empty WAL (this format's clean marker) next to stale IDs/Codes CRCs.
            try writeWalSentinel()
            guard mapSection(.ids) != nil else {
                throw ErrorBuilder(.capacityExceeded, operation: "mmap_append_begin")
                    .message("Cannot grow IDs section")
                    .info("list_id", "\(listID)")
                    .build()
            }
            guard mapSection(.codes) != nil else {
                throw ErrorBuilder(.capacityExceeded, operation: "mmap_append_begin")
                    .message("Cannot grow codes section")
                    .info("list_id", "\(listID)")
                    .build()
            }
            if vecsStride > 0 {
                guard mapSection(.vecs) != nil else {
                    throw ErrorBuilder(.capacityExceeded, operation: "mmap_append_begin")
                        .message("Cannot grow vecs section")
                        .info("list_id", "\(listID)")
                        .build()
                }
            }
            let newCap = max(need, max(cap * 2, 256))
            let newIDsOff = alignUp(tailIDs, 64)
            let newCodesOff = alignUp(tailCodes, 64)
            let newVecsOff = alignUp(tailVecs, 64)
            let idsBytes = UInt64(newCap * idsStride)
            let codesBytes = UInt64(newCap * codesStride)
            let vecsBytes = UInt64(newCap * vecsStride)
            debugLog("growth list=\(listID) newCap=\(newCap) newOff(ids=\(newIDsOff),codes=\(newCodesOff),vecs=\(newVecsOff)) bytes(ids=\(idsBytes),codes=\(codesBytes),vecs=\(vecsBytes))")
            // Compute tails relative to section base (offsets stored in descriptors are section-relative)
            try ensureFileCapacity(for: .ids, tail: newIDsOff &+ idsBytes)
            try ensureFileCapacity(for: .codes, tail: newCodesOff &+ codesBytes)
            if vecsStride > 0 { try ensureFileCapacity(for: .vecs, tail: newVecsOff &+ vecsBytes) }
            // After possible remap, refresh bases
            guard let idsBase2 = secIDs, let codesBase2 = secCodes else {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                    .message("Section base pointers unavailable after remap")
                    .build()
            }
            let oldIDs = idsBase2.advanced(by: Int(currIDsOff))
            let newIDs = idsBase2.advanced(by: Int(newIDsOff))
            if oldLen > 0 { memcpy(newIDs, oldIDs, oldLen * idsStride) }
            let oldCodes = codesBase2.advanced(by: Int(currCodesOff))
            let newCodes = codesBase2.advanced(by: Int(newCodesOff))
            if oldLen > 0 { memcpy(newCodes, oldCodes, oldLen * codesStride) }
            var newVecsOffUsed: UInt64 = 0
            if vecsStride > 0 {
                guard let vecsBase = secVecs else {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("Vecs section base pointer unavailable after remap")
                        .build()
                }
                let oldVecs = vecsBase.advanced(by: Int(currVecsOff))
                let newVecs = vecsBase.advanced(by: Int(newVecsOff))
                if oldLen > 0 { memcpy(newVecs, oldVecs, oldLen * vecsStride) }
                newVecsOffUsed = newVecsOff
                tailVecs = alignUp(newVecsOff &+ vecsBytes, 64)
            }
            // Sanity: ensure new offsets + capacity bytes fit inside the section sizes
            if let eIDs = mapSection(.ids) {
                let idsSecSize = eIDs.size
                let idsNeeded = (newIDsOff &+ idsBytes)
                if idsNeeded > idsSecSize {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("IDs offset/capacity exceed section size")
                        .info("offset", "\(newIDsOff)")
                        .info("needed", "\(idsNeeded)")
                        .info("section_size", "\(idsSecSize)")
                        .build()
                }
            }
            if let eCodes = mapSection(.codes) {
                let codesSecSize = eCodes.size
                let codesNeeded = (newCodesOff &+ codesBytes)
                if codesNeeded > codesSecSize {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("Codes offset/capacity exceed section size")
                        .info("offset", "\(newCodesOff)")
                        .info("needed", "\(codesNeeded)")
                        .info("section_size", "\(codesSecSize)")
                        .build()
                }
            }
            if vecsStride > 0, let eVecs = mapSection(.vecs) {
                let vecsSecSize = eVecs.size
                let vecsNeeded = (newVecsOff &+ vecsBytes)
                if vecsNeeded > vecsSecSize {
                    throw ErrorBuilder(.mmapError, operation: "mmap_append_begin")
                        .message("Vecs offset/capacity exceed section size")
                        .info("offset", "\(newVecsOff)")
                        .info("needed", "\(vecsNeeded)")
                        .info("section_size", "\(vecsSecSize)")
                        .build()
                }
            }

            try writeListDescOffsets(listID: listID, idsOff: newIDsOff, codesOff: newCodesOff, vecsOff: newVecsOffUsed, newCapacity: newCap, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride)
            tailIDs = alignUp(newIDsOff &+ idsBytes, 64); tailCodes = alignUp(newCodesOff &+ codesBytes, 64)
            currIDsOff = newIDsOff
            currCodesOff = newCodesOff
            currVecsOff = newVecsOffUsed
            // Read back descriptor packed
            let vIDs = readLE64(UnsafeRawPointer(rec.advanced(by: 16)))
            let vCodes = readLE64(UnsafeRawPointer(rec.advanced(by: 24)))
            let vVecs = readLE64(UnsafeRawPointer(rec.advanced(by: 32)))
            debugLog("descriptor_written list=\(listID) verifyOff(ids=\(vIDs),codes=\(vCodes),vecs=\(vVecs))")
        }
        // Use current (possibly updated) section-relative offsets
        let idsOffset = currIDsOff &+ UInt64(oldLen * idsStride)
        let codesOffset = currCodesOff &+ UInt64(oldLen * codesStride)
        let vecsOffset = currVecsOff &+ UInt64((vecsStride > 0 ? oldLen * vecsStride : 0))
        debugLog("reservation list=\(listID) oldLen=\(oldLen) addLen=\(addLen) resOff(ids=\(idsOffset),codes=\(codesOffset),vecs=\(vecsOffset))")
        return AppendReservation(listID: listID, oldLen: oldLen, addLen: addLen, idsFileOffset: idsOffset, codesFileOffset: codesOffset, vecsFileOffset: vecsOffset, idsStride: idsStride, codesStride: codesStride, vecsStride: vecsStride)
    }

    public func mmap_append_commit(_ res: AppendReservation, idsSrc: UnsafeRawPointer?, codesSrc: UnsafeRawPointer?, vecsSrc: UnsafeRawPointer?) throws {
        guard !opts.readOnly, res.addLen > 0 else { return }
        debugLog("commit_begin list=\(res.listID) oldLen=\(res.oldLen) addLen=\(res.addLen) resOff(ids=\(res.idsFileOffset),codes=\(res.codesFileOffset),vecs=\(res.vecsFileOffset)) strides(ids=\(res.idsStride),codes=\(res.codesStride),vecs=\(res.vecsStride))")
        try writeWalAppend(listID: res.listID, oldLen: res.oldLen, delta: res.addLen, idsOff: res.idsFileOffset, codesOff: res.codesFileOffset, vecsOff: res.vecsFileOffset)
        if let ids = idsSrc, let idsBase = secIDs {
            guard let e = tocByType[.ids] else {
                throw ErrorBuilder(.invalidFormat, operation: "mmap_append_commit").message("IDs section not found in TOC").build()
            }
            let secSize = e.size
            let bytes = UInt64(res.addLen &* res.idsStride)
            let relOff = res.idsFileOffset
            debugLog("commit_ids list=\(res.listID) relOff=\(relOff) bytes=\(bytes) secSize=\(secSize)")
            if relOff &+ bytes > secSize {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_commit")
                    .message("IDs write exceeds mapped section size")
                    .info("offset", "\(relOff)")
                    .info("bytes", "\(bytes)")
                    .info("limit", "\(secSize)")
                    .build()
            }
            let dst = idsBase.advanced(by: Int(relOff))
            memcpy(dst, ids, Int(bytes))
            try msyncPageAligned(dst, Int(bytes))
            // P4: IDs section CRC freshness is deferred to flush()/close() (or unclean-reopen
            // repair) instead of recomputed here. Recomputing the whole section's CRC on every
            // commit made N appends O(N^2) — the WAL append/commit records above already make
            // this write crash-recoverable; only checksum *freshness* moves later.
        }
        if let codes = codesSrc, let codesBase = secCodes {
            guard let e = tocByType[.codes] else {
                throw ErrorBuilder(.invalidFormat, operation: "mmap_append_commit").message("Codes section not found in TOC").build()
            }
            let secSize = e.size
            let bytes = UInt64(res.addLen &* res.codesStride)
            let relOff = res.codesFileOffset
            debugLog("commit_codes list=\(res.listID) relOff=\(relOff) bytes=\(bytes) secSize=\(secSize)")
            if relOff &+ bytes > secSize {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_commit")
                    .message("Codes write exceeds mapped section size")
                    .info("offset", "\(relOff)")
                    .info("bytes", "\(bytes)")
                    .info("limit", "\(secSize)")
                    .build()
            }
            let dst = codesBase.advanced(by: Int(relOff))
            memcpy(dst, codes, Int(bytes))
            try msyncPageAligned(dst, Int(bytes))
            // P4: Codes section CRC freshness deferred — see the IDs branch above.
        }
        if res.vecsStride > 0, let vecs = vecsSrc, let vecsBase = secVecs {
            guard let e = tocByType[.vecs] else {
                throw ErrorBuilder(.invalidFormat, operation: "mmap_append_commit").message("Vecs section not found in TOC").build()
            }
            let secSize = e.size
            let bytes = UInt64(res.addLen &* res.vecsStride)
            let relOff = res.vecsFileOffset
            debugLog("commit_vecs list=\(res.listID) relOff=\(relOff) bytes=\(bytes) secSize=\(secSize)")
            if relOff &+ bytes > secSize {
                throw ErrorBuilder(.mmapError, operation: "mmap_append_commit")
                    .message("Vecs write exceeds mapped section size")
                    .info("offset", "\(relOff)")
                    .info("bytes", "\(bytes)")
                    .info("limit", "\(secSize)")
                    .build()
            }
            let dst = vecsBase.advanced(by: Int(relOff))
            memcpy(dst, vecs, Int(bytes))
            try msyncPageAligned(dst, Int(bytes))
            // P4: Vecs section CRC freshness deferred — see the IDs branch above.
        }
        let newLen = res.oldLen + res.addLen
        // Update length in packed ListsDesc record and sync. This write (and its msync) is what
        // makes the length bump durable at commit-return time — unchanged by P4, and still
        // independent of ListsDesc's *CRC*, which (like IDs/Codes/Vecs above) is no longer
        // refreshed here; it is recomputed by flush()/close()/unclean-reopen repair instead.
        if let base = listsDescBase {
            let rec = base.advanced(by: res.listID * 64)
            writeLE32(rec.advanced(by: 4), UInt32(truncatingIfNeeded: newLen))
            try msyncPageAligned(rec, 64)
        }
        try writeWalCommit(listID: res.listID, newLen: newLen)
    }

    public func mmap_wal_replay() throws {
        guard !opts.readOnly else { return }
        let fd = Darwin.open(walPath, O_RDONLY | O_CLOEXEC)
        if fd < 0 { return }
        defer { _ = Darwin.close(fd) }
        var lastCommitForList: [UInt32: UInt32] = [:]
        let recordSizeAppend = MemoryLayout<WalAppend>.size
        let recordSizeCommit = MemoryLayout<WalCommit>.size
        var offset: off_t = 0
        func readExact(_ n: Int) -> [UInt8]? {
            var buf = [UInt8](repeating: 0, count: n)
            var got = 0
            while got < n {
                let r = buf.withUnsafeMutableBytes { pread(fd, $0.baseAddress!.advanced(by: got), n - got, offset) }
                if r <= 0 { return nil }
                got += r; offset += off_t(r)
            }
            return buf
        }
        while true {
            guard let tagBytes = readExact(4) else { break }
            let tag = tagBytes.withUnsafeBytes { UInt32(littleEndian: $0.load(as: UInt32.self)) }
            if tag == WAL_GROWTH_SENTINEL_TAG {
                // I2: no payload beyond the tag — skip and keep replaying. Unlike an unrecognized
                // tag, this must NOT stop replay: real append/commit records for the growth's
                // eventual commit (or from unrelated later commits) can follow it in the log.
                continue
            } else if tag == WAL_APPEND_TAG {
                guard let rest = readExact(recordSizeAppend - 4) else { break }
                // rest = listID(4) oldLen(4) delta(4) idsOff(8) codesOff(8) vecsOff(8) crc32(4) = 40 bytes
                let storedCRC = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 36, as: UInt32.self)) }
                var payload = [UInt8](repeating: 0, count: recordSizeAppend - 4) // tag + listID..vecsOff = 40 bytes
                tagBytes.withUnsafeBytes { payload.replaceSubrange(0..<4, with: $0) }
                payload.replaceSubrange(4..<(recordSizeAppend - 4), with: rest[0..<(recordSizeAppend - 8)])
                let calc = payload.withUnsafeBytes { CRC32.hash($0.baseAddress!, payload.count) }
                // A torn/corrupt append record means the log is unreliable from here on —
                // stop replay, exactly like the WAL_COMMIT_TAG branch already does on its
                // own CRC mismatch. Append records don't themselves drive recovered state
                // (only WAL_COMMIT_TAG's newLen does); this CRC is validated purely to
                // detect corruption and halt before trusting anything that follows it.
                if calc != storedCRC { break }
            } else if tag == WAL_COMMIT_TAG {
                guard let rest = readExact(recordSizeCommit - 4) else { break }
                let listID = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 0, as: UInt32.self)) }
                let newLen = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 4, as: UInt32.self)) }
                let crc32 = rest.withUnsafeBytes { UInt32(littleEndian: $0.load(fromByteOffset: 8, as: UInt32.self)) }
                var tmp = [UInt8](repeating: 0, count: 8)
                withUnsafeBytes(of: listID.littleEndian) { tmp.replaceSubrange(0..<4, with: $0) }
                withUnsafeBytes(of: newLen.littleEndian) { tmp.replaceSubrange(4..<8, with: $0) }
                let calc = tmp.withUnsafeBytes { CRC32.hash($0.baseAddress!, 8) }
                if calc == crc32 { lastCommitForList[listID] = newLen } else { break }
            } else { break }
        }
        if let base = listsDescBase {
            for (lid, nl) in lastCommitForList {
                let i = Int(lid)
                if i >= 0 && i < kc {
                    let rec = base.advanced(by: i * 64)
                    writeLE32(rec.advanced(by: 4), nl)
                    try msyncPageAligned(rec, 64)
                }
            }
        }
    }

    /// `throws` (fix round 1): now propagates a failed `msyncPageAligned` instead of discarding
    /// it. Its only call site (`mmap_append_begin`'s growth branch) is itself `throws`, so this
    /// was the one call site among the 10 audited for the msync-failure fix that did NOT already
    /// sit in a throwing context -- promoting it to `throws` was required to surface the error.
    private func writeListDescOffsets(listID: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64, newCapacity: Int, idsStride: Int, codesStride: Int, vecsStride: Int) throws {
        guard let base = listsDescBase else { return }
        let rec = base.advanced(by: listID * 64)
        // Write packed LE fields
        writeLE64(rec.advanced(by: 16), idsOff)
        writeLE64(rec.advanced(by: 24), codesOff)
        writeLE64(rec.advanced(by: 32), vecsOff)
        writeLE32(rec.advanced(by: 8), UInt32(truncatingIfNeeded: newCapacity))
        writeLE32(rec.advanced(by: 40), UInt32(truncatingIfNeeded: idsStride))
        writeLE32(rec.advanced(by: 44), UInt32(truncatingIfNeeded: codesStride))
        writeLE32(rec.advanced(by: 48), UInt32(truncatingIfNeeded: vecsStride))
        try msyncPageAligned(rec, 64)
        _ = try? updateSectionCRC(.listsDesc)
    }

    private func ensureFileCapacity(for ty: SectionType, tail: UInt64) throws {
        guard let e = mapSection(ty) else {
            throw ErrorBuilder(.mmapError, operation: "ensure_file_capacity")
                .message("Cannot grow section: not found in TOC")
                .info("section_type", "\(ty.rawValue)")
                .build()
        }
        let off = e.offset
        let needEnd = off + tail
        if needEnd > fileSize {
            // Ensure this section is the last by offset
            for (t, other) in tocByType {
                if t == ty { continue }
                let oOff = other.offset
                if oOff > off {
                    throw ErrorBuilder(.mmapError, operation: "ensure_file_capacity")
                        .message("Cannot grow section: not the last section by offset")
                        .info("section_type", "\(ty.rawValue)")
                        .info("section_offset", "\(off)")
                        .info("later_section_offset", "\(oOff)")
                        .build()
                }
            }
            // Extend file size
            let newSize = alignUp(needEnd, UInt64(getpagesize()))
            if ftruncate(fd, off_t(newSize)) != 0 {
                throw ErrorBuilder(.fileIOError, operation: "ensure_file_capacity")
                    .message("Failed to extend index file")
                    .info("current_size", "\(fileSize)")
                    .info("new_size", "\(newSize)")
                    .info("errno", "\(errno)")
                    .build()
            }
            // Remap
            _ = msync(base, Int(fileSize), MS_SYNC)
            _ = munmap(base, Int(fileSize))
            // I3: `base` is dangling from this point until the remap below succeeds. If `mmap()`
            // fails, this stays `false` for the rest of the handle's life — flush()/close() check
            // it before ever dereferencing `base` again. Deliberately not attempting any repair
            // of the wider growth path here (out of scope for this fix — see the task-3 brief).
            mappingValid = false
            let newMap = mmap(nil, Int(newSize), prot, MAP_SHARED, fd, 0)
            if newMap == MAP_FAILED {
                throw ErrorBuilder(.mmapError, operation: "ensure_file_capacity")
                    .message("Failed to remap extended index file")
                    .info("new_size", "\(newSize)")
                    .info("errno", "\(errno)")
                    .build()
            }
            base = newMap!
            fileSize = newSize
            mappingValid = true
            // Rebuild TOC map (packed)
            tocByType.removeAll(keepingCapacity: true)
            let tocOff = Int(toHost(header.toc_offset, fileEndian: fileEndian))
            let DISK_TOC_ENTRY_SIZE = 36
            for i in 0..<tocCount {
                let te = UnsafeRawPointer(base).advanced(by: tocOff + i * DISK_TOC_ENTRY_SIZE)
                let tyRaw = readLE32(te.advanced(by: 0))
                guard let ty2 = SectionType(rawValue: tyRaw) else {
                    throw ErrorBuilder(.invalidFormat, operation: "ensure_file_capacity")
                        .message("Unknown section type after remap")
                        .info("toc_index", "\(i)")
                        .build()
                }
                // Packed: offset@+4, size@+12, align@+20, flags@+24, crc@+28 (matches the
                // canonical writer/indexInit layout — see indexInit() above).
                let off = readLE64(te.advanced(by: 4))
                let sz  = readLE64(te.advanced(by: 12))
                let al  = readLE32(te.advanced(by: 20))
                let flags = readLE32(te.advanced(by: 24))
                let crc = readLE32(te.advanced(by: 28))
                tocByType[ty2] = HostTOCEntry(type: ty2, offset: off, size: sz, align: al, flags: flags, crc32: crc)
            }
            // Rebuild section base pointers
            if let e = mapSection(.centroids) { secCentroids = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.codebooks) { secCodebooks = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.centroidNorms) { secCentroidNorms = UnsafePointer(UnsafeRawPointer(base).advanced(by: Int(e.offset)).assumingMemoryBound(to: Float.self)) }
            if let e = mapSection(.listsDesc) { listsDescBase = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.ids) { secIDs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.codes) { secCodes = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.vecs) { secVecs = UnsafeMutableRawPointer(mutating: UnsafeRawPointer(base).advanced(by: Int(e.offset))) }
            if let e = mapSection(.normsInv) { secNormsInv = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            if let e = mapSection(.normsSq) { secNormsSq = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            if let e = mapSection(.idMap) { secIDMap = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            if let e = mapSection(.tombstones) { secTombstones = UnsafeRawPointer(base).advanced(by: Int(e.offset)) }
            // Recompute tails
            if let descs = listsDescBase?.assumingMemoryBound(to: UInt8.self) {
                var idsMax: UInt64 = 0, codesMax: UInt64 = 0, vecsMax: UInt64 = 0
                for i in 0..<kc {
                    let rec = UnsafeRawPointer(descs.advanced(by: i * 64))
                    let cap  = UInt64(readLE32(rec.advanced(by: 8)))
                    let idsOff   = readLE64(rec.advanced(by: 16))
                    let codesOff = readLE64(rec.advanced(by: 24))
                    let vecsOff  = readLE64(rec.advanced(by: 32))
                    let idsStride = UInt64(readLE32(rec.advanced(by: 40)))
                    let codesStride = UInt64(readLE32(rec.advanced(by: 44)))
                    let vecsStride = UInt64(readLE32(rec.advanced(by: 48)))
                    idsMax   = max(idsMax, idsOff   &+ cap &* idsStride)
                    codesMax = max(codesMax, codesOff &+ cap &* codesStride)
                    vecsMax  = max(vecsMax, vecsOff  &+ cap &* vecsStride)
                }
                tailIDs = idsMax; tailCodes = codesMax; tailVecs = vecsMax
            }
        }
    }

    private func writeWalAppend(listID: Int, oldLen: Int, delta: Int, idsOff: UInt64, codesOff: UInt64, vecsOff: UInt64) throws {
        var rec = WalAppend(tag: WAL_APPEND_TAG.littleEndian, listID: UInt32(listID).littleEndian, oldLen: UInt32(oldLen).littleEndian, delta: UInt32(delta).littleEndian, idsOff: idsOff.littleEndian, codesOff: codesOff.littleEndian, vecsOff: vecsOff.littleEndian, crc32: 0)
        withUnsafeBytes(of: rec.tag) { walScratch.replaceSubrange(0..<4, with: $0) }
        withUnsafeBytes(of: rec.listID) { walScratch.replaceSubrange(4..<8, with: $0) }
        withUnsafeBytes(of: rec.oldLen) { walScratch.replaceSubrange(8..<12, with: $0) }
        withUnsafeBytes(of: rec.delta) { walScratch.replaceSubrange(12..<16, with: $0) }
        withUnsafeBytes(of: rec.idsOff) { walScratch.replaceSubrange(16..<24, with: $0) }
        withUnsafeBytes(of: rec.codesOff) { walScratch.replaceSubrange(24..<32, with: $0) }
        withUnsafeBytes(of: rec.vecsOff) { walScratch.replaceSubrange(32..<40, with: $0) }
        let crc = walScratch.withUnsafeBytes { CRC32.hash($0.baseAddress!, 40) }
        rec.crc32 = crc.littleEndian
        var r = rec
        let wrote = withUnsafeBytes(of: &r) { write(walFD, $0.baseAddress!, $0.count) }
        if wrote != MemoryLayout<WalAppend>.size {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_append")
                .message("Failed to write WAL append record")
                .info("expected_bytes", "\(MemoryLayout<WalAppend>.size)")
                .info("written_bytes", "\(wrote)")
                .info("errno", "\(errno)")
                .build()
        }
        if fsync(walFD) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_append")
                .message("Failed to fsync WAL file")
                .info("errno", "\(errno)")
                .build()
        }
    }

    private func writeWalCommit(listID: Int, newLen: Int) throws {
        var rec = WalCommit(tag: WAL_COMMIT_TAG.littleEndian, listID: UInt32(listID).littleEndian, newLen: UInt32(newLen).littleEndian, crc32: 0)
        withUnsafeBytes(of: rec.listID) { walScratch.replaceSubrange(0..<4, with: $0) }
        withUnsafeBytes(of: rec.newLen) { walScratch.replaceSubrange(4..<8, with: $0) }
        let crc = walScratch.withUnsafeBytes { CRC32.hash($0.baseAddress!, 8) }
        rec.crc32 = crc.littleEndian
        var r = rec
        let wrote = withUnsafeBytes(of: &r) { write(walFD, $0.baseAddress!, $0.count) }
        if wrote != MemoryLayout<WalCommit>.size {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_commit")
                .message("Failed to write WAL commit record")
                .info("expected_bytes", "\(MemoryLayout<WalCommit>.size)")
                .info("written_bytes", "\(wrote)")
                .info("errno", "\(errno)")
                .build()
        }
        if fsync(walFD) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_commit")
                .message("Failed to fsync WAL file")
                .info("errno", "\(errno)")
                .build()
        }
    }

    /// Fix round 1, I2: writes the 4-byte growth sentinel (see `WAL_GROWTH_SENTINEL_TAG`) and
    /// fsyncs. No payload, no CRC of its own — its only job is to make `st_size > 0` true before
    /// `mmap_append_begin`'s growth branch mutates IDs/Codes/Vecs directly. Guards on `walFD >= 0`
    /// defensively; callers only reach this from an already-writable append path, where
    /// `indexInit()` guarantees the WAL fd is open.
    private func writeWalSentinel() throws {
        guard walFD >= 0 else { return }
        var tag = WAL_GROWTH_SENTINEL_TAG.littleEndian
        let wrote = withUnsafeBytes(of: &tag) { write(walFD, $0.baseAddress!, $0.count) }
        if wrote != MemoryLayout<UInt32>.size {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_sentinel")
                .message("Failed to write WAL growth sentinel record")
                .info("expected_bytes", "\(MemoryLayout<UInt32>.size)")
                .info("written_bytes", "\(wrote)")
                .info("errno", "\(errno)")
                .build()
        }
        if fsync(walFD) != 0 {
            throw ErrorBuilder(.fileIOError, operation: "wal_write_sentinel")
                .message("Failed to fsync WAL file after growth sentinel")
                .info("errno", "\(errno)")
                .build()
        }
    }
}
// Little-endian readers for packed on-disk records
@inline(__always) private func readLE32(_ p: UnsafeRawPointer) -> UInt32 {
    var v: UInt32 = 0
    memcpy(&v, p, 4)
    return UInt32(littleEndian: v)
}

@inline(__always) private func readLE64(_ p: UnsafeRawPointer) -> UInt64 {
    var v: UInt64 = 0
    memcpy(&v, p, 8)
    return UInt64(littleEndian: v)
}
