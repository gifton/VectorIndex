import Foundation
import simd
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public extension IndexOps.Support {
    enum Norms {
        // MARK: - Norm Mode
        public enum NormMode: UInt8 {
            case none = 0
            case inv  = 1
            case sq   = 2
            case both = 3
            @inlinable public var needsInv: Bool { self == .inv || self == .both }
            @inlinable public var needsSq:  Bool { self == .sq  || self == .both }
        }

        // MARK: - Data Type
        public enum NormDType: UInt8 {
            case float32 = 0
            case float16 = 1
            case bfloat16 = 2
            @inlinable public var byteSize: Int {
                switch self {
                case .float32:  return 4
                case .float16:  return 2
                case .bfloat16: return 2
                }
            }
        }

        // MARK: - Norm Cache Storage
        public struct NormCache {
            public var count: Int
            public var dimension: Int
            public var mode: NormMode
            public var invDType: NormDType
            public var epsilon: Float

            public private(set) var invNorms: UnsafeMutablePointer<Float>? // when invDType == .float32
            public private(set) var sqNorms:  UnsafeMutablePointer<Float>?

            public private(set) var invRaw: UnsafeMutableRawPointer?
            private var mmapBase: UnsafeMutableRawPointer?
            private var mmapSize: Int = 0
            private var mapped: Bool { mmapBase != nil }

            public init(count: Int, dimension: Int, mode: NormMode, invDType: NormDType = .float32, epsilon: Float = 1e-12) {
                self.count = count
                self.dimension = dimension
                self.mode = mode
                self.invDType = invDType
                self.epsilon = epsilon
                self.invNorms = nil
                self.sqNorms = nil
                self.invRaw = nil
                self.mmapBase = nil
                self.mmapSize = 0
            }

            public mutating func allocate() {
                precondition(!mapped, "Cannot allocate on a memory-mapped cache")
                if mode.needsInv {
                    let bytes = count * invDType.byteSize
                    self.invRaw = allocateAligned(bytes: bytes, align: 64)
                    if invDType == .float32 { self.invNorms = self.invRaw!.assumingMemoryBound(to: Float.self) }
                }
                if mode.needsSq {
                    let bytes = count * MemoryLayout<Float>.stride
                    let ptr = allocateAligned(bytes: bytes, align: 64)
                    self.sqNorms = ptr.assumingMemoryBound(to: Float.self)
                }
            }

            public func deallocate() {
                if mapped {
                    if let base = mmapBase { munmap(base, mmapSize) }
                } else {
                    if mode.needsInv, let p = invRaw { free(p) }
                    if mode.needsSq,  let p = sqNorms { free(p) }
                }
            }

            @inlinable public func invPointer_f16() -> UnsafeMutablePointer<Float16>? {
                (invDType == .float16) ? invRaw?.assumingMemoryBound(to: Float16.self) : nil
            }
            @inlinable public func invPointer_bf16() -> UnsafeMutablePointer<UInt16>? {
                (invDType == .bfloat16) ? invRaw?.assumingMemoryBound(to: UInt16.self) : nil
            }

            internal mutating func setMapped(base: UnsafeMutableRawPointer, size: Int, invPtr: UnsafeMutableRawPointer?, sqPtr: UnsafeMutablePointer<Float>?) {
                self.mmapBase = base
                self.mmapSize = size
                self.invRaw = invPtr
                self.invNorms = (invDType == .float32 && invPtr != nil) ? invPtr!.assumingMemoryBound(to: Float.self) : nil
                self.sqNorms = sqPtr
            }
        }

        // MARK: - Core math (L2 norm squared)
        @inlinable
        public static func l2NormSquared(vector x: UnsafePointer<Float>, dimension d: Int) -> Float {
            if d == 0 { return 0 }
            var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero, a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
            let d16 = d & ~15
            var j = 0
            while j < d16 {
                let v0 = load4(x.advanced(by: j +  0))
                let v1 = load4(x.advanced(by: j +  4))
                let v2 = load4(x.advanced(by: j +  8))
                let v3 = load4(x.advanced(by: j + 12))
                a0 += v0 * v0
                a1 += v1 * v1
                a2 += v2 * v2
                a3 += v3 * v3
                j += 16
            }
            var sum = hsum4(a0 + a1 + a2 + a3)
            let d4 = d & ~3
            while j < d4 {
                let v = load4(x.advanced(by: j))
                sum += hsum4(v * v)
                j += 4
            }
            while j < d { let v = x[j]; sum += v * v; j += 1 }
            return sum
        }

        // MARK: - Query norms
        @inlinable
        public static func queryInvNorm(query q: UnsafePointer<Float>, dimension d: Int, epsilon: Float = 1e-12) -> Float {
            let s = l2NormSquared(vector: q, dimension: d)
            let protected = max(s, epsilon)
            return 1.0 / sqrtf(protected)
        }
        @inlinable
        public static func querySquaredNorm(query q: UnsafePointer<Float>, dimension d: Int) -> Float {
            l2NormSquared(vector: q, dimension: d)
        }

        // MARK: - Batch Norm Building / Append / Update
        @inlinable
        public static func normsBuild(
            vectors xb: UnsafePointer<Float>, count n: Int, dimension d: Int,
            mode: NormMode, epsilon: Float = 1e-12,
            invOut invRaw: UnsafeMutableRawPointer?, sqOut sq: UnsafeMutablePointer<Float>?, invDType: NormDType = .float32
        ) {
            if n == 0 || mode == .none { return }
            verifyAlignment(UnsafeRawPointer(xb), 64, "vectors")
            verifyAlignment(invRaw, 64, "invOut")
            verifyAlignment(UnsafeRawPointer(sq), 64, "sqOut")
            for i in 0..<n {
                let row = xb.advanced(by: i * d)
                let s = l2NormSquared(vector: row, dimension: d)
                if mode.needsSq, let sqp = sq { sqp[i] = s }
                if mode.needsInv, let dst = invRaw {
                    let inv = 1.0 / sqrtf(max(s, epsilon))
                    switch invDType {
                    case .float32: dst.assumingMemoryBound(to: Float.self)[i] = inv
                    case .float16: dst.assumingMemoryBound(to: Float16.self)[i] = f32_to_f16(inv)
                    case .bfloat16: dst.assumingMemoryBound(to: UInt16.self)[i] = floatToBF16(inv)
                    }
                }
            }
        }

        @inlinable
        public static func normsAppend(
            newVectors xb: UnsafePointer<Float>, count m: Int, dimension d: Int,
            mode: NormMode, epsilon: Float = 1e-12,
            invOut invRaw: UnsafeMutableRawPointer?, sqOut sq: UnsafeMutablePointer<Float>?, invDType: NormDType = .float32
        ) {
            if m == 0 || mode == .none { return }
            for i in 0..<m {
                let row = xb.advanced(by: i * d)
                let s = l2NormSquared(vector: row, dimension: d)
                if mode.needsSq, let sqp = sq { sqp[i] = s }
                if mode.needsInv, let dst = invRaw {
                    let inv = 1.0 / sqrtf(max(s, epsilon))
                    switch invDType {
                    case .float32: dst.assumingMemoryBound(to: Float.self)[i] = inv
                    case .float16: dst.assumingMemoryBound(to: Float16.self)[i] = f32_to_f16(inv)
                    case .bfloat16: dst.assumingMemoryBound(to: UInt16.self)[i] = floatToBF16(inv)
                    }
                }
            }
        }

        @inlinable
        public static func normsUpdate(
            updatedVectors xb: UnsafePointer<Float>, ids: UnsafePointer<Int>, count m: Int, dimension d: Int,
            mode: NormMode, epsilon: Float = 1e-12,
            invOut invRaw: UnsafeMutableRawPointer?, sqOut sq: UnsafeMutablePointer<Float>?, invDType: NormDType = .float32
        ) {
            if m == 0 || mode == .none { return }
            for i in 0..<m {
                let row = xb.advanced(by: i * d)
                let id = ids[i]
                let s = l2NormSquared(vector: row, dimension: d)
                if mode.needsSq, let sqp = sq { sqp[id] = s }
                if mode.needsInv, let dst = invRaw {
                    let inv = 1.0 / sqrtf(max(s, epsilon))
                    switch invDType {
                    case .float32: dst.assumingMemoryBound(to: Float.self)[id] = inv
                    case .float16: dst.assumingMemoryBound(to: Float16.self)[id] = f32_to_f16(inv)
                    case .bfloat16: dst.assumingMemoryBound(to: UInt16.self)[id] = floatToBF16(inv)
                    }
                }
            }
        }

        // MARK: - Conversions
        @inlinable public static func convertInvNorms_f32_to_f16(input src: UnsafePointer<Float>, output dst: UnsafeMutablePointer<Float16>, count n: Int) {
            for i in 0..<n { dst[i] = f32_to_f16(src[i]) }
        }
        @inlinable public static func convertInvNorms_f32_to_bf16(input src: UnsafePointer<Float>, output dst: UnsafeMutablePointer<UInt16>, count n: Int) {
            for i in 0..<n { dst[i] = floatToBF16(src[i]) }
        }
        @inlinable public static func widenInvNorms_f16_to_f32(input src: UnsafePointer<Float16>, output dst: UnsafeMutablePointer<Float>, count n: Int) {
            for i in 0..<n { dst[i] = Float(src[i]) }
        }

        // MARK: - bfloat16 helpers
        @inline(__always) public static func floatToBF16(_ value: Float) -> UInt16 {
            let bits = value.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            return UInt16(truncatingIfNeeded: rounded >> 16)
        }
        @inline(__always) public static func bf16ToFloat(_ bf16: UInt16) -> Float {
            let bits = UInt32(bf16) << 16
            return Float(bitPattern: bits)
        }
        @usableFromInline static func f32_to_f16(_ x: Float) -> Float16 {
            let maxF16 = Float(Float16.greatestFiniteMagnitude)
            let minF16 = -maxF16
            let clamped = max(minF16, min(maxF16, x))
            return Float16(clamped)
        }

        // MARK: - Mmap header and IO
        public struct NormCacheHeader {
            public var magic: UInt32
            public var version: UInt32
            public var mode: UInt8
            public var invDType: UInt8
            public var sqDType: UInt8
            public var _reserved0: UInt8
            public var dimension: UInt32
            public var count: UInt64
            public var epsilon: Float
            public var checksum: UInt64
            public var _pad: (UInt64, UInt64)
            public static let size: Int = 64
            public init(mode: NormMode, invDType: NormDType, dimension: Int, count: Int, epsilon: Float, checksum: UInt64) {
                self.magic = 0x4E524D43
                self.version = 1
                self.mode = mode.rawValue
                self.invDType = invDType.rawValue
                self.sqDType = 0
                self._reserved0 = 0
                self.dimension = UInt32(dimension)
                self.count = UInt64(count)
                self.epsilon = epsilon
                self.checksum = checksum
                self._pad = (0, 0)
            }
        }

        public static func save(cache: NormCache, path: String) throws {
            let fd = open(path, O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
            guard fd >= 0 else { throw NSError(domain: "NormCache.save", code: 1, userInfo: [NSLocalizedDescriptionKey: "open failed"]) }
            defer { close(fd) }

            let n = cache.count
            let mode = cache.mode
            let invDT = cache.invDType
            let invBytes = mode.needsInv ? n * invDT.byteSize : 0
            let sqBytes  = mode.needsSq  ? n * MemoryLayout<Float>.stride : 0
            let dataBytes = invBytes + sqBytes

            var hdr = NormCacheHeader(mode: mode, invDType: invDT, dimension: cache.dimension, count: n, epsilon: cache.epsilon, checksum: 0)
            try writeAll(fd: fd, ptr: &hdr, count: NormCacheHeader.size)

            if mode.needsInv {
                guard let invRaw = cache.invRaw else { throw NSError(domain: "NormCache.save", code: 2, userInfo: [NSLocalizedDescriptionKey: "invRaw is nil"]) }
                try writeAll(fd: fd, ptr: invRaw, count: invBytes)
            }
            if mode.needsSq {
                guard let sq = cache.sqNorms else { throw NSError(domain: "NormCache.save", code: 3, userInfo: [NSLocalizedDescriptionKey: "sqNorms is nil"]) }
                try writeAll(fd: fd, ptr: sq, count: sqBytes)
            }

            let dataOffset = NormCacheHeader.size
            let fileSize = off_t(dataOffset + dataBytes)
            let map = mmap(nil, dataBytes, PROT_READ, MAP_FILE | MAP_SHARED, fd, off_t(dataOffset))
            guard map != MAP_FAILED else { throw NSError(domain: "NormCache.save", code: 4, userInfo: [NSLocalizedDescriptionKey: "mmap for checksum failed"]) }
            let crc = crc64_ecma(buffer: map!, length: dataBytes)
            munmap(map, dataBytes)

            hdr.checksum = crc
            guard lseek(fd, 0, SEEK_SET) == 0 else { throw NSError(domain: "NormCache.save", code: 5, userInfo: [NSLocalizedDescriptionKey: "lseek failed"]) }
            try writeAll(fd: fd, ptr: &hdr, count: NormCacheHeader.size)
            ftruncate(fd, fileSize)
        }

        public static func load(path: String) throws -> NormCache {
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw NSError(domain: "NormCache.load", code: 10, userInfo: [NSLocalizedDescriptionKey: "open failed"]) }
            defer { close(fd) }
            var st = stat()
            guard fstat(fd, &st) == 0 else { throw NSError(domain: "NormCache.load", code: 11, userInfo: [NSLocalizedDescriptionKey: "fstat failed"]) }
            let fileSize = Int(st.st_size)
            guard fileSize >= NormCacheHeader.size else { throw NSError(domain: "NormCache.load", code: 12, userInfo: [NSLocalizedDescriptionKey: "file too small"]) }

            let base = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
            guard base != MAP_FAILED else { throw NSError(domain: "NormCache.load", code: 13, userInfo: [NSLocalizedDescriptionKey: "mmap failed"]) }

            let hdrPtr = base!.assumingMemoryBound(to: NormCacheHeader.self)
            let hdr = hdrPtr.pointee
            guard hdr.magic == 0x4E524D43, hdr.version == 1 else {
                munmap(base, fileSize)
                throw NSError(domain: "NormCache.load", code: 14, userInfo: [NSLocalizedDescriptionKey: "bad magic/version"]) }

            let mode = NormMode(rawValue: hdr.mode) ?? .none
            let invDT = NormDType(rawValue: hdr.invDType) ?? .float32
            let n = Int(hdr.count)
            let d = Int(hdr.dimension)
            let eps = hdr.epsilon

            let dataOffset = NormCacheHeader.size
            let invBytes = mode.needsInv ? n * invDT.byteSize : 0
            let sqBytes  = mode.needsSq  ? n * MemoryLayout<Float>.stride : 0
            let dataBytes = invBytes + sqBytes
            guard fileSize >= dataOffset + dataBytes else { munmap(base, fileSize); throw NSError(domain: "NormCache.load", code: 15, userInfo: [NSLocalizedDescriptionKey: "truncated file"]) }

            let dataPtr = UnsafeMutableRawPointer(mutating: base!.advanced(by: dataOffset))
            let crc = crc64_ecma(buffer: dataPtr, length: dataBytes)
            guard crc == hdr.checksum else { munmap(base, fileSize); throw NSError(domain: "NormCache.load", code: 16, userInfo: [NSLocalizedDescriptionKey: "checksum mismatch"]) }

            var cache = NormCache(count: n, dimension: d, mode: mode, invDType: invDT, epsilon: eps)
            var invPtr: UnsafeMutableRawPointer? = nil
            var sqPtr: UnsafeMutablePointer<Float>? = nil
            if mode.needsInv { invPtr = dataPtr }
            if mode.needsSq  { sqPtr = dataPtr.advanced(by: invBytes).assumingMemoryBound(to: Float.self) }
            cache.setMapped(base: UnsafeMutableRawPointer(mutating: base!), size: fileSize, invPtr: invPtr, sqPtr: sqPtr)
            return cache
        }

        // MARK: - Telemetry struct
        public struct Telemetry {
            public let vectorsProcessed: Int
            public let dimension: Int
            public let mode: NormMode
            public let invDType: NormDType
            public let zeroNormCount: Int
            public let nearZeroCount: Int
            public let executionTimeNanos: UInt64
            @inlinable public var throughputVecsPerSec: Double {
                let sec = Double(executionTimeNanos) / 1e9
                return sec > 0 ? Double(vectorsProcessed) / sec : 0
            }
        }

        // MARK: - CRC64-ECMA
        @inline(__always) private static func crc64_ecma(buffer: UnsafeRawPointer, length: Int) -> UInt64 {
            let poly: UInt64 = 0x42F0E1EBA9EA3693
            var table = [UInt64](repeating: 0, count: 256)
            for i in 0..<256 {
                var crc = UInt64(i) << 56
                for _ in 0..<8 { crc = (crc & 0x8000000000000000) != 0 ? ((crc << 1) ^ poly) : (crc << 1) }
                table[i] = crc
            }
            var crc: UInt64 = 0
            let bytes = buffer.assumingMemoryBound(to: UInt8.self)
            for i in 0..<length {
                let idx = Int(((crc >> 56) ^ UInt64(bytes[i])) & 0xFF)
                crc = table[idx] ^ (crc << 8)
            }
            return crc
        }

        // MARK: - POSIX write helpers
        @inline(__always) private static func writeAll<T>(fd: Int32, ptr: UnsafeMutablePointer<T>, count: Int) throws {
            try writeAll(fd: fd, raw: UnsafeRawPointer(ptr), count: count)
        }
        @inline(__always) private static func writeAll(fd: Int32, ptr: UnsafeMutableRawPointer, count: Int) throws {
            try writeAll(fd: fd, raw: UnsafeRawPointer(ptr), count: count)
        }
        @inline(__always) private static func writeAll(fd: Int32, raw: UnsafeRawPointer, count: Int) throws {
            var remaining = count
            var p = raw.assumingMemoryBound(to: UInt8.self)
            while remaining > 0 {
                let w = Darwin.write(fd, p, remaining)
                if w < 0 { throw NSError(domain: "NormCache.write", code: 20, userInfo: [NSLocalizedDescriptionKey: "write failed"]) }
                remaining -= w
                p = p.advanced(by: w)
            }
        }

        // MARK: - Low-level helpers
        @inline(__always) private static func allocateAligned(bytes: Int, align: Int) -> UnsafeMutableRawPointer {
            precondition(bytes >= 0)
            var p: UnsafeMutableRawPointer?
            let r = posix_memalign(&p, align, bytes)
            precondition(r == 0 && p != nil, "posix_memalign failed")
            return p!
        }
        @usableFromInline static func load4(_ p: UnsafePointer<Float>) -> SIMD4<Float> {
            if Int(bitPattern: p) & (MemoryLayout<SIMD4<Float>>.alignment - 1) == 0 {
                return p.withMemoryRebound(to: SIMD4<Float>.self, capacity: 1) { $0.pointee }
            } else { return SIMD4<Float>(p[0], p[1], p[2], p[3]) }
        }
        @usableFromInline static func hsum4(_ v: SIMD4<Float>) -> Float { v[0]+v[1]+v[2]+v[3] }
        @usableFromInline static func verifyAlignment(_ ptr: UnsafeRawPointer?, _ alignment: Int, _ label: String) {
            #if DEBUG
            if let p = ptr { assert(Int(bitPattern: p) % alignment == 0, "\(label) must be \(alignment)-byte aligned") }
            #endif
        }
    }
}
