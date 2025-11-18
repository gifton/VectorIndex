//
//  Telemetry.swift
//  VectorIndex
//
//  Kernel #46: Index Stats & Telemetry
//  Low-overhead instrumentation for performance monitoring and tuning
//
//  Compile-time guard:
//    Define -D VINDEX_TELEM to enable instrumentation. When not defined,
//    all APIs exist but are optimized to no-ops to keep call sites simple.
//
//  Features:
//    - Thread-local storage (pthread TLS)
//    - Lock-striped histograms for latency/size distributions
//    - Sampling support for production overhead control
//    - JSON snapshot export
//    - Ring buffer for recent queries
//    - Power-of-2 histogram buckets for percentile calculation
//

import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// MARK: - Internal Types (Benchmarking/Debugging Only)

/// Timer identifiers for different query stages
internal enum TelemetryTimerId: Int, CaseIterable {
  case t_lut_build = 0
  case t_scan_adc
  case t_score_flat
  case t_topk
  case t_merge
  case t_dedup
  case t_reservoir
  case t_rerank
  case t_total
}

/// Optimization flags tracking which code paths were used
internal struct TelemetryFlags: OptionSet, Sendable {
  let rawValue: UInt64
  static let used_dot_trick         = TelemetryFlags(rawValue: 1 << 0)
  static let used_cosine            = TelemetryFlags(rawValue: 1 << 1)
  static let used_interleaved_codes = TelemetryFlags(rawValue: 1 << 2)
  static let used_u4                = TelemetryFlags(rawValue: 1 << 3)
  static let used_prefetch          = TelemetryFlags(rawValue: 1 << 4)
  static let used_heap_merge        = TelemetryFlags(rawValue: 1 << 5)
}

/// Per-query statistics (returned to caller after query completion)
internal struct QueryStats {
  // Identity / configuration
  var metric: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                      UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) // 16 bytes
  var d: Int32 = 0
  var m: Int32 = 0
  var ks: Int32 = 0
  var nprobe: Int32 = 0
  var C: Int32 = 0
  var K: Int32 = 0

  // Work
  var kc_scored: UInt64 = 0
  var lists_routed: UInt64 = 0
  var lists_scanned: UInt64 = 0
  var codes_scanned: UInt64 = 0
  var vecs_scored: UInt64 = 0
  var candidates_emitted: UInt64 = 0
  var candidates_unique: UInt64 = 0
  var candidates_kept: UInt64 = 0
  var topk_selected: UInt64 = 0

  // Saturation / quality
  var reservoir_tau: Double = 0
  var heap_sifts: UInt64 = 0
  var quickselect_calls: UInt64 = 0
  var dup_ratio: Double = 0
  var beam_expansions: UInt64 = 0

  // Bytes
  var bytes_lut: UInt64 = 0
  var bytes_codes: UInt64 = 0
  var bytes_vecs: UInt64 = 0
  var bytes_ids: UInt64 = 0
  var bytes_norms: UInt64 = 0

  // Timers (ns)
  var t_lut_build: UInt64 = 0
  var t_scan_adc: UInt64 = 0
  var t_score_flat: UInt64 = 0
  var t_topk: UInt64 = 0
  var t_merge: UInt64 = 0
  var t_dedup: UInt64 = 0
  var t_reservoir: UInt64 = 0
  var t_rerank: UInt64 = 0
  var t_total: UInt64 = 0

  // Flags
  var flags: TelemetryFlags = []
}

/// Query context (passed at begin_query)
public struct QueryCtx {
  var metric: String?
  var d: Int32 = 0
  var m: Int32 = 0
  var ks: Int32 = 0
  var nprobe: Int32 = 0
  var C: Int32 = 0
  var K: Int32 = 0
  init(metric: String? = nil, d: Int32, m: Int32, ks: Int32, nprobe: Int32, C: Int32, K: Int32) {
    self.metric = metric; self.d = d; self.m = m; self.ks = ks; self.nprobe = nprobe; self.C = C; self.K = K
  }
}

/// Telemetry configuration
public struct TelemetryConfig {
  var enabled: Bool
  var sampleRate: Double           // [0,1]
  var maxHistBuckets: Int          // default 64, capped at 128
  var sink: ((QueryStats) -> Void)?// optional callback per query
  var persistSnapshot: Bool
  var persistPath: String?
  init(enabled: Bool = false, sampleRate: Double = 0.0, maxHistBuckets: Int = 64,
              sink: ((QueryStats) -> Void)? = nil, persistSnapshot: Bool = false, persistPath: String? = nil) {
    self.enabled = enabled; self.sampleRate = sampleRate; self.maxHistBuckets = maxHistBuckets
    self.sink = sink; self.persistSnapshot = persistSnapshot; self.persistPath = persistPath
  }
}

/// Global telemetry aggregates (snapshot-able)
public struct TelemetryGlobal {
  // Totals
  var queries_total: UInt64 = 0
  var queries_sampled: UInt64 = 0

  // Work sums
  var work_kc_scored: UInt64 = 0
  var work_lists_routed: UInt64 = 0
  var work_lists_scanned: UInt64 = 0
  var work_codes_scanned: UInt64 = 0
  var work_vecs_scored: UInt64 = 0
  var work_candidates_emitted: UInt64 = 0
  var work_candidates_unique: UInt64 = 0
  var work_candidates_kept: UInt64 = 0
  var work_topk_selected: UInt64 = 0

  // Bytes sums
  var bytes_lut: UInt64 = 0
  var bytes_codes: UInt64 = 0
  var bytes_vecs: UInt64 = 0
  var bytes_ids: UInt64 = 0
  var bytes_norms: UInt64 = 0

  // Time sums
  var time_ns: [UInt64] = Array(repeating: 0, count: TelemetryTimerId.allCases.count)

  // Flags counters
  var flag_used_dot_trick: UInt64 = 0
  var flag_used_cosine: UInt64 = 0
  var flag_used_interleaved_codes: UInt64 = 0
  var flag_used_u4: UInt64 = 0
  var flag_used_prefetch: UInt64 = 0
  var flag_used_heap_merge: UInt64 = 0

  // Ring
  var ring_cap: UInt32 = 1024
}

/// Light query stats for ring buffer (compact representation)
internal struct QueryStatsLight {
  var metric: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                      UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  var d: Int32 = 0
  var m: Int32 = 0
  var ks: Int32 = 0
  var nprobe: Int32 = 0
  var C: Int32 = 0
  var K: Int32 = 0
  var lists_scanned: UInt64 = 0
  var codes_scanned: UInt64 = 0
  var kc_scored: UInt64 = 0
  var t_total: UInt64 = 0
}

// MARK: - API Surface (names follow spec)

internal func telem_init(_ cfg: TelemetryConfig?) { Telemetry._init(cfg ?? TelemetryConfig()) }
internal func telem_shutdown() { Telemetry._shutdown() }
internal func telem_thread_init() { Telemetry._threadInit() }
internal func telem_begin_query(_ qctx: QueryCtx?) { Telemetry._beginQuery(qctx) }
internal func telem_end_query(_ out: UnsafeMutablePointer<QueryStats>?) { Telemetry._endQuery(copyTo: out) }
internal func telem_snapshot_json(_ buf: UnsafeMutablePointer<CChar>?, _ cap: Int) -> Int {
  Telemetry._snapshotJSON(into: buf, cap: cap)
}
internal func telem_snapshot_struct(_ out: UnsafeMutablePointer<TelemetryGlobal>?) -> Int {
  Telemetry._snapshotStruct(out)
}
internal func telem_snapshot_to_file(_ path: UnsafePointer<CChar>?) -> Int {
  Telemetry._snapshotToFile(path)
}

// MARK: - Event Helpers (Swift analogs of TELEM_*)

public enum TelemetryCounter {
  case kc_scored, lists_routed, lists_scanned, codes_scanned, vecs_scored
  case candidates_emitted, candidates_unique, candidates_kept, topk_selected
  case heap_sifts, quickselect_calls, beam_expansions
}

public enum TelemetryBytes { case lut, codes, vecs, ids, norms }
public enum TelemetryDoubleField { case reservoir_tau, dup_ratio }
public enum TelemetryU64Field { case candidates_emitted, candidates_unique, candidates_kept }

internal func TELEM_INC(_ c: TelemetryCounter, _ v: UInt64 = 1) { Telemetry._inc(c, v) }
internal func TELEM_ADD_BYTES(_ b: TelemetryBytes, _ v: UInt64) { Telemetry._addBytes(b, v) }
internal func TELEM_SET(_ f: TelemetryDoubleField, _ value: Double) { Telemetry._setDouble(f, value) }
internal func TELEM_SET64(_ f: TelemetryU64Field, _ value: UInt64) { Telemetry._setU64(f, value) }
internal func TELEM_FLAG(_ f: TelemetryFlags) { Telemetry._flag(f) }

/// RAII timer guard (automatically stops timer on deinit)
public struct TelemetryTimerGuard: ~Copyable {
  internal let id: TelemetryTimerId
  internal let t0: UInt64
  init(_ id: TelemetryTimerId) { self.id = id; self.t0 = Telemetry._nowNs() }
  deinit { Telemetry._addTimer(id, delta: Telemetry._nowNs() &- t0) }
}
@discardableResult internal func TELEM_TIMER_GUARD(_ id: TelemetryTimerId) -> TelemetryTimerGuard { TelemetryTimerGuard(id) }

/// Manual timer token (start/end pair)
public struct TelemetryTimerToken {
  internal let id: TelemetryTimerId
  internal let t0: UInt64
}
internal func TELEM_TIMER_START(_ id: TelemetryTimerId) -> TelemetryTimerToken { .init(id: id, t0: Telemetry._nowNs()) }
internal func TELEM_TIMER_END(_ token: TelemetryTimerToken) { Telemetry._addTimer(token.id, delta: Telemetry._nowNs() &- token.t0) }

// MARK: - Implementation

#if VINDEX_TELEM

@usableFromInline
internal enum Telemetry {
  // --- Constants
  static let STRIPES = 8
  static let RECENT_RING = 1024
  static let TIMER_COUNT = TelemetryTimerId.allCases.count

  // --- Time base
  private static var timebase: mach_timebase_info_data_t = {
    var info = mach_timebase_info_data_t(numer: 0, denom: 0)
    mach_timebase_info(&info)
    return info
  }()

  @inline(__always)
  @usableFromInline
  static func _nowNs() -> UInt64 {
    let t = mach_absolute_time()
    // Multiply first, then divide; use &* for wraparound-safe multiply (denom>0).
    return (t &* UInt64(timebase.numer)) / UInt64(timebase.denom)
  }

  // --- Histograms (power-of-2 buckets with striped locks)
  final class Stripe {
    let lock = NSLock()
    var buckets: [UInt64]
    init(_ n: Int) { buckets = Array(repeating: 0, count: n) }
  }

  final class Histogram {
    let bucketCount: Int
    let stripes: [Stripe]
    init(buckets: Int) {
      let c = max(1, min(buckets, 128))
      self.bucketCount = c
      self.stripes = (0..<STRIPES).map { _ in Stripe(c) }
    }

    @inline(__always) static func ceilPow2Bucket(_ v: UInt64) -> Int {
      if v == 0 { return 0 }
      let lz = v.leadingZeroBitCount
      // 1 + floor(log2(v)) ; cap by caller
      return 1 + (63 - lz)
    }

    @inline(__always) func add(_ v: UInt64) {
      var idx = Histogram.ceilPow2Bucket(v)
      if idx >= bucketCount { idx = bucketCount - 1 }
      let sidx = Int(bitPattern: pthread_self()) % STRIPES
      let s = stripes[sidx]
      s.lock.lock()
      s.buckets[idx] &+= 1
      s.lock.unlock()
    }

    func snapshot() -> [UInt64] {
      var out = Array(repeating: 0 as UInt64, count: bucketCount)
      for s in stripes {
        s.lock.lock()
        for i in 0..<bucketCount { out[i] &+= s.buckets[i] }
        s.lock.unlock()
      }
      return out
    }

    func percentileNs(_ p: Double) -> UInt64 {
      let b = snapshot()
      let total = b.reduce(0, &+)
      if total == 0 { return 0 }
      let target: UInt64 = {
        if p <= 0 { return 1 }
        if p >= 1 { return total }
        return UInt64(p * Double(total))
      }()
      var acc: UInt64 = 0
      for i in 0..<bucketCount {
        acc &+= b[i]
        if acc >= target {
          if i == 0 { return 0 }
          return (i == 1) ? 1 : ((1 as UInt64) << i) &- 1
        }
      }
      return ((1 as UInt64) << (bucketCount - 1)) &- 1
    }
  }

  // --- Global (protected by locks on merge)
  static var cfg = TelemetryConfig()
  static var inited = false

  static var g = TelemetryGlobal()
  private static let mergeLock = NSLock()
  private static let ringLock = NSLock()

  private static var histLatency = Histogram(buckets: 64)
  private static var histCodes   = Histogram(buckets: 64)
  private static var histLists   = Histogram(buckets: 64)

  private static var ring = Array(repeating: QueryStatsLight(), count: RECENT_RING)
  private static var ringWrite: UInt64 = 0

  // --- TLS (pthread)
  private static var tlsKey: pthread_key_t = {
    var key = pthread_key_t()
    pthread_key_create(&key) { raw in
      raw?.deinitialize(count: 1)
      raw?.deallocate()
    }
    return key
  }()

  final class TLS {
    var active: Bool = false
    var qStartNs: UInt64 = 0
    var qs = QueryStats()
    var timers = Array(repeating: 0 as UInt64, count: TIMER_COUNT)
    // sampler seed (thread-local)
    var seed: UInt64 = 0
  }

  @inline(__always) static func tls() -> TLS {
    if let p = pthread_getspecific(tlsKey) {
      return Unmanaged<TLS>.fromOpaque(p).takeUnretainedValue()
    } else {
      let t = TLS()
      let u = Unmanaged.passRetained(t).toOpaque()
      pthread_setspecific(tlsKey, u)
      return t
    }
  }

  // --- Sampling
  @inline(__always) static func xorshift64star(_ s: inout UInt64) -> UInt64 {
    var x = s
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27
    s = x
    return x &* 2685821657736338717
  }

  @inline(__always) static func shouldSample(_ rate: Double) -> Bool {
    if rate >= 1 { return true }
    if rate <= 0 { return false }
    let T = tls()
    if T.seed == 0 {
      // Derive from address entropy and monotonic time
      T.seed = _nowNs() ^ UInt64(bitPattern: Int64(bitPattern: UInt64(UInt(bitPattern: Unmanaged.passUnretained(T).toOpaque().hashValue)))) | 1
    }
    var r = xorshift64star(&T.seed)
    // use top 53 bits for a double in [0,1)
    r >>= 11
    let u = Double(r) * (1.0 / 9007199254740992.0)
    return u < rate
  }

  // --- API impl

  static func _init(_ config: TelemetryConfig) {
    guard !inited else { return }
    cfg = config
    let buckets = max(1, min(cfg.maxHistBuckets, 128))
    histLatency = Histogram(buckets: buckets)
    histCodes   = Histogram(buckets: buckets)
    histLists   = Histogram(buckets: buckets)
    g.ring_cap = UInt32(RECENT_RING)
    inited = true
  }

  static func _shutdown() {
    guard inited else { return }
    // nothing to deallocate explicitly (Swift-managed)
    inited = false
  }

  static func _threadInit() {
    // TLS initialized lazily; nothing to do.
  }

  static func _beginQuery(_ qctx: QueryCtx?) {
    // increment queries_total (coarse; guarded)
    mergeLock.lock()
    g.queries_total &+= 1
    mergeLock.unlock()

    let T = tls()
    T.active = cfg.enabled && shouldSample(cfg.sampleRate)
    T.qStartNs = _nowNs()
    T.timers.withUnsafeMutableBufferPointer { ptr in ptr.initialize(repeating: 0) }
    T.qs = QueryStats() // zero it

    if T.active {
      mergeLock.lock(); g.queries_sampled &+= 1; mergeLock.unlock()
      if let name = qctx?.metric, !name.isEmpty {
        var bytes = Array(name.utf8.prefix(16))
        if bytes.count < 16 { bytes += Array(repeating: 0, count: 16 - bytes.count) }
        T.qs.metric = (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                       bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15])
      }
      if let q = qctx {
        T.qs.d = q.d; T.qs.m = q.m; T.qs.ks = q.ks
        T.qs.nprobe = q.nprobe; T.qs.C = q.C; T.qs.K = q.K
      }
    }
  }

  static func _endQuery(copyTo out: UnsafeMutablePointer<QueryStats>?) {
    let T = tls()
    let tTotal = _nowNs() &- T.qStartNs
    if T.active {
      // fold timers
      T.qs.t_lut_build  &+= T.timers[TelemetryTimerId.t_lut_build.rawValue]
      T.qs.t_scan_adc   &+= T.timers[TelemetryTimerId.t_scan_adc.rawValue]
      T.qs.t_score_flat &+= T.timers[TelemetryTimerId.t_score_flat.rawValue]
      T.qs.t_topk       &+= T.timers[TelemetryTimerId.t_topk.rawValue]
      T.qs.t_merge      &+= T.timers[TelemetryTimerId.t_merge.rawValue]
      T.qs.t_dedup      &+= T.timers[TelemetryTimerId.t_dedup.rawValue]
      T.qs.t_reservoir  &+= T.timers[TelemetryTimerId.t_reservoir.rawValue]
      T.qs.t_rerank     &+= T.timers[TelemetryTimerId.t_rerank.rawValue]
      T.qs.t_total      &+= tTotal

      // merge (single lock)
      mergeLock.lock()
      defer { mergeLock.unlock() }

      // work
      g.work_kc_scored          &+= T.qs.kc_scored
      g.work_lists_routed       &+= T.qs.lists_routed
      g.work_lists_scanned      &+= T.qs.lists_scanned
      g.work_codes_scanned      &+= T.qs.codes_scanned
      g.work_vecs_scored        &+= T.qs.vecs_scored
      g.work_candidates_emitted &+= T.qs.candidates_emitted
      g.work_candidates_unique  &+= T.qs.candidates_unique
      g.work_candidates_kept    &+= T.qs.candidates_kept
      g.work_topk_selected      &+= T.qs.topk_selected

      // bytes
      g.bytes_lut   &+= T.qs.bytes_lut
      g.bytes_codes &+= T.qs.bytes_codes
      g.bytes_vecs  &+= T.qs.bytes_vecs
      g.bytes_ids   &+= T.qs.bytes_ids
      g.bytes_norms &+= T.qs.bytes_norms

      // time sums
      g.time_ns[TelemetryTimerId.t_lut_build.rawValue]  &+= T.qs.t_lut_build
      g.time_ns[TelemetryTimerId.t_scan_adc.rawValue]   &+= T.qs.t_scan_adc
      g.time_ns[TelemetryTimerId.t_score_flat.rawValue] &+= T.qs.t_score_flat
      g.time_ns[TelemetryTimerId.t_topk.rawValue]       &+= T.qs.t_topk
      g.time_ns[TelemetryTimerId.t_merge.rawValue]      &+= T.qs.t_merge
      g.time_ns[TelemetryTimerId.t_dedup.rawValue]      &+= T.qs.t_dedup
      g.time_ns[TelemetryTimerId.t_reservoir.rawValue]  &+= T.qs.t_reservoir
      g.time_ns[TelemetryTimerId.t_rerank.rawValue]     &+= T.qs.t_rerank
      g.time_ns[TelemetryTimerId.t_total.rawValue]      &+= T.qs.t_total

      // flags (count queries that used each flag)
      if T.qs.flags.contains(.used_dot_trick) { g.flag_used_dot_trick &+= 1 }
      if T.qs.flags.contains(.used_cosine) { g.flag_used_cosine &+= 1 }
      if T.qs.flags.contains(.used_interleaved_codes) { g.flag_used_interleaved_codes &+= 1 }
      if T.qs.flags.contains(.used_u4) { g.flag_used_u4 &+= 1 }
      if T.qs.flags.contains(.used_prefetch) { g.flag_used_prefetch &+= 1 }
      if T.qs.flags.contains(.used_heap_merge) { g.flag_used_heap_merge &+= 1 }

      // histograms
      histLatency.add(T.qs.t_total)
      histCodes.add(T.qs.codes_scanned)
      histLists.add(T.qs.lists_scanned)

      // ring buffer (coarse)
      ringLock.lock()
      let w = ringWrite
      ringWrite &+= 1
      var light = QueryStatsLight()
      light.metric = T.qs.metric
      light.d = T.qs.d; light.m = T.qs.m; light.ks = T.qs.ks
      light.nprobe = T.qs.nprobe; light.C = T.qs.C; light.K = T.qs.K
      light.lists_scanned = T.qs.lists_scanned
      light.codes_scanned = T.qs.codes_scanned
      light.kc_scored     = T.qs.kc_scored
      light.t_total       = T.qs.t_total
      ring[Int(w % UInt64(RECENT_RING))] = light
      ringLock.unlock()

      // sink callback
      if let sink = cfg.sink { sink(T.qs) }

      // persistence
      if cfg.persistSnapshot, let path = cfg.persistPath {
        _ = _snapshotToFile(path.cString(using: .utf8))
      }

      // return copy if requested
      if let out = out { out.pointee = T.qs }
    } else {
      if let out = out { out.pointee = QueryStats() }
    }
    T.active = false
  }

  // --- Event helpers

  @inline(__always) static func _inc(_ c: TelemetryCounter, _ v: UInt64) {
    let T = tls(); guard T.active else { return }
    switch c {
      case .kc_scored: T.qs.kc_scored &+= v
      case .lists_routed: T.qs.lists_routed &+= v
      case .lists_scanned: T.qs.lists_scanned &+= v
      case .codes_scanned: T.qs.codes_scanned &+= v
      case .vecs_scored: T.qs.vecs_scored &+= v
      case .candidates_emitted: T.qs.candidates_emitted &+= v
      case .candidates_unique: T.qs.candidates_unique &+= v
      case .candidates_kept: T.qs.candidates_kept &+= v
      case .topk_selected: T.qs.topk_selected &+= v
      case .heap_sifts: T.qs.heap_sifts &+= v
      case .quickselect_calls: T.qs.quickselect_calls &+= v
      case .beam_expansions: T.qs.beam_expansions &+= v
    }
  }

  @inline(__always) static func _addBytes(_ b: TelemetryBytes, _ v: UInt64) {
    let T = tls(); guard T.active else { return }
    switch b {
      case .lut:   T.qs.bytes_lut &+= v
      case .codes: T.qs.bytes_codes &+= v
      case .vecs:  T.qs.bytes_vecs &+= v
      case .ids:   T.qs.bytes_ids &+= v
      case .norms: T.qs.bytes_norms &+= v
    }
  }

  @inline(__always) static func _setDouble(_ f: TelemetryDoubleField, _ value: Double) {
    let T = tls(); guard T.active else { return }
    switch f {
      case .reservoir_tau: T.qs.reservoir_tau = value
      case .dup_ratio:     T.qs.dup_ratio = value
    }
  }

  @inline(__always) static func _setU64(_ f: TelemetryU64Field, _ value: UInt64) {
    let T = tls(); guard T.active else { return }
    switch f {
      case .candidates_emitted: T.qs.candidates_emitted = value
      case .candidates_unique:  T.qs.candidates_unique = value
      case .candidates_kept:    T.qs.candidates_kept = value
    }
  }

  @inline(__always) static func _flag(_ f: TelemetryFlags) {
    let T = tls(); guard T.active else { return }
    T.qs.flags.insert(f)
  }

  @inline(__always) static func _addTimer(_ id: TelemetryTimerId, delta: UInt64) {
    let T = tls(); guard T.active else { return }
    T.timers[id.rawValue] &+= delta
  }

  // --- Snapshot

  static func _snapshotStruct(_ out: UnsafeMutablePointer<TelemetryGlobal>?) -> Int {
    guard let out = out else { return -1 }
    mergeLock.lock()
    out.pointee = g
    mergeLock.unlock()
    return 0
  }

  static func _snapshotJSON(into buf: UnsafeMutablePointer<CChar>?, cap: Int) -> Int {
    guard let buf = buf, cap > 0 else { return 0 }

    // Read snapshot under lock
    mergeLock.lock()
    let gg = g
    mergeLock.unlock()

    let p50 = histLatency.percentileNs(0.50)
    let p90 = histLatency.percentileNs(0.90)
    let p99 = histLatency.percentileNs(0.99)

    // Assemble JSON with minimal allocations
    var s = "{"
    s += "\"enabled\":\(cfg.enabled),"
    s += String(format: "\"sample_rate\":%.6f,", cfg.sampleRate)
    s += "\"global\":{"
    s += "\"queries_total\":\(gg.queries_total),"
    s += "\"queries_sampled\":\(gg.queries_sampled),"

    s += "\"work\":{"
    s += "\"kc_scored\":\(gg.work_kc_scored),"
    s += "\"lists_routed\":\(gg.work_lists_routed),"
    s += "\"lists_scanned\":\(gg.work_lists_scanned),"
    s += "\"codes_scanned\":\(gg.work_codes_scanned),"
    s += "\"vecs_scored\":\(gg.work_vecs_scored),"
    s += "\"candidates_emitted\":\(gg.work_candidates_emitted),"
    s += "\"candidates_unique\":\(gg.work_candidates_unique),"
    s += "\"candidates_kept\":\(gg.work_candidates_kept),"
    s += "\"topk_selected\":\(gg.work_topk_selected)"
    s += "},"

    s += "\"bytes\":{"
    s += "\"lut\":\(gg.bytes_lut),"
    s += "\"codes\":\(gg.bytes_codes),"
    s += "\"vecs\":\(gg.bytes_vecs),"
    s += "\"ids\":\(gg.bytes_ids),"
    s += "\"norms\":\(gg.bytes_norms)"
    s += "},"

    s += "\"time_ns_sum\":{"
    s += "\"t_lut\":\(gg.time_ns[TelemetryTimerId.t_lut_build.rawValue]),"
    s += "\"t_scan_adc\":\(gg.time_ns[TelemetryTimerId.t_scan_adc.rawValue]),"
    s += "\"t_score_flat\":\(gg.time_ns[TelemetryTimerId.t_score_flat.rawValue]),"
    s += "\"t_topk\":\(gg.time_ns[TelemetryTimerId.t_topk.rawValue]),"
    s += "\"t_merge\":\(gg.time_ns[TelemetryTimerId.t_merge.rawValue]),"
    s += "\"t_dedup\":\(gg.time_ns[TelemetryTimerId.t_dedup.rawValue]),"
    s += "\"t_reservoir\":\(gg.time_ns[TelemetryTimerId.t_reservoir.rawValue]),"
    s += "\"t_rerank\":\(gg.time_ns[TelemetryTimerId.t_rerank.rawValue]),"
    s += "\"t_total\":\(gg.time_ns[TelemetryTimerId.t_total.rawValue])"
    s += "},"

    s += "\"q_latency_ns\":{"
    s += "\"p50\":\(p50),\"p90\":\(p90),\"p99\":\(p99)"
    s += "},"

    s += "\"flags\":{"
    s += "\"used_dot_trick\":\(gg.flag_used_dot_trick),"
    s += "\"used_cosine\":\(gg.flag_used_cosine),"
    s += "\"used_interleaved_codes\":\(gg.flag_used_interleaved_codes),"
    s += "\"used_u4\":\(gg.flag_used_u4),"
    s += "\"used_prefetch\":\(gg.flag_used_prefetch),"
    s += "\"used_heap_merge\":\(gg.flag_used_heap_merge)"
    s += "}"

    s += "}" // end global

    // recent (up to 16)
    ringLock.lock()
    let wrote = ringWrite
    let emitN = Int(min(UInt64(16), wrote))
    var recent = "["
    for k in 0..<emitN {
      let rec = ring[Int((wrote - 1 - UInt64(k)) % UInt64(RECENT_RING))]
      if k > 0 { recent += "," }
      // metric as string (trim zeros)
      let nameBytes: [UInt8] = [rec.metric.0, rec.metric.1, rec.metric.2, rec.metric.3, rec.metric.4, rec.metric.5, rec.metric.6, rec.metric.7,
                                 rec.metric.8, rec.metric.9, rec.metric.10, rec.metric.11, rec.metric.12, rec.metric.13, rec.metric.14, rec.metric.15]
      let name = String(bytes: nameBytes.prefix { $0 != 0 }, encoding: .utf8) ?? ""
      recent += "{\"metric\":\"\(name)\",\"d\":\(rec.d),\"m\":\(rec.m),\"ks\":\(rec.ks),"
      recent += "\"nprobe\":\(rec.nprobe),\"C\":\(rec.C),\"K\":\(rec.K),"
      recent += "\"lists_scanned\":\(rec.lists_scanned),\"codes_scanned\":\(rec.codes_scanned),"
      recent += "\"kc_scored\":\(rec.kc_scored),\"t_total\":\(rec.t_total)}"
    }
    ringLock.unlock()
    s += ",\"recent\":\(recent)]"
    s += "}"

    // copy to C buffer
    let utf8 = Array(s.utf8)
    let n = min(utf8.count, max(0, cap - 1))
    utf8.withUnsafeBytes {
      memcpy(buf, $0.baseAddress!, n)
    }
    buf[n] = 0
    return n
  }

  static func _snapshotToFile(_ cpath: UnsafePointer<CChar>?) -> Int {
    guard let cpath = cpath, let path = String(validatingUTF8: cpath) else { return -1 }
    var buf = [CChar](repeating: 0, count: 1 << 16)
    let n = _snapshotJSON(into: &buf, cap: buf.count)
    let data = Data(bytes: buf, count: n)
    let url = URL(fileURLWithPath: path)
    do {
      // atomic write via tmp + replace
      let dir = url.deletingLastPathComponent()
      let tmp = dir.appendingPathComponent(".telem.tmp.\(UInt64(_nowNs()))")
      try data.write(to: tmp, options: .atomic)
      // Move atomically to final location
      if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.replaceItemAt(url, withItemAt: tmp)
      } else {
        try FileManager.default.moveItem(at: tmp, to: url)
      }
      return 0
    } catch {
      return -1
    }
  }
}

#else // !VINDEX_TELEM (stubs)

@usableFromInline
internal enum Telemetry {
  @inline(__always) static func _init(_ config: TelemetryConfig) {}
  @inline(__always) static func _shutdown() {}
  @inline(__always) static func _threadInit() {}
  @inline(__always) static func _beginQuery(_ qctx: QueryCtx?) {}
  @inline(__always) static func _endQuery(copyTo out: UnsafeMutablePointer<QueryStats>?) {
    if let out = out { out.pointee = QueryStats() }
  }
  @inline(__always) static func _snapshotStruct(_ out: UnsafeMutablePointer<TelemetryGlobal>?) -> Int {
    if let out = out { out.pointee = TelemetryGlobal() }; return 0
  }
  @inline(__always) static func _snapshotJSON(into buf: UnsafeMutablePointer<CChar>?, cap: Int) -> Int {
    guard let buf = buf, cap > 0 else { return 0 }
    buf[0] = 0; return 0
  }
  @inline(__always) static func _snapshotToFile(_ path: UnsafePointer<CChar>?) -> Int { 0 }

  // events
  @inline(__always) static func _inc(_ c: TelemetryCounter, _ v: UInt64) {}
  @inline(__always) static func _addBytes(_ b: TelemetryBytes, _ v: UInt64) {}
  @inline(__always) static func _setDouble(_ f: TelemetryDoubleField, _ value: Double) {}
  @inline(__always) static func _setU64(_ f: TelemetryU64Field, _ value: UInt64) {}
  @inline(__always) static func _flag(_ f: TelemetryFlags) {}
  @usableFromInline
  @inline(__always)
  static func _nowNs() -> UInt64 { 0 }
  @inline(__always) static func _addTimer(_ id: TelemetryTimerId, delta: UInt64) {}
}

#endif
