//
//  Telemetry.swift
//  VectorIndex
//
//  Kernel #46: Index Stats & Telemetry
//
//  Phase-2 cleanup (2026-07): the VINDEX_TELEM-gated TLS/histogram/JSON-snapshot
//  implementation below never compiled (12 Swift 6 strict-concurrency + stale-API
//  errors under -D VINDEX_TELEM) and was never reachable from any call site in a
//  shipping build. It has been removed. The types below are kept -- deprecated,
//  not deleted, because they are `public` -- for source compatibility only; see
//  the `@available` message on each for the Phase-4 removal plan.
//
//  The project's actual telemetry surface is the per-kernel push-callback
//  recorders (HNSWTelemetryRecorder, GlobalTelemetryRecorder,
//  IndexOps.Scoring.Cosine/InnerProduct.TelemetryRecorder, L2SqrTelemetryRecorder,
//  IndexOps.Selection.TopKTelemetryRecorder, RangeScanTelemetryRecorder) plus
//  CandidateDedup.DefaultVisitedSet.getTelemetry(). See docs/cleanup-0.2.0-plan.md.
//

import Foundation

// MARK: - Internal Types (dead; retained only as storage for the deprecated public types below)

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
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: the VINDEX_TELEM implementation that consumed this never compiled and had no reachable call site. Scheduled for removal in Phase 4.")
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
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: the VINDEX_TELEM implementation that consumed this never compiled and had no reachable call site. Scheduled for removal in Phase 4.")
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
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: the VINDEX_TELEM implementation that consumed this never compiled and had no reachable call site. Scheduled for removal in Phase 4.")
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

// MARK: - Event Helpers (dead; retained only as storage for the deprecated public types below)

@available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever incremented this. Scheduled for removal in Phase 4.")
public enum TelemetryCounter {
  case kc_scored, lists_routed, lists_scanned, codes_scanned, vecs_scored
  case candidates_emitted, candidates_unique, candidates_kept, topk_selected
  case heap_sifts, quickselect_calls, beam_expansions
}

@available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever incremented this. Scheduled for removal in Phase 4.")
public enum TelemetryBytes { case lut, codes, vecs, ids, norms }
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever set this. Scheduled for removal in Phase 4.")
public enum TelemetryDoubleField { case reservoir_tau, dup_ratio }
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever set this. Scheduled for removal in Phase 4.")
public enum TelemetryU64Field { case candidates_emitted, candidates_unique, candidates_kept }

/// RAII timer guard (automatically stops timer on deinit)
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever constructed this. Scheduled for removal in Phase 4.")
public struct TelemetryTimerGuard: ~Copyable {
  internal let id: TelemetryTimerId
  internal let t0: UInt64
  init(_ id: TelemetryTimerId) { self.id = id; self.t0 = Telemetry._nowNs() }
  deinit { Telemetry._addTimer(id, delta: Telemetry._nowNs() &- t0) }
}

/// Manual timer token (start/end pair)
@available(*, deprecated, message: "Dead since the Phase 2 cleanup: no reachable call site ever constructed this. Scheduled for removal in Phase 4.")
public struct TelemetryTimerToken {
  internal let id: TelemetryTimerId
  internal let t0: UInt64
}

// MARK: - Implementation (stub only; the VINDEX_TELEM-gated real implementation
// and its 12 compile errors -- 10 Swift 6 strict-concurrency violations on bare
// mutable global state, 2 stale pthread-TLS-destructor API calls -- were removed
// in the Phase 2 cleanup. `_nowNs`/`_addTimer` are kept only because the
// deprecated-but-still-public `TelemetryTimerGuard` above references them.)

@usableFromInline
internal enum Telemetry {
  @usableFromInline
  @inline(__always)
  static func _nowNs() -> UInt64 { 0 }
  @inline(__always) static func _addTimer(_ id: TelemetryTimerId, delta: UInt64) {}
}
