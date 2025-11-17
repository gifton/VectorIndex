// ===----------------------------------------------------------------------===//
// VectorIndex – Kernel #39: Candidate Reservoir Buffer
// Path: Sources/VectorIndex/Operations/Reservoir/CandidateReservoir.swift
//
// Single-writer reservoir for buffering (id, score) during scans.
// Modes: Block, Heap, Adaptive (Block→Heap at occupancy threshold).
// Integrates with a VisitedSet (kernel #32) via `VisitedSet` protocol.
// Deterministic tie-breaking by id.
// Thread-safety: Single-writer (per-query instance). Not thread-safe for mutation.
// ===----------------------------------------------------------------------===//

import Foundation

// MARK: - Metric & Mode

/// Distance/similarity metric ordering.
public enum ReservoirMetric: Sendable {
  /// Smaller is better
  case l2
  /// Larger is better
  case innerProduct
  /// Larger is better
  case cosine
}

/// Reservoir buffering modes.
public enum ReservoirMode: Sendable {
  case heap        // O(log C) insert, threshold pruning by root (worst-at-root)
  case block       // Append until C(1+α), then quickselect prune to C
  case adaptive    // Start Block, switch to Heap when occupancy > threshold
}

// MARK: - Options & Telemetry

/// Configuration options for reservoir behavior.
public struct ReservoirOptions: Sendable {
  public var mode: ReservoirMode
  /// Extra headroom fraction for Block mode (α). Typical 0.1–0.2.
  public var reserveExtra: Float
  /// Occupancy threshold (0…1) for Adaptive to switch Block→Heap.
  public var adaptiveThreshold: Float
  /// Deterministic tie-breaking when scores are equal (default true).
  public var stableTies: Bool
  /// Initial mode when using `.adaptive` (default .block).
  public var adaptiveInitialMode: ReservoirMode
  /// Enable simple telemetry counters.
  public var telemetry: Bool

  public static let `default` = ReservoirOptions(
    mode: .adaptive,
    reserveExtra: 0.10,
    adaptiveThreshold: 0.75,
    stableTies: true,
    adaptiveInitialMode: .block,
    telemetry: false
  )

  @inlinable
  public init(
    mode: ReservoirMode = .adaptive,
    reserveExtra: Float = 0.10,
    adaptiveThreshold: Float = 0.75,
    stableTies: Bool = true,
    adaptiveInitialMode: ReservoirMode = .block,
    telemetry: Bool = false
  ) {
    self.mode = mode
    self.reserveExtra = reserveExtra
    self.adaptiveThreshold = adaptiveThreshold
    self.stableTies = stableTies
    self.adaptiveInitialMode = adaptiveInitialMode
    self.telemetry = telemetry
  }
}

/// Lightweight counters for perf/behavior tracking.
public struct ReservoirTelemetry: Sendable {
  public var pushed: Int64 = 0
  public var accepted: Int64 = 0
  public var rejectedTau: Int64 = 0
  public var rejectedDedup: Int64 = 0
  public var rejectedInvalid: Int64 = 0
  public var prunes: Int64 = 0
  public var modeSwitches: Int64 = 0

  public init() {}
}

// MARK: - CandidateReservoir

/// Single-writer candidate reservoir. Public API is Sendable; the type itself
/// contains mutable state and is not thread-safe for concurrent mutation.
/// Create one per query thread.
public final class CandidateReservoir: @unchecked Sendable {

  // MARK: Storage (SoA)
  @usableFromInline internal var scores: [Float]
  @usableFromInline internal var ids: [Int64]

  // MARK: Configuration
  @usableFromInline internal var capacityC: Int
  @usableFromInline internal var metric: ReservoirMetric
  @usableFromInline internal var opts: ReservoirOptions
  @usableFromInline internal var currentMode: ReservoirMode

  // MARK: State
  /// Current count of buffered candidates (≤ bufferCapacity).
  @usableFromInline internal var size: Int = 0
  /// Buffer capacity allowing headroom in Block/Adaptive Block phase.
  @usableFromInline internal var bufferCapacity: Int
  /// Current acceptance threshold (score of worst candidate among top‑C).
  /// Meaningful in Heap, and after Block prune; undefined when size < C.
  @usableFromInline internal var tau: Float = .infinity

  // MARK: Telemetry
  public var telemetry: ReservoirTelemetry = .init()

  // MARK: Constants
  @usableFromInline internal static let scoreEps: Float = 1e-6

  // MARK: - Init / Reset

  /// Initialize a reservoir with capacity `C` and metric.
  ///
  /// Thread-safety: single-writer; read-only snapshots allowed between mutations.
  @inlinable
  public init(capacity: Int, metric: ReservoirMetric, options: ReservoirOptions = .default) {
    precondition(capacity > 0, "Reservoir capacity must be > 0")
    self.capacityC = capacity
    self.metric = metric
    self.opts = options
    self.currentMode = (options.mode == .adaptive) ? options.adaptiveInitialMode : options.mode

    let headroom = max(1, Int(ceil(Float(capacity) * max(0, options.reserveExtra))))
    self.bufferCapacity = (self.currentMode == .heap) ? capacity : capacity &+ headroom

    // Pre-allocate SoA buffers (no hot-path allocations).
    self.scores = [Float](repeating: 0, count: self.bufferCapacity)
    self.ids = [Int64](repeating: 0, count: self.bufferCapacity)

    // Initial tau sentinel (only meaningful when we have ≥ C items).
    self.tau = worstSentinel(for: metric)
  }

  /// Reset for a new query. Optionally change capacity `newCapacity`.
  ///
  /// O(1) when buffers are large enough (no reallocation).
  @inlinable
  public func reset(newCapacity: Int? = nil) {
    if let nc = newCapacity {
      precondition(nc > 0, "Reservoir capacity must be > 0")
      capacityC = nc
      // Resize buffer if needed (keep headroom policy).
      let headroom = max(1, Int(ceil(Float(capacityC) * max(0, opts.reserveExtra))))
      let newBufferCapacity = (currentMode == .heap) ? capacityC : capacityC &+ headroom
      if newBufferCapacity > bufferCapacity {
        scores = [Float](repeating: 0, count: newBufferCapacity)
        ids = [Int64](repeating: 0, count: newBufferCapacity)
        bufferCapacity = newBufferCapacity
      }
    }

    size = 0
    currentMode = (opts.mode == .adaptive) ? opts.adaptiveInitialMode : opts.mode
    // Reset tau
    tau = worstSentinel(for: metric)
    if opts.telemetry { telemetry = .init() }
  }

  // MARK: - Public Query

  /// Current number of buffered candidates.
  @inlinable
  public var count: Int { size }

  /// Read-only snapshot; pointers are valid only for the duration of `body`.
  /// Do not mutate the reservoir while the closure is executing.
  @inlinable
  public func withSnapshot<R>(
    _ body: (UnsafePointer<Float>, UnsafePointer<Int64>, Int) throws -> R
  ) rethrows -> R {
    try scores.withUnsafeBufferPointer { sp in
      try ids.withUnsafeBufferPointer { ip in
        try body(sp.baseAddress!, ip.baseAddress!, size)
      }
    }
  }

  // MARK: - Push Batch

  /// Push a batch of candidates. Returns number of accepted (unique, valid) items.
  ///
  /// - Parameters:
  ///   - ids:    pointer to candidate IDs [count]
  ///   - scores: pointer to candidate scores [count]
  ///   - count:  number of items in batch
  ///   - visited: optional dedup set (Kernel #32)
  ///
  /// Behavior by mode:
  ///   - Block: append until capacityWithHeadroom, quickselect prune to C
  ///   - Heap: keep C best by maintaining worst-at-root heap
  ///   - Adaptive: Block initially; when |R|/C > threshold, prune + heapify
  @inlinable
  @discardableResult
  public func pushBatch(
    ids idPtr: UnsafePointer<Int64>,
    scores scorePtr: UnsafePointer<Float>,
    count n: Int,
    visited: (any VisitedSet)? = nil
  ) -> Int {
    guard n > 0 else { return 0 }

    var acceptedInBatch = 0

    // Fast-path locals
    let C = capacityC

    // Process candidates sequentially (batch-sized loop for cache prefetching).
    var i = 0
    while i < n {
      let cid = idPtr[i]
      let s = scorePtr[i]
      telemetry.pushed &+= 1

      // Drop NaN/Inf scores
      if !s.isFinite {
        telemetry.rejectedInvalid &+= 1
        i &+= 1
        continue
      }

      // Dedup check (if provided)
      if let vs = visited, !vs.testAndSet(id: cid) {
        telemetry.rejectedDedup &+= 1
        i &+= 1
        continue
      }

      switch currentMode {
      case .block:
        // Append (no threshold check); prune when exceeding headroom.
        appendUnsorted(id: cid, score: s)
        acceptedInBatch &+= 1

        if size >= bufferCapacity {
          pruneToTopC() // sets tau
        }

      case .adaptive:
        // In adaptive block phase until switch
        appendUnsorted(id: cid, score: s)
        acceptedInBatch &+= 1

        // Check occupancy periodically to keep overhead low.
        if (size & 63) == 0 {
          let occ = Float(size) / Float(C)
          if occ > opts.adaptiveThreshold {
            // Switch to heap: ensure we have exactly top‑C, then heapify (worst-at-root).
            if size > C { pruneToTopC() }
            heapifyWorstRoot()
            currentMode = .heap
            telemetry.modeSwitches &+= 1
          }
        }

      case .heap:
        // Heap keeps worst at root; accept only if better than worst (or size<C).
        if size < C {
          heapInsert(id: cid, score: s)
          acceptedInBatch &+= 1
        } else {
          // Compare against tau (root is worst).
          let worstID = ids[0]
          if isBetter(scoreA: s, idA: cid, scoreB: tau, idB: worstID) {
            replaceRoot(id: cid, score: s)
            acceptedInBatch &+= 1
          } else {
            telemetry.rejectedTau &+= 1
          }
        }
      }

      i &+= 1
    }

    return acceptedInBatch
  }

  // MARK: - Extract Top‑K (read-only; does not modify reservoir)

  /// Extracts top‑K results (best-first) into caller-provided buffers.
  /// K must be ≤ current `count`.
  ///
  /// Complexity: O(count log count) for full sort; acceptable since it's off the hot path.
  /// If you need partial select, you can adapt this to a k‑select then sort K.
  @inlinable
  public func extractTopK(
    k: Int,
    topScores outScores: UnsafeMutablePointer<Float>,
    topIDs outIDs: UnsafeMutablePointer<Int64>
  ) {
    precondition(k >= 0 && k <= size, "k must be in [0, count]")

    // Copy to local work buffers (read-only operation per spec).
    var ws = [Float](repeating: 0, count: size)
    var wi = [Int64](repeating: 0, count: size)

    // Use .update(from:count:) per project deprecation guidance.
    scores.withUnsafeBufferPointer { sp in
      ws.withUnsafeMutableBufferPointer { wp in
        wp.baseAddress!.update(from: sp.baseAddress!, count: size)
      }
    }
    ids.withUnsafeBufferPointer { ip in
      wi.withUnsafeMutableBufferPointer { wp in
        wp.baseAddress!.update(from: ip.baseAddress!, count: size)
      }
    }

    // Sort entire set by "better first" comparator (deterministic).
    ws.indices.sorted { a, b in
      isBetter(scoreA: ws[a], idA: wi[a], scoreB: ws[b], idB: wi[b])
    }.prefix(k).enumerated().forEach { (j, idx) in
      outScores[j] = ws[idx]
      outIDs[j] = wi[idx]
    }
  }

  // MARK: - Internal helpers (inlinable-visible)

  /// Append element without ordering (Block/Adaptive Block).
  @usableFromInline
  internal func appendUnsorted(id: Int64, score: Float) {
    // Ensure buffer room (contract: caller prunes before overflow).
    if size >= bufferCapacity {
      // Defensive: grow (should not happen if prune performed).
      let newCap = max(bufferCapacity &* 2, bufferCapacity &+ 1)
      scores.append(contentsOf: repeatElement(0, count: newCap - bufferCapacity))
      ids.append(contentsOf: repeatElement(0, count: newCap - bufferCapacity))
      bufferCapacity = newCap
    }
    ids[size] = id
    scores[size] = score
    size &+= 1
  }

  /// Prune buffer to top‑C via in-place quickselect partition.
  /// Post: first C elements contain the C best in unspecified order; `size = C`.
  /// Also updates `tau` (worst among top‑C).
  @usableFromInline
  internal func pruneToTopC() {
    let C = capacityC
    guard size > C else {
      // Not full yet; leave tau sentinel.
      return
    }
    quickselectTop(countKeep: C)
    size = C
    // Update tau: find worst within first C.
    var worstScore = scores[0]
    var worstID = ids[0]
    var j = 1
    while j < C {
      let s = scores[j]
      let id = ids[j]
      if isWorse(scoreA: s, idA: id, scoreB: worstScore, idB: worstID) {
        worstScore = s
        worstID = id
      }
      j &+= 1
    }
    tau = worstScore
    // Optional: move worst to index 0 to match heap root semantics? Not required here.
  }

  /// Build a worst-at-root heap in place over first `size` elements.
  /// For L2: max-heap (largest/worst at root). For IP/Cos: min-heap (smallest/worst at root).
  @usableFromInline
  internal func heapifyWorstRoot() {
    // Floyd's heapify: sift down from last non-leaf.
    var i = (size >> 1) &- 1
    while i >= 0 {
      heapSiftDown(from: i, heapSize: size)
      if i == 0 { break }
      i &-= 1
    }
    // Root is worst; set tau
    if size > 0 { tau = scores[0] } else { tau = worstSentinel(for: metric) }
  }

  /// Insert into heap (size < C).
  @usableFromInline
  internal func heapInsert(id: Int64, score: Float) {
    var i = size
    ids[i] = id
    scores[i] = score
    size &+= 1
    // Sift-up: bubble worse toward root so that parent is always ≥ child in "worse" order.
    while i > 0 {
      let p = (i &- 1) >> 1
      // If parent is already worse-or-equal than child, stop.
      if isWorse(scoreA: scores[p], idA: ids[p], scoreB: scores[i], idB: ids[i]) {
        break
      }
      swapAt(i, p)
      i = p
    }
    tau = scores[0]
  }

  /// Replace heap root (reservoir is full and new item is better than root).
  @usableFromInline
  internal func replaceRoot(id: Int64, score: Float) {
    ids[0] = id
    scores[0] = score
    heapSiftDown(from: 0, heapSize: size)
    tau = scores[0]
  }

  /// Sift-down: ensure worst-at-root heap property from `i` downward.
  @usableFromInline
  internal func heapSiftDown(from start: Int, heapSize: Int) {
    var i = start
    while true {
      let l = (i << 1) &+ 1
      if l >= heapSize { break }
      var worst = l
      let r = l &+ 1
      if r < heapSize {
        // pick worse child
        if isWorse(scoreA: scores[r], idA: ids[r], scoreB: scores[worst], idB: ids[worst]) {
          worst = r
        }
      }
      // If child 'worst' is worse than parent i, swap
      if isWorse(scoreA: scores[worst], idA: ids[worst], scoreB: scores[i], idB: ids[i]) {
        swapAt(i, worst)
        i = worst
      } else {
        break
      }
    }
  }

  /// Swap aligned (id, score) pairs at indices `a` and `b`.
  @usableFromInline
  internal func swapAt(_ a: Int, _ b: Int) {
    if a == b { return }
    let tmpS = scores[a]; scores[a] = scores[b]; scores[b] = tmpS
    let tmpI = ids[a]; ids[a] = ids[b]; ids[b] = tmpI
  }

  // MARK: - Selection: quickselect (top C best in first C positions)

  /// Partition buffer so that the first `countKeep` elements are the best ones (unordered).
  @usableFromInline
  internal func quickselectTop(countKeep k: Int) {
    var left = 0
    var right = size &- 1
    let target = k &- 1 // 0-based index

    while left <= right {
      let pivotIndex = medianOfThreeIndex(left, (left &+ right) >> 1, right)
      let newPivot = partitionAroundPivot(left: left, right: right, pivotIndex: pivotIndex)
      if newPivot == target { return }
      if target < newPivot {
        right = newPivot &- 1
      } else {
        left = newPivot &+ 1
      }
    }
  }

  /// Partition [left, right] so that elements better than pivot are left of `storeIdx`,
  /// worse to the right; returns final pivot index.
  @usableFromInline
  internal func partitionAroundPivot(left: Int, right: Int, pivotIndex: Int) -> Int {
    // Move pivot to end
    swapAt(pivotIndex, right)
    let pivotScore = scores[right]
    let pivotID = ids[right]

    var store = left
    var i = left
    while i < right {
      if isBetter(scoreA: scores[i], idA: ids[i], scoreB: pivotScore, idB: pivotID) {
        swapAt(i, store)
        store &+= 1
      }
      i &+= 1
    }
    // Move pivot to its final place
    swapAt(store, right)
    return store
  }

  /// Median-of-three index selection (deterministic).
  @usableFromInline
  internal func medianOfThreeIndex(_ a: Int, _ b: Int, _ c: Int) -> Int {
    let sa = scores[a], ia = ids[a]
    let sb = scores[b], ib = ids[b]
    let sc = scores[c], ic = ids[c]

    // Compare pairs with "better" to pick the median in terms of total order.
    // We need the pivot near the middle of the distribution to avoid worst-case.
    // We can rank a,b,c by better() and pick the middle.
    let ab = isBetter(scoreA: sa, idA: ia, scoreB: sb, idB: ib)
    let bc = isBetter(scoreA: sb, idA: ib, scoreB: sc, idB: ic)
    let ac = isBetter(scoreA: sa, idA: ia, scoreB: sc, idB: ic)

    // Cases enumerated to return median index.
    if ab {
      // a better than b
      if bc {
        // b better than c → a better than b better than c -> median is b
        return b
      } else {
        // b !> c  → compare a vs c
        return ac ? c : a
      }
    } else {
      // b better than a
      if ac {
        // a better than c -> b better than a better than c -> median is a
        return a
      } else {
        // a !> c -> compare b vs c
        return bc ? c : b
      }
    }
  }

  // MARK: - Ordering predicates

  /// Returns true if (scoreA, idA) is strictly **better** than (scoreB, idB)
  /// by the metric, with deterministic tie-breaking by ID when |Δ|≤eps.
  @usableFromInline
  internal func isBetter(scoreA: Float, idA: Int64, scoreB: Float, idB: Int64) -> Bool {
    let eps = CandidateReservoir.scoreEps
    switch metric {
      case .l2:
        // Smaller score is better
        let diff = scoreA - scoreB
        if diff < -eps { return true }
        if diff > eps { return false }
        // Tie: smaller ID wins
        return idA < idB
      case .innerProduct, .cosine:
        // Larger score is better
        let diff = scoreA - scoreB
        if diff > eps { return true }
        if diff < -eps { return false }
        // Tie: smaller ID wins
        return idA < idB
    }
  }

  /// Returns true if (scoreA, idA) is **worse** than (scoreB, idB).
  @usableFromInline
  internal func isWorse(scoreA: Float, idA: Int64, scoreB: Float, idB: Int64) -> Bool {
    isBetter(scoreA: scoreB, idA: idB, scoreB: scoreA, idB: idA)
  }

  /// Worst sentinel used for initial `tau`.
  @usableFromInline
  internal func worstSentinel(for metric: ReservoirMetric) -> Float {
    switch metric {
      case .l2:          return -.infinity  // any real score will be worse than this when heap is built
      case .innerProduct, .cosine:
        return .infinity
    }
  }
}

// MARK: - Convenience API

extension CandidateReservoir {
  /// Convenience: extract top‑K as Swift arrays (allocates result buffers).
  @inlinable
  public func extractTopK(k: Int) -> (ids: [Int64], scores: [Float]) {
    precondition(k >= 0 && k <= size, "k must be in [0, count]")
    var outI = [Int64](repeating: 0, count: k)
    var outS = [Float](repeating: 0, count: k)
    outS.withUnsafeMutableBufferPointer { sp in
      outI.withUnsafeMutableBufferPointer { ip in
        extractTopK(k: k, topScores: sp.baseAddress!, topIDs: ip.baseAddress!)
      }
    }
    return (outI, outS)
  }
}
