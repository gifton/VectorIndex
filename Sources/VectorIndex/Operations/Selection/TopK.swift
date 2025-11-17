import Foundation
import Dispatch
import VectorCore

public extension IndexOps {
    enum Selection {
        // MARK: - Heap Ordering
        public enum HeapOrdering {
            case min, max
            @inline(__always)
            func shouldReplace(_ newScore: Float, _ newId: Int32, rootScore: Float, rootId: Int32) -> Bool {
                switch self {
                case .min: return (newScore < rootScore) || (newScore == rootScore && newId < rootId)
                case .max: return (newScore > rootScore) || (newScore == rootScore && newId < rootId)
                }
            }
            @inline(__always)
            func isWorse(_ aScore: Float, _ aId: Int32, than bScore: Float, _ bId: Int32) -> Bool {
                switch self {
                case .min: return (aScore > bScore) || (aScore == bScore && aId > bId)
                case .max: return (aScore < bScore) || (aScore == bScore && aId > bId)
                }
            }
            @inline(__always)
            func isBetter(_ aScore: Float, _ aId: Int32, than bScore: Float, _ bId: Int32) -> Bool {
                switch self {
                case .min: return (aScore < bScore) || (aScore == bScore && aId < bId)
                case .max: return (aScore > bScore) || (aScore == bScore && aId < bId)
                }
            }
        }

        public struct TopKTelemetry {
            public let algorithm: TopKConfig.Algorithm
            public let candidatesProcessed: Int
            public let k: Int
            public let comparisons: Int
            public let heapPushes: Int
            public let siftOperations: Int
            public let executionTimeNanos: UInt64
            public var throughputCandidatesPerSec: Double {
                let sec = Double(executionTimeNanos) / 1e9
                return sec > 0 ? Double(candidatesProcessed) / sec : 0
            }
        }
        public enum TopKTelemetryRecorder {
            public nonisolated(unsafe) static var sink: ((TopKTelemetry) -> Void)?
            @inline(__always) public static func record(_ t: TopKTelemetry) { sink?(t) }
        }

        @usableFromInline struct HeapCounters { @usableFromInline var pushes=0, sifts=0, comps=0 }

        // MARK: Fixed-size Top-K Heap (SoA)
        public struct TopKHeap {
            public let ordering: HeapOrdering
            public let capacity: Int
            public internal(set) var count: Int
            private var scores: UnsafeMutablePointer<Float>
            private var ids: UnsafeMutablePointer<Int32>
            public init(capacity k: Int, ordering: HeapOrdering) {
                self.ordering = ordering
                self.capacity = max(0, k)
                self.count = 0
                let c = max(self.capacity, 1)
                self.scores = _allocateAligned(count: c, align: 64, of: Float.self)
                self.ids    = _allocateAligned(count: c, align: 64, of: Int32.self)
            }
            public mutating func clear() { count = 0 }
            public func deallocate() {
                free(UnsafeMutableRawPointer(scores))
                free(UnsafeMutableRawPointer(ids))
            }
            public var root: (score: Float, id: Int32)? { count > 0 ? (scores[0], ids[0]) : nil }
            public mutating func push(score: Float, id: Int32) { var c=HeapCounters(); _push(score: score, id: id, counters: &c) }
            public func extractSorted() -> [(score: Float, id: Int32)] {
                var out = [(Float, Int32)](); out.reserveCapacity(count)
                for i in 0..<count { out.append((scores[i], ids[i])) }
                out.sort { a, b in ordering.isBetter(a.0, a.1, than: b.0, b.1) }
                return out
            }
            @inline(__always) mutating func heapify(counters: inout HeapCounters) { var i=(count/2)-1; while i>=0 { counters.sifts &+= _siftDown(from: i); if i==0 {break}; i &-= 1 } }
            @inline(__always) mutating func replaceRoot(score: Float, id: Int32, counters: inout HeapCounters) { scores[0]=score; ids[0]=id; counters.sifts &+= _siftDown(from: 0); counters.pushes &+= 1 }
            @inline(__always) mutating func _push(score: Float, id: Int32, counters: inout HeapCounters) {
                if capacity == 0 { return }
                if count < capacity { scores[count]=score; ids[count]=id; counters.sifts &+= _siftUp(from: count); count &+= 1; counters.pushes &+= 1; return }
                if ordering.shouldReplace(score, id, rootScore: scores[0], rootId: ids[0]) { replaceRoot(score: score, id: id, counters: &counters) }
            }
            @inline(__always) private mutating func _siftUp(from idx: Int) -> Int {
                guard idx>0 else { return 0 }
                var sifts=0; var child=idx; let score=scores[child]; let id=ids[child]
                while child>0 { let parent=(child-1)>>1; let pScore=scores[parent], pId=ids[parent]
                    if ordering.isWorse(score, id, than: pScore, pId) { scores[child]=pScore; ids[child]=pId; child=parent; sifts &+= 1 } else { break }
                }
                scores[child]=score; ids[child]=id; return sifts
            }
            @inline(__always) private mutating func _siftDown(from idx: Int) -> Int {
                var sifts=0; var i=idx; let s=scores[i], id=ids[i]
                while true { let left=(i<<1)&+1; if left>=count { break }; let right=left&+1
                    var wIdx=left; var wScore=scores[left], wId=ids[left]
                    if right<count { let rScore=scores[right], rId=ids[right]; if ordering.isWorse(rScore, rId, than: wScore, wId) { wIdx=right; wScore=rScore; wId=rId } }
                    if ordering.isWorse(wScore, wId, than: s, id) { scores[i]=wScore; ids[i]=wId; i=wIdx; sifts &+= 1 } else { break }
                }
                scores[i]=s; ids[i]=id; return sifts
            }
            @inline(__always) internal mutating func _directWrite(at i: Int, score: Float, id: Int32) { scores[i]=score; ids[i]=id }
        }

        @inline(__always) private static func _allocateAligned<T>(count: Int, align: Int, of: T.Type) -> UnsafeMutablePointer<T> {
            var p: UnsafeMutableRawPointer?
            let r = posix_memalign(&p, align, max(count, 1)*MemoryLayout<T>.stride)
            precondition(r == 0 && p != nil, "posix_memalign failed")
            return p!.bindMemory(to: T.self, capacity: max(count, 1))
        }

        public struct TopKConfig: Sendable {
            public enum Algorithm: Sendable { case streaming, hybrid }
            public let enableTelemetry: Bool
            public let forceAlgorithm: Algorithm?
            public let hybridThreshold: Int
            public static let `default` = TopKConfig(enableTelemetry: false, forceAlgorithm: nil, hybridThreshold: 16_384)
            public init(enableTelemetry: Bool = false, forceAlgorithm: Algorithm? = nil, hybridThreshold: Int = 16_384) {
                self.enableTelemetry = enableTelemetry; self.forceAlgorithm = forceAlgorithm; self.hybridThreshold = hybridThreshold
            }
        }

        // MARK: Core API
        public static func selectTopK(
            scores: UnsafePointer<Float>, ids: UnsafePointer<Int32>?, count n: Int, k: Int,
            ordering: HeapOrdering, config: TopKConfig = .default
        ) -> TopKHeap {
            let kEff = max(0, min(k, n))
            var counters = HeapCounters()
            let t0 = config.enableTelemetry ? DispatchTime.now().uptimeNanoseconds : 0
            let algo: TopKConfig.Algorithm = {
                if let forced = config.forceAlgorithm { return forced }
                if n > config.hybridThreshold && kEff > 0 && kEff < n/100 { return .hybrid }
                return .streaming
            }()
            let heap: TopKHeap
            switch algo {
            case .streaming:
                heap = _streaming(scores: scores, ids: ids, n: n, k: kEff, ordering: ordering, counters: &counters)
            case .hybrid:
                heap = _hybrid(scores: scores, ids: ids, n: n, k: kEff, ordering: ordering, counters: &counters)
            }
            if config.enableTelemetry {
                let t1 = DispatchTime.now().uptimeNanoseconds
                TopKTelemetryRecorder.record(TopKTelemetry(algorithm: algo, candidatesProcessed: n, k: kEff, comparisons: counters.comps, heapPushes: counters.pushes, siftOperations: counters.sifts, executionTimeNanos: t1 &- t0))
            }
            return heap
        }

        public static func selectTopK_streaming(scores s: UnsafePointer<Float>, ids idsOpt: UnsafePointer<Int32>?, count n: Int, k: Int, ordering: HeapOrdering) -> TopKHeap {
            var tmp=HeapCounters(); return _streaming(scores: s, ids: idsOpt, n: n, k: k, ordering: ordering, counters: &tmp)
        }
        @inline(__always) private static func _streaming(scores s: UnsafePointer<Float>, ids idsOpt: UnsafePointer<Int32>?, n: Int, k: Int, ordering: HeapOrdering, counters: inout HeapCounters) -> TopKHeap {
            var heap = TopKHeap(capacity: k, ordering: ordering); guard k>0 && n>0 else { return heap }
            let m = min(k, n)
            for i in 0..<m { let id = idsOpt?.advanced(by: i).pointee ?? Int32(i); heap._directWrite(at: i, score: s[i], id: id) }
            heap.count = m; heap.heapify(counters: &counters); counters.pushes &+= m
            if m < n { for i in m..<n { let id = idsOpt?.advanced(by: i).pointee ?? Int32(i); counters.comps &+= 1; let r = heap.root!; if ordering.shouldReplace(s[i], id, rootScore: r.score, rootId: r.id) { heap.replaceRoot(score: s[i], id: id, counters: &counters) } } }
            return heap
        }

        public static func selectTopK_hybrid(scores s: UnsafePointer<Float>, ids idsOpt: UnsafePointer<Int32>?, count n: Int, k: Int, ordering: HeapOrdering) -> TopKHeap {
            var tmp=HeapCounters(); return _hybrid(scores: s, ids: idsOpt, n: n, k: k, ordering: ordering, counters: &tmp)
        }
        @inline(__always) private static func _hybrid(scores s: UnsafePointer<Float>, ids idsOpt: UnsafePointer<Int32>?, n: Int, k: Int, ordering: HeapOrdering, counters: inout HeapCounters) -> TopKHeap {
            var heap = TopKHeap(capacity: k, ordering: ordering); guard k>0 && n>0 else { return heap }
            var ws=[Float](repeating: 0, count: n); var wi=[Int32](repeating: 0, count: n)
            for i in 0..<n { ws[i]=s[i]; wi[i]=idsOpt?.advanced(by: i).pointee ?? Int32(i) }
            _quickselectTopK(&ws, &wi, k, ordering: ordering, counters: &counters)
            for i in 0..<k { heap._directWrite(at: i, score: ws[i], id: wi[i]) }
            heap.count=k; heap.heapify(counters: &counters); counters.pushes &+= k; return heap
        }

        @inline(__always) private static func _medianOfThree(_ scores: [Float], _ ids: [Int32], _ a: Int, _ b: Int, _ c: Int, ordering: HeapOrdering) -> Int {
            func better(_ i: Int, _ j: Int) -> Bool { ordering.isBetter(scores[i], ids[i], than: scores[j], ids[j]) }
            let ab=better(a, b), ac=better(a, c), bc=better(b, c)
            if ab { if ac { return a } ; return bc ? b : c } else { if !bc { return b } ; return ac ? a : c }
        }
        @inline(__always) private static func _partition(_ scores: inout [Float], _ ids: inout [Int32], left: Int, right: Int, pivotIdx: Int, ordering: HeapOrdering, counters: inout HeapCounters) -> Int {
            let pScore=scores[pivotIdx], pId=ids[pivotIdx]; scores.swapAt(pivotIdx, right); ids.swapAt(pivotIdx, right)
            var store=left; for i in left..<right { counters.comps &+= 1; if ordering.isBetter(scores[i], ids[i], than: pScore, pId) { scores.swapAt(i, store); ids.swapAt(i, store); store &+= 1 } }
            scores.swapAt(store, right); ids.swapAt(store, right); return store
        }
        @inline(__always) private static func _quickselectTopK(_ scores: inout [Float], _ ids: inout [Int32], _ k: Int, ordering: HeapOrdering, counters: inout HeapCounters) {
            var left=0, right=scores.count-1; let target=max(0, min(k, scores.count))
            while left<=right { if left==right { break } ; let mid=(left+right)>>1; let piv=_medianOfThree(scores, ids, left, mid, right, ordering: ordering); let pf=_partition(&scores, &ids, left: left, right: right, pivotIdx: piv, ordering: ordering, counters: &counters); if pf==target { break } ; if pf<target { left=pf+1 } else { if pf==0 { break }; right=pf-1 } }
        }

        // Convenience using project metric
        public static func ordering(for metric: SupportedDistanceMetric) -> HeapOrdering {
            switch metric { case .euclidean, .manhattan, .chebyshev: return .min; case .dotProduct, .cosine: return .max }
        }
    }
}
