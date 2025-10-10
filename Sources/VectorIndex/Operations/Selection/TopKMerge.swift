import Foundation

public extension IndexOps.Selection {
    private struct MergeNode { let list: Int; var index: Int; let score: Float; let id: Int32 }
    /// K-way merge of multiple sorted best→worst lists into a single top-K, deterministic and stable by id.
    /// - Parameters:
    ///   - lists: Each list must be sorted best→worst according to `ordering`.
    ///   - k: Number of results to output (clamped to total available).
    ///   - ordering: HeapOrdering semantics for comparison.
    /// - Returns: A merged list of up to K items sorted best→worst.
    static func mergeTopK(
        lists: [[(score: Float, id: Int32)]],
        k: Int,
        ordering: HeapOrdering
    ) -> [(score: Float, id: Int32)] {
        let L = lists.count
        guard k > 0, L > 0 else { return [] }

        // Min- or max-heap of heads depending on ordering; we always want to pop the current "best" item.
        var heap: [MergeNode] = []
        heap.reserveCapacity(L)

        // Initialize heap with the head of each non-empty list
        for l in 0..<L {
            if !lists[l].isEmpty {
                let h = lists[l][0]
                heapAppend(&heap, MergeNode(list: l, index: 0, score: h.score, id: h.id), ordering)
            }
        }

        var out: [(score: Float, id: Int32)] = []
        out.reserveCapacity(k)

        while !heap.isEmpty && out.count < k {
            let best = heapPop(&heap, ordering)
            out.append((best.score, best.id))
            let nextIdx = best.index + 1
            if nextIdx < lists[best.list].count {
                let nxt = lists[best.list][nextIdx]
                heapAppend(&heap, MergeNode(list: best.list, index: nextIdx, score: nxt.score, id: nxt.id), ordering)
            }
        }
        return out
    }

    /// Merge K-way from TopKHeap inputs. Heaps may be unsorted; we extract sorted snapshots first.
    /// - Parameters:
    ///   - heaps: Input heaps built with consistent ordering.
    ///   - k: Desired number of results.
    /// - Returns: Up to K merged results sorted best→worst.
    static func mergeTopK(
        heaps: [TopKHeap],
        k: Int
    ) -> [(score: Float, id: Int32)] {
        guard let ord = heaps.first?.ordering else { return [] }
        // Extract best→worst from each heap
        var lists: [[(score: Float, id: Int32)]] = []
        lists.reserveCapacity(heaps.count)
        for h in heaps { lists.append(h.extractSorted()) }
        return mergeTopK(lists: lists, k: k, ordering: ord)
    }

    // MARK: - Internal binary heap helpers over Node

    @inline(__always)
    private static func better(_ a: MergeNode, _ b: MergeNode, _ ordering: HeapOrdering) -> Bool {
        // Use ordering.isBetter (score,id). If identical score+id from different lists, break by smaller list index for determinism.
        if ordering.isBetter(a.score, a.id, than: b.score, b.id) { return true }
        if a.score == b.score && a.id == b.id { return a.list < b.list }
        return false
    }

    @inline(__always)
    private static func heapAppend(_ heap: inout [MergeNode], _ node: MergeNode, _ ordering: HeapOrdering) {
        heap.append(node)
        var i = heap.count - 1
        while i > 0 {
            let p = (i - 1) >> 1
            if better(heap[i], heap[p], ordering) {
                heap.swapAt(i, p); i = p
            } else { break }
        }
    }

    @inline(__always)
    private static func heapPop(_ heap: inout [MergeNode], _ ordering: HeapOrdering) -> MergeNode {
        let top = heap[0]
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var i = 0
            while true {
                let l = (i << 1) &+ 1
                if l >= heap.count { break }
                let r = l &+ 1
                var b = l
                if r < heap.count, better(heap[r], heap[l], ordering) { b = r }
                if better(heap[b], heap[i], ordering) { heap.swapAt(b, i); i = b } else { break }
            }
        }
        return top
    }
}
