import XCTest
@testable import VectorIndex

final class IVFProbeMonotonicTests: XCTestCase {
    struct LCG { var s: UInt64; mutating func next()->UInt64{ s = 2862933555777941757 &* s &+ 3037000493; return s }; mutating func f()->Float{ Float(next()>>11)/Float(1<<53) } }
    func gen(_ n: Int, dim: Int, seed: UInt64) -> [[Float]] {
        var r = LCG(s: seed); var arr: [[Float]] = []
        for _ in 0..<n { var v = (0..<dim).map{ _ in r.f()*2-1 }; let norm = sqrt(v.reduce(0){$0+$1*$1}); if norm>0 { v = v.map{$0/norm} }; arr.append(v) }
        return arr
    }
    func flatTopK(_ q: [Float], data: [[Float]], ids: [String], k: Int) -> [String] {
        var s: [(String,Float)] = []; s.reserveCapacity(data.count)
        for (i,v) in data.enumerated(){ var sum:Float=0; for j in 0..<q.count { let d=q[j]-v[j]; sum+=d*d }; s.append((ids[i], sqrt(sum))) }
        s.sort{ $0.1 < $1.1 }; return Array(s.prefix(k).map{$0.0})
    }
    func recall(_ k:Int, _ t:[String], _ a:[String])->Float{ Float(Set(t.prefix(k)).intersection(Set(a.prefix(k))).count)/Float(k) }

    func testRecallImprovesWithNprobe() async throws {
        let dim=32, n=300, q=20, k=5
        let data = gen(n, dim: dim, seed: 17)
        let queries = gen(q, dim: dim, seed: 19)
        let ids = (0..<n).map{ "id\($0)" }

        let ivf1 = IVFIndex(dimension: dim, metric: .euclidean, config: .init(nlist: 32, nprobe: 1))
        for i in 0..<n { try await ivf1.insert(id: ids[i], vector: data[i], metadata: nil) }
        try await ivf1.optimize()

        let ivf2 = IVFIndex(dimension: dim, metric: .euclidean, config: .init(nlist: 32, nprobe: 8))
        for i in 0..<n { try await ivf2.insert(id: ids[i], vector: data[i], metadata: nil) }
        try await ivf2.optimize()

        var r1:[Float]=[], r2:[Float]=[]
        for qi in 0..<q {
            let truth = flatTopK(queries[qi], data: data, ids: ids, k: k)
            let a1 = try await ivf1.search(query: queries[qi], k: k, filter: nil).map{ $0.id }
            let a2 = try await ivf2.search(query: queries[qi], k: k, filter: nil).map{ $0.id }
            r1.append(recall(k, truth, a1)); r2.append(recall(k, truth, a2))
        }
        let avg1 = r1.reduce(0,+)/Float(r1.count)
        let avg2 = r2.reduce(0,+)/Float(r2.count)
        XCTAssertGreaterThanOrEqual(avg2, avg1, "Recall with higher nprobe should be >= lower nprobe")
    }
}
