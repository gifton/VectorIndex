import XCTest
@testable import VectorIndex

final class HNSWParamSweepTests: XCTestCase {
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

    func testRecallImprovesWithEfSearch() async throws {
        let dim=32, n=300, q=20, k=5
        let data = gen(n, dim: dim, seed: 101)
        let queries = gen(q, dim: dim, seed: 202)
        let ids = (0..<n).map{ "id\($0)" }

        let low = HNSWIndex(dimension: dim, metric: .euclidean, config: .init(m: 12, efConstruction: 100, efSearch: 16))
        for i in 0..<n { try await low.insert(id: ids[i], vector: data[i], metadata: nil) }

        let high = HNSWIndex(dimension: dim, metric: .euclidean, config: .init(m: 12, efConstruction: 100, efSearch: 64))
        for i in 0..<n { try await high.insert(id: ids[i], vector: data[i], metadata: nil) }

        var rLow:[Float]=[], rHigh:[Float]=[]
        for qi in 0..<q {
            let truth = flatTopK(queries[qi], data: data, ids: ids, k: k)
            let aLow = try await low.search(query: queries[qi], k: k, filter: nil).map{ $0.id }
            let aHigh = try await high.search(query: queries[qi], k: k, filter: nil).map{ $0.id }
            rLow.append(recall(k, truth, aLow)); rHigh.append(recall(k, truth, aHigh))
        }
        let avgLow = rLow.reduce(0,+)/Float(rLow.count)
        let avgHigh = rHigh.reduce(0,+)/Float(rHigh.count)
        XCTAssertGreaterThanOrEqual(avgHigh, avgLow, "Recall with higher efSearch should be >= lower efSearch")
    }

    func testRecallImprovesWithEfConstruction() async throws {
        let dim=32, n=300, q=20, k=5
        let data = gen(n, dim: dim, seed: 303)
        let queries = gen(q, dim: dim, seed: 404)
        let ids = (0..<n).map{ "id\($0)" }

        let low = HNSWIndex(dimension: dim, metric: .euclidean, config: .init(m: 12, efConstruction: 50, efSearch: 32))
        for i in 0..<n { try await low.insert(id: ids[i], vector: data[i], metadata: nil) }

        let high = HNSWIndex(dimension: dim, metric: .euclidean, config: .init(m: 12, efConstruction: 200, efSearch: 32))
        for i in 0..<n { try await high.insert(id: ids[i], vector: data[i], metadata: nil) }

        var rLow:[Float]=[], rHigh:[Float]=[]
        for qi in 0..<q {
            let truth = flatTopK(queries[qi], data: data, ids: ids, k: k)
            let aLow = try await low.search(query: queries[qi], k: k, filter: nil).map{ $0.id }
            let aHigh = try await high.search(query: queries[qi], k: k, filter: nil).map{ $0.id }
            rLow.append(recall(k, truth, aLow)); rHigh.append(recall(k, truth, aHigh))
        }
        let avgLow = rLow.reduce(0,+)/Float(rLow.count)
        let avgHigh = rHigh.reduce(0,+)/Float(rHigh.count)
        XCTAssertGreaterThanOrEqual(avgHigh, avgLow, "Recall with higher efConstruction should be >= lower efConstruction")
    }
}
