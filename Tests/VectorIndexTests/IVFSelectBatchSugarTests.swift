import XCTest
@testable import VectorIndex

final class IVFSelectBatchSugarTests: XCTestCase {

    func testL2ShapeAndParity() {
        let b = 6
        let d = 64
        let kc = 300
        let nprobe = 10

        // Random data (stable seed via deterministic RNG not required; just parity within run)
        let Q = (0..<(b*d)).map { _ in Float.random(in: -1...1) }
        let centroids = (0..<(kc*d)).map { _ in Float.random(in: -1...1) }

        // Sugar path
        let (ids2D, scores2D) = IndexOps.Batch.ivfSelectNprobe(
            Q: Q, b: b, d: d,
            centroids: centroids, kc: kc,
            metric: .l2, nprobe: nprobe,
            opts: IVFSelectOpts(),
            gatherScores: true
        )
        XCTAssertEqual(ids2D.count, b)
        XCTAssertEqual(ids2D[0].count, nprobe)
        XCTAssertNotNil(scores2D)
        XCTAssertEqual(scores2D!.count, b)
        XCTAssertEqual(scores2D![0].count, nprobe)

        // Parity vs single-query API
        for i in 0..<b {
            let q = Array(Q[(i*d)..<((i+1)*d)])
            var singleIDs = [Int32](repeating: -1, count: nprobe)
            var singleScores: [Float]? = [Float](repeating: 0, count: nprobe)
            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .l2, nprobe: nprobe,
                opts: IVFSelectOpts(),
                listIDsOut: &singleIDs,
                listScoresOut: &singleScores
            )
            XCTAssertEqual(ids2D[i], singleIDs, "IDs mismatch for query \(i)")
            if let sc = scores2D, let ss = singleScores {
                for j in 0..<nprobe {
                    XCTAssertEqual(sc[i][j], ss[j], accuracy: 1e-4)
                }
            }
        }
    }

    func testCosineParityNoScores() {
        let b = 4, d = 32, kc = 200, nprobe = 8
        let Q = (0..<(b*d)).map { _ in Float.random(in: -1...1) }
        let centroids = (0..<(kc*d)).map { _ in Float.random(in: -1...1) }

        // Sugar without scores
        let (ids2D, scores2D) = IndexOps.Batch.ivfSelectNprobe(
            Q: Q, b: b, d: d,
            centroids: centroids, kc: kc,
            metric: .cosine, nprobe: nprobe,
            gatherScores: false
        )
        XCTAssertNil(scores2D)

        // Check parity of IDs only vs single-query path
        for i in 0..<b {
            let q = Array(Q[(i*d)..<((i+1)*d)])
            var singleIDs = [Int32](repeating: -1, count: nprobe)
            var nilScores: [Float]? = nil
            ivf_select_nprobe_f32(
                q: q, d: d, centroids: centroids, kc: kc,
                metric: .cosine, nprobe: nprobe,
                opts: IVFSelectOpts(),
                listIDsOut: &singleIDs,
                listScoresOut: &nilScores
            )
            XCTAssertEqual(ids2D[i], singleIDs, "IDs mismatch for query \(i)")
        }
    }
}

