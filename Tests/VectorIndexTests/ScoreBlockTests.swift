import XCTest
@testable import VectorIndex
import VectorCore

// First direct coverage of `IndexOps.Scoring.ScoreBlock.run` (P6c, Task 11).
//
// Prior to this task the kernel had zero dedicated tests: it was only
// exercised indirectly through callers (IVFIndex centroid probing,
// ExactRerank's cosine path, etc). Each test here hand-computes the expected
// per-row score with scalar reference math independent of any VectorIndex
// kernel, then compares against `ScoreBlock.run`'s output.
//
// euclidean/dotProduct/cosine hit ScoreBlock's explicit switch cases
// (L2Sqr/InnerProduct/Cosine kernels). manhattan has no explicit case and
// falls through to the `default:` branch -- the allocating scalar fallback
// this task hoists (`qArr`/`tmp` moved above the per-row loop). Running this
// suite both before and after the hoist proves the hoist is behavior-
// preserving.
final class ScoreBlockTests: XCTestCase {

    private let n = 32
    private let d = 16

    /// Deterministic LCG filler (same recipe as ResidualKernelTests) so the
    /// block contents are fixed across runs without relying on SystemRandom.
    private func seededVectors(count: Int, seed: UInt64) -> [Float] {
        var s = seed
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            s = 2862933555777941757 &* s &+ 3037000493
            let u = Float(s >> 40) / Float(1 << 24) // [0, 1)
            out[i] = u * 2 - 1                      // [-1, 1)
        }
        return out
    }

    private func makeQueryAndBlock() -> (q: [Float], xb: [Float]) {
        let q = seededVectors(count: d, seed: 0x9E3779B97F4A7C15)
        let xb = seededVectors(count: n * d, seed: 0xD1B54A32D192ED03)
        return (q, xb)
    }

    private func runScoreBlock(q: [Float], xb: [Float], metric: SupportedDistanceMetric) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        q.withUnsafeBufferPointer { qp in
            xb.withUnsafeBufferPointer { xbp in
                out.withUnsafeMutableBufferPointer { op in
                    IndexOps.Scoring.ScoreBlock.run(
                        q: qp.baseAddress!, xb: xbp.baseAddress!, n: n, d: d,
                        metric: metric, out: op.baseAddress!)
                }
            }
        }
        return out
    }

    // MARK: - Euclidean (explicit case -> L2Sqr; writes L2^2, no sqrt)

    func testEuclideanMatchesScalarReference() {
        let (q, xb) = makeQueryAndBlock()
        let scores = runScoreBlock(q: q, xb: xb, metric: .euclidean)

        for i in 0..<n {
            var expected: Float = 0
            for j in 0..<d {
                let diff = q[j] - xb[i * d + j]
                expected += diff * diff
            }
            XCTAssertEqual(scores[i], expected, accuracy: 1e-4, "row \(i)")
        }
    }

    // MARK: - Dot product (explicit case -> InnerProduct; raw inner product)

    func testDotProductMatchesScalarReference() {
        let (q, xb) = makeQueryAndBlock()
        let scores = runScoreBlock(q: q, xb: xb, metric: .dotProduct)

        for i in 0..<n {
            var expected: Float = 0
            for j in 0..<d {
                expected += q[j] * xb[i * d + j]
            }
            XCTAssertEqual(scores[i], expected, accuracy: 1e-4, "row \(i)")
        }
    }

    // MARK: - Cosine (explicit case -> Cosine; on-the-fly norms, [-1,1])

    func testCosineMatchesScalarReference() {
        let (q, xb) = makeQueryAndBlock()
        let scores = runScoreBlock(q: q, xb: xb, metric: .cosine)

        let epsilon: Float = 1e-12
        var qSumSq: Float = 0
        for j in 0..<d { qSumSq += q[j] * q[j] }
        let qInv = 1.0 / (qSumSq.squareRoot() + epsilon)

        for i in 0..<n {
            var dot: Float = 0
            var xSumSq: Float = 0
            for j in 0..<d {
                let x = xb[i * d + j]
                dot += q[j] * x
                xSumSq += x * x
            }
            let xInv = 1.0 / (xSumSq.squareRoot() + epsilon)
            let expected = max(-1, min(1, dot * qInv * xInv))
            XCTAssertEqual(scores[i], expected, accuracy: 1e-4, "row \(i)")
        }
    }

    // MARK: - Manhattan (no explicit case -> `default:` scalar fallback)

    func testManhattanMatchesScalarReference_ExercisesDefaultFallback() {
        let (q, xb) = makeQueryAndBlock()
        let scores = runScoreBlock(q: q, xb: xb, metric: .manhattan)

        for i in 0..<n {
            var expected: Float = 0
            for j in 0..<d {
                expected += abs(q[j] - xb[i * d + j])
            }
            XCTAssertEqual(scores[i], expected, accuracy: 1e-4, "row \(i)")
        }
    }
}
