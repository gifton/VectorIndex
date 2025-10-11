import Foundation
@testable import VectorIndex

print("Starting quick PQ training test...")

// Minimal test data
let n: Int64 = 100
let d = 32
let m = 2
let ks = 16

var x = [Float](repeating: 0, count: Int(n) * d)
for i in 0..<(Int(n) * d) {
    x[i] = Float.random(in: -1...1)
}

var codebooks = [Float]()
var norms: [Float]? = []

print("Training PQ codebook: n=\(n), d=\(d), m=\(m), ks=\(ks)")

let start = Date()
do {
    let stats = try pq_train_f32(
        x: x, n: n, d: d, m: m, ks: ks,
        codebooksOut: &codebooks,
        centroidNormsOut: &norms
    )
    let elapsed = Date().timeIntervalSince(start)

    print("✅ SUCCESS!")
    print("   Time: \(elapsed)s")
    print("   Distortion: \(stats.distortion)")
    print("   Codebooks: \(codebooks.count) values")
    print("   Norms: \(norms?.count ?? 0) values")
    print("   Iterations: \(stats.itersPerSubspace)")

    // Validate
    assert(codebooks.count == m * ks * (d/m), "Codebook size mismatch")
    assert(norms?.count == m * ks, "Norms size mismatch")
    assert(stats.distortion > 0, "Invalid distortion")

    print("✅ All validations passed!")
} catch {
    print("❌ ERROR: \(error)")
}
