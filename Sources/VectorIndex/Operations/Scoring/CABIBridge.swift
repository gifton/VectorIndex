import Foundation

// Global C ABI entrypoints that forward to operations

@_cdecl("ip_f32_block")
public func ip_f32_block_abi(
    q: UnsafePointer<Float>?,
    xb: UnsafePointer<Float>?,
    n: Int32,
    d: Int32,
    out: UnsafeMutablePointer<Float>?
) {
    guard let qp = q, let xbp = xb, let outp = out else { return }
    IndexOps.Scoring.InnerProduct.run(q: qp, xb: xbp, n: Int(n), d: Int(d), out: outp)
}

@_cdecl("l2sqr_f32_block")
public func l2sqr_f32_block_abi(
    q: UnsafePointer<Float>?,
    xb: UnsafePointer<Float>?,
    n: Int32,
    d: Int32,
    out: UnsafeMutablePointer<Float>?,
    xb_norm: UnsafePointer<Float>?,
    q_norm: Float
) {
    guard let qp = q, let xbp = xb, let outp = out else { return }
    let qnOpt: Float? = q_norm.isNaN ? nil : q_norm
    IndexOps.Scoring.L2Sqr.run(q: qp, xb: xbp, n: Int(n), d: Int(d), out: outp, xb_norm: xb_norm, q_norm: qnOpt)
}
