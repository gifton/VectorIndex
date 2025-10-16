import Foundation
import VectorIndex

// Simple micro-benchmark for L2^2 kernel (#01)

struct Cfg {
    var n: Int = 20000
    var d: Int = 1024
    var threads: Int = 0 // 0 = auto
    var trials: Int = 5
    var algo: String = "auto" // auto|direct|dot
    var seed: UInt64 = 42
}

func parseArgs() -> Cfg {
    var c = Cfg()
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    func next() -> String? { guard i+1 < args.count else { return nil }; i += 1; return args[i] }
    while i < args.count {
        switch args[i] {
        case "--n": if let v = next(), let iv = Int(v) { c.n = iv }
        case "--d": if let v = next(), let iv = Int(v) { c.d = iv }
        case "--threads": if let v = next(), let iv = Int(v) { c.threads = iv }
        case "--trials": if let v = next(), let iv = Int(v) { c.trials = iv }
        case "--algo": if let v = next() { c.algo = v }
        case "--seed": if let v = next(), let uv = UInt64(v) { c.seed = uv }
        default: break
        }
        i += 1
    }
    return c
}

@inline(__always)
func randFloats(count: Int, seed: inout UInt64) -> [Float] {
    func next(_ s: inout UInt64) -> UInt64 { s = 2862933555777941757 &* s &+ 3037000493; return s }
    var v = [Float](repeating: 0, count: count)
    for i in 0..<count { v[i] = Float(next(&seed) >> 11) / Float(1<<53) * 2 - 1 }
    return v
}

@inline(__always)
func norms(xb: UnsafePointer<Float>, n: Int, d: Int) -> [Float] {
    var out = [Float](repeating: 0, count: n)
    for i in 0..<n {
        var s: Float = 0
        let row = xb + i*d
        for j in 0..<d { let v = row[j]; s += v*v }
        out[i] = s
    }
    return out
}

func runTrial(c: Cfg, q: UnsafePointer<Float>, xb: UnsafePointer<Float>, xbNorm: UnsafePointer<Float>?, qNorm: Float, opts: inout L2SqrOpts, out: UnsafeMutablePointer<Float>) -> Double {
    let t0 = DispatchTime.now().uptimeNanoseconds
    l2sqr_f32_block(q, xb, c.n, c.d, out, xbNorm, qNorm, &opts)
    let t1 = DispatchTime.now().uptimeNanoseconds
    return Double(t1 &- t0) / 1e9
}

func main() {
    let cfg = parseArgs()
    print("L2Sqr microbench — n=\(cfg.n) d=\(cfg.d) trials=\(cfg.trials) threads=\(cfg.threads) algo=\(cfg.algo)")

    // Allocate data
    var seed = cfg.seed
    var q = randFloats(count: cfg.d, seed: &seed)
    var xb = randFloats(count: cfg.n * cfg.d, seed: &seed)
    let outCount = cfg.n
    var out = [Float](repeating: 0, count: outCount)

    q.withUnsafeMutableBufferPointer { qbuf in
        xb.withUnsafeMutableBufferPointer { xbbuf in
            out.withUnsafeMutableBufferPointer { obuf in
                // Precompute norms
                let qn: Float = qbuf.baseAddress!.withMemoryRebound(to: Float.self, capacity: cfg.d) { qp in
                    var s: Float = 0
                    for j in 0..<cfg.d { let v = qp[j]; s += v*v }
                    return s
                }
                let xbn = norms(xb: xbbuf.baseAddress!, n: cfg.n, d: cfg.d)
                xbn.withUnsafeBufferPointer { xbnBuf in
                    var opts = L2SqrOpts.default
                    opts.numThreads = Int32(cfg.threads)
                    switch cfg.algo {
                    case "direct": opts.algo = .direct
                    case "dot": opts.algo = .dotTrick
                    default: opts.algo = .auto
                    }

                    // Warmup
                    _ = runTrial(c: cfg, q: qbuf.baseAddress!, xb: xbbuf.baseAddress!, xbNorm: xbnBuf.baseAddress!, qNorm: qn, opts: &opts, out: obuf.baseAddress!)

                    // Measure
                    var times: [Double] = []
                    times.reserveCapacity(cfg.trials)
                    for _ in 0..<cfg.trials {
                        let sec = runTrial(c: cfg, q: qbuf.baseAddress!, xb: xbbuf.baseAddress!, xbNorm: xbnBuf.baseAddress!, qNorm: qn, opts: &opts, out: obuf.baseAddress!)
                        times.append(sec)
                    }
                    let avg = times.reduce(0,+)/Double(times.count)
                    let bytes = Double(cfg.n * cfg.d * MemoryLayout<Float>.stride + cfg.d*MemoryLayout<Float>.stride + cfg.n*MemoryLayout<Float>.stride)
                    let throughput = Double(cfg.n) / avg
                    let bw = bytes/avg / 1e9
                    print(String(format: "avg: %.3f s, rows/sec: %.1f M, BW: %.2f GB/s", avg, throughput/1e6, bw))
                }
            }
        }
    }
}

main()
