// Optional C backend for PQ encode (AoS layout only). Falls back to Swift otherwise.

import Foundation
#if canImport(CPQEncode)
import CPQEncode
#endif

@inline(__always)
@usableFromInline internal func _envFlag(_ name: String) -> Bool {
    guard let v = getenv(name) else { return false }
    let s = String(cString: v).lowercased()
    return s == "1" || s == "true" || s == "yes" || s == "on"
}

@inline(__always)
@usableFromInline internal var _useCPQEncode: Bool {
    #if canImport(CPQEncode)
    // Default ON on Apple platforms; can be disabled via env.
    return !_envFlag("VECTORINDEX_DISABLE_C_PQ")
    #else
    return false
    #endif
}

extension PQEncodeOpts {
    @inline(__always) @usableFromInline internal func _toC(ks: Int, layout: PQCodeLayout) -> CPQEncode.PQEncodeOpts {
        var c = CPQEncode.PQEncodeOpts(
            layout: CPQEncode.PQLayout(rawValue: 0),
            use_dot_trick: self.useDotTrick,
            precompute_x_norm2: self.useDotTrick,
            prefetch_distance: 8,
            num_threads: 0,
            soa_block_B: 0,
            interleave_g: 0
        )
        // We only support AoS in C path; layout field is set to AOS (0).
        return c
    }
}
