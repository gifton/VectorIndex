import Foundation

// ExternalIDRegistry maps public VectorID (String) <-> compact UInt64 handles.
// This allows #50 (IDMap) to operate on UInt64 while public APIs remain String-based.

public final class ExternalIDRegistry {
    private var strToU64: [String: UInt64] = [:]
    private var u64ToStr: [UInt64: String] = [:]
    private var next: UInt64 = 1 // reserve 0 for "unset"
    private let lock = NSLock()

    public init() {}

    public func getOrCreate(_ s: String) -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        if let v = strToU64[s] { return v }
        let v = next; next &+= 1
        strToU64[s] = v
        u64ToStr[v] = s
        return v
    }

    public func getString(_ v: UInt64) -> String? {
        lock.lock(); defer { lock.unlock() }
        return u64ToStr[v]
    }

    public func getOrCreateBatch(_ ss: [String]) -> [UInt64] {
        var out: [UInt64] = []
        out.reserveCapacity(ss.count)
        for s in ss { out.append(getOrCreate(s)) }
        return out
    }
}

