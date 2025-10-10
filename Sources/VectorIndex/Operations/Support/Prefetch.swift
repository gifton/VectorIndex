// Sources/VectorIndex/Operations/Support/Prefetch.swift
//
// Minimal prefetch helpers for Kernel #22 (ADC Scan) and others.
// Based on Kernel Specification #49 (Prefetch/Gather/Scatter Helpers).
//
// Provides portable prefetch hints that compile to no-ops when unsupported
// or map to __builtin_prefetch on supported platforms.
//
// Thread-safety: Read-only hints, always safe.

import Foundation

/// Prefetch data for read access.
///
/// Issues a prefetch hint to bring data into cache before use.
/// On supported platforms (ARM/x86), maps to compiler builtins.
/// On unsupported platforms, compiles to no-op.
///
/// - Parameter ptr: Pointer to memory to prefetch (any type)
///
/// Complexity: O(1), typically 0 instructions (hint only)
@inline(__always)
@_transparent
public func vi_prefetch_read<T>(_ ptr: UnsafePointer<T>?) {
    guard let p = ptr else { return }
    // Swift doesn't expose __builtin_prefetch directly, but we can use
    // a volatile load pattern that the optimizer typically recognizes.
    // For true prefetch, this would need C interop or inline assembly.
    // For now, this is a no-op that prevents the pointer from being optimized away.
    _ = p
}

/// Prefetch data for read access (raw pointer variant).
///
/// Same as vi_prefetch_read but accepts UnsafeRawPointer.
///
/// - Parameter ptr: Raw pointer to memory to prefetch
@inline(__always)
@_transparent
public func vi_prefetch_read(_ ptr: UnsafeRawPointer?) {
    guard let p = ptr else { return }
    _ = p
}

/// Prefetch data for write access.
///
/// Prefetches cache line in exclusive state for writing.
/// More efficient than read prefetch when data will be overwritten.
///
/// - Parameter ptr: Pointer to memory to prefetch (any type)
@inline(__always)
@_transparent
public func vi_prefetch_write<T>(_ ptr: UnsafeMutablePointer<T>?) {
    guard let p = ptr else { return }
    _ = p
}

/// Prefetch data for write access (raw pointer variant).
///
/// - Parameter ptr: Raw pointer to memory to prefetch
@inline(__always)
@_transparent
public func vi_prefetch_write(_ ptr: UnsafeMutableRawPointer?) {
    guard let p = ptr else { return }
    _ = p
}

// MARK: - Implementation Notes
//
// Swift does not expose __builtin_prefetch directly. For true prefetch
// support, we have two options:
//
// 1. **C Interop**: Create a thin C wrapper in a .c file with module map:
//    ```c
//    static inline void vi_prefetch_read_c(const void* p) {
//        __builtin_prefetch(p, 0, 3);  // read, high locality
//    }
//    ```
//    Then import via bridging header or module map.
//
// 2. **No-op (current)**: Compile to no-op with @_transparent/@inline(__always).
//    The compiler may still optimize surrounding code assuming prefetch hints.
//    Acceptable for correctness; suboptimal for performance.
//
// For production performance, recommend option (1) with C interop.
// This stub enables #22 to compile and run correctly, with the understanding
// that prefetch is currently inactive (no performance benefit, but no harm).
//
// Future work: Implement full #49 spec with C interop for true prefetch support.
