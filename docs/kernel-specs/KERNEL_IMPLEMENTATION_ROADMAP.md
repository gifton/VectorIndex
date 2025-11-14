# Kernel Implementation Roadmap

## ✅ Completed Kernels (29/35 = 83%)

### Core Distance Microkernels
- ✅ **#01** - L2 Squared Distance Microkernel
- ✅ **#02** - Inner Product Microkernel
- ✅ **#03** - Cosine Distance Microkernel
- ✅ **#04** - Score Block (batch distance computation)

### Top-K & Candidate Selection
- ✅ **#05** - Partial Top-K Selection
- ✅ **#06** - Top-K Merge (k-way merge)
- ✅ **#07** - Range/Threshold Query
- ✅ **#08** - ID Filter (bitset filtering)
- ✅ **#32** - Candidate Deduplication
- ✅ **#39** - Candidate Reservoir (adaptive thresholding)

### K-Means & Clustering
- ✅ **#11** - K-Means++ Seeding
- ✅ **#12** - Mini-Batch K-Means

### Product Quantization (PQ)
- ✅ **#19** - PQ Codebook Training
- ✅ **#20** - PQ Encoding (u8/u4)
- ✅ **#21** - PQ LUT Construction
- ✅ **#22** - ADC Scan (Asymmetric Distance Computation)
- ✅ **#23** - Residual Computation (IVF-PQ)

### Support Kernels
- ✅ **#S2** - RNG & Dtype Helpers (Xoroshiro128**, Philox4x32, dtype conversions)
- ✅ **#46** - Telemetry (Low-overhead instrumentation, histograms, JSON export)

### Vector Transforms & Utilities
- ✅ **#09** - Norm Cache
- ✅ **#10** - MIPS to L2 Transform
- ✅ **#49** - Prefetch Helpers

---

## 🚧 Remaining Kernels (6/35 = 17%)

Prioritized by **dependency order, complexity, and impact**:

### 🟡 **Tier 2: IVF Core Operations (Primary Functionality)**

#### **1. 29_ivf_select_nprobe** - IVF List Selection (nprobe routing)
- **Priority**: CRITICAL - Core IVF query operation
- **Complexity**: Medium-High
- **Impact**: Enables efficient IVF search with recall/performance trade-off
- **Dependencies**:
  - Kernels #04 (score block) ✅
  - Kernel #05 (partial top-k) ✅
- **Provides**:
  - Select which IVF lists to probe during query
  - Support for L2, IP, and cosine metrics
  - Optional beam search expansion
  - Batch query processing
- **Performance Target**: 50 μs for kc=10K, nprobe=50
- **Estimated Effort**: 3-4 days
- **Why Third**: Core operation that makes IVF useful

#### **4. 40_exact_rerank** - Exact Re-rank on Top-C
- **Priority**: HIGH - Improves search accuracy
- **Complexity**: Low-Medium
- **Impact**: Significant accuracy improvement with minimal cost
- **Dependencies**:
  - Kernels #01/#02/#03 (distance microkernels) ✅
  - Kernel #05 (partial top-k) ✅
- **Provides**:
  - Exact distance computation on candidate set
  - Recall improvement (95% → 99%+)
  - Small overhead (~10-20% of query time)
- **Estimated Effort**: 1-2 days
- **Why Fourth**: High impact, relatively simple, improves accuracy

#### **5. 30_ivf_append** - IVF List Append/Insert Operations
- **Priority**: HIGH - IVF maintenance
- **Complexity**: Medium-High
- **Impact**: Enables dynamic IVF index updates
- **Dependencies**:
  - Kernel #29 (for assignment) ⚠️
  - Kernel #23 (residuals) ✅
  - Kernel #20 (PQ encode) ✅
- **Provides**:
  - Append vectors to IVF lists
  - Batch insertion optimization
  - List growth management
  - Crash-safe append (with WAL)
- **Estimated Effort**: 3-4 days
- **Why Fifth**: Builds on #29, enables dynamic updates

---

### 🟢 **Tier 3: Infrastructure & Optimization (Polish)**

#### **6. 48_memory_layout_transforms** - Memory Layout Transformations
- **Priority**: MEDIUM - Performance optimization
- **Complexity**: Medium
- **Impact**: Enables better memory access patterns
- **Dependencies**: None (standalone utility)
- **Provides**:
  - AoS ↔ SoA transformations
  - Interleaved code layouts
  - Cache-friendly data arrangement
  - SIMD-aligned allocations
- **Estimated Effort**: 2-3 days
- **Why Sixth**: Nice optimization, not blocking

#### **7. 50_id_remap** - ID Remapping (External ↔ Internal)
- **Priority**: MEDIUM - Infrastructure
- **Complexity**: Medium
- **Impact**: Better ID management and deletion support
- **Dependencies**: None
- **Provides**:
  - External ID → internal dense handle mapping
  - Support for deletions without gaps
  - Compact ID storage
  - Fast bidirectional lookup
- **Estimated Effort**: 2-3 days
- **Why Seventh**: Infrastructure piece, not critical path

#### **8. S_serialization_mmap** - Serialization & Mmap Layout
- **Priority**: MEDIUM - Persistence
- **Complexity**: HIGH - Most complex remaining
- **Impact**: Zero-copy loading, persistence
- **Dependencies**:
  - All other kernels (consolidates everything)
  - Kernel #50 (ID remap) ⚠️
  - Kernel #46 (telemetry snapshot) ⚠️
- **Provides**:
  - Stable on-disk format
  - mmap-friendly layout
  - Zero-copy query-time access
  - Crash-safe updates with WAL
  - TOC-based extensibility
- **Estimated Effort**: 5-7 days
- **Why Last**: Most complex, depends on everything else

---

## 📊 **Summary Statistics**

### Completion Status
- **Total Kernels**: 35
- **Completed**: 29 (83%)
- **Remaining**: 6 (17%)

### By Priority Tier
- **Tier 1 (Foundation)**: 0 kernels - COMPLETE! ✅
- **Tier 2 (IVF Core)**: 3 kernels - ~7-10 days
- **Tier 3 (Infrastructure)**: 3 kernels - ~9-13 days

**Total Estimated Effort**: 16-23 days of focused work

### Critical Path
```
✅ S_rng_dtype_helpers (3d) - DONE!
    ↓
✅ 46_telemetry (3d) - DONE!
    ↓
29_ivf_select_nprobe (4d)
    ↓
40_exact_rerank (2d) + 30_ivf_append (4d)
    ↓
48_memory_layout + 50_id_remap (5d)
    ↓
S_serialization_mmap (6d)

Total: ~21 days remaining
```

---

## 🎯 **Recommended Order**

Based on dependencies, impact, and difficulty:

1. ✅ **DONE**: Kernel #23 (Residuals)
2. ✅ **DONE**: Kernel #S2 (RNG & Dtype Helpers)
3. ✅ **DONE**: Kernel #46 (Telemetry) - Just completed! 🎉

4. **29_ivf_select_nprobe** - Core IVF query
   - Enables efficient search
   - Recall/performance trade-off
   - High impact

5. **40_exact_rerank** - Quick win
   - Improves accuracy significantly
   - Relatively simple
   - Works with existing kernels

6. **30_ivf_append** - Dynamic updates
   - IVF maintenance
   - Batch insertion
   - Builds on #29

7. **48_memory_layout_transforms** - Optimization
   - Better memory patterns
   - Performance enhancement
   - Standalone utility

8. **50_id_remap** - Infrastructure
   - Better ID management
   - Deletion support
   - Nice to have

9. **S_serialization_mmap** - Final piece
   - Persistence layer
   - Zero-copy loading
   - Most complex, save for last

---

## 🚀 **Next Action**

**Recommend starting with: 29_ivf_select_nprobe (Kernel #29)**

This is the next priority:
- Core IVF query operation
- Enables efficient search with recall/performance trade-off
- High impact on production performance
- Medium-high complexity (~3-4 days)

After that, tackle **40_exact_rerank** for accuracy improvements.

---

## 📝 **Notes**

- All remaining kernels are marked **MUST** priority
- **Tier 1 (Foundation) is COMPLETE!** ✅
- Focus on Tier 2 (IVF Core) for production functionality
- Tier 3 can be done as optimization/polish
- **Current completion**: 83% - Exceptional progress! 🎉
- **Kernel #46 (Telemetry)** was just completed (9.5/10) ✨
  - Zero-overhead when disabled (#if guard)
  - Lock-free hot path with thread-local storage
  - Lock-striped histograms for P50/P90/P99
  - JSON export with atomic file writes
  - Production-ready instrumentation
- **Kernel #S2 (RNG & Dtype)** completed with Phase 1 (7.5/10)
  - Critical bf16 bug fixed
  - Comprehensive test suite (22 tests)
  - Ready for Phase 2 SIMD optimizations

---

**Last Updated**: October 10, 2025
**Status**: Kernel #46 (Telemetry) completed and marked as DONE
**Next Target**: 29_ivf_select_nprobe (Kernel #29)
<!-- moved to docs/kernel-specs/ -->
