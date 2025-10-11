# ✅ DONE — Kernel Specification #22: ADC (Asymmetric Distance Computation) Scan

**ID**: 22
**Priority**: MUST
**Role**: Q (Query)
**Status**: Specification

---

## Purpose

Compute approximate L2 distances between a query and PQ-encoded database vectors using precomputed lookup tables (LUTs). This is the core throughput-critical kernel in PQ and IVF-PQ search, enabling 50-100M+ vectors/sec scan rates.

**Key Benefits**:
1. **Speed**: O(m) distance computation vs O(d) for exact L2
2. **Throughput**: 50-100M vectors/sec on modern CPUs
3. **Memory efficiency**: Scan compressed codes (8-32 bytes) instead of full vectors (4 KB)

**Typical Use Case**: Scan 10M PQ-encoded vectors (8 bytes each) in ~100-200 ms, achieving 100× speedup over exact search.

---

## Mathematical Foundations

### 1. Asymmetric Distance Computation (ADC)

**Problem**: Given query **q** ∈ ℝᵈ and PQ-encoded database vector with codes **c** = (c₁, ..., cₘ), estimate ‖**q** - **x**‖².

**PQ Approximation**:
```
dist²(q, x) ≈ Σⱼ₌₁ᵐ ‖qⱼ - Cⱼ[cⱼ]‖²
```

**With Precomputed LUT** (from kernel #21):
```
LUT[j][k] = ‖qⱼ - Cⱼ[k]‖²
```

**ADC Distance**:
```
dist²(q, x) ≈ Σⱼ₌₁ᵐ LUT[j][codes[j]]
```

**Complexity**:
- m table lookups (typically m=8)
- m floating-point additions
- **Total**: O(m) operations vs O(d) for exact L2

### 2. Code Layouts

**AoS (Array of Structures)**:
```
codes[i*m + j] = code for vector i, subspace j
```

For vector i, codes are stored contiguously: [c₀, c₁, ..., cₘ₋₁]

**Interleaved Block (group size g)**:
```
codes[(i/g)*m*g + j*g + (i%g)] = code for vector i, subspace j
```

Vectors are grouped in blocks of g, with all g vectors' codes for subspace j stored together.

**Example** (m=8, g=4):
```
Block 0 (vectors 0-3):
  [c₀₀, c₀₁, c₀₂, c₀₃]  // subspace 0, vectors 0-3
  [c₁₀, c₁₁, c₁₂, c₁₃]  // subspace 1, vectors 0-3
  ...
  [c₇₀, c₇₁, c₇₂, c₇₃]  // subspace 7, vectors 0-3

Block 1 (vectors 4-7):
  [c₀₄, c₀₅, c₀₆, c₀₇]  // subspace 0, vectors 4-7
  ...
```

**Benefit**: Interleaved layout enables processing g vectors in parallel with better cache locality.

### 3. 4-bit Code Handling

For ks=16, codes are packed as nibbles (4 bits):

**Packed Layout**:
```
byte = (code_low & 0xF) | ((code_high & 0xF) << 4)
```

**Unpacking**:
```
code_low = byte & 0xF
code_high = (byte >> 4) & 0xF
```

**Storage**: For m=8 subspaces, 4 bytes per vector (vs 8 bytes for u8).

### 4. Query Norm Bias

If LUT was built with `include_q_norm = false` (kernel #21), add query norm as a bias:

```
dist²(q, x) = Σⱼ₌₁ᵐ LUT[j][codes[j]] + ‖q‖²
```

The bias ‖**q**‖² is constant for all vectors and added once per distance computation.

---

## API Signatures

### 1. 8-bit ADC Scan (ks=256)

```c
void adc_scan_u8(
    const uint8_t* codes,              // [n × m] or interleaved codes
    int64_t n,                         // number of vectors
    int m,                             // number of subspaces
    int ks,                            // codebook size (must be 256)
    const float* lut,                  // [m × ks] lookup table
    float* out,                        // [n] output distances
    const ADCScanOpts* opts            // options (nullable)
);
```

**Parameters**:

- `codes`: PQ codes from kernel #20
  - AoS layout: `codes[i*m + j]` = code for vector i, subspace j
  - Interleaved: depends on group size g (see opts)
- `n`: Number of vectors to scan
- `m`: Number of subspaces (typically 8, 16, or 32)
- `ks`: Codebook size per subspace (must be 256 for u8)
- `lut`: Lookup table from kernel #21, layout `[m][ks]`
  - `lut[j*ks + k]` = distance from query subspace j to centroid k
- `out`: Output buffer for distances, **must be preallocated** to n floats
  - `out[i]` = approximate distance from query to vector i
- `opts`: Optional configuration (nullable, use defaults if null)

### 2. 4-bit ADC Scan (ks=16)

```c
void adc_scan_u4(
    const uint8_t* codes,              // [n × m/2] or interleaved packed codes
    int64_t n,                         // number of vectors
    int m,                             // number of subspaces (must be even)
    int ks,                            // codebook size (must be 16)
    const float* lut,                  // [m × ks] lookup table
    float* out,                        // [n] output distances
    const ADCScanOpts* opts            // options (nullable)
);
```

**Parameters**: Same as u8, except:
- `m`: Must be even (for byte-aligned packing)
- `ks`: Must be 16 for u4
- `codes`: Packed nibbles, m/2 bytes per vector

### 3. Options

```c
typedef struct {
    ADCLayout layout;              // code layout (default: ADC_LAYOUT_AOS)
    int group_size;                // interleaved group size (for INTERLEAVED_BLOCK)
    int stride;                    // stride for padded AoS (0 = tight, default: 0)
    float add_bias;                // query norm bias (default: 0.0)
    bool strict_fp;                // use Kahan summation for m≥64 (default: false)
    int prefetch_distance;         // prefetch lookahead (default: 8)
    int num_threads;               // parallelism (0 = auto, default: 0)
} ADCScanOpts;

typedef enum {
    ADC_LAYOUT_AOS,                // [n][m] row-major codes
    ADC_LAYOUT_INTERLEAVED_BLOCK   // [n/g][m][g] interleaved groups
} ADCLayout;
```

**Options Explained**:

- **layout**: Code memory layout
  - AoS: Standard row-major, `codes[i*m + j]`
  - INTERLEAVED_BLOCK: Blocked by group size g

- **group_size**: Number of vectors per interleaved block (typical: 4 or 8)
  - Only used if layout = INTERLEAVED_BLOCK
  - Must divide n evenly (or handle remainder separately)

- **stride**: For padded AoS layouts
  - 0 (default): tight packing, stride = m
  - >0: stride = specified value (e.g., round up to power of 2)

- **add_bias**: Query norm bias to add to each distance
  - Use when LUT was built with `include_q_norm = false`
  - Value: ‖**q**‖² = Σⱼ q_norms[j]

- **strict_fp**: Strict floating-point mode
  - true: Use Kahan summation for m≥64 to reduce accumulation error
  - false: Standard summation (faster but less accurate for large m)

- **prefetch_distance**: Software prefetch lookahead (for code data)

- **num_threads**: Parallelism level
  - 0 (default): Auto-detect (use all cores)
  - >0: Use specified number of threads
  - Parallelizes over vectors (chunks of n)

---

## Algorithm Details

### 1. AoS Scan (8-bit)

**Pseudocode**:
```
adc_scan_u8(codes, n, m, ks, lut, out, opts):
    for i in 0..n-1:
        sum = 0
        for j in 0..m-1:
            code = codes[i*m + j]
            sum += lut[j*ks + code]

        out[i] = sum + opts.add_bias
```

**Optimized with ILP** (Instruction-Level Parallelism):
```c
void adc_scan_u8_aos(const uint8_t* codes, int64_t n, int m, int ks,
                      const float* lut, float* out, const ADCScanOpts* opts) {
    float bias = opts ? opts->add_bias : 0.0f;

    for (int64_t i = 0; i < n; i++) {
        const uint8_t* code_vec = codes + i*m;

        // Use multiple accumulators for ILP
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

        int j = 0;
        // Process 8 codes at a time
        for (; j + 7 < m; j += 8) {
            sum0 += lut[j*ks + code_vec[j]];
            sum1 += lut[(j+1)*ks + code_vec[j+1]];
            sum2 += lut[(j+2)*ks + code_vec[j+2]];
            sum3 += lut[(j+3)*ks + code_vec[j+3]];
            sum0 += lut[(j+4)*ks + code_vec[j+4]];
            sum1 += lut[(j+5)*ks + code_vec[j+5]];
            sum2 += lut[(j+6)*ks + code_vec[j+6]];
            sum3 += lut[(j+7)*ks + code_vec[j+7]];
        }

        // Handle remainder
        for (; j < m; j++) {
            sum0 += lut[j*ks + code_vec[j]];
        }

        out[i] = sum0 + sum1 + sum2 + sum3 + bias;
    }
}
```

### 2. Interleaved Scan (8-bit)

**Layout**: Vectors grouped in blocks of g, subspaces interleaved.

**Pseudocode**:
```
adc_scan_u8_interleaved(codes, n, m, ks, g, lut, out, opts):
    num_blocks = ceil(n / g)

    for block in 0..num_blocks-1:
        base_idx = block * g
        block_size = min(g, n - base_idx)

        // Initialize accumulators for g vectors
        sums[g] = {0}

        // Process all subspaces
        for j in 0..m-1:
            code_offset = block*m*g + j*g
            lut_row = lut[j*ks : (j+1)*ks]

            for v in 0..block_size-1:
                code = codes[code_offset + v]
                sums[v] += lut_row[code]

        // Write outputs
        for v in 0..block_size-1:
            out[base_idx + v] = sums[v] + opts.add_bias
```

**Optimized** (process g vectors in parallel):
```c
void adc_scan_u8_interleaved(const uint8_t* codes, int64_t n, int m, int ks, int g,
                              const float* lut, float* out, float bias) {
    int64_t num_blocks = (n + g - 1) / g;

    for (int64_t block = 0; block < num_blocks; block++) {
        int64_t base_idx = block * g;
        int block_size = (int)fmin(g, n - base_idx);

        // Accumulate for g vectors
        float sums[8] = {0};  // assumes g ≤ 8

        const uint8_t* block_codes = codes + block*m*g;

        for (int j = 0; j < m; j++) {
            const float* lut_row = lut + j*ks;
            const uint8_t* subspace_codes = block_codes + j*g;

            // Process g codes for this subspace
            for (int v = 0; v < block_size; v++) {
                uint8_t code = subspace_codes[v];
                sums[v] += lut_row[code];
            }
        }

        // Write outputs
        for (int v = 0; v < block_size; v++) {
            out[base_idx + v] = sums[v] + bias;
        }
    }
}
```

### 3. 4-bit Scan with Unpacking

**Pseudocode**:
```
adc_scan_u4(codes, n, m, ks, lut, out, opts):
    for i in 0..n-1:
        sum = 0

        for j in 0..m-1 step 2:
            byte = codes[i*(m/2) + j/2]

            // Unpack two codes
            code0 = byte & 0xF
            code1 = (byte >> 4) & 0xF

            sum += lut[j*ks + code0]
            sum += lut[(j+1)*ks + code1]

        out[i] = sum + opts.add_bias
```

**Optimized** (batch unpacking):
```c
void adc_scan_u4_aos(const uint8_t* codes, int64_t n, int m, int ks,
                      const float* lut, float* out, float bias) {
    int m_bytes = m / 2;

    for (int64_t i = 0; i < n; i++) {
        const uint8_t* code_vec = codes + i*m_bytes;

        float sum = 0;

        // Process pairs
        for (int j = 0; j < m; j += 2) {
            uint8_t byte = code_vec[j/2];
            uint8_t code0 = byte & 0xF;
            uint8_t code1 = (byte >> 4) & 0xF;

            sum += lut[j*ks + code0];
            sum += lut[(j+1)*ks + code1];
        }

        out[i] = sum + bias;
    }
}
```

### 4. Kahan Summation (Strict FP)

For large m (m ≥ 64), accumulation error can be significant. Use Kahan summation:

```c
float adc_scan_kahan(const uint8_t* code_vec, int m, int ks, const float* lut, float bias) {
    float sum = 0;
    float compensation = 0;  // error compensation

    for (int j = 0; j < m; j++) {
        uint8_t code = code_vec[j];
        float value = lut[j*ks + code];

        // Kahan summation
        float y = value - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    return sum + bias;
}
```

**Complexity**: Same O(m) but ~2× more operations due to compensation tracking.

**When to Use**: Only for strict_fp mode with large m (m ≥ 64).

---

## Implementation Strategies

### 1. Instruction-Level Parallelism (ILP)

**Multiple Accumulators**: Hide memory latency by using 2-4 independent accumulators:

```c
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

for (int j = 0; j < m; j += 4) {
    sum0 += lut[j*ks + code_vec[j]];
    sum1 += lut[(j+1)*ks + code_vec[j+1]];
    sum2 += lut[(j+2)*ks + code_vec[j+2]];
    sum3 += lut[(j+3)*ks + code_vec[j+3]];
}

float total = sum0 + sum1 + sum2 + sum3;
```

**Benefit**: Allows CPU to execute lookups in parallel, hiding ~5-10 cycle latency per lookup.

### 2. Loop Unrolling

**Unroll by 8-16 iterations**:

```c
int j = 0;
// Unrolled loop (16 iterations)
for (; j + 15 < m; j += 16) {
    sum0 += lut[(j+0)*ks + code_vec[j+0]];
    sum1 += lut[(j+1)*ks + code_vec[j+1]];
    sum2 += lut[(j+2)*ks + code_vec[j+2]];
    sum3 += lut[(j+3)*ks + code_vec[j+3]];
    sum0 += lut[(j+4)*ks + code_vec[j+4]];
    sum1 += lut[(j+5)*ks + code_vec[j+5]];
    sum2 += lut[(j+6)*ks + code_vec[j+6]];
    sum3 += lut[(j+7)*ks + code_vec[j+7]];
    sum0 += lut[(j+8)*ks + code_vec[j+8]];
    sum1 += lut[(j+9)*ks + code_vec[j+9]];
    sum2 += lut[(j+10)*ks + code_vec[j+10]];
    sum3 += lut[(j+11)*ks + code_vec[j+11]];
    sum0 += lut[(j+12)*ks + code_vec[j+12]];
    sum1 += lut[(j+13)*ks + code_vec[j+13]];
    sum2 += lut[(j+14)*ks + code_vec[j+14]];
    sum3 += lut[(j+15)*ks + code_vec[j+15]];
}

// Remainder
for (; j < m; j++) {
    sum0 += lut[j*ks + code_vec[j]];
}
```

### 3. Prefetching

**Prefetch Code Stream**:
```c
const int PREFETCH_DIST = 8;

for (int64_t i = 0; i < n; i++) {
    // Prefetch future codes
    if (i + PREFETCH_DIST < n) {
        __builtin_prefetch(codes + (i + PREFETCH_DIST)*m, 0, 3);
    }

    // Process current vector
    float dist = adc_scan_vector(codes + i*m, m, ks, lut);
    out[i] = dist;
}
```

**LUT Prefetching**: LUT should fit in L1 cache (8 KB for m=8, ks=256), so explicit prefetching usually not needed.

### 4. Tiling

**Process vectors in tiles**:

```c
const int TILE_SIZE = 1024;

for (int64_t tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
    int64_t tile_end = min(tile_start + TILE_SIZE, n);

    // Process tile
    for (int64_t i = tile_start; i < tile_end; i++) {
        out[i] = adc_scan_vector(codes + i*m, m, ks, lut);
    }
}
```

**Benefit**: Better cache locality for output buffer.

### 5. Parallelism

**Parallel over Vectors**:
```c
#pragma omp parallel for num_threads(opts->num_threads) schedule(static)
for (int64_t i = 0; i < n; i++) {
    out[i] = adc_scan_vector(codes + i*m, m, ks, lut, bias);
}
```

**Scaling**: Near-linear scaling up to number of physical cores (no synchronization needed).

---

## Performance Characteristics

### 1. Computational Complexity

**Per Vector**:
- m table lookups (pointer indirection + load)
- m floating-point additions
- 1 bias addition (if applicable)

**Memory Operations**:
- Read m bytes (u8) or m/2 bytes (u4) for codes
- m random accesses to LUT (but LUT in L1 cache)

**Latency** (per vector):
- Best case: ~5-10 ns (m=8, hot cache)
- Typical: ~10-20 ns (m=8, cold codes)

**Throughput** (full scan):
- Limited by code memory bandwidth and lookup latency

### 2. Memory Bandwidth

**Reads per Vector**:
- Codes: m bytes (u8) or m/2 bytes (u4)
- LUT: m × 4 bytes (if cache misses, but should be in L1)

**Writes per Vector**:
- Output distance: 4 bytes

**Example** (m=8, u8):
- Reads: 8 bytes (codes) + 0 bytes (LUT in cache) = 8 bytes
- Writes: 4 bytes
- Total: 12 bytes per vector

**Throughput** (bandwidth-limited):
- Memory bandwidth: 200 GB/s (typical M2 Max)
- Bytes per vector: 12 bytes
- Max throughput: 200 GB/s / 12 B ≈ 16.7 billion vectors/sec

**Actual**: Typically 50-100M vectors/sec due to lookup latency and cache effects.

### 3. Performance Targets (Apple M2 Max, 8 P-cores)

| Configuration | Throughput | Time (10M vectors) | Notes |
|---------------|------------|-------------------|-------|
| m=8, u8, AoS | 80M vec/s | 125 ms | Good cache locality |
| m=8, u8, Interleaved(g=4) | 100M vec/s | 100 ms | Better ILP |
| m=8, u4, AoS | 120M vec/s | 83 ms | Less bandwidth |
| m=16, u8, AoS | 60M vec/s | 167 ms | More lookups |
| m=32, u8, AoS | 40M vec/s | 250 ms | Lookup-bound |

**Scaling**:
- **n**: Linear (O(n))
- **m**: Linear (O(m)), but diminishing returns for large m due to lookup latency
- **Threads**: Near-linear up to physical cores

### 4. ADC vs Exact Search Speedup

| Dimension | ADC (m=8) | Exact L2 | Speedup |
|-----------|-----------|----------|---------|
| d=512 | 100M vec/s | 1M vec/s | 100× |
| d=1024 | 100M vec/s | 500K vec/s | 200× |
| d=1536 | 100M vec/s | 350K vec/s | 285× |
| d=2048 | 100M vec/s | 250K vec/s | 400× |

**Note**: Exact L2 throughput decreases with dimension, ADC throughput remains constant.

---

## Numerical Considerations

### 1. Floating-Point Accumulation

**Standard Summation** (fast mode):
```c
float sum = 0;
for (int j = 0; j < m; j++) {
    sum += lut[j*ks + codes[j]];
}
```

**Error**: O(m·ε) where ε ≈ 2⁻²³ ≈ 1.2×10⁻⁷ for float32.
- For m=8: error ≈ 10⁻⁶ (negligible)
- For m=64: error ≈ 8×10⁻⁶ (may be significant)

**Kahan Summation** (strict mode, m ≥ 64):
```c
float sum = 0, c = 0;
for (int j = 0; j < m; j++) {
    float y = lut[j*ks + codes[j]] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

**Error**: O(ε²) instead of O(m·ε), much more accurate for large m.

**Trade-off**: ~2× slower due to extra operations.

### 2. Bias Addition

**Bias Term**: When LUT excludes query norm, add as bias:

```c
out[i] = adc_distance + query_norm;
```

**Numerical Equivalence**: Should produce identical results to including query norm in LUT (within floating-point precision).

### 3. Determinism

**Guarantee**: With fixed inputs and `strict_fp = false`, results are deterministic across runs (same floating-point rounding).

**With Parallelism**: Output order is deterministic (sorted by index), but floating-point operations are performed in parallel, so results are bitwise identical.

---

## Correctness Testing

### 1. Scalar Reference

**Test 1: Exact Match**
```swift
func testADCScanCorrectness() {
    let n = 1_000
    let m = 8
    let ks = 256

    let codes = generateRandomCodes(n: n, m: m, ks: ks)
    let lut = generateRandomLUT(m: m, ks: ks)

    // Optimized scan
    var out_fast = [Float](repeating: 0, count: n)
    adc_scan_u8(codes, n, m, ks, lut, &out_fast, nil)

    // Scalar reference
    var out_ref = [Float](repeating: 0, count: n)
    for i in 0..<n {
        var sum: Float = 0
        for j in 0..<m {
            let code = Int(codes[i*m + j])
            sum += lut[j*ks + code]
        }
        out_ref[i] = sum
    }

    // Should match exactly
    for i in 0..<n {
        assert(out_fast[i] == out_ref[i], "Mismatch at \(i)")
    }
}
```

### 2. Layout Parity

**Test 2: AoS vs Interleaved**
```swift
func testLayoutParity() {
    let n = 10_000
    let m = 8
    let ks = 256
    let g = 4

    let codes_aos = generateRandomCodes(n: n, m: m, ks: ks)
    let lut = generateRandomLUT(m: m, ks: ks)

    // Convert to interleaved
    let codes_interleaved = convertToInterleaved(codes_aos, n: n, m: m, g: g)

    // Scan AoS
    var out_aos = [Float](repeating: 0, count: n)
    adc_scan_u8(codes_aos, n, m, ks, lut, &out_aos, nil)

    // Scan Interleaved
    var out_interleaved = [Float](repeating: 0, count: n)
    var opts = ADCScanOpts(layout: ADC_LAYOUT_INTERLEAVED_BLOCK, group_size: g)
    adc_scan_u8(codes_interleaved, n, m, ks, lut, &out_interleaved, &opts)

    // Should match exactly
    for i in 0..<n {
        assert(out_aos[i] == out_interleaved[i], "Layout mismatch at \(i)")
    }
}
```

### 3. u4 vs u8 Consistency

**Test 3: 4-bit Correctness**
```swift
func testU4Correctness() {
    let n = 5_000
    let m = 8
    let ks = 16  // u4

    // Generate codes in range [0, 15]
    let codes_u8 = generateRandomCodes(n: n, m: m, ks: ks)

    // Pack to u4
    var codes_u4 = [UInt8](repeating: 0, count: n * m / 2)
    for i in 0..<n {
        for j in 0..<(m/2) {
            let c0 = codes_u8[i*m + j*2]
            let c1 = codes_u8[i*m + j*2 + 1]
            codes_u4[i*(m/2) + j] = (c0 & 0xF) | ((c1 & 0xF) << 4)
        }
    }

    let lut = generateRandomLUT(m: m, ks: ks)

    // Scan u8 (reference)
    var out_u8 = [Float](repeating: 0, count: n)
    adc_scan_u8(codes_u8, n, m, ks, lut, &out_u8, nil)

    // Scan u4
    var out_u4 = [Float](repeating: 0, count: n)
    adc_scan_u4(codes_u4, n, m, ks, lut, &out_u4, nil)

    // Should match exactly
    for i in 0..<n {
        assert(out_u4[i] == out_u8[i], "u4 mismatch at \(i)")
    }
}
```

### 4. Bias Addition

**Test 4: Query Norm Bias**
```swift
func testBiasAddition() {
    let n = 1_000
    let m = 8
    let ks = 256
    let bias: Float = 123.45

    let codes = generateRandomCodes(n: n, m: m, ks: ks)
    let lut = generateRandomLUT(m: m, ks: ks)

    // Scan without bias
    var out_no_bias = [Float](repeating: 0, count: n)
    adc_scan_u8(codes, n, m, ks, lut, &out_no_bias, nil)

    // Scan with bias
    var out_with_bias = [Float](repeating: 0, count: n)
    var opts = ADCScanOpts(add_bias: bias)
    adc_scan_u8(codes, n, m, ks, lut, &out_with_bias, &opts)

    // Check: out_with_bias[i] = out_no_bias[i] + bias
    for i in 0..<n {
        let expected = out_no_bias[i] + bias
        assert(abs(out_with_bias[i] - expected) < 1e-5, "Bias mismatch at \(i)")
    }
}
```

### 5. End-to-End with Encoding

**Test 5: Encode → LUT → ADC → Verify**
```swift
func testEndToEndADC() {
    let n = 1_000
    let d = 512
    let m = 8
    let ks = 256

    let vectors = generateRandomVectors(n: n, d: d)
    let query = generateRandomVector(d: d)
    let codebooks = trainPQCodebooks(vectors, m: m, ks: ks)

    // Encode vectors (kernel #20)
    var codes = [UInt8](repeating: 0, count: n * m)
    pq_encode_u8_f32(vectors, n, d, m, ks, codebooks, &codes, nil)

    // Build LUT (kernel #21)
    var lut = [Float](repeating: 0, count: m * ks)
    pq_lut_l2_f32(query, d, m, ks, codebooks, &lut, nil, nil, nil)

    // ADC scan (kernel #22)
    var adc_dists = [Float](repeating: 0, count: n)
    adc_scan_u8(codes, n, m, ks, lut, &adc_dists, nil)

    // Compute exact distances
    var exact_dists = [Float](repeating: 0, count: n)
    for i in 0..<n {
        exact_dists[i] = l2Distance(query, vectors[i*d..<(i+1)*d])
    }

    // ADC should approximate exact within reasonable error
    for i in 0..<n {
        let error = abs(adc_dists[i] - exact_dists[i]) / exact_dists[i]
        assert(error < 0.3, "ADC error \(error) too large at \(i)")  // 30% relative error threshold
    }
}
```

### 6. Throughput Benchmark

**Test 6: Performance Regression**
```swift
func testADCThroughput() {
    let n = 10_000_000
    let m = 8
    let ks = 256

    let codes = generateRandomCodes(n: n, m: m, ks: ks)
    let lut = generateRandomLUT(m: m, ks: ks)

    var out = [Float](repeating: 0, count: n)

    let start = Date()
    adc_scan_u8(codes, n, m, ks, lut, &out, nil)
    let elapsed = Date().timeIntervalSince(start)

    let throughput = Double(n) / elapsed
    print("ADC scan throughput: \(throughput / 1_000_000) M vectors/sec")

    // Expect > 50M vectors/sec on M2 Max
    assert(throughput > 50_000_000, "Throughput \(throughput) below target")
}
```

---

## Integration Patterns

### 1. IVF-PQ Query (Complete Pipeline)

**Full Query Workflow**:
```swift
func queryIVFPQ(query: [Float], index: IVFPQIndex, k: Int, nprobe: Int) -> [(id: Int, dist: Float)] {
    // 1. Select nprobe IVF lists (kernel #29)
    let probe_lists = selectIVFLists(query, index.coarse_centroids, nprobe)

    var all_candidates: [(id: Int, dist: Float)] = []

    // 2. For each probed list
    for list_id in probe_lists {
        let ivf_list = index.ivf_lists[list_id]
        let n_list = ivf_list.count / index.m

        // 3. Build residual LUT (kernel #21)
        var lut = [Float](repeating: 0, count: index.m * index.ks)
        pq_lut_residual_l2_f32(
            query,
            index.coarse_centroids[list_id],
            index.dimension, index.m, index.ks,
            index.pq_codebooks, &lut,
            index.centroid_norms, nil
        )

        // 4. ADC scan (kernel #22)
        var distances = [Float](repeating: 0, count: n_list)
        adc_scan_u8(
            ivf_list, n_list, index.m, index.ks,
            lut, &distances, nil
        )

        // 5. Collect top-k from this list (kernel #05)
        let list_topk = partialTopK(distances, k: k)
        for (local_id, dist) in list_topk {
            let global_id = index.id_map[list_id][local_id]
            all_candidates.append((id: global_id, dist: dist))
        }
    }

    // 6. Merge candidates across all lists (kernel #06)
    let final_topk = mergeTopK(all_candidates, k: k)

    // 7. Optional: exact rerank (kernel #40)
    if index.enable_rerank {
        return rerankExact(final_topk, query, index.vectors)
    }

    return final_topk
}
```

### 2. Flat PQ Query

**Without IVF**:
```swift
func queryFlatPQ(query: [Float], index: FlatPQIndex, k: Int) -> [(id: Int, dist: Float)] {
    // 1. Build LUT (kernel #21)
    var lut = [Float](repeating: 0, count: index.m * index.ks)
    pq_lut_l2_f32(
        query, index.dimension, index.m, index.ks,
        index.pq_codebooks, &lut,
        index.centroid_norms, nil, nil
    )

    // 2. ADC scan all codes (kernel #22)
    var distances = [Float](repeating: 0, count: index.num_vectors)
    adc_scan_u8(
        index.codes, index.num_vectors, index.m, index.ks,
        lut, &distances, nil
    )

    // 3. Top-k selection (kernel #05)
    let topk = partialTopK(distances, k: k)

    return topk
}
```

### 3. Batch Query Processing

**Process multiple queries in parallel**:
```c
void batch_adc_scan(
    const float* queries,              // [n_queries × d]
    int n_queries,
    const uint8_t* codes,              // [n_vectors × m]
    int64_t n_vectors,
    int d, int m, int ks,
    const float* codebooks,
    float* distances                   // [n_queries × n_vectors]
) {
    #pragma omp parallel for
    for (int q = 0; q < n_queries; q++) {
        // Build LUT
        float lut[m * ks];
        pq_lut_l2_f32(queries + q*d, d, m, ks, codebooks, lut, NULL, NULL, NULL);

        // ADC scan
        adc_scan_u8(codes, n_vectors, m, ks, lut, distances + q*n_vectors, NULL);
    }
}
```

---

## Coding Guidelines

### 1. API Design

**Consistent Naming**:
```c
// All ADC scan functions prefixed with adc_scan_
void adc_scan_u8(...);
void adc_scan_u4(...);
void adc_scan_u8_aos(...);      // internal
void adc_scan_u8_interleaved(...);  // internal
```

### 2. Performance Optimization

**ILP Template**:
```c
// Always use 4 accumulators for m ≥ 16
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

int j = 0;
for (; j + 3 < m; j += 4) {
    sum0 += lut[j*ks + codes[j]];
    sum1 += lut[(j+1)*ks + codes[j+1]];
    sum2 += lut[(j+2)*ks + codes[j+2]];
    sum3 += lut[(j+3)*ks + codes[j+3]];
}

// Remainder
for (; j < m; j++) {
    sum0 += lut[j*ks + codes[j]];
}

float total = sum0 + sum1 + sum2 + sum3;
```

### 3. Telemetry Integration

**Metrics Collection** (#46):
```c
void emit_adc_scan_telemetry(int64_t n, int m, int ks, bool u4, double time_sec) {
    telemetry_emit("adc_scan.vectors", n);
    telemetry_emit("adc_scan.subspaces", m);
    telemetry_emit("adc_scan.codebook_size", ks);
    telemetry_emit("adc_scan.u4", u4 ? 1 : 0);
    telemetry_emit("adc_scan.time_sec", time_sec);
    telemetry_emit("adc_scan.throughput_vec_per_sec", (double)n / time_sec);
    telemetry_emit("adc_scan.lookups_per_sec", (double)(n * m) / time_sec);
}
```

---

## Example Usage

### Example 1: Basic ADC Scan

```c
#include "adc_scan.h"

int main() {
    int64_t n = 1000000;
    int m = 8;
    int ks = 256;

    // Load codes and LUT
    uint8_t* codes = load_codes("codes.bin", n, m);
    float lut[8 * 256];
    load_lut("lut.bin", lut, m, ks);

    // Allocate output
    float* distances = malloc(n * sizeof(float));

    // ADC scan
    adc_scan_u8(codes, n, m, ks, lut, distances, NULL);

    // Find top-k
    topk_select(distances, n, 10);

    free(codes);
    free(distances);
    return 0;
}
```

### Example 2: Interleaved Layout

```c
#include "adc_scan.h"

void scan_interleaved(const uint8_t* codes, int64_t n, int m) {
    int ks = 256;
    int g = 4;  // group size

    float lut[8 * 256];
    load_lut("lut.bin", lut, m, ks);

    float* distances = malloc(n * sizeof(float));

    // Configure for interleaved layout
    ADCScanOpts opts = {
        .layout = ADC_LAYOUT_INTERLEAVED_BLOCK,
        .group_size = g,
        .stride = 0,
        .add_bias = 0.0f,
        .strict_fp = false,
        .prefetch_distance = 8,
        .num_threads = 0
    };

    adc_scan_u8(codes, n, m, ks, lut, distances, &opts);

    free(distances);
}
```

### Example 3: Swift Integration

```swift
import Foundation

func adcScan(
    codes: [UInt8],
    n: Int,
    m: Int,
    ks: Int,
    lut: [Float],
    bias: Float = 0.0
) -> [Float] {
    var distances = [Float](repeating: 0, count: n)

    codes.withUnsafeBufferPointer { codePtr in
        lut.withUnsafeBufferPointer { lutPtr in
            distances.withUnsafeMutableBufferPointer { distPtr in
                var opts = ADCScanOpts(add_bias: bias)

                adc_scan_u8(
                    codePtr.baseAddress!,
                    Int64(n),
                    Int32(m),
                    Int32(ks),
                    lutPtr.baseAddress!,
                    distPtr.baseAddress!,
                    &opts
                )
            }
        }
    }

    return distances
}
```

---

## Summary

**Kernel #22** provides ultra-fast ADC scanning for PQ-encoded vectors:

1. **Performance**: 50-100M vectors/sec on M2 Max, enabling 100-400× speedup over exact L2
2. **Formats**:
   - 8-bit (ks=256): 1 byte per subspace
   - 4-bit (ks=16): 4 bits per subspace (packed)
3. **Layouts**:
   - AoS: Standard row-major
   - Interleaved: Better ILP and cache locality
4. **Key Optimizations**:
   - Multiple accumulators for ILP (hide lookup latency)
   - Loop unrolling (8-16 iterations)
   - Prefetching for code stream
   - Kahan summation for large m (strict mode)
5. **Integration**:
   - Consumes LUTs from kernel #21
   - Processes codes from kernel #20
   - Produces distances for top-k selection (kernel #05)

**Dependencies**:
- Kernel #20 (PQ encoding, codes)
- Kernel #21 (LUT construction)
- #49 (prefetch helpers)

**Typical Use**: Scan 10M vectors (m=8, ks=256) in ~100-200 ms, achieving 100× faster search than exact L2 distance with 95%+ recall.
