ID: S1 — Serialization & Mmap Layout (MUST)

Purpose
- Define a stable, mmap‑friendly on‑disk layout for VectorIndex (IVF‑Flat / IVF‑PQ), enabling zero‑copy query‑time access and crash‑safe updates.

Scope
- Covers: global header/TOC, coarse centroids, PQ codebooks + centroid norms, list descriptors + payloads (IDs, codes, vecs), norm caches (#09), ID remap (#50), tombstones (#43), telemetry snapshot (#46), and WAL/commit protocol.
- Excludes: graph track specifics (handled by graph serialization if shipped later).

Design Goals
- Zero‑copy scan: #22 and #04 can read directly via mmapped pointers with native alignment.
- Deterministic, versioned format with forward/backward compatibility and explicit endianness.
- Crash‑consistency for append‑only updates; compaction produces a new file atomically.

Container
- Single file container (`.vindex`) with monotonic `generation`. Optional side WAL (`.vindex.wal`) for streaming appends.

Header (fixed, 256 bytes)
- `magic` (8): "VINDEX\0\0"
- `version_major` (u16), `version_minor` (u16)
- `endianness` (u8): 1=little, 2=big (file encoding)
- `arch` (u8): 0=generic
- `flags` (u32): bit0=IVFFlat present, bit1=IVFPQ present, bit2=PQ4, bit3=PQ8, bit4=cosine_norms_present, bit5=l2_sqnorms_present
- `d` (u32), `m` (u16), `ks` (u16), `kc` (u32)
- `id_bits` (u8) {32,64}; `code_group_g` (u8) {4,8}; reserved (6 bytes)
- `N_total` (u64)
- `generation` (u64)
- `toc_offset` (u64), `toc_entries` (u32)
- `header_crc32` (u32) over bytes 0..(end-4) after setting crc to 0
- 256 bytes total (pad with zeros)

TOC (table of contents)
- Array of entries, each:
  - `type` (u32): enum: 1=Centroids, 2=Codebooks, 3=CentroidNorms, 4=ListsDesc, 5=IDs, 6=Codes, 7=Vecs, 8=NormsInv, 9=NormsSq, 10=IDMap, 11=Tombstones, 12=Telemetry, 13=FreeList, 14=WALAnchor
  - `offset` (u64), `size` (u64) — file offsets in bytes
  - `align` (u32) (e.g., 64 or 4096), `flags` (u32)
  - `crc32` (u32)
  - `reserved` (u32)

Sections & Layout
- Centroids (type=1): AoS `[kc][d]` f32, 64‑B aligned. Optional centroid norms (type=3) f32 `[kc]`.
- Codebooks (type=2): PQ codebooks `[m][ks][dsub]` f32 contiguous, 64‑B aligned. Optional centroid_norms `[m*ks]` next to codebooks or separate (type=3).
- ListsDesc (type=4): array of `kc` descriptors (see below). 64‑B aligned.
- IDs (type=5): concatenated per‑list ID arrays (u32/u64). Each list descriptor points to its region via offset/length.
- Codes (type=6): concatenated per‑list PQ code arrays. Layout AoS or InterleavedBlock(g) as configured (g in header). PQ4 packed two nibbles/byte.
- Vecs (type=7): concatenated per‑list f32 AoS or SoA‑blocked (choose one per index; declared in descriptor flags). Present for IVF‑Flat.
- NormsInv (type=8): inverse norms (#09) `[N_total]` dtype f16/bf16/f32; NormsSq (type=9) `[N_total]` f32.
- IDMap (type=10): dense internal↔external mapping (#50): `ext_ids[N_total]` (u64) and optional reverse map `int_ids` indexed by external id domain if configured.
- Tombstones (type=11): bitset length `N_total`; 1=tombstoned.
- Telemetry (type=12): snapshot of counters (#46) (optional, read‑only advisory).
- FreeList (type=13): reserved; not used in P0. WALAnchor (type=14): pointer to WAL metadata.

List Descriptor (per list)
- `format` (u8): 0=Empty, 1=Flat, 2=PQ8, 3=PQ4
- `group_g` (u8): 4 or 8 (for PQ); 0 otherwise
- `id_bits` (u8): 32/64; `reserved` (u8)
- `length` (u32), `capacity` (u32)
- `ids_offset` (u64), `codes_offset` (u64), `vecs_offset` (u64)
- `ids_stride` (u32), `codes_stride` (u32), `vecs_stride` (u32), `reserved2` (u32)

Endianness & Compatibility
- File is stored little‑endian by default. Readers must handle both endiannesses by swapping multi‑byte fields when `endianness` mismatches host.
- SemVer policy: `version_major` bump may change layouts; minor bumps add optional sections/flags. Unknown TOC entries are skipped.

Alignment
- All sections aligned to `max(64, page_size)`; inner arrays (IDs, Codes, Vecs) aligned to 64 bytes to support NEON loads and write‑combining.

Update Protocol (Append‑only)
- For streaming appends (#30):
  1) Reserve space in target sections (IDs/Codes/Vecs) by extending file and writing data at `tail` offsets.
  2) Write append record to WAL: `{list_id, old_len, delta, data_offsets}` with crc; fsync WAL.
  3) Write payload bytes; fsync data.
  4) Atomically update `length` in ListsDesc with release semantics; write `commit` record to WAL; fsync.
  5) On crash: replay WAL to reconcile lengths to the last committed payload; discard partial tails.
- Periodic compaction: write a fresh `.vindex.tmp` with compacted lists and updated TOC; fsync; atomic rename over old; bump `generation`.

APIs
- Load / mmap:
  - `IndexMmap* index_mmap_open(const char* path, MmapOpts* opts);` (parses header, TOC, validates CRCs; maps sections with alignment; sets pointers)
  - `void index_mmap_close(IndexMmap* idx);`
  - `const float* mmap_centroids(const IndexMmap* idx, int* kc_out, int* d_out);`
  - `const float* mmap_codebooks(const IndexMmap* idx, int* m_out, int* ks_out, int* dsub_out);`
  - `const ListDesc* mmap_lists(const IndexMmap* idx, int* kc_out);` and accessors to list payloads.
- Update (append):
  - `int mmap_append_begin(IndexMmap* idx, int list_id, int add_len, AppendReservation* res);` // returns offsets for IDs/Codes/Vecs
  - `int mmap_append_commit(IndexMmap* idx, int list_id, const AppendReservation* res);`
  - `int mmap_wal_replay(IndexMmap* idx);`

Safety & Concurrency
- Readers: map read‑only; snapshot `length` before scanning a list; scan up to snapshot length.
- Writers: single process maintains WAL; per‑list locks coordinate writers (#30). Readers never see torn data due to commit protocol.

Telemetry (#46)
- Store a snapshot: total bytes, section sizes, list length histogram, last generation, last compaction time, WAL replay count.

Tests
- Round‑trip: build index in memory → serialize → mmap open → scan (#22/#04) equals in‑memory results.
- Crash simulation: kill process between WAL/data/commit steps; ensure replay yields consistent state.
- Endianness: write big‑endian test file; read on little‑endian host and verify contents.
- Alignment: verify section offsets are multiples of 64 and page_size; NEON path can read without faults.

Integration
- #22/#04 access codes/vecs/centroids in place; #30 uses append protocol; #09 norms live in dedicated sections; #50 ID maps stored alongside.

