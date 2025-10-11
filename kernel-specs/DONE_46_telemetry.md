ID: 46 — Index Stats & Telemetry (MUST)

Purpose
- Collect low-overhead, per-query and global statistics to guide tuning (e.g., adaptive search width #42), monitor performance/health, and support persistence snapshots (S1).

Role
- M

Design Goals
- Near-zero overhead when disabled; thread-local accumulation when enabled; deterministic semantics; JSON-exportable snapshots; optional persistence into mmap file (read-only advisory).

Data Model
- Scopes: `Query`, `Build`, `Maintenance`.
- Structures:
  - `TelemetryGlobal`: process-wide counters and histograms; lock-light aggregation.
  - `TelemetryTLS`: thread-local counters reset per query; merged on `end_query`.
  - `QueryStats`: returned to caller for a finished query.
- Histograms: fixed bucket hist with power-of-two buckets for sizes/latencies; P50/P90/P99 derived.

Key Counters (Query)
- Work:
  - `kc_scored`, `lists_routed` (nprobe), `lists_scanned`, `codes_scanned`, `vecs_scored` (re-rank), `candidates_emitted`, `candidates_unique`, `candidates_kept` (reservoir), `topk_selected`.
- Time (ns): `t_lut_build`, `t_scan_adc`, `t_score_flat`, `t_topk`, `t_merge`, `t_dedup`, `t_reservoir`, `t_rerank`, `t_total`.
- Bytes: `bytes_lut`, `bytes_codes`, `bytes_vecs`, `bytes_ids`, `bytes_norms`.
- Path flags (bitfield): `used_dot_trick`, `used_cosine`, `used_interleaved_codes`, `used_u4`, `used_prefetch`, `used_heap_merge`.
- Saturation/quality proxies: `reservoir_tau` (final threshold), `heap_sifts`, `quickselect_calls`, `dup_ratio`, `beam_expansions`.

Key Counters (Build/Maintenance)
- PQ train: iterations, empties repaired, distortion trajectory.
- Append: growth events, bytes written, wal_replays (S1), list length histogram.
- Norm cache: rows built/updated, zero-norms.

APIs
- Init/shutdown
  - `void telem_init(const TelemetryConfig* cfg);` // enable flags, sampling rate, sinks
  - `void telem_shutdown();`
- Per-thread / per-query
  - `void telem_thread_init();`
  - `void telem_begin_query(const QueryCtx* qctx);` // captures metric, d, m, ks, nprobe target
  - `void telem_end_query(QueryStats* out);` // merges TLS → global; returns a copy
- Event helpers (macros/inlines)
  - `TELEM_INC(name, v)`, `TELEM_ADD_BYTES(name, v)`, `TELEM_FLAG(bit)`, `TELEM_TIMER_GUARD(name)` (RAII timer), `TELEM_SET(name, v)`
- Export
  - `size_t telem_snapshot_json(char* buf, size_t cap);`
  - `int telem_snapshot_struct(TelemetryGlobal* out);`
  - `int telem_snapshot_to_file(const char* path);`

Configuration (`TelemetryConfig`)
- `enabled` (bool), `sample_rate` (e.g., 1.0, 0.1), `max_hist_buckets`, `sink_cb` (optional callback per QueryStats), `persist_snapshot` (bool), `persist_path`.

Implementation Notes
- TLS accumulation via thread_local struct; merging uses atomic fetch-add on 64-bit counters and striped locks for histograms.
- Timers use `mach_absolute_time()` on Apple; convert to ns with cached ratio; fall back to `clock_gettime` elsewhere.
- Compile-time guard: `#if VINDEX_TELEM` to compile out macros entirely.

QueryStats Schema (subset)
- `{ metric, d, m, ks, nprobe, C, K, kc_scored, lists_scanned, codes_scanned, vecs_scored, candidates_emitted, candidates_unique, candidates_kept, reservoir_tau, bytes_{lut,codes,vecs,ids,norms}, t_{lut,scan_adc,score_flat,topk,merge,dedup,reservoir,rerank,total}, flags }`

Persistence (S1)
- Optional TOC entry type=12 stores: rolling aggregates (since generation), list length histogram, last N (e.g., 1024) QueryStats summaries (metric/dim + coarse counters) to aid cold-start tuning.

Determinism & Overhead
- Counters are additive and do not affect algorithmic decisions unless caller reads them for adaptive control (#42). When disabled, macros are no-ops; overhead should be below 1–2% when enabled.

Tests
- Unit: macro compilation on/off, timer accuracy calibration, JSON snapshot schema validity.
- Integration: counters from #21, #22, #29, #32, #39, #40 increment as expected on synthetic workloads.
- Performance: measure overhead under hot path (#22) with telemetry on/off; target ≤2% overhead at sample_rate=1.0.

Reuse / Integration
- Emitted by kernels: #04/#05/#06/#21/#22/#23/#29/#30/#32/#39/#40 and S1/S2.

