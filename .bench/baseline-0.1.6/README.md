# Baseline: v0.1.6

These numbers were captured at v0.1.6, **before** the A9 `pruneNeighbors` diversity fix
(`da605ad`, "apply insertion's diversity heuristic in pruneNeighbors"), which intentionally
changes HNSW graph topology. Recall metrics here (`knn_graph_uniform.json` 0.957,
`knn_graph_clusters.json` 0.784, `hnsw_search.json`) are expected to move as a result.

Phase 3's no-regression gate should treat recall movement against this baseline as an expected
improvement from the diversity fix, not as drift -- or re-capture a post-Phase-1 reference to
compare against instead.
