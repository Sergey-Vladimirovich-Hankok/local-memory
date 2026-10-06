# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-06

### Added
- HTTP transports: `serve --transport http` (plain REST, stdlib-only
  ThreadingHTTPServer: `/health`, `/status`, `/overview`, `/search`,
  `/ingest`, `/fetch`, `/tools`) and `serve --transport mcp-http`
  (MCP streamable-http). Loopback default; off-loopback binding refuses to
  start without a bearer token (`MEMORY_HTTP_TOKEN`).
- sqlite-vec vector backend (optional extra `[vector]`): in-database KNN
  search with normalized vectors, `local-memory migrate-vectors [--remove-npy]`
  to copy the `.npy` cache into the vec table, `MEMORY_VECTOR_BACKEND=auto|npy|sqlitevec`
  for selection. The `.npy` numpy-matmul path remains the fallback.
- `local-memory config --client claude|opencode|cursor|generic` — generates
  the MCP client snippet (guarded by tests against drift from `examples/`);
  `--setup` runs the registration command (claude), `--dry-run` prints it.
- MCP stdio server with six tools: `memory_overview`, `memory_search`,
  `memory_search_semantic`, `memory_fetch`, `memory_status`, `memory_ingest`.
- CLI: `init`, `serve`, `search`, `ingest`, `fetch`, `build`, `status`.
- Keyword search: literal (exact), FTS5, and LIKE engines with a context window.
- Semantic search: OpenAI-compatible embeddings API with a numpy matrix cache,
  TF-IDF fallback (optional `scikit-learn`), keyword fallback.
- SQLite schema with FTS5 triggers and a materialized content-dedup table.
- Test suite with fully synthetic fixtures (no real data, no network).

### Fixed
- `serve --transport mcp-http` now enforces the same bind policy as REST:
  off-loopback binding without a token refuses to start (security).
- Ingest race: parallel writes no longer produce duplicate positions
  (`UNIQUE(session_id, position)` + per-process lock + retry).
- Fresh databases work without an explicit `init` — `connect()` ensures the
  schema, so `/search`, `/overview`, `/fetch` return 200 instead of 500.
- Embedding cache key now includes the database filename, so two databases in
  the same directory never silently share vectors.
- HTTP: `limit` is clamped to `[1, 1000]` (negative values no longer mean
  unlimited), invalid `Content-Length` returns 400 instead of 500, and
  `X-Token`/`Authorization` headers are stripped of whitespace.
