# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MCP stdio server with six tools: `memory_overview`, `memory_search`,
  `memory_search_semantic`, `memory_fetch`, `memory_status`, `memory_ingest`.
- CLI: `init`, `serve`, `search`, `ingest`, `fetch`, `build`, `status`.
- Keyword search: literal (exact), FTS5, and LIKE engines with a context window.
- Semantic search: OpenAI-compatible embeddings API with a numpy matrix cache,
  TF-IDF fallback (optional `scikit-learn`), keyword fallback.
- SQLite schema with FTS5 triggers and a materialized content-dedup table.
- Test suite with fully synthetic fixtures (no real data, no network).
