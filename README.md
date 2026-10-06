# local-memory

**Local-first long-term memory for AI agents. SQLite, offline, zero cloud.**

> Your agent's memory stays on your machine. SQLite. No cloud. No API keys.
> Works offline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/badge/CI-pytest-green.svg)](.github/workflows/ci.yml)

local-memory is a standalone [MCP](https://modelcontextprotocol.io) server
(stdio transport) that gives any AI agent durable long-term memory: search over
past sessions (keyword, semantic, exact), a catalog of sessions, full session
reads, and a write channel for new memories. It works with Claude Code,
opencode, Cursor, Cline, Codex, or any other MCP client.

The engine is small and boring on purpose: one SQLite file, the Python
standard library, and numpy. The only other dependency is the `mcp` package,
which powers the stdio server.

---

## Why

AI agents are stateless by default. Every new session starts from zero: the
decisions made yesterday, the bugs already chased away, the project's
conventions, the exact error message you fixed last week — all of it is gone
unless you copy-paste it back in by hand.

The usual fix is a cloud memory API. But that means your conversations —
often your most sensitive work context — are stored, indexed, and processed on
someone else's servers. For local work that is a bad trade: the "memory" you
paid for is also the leak you did not want.

local-memory flips the trade. The memory is a plain SQLite database on your
machine, readable with any SQL client, back-upable with any backup tool, and
searchable with your own embeddings model (or without any model at all). The
agent gets the memory interface it wants; you keep the data you already had.

---

## How it works

```
                 +------------------------------------------------+
                 |              your machine (offline)            |
                 |                                                |
  MCP client     |   +----------------+          +-------------+  |
  (Claude Code,  |   |  local-memory  |  stdio   |   SQLite    |  |
   opencode,     +-->|  MCP server    |<-------->|  memory.db  |  |
   Cursor, ...)   |   |               |          |  + FTS5     |  |
   via stdio      |   +-------+-------+          +------+------+ |
                 |           |                              ^      |
                 |           | embeddings (optional,        |      |
                 |           | only if YOU configure it)    |      |
                 |           v                              |      |
                 |   +----------------+          +-------------+  |
                 |   | embeddings API |          | numpy matrix |  |
                 |   | (OpenAI-style) |          | cache (.npy) |  |
                 |   +----------------+          +-------------+  |
                 +-------------------------------------------------+
```

One database, three search engines, chosen automatically:

| Mode      | When used                              | How                                    |
|-----------|----------------------------------------|----------------------------------------|
| exact     | query looks like code (`a=b`, `fn(x)`) | literal LIKE, then FTS5                |
| keyword   | default for `memory_search`            | FTS5 with prefix expansion, LIKE safe  |
| semantic  | `memory_search_semantic`               | embeddings -> TF-IDF -> keyword        |

Every result carries a **context window**: the neighboring chunks around the
hit, so the agent sees what came before and after without an extra round
trip.

---

## Tools

| Tool                      | Args                                        | Returns |
|---------------------------|---------------------------------------------|---------|
| `memory_overview`         | `days=7`                                    | recent sessions: id, title, counts, times |
| `memory_search`           | `query`, `limit=5`                          | keyword hits + context, engine used      |
| `memory_search_semantic`  | `query`, `limit=5`                          | semantic hits + context, engine used     |
| `memory_fetch`            | `session_id`, `position=0`, `limit=20`      | consecutive chunks of one session        |
| `memory_status`           | —                                           | db statistics (counts, size, cache)      |
| `memory_ingest`           | `session_id`, `content`, `project`, `metadata` | write result (position, ok)          |

Result rows (a public contract — stable across versions):

```json
{
  "session_id": "alpha",
  "position": 3,
  "content": "the postgres connection pool timed out under load",
  "score": 0.83,
  "context": [
    {"position": 2, "content": "..."},
    {"position": 3, "content": "the postgres connection pool timed out under load"},
    {"position": 4, "content": "..."}
  ]
}
```

## Quickstart

### 1. Install

From source (recommended until the package is published on PyPI):

```bash
git clone https://github.com/Sergey-Vladimirovich-Hankok/local-memory
cd local-memory
pip install -e .            # core (keyword + FTS search)
pip install -e ".[semantic]"  # + TF-IDF semantic search fallback (scikit-learn)
```

Once available on PyPI:

```bash
pip install local-memory
# optional: TF-IDF semantic search fallback
pip install local-memory[semantic]
```

### 2. Initialize the database

```bash
local-memory init          # creates ~/.local/share/local-memory/memory.db
local-memory status        # {"exists": true, "sessions": 0, ...}
```

### 3. Write a memory

```bash
local-memory ingest --session-id demo --project myproject \
  --text "the postgres connection pool timed out under load, raised max size"
```

### 4. Search it

```bash
local-memory search "postgres connection pool" -k 5
local-memory search "pool timeout" --semantic
```

### 5. Connect an MCP client

Claude Code:

```bash
claude mcp add local-memory -- local-memory serve
```

opencode (`opencode.json`):

```json
{
  "mcp": {
    "local-memory": {
      "command": "uvx",
      "args": ["local-memory", "serve"]
    }
  }
}
```

Cursor / any generic MCP client (`mcpServers` snippet):

```json
{
  "mcpServers": {
    "local-memory": {
      "command": "uvx",
      "args": ["local-memory", "serve"]
    }
  }
}
```

> If you installed from source (step 1) instead of PyPI, `uvx` cannot see the
> package — point `command` at the installed binary directly:
> `"command": "local-memory"` (see `examples/generic_mcp.json`).

Ready-made snippets live in [`examples/`](examples/).

---

## Configuration

Everything is configured via environment variables. There is no config file —
one process, one database, zero moving parts.

| Variable             | Default                                  | Meaning |
|----------------------|------------------------------------------|---------|
| `MEMORY_DB_PATH`     | `~/.local/share/local-memory/memory.db`  | path to the SQLite database |
| `EMBED_API_URL`      | *(empty = embeddings disabled)*          | OpenAI-compatible `/v1/embeddings` endpoint |
| `EMBED_MODEL`        | `local-embedder`                         | model name sent to the embeddings API |
| `EMBED_OFF`          | *(unset)*                                | `1` forces keyword/TF-IDF only |
| `MEMORY_CTX_WINDOW`  | `3`                                      | context window width (neighbors per side) |
| `MEMORY_MAX_CHUNKS`  | `1000000`                                | cap on rows scanned per search |
| `MEMORY_MATRIX_CAP`  | `5000`                                   | max new chunks embedded per call |

---

## Ingesting memory

`memory_ingest` / `local-memory ingest` is the write channel. A client hook or
plugin calls it once per message (or chunk) it wants the agent to remember.
Positions auto-increment per session; the FTS index is updated by a database
trigger, so ingested text is searchable immediately.

CLI:

```bash
local-memory ingest --session-id my-project --project my-project \
  --text "decided: use SQLite WAL mode for the memory db" \
  --metadata '{"source": "standup", "date": "2026-10-06"}'
local-memory ingest --session-id my-project --file notes.md
```

MCP tool:

```json
{
  "session_id": "my-project",
  "content": "decided: use SQLite WAL mode for the memory db",
  "project": "my-project",
  "metadata": {"source": "standup"}
}
```

Example hook script (call from your agent's post-message hook):

```bash
#!/usr/bin/env bash
# remember.sh — feed one message into local-memory
local-memory ingest \
  --session-id "project-${1}" \
  --text "${2}" \
  --project "hooks" >/dev/null
```

---

## Semantic search (optional)

`memory_search_semantic` tries, in order:

1. **Embeddings** — any OpenAI-compatible `/v1/embeddings` endpoint
   (llama.cpp `llama-server`, Ollama, vLLM, a local transformer service...).
   Point `EMBED_API_URL` at it; vectors are cached on disk next to the
   database (`embed_cache_*.npy`) and reused, so repeated queries are fast.
2. **TF-IDF** — character n-gram cosine similarity, offline, needs the
   optional `scikit-learn` extra.
3. **Keyword fallback** — the exact/FTS5/LIKE engines, with
   `"engine": "keyword-fallback"` in the response.

Prebuilding the cache (only needed for the embeddings path):

```bash
export EMBED_API_URL=http://localhost:8080/v1/embeddings
export EMBED_MODEL=my-local-embedding-model
local-memory build            # embed all chunks once
```

If no embeddings endpoint is configured, semantic search silently degrades to
TF-IDF/keyword. Nothing is sent over the network unless you configure a
`EMBED_API_URL`.

---

## Privacy

- The database, FTS index, and embedding caches are plain files under
  `MEMORY_DB_PATH` (default `~/.local/share/local-memory/`). Back them up or
  delete them like any other file.
- No telemetry, no update checks, no phone-home of any kind.
- The only network access in the entire codebase is the embeddings call, and
  it only happens if you set `EMBED_API_URL` yourself.
- `memory_ingest` stores content as-is, unencrypted, in SQLite. If your data
  is sensitive, protect the file with filesystem permissions.
- Nothing is ever sent to the authors of this project.

---

## Testing

The test suite runs fully offline against a synthetic in-memory-style fixture
database (three fake sessions, ~30 chunks). No real conversation data is read,
no network is touched.

```bash
python3 -m pytest tests/ -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: local-first is the
product promise; keep dependencies light; keep fixtures synthetic; add tests.

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities privately via the
repository's Security tab — not in public issues.

## License

[MIT](LICENSE) — do what you want, just keep the notice.

Attribution — if you use or copy this code, keep the author notice in the
source headers, the repository link
(https://github.com/Sergey-Vladimirovich-Hankok/local-memory), and the
contact email (kokgfnu@gmail.com). MIT license covers permissions;
attribution keeps the work traceable. See [NOTICE](NOTICE).

---

## Roadmap

- Qdrant / sqlite-vec backend for the embeddings matrix (still local).
- REST mode (HTTP transport alongside stdio) for non-MCP clients.
- Plugin packages for specific agents (Claude Code, opencode, Cursor).
- Session summaries via any local LLM (opt-in, offline).
- Not on the roadmap: cloud sync, web UI, Rust rewrites. If it must be local,
  it stays local.

## Disclaimer

local-memory is provided for educational and personal use. You are responsible
for what you ingest into it and for protecting the resulting database file.
The authors are not liable for lost, leaked, or misused data — the database is
yours, and so is the responsibility.

