# local-memory for Claude Code

## 1. Install

```bash
pip install local-memory
local-memory init
```

## 2. Register the MCP server

```bash
claude mcp add local-memory -- local-memory serve
```

## 3. Use it

Ask Claude things like:

- "search your memory for the postgres connection pool timeout"
- "show me a summary of recent sessions"
- "remember that the deploy pipeline runs migrations before rollout"

Claude will call `memory_search`, `memory_overview`, `memory_ingest`, etc.
automatically when useful.

## Verify

```bash
claude mcp list          # local-memory should appear
local-memory status      # database statistics
```
