# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""MCP stdio server: exposes the local-memory engine as six tools.

Run with `local-memory serve` (or `python -m local_memory serve`). All tools
are async, return JSON-serializable dicts, and convert errors into a
{'error': ...} payload instead of crashing the stdio transport.
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from . import __version__, core

mcp = FastMCP(
    'local-memory',
    instructions=(
        'Local-first long-term memory for AI agents. SQLite, offline, zero cloud. '
        'Tools: memory_overview, memory_search, memory_search_semantic, '
        'memory_fetch, memory_status, memory_ingest.'
    ),
)


def _safe(fn, *args, **kwargs):
    """Tool boundary: never let an exception escape to the transport."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - by design, tool-level error capture
        return {'error': f'{type(e).__name__}: {e}'}


@mcp.tool()
async def memory_overview(days: int = 7) -> dict:
    """List recent sessions (newest activity first).

    Args:
        days: look-back window in days (default 7).

    Returns:
        {"sessions": [{session_id, title, message_count, start_time, last_active}]}
    """
    return _safe(lambda: {'sessions': core.overview(days=days)})


@mcp.tool()
async def memory_search(query: str, limit: int = 5) -> dict:
    """Keyword search over all stored sessions (exact / FTS5 / LIKE).

    Args:
        query: search text (code snippets are searched literally first).
        limit: max results (default 5).

    Returns:
        {"results": [{session_id, position, content, score, context}],
         "engine": "exact" | "fts" | "like" | "empty"}
    """
    return _safe(lambda: core.keyword_search(query, limit=limit))


@mcp.tool()
async def memory_search_semantic(query: str, limit: int = 5) -> dict:
    """Semantic search: embeddings API -> TF-IDF -> keyword fallback.

    Args:
        query: natural-language search text.
        limit: max results (default 5).

    Returns:
        {"results": [{session_id, position, content, score, context}],
         "engine": "embeddings" | "tfidf" | "keyword-fallback" | ...}
    """
    return _safe(lambda: core.semantic_search(query, limit=limit))


@mcp.tool()
async def memory_fetch(session_id: str, position: int = 0, limit: int = 20) -> dict:
    """Read consecutive chunks of one session (content, reasoning, tool calls).

    Args:
        session_id: session to read.
        position: starting chunk position (default 0).
        limit: max chunks (default 20).

    Returns:
        {"session_id": str, "chunks": [{position, content, ...}]}
    """
    return _safe(lambda: core.fetch(session_id, position=position, limit=limit))


@mcp.tool()
async def memory_status() -> dict:
    """Database statistics: session/chunk counts, size, cached embeddings."""
    return _safe(core.status)


@mcp.tool()
async def memory_ingest(session_id: str, content: str,
                        project: str = '', metadata: dict | None = None) -> dict:
    """Write one chunk into a session (append; position auto-increments).

    The write channel for clients: call once per message/chunk you want the
    agent to remember.

    Args:
        session_id: session id to append to.
        content: the text to store.
        project: optional project tag (empty = keep existing).
        metadata: optional JSON object stored with the chunk.

    Returns:
        {"session_id": str, "position": int, "ok": true}
    """
    return _safe(lambda: core.ingest(
        session_id, content, project=project, metadata=metadata))


def _self_test() -> dict:
    """Sanity payload used by smoke tests and the CLI --version flow."""
    return {'name': 'local-memory', 'version': __version__,
            'tools': [t for t in (
                'memory_overview', 'memory_search', 'memory_search_semantic',
                'memory_fetch', 'memory_status', 'memory_ingest')]}


def tools_manifest() -> str:
    """JSON manifest of registered tools (for docs / debugging)."""
    return json.dumps(_self_test(), ensure_ascii=False)


def run_http(host: str = '127.0.0.1', port: int = 8000) -> None:
    """MCP over HTTP (streamable-http transport), for non-stdio MCP clients."""
    mcp.settings.host = host
    mcp.settings.port = int(port)
    mcp.run(transport='streamable-http')


def main() -> None:
    """stdio transport entry point (called by the CLI `serve` subcommand)."""
    mcp.run()
