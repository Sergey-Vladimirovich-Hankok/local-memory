# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""MCP tool handler tests: call the async handlers directly (no stdio transport).

Verifies the six tools return JSON-serializable payloads and that errors
surface as {'error': ...} instead of exceptions.
"""
from __future__ import annotations

import asyncio
import json

from local_memory import server


def run(coro):
    return asyncio.run(coro)


def test_status_shape(db):
    out = run(server.memory_status())
    assert out['exists'] is True
    assert out['sessions'] == 3
    assert out['chunks'] == 32
    assert out['unique_chunks'] == 31  # 32 chunks, one shared duplicate pair
    json.dumps(out)  # must be JSON-serializable


def test_overview_lists_three_sessions(db):
    out = run(server.memory_overview(days=7))
    ids = {s['id'] for s in out['sessions']}
    assert {'alpha', 'beta', 'gamma'} <= ids


def test_search_tool(db):
    out = run(server.memory_search('postgres connection pool', limit=3))
    assert out['engine'] in ('fts', 'like', 'exact')
    assert len(out['results']) <= 3
    json.dumps(out)


def test_semantic_tool_offline(db):
    out = run(server.memory_search_semantic('postgres timeout', limit=3))
    assert out['engine'] == 'tfidf'  # EMBED_OFF=1 in the fixture
    assert out['results']


def test_fetch_tool(db):
    out = run(server.memory_fetch('alpha', position=0, limit=3))
    assert out['session_id'] == 'alpha'
    assert len(out['chunks']) == 3
    assert out['chunks'][0]['position'] == 0


def test_ingest_tool_roundtrip(db):
    out = run(server.memory_ingest(
        'tool-sess', 'a brand new memory from the tool layer', project='demo'))
    assert out['ok'] is True and out['position'] == 0
    found = run(server.memory_search('brand new memory from the tool layer'))
    assert found['results'] and found['results'][0]['session_id'] == 'tool-sess'


def test_ingest_tool_error_is_captured(db):
    out = run(server.memory_ingest('', 'content'))
    assert 'error' in out
    assert 'ValueError' in out['error']


def test_fetch_unknown_session_is_empty(db):
    out = run(server.memory_fetch('ghost-session'))
    assert out['chunks'] == []


def test_tools_are_registered(db):
    names = server.tools_manifest()
    assert 'memory_overview' in names
    assert 'memory_ingest' in names


def test_all_tools_json_serializable(db):
    payloads = [
        run(server.memory_overview(days=1)),
        run(server.memory_search('redis')),
        run(server.memory_search_semantic('redis')),
        run(server.memory_fetch('beta')),
        run(server.memory_status()),
    ]
    for p in payloads:
        json.dumps(p)
