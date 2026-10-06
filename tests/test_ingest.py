# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Ingest tests: append semantics, upserts, FTS round-trip, metadata."""
from __future__ import annotations

import sqlite3

from local_memory import core, search


def test_ingest_appends_and_increments_position(db):
    first = core.ingest('new-sess', 'zebra unicorn quantum')
    second = core.ingest('new-sess', 'second chunk of the new session')
    assert first['position'] == 0
    assert second['position'] == 1
    assert first['session_id'] == 'new-sess'


def test_ingest_roundtrip_searchable(db):
    core.ingest('roundtrip', 'the zeppelin carrying hot cocoa left at dawn',
                project='travel')
    out = core.keyword_search('zeppelin cocoa')
    assert out['results']
    assert out['results'][0]['session_id'] == 'roundtrip'


def test_ingest_updates_session_counters(db):
    core.ingest('counters', 'first line about lighthouses')
    core.ingest('counters', 'second line about lighthouses')
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            'SELECT message_count, total_chars, project FROM sessions '
            "WHERE id = 'counters'").fetchone()
        assert row[0] == 2
        assert row[1] == len('first line about lighthouses') + len('second line about lighthouses')
        assert row[2] == ''
    finally:
        conn.close()


def test_ingest_keeps_existing_project(db):
    core.ingest('proj', 'line one', project='kept')
    core.ingest('proj', 'line two')  # empty project must not clobber
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute(
            "SELECT project FROM sessions WHERE id = 'proj'").fetchone()[0] == 'kept'
    finally:
        conn.close()


def test_ingest_is_fts_indexed_via_trigger(db):
    core.ingest('fts-sess', 'crystalline hummingbird observatory')
    conn = sqlite3.connect(str(db))
    try:
        rows = search.fts_search(conn, 'crystalline hummingbird', 5)
        assert rows, 'new chunk must be visible to FTS immediately'
        assert any('hummingbird' in r[3] for r in rows)
    finally:
        conn.close()


def test_ingest_stores_metadata(db):
    core.ingest('meta-sess', 'metadata probe chunk',
                metadata={'source': 'hook', 'n': 7})
    conn = sqlite3.connect(str(db))
    try:
        meta = conn.execute(
            "SELECT metadata FROM chunks WHERE content = 'metadata probe chunk'"
        ).fetchone()[0]
        assert '"source"' in meta and 'hook' in meta
    finally:
        conn.close()


def test_ingest_validates_input(db):
    import pytest
    with pytest.raises(ValueError):
        core.ingest('', 'content')
    with pytest.raises(ValueError):
        core.ingest('x', '')


def test_parallel_ingest_positions_are_unique(db):
    # regression: position = MAX(position)+1 was racy under concurrent
    # writers (ThreadingHTTPServer) and produced duplicate positions
    import concurrent.futures as cf

    def one(i):
        return core.ingest('race-sess', f'parallel chunk {i}')['position']

    with cf.ThreadPoolExecutor(max_workers=20) as pool:
        positions = list(pool.map(one, range(20)))
    assert sorted(positions) == list(range(20))


def test_duplicate_position_insert_is_rejected(db):
    # the UNIQUE index on (session_id, position) is the last line of defense
    import pytest
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO chunks (session_id, position, content) "
            "VALUES ('dup-sess', 0, 'first')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chunks (session_id, position, content) "
                "VALUES ('dup-sess', 0, 'second')")
        conn.rollback()
    finally:
        conn.close()


def test_legacy_duplicate_positions_are_deduped_on_open(tmp_path):
    # simulate an older database (pre-unique-index) that already contains
    # racy duplicate positions; ensure_schema must keep the newest row per
    # slot and still create the index
    path = tmp_path / 'legacy.db'
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            'CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'session_id TEXT, position INTEGER, content TEXT, '
            'reasoning TEXT, tool_calls TEXT, tool_call_results TEXT, '
            'prev_id INTEGER, next_id INTEGER, metadata TEXT DEFAULT \'{}\')')
        conn.execute(
            "INSERT INTO chunks (session_id, position, content) "
            "VALUES ('legacy', 0, 'old row')")
        conn.execute(
            "INSERT INTO chunks (session_id, position, content) "
            "VALUES ('legacy', 0, 'new row')")
        conn.commit()
    finally:
        conn.close()
    reopened = core.connect(path)
    try:
        rows = reopened.execute(
            "SELECT content FROM chunks WHERE session_id = 'legacy' "
            'ORDER BY position').fetchall()
        assert [r['content'] for r in rows] == ['new row']
    finally:
        reopened.close()


def test_fetch_returns_parsed_tool_calls(db):
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO chunks (session_id, position, content, reasoning, tool_calls) "
            "VALUES ('tool-sess', 0, 'ran the migration', 'decided to be careful', "
            "'[{\"name\": \"bash\", \"args\": {\"command\": \"migrate\"}}]')")
        conn.commit()
    finally:
        conn.close()
    out = core.fetch('tool-sess')
    chunk = out['chunks'][0]
    assert chunk['reasoning'] == 'decided to be careful'
    assert chunk['tool_calls'][0]['name'] == 'bash'
    assert out['session_id'] == 'tool-sess'


def test_fetch_unknown_session_is_empty(db):
    out = core.fetch('no-such-session')
    assert out['chunks'] == []


def test_overview_lists_recent_sessions(db):
    out = core.overview(days=7)
    ids = {s['id'] for s in out}
    assert {'alpha', 'beta', 'gamma'} <= ids
    for s in out:
        for key in ('id', 'title', 'message_count', 'start_time', 'last_active'):
            assert key in s
