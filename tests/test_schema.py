"""Schema tests: idempotency, table presence, FTS trigger behavior."""
from __future__ import annotations

import sqlite3

from local_memory import schema


def _conn(db):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def test_ensure_schema_is_idempotent(db):
    conn = _conn(db)
    try:
        schema.ensure_schema(conn)
        schema.ensure_schema(conn)  # second run must not raise
    finally:
        conn.close()


def test_all_tables_exist(db):
    conn = _conn(db)
    try:
        names = {r['name'] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        for expected in ('sessions', 'chunks', 'memories_fts', 'session_summaries',
                         'chunk_embeddings', 'chunks_unique'):
            assert expected in names, f'missing table {expected}'
    finally:
        conn.close()


def test_fts_trigger_insert_and_delete(db):
    conn = _conn(db)
    try:
        before = conn.execute('SELECT COUNT(*) c FROM memories_fts').fetchone()['c']
        conn.execute(
            "INSERT INTO chunks (session_id, position, content) VALUES ('x', 0, 'trigger probe text')")
        conn.commit()
        after = conn.execute('SELECT COUNT(*) c FROM memories_fts').fetchone()['c']
        assert after == before + 1
        rowid = conn.execute(
            "SELECT id FROM chunks WHERE content = 'trigger probe text'").fetchone()['id']
        conn.execute('DELETE FROM chunks WHERE id = ?', (rowid,))
        conn.commit()
        final = conn.execute('SELECT COUNT(*) c FROM memories_fts').fetchone()['c']
        assert final == before
    finally:
        conn.close()


def test_seed_counts(db):
    conn = _conn(db)
    try:
        sessions = conn.execute('SELECT COUNT(*) c FROM sessions').fetchone()['c']
        chunks = conn.execute('SELECT COUNT(*) c FROM chunks').fetchone()['c']
        assert sessions == 3
        assert chunks == 32  # 3 x 10 + 2 shared duplicates
    finally:
        conn.close()
