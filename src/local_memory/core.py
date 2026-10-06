"""Engine glue: DB path, chunk dedup, search dispatch, tool-level operations.

Single entry point for server.py and cli.py. Functions take an explicit `db`
path when possible; the default comes from the MEMORY_DB_PATH environment
variable (or ~/.local/share/local-memory/memory.db).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import embeddings, schema
from . import search as search_mod

DEFAULT_DB = Path.home() / '.local' / 'share' / 'local-memory' / 'memory.db'
TFIDF_ROW_CAP = 50000


def db_path(db=None) -> Path:
    if db:
        return Path(db).expanduser()
    env = os.environ.get('MEMORY_DB_PATH', '').strip()
    return Path(env).expanduser() if env else DEFAULT_DB


def max_chunks() -> int:
    try:
        return int(os.environ.get('MEMORY_MAX_CHUNKS', '1000000'))
    except (TypeError, ValueError):
        return 1000000


def connect(db=None) -> sqlite3.Connection:
    """Open the database (creating the file/dir if needed) with Row factory."""
    path = db_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init(db=None) -> dict:
    """Create the database and full schema. Idempotent."""
    path = db_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        schema.ensure_schema(conn)
    finally:
        conn.close()
    return {'path': str(path), 'ok': True}


def load_chunks(conn: sqlite3.Connection) -> list:
    """Deduplicated chunk rows (newest occurrence per content), newest first.

    Materializes chunks_unique so the hot path is an index scan instead of a
    full-table GROUP BY. Falls back to the GROUP BY query if the materialized
    table is unavailable (e.g. pre-init database).
    """
    cap = max_chunks()
    try:
        max_synced = conn.execute(
            'SELECT COALESCE(MAX(id), 0) FROM chunks_unique').fetchone()[0]
        max_total = conn.execute(
            'SELECT COALESCE(MAX(id), 0) FROM chunks').fetchone()[0]
        if max_synced < max_total:
            conn.execute("""
                INSERT INTO chunks_unique (id, session_id, position, content,
                    reasoning, tool_calls, tool_call_results, prev_id, next_id, metadata)
                SELECT c.id, c.session_id, c.position, c.content, c.reasoning,
                    c.tool_calls, c.tool_call_results, c.prev_id, c.next_id, c.metadata
                FROM chunks c WHERE c.id > ?
                ON CONFLICT(content) DO UPDATE SET
                    id=excluded.id, session_id=excluded.session_id,
                    position=excluded.position, reasoning=excluded.reasoning,
                    tool_calls=excluded.tool_calls,
                    tool_call_results=excluded.tool_call_results,
                    prev_id=excluded.prev_id, next_id=excluded.next_id,
                    metadata=excluded.metadata
            """, (max_synced,))
            conn.commit()
        rows = conn.execute(
            'SELECT id, session_id, position, content FROM chunks_unique '
            'ORDER BY id DESC LIMIT ?',
            (cap,),
        ).fetchall()
        if rows:
            return rows
    except sqlite3.Error:
        pass
    return conn.execute(
        'SELECT id, session_id, position, content FROM chunks '
        'WHERE id IN (SELECT MAX(id) FROM chunks GROUP BY content) '
        'ORDER BY id DESC LIMIT ?',
        (cap,),
    ).fetchall()


def keyword_search(query: str, limit: int = 5, db=None) -> dict:
    """Keyword search (exact/fts/like). Returns {'results': [...], 'engine': str}."""
    conn = connect(db)
    try:
        results, engine = search_mod.keyword_search(conn, query, limit)
        return {'results': results, 'engine': engine}
    finally:
        conn.close()


def semantic_search(query: str, limit: int = 5, db=None) -> dict:
    """Semantic search: embeddings -> TF-IDF -> keyword fallback.

    Returns {'results': [...], 'engine': str} where engine is one of
    'embeddings' | 'tfidf' | 'exact' | 'fts' | 'like' | 'keyword-fallback' | 'empty'.
    """
    conn = connect(db)
    try:
        if not query or not query.strip():
            return {'results': [], 'engine': 'empty'}
        rows = load_chunks(conn)
        if not rows:
            return {'results': [], 'engine': 'empty'}
        try:
            results = embeddings.embed_search(
                rows, query, limit, db_path=db_path(db))
            search_mod.add_context(results, conn)
            return {'results': results, 'engine': 'embeddings'}
        except Exception:
            pass
        try:
            results = search_mod.tfidf_search(rows[:TFIDF_ROW_CAP], query, limit)
            search_mod.add_context(results, conn)
            return {'results': results, 'engine': 'tfidf'}
        except ImportError:
            pass
        results, engine = search_mod.keyword_search(conn, query, limit)
        return {'results': results, 'engine': 'keyword-fallback'}
    finally:
        conn.close()


def overview(days: int = 7, db=None) -> list[dict]:
    """Recent sessions, newest activity first (capped at 50)."""
    conn = connect(db)
    try:
        threshold = (
            datetime.now(timezone.utc) - timedelta(days=max(0, int(days)))
        ).strftime('%Y-%m-%dT%H:%M:%S')
        rows = conn.execute(
            'SELECT s.id, COALESCE(t.title, \'\') AS title, '
            'COALESCE(s.message_count, 0) AS message_count, s.start_time, '
            'COALESCE(t.last_active, s.end_time, s.start_time) AS last_active '
            'FROM sessions s '
            'LEFT JOIN session_summaries t ON t.session_id = s.id '
            'WHERE COALESCE(t.last_active, s.end_time, s.start_time) >= ? '
            'ORDER BY last_active DESC, s.start_time DESC LIMIT 50',
            (threshold,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch(session_id: str, position: int = 0, limit: int = 20, db=None) -> dict:
    """Consecutive chunks of one session, with parsed tool_calls when possible."""
    conn = connect(db)
    try:
        rows = conn.execute(
            'SELECT position, content, reasoning, tool_calls, tool_call_results '
            'FROM chunks WHERE session_id = ? AND position >= ? '
            'ORDER BY position LIMIT ?',
            (session_id, int(position), int(limit)),
        ).fetchall()
        chunks: list[dict] = []
        for r in rows:
            item = {'position': r['position'], 'content': r['content']}
            if r['reasoning']:
                item['reasoning'] = r['reasoning']
            for field in ('tool_calls', 'tool_call_results'):
                raw = r[field]
                if raw:
                    try:
                        item[field] = json.loads(raw)
                    except (TypeError, ValueError):
                        item[field] = raw
            chunks.append(item)
        return {'session_id': session_id, 'chunks': chunks}
    finally:
        conn.close()


def status(db=None) -> dict:
    """Database statistics: row counts, size, cached embeddings."""
    path = db_path(db)

    def _zero() -> dict:
        return {'db_path': str(path), 'exists': False, 'sessions': 0, 'chunks': 0,
                'unique_chunks': 0, 'summaries': 0, 'db_size_bytes': 0,
                'embeddings_cached': 0}

    if not path.exists():
        return _zero()
    conn = sqlite3.connect(str(path))
    try:
        def _count(table: str) -> int:
            try:
                return int(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
            except sqlite3.Error:
                return 0

        try:
            unique = int(conn.execute(
                'SELECT COUNT(DISTINCT content) FROM chunks').fetchone()[0])
        except sqlite3.Error:
            unique = 0
        return {
            'db_path': str(path),
            'exists': True,
            'sessions': _count('sessions'),
            'chunks': _count('chunks'),
            'unique_chunks': unique,
            'summaries': _count('session_summaries'),
            'db_size_bytes': path.stat().st_size,
            'embeddings_cached': _count('chunk_embeddings'),
        }
    finally:
        conn.close()


def ingest(session_id: str, content: str, project: str = '',
           metadata=None, db=None) -> dict:
    """Append one chunk to a session (position = max+1). Upserts sessions.

    This is the write channel: a client hook or plugin calls it once per
    message/chunk. The FTS index is maintained by schema triggers.
    """
    if not session_id:
        raise ValueError('session_id is required')
    if not content:
        raise ValueError('content must not be empty')
    path = db_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        schema.ensure_schema(conn)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        last = conn.execute(
            'SELECT MAX(position) FROM chunks WHERE session_id = ?',
            (session_id,),
        ).fetchone()[0]
        position = int(last) + 1 if last is not None else 0
        meta = json.dumps(metadata) if metadata is not None else '{}'
        conn.execute(
            'INSERT INTO chunks (session_id, position, content, metadata) '
            'VALUES (?, ?, ?, ?)',
            (session_id, position, content, meta),
        )
        conn.execute("""
            INSERT INTO sessions (id, project, start_time, message_count, total_chars)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                project = CASE WHEN excluded.project != ''
                               THEN excluded.project ELSE sessions.project END,
                end_time = excluded.start_time,
                message_count = sessions.message_count + 1,
                total_chars = sessions.total_chars + excluded.total_chars
        """, (session_id, project, now, len(content)))
        conn.execute("""
            INSERT INTO session_summaries
                (session_id, title, message_count, start_time, last_active)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active = excluded.last_active,
                title = COALESCE(NULLIF(session_summaries.title, ''),
                                 excluded.title),
                message_count = COALESCE(session_summaries.message_count, 0) + 1
        """, (session_id, session_id, now, now))
        conn.commit()
        return {'session_id': session_id, 'position': position, 'ok': True}
    finally:
        conn.close()


def build(limit: int | None = None, db=None) -> dict:
    """Prebuild the embedding cache (requires EMBED_API_URL). Returns counts."""
    return embeddings.build(db_path(db), limit=limit)
