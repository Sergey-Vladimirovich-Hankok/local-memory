# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Optional sqlite-vec backend: in-database vector search (extra: [vector]).

The .npy matrix cache (embeddings.py) remains the permanent fallback.
Backend selection:

    MEMORY_VECTOR_BACKEND = auto (default) | npy | sqlitevec

auto uses sqlite-vec when `import sqlite_vec` succeeds and the per-model
vec table is populated; otherwise the legacy .npy matmul path runs
unchanged. Vectors are stored normalized, so cosine similarity of the
L2 distance reported by sqlite-vec is `1 - d^2 / 2`.
"""
from __future__ import annotations

import os
import re
import sqlite3

import numpy as np


def available() -> bool:
    """True when the optional sqlite-vec extension can be imported."""
    try:
        import sqlite_vec  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend() -> str:
    """'sqlitevec' | 'npy' from MEMORY_VECTOR_BACKEND (default auto)."""
    env = os.environ.get('MEMORY_VECTOR_BACKEND', 'auto').strip().lower()
    if env in ('', 'auto'):
        return 'sqlitevec' if available() else 'npy'
    if env in ('npy', 'sqlitevec'):
        return env
    return 'npy'


def load_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into a connection (caller must keep it open)."""
    import sqlite_vec
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def table_name(model_key: str) -> str:
    """One vec0 table per embedding model: vec_<sanitized_key>.

    Model keys may contain characters that are invalid in SQL identifiers
    (e.g. 'text-embedding-0.6b'), so sanitize to [A-Za-z0-9_].
    """
    safe = re.sub(r'[^0-9A-Za-z_]', '_', model_key)
    return f'vec_{safe}'


def ensure_table(conn: sqlite3.Connection, model_key: str, dim: int) -> None:
    """Create the vec0 table for the model if missing. Idempotent."""
    name = table_name(model_key)
    existing = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,)).fetchone()
    if existing is None:
        conn.execute(
            f'CREATE VIRTUAL TABLE {name} USING vec0('
            f'chunk_id integer primary key, embedding float[{int(dim)}])')
        conn.commit()


def upsert(conn: sqlite3.Connection, model_key: str, ids, mat) -> int:
    """Replace vectors for `ids` (vec0 has no UPSERT: delete + insert).

    Returns the number of rows written.
    """
    name = table_name(model_key)
    ids = np.asarray(ids, dtype=np.int64).ravel()
    mat = np.asarray(mat, dtype=np.float32)
    if len(ids) == 0:
        return 0
    placeholders = ','.join('?' * len(ids))
    conn.execute(f'DELETE FROM {name} WHERE chunk_id IN ({placeholders})',
                 ids.tolist())
    conn.executemany(
        f'INSERT INTO {name} VALUES (?, ?)',
        [(int(cid), np.asarray(v, dtype=np.float32).tobytes())
         for cid, v in zip(ids, mat)],
    )
    conn.commit()
    return int(len(ids))


def knn(conn: sqlite3.Connection, model_key: str, query: np.ndarray,
        k: int) -> list[tuple[int, float]]:
    """Nearest `k` rows as (chunk_id, l2_distance), nearest first."""
    name = table_name(model_key)
    q = np.asarray(query, dtype=np.float32).tobytes()
    rows = conn.execute(
        f'SELECT chunk_id, distance FROM {name} '
        'WHERE embedding MATCH ? AND k = ? ORDER BY distance',
        (q, int(k)),
    ).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]


def table_count(conn: sqlite3.Connection, model_key: str) -> int:
    """Row count of the model's vec table (raises if the table is missing)."""
    row = conn.execute(
        f'SELECT COUNT(*) FROM {table_name(model_key)}').fetchone()
    return int(row[0]) if row else 0


def describe(db_path, model_key: str) -> str:
    """Storage actually usable for the model: 'sqlitevec' | 'npy' | 'none'."""
    if resolve_backend() == 'sqlitevec' and available():
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                load_extension(conn)
                if table_count(conn, model_key) > 0:
                    return 'sqlitevec'
            finally:
                conn.close()
        except sqlite3.Error:
            pass
    try:
        from . import embeddings
        ids_path, _ = embeddings.cache_paths(db_path)
        if ids_path.exists():
            return 'npy'
    except Exception:
        pass
    return 'none'
