# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Optional semantic search: OpenAI-compatible /v1/embeddings API + numpy cache.

Strictly opt-in and offline-safe:
    - no EMBED_API_URL  -> EmbeddingError -> caller falls back to TF-IDF/keyword
    - EMBED_OFF=1       -> EmbeddingError immediately
    - no network call is ever made unless the operator sets EMBED_API_URL

Chunk vectors are cached next to the database as embed_cache_<model>.{ids,mat}.npy
(mmap'd on load) and mirrored in the chunk_embeddings table for durability.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import urllib.request
from pathlib import Path

import numpy as np

BATCH = 8
MAX_TEXT_LEN = 1200
API_TIMEOUT = 60


class EmbeddingError(RuntimeError):
    """Raised when semantic search is unavailable (no API, offline, no cache)."""


def api_url() -> str:
    return os.environ.get('EMBED_API_URL', '').strip()


def model_name() -> str:
    return os.environ.get('EMBED_MODEL', 'local-embedder')


def is_off() -> bool:
    return os.environ.get('EMBED_OFF', '') not in ('', '0')


def matrix_cap() -> int:
    try:
        return int(os.environ.get('MEMORY_MATRIX_CAP', '5000'))
    except (TypeError, ValueError):
        return 5000


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', 'replace')).hexdigest()


def model_key() -> str:
    return model_name().replace('/', '__').replace('\\', '__')


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cache_paths(db_path) -> tuple[Path, Path]:
    base = Path(db_path).parent / f'embed_cache_{model_key()}'
    return Path(f'{base}.ids.npy'), Path(f'{base}.mat.npy')


def load_matrix_cache(db_path):
    """(ids, mat) from disk, or None if missing/corrupt. mat is mmap'd."""
    ids_path, mat_path = cache_paths(db_path)
    if not (ids_path.exists() and mat_path.exists()):
        return None
    try:
        ids = np.load(ids_path)
        mat = np.load(mat_path, mmap_mode='r')
        if len(ids) != mat.shape[0]:
            return None
        return ids, mat
    except Exception:
        return None


def save_matrix_cache(db_path, ids, mat) -> None:
    ids_path, mat_path = cache_paths(db_path)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_i, tmp_m = f'{ids_path}.tmp.npy', f'{mat_path}.tmp.npy'
    np.save(tmp_i, np.asarray(ids, dtype=np.int64))
    np.save(tmp_m, np.asarray(mat, dtype=np.float32))
    os.replace(tmp_i, ids_path)
    os.replace(tmp_m, mat_path)


def _api_embed_one(batch_texts: list[str]) -> list[np.ndarray]:
    """Embed one batch; on error split in half and retry (fail fast at size 1)."""
    payload = {'model': model_name(), 'input': [t[:MAX_TEXT_LEN] for t in batch_texts]}
    req = urllib.request.Request(
        api_url(),
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return [np.asarray(d['embedding'], dtype=np.float32) for d in data['data']]
    except Exception as e:
        if len(batch_texts) <= 1:
            raise EmbeddingError(f'embeddings API failed: {e}') from e
        mid = len(batch_texts) // 2
        return _api_embed_one(batch_texts[:mid]) + _api_embed_one(batch_texts[mid:])


def api_embed(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts via the configured OpenAI-compatible endpoint."""
    if not api_url():
        raise EmbeddingError('EMBED_API_URL is not set')
    out: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH):
        out.extend(_api_embed_one(texts[i:i + BATCH]))
    return out


def _select_rows(conn: sqlite3.Connection, limit: int | None) -> list:
    """Deduplicated chunk rows (newest occurrence per content), newest first."""
    if limit is not None:
        return conn.execute(
            'SELECT id, session_id, position, content FROM chunks '
            'WHERE id IN (SELECT MAX(id) FROM chunks GROUP BY content) '
            'ORDER BY id DESC LIMIT ?',
            (int(limit),),
        ).fetchall()
    return conn.execute(
        'SELECT id, session_id, position, content FROM chunks '
        'WHERE id IN (SELECT MAX(id) FROM chunks GROUP BY content) '
        'ORDER BY id DESC LIMIT ?',
        (int(os.environ.get('MEMORY_MAX_CHUNKS', '1000000')),),
    ).fetchall()


def embed_search(rows: list, query: str, limit: int = 5, db_path=None) -> list[dict]:
    """Semantic search over deduplicated chunk rows.

    Uses the embeddings API for the query and any missing chunks, then cosine
    similarity against the matrix cache. Raises EmbeddingError when unavailable.
    """
    if is_off():
        raise EmbeddingError('EMBED_OFF is set')
    if not api_url():
        raise EmbeddingError('EMBED_API_URL is not set')
    if not rows:
        raise EmbeddingError('no chunks to search')

    ids = [r['id'] for r in rows]
    cached = load_matrix_cache(db_path) if db_path is not None else None
    if cached is not None:
        known = {int(i) for i in cached[0]}
        base_ids, base_mat = cached
    else:
        known = set()
        base_ids = np.asarray([], dtype=np.int64)
        base_mat = np.empty((0, 0), dtype=np.float32)

    new_ids = sorted(i for i in ids if i not in known)
    if len(new_ids) > matrix_cap():
        raise EmbeddingError(
            f'{len(new_ids)} chunks missing embeddings (cap {matrix_cap()}); '
            'run: local-memory build')
    if new_ids:
        content_by_id = {r['id']: r['content'] for r in rows}
        vecs = api_embed([content_by_id[cid] for cid in new_ids])
        mat_new = np.stack([_norm(v) for v in vecs], dtype=np.float32)
        all_ids = np.concatenate([base_ids, np.asarray(new_ids, dtype=np.int64)])
        all_mat = mat_new if base_mat.size == 0 else np.concatenate(
            [np.asarray(base_mat, dtype=np.float32), mat_new], axis=0)
        save_matrix_cache(db_path, all_ids, all_mat)
    else:
        all_ids, all_mat = base_ids, base_mat

    if len(all_ids) == 0:
        raise EmbeddingError('no embeddings cached; run: local-memory build')
    if all_mat.shape[1] == 0:
        raise EmbeddingError('empty embedding matrix')

    q = _norm(np.asarray(api_embed([query])[0], dtype=np.float32))
    if q.shape[0] != all_mat.shape[1]:
        raise EmbeddingError(
            f'embedding dimension mismatch: query {q.shape[0]} vs cache {all_mat.shape[1]}')
    sims = np.asarray(all_mat, dtype=np.float32) @ q
    order = np.searchsorted(np.asarray(all_ids, dtype=np.int64), np.asarray(ids, dtype=np.int64))
    sims_ordered = sims[order]
    top = np.argsort(sims_ordered)[-limit:][::-1]

    results: list[dict] = []
    for idx in top:
        score = float(sims_ordered[idx])
        if score < 0.01:
            continue
        row = rows[idx]
        results.append({
            'session_id': row['session_id'],
            'position': row['position'],
            'content': (row['content'] or '')[:500],
            'score': round(score, 4),
        })
    if not results:
        raise EmbeddingError('no semantic matches above the score threshold')
    return results


def build(db_path, limit: int | None = None) -> dict:
    """Embed new/changed chunks and persist the matrix cache. Returns counts."""
    if is_off():
        raise EmbeddingError('EMBED_OFF is set')
    if not api_url():
        raise EmbeddingError('EMBED_API_URL is not set')
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _select_rows(conn, limit)
    finally:
        conn.close()
    if not rows:
        return {'embedded': 0, 'total': 0}

    cached = load_matrix_cache(db_path)
    known = {int(i) for i in cached[0]} if cached is not None else set()
    todo = [r['id'] for r in rows if r['id'] not in known]
    if todo:
        content_by_id = {r['id']: r['content'] for r in rows}
        vecs = api_embed([content_by_id[cid] for cid in todo])
        new_ids = np.asarray(todo, dtype=np.int64)
        new_mat = np.stack([_norm(v) for v in vecs], dtype=np.float32)
        if cached is not None:
            all_ids = np.concatenate([cached[0], new_ids])
            all_mat = np.concatenate([np.asarray(cached[1], dtype=np.float32), new_mat], axis=0)
        else:
            all_ids, all_mat = new_ids, new_mat
        order = np.argsort(all_ids)
        save_matrix_cache(db_path, all_ids[order], all_mat[order])
    return {'embedded': len(todo), 'total': len(rows)}
