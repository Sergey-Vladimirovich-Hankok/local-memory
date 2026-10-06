"""Keyword search: LIKE, FTS5, TF-IDF (sklearn), exact match, and context window.

Pure SQL plus optional scikit-learn; no network access. The result row format
is a public contract and must not change:

    {"session_id": str, "position": int, "content": str, "score": float,
     "context": [{"position": int, "content": str}, ...]}

Engines (cheapest first):
    exact   - literal LIKE, then FTS5; used when the query looks like code
    fts     - FTS5 MATCH with prefix expansion
    like    - plain SQL LIKE fallback (no word-segmentation assumptions)
    tfidf   - character n-gram TF-IDF cosine similarity (needs sklearn)
"""
from __future__ import annotations

import os
import re
import sqlite3

# Unicode class covering Latin + Cyrillic word characters, written as escapes
# so the source file itself stays ASCII-only.
_WORD_RE = re.compile(r'[^A-Za-z0-9\u0410-\u042f\u0430-\u044f\u0451\u0401]+')
_CODE_CHARS = set('=()[]{}_.:;<>!#|&*+\\/')


def ctx_window() -> int:
    """Context width (positions on each side of a hit), from MEMORY_CTX_WINDOW."""
    try:
        return max(0, int(os.environ.get('MEMORY_CTX_WINDOW', '3')))
    except (TypeError, ValueError):
        return 3


def looks_like_code(query: str) -> bool:
    """Heuristic: queries with code punctuation are searched exactly first."""
    return any(ch in query for ch in _CODE_CHARS)


def fetch_contexts(conn: sqlite3.Connection, items: list[tuple]) -> list[list[dict]]:
    """Fetch context windows for many (session_id, position) pairs in ONE query."""
    if not items:
        return []
    window = ctx_window()
    clauses: list[str] = []
    params: list = []
    for sid, pos in items:
        clauses.append('(session_id = ? AND position BETWEEN ? AND ?)')
        params.extend([sid, pos - window, pos + window])
    where = ' OR '.join(clauses)
    rows = conn.execute(
        f'SELECT session_id, position, content FROM chunks WHERE {where} '
        'GROUP BY session_id, position ORDER BY session_id, position',
        params,
    ).fetchall()
    by_session: dict = {}
    for sid, pos, content in rows:
        by_session.setdefault(sid, []).append(
            {'position': pos, 'content': (content or '')[:500]})
    return [by_session.get(sid, []) for sid, _ in items]


def add_context(results: list[dict], conn: sqlite3.Connection) -> list[dict]:
    """Attach a 'context' window (neighbors, content[:500]) to each result row."""
    items = [(r['session_id'], r['position']) for r in results]
    contexts = fetch_contexts(conn, items)
    for row, ctx in zip(results, contexts):
        row['context'] = ctx
    return results


def like_search(conn: sqlite3.Connection, query: str, limit: int) -> list:
    """Raw rows: literal substring match, newest first."""
    esc = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return conn.execute(
        "SELECT id, session_id, position, content FROM chunks "
        "WHERE content LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
        (f'%{esc}%', limit),
    ).fetchall()


def fts_search(conn: sqlite3.Connection, query: str, limit: int) -> list:
    """Raw rows: FTS5 MATCH with prefix expansion; [] on no usable terms or errors."""
    words = [w for w in _WORD_RE.split(query) if len(w) >= 2]
    if not words:
        return []
    terms = ' OR '.join(f'"{w}" OR {w}*' for w in words)
    try:
        return conn.execute(
            'SELECT c.id, c.session_id, c.position, c.content '
            'FROM memories_fts f JOIN chunks c ON c.id = f.rowid '
            'WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?',
            (terms, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def rows_to_results(rows: list) -> list[dict]:
    """Convert raw (id, session_id, position, content) rows to the public format."""
    return [
        {
            'session_id': r[1],
            'position': r[2],
            'content': (r[3] or '')[:500],
            'score': round(1.0 - i * 0.001, 4),
        }
        for i, r in enumerate(rows)
    ]


def exact_search(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """Literal search (LIKE first, FTS5 fallback). No context attached."""
    rows = like_search(conn, query, limit) or fts_search(conn, query, limit)
    return rows_to_results(rows)


def tfidf_search(rows: list, query: str, limit: int) -> list[dict]:
    """Character n-gram TF-IDF cosine similarity over deduplicated chunk rows.

    Requires scikit-learn; raises ImportError if it is not installed.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [(r['content'] or '') for r in rows]
    vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(2, 4), max_features=50000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts + [query])
    sims = cosine_similarity(matrix[-1:], matrix[:-1])[0]
    order = np.argsort(sims)[-limit:][::-1]
    results: list[dict] = []
    for idx in order:
        score = float(sims[idx])
        if score < 0.01:
            continue
        row = rows[idx]
        results.append({
            'session_id': row['session_id'],
            'position': row['position'],
            'content': (row['content'] or '')[:500],
            'score': round(score, 4),
        })
    return results


def keyword_search(conn: sqlite3.Connection, query: str, limit: int = 5):
    """Dispatch a keyword query: code-like -> exact, else FTS5, else LIKE.

    Returns (results_with_context, engine) where engine is one of
    'exact' | 'fts' | 'like' | 'empty'.
    """
    if not query or not query.strip():
        return [], 'empty'
    if looks_like_code(query):
        results = exact_search(conn, query, limit)
        if results:
            return add_context(results, conn), 'exact'
    results = fts_search(conn, query, limit)
    if results:
        return add_context(rows_to_results(results), conn), 'fts'
    results = like_search(conn, query, limit)
    if results:
        return add_context(rows_to_results(results), conn), 'like'
    return [], 'empty'
