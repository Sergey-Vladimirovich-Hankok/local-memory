"""Keyword search tests: dispatch, result format, context window, fallbacks."""
from __future__ import annotations

import sqlite3

from local_memory import core, search


def test_keyword_finds_postgres(db):
    out = core.keyword_search('postgres connection pool')
    assert out['engine'] in ('fts', 'like', 'exact')
    assert out['results'], 'expected results for a seeded keyword'
    sids = {r['session_id'] for r in out['results']}
    assert 'alpha' in sids


def test_result_row_format(db):
    out = core.keyword_search('redis cache eviction')
    assert out['results']
    row = out['results'][0]
    for key in ('session_id', 'position', 'content', 'score', 'context'):
        assert key in row, f'missing key {key}'
    assert isinstance(row['position'], int)
    assert isinstance(row['score'], float)
    assert isinstance(row['context'], list)


def test_context_window_contains_neighbors(db):
    out = core.keyword_search('deploy pipeline migrations')
    hit = out['results'][0]
    positions = {c['position'] for c in hit['context']}
    assert hit['position'] in positions
    assert len(hit['context']) >= 3, 'window should include neighbors'
    for c in hit['context']:
        assert len(c['content']) <= 500


def test_code_like_query_goes_exact(db):
    out = core.keyword_search('attempts=3')
    assert out['engine'] == 'exact'
    assert any('retry' in r['content'] for r in out['results'])


def test_empty_query(db):
    out = core.keyword_search('')
    assert out == {'results': [], 'engine': 'empty'}


def test_unknown_query_is_empty(db):
    out = core.keyword_search('zzzqqqxxxyyy')
    assert out['engine'] == 'empty'
    assert out['results'] == []


def test_like_search_returns_raw_rows(db):
    conn = sqlite3.connect(str(db))
    try:
        rows = search.like_search(conn, 'vacuum job', 5)
        assert rows and rows[0][3].startswith('the nightly vacuum job')
    finally:
        conn.close()


def test_dedup_keeps_newest_occurrence(db):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = core.load_chunks(conn)
        shared = [r for r in rows if 'shared ops runbook' in r['content']]
        assert len(shared) == 1, 'duplicate content must collapse to one row'
        assert shared[0]['session_id'] == 'gamma'  # newer chunk wins
    finally:
        conn.close()
