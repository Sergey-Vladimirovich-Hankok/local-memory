# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Semantic search tests: TF-IDF path, fallbacks, and mocked-embeddings path.

No network is ever touched: the real API is replaced with a deterministic
in-process fake, or EMBED_OFF / missing API URL forces the offline engines.
"""
from __future__ import annotations

import numpy as np
import pytest

from local_memory import core, embeddings


def test_tfidf_when_embed_off(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '1')
    out = core.semantic_search('postgres timeout under load')
    assert out['engine'] == 'tfidf'
    assert out['results']
    assert out['results'][0]['session_id'] == 'alpha'


def test_tfidf_when_no_api_url(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.delenv('EMBED_API_URL', raising=False)
    out = core.semantic_search('postgres timeout under load')
    assert out['engine'] == 'tfidf'
    assert out['results']


def test_keyword_fallback_without_sklearn(db, monkeypatch):
    import sys
    monkeypatch.setenv('EMBED_OFF', '1')
    monkeypatch.setitem(sys.modules, 'sklearn.feature_extraction.text', None)
    out = core.semantic_search('postgres timeout')
    assert out['engine'] == 'keyword-fallback'
    assert out['results']


def _fake_api_embed(texts):
    """Deterministic 4-dim vectors: keyword-presence one-hot style."""
    out = []
    for t in texts:
        v = np.zeros(4, dtype=np.float32)
        if 'postgres' in t:
            v[0] = 1.0
        if 'react' in t or 'render' in t:
            v[1] = 1.0
        if 'docker' in t or 'image' in t:
            v[2] = 1.0
        if 'timeout' in t or 'reconnect' in t:
            v[3] = 1.0
        n = float(np.linalg.norm(v))
        out.append(v / n if n > 0 else np.zeros(4, dtype=np.float32))
    return out


def test_embedded_search_with_mocked_api(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.setenv('EMBED_API_URL', 'http://mock.invalid/v1/embeddings')
    monkeypatch.setattr(embeddings, 'api_embed', _fake_api_embed)
    out = core.semantic_search('postgres timeout')
    assert out['engine'] == 'embeddings'
    assert out['results']
    top = out['results'][0]
    assert 'postgres' in top['content']
    assert top['score'] >= 0.7
    # matrix cache must have been materialized next to the db, keyed by
    # the database FILE name (memory.db -> 'memory')
    assert (db.parent / 'embed_cache_memory_local-embedder.ids.npy').exists()


def test_embedded_search_second_call_uses_cache(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.setenv('EMBED_API_URL', 'http://mock.invalid/v1/embeddings')
    calls = {'n': 0}

    def counting_embed(texts):
        calls['n'] += len(texts)
        return _fake_api_embed(texts)

    monkeypatch.setattr(embeddings, 'api_embed', counting_embed)
    first = core.semantic_search('postgres')
    calls_first = calls['n']
    second = core.semantic_search('postgres')
    assert first['engine'] == second['engine'] == 'embeddings'
    # second call embeds only the query, not the whole corpus again
    assert calls['n'] - calls_first <= 1


def test_build_with_mocked_api(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.setenv('EMBED_API_URL', 'http://mock.invalid/v1/embeddings')
    monkeypatch.setattr(embeddings, 'api_embed', _fake_api_embed)
    out = core.build(limit=None)
    assert out['embedded'] >= 31  # all unique chunks
    assert out['total'] >= 31
    # corpus is fully cached: a semantic search now embeds only the query
    out2 = core.semantic_search('postgres timeout')
    assert out2['engine'] == 'embeddings'
    assert 'postgres' in out2['results'][0]['content']


def test_two_dbs_same_dir_have_independent_caches(tmp_path, monkeypatch):
    # regression: the cache key used to be (dir, model) only, so a second
    # database in the same directory silently inherited the first one's
    # vectors and reported 'embedded: 0'
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.setenv('EMBED_API_URL', 'http://mock.invalid/v1/embeddings')
    monkeypatch.setattr(embeddings, 'api_embed', _fake_api_embed)
    db1, db2 = tmp_path / 'first.db', tmp_path / 'second.db'
    core.ingest('s1', 'postgres timeout in first db', db=db1)
    core.ingest('s2', 'react render loop in second db', db=db2)
    out1 = core.build(db=db1)
    out2 = core.build(db=db2)
    assert out1['embedded'] == 1 and out1['total'] == 1
    assert out2['embedded'] == 1, 'second db must embed its own chunks'
    assert out2['total'] == 1
    assert (tmp_path / 'embed_cache_first_local-embedder.ids.npy').exists()
    assert (tmp_path / 'embed_cache_second_local-embedder.ids.npy').exists()


def test_build_without_api_fails_cleanly(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.delenv('EMBED_API_URL', raising=False)
    with pytest.raises(embeddings.EmbeddingError):
        core.build()


def test_embed_search_raises_when_off(db, monkeypatch):
    monkeypatch.setenv('EMBED_OFF', '1')
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_search(
            [{'id': 1, 'session_id': 'a', 'position': 0, 'content': 'x'}],
            'q', db_path=db)
