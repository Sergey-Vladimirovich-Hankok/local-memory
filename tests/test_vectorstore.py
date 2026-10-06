# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankoc
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""sqlite-vec backend tests (optional extra: [vector]).

Skipped entirely when sqlite_vec is not installed. The embeddings API is
replaced with a deterministic in-process fake — no network is ever touched.
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pytest

from local_memory import core, embeddings, vectorstore

pytest.importorskip('sqlite_vec')


def _fake_api_embed(texts):
    """Deterministic 4-dim keyword vectors (same as tests/test_semantic.py)."""
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


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EMBED_OFF', '')
    monkeypatch.setenv('EMBED_API_URL', 'http://mock.invalid/v1/embeddings')
    monkeypatch.setattr(embeddings, 'api_embed', _fake_api_embed)


def test_table_name_sanitizes_model_key():
    assert vectorstore.table_name('text-embedding-0.6b') == 'vec_text_embedding_0_6b'
    assert vectorstore.table_name('local-embedder') == 'vec_local_embedder'
    assert vectorstore.table_name('qwen3/8b') == 'vec_qwen3_8b'


def test_build_populates_vec_table(db, monkeypatch):
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'sqlitevec')
    _enable(monkeypatch)
    core.build()
    conn = sqlite3.connect(str(db))
    vectorstore.load_extension(conn)
    try:
        assert vectorstore.table_count(conn, embeddings.model_key()) >= 31
    finally:
        conn.close()


def test_sqlitevec_search_matches_contract(db, monkeypatch):
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'sqlitevec')
    _enable(monkeypatch)
    out = core.semantic_search('postgres timeout')
    assert out['engine'] == 'embeddings'
    assert out['results']
    assert 'postgres' in out['results'][0]['content']
    for key in ('session_id', 'position', 'content', 'score', 'context'):
        assert key in out['results'][0]


def test_migrate_vectors_copies_npy_cache(db, monkeypatch):
    # 1) build the .npy cache with the npy backend (vec table untouched)
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'npy')
    _enable(monkeypatch)
    core.build()

    # 2) migrate into the vec table
    report = core.migrate_vectors()
    assert report['migrated'] >= 31
    assert report['vector_backend'] == 'sqlitevec'
    assert report['npy_removed'] is False

    # 3) search now runs on sqlite-vec with the same top result
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'sqlitevec')
    assert core.status()['vector_backend'] == 'sqlitevec'
    out = core.semantic_search('postgres timeout')
    assert out['engine'] == 'embeddings'
    assert 'postgres' in out['results'][0]['content']


def test_auto_degrades_to_npy_when_module_missing(db, monkeypatch):
    monkeypatch.setitem(sys.modules, 'sqlite_vec', None)
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'auto')
    assert vectorstore.available() is False
    assert vectorstore.resolve_backend() == 'npy'
    _enable(monkeypatch)
    out = core.semantic_search('postgres timeout')
    assert out['engine'] == 'embeddings'
    assert 'postgres' in out['results'][0]['content']


def test_second_model_gets_own_table(db, monkeypatch):
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'sqlitevec')
    _enable(monkeypatch)
    model1 = embeddings.model_key()
    core.build()

    def fake8(texts):
        out = []
        for t in texts:
            v = np.zeros(8, dtype=np.float32)
            if 'postgres' in t:
                v[0] = 1.0
            if 'timeout' in t or 'reconnect' in t:
                v[1] = 1.0
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else np.zeros(8, dtype=np.float32))
        return out

    # a different model (different name + dimension) must not clash
    monkeypatch.setenv('EMBED_MODEL', 'fake-8d')
    monkeypatch.setattr(embeddings, 'api_embed', fake8)
    core.build()
    out = core.semantic_search('postgres timeout')
    assert out['engine'] == 'embeddings'

    conn = sqlite3.connect(str(db))
    vectorstore.load_extension(conn)
    try:
        assert vectorstore.table_count(conn, model1) >= 31
        assert vectorstore.table_count(conn, 'fake-8d') >= 31
    finally:
        conn.close()


def test_migrate_without_npy_cache_fails_cleanly(db, monkeypatch):
    monkeypatch.setenv('MEMORY_VECTOR_BACKEND', 'sqlitevec')
    with pytest.raises(ValueError, match='no .npy embedding cache'):
        core.migrate_vectors()
