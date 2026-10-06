# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Core engine tests: first-run behavior without an explicit `init`."""
from __future__ import annotations

from local_memory import core


def test_fresh_db_works_without_init(tmp_path, monkeypatch):
    # regression: connect() never created the schema, so a fresh database
    # returned 500 'no such table: chunks' on the very first search
    path = tmp_path / 'fresh.db'
    monkeypatch.setenv('MEMORY_DB_PATH', str(path))
    monkeypatch.setenv('EMBED_OFF', '1')
    assert not path.exists()
    out = core.keyword_search('anything at all')
    assert out['results'] == []
    assert core.overview(days=7) == []
    assert core.fetch('no-such-session')['chunks'] == []
    # the write channel works right away, and the chunk is searchable
    out = core.ingest('first-run', 'zebra unicorn quantum')
    assert out['ok'] is True and out['position'] == 0
    found = core.keyword_search('zebra unicorn')
    assert found['results']
    assert found['results'][0]['session_id'] == 'first-run'


def test_connect_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv('MEMORY_DB_PATH', str(tmp_path / 'idem.db'))
    core.ingest('s', 'line one')
    core.ingest('s', 'line two')  # second connect must not clobber
    out = core.fetch('s')
    assert [c['position'] for c in out['chunks']] == [0, 1]
