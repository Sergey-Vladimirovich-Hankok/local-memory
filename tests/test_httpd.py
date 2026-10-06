# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankoc
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""REST transport tests: plain stdlib HTTP server on an ephemeral port.

No network is ever touched: the server is started in-process on 127.0.0.1:0
and driven with urllib from the stdlib. The `db` fixture (conftest.py)
points the server at a throwaway database.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from local_memory import core, httpd


@pytest.fixture()
def base_url(db):
    srv = httpd.make_server('127.0.0.1', 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()
    srv.server_close()


def req(base_url: str, path: str, method: str = 'GET', body=None,
        headers=None) -> tuple[int, dict]:
    data = json.dumps(body).encode('utf-8') if body is not None else None
    r = urllib.request.Request(base_url + path, data=data, method=method)
    if data is not None:
        r.add_header('Content-Type', 'application/json')
    for key, value in (headers or {}).items():
        r.add_header(key, value)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))


def test_health(base_url):
    code, body = req(base_url, '/health')
    assert code == 200
    assert body['ok'] is True
    assert body['version']


def test_status_matches_core(base_url):
    code, body = req(base_url, '/status')
    assert code == 200
    expected = core.status()
    for key in ('exists', 'sessions', 'chunks', 'unique_chunks',
                'vector_backend'):
        assert body[key] == expected[key]
    assert body['exists'] is True


def test_overview_lists_fixture_sessions(base_url):
    code, body = req(base_url, '/overview?days=7')
    assert code == 200
    ids = {s['id'] for s in body['sessions']}
    assert {'alpha', 'beta', 'gamma'} <= ids


def test_search_keyword(base_url):
    code, body = req(base_url, '/search', 'POST',
                     {'query': 'postgres connection pool'})
    assert code == 200
    assert body['engine'] in ('exact', 'fts', 'like')
    assert body['results']
    assert 'alpha' in {r['session_id'] for r in body['results']}


def test_search_semantic_falls_back_offline(base_url):
    # EMBED_OFF=1 (set by the db fixture) forces the TF-IDF engine
    code, body = req(base_url, '/search', 'POST',
                     {'query': 'postgres timeout', 'semantic': True})
    assert code == 200
    assert body['engine'] == 'tfidf'
    assert body['results']


def test_ingest_then_fetch(base_url):
    code, body = req(base_url, '/ingest', 'POST',
                     {'session_id': 'rest-test',
                      'content': 'rest ingest works', 'project': 'rest'})
    assert code == 200
    assert body['ok'] is True
    assert body['position'] == 0
    code, body = req(base_url, '/fetch', 'POST', {'session_id': 'rest-test'})
    assert code == 200
    assert body['chunks']
    assert body['chunks'][0]['content'] == 'rest ingest works'


def test_error_codes_and_bodies(base_url):
    code, body = req(base_url, '/search', 'POST', {})
    assert code == 400 and 'error' in body
    code, body = req(base_url, '/nope')
    assert code == 404 and 'error' in body
    code, body = req(base_url, '/search')
    assert code == 405 and 'error' in body
    code, body = req(base_url, '/health', 'POST')
    assert code == 405 and 'error' in body


def test_invalid_json_body_is_400(base_url):
    r = urllib.request.Request(base_url + '/search', data=b'{not json',
                               method='POST')
    r.add_header('Content-Type', 'application/json')
    try:
        urllib.request.urlopen(r, timeout=10)
        raise AssertionError('expected HTTP 400')
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert 'error' in json.loads(e.read().decode('utf-8'))


def test_bind_policy_refuses_offloopback_without_token():
    with pytest.raises(RuntimeError):
        httpd.make_server('0.0.0.0', 0)


def test_bind_policy_allows_offloopback_with_token():
    srv = httpd.make_server('0.0.0.0', 0, token='sekrit')
    srv.server_close()


def test_token_required_when_set(db):
    srv = httpd.make_server('127.0.0.1', 0, token='sekrit')
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    try:
        code, body = req(base, '/health')
        assert code == 401 and 'error' in body
        code, body = req(base, '/health',
                         headers={'Authorization': 'Bearer sekrit'})
        assert code == 200 and body['ok'] is True
        code, body = req(base, '/health', headers={'X-Token': 'sekrit'})
        assert code == 200 and body['ok'] is True
        code, body = req(base, '/health',
                         headers={'Authorization': 'Bearer wrong'})
        assert code == 401
    finally:
        srv.shutdown()
        srv.server_close()
