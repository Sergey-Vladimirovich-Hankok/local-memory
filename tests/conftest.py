# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Synthetic test fixtures: a throwaway memory.db with 3 sessions, ~30 chunks.

NO real data is ever read from the host. Each test gets a fresh database in
tmp_path; MEMORY_DB_PATH points at it and EMBED_OFF forces the offline path.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_memory import schema  # noqa: E402


def _ts(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)
            ).strftime('%Y-%m-%dT%H:%M:%S')


ALPHA = [
    'the postgres connection pool timed out under load and we raised the max size',
    'redis cache eviction policy was changed from lru to allkeys-lru in prod',
    'deploy pipeline now runs migrations before rolling out the new service',
    'def retry(fn, attempts=3): keep retrying with exponential backoff',
    'the database migration failed on a lock held by a long running report',
    'we moved the health check endpoint to a dedicated port for the lb',
    'connection strings are read from environment variables at startup',
    'the nightly vacuum job trims the audit table and frees disk space',
    'we added slow query logging to catch the worst offenders',
    'rollback plan: restore the last good release tag and restart the fleet',
]

BETA = [
    'react hooks re-render loop caused by a stale closure in the fetch effect',
    'memoization of the table rows cut the render time in half',
    'websocket disconnect handler now reconnects with jittered backoff',
    'the bundle size grew after inlining the icon font, reverted that change',
    'we switched the router to lazy loading to shrink the initial chunk',
    'state is lifted into a reducer to keep the form logic testable',
    'the dropdown flickered because two effects wrote to the same state',
    'virtualized the long list and scroll performance is fine now',
    'we added suspense boundaries around the dashboard widgets',
    'the dev server proxy strips the authorization header by default',
]

GAMMA = [
    'the docker image size dropped from two to one point one gigabytes',
    'terraform state lock expired mid-apply and left a partial resource set',
    'kubernetes ingress controller rate limited the mobile app traffic',
    'helm upgrade failed on a value collision between the two subcharts',
    'node pool autoscaler was disabled while we tuned the bin packing',
    'we pinned the base image digest to stop silent drift between builds',
    'the service mesh mTLS handshake latency spiked after the ca rotation',
    'spot instance interruption drained the queue faster than workers joined',
    'we moved secrets into the vault agent injector for the sidecars',
    'load balancer health checks now hit the ready probe instead of root',
]


def _seed(conn: sqlite3.Connection) -> None:
    data = [('alpha', 'backend', 1.0, ALPHA),
            ('beta', 'frontend', 2.0, BETA),
            ('gamma', 'infra', 3.0, GAMMA)]
    for sid, project, hours, lines in data:
        start = _ts(hours)
        end = _ts(hours - 0.5)
        conn.execute(
            'INSERT INTO sessions (id, project, start_time, end_time, '
            'message_count, total_chars) VALUES (?,?,?,?,?,?)',
            (sid, project, start, end, len(lines), sum(len(l) for l in lines)))
        for pos, line in enumerate(lines):
            conn.execute(
                'INSERT INTO chunks (session_id, position, content) VALUES (?,?,?)',
                (sid, pos, line))
        conn.execute(
            'INSERT INTO session_summaries (session_id, title, message_count, '
            'start_time, last_active) VALUES (?,?,?,?,?)',
            (sid, f'{project} session {sid}', len(lines), start, end))
    # one shared line in two sessions: exercises content dedup
    conn.execute(
        'INSERT INTO chunks (session_id, position, content) VALUES (?,?,?)',
        ('alpha', 10, 'shared ops runbook line used by two sessions'))
    conn.execute(
        'INSERT INTO chunks (session_id, position, content) VALUES (?,?,?)',
        ('gamma', 10, 'shared ops runbook line used by two sessions'))


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh synthetic memory.db; env points local-memory at it."""
    path = tmp_path / 'memory.db'
    monkeypatch.setenv('MEMORY_DB_PATH', str(path))
    monkeypatch.setenv('EMBED_OFF', '1')
    monkeypatch.delenv('EMBED_API_URL', raising=False)
    conn = sqlite3.connect(str(path))
    try:
        schema.ensure_schema(conn)
        _seed(conn)
        conn.commit()
    finally:
        conn.close()
    return path
