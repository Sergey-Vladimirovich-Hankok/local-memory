"""CLI tests: run the real subprocess entry point against a temp database."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cli(args, env_extra=None, expect_ok=True):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, '-m', 'local_memory', *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)
    if expect_ok:
        assert proc.returncode == 0, f'stderr: {proc.stderr}'
    return proc


def test_version():
    proc = cli(['--version'], expect_ok=False)
    assert '0.1.0' in (proc.stdout + proc.stderr)


def test_init_is_idempotent(tmp_path):
    dbp = str(tmp_path / 'mem' / 'memory.db')
    proc1 = cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    assert json.loads(proc1.stdout)['ok'] is True
    proc2 = cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    assert json.loads(proc2.stdout)['ok'] is True
    assert Path(dbp).exists()


def test_status_json(tmp_path):
    dbp = str(tmp_path / 'memory.db')
    cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    proc = cli(['status'], env_extra={'MEMORY_DB_PATH': dbp})
    out = json.loads(proc.stdout)
    assert out['exists'] is True and out['sessions'] == 0


def test_ingest_then_search_roundtrip(tmp_path):
    dbp = str(tmp_path / 'memory.db')
    cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    proc = cli(['ingest', '--session-id', 'cli-sess',
                '--text', 'the quick brown fox jumps over the lazy dog',
                '--project', 'cli-demo'], env_extra={'MEMORY_DB_PATH': dbp})
    assert json.loads(proc.stdout)['position'] == 0
    found = cli(['search', 'brown fox', '-k', '3'],
                env_extra={'MEMORY_DB_PATH': dbp})
    results = json.loads(found.stdout)['results']
    assert results and results[0]['session_id'] == 'cli-sess'


def test_fetch_cli(tmp_path):
    dbp = str(tmp_path / 'memory.db')
    cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    cli(['ingest', '--session-id', 's1', '--text', 'first line'],
        env_extra={'MEMORY_DB_PATH': dbp})
    cli(['ingest', '--session-id', 's1', '--text', 'second line'],
        env_extra={'MEMORY_DB_PATH': dbp})
    proc = cli(['fetch', '--session-id', 's1', '--limit', '1'],
               env_extra={'MEMORY_DB_PATH': dbp})
    out = json.loads(proc.stdout)
    assert len(out['chunks']) == 1
    assert out['chunks'][0]['content'] == 'first line'


def test_ingest_requires_text_or_file(tmp_path):
    dbp = str(tmp_path / 'memory.db')
    cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    proc = cli(['ingest', '--session-id', 's1'],
               env_extra={'MEMORY_DB_PATH': dbp}, expect_ok=False)
    assert proc.returncode == 2
    assert 'provide --text or --file' in proc.stderr


def test_build_without_api_fails_cleanly(tmp_path):
    dbp = str(tmp_path / 'memory.db')
    cli(['init'], env_extra={'MEMORY_DB_PATH': dbp})
    env = {'MEMORY_DB_PATH': dbp, 'EMBED_OFF': '', 'EMBED_API_URL': ''}
    proc = cli(['build'], env_extra=env, expect_ok=False)
    assert proc.returncode == 1
    assert 'EMBED_API_URL' in proc.stderr
