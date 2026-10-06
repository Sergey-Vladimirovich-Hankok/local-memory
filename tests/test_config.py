# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankoc
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""configgen tests: per-client MCP snippets must match examples/*.json.

The examples are the copy-paste source of truth; these tests guard against
the generated snippets drifting from them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_memory import configgen

ROOT = Path(__file__).resolve().parent.parent


def example(name: str) -> dict:
    return json.loads((ROOT / 'examples' / name).read_text(encoding='utf-8'))


def test_opencode_snippet_matches_example():
    assert configgen.print_config('opencode') == example('opencode.json')


def test_cursor_snippet_matches_example():
    assert configgen.print_config('cursor') == example('cursor_mcp.json')


def test_generic_snippet_matches_example():
    assert configgen.print_config('generic') == example('generic_mcp.json')


def test_claude_snippet_is_a_registration_command():
    cfg = configgen.print_config('claude')
    assert cfg['command'] == 'claude'
    assert cfg['args'][:3] == ['mcp', 'add', 'local-memory']
    assert cfg['args'][-1] == 'serve'


def test_setup_command_claude():
    cmd = configgen.setup_command('claude')
    cfg = configgen.print_config('claude')
    assert cmd == [cfg['command'], *cfg['args']]


def test_setup_command_rejects_clients_without_cli_registration():
    with pytest.raises(ValueError):
        configgen.setup_command('opencode')
    with pytest.raises(ValueError):
        configgen.setup_command('cursor')


def test_unknown_client_rejected():
    with pytest.raises(ValueError):
        configgen.print_config('bogus')


def test_cli_config_print_opencode():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    proc = subprocess.run(
        [sys.executable, '-m', 'local_memory', 'config', '--client', 'opencode'],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == example('opencode.json')


def test_cli_config_unknown_client_fails():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    proc = subprocess.run(
        [sys.executable, '-m', 'local_memory', 'config', '--client', 'bogus'],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert 'unknown client' in (proc.stderr + proc.stdout)
