# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Per-client configuration generator: `local-memory config --client ...`.

The snippets mirror examples/*.json — the test suite guards against drift.
Supported clients: claude, opencode, cursor, generic.
"""
from __future__ import annotations

REPO_GIT = 'git+https://github.com/Sergey-Vladimirovich-Hankok/local-memory'
CLIENTS = ('claude', 'opencode', 'cursor', 'generic')


def _normalize(client: str) -> str:
    c = (client or '').strip().lower()
    if c not in CLIENTS:
        raise ValueError(
            f'unknown client {client!r}; supported: {", ".join(CLIENTS)}')
    return c


def _uvx_spec() -> dict:
    return {
        'command': 'uvx',
        'args': ['--from', REPO_GIT, 'local-memory', 'serve'],
    }


def print_config(client: str) -> dict:
    """The MCP client snippet for `client` (JSON-serializable).

    Raises:
        ValueError: unknown client.
    """
    c = _normalize(client)
    if c == 'opencode':
        return {'mcp': {'local-memory': _uvx_spec()}}
    if c == 'cursor':
        return {'mcpServers': {'local-memory': _uvx_spec()}}
    if c == 'generic':
        return {'mcpServers': {'local-memory': {
            'command': 'local-memory',
            'args': ['serve'],
            'env': {'MEMORY_DB_PATH': '/absolute/path/to/memory.db'},
        }}}
    # claude: registration command (see examples/claude_code.md)
    return {
        'command': 'claude',
        'args': ['mcp', 'add', 'local-memory', '--',
                 'uvx', '--from', REPO_GIT, 'local-memory', 'serve'],
    }


def setup_command(client: str) -> list[str]:
    """The command `config --setup` would run for `client` (claude only).

    Raises:
        ValueError: client has no CLI registration step.
    """
    c = _normalize(client)
    if c != 'claude':
        raise ValueError(
            f'`config --setup` has no CLI registration step for {c!r}; '
            f'use `config --client {c}` and merge the JSON into your config')
    cfg = print_config('claude')
    return [cfg['command'], *cfg['args']]
