# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankoc
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""MCP over HTTP (streamable-http transport) end-to-end test.

Brings the real FastMCP server up on a loopback port in a daemon thread and
talks to it with the official `mcp` SDK client. Skipped when the port is
busy or the server does not come up within the deadline.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip('mcp')

PORT = 8917


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


@pytest.mark.asyncio
async def test_mcp_http_end_to_end(db):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from local_memory import server as srv

    if not _port_free(PORT):
        pytest.skip(f'port {PORT} is busy')
    thread = threading.Thread(
        target=lambda: srv.run_http('127.0.0.1', PORT), daemon=True)
    thread.start()
    deadline = time.time() + 10
    up = False
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', PORT), timeout=0.25):
                up = True
                break
        except OSError:
            time.sleep(0.1)
    if not up:
        pytest.skip('streamable-http server did not come up')

    async with streamablehttp_client(f'http://127.0.0.1:{PORT}/mcp') as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {'memory_overview', 'memory_search', 'memory_search_semantic',
                    'memory_fetch', 'memory_status', 'memory_ingest'} <= names
            res = await session.call_tool('memory_status', {})
            item = res.content[0]
            text = getattr(item, 'text', None) or str(item)
            assert 'exists' in text and 'true' in text.lower()
