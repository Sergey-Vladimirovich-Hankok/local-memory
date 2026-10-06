# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Plain REST transport: stdlib ThreadingHTTPServer, zero new dependencies.

Endpoints (JSON, same shapes as the CLI/MCP tools):

    GET  /health     -> {'ok': true, 'version': ..., 'engine_ready': true}
    GET  /status     -> core.status()
    GET  /overview   -> {'sessions': [...]}            (?days=7)
    POST /search     -> {'query', 'limit'?, 'semantic'?}
    POST /ingest     -> {'session_id', 'content', 'project'?, 'metadata'?}
    POST /fetch      -> {'session_id', 'position'?, 'limit'?}
    GET  /tools      -> MCP tools manifest

Security: loopback (127.0.0.1/::1/localhost) is the default and needs no
token. Binding any other address REQUIRES a bearer token, otherwise the
server refuses to start (fail fast). With a token, every request must send
`Authorization: Bearer <token>` or `X-Token: <token>` (constant-time
compare). No TLS: put a reverse proxy or SSH tunnel in front for networks.
"""
from __future__ import annotations

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__, core
from . import server as mcp_server

# NB: '0.0.0.0' and '' are NOT loopback (they bind every interface) and are
# deliberately absent, so they fall through to the token requirement.
_LOOPBACK_HOSTS = {'127.0.0.1', '::1', 'localhost'}


class HttpError(Exception):
    """Route-level error with an HTTP status code."""

    def __init__(self, code: int, message: str, allow: str | None = None):
        super().__init__(message)
        self.code = code
        self.allow = allow


def validate_bind_policy(host: str, token: str | None) -> None:
    """Refuse to listen off-loopback without a token (fail fast)."""
    if host in _LOOPBACK_HOSTS:
        return
    if not token:
        raise RuntimeError(
            f'refusing to bind {host!r} without a token: '
            'pass --token (or MEMORY_HTTP_TOKEN) or bind 127.0.0.1')


class _Handler(BaseHTTPRequestHandler):
    server_version = f'local-memory/{__version__}'
    auth_token = ''  # set per-server by make_server()

    def log_message(self, format, *args):  # noqa: D102 - keep stdout/stderr clean
        pass

    # -- plumbing ---------------------------------------------------------
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, err: HttpError) -> None:
        self._send(err.code, {'error': str(err)})

    def _authorized(self) -> None:
        if not self.auth_token:
            return
        candidate = self.headers.get('Authorization', '').strip()
        if candidate.startswith('Bearer '):
            candidate = candidate[len('Bearer '):].strip()
        else:
            candidate = self.headers.get('X-Token', '').strip()
        if not hmac.compare_digest(
                candidate.encode('utf-8'), self.auth_token.encode('utf-8')):
            raise HttpError(401, 'unauthorized: missing or invalid bearer token')

    def _body(self) -> dict:
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            raise HttpError(400, 'invalid Content-Length header')
        if length < 0:
            raise HttpError(400, 'invalid Content-Length header')
        raw = self.rfile.read(length) if length > 0 else b''
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as e:
            raise HttpError(400, f'invalid JSON body: {e}') from e
        if not isinstance(data, dict):
            raise HttpError(400, 'JSON body must be an object')
        return data

    @staticmethod
    def _int_field(data: dict, name: str, default: int,
                   minimum: int = 0, maximum: int | None = None) -> int:
        value = data.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise HttpError(400, f'field {name!r} must be an integer')
        if value < minimum:
            raise HttpError(400, f'field {name!r} must be >= {minimum}')
        if maximum is not None and value > maximum:
            raise HttpError(400, f'field {name!r} must be <= {maximum}')
        return value

    # -- routes -----------------------------------------------------------
    def _route_get(self) -> None:
        self._authorized()
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/') or '/'
        if path == '/health':
            self._send(200, {'ok': True, 'version': __version__,
                             'engine_ready': True})
        elif path == '/status':
            self._send(200, core.status())
        elif path == '/overview':
            days_raw = parse_qs(parsed.query).get('days', ['7'])[0]
            try:
                days = int(days_raw)
            except ValueError:
                raise HttpError(400, 'query param days must be an integer')
            self._send(200, {'sessions': core.overview(days=days)})
        elif path == '/tools':
            self._send(200, json.loads(mcp_server.tools_manifest()))
        elif path in ('/search', '/ingest', '/fetch'):
            raise HttpError(405, f'{path} requires POST', allow='POST')
        else:
            raise HttpError(404, f'unknown path: {path}')

    def _route_post(self) -> None:
        self._authorized()
        path = urlparse(self.path).path.rstrip('/') or '/'
        data = self._body()
        if path == '/search':
            query = data.get('query')
            if not isinstance(query, str) or not query.strip():
                raise HttpError(400, 'field "query" (non-empty string) is required')
            limit = self._int_field(data, 'limit', 5, minimum=1, maximum=1000)
            semantic = bool(data.get('semantic', False))
            fn = core.semantic_search if semantic else core.keyword_search
            self._send(200, fn(query, limit=limit))
        elif path == '/ingest':
            session_id = data.get('session_id')
            content = data.get('content')
            if not isinstance(session_id, str) or not session_id:
                raise HttpError(400, 'field "session_id" (non-empty string) is required')
            if not isinstance(content, str) or not content:
                raise HttpError(400, 'field "content" (non-empty string) is required')
            project = data.get('project', '')
            if not isinstance(project, str):
                raise HttpError(400, 'field "project" must be a string')
            self._send(200, core.ingest(
                session_id, content, project=project,
                metadata=data.get('metadata')))
        elif path == '/fetch':
            session_id = data.get('session_id')
            if not isinstance(session_id, str) or not session_id:
                raise HttpError(400, 'field "session_id" (non-empty string) is required')
            self._send(200, core.fetch(
                session_id,
                position=self._int_field(data, 'position', 0, minimum=0),
                limit=self._int_field(data, 'limit', 20, minimum=1,
                                     maximum=1000)))
        elif path in ('/health', '/status', '/overview', '/tools'):
            raise HttpError(405, f'{path} is GET-only', allow='GET')
        else:
            raise HttpError(404, f'unknown path: {path}')

    # -- http.server dispatch ---------------------------------------------
    def do_GET(self):  # noqa: N802 - stdlib naming
        try:
            self._route_get()
        except HttpError as e:
            self._fail(e)
        except Exception as e:  # noqa: BLE001 - HTTP boundary: report, never crash
            self._send(500, {'error': f'{type(e).__name__}: {e}'})

    def do_POST(self):  # noqa: N802 - stdlib naming
        try:
            self._route_post()
        except HttpError as e:
            self._fail(e)
        except Exception as e:  # noqa: BLE001 - HTTP boundary: report, never crash
            self._send(500, {'error': f'{type(e).__name__}: {e}'})


def make_server(host: str, port: int, token: str | None = None) -> ThreadingHTTPServer:
    """Bind (validating the security policy first) and return the server."""
    validate_bind_policy(host, token)
    handler = type('BoundHandler', (_Handler,), {'auth_token': token or ''})
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str, port: int, token: str | None = None) -> None:
    """Start the REST server and block until interrupted."""
    server = make_server(host, port, token)
    print(json.dumps({'ok': True, 'host': server.server_address[0],
                      'port': server.server_address[1],
                      'token_required': bool(token)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
