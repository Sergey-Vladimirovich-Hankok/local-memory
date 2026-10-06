# local-memory — local-first long-term memory for AI agents
# Copyright (c) 2026 Sergey Vladimirovich Hankok
# Repository: https://github.com/Sergey-Vladimirovich-Hankok/local-memory
# Author: kokgfnu@gmail.com
# If you use, copy, fork or build upon this code, please keep this
# attribution notice and a link to the repository above.

"""Command-line interface: local-memory {serve,init,ingest,search,fetch,build,status}.

All human-facing commands print JSON to stdout and return a process exit code.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, core


def emit(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='local-memory',
        description='Local-first long-term memory MCP server for AI agents.')
    parser.add_argument('--version', action='version',
                        version=f'local-memory {__version__}')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('serve', help='run the MCP stdio server (for MCP clients)')
    sub.add_parser('init', help='create the database and schema (idempotent)')
    sub.add_parser('status', help='print database statistics as JSON')

    p_search = sub.add_parser('search', help='keyword or semantic search')
    p_search.add_argument('query')
    p_search.add_argument('-k', '--limit', type=int, default=5)
    p_search.add_argument('--semantic', action='store_true',
                          help='try embeddings, then TF-IDF, then keyword')

    p_ingest = sub.add_parser('ingest', help='append one chunk to a session')
    p_ingest.add_argument('--session-id', required=True)
    p_ingest.add_argument('--text', default=None, help='chunk text')
    p_ingest.add_argument('--file', default=None, help='read chunk text from file')
    p_ingest.add_argument('--project', default='')
    p_ingest.add_argument('--metadata', default=None, help='JSON object as a string')

    p_fetch = sub.add_parser('fetch', help='read consecutive chunks of a session')
    p_fetch.add_argument('--session-id', required=True)
    p_fetch.add_argument('--position', type=int, default=0)
    p_fetch.add_argument('--limit', type=int, default=20)

    p_build = sub.add_parser('build',
                             help='prebuild the embedding cache (needs EMBED_API_URL)')
    p_build.add_argument('--limit', type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == 'serve':
            from .server import mcp
            mcp.run()
            return 0
        if args.cmd == 'init':
            emit(core.init())
            return 0
        if args.cmd == 'status':
            emit(core.status())
            return 0
        if args.cmd == 'search':
            fn = core.semantic_search if args.semantic else core.keyword_search
            emit(fn(args.query, limit=args.limit))
            return 0
        if args.cmd == 'ingest':
            if args.file:
                with open(args.file, encoding='utf-8') as fh:
                    content = fh.read()
            elif args.text is not None:
                content = args.text
            else:
                print('error: provide --text or --file', file=sys.stderr)
                return 2
            metadata = json.loads(args.metadata) if args.metadata else None
            emit(core.ingest(args.session_id, content,
                             project=args.project, metadata=metadata))
            return 0
        if args.cmd == 'fetch':
            emit(core.fetch(args.session_id,
                            position=args.position, limit=args.limit))
            return 0
        if args.cmd == 'build':
            emit(core.build(limit=args.limit))
            return 0
    except Exception as e:  # noqa: BLE001 - CLI boundary: report, never traceback
        print(json.dumps({'error': f'{type(e).__name__}: {e}'}), file=sys.stderr)
        return 1
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
