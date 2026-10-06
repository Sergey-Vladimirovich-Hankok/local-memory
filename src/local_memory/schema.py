"""Idempotent DDL for the local-memory database schema.

Tables:
    sessions          - one row per conversation/session (catalog)
    chunks            - ordered memory chunks (messages, reasoning, tool calls)
    memories_fts      - FTS5 index over chunks.content (maintained by triggers)
    session_summaries - optional per-session titles/summaries
    chunk_embeddings  - persisted embedding vectors (model, chunk_id, hash, vec)
    chunks_unique     - materialized content-dedup view of chunks (hot path)
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY,
    project TEXT DEFAULT '',
    start_time TEXT NOT NULL,
    end_time TEXT,
    message_count INTEGER DEFAULT 0,
    total_chars INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    position INTEGER,
    content TEXT,
    reasoning TEXT,
    tool_calls TEXT,
    tool_call_results TEXT,
    prev_id INTEGER,
    next_id INTEGER,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_content ON chunks(content);
CREATE INDEX IF NOT EXISTS idx_chunks_session_pos ON chunks(session_id, position);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
END;

CREATE TABLE IF NOT EXISTS session_summaries(
    session_id TEXT PRIMARY KEY,
    title TEXT,
    message_count INTEGER,
    start_time TEXT,
    last_active TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS chunk_embeddings(
    model TEXT,
    chunk_id INTEGER,
    content_hash TEXT,
    vec BLOB,
    PRIMARY KEY(model, chunk_id)
);

CREATE TABLE IF NOT EXISTS chunks_unique(
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    position INTEGER,
    content TEXT,
    reasoning TEXT,
    tool_calls TEXT,
    tool_call_results TEXT,
    prev_id INTEGER,
    next_id INTEGER,
    metadata TEXT DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_unique_content ON chunks_unique(content);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indexes and triggers if missing. Idempotent."""
    conn.executescript(SCHEMA)
    conn.commit()
