"""Durable, concurrent incoming-signal queue backed by SQLite."""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import AGENT_DIR


QUEUE_PATH = AGENT_DIR / "signal_queue.sqlite3"
_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS: set[str] = set()


def _utcstamp(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else timestamp
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(QUEUE_PATH, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        yield connection
    finally:
        connection.close()


def initialize_signal_queue() -> None:
    path_key = str(QUEUE_PATH.resolve())
    if path_key in _INITIALIZED_PATHS:
        return
    with _INIT_LOCK:
        if path_key in _INITIALIZED_PATHS:
            return
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_queue (
                    id TEXT PRIMARY KEY,
                    raw_text TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    claimed_at REAL,
                    created_at REAL NOT NULL,
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(signal_queue)").fetchall()
            }
            migrations = {
                "transport": "TEXT NOT NULL DEFAULT 'discord'",
                "reply_target": "TEXT NOT NULL DEFAULT ''",
                "is_group": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE signal_queue ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_signal_queue_ready
                ON signal_queue(status, available_at, created_at)
                """
            )
        _INITIALIZED_PATHS.add(path_key)


def enqueue_signal(
    raw_text: str,
    user_id: str,
    channel_id: str,
    message_id: str,
    limit: int = 20_000,
    transport: str = "discord",
    reply_target: str = "",
    is_group: bool = False,
) -> Dict[str, Any]:
    initialize_signal_queue()
    now = time.time()
    transport_name = str(transport or "discord").strip().lower()
    if transport_name == "discord":
        # Preserve pre-transport IDs so a Discord redelivery cannot duplicate an
        # item that was queued before this migration.
        stable_source = (
            f"{channel_id}:{message_id}"
            if message_id
            else f"{channel_id}:{user_id}:{now}:{raw_text}"
        )
    else:
        stable_source = (
            f"{transport_name}:{channel_id}:{message_id}"
            if message_id
            else f"{transport_name}:{channel_id}:{user_id}:{now}:{raw_text}"
        )
    signal_id = hashlib.sha256(stable_source.encode("utf-8")).hexdigest()[:24]
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT status FROM signal_queue WHERE id = ?", (signal_id,)
        ).fetchone()
        if existing:
            connection.execute("COMMIT")
            return {
                "id": signal_id,
                "accepted": False,
                "duplicate_delivery": True,
                "status": str(existing["status"]),
            }
        active = connection.execute(
            "SELECT COUNT(*) AS total FROM signal_queue WHERE status IN ('queued', 'processing')"
        ).fetchone()
        if int(active["total"] or 0) >= max(1, int(limit)):
            connection.execute("COMMIT")
            return {
                "id": signal_id,
                "accepted": False,
                "queue_full": True,
                "status": "rejected",
            }
        connection.execute(
            """
            INSERT INTO signal_queue (
                id, raw_text, user_id, channel_id, message_id,
                status, attempts, available_at, created_at,
                transport, reply_target, is_group
            ) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                str(raw_text or ""),
                str(user_id or ""),
                str(channel_id or ""),
                str(message_id or ""),
                now,
                now,
                transport_name,
                str(reply_target or ""),
                1 if is_group else 0,
            ),
        )
        connection.execute("COMMIT")
    return {
        "id": signal_id,
        "raw_text": str(raw_text or ""),
        "user_id": str(user_id or ""),
        "channel_id": str(channel_id or ""),
        "message_id": str(message_id or ""),
        "transport": transport_name,
        "reply_target": str(reply_target or ""),
        "is_group": bool(is_group),
        "created_at": _utcstamp(now),
        "accepted": True,
        "status": "queued",
    }


def recover_inflight_signals() -> int:
    """Return interrupted work to the queue when this process starts."""
    initialize_signal_queue()
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE signal_queue
            SET status = 'queued', claimed_at = NULL, available_at = ?
            WHERE status = 'processing'
            """,
            (time.time(),),
        )
        return int(cursor.rowcount or 0)


def claim_next_signal(max_attempts: int = 3, stale_seconds: int = 600) -> Dict[str, Any]:
    initialize_signal_queue()
    now = time.time()
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        stale_before = now - max(30, int(stale_seconds))
        connection.execute(
            """
            UPDATE signal_queue
            SET status = 'queued', claimed_at = NULL, available_at = ?
            WHERE status = 'processing' AND claimed_at < ? AND attempts < ?
            """,
            (now, stale_before, max(1, int(max_attempts))),
        )
        connection.execute(
            """
            UPDATE signal_queue
            SET status = 'dead', last_error = 'claim timeout after maximum attempts'
            WHERE status = 'processing' AND claimed_at < ? AND attempts >= ?
            """,
            (stale_before, max(1, int(max_attempts))),
        )
        row = connection.execute(
            """
            SELECT * FROM signal_queue
            WHERE status = 'queued' AND available_at <= ? AND attempts < ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now, max(1, int(max_attempts))),
        ).fetchone()
        if not row:
            connection.execute("COMMIT")
            return {}
        connection.execute(
            """
            UPDATE signal_queue
            SET status = 'processing', attempts = attempts + 1, claimed_at = ?
            WHERE id = ? AND status = 'queued'
            """,
            (now, row["id"]),
        )
        claimed = connection.execute(
            "SELECT * FROM signal_queue WHERE id = ?", (row["id"],)
        ).fetchone()
        connection.execute("COMMIT")
    item = dict(claimed)
    item["created_at"] = _utcstamp(float(item["created_at"]))
    item["started_at"] = _utcstamp(float(item["claimed_at"]))
    return item


def complete_signal(signal_id: str) -> None:
    if not signal_id:
        return
    initialize_signal_queue()
    with _connection() as connection:
        connection.execute("DELETE FROM signal_queue WHERE id = ?", (str(signal_id),))


def fail_signal(
    signal_id: str,
    error: str,
    max_attempts: int = 3,
    retry_base_seconds: int = 5,
) -> Dict[str, Any]:
    initialize_signal_queue()
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT attempts FROM signal_queue WHERE id = ?", (str(signal_id),)
        ).fetchone()
        if not row:
            connection.execute("COMMIT")
            return {"status": "missing"}
        attempts = int(row["attempts"] or 0)
        if attempts >= max(1, int(max_attempts)):
            status = "dead"
            available_at = time.time()
        else:
            status = "queued"
            delay = max(1, int(retry_base_seconds)) * (2 ** max(0, attempts - 1))
            available_at = time.time() + min(delay, 300)
        connection.execute(
            """
            UPDATE signal_queue
            SET status = ?, available_at = ?, claimed_at = NULL, last_error = ?
            WHERE id = ?
            """,
            (status, available_at, str(error or "")[:1000], str(signal_id)),
        )
        connection.execute("COMMIT")
    return {"status": status, "attempts": attempts, "available_at": _utcstamp(available_at)}


def queue_stats() -> Dict[str, int]:
    initialize_signal_queue()
    with _connection() as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS total FROM signal_queue GROUP BY status"
        ).fetchall()
    counts = {str(row["status"]): int(row["total"] or 0) for row in rows}
    return {
        "queued": counts.get("queued", 0),
        "processing": counts.get("processing", 0),
        "dead": counts.get("dead", 0),
        "depth": counts.get("queued", 0) + counts.get("processing", 0),
    }


def queue_depth() -> int:
    return queue_stats()["depth"]


def dead_letter_count() -> int:
    return queue_stats()["dead"]


def list_dead_signals(limit: int = 25) -> list[Dict[str, Any]]:
    initialize_signal_queue()
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, raw_text, transport, reply_target, attempts, created_at, last_error
            FROM signal_queue
            WHERE status = 'dead'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(250, int(limit))),),
        ).fetchall()
    return [dict(row) for row in rows]


def retry_dead_signals(limit: int = 100) -> int:
    """Return a bounded number of dead letters to the queue for operator recovery."""
    initialize_signal_queue()
    now = time.time()
    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            """
            SELECT id FROM signal_queue
            WHERE status = 'dead'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (max(1, min(1_000, int(limit))),),
        ).fetchall()
        identifiers = [str(row["id"]) for row in rows]
        if identifiers:
            placeholders = ",".join("?" for _ in identifiers)
            connection.execute(
                f"""
                UPDATE signal_queue
                SET status = 'queued', attempts = 0, available_at = ?,
                    claimed_at = NULL, last_error = ''
                WHERE id IN ({placeholders})
                """,
                (now, *identifiers),
            )
        connection.execute("COMMIT")
    return len(identifiers)
