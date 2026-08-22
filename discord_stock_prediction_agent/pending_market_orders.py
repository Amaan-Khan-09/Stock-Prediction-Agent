"""Durable queue for Alpaca market orders waiting to be submitted.

Queued orders live separately from prediction/position state. SQLite provides
atomic writes across worker threads and preserves orders across agent restarts.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AGENT_DIR


QUEUE_PATH = AGENT_DIR / "pending_market_orders.sqlite3"
_DB_LOCK = threading.RLock()


def _utcstamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _client_order_id(queue_id: str, side: str) -> str:
    digest = hashlib.sha256(str(queue_id).encode("utf-8")).hexdigest()[:28]
    return f"dsa-queued{side.lower()}-{digest}"[:48]


def _connect() -> sqlite3.Connection:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(QUEUE_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_market_orders (
            queue_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            stop_loss_pct REAL NOT NULL DEFAULT 1.0,
            status TEXT NOT NULL DEFAULT 'queued',
            client_order_id TEXT NOT NULL UNIQUE,
            broker_order_id TEXT NOT NULL DEFAULT '',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    existing_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(pending_market_orders)").fetchall()
    }
    migrations = {
        "action": "TEXT NOT NULL DEFAULT ''",
        "order_type": "TEXT NOT NULL DEFAULT 'market'",
        "limit_price": "REAL",
        "stop_price": "REAL",
        "time_in_force": "TEXT NOT NULL DEFAULT 'DAY'",
    }
    for column, definition in migrations.items():
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE pending_market_orders ADD COLUMN {column} {definition}"
            )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_market_status_created "
        "ON pending_market_orders(status, created_at)"
    )
    connection.commit()
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        yield connection
    finally:
        connection.close()


def _row(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


def enqueue_market_order(
    symbol: str,
    side: str,
    qty: float,
    reason: str = "",
    stop_loss_pct: float = 1.0,
    queue_id: str = "",
    action: str = "",
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
    time_in_force: str = "DAY",
) -> Dict[str, Any]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_side = str(side or "").strip().lower()
    quantity = float(qty)
    if not normalized_symbol:
        raise ValueError("symbol is required")
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if quantity <= 0:
        raise ValueError("qty must be greater than zero")

    key = str(queue_id or f"queued-{normalized_side}:{uuid.uuid4().hex}")
    client_id = _client_order_id(key, normalized_side)
    now = _utcstamp()
    with _DB_LOCK, _database() as connection:
        cursor = connection.execute(
            """
            INSERT INTO pending_market_orders (
                queue_id, symbol, side, qty, reason, stop_loss_pct, status,
                client_order_id, created_at, updated_at, action, order_type,
                limit_price, stop_price, time_in_force
            ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(queue_id) DO NOTHING
            """,
            (
                key,
                normalized_symbol,
                normalized_side,
                quantity,
                str(reason or ""),
                max(0.0, float(stop_loss_pct)),
                client_id,
                now,
                now,
                str(action or "").upper(),
                str(order_type or "market").lower(),
                float(limit_price) if limit_price is not None else None,
                float(stop_price) if stop_price is not None else None,
                str(time_in_force or "DAY").upper(),
            ),
        )
        # rowcount is 0 when ON CONFLICT DO NOTHING suppressed the insert --
        # i.e. an order with this same deterministic queue_id was already
        # queued, so this call is a duplicate resubmission rather than a new
        # order. Callers use this to tell the user "already queued" instead
        # of silently no-op-ing.
        inserted = cursor.rowcount > 0
        row = connection.execute(
            "SELECT * FROM pending_market_orders WHERE queue_id = ?", (key,)
        ).fetchone()
        connection.commit()
    result = _row(row)
    result["_inserted"] = inserted
    return result


def list_queued_market_orders(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = (
        "SELECT * FROM pending_market_orders WHERE status = 'queued' "
        "ORDER BY created_at, queue_id"
    )
    parameters: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        parameters = (max(1, int(limit)),)
    with _DB_LOCK, _database() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_row(row) for row in rows]


def mark_attempt(queue_id: str, error: str = "") -> None:
    with _DB_LOCK, _database() as connection:
        connection.execute(
            """
            UPDATE pending_market_orders
            SET attempts = attempts + 1, last_error = ?, updated_at = ?
            WHERE queue_id = ?
            """,
            (str(error or ""), _utcstamp(), str(queue_id)),
        )
        connection.commit()


def mark_failed(queue_id: str, error: str) -> None:
    with _DB_LOCK, _database() as connection:
        connection.execute(
            """
            UPDATE pending_market_orders
            SET status = 'failed', attempts = attempts + 1,
                last_error = ?, updated_at = ?
            WHERE queue_id = ?
            """,
            (str(error or ""), _utcstamp(), str(queue_id)),
        )
        connection.commit()


def remove_queued_market_order(queue_id: str) -> None:
    with _DB_LOCK, _database() as connection:
        connection.execute(
            "DELETE FROM pending_market_orders WHERE queue_id = ?", (str(queue_id),)
        )
        connection.commit()


def queue_summary() -> Dict[str, int]:
    with _DB_LOCK, _database() as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS total FROM pending_market_orders GROUP BY status"
        ).fetchall()
    counts = {str(row["status"]): int(row["total"]) for row in rows}
    return {
        "queued": counts.get("queued", 0),
        "failed": counts.get("failed", 0),
        "total": sum(counts.values()),
    }
