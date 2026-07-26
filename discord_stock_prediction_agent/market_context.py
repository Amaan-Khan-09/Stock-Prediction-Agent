"""Non-blocking market regime signal used by the Discord decision engine."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from .config import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(PROJECT_ROOT / "tools"))

try:
    from historical_price_service import fetch_price_history
except Exception:  # pragma: no cover - project import safety
    fetch_price_history = None


_DEFAULT_CONTEXT: Dict[str, Any] = {
    "regime": "unknown",
    "benchmark": config.benchmark or "SPY",
    "five_day_return_pct": 0.0,
    "twenty_day_return_pct": 0.0,
    "score_adjustment_buy": 0.0,
    "score_adjustment_sell": 0.0,
    "reason": "Market context is warming up.",
}

_CACHE_TTL_SECONDS = 30 * 60
_lock = threading.Lock()
_cached_context: Dict[str, Any] = dict(_DEFAULT_CONTEXT)
_last_refresh_started = 0.0
_refreshing = False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_market_context() -> Dict[str, Any]:
    if fetch_price_history is None:
        return {
            **_DEFAULT_CONTEXT,
            "reason": "Project price-history provider is unavailable.",
        }

    try:
        hist, err = fetch_price_history(config.benchmark or "SPY", min_days=90)
    except Exception as exc:
        return {
            **_DEFAULT_CONTEXT,
            "reason": f"Market context refresh failed: {type(exc).__name__}: {exc}",
        }

    if not hist or len(hist) < 25:
        return {
            **_DEFAULT_CONTEXT,
            "reason": err or "Not enough benchmark history.",
        }

    closes = [_as_float(bar.get("close")) for bar in hist if _as_float(bar.get("close")) > 0]
    if len(closes) < 25:
        return {
            **_DEFAULT_CONTEXT,
            "reason": "Benchmark history did not contain enough usable closes.",
        }

    last = closes[-1]
    five_day = (last - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0.0
    twenty_day = (last - closes[-21]) / closes[-21] * 100 if len(closes) >= 21 else 0.0

    if twenty_day >= 1.0 and five_day >= -0.5:
        regime = "risk_on"
        buy_adj = 4.0
        sell_adj = -3.0
    elif twenty_day <= -1.0 or five_day <= -1.5:
        regime = "risk_off"
        buy_adj = -5.0
        sell_adj = 5.0
    else:
        regime = "neutral"
        buy_adj = 0.0
        sell_adj = 0.0

    return {
        "regime": regime,
        "benchmark": config.benchmark or "SPY",
        "five_day_return_pct": round(five_day, 4),
        "twenty_day_return_pct": round(twenty_day, 4),
        "score_adjustment_buy": buy_adj,
        "score_adjustment_sell": sell_adj,
        "reason": "Refreshed in background.",
    }


def _refresh_worker() -> None:
    global _cached_context, _refreshing
    try:
        context = _build_market_context()
    except Exception as exc:  # pragma: no cover - defensive
        context = {
            **_DEFAULT_CONTEXT,
            "reason": f"Market context refresh failed: {type(exc).__name__}: {exc}",
        }
    with _lock:
        _cached_context = context
        _refreshing = False


def _start_refresh_if_needed(force: bool = False) -> None:
    global _last_refresh_started, _refreshing
    now = time.monotonic()
    with _lock:
        stale = now - _last_refresh_started >= _CACHE_TTL_SECONDS
        if _refreshing or (not force and not stale):
            return
        _refreshing = True
        _last_refresh_started = now
    thread = threading.Thread(target=_refresh_worker, name="market-context", daemon=True)
    thread.start()


def get_market_context() -> Dict[str, Any]:
    """Return cached context immediately and refresh it in the background.

    Discord gateway heartbeats must never wait on external market-data calls.
    This function therefore returns the last known context right away. If the
    cache is stale, a background refresh is started for future messages.
    """
    _start_refresh_if_needed()
    with _lock:
        return dict(_cached_context)


def refresh_market_context_async() -> None:
    """Warm or refresh the market context without blocking the caller."""
    _start_refresh_if_needed(force=True)
