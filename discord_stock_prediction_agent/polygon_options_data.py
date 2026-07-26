"""Polygon exact option-contract validation.

This module is intentionally small and dependency-light. It validates strike
signals by building the exact Polygon option ticker from root/expiry/side/strike
and reading historical aggregate bars for that exact contract.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from .config import config
from .options_parser import ParsedOptionSignal
from .options_symbol import build_occ_symbol


def build_polygon_option_ticker(option: ParsedOptionSignal) -> str:
    expiry = date.fromisoformat(str(option.expiry_date))
    occ_symbol = build_occ_symbol(option.root, expiry, option.side, float(option.strike))
    return f"O:{occ_symbol}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decision_from_return(return_pct: float) -> str:
    if return_pct > 0:
        return "BUY"
    if return_pct < 0:
        return "SELL"
    return "HOLD"


def validate_exact_strike_with_polygon(
    option: ParsedOptionSignal,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    if not config.polygon_api_key:
        return {
            "status": "SKIPPED",
            "decision": "REVIEW",
            "message": "POLYGON_API_KEY is not configured.",
        }
    if option.strike is None or not option.expiry_date or not option.side:
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "message": "Exact strike validation requires strike, expiry, and CALL/PUT side.",
        }

    try:
        polygon_ticker = build_polygon_option_ticker(option)
    except Exception as exc:
        return {
            "status": "FAILED",
            "decision": "REVIEW",
            "message": f"Could not build Polygon option ticker: {type(exc).__name__}: {exc}",
        }

    encoded_ticker = quote(polygon_ticker, safe="")
    url = (
        f"{config.polygon_base_url}/v2/aggs/ticker/{encoded_ticker}/range/1/day/"
        f"{start_date}/{end_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": config.polygon_api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=config.polygon_timeout_seconds)
    except Exception as exc:
        return {
            "status": "ERROR",
            "decision": "REVIEW",
            "polygon_ticker": polygon_ticker,
            "message": f"Polygon request error: {type(exc).__name__}: {exc}",
        }
    if not (200 <= resp.status_code < 300):
        return {
            "status": "ERROR",
            "decision": "REVIEW",
            "polygon_ticker": polygon_ticker,
            "http_status": resp.status_code,
            "message": f"Polygon HTTP {resp.status_code}: {resp.text[:240]}",
        }

    data = resp.json() if resp.content else {}
    bars = list(data.get("results") or [])
    if not bars:
        return {
            "status": "NO_DATA",
            "decision": "REVIEW",
            "polygon_ticker": polygon_ticker,
            "message": (
                "Polygon returned no historical aggregate bars for the exact "
                "strike/expiry contract in the selected window."
            ),
        }

    first = bars[0]
    last = bars[-1]
    entry_price = _as_float(first.get("c", first.get("o")))
    exit_price = _as_float(last.get("c", last.get("o")))
    if entry_price <= 0 or exit_price <= 0:
        return {
            "status": "NO_DATA",
            "decision": "REVIEW",
            "polygon_ticker": polygon_ticker,
            "message": "Polygon bars were returned, but usable option prices were missing.",
        }

    qty = max(1, int(_as_float(option.quantity, 1)))
    multiplier = 100
    pnl = (exit_price - entry_price) * qty * multiplier
    return_pct = ((exit_price - entry_price) / entry_price) * 100.0
    decision = _decision_from_return(return_pct)
    return {
        "status": "SUCCESS",
        "decision": decision,
        "polygon_ticker": polygon_ticker,
        "bars": len(bars),
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "return_pct": round(return_pct, 4),
        "profit_loss": round(pnl, 2),
        "win_rate": 1.0 if pnl > 0 else 0.0,
        "message": "Polygon exact-strike historical aggregate validation succeeded.",
    }
