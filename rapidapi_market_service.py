"""
RapidAPI Market Service -- Live market health check via TradingView endpoint.

Security rules (DO NOT violate):
- Never hardcode API keys -- read from .env only.
- Never log x-rapidapi-key values.
- Never use Yahoo Finance / yfinance / yahooquery as fallback.
- RapidAPI host must remain trading-view.p.rapidapi.com.
- Historical predictions: live market snapshot is NOT passed to Gemini for past dates.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_MOVERS_PATH = "/market/get-movers"


def _key() -> str:
    return os.getenv("RAPIDAPI_KEY", "")


def _host() -> str:
    return os.getenv("RAPIDAPI_HOST", "trading-view.p.rapidapi.com")


def _base() -> str:
    return os.getenv("RAPIDAPI_BASE_URL", "https://trading-view.p.rapidapi.com")


def _headers() -> Dict[str, str]:
    return {
        "x-rapidapi-key":  _key(),
        "x-rapidapi-host": _host(),
        "Content-Type":    "application/json",
    }


def run_rapidapi_market_health_check(exchange: str = "US") -> Dict[str, Any]:
    """
    Call GET /market/get-movers to prove the RapidAPI key is live.

    Returns a dict with:
      called                    -- True if HTTP request was made
      endpoint                  -- Full URL called
      http_status               -- HTTP response code (0 on exception)
      total_count               -- totalCount from response (int | None)
      top_symbols               -- list of up to 5 symbol strings
      key_present               -- bool: was RAPIDAPI_KEY found in env
      role                      -- "market_intelligence_health_check"
      used_in_prediction_context -- False: live snapshot NOT sent to Gemini
                                    for historical prediction dates
      error                     -- None or error string
    """
    key_present = bool(_key())
    endpoint    = f"{_base()}{_MOVERS_PATH}"

    result: Dict[str, Any] = {
        "called":                     False,
        "endpoint":                   endpoint,
        "http_status":                0,
        "latency_ms":                 None,
        "total_count":                None,
        "top_symbols":                [],
        "key_present":                key_present,
        "role":                       "market_intelligence_health_check",
        "used_in_prediction_context": False,
        "error":                      None,
    }

    if not key_present:
        result["error"] = "RAPIDAPI_KEY not set in .env"
        return result

    params = {"exchange": exchange, "name": "volume_gainers", "locale": "en"}
    try:
        t0  = time.monotonic()
        r   = requests.get(endpoint, headers=_headers(), params=params, timeout=15)
        ms  = int((time.monotonic() - t0) * 1000)
        result["called"]      = True
        result["http_status"] = r.status_code
        result["latency_ms"]  = ms
        if r.status_code == 200:
            data              = r.json()
            result["total_count"]  = data.get("totalCount")
            syms                   = data.get("symbols", [])
            result["top_symbols"]  = [s.get("s", "") for s in syms[:5]]
        else:
            result["error"] = f"HTTP {r.status_code}"
    except Exception as exc:
        result["called"] = True
        result["error"]  = f"{type(exc).__name__}: {exc}"

    return result


def run_rapidapi_instrument_check(symbol: str, exchange: str = "US") -> Dict[str, Any]:
    """
    Run the market health check and tag which symbol was being checked.
    The movers endpoint is used since instrument-level endpoints require a higher plan.
    Data is NOT passed to Gemini for historical predictions.
    """
    hc = run_rapidapi_market_health_check(exchange=exchange)
    hc["symbol_checked"] = symbol
    return hc


def fetch_top_movers_for_gemini() -> Dict[str, Any]:
    """
    Fetch top market movers for inclusion in Gemini prompt.

    ONLY safe to call for LIVE predictions (origin_date within last 7 days).
    Returns symbol names and volume/direction — no prices, so no leakage risk.
    """
    key_present = bool(_key())
    if not key_present:
        return {"status": "FAILED", "error": "RAPIDAPI_KEY not set", "context_text": ""}

    mover_types = [
        ("percent_change_gainers", "TOP % GAINERS"),
        ("percent_change_losers",  "TOP % LOSERS"),
        ("volume_gainers",         "HIGHEST VOLUME"),
    ]

    results: Dict[str, List[str]] = {}
    any_success = False
    for name, label in mover_types:
        params = {"exchange": "US", "name": name, "locale": "en"}
        try:
            r = requests.get(
                f"{_base()}{_MOVERS_PATH}",
                headers=_headers(),
                params=params,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                syms = data.get("symbols", [])
                # Extract top 10 symbol tickers only (no prices — leakage-safe)
                top = [
                    s.get("s", "").split(":")[-1].upper()
                    for s in syms[:10]
                    if isinstance(s, dict) and s.get("s")
                ]
                results[label] = top
                any_success = True
            else:
                results[label] = []
        except Exception:
            results[label] = []

    if not any_success:
        return {"status": "FAILED", "error": "All mover requests failed", "context_text": ""}

    lines = [
        "LIVE MARKET CONTEXT (today's movers — symbol names + direction only, no prices, no leakage):",
    ]
    for label, syms in results.items():
        if syms:
            lines.append(f"  {label}: {', '.join(syms)}")
    lines.append(
        "Use this to assess if the symbol is in a high-momentum market regime. "
        "If the symbol appears in GAINERS or LOSERS, weight momentum signals accordingly."
    )

    return {
        "status":       "SUCCESS",
        "context_text": "\n".join(lines),
        "data":         results,
    }
