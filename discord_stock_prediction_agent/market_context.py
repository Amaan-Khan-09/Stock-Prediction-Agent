"""Non-blocking market regime signal used by the Discord decision engine."""
from __future__ import annotations

import os
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

try:
    from tools.tradingview_calendar import get_economic_calendar
    from tools.tradingview_news import get_stock_market_news
except Exception:  # pragma: no cover - optional external evidence
    get_economic_calendar = None
    get_stock_market_news = None


_DEFAULT_CONTEXT: Dict[str, Any] = {
    "regime": "unknown",
    "benchmark": config.benchmark or "SPY",
    "five_day_return_pct": 0.0,
    "twenty_day_return_pct": 0.0,
    "vix_level": None,
    "volatility_regime": "unknown",
    "news_risk_score": 0.0,
    "macro_event_count": 0,
    "geopolitical_mentions": 0,
    "evidence": [],
    "evidence_availability": {
        "price": False,
        "volatility": False,
        "news": False,
        "macro_calendar": False,
        "earnings": "handled_per_symbol_by_prediction_agent",
    },
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


def _return_pct(closes: list[float], periods: int) -> float:
    if len(closes) <= periods or closes[-periods - 1] <= 0:
        return 0.0
    return (closes[-1] - closes[-periods - 1]) / closes[-periods - 1] * 100


def _flatten_records(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "events", "result", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []


def _headline_risk(articles: list[dict]) -> tuple[float, int]:
    risk_terms = (
        "war", "conflict", "sanction", "tariff", "missile", "attack",
        "invasion", "geopolitical", "recession", "default", "crisis",
    )
    relief_terms = ("ceasefire", "peace agreement", "de-escalation", "rate cut")
    risk_hits = 0
    relief_hits = 0
    for article in articles[:50]:
        text = " ".join(
            str(article.get(key) or "") for key in ("title", "headline", "description")
        ).lower()
        risk_hits += sum(1 for term in risk_terms if term in text)
        relief_hits += sum(1 for term in relief_terms if term in text)
    score = max(-3.0, min(3.0, float(risk_hits - relief_hits)))
    return score, risk_hits


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
    five_day = _return_pct(closes, 5)
    twenty_day = _return_pct(closes, 20)

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

    evidence = [
        f"{config.benchmark or 'SPY'} 5-day return {five_day:+.2f}%",
        f"{config.benchmark or 'SPY'} 20-day return {twenty_day:+.2f}%",
    ]
    availability: Dict[str, Any] = dict(_DEFAULT_CONTEXT["evidence_availability"])
    availability["price"] = True

    vix_level = None
    volatility_regime = "unknown"
    try:
        vix_hist, _ = fetch_price_history("VIX", min_days=40)
        vix_closes = [
            _as_float(bar.get("close")) for bar in (vix_hist or [])
            if _as_float(bar.get("close")) > 0
        ]
        if vix_closes:
            vix_level = vix_closes[-1]
            availability["volatility"] = True
            if vix_level >= 30:
                volatility_regime = "high"
                buy_adj -= 4.0
                sell_adj += 4.0
            elif vix_level >= 20:
                volatility_regime = "elevated"
                buy_adj -= 2.0
                sell_adj += 2.0
            else:
                volatility_regime = "normal"
            evidence.append(f"VIX {vix_level:.2f} ({volatility_regime})")
    except Exception:
        pass

    news_risk_score = 0.0
    geopolitical_mentions = 0
    macro_event_count = 0
    if os.getenv("RAPIDAPI_KEY"):
        try:
            news = get_stock_market_news() if get_stock_market_news else {}
            articles = _flatten_records(news.get("data")) if news.get("status") == "SUCCESS" else []
            if articles:
                news_risk_score, geopolitical_mentions = _headline_risk(articles)
                availability["news"] = True
                buy_adj -= max(0.0, news_risk_score)
                sell_adj += max(0.0, news_risk_score)
                evidence.append(
                    f"News risk {news_risk_score:+.0f}; {geopolitical_mentions} risk-keyword mention(s)"
                )
        except Exception:
            pass
        try:
            calendar = get_economic_calendar() if get_economic_calendar else {}
            events = _flatten_records(calendar.get("data")) if calendar.get("status") == "SUCCESS" else []
            if events:
                macro_event_count = len(events)
                availability["macro_calendar"] = True
                evidence.append(f"Economic calendar: {macro_event_count} event(s) available")
        except Exception:
            pass

    return {
        "regime": regime,
        "benchmark": config.benchmark or "SPY",
        "five_day_return_pct": round(five_day, 4),
        "twenty_day_return_pct": round(twenty_day, 4),
        "vix_level": round(vix_level, 4) if vix_level is not None else None,
        "volatility_regime": volatility_regime,
        "news_risk_score": news_risk_score,
        "macro_event_count": macro_event_count,
        "geopolitical_mentions": geopolitical_mentions,
        "evidence": evidence,
        "evidence_availability": availability,
        "score_adjustment_buy": round(max(-10.0, min(10.0, buy_adj)), 2),
        "score_adjustment_sell": round(max(-10.0, min(10.0, sell_adj)), 2),
        "reason": "Refreshed from available price, volatility, news, and calendar evidence.",
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
