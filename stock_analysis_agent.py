"""
Stock Analysis Agent
Analyzes individual stocks by fetching historical data, news, and technical indicators
to generate probability-based investment recommendations
"""

import os
import sys
import time
import warnings
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Suppress FutureWarning for google generative AI libraries
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

_AGENT_ROOT = Path(__file__).resolve().parent
load_dotenv(_AGENT_ROOT / ".env", override=True)

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

# Import tools
from stock_historical_data import get_year_historical_data, analyze_historical_performance
from tradingview_news import get_stock_news, get_stock_market_news
from tradingview_technical_analysis import get_technical_analysis, get_technical_indicators
from tradingview_price import get_price, format_symbol
from tradingview_market_data import get_market_data_exact, get_market_data_multi_exchange, get_company_info, get_analyst_recommendations
from tradingview_calendar import get_economic_calendar, get_earnings_calendar, get_dividends_calendar
from tradingview_leaderboards import get_stock_gainers, get_stock_losers, get_most_active_stocks
from world_economy import get_g20_gdp_growth, get_world_cpi, get_interest_rates
from tradingview_community import get_community_data
from tradingview_search import find_ticker_by_company_name


def _clamp_score(value: float) -> int:
    try:
        value_f = float(value)
    except Exception:
        return 0
    if value_f < 0:
        return 0
    if value_f > 100:
        return 100
    return int(round(value_f))


def _payload(result: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Tool responses in this repo commonly look like:
    {"status":"SUCCESS", "data":{"success": true, "data": {...}}}
    """
    if not isinstance(result, dict):
        return None
    if str(result.get("status", "")).upper() != "SUCCESS":
        return None
    data = result.get("data")
    if isinstance(data, dict) and data.get("success") and isinstance(data.get("data"), dict):
        return data.get("data")
    return None


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _history_from_historical_result(historical_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = _payload(historical_result) or {}
    history = payload.get("history", [])
    return history if isinstance(history, list) else []


def _ts_from_date_str(date_str: str) -> int | None:
    """
    Accepts YYYY-MM-DD and returns unix timestamp (seconds) at 00:00 UTC.
    """
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _close_at_or_before(history: List[Dict[str, Any]], ts: int) -> float | None:
    best_t = None
    best_close = None
    for pt in history:
        if not isinstance(pt, dict):
            continue
        t = pt.get("time", pt.get("t"))
        c = pt.get("close", pt.get("c"))
        try:
            t_i = int(float(t))
        except Exception:
            continue
        if t_i <= ts:
            c_f = _safe_float(c)
            if c_f is None:
                continue
            if best_t is None or t_i > best_t:
                best_t = t_i
                best_close = c_f
    return best_close


def _compute_rsi(closes: List[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses += -delta
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _technical_score_from_history(history: List[Dict[str, Any]], cutoff_ts: int) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    pts = [pt for pt in history if isinstance(pt, dict)]
    pts_sorted = sorted(
        [pt for pt in pts if _safe_float(pt.get("close", pt.get("c"))) is not None and _safe_float(pt.get("time", pt.get("t"))) is not None],
        key=lambda x: int(float(x.get("time", x.get("t")))),
    )
    pts_sorted = [pt for pt in pts_sorted if int(float(pt.get("time", pt.get("t")))) <= cutoff_ts]
    closes = [float(pt.get("close", pt.get("c"))) for pt in pts_sorted]

    if len(closes) < 60:
        missing.append("history.closes_60")
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": missing}

    # Momentum via MA20 vs MA50
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    last = closes[-1]
    score = 50.0

    if ma20 > ma50:
        score += 10
        factors.append({"factor": "MA20 > MA50", "impact": +10, "value": {"ma20": round(ma20, 2), "ma50": round(ma50, 2)}})
    else:
        score -= 10
        factors.append({"factor": "MA20 <= MA50", "impact": -10, "value": {"ma20": round(ma20, 2), "ma50": round(ma50, 2)}})

    # Price vs MA50
    if last > ma50:
        score += 6
        factors.append({"factor": "Price > MA50", "impact": +6, "value": {"price": round(last, 2), "ma50": round(ma50, 2)}})
    else:
        score -= 6
        factors.append({"factor": "Price <= MA50", "impact": -6, "value": {"price": round(last, 2), "ma50": round(ma50, 2)}})

    rsi = _compute_rsi(closes, period=14)
    if rsi is None:
        missing.append("history.rsi_14")
    else:
        if rsi >= 70:
            score -= 8
            factors.append({"factor": "RSI overbought", "impact": -8, "value": round(rsi, 1)})
        elif rsi <= 30:
            score += 8
            factors.append({"factor": "RSI oversold", "impact": +8, "value": round(rsi, 1)})
        else:
            factors.append({"factor": "RSI neutral", "impact": 0, "value": round(rsi, 1)})

    s = _clamp_score(score)
    return {"score": s, "signal": _signal_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def _risk_score_from_history(history: List[Dict[str, Any]], cutoff_ts: int) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    pts = [pt for pt in history if isinstance(pt, dict)]
    pts_sorted = sorted(
        [pt for pt in pts if _safe_float(pt.get("close", pt.get("c"))) is not None and _safe_float(pt.get("time", pt.get("t"))) is not None],
        key=lambda x: int(float(x.get("time", x.get("t")))),
    )
    pts_sorted = [pt for pt in pts_sorted if int(float(pt.get("time", pt.get("t")))) <= cutoff_ts]
    closes = [float(pt.get("close", pt.get("c"))) for pt in pts_sorted]
    if len(closes) < 40:
        missing.append("history.closes_40")
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": missing}

    # Daily returns volatility (simple)
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            continue
        rets.append((closes[i] / closes[i - 1]) - 1.0)
    if len(rets) < 20:
        missing.append("history.returns_20")
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": missing}

    mean = sum(rets[-20:]) / 20
    var = sum((x - mean) ** 2 for x in rets[-20:]) / 19
    vol = (var ** 0.5) * (252 ** 0.5)  # annualized

    score = 50.0
    # vol thresholds (annualized)
    if vol >= 0.60:
        score += 30
        factors.append({"factor": "High volatility (ann.)", "impact": +30, "value": round(vol, 3)})
    elif vol >= 0.40:
        score += 18
        factors.append({"factor": "Elevated volatility (ann.)", "impact": +18, "value": round(vol, 3)})
    elif vol >= 0.25:
        score += 8
        factors.append({"factor": "Moderate volatility (ann.)", "impact": +8, "value": round(vol, 3)})
    else:
        score -= 6
        factors.append({"factor": "Low volatility (ann.)", "impact": -6, "value": round(vol, 3)})

    # Drawdown over last 60 days
    window = closes[-60:] if len(closes) >= 60 else closes
    peak = window[0]
    max_dd = 0.0
    for c in window:
        if c > peak:
            peak = c
        dd = (peak - c) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    if max_dd >= 0.30:
        score += 20
        factors.append({"factor": "Large drawdown (60d)", "impact": +20, "value": round(max_dd * 100, 1)})
    elif max_dd >= 0.15:
        score += 10
        factors.append({"factor": "Meaningful drawdown (60d)", "impact": +10, "value": round(max_dd * 100, 1)})
    else:
        factors.append({"factor": "Contained drawdown (60d)", "impact": 0, "value": round(max_dd * 100, 1)})

    s = _clamp_score(score)
    return {"score": s, "signal": _risk_label_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def _write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    from jsonl_store import append_jsonl

    append_jsonl(path, obj)


def _signal_from_score(score: int) -> str:
    if score <= 0:
        return "Unavailable"
    if score >= 67:
        return "Bullish"
    if score <= 33:
        return "Bearish"
    return "Neutral"


def _risk_label_from_score(score: int) -> str:
    if score <= 0:
        return "Unavailable"
    if score >= 67:
        return "High"
    if score <= 33:
        return "Low"
    return "Medium"


def fundamental_score(market_data_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fundamental score uses only validated market-data fields (TTM margins, EPS/revenue when available).
    """
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    payload = _payload(market_data_result) or {}
    ttm = payload.get("ttm") if isinstance(payload.get("ttm"), dict) else {}
    indicators = payload.get("indicators") if isinstance(payload.get("indicators"), dict) else {}

    eps = ttm.get("earnings_per_share_ttm")
    revenue = ttm.get("total_revenue_ttm")
    net_margin = ttm.get("net_margin_ttm")
    gross_margin = ttm.get("gross_margin_ttm")
    market_cap = indicators.get("market_cap_calc")

    score = 50.0
    used_any = False

    if eps is None:
        missing.append("ttm.earnings_per_share_ttm")
    else:
        used_any = True
        try:
            eps_f = float(eps)
            delta = 10 if eps_f > 0 else -10
            score += delta
            factors.append({"factor": "EPS (TTM)", "impact": delta, "value": eps_f})
        except Exception:
            missing.append("ttm.earnings_per_share_ttm")

    if revenue is None:
        missing.append("ttm.total_revenue_ttm")
    else:
        used_any = True
        try:
            rev_f = float(revenue)
            delta = 5 if rev_f > 0 else -5
            score += delta
            factors.append({"factor": "Revenue (TTM)", "impact": delta, "value": rev_f})
        except Exception:
            missing.append("ttm.total_revenue_ttm")

    if net_margin is None:
        missing.append("ttm.net_margin_ttm")
    else:
        used_any = True
        try:
            nm = float(net_margin)
            if nm >= 20:
                delta = 15
            elif nm >= 10:
                delta = 8
            elif nm >= 0:
                delta = 2
            else:
                delta = -15
            score += delta
            factors.append({"factor": "Net Margin (TTM)", "impact": delta, "value": nm})
        except Exception:
            missing.append("ttm.net_margin_ttm")

    if gross_margin is None:
        missing.append("ttm.gross_margin_ttm")
    else:
        used_any = True
        try:
            gm = float(gross_margin)
            if gm >= 50:
                delta = 10
            elif gm >= 35:
                delta = 5
            elif gm >= 20:
                delta = 1
            else:
                delta = -8
            score += delta
            factors.append({"factor": "Gross Margin (TTM)", "impact": delta, "value": gm})
        except Exception:
            missing.append("ttm.gross_margin_ttm")

    if market_cap is None:
        missing.append("indicators.market_cap_calc")
    else:
        used_any = True
        try:
            mc = float(market_cap)
            # Scale-neutral mild boost for large caps (stability proxy)
            delta = 5 if mc >= 50_000_000_000 else (2 if mc >= 10_000_000_000 else 0)
            score += delta
            factors.append({"factor": "Market Cap", "impact": delta, "value": mc})
        except Exception:
            missing.append("indicators.market_cap_calc")

    if not used_any:
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": sorted(set(missing))}

    s = _clamp_score(score)
    return {"score": s, "signal": _signal_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def valuation_score(market_data_result: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    payload = _payload(market_data_result) or {}
    indicators = payload.get("indicators") if isinstance(payload.get("indicators"), dict) else {}
    pe = indicators.get("price_earnings")
    if pe is None:
        missing.append("indicators.price_earnings")
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": missing}

    try:
        pe_f = float(pe)
    except Exception:
        missing.append("indicators.price_earnings")
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": missing}

    # Simple P/E heuristic: lower is generally “cheaper” but too low can be distressed.
    score = 50.0
    if pe_f <= 0:
        score = 0
        factors.append({"factor": "P/E", "impact": -50, "value": pe_f})
        return {"score": 0, "signal": "Unavailable", "factors": factors, "missing_fields": missing}
    if pe_f < 10:
        score = 70
        factors.append({"factor": "P/E (low)", "impact": +20, "value": pe_f})
    elif pe_f <= 18:
        score = 80
        factors.append({"factor": "P/E (reasonable)", "impact": +30, "value": pe_f})
    elif pe_f <= 28:
        score = 60
        factors.append({"factor": "P/E (elevated)", "impact": +10, "value": pe_f})
    elif pe_f <= 45:
        score = 35
        factors.append({"factor": "P/E (high)", "impact": -15, "value": pe_f})
    else:
        score = 20
        factors.append({"factor": "P/E (very high)", "impact": -30, "value": pe_f})

    s = _clamp_score(score)
    return {"score": s, "signal": _signal_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def risk_score(market_data_result: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    payload = _payload(market_data_result) or {}
    indicators = payload.get("indicators") if isinstance(payload.get("indicators"), dict) else {}

    beta = indicators.get("beta_1_year")
    week_high = indicators.get("price_52_week_high")
    week_low = indicators.get("price_52_week_low")

    used_any = False
    score = 50.0  # higher => higher risk

    if beta is None:
        missing.append("indicators.beta_1_year")
    else:
        used_any = True
        try:
            b = float(beta)
            if b >= 1.6:
                delta = +25
            elif b >= 1.2:
                delta = +12
            elif b >= 0.9:
                delta = 0
            else:
                delta = -8
            score += delta
            factors.append({"factor": "Beta (1Y)", "impact": delta, "value": b})
        except Exception:
            missing.append("indicators.beta_1_year")

    vol_range = None
    if week_high is None:
        missing.append("indicators.price_52_week_high")
    if week_low is None:
        missing.append("indicators.price_52_week_low")

    try:
        if week_high is not None and week_low is not None:
            wh = float(week_high)
            wl = float(week_low)
            if wl > 0:
                used_any = True
                vol_range = ((wh - wl) / wl) * 100.0
    except Exception:
        pass

    if vol_range is not None:
        if vol_range >= 80:
            delta = +25
        elif vol_range >= 50:
            delta = +15
        elif vol_range >= 30:
            delta = +5
        else:
            delta = -5
        score += delta
        factors.append({"factor": "52W Range Volatility %", "impact": delta, "value": round(vol_range, 1)})

    if not used_any:
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": sorted(set(missing))}

    s = _clamp_score(score)
    return {"score": s, "signal": _risk_label_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def technical_score(technical_result: Dict[str, Any], indicators_result: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    tech_payload = _payload(technical_result) or {}
    ind_payload = _payload(indicators_result) or {}

    recommendation = None
    if isinstance(tech_payload, dict):
        recommendation = tech_payload.get("recommendation") or tech_payload.get("summary", {}).get("RECOMMENDATION")
    if not recommendation:
        missing.append("technical.recommendation")

    rec_map = {
        "STRONG_BUY": 85,
        "BUY": 72,
        "NEUTRAL": 50,
        "SELL": 28,
        "STRONG_SELL": 15,
    }
    score = None
    if recommendation:
        rec_norm = str(recommendation).upper().replace(" ", "_")
        if rec_norm in rec_map:
            score = float(rec_map[rec_norm])
            factors.append({"factor": "TA Recommendation", "impact": 0, "value": rec_norm})
        else:
            # Unknown string from API; treat as missing rather than guessing.
            missing.append("technical.recommendation")

    # Optional RSI adjustment if available
    rsi = None
    if isinstance(ind_payload, dict):
        rsi = ind_payload.get("RSI") or ind_payload.get("rsi")
    if rsi is not None:
        try:
            r = float(rsi)
            if score is None:
                score = 50.0
            # Overbought/oversold heuristic
            if r >= 70:
                score -= 8
                factors.append({"factor": "RSI overbought", "impact": -8, "value": r})
            elif r <= 30:
                score += 8
                factors.append({"factor": "RSI oversold", "impact": +8, "value": r})
        except Exception:
            missing.append("indicators.rsi")
    else:
        missing.append("indicators.rsi")

    if score is None:
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": sorted(set(missing))}

    s = _clamp_score(score)
    return {"score": s, "signal": _signal_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def sentiment_score(news_result: Dict[str, Any], community_result: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    score = None

    # News: keyword sentiment on titles (no inference beyond explicit text)
    news_payload = _payload(news_result)  # may be None for list-style responses
    articles = None
    if isinstance(news_payload, list):
        articles = news_payload
    elif isinstance(news_payload, dict):
        articles = news_payload.get("data") if isinstance(news_payload.get("data"), list) else news_payload.get("articles")
    # Also handle new format: news_result["data"] is a direct list of articles
    if articles is None and isinstance(news_result, dict) and news_result.get("status") == "SUCCESS":
        raw_data = news_result.get("data")
        if isinstance(raw_data, list):
            articles = raw_data
    if not isinstance(articles, list):
        missing.append("news.articles")
    else:
        bulls = 0
        bears = 0
        _BULL_KW = [
            "beats", "beat", "upgrade", "upgraded", "raises guidance", "record",
            "surge", "rally", "gains", "gain", "rises", "rise", "jumped", "jumps",
            "soars", "soar", "buying opportunity", "outperform", "buy", "strong",
            "positive", "growth", "breakout", "bullish",
        ]
        _BEAR_KW = [
            "misses", "miss", "downgrade", "downgraded", "cuts guidance", "plunge",
            "lawsuit", "probe", "lower", "falls", "fall", "drops", "drop", "tumbles",
            "decline", "declines", "sell", "weak", "warning", "loss", "bearish",
            "concern", "cut", "layoffs",
        ]
        for a in articles[:50]:
            if not isinstance(a, dict):
                continue
            title = (a.get("title") or a.get("headline") or "").lower()
            if not title:
                continue
            if any(k in title for k in _BULL_KW):
                bulls += 1
            if any(k in title for k in _BEAR_KW):
                bears += 1

        if bulls or bears:
            score = 50.0 + (bulls - bears) * 6.0
            factors.append({"factor": "News title balance", "impact": _clamp_score(score) - 50, "value": {"bull": bulls, "bear": bears}})
        elif len(articles) > 0:
            # Articles present but no strong sentiment keywords — treat as neutral
            score = 50.0
            factors.append({"factor": "News neutral (no keywords matched)", "impact": 0, "value": {"articles": len(articles)}})

    # Community sentiment (if available) as mild modifier
    comm_payload = _payload(community_result) or {}
    if not comm_payload:
        missing.append("community.sentiment")
    else:
        # Unknown schema; only use if numeric signal exists.
        possible = None
        for key in ["sentiment", "score", "bullish", "bearish"]:
            if key in comm_payload:
                possible = comm_payload.get(key)
                break
        if possible is not None:
            try:
                val = float(possible)
                if score is None:
                    score = 50.0
                # Normalize [-1..1] or [0..100]
                if -1 <= val <= 1:
                    delta = val * 10
                else:
                    delta = (val - 50) / 10
                score += delta
                factors.append({"factor": "Community signal", "impact": int(round(delta)), "value": val})
            except Exception:
                missing.append("community.sentiment")

    if score is None:
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": sorted(set(missing))}

    s = _clamp_score(score)
    return {"score": s, "signal": _signal_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def macro_score(gdp_result: Dict[str, Any], interest_rates_result: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    factors: List[Dict[str, Any]] = []

    score = None

    def _first_numeric(obj) -> float | None:
        if obj is None:
            return None
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, str):
            try:
                return float(obj.replace("%", "").strip())
            except Exception:
                return None
        if isinstance(obj, dict):
            for k in ["value", "close", "c", "latest", "current"]:
                if k in obj:
                    v = _first_numeric(obj.get(k))
                    if v is not None:
                        return v
            for v in obj.values():
                v2 = _first_numeric(v)
                if v2 is not None:
                    return v2
        if isinstance(obj, list):
            for it in obj[:10]:
                v = _first_numeric(it)
                if v is not None:
                    return v
        return None

    gdp_val = _first_numeric((gdp_result or {}).get("data"))
    ir_val = _first_numeric((interest_rates_result or {}).get("data"))

    if gdp_val is None:
        missing.append("macro.gdp_growth")
    if ir_val is None:
        missing.append("macro.interest_rates")

    if gdp_val is not None or ir_val is not None:
        score = 50.0
        if gdp_val is not None:
            # Higher growth => better macro backdrop
            delta = 0
            if gdp_val >= 3.0:
                delta = +12
            elif gdp_val >= 1.5:
                delta = +6
            elif gdp_val >= 0:
                delta = 0
            else:
                delta = -10
            score += delta
            factors.append({"factor": "GDP growth proxy", "impact": delta, "value": gdp_val})
        if ir_val is not None:
            # Higher rates => tighter conditions
            delta = 0
            if ir_val >= 6.0:
                delta = -12
            elif ir_val >= 4.0:
                delta = -6
            elif ir_val >= 2.0:
                delta = 0
            else:
                delta = +4
            score += delta
            factors.append({"factor": "Interest rate proxy", "impact": delta, "value": ir_val})

    if score is None:
        return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": sorted(set(missing))}

    s = _clamp_score(score)
    return {"score": s, "signal": _signal_from_score(s), "factors": factors, "missing_fields": sorted(set(missing))}


def compute_intelligence_scores(
    *,
    market_data_result: Dict[str, Any],
    technical_result: Dict[str, Any],
    indicators_result: Dict[str, Any],
    news_result: Dict[str, Any],
    community_result: Dict[str, Any],
    gdp_result: Dict[str, Any],
    interest_rates_result: Dict[str, Any],
) -> Dict[str, Any]:
    f = fundamental_score(market_data_result)
    t = technical_score(technical_result, indicators_result)
    v = valuation_score(market_data_result)
    r = risk_score(market_data_result)
    m = macro_score(gdp_result, interest_rates_result)
    s = sentiment_score(news_result, community_result)

    # Confidence engine (decision reliability, not "API availability").
    # - penalize missing engines
    # - penalize score disagreement
    # - penalize explicit contradictions (bull vs bear signals)
    # - penalize high risk regimes
    # - optionally incorporate historical calibration when available
    engines = [f, t, v, r, m, s]
    available = sum(1 for e in engines if e.get("signal") != "Unavailable")
    completeness = available / len(engines)
    completeness_score = 0 if available == 0 else _clamp_score(40 + completeness * 60)

    # Simple verdict from weighted average (risk penalizes)
    weights = {
        "fundamental": 0.20,
        "technical": 0.25,
        "valuation": 0.15,
        "macro": 0.10,
        "sentiment": 0.10,
        "risk_penalty": 0.20,
    }

    base_weighted_sum = 0.0
    denom = 0.0
    per_engine_weighted = {}
    for key, eng in [("fundamental", f), ("technical", t), ("valuation", v), ("macro", m), ("sentiment", s)]:
        if eng.get("signal") != "Unavailable":
            w = float(weights[key])
            val = float(eng.get("score", 0))
            per_engine_weighted[key] = {"weight": w, "score": val, "weighted": val * w}
            base_weighted_sum += val * w
            denom += w
    base = base_weighted_sum / denom if denom > 0 else 0.0

    risk_pen = 0.0
    if r.get("signal") != "Unavailable":
        risk_pen = r["score"] * weights["risk_penalty"]  # higher risk => larger penalty

    composite = _clamp_score(base - risk_pen / 100 * 30)

    if available == 0:
        composite = 0
        verdict = "REVIEW"
    elif composite >= 67:
        verdict = "BUY"
    elif composite <= 33:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    # Decision trace & factor aggregation (auditable, computed from validated signals only).
    def _impact_num(x) -> int | None:
        try:
            return int(round(float(x)))
        except Exception:
            return None

    engine_map = {
        "fundamental": f,
        "technical": t,
        "valuation": v,
        "macro": m,
        "sentiment": s,
        "risk": r,
    }

    alpha_positive = []
    alpha_negative = []
    risk_contributors = []
    for eng_name, eng in engine_map.items():
        for fac in (eng.get("factors") or [])[:12]:
            if not isinstance(fac, dict):
                continue
            impact = _impact_num(fac.get("impact"))
            if impact is None or impact == 0:
                continue
            item = {
                "engine": eng_name,
                "factor": fac.get("factor"),
                "impact": impact,
                "value": fac.get("value"),
            }
            if eng_name == "risk":
                # Risk factors are not alpha drivers. Always treat as risk contributors.
                risk_contributors.append(item)
                continue
            if impact > 0:
                alpha_positive.append(item)
            else:
                alpha_negative.append(item)

    alpha_positive = sorted(alpha_positive, key=lambda x: x.get("impact", 0), reverse=True)[:8]
    alpha_negative = sorted(alpha_negative, key=lambda x: x.get("impact", 0))[:8]
    # For risk contributors, larger positive impact => more risk; show highest first.
    risk_contributors = sorted(risk_contributors, key=lambda x: x.get("impact", 0), reverse=True)[:8]

    missing_engines = [k for k, e in engine_map.items() if e.get("signal") == "Unavailable"]
    confidence_penalties = []
    missing_penalty = 0
    if missing_engines:
        # 10 points each, capped.
        missing_penalty = min(50, 10 * len(missing_engines))
        for eng in missing_engines:
            confidence_penalties.append({"engine": eng, "penalty": 10, "reason": "Engine unavailable (missing/failed inputs)."})

    # Disagreement penalty (dispersion among non-risk engines)
    import math

    agreement_penalty = 0
    non_risk_scores = []
    for k in ["fundamental", "technical", "valuation", "macro", "sentiment"]:
        eng = engine_map.get(k) or {}
        if eng.get("signal") != "Unavailable":
            try:
                non_risk_scores.append(float(eng.get("score", 0)))
            except Exception:
                pass
    if len(non_risk_scores) >= 2:
        mean = sum(non_risk_scores) / len(non_risk_scores)
        var = sum((x - mean) ** 2 for x in non_risk_scores) / max(1, (len(non_risk_scores) - 1))
        std = math.sqrt(var)
        # std <= 12 => fine; std >= 28 => heavy penalty
        if std > 12:
            agreement_penalty = _clamp_score((std - 12) * 2.0)  # 0..100
            agreement_penalty = min(35, agreement_penalty)

    # Contradiction penalty (bullish vs bearish engine signals)
    bullish = 0
    bearish = 0
    for k in ["fundamental", "technical", "valuation", "macro", "sentiment"]:
        sig = (engine_map.get(k) or {}).get("signal")
        if sig == "Bullish":
            bullish += 1
        elif sig == "Bearish":
            bearish += 1
    contradiction_penalty = 0
    if bullish > 0 and bearish > 0:
        # more mixed signals => less reliable
        contradiction_penalty = min(25, 8 + 4 * min(bullish, bearish))

    # Risk regime penalty (high risk reduces decision reliability)
    risk_regime_penalty = 0
    if r.get("signal") != "Unavailable":
        try:
            risk_regime_penalty = int(round(float(r.get("score", 0)) * 0.25))  # up to 25
            risk_regime_penalty = min(25, max(0, risk_regime_penalty))
        except Exception:
            risk_regime_penalty = 0

    # Historical calibration (optional; requires prior evaluated runs stored in a local JSONL)
    calibration = {"available": False, "expected_accuracy": None, "note": "No evaluated history available yet."}
    calibration_penalty = 0
    try:
        hist_path = os.path.join(os.path.dirname(__file__), "evaluation_runs.jsonl")
        if os.path.exists(hist_path):
            evaluated = []
            with open(hist_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json

                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("evaluated") is True and "correct" in obj:
                        evaluated.append(
                            {
                                "confidence": obj.get("confidence", 0),
                                "correct": bool(obj.get("correct")),
                            }
                        )
                    elif obj.get("valid") is True and obj.get("outcome") in {"WIN", "LOSS"}:
                        evaluated.append(
                            {
                                "confidence": obj.get("confidence", 0),
                                "correct": obj.get("outcome") == "WIN",
                            }
                        )
            if len(evaluated) >= 20:
                # bucket by confidence deciles and compute expected accuracy for current bucket
                bucket = int(min(9, max(0, (completeness_score // 10))))
                in_bucket = [x for x in evaluated if int(min(9, max(0, (int(x.get("confidence", 0)) // 10)))) == bucket]
                if len(in_bucket) >= 10:
                    acc = sum(1 for x in in_bucket if bool(x.get("correct")) is True) / len(in_bucket)
                    calibration = {"available": True, "expected_accuracy": _clamp_score(acc * 100), "note": f"Bucket {bucket*10}-{bucket*10+9} accuracy from local evaluated runs."}
                    # If history says this bucket is weak, reduce confidence.
                    if calibration["expected_accuracy"] is not None:
                        calibration_penalty = max(0, int(round((100 - calibration["expected_accuracy"]) * 0.20)))
    except Exception:
        pass

    confidence_raw = 0 if available == 0 else _clamp_score(
        100
        - missing_penalty
        - agreement_penalty
        - contradiction_penalty
        - risk_regime_penalty
        - calibration_penalty
    )
    # Blend with completeness_score so confidence never exceeds data coverage by too much.
    confidence = 0 if available == 0 else min(confidence_raw, completeness_score + 10)

    # Source reliability (heuristic; used for transparency, not for overriding scores yet).
    source_reliability = {
        "fundamental": "High",
        "valuation": "Medium",
        "technical": "Medium",
        "macro": "Medium",
        "sentiment": "Low",
        "risk": "Medium",
    }

    decision_trace = {
        "base_components": [
            {
                "engine": k,
                "score": int(round(v["score"])),
                "weight": v["weight"],
                "weighted": round(v["weighted"], 2),
            }
            for k, v in per_engine_weighted.items()
        ],
        "base_weighted_sum": round(base_weighted_sum, 2),
        "base_weight_denominator": round(denom, 2),
        "base_score": _clamp_score(base),
        "risk_penalty_score": int(r.get("score", 0)) if r.get("signal") != "Unavailable" else 0,
        "composite_score": composite,
    }

    # Structured multi-agent layer (deterministic; consumes only validated scores/signals/factors).
    def _missing_inputs_for(keys: List[str]) -> List[str]:
        missing_all = []
        for k in keys:
            eng = engine_map.get(k) or {}
            missing_all.extend(list(eng.get("missing_fields") or []))
        # Deduplicate while keeping it short
        seen = set()
        out = []
        for m0 in missing_all:
            if m0 in seen:
                continue
            seen.add(m0)
            out.append(m0)
            if len(out) >= 12:
                break
        return out

    def _top_factor_names(eng_key: str, limit: int = 3) -> List[str]:
        eng = engine_map.get(eng_key) or {}
        out = []
        for fac in (eng.get("factors") or [])[:limit]:
            if isinstance(fac, dict) and fac.get("factor"):
                imp = _impact_num(fac.get("impact"))
                if imp is None:
                    out.append(str(fac.get("factor")))
                else:
                    out.append(f"{fac.get('factor')} ({imp:+})")
        return out

    bull_thesis = []
    if f.get("signal") == "Bullish":
        bull_thesis.append("Fundamentals supportive (profitability/margins/EPS positive where available).")
    if t.get("signal") == "Bullish":
        bull_thesis.append("Technical posture bullish per TA/RSI signals available.")
    if m.get("signal") == "Bullish":
        bull_thesis.append("Macro backdrop supportive based on available proxies.")
    bull_thesis = bull_thesis[:3]

    bear_thesis = []
    if v.get("signal") == "Bearish":
        bear_thesis.append("Valuation expensive (elevated P/E proxy where available).")
    if t.get("signal") == "Bearish":
        bear_thesis.append("Technical posture bearish per TA/RSI signals available.")
    if s.get("signal") == "Bearish":
        bear_thesis.append("News/community sentiment skew negative based on explicit titles/signals only.")
    bear_thesis = bear_thesis[:3]

    # Risk agent (risk-only)
    risk_drivers = _top_factor_names("risk", limit=3)
    risk_agent_out = {
        "agent": "risk",
        "risk_level": r.get("signal", "Unavailable"),
        "risk_score": r.get("score", 0),
        "risk_drivers": risk_drivers,
        "missing_inputs": _missing_inputs_for(["risk"]),
    }

    # Critic agent: identify contradictions & weak points from missing inputs (no freeform speculation).
    contradictions = []
    if f.get("signal") == "Bullish" and v.get("signal") == "Bearish":
        contradictions.append("Strong fundamentals but valuation is expensive (quality vs price conflict).")
    if t.get("signal") == "Bullish" and r.get("signal") == "High":
        contradictions.append("Bullish technicals but risk regime is high (volatility/beta).")
    if t.get("signal") == "Bearish" and f.get("signal") == "Bullish":
        contradictions.append("Short-term technical weakness against strong fundamentals (time-horizon mismatch).")
    contradictions = contradictions[:4]

    critic_flags = []
    if missing_engines:
        critic_flags.append(f"Missing engines: {', '.join(missing_engines)}")
    if agreement_penalty >= 20:
        critic_flags.append("High score dispersion across engines (low agreement).")
    if contradiction_penalty >= 12:
        critic_flags.append("Mixed bullish/bearish signals detected.")
    critic_flags = critic_flags[:4]

    critic_agent_out = {
        "agent": "critic",
        "contradictions": contradictions,
        "flags": critic_flags,
        "missing_inputs": _missing_inputs_for(["fundamental", "technical", "valuation", "macro", "sentiment", "risk"]),
    }

    # Bull/Bear agents
    bull_agent_out = {
        "agent": "bull",
        "verdict": "BUY" if composite >= 67 else "HOLD",
        "confidence": confidence,
        "thesis": bull_thesis if bull_thesis else ["Insufficient bullish signals from validated engines."],
        "top_signals": _top_factor_names("fundamental", limit=2) + _top_factor_names("technical", limit=1),
        "missing_inputs": _missing_inputs_for(["fundamental", "technical", "macro"]),
    }

    bear_agent_out = {
        "agent": "bear",
        "verdict": "SELL" if composite <= 33 else "HOLD",
        "confidence": confidence,
        "thesis": bear_thesis if bear_thesis else ["Insufficient bearish signals from validated engines."],
        "top_signals": _top_factor_names("valuation", limit=2) + _top_factor_names("technical", limit=1),
        "missing_inputs": _missing_inputs_for(["valuation", "technical", "sentiment"]),
    }

    # Final decision agent: adjudicate using composite + critic flags + risk regime.
    overrides = []
    if r.get("signal") == "High":
        if verdict == "BUY" and confidence < 70:
            overrides.append("High risk regime reduces conviction; consider position sizing discipline.")
    final_agent_out = {
        "agent": "final",
        "verdict": verdict,
        "composite_score": composite,
        "confidence": confidence,
        "overrides": overrides[:3],
        "missing_inputs": _missing_inputs_for(["fundamental", "technical", "valuation", "macro", "sentiment", "risk"]),
    }

    agents = {
        "bull": bull_agent_out,
        "bear": bear_agent_out,
        "risk": risk_agent_out,
        "critic": critic_agent_out,
        "final": final_agent_out,
    }

    # Simple probability view (bounded, derived from composite + risk + contradiction penalties).
    # This is not a predictive probability; it's a structured "confidence-weighted inclination" score.
    bull_raw = max(0.0, composite - 40.0)
    bear_raw = max(0.0, 60.0 - composite)
    hold_raw = 10.0 + contradiction_penalty + agreement_penalty
    # risk pushes toward HOLD rather than BUY/SELL
    if r.get("signal") != "Unavailable":
        hold_raw += float(r.get("score", 0)) * 0.10
    total = bull_raw + bear_raw + hold_raw
    if total <= 0:
        probs = {"buy": 0, "hold": 100, "sell": 0, "note": "Insufficient signals; default HOLD."}
    else:
        probs = {
            "buy": _clamp_score((bull_raw / total) * 100),
            "hold": _clamp_score((hold_raw / total) * 100),
            "sell": _clamp_score((bear_raw / total) * 100),
            "note": "Derived from composite score + penalties (not a predictive probability).",
        }

    return {
        "verdict": {"value": verdict, "score": composite},
        "confidence": {
            "score": confidence,
            "note": "Computed from completeness + agreement + contradictions + risk regime (+ optional calibration)."
            if available > 0
            else "Unavailable: no engines produced valid signals.",
            "penalties": confidence_penalties,
            "breakdown": {
                "completeness_score": completeness_score,
                "missing_penalty": missing_penalty,
                "agreement_penalty": agreement_penalty,
                "contradiction_penalty": contradiction_penalty,
                "risk_regime_penalty": risk_regime_penalty,
                "calibration": calibration,
            },
        },
        "decision_trace": decision_trace,
        "source_reliability": source_reliability,
        "alpha_positive_drivers": alpha_positive,
        "alpha_negative_drivers": alpha_negative,
        "risk_contributors": risk_contributors,
        "agents": agents,
        "probabilities": probs,
        "scores": {
            "fundamental": f,
            "technical": t,
            "valuation": v,
            "risk": r,
            "macro": m,
            "sentiment": s,
        },
    }


class StockAnalysisAgent:
    """
    Agent for analyzing individual stocks and generating probability-based recommendations.
    """
    
    def __init__(self, user_id: str = None):
        """Initialize the stock analysis agent with Gemini AI."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and genai is not None:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                print("Stock Analysis Agent initialized with Gemini AI")
            except Exception as e:
                print(f"Error initializing Gemini AI: {e}")
                self.model = None
        elif api_key and genai is None:
            print("Warning: google-generativeai is not installed (cannot import google.generativeai).")
            self.model = None
        else:
            print("Warning: GOOGLE_API_KEY not found")
            self.model = None
        
        self.user_id = user_id or "default_user"
    
    def _convert_to_ticker(self, input_symbol: str) -> tuple:
        """
        Convert company name to ticker symbol if needed.
        
        Args:
            input_symbol (str): Company name or ticker symbol (e.g., "Apple" or "AAPL")
        
        Returns:
            tuple: (ticker_symbol, conversion_info)
        """
        # Check if it looks like a ticker symbol (1-5 uppercase letters, maybe with numbers)
        import re
        ticker_pattern = r'^[A-Z]{1,5}[0-9]{0,2}$'
        
        if re.match(ticker_pattern, input_symbol):
            # It looks like a ticker symbol, use as-is
            return input_symbol, {
                "original_input": input_symbol,
                "converted": False,
                "type": "ticker_symbol"
            }
        
        # It looks like a company name, search for ticker
        ticker_result = find_ticker_by_company_name(input_symbol)
        
        if ticker_result['status'] == 'SUCCESS':
            return ticker_result['ticker'], {
                "original_input": input_symbol,
                "converted": True,
                "type": "company_name",
                "company_name": ticker_result['company_name'],
                "exchange": ticker_result['exchange']
            }
        else:
            # Search failed, try using the input as-is
            return input_symbol, {
                "original_input": input_symbol,
                "converted": False,
                "type": "unknown",
                "error": ticker_result.get('message', 'Search failed')
            }
    
    def analyze_investment_scenario(self, symbol: str, investment_amount: float = 10000, days: int = 100) -> Dict[str, Any]:
        """
        Perform comprehensive investment scenario analysis with best/worst/base cases.
        
        Args:
            symbol (str): Stock symbol or company name to analyze (e.g., "AAPL" or "Apple")
            investment_amount (float): Investment amount in USD (default: $10,000)
            days (int): Investment timeframe in days (default: 100 days)
        
        Returns:
            dict: Comprehensive scenario analysis with realistic ranges
        """
        execution_steps = []
        start_time = time.time()
        conversion_info = {}
        
        try:
            # Convert company name to ticker symbol if needed
            ticker_symbol, conversion_info = self._convert_to_ticker(symbol)
            
            execution_steps.append({
                "step": 0,
                "action": f"Input conversion: {symbol} -> {ticker_symbol}",
                "status": "SUCCESS",
                "conversion_info": conversion_info,
                "duration": f"{time.time() - start_time:.2f}s"
            })
            
            # Reset start time for actual analysis
            start_time = time.time()
            # Step 1: Get current price
            step_start = time.time()
            price_result = get_price(ticker_symbol)
            execution_steps.append({
                "step": 1,
                "action": "Fetch Current Price",
                "status": price_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 2: Get comprehensive market data for volatility metrics
            step_start = time.time()
            market_data_result = get_market_data_multi_exchange(ticker_symbol, ["NASDAQ", "NYSE"])
            execution_steps.append({
                "step": 2,
                "action": "Fetch Market Data (Volatility Metrics)",
                "status": market_data_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 3: Get 1-year historical data for volatility analysis
            step_start = time.time()
            historical_result = get_year_historical_data(ticker_symbol)
            execution_steps.append({
                "step": 3,
                "action": "Fetch Historical Data (Volatility Analysis)",
                "status": historical_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 4: Get technical analysis for trend direction
            step_start = time.time()
            technical_result = get_technical_analysis(ticker_symbol)
            execution_steps.append({
                "step": 4,
                "action": "Fetch Technical Analysis (Trend)",
                "status": technical_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 5: Get technical indicators for volatility
            step_start = time.time()
            indicators_result = get_technical_indicators(ticker_symbol)
            execution_steps.append({
                "step": 5,
                "action": "Fetch Technical Indicators (Volatility)",
                "status": indicators_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 6: Get news for sentiment analysis
            step_start = time.time()
            news_result = get_stock_news(ticker_symbol)
            if news_result.get('status') != 'SUCCESS':
                news_result = get_stock_market_news()
            execution_steps.append({
                "step": 6,
                "action": "Fetch News (Sentiment)",
                "status": news_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 7: Generate scenario analysis using AI
            step_start = time.time()
            scenario_analysis = self._generate_scenario_analysis(
                ticker_symbol, investment_amount, days, price_result, market_data_result,
                historical_result, technical_result, indicators_result, news_result
            )
            execution_steps.append({
                "step": 7,
                "action": "Generate Investment Scenario Analysis",
                "status": "SUCCESS",
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Compile scenario report
            report = self._compile_scenario_report(
                ticker_symbol, investment_amount, days, price_result, market_data_result,
                historical_result, scenario_analysis, conversion_info
            )
            
            return {
                "status": "SUCCESS",
                "input": symbol,
                "ticker": ticker_symbol,
                "investment_amount": investment_amount,
                "timeframe_days": days,
                "report": report,
                "scenario_analysis": scenario_analysis,
                "execution_steps": execution_steps,
                "total_time": f"{time.time() - start_time:.2f}s",
                "analysis_type": "investment scenario analysis",
                "tools_used": 6,
                "conversion_info": conversion_info
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error analyzing investment scenario: {str(e)}",
                "execution_steps": execution_steps,
                "total_time": f"{time.time() - start_time:.2f}s"
            }

    def analyze_stock_one_month(self, symbol: str) -> Dict[str, Any]:
        """
        Perform focused 1-month stock analysis with clear prediction.
        
        Args:
            symbol (str): Stock symbol or company name to analyze (e.g., "AAPL" or "Apple")
        
        Returns:
            dict: Focused 1-month analysis with clear increase/decrease prediction
        """
        execution_steps = []
        start_time = time.time()
        
        try:
            # Convert company name to ticker symbol if needed
            ticker_symbol, conversion_info = self._convert_to_ticker(symbol)
            
            execution_steps.append({
                "step": 0,
                "action": f"Input conversion: {symbol} -> {ticker_symbol}",
                "status": "SUCCESS",
                "conversion_info": conversion_info,
                "duration": f"{time.time() - start_time:.2f}s"
            })
            
            # Reset start time for actual analysis
            start_time = time.time()
            # Step 1: Get current price
            step_start = time.time()
            price_result = get_price(ticker_symbol)
            execution_steps.append({
                "step": 1,
                "action": "Fetch Current Price",
                "status": price_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 2: Get comprehensive market data (15 sections) for fundamentals
            step_start = time.time()
            market_data_result = get_market_data_multi_exchange(ticker_symbol, ["NASDAQ", "NYSE"])
            execution_steps.append({
                "step": 2,
                "action": "Fetch Comprehensive Market Data",
                "status": market_data_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 3: Get 1-year historical data for pattern analysis
            step_start = time.time()
            historical_result = get_year_historical_data(ticker_symbol)
            execution_steps.append({
                "step": 3,
                "action": "Fetch 1-Year Historical Data (Pattern Analysis)",
                "status": historical_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 4: Get technical analysis for signals
            step_start = time.time()
            technical_result = get_technical_analysis(ticker_symbol)
            execution_steps.append({
                "step": 4,
                "action": "Fetch Technical Analysis",
                "status": technical_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 5: Get technical indicators for detailed signals
            step_start = time.time()
            indicators_result = get_technical_indicators(ticker_symbol)
            execution_steps.append({
                "step": 5,
                "action": "Fetch Technical Indicators",
                "status": indicators_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 6: Get stock news for recent sentiment
            step_start = time.time()
            news_result = get_stock_news(ticker_symbol)
            if news_result.get('status') != 'SUCCESS':
                news_result = get_stock_market_news()
            execution_steps.append({
                "step": 6,
                "action": "Fetch Recent News",
                "status": news_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 7: Get analyst recommendations
            step_start = time.time()
            analyst_recommendations_result = get_analyst_recommendations(ticker_symbol)
            execution_steps.append({
                "step": 7,
                "action": "Fetch Analyst Recommendations",
                "status": analyst_recommendations_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 8: Get economic calendar for market context
            step_start = time.time()
            from datetime import datetime, timedelta
            today = datetime.now()
            month_later = today + timedelta(days=30)
            from_timestamp = int(today.timestamp())
            to_timestamp = int(month_later.timestamp())
            economic_calendar_result = get_economic_calendar(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            execution_steps.append({
                "step": 8,
                "action": "Fetch Economic Calendar (30-day context)",
                "status": economic_calendar_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 9: Generate focused 1-month prediction using AI
            step_start = time.time()
            prediction_analysis = self._generate_one_month_prediction(
                ticker_symbol, price_result, market_data_result, historical_result,
                technical_result, indicators_result, news_result,
                analyst_recommendations_result, economic_calendar_result
            )
            execution_steps.append({
                "step": 9,
                "action": "Generate 1-Month Prediction",
                "status": "SUCCESS",
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Compile focused report
            report = self._compile_one_month_report(
                ticker_symbol, price_result, market_data_result, historical_result,
                technical_result, indicators_result, news_result,
                analyst_recommendations_result, economic_calendar_result,
                prediction_analysis, conversion_info
            )
            
            return {
                "status": "SUCCESS",
                "input": symbol,
                "ticker": ticker_symbol,
                "report": report,
                "prediction": prediction_analysis,
                "execution_steps": execution_steps,
                "total_time": f"{time.time() - start_time:.2f}s",
                "analysis_type": "1-month prediction",
                "tools_used": 8,
                "conversion_info": conversion_info
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error analyzing stock: {str(e)}",
                "execution_steps": execution_steps,
                "total_time": f"{time.time() - start_time:.2f}s"
            }

    def analyze_stock(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single stock using ALL available tools.
        
        Args:
            symbol (str): Stock symbol to analyze (e.g., "AAPL", "TSLA")
        
        Returns:
            dict: Comprehensive analysis report with probability assessment
        """
        execution_steps = []
        start_time = time.time()
        
        try:
            # Step 1: Get current price
            step_start = time.time()
            price_result = get_price(symbol)
            execution_steps.append({
                "step": 1,
                "action": "Fetch Current Price",
                "status": price_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 2: Get comprehensive market data (NEW - 15 data sections)
            step_start = time.time()
            market_data_result = get_market_data_multi_exchange(symbol, ["NASDAQ", "NYSE"])
            execution_steps.append({
                "step": 2,
                "action": "Fetch Comprehensive Market Data (15 sections)",
                "status": market_data_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 3: Get company information (NEW)
            step_start = time.time()
            company_info_result = get_company_info(symbol)
            execution_steps.append({
                "step": 3,
                "action": "Fetch Company Information",
                "status": company_info_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 4: Get analyst recommendations (NEW)
            step_start = time.time()
            analyst_recommendations_result = get_analyst_recommendations(symbol)
            execution_steps.append({
                "step": 4,
                "action": "Fetch Analyst Recommendations",
                "status": analyst_recommendations_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 5: Get 1-year historical data
            step_start = time.time()
            historical_result = get_year_historical_data(symbol)
            execution_steps.append({
                "step": 5,
                "action": "Fetch 1-Year Historical Data",
                "status": historical_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 6: Get stock news (try symbol-specific first, then general stock market news)
            step_start = time.time()
            news_result = get_stock_news(symbol)
            # If symbol-specific news fails, fall back to general stock market news
            if news_result.get('status') != 'SUCCESS':
                news_result = get_stock_market_news()
            execution_steps.append({
                "step": 6,
                "action": "Fetch Current News",
                "status": news_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 7: Get technical analysis
            step_start = time.time()
            technical_result = get_technical_analysis(symbol)
            execution_steps.append({
                "step": 7,
                "action": "Fetch Technical Analysis",
                "status": technical_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 8: Get detailed technical indicators
            step_start = time.time()
            indicators_result = get_technical_indicators(symbol)
            execution_steps.append({
                "step": 8,
                "action": "Fetch Technical Indicators",
                "status": indicators_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 9: Get community data (NEW)
            step_start = time.time()
            community_result = get_community_data(symbol)
            execution_steps.append({
                "step": 9,
                "action": "Fetch Community Sentiment Data",
                "status": community_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 10: Get economic calendar (NEW - market context)
            step_start = time.time()
            from datetime import datetime, timedelta
            today = datetime.now()
            week_later = today + timedelta(days=7)
            from_timestamp = int(today.timestamp())
            to_timestamp = int(week_later.timestamp())
            economic_calendar_result = get_economic_calendar(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            execution_steps.append({
                "step": 10,
                "action": "Fetch Economic Calendar (market context)",
                "status": economic_calendar_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 11: Get earnings calendar (NEW - company-specific)
            step_start = time.time()
            earnings_calendar_result = get_earnings_calendar(
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp
            )
            execution_steps.append({
                "step": 11,
                "action": "Fetch Earnings Calendar (company events)",
                "status": earnings_calendar_result.get("status"),
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 12: Get market leaderboards (NEW - market comparison)
            step_start = time.time()
            gainers_result = get_stock_gainers()
            losers_result = get_stock_losers()
            active_result = get_most_active_stocks()
            execution_steps.append({
                "step": 12,
                "action": "Fetch Market Leaderboards (comparison context)",
                "status": "SUCCESS" if all([
                    gainers_result.get("status") == "SUCCESS",
                    losers_result.get("status") == "SUCCESS", 
                    active_result.get("status") == "SUCCESS"
                ]) else "PARTIAL",
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 13: Get world economy indicators (NEW - macro context)
            step_start = time.time()
            gdp_result = get_g20_gdp_growth()
            interest_rates_result = get_interest_rates()
            execution_steps.append({
                "step": 13,
                "action": "Fetch World Economy Indicators (macro context)",
                "status": "SUCCESS" if all([
                    gdp_result.get("status") == "SUCCESS",
                    interest_rates_result.get("status") == "SUCCESS"
                ]) else "PARTIAL",
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Step 14: Generate comprehensive probability analysis using AI
            step_start = time.time()
            probability_analysis = self._generate_comprehensive_analysis(
                symbol, price_result, market_data_result, company_info_result,
                analyst_recommendations_result, historical_result, news_result,
                technical_result, indicators_result, community_result,
                economic_calendar_result, earnings_calendar_result,
                gainers_result, losers_result, active_result,
                gdp_result, interest_rates_result
            )
            execution_steps.append({
                "step": 14,
                "action": "Generate Comprehensive Probability Analysis",
                "status": "SUCCESS",
                "duration": f"{time.time() - step_start:.2f}s"
            })
            
            # Compile final comprehensive report
            report = self._compile_comprehensive_report(
                symbol, price_result, market_data_result, company_info_result,
                analyst_recommendations_result, historical_result, news_result,
                technical_result, indicators_result, community_result,
                economic_calendar_result, earnings_calendar_result,
                gainers_result, losers_result, active_result,
                gdp_result, interest_rates_result, probability_analysis
            )

            intelligence = compute_intelligence_scores(
                market_data_result=market_data_result,
                technical_result=technical_result,
                indicators_result=indicators_result,
                news_result=news_result,
                community_result=community_result,
                gdp_result=gdp_result,
                interest_rates_result=interest_rates_result,
            )

            # If ALL core engines report Unavailable, mark for REVIEW
            scores_dict = intelligence.get("scores", {})
            all_unavailable = all(
                scores_dict.get(k, {}).get("signal") == "Unavailable"
                for k in ("fundamental", "technical", "valuation", "risk")
            )
            if all_unavailable:
                intelligence["verdict"] = {"value": "REVIEW", "score": 0}
                intelligence["data_error"] = {
                    "type": "ALL_DATA_UNAVAILABLE",
                    "message": "RapidAPI endpoints unavailable — use strategy prediction agent for options analysis.",
                }

            return {
                "status": "SUCCESS",
                "symbol": symbol,
                "intelligence": intelligence,
                "report": report,
                "execution_steps": execution_steps,
                "total_time": f"{time.time() - start_time:.2f}s",
                "tools_used": 13
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error analyzing stock: {str(e)}",
                "execution_steps": execution_steps,
                "total_time": f"{time.time() - start_time:.2f}s"
            }
    
    def _generate_scenario_analysis(self, symbol: str, investment_amount: float, days: int,
                                  price_result: Dict, market_data_result: Dict,
                                  historical_result: Dict, technical_result: Dict,
                                  indicators_result: Dict, news_result: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive investment scenario analysis with realistic ranges.
        
        Args:
            symbol: Stock symbol
            investment_amount: Investment amount in USD
            days: Investment timeframe in days
            price_result: Current price data
            market_data_result: Market data for volatility
            historical_result: Historical data for volatility analysis
            technical_result: Technical analysis for trend
            indicators_result: Technical indicators for volatility
            news_result: News for sentiment
        
        Returns:
            dict: Comprehensive scenario analysis with best/worst/base cases
        """
        if not self.model:
            return {
                "status": "ERROR",
                "message": "AI model not available"
            }
        
        try:
            # Prepare data summary for scenario analysis
            data_summary = f"""
INVESTMENT SCENARIO ANALYSIS FOR: {symbol}
Investment Amount: ${investment_amount:,.2f}
Timeframe: {days} days

CURRENT SITUATION:
"""
            
            # Add current price
            if price_result.get('status') == 'SUCCESS':
                price_data = price_result.get('data', {})
                if price_data.get('success'):
                    actual_data = price_data.get('data', {})
                    current = actual_data.get('current', {})
                    info = actual_data.get('info', {})
                    current_price = current.get('close', 0)
                    shares = investment_amount / current_price if current_price > 0 else 0
                    
                    data_summary += f"""
- Current Price: ${current_price:.2f}
- Investment Amount: ${investment_amount:,.2f}
- Shares Purchased: {shares:.2f}
- Company: {info.get('description', 'N/A')}
"""
            
            # Add volatility metrics from market data
            if market_data_result.get('status') == 'SUCCESS':
                market_data = market_data_result.get('data', {})
                if market_data.get('success'):
                    inner_data = market_data.get('data', {})
                    data_summary += f"""
VOLATILITY METRICS:
"""
                    if 'indicators' in inner_data:
                        indicators = inner_data['indicators']
                        beta = indicators.get('beta_1_year', 1.0)
                        week_high = indicators.get('price_52_week_high', 0)
                        week_low = indicators.get('price_52_week_low', 0)
                        volatility_range = ((week_high - week_low) / week_low * 100) if week_low > 0 else 0
                        
                        data_summary += f"""
- Beta (1Y): {beta:.2f} (Market sensitivity)
- 52-Week Range: ${week_low:.2f} - ${week_high:.2f}
- Historical Volatility Range: {volatility_range:.1f}%
- P/E Ratio: {indicators.get('price_earnings', 'N/A')}
"""
            
            # Add historical volatility context
            if historical_result.get('status') == 'SUCCESS':
                data_summary += """
HISTORICAL VOLATILITY ANALYSIS:
- 1-Year historical data available for volatility calculation
- Historical price patterns and volatility trends accessible
- Support and resistance levels identifiable
"""
            
            # Add technical trend
            if technical_result.get('status') == 'SUCCESS':
                data_summary += """
TECHNICAL TREND ANALYSIS:
- Current trend direction available
- Chart patterns and momentum signals accessible
"""
            
            # Add indicators for volatility
            if indicators_result.get('status') == 'SUCCESS':
                data_summary += """
TECHNICAL INDICATORS (Volatility):
- RSI (Relative Strength Index) available
- MACD and momentum indicators accessible
- Volatility indicators (ATR, Bollinger Bands) available
"""
            
            # Add sentiment context
            if news_result.get('status') == 'SUCCESS':
                data_summary += """
MARKET SENTIMENT:
- Recent news sentiment analysis available
- Current market conditions and company-specific news
"""
            
            prompt = f"""
            Based on the data above, provide a COMPREHENSIVE INVESTMENT SCENARIO ANALYSIS for {symbol}.
            
            Investment: ${investment_amount:,.2f}
            Timeframe: {days} days
            Current shares: {investment_amount / price_result.get('data', {}).get('data', {}).get('current', {}).get('close', 1):.2f}
            
            Provide THREE SCENARIOS with REALISTIC ranges based on historical volatility:
            
            1. BEST CASE SCENARIO (Optimistic but realistic):
               - Expected price increase: X% to Y%
               - Investment value: $[amount]
               - Profit: $[amount]
               - Probability: X%
               - What conditions would make this happen?
            
            2. BASE CASE SCENARIO (Most likely outcome):
               - Expected price change: X% to Y%
               - Investment value: $[amount]
               - Profit/Loss: $[amount]
               - Probability: X%
               - Why is this most likely?
            
            3. WORST CASE SCENARIO (Risk management):
               - Maximum expected decline: X% to Y%
               - Investment value: $[amount]
               - Maximum loss: $[amount]
               - Probability: X%
               - What could cause this?
            
            RISK ANALYSIS:
            - Maximum drawdown risk: X%
            - Stop-loss recommendation: $[price]
            - Position size recommendation: X% of portfolio
            - Risk/reward ratio: X:Y
            
            CONFIDENCE FACTORS:
            - Historical volatility support: [High/Medium/Low]
            - Technical trend confirmation: [Strong/Weak/Neutral]
            - Market sentiment alignment: [Positive/Negative/Neutral]
            - Overall confidence in scenarios: [X]%
            
            IMPORTANT: Be realistic and conservative. Use historical volatility data to calculate realistic ranges.
            Don't promise unrealistic returns. Focus on risk management and realistic expectations.
            
            Format as a clean, professional investment analysis that helps investors make informed decisions.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "status": "SUCCESS",
                "analysis": response.text,
                "investment_amount": investment_amount,
                "timeframe_days": days,
                "focus": "comprehensive scenario analysis with risk management"
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error generating scenario analysis: {str(e)}"
            }

    def _compile_scenario_report(self, symbol: str, investment_amount: float, days: int,
                               price_result: Dict, market_data_result: Dict,
                               historical_result: Dict, scenario_analysis: Dict,
                               conversion_info: Dict) -> str:
        """
        Compile comprehensive scenario analysis report.
        """
        # Calculate shares based on current price
        shares = 0
        current_price = 0
        if price_result.get('status') == 'SUCCESS':
            price_data = price_result.get('data', {})
            if price_data.get('success'):
                actual_data = price_data.get('data', {})
                current = actual_data.get('current', {})
                current_price = current.get('close', 0)
                shares = investment_amount / current_price if current_price > 0 else 0
        
        report = f"""
{'='*80}
COMPREHENSIVE INVESTMENT SCENARIO ANALYSIS: {symbol}
{'='*80}

INPUT CONVERSION:
- Original Input: {conversion_info.get('original_input', 'N/A')}
- Converted: {conversion_info.get('converted', False)}
- Input Type: {conversion_info.get('type', 'N/A')}
"""
        
        if conversion_info.get('converted'):
            report += f"- Company Name: {conversion_info.get('company_name', 'N/A')}\n"
            report += f"- Exchange: {conversion_info.get('exchange', 'N/A')}\n"
        
        report += f"""
INVESTMENT DETAILS:
- Investment Amount: ${investment_amount:,.2f}
- Timeframe: {days} days
- Current Price: ${current_price:.2f}
- Shares Purchased: {shares:.2f}
- Analysis Type: Comprehensive Scenario Planning
- Tools Used: 6 (Price, Volatility Metrics, Historical Analysis, 
             Technical Trend, Volatility Indicators, Market Sentiment)

"""
        
        # Add volatility metrics
        if market_data_result.get('status') == 'SUCCESS':
            market_data = market_data_result.get('data', {})
            if market_data.get('success'):
                inner_data = market_data.get('data', {})
                if 'indicators' in inner_data:
                    indicators = inner_data['indicators']
                    beta = indicators.get('beta_1_year', 1.0)
                    week_high = indicators.get('price_52_week_high', 0)
                    week_low = indicators.get('price_52_week_low', 0)
                    volatility_range = ((week_high - week_low) / week_low * 100) if week_low > 0 else 0
                    
                    report += f"VOLATILITY PROFILE:\n"
                    report += f"Beta (Market Sensitivity): {beta:.2f}\n"
                    report += f"52-Week Range: ${week_low:.2f} - ${week_high:.2f}\n"
                    report += f"Historical Volatility: {volatility_range:.1f}%\n"
                    report += f"P/E Ratio: {indicators.get('price_earnings', 'N/A')}\n\n"
        
        # Add the AI scenario analysis
        if scenario_analysis.get('status') == 'SUCCESS':
            report += f"INVESTMENT SCENARIOS:\n"
            report += f"{scenario_analysis.get('analysis')}\n\n"
        else:
            report += f"Scenario Analysis: {scenario_analysis.get('message', 'Not available')}\n\n"
        
        report += f"{'='*80}\n"
        report += f"DISCLAIMER: This analysis is for educational purposes only.\n"
        report += f"Past performance does not guarantee future results.\n"
        report += f"Invest based on your own research and risk tolerance.\n"
        report += f"{'='*80}\n"
        
        return report

    def _generate_one_month_prediction(self, symbol: str, price_result: Dict,
                                     market_data_result: Dict, historical_result: Dict,
                                     technical_result: Dict, indicators_result: Dict,
                                     news_result: Dict, analyst_recommendations_result: Dict,
                                     economic_calendar_result: Dict) -> Dict[str, Any]:
        """
        Generate focused 1-month price prediction based on historical patterns and key factors.
        
        Args:
            symbol: Stock symbol
            price_result: Current price data
            market_data_result: Comprehensive market data
            historical_result: Historical data for pattern analysis
            technical_result: Technical analysis
            indicators_result: Technical indicators
            news_result: News sentiment
            analyst_recommendations_result: Analyst recommendations
            economic_calendar_result: Economic calendar for context
        
        Returns:
            dict: Focused 1-month prediction with clear increase/decrease and percentage
        """
        if not self.model:
            return {
                "status": "ERROR",
                "message": "AI model not available"
            }
        
        try:
            # Prepare focused data summary for 1-month prediction
            data_summary = f"""
1-MONTH STOCK PRICE PREDICTION FOR: {symbol}

CURRENT SITUATION:
"""
            
            # Add current price - check both status and nested success flag
            price_available = False
            if price_result.get('status') == 'SUCCESS':
                price_data = price_result.get('data', {})
                if price_data.get('success'):
                    actual_data = price_data.get('data', {})
                    current = actual_data.get('current', {})
                    info = actual_data.get('info', {})
                    if current.get('close') or info.get('description'):
                        price_available = True
                        data_summary += f"""
- Current Price: ${current.get('close', 'N/A')}
- Company: {info.get('description', 'N/A')}
- Exchange: {info.get('exchange', 'N/A')}
- Volume: {current.get('volume', 'N/A')}
"""
            
            if not price_available:
                data_summary += f"""
- Current Price: Data not available from API
- Company: {symbol}
- Note: This stock may not be available on the TradingView data feed
"""
            
            # Add comprehensive market data fundamentals - check nested success flag
            market_data_available = False
            if market_data_result.get('status') == 'SUCCESS':
                market_data = market_data_result.get('data', {})
                if market_data.get('success'):
                    inner_data = market_data.get('data', {})
                    
                    if 'indicators' in inner_data:
                        indicators = inner_data['indicators']
                        if indicators.get('market_cap_calc') or indicators.get('price_earnings'):
                            market_data_available = True
                            data_summary += f"\nFUNDAMENTAL DATA:\n"
                            data_summary += f"""
- Market Cap: ${indicators.get('market_cap_calc', 0):,.0f}
- P/E Ratio: {indicators.get('price_earnings', 'N/A')}
- 52-Week High: ${indicators.get('price_52_week_high', 'N/A')}
- 52-Week Low: ${indicators.get('price_52_week_low', 'N/A')}
- Beta (1Y): {indicators.get('beta_1_year', 'N/A')}
"""
                    
                    if 'ttm' in inner_data:
                        ttm = inner_data['ttm']
                        if ttm.get('earnings_per_share_ttm') or ttm.get('total_revenue_ttm'):
                            market_data_available = True
                            if not data_summary.endswith("FUNDAMENTAL DATA:\n"):
                                data_summary += f"\nFUNDAMENTAL DATA:\n"
                            data_summary += f"""
- EPS (TTM): ${ttm.get('earnings_per_share_ttm', 'N/A')}
- Revenue (TTM): ${ttm.get('total_revenue_ttm', 0):,.0f}
- Net Margin (TTM): {ttm.get('net_margin_ttm', 'N/A')}%
- Gross Margin (TTM): {ttm.get('gross_margin_ttm', 'N/A')}%
"""
            
            # Add historical data availability
            if historical_result.get('status') == 'SUCCESS':
                hist_data = historical_result.get('data', {})
                if hist_data.get('success') or hist_data.get('prices'):
                    data_summary += """
HISTORICAL PATTERNS:
- 1-Year Historical Data: Available for pattern analysis
- Price trends and patterns can be analyzed
- Support and resistance levels identifiable
"""
            
            # Add technical signals
            if technical_result.get('status') == 'SUCCESS':
                tech_data = technical_result.get('data', {})
                if tech_data.get('success') or tech_data.get('summary'):
                    data_summary += """
TECHNICAL ANALYSIS:
- Technical signals and indicators available
- Chart patterns and trend analysis possible
"""
            
            # Add indicators
            if indicators_result.get('status') == 'SUCCESS':
                ind_data = indicators_result.get('data', {})
                if ind_data.get('success') or ind_data.get('indicators'):
                    data_summary += """
TECHNICAL INDICATORS:
- RSI, MACD, moving averages available
- Momentum and volatility indicators accessible
"""
            
            # Add news sentiment
            if news_result.get('status') == 'SUCCESS':
                news_data = news_result.get('data', {})
                has_news = (isinstance(news_data, list) and len(news_data) > 0) or (
                    isinstance(news_data, dict) and (news_data.get('success') or news_data.get('news'))
                )
                if has_news:
                    data_summary += """
RECENT NEWS SENTIMENT:
- Current news articles and sentiment available
- Recent company developments and market news
"""
            
            # Add analyst views
            if analyst_recommendations_result.get('status') == 'SUCCESS':
                analyst_data = analyst_recommendations_result.get('data', {})
                if analyst_data.get('success') or analyst_data.get('recommendations'):
                    data_summary += """
ANALYST RECOMMENDATIONS:
- Professional analyst ratings available
- Price targets and consensus estimates
"""
            
            # Add economic context
            if economic_calendar_result.get('status') == 'SUCCESS':
                econ_data = economic_calendar_result.get('data', {})
                if econ_data.get('success') or econ_data.get('events'):
                    data_summary += """
ECONOMIC CONTEXT (Next 30 Days):
- Upcoming economic events scheduled
- Market-moving announcements in timeline
"""
            
            prompt = f"""
            Based on the data above, provide a CLEAN, FOCUSED 1-month price prediction for {symbol}.
            
            IMPORTANT NOTES:
            - The data above includes ONLY the information that was successfully retrieved from the API
            - Some data sections may be missing if the API doesn't have that information for this stock
            - If you have at least the current price and company name, you should provide a prediction based on available data
            - If you have very limited data (e.g., only current price), provide a prediction with LOW confidence and explain the limitations
            - Only state "cannot provide prediction" if you have absolutely no data (not even current price)
            
            If you have sufficient data to make a prediction, focus on these specific requirements:
            
            1. CLEAR PREDICTION: Will the stock INCREASE or DECREASE in the next 30 days?
            2. SPECIFIC PERCENTAGE: How much will it increase or decrease? (Give a specific percentage range)
            3. HISTORICAL PATTERNS: What historical patterns support this prediction?
            4. KEY FACTORS: What are the 3-5 most important factors driving this prediction?
            5. CONFIDENCE LEVEL: How confident are you in this prediction? (0-100%)
            6. TARGET PRICE: What is your target price in 30 days?
            
            Format your response as a CLEAN, ACTIONABLE analysis:
            
            PREDICTION: [INCREASE/DECREASE] by [X% to Y%]
            TARGET PRICE: $[price] in 30 days
            CONFIDENCE: [X]%
            
            HISTORICAL PATTERNS SUPPORTING THIS:
            - [Pattern 1]
            - [Pattern 2]
            - [Pattern 3]
            
            KEY FACTORS:
            1. [Factor 1] - [Why it matters]
            2. [Factor 2] - [Why it matters]
            3. [Factor 3] - [Why it matters]
            
            RISK CONSIDERATIONS:
            - [Risk 1]
            - [Risk 2]
            
            Keep it concise, clear, and focused on the 1-month timeframe.
            """
            
            # Combine data_summary with prompt
            full_prompt = data_summary + "\n" + prompt
            
            response = self.model.generate_content(full_prompt)
            analysis_text = response.text

            # Lightweight policy enforcement: flag forbidden speculative tokens.
            violations = []
            lowered = (analysis_text or "").lower()
            for token in ["(inferred", "inferred", "assumed", "likely", "probably"]:
                if token in lowered:
                    violations.append(token)
            if violations:
                analysis_text = (
                    "MODEL OUTPUT WARNING: speculative language detected "
                    f"{sorted(set(violations))}. Treat those lines as non-auditable.\n\n"
                    + (analysis_text or "")
                )

            return {
                "status": "SUCCESS",
                "analysis": analysis_text,
                "timeframe": "1-month",
                "focus": "price prediction with historical patterns"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error generating prediction: {str(e)}"
            }

    def evaluate_price_only_cutoff(self, symbol: str, *, as_of_date: str, horizon_days: int = 30) -> Dict[str, Any]:
        """
        Evaluation Lab runner (price-only, leakage-safe):
        - Uses only historical prices <= as_of_date to compute Technical + Risk engines.
        - Fundamental/Valuation/Macro/Sentiment are marked Unavailable to avoid leaking current APIs.
        - Computes realized return over (as_of_date, as_of_date+horizon] when available.
        - Logs runs to evaluation_runs.jsonl for calibration dashboards.
        """
        start_time = time.time()
        cutoff_ts = _ts_from_date_str(as_of_date)
        if cutoff_ts is None:
            return {"status": "ERROR", "message": "Invalid as_of_date. Use YYYY-MM-DD."}

        hist = get_year_historical_data(symbol)
        history = _history_from_historical_result(hist)
        if hist.get("status") != "SUCCESS" or not history:
            return {"status": "ERROR", "message": "Historical data unavailable for evaluation.", "historical_result": hist}

        tech = _technical_score_from_history(history, cutoff_ts)
        risk = _risk_score_from_history(history, cutoff_ts)

        def _stub(name: str) -> Dict[str, Any]:
            return {"score": 0, "signal": "Unavailable", "factors": [], "missing_fields": [f"{name}.unavailable_in_price_only_mode"]}

        f0 = _stub("fundamental")
        v0 = _stub("valuation")
        m0 = _stub("macro")
        s0 = _stub("sentiment")

        base_intel = compute_intelligence_scores(
            market_data_result={"status": "ERROR"},
            technical_result={"status": "ERROR"},
            indicators_result={"status": "ERROR"},
            news_result={"status": "ERROR"},
            community_result={"status": "ERROR"},
            gdp_result={"status": "ERROR"},
            interest_rates_result={"status": "ERROR"},
        )
        base_intel["scores"]["technical"] = tech
        base_intel["scores"]["risk"] = risk
        base_intel["scores"]["fundamental"] = f0
        base_intel["scores"]["valuation"] = v0
        base_intel["scores"]["macro"] = m0
        base_intel["scores"]["sentiment"] = s0

        if tech.get("signal") == "Unavailable" and risk.get("signal") == "Unavailable":
            composite = 0
            verdict = "HOLD"
        else:
            comp = float(tech.get("score", 0) or 0)
            pen = float(risk.get("score", 0) or 0) * 0.30
            composite = _clamp_score(comp - pen)
            verdict = "BUY" if composite >= 67 else ("SELL" if composite <= 33 else "HOLD")

        base_intel["verdict"] = {"value": verdict, "score": composite}
        base_intel["confidence"] = {
            "score": _clamp_score(70 - (risk.get("score", 0) or 0) * 0.20),
            "note": "Price-only cutoff mode: confidence derived from risk regime + limited signal coverage.",
            "penalties": [{"engine": "fundamental/valuation/macro/sentiment", "penalty": 10, "reason": "Unavailable in price-only cutoff mode."}],
            "breakdown": {"mode": "price_only_cutoff"},
        }

        horizon_ts = cutoff_ts + int(horizon_days) * 86400
        px0 = _close_at_or_before(history, cutoff_ts)
        px1 = _close_at_or_before(history, horizon_ts)
        evaluated = False
        actual_return = None
        correct = None
        if px0 is not None and px1 is not None and px0 > 0:
            evaluated = True
            actual_return = (px1 / px0) - 1.0
            if verdict == "BUY":
                correct = actual_return > 0
            elif verdict == "SELL":
                correct = actual_return < 0
            else:
                correct = abs(actual_return) < 0.01

        run_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "mode": "price_only_cutoff",
            "as_of_date": as_of_date,
            "horizon_days": horizon_days,
            "verdict": verdict,
            "confidence": base_intel.get("confidence", {}).get("score"),
            "composite_score": composite,
            "evaluated": evaluated,
            "actual_return": actual_return,
            "correct": correct,
        }
        try:
            path = os.path.join(os.path.dirname(__file__), "evaluation_runs.jsonl")
            _write_jsonl(path, run_obj)
        except Exception:
            pass

        return {
            "status": "SUCCESS",
            "symbol": symbol,
            "mode": "price_only_cutoff",
            "as_of_date": as_of_date,
            "horizon_days": horizon_days,
            "intelligence": base_intel,
            "outcome": {"evaluated": evaluated, "actual_return": actual_return, "correct": correct, "px0": px0, "px1": px1},
            "total_time": f"{time.time() - start_time:.2f}s",
        }

    def _compile_one_month_report(self, symbol: str, price_result: Dict,
                                  market_data_result: Dict, historical_result: Dict,
                                  technical_result: Dict, indicators_result: Dict,
                                  news_result: Dict, analyst_recommendations_result: Dict,
                                  economic_calendar_result: Dict, prediction_analysis: Dict,
                                  conversion_info: Dict) -> str:
        """
        Compile clean, focused 1-month prediction report.
        """
        report = f"""
{'='*80}
1-MONTH STOCK PRICE PREDICTION: {symbol}
{'='*80}

INPUT CONVERSION:
- Original Input: {conversion_info.get('original_input', 'N/A')}
- Converted: {conversion_info.get('converted', False)}
- Input Type: {conversion_info.get('type', 'N/A')}
"""
        
        if conversion_info.get('converted'):
            report += f"- Company Name: {conversion_info.get('company_name', 'N/A')}\n"
            report += f"- Exchange: {conversion_info.get('exchange', 'N/A')}\n"
        
        report += f"""
ANALYSIS TYPE: Focused 1-Month Prediction
TOOLS USED: 8 (Price, Market Data, Historical Patterns, Technical Analysis, 
             Indicators, News, Analyst Views, Economic Context)

"""
        
        # Add current price information
        if price_result.get('status') == 'SUCCESS':
            price_data = price_result.get('data', {})
            if price_data.get('success'):
                actual_data = price_data.get('data', {})
                current = actual_data.get('current', {})
                info = actual_data.get('info', {})
                
                report += f"CURRENT SITUATION:\n"
                report += f"Symbol: {symbol}\n"
                report += f"Company: {info.get('description', 'N/A')}\n"
                report += f"Current Price: ${current.get('close', 'N/A')}\n"
                report += f"52-Week Range: ${market_data_result.get('data', {}).get('data', {}).get('indicators', {}).get('price_52_week_low', 'N/A')} - ${market_data_result.get('data', {}).get('data', {}).get('indicators', {}).get('price_52_week_high', 'N/A')}\n\n"
        
        # Add the AI prediction
        if prediction_analysis.get('status') == 'SUCCESS':
            report += f"1-MONTH PREDICTION:\n"
            report += f"{prediction_analysis.get('analysis')}\n\n"
        else:
            report += f"Prediction: {prediction_analysis.get('message', 'Not available')}\n\n"
        
        report += f"{'='*80}\n"
        report += f"Analysis Method: Historical Patterns + Key Factors\n"
        report += f"Timeframe: Next 30 Days\n"
        report += f"{'='*80}\n"
        
        return report

    def _generate_comprehensive_analysis(self, symbol: str, price_result: Dict,
                                        market_data_result: Dict, company_info_result: Dict,
                                        analyst_recommendations_result: Dict, historical_result: Dict,
                                        news_result: Dict, technical_result: Dict,
                                        indicators_result: Dict, community_result: Dict,
                                        economic_calendar_result: Dict, earnings_calendar_result: Dict,
                                        gainers_result: Dict, losers_result: Dict, active_result: Dict,
                                        gdp_result: Dict, interest_rates_result: Dict) -> Dict[str, Any]:
        """
        Use AI to generate comprehensive probability-based analysis from ALL data sources.
        
        Args:
            symbol: Stock symbol
            price_result: Current price data
            market_data_result: Comprehensive market data (15 sections)
            company_info_result: Company information
            analyst_recommendations_result: Analyst recommendations
            historical_result: Historical data
            news_result: News data
            technical_result: Technical analysis
            indicators_result: Technical indicators
            community_result: Community sentiment data
            economic_calendar_result: Economic calendar
            earnings_calendar_result: Earnings calendar
            gainers_result: Stock gainers leaderboard
            losers_result: Stock losers leaderboard
            active_result: Most active stocks
            gdp_result: GDP growth data
            interest_rates_result: Interest rates data
        
        Returns:
            dict: Comprehensive probability analysis with buy/sell/hold probabilities
        """
        if not self.model:
            return {
                "status": "ERROR",
                "message": "AI model not available"
            }
        
        try:
            # Prepare comprehensive data summary for AI with actual data from ALL sources
            data_summary = f"""
COMPREHENSIVE STOCK ANALYSIS FOR: {symbol}

GROUNDING & RELIABILITY RULES (MANDATORY):
- Do NOT infer missing data. If a data source is unavailable, say "Unavailable" and move on.
- Do NOT use words like "inferred", "assumed", "likely" for missing analyst/news/sentiment fields.
- Keep outputs compact and structured; avoid long essay paragraphs.
- If you cannot support a claim from the provided snapshot, omit the claim.

"""
            
            # Add price data if available
            if price_result.get('status') == 'SUCCESS':
                price_data = price_result.get('data', {})
                if price_data.get('success'):
                    actual_data = price_data.get('data', {})
                    current = actual_data.get('current', {})
                    info = actual_data.get('info', {})
                    data_summary += f"""
CURRENT PRICE DATA:
- Company: {info.get('description', 'N/A')}
- Current Price: ${current.get('close', 'N/A')}
- Open: ${current.get('open', 'N/A')}
- High: ${current.get('max', 'N/A')}
- Low: ${current.get('min', 'N/A')}
- Volume: {current.get('volume', 'N/A')}
- Exchange: {info.get('exchange', 'N/A')}
"""
                else:
                    data_summary += f"Current Price Data: Available but API returned error - {price_data.get('msg', 'Unknown error')}\n"
            else:
                data_summary += f"Current Price Data: {price_result.get('message', 'Not available')}\n"
            
            # Add comprehensive market data (NEW)
            if market_data_result.get('status') == 'SUCCESS':
                market_data = market_data_result.get('data', {})
                if market_data.get('success'):
                    inner_data = market_data.get('data', {})
                    data_summary += f"""
COMPREHENSIVE MARKET DATA (15 sections):
- Available sections: {list(inner_data.keys())}
"""
                    if 'company' in inner_data:
                        company = inner_data['company']
                        data_summary += f"""
  Company Details:
  - CEO: {company.get('ceo', 'N/A')}
  - Sector: {company.get('sector', 'N/A')}
  - Industry: {company.get('industry', 'N/A')}
  - Employees: {company.get('number_of_employees', 'N/A')}
  - Founded: {company.get('founded', 'N/A')}
  - Country: {company.get('country', 'N/A')}
"""
                    if 'indicators' in inner_data:
                        indicators = inner_data['indicators']
                        data_summary += f"""
  Key Indicators:
  - Market Cap: ${indicators.get('market_cap_calc', 0):,.0f}
  - P/E Ratio: {indicators.get('price_earnings', 'N/A')}
  - 52-Week High: ${indicators.get('price_52_week_high', 'N/A')}
  - 52-Week Low: ${indicators.get('price_52_week_low', 'N/A')}
  - Beta (1Y): {indicators.get('beta_1_year', 'N/A')}
"""
                    if 'ttm' in inner_data:
                        ttm = inner_data['ttm']
                        data_summary += f"""
  TTM Metrics:
  - EPS (TTM): ${ttm.get('earnings_per_share_ttm', 'N/A')}
  - Revenue (TTM): ${ttm.get('total_revenue_ttm', 0):,.0f}
  - Net Margin (TTM): {ttm.get('net_margin_ttm', 'N/A')}%
  - Gross Margin (TTM): {ttm.get('gross_margin_ttm', 'N/A')}%
"""
                else:
                    data_summary += f"Comprehensive Market Data: Available but API returned error - {market_data.get('msg', 'Unknown error')}\n"
            else:
                data_summary += f"Comprehensive Market Data: {market_data_result.get('message', 'Not available')}\n"
            
            # Add company information (NEW)
            if company_info_result.get('status') == 'SUCCESS':
                data_summary += "Company Information: Available\n"
            else:
                data_summary += f"Company Information: {company_info_result.get('message', 'Not available')}\n"
            
            # Add analyst recommendations (NEW)
            if analyst_recommendations_result.get('status') == 'SUCCESS':
                analyst_data = analyst_recommendations_result.get('data', {})
                if analyst_data.get('success'):
                    data_summary += "Analyst Recommendations: Available with ratings and price targets\n"
                else:
                    data_summary += f"Analyst Recommendations: Available but API returned error\n"
            else:
                data_summary += f"Analyst Recommendations: {analyst_recommendations_result.get('message', 'Not available')}\n"
            
            # Add historical data summary
            if historical_result.get('status') == 'SUCCESS':
                hist_data = historical_result.get('data', {})
                if hist_data.get('success'):
                    data_summary += "Historical Data (1Y): Available with price history\n"
                else:
                    data_summary += f"Historical Data (1Y): Available but API returned error - {hist_data.get('msg', 'Unknown error')}\n"
            else:
                data_summary += f"Historical Data (1Y): {historical_result.get('message', 'Not available')}\n"
            
            # Add news summary
            if news_result.get('status') == 'SUCCESS':
                news_data = news_result.get('data', {})
                if isinstance(news_data, list):
                    data_summary += f"News Data: Available with {len(news_data)} recent news articles\n"
                elif isinstance(news_data, dict) and news_data.get('success'):
                    data_summary += "News Data: Available with recent news articles\n"
                else:
                    data_summary += "News Data: Available\n"
            else:
                data_summary += f"News Data: {news_result.get('message', 'Not available')}\n"
            
            # Add technical analysis summary
            if technical_result.get('status') == 'SUCCESS':
                tech_data = technical_result.get('data', {})
                if tech_data.get('success'):
                    data_summary += "Technical Analysis: Available with signals and indicators\n"
                else:
                    data_summary += f"Technical Analysis: Available but API returned error - {tech_data.get('msg', 'Unknown error')}\n"
            else:
                data_summary += f"Technical Analysis: {technical_result.get('message', 'Not available')}\n"
            
            # Add indicators summary
            if indicators_result.get('status') == 'SUCCESS':
                ind_data = indicators_result.get('data', {})
                if ind_data.get('success'):
                    data_summary += "Technical Indicators: Available with detailed indicator values\n"
                else:
                    data_summary += f"Technical Indicators: Available but API returned error - {ind_data.get('msg', 'Unknown error')}\n"
            else:
                data_summary += f"Technical Indicators: {indicators_result.get('message', 'Not available')}\n"
            
            # Add community sentiment (NEW)
            if community_result.get('status') == 'SUCCESS':
                data_summary += "Community Sentiment Data: Available with social sentiment indicators\n"
            else:
                data_summary += f"Community Sentiment Data: {community_result.get('message', 'Not available')}\n"
            
            # Add economic calendar (NEW)
            if economic_calendar_result.get('status') == 'SUCCESS':
                econ_data = economic_calendar_result.get('data', {})
                if econ_data.get('success'):
                    events = econ_data.get('data', [])
                    data_summary += f"Economic Calendar: Available with {len(events) if isinstance(events, list) else 0} upcoming economic events\n"
                else:
                    data_summary += f"Economic Calendar: Available but API returned error\n"
            else:
                data_summary += f"Economic Calendar: {economic_calendar_result.get('message', 'Not available')}\n"
            
            # Add earnings calendar (NEW)
            if earnings_calendar_result.get('status') == 'SUCCESS':
                earnings_data = earnings_calendar_result.get('data', {})
                if earnings_data.get('success'):
                    events = earnings_data.get('data', [])
                    data_summary += f"Earnings Calendar: Available with {len(events) if isinstance(events, list) else 0} upcoming earnings events\n"
                else:
                    data_summary += f"Earnings Calendar: Available but API returned error\n"
            else:
                data_summary += f"Earnings Calendar: {earnings_calendar_result.get('message', 'Not available')}\n"
            
            # Add market leaderboards (NEW)
            if all([gainers_result.get('status') == 'SUCCESS', losers_result.get('status') == 'SUCCESS', active_result.get('status') == 'SUCCESS']):
                data_summary += "Market Leaderboards: Available with gainers, losers, and most active stocks for market comparison\n"
            else:
                data_summary += "Market Leaderboards: Partially available or not available\n"
            
            # Add world economy indicators (NEW)
            if all([gdp_result.get('status') == 'SUCCESS', interest_rates_result.get('status') == 'SUCCESS']):
                data_summary += "World Economy Indicators: Available with GDP growth and interest rates for macro context\n"
            else:
                data_summary += "World Economy Indicators: Partially available or not available\n"
            
            prompt = f"""
            Analyze the following COMPREHENSIVE stock data from multiple sources and provide a probability assessment:
            
            {data_summary}
            
            Based on ALL the available data sources (price data, comprehensive market data, company info, analyst recommendations, 
            historical data, news, technical analysis, community sentiment, economic calendar, earnings calendar, market leaderboards, 
            and world economy indicators), provide:
            
            1. Probability of price increase (0-100%)
            2. Probability of price decrease (0-100%)
            3. Probability of price staying stable (0-100%)
            4. Overall recommendation (BUY/SELL/HOLD)
            5. Confidence level (0-100%)
            6. Key factors influencing your decision (consider all data sources)
            7. Risk assessment (LOW/MEDIUM/HIGH)
            8. Time horizon for the recommendation
            9. How market context and economic indicators affect this stock
            10. How this stock compares to market leaders (gainers/losers)
            
            Format your response as a structured analysis with clear percentages and reasoning that integrates insights from 
            all available data sources for a comprehensive investment decision.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "status": "SUCCESS",
                "analysis": response.text,
                "raw_data": {
                    "price": {
                        "status": price_result.get("status"),
                        "success": price_result.get("data", {}).get("success", False),
                        "has_data": price_result.get("data", {}).get("data") is not None
                    },
                    "market_data": {
                        "status": market_data_result.get("status"),
                        "success": market_data_result.get("data", {}).get("success", False),
                        "has_data": market_data_result.get("data", {}).get("data") is not None
                    },
                    "company_info": {
                        "status": company_info_result.get("status"),
                        "success": company_info_result.get("data", {}).get("success", False),
                        "has_data": company_info_result.get("data", {}).get("data") is not None
                    },
                    "analyst_recommendations": {
                        "status": analyst_recommendations_result.get("status"),
                        "success": analyst_recommendations_result.get("data", {}).get("success", False),
                        "has_data": analyst_recommendations_result.get("data", {}).get("data") is not None
                    },
                    "historical": {
                        "status": historical_result.get("status"),
                        "success": historical_result.get("data", {}).get("success", False),
                        "has_data": historical_result.get("data", {}).get("data") is not None
                    },
                    "news": {
                        "status": news_result.get("status"),
                        "success": news_result.get("data", {}).get("success", False),
                        "has_data": news_result.get("data", {}).get("data") is not None
                    },
                    "technical": {
                        "status": technical_result.get("status"),
                        "success": technical_result.get("data", {}).get("success", False),
                        "has_data": technical_result.get("data", {}).get("data") is not None
                    },
                    "indicators": {
                        "status": indicators_result.get("status"),
                        "success": indicators_result.get("data", {}).get("success", False),
                        "has_data": indicators_result.get("data", {}).get("data") is not None
                    },
                    "community": {
                        "status": community_result.get("status"),
                        "success": community_result.get("data", {}).get("success", False),
                        "has_data": community_result.get("data", {}).get("data") is not None
                    },
                    "economic_calendar": {
                        "status": economic_calendar_result.get("status"),
                        "success": economic_calendar_result.get("data", {}).get("success", False),
                        "has_data": economic_calendar_result.get("data", {}).get("data") is not None
                    },
                    "earnings_calendar": {
                        "status": earnings_calendar_result.get("status"),
                        "success": earnings_calendar_result.get("data", {}).get("success", False),
                        "has_data": earnings_calendar_result.get("data", {}).get("data") is not None
                    },
                    "market_leaderboards": {
                        "status": "SUCCESS" if all([
                            gainers_result.get("status") == "SUCCESS",
                            losers_result.get("status") == "SUCCESS",
                            active_result.get("status") == "SUCCESS"
                        ]) else "PARTIAL",
                        "gainers": gainers_result.get("status"),
                        "losers": losers_result.get("status"),
                        "active": active_result.get("status")
                    },
                    "world_economy": {
                        "status": "SUCCESS" if all([
                            gdp_result.get("status") == "SUCCESS",
                            interest_rates_result.get("status") == "SUCCESS"
                        ]) else "PARTIAL",
                        "gdp": gdp_result.get("status"),
                        "interest_rates": interest_rates_result.get("status")
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error generating probability analysis: {str(e)}"
            }
    
    def _compile_comprehensive_report(self, symbol: str, price_result: Dict,
                                     market_data_result: Dict, company_info_result: Dict,
                                     analyst_recommendations_result: Dict, historical_result: Dict,
                                     news_result: Dict, technical_result: Dict,
                                     indicators_result: Dict, community_result: Dict,
                                     economic_calendar_result: Dict, earnings_calendar_result: Dict,
                                     gainers_result: Dict, losers_result: Dict, active_result: Dict,
                                     gdp_result: Dict, interest_rates_result: Dict,
                                     probability_analysis: Dict) -> str:
        """
        Compile ALL data into a comprehensive report.
        
        Args:
            symbol: Stock symbol
            price_result: Current price data
            market_data_result: Comprehensive market data (15 sections)
            company_info_result: Company information
            analyst_recommendations_result: Analyst recommendations
            historical_result: Historical data
            news_result: News data
            technical_result: Technical analysis
            indicators_result: Technical indicators
            community_result: Community sentiment data
            economic_calendar_result: Economic calendar
            earnings_calendar_result: Earnings calendar
            gainers_result: Stock gainers leaderboard
            losers_result: Stock losers leaderboard
            active_result: Most active stocks
            gdp_result: GDP growth data
            interest_rates_result: Interest rates data
            probability_analysis: AI-generated probability analysis
        
        Returns:
            str: Comprehensive report
        """
        report = f"""
{'='*80}
COMPREHENSIVE STOCK ANALYSIS REPORT: {symbol}
{'='*80}

DATA SOURCES STATUS (13 Tools Used):
- Current Price: {price_result.get('status')}
- 1-Year Historical Data: {historical_result.get('status')}
- Current News: {news_result.get('status')}
- Technical Analysis: {technical_result.get('status')}
- Technical Indicators: {indicators_result.get('status')}

"""
        
        # Add current price information if available
        if price_result.get('status') == 'SUCCESS':
            price_data = price_result.get('data', {})
            if price_data.get('success'):
                actual_data = price_data.get('data', {})
                current = actual_data.get('current', {})
                info = actual_data.get('info', {})
                
                report += f"CURRENT PRICE INFORMATION:\n"
                report += f"Symbol: {symbol}\n"
                report += f"Formatted Symbol: {price_result.get('formatted_symbol')}\n"
                report += f"Company: {info.get('description', 'N/A')}\n"
                report += f"Current Price: ${current.get('close', 'N/A')}\n"
                report += f"Open: ${current.get('open', 'N/A')}\n"
                report += f"High: ${current.get('max', 'N/A')}\n"
                report += f"Low: ${current.get('min', 'N/A')}\n"
                report += f"Volume: {current.get('volume', 'N/A')}\n"
                report += f"Exchange: {info.get('exchange', 'N/A')}\n\n"
            else:
                report += f"CURRENT PRICE INFORMATION:\n"
                report += f"Symbol: {symbol}\n"
                report += f"Error: {price_data.get('msg', 'Data not available')}\n\n"
        
        # Add probability analysis
        if probability_analysis.get('status') == 'SUCCESS':
            report += f"PROBABILITY ANALYSIS:\n"
            report += f"{probability_analysis.get('analysis')}\n\n"
        else:
            report += f"Probability Analysis: {probability_analysis.get('message', 'Not available')}\n\n"
        
        # Add technical analysis summary
        if technical_result.get('status') == 'SUCCESS':
            tech_data = technical_result.get('data', {})
            if tech_data.get('success'):
                actual_tech_data = tech_data.get('data', {})
                report += f"TECHNICAL ANALYSIS SUMMARY:\n"
                report += f"Status: Data available\n"
                if isinstance(actual_tech_data, dict):
                    # Add some key technical analysis fields if available
                    if 'recommendation' in actual_tech_data:
                        report += f"Recommendation: {actual_tech_data.get('recommendation', 'N/A')}\n"
                    if 'signal' in actual_tech_data:
                        report += f"Signal: {actual_tech_data.get('signal', 'N/A')}\n"
                    report += f"Data keys: {list(actual_tech_data.keys())}\n"
                report += "\n"
            else:
                report += f"TECHNICAL ANALYSIS SUMMARY:\n"
                report += f"Error: {tech_data.get('msg', 'Data not available')}\n\n"
        
        # Add news summary
        if news_result.get('status') == 'SUCCESS':
            news_data = news_result.get('data', {})
            report += f"NEWS SUMMARY:\n"
            if isinstance(news_data, list):
                report += f"Status: Data available\n"
                report += f"Number of articles: {len(news_data)}\n"
            elif isinstance(news_data, dict) and news_data.get('success'):
                actual_news_data = news_data.get('data', {})
                report += f"Status: Data available\n"
                if isinstance(actual_news_data, list):
                    report += f"Number of articles: {len(actual_news_data)}\n"
                elif isinstance(actual_news_data, dict):
                    report += f"Data keys: {list(actual_news_data.keys())}\n"
            else:
                report += f"Status: Available\n"
            report += "\n"
        
        report += f"{'='*80}\n"
        report += f"Report generated by Stock Analysis Agent\n"
        report += f"{'='*80}\n"
        
        return report
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query and extract stock symbol for analysis.
        
        Args:
            query: User's query (e.g., "Analyze AAPL" or "What about TSLA?")
        
        Returns:
            dict: Analysis result
        """
        # Extract stock symbol from query
        symbol = self._extract_symbol(query)
        
        if not symbol:
            return {
                "status": "ERROR",
                "message": "Please provide a stock symbol (e.g., 'Analyze AAPL' or 'What about TSLA?')"
            }
        
        return self.analyze_stock(symbol)
    
    def _extract_symbol(self, query: str) -> str:
        """
        Extract stock symbol from user query.
        
        Args:
            query: User's query text
        
        Returns:
            str: Extracted stock symbol or None
        """
        # Simple extraction - look for uppercase stock symbols (1-5 letters)
        import re
        matches = re.findall(r'\b[A-Z]{1,5}\b', query)
        
        # Filter out common words
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HAVE', 'BEEN', 'WILL', 'WITH', 'THIS', 'THAT', 'FROM', 'WHAT', 'WHEN', 'ABOUT'}
        symbols = [match for match in matches if match not in common_words]
        
        if symbols:
            return symbols[0]  # Return first potential symbol
        
        return None


if __name__ == "__main__":
    # Test the agent
    print("Testing Stock Analysis Agent...")
    print("Choose analysis type:")
    print("1. Comprehensive Analysis (13 tools)")
    print("2. 1-Month Prediction (8 tools - focused)")
    print("3. Investment Scenario Analysis (100-day with $10K example)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    agent = StockAnalysisAgent()
    
    if choice == "3":
        print("\nRunning Investment Scenario Analysis...")
        investment = input("Enter investment amount (default $10,000): ").strip()
        days = input("Enter timeframe in days (default 100): ").strip()
        
        try:
            investment_amount = float(investment) if investment else 10000
            timeframe = int(days) if days else 100
        except:
            investment_amount = 10000
            timeframe = 100
        
        result = agent.analyze_investment_scenario("AAPL", investment_amount, timeframe)
    elif choice == "2":
        print("\nRunning 1-Month Prediction Analysis...")
        result = agent.analyze_stock_one_month("AAPL")
    else:
        print("\nRunning Comprehensive Analysis...")
        result = agent.analyze_stock("AAPL")
    
    print(f"\nStatus: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(result['report'])
    else:
        print(f"Error: {result.get('message')}")
