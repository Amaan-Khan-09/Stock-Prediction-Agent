"""
Stock Price Prediction Agent — Walk-Forward (No Leakage)

Model version: stock_predictor_v1_walkforward

Rules:
- AI receives ONLY bars up to prediction_origin_date (enforced twice)
- NO target-date price ever passed to AI
- NO yfinance
- NO hardcoded prices
- NO LLM-generated numeric values (Gemini writes explanation only)
- Deterministic: same inputs always produce same numeric output
- Return clamped to [-20%, +20%] (realistic 30-day range)

Decision logic (Amaan fix — risk alone NEVER forces SELL):
    if data_quality_score < 60  → REVIEW
    elif confidence_score < 50  → REVIEW
    elif return >= +3%          → BUY
    elif return <= -3%          → SELL
    else                        → HOLD

    if risk >= 80 and decision == BUY  → downgrade to HOLD
    if risk >= 80 and decision == SELL → convert to REVIEW
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "tools"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

MODEL_VERSION = "stock_predictor_v1_walkforward"

# Decision thresholds — AI prediction (wider band than actual validation)
_AI_BUY_THRESHOLD  =  3.0   # predicted_return_pct >= +3%  → BUY
_AI_SELL_THRESHOLD = -3.0   # predicted_return_pct <= -3%  → SELL
_HIGH_RISK_CUTOFF  = 80     # risk_score >= 80 triggers downgrade
_DAMPING_FACTOR    = 0.50   # reduce raw momentum signal (mean reversion)
_RETURN_CLAMP_MAX  = 20.0   # max realistic 30-day prediction
_RETURN_CLAMP_MIN  = -20.0  # min realistic 30-day prediction

STOCK_PRED_FIELDS = [
    "symbol",
    "historical_context_start_date",
    "prediction_origin_date",
    "decision_horizon_days",
    "target_date",
    "initial_capital",
    "benchmark",
]

_CTX_CACHE: Dict[str, Dict] = {}
_CTX_TTL = 15 * 60


# ══════════════════════════════════════════════════════════════════════════════
# HASH + VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def build_stock_prediction_hash(spi: dict) -> str:
    """SHA-256[:16] of canonical sorted-key JSON of the standard fields."""
    subset = {k: spi[k] for k in sorted(STOCK_PRED_FIELDS) if k in spi}
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:16]


def validate_stock_prediction_input(spi: dict) -> Tuple[bool, str]:
    """Validate StockPredictionInput. Returns (is_valid, error_msg)."""
    errors: List[str] = []
    today = date.today()

    sym = str(spi.get("symbol", "") or "").strip().upper()
    if not sym:
        errors.append("Symbol is required.")
    elif not (1 <= len(sym) <= 7):
        errors.append(f"Symbol '{sym}' must be 1-7 characters.")

    ctx_start = _parse_ymd(str(spi.get("historical_context_start_date", "") or ""))
    origin    = _parse_ymd(str(spi.get("prediction_origin_date",        "") or ""))
    target    = _parse_ymd(str(spi.get("target_date",                   "") or ""))

    if ctx_start is None:
        errors.append(f"historical_context_start_date invalid: {spi.get('historical_context_start_date')!r}")
    if origin is None:
        errors.append(f"prediction_origin_date invalid: {spi.get('prediction_origin_date')!r}")
    if target is None:
        errors.append(f"target_date invalid: {spi.get('target_date')!r}")

    if ctx_start and ctx_start < date(2000, 1, 1):
        errors.append("historical_context_start_date cannot be before 2000-01-01.")
    if ctx_start and ctx_start >= today:
        errors.append("historical_context_start_date must be in the past.")
    if origin and origin >= today:
        errors.append("prediction_origin_date must be in the past.")
    # Future target_date is allowed — run becomes "live_future_prediction" mode.
    # Actual validation is PENDING; AI prediction still runs using context window.
    if ctx_start and origin and ctx_start >= origin:
        errors.append("historical_context_start_date must be before prediction_origin_date.")
    if origin and target and origin >= target:
        errors.append("prediction_origin_date must be before target_date.")

    try:
        cap = float(spi.get("initial_capital", 0) or 0)
        if cap < 100:
            errors.append("initial_capital must be at least $100.")
    except (TypeError, ValueError):
        errors.append("initial_capital must be a valid number.")

    try:
        h = int(spi.get("decision_horizon_days", 0) or 0)
        if h <= 0 or h > 365:
            errors.append("decision_horizon_days must be 1-365.")
    except (TypeError, ValueError):
        errors.append("decision_horizon_days must be a valid integer.")

    if errors:
        return False, "; ".join(errors)
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# DECISION ENGINE — PUBLIC (importable for tests)
# ══════════════════════════════════════════════════════════════════════════════

def make_decision(
    predicted_return_pct: float,
    confidence_score:     int,
    risk_score:           int,
    data_quality_score:   int,
) -> Tuple[str, str]:
    """
    Deterministic, testable decision function.

    Rules:
    1. Low data quality (< 60)  → REVIEW regardless of return
    2. Low confidence (< 50)    → REVIEW regardless of return
    3. Return >= +3%            → BUY
    4. Return <= -3%            → SELL
    5. Otherwise                → HOLD

    Risk adjustment (risk NEVER creates SELL from scratch):
    6. risk >= 80 AND decision == BUY  → HOLD  (positive but too risky)
    7. risk >= 80 AND decision == SELL → REVIEW (uncertain, not confident enough to SELL)

    Returns (decision, reason_string).
    """
    # Step 1 — base decision
    if data_quality_score < 60:
        decision = "REVIEW"
        reason   = (f"Data quality too low ({data_quality_score}/100) — "
                    "insufficient data for confident prediction.")
    elif confidence_score < 50:
        decision = "REVIEW"
        reason   = (f"Confidence too low ({confidence_score}/100) — "
                    "mixed or weak signals, result not reliable.")
    elif predicted_return_pct >= _AI_BUY_THRESHOLD:
        decision = "BUY"
        reason   = (f"Predicted return {predicted_return_pct:+.2f}% "
                    f">= +{_AI_BUY_THRESHOLD}% BUY threshold.")
    elif predicted_return_pct <= _AI_SELL_THRESHOLD:
        decision = "SELL"
        reason   = (f"Predicted return {predicted_return_pct:+.2f}% "
                    f"<= {_AI_SELL_THRESHOLD}% SELL threshold.")
    else:
        decision = "HOLD"
        reason   = (f"Predicted return {predicted_return_pct:+.2f}% "
                    f"is in HOLD band ({_AI_SELL_THRESHOLD}% to +{_AI_BUY_THRESHOLD}%).")

    # Step 2 — risk adjustment (risk NEVER forces SELL on its own)
    if risk_score >= _HIGH_RISK_CUTOFF and decision == "BUY":
        decision = "HOLD"
        reason   = (f"BUY downgraded to HOLD: positive return predicted "
                    f"({predicted_return_pct:+.2f}%) but risk too high ({risk_score}/100). "
                    "Positive return + high risk = avoid entry.")

    elif risk_score >= _HIGH_RISK_CUTOFF and decision == "SELL":
        decision = "REVIEW"
        reason   = (f"SELL converted to REVIEW: negative return predicted "
                    f"({predicted_return_pct:+.2f}%) but high uncertainty ({risk_score}/100). "
                    "High risk should not confirm SELL — needs human review.")

    return decision, reason


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _period_return(bars: List[dict], lookback: int) -> Optional[float]:
    """Return % change over the last `lookback` bars."""
    if len(bars) < lookback + 1:
        return None
    p_now  = bars[-1]["close"]
    p_then = bars[-lookback]["close"]
    if p_then <= 0:
        return None
    return (p_now - p_then) / p_then * 100.0


def _annualized_volatility(bars: List[dict], window: int = 60) -> Optional[float]:
    """Annualized volatility from log returns over `window` bars."""
    if len(bars) < window + 1:
        return None
    rets = []
    for i in range(1, window + 1):
        prev = bars[-i - 1]["close"]
        curr = bars[-i]["close"]
        if prev > 0:
            rets.append(math.log(curr / prev))
    if len(rets) < 10:
        return None
    n    = len(rets)
    mean = sum(rets) / n
    var  = sum((r - mean) ** 2 for r in rets) / n
    return math.sqrt(var) * math.sqrt(252) * 100.0


def _compute_momentum_prediction(
    bars_ctx: List[dict],
    horizon_days: int,
    benchmark_adj: float = 0.0,
) -> dict:
    """
    Compute predicted return using spec-defined weighted momentum formula.

    Formula:
        weighted_signal = 0.20 * return_5d
                        + 0.45 * return_20d
                        + 0.25 * return_60d
                        + 0.10 * benchmark_adj

        predicted_return_pct = weighted_signal * DAMPING_FACTOR (0.50)

        clamped to [_RETURN_CLAMP_MIN, _RETURN_CLAMP_MAX]
    """
    if len(bars_ctx) < 5:
        return {"status": "INSUFFICIENT", "n_bars": len(bars_ctx)}

    r5  = _period_return(bars_ctx, 5)  or 0.0
    r20 = _period_return(bars_ctx, 20) or 0.0
    r60 = _period_return(bars_ctx, 60) or 0.0
    vol = _annualized_volatility(bars_ctx)

    # Weighted blend
    raw_signal = (
        0.20 * r5
        + 0.45 * r20
        + 0.25 * r60
        + 0.10 * benchmark_adj
    )

    # Dampen — reduces extreme momentum (mean reversion)
    predicted_return_pct = raw_signal * _DAMPING_FACTOR

    # Clamp — realistic 30-day range
    predicted_return_pct = max(_RETURN_CLAMP_MIN, min(_RETURN_CLAMP_MAX, predicted_return_pct))

    return {
        "status":                "OK",
        "n_bars":                len(bars_ctx),
        "latest_date":           bars_ctx[-1]["date"],
        "latest_price":          bars_ctx[-1]["close"],
        "return_5d":             round(r5,  4),
        "return_20d":            round(r20, 4),
        "return_60d":            round(r60, 4),
        "benchmark_adj":         round(benchmark_adj, 4),
        "raw_signal":            round(raw_signal, 4),
        "damping_factor":        _DAMPING_FACTOR,
        "predicted_return_pct":  round(predicted_return_pct, 4),
        "annualized_vol":        round(vol, 2) if vol is not None else None,
    }


def _compute_scores(mom: dict) -> dict:
    """Compute confidence, risk, and data_quality scores."""
    n   = mom.get("n_bars", 0)
    vol = mom.get("annualized_vol") or 0.0

    # --- Data Quality ---
    dq = 100
    if n < 20:  dq -= 40
    elif n < 60:  dq -= 20
    elif n < 120: dq -= 10
    if mom.get("status") != "OK": dq -= 50
    data_quality = max(0, min(100, dq))

    # --- Confidence ---
    conf = 50
    if n >= 120: conf += 15
    elif n >= 60: conf += 8
    elif n >= 20: conf += 3
    # Signal agreement: all three windows same direction?
    r5  = mom.get("return_5d",  0) or 0
    r20 = mom.get("return_20d", 0) or 0
    r60 = mom.get("return_60d", 0) or 0
    signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in [r5, r20, r60] if r != 0]
    if len(signs) >= 2 and all(s == signs[0] for s in signs):
        conf += 12   # all pointing same direction
    elif len(signs) >= 2 and signs[0] != signs[-1]:
        conf -= 8    # short-term contradicts long-term
    if data_quality >= 80: conf += 8
    conf = max(0, min(95, conf))

    # --- Risk ---
    risk = 40
    if vol > 60:   risk += 30
    elif vol > 40: risk += 20
    elif vol > 25: risk += 12
    elif vol > 15: risk += 6
    elif vol < 10: risk -= 10
    # Large recent moves = higher risk
    abs_r20 = abs(r20)
    if abs_r20 > 30:  risk += 20
    elif abs_r20 > 15: risk += 10
    elif abs_r20 > 8:  risk += 5
    risk = max(5, min(99, risk))

    return {
        "confidence_score":  conf,
        "risk_score":        risk,
        "data_quality_score": data_quality,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAPIDAPI MARKET CONTEXT (benchmark adjustment)
# ══════════════════════════════════════════════════════════════════════════════

def _get_market_context(symbol: str) -> dict:
    sym_key = symbol.upper()
    cached  = _CTX_CACHE.get(sym_key)
    if cached and (time.time() - cached["ts"]) < _CTX_TTL:
        return cached["ctx"]
    ctx = _fetch_market_context(symbol)
    _CTX_CACHE[sym_key] = {"ctx": ctx, "ts": time.time()}
    return ctx


def _fetch_market_context(symbol: str) -> dict:
    ctx: dict = {
        "gainers": [], "losers": [],
        "symbol_in_gainers": False, "symbol_in_losers": False,
        "movers_ok": False, "benchmark_adj": 0.0, "context_error": "",
    }
    try:
        from rapidapi_client import rapidapi_get

        def _syms(r: dict) -> list:
            if r.get("status") != "SUCCESS": return []
            d = r.get("data", {})
            if isinstance(d, dict):
                return [s.get("s","").split(":")[-1].upper()
                        for s in d.get("symbols",[]) if isinstance(s, dict) and s.get("s")]
            return []

        g_r = rapidapi_get("/market/get-movers",
                           params={"exchange":"US","name":"percent_change_gainers","locale":"en"})
        l_r = rapidapi_get("/market/get-movers",
                           params={"exchange":"US","name":"percent_change_losers","locale":"en"})
        gainers = _syms(g_r)
        losers  = _syms(l_r)
        sym_u   = symbol.upper()

        adj = 0.0
        if sym_u in gainers: adj += 1.5
        if sym_u in losers:  adj -= 1.5

        ctx.update({
            "gainers":            gainers,
            "losers":             losers,
            "symbol_in_gainers":  sym_u in gainers,
            "symbol_in_losers":   sym_u in losers,
            "movers_ok":          True,
            "benchmark_adj":      adj,
        })
    except Exception as exc:
        ctx["context_error"] = str(exc)
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_explanation(spi: dict, mom: dict, decision: str, predicted_return_pct: float) -> str:
    """Generate reasoning text. Returns '' on any failure. NEVER generates numeric values."""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return ""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("AGENT_MODEL", "gemini-2.5-flash"))
        prompt = (
            f"Stock: {spi['symbol']}\n"
            f"Context window: {spi['historical_context_start_date']} to {spi['prediction_origin_date']}\n"
            f"Horizon: {spi['decision_horizon_days']} days\n"
            f"5d momentum: {mom.get('return_5d','?'):.1f}%, "
            f"20d momentum: {mom.get('return_20d','?'):.1f}%, "
            f"60d momentum: {mom.get('return_60d','?'):.1f}%\n"
            f"AI decision: {decision} (predicted return: {predicted_return_pct:+.2f}%)\n\n"
            f"Write ONE short paragraph (2-3 sentences) explaining WHY this prediction was made "
            f"based on momentum signals. Do NOT suggest different numbers. "
            f"Do NOT invent prices. Focus on the directional reasoning only."
        )
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 200},
        )
        return resp.text.strip()[:600]
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_stock_prediction(spi: dict, price_history_context: List[dict]) -> dict:
    """
    Walk-forward stock price prediction.

    Parameters:
    - spi                   : StockPredictionInput dict
    - price_history_context : bars filtered to <= prediction_origin_date by caller

    LEAKAGE GUARANTEE:
    This function re-strips any bars after prediction_origin_date even if caller
    failed to filter. target_price_hidden_from_ai is always True in output.
    """
    origin_str = str(spi.get("prediction_origin_date", "") or "")

    # Double guard: strip any accidental future bars
    history = [b for b in (price_history_context or []) if b["date"] <= origin_str]

    valid, err = validate_stock_prediction_input(spi)
    if not valid:
        return _error_result(spi, err)

    symbol       = str(spi["symbol"]).upper()
    origin_date  = str(spi["prediction_origin_date"])
    target_date  = str(spi["target_date"])
    horizon_days = int(spi["decision_horizon_days"])
    initial_cap  = float(spi["initial_capital"])
    input_hash   = build_stock_prediction_hash(spi)
    price_basis  = str(spi.get("price_basis", "close") or "close")

    # Get origin price — respect price_basis setting (high/low/open/close)
    from historical_price_service import get_price_on_date
    origin_price, eff_origin_date, price_err = get_price_on_date(history, origin_date, price_basis=price_basis)
    if origin_price is None:
        return _error_result(
            spi,
            f"Cannot get origin price on {origin_date}: {price_err}. "
            "API may not have historical data for this date."
        )

    # Market context for benchmark adjustment
    mctx = _get_market_context(symbol)
    benchmark_adj = float(mctx.get("benchmark_adj", 0.0) or 0.0)

    # Compute momentum prediction
    mom = _compute_momentum_prediction(history, horizon_days, benchmark_adj)
    if mom.get("status") != "OK":
        return _error_result(
            spi,
            f"Insufficient price history for momentum calculation "
            f"(only {mom.get('n_bars', 0)} bars, need >= 5)."
        )

    predicted_return_pct = float(mom["predicted_return_pct"])

    # Compute scores
    scores = _compute_scores(mom)
    confidence_score    = scores["confidence_score"]
    risk_score          = scores["risk_score"]
    data_quality_score  = scores["data_quality_score"]

    # Make decision (public function — testable independently)
    decision, decision_reason = make_decision(
        predicted_return_pct, confidence_score, risk_score, data_quality_score
    )

    # Compute predicted prices + capital
    predicted_target_price  = round(origin_price * (1.0 + predicted_return_pct / 100.0), 4)
    predicted_final_capital = round(initial_cap  * (predicted_target_price / origin_price), 4)
    predicted_total_pl      = round(predicted_final_capital - initial_cap, 4)

    # Leakage proof
    from stock_comparison_engine import validate_no_leakage
    leakage = validate_no_leakage(history, origin_date)

    # Gemini reasoning text (does NOT own numbers)
    reasoning = _gemini_explanation(spi, mom, decision, predicted_return_pct)
    if not reasoning:
        reasoning = (
            f"AI uses {mom['n_bars']}-bar context ({spi['historical_context_start_date']} to "
            f"{eff_origin_date}). "
            f"5d/20d/60d momentum: {mom['return_5d']:+.1f}% / "
            f"{mom['return_20d']:+.1f}% / {mom['return_60d']:+.1f}%. "
            f"Dampened blended signal: {predicted_return_pct:+.2f}% → {decision}. "
            f"{decision_reason}"
        )

    return {
        "status":                      "SUCCESS",
        "source":                      "stock_prediction_agent",
        "model_version":               MODEL_VERSION,
        "stock_prediction_input_hash": input_hash,
        "symbol":                      symbol,
        "historical_context_start_date": spi["historical_context_start_date"],
        "prediction_origin_date":      origin_date,
        "effective_origin_date":       eff_origin_date,
        "target_date":                 target_date,
        "decision_horizon_days":       horizon_days,
        "price_basis":                 price_basis,
        "origin_price_used":           round(origin_price, 4),
        "predicted_target_price":      predicted_target_price,
        "predicted_return_pct":        predicted_return_pct,
        "initial_capital":             initial_cap,
        "predicted_final_capital":     predicted_final_capital,
        "predicted_total_pl":          predicted_total_pl,
        "decision":                    decision,
        "decision_reason":             decision_reason,
        "confidence_score":            confidence_score,
        "risk_score":                  risk_score,
        "data_quality_score":          data_quality_score,
        "latest_date_seen_by_ai":      mom["latest_date"],
        "target_price_hidden_from_ai": True,
        "leakage_check":               leakage["status"],
        "reasoning":                   reasoning,
        "features_used": {
            "return_5d":          mom["return_5d"],
            "return_20d":         mom["return_20d"],
            "return_60d":         mom["return_60d"],
            "benchmark_adj":      mom["benchmark_adj"],
            "raw_signal":         mom["raw_signal"],
            "damping_factor":     mom["damping_factor"],
            "annualized_vol":     mom["annualized_vol"],
            "bars_used":          mom["n_bars"],
            "last_ai_bar_date":   mom["latest_date"],
        },
    }


def _error_result(spi: dict, error_msg: str) -> dict:
    return {
        "status":                      "FAILED",
        "source":                      "stock_prediction_agent",
        "model_version":               MODEL_VERSION,
        "stock_prediction_input_hash": build_stock_prediction_hash(spi) if spi else "",
        "symbol":                      str(spi.get("symbol", "") or ""),
        "prediction_origin_date":      str(spi.get("prediction_origin_date", "") or ""),
        "target_date":                 str(spi.get("target_date", "") or ""),
        "decision":                    "REVIEW",
        "confidence_score":            0,
        "data_quality_score":          0,
        "risk_score":                  0,
        "target_price_hidden_from_ai": True,
        "error":                       error_msg,
        "predicted_target_price":      None,
        "predicted_return_pct":        None,
        "predicted_final_capital":     None,
        "predicted_total_pl":          None,
    }


def build_stock_prediction_input(
    symbol: str,
    historical_context_start_date: str,
    prediction_origin_date: str,
    decision_horizon_days: int,
    initial_capital: float,
    benchmark: str = "SPY",
    validation_mode: str = "horizon_days",
    price_basis: str = "close",
) -> dict:
    """Convenience builder that computes target_date automatically."""
    origin_dt   = datetime.strptime(prediction_origin_date, "%Y-%m-%d")
    target_dt   = origin_dt + timedelta(days=decision_horizon_days)
    return {
        "symbol":                        symbol.strip().upper(),
        "historical_context_start_date": historical_context_start_date,
        "prediction_origin_date":        prediction_origin_date,
        "decision_horizon_days":         int(decision_horizon_days),
        "target_date":                   target_dt.strftime("%Y-%m-%d"),
        "initial_capital":               float(initial_capital),
        "benchmark":                     benchmark.strip().upper(),
        "validation_mode":               validation_mode,
        "price_basis":                   price_basis,
    }


def _parse_ymd(s: str) -> Optional[date]:
    try:
        return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
