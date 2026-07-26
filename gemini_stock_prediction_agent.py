"""
Gemini Stock Prediction Agent — Walk-Forward (No Leakage)

Uses Google Gemini as the AI reasoning engine.
- Builds rich feature packet: RSI, MACD, Bollinger Bands, ATR, SMA, volume
- Enforces strict leakage prevention (no target price in Gemini context)
- Applies calibration from past validated records
- Validates and recalculates Gemini numeric output
- Applies deterministic decision guardrails

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY  — required for Gemini
    GEMINI_MODEL                      — default: gemini-2.5-flash
    AI_PROVIDER                       — "gemini" or "baseline" (default: gemini)
    ALLOW_BASELINE_FALLBACK           — "true" / "false" (default: false)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

# ─── Config ───────────────────────────────────────────────────────────────────
AI_PROVIDER             = os.getenv("AI_PROVIDER", "gemini").lower()
ALLOW_BASELINE_FALLBACK = os.getenv("ALLOW_BASELINE_FALLBACK", "false").lower() == "true"
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", os.getenv("AGENT_MODEL", "gemini-2.5-flash"))
_GEMINI_KEY             = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

STOCK_EVAL_FILE = ROOT / "stock_prediction_evaluation_runs.jsonl"

_HIGH_RISK_CUTOFF  = 80
_RETURN_CLAMP_MAX  = 20.0
_RETURN_CLAMP_MIN  = -20.0


def _get_horizon_thresholds(horizon_days: int) -> Tuple[float, float]:
    """Return (buy_threshold, sell_threshold) calibrated to prediction horizon."""
    if horizon_days <= 10:
        return 1.25, -1.25
    elif horizon_days <= 30:
        return 2.0, -2.0
    else:
        return 2.5, -2.5


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_sma(closes: List[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 4)


def _calc_ema(closes: List[float], period: int) -> List[float]:
    if not closes:
        return []
    k = 2.0 / (period + 1)
    emas = [closes[0]]
    for p in closes[1:]:
        emas.append(p * k + emas[-1] * (1 - k))
    return emas


def _calc_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _calc_macd(
    closes: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < slow + signal_period:
        return None, None, None
    fast_ema  = _calc_ema(closes, fast)
    slow_ema  = _calc_ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    if len(macd_line) < signal_period:
        return None, None, None
    sig_ema  = _calc_ema(macd_line, signal_period)
    macd_v   = round(macd_line[-1], 4)
    signal_v = round(sig_ema[-1], 4)
    hist_v   = round(macd_v - signal_v, 4)
    return macd_v, signal_v, hist_v


def _calc_bollinger_bands(
    closes: List[float], period: int = 20
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    sma = sum(window) / period
    std = math.sqrt(sum((c - sma) ** 2 for c in window) / period)
    return round(sma + 2 * std, 4), round(sma, 4), round(sma - 2 * std, 4)


def _calc_atr(bars: List[dict], period: int = 14) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        high = float(bars[i].get("high") or bars[i].get("close") or 0)
        low  = float(bars[i].get("low")  or bars[i].get("close") or 0)
        prev = float(bars[i - 1].get("close") or 0)
        if prev <= 0:
            continue
        trs.append(max(high - low, abs(high - prev), abs(low - prev)))
    if len(trs) < period:
        return None
    return round(sum(trs[-period:]) / period, 4)


def _period_return(closes: List[float], lookback: int) -> Optional[float]:
    if len(closes) < lookback + 1:
        return None
    p_now, p_then = closes[-1], closes[-lookback]
    if p_then <= 0:
        return None
    return round((p_now - p_then) / p_then * 100.0, 4)


def _annualized_volatility(closes: List[float], window: int = 60) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    rets = []
    for i in range(max(1, len(closes) - window), len(closes)):
        prev, curr = closes[i - 1], closes[i]
        if prev > 0:
            rets.append(math.log(curr / prev))
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var  = sum((r - mean) ** 2 for r in rets) / len(rets)
    return round(math.sqrt(var) * math.sqrt(252) * 100.0, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE PACKET
# ═══════════════════════════════════════════════════════════════════════════════

def build_stock_feature_packet(
    normalized_input: dict,
    ai_visible_bars: List[dict],
    benchmark_bars: Optional[List[dict]] = None,
) -> dict:
    """
    Build deterministic feature packet from real bars only.
    Leakage prevention: caller must pass only bars up to effective_origin_date.
    """
    if not ai_visible_bars:
        return {"status": "INSUFFICIENT", "n_bars": 0, "error": "No bars provided"}

    price_basis = str(normalized_input.get("price_basis", "close") or "close")

    def _price(bar: dict) -> float:
        return float(bar.get(price_basis) or bar.get("close") or 0)

    closes = [_price(b) for b in ai_visible_bars if _price(b) > 0]
    n = len(closes)
    if n < 5:
        return {"status": "INSUFFICIENT", "n_bars": n, "error": "Too few valid price bars"}

    origin_price = closes[-1]
    ctx_start    = str(normalized_input.get("historical_context_start_date", "") or "")
    origin_str   = str(normalized_input.get("prediction_origin_date", "") or "")

    # Returns
    r1d   = _period_return(closes, 1)
    r5d   = _period_return(closes, 5)
    r10d  = _period_return(closes, 10)
    r20d  = _period_return(closes, 20)
    r60d  = _period_return(closes, 60)
    r120d = _period_return(closes, 120)
    r_full = round((closes[-1] - closes[0]) / closes[0] * 100.0, 4) if closes[0] > 0 else None

    # Trend / SMA
    sma10  = _calc_sma(closes, 10)
    sma20  = _calc_sma(closes, 20)
    sma50  = _calc_sma(closes, 50)
    sma100 = _calc_sma(closes, 100)

    def _pct_diff(price, ref):
        if price and ref and ref > 0:
            return round((price - ref) / ref * 100.0, 4)
        return None

    pvs20 = _pct_diff(origin_price, sma20)
    pvs50 = _pct_diff(origin_price, sma50)
    trend_regime = (
        "bullish" if pvs20 and pvs50 and pvs20 > 2 and pvs50 > 2
        else "bearish" if pvs20 and pvs50 and pvs20 < -2 and pvs50 < -2
        else "sideways"
    )

    # Momentum
    rsi14 = _calc_rsi(closes, 14)
    macd, macd_signal, macd_hist = _calc_macd(closes)

    # Volatility
    ann_vol = _annualized_volatility(closes)
    atr14   = _calc_atr(ai_visible_bars, 14)
    bb_upper, bb_mid, bb_lower = _calc_bollinger_bands(closes, 20)
    vol_regime = (
        "low" if ann_vol and ann_vol < 20
        else "high" if ann_vol and ann_vol > 50
        else "medium"
    )
    bb_position = None
    if bb_upper and bb_lower and (bb_upper - bb_lower) > 0:
        bb_position = round((origin_price - bb_lower) / (bb_upper - bb_lower), 4)

    peak = max(closes)
    max_dd = round((origin_price - peak) / peak * 100.0, 4) if peak > 0 else None
    low_   = min(closes)
    dist_from_low = round((origin_price - low_) / low_ * 100.0, 4) if low_ > 0 else None

    # Volume
    vols = [float(b.get("volume") or 0) for b in ai_visible_bars]
    avg_vol20 = round(sum(vols[-20:]) / 20, 0) if len(vols) >= 20 else None
    avg_vol60 = round(sum(vols[-60:]) / 60, 0) if len(vols) >= 60 else None
    latest_vol = vols[-1] if vols else None
    vol_ratio20 = round(latest_vol / avg_vol20, 4) if avg_vol20 and avg_vol20 > 0 and latest_vol else None

    # Support / Resistance
    window20         = closes[-20:] if n >= 20 else closes
    recent_support   = round(min(window20), 4)
    recent_resistance = round(max(window20), 4)
    dist_to_sup  = round((origin_price - recent_support) / origin_price * 100.0, 4) if origin_price > 0 else None
    dist_to_res  = round((recent_resistance - origin_price) / origin_price * 100.0, 4) if origin_price > 0 else None

    # Benchmark relative strength
    bm_r20 = bm_r60 = rel20 = rel60 = None
    if benchmark_bars:
        bm_closes = [float(b.get("close") or 0) for b in benchmark_bars if b.get("close")]
        if bm_closes:
            bm_r20 = _period_return(bm_closes, 20)
            bm_r60 = _period_return(bm_closes, 60)
            if bm_r20 is not None and r20d is not None:
                rel20 = round(r20d - bm_r20, 4)
            if bm_r60 is not None and r60d is not None:
                rel60 = round(r60d - bm_r60, 4)

    # Reversal detection
    oversold_signal   = rsi14 is not None and rsi14 < 30
    overbought_signal = rsi14 is not None and rsi14 > 70
    near_support      = dist_to_sup is not None and dist_to_sup < 3.0
    near_resistance   = dist_to_res is not None and dist_to_res < 3.0
    momentum_divergence = (
        r5d is not None and r20d is not None and r60d is not None
        and r5d > 0 and r20d < 0 and r60d < 0
    )
    reversal_risk = "low"
    if oversold_signal and near_support and r60d is not None and r60d < -20:
        reversal_risk = "high"
    elif oversold_signal or (near_support and r20d is not None and r20d < -10):
        reversal_risk = "medium"

    dq = 100
    if n < 20:   dq -= 40
    elif n < 60: dq -= 20
    elif n < 120: dq -= 10

    return {
        "status":              "OK",
        "n_bars":              n,
        "data_quality_score":  max(0, min(100, dq)),
        "symbol":              normalized_input.get("symbol", ""),
        "benchmark":           normalized_input.get("benchmark", ""),
        "price_basis":         price_basis,
        "origin_price":        round(origin_price, 4),
        "ctx_start":           ctx_start,
        "origin_date":         origin_str,
        "horizon_days":        normalized_input.get("decision_horizon_days", 30),
        "returns": {
            "return_1d":            r1d,
            "return_5d":            r5d,
            "return_10d":           r10d,
            "return_20d":           r20d,
            "return_60d":           r60d,
            "return_120d":          r120d,
            "full_context_return":  r_full,
        },
        "trend": {
            "sma_10":               sma10,
            "sma_20":               sma20,
            "sma_50":               sma50,
            "sma_100":              sma100,
            "price_vs_sma_20_pct":  pvs20,
            "price_vs_sma_50_pct":  pvs50,
            "trend_regime":         trend_regime,
        },
        "momentum": {
            "RSI_14":               rsi14,
            "MACD":                 macd,
            "MACD_signal":          macd_signal,
            "MACD_histogram":       macd_hist,
            "rate_of_change_20":    r20d,
        },
        "volatility": {
            "annualized_volatility": ann_vol,
            "ATR_14":               atr14,
            "BB_upper":             bb_upper,
            "BB_mid":               bb_mid,
            "BB_lower":             bb_lower,
            "BB_position":          bb_position,
            "max_drawdown_context": max_dd,
            "dist_from_low_pct":    dist_from_low,
            "volatility_regime":    vol_regime,
        },
        "volume": {
            "avg_volume_20":  avg_vol20,
            "avg_volume_60":  avg_vol60,
            "latest_volume":  latest_vol,
            "volume_ratio_20": vol_ratio20,
        },
        "support_resistance": {
            "recent_support":           recent_support,
            "recent_resistance":        recent_resistance,
            "dist_to_support_pct":      dist_to_sup,
            "dist_to_resistance_pct":   dist_to_res,
        },
        "benchmark_comparison": {
            "benchmark_return_20d":  bm_r20,
            "benchmark_return_60d":  bm_r60,
            "relative_strength_20d": rel20,
            "relative_strength_60d": rel60,
        },
        "reversal_signals": {
            "oversold_signal":      oversold_signal,
            "overbought_signal":    overbought_signal,
            "near_support":         near_support,
            "near_resistance":      near_resistance,
            "momentum_divergence":  momentum_divergence,
            "reversal_risk":        reversal_risk,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def build_calibration_summary(symbol: str = "", horizon_days: int = 0) -> dict:
    """Read past validated records and build a calibration summary for Gemini."""
    records: List[dict] = []
    if STOCK_EVAL_FILE.exists():
        try:
            with open(STOCK_EVAL_FILE, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass

    if not records:
        return {
            "total_validated_records": 0,
            "note": "No calibration records yet — this is a fresh model with no history.",
        }

    total  = len(records)
    dec_ok = sum(1 for r in records if r.get("comparison", {}).get("decision_match") is True)
    dir_ok = sum(1 for r in records if r.get("comparison", {}).get("directional_match") is True)

    ret_errors = [
        abs(float(r["comparison"]["abs_return_error_pct"]))
        for r in records
        if r.get("comparison", {}).get("abs_return_error_pct") is not None
    ]
    avg_ret_err = round(sum(ret_errors) / len(ret_errors), 2) if ret_errors else None

    raw_errors = [
        float(r["comparison"]["return_error_pct"])
        for r in records
        if r.get("comparison", {}).get("return_error_pct") is not None
    ]
    overpred  = sum(1 for e in raw_errors if e < 0)   # AI predicted too high
    underpred = sum(1 for e in raw_errors if e > 0)   # AI predicted too low

    sym_records = [
        r for r in records
        if r.get("stock_prediction_input", {}).get("symbol") == symbol
    ] if symbol else []

    recent_conflicts = [
        {
            "symbol":   r.get("stock_prediction_input", {}).get("symbol", ""),
            "ai_dec":   r.get("ai_prediction", {}).get("decision", ""),
            "act_dec":  r.get("actual_validation", {}).get("actual_decision", ""),
            "ret_err":  r.get("comparison", {}).get("return_error_pct"),
        }
        for r in records
        if r.get("comparison", {}).get("decision_match") is False
    ][-5:]

    if overpred > underpred * 1.5:
        bias_note = "Model has SELL bias — overpredicts downside. Consider adjusting toward neutral/BUY."
    elif underpred > overpred * 1.5:
        bias_note = "Model has BUY bias — underpredicts downside."
    else:
        bias_note = "No strong directional bias detected."

    return {
        "total_validated_records": total,
        "decision_match_pct":     round(dec_ok / total * 100, 1),
        "direction_match_pct":    round(dir_ok / total * 100, 1),
        "avg_return_error_pp":    avg_ret_err,
        "overprediction_rate":    round(overpred / total * 100, 1),
        "underprediction_rate":   round(underpred / total * 100, 1),
        "symbol_record_count":    len(sym_records),
        "recent_conflicts":       recent_conflicts,
        "bias_note":              bias_note,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL SCORE ENGINE — deterministic bull/bear/uncertainty from feature packet
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_signal_scores(feature_packet: dict) -> dict:
    """
    Compute deterministic bullish/bearish/uncertainty scores (each 0-100) from
    the feature packet. Each indicator casts a vote (bull, bear, uncertain) that
    sums to 1.0 per indicator. Final score = mean vote × 100 across all indicators.
    """
    trend     = feature_packet.get("trend", {}) or {}
    momentum  = feature_packet.get("momentum", {}) or {}
    volatility = feature_packet.get("volatility", {}) or {}
    returns   = feature_packet.get("returns", {}) or {}
    reversal  = feature_packet.get("reversal_signals", {}) or {}

    signals: List[Tuple[float, float, float]] = []  # (bull, bear, uncert) each vote sums to 1.0

    # 1. Trend regime + SMA alignment
    regime = trend.get("trend_regime", "sideways")
    pvs20  = trend.get("price_vs_sma_20_pct") or 0.0
    pvs50  = trend.get("price_vs_sma_50_pct") or 0.0
    if regime == "bullish" and pvs20 > 2 and pvs50 > 2:
        signals.append((1.0, 0.0, 0.0))
    elif regime == "bearish" and pvs20 < -2 and pvs50 < -2:
        signals.append((0.0, 1.0, 0.0))
    elif pvs20 > 1:
        signals.append((0.6, 0.0, 0.4))
    elif pvs20 < -1:
        signals.append((0.0, 0.6, 0.4))
    else:
        signals.append((0.0, 0.0, 1.0))

    # 2. RSI
    rsi = momentum.get("RSI_14")
    if rsi is not None:
        if 55 <= rsi < 70:
            signals.append((1.0, 0.0, 0.0))
        elif 30 < rsi <= 45:
            signals.append((0.0, 1.0, 0.0))
        elif rsi >= 70:
            signals.append((0.5, 0.0, 0.5))   # overbought → uncertain
        elif rsi <= 30:
            signals.append((0.0, 0.4, 0.6))   # oversold → uncertain (reversal risk)
        else:
            signals.append((0.0, 0.0, 1.0))

    # 3. MACD line + histogram
    macd_v = momentum.get("MACD")
    macd_h = momentum.get("MACD_histogram")
    if macd_v is not None and macd_h is not None:
        if macd_v > 0 and macd_h > 0:
            signals.append((1.0, 0.0, 0.0))
        elif macd_v < 0 and macd_h < 0:
            signals.append((0.0, 1.0, 0.0))
        elif macd_h > 0:
            signals.append((0.6, 0.0, 0.4))
        elif macd_h < 0:
            signals.append((0.0, 0.6, 0.4))
        else:
            signals.append((0.0, 0.0, 1.0))

    # 4. Recent returns (20d + 60d)
    r20 = returns.get("return_20d")
    r60 = returns.get("return_60d")
    if r20 is not None and r60 is not None:
        if r20 > 3 and r60 > 5:
            signals.append((1.0, 0.0, 0.0))
        elif r20 < -3 and r60 < -5:
            signals.append((0.0, 1.0, 0.0))
        elif r20 > 1:
            signals.append((0.6, 0.0, 0.4))
        elif r20 < -1:
            signals.append((0.0, 0.6, 0.4))
        else:
            signals.append((0.0, 0.0, 1.0))

    # 5. Bollinger band position
    bb_pos = volatility.get("BB_position")
    if bb_pos is not None:
        if bb_pos > 0.8:
            signals.append((0.0, 0.4, 0.6))   # near upper band → bearish tilt + uncertain
        elif bb_pos < 0.2:
            signals.append((0.4, 0.0, 0.6))   # near lower band → bullish tilt + uncertain
        else:
            signals.append((0.0, 0.0, 1.0))

    # 6. Volatility regime
    vol_regime = volatility.get("volatility_regime", "medium")
    if vol_regime == "high":
        signals.append((0.0, 0.0, 1.0))
    elif vol_regime == "low":
        pass   # no vote — low vol is neutral

    # 7. Reversal risk
    rev_risk = reversal.get("reversal_risk", "low")
    if rev_risk == "high":
        signals.append((0.0, 0.0, 1.0))
    elif rev_risk == "medium":
        signals.append((0.0, 0.0, 0.8))

    if not signals:
        return {"bullish_score": 50, "bearish_score": 50, "uncertainty_score": 50,
                "dominant": "uncertain", "dominant_score": 50}

    n = len(signals)
    bull_s  = round(sum(s[0] for s in signals) / n * 100)
    bear_s  = round(sum(s[1] for s in signals) / n * 100)
    uncert_s = round(sum(s[2] for s in signals) / n * 100)

    if bull_s > bear_s and bull_s > uncert_s:
        dominant, dom_score = "bullish", bull_s
    elif bear_s > bull_s and bear_s > uncert_s:
        dominant, dom_score = "bearish", bear_s
    else:
        dominant, dom_score = "uncertain", uncert_s

    return {
        "bullish_score":    bull_s,
        "bearish_score":    bear_s,
        "uncertainty_score": uncert_s,
        "dominant":          dominant,
        "dominant_score":    dom_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION GUARDRAILS (deterministic — dynamic thresholds by horizon)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_guardrails(
    predicted_return_pct: float,
    confidence_score: int,
    risk_score: int,
    data_quality_score: int,
    horizon_days: int = 30,
    signal_scores: Optional[dict] = None,
) -> Tuple[str, str]:
    """Deterministic post-Gemini guardrails with horizon-calibrated thresholds.
    Risk score NEVER creates SELL. HOLD can be overridden by strong signal consensus.
    """
    if data_quality_score < 60:
        return "REVIEW", f"Data quality too low ({data_quality_score}/100) — insufficient data."
    if confidence_score < 50:
        return "REVIEW", f"Confidence too low ({confidence_score}/100) — mixed signals."

    buy_thr, sell_thr = _get_horizon_thresholds(horizon_days)

    if predicted_return_pct >= buy_thr:
        decision, reason = "BUY",  (
            f"Predicted return {predicted_return_pct:+.2f}% >= +{buy_thr}% BUY threshold "
            f"({horizon_days}d horizon)."
        )
    elif predicted_return_pct <= sell_thr:
        decision, reason = "SELL", (
            f"Predicted return {predicted_return_pct:+.2f}% <= {sell_thr}% SELL threshold "
            f"({horizon_days}d horizon)."
        )
    else:
        decision, reason = "HOLD", (
            f"Return {predicted_return_pct:+.2f}% in HOLD band (±{abs(sell_thr):.2f}% "
            f"for {horizon_days}d horizon)."
        )

    # Post-Gemini HOLD override: strong signal consensus + confidence overrides weak HOLD
    if decision == "HOLD" and signal_scores and confidence_score >= 60:
        dominant   = signal_scores.get("dominant", "uncertain")
        dom_score  = signal_scores.get("dominant_score", 0)
        near_thr   = abs(predicted_return_pct) >= abs(sell_thr) * 0.75
        if dominant == "bullish" and dom_score >= 65 and near_thr:
            decision = "BUY"
            reason   = (
                f"HOLD→BUY override: bullish signals {dom_score}/100, "
                f"return {predicted_return_pct:+.2f}% near threshold. "
                f"[was: {reason}]"
            )
        elif dominant == "bearish" and dom_score >= 65 and near_thr:
            decision = "SELL"
            reason   = (
                f"HOLD→SELL override: bearish signals {dom_score}/100, "
                f"return {predicted_return_pct:+.2f}% near threshold. "
                f"[was: {reason}]"
            )

    if risk_score >= _HIGH_RISK_CUTOFF and decision == "BUY":
        return "HOLD",   f"BUY downgraded to HOLD: high risk ({risk_score}/100) despite positive return."
    if risk_score >= _HIGH_RISK_CUTOFF and decision == "SELL":
        return "REVIEW", f"SELL → REVIEW: high uncertainty ({risk_score}/100) — needs human review."

    return decision, reason


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_gemini_prompt(
    normalized_input: dict,
    feature_packet: dict,
    calibration_summary: dict,
) -> str:
    fp_json  = json.dumps(feature_packet, indent=2, default=str)
    cal_json = json.dumps(calibration_summary, indent=2, default=str)
    horizon_days = int(normalized_input.get("decision_horizon_days") or 30)
    buy_thr, sell_thr = _get_horizon_thresholds(horizon_days)
    inp_json = json.dumps({
        "symbol":                        normalized_input.get("symbol"),
        "benchmark":                     normalized_input.get("benchmark"),
        "price_basis":                   normalized_input.get("price_basis"),
        "initial_capital":               normalized_input.get("initial_capital"),
        "historical_context_start_date": normalized_input.get("historical_context_start_date"),
        "effective_origin_date":         normalized_input.get("prediction_origin_date"),
        "target_date":                   normalized_input.get("target_date"),
        "horizon_days":                  horizon_days,
        "buy_threshold_pct":             buy_thr,
        "sell_threshold_pct":            sell_thr,
    }, indent=2)

    return f"""You are a financial decision-support reasoning agent for a walk-forward backtesting system.

STRICT RULES:
1. You must analyze ONLY the structured data provided below.
2. You must NOT invent prices, news events, earnings, or external facts not in the data.
3. You are NOT giving investment advice. This is a model prediction for validation/research only.
4. You must return ONLY valid JSON — no text before or after the JSON.
5. The target_date price is UNKNOWN to you — do NOT reference any price after the origin date.
6. Do NOT use calibration accuracy stats to fake a better prediction. Use them to avoid known failure patterns.

IMPORTANT — SELL BIAS WARNING:
The calibration summary below may show a SELL bias in past predictions.
If momentum looks negative but RSI is oversold (< 35), price is near support, and reversal_risk is medium/high,
do NOT blindly predict SELL. Consider HOLD or even a muted negative return with REVIEW decision.
Strong downside momentum followed by extreme oversold RSI historically precedes reversals.

DECISION QUALITY — AVOID HOLD OVERUSE:
Use HOLD ONLY when ALL of the following are true:
- Expected price movement is genuinely near zero (predicted_return_pct within 0.5% of 0)
- Bullish and bearish signals are nearly equal and balanced (no dominant direction)
- Confidence is genuinely weak (< 55) with no clear trend

Use REVIEW (not HOLD) when:
- Uncertainty is high but there IS a directional lean — flag for human review, not neutral

Do NOT default to HOLD when:
- Trend regime + SMA position agree on a direction
- RSI clearly signals momentum (>60 bullish, <40 bearish)
- MACD line AND histogram both point the same direction
- 20d and 60d returns consistently agree on direction

Horizon-calibrated thresholds for this prediction:
- BUY if predicted_return_pct >= +{buy_thr}%
- SELL if predicted_return_pct <= {sell_thr}%
- Only predict HOLD if return is genuinely in the ±{abs(sell_thr)}% neutral zone

If meaningful positive return ({buy_thr}%+) is predicted WITH bullish trend AND positive MACD → commit to BUY.
If meaningful negative return ({sell_thr}% or worse) WITH bearish trend AND negative MACD → commit to SELL.

NORMALIZED INPUT:
{inp_json}

FEATURE PACKET (computed from real historical bars — no future data):
{fp_json}

CALIBRATION SUMMARY (past model performance — use to avoid known failure patterns):
{cal_json}

DATA LIMITATIONS:
- Historical bars come from external_historical_provider (NASDAQ public API)
- Approximately 16 months of history available
- No real-time news, earnings, or macro data
- RapidAPI market movers data NOT available for historical origins (would be future leakage)

REQUIRED JSON RESPONSE (return ONLY this JSON, no other text):
{{
  "status": "SUCCESS",
  "ai_provider": "gemini",
  "predicted_return_pct": <float, clamped to -20.0 to +20.0>,
  "confidence_score": <integer 0-100>,
  "risk_score": <integer 0-100>,
  "trend_assessment": "<1-2 sentences on trend regime>",
  "momentum_assessment": "<1-2 sentences on RSI/MACD/momentum>",
  "volatility_assessment": "<1-2 sentences on ATR/BB/vol regime>",
  "benchmark_assessment": "<1 sentence on relative strength vs benchmark>",
  "bull_case": "<1 sentence on best-case scenario>",
  "bear_case": "<1 sentence on worst-case scenario>",
  "main_reason": "<2-3 sentences explaining the prediction direction>",
  "why_not_opposite_decision": "<1-2 sentences on why opposite direction was rejected>",
  "key_features_used": ["list of top 3-5 features that most influenced this prediction"],
  "data_limitations": ["list of 1-3 known data gaps that could affect accuracy"],
  "needs_human_review": <true if confidence < 60 or strong reversal risk, else false>
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json(raw_text: str) -> str:
    """
    Extract clean JSON object from Gemini response.
    Most reliable strategy: find first '{' and last '}' in the text.
    Handles thinking tags, markdown fences, preamble text.
    """
    import re
    # Remove <think>...</think> or <thinking>...</thinking> blocks
    raw_text = re.sub(r"<think(?:ing)?>\s*.*?\s*</think(?:ing)?>", "", raw_text, flags=re.DOTALL)
    # Find the outermost JSON object
    start = raw_text.find("{")
    end   = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end + 1].strip()
    return raw_text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI API CALL
# ═══════════════════════════════════════════════════════════════════════════════

def _call_gemini(
    normalized_input: dict,
    feature_packet: dict,
    calibration_summary: dict,
    origin_price: float,
    initial_capital: float,
) -> dict:
    """Call Gemini, validate response, recalculate numerics. Returns result dict."""
    t0 = time.time()

    if not _GEMINI_KEY:
        return {
            "status":            "FAILED",
            "gemini_used":       False,
            "error":             "GEMINI_API_KEY missing. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env",
            "gemini_latency_ms": 0,
        }

    prompt_content = _build_gemini_prompt(normalized_input, feature_packet, calibration_summary)
    prompt_hash    = hashlib.sha256(prompt_content.encode()).hexdigest()[:16]

    try:
        try:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=_GEMINI_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_content,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw_text = (response.text or "").strip()
        except ImportError:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=_GEMINI_KEY)
            model = legacy_genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt_content,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                },
            )
            raw_text = (getattr(response, "text", "") or "").strip()

        gemini_latency_ms = round((time.time() - t0) * 1000, 0)

        # Extract JSON from response (handles thinking tags, markdown fences, plain JSON)
        json_text = _extract_json(raw_text)

        try:
            gemini_json = json.loads(json_text)
        except Exception as parse_err:
            return {
                "status":            "FAILED",
                "gemini_used":       True,
                "error":             f"Gemini JSON parse failed: {parse_err}",
                "gemini_raw_text":   raw_text[:500],
                "gemini_latency_ms": gemini_latency_ms,
            }

        # Recalculate numerics from Gemini's predicted_return_pct
        pred_return  = float(gemini_json.get("predicted_return_pct") or 0.0)
        pred_return  = max(_RETURN_CLAMP_MIN, min(_RETURN_CLAMP_MAX, pred_return))
        pred_return  = round(pred_return, 4)

        pred_target  = round(origin_price * (1.0 + pred_return / 100.0), 4)
        pred_capital = round(initial_capital * (pred_target / origin_price), 4)
        pred_pl      = round(pred_capital - initial_capital, 4)

        conf  = max(0, min(100, int(gemini_json.get("confidence_score") or 50)))
        risk  = max(0, min(100, int(gemini_json.get("risk_score") or 50)))
        dq    = feature_packet.get("data_quality_score", 70)
        h_days = int(normalized_input.get("decision_horizon_days") or 30)

        sig_scores = _compute_signal_scores(feature_packet)
        decision, decision_reason = _apply_guardrails(
            pred_return, conf, risk, dq,
            horizon_days=h_days,
            signal_scores=sig_scores,
        )

        output_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:16]

        return {
            "status":                   "SUCCESS",
            "gemini_used":              True,
            "ai_provider":              "gemini",
            "model_name":               GEMINI_MODEL,
            "gemini_latency_ms":        gemini_latency_ms,
            "prompt_hash":              prompt_hash,
            "gemini_output_hash":       output_hash,
            "predicted_return_pct":     pred_return,
            "predicted_target_price":   pred_target,
            "predicted_final_capital":  pred_capital,
            "predicted_total_pl":       pred_pl,
            "decision":                 decision,
            "decision_reason":          decision_reason,
            "confidence_score":         conf,
            "risk_score":               risk,
            "data_quality_score":       dq,
            "signal_scores":            sig_scores,
            "reasoning":                gemini_json.get("main_reason", ""),
            "trend_assessment":         gemini_json.get("trend_assessment", ""),
            "momentum_assessment":      gemini_json.get("momentum_assessment", ""),
            "volatility_assessment":    gemini_json.get("volatility_assessment", ""),
            "benchmark_assessment":     gemini_json.get("benchmark_assessment", ""),
            "bull_case":                gemini_json.get("bull_case", ""),
            "bear_case":                gemini_json.get("bear_case", ""),
            "why_not_opposite":         gemini_json.get("why_not_opposite_decision", ""),
            "key_features_used":        gemini_json.get("key_features_used", []),
            "data_limitations":         gemini_json.get("data_limitations", []),
            "needs_human_review":       bool(gemini_json.get("needs_human_review", False)),
            "gemini_raw_json":          gemini_json,
        }

    except Exception as exc:
        gemini_latency_ms = round((time.time() - t0) * 1000, 0)
        return {
            "status":            "FAILED",
            "gemini_used":       True,
            "error":             f"Gemini API error: {type(exc).__name__}: {exc}",
            "gemini_latency_ms": gemini_latency_ms,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_no_leakage(ai_visible_bars: List[dict], origin_date: str, ctx_start: str) -> dict:
    if not ai_visible_bars:
        return {"status": "NO_BARS", "note": "No bars to check"}
    dates = [b["date"] for b in ai_visible_bars]
    min_d, max_d = min(dates), max(dates)
    if max_d > origin_date:
        return {"status": "LEAKAGE_DETECTED", "max_date": max_d, "origin": origin_date}
    if min_d < ctx_start:
        return {"status": "LEAKAGE_DETECTED", "min_date": min_d, "ctx_start": ctx_start}
    return {"status": "CLEAN", "min_date": min_d, "max_date": max_d}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_gemini_stock_prediction(spi: dict, price_history_context: List[dict]) -> dict:
    """
    Walk-forward stock prediction using Gemini.

    Parameters:
    - spi                   : StockPredictionInput dict
    - price_history_context : bars filtered to [ctx_start, origin_date] by caller

    Returns a result dict compatible with the existing app's ai_result schema,
    plus additional Gemini-specific fields (ai_provider, gemini_used, etc.).
    """
    from stock_prediction_agent import (
        validate_stock_prediction_input,
        build_stock_prediction_hash,
        _parse_ymd,
    )
    from historical_price_service import get_price_on_date

    origin_str  = str(spi.get("prediction_origin_date", "") or "")
    ctx_start   = str(spi.get("historical_context_start_date", "") or "")
    target_str  = str(spi.get("target_date", "") or "")
    symbol      = str(spi.get("symbol", "") or "").upper()
    horizon_days = int(spi.get("decision_horizon_days", 30) or 30)
    initial_cap  = float(spi.get("initial_capital", 50000) or 50000)
    input_hash   = build_stock_prediction_hash(spi)

    # Double guard: strip future bars
    history = [b for b in (price_history_context or []) if b["date"] <= origin_str]

    valid, err = validate_stock_prediction_input(spi)
    if not valid:
        return _error_result(spi, input_hash, err)

    if not history:
        return _error_result(spi, input_hash, f"No price bars available up to {origin_str}")

    # Leakage check
    leakage = _validate_no_leakage(history, origin_str, ctx_start)
    if leakage["status"] == "LEAKAGE_DETECTED":
        return _error_result(spi, input_hash, f"Leakage detected: {leakage}")

    # Origin price
    origin_price, eff_origin_date, price_err = get_price_on_date(history, origin_str)
    if origin_price is None:
        return _error_result(spi, input_hash, f"Cannot get origin price on {origin_str}: {price_err}")

    # Build feature packet
    feature_packet = build_stock_feature_packet(spi, history)
    if feature_packet.get("status") != "OK":
        return _error_result(spi, input_hash, f"Feature packet failed: {feature_packet.get('error')}")

    # Calibration summary
    calibration_summary = build_calibration_summary(symbol=symbol, horizon_days=horizon_days)

    # Call Gemini
    gemini_result = _call_gemini(spi, feature_packet, calibration_summary, origin_price, initial_cap)

    if gemini_result.get("status") != "SUCCESS":
        return _error_result(
            spi, input_hash,
            f"Gemini prediction unavailable. {gemini_result.get('error', 'Unknown error')}",
            extra={
                "gemini_used":       gemini_result.get("gemini_used", False),
                "ai_provider":       "gemini",
                "gemini_latency_ms": gemini_result.get("gemini_latency_ms", 0),
            },
        )

    pred_return  = gemini_result["predicted_return_pct"]
    pred_target  = gemini_result["predicted_target_price"]
    pred_capital = gemini_result["predicted_final_capital"]
    pred_pl      = gemini_result["predicted_total_pl"]

    from stock_comparison_engine import validate_no_leakage as _ck
    leakage_check = _ck(history, origin_str).get("status", "UNKNOWN")

    # Gemini reasoning (combine assessments)
    reasoning_parts = [
        gemini_result.get("reasoning") or "",
        gemini_result.get("trend_assessment") or "",
        gemini_result.get("momentum_assessment") or "",
    ]
    reasoning = " | ".join(p for p in reasoning_parts if p)[:800] or "Gemini prediction completed."

    return {
        "status":                       "SUCCESS",
        "source":                       "gemini_stock_prediction_agent",
        "model_version":                "gemini_stock_prediction_agent_v1",
        "ai_provider":                  "gemini",
        "gemini_used":                  True,
        "model_name":                   GEMINI_MODEL,
        "gemini_latency_ms":            gemini_result.get("gemini_latency_ms", 0),
        "prompt_hash":                  gemini_result.get("prompt_hash", ""),
        "gemini_output_hash":           gemini_result.get("gemini_output_hash", ""),
        "stock_prediction_input_hash":  input_hash,
        "symbol":                       symbol,
        "historical_context_start_date": spi.get("historical_context_start_date"),
        "prediction_origin_date":       origin_str,
        "effective_origin_date":        eff_origin_date,
        "target_date":                  target_str,
        "decision_horizon_days":        horizon_days,
        "price_basis":                  str(spi.get("price_basis", "close") or "close"),
        "origin_price_used":            round(origin_price, 4),
        "predicted_target_price":       pred_target,
        "predicted_return_pct":         pred_return,
        "initial_capital":              initial_cap,
        "predicted_final_capital":      pred_capital,
        "predicted_total_pl":           pred_pl,
        "decision":                     gemini_result["decision"],
        "decision_reason":              gemini_result["decision_reason"],
        "confidence_score":             gemini_result["confidence_score"],
        "risk_score":                   gemini_result["risk_score"],
        "data_quality_score":           gemini_result["data_quality_score"],
        "signal_scores":                gemini_result.get("signal_scores", {}),
        "latest_date_seen_by_ai":       history[-1]["date"] if history else origin_str,
        "target_price_hidden_from_ai":  True,
        "leakage_check":                leakage_check,
        "reasoning":                    reasoning,
        "trend_assessment":             gemini_result.get("trend_assessment", ""),
        "momentum_assessment":          gemini_result.get("momentum_assessment", ""),
        "volatility_assessment":        gemini_result.get("volatility_assessment", ""),
        "benchmark_assessment":         gemini_result.get("benchmark_assessment", ""),
        "bull_case":                    gemini_result.get("bull_case", ""),
        "bear_case":                    gemini_result.get("bear_case", ""),
        "why_not_opposite":             gemini_result.get("why_not_opposite", ""),
        "key_features_used":            gemini_result.get("key_features_used", []),
        "data_limitations":             gemini_result.get("data_limitations", []),
        "needs_human_review":           gemini_result.get("needs_human_review", False),
        "_feature_packet":              feature_packet,
        "_calibration_summary":         calibration_summary,
        "_gemini_raw_json":             gemini_result.get("gemini_raw_json", {}),
        "features_used": {
            "RSI_14":           feature_packet["momentum"]["RSI_14"],
            "MACD":             feature_packet["momentum"]["MACD"],
            "return_5d":        feature_packet["returns"]["return_5d"],
            "return_20d":       feature_packet["returns"]["return_20d"],
            "return_60d":       feature_packet["returns"]["return_60d"],
            "annualized_vol":   feature_packet["volatility"]["annualized_volatility"],
            "trend_regime":     feature_packet["trend"]["trend_regime"],
            "reversal_risk":    feature_packet["reversal_signals"]["reversal_risk"],
            "bars_used":        feature_packet["n_bars"],
            "last_ai_bar_date": history[-1]["date"] if history else origin_str,
        },
    }


def _error_result(spi: dict, input_hash: str, error_msg: str, extra: dict = None) -> dict:
    result = {
        "status":                       "FAILED",
        "source":                       "gemini_stock_prediction_agent",
        "model_version":                "gemini_stock_prediction_agent_v1",
        "ai_provider":                  "gemini",
        "gemini_used":                  False,
        "stock_prediction_input_hash":  input_hash,
        "symbol":                       str(spi.get("symbol", "") or ""),
        "prediction_origin_date":       str(spi.get("prediction_origin_date", "") or ""),
        "target_date":                  str(spi.get("target_date", "") or ""),
        "decision":                     "REVIEW",
        "confidence_score":             0,
        "data_quality_score":           0,
        "risk_score":                   0,
        "target_price_hidden_from_ai":  True,
        "error":                        error_msg,
        "predicted_target_price":       None,
        "predicted_return_pct":         None,
        "predicted_final_capital":      None,
        "predicted_total_pl":           None,
    }
    if extra:
        result.update(extra)
    return result
