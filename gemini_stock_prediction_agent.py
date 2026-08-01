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
        return 1.0, -1.0
    elif horizon_days <= 30:
        return 1.5, -1.5
    elif horizon_days <= 60:
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

    # Effective context start = first bar actually present (may be later than requested)
    _first_bar_date = ""
    for _b in ai_visible_bars:
        _bd = str(_b.get("date") or _b.get("datetime") or "")
        if _bd:
            _first_bar_date = _bd[:10]
            break

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
    elif overbought_signal and near_resistance and r20d is not None and r20d > 10:
        reversal_risk = "high"   # extreme overbought surge near resistance → high reversal risk
    elif oversold_signal or (near_support and r20d is not None and r20d < -10):
        reversal_risk = "medium"
    elif overbought_signal and near_resistance:
        reversal_risk = "medium"  # overbought + at resistance → medium reversal risk

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
        "requested_ctx_start": ctx_start,
        "effective_ctx_start": _first_bar_date or ctx_start,
        "ctx_start":           _first_bar_date or ctx_start,
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
        _dir_bias = "Model has SELL bias — overpredicts downside. Consider adjusting toward neutral/BUY."
    elif underpred > overpred * 1.5:
        _dir_bias = "Model has BUY bias — underpredicts downside."
    else:
        _dir_bias = "No strong BUY/SELL directional bias detected."

    # ── Failure category analysis ─────────────────────────────────────────
    hold_count  = sum(1 for r in records
                      if (r.get("ai_prediction", {}).get("decision") or "").upper() == "HOLD")
    hold_rate   = round(hold_count / total * 100, 1)

    # bias_note now includes HOLD bias if applicable
    if hold_rate > 55:
        bias_note = f"{_dir_bias} HOLD bias CRITICAL: {hold_rate}% HOLD rate — model over-selects neutral."
    elif hold_rate > 45:
        bias_note = f"{_dir_bias} HOLD bias WARNING: {hold_rate}% HOLD rate — above ideal range."
    else:
        bias_note = _dir_bias

    false_buy_count  = 0
    false_sell_count = 0
    over_hold_count  = 0
    wrong_magnitude  = 0
    trend_reversal   = 0

    for r in records:
        ai_dec  = (r.get("ai_prediction", {}).get("decision") or "").upper()
        act_dec = (r.get("actual_validation", {}).get("actual_decision") or "").upper()
        if ai_dec == act_dec:
            continue
        if ai_dec == "HOLD" and act_dec in ("BUY", "SELL"):
            over_hold_count += 1
        elif ai_dec == "BUY" and act_dec == "SELL":
            actual_ret = r.get("actual_validation", {}).get("actual_return_pct")
            if actual_ret is not None and float(actual_ret) < -5:
                trend_reversal += 1
            else:
                false_buy_count += 1
        elif ai_dec == "SELL" and act_dec == "BUY":
            false_sell_count += 1

    top_failure = "NONE"
    failure_counts = {
        "OVER_HOLD":      over_hold_count,
        "FALSE_BUY":      false_buy_count,
        "FALSE_SELL":     false_sell_count,
        "WRONG_MAGNITUDE":wrong_magnitude,
        "TREND_REVERSAL_MISSED": trend_reversal,
    }
    if any(v > 0 for v in failure_counts.values()):
        top_failure = max(failure_counts, key=failure_counts.get)

    hold_warning = ""
    if hold_rate > 55:
        hold_warning = (
            f"CRITICAL: HOLD rate is {hold_rate}% — you are severely over-selecting HOLD. "
            "Commit to BUY or SELL when indicators clearly agree."
        )
    elif hold_rate > 45:
        hold_warning = (
            f"CAUTION: HOLD rate is {hold_rate}% — you are over-selecting HOLD. "
            "Commit to BUY or SELL when indicators clearly agree."
        )

    cal_summary = {
        "total_validated_records": total,
        "decision_match_pct":     round(dec_ok / total * 100, 1),
        "direction_match_pct":    round(dir_ok / total * 100, 1),
        "avg_return_error_pp":    avg_ret_err,
        "overprediction_rate":    round(overpred / total * 100, 1),
        "underprediction_rate":   round(underpred / total * 100, 1),
        "hold_rate_pct":          hold_rate,
        "false_buy_count":        false_buy_count,
        "false_sell_count":       false_sell_count,
        "over_hold_count":        over_hold_count,
        "trend_reversal_missed":  trend_reversal,
        "top_failure_reason":     top_failure,
        "symbol_record_count":    len(sym_records),
        "recent_conflicts":       recent_conflicts,
        "bias_note":              bias_note,
        "hold_rate_warning":      hold_warning,
    }
    # Enrich with per-symbol calibration profile if available
    cal_profile_path = ROOT / "data" / "calibration_profiles.json"
    if cal_profile_path.exists() and symbol:
        try:
            cal_profiles = json.loads(cal_profile_path.read_text(encoding="utf-8"))
            sym_profile  = cal_profiles.get("profiles", {}).get(symbol, {})
            if sym_profile.get("calibration_available"):
                cal_summary["symbol_calibration"] = {
                    "match_pct":            sym_profile.get("match_pct"),
                    "hold_rate_pct":        sym_profile.get("hold_rate_pct"),
                    "false_buy_rate_pct":   sym_profile.get("false_buy_rate_pct"),
                    "false_sell_rate_pct":  sym_profile.get("false_sell_rate_pct"),
                    "avg_return_error_pp":  sym_profile.get("avg_return_error_pp"),
                    "top_failure_categories": sym_profile.get("top_failure_categories", []),
                    "volatility_warning":   sym_profile.get("volatility_warning"),
                    "news_earnings_warning":sym_profile.get("news_earnings_warning"),
                    "calibration_summary":  sym_profile.get("calibration_summary", ""),
                }
        except Exception as exc:
            # Degrade gracefully (a malformed calibration file must never break
            # prediction) but don't hide that this enrichment was skipped --
            # callers can surface it instead of silently looking complete.
            cal_summary["symbol_calibration_error"] = f"{type(exc).__name__}: {exc}"
    return cal_summary


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
        elif rsi >= 78:
            signals.append((0.0, 0.5, 0.5))   # extreme overbought → bearish lean + high uncertainty
        elif rsi >= 70:
            signals.append((0.3, 0.0, 0.7))   # overbought → mostly uncertain, mild bull caution
        elif rsi <= 22:
            signals.append((0.5, 0.0, 0.5))   # extreme oversold → bullish lean + high uncertainty
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

    # 4. Recent returns (20d + 60d) — stronger conviction when both agree strongly
    r20 = returns.get("return_20d")
    r60 = returns.get("return_60d")
    if r20 is not None and r60 is not None:
        if r20 > 5 and r60 > 8:          # very strong uptrend
            signals.append((1.0, 0.0, 0.0))
            signals.append((1.0, 0.0, 0.0))  # double weight for strong consensus
        elif r20 > 3 and r60 > 5:
            signals.append((1.0, 0.0, 0.0))
        elif r20 < -5 and r60 < -8:      # very strong downtrend
            signals.append((0.0, 1.0, 0.0))
            signals.append((0.0, 1.0, 0.0))  # double weight for strong consensus
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
    reversal_risk: str = "low",
    rsi: Optional[float] = None,
) -> Tuple[str, str]:
    """Deterministic post-Gemini guardrails — signal scores now gate BUY/SELL."""
    # Gate 1: data quality
    if data_quality_score < 60:
        return "REVIEW", f"Data quality too low ({data_quality_score}/100) — insufficient data."

    # Gate 2: confidence
    if confidence_score < 45:   # was 55
        return "REVIEW", f"Confidence too low ({confidence_score}/100) — mixed signals."

    # Gate 3: high uncertainty blocks directional decisions
    uncert_s = (signal_scores or {}).get("uncertainty_score", 50)
    if uncert_s > 80 and confidence_score < 50:   # was >70 and <65
        return "REVIEW", f"Uncertainty too high ({uncert_s}/100) — needs human review."

    buy_thr, sell_thr = _get_horizon_thresholds(horizon_days)
    bull_s = (signal_scores or {}).get("bullish_score", 50)
    bear_s = (signal_scores or {}).get("bearish_score", 50)

    # Strong consensus override: when bear/bull score is overwhelming (>=70),
    # lower the uncertainty barrier so clear trending markets don't get stuck in HOLD.
    _strong_bear = signal_scores and bear_s >= 70 and (bear_s - bull_s) >= 25
    _strong_bull = signal_scores and bull_s >= 70 and (bull_s - bear_s) >= 25
    _uncert_limit = 55 if (_strong_bear or _strong_bull) else 75

    if predicted_return_pct >= buy_thr:
        # BUY requires signal confirmation
        if signal_scores and bull_s >= 45 and (bull_s - bear_s) >= 5 and uncert_s <= _uncert_limit:
            # Gate 4: overbought reversal block — only for SHORT horizons (≤ 45 days)
            # For horizons > 45 days overbought conditions in strong trends continue more often than reverse.
            _rsi_str = f"{round(rsi, 1)}" if rsi is not None else "N/A"
            if reversal_risk == "high" and horizon_days <= 45:
                decision = "REVIEW"
                reason   = (
                    f"BUY blocked → REVIEW: reversal_risk=high "
                    f"(RSI={_rsi_str}, overbought+near_resistance, horizon={horizon_days}d ≤ 45). "
                    f"Short-horizon overbought reversal risk overrides bullish signals. Needs human review."
                )
            elif reversal_risk == "medium" and rsi is not None and rsi >= 75 and horizon_days <= 45:
                decision = "HOLD"
                reason   = (
                    f"BUY→HOLD: reversal_risk=medium, RSI={_rsi_str} (>=75). "
                    f"Overbought conditions near resistance require caution."
                )
            else:
                decision = "BUY"
                reason   = (
                    f"Predicted return {predicted_return_pct:+.2f}% >= +{buy_thr}% BUY threshold "
                    f"({horizon_days}d). Bullish score {bull_s}/100 confirms."
                )
        else:
            decision = "HOLD"
            reason   = (
                f"Return {predicted_return_pct:+.2f}% above buy threshold but signal scores "
                f"insufficient (bull={bull_s}, bear={bear_s}, uncert={uncert_s})."
            )
    elif predicted_return_pct <= sell_thr:
        # SELL requires signal confirmation; strong consensus lowers uncertainty requirement.
        if signal_scores and bear_s >= 45 and (bear_s - bull_s) >= 5 and uncert_s <= _uncert_limit:
            decision = "SELL"
            reason   = (
                f"Predicted return {predicted_return_pct:+.2f}% <= {sell_thr}% SELL threshold "
                f"({horizon_days}d). Bearish score {bear_s}/100 confirms."
                + (" [STRONG CONSENSUS]" if _strong_bear else "")
            )
        else:
            decision = "HOLD"
            reason   = (
                f"Return {predicted_return_pct:+.2f}% below sell threshold but signal scores "
                f"insufficient (bull={bull_s}, bear={bear_s}, uncert={uncert_s})."
            )
    else:
        decision = "HOLD"
        reason   = (
            f"Return {predicted_return_pct:+.2f}% in HOLD band "
            f"(±{abs(sell_thr):.2f}% for {horizon_days}d horizon)."
        )

    # HOLD override: strong signal consensus overrides neutral return (reversal risk must be low).
    # For overwhelming consensus (>=70 score, >=25 gap), apply slightly relaxed thresholds;
    # otherwise keep original thresholds (confidence>=60, dom_score>=45, near_thr factor 0.50).
    if decision == "HOLD" and signal_scores and reversal_risk == "low":
        dominant   = signal_scores.get("dominant", "uncertain")
        dom_score  = signal_scores.get("dominant_score", 0)
        _conf_thr  = 55 if (_strong_bear or _strong_bull) else 60
        _dom_thr   = 42 if (_strong_bear or _strong_bull) else 45
        _nthr_fac  = 0.40 if (_strong_bear or _strong_bull) else 0.50
        near_thr   = abs(predicted_return_pct) >= abs(sell_thr) * _nthr_fac
        if confidence_score >= _conf_thr and near_thr:
            if dominant == "bullish" and dom_score >= _dom_thr:
                decision = "BUY"
                reason   = (
                    f"HOLD→BUY override: bullish {dom_score}/100, "
                    f"return {predicted_return_pct:+.2f}% near threshold. [{reason}]"
                )
            elif dominant == "bearish" and dom_score >= _dom_thr:
                decision = "SELL"
                reason   = (
                    f"HOLD→SELL override: bearish {dom_score}/100, "
                    f"return {predicted_return_pct:+.2f}% near threshold. [{reason}]"
                )

    # Risk gates (risk alone never creates SELL)
    if risk_score >= _HIGH_RISK_CUTOFF and decision == "BUY":
        return "HOLD", f"BUY downgraded to HOLD: high risk ({risk_score}/100) despite positive return."
    if risk_score >= _HIGH_RISK_CUTOFF and decision == "SELL":
        return "REVIEW", f"SELL → REVIEW: high uncertainty ({risk_score}/100) — needs human review."

    return decision, reason


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_symbol_calibration_section(calibration_summary: dict) -> str:
    """Build per-symbol calibration section for Gemini prompt if available."""
    sym_cal = calibration_summary.get("symbol_calibration", {})
    if not sym_cal:
        return ""
    lines = [
        "SYMBOL-SPECIFIC CALIBRATION PROFILE (highest priority — this symbol's known failure patterns):",
        f"  Decision match rate:    {sym_cal.get('match_pct', 'N/A')}%",
        f"  Hold rate:              {sym_cal.get('hold_rate_pct', 'N/A')}%",
        f"  False BUY rate:         {sym_cal.get('false_buy_rate_pct', 'N/A')}%",
        f"  False SELL rate:        {sym_cal.get('false_sell_rate_pct', 'N/A')}%",
        f"  Avg return error:       {sym_cal.get('avg_return_error_pp', 'N/A')} pp",
        f"  Top failure categories: {', '.join(sym_cal.get('top_failure_categories', [])) or 'NONE'}",
    ]
    if sym_cal.get("volatility_warning"):
        lines.append("  WARNING: Volatility shifts frequently missed on this symbol — weight ATR/BB heavily.")
    if sym_cal.get("news_earnings_warning"):
        lines.append("  WARNING: Earnings/news events caused AI errors — check calendar context.")
    if sym_cal.get("calibration_summary"):
        lines.append(f"  Summary: {sym_cal['calibration_summary']}")
    # Specific guidance from hold rate
    hold_r = sym_cal.get("hold_rate_pct") or 0
    if hold_r > 50:
        lines.append(
            f"  CRITICAL: This symbol has a {hold_r}% HOLD rate — you over-hold on it. "
            "When trend + momentum agree, COMMIT to BUY or SELL. Do NOT default to HOLD."
        )
    return "\n".join(lines) + "\n\n"


def _build_gemini_prompt(
    normalized_input: dict,
    feature_packet: dict,
    calibration_summary: dict,
    signal_scores: dict = None,
    earnings_warning: str = "",
    market_movers_context: str = "",
) -> str:
    fp_json  = json.dumps(feature_packet, indent=2, default=str)
    cal_json = json.dumps(calibration_summary, indent=2, default=str)
    signal_scores_for_prompt = signal_scores or {}
    horizon_days = int(normalized_input.get("decision_horizon_days") or 30)

    # Compute actual data window description from feature packet (never hardcode)
    _n_bars_fp    = int(feature_packet.get("n_bars", 0) or 0)
    _ctx_start_fp = feature_packet.get("effective_ctx_start") or feature_packet.get("ctx_start") or "unknown"
    _years_fp     = round(_n_bars_fp / 252, 1) if _n_bars_fp else 0
    _data_window  = f"{_n_bars_fp} bars (~{_years_fp} years, from {_ctx_start_fp})"
    buy_thr, sell_thr = _get_horizon_thresholds(horizon_days)
    inp_json = json.dumps({
        "symbol":                        normalized_input.get("symbol"),
        "benchmark":                     normalized_input.get("benchmark"),
        "price_basis":                   normalized_input.get("price_basis"),
        "initial_capital":               normalized_input.get("initial_capital"),
        "requested_ctx_start":           normalized_input.get("historical_context_start_date"),
        "effective_ctx_start":           feature_packet.get("effective_ctx_start") or normalized_input.get("historical_context_start_date"),
        "effective_origin_date":         normalized_input.get("prediction_origin_date"),
        "target_date":                   normalized_input.get("target_date"),
        "horizon_days":                  horizon_days,
        "buy_threshold_pct":             buy_thr,
        "sell_threshold_pct":            sell_thr,
    }, indent=2)

    # Build failure category note for prompt
    hold_rate    = calibration_summary.get("hold_rate_pct", 0) or 0
    top_failure  = calibration_summary.get("top_failure_reason", "NONE") or "NONE"
    hold_warning = calibration_summary.get("hold_rate_warning", "") or ""
    over_hold_n  = calibration_summary.get("over_hold_count", 0) or 0
    false_buy_n  = calibration_summary.get("false_buy_count", 0) or 0
    false_sell_n = calibration_summary.get("false_sell_count", 0) or 0
    match_pct    = calibration_summary.get("decision_match_pct", 0) or 0

    failure_guidance = ""
    if top_failure == "OVER_HOLD":
        failure_guidance = (
            f"CRITICAL — OVER_HOLD is the top failure ({over_hold_n} cases). "
            "You have been refusing to commit to BUY/SELL when indicators were clear. "
            "When trend + RSI + MACD all agree, COMMIT to BUY or SELL — do not hide in HOLD."
        )
    elif top_failure == "FALSE_BUY":
        failure_guidance = (
            f"CAUTION — FALSE_BUY occurred {false_buy_n} times. "
            "You predicted BUY but market fell. Add reversal_risk check before committing BUY. "
            "If price is near resistance AND RSI>70, use REVIEW instead of BUY."
        )
    elif top_failure == "FALSE_SELL":
        failure_guidance = (
            f"CAUTION — FALSE_SELL occurred {false_sell_n} times. "
            "You predicted SELL but market rose. When RSI<35 and price near support, avoid SELL. "
            "Use REVIEW or HOLD when oversold signals conflict with bearish trend."
        )
    elif top_failure == "TREND_REVERSAL_MISSED":
        failure_guidance = (
            "CAUTION — Trend reversals were missed. When RSI diverges from price trend direction, "
            "this often signals an impending reversal. Weight reversal_signals heavily."
        )

    return f"""You are a financial decision-support reasoning agent for a walk-forward backtesting system.

STRICT RULES:
1. Analyze ONLY the structured data provided below.
2. Do NOT invent prices, news events, earnings, or external facts not in the data.
3. NOT investment advice — this is a model prediction for validation/research only.
4. Return ONLY valid JSON — no text before or after the JSON.
5. The target_date price is UNKNOWN to you — do NOT reference any price after the origin date.
6. Do NOT use calibration accuracy stats to fake a better prediction. Use them to avoid known failure patterns.

IMPORTANT — SELL BIAS WARNING (NARROW SCOPE):
This warning applies ONLY when ALL THREE are simultaneously true:
  1. RSI is EXTREME oversold (< 28, not just < 35)
  2. Price is explicitly near a major support level (near_support = true)
  3. reversal_risk = "high"
In that specific case, use HOLD or REVIEW instead of SELL.
In ALL OTHER bearish cases — commit to SELL if indicators agree.

STRONG BEARISH MANDATE — when ALL of these are true, you MUST predict SELL with large negative return:
- trend_regime = "bearish"
- price below SMA20 AND SMA50
- MACD < 0 AND MACD_histogram < 0
- 20d return < -3% AND 60d return < -5%
In this case, predict return between -8% and -18% (proportional to ATR and return momentum).
Do NOT default to HOLD when all four bearish conditions are met.

STRONG BULLISH MANDATE — when ALL of these are true, you MUST predict BUY with large positive return:
- trend_regime = "bullish"
- price above SMA20 AND SMA50
- MACD > 0 AND MACD_histogram > 0
- 20d return > +3% AND 60d return > +5%
In this case, predict return between +8% and +18% (proportional to ATR and return momentum).
Do NOT default to HOLD when all four bullish conditions are met.

LONG HORIZON GUIDANCE — when horizon_days > 60:
- Predicting ±1-2% return for a 60-90 day window is almost certainly wrong. Markets move 5-25% over 3 months.
- If ANY directional bias exists (bullish OR bearish trend), predict at least ±5% minimum.
- For strong trend + positive MACD + RSI > 55: predict +8% to +18% (bullish).
- For strong trend + negative MACD + RSI < 45: predict -8% to -18% (bearish).
- Only predict ±2-5% if signals are genuinely split 50/50 with no clear winner.
- A horizon of 72 days with a bullish stock going up consistently = at minimum +8% prediction, not HOLD.

DECISION QUALITY — COMMIT TO BUY OR SELL:
HOLD is only correct when signals genuinely cancel out. It is NOT a safe default.
The system currently has a {hold_rate:.0f}% HOLD rate — this is a critical failure. You MUST be more decisive.

Use HOLD ONLY when ALL of these are true simultaneously:
- Predicted return is within ±1.5% of zero (genuinely flat)
- Bull and bear signals are nearly balanced (neither side > 10pp advantage)
- RSI is between 42-58 (genuinely neutral zone)

OVERBOUGHT REVERSAL GATE (applies only for horizon_days ≤ 45):
If RSI_14 > 75 AND near_resistance = true AND reversal_risk = "medium" or "high" AND horizon_days ≤ 45:
- Do NOT predict BUY. Output REVIEW.
- These exact conditions (RSI > 75 + at resistance) historically precede short-term reversals (1-3 weeks).
- For SHORT horizons (≤ 45 days) this gate applies firmly.

OVERBOUGHT FOR LONG HORIZONS (horizon_days > 45):
If RSI_14 > 75 AND near_resistance = true AND reversal_risk = "high" AND horizon_days > 45:
- RSI overbought is a SHORT-TERM signal only. Over 60-90 days, overbought stocks continue higher more often than they reverse.
- Do NOT output HOLD or REVIEW just because RSI is high. The gate does NOT apply for long horizons.
- If trend_regime = "bullish" AND MACD > 0 AND 60d return > +5%: predict BUY with return +8% to +15%.
- You may note the overbought condition as a risk factor in the rationale, but still commit to BUY if underlying trend is strong.
- A stock that is RSI=79, bullish trend, positive MACD, +22% actual return over 72 days is a BUY — not a HOLD.

USE BUY when ALL of these hold (not just ANY):
- Trend regime = bullish AND RSI > 55 AND RSI < 75 AND MACD histogram > 0
- reversal_risk is "low" (not "medium" or "high")
- NOT (near_resistance = true AND RSI > 70)
- 20d return > 0 AND price above SMA50

USE SELL when ANY of these combinations exist:
- Trend regime = bearish AND RSI < 45 AND MACD histogram < 0
- 20d return < -3% AND 60d return < -5% AND price below SMA50 (COMMIT — do not HOLD)
- RSI crossed below 50 from above AND MACD histogram turning negative AND trend = bearish

RETURN MAGNITUDE GUIDANCE (critical — calibrate your predicted_return_pct to volatility):
- If ATR/price > 1.5% per day AND trend = bearish: predict return between -10% and -18%
- If ATR/price 0.8%-1.5% per day AND trend = bearish: predict return between -5% and -12%
- If ATR/price < 0.8% per day AND trend = bearish: predict return between -2% and -7%
- If ATR/price > 1.5% per day AND trend = bullish: predict return between +10% and +18%
- If ATR/price 0.8%-1.5% per day AND trend = bullish: predict return between +5% and +12%
- If ATR/price < 0.8% per day AND trend = bullish: predict return between +2% and +7%
- NEVER predict 0% when strong directional trend exists
- For horizon > 60 days: NEVER predict less than ±5% if trend is directional

Do NOT default to HOLD when:
- Trend regime is clearly bullish or bearish
- RSI signals momentum (>58 = bullish, <42 = bearish)
- MACD line AND histogram agree on direction
- 20d and 60d returns consistently agree
- reversal_risk is "low" (no overbought/oversold near S/R)

Horizon-calibrated thresholds for this prediction:
- BUY if predicted_return_pct >= +{buy_thr}%
- SELL if predicted_return_pct <= {sell_thr}%
- Only predict HOLD if return is genuinely in the ±{abs(sell_thr)}% neutral zone

If meaningful positive return ({buy_thr}%+) is predicted WITH bullish trend AND positive MACD → commit to BUY.
If meaningful negative return ({sell_thr}% or worse) WITH bearish trend AND negative MACD → commit to SELL.

CALIBRATION FAILURE CONTEXT (learn from these to avoid repeating the same mistakes):
- Current decision match rate: {match_pct}%
- Hold rate: {hold_rate}% {('(TOO HIGH — commit more)' if hold_rate > 45 else '')}
- Top failure reason: {top_failure}
- Over-hold cases: {over_hold_n} | False-buy cases: {false_buy_n} | False-sell cases: {false_sell_n}
{f'- {hold_warning}' if hold_warning else ''}
{f'- {failure_guidance}' if failure_guidance else ''}

NORMALIZED INPUT:
{inp_json}

FEATURE PACKET (computed from real historical bars — no future data):
{fp_json}

CALIBRATION SUMMARY (full past performance — use to avoid known failure patterns):
{cal_json}

{_build_symbol_calibration_section(calibration_summary)}
DATA LIMITATIONS:
- Historical bars come from open-source market data provider (primary) → NASDAQ external → RapidAPI TradingView (plan-dependent)
- {_data_window} of price history available
- No real-time news or macro data
- Options chain live data: NOT available in historical context (would be future leakage for past dates)
{f'EARNINGS ALERT: {earnings_warning}' if earnings_warning else '- Earnings calendar: checked — no events detected in prediction window.'}
{market_movers_context if market_movers_context else '- Market movers: NOT included (historical prediction — live snapshot would be future leakage).'}

SIGNAL SCORES (deterministic, computed from feature packet — these are the objective indicators):
{json.dumps(signal_scores_for_prompt, indent=2)}

REQUIRED JSON RESPONSE (return ONLY this JSON, no other text):
{{
  "status": "SUCCESS",
  "ai_provider": "gemini",
  "predicted_return_pct": <float, clamped to -20.0 to +20.0>,
  "confidence_score": <integer 0-100>,
  "risk_score": <integer 0-100>,
  "decision_threshold_used": {{"buy_threshold": {buy_thr}, "sell_threshold": {sell_thr}}},
  "bullish_score_assessment": <integer 0-100, your assessment of bullish evidence>,
  "bearish_score_assessment": <integer 0-100, your assessment of bearish evidence>,
  "primary_signal": "<single most important indicator driving your prediction>",
  "trend_assessment": "<3-4 sentences: What is the current trend regime (bullish/bearish/sideways)? Where is price relative to SMA20, SMA50, SMA200? How long has this trend been in place? Is the trend strengthening or weakening based on the last 20-60 days of price action?>",
  "momentum_assessment": "<3-4 sentences: What is the exact RSI value and what does it signal (overbought/oversold/neutral)? What direction is MACD histogram pointing and is it accelerating or decelerating? How does the Rate of Change (ROC) confirm or conflict with RSI? Is momentum building or exhausting?>",
  "volatility_assessment": "<2-3 sentences: What does ATR tell us about current volatility relative to its average? Are Bollinger Bands expanding or contracting? Does current volatility support or argue against a directional move in the prediction window?>",
  "benchmark_assessment": "<2-3 sentences: How has this stock performed relative to its benchmark over the past 20 and 60 days? Is it outperforming or underperforming? What does relative strength tell us about market sentiment toward this stock?>",
  "bull_case": "<2-3 sentences: What is the specific price target and catalyst for a bullish outcome? Which combination of indicators most strongly supports upside? What must hold true for this scenario to play out?>",
  "bear_case": "<2-3 sentences: What is the specific price level and catalyst for a bearish outcome? Which indicators signal downside risk? What would trigger further selling pressure?>",
  "main_reason": "<6-8 sentences: Start with WHAT decision was made and WHY that specific decision was reached. Explain WHERE price is positioned (support/resistance/trend levels). Explain HOW the key indicators (RSI, MACD, trend) combine to support this call. State WHEN this analysis is most likely to play out (early, mid, or late in the prediction window). Explain WHAT RISK factors exist. State what CONFIRMATION signal would strengthen conviction. End with the overall confidence level and what would change the call.>",
  "hold_reason": "<only if decision is HOLD — 3-4 sentences explaining exactly which bullish signals cancel which bearish signals and why neither side has sufficient edge to commit>",
  "why_not_opposite_decision": "<3-4 sentences: Explain specifically why the opposite direction was rejected. Which indicators argued for the opposite? Why were those signals overruled? What would need to change for the opposite call to be valid?>",
  "invalidating_conditions": ["<specific price level or indicator reading that would invalidate this call>", "<second invalidating condition>", "<third invalidating condition>"],
  "calibration_notes": "<2 sentences on what past failure patterns you avoided in this prediction and how calibration history influenced your confidence level>",
  "key_features_used": ["list of top 5 features that most influenced this prediction — be specific, e.g. 'RSI_14=67.3 (bullish momentum zone)' not just 'RSI'"],
  "data_limitations": ["list of 1-3 specific data gaps that could affect accuracy — be specific about what is missing and how it matters"],
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
    earnings_warning: str = "",
    market_movers_context: str = "",
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

    sig_scores     = _compute_signal_scores(feature_packet)
    prompt_content = _build_gemini_prompt(normalized_input, feature_packet, calibration_summary, signal_scores=sig_scores, earnings_warning=earnings_warning, market_movers_context=market_movers_context)
    prompt_hash    = hashlib.sha256(prompt_content.encode()).hexdigest()[:16]

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

        gemini_latency_ms = round((time.time() - t0) * 1000, 0)
        raw_text = (response.text or "").strip()

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
        dq    = min(100, int(feature_packet.get("data_quality_score", 70) or 70))
        h_days = int(normalized_input.get("decision_horizon_days") or 30)

        _rev_risk = feature_packet.get("reversal_signals", {}).get("reversal_risk", "low")
        _rsi_val  = feature_packet.get("momentum", {}).get("RSI_14")
        decision, decision_reason = _apply_guardrails(
            pred_return, conf, risk, dq,
            horizon_days=h_days,
            signal_scores=sig_scores,
            reversal_risk=_rev_risk,
            rsi=_rsi_val,
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

    # Pre-flight assertion: block if any bar date is after origin date
    _target_in_history = any(b["date"] > origin_str for b in history)
    if _target_in_history:
        return _error_result(spi, input_hash, "LEAKAGE_BLOCKED: bars after origin date detected in history.")

    # Origin price — respect price_basis setting (high/low/open/close)
    _price_basis = str(spi.get("price_basis", "close") or "close")
    origin_price, eff_origin_date, price_err = get_price_on_date(history, origin_str, price_basis=_price_basis)
    if origin_price is None:
        return _error_result(spi, input_hash, f"Cannot get origin price on {origin_str}: {price_err}")

    # Optional context enrichments (benchmark bars, earnings calendar, market
    # movers) must never break a prediction if they fail -- but a silently
    # swallowed failure here previously looked identical to "this symbol
    # simply has no earnings/no benchmark data," when the real cause could be
    # a flaky API, missing key, or bad response. Track failures explicitly so
    # they can be surfaced instead of just disappearing.
    _context_enrichment_warnings: Dict[str, str] = {}

    # Fetch benchmark bars so relative strength is computed for Gemini
    benchmark_sym = str(spi.get("benchmark", "") or "").strip().upper()
    _benchmark_bars: Optional[List[dict]] = None
    if benchmark_sym and benchmark_sym != symbol:
        try:
            from historical_price_service import fetch_price_history_for_range as _fpr
            _bm_bars, _bm_err, _bm_cov = _fpr(benchmark_sym, ctx_start, origin_str)
            if _bm_bars:
                _benchmark_bars = [b for b in _bm_bars if b["date"] <= origin_str]
            else:
                _context_enrichment_warnings["benchmark_bars"] = _bm_err or "No benchmark bars returned."
        except Exception as exc:
            _benchmark_bars = None
            _context_enrichment_warnings["benchmark_bars"] = f"{type(exc).__name__}: {exc}"

    # Build feature packet
    feature_packet = build_stock_feature_packet(spi, history, benchmark_bars=_benchmark_bars)
    if feature_packet.get("status") != "OK":
        return _error_result(spi, input_hash, f"Feature packet failed: {feature_packet.get('error')}")

    # Calibration summary
    calibration_summary = build_calibration_summary(symbol=symbol, horizon_days=horizon_days)
    if calibration_summary.get("symbol_calibration_error"):
        _context_enrichment_warnings["symbol_calibration"] = calibration_summary["symbol_calibration_error"]

    # Fetch earnings calendar for prediction window
    _earnings_warning = ""
    try:
        from tools.tradingview_calendar import get_earnings_calendar
        from datetime import datetime as _dt
        _origin_ts = int(_dt.strptime(origin_str[:10], "%Y-%m-%d").timestamp())
        _target_ts = int(_dt.strptime(target_str[:10], "%Y-%m-%d").timestamp()) if target_str else _origin_ts + 90*86400
        _earn_data = get_earnings_calendar(_origin_ts, _target_ts)
        if _earn_data.get("status") == "SUCCESS":
            _earn_events = _earn_data.get("data", []) or []
            if isinstance(_earn_events, list):
                _sym_earnings = [e for e in _earn_events if str(e.get("ticker","") or "").upper() == symbol]
                if _sym_earnings:
                    _earn_dates = [str(e.get("date","") or e.get("time",""))[:10] for e in _sym_earnings]
                    _earnings_warning = f"EARNINGS WARNING: {symbol} has earnings in prediction window on {', '.join(_earn_dates)}. Earnings events cause high volatility and often invalidate technical predictions. Raise risk_score by 15-20 and lower confidence_score accordingly."
        else:
            _context_enrichment_warnings["earnings_calendar"] = str(_earn_data.get("message") or _earn_data.get("status") or "unavailable")
    except Exception as exc:
        _context_enrichment_warnings["earnings_calendar"] = f"{type(exc).__name__}: {exc}"

    # Fetch live market movers for LIVE predictions only (leakage-safe: origin within last 7 days)
    _market_movers_context = ""
    try:
        from datetime import date as _date_cls
        _origin_date_obj = _parse_ymd(origin_str)
        _days_ago = (_date_cls.today() - _origin_date_obj).days if _origin_date_obj else 999
        if _days_ago <= 7:
            from rapidapi_market_service import fetch_top_movers_for_gemini
            _movers_result = fetch_top_movers_for_gemini()
            if _movers_result.get("status") == "SUCCESS":
                _market_movers_context = _movers_result.get("context_text", "")
            else:
                _context_enrichment_warnings["market_movers"] = str(_movers_result.get("message") or _movers_result.get("status") or "unavailable")
    except Exception as exc:
        _context_enrichment_warnings["market_movers"] = f"{type(exc).__name__}: {exc}"

    # Call Gemini
    gemini_result = _call_gemini(
        spi, feature_packet, calibration_summary, origin_price, initial_cap,
        earnings_warning=_earnings_warning,
        market_movers_context=_market_movers_context,
    )

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
        "requested_ctx_start":          spi.get("historical_context_start_date"),
        "effective_ctx_start":          feature_packet.get("effective_ctx_start") or spi.get("historical_context_start_date"),
        "historical_context_start_date": feature_packet.get("effective_ctx_start") or spi.get("historical_context_start_date"),
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
        "market_movers_used":           bool(_market_movers_context),
        "context_enrichment_warnings":  _context_enrichment_warnings,
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
