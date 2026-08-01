"""
Stock Comparison Engine
Standalone pure-math module for computing actual stock metrics and comparing AI vs actual.

No API calls. No Streamlit. Importable anywhere including test harness.

Key entry points:
- compute_actual_metrics(origin_price, target_price, initial_capital)
    Pure formula — testable without any API key.
- compare_ai_vs_actual(ai_prediction, actual_validation)
    Compare AI output vs actual validation, return structured comparison.
- validate_no_leakage(bars, cutoff_date)
    Assert no bar in bars has date > cutoff_date.
"""
from __future__ import annotations

from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_BUY_THRESHOLD  =  2.0   # return_pct > +2% → BUY
_SELL_THRESHOLD = -2.0   # return_pct < -2% → SELL


# ══════════════════════════════════════════════════════════════════════════════
# PURE FORMULA — NO API NEEDED
# ══════════════════════════════════════════════════════════════════════════════

def compute_actual_metrics(
    origin_price: float,
    target_price: float,
    initial_capital: float,
) -> dict:
    """
    Compute actual return, final capital, P&L, and decision from raw prices.

    Formula (canonical):
        actual_return_pct    = ((target_price - origin_price) / origin_price) * 100
        actual_final_capital = initial_capital * (target_price / origin_price)
        actual_total_pl      = actual_final_capital - initial_capital

    Decision thresholds:
        actual_return_pct >  +2%  → BUY
        actual_return_pct <  -2%  → SELL
        -2% <= actual_return_pct <= +2% → HOLD

    Returns dict with all computed values. Does NOT round — caller rounds if needed.
    """
    if origin_price <= 0:
        return {
            "status":  "FAILED",
            "error":   f"origin_price must be > 0 (got {origin_price})",
        }
    if target_price <= 0:
        return {
            "status":  "FAILED",
            "error":   f"target_price must be > 0 (got {target_price})",
        }
    if initial_capital <= 0:
        return {
            "status":  "FAILED",
            "error":   f"initial_capital must be > 0 (got {initial_capital})",
        }

    actual_return_pct    = (target_price - origin_price) / origin_price * 100.0
    actual_final_capital = initial_capital * (target_price / origin_price)
    actual_total_pl      = actual_final_capital - initial_capital

    if actual_return_pct > _BUY_THRESHOLD:
        decision = "BUY"
    elif actual_return_pct < _SELL_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    return {
        "status":                "SUCCESS",
        "origin_price":          origin_price,
        "target_price":          target_price,
        "initial_capital":       initial_capital,
        "actual_return_pct":     actual_return_pct,
        "actual_final_capital":  actual_final_capital,
        "actual_total_pl":       actual_total_pl,
        "actual_decision":       decision,
    }


def decision_from_return(return_pct: float) -> str:
    """Deterministic BUY / SELL / HOLD from return_pct. Canonical single source of truth."""
    if return_pct > _BUY_THRESHOLD:  return "BUY"
    if return_pct < _SELL_THRESHOLD: return "SELL"
    return "HOLD"


# ══════════════════════════════════════════════════════════════════════════════
# LEAKAGE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_no_leakage(bars: List[dict], cutoff_date: str) -> dict:
    """
    Verify no bar in bars has date > cutoff_date.

    Returns:
        {"status": "CLEAN", "n_bars": N, "cutoff_date": cutoff_date}
        {"status": "LEAKAGE_DETECTED", "leaked_bars": [...], ...}
    """
    leaked = [b for b in bars if b.get("date", "") > cutoff_date]
    if leaked:
        return {
            "status":        "LEAKAGE_DETECTED",
            "cutoff_date":   cutoff_date,
            "n_bars":        len(bars),
            "leaked_bars":   leaked,
            "error":         (
                f"{len(leaked)} bar(s) found after cutoff_date {cutoff_date}. "
                f"Latest leaked date: {max(b['date'] for b in leaked)}. "
                "Future data must not be passed to AI prediction."
            ),
        }
    return {
        "status":      "CLEAN",
        "n_bars":      len(bars),
        "cutoff_date": cutoff_date,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI vs ACTUAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_ai_vs_actual(
    ai_prediction:     dict,
    actual_validation: dict,
) -> dict:
    """
    Compare AI prediction against actual historical validation.

    Blocks comparison if:
    - AI status != SUCCESS
    - actual status != SUCCESS
    - hash mismatch
    - missing target prices
    - missing final capitals
    - missing return percentages

    Returns structured comparison dict. Never raises.
    """
    # ── Status guards ─────────────────────────────────────────────────────────
    if not isinstance(ai_prediction, dict):
        return _fail_cmp("ai_prediction is not a dict")
    if not isinstance(actual_validation, dict):
        return _fail_cmp("actual_validation is not a dict")

    if ai_prediction.get("status") != "SUCCESS":
        return _fail_cmp(
            f"AI prediction status is '{ai_prediction.get('status', 'MISSING')}' "
            "— comparison blocked until AI prediction succeeds."
        )
    if actual_validation.get("status") != "SUCCESS":
        return _fail_cmp(
            f"Actual validation status is '{actual_validation.get('status', 'MISSING')}' "
            "— comparison blocked until actual validation succeeds."
        )

    # ── Hash match guard ───────────────────────────────────────────────────────
    ai_hash  = str(ai_prediction.get("stock_prediction_input_hash", "") or "")
    val_hash = str(actual_validation.get("stock_prediction_input_hash", "") or "")
    if ai_hash and val_hash and ai_hash != val_hash:
        return _fail_cmp(
            f"Hash mismatch: AI hash={ai_hash!r}, validation hash={val_hash!r}. "
            "Both must be derived from the same StockPredictionInput."
        )

    # ── Required values ───────────────────────────────────────────────────────
    ai_pred_price = ai_prediction.get("predicted_target_price")
    ai_pred_ret   = ai_prediction.get("predicted_return_pct")
    ai_pred_cap   = ai_prediction.get("predicted_final_capital")
    ai_pred_pl    = ai_prediction.get("predicted_total_pl")
    ai_decision   = str(ai_prediction.get("decision", "") or "REVIEW")

    actual_price  = actual_validation.get("target_price")
    actual_ret    = actual_validation.get("actual_return_pct")
    actual_cap    = actual_validation.get("actual_final_capital")
    actual_pl     = actual_validation.get("actual_total_pl")
    actual_dec    = str(actual_validation.get("actual_decision", "") or "REVIEW")

    missing = []
    if ai_pred_price  is None: missing.append("ai_prediction.predicted_target_price")
    if ai_pred_ret    is None: missing.append("ai_prediction.predicted_return_pct")
    if ai_pred_cap    is None: missing.append("ai_prediction.predicted_final_capital")
    if actual_price   is None: missing.append("actual_validation.target_price")
    if actual_ret     is None: missing.append("actual_validation.actual_return_pct")
    if actual_cap     is None: missing.append("actual_validation.actual_final_capital")

    if missing:
        return _fail_cmp(f"Missing required fields: {', '.join(missing)}")

    if float(ai_pred_price) <= 0:
        return _fail_cmp(f"AI predicted_target_price is invalid ({ai_pred_price})")
    if float(actual_price) <= 0:
        return _fail_cmp(f"Actual target_price is invalid ({actual_price})")

    # ── Compute errors ────────────────────────────────────────────────────────
    ai_pred_price = float(ai_pred_price)
    ai_pred_ret   = float(ai_pred_ret)
    ai_pred_cap   = float(ai_pred_cap)
    ai_pred_pl    = float(ai_pred_pl) if ai_pred_pl is not None else (ai_pred_cap - float(actual_validation.get("initial_capital", 0) or 0))
    actual_price  = float(actual_price)
    actual_ret    = float(actual_ret)
    actual_cap    = float(actual_cap)
    actual_pl     = float(actual_pl) if actual_pl is not None else 0.0

    target_price_error      = ai_pred_price - actual_price
    return_error_pct        = ai_pred_ret   - actual_ret
    absolute_return_error   = abs(return_error_pct)
    capital_error           = ai_pred_cap   - actual_cap
    absolute_capital_error  = abs(capital_error)
    pl_error                = ai_pred_pl    - actual_pl

    capital_error_pct = (capital_error / actual_cap * 100.0) if actual_cap else None

    # ── Decision and direction match ──────────────────────────────────────────
    decision_match    = (ai_decision == actual_dec)
    directional_match = (
        (ai_pred_ret > 0 and actual_ret > 0) or
        (ai_pred_ret < 0 and actual_ret < 0) or
        (abs(ai_pred_ret) <= _BUY_THRESHOLD and abs(actual_ret) <= _BUY_THRESHOLD)
    )

    # ── Quality classification ────────────────────────────────────────────────
    if not decision_match:
        quality_classification = "DECISION_MISMATCH"
    elif absolute_return_error <= 10.0:
        quality_classification = "CLEAN_MATCH"
    elif absolute_return_error <= 15.0:
        quality_classification = "DIRECTION_MATCH_MINOR_ERROR"
    elif absolute_return_error <= 30.0:
        quality_classification = "LOW_QUALITY_MATCH"
    elif absolute_return_error <= 75.0:
        quality_classification = "MAGNITUDE_FAILURE"
    else:
        quality_classification = "SEVERE_MAGNITUDE_FAILURE"

    agreement = "MATCH" if decision_match else "CONFLICT"
    # REVIEW on conflict: do not blindly use actual_dec — signal unreliable when engines disagree
    if agreement == "MATCH":
        final_decision = ai_decision   # both agree, use the agreed decision
    else:
        final_decision = "REVIEW"      # conflict means uncertainty — human review required

    note = _build_note(agreement, ai_decision, actual_dec, return_error_pct, capital_error)

    return {
        "status":                  "SUCCESS",
        "agreement":               agreement,
        "final_decision":          final_decision,
        "decision_match":          decision_match,
        "directional_match":       directional_match,
        "ai_decision":             ai_decision,
        "actual_decision":         actual_dec,
        "ai_predicted_price":      round(ai_pred_price, 4),
        "actual_price":            round(actual_price, 4),
        "target_price_error":      round(target_price_error, 4),
        "abs_target_price_error":  round(abs(target_price_error), 4),
        "ai_predicted_return_pct": round(ai_pred_ret, 4),
        "actual_return_pct":       round(actual_ret, 4),
        "return_error_pct":        round(return_error_pct, 4),
        "absolute_return_error_pct": round(absolute_return_error, 4),
        "ai_predicted_capital":    round(ai_pred_cap, 4),
        "actual_capital":          round(actual_cap, 4),
        "capital_error":           round(capital_error, 4),
        "absolute_capital_error":  round(absolute_capital_error, 4),
        "capital_error_pct":       round(capital_error_pct, 4) if capital_error_pct is not None else None,
        "pl_error":                round(pl_error, 4),
        "abs_pl_error":            round(abs(pl_error), 4),
        "agreement_note":          note,
        # Legacy aliases for callers that use old field names
        "price_error":             round(target_price_error, 4),
        "abs_price_error":         round(abs(target_price_error), 4),
        "abs_return_error_pct":    round(absolute_return_error, 4),
        "abs_capital_error":       round(absolute_capital_error, 4),
        # Quality classification — never treat huge return error as clean success
        "quality_classification":  quality_classification,
    }


def _fail_cmp(reason: str) -> dict:
    return {
        "status":                "FAILED",
        "agreement":             "UNVERIFIED",
        "final_decision":        "REVIEW",
        "decision_match":        False,
        "directional_match":     False,
        "quality_classification": "UNVERIFIED",
        "error":                 reason,
    }


def _build_note(agreement: str, ai_dec: str, act_dec: str,
                ret_err: float, cap_err: float) -> str:
    if agreement == "MATCH":
        return (
            f"AI predicted {ai_dec} and actual result was {act_dec}. "
            f"Return error: {ret_err:+.2f}pp, capital error: ${cap_err:+,.2f}."
        )
    return (
        f"AI predicted {ai_dec} but actual result was {act_dec} — CONFLICT. "
        f"Ground-truth decision ({act_dec}) is used as final signal. "
        f"Return error: {ret_err:+.2f}pp, capital error: ${cap_err:+,.2f}."
    )
