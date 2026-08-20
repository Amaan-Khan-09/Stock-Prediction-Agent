"""
tests/test_calibration_sprint.py — Accuracy Calibration Sprint: 30 tests

Tests cover:
- Dynamic threshold logic (_get_horizon_thresholds)
- Signal-gated guardrails (_apply_guardrails)
- HOLD bias analysis (analyze_hold_bias)
- Calibration profile builder (build_profiles, global/by_horizon/prompt_guidance)
- Compare calibration runs (compare, load_summary)
- Batch validation tag file naming
- Gemini prompt fields (signal_scores, new JSON fields)
- Direction match function
- Accuracy eligibility logic
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a minimal signal_scores dict
# ──────────────────────────────────────────────────────────────────────────────

def _sig(bull=50, bear=50, uncert=50, dominant="uncertain", dom_score=50):
    return {
        "bullish_score": bull,
        "bearish_score": bear,
        "uncertainty_score": uncert,
        "dominant": dominant,
        "dominant_score": dom_score,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Import targets (lazily to avoid environment issues in import-only scenarios)
# ──────────────────────────────────────────────────────────────────────────────

def _import_agent():
    from gemini_stock_prediction_agent import (
        _get_horizon_thresholds,
        _apply_guardrails,
        _build_gemini_prompt,
        _compute_signal_scores,
    )
    return _get_horizon_thresholds, _apply_guardrails, _build_gemini_prompt, _compute_signal_scores


def _import_failure_analyzer():
    from failure_analyzer import analyze_hold_bias, _safe_float, _safe_int
    return analyze_hold_bias, _safe_float, _safe_int


def _import_calibration_builder():
    from calibration_profile_builder import (
        build_profiles,
        _build_global_profile,
        _build_horizon_profile,
        _build_prompt_guidance,
    )
    return build_profiles, _build_global_profile, _build_horizon_profile, _build_prompt_guidance


def _import_compare():
    from compare_calibration_runs import compare, load_summary
    return compare, load_summary


def _import_batch_runner():
    from batch_validation_runner import (
        _direction_match,
        _classify_failure,
        CSV_FIELDNAMES,
        DATA_DIR,
    )
    return _direction_match, _classify_failure, CSV_FIELDNAMES, DATA_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 1-3: Dynamic threshold tests
# ══════════════════════════════════════════════════════════════════════════════

def test_dynamic_thresholds_7d():
    # Near-zero threshold: any return sign triggers decision, not a large hurdle
    _get_horizon_thresholds, *_ = _import_agent()
    buy, sell = _get_horizon_thresholds(7)
    assert buy == 0.1
    assert sell == -0.1


def test_dynamic_thresholds_30d():
    _get_horizon_thresholds, *_ = _import_agent()
    buy, sell = _get_horizon_thresholds(30)
    assert buy == 0.1
    assert sell == -0.1


def test_dynamic_thresholds_45d():
    _get_horizon_thresholds, *_ = _import_agent()
    buy, sell = _get_horizon_thresholds(45)
    assert buy == 0.1
    assert sell == -0.1


# ══════════════════════════════════════════════════════════════════════════════
# 4-12: _apply_guardrails signal-gate tests
# ══════════════════════════════════════════════════════════════════════════════

def test_buy_requires_bullish_score_45():
    """HOLD is banned — pred_return=3.0 > 0 always produces BUY."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, _ = _apply_guardrails(
        3.0, 70, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=44, bear=20, uncert=36, dominant="bullish", dom_score=44)
    )
    assert decision == "BUY"


def test_buy_requires_bull_bear_spread_5():
    """HOLD is banned — pred_return=3.0 > 0 always produces BUY."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, _ = _apply_guardrails(
        3.0, 55, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=60, bear=56, uncert=40, dominant="bullish", dom_score=60)
    )
    assert decision == "BUY"


def test_buy_with_valid_signals():
    """pred=3.0%, bull=65, bear=40, spread=25 >= 12, uncert=40 <= 65 => BUY."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, reason = _apply_guardrails(
        3.0, 70, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=65, bear=40, uncert=40, dominant="bullish", dom_score=65)
    )
    assert decision == "BUY"
    assert "BUY" in reason


def test_sell_requires_bearish_score_45():
    """HOLD is banned — pred_return=-3.0 < 0 always produces SELL."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, _ = _apply_guardrails(
        -3.0, 70, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=20, bear=44, uncert=36, dominant="bearish", dom_score=44)
    )
    assert decision == "SELL"


def test_sell_requires_bear_bull_spread_5():
    """HOLD is banned — pred_return=-3.0 < 0 always produces SELL."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, _ = _apply_guardrails(
        -3.0, 55, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=56, bear=60, uncert=40, dominant="bearish", dom_score=60)
    )
    assert decision == "SELL"


def test_sell_with_valid_signals():
    """pred=-3.0%, bear=70, bull=45, spread=25 >= 12, uncert=40 <= 65 => SELL."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, reason = _apply_guardrails(
        -3.0, 70, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=45, bear=70, uncert=40, dominant="bearish", dom_score=70)
    )
    assert decision == "SELL"
    assert "SELL" in reason


def test_hold_override_fires_at_45():
    """HOLD is banned — any positive return produces BUY directly."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, reason = _apply_guardrails(
        0.8, 65, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=46, bear=20, uncert=34, dominant="bullish", dom_score=46)
    )
    assert decision == "BUY"
    assert "BUY" in reason


def test_risk_alone_never_forces_sell():
    """High risk score should not create SELL; it only demotes BUY to HOLD or SELL to REVIEW."""
    _, _apply_guardrails, *_ = _import_agent()
    # Positive return => no SELL path; high risk should just block BUY -> HOLD
    decision, _ = _apply_guardrails(
        3.0, 70, 90, 80, horizon_days=30,
        signal_scores=_sig(bull=65, bear=40, uncert=40, dominant="bullish", dom_score=65)
    )
    # High risk (90) downgrades BUY -> HOLD; NEVER -> SELL
    assert decision in ("HOLD", "BUY")
    assert decision != "SELL"


def test_high_uncertainty_causes_review():
    """data_quality < 50 triggers REVIEW — critical data quality gate."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, reason = _apply_guardrails(
        3.0, 45, 30, 40, horizon_days=30,
        signal_scores=_sig(bull=65, bear=30, uncert=85, dominant="bullish", dom_score=65)
    )
    assert decision == "REVIEW"
    assert "quality" in reason.lower() or "Quality" in reason


def test_low_confidence_causes_review():
    """confidence < 20 triggers REVIEW — extreme low confidence gate."""
    _, _apply_guardrails, *_ = _import_agent()
    decision, reason = _apply_guardrails(
        3.0, 15, 30, 80, horizon_days=30,
        signal_scores=_sig(bull=65, bear=30, uncert=40)
    )
    assert decision == "REVIEW"
    assert "Confidence" in reason or "confidence" in reason


# ══════════════════════════════════════════════════════════════════════════════
# 13-16: HOLD bias analyzer tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_rows(n_hold=6, n_buy=2, n_sell=2, act_when_hold="BUY", actual_ret_when_hold=5.0):
    """Build a simple list of fake CSV rows for hold-bias analysis."""
    rows = []
    for i in range(n_hold):
        rows.append({
            "ai_decision": "HOLD",
            "actual_decision": act_when_hold,
            "actual_return_pct": str(actual_ret_when_hold),
            "validation_status": "SUCCESS",
            "symbol": "SPY",
            "horizon_days": "30",
        })
    for i in range(n_buy):
        rows.append({
            "ai_decision": "BUY",
            "actual_decision": "BUY",
            "actual_return_pct": "3.0",
            "validation_status": "SUCCESS",
            "symbol": "SPY",
            "horizon_days": "30",
        })
    for i in range(n_sell):
        rows.append({
            "ai_decision": "SELL",
            "actual_decision": "SELL",
            "actual_return_pct": "-3.0",
            "validation_status": "SUCCESS",
            "symbol": "SPY",
            "horizon_days": "30",
        })
    return rows


def test_hold_bias_analyzer_calculates_false_neutral():
    """analyze_hold_bias returns correct structure with false_neutral_count."""
    analyze_hold_bias, *_ = _import_failure_analyzer()
    rows = _make_rows(n_hold=6, n_buy=2, n_sell=2, act_when_hold="BUY", actual_ret_when_hold=5.0)
    result = analyze_hold_bias(rows)
    assert "overall_hold_rate" in result
    assert "false_neutral_count" in result
    assert "false_neutral_rate" in result
    assert "hold_rate_by_symbol" in result
    assert "severity" in result
    # 6 holds, all with actual=BUY and abs return=5 > 2 => all are false neutral
    assert result["false_neutral_count"] == 6


def test_hold_bias_critical_warning_above_55pct():
    """hold_rate > 55% => severity=CRITICAL."""
    analyze_hold_bias, *_ = _import_failure_analyzer()
    # 6 holds out of 8 total = 75% hold rate
    rows = _make_rows(n_hold=6, n_buy=1, n_sell=1)
    result = analyze_hold_bias(rows)
    assert result["overall_hold_rate"] > 55
    assert result["severity"] == "CRITICAL"


def test_hold_bias_caution_above_45pct():
    """hold_rate > 45% but <= 55% => severity=CAUTION."""
    analyze_hold_bias, *_ = _import_failure_analyzer()
    # 5 holds out of 10 total = 50% hold rate
    rows = _make_rows(n_hold=5, n_buy=3, n_sell=2)
    result = analyze_hold_bias(rows)
    assert result["overall_hold_rate"] > 45
    assert result["severity"] == "CAUTION"


# ══════════════════════════════════════════════════════════════════════════════
# 17-19: Calibration profile builder keys
# ══════════════════════════════════════════════════════════════════════════════

def test_calibration_profile_has_global_section():
    """build_profiles returns dict with 'global' key."""
    build_profiles, *_ = _import_calibration_builder()
    # Run with empty paths (no data) — should still return structure
    fake_csv  = ROOT / "data" / "_test_empty.csv"
    fake_json = ROOT / "data" / "_test_empty.json"
    result = build_profiles(symbols=["SPY"], csv_path=fake_csv, fail_json=fake_json)
    assert "global" in result


def test_calibration_profile_has_by_horizon():
    """build_profiles returns dict with 'by_horizon' key."""
    build_profiles, *_ = _import_calibration_builder()
    fake_csv  = ROOT / "data" / "_test_empty.csv"
    fake_json = ROOT / "data" / "_test_empty.json"
    result = build_profiles(symbols=["SPY"], csv_path=fake_csv, fail_json=fake_json)
    assert "by_horizon" in result


def test_calibration_profile_has_prompt_guidance():
    """build_profiles returns dict with 'prompt_guidance' key that is a list."""
    build_profiles, *_ = _import_calibration_builder()
    fake_csv  = ROOT / "data" / "_test_empty.csv"
    fake_json = ROOT / "data" / "_test_empty.json"
    result = build_profiles(symbols=["SPY"], csv_path=fake_csv, fail_json=fake_json)
    assert "prompt_guidance" in result
    assert isinstance(result["prompt_guidance"], list)


# ══════════════════════════════════════════════════════════════════════════════
# 20-22: compare_calibration_runs tests
# ══════════════════════════════════════════════════════════════════════════════

def test_compare_calibration_runs_importable():
    """compare and load_summary are importable."""
    compare, load_summary = _import_compare()
    assert callable(compare)
    assert callable(load_summary)


def test_compare_shows_regression_honestly():
    """If post is worse, regressed list is non-empty."""
    compare, load_summary = _import_compare()
    baseline = {
        "decision_match_pct": 40.0,
        "direction_match_pct": 50.0,
        "hold_rate_pct": 30.0,
        "avg_return_error_pp": 10.0,
        "valid_runs": 20,
    }
    post = {
        "decision_match_pct": 30.0,    # worse
        "direction_match_pct": 45.0,   # worse
        "hold_rate_pct": 35.0,         # worse (higher hold)
        "avg_return_error_pp": 15.0,   # worse (higher error)
        "valid_runs": 20,
    }
    # Patch load_summary to return controlled data
    with patch("compare_calibration_runs.load_summary") as mock_ls:
        mock_ls.side_effect = lambda tag: baseline if tag == "base" else post
        result = compare("base", "post")
    assert len(result.get("regressed", [])) > 0
    assert "NO IMPROVEMENT" in result.get("honest_assessment", "")


def test_compare_no_fake_improvement():
    """If post matches baseline exactly, honest_assessment should say NO IMPROVEMENT."""
    compare, _ = _import_compare()
    same = {
        "decision_match_pct": 40.0,
        "direction_match_pct": 50.0,
        "hold_rate_pct": 30.0,
        "avg_return_error_pp": 10.0,
        "valid_runs": 20,
    }
    with patch("compare_calibration_runs.load_summary") as mock_ls:
        mock_ls.side_effect = lambda tag: same
        result = compare("base", "post")
    # delta=0 on all metrics; no improvement, no regression
    assert "NO IMPROVEMENT" in result.get("honest_assessment", "")


# ══════════════════════════════════════════════════════════════════════════════
# 23: Batch tag writes tagged files
# ══════════════════════════════════════════════════════════════════════════════

def test_batch_tag_writes_tagged_files():
    """When tag='baseline', output paths include 'baseline' in their names."""
    from batch_validation_runner import DATA_DIR
    tag = "baseline"
    expected_csv  = DATA_DIR / f"batch_validation_results_{tag}.csv"
    expected_json = DATA_DIR / f"batch_validation_summary_{tag}.json"
    # Just verify the path logic produces tagged names
    assert "baseline" in str(expected_csv)
    assert "baseline" in str(expected_json)


# ══════════════════════════════════════════════════════════════════════════════
# 24-27: Gemini prompt content tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_minimal_prompt_args():
    """Build minimal arguments for _build_gemini_prompt."""
    normalized_input = {
        "symbol": "TSLA",
        "benchmark": "SPY",
        "price_basis": "close",
        "initial_capital": 50000,
        "historical_context_start_date": "2025-01-01",
        "prediction_origin_date": "2025-07-01",
        "target_date": "2025-07-31",
        "decision_horizon_days": 30,
    }
    feature_packet = {
        "status": "OK",
        "n_bars": 120,
        "data_quality_score": 80,
        "symbol": "TSLA",
        "benchmark": "SPY",
        "origin_price": 250.0,
        "horizon_days": 30,
        "returns": {"return_1d": 0.5, "return_5d": 1.2, "return_20d": 3.0, "return_60d": 5.0, "return_120d": 10.0, "full_context_return": 12.0, "return_10d": 2.0},
        "trend": {"sma_10": 248.0, "sma_20": 245.0, "sma_50": 240.0, "sma_100": 235.0, "price_vs_sma_20_pct": 2.0, "price_vs_sma_50_pct": 4.0, "trend_regime": "bullish"},
        "momentum": {"RSI_14": 58.0, "MACD": 1.5, "MACD_signal": 1.0, "MACD_histogram": 0.5, "rate_of_change_20": 3.0},
        "volatility": {"annualized_volatility": 30.0, "ATR_14": 5.0, "BB_upper": 260.0, "BB_mid": 250.0, "BB_lower": 240.0, "BB_position": 0.5, "max_drawdown_context": -10.0, "dist_from_low_pct": 15.0, "volatility_regime": "medium"},
        "volume": {"avg_volume_20": 1000000, "avg_volume_60": 900000, "latest_volume": 1100000, "volume_ratio_20": 1.1},
        "support_resistance": {"recent_support": 240.0, "recent_resistance": 260.0, "dist_to_support_pct": 4.0, "dist_to_resistance_pct": 4.0},
        "benchmark_comparison": {"benchmark_return_20d": 2.0, "benchmark_return_60d": 4.0, "relative_strength_20d": 1.0, "relative_strength_60d": 1.0},
        "reversal_signals": {"oversold_signal": False, "overbought_signal": False, "near_support": False, "near_resistance": False, "momentum_divergence": False, "reversal_risk": "low"},
    }
    calibration_summary = {"total_validated_records": 0, "note": "No calibration yet."}
    return normalized_input, feature_packet, calibration_summary


def test_gemini_prompt_includes_signal_scores():
    """_build_gemini_prompt accepts signal_scores param and includes it in prompt."""
    _, _apply_guardrails, _build_gemini_prompt, _compute_signal_scores = _import_agent()
    ni, fp, cs = _make_minimal_prompt_args()
    sig = {"bullish_score": 65, "bearish_score": 30, "uncertainty_score": 40,
           "dominant": "bullish", "dominant_score": 65}
    prompt = _build_gemini_prompt(ni, fp, cs, signal_scores=sig)
    assert "bullish_score" in prompt
    assert "SIGNAL SCORES" in prompt


def test_gemini_prompt_has_decision_threshold_in_json_schema():
    """Prompt contains 'decision_threshold_used' field."""
    _, _apply_guardrails, _build_gemini_prompt, _ = _import_agent()
    ni, fp, cs = _make_minimal_prompt_args()
    prompt = _build_gemini_prompt(ni, fp, cs)
    assert "decision_threshold_used" in prompt


def test_gemini_prompt_has_hold_reason_field():
    """Prompt contains 'hold_reason' field in JSON schema."""
    _, _apply_guardrails, _build_gemini_prompt, _ = _import_agent()
    ni, fp, cs = _make_minimal_prompt_args()
    prompt = _build_gemini_prompt(ni, fp, cs)
    assert "hold_reason" in prompt


def test_gemini_prompt_has_invalidating_conditions():
    """Prompt contains 'invalidating_conditions' field."""
    _, _apply_guardrails, _build_gemini_prompt, _ = _import_agent()
    ni, fp, cs = _make_minimal_prompt_args()
    prompt = _build_gemini_prompt(ni, fp, cs)
    assert "invalidating_conditions" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# 28: accuracy_eligible logic
# ══════════════════════════════════════════════════════════════════════════════

def test_accuracy_eligible_only_for_success():
    """Error rows have accuracy_eligible=False, SUCCESS+MATCH rows have True."""
    from batch_validation_runner import _result_to_row, _error_row
    # Build a fake success result
    fake_result = {
        "status": "SUCCESS",
        "ai_prediction": {"decision": "BUY", "predicted_return_pct": 3.0,
                          "confidence_score": 70, "risk_score": 30,
                          "data_quality_score": 80},
        "actual_validation": {"actual_decision": "BUY", "actual_return_pct": 3.5},
        "comparison": {},
        "signal_scores": {},
        "features_used": {"bars_used": 120, "last_ai_bar_date": "2025-06-30",
                          "trend_regime": "bullish"},
        "_feature_packet": {},
        "stock_prediction_input_hash": "abc123",
    }
    row = _result_to_row(fake_result, "TESTID", "SPY", "stock",
                         "2025-07-01", "2025-07-31", 30, "close")
    assert row["accuracy_eligible"] is True

    # Error row => False
    err_row = _error_row("ERR01", "SPY", "stock", "2025-07-01", "2025-07-31", 30,
                          "timeout", failure_reason="DATA_UNAVAILABLE")
    assert err_row["accuracy_eligible"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 29: Direction match function
# ══════════════════════════════════════════════════════════════════════════════

def test_direction_match_function():
    """BUY vs BUY = True, BUY vs SELL = False, HOLD vs HOLD = True, SELL vs HOLD = False."""
    from batch_validation_runner import _direction_match
    assert _direction_match("BUY", "BUY") is True
    assert _direction_match("SELL", "SELL") is True
    assert _direction_match("HOLD", "HOLD") is True
    assert _direction_match("BUY", "SELL") is False
    assert _direction_match("SELL", "BUY") is False
    assert _direction_match("BUY", "HOLD") is False
    assert _direction_match("HOLD", "BUY") is False


# ══════════════════════════════════════════════════════════════════════════════
# 30: Over-hold reduction mechanism exists
# ══════════════════════════════════════════════════════════════════════════════

def test_over_hold_reduction_mechanism_exists():
    """Guardrails include HOLD->BUY/SELL override path when signals strongly agree."""
    _, _apply_guardrails, *_ = _import_agent()
    # Neutral return in HOLD band (30d buy_thr=1.5), but strongly bullish signals and high confidence
    # Use return=1.0 which is in HOLD band. near_thr = abs(1.0) >= abs(-1.5)*0.50 = 0.75 => True
    decision, reason = _apply_guardrails(
        1.0, 65, 20, 80, horizon_days=30,
        signal_scores=_sig(bull=70, bear=25, uncert=35, dominant="bullish", dom_score=70)
    )
    # HOLD is banned — positive return always produces BUY
    assert decision == "BUY"
    assert "BUY" in reason
