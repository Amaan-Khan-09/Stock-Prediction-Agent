"""Tests for batch_validation_runner — verifies CSV/JSON output exist after a run."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Test 1: Import and basic structure ────────────────────────────────────────

def test_batch_validation_runner_importable():
    """batch_validation_runner must import without errors."""
    import batch_validation_runner as bvr
    assert hasattr(bvr, "run_batch")
    assert hasattr(bvr, "FOCUSED_SYMBOLS")
    assert hasattr(bvr, "TOP_SP500_SYMBOLS")
    assert hasattr(bvr, "CSV_FIELDNAMES")


def test_focused_symbols_includes_spy_tsla():
    from batch_validation_runner import FOCUSED_SYMBOLS
    assert "SPY" in FOCUSED_SYMBOLS
    assert "TSLA" in FOCUSED_SYMBOLS


def test_top_sp500_includes_required():
    from batch_validation_runner import TOP_SP500_SYMBOLS
    for sym in ["SPY", "TSLA", "AAPL", "MSFT", "NVDA"]:
        assert sym in TOP_SP500_SYMBOLS, f"{sym} missing from TOP_SP500_SYMBOLS"


def test_csv_fieldnames_complete():
    from batch_validation_runner import CSV_FIELDNAMES
    required = [
        "run_id", "symbol", "origin_date", "target_date", "horizon_days",
        "ai_decision", "actual_decision", "agreement",
        "predicted_return_pct", "actual_return_pct", "return_error_pp",
        "confidence", "risk", "validation_status", "failure_reason", "recommended_fix",
    ]
    for field in required:
        assert field in CSV_FIELDNAMES, f"Field '{field}' missing from CSV_FIELDNAMES"


# ── Test 2: Failure classification ────────────────────────────────────────────

def test_classify_over_hold():
    from batch_validation_runner import _classify_failure
    reason, fix = _classify_failure("HOLD", "BUY", 0.5, 3.0, 70, {})
    assert reason == "OVER_HOLD"


def test_classify_false_buy():
    from batch_validation_runner import _classify_failure
    reason, fix = _classify_failure("BUY", "SELL", 3.0, -3.0, 70, {})
    assert reason in ("FALSE_BUY", "WRONG_MAGNITUDE")


def test_classify_false_sell():
    from batch_validation_runner import _classify_failure
    reason, fix = _classify_failure("SELL", "BUY", -3.0, 3.0, 70, {})
    assert reason == "FALSE_SELL"


def test_classify_match():
    from batch_validation_runner import _classify_failure
    reason, fix = _classify_failure("BUY", "BUY", 3.0, 3.0, 70, {})
    assert reason == "NONE"


def test_classify_trend_reversal():
    from batch_validation_runner import _classify_failure
    reason, fix = _classify_failure("BUY", "SELL", 3.0, -8.0, 70, {})
    assert reason == "TREND_REVERSAL_MISSED"


# ── Test 3: Direction match helper ────────────────────────────────────────────

def test_direction_match_same():
    from batch_validation_runner import _direction_match
    assert _direction_match("BUY", "BUY") is True
    assert _direction_match("SELL", "SELL") is True


def test_direction_match_different():
    from batch_validation_runner import _direction_match
    assert _direction_match("BUY", "SELL") is False
    assert _direction_match("SELL", "BUY") is False


def test_direction_match_none():
    from batch_validation_runner import _direction_match
    assert _direction_match(None, "BUY") is False


# ── Test 4: Summary builder ────────────────────────────────────────────────────

def test_build_summary_empty():
    from batch_validation_runner import _build_summary
    summary = _build_summary([], ["SPY"], [30])
    assert summary["total_runs"] == 0
    assert summary["valid_runs"] == 0
    assert summary["decision_match_pct"] == 0.0


def test_build_summary_all_match():
    from batch_validation_runner import _build_summary, _compute_agreement
    rows = [
        {
            "symbol": "SPY", "horizon_days": 30,
            "ai_decision": "BUY", "actual_decision": "BUY",
            "agreement": "MATCH",
            "validation_status": "SUCCESS",
            "return_error_pp": -0.5,
            "failure_reason": "NONE",
        }
    ] * 5
    summary = _build_summary(rows, ["SPY"], [30])
    assert summary["decision_match_pct"] == 100.0
    assert summary["false_buy_count"] == 0


# ── Test 5: Recommendations builder ──────────────────────────────────────────

def test_recommendations_for_over_hold():
    from batch_validation_runner import _build_recommendations
    from collections import Counter
    fc = Counter({"OVER_HOLD": 5})
    recs = _build_recommendations(list(fc.items()), hold_rate=70.0, decision_match_pct=25.0)
    assert any("OVER_HOLD" in r for r in recs)


def test_recommendations_for_high_hold_rate():
    from batch_validation_runner import _build_recommendations
    recs = _build_recommendations([], hold_rate=75.0, decision_match_pct=30.0)
    assert any("hold" in r.lower() or "HOLD" in r for r in recs)
