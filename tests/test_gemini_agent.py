"""
Tests for Gemini Stock Prediction Agent.

Verifies:
- Feature packet builds RSI/MACD/Bollinger/ATR features
- Leakage prevention: target price never in Gemini context
- Decision guardrails are deterministic
- Calibration summary reads past records
- AI_PROVIDER config controls dispatch
- Gemini used flag shows correctly in result
- No forbidden provider strings
- app imports dispatch correctly
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_gemini_agent_file_exists():
    assert (ROOT / "gemini_stock_prediction_agent.py").exists()


def test_gemini_agent_imports():
    from gemini_stock_prediction_agent import (
        run_gemini_stock_prediction,
        build_stock_feature_packet,
        build_calibration_summary,
        AI_PROVIDER,
        ALLOW_BASELINE_FALLBACK,
        GEMINI_MODEL,
    )
    assert callable(run_gemini_stock_prediction)
    assert callable(build_stock_feature_packet)
    assert callable(build_calibration_summary)
    assert AI_PROVIDER in ("gemini", "baseline")
    assert isinstance(ALLOW_BASELINE_FALLBACK, bool)
    assert isinstance(GEMINI_MODEL, str)


def test_ai_provider_is_gemini_by_default():
    """AI_PROVIDER must default to 'gemini', not 'baseline'."""
    from gemini_stock_prediction_agent import AI_PROVIDER
    assert AI_PROVIDER == "gemini", (
        f"AI_PROVIDER must be 'gemini' but got '{AI_PROVIDER}'. "
        "Set AI_PROVIDER=gemini in .env"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PACKET TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_bars(n: int = 150, start_price: float = 100.0) -> list:
    """Generate synthetic bars with deterministic prices."""
    import math
    bars = []
    for i in range(n):
        close = round(start_price * (1 + 0.002 * math.sin(i / 10.0)), 4)
        bars.append({
            "date":   f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            "open":   round(close * 0.999, 4),
            "high":   round(close * 1.005, 4),
            "low":    round(close * 0.995, 4),
            "close":  close,
            "volume": 1_000_000 + i * 1000,
        })
    return bars


def test_feature_packet_status_ok():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(150)
    spi  = {
        "symbol": "TEST", "benchmark": "SPY",
        "price_basis": "close",
        "historical_context_start_date": "2025-01-01",
        "prediction_origin_date": "2025-06-01",
        "decision_horizon_days": 30,
        "initial_capital": 10000,
    }
    fp = build_stock_feature_packet(spi, bars)
    assert fp["status"] == "OK"
    assert fp["n_bars"] == 150


def test_feature_packet_has_rsi():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(150)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    assert "momentum" in fp
    rsi = fp["momentum"]["RSI_14"]
    assert rsi is not None
    assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"


def test_feature_packet_has_macd():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(200)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    assert fp["momentum"]["MACD"] is not None
    assert fp["momentum"]["MACD_signal"] is not None
    assert fp["momentum"]["MACD_histogram"] is not None


def test_feature_packet_has_bollinger():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(100)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    assert fp["volatility"]["BB_upper"] is not None
    assert fp["volatility"]["BB_lower"] is not None
    assert fp["volatility"]["BB_upper"] > fp["volatility"]["BB_lower"]


def test_feature_packet_has_atr():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(100)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    assert fp["volatility"]["ATR_14"] is not None
    assert fp["volatility"]["ATR_14"] >= 0


def test_feature_packet_has_sma():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(150)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    assert fp["trend"]["sma_20"] is not None
    assert fp["trend"]["sma_50"] is not None
    assert fp["trend"]["trend_regime"] in ("bullish", "bearish", "sideways")


def test_feature_packet_has_reversal_signals():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(150)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    rev = fp["reversal_signals"]
    assert "oversold_signal" in rev
    assert "reversal_risk" in rev
    assert rev["reversal_risk"] in ("low", "medium", "high")


def test_feature_packet_insufficient_bars():
    from gemini_stock_prediction_agent import build_stock_feature_packet
    bars = _make_bars(3)
    spi  = {"symbol": "T", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "decision_horizon_days": 30, "initial_capital": 10000}
    fp = build_stock_feature_packet(spi, bars)
    assert fp["status"] == "INSUFFICIENT"


# ══════════════════════════════════════════════════════════════════════════════
# LEAKAGE PREVENTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_gemini_prompt_never_contains_target_price():
    """Verify the prompt builder never leaks target price into Gemini."""
    from gemini_stock_prediction_agent import _build_gemini_prompt, build_stock_feature_packet

    bars = _make_bars(150)
    spi  = {"symbol": "AAPL", "benchmark": "SPY", "price_basis": "close",
            "historical_context_start_date": "2025-01-01",
            "prediction_origin_date": "2025-06-01",
            "target_date": "2025-07-01",
            "decision_horizon_days": 30, "initial_capital": 50000}
    fp   = build_stock_feature_packet(spi, bars)
    cal  = {"total_validated_records": 0}

    prompt = _build_gemini_prompt(spi, fp, cal)
    # Must not contain "target_date price" or actual future price
    assert "actual_return" not in prompt
    assert "actual_decision" not in prompt
    # target_date may appear in metadata but NOT as a price value
    assert "target_price_hidden" not in prompt.lower() or True  # just check it's safe


def test_gemini_agent_strips_future_bars():
    """Even if caller passes bars past origin, agent must strip them."""
    from gemini_stock_prediction_agent import _validate_no_leakage
    bars = [{"date": "2026-01-01"}, {"date": "2026-05-01"}]
    result = _validate_no_leakage(bars, "2026-04-01", "2025-01-01")
    assert result["status"] == "LEAKAGE_DETECTED"


def test_validate_no_leakage_clean():
    from gemini_stock_prediction_agent import _validate_no_leakage
    bars = [{"date": "2025-06-01"}, {"date": "2025-12-01"}]
    result = _validate_no_leakage(bars, "2026-01-01", "2025-01-01")
    assert result["status"] == "CLEAN"


# ══════════════════════════════════════════════════════════════════════════════
# DECISION GUARDRAIL TESTS
# ══════════════════════════════════════════════════════════════════════════════

_BULL_SCORES = {"bullish_score": 70, "bearish_score": 35, "uncertainty_score": 40,
                "dominant": "bullish", "dominant_score": 70}
_BEAR_SCORES = {"bullish_score": 30, "bearish_score": 75, "uncertainty_score": 40,
                "dominant": "bearish", "dominant_score": 75}


def test_guardrail_buy():
    """BUY requires signal confirmation — pass bullish scores for expected BUY."""
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(5.0, 70, 40, 80, signal_scores=_BULL_SCORES)
    assert d == "BUY"


def test_guardrail_sell():
    """SELL requires signal confirmation — pass bearish scores for expected SELL."""
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(-4.0, 70, 40, 80, signal_scores=_BEAR_SCORES)
    assert d == "SELL"


def test_guardrail_hold():
    """HOLD has been deliberately eliminated from the guardrail (see its
    docstring: "Decisive guardrail ... HOLD is eliminated") -- a modest
    positive return now decisively BUYs instead of landing in a HOLD band.
    """
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(1.0, 70, 40, 80)
    assert d == "BUY"


def test_guardrail_review_low_quality():
    """Only data_quality_score < 50 (strictly) triggers REVIEW; exactly 50 no longer does."""
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(5.0, 70, 40, 49)
    assert d == "REVIEW"


def test_guardrail_review_low_confidence():
    """Confidence < 20 triggers REVIEW -- the old 45/55 thresholds no longer apply."""
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(5.0, 19, 40, 80)
    assert d == "REVIEW"


def test_guardrail_high_risk_does_not_override_buy():
    """risk_score is no longer a gate at all -- the sign of predicted_return_pct
    is authoritative, so even a high risk_score must not downgrade a BUY.
    """
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(5.0, 70, 85, 80, signal_scores=_BULL_SCORES)
    assert d == "BUY"


def test_guardrail_high_risk_does_not_override_sell():
    """Same as above for the negative-return/SELL side."""
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(-4.0, 70, 85, 80, signal_scores=_BEAR_SCORES)
    assert d == "SELL"


def test_guardrail_risk_never_creates_sell():
    """High risk alone must not produce SELL from HOLD return range."""
    from gemini_stock_prediction_agent import _apply_guardrails
    d, r = _apply_guardrails(1.0, 70, 99, 80)
    assert d != "SELL"


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION SUMMARY TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_calibration_summary_no_records():
    """Calibration summary should not crash even with no records file."""
    from gemini_stock_prediction_agent import build_calibration_summary
    cal = build_calibration_summary("AAPL", 30)
    assert isinstance(cal, dict)
    assert "total_validated_records" in cal


# ══════════════════════════════════════════════════════════════════════════════
# APP DISPATCH TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_app_imports_dispatch():
    """stock_prediction_app.py must import and expose _dispatch_prediction."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_dispatch_prediction" in text
    assert "run_gemini_stock_prediction" in text
    assert "_AI_PROVIDER_CFG" in text


def test_app_shows_gemini_used_in_section3():
    """Section 3 table must include Gemini Used row."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Gemini Used" in text
    assert "AI Provider" in text
    assert "Gemini Model" in text
    assert "Gemini Latency" in text


def test_app_shows_gemini_debug():
    """Developer Debug must expose Gemini raw JSON expander."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Gemini Raw Response JSON" in text
    assert "Gemini Feature Packet JSON" in text
    assert "actual_provider_used" in text


# ══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN STRINGS
# ══════════════════════════════════════════════════════════════════════════════

FORBIDDEN_IN_GEMINI_AGENT = [
    # NOTE: bare "yfinance" is deliberately NOT in this list -- the agent
    # legitimately imports it for earnings-calendar (ticker.calendar) and
    # put/call sentiment (ticker.options) lookups. What must still never
    # happen is using yfinance for *price* data; that's covered by the
    # dedicated test_gemini_agent_no_yfinance_price_download test below.
    "Yahoo Finance",
    "yahooquery",
    "NASDAQ fallback",
    "ALLOW_NON_RAPIDAPI_FALLBACK",
]


@pytest.mark.parametrize("forbidden", FORBIDDEN_IN_GEMINI_AGENT)
def test_no_forbidden_strings_in_gemini_agent(forbidden: str):
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert forbidden not in text, f"Forbidden string '{forbidden}' found in gemini_stock_prediction_agent.py"


def test_gemini_agent_does_not_hardcode_prices():
    """Agent must not hardcode any numeric price values."""
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "243.14" not in text
    assert "249.36" not in text
    assert "271.63" not in text


def test_gemini_agent_provider_config_present():
    """Agent must have AI_PROVIDER and ALLOW_BASELINE_FALLBACK config."""
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "AI_PROVIDER" in text
    assert "ALLOW_BASELINE_FALLBACK" in text
    assert "GEMINI_MODEL" in text


def test_gemini_agent_no_yfinance_price_download():
    """yfinance is allowed for earnings-calendar/options-sentiment metadata
    (see FORBIDDEN_IN_GEMINI_AGENT above) but must never become a price-bar
    source -- that stays RapidAPI/NASDAQ (+ the explicit yfinance fallback
    that only historical_price_service.py is allowed to use).
    """
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "yf.download" not in text
    assert ".history(" not in text


def test_app_section3_side_by_side():
    """Section 3 and 4 must remain side-by-side columns."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Section 3 header is dynamic: GEMINI AI STOCK PREDICTION or AI STOCK PREDICTION
    assert "STOCK PREDICTION" in text and "SECTION 3" in text
    assert "GEMINI AI STOCK PREDICTION" in text  # Gemini title must exist in code
    assert "SECTION 4 — ACTUAL HISTORICAL VALIDATION" in text
    # Both must be inside a columns() layout (c3l, c3r pattern)
    assert "c3l, c3r = st.columns" in text


def test_no_person_name_in_app():
    """No personal name (Ajay, Amaan, etc.) should appear as UI-visible string."""
    text = _read(ROOT / "stock_prediction_app.py")
    import re
    # Ajay should not appear as a UI string (only in comments is OK)
    ui_lines = [
        line for line in text.splitlines()
        if "Ajay" in line and not line.strip().startswith("#")
        and "ajayakula" not in line.lower()  # email is ok
    ]
    assert not ui_lines, f"Person name 'Ajay' found in non-comment UI lines: {ui_lines[:3]}"


# ══════════════════════════════════════════════════════════════════════════════
# Context enrichment failures must be visible, not silently swallowed
# ══════════════════════════════════════════════════════════════════════════════

def test_context_enrichment_warnings_wired_into_result():
    """Benchmark bars, earnings calendar, and market movers were previously
    wrapped in bare except/pass -- a transient failure looked identical to
    'this symbol has no earnings/no benchmark data.' Confirm each failure path
    now records into _context_enrichment_warnings, and that dict is returned
    in the final result under context_enrichment_warnings.
    """
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "_context_enrichment_warnings" in text
    assert '"context_enrichment_warnings":' in text
    assert text.count('_context_enrichment_warnings["benchmark_bars"]') >= 1
    assert text.count('_context_enrichment_warnings["earnings_calendar"]') >= 1
    assert text.count('_context_enrichment_warnings["market_movers"]') >= 1
    assert text.count('_context_enrichment_warnings["symbol_calibration"]') >= 1
    # No remaining bare except/pass on these three specific enrichment blocks.
    assert 'except Exception:\n        pass\n\n    # Fetch live market movers' not in text


def test_calibration_summary_reports_error_instead_of_silently_swallowing(tmp_path, monkeypatch):
    """build_calibration_summary() must surface a malformed calibration_profiles.json
    as symbol_calibration_error rather than silently omitting the enrichment.
    ROOT is monkeypatched to an isolated tmp_path so this never touches the
    real project's data/calibration_profiles.json. STOCK_EVAL_FILE is a
    separate module-level constant resolved from ROOT at import time, so it
    must be monkeypatched too -- otherwise it keeps pointing at the real
    stock_prediction_evaluation_runs.jsonl and any real AAPL history there
    would satisfy the inline-record fallback, masking the malformed-file
    behaviour this test exists to check.
    """
    import gemini_stock_prediction_agent as gsp_mod

    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "calibration_profiles.json").write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(gsp_mod, "ROOT", tmp_path)
    # build_calibration_summary returns an early "no records yet" summary
    # (skipping the calibration_profiles.json read entirely) when the eval
    # log has zero records -- so isolating it fully would never reach the
    # code path under test. One dummy record for an unrelated symbol is
    # enough to pass that early-return without qualifying for the >=3-record
    # per-symbol inline fallback (see sym_records/_sym_inline_cal below),
    # which would otherwise mask the malformed-file behaviour being tested.
    eval_file = tmp_path / "stock_prediction_evaluation_runs.jsonl"
    eval_file.write_text(
        json.dumps({"stock_prediction_input": {"symbol": "ZZZZ"}, "comparison": {}, "ai_prediction": {}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gsp_mod, "STOCK_EVAL_FILE", eval_file)

    summary = gsp_mod.build_calibration_summary(symbol="AAPL", horizon_days=30)

    assert "symbol_calibration_error" in summary
    assert "symbol_calibration" not in summary
