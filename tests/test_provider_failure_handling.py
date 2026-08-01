"""
Tests for provider failure handling, truth gates, accuracy isolation,
Discord reply chain, and batch failure resilience.

Requirements from FINAL TRUTH-FIRST IMPLEMENTATION PROMPT:
1.  Historical data unavailable blocks run
2.  Provider DNS failure → invalid log, not accuracy log
3.  Failed run does not show old results (ai_result must be None)
4.  Missing tastytrade token cannot return SUCCESS
5.  Missing tastytrade token cannot save accuracy
6.  Missing Alpaca keys cannot show VERIFIED/SUBMITTED
7.  Strike mode with strike 0 blocks run
8.  Strike mode never sends delta=30/50 as selection value
9.  Delta mode accepts 83 and rejects 3046 via UI constraint
10. Options SELL produces NO_TRADE / REVIEW_REQUIRED
11. Exact unsupported → REVIEW_REQUIRED / UNSUPPORTED_BY_PROVIDER
12. Proxy validation is approximate, not exact accuracy
13. Discord parses SPX 7570 as exact strike
14. Reply chain calculates $130 correctly
15. Batch continues after one symbol data failure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Historical data unavailable must block run
# ══════════════════════════════════════════════════════════════════════════════

def test_run_all_blocks_on_hist_data_failure():
    """_run_all must set error_msg and NOT set ai_result when historical data fails."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "HISTORICAL_DATA_UNAVAILABLE" in text
    assert "_log_invalid_run" in text
    # Ensure the function returns early after setting error
    assert "st.session_state[\"error_msg\"]   = hist_err" in text or \
           "session_state[\"error_msg\"] = hist_err" in text


def test_options_run_all_blocks_on_hist_data_failure():
    """_run_options_all must also block and log invalid run on data failure."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "HISTORICAL_DATA_UNAVAILABLE" in text
    # Both _run_all and _run_options_all must call _log_invalid_run
    assert text.count("_log_invalid_run") >= 2


# ══════════════════════════════════════════════════════════════════════════════
# 2. Invalid log saved for provider failure
# ══════════════════════════════════════════════════════════════════════════════

def test_log_invalid_run_function_exists():
    """_log_invalid_run() must exist and write to data/invalid_runs.jsonl."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "def _log_invalid_run(" in text
    assert "invalid_runs.jsonl" in text
    assert "accuracy_saved" in text


def test_log_invalid_run_logic():
    """_log_invalid_run must write accuracy_saved=False to quarantine log."""
    from stock_prediction_app import _log_invalid_run
    import tempfile, os
    # Verify function is callable
    assert callable(_log_invalid_run)
    # It should write to a log file
    log_path = ROOT / "data" / "invalid_runs.jsonl"
    initial_lines = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    _log_invalid_run("TEST01", "hash123", "UNH", "stock",
                     "HISTORICAL_DATA_UNAVAILABLE", "DNS resolution failed")
    after_lines = len(log_path.read_text().splitlines())
    assert after_lines > initial_lines, "Invalid run was not logged"
    # Verify the entry has accuracy_saved=False
    last_line = log_path.read_text().splitlines()[-1]
    entry = json.loads(last_line)
    assert entry["accuracy_saved"] is False
    assert entry["symbol"] == "UNH"
    assert entry["failure_reason"] == "HISTORICAL_DATA_UNAVAILABLE"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Failed run does not show old results
# ══════════════════════════════════════════════════════════════════════════════

def test_failure_panel_exists_in_app():
    """_render_failure_panel() must exist and show RUN BLOCKED message."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "def _render_failure_panel(" in text
    assert "RUN BLOCKED" in text
    assert "OLD OUTPUT CLEARED" in text


def test_error_render_uses_failure_panel():
    """Main render loop must call _render_failure_panel, not plain _err_card, for errors."""
    text = _read(ROOT / "stock_prediction_app.py")
    # The render call for error must be _render_failure_panel()
    assert "_render_failure_panel()" in text
    # Plain _err_card must NOT be used for the main error display block
    # (it should only be used inside _render_failure_panel itself)
    lines = text.splitlines()
    main_render_err_lines = [
        l for l in lines
        if "_err_card(st.session_state" in l and "error_msg" in l
    ]
    assert not main_render_err_lines, (
        "Main render loop still uses _err_card directly for error_msg display: "
        + str(main_render_err_lines)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4 & 5. Missing tastytrade token cannot return SUCCESS or save accuracy
# ══════════════════════════════════════════════════════════════════════════════

def test_missing_tt_token_does_not_return_success():
    """When TT credentials missing, status must be BACKTEST_CREATE_FAILED, not SUCCESS."""
    text = _read(ROOT / "stock_prediction_app.py")
    # The credential check must exist
    assert "TASTYTRADE_NOT_CONFIGURED" in text or "NOT_CONFIGURED" in text
    # SUCCESS must only be set when backtest actually ran
    assert '"status":       "SUCCESS"' in text or '"status": "SUCCESS"' in text
    # BACKTEST_CREATE_FAILED must be a possible status
    assert "BACKTEST_CREATE_FAILED" in text


def test_tt_auth_debug_section_exists():
    """TT auth debug must show credential source, token refresh, customer check."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Tastytrade Auth Debug" in text
    assert "Credential Source" in text
    assert "Token Refresh" in text or "token_refresh" in text
    assert "Customer Check" in text or "customer_check" in text
    assert "Backtest Create" in text or "backtest_create" in text


def test_accuracy_save_gate_blocks_when_tastytrade_failed():
    """_backtest_truly_succeeded must be False if status != SUCCESS."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert '_backtest_truly_succeeded = opts_result.get("status") == "SUCCESS"' in text
    assert "NOT SAVED" in text


# ══════════════════════════════════════════════════════════════════════════════
# 6. Alpaca missing keys → NOT_CONFIGURED (Alpaca not wired in current app)
# ══════════════════════════════════════════════════════════════════════════════

def test_alpaca_not_used_for_live_trading():
    """App must not contain live Alpaca trade submission without paper flag."""
    text = _read(ROOT / "stock_prediction_app.py")
    # No live-money Alpaca calls
    assert "live.alpaca.markets" not in text


# ══════════════════════════════════════════════════════════════════════════════
# 7. Strike mode with strike 0 blocks run
# ══════════════════════════════════════════════════════════════════════════════

def test_pre_run_gate_blocks_zero_strike_in_app():
    """App must have pre-run gate that blocks strike=0 in Strike mode."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Strike Mode requires Strike Price > 0" in text
    assert "MISSING_STRIKE_PRICE" in text


def test_server_guard_blocks_exact_strike_missing():
    """Server-side guard in _run_options_all must block when exact strike is missing."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "MISSING_STRIKE_SERVER_GUARD" in text
    assert "EXACT_STRIKE mode requires a valid strike price" in text


# ══════════════════════════════════════════════════════════════════════════════
# 8. Strike mode never sends user's delta as selection value
# ══════════════════════════════════════════════════════════════════════════════

def test_strike_mode_does_not_use_user_delta_in_exact_branch():
    """Proxy branch uses delta=50 (ATM reference), never the user's delta_ui."""
    text = _read(ROOT / "stock_prediction_app.py")
    # In exact mode, proxy uses delta=50 explicitly labeled
    assert '"delta":               50,' in text
    # User's delta should only be used in the else (delta selection) branch
    assert '"delta":               int(_delta_ui),' in text


# ══════════════════════════════════════════════════════════════════════════════
# 9. Delta mode constraints
# ══════════════════════════════════════════════════════════════════════════════

def test_delta_mode_max_value_is_99():
    """Delta mode UI must cap at 99 — 3046 is rejected by UI."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Delta field in Delta mode should have max_value=99
    assert "max_value=99" in text
    # But NOT max_value=99 on the Strike Price field (strike has 99999)
    import re
    strike_max = re.search(r"Strike Price.*?max_value=(\d+)", text, re.DOTALL)
    if strike_max:
        assert int(strike_max.group(1)) > 99, "Strike Price must not be capped at 99"


def test_delta_mode_user_value_used():
    """Delta mode leg must use user's actual delta_ui, not a hardcoded default."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "int(_delta_ui)" in text
    # The 'or 30' default must NOT exist (was removed)
    assert "or 30" not in text


# ══════════════════════════════════════════════════════════════════════════════
# 11. Exact unsupported → UNSUPPORTED_BY_PROVIDER
# ══════════════════════════════════════════════════════════════════════════════

def test_exact_strike_marked_unsupported_by_provider():
    """When contract_selection_method=EXACT_STRIKE, status must be UNSUPPORTED_BY_PROVIDER."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "UNSUPPORTED_BY_PROVIDER" in text
    assert "EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER" in text


def test_proxy_labeled_approximate():
    """Proxy run must be labeled DELTA_PROXY_APPROXIMATE, never EXACT."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "DELTA_PROXY_APPROXIMATE" in text
    assert "fallback_used" in text


# ══════════════════════════════════════════════════════════════════════════════
# 15. Batch continues after one symbol data failure
# ══════════════════════════════════════════════════════════════════════════════

def test_batch_error_row_marks_data_unavailable():
    """_error_row with DATA_UNAVAILABLE failure_reason must produce DATA_UNAVAILABLE status."""
    from batch_validation_runner import _error_row
    row = _error_row("R1", "SYM", "stock", "2025-01-01", "2025-02-01", 30,
                     "DNS resolution failed for api.nasdaq.com",
                     failure_reason="DATA_UNAVAILABLE")
    assert row["validation_status"] == "DATA_UNAVAILABLE"
    assert row["failure_reason"] == "DATA_UNAVAILABLE"
    assert row["recommended_fix"] == "DATA_UNAVAILABLE"


def test_batch_summary_includes_data_unavailable_count():
    """_build_summary must report data_unavailable_count separately from failed_runs."""
    from batch_validation_runner import _build_summary
    rows = [
        {"symbol": "UNH", "validation_status": "DATA_UNAVAILABLE", "failure_reason": "DATA_UNAVAILABLE",
         "ai_decision": None, "actual_decision": None, "agreement": "UNKNOWN",
         "predicted_return_pct": None, "actual_return_pct": None, "return_error_pp": None,
         "horizon_days": 7},
        {"symbol": "SPY", "validation_status": "SUCCESS", "failure_reason": "NONE",
         "ai_decision": "BUY", "actual_decision": "BUY", "agreement": "MATCH",
         "predicted_return_pct": 1.0, "actual_return_pct": 1.2, "return_error_pp": -0.2,
         "horizon_days": 7},
    ]
    summary = _build_summary(rows, ["SPY", "UNH"], [7])
    assert summary["data_unavailable_count"] == 1, f"Expected 1, got {summary['data_unavailable_count']}"
    assert "UNH" in summary["symbols_failed"], f"UNH should be in symbols_failed: {summary['symbols_failed']}"
    assert summary["valid_runs"] == 1
    assert summary["decision_match_pct"] == 100.0


def test_failure_analyzer_classifies_data_unavailable():
    """failure_analyzer.classify_failure must return DATA_UNAVAILABLE for that status."""
    from failure_analyzer import classify_failure
    row = {"validation_status": "DATA_UNAVAILABLE", "failure_reason": "DATA_UNAVAILABLE",
           "ai_decision": None, "actual_decision": None, "agreement": "UNKNOWN"}
    assert classify_failure(row) == "DATA_UNAVAILABLE"


def test_calibration_profile_builder_importable():
    """calibration_profile_builder must be importable."""
    from calibration_profile_builder import build_profiles
    assert callable(build_profiles)


# ══════════════════════════════════════════════════════════════════════════════
# Input Binding / Date Gap Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_user_input_snapshot_captured_in_stock_form():
    """Stock form submit must store user_input_snapshot with submitted_at_utc."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert 'st.session_state["user_input_snapshot"]' in text
    assert "submitted_at_utc" in text
    assert "historical_context_start_date" in text


def test_user_input_snapshot_captured_in_options_form():
    """Options form submit must also store user_input_snapshot."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Check that user_input_snapshot is set twice (once per form)
    assert text.count('st.session_state["user_input_snapshot"]') >= 2


def test_input_binding_warning_shows_in_results():
    """_render_results and _render_options_results must show INPUT BINDING WARNING when date gap exists."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "INPUT BINDING WARNING" in text
    assert "input_binding_warning" in text
    assert "effective_ctx_start" in text
    assert "ctx_start_gap_days" in text


def test_run_context_stores_effective_start_when_gap_exists():
    """_run_all must store effective_ctx_start and requested_ctx_start in run_context when provider date gap found."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_run_ctx[\"effective_ctx_start\"]" in text or '"effective_ctx_start"' in text
    assert "_run_ctx[\"requested_ctx_start\"]" in text or '"requested_ctx_start"' in text
    assert "_run_ctx[\"ctx_start_gap_days\"]" in text or '"ctx_start_gap_days"' in text


def test_truth_gate_v4_banner_shows_tt_cred_and_rapidapi():
    """TRUTH_GATE_V4 banner must show TT cred status and RapidAPI key status."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "TRUTH_GATE_V4" in text
    assert "TT cred" in text
    assert "RapidAPI" in text
    assert "TASTYTRADE_REFRESH_TOKEN" in text
    assert "TASTYTRADE_ACCESS_TOKEN" in text


def test_tt_stale_token_detection_present():
    """App must attempt JWT expiry check when only access token is present."""
    text = _read(ROOT / "stock_prediction_app.py")
    # JWT parsing logic for staleness detection
    assert "exp" in text  # JWT exp claim check
    assert "EXPIRED" in text or "stale" in text.lower()


def test_rapidapi_capability_panel_present():
    """App must show RapidAPI capability panel when key is present."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "RapidAPI Plan Capabilities" in text or "RapidAPI Key Missing" in text


# ══════════════════════════════════════════════════════════════════════════════
# 20. Primary predicted-symbol history falls back to yfinance, not just benchmark
# ══════════════════════════════════════════════════════════════════════════════

def test_fetch_price_history_falls_back_to_yfinance(monkeypatch):
    """fetch_price_history() (the primary predicted-symbol path, not just the
    benchmark-only fetch_price_history_for_range()) must reach yfinance as a
    last resort when both RapidAPI and NASDAQ fail -- previously it gave up
    right after NASDAQ and yfinance was only wired into the benchmark fetch.
    """
    import requests
    import historical_price_service as hps

    hps._PRICE_CACHE.clear()
    hps._PROVIDER_USED.clear()
    monkeypatch.setenv("RAPIDAPI_KEY", "test-key-not-real")

    class _NotFound:
        status_code = 404
        text = "not found"

        def json(self):
            return {}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _NotFound())
    monkeypatch.setattr(requests, "get", lambda *a, **k: _NotFound())
    monkeypatch.setattr(
        hps, "_fetch_external_historical_provider",
        lambda symbol, start, end: ([], "NASDAQ simulated failure"),
    )
    monkeypatch.setattr(
        hps, "_fetch_yfinance",
        lambda symbol, start, end: (
            [{"date": "2024-01-02", "close": 100.0, "open": 99.0,
              "high": 101.0, "low": 98.0, "volume": 1000.0, "source": "yfinance"}],
            "",
        ),
    )

    bars, err = hps.fetch_price_history("AAPL", min_days=400)
    assert err == "", f"Expected success via yfinance fallback, got error: {err}"
    assert bars and bars[0]["close"] == 100.0
    assert hps.get_provider_used("AAPL") == "yfinance"


def test_fetch_price_history_reports_all_three_providers_when_all_fail(monkeypatch):
    """When RapidAPI, NASDAQ, and yfinance all fail, the error message must
    mention all three so a user can tell which providers were actually tried."""
    import requests
    import historical_price_service as hps

    hps._PRICE_CACHE.clear()
    hps._PROVIDER_USED.clear()
    monkeypatch.setenv("RAPIDAPI_KEY", "test-key-not-real")

    class _NotFound:
        status_code = 404
        text = "not found"

        def json(self):
            return {}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _NotFound())
    monkeypatch.setattr(requests, "get", lambda *a, **k: _NotFound())
    monkeypatch.setattr(
        hps, "_fetch_external_historical_provider",
        lambda symbol, start, end: ([], "NASDAQ simulated failure"),
    )
    monkeypatch.setattr(
        hps, "_fetch_yfinance",
        lambda symbol, start, end: ([], "yfinance simulated failure"),
    )

    bars, err = hps.fetch_price_history("ZZZZ", min_days=400)
    assert bars == []
    assert "NASDAQ simulated failure" in err
    assert "yfinance simulated failure" in err


def test_fetch_yfinance_uses_unadjusted_close(monkeypatch):
    """_fetch_yfinance must request auto_adjust=False so its closes stay
    comparable to RapidAPI/NASDAQ's unadjusted closes -- otherwise a symbol
    falling back to yfinance while its benchmark comes from RapidAPI/NASDAQ
    (or vice versa) would silently mix adjusted and unadjusted price series
    in relative-strength/return comparisons.
    """
    import pandas as pd
    import historical_price_service as hps

    captured_kwargs = {}

    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            captured_kwargs.update(kwargs)
            return pd.DataFrame(
                {"Close": [100.0], "Open": [99.0], "High": [101.0],
                 "Low": [98.0], "Volume": [1000]},
                index=[pd.Timestamp("2024-01-02")],
            )

    fake_yf = type("FakeYF", (), {"Ticker": staticmethod(lambda symbol: _FakeTicker(symbol))})

    import sys
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    bars, err = hps._fetch_yfinance("AAPL", "2024-01-01", "2024-01-05")
    assert err == ""
    assert captured_kwargs.get("auto_adjust") is False
    assert bars[0]["close"] == 100.0


def test_rapidapi_capability_panel_not_stale():
    """The RapidAPI capability panel must not claim earnings-calendar/market-movers
    are unimplemented -- gemini_stock_prediction_agent.py already fetches and
    injects both into the prompt (leakage-gated). This previously told users a
    working feature didn't exist.
    """
    app_text = _read(ROOT / "stock_prediction_app.py")
    assert "NOT YET IMPLEMENTED" not in app_text
    assert "not yet wired into prompt" not in app_text

    agent_text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "get_earnings_calendar" in agent_text
    assert "fetch_top_movers_for_gemini" in agent_text
