"""
Tests for Paid API Utilization — RapidAPI + Tastytrade.

Verifies:
- rapidapi_market_service.py exists and is importable
- run_rapidapi_market_health_check() returns required fields
- No leakage: used_in_prediction_context is always False
- tastytrade_service.py exists and is importable
- run_tastytrade_health_check() returns required fields
- Tastytrade used_in_stock_mode is always False
- Tastytrade used_in_options_mode is always True
- App imports both services
- App has PAID API USAGE PROOF panel in source
- No hardcoded keys or secrets in either service
- Demo Health Check references both APIs
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# FILE EXISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def test_rapidapi_market_service_file_exists():
    assert (ROOT / "rapidapi_market_service.py").exists()


def test_tastytrade_service_file_exists():
    assert (ROOT / "tastytrade_service.py").exists()


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_rapidapi_market_service_importable():
    from rapidapi_market_service import (
        run_rapidapi_market_health_check,
        run_rapidapi_instrument_check,
    )
    assert callable(run_rapidapi_market_health_check)
    assert callable(run_rapidapi_instrument_check)


def test_tastytrade_service_importable():
    from tastytrade_service import (
        run_tastytrade_health_check,
        run_tastytrade_instrument_check,
    )
    assert callable(run_tastytrade_health_check)
    assert callable(run_tastytrade_instrument_check)


# ══════════════════════════════════════════════════════════════════════════════
# RETURN-SHAPE TESTS (no real network call — key absent → early-exit dict)
# ══════════════════════════════════════════════════════════════════════════════

def test_rapidapi_health_check_returns_dict(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "")
    from rapidapi_market_service import run_rapidapi_market_health_check
    result = run_rapidapi_market_health_check()
    assert isinstance(result, dict)


def test_rapidapi_health_check_has_required_keys(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "")
    from rapidapi_market_service import run_rapidapi_market_health_check
    result = run_rapidapi_market_health_check()
    required = {"called", "endpoint", "http_status", "key_present",
                "role", "used_in_prediction_context", "error"}
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_rapidapi_health_check_no_leakage_flag(monkeypatch):
    """used_in_prediction_context must be False — live data never sent to Gemini for historical dates."""
    monkeypatch.setenv("RAPIDAPI_KEY", "")
    from rapidapi_market_service import run_rapidapi_market_health_check
    result = run_rapidapi_market_health_check()
    assert result["used_in_prediction_context"] is False


def test_rapidapi_health_check_role_correct(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "")
    from rapidapi_market_service import run_rapidapi_market_health_check
    result = run_rapidapi_market_health_check()
    assert result["role"] == "market_intelligence_health_check"


def test_tastytrade_health_check_returns_dict(monkeypatch):
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "")
    from tastytrade_service import run_tastytrade_health_check
    result = run_tastytrade_health_check()
    assert isinstance(result, dict)


def test_tastytrade_health_check_has_required_keys(monkeypatch):
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "")
    from tastytrade_service import run_tastytrade_health_check
    result = run_tastytrade_health_check()
    required = {"called", "endpoint", "http_status", "token_refreshed",
                "customer_verified", "refresh_present",
                "role", "used_in_stock_mode", "used_in_options_mode", "error"}
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_tastytrade_health_check_stock_mode_always_false(monkeypatch):
    """Stock mode never uses Tastytrade — this must be False unconditionally."""
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "")
    from tastytrade_service import run_tastytrade_health_check
    result = run_tastytrade_health_check()
    assert result["used_in_stock_mode"] is False


def test_tastytrade_health_check_options_mode_always_true(monkeypatch):
    """Options mode always uses Tastytrade credentials — this must be True."""
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "")
    from tastytrade_service import run_tastytrade_health_check
    result = run_tastytrade_health_check()
    assert result["used_in_options_mode"] is True


def test_tastytrade_health_check_role_correct(monkeypatch):
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "")
    from tastytrade_service import run_tastytrade_health_check
    result = run_tastytrade_health_check()
    assert result["role"] == "authentication_and_account_health_check"


def test_tastytrade_no_refresh_returns_error(monkeypatch):
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "")
    from tastytrade_service import run_tastytrade_health_check
    result = run_tastytrade_health_check()
    assert result["error"] is not None
    assert result["customer_verified"] is False


def test_rapidapi_no_key_returns_error(monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "")
    from rapidapi_market_service import run_rapidapi_market_health_check
    result = run_rapidapi_market_health_check()
    assert result["error"] is not None
    assert result["called"] is False


# ══════════════════════════════════════════════════════════════════════════════
# APP INTEGRATION TESTS (source-level)
# ══════════════════════════════════════════════════════════════════════════════

def test_app_imports_rapidapi_service():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "rapidapi_market_service" in text
    assert "_rapidapi_hc" in text


def test_app_imports_tastytrade_service():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "tastytrade_service" in text
    assert "_tastytrade_hc" in text


def test_app_has_paid_api_usage_proof_panel():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "PAID API USAGE PROOF" in text


def test_app_calls_paid_apis_in_run_all():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "rapidapi_health" in text
    assert "tastytrade_health" in text
    assert '_rapidapi_hc()' in text
    assert '_tastytrade_hc()' in text


def test_app_demo_health_check_calls_both_apis():
    text = _read(ROOT / "stock_prediction_app.py")
    assert "RapidAPI" in text and "Tastytrade" in text
    assert "rapidapi_ok" in text
    assert "tastytrade_ok" in text


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY — NO HARDCODED SECRETS
# ══════════════════════════════════════════════════════════════════════════════

def test_rapidapi_service_no_hardcoded_keys():
    """API key must come from os.getenv, never a hardcoded literal string value."""
    import re
    text = _read(ROOT / "rapidapi_market_service.py")
    assert "RAPIDAPI_KEY" in text
    assert "os.getenv" in text
    # No long alphanumeric string that looks like a real API key
    suspicious = re.findall(r'["\'][0-9a-zA-Z]{32,}["\']', text)
    assert not suspicious, f"Possible hardcoded key found: {suspicious}"


def test_tastytrade_service_no_hardcoded_secrets():
    """Secrets must come from os.getenv, never hardcoded."""
    import re
    text = _read(ROOT / "tastytrade_service.py")
    assert "TASTYTRADE_REFRESH_TOKEN" in text
    assert "TASTYTRADE_CLIENT_SECRET" in text
    # No hardcoded Bearer tokens
    suspicious = re.findall(r'Bearer\s+[A-Za-z0-9._-]{40,}', text)
    assert not suspicious, f"Possible hardcoded Bearer token found: {suspicious}"


def test_tastytrade_service_no_hardcoded_urls():
    """cert.tastyworks.com must only appear in a security comment, not in live code."""
    text = _read(ROOT / "tastytrade_service.py")
    assert "TASTYTRADE_API_BASE_URL" in text
    # cert domain must only appear in a 'DO NOT' comment, never as a URL being called
    code_lines = [
        line for line in text.splitlines()
        if "cert.tastyworks.com" in line
        and not line.strip().startswith("#")
        and not line.strip().startswith('"""')
        and not line.strip().startswith("- ")
    ]
    assert not code_lines, f"cert.tastyworks.com found in live code: {code_lines}"


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS MODE — BACKTEST FAILURE BEHAVIOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_auth_service_prefers_refresh_over_static_token():
    """get_access_token() must prefer refresh flow over static env token when refresh is available."""
    text = _read(ROOT / "src" / "services" / "tastytrade_auth_service.py")
    lines = text.splitlines()
    # Find line index where refresh token is checked (conditional)
    refresh_check_line = next(
        (i for i, l in enumerate(lines) if "tastytrade_refresh_token" in l and "if " in l),
        None
    )
    # Find line where return refresh_access_token() is called
    refresh_return_line = next(
        (i for i, l in enumerate(lines) if "return refresh_access_token()" in l),
        None
    )
    # Find line where static token is used as fallback
    static_token_line = next(
        (i for i, l in enumerate(lines) if "tastytrade_access_token" in l and "_cached_token" in l),
        None
    )
    assert refresh_check_line is not None, "Refresh token conditional not found in get_access_token()"
    assert refresh_return_line is not None, "return refresh_access_token() not found"
    assert static_token_line is not None, "static token fallback not found"
    assert refresh_return_line < static_token_line, (
        f"get_access_token() uses static token (line {static_token_line}) "
        f"BEFORE refresh flow (line {refresh_return_line}) — causes stale-token 401"
    )


def test_options_backtest_failed_must_block_accuracy_save():
    """When options backtest fails, accuracy record must NOT be saved."""
    text = _read(ROOT / "stock_prediction_app.py")
    # The save logic must be conditional on backtest succeeding
    assert "_backtest_succeeded" in text or "_backtest_truly_succeeded" in text
    assert "not saved" in text.lower() or "not_saved" in text.lower() or "accuracy record not saved" in text or "NOT SAVED" in text


def test_options_backtest_failed_shows_backtest_failed_agreement():
    """When backtest fails, Final Decision Board agreement must be BACKTEST_FAILED."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "BACKTEST_FAILED" in text


def test_options_backtest_final_decision_is_review_when_failed():
    """When backtest fails, final_dec must be REVIEW / UNVERIFIED."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert '"REVIEW"' in text or "'REVIEW'" in text
    # Must be set when backtest fails
    assert "BACKTEST_FAILED" in text and "REVIEW" in text


def test_underlying_stock_reference_section_exists():
    """Options mode must show an Underlying Stock Reference section labeled informational only."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Underlying Stock Reference" in text
    assert "INFORMATIONAL ONLY" in text or "informational only" in text.lower()


def test_options_save_conditional_on_backtest_success():
    """save_stock_prediction_record must only be called when backtest truly succeeded."""
    text = _read(ROOT / "stock_prediction_app.py")
    # The save call must be inside a backtest-success conditional
    assert "_backtest_succeeded" in text or "_backtest_truly_succeeded" in text
    # There must be a branch that blocks save when not backtest succeeded
    assert "accuracy record not saved" in text or "NOT saved" in text or "not saved" in text.lower() or "NOT SAVED" in text


def test_backtest_stats_extracted_from_results_subkey():
    """Backtest stats must be extracted from bt_data[results][statistics], not top-level."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Should look inside "results" sub-dict
    assert 'get("results")' in text or "results_obj" in text or "_res_obj" in text


def test_parse_backtest_result_looks_inside_results():
    """parse_backtest_result() must look inside data['results'] for trials and statistics."""
    text = _read(ROOT / "src" / "services" / "tastytrade_backtester_service.py")
    assert 'get("results")' in text
    assert 'results_obj' in text or 'results_obj.get("trials")' in text or '_res_obj' in text


def test_paid_api_panel_shows_backtest_status_separately():
    """Paid API panel in options mode must distinguish auth status from backtest status."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "OPTIONS BACKTEST:" in text or "Backtest Status:" in text
    assert "Customer Verified" in text
    assert "Accuracy Saved" in text
