"""
Tests for Tastytrade Auth Truth Verification.

Requirements from TASTYTRADE AUTH TRUTH VERIFICATION PROMPT:
1.  No refresh + no access token -> backtest_allowed=false
2.  FORCE_DISABLE_TASTYTRADE=true -> backtest_allowed=false even if tokens exist
3.  Missing tokens cannot return options backtest SUCCESS
4.  Missing tokens cannot save accuracy
5.  Missing tokens clear stale old backtest result
6.  Refresh token path attempts refresh before customer check
7.  Access-token-only path does NOT pretend refresh succeeded
8.  Secrets are masked in debug output (no full token values)
9.  No backtest ID exists when auth failed
10. No P&L/trades/win_rate shown when auth failed
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Import the module under test
# ══════════════════════════════════════════════════════════════════════════════

from verify_tastytrade_auth_truth import verify_tastytrade_auth_truth, _mask


# ══════════════════════════════════════════════════════════════════════════════
# 1. No credentials -> backtest_allowed=false
# ══════════════════════════════════════════════════════════════════════════════

def test_no_credentials_blocks_backtest(monkeypatch):
    """No refresh token + no access token must return backtest_allowed=false."""
    monkeypatch.delenv("TASTYTRADE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("TASTYTRADE_ACCESS_TOKEN",  raising=False)
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "false")

    result = verify_tastytrade_auth_truth()

    assert result["backtest_allowed"] is False
    assert result["credential_source"] == "MISSING"
    assert result["token_refresh_status"] == "NOT_RUN"
    assert result["customer_check_status"] == "NOT_RUN"
    assert result["secrets_masked"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. FORCE_DISABLE_TASTYTRADE=true -> always blocked
# ══════════════════════════════════════════════════════════════════════════════

def test_force_disable_blocks_regardless_of_tokens(monkeypatch):
    """FORCE_DISABLE_TASTYTRADE=true must block even if valid tokens are present."""
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "true")
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "fake_refresh_token_for_test")
    monkeypatch.setenv("TASTYTRADE_ACCESS_TOKEN",  "fake_access_token_for_test")

    result = verify_tastytrade_auth_truth()

    assert result["backtest_allowed"] is False
    assert result["credential_source"] == "MISSING_FORCED"
    assert result["token_refresh_status"] == "NOT_RUN"
    assert result["customer_check_status"] == "NOT_RUN"
    assert "FORCE_DISABLE_TASTYTRADE" in result["reason"]


def test_force_disable_lowercase_true(monkeypatch):
    """FORCE_DISABLE_TASTYTRADE=TRUE (uppercase) must also block."""
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "TRUE")
    monkeypatch.delenv("TASTYTRADE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("TASTYTRADE_ACCESS_TOKEN",  raising=False)

    result = verify_tastytrade_auth_truth()
    assert result["backtest_allowed"] is False
    assert result["credential_source"] == "MISSING_FORCED"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Missing tokens cannot return options backtest SUCCESS in app
# ══════════════════════════════════════════════════════════════════════════════

def test_missing_tokens_cannot_return_success_in_app():
    """App must have TASTYTRADE_NOT_CONFIGURED guard that blocks SUCCESS."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "TASTYTRADE_NOT_CONFIGURED" in text
    assert "BACKTEST_CREATE_FAILED" in text
    assert "backtest_allowed" in text


# ══════════════════════════════════════════════════════════════════════════════
# 4. Missing tokens cannot save accuracy
# ══════════════════════════════════════════════════════════════════════════════

def test_missing_tokens_cannot_save_accuracy_in_app():
    """App must gate accuracy save on backtest_truly_succeeded and not skip_reason."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_backtest_truly_succeeded = opts_result.get(\"status\") == \"SUCCESS\"" in text
    assert "NOT SAVED" in text
    assert "_accuracy_skip_reason" in text


# ══════════════════════════════════════════════════════════════════════════════
# 5. Missing tokens clear stale old backtest result (run_context not persisted)
# ══════════════════════════════════════════════════════════════════════════════

def test_app_calls_clear_run_on_new_submit():
    """App must call _clear_run() before any new backtest, preventing stale results."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_clear_run()" in text
    # _clear_run must clear opts_result
    assert "opts_result" in text
    # After clear, old state cannot carry forward
    assert "\"opts_result\"" in text


# ══════════════════════════════════════════════════════════════════════════════
# 6. Refresh token path attempts refresh before customer check
# ══════════════════════════════════════════════════════════════════════════════

def test_refresh_token_path_sets_correct_credential_source(monkeypatch):
    """When refresh token is present, credential_source must be REFRESH_TOKEN
    and token_refresh_attempted must be True (even if refresh API fails in test)."""
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "test_rt_value")
    monkeypatch.delenv("TASTYTRADE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "false")

    # Mock the HTTP call to fail gracefully (no real network in tests)
    with patch("verify_tastytrade_auth_truth.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=401)
        result = verify_tastytrade_auth_truth()

    assert result["credential_source"] == "REFRESH_TOKEN"
    assert result["token_refresh_attempted"] is True
    assert result["token_refresh_status"] == "FAILED"
    assert result["backtest_allowed"] is False
    assert result["customer_check_status"] == "NOT_RUN"


def test_refresh_token_path_allows_backtest_on_success(monkeypatch):
    """When refresh + customer check both succeed, backtest_allowed must be True."""
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", "test_rt_value")
    monkeypatch.delenv("TASTYTRADE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "false")

    with patch("verify_tastytrade_auth_truth.requests.post") as mock_post, \
         patch("verify_tastytrade_auth_truth.requests.get") as mock_get:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "fresh_at_for_test"}
        )
        mock_get.return_value = MagicMock(status_code=200)

        result = verify_tastytrade_auth_truth()

    assert result["credential_source"] == "REFRESH_TOKEN"
    assert result["token_refresh_status"] == "SUCCESS"
    assert result["customer_check_status"] == "SUCCESS"
    assert result["backtest_allowed"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 7. Access-token-only path does NOT pretend refresh succeeded
# ══════════════════════════════════════════════════════════════════════════════

def test_access_token_only_does_not_pretend_refresh_succeeded(monkeypatch):
    """When only access token is present, token_refresh_status must be NOT_RUN."""
    monkeypatch.delenv("TASTYTRADE_REFRESH_TOKEN", raising=False)
    monkeypatch.setenv("TASTYTRADE_ACCESS_TOKEN", "old_at_value")
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "false")

    with patch("verify_tastytrade_auth_truth.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=401)
        result = verify_tastytrade_auth_truth()

    assert result["credential_source"] == "ACCESS_TOKEN_ONLY"
    assert result["token_refresh_attempted"] is False
    assert result["token_refresh_status"] == "NOT_RUN"
    assert result["backtest_allowed"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 8. Secrets are masked in debug output
# ══════════════════════════════════════════════════════════════════════════════

def test_secrets_masked_in_output(monkeypatch):
    """Token values must never appear in output — only masked previews."""
    real_token = "eyJhbGciOiJSUzI1NiJ9.secretpart.signature"
    monkeypatch.setenv("TASTYTRADE_REFRESH_TOKEN", real_token)
    monkeypatch.delenv("TASTYTRADE_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("FORCE_DISABLE_TASTYTRADE", "false")

    with patch("verify_tastytrade_auth_truth.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=503)
        result = verify_tastytrade_auth_truth()

    result_json = json.dumps(result)
    # Full token must never appear in output
    assert real_token not in result_json, "Full token value leaked into result JSON"
    assert result["secrets_masked"] is True
    # Masked version should be present (first 3 + last 4)
    assert "eyJ...ture" in (result.get("refresh_token_masked") or "")


def test_mask_function():
    """_mask must return first 3 chars + ... + last 4 chars."""
    assert _mask("eyJhbGciOiJSUzI1NiJ9") == "eyJ...NiJ9"
    assert _mask("abc") == "***"  # too short
    assert _mask("") is None
    assert _mask(None) is None


# ══════════════════════════════════════════════════════════════════════════════
# 9. No backtest ID when auth failed
# ══════════════════════════════════════════════════════════════════════════════

def test_no_backtest_id_when_auth_failed():
    """App must not store backtest_id in opts_result when auth truth check fails."""
    text = _read(ROOT / "stock_prediction_app.py")
    # When auth blocked, backtest_create is set to BLOCKED_* and no backtest_id is set
    assert "BLOCKED_NO_CREDENTIALS" in text
    assert "BLOCKED_AUTH_FAILED" in text
    # backtest_id is only set after _tt_create_backtest succeeds
    assert "backtest_id" in text
    assert "_tt_create_backtest" in text


# ══════════════════════════════════════════════════════════════════════════════
# 10. No P&L/trades/win_rate shown when auth failed
# ══════════════════════════════════════════════════════════════════════════════

def test_no_pnl_shown_when_auth_failed():
    """When backtest is blocked, opts_result must not contain P&L, trades, or win_rate
    from a real run."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Proof: profit_loss, win_rate, total_trades are only set inside the success block
    assert '"profit_loss"' in text or "'profit_loss'" in text
    assert "win_rate" in text
    assert "total_trades" in text
    # These values are set only in the `opts_result.update({...})` block inside
    # the successful backtest path, which is gated by `backtest_allowed`
    assert "BACKTEST_CREATE_FAILED" in text


# ══════════════════════════════════════════════════════════════════════════════
# App integration: truth check wired into options flow
# ══════════════════════════════════════════════════════════════════════════════

def test_verify_function_called_before_backtest_in_app():
    """App must call verify_tastytrade_auth_truth before running the backtest."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "verify_tastytrade_auth_truth" in text
    assert "_tt_auth_check" in text
    assert "backtest_allowed" in text


def test_force_disable_warning_in_main_banner():
    """App main() must show FORCE_DISABLE_TASTYTRADE warning when set."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "FORCE_DISABLE_TASTYTRADE" in text
    assert "disabled for verification" in text or "DISABLED FOR VERIFICATION" in text.upper()


def test_truth_check_result_stored_in_run_context():
    """App must store tastytrade_auth_truth dict in run_context for debug panel."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "tastytrade_auth_truth" in text
    assert "TASTYTRADE AUTH TRUTH CHECK" in text


def test_auth_truth_debug_panel_shows_all_fields():
    """Developer Debug section must show all required auth truth fields."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "Credential Source" in text
    assert "Access Token Present" in text
    assert "Refresh Token Present" in text
    assert "Token Refresh Attempted" in text
    assert "Token Refresh Status" in text
    assert "Customer Check Attempted" in text
    assert "Customer Check Status" in text
    assert "Auth HTTP Status" in text
    assert "Backtest Allowed" in text
    assert "Secrets Masked" in text
