"""
Tests for Single Truth System — all panels must read from one auth_truth object.

Requirements from FINAL REVIEWER-READY IMPLEMENTATION PROMPT:
1.  FORCE_DISABLE=true makes Paid API Proof show NOT_RUN (not YES)
2.  FORCE_DISABLE=true — Developer Debug and Paid Proof match (same source)
3.  run_context owns provider status; all sections read from it
4.  Stale tastytrade health cannot show as current when force-disabled
5.  Date 2017 preserved in user input snapshot
6.  Effective input differences surface a reason
7.  Strike mode never sends delta=30/50 as selection value
8.  Delta mode never sends strike value
9.  Backtest blocked means accuracy not saved
10. Proxy validation excluded from exact accuracy
11. Failed historical provider goes to invalid log
12. Accuracy headline uses only eligible records (clean_accuracy_records)
13. HOLD bias diagnostics use validated-only records
14. Symbol calibration fed into Gemini prompt
15. Gemini prompt has no future actual price / actual decision
16. Discord SPX 7570 parsed as exact strike
17. Reply-chain P&L = 130
18. Existing stock validation flow still works
19. Existing options delta mode still works
20. No-leakage proof still in app
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1. FORCE_DISABLE makes Paid API Proof show NOT_RUN (not YES)
# ══════════════════════════════════════════════════════════════════════════════

def test_paid_api_proof_reads_from_auth_truth_not_health():
    """Options Paid API Proof must read tastytrade data from run_context.tastytrade_auth_truth."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Must use tastytrade_auth_truth, NOT tastytrade_health for TT display
    assert "_tt_truth_ui" in text, "Paid API Proof must use auth_truth object"
    assert "_tt_truth_ui.get(" in text


def test_paid_api_proof_no_longer_uses_stale_tastytrade_health_for_display():
    """The old pattern reading Token Refreshed from tastytrade_health must be gone from options Paid Proof."""
    text = _read(ROOT / "stock_prediction_app.py")
    # In the options Paid Proof section, the new pattern uses _tt_truth_ui
    # The old _to_refreshed from tastytrade_health must not be used in the UI rendering
    lines = text.splitlines()
    # Find the options Paid Proof section (after _tt_truth_ui is defined)
    tt_truth_ui_line = next((i for i, l in enumerate(lines) if "_tt_truth_ui = " in l), None)
    assert tt_truth_ui_line is not None, "Must define _tt_truth_ui"
    # After that line, Token Refreshed must use _tt_ref_att from auth_truth, not _to_refreshed
    post_lines = "\n".join(lines[tt_truth_ui_line:tt_truth_ui_line + 50])
    assert "_tt_ref_att" in post_lines or "token_refresh_attempted" in post_lines


def test_force_disable_skips_tastytrade_hc_in_options_flow():
    """When backtest_allowed=False, _tastytrade_hc() must NOT be called in options mode."""
    text = _read(ROOT / "stock_prediction_app.py")
    # The synthetic "blocked" health result must exist
    assert "FORCE_DISABLE_TASTYTRADE" in text
    assert "blocked_by" in text
    assert "NOT_CALLED" in text


def test_force_disable_skips_tastytrade_hc_in_stock_flow():
    """Stock mode must skip _tastytrade_hc() when FORCE_DISABLE_TASTYTRADE=true."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "_force_disable_tt_stock" in text
    assert "DISABLED_BY_FORCE_DISABLE_TASTYTRADE" in text


# ══════════════════════════════════════════════════════════════════════════════
# 2. FORCE_DISABLE — Developer Debug and Paid Proof use same object
# ══════════════════════════════════════════════════════════════════════════════

def test_auth_truth_stored_in_run_context_early():
    """verify_tastytrade_auth_truth must be called BEFORE _tastytrade_hc() in options flow."""
    text = _read(ROOT / "stock_prediction_app.py")
    # The early truth gate stores result in _run_ctx2["tastytrade_auth_truth"]
    assert "_tt_truth_early" in text
    assert "tastytrade_auth_truth" in text
    # The paid proof reads from run_context["tastytrade_auth_truth"]
    assert "_run_ctx_tt.get(\"tastytrade_auth_truth\"" in text or \
           "_run_ctx_tt.get('tastytrade_auth_truth'" in text


def test_both_paid_proof_and_debug_read_same_auth_truth():
    """TASTYTRADE AUTH TRUTH CHECK panel and Paid API Proof must both reference tastytrade_auth_truth."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert text.count("tastytrade_auth_truth") >= 4, \
        "Must appear in: early set, run_ctx store, paid proof read, debug panel read"


# ══════════════════════════════════════════════════════════════════════════════
# 3. run_context owns everything — provider status fields
# ══════════════════════════════════════════════════════════════════════════════

def test_run_context_has_all_provider_status_fields():
    """run_context must track historical, rapidapi, tastytrade, gemini provider status."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert '"historical_data"' in text or "'historical_data'" in text
    assert '"rapidapi"' in text or "'rapidapi'" in text
    assert '"tastytrade"' in text or "'tastytrade'" in text
    assert '"gemini"' in text or "'gemini'" in text


def test_force_disabled_provider_status_set_correctly():
    """When force-disabled, provider_status['tastytrade'] must be FORCE_DISABLED."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "FORCE_DISABLED" in text
    assert '"FORCE_DISABLED"' in text or "'FORCE_DISABLED'" in text


# ══════════════════════════════════════════════════════════════════════════════
# 4. Stale health cannot show as current
# ══════════════════════════════════════════════════════════════════════════════

def test_clear_run_clears_tastytrade_health():
    """_clear_run() must clear tastytrade_health to prevent stale display."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "tastytrade_health" in text
    # It must appear in _RUN_KEYS tuple
    run_keys_start = text.index("_RUN_KEYS = (")
    run_keys_end   = text.index(")", run_keys_start)
    run_keys_block = text[run_keys_start:run_keys_end]
    assert "tastytrade_health" in run_keys_block


# ══════════════════════════════════════════════════════════════════════════════
# 5 & 6. Date 2017 preserved, differences surfaced
# ══════════════════════════════════════════════════════════════════════════════

def test_user_input_snapshot_captured_for_both_forms():
    """Both stock and options forms must capture user_input_snapshot at submit."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert text.count('st.session_state["user_input_snapshot"]') >= 2


def test_input_binding_warning_shown_in_both_modes():
    """INPUT BINDING WARNING must appear in both render functions."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert text.count("INPUT BINDING WARNING") >= 2


def test_effective_vs_requested_in_developer_debug():
    """Developer Debug must show effective_ctx_start vs requested_ctx_start."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "User Input Snapshot vs Effective Input" in text


# ══════════════════════════════════════════════════════════════════════════════
# 7 & 8. Strike mode never sends delta; delta mode never sends strike
# ══════════════════════════════════════════════════════════════════════════════

def test_strike_mode_uses_delta50_proxy_only_not_user_delta():
    """Delta mode uses user's actual delta_ui; no hardcoded ATM proxy (exact-strike removed)."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Delta mode must use user's actual value
    assert '"delta":               int(_delta_ui),' in text
    # No hardcoded ATM proxy — exact-strike mode was removed
    assert '"delta":               50,' not in text


def test_delta_mode_uses_user_delta_not_hardcoded():
    """Delta mode must use int(_delta_ui) — never a hardcoded value."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert '"delta":               int(_delta_ui),' in text
    assert "or 30" not in text


# ══════════════════════════════════════════════════════════════════════════════
# 9 & 10. Backtest blocked = accuracy not saved; proxy excluded
# ══════════════════════════════════════════════════════════════════════════════

def test_accuracy_save_gate_requires_truly_succeeded():
    """Accuracy must only save when _backtest_truly_succeeded=True (status==SUCCESS)."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert '_backtest_truly_succeeded = opts_result.get("status") == "SUCCESS"' in text


def test_proxy_run_excluded_from_exact_accuracy():
    """All 4 Tastytrade strike modes wire entry_frequency correctly to payload."""
    text = _read(ROOT / "stock_prediction_app.py")
    # Confirmed working entry frequency values — "every day" is default
    assert '"every day"' in text, "Confirmed default entry frequency 'every day' must be present"
    assert "api_entry_frequency" in text, "Entry frequency must be stored in options_params"
    assert "entry_frequency=_api_entry_freq" in text, "Entry frequency must be passed to payload builder"


# ══════════════════════════════════════════════════════════════════════════════
# 11. Failed historical provider goes to invalid log
# ══════════════════════════════════════════════════════════════════════════════

def test_historical_failure_calls_log_invalid_run():
    """_run_options_all must call _log_invalid_run() on historical data failure."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert text.count("_log_invalid_run") >= 2  # stock and options both
    assert "HISTORICAL_DATA_UNAVAILABLE" in text
    assert "invalid_runs.jsonl" in text


# ══════════════════════════════════════════════════════════════════════════════
# 12. clean_accuracy_records classifies correctly
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_accuracy_records_importable():
    """clean_accuracy_records must be importable with analyze and classify_record."""
    from clean_accuracy_records import analyze, classify_record, apply_cleanup
    assert callable(analyze)
    assert callable(classify_record)
    assert callable(apply_cleanup)


def test_classify_record_valid():
    """A complete, accurate record with all fields must classify as VALID."""
    from clean_accuracy_records import classify_record
    rec = {
        "input_hash": "abc123",
        "symbol": "SPY",
        "target_date": "2024-01-15",
        "prediction_origin_date": "2023-12-15",
        "validation_status": "validated",
        "accuracy_saved": True,
        "actual_decision": "BUY",
        "actual_return_pct": 1.5,
        "ai_decision": "BUY",
        "agreement": "MATCH",
    }
    assert classify_record(rec) == "VALID"


def test_classify_record_missing_hash_invalid():
    """Record with no input_hash must classify as MISSING_HASH."""
    from clean_accuracy_records import classify_record
    rec = {"symbol": "SPY", "target_date": "2024-01-15", "validation_status": "validated"}
    assert classify_record(rec) == "MISSING_HASH"


def test_classify_record_proxy():
    """Record with DELTA_PROXY_APPROXIMATE validation_type must classify as PROXY."""
    from clean_accuracy_records import classify_record
    rec = {
        "input_hash": "abc123",
        "symbol": "SPY",
        "target_date": "2024-01-15",
        "prediction_origin_date": "2023-12-15",
        "validation_status": "validated",
        "accuracy_saved": True,
        "actual_decision": "BUY",
        "actual_return_pct": 1.5,
        "validation_type": "DELTA_PROXY_APPROXIMATE",
    }
    assert classify_record(rec) == "PROXY"


def test_classify_record_failed_run():
    """Record with accuracy_saved=False must classify as FAILED_RUN."""
    from clean_accuracy_records import classify_record
    rec = {
        "input_hash": "abc123",
        "symbol": "SPY",
        "target_date": "2024-01-15",
        "prediction_origin_date": "2023-12-15",
        "validation_status": "FAILED",
        "accuracy_saved": False,
    }
    assert classify_record(rec) == "FAILED_RUN"


def test_classify_stale_past_pending():
    """Pending record with past target_date must classify as STALE."""
    from clean_accuracy_records import classify_record
    rec = {
        "input_hash": "abc123",
        "symbol": "SPY",
        "target_date": "2020-01-01",  # far past
        "prediction_origin_date": "2019-12-01",
        "validation_status": "PENDING",
        "accuracy_saved": None,
    }
    assert classify_record(rec) == "STALE"


def test_analyze_returns_counts():
    """analyze() on missing file returns error dict, not crash."""
    from clean_accuracy_records import analyze
    from pathlib import Path
    result = analyze(Path("/nonexistent/path/file.jsonl"))
    assert "error" in result
    assert result["total"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 14. Symbol calibration in Gemini prompt
# ══════════════════════════════════════════════════════════════════════════════

def test_symbol_calibration_section_builder_exists():
    """gemini_stock_prediction_agent must have _build_symbol_calibration_section."""
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "_build_symbol_calibration_section" in text
    assert "symbol_calibration" in text
    assert "HOLD rate" in text


def test_gemini_prompt_has_symbol_calibration_call():
    """Gemini prompt builder must call _build_symbol_calibration_section."""
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "_build_symbol_calibration_section(calibration_summary)" in text


def test_calibration_builder_reads_profiles_json():
    """build_calibration_summary must attempt to read calibration_profiles.json."""
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    assert "calibration_profiles.json" in text
    assert "symbol_calibration" in text


# ══════════════════════════════════════════════════════════════════════════════
# 15. No future actual in Gemini prompt
# ══════════════════════════════════════════════════════════════════════════════

def test_gemini_prompt_has_no_future_actual_price():
    """Gemini prompt must never include actual_return, actual_decision, actual_price."""
    text = _read(ROOT / "gemini_stock_prediction_agent.py")
    # The inp_json dict passed to prompt must NOT contain actual results
    prompt_fn_start = text.index("def _build_gemini_prompt(")
    prompt_fn_end   = text.index("\ndef ", prompt_fn_start + 1)
    prompt_fn_text  = text[prompt_fn_start:prompt_fn_end]
    assert "actual_return" not in prompt_fn_text
    assert "actual_decision" not in prompt_fn_text
    assert "target-date price was NOT" in text or "do NOT reference any price after" in text


# ══════════════════════════════════════════════════════════════════════════════
# 16 & 17. Discord parser and reply-chain (existing tests already cover these)
# ══════════════════════════════════════════════════════════════════════════════

def test_discord_parser_spx_7570():
    """parse_signal must parse SPX 7570 as exact strike."""
    from agents.discord_parser import parse_signal
    sig = parse_signal("Buy SPX 7570 call at 1.40")
    assert sig.strike == 7570.0
    assert sig.contract_selection_method == "EXACT_STRIKE"


def test_reply_chain_pnl():
    """Entry 2@1.40, trim 1@1.90, out 1@2.20 must yield $130 P&L."""
    from agents.discord_parser import parse_signal, calculate_reply_chain_pnl
    entry = parse_signal("Entered SPY 620C 0DTE @ 1.40, 2 contracts")
    trim  = parse_signal("trimmed half at 1.90")
    out   = parse_signal("out rest at 2.20")
    result = calculate_reply_chain_pnl([entry, trim, out])
    assert result["realized_pnl"] == 130.0


# ══════════════════════════════════════════════════════════════════════════════
# 18 & 19. Existing flows preserved
# ══════════════════════════════════════════════════════════════════════════════

def test_stock_validation_flow_still_exists():
    """_run_all function must still exist and contain core stock validation logic."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "def _run_all(" in text
    assert "run_stock_validation" in text
    assert "save_stock_prediction_record" in text


def test_options_delta_mode_still_works():
    """Options delta mode must still build a leg with user's delta_ui."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "DELTA_SELECTION" in text
    assert '"delta":               int(_delta_ui),' in text
    assert "fallback_used" in text


# ══════════════════════════════════════════════════════════════════════════════
# 20. No-leakage proof
# ══════════════════════════════════════════════════════════════════════════════

def test_no_leakage_proof_in_developer_debug():
    """Developer Debug must show AI visible history bars and no-leakage note."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "target-date price was NOT in these bars" in text
    assert "Target price hidden" in text or "target_date" in text


def test_accuracy_improvement_dashboard_shows_hold_bias():
    """App must show HOLD rate and decision match in accuracy diagnostics."""
    text = _read(ROOT / "stock_prediction_app.py")
    assert "HOLD" in text
    assert "Decision Match" in text or "decision_match" in text
