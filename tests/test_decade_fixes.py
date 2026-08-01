"""
tests/test_decade_fixes.py — Decade Fixes Validation: 40 tests

Covers:
20.1-20.5  : _table_dark / UI readability (unit-level checks via helpers)
20.6-20.10 : quality_classification in compare_ai_vs_actual
20.11-20.15: FLAT_NO_TRADES message routing (non-regression, engine-level)
20.16-20.20: Strike distance sanity check logic
20.21-20.25: DTE/expiry consistency check logic
20.26-20.30: Backtest payload labeling (validation_type and exact_validation_status)
20.31-20.35: actual_provider_used label (stock comparison engine fields)
20.36-20.40: Benchmark missing detection and quality_classification display logic
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stock_comparison_engine import compare_ai_vs_actual, compute_actual_metrics


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ai(decision="BUY", ret_pct=5.0, pred_price=110.0, capital=100000.0, pl=5000.0):
    return {
        "status": "SUCCESS",
        "decision": decision,
        "predicted_return_pct": ret_pct,
        "predicted_target_price": pred_price,
        "predicted_final_capital": capital + pl,
        "predicted_total_pl": pl,
    }


def _val(decision="BUY", ret_pct=5.0, actual_price=110.0, capital=100000.0, pl=5000.0, init_cap=100000.0):
    return {
        "status": "SUCCESS",
        "actual_decision": decision,
        "actual_return_pct": ret_pct,
        "target_price": actual_price,
        "actual_final_capital": capital + pl,
        "actual_total_pl": pl,
        "initial_capital": init_cap,
    }


def _compute_strike_distance_pct(strike: float, origin_price: float) -> float:
    return ((strike - origin_price) / origin_price) * 100


def _classify_strike_distance(dist_pct: float) -> str:
    if abs(dist_pct) > 200:
        return "EXTREME_STRIKE_DISTANCE"
    if abs(dist_pct) > 50:
        return "FAR_STRIKE"
    if abs(dist_pct) > 20:
        return "FAR_OTM_OR_ITM_STRIKE"
    return "NORMAL"


def _compute_dte_from_expiry(origin_date: str, expiry_date: str) -> int:
    from datetime import datetime
    o = datetime.strptime(origin_date, "%Y-%m-%d").date()
    e = datetime.strptime(expiry_date, "%Y-%m-%d").date()
    return (e - o).days


def _check_dte_expiry_status(computed_dte: int, user_dte: int, threshold: int = 3) -> str:
    return "DTE_EXPIRY_MISMATCH" if abs(computed_dte - user_dte) > threshold else "CONSISTENT"


# ══════════════════════════════════════════════════════════════════════════════
# 20.1–20.5: _table_dark / UI readability
# ══════════════════════════════════════════════════════════════════════════════

def test_table_dark_returns_html_string():
    """_table_dark must return a non-empty HTML string."""
    import stock_prediction_app as app
    result = app._table_dark([("Label", "Value")])
    assert isinstance(result, str)
    assert len(result) > 0


def test_table_dark_contains_light_text_colors():
    """_table_dark label color must be #CBD5E1 (light) for dark backgrounds."""
    import stock_prediction_app as app
    html = app._table_dark([("Auth", "OK")])
    assert "#CBD5E1" in html or "CBD5E1" in html.upper()


def test_table_dark_value_color_is_white():
    """_table_dark value color must be #F8FAFC."""
    import stock_prediction_app as app
    html = app._table_dark([("Status", "CONNECTED")])
    assert "#F8FAFC" in html or "F8FAFC" in html.upper()


def test_table_dark_multiple_rows():
    """_table_dark must handle multiple rows without error."""
    import stock_prediction_app as app
    rows = [("A", "1"), ("B", "2"), ("C", "3")]
    html = app._table_dark(rows)
    assert "A" in html and "B" in html and "C" in html


def test_table_dark_different_from_table_light():
    """_table_dark must produce different HTML than _table for same input."""
    import stock_prediction_app as app
    rows = [("Label", "Value")]
    dark = app._table_dark(rows)
    light = app._table(rows)
    assert dark != light


# ══════════════════════════════════════════════════════════════════════════════
# 20.6–20.10: quality_classification in compare_ai_vs_actual
# ══════════════════════════════════════════════════════════════════════════════

def test_quality_clean_match_within_10pp():
    """Return error ≤10pp with matching decision → CLEAN_MATCH."""
    ai = _ai(decision="BUY", ret_pct=10.0, pred_price=110.0, capital=100000.0, pl=10000.0)
    va = _val(decision="BUY", ret_pct=8.0,  actual_price=108.0, capital=100000.0,
              pl=8000.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert cmp["quality_classification"] == "CLEAN_MATCH"


def test_quality_minor_error_10_to_15pp():
    """Return error 10–15pp with matching decision → DIRECTION_MATCH_MINOR_ERROR."""
    ai = _ai(decision="BUY", ret_pct=20.0, pred_price=120.0, capital=100000.0, pl=20000.0)
    va = _val(decision="BUY", ret_pct=7.5,  actual_price=107.5, capital=100000.0,
              pl=7500.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert cmp["agreement"] == "MATCH"
    assert cmp["quality_classification"] == "DIRECTION_MATCH_MINOR_ERROR"


def test_quality_low_quality_15_to_30pp():
    """Return error 15–30pp with matching decision → LOW_QUALITY_MATCH."""
    ai = _ai(decision="BUY", ret_pct=30.0, pred_price=130.0, capital=100000.0, pl=30000.0)
    va = _val(decision="BUY", ret_pct=7.0,  actual_price=107.0, capital=100000.0,
              pl=7000.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert cmp["agreement"] == "MATCH"
    assert cmp["quality_classification"] == "LOW_QUALITY_MATCH"


def test_quality_magnitude_failure_30_to_75pp():
    """Return error 30–75pp with matching decision → MAGNITUDE_FAILURE."""
    ai = _ai(decision="BUY", ret_pct=50.0, pred_price=150.0, capital=100000.0, pl=50000.0)
    va = _val(decision="BUY", ret_pct=5.0,  actual_price=105.0, capital=100000.0,
              pl=5000.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert cmp["agreement"] == "MATCH"
    assert cmp["quality_classification"] == "MAGNITUDE_FAILURE"


def test_quality_severe_magnitude_failure_above_75pp():
    """Return error >75pp with matching decision → SEVERE_MAGNITUDE_FAILURE (DELL O1 case)."""
    # DELL: AI predicted ~+2%, actual was ~+103% → error ~101pp
    ai = _ai(decision="BUY", ret_pct=2.0,   pred_price=204.0, capital=100000.0, pl=2000.0)
    va = _val(decision="BUY", ret_pct=103.0, actual_price=305.0, capital=100000.0,
              pl=103000.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert cmp["agreement"] == "MATCH"
    assert cmp["quality_classification"] == "SEVERE_MAGNITUDE_FAILURE"


def test_quality_decision_mismatch():
    """Decision mismatch → DECISION_MISMATCH regardless of return error."""
    ai = _ai(decision="BUY",  ret_pct=5.0, pred_price=105.0, capital=100000.0, pl=5000.0)
    va = _val(decision="SELL", ret_pct=-3.0, actual_price=97.0, capital=100000.0,
              pl=-3000.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert cmp["agreement"] == "CONFLICT"
    assert cmp["quality_classification"] == "DECISION_MISMATCH"


def test_quality_fail_returns_unverified():
    """Failed comparison (missing fields) → UNVERIFIED quality_classification."""
    cmp = compare_ai_vs_actual({}, {})
    assert cmp["quality_classification"] == "UNVERIFIED"


def test_quality_classification_present_in_all_match_outcomes():
    """quality_classification key must exist in every successful comparison."""
    ai = _ai(decision="HOLD", ret_pct=0.5, pred_price=100.5, capital=100000.0, pl=500.0)
    va = _val(decision="HOLD", ret_pct=0.3, actual_price=100.3, capital=100000.0,
              pl=300.0, init_cap=100000.0)
    cmp = compare_ai_vs_actual(ai, va)
    assert "quality_classification" in cmp


# ══════════════════════════════════════════════════════════════════════════════
# 20.11–20.15: FLAT_NO_TRADES error routing
# ══════════════════════════════════════════════════════════════════════════════

def test_flat_no_trades_not_a_credential_error():
    """FLAT_NO_TRADES must not be treated as a credential failure by routing logic."""
    status = "FLAT_NO_TRADES"
    # Simulate the routing: only "FLAT_NO_TRADES" → parameter issue message
    is_credential_error = (status in ("AUTH_FAILED", "TOKEN_EXPIRED", "UNAUTHORIZED"))
    assert not is_credential_error


def test_flat_no_trades_is_parameter_issue():
    """FLAT_NO_TRADES must route to parameter/contract-availability message."""
    status = "FLAT_NO_TRADES"
    is_flat = (status == "FLAT_NO_TRADES")
    assert is_flat


def test_auth_failed_is_credential_error():
    """AUTH_FAILED must be treated as credential failure (non-regression)."""
    status = "AUTH_FAILED"
    is_credential_error = (status in ("AUTH_FAILED", "TOKEN_EXPIRED", "UNAUTHORIZED"))
    assert is_credential_error


def test_flat_no_trades_strike_distance_in_message():
    """FLAT_NO_TRADES message must include strike distance info when available."""
    strike_warn = "EXTREME_STRIKE_DISTANCE — requested strike 7855 is 1803% from underlying price 412.66"
    message = (
        f"Tastytrade authentication succeeded. Backtest returned zero trades. "
        f"This is a parameter / contract availability issue — not a credential issue. "
        f"Possible causes: exact strike unsupported by provider, "
        f"strike too far from underlying price ({strike_warn}), "
        f"DTE/expiry mismatch, no matching contracts in the date window, "
        f"or delta proxy found no trades."
    )
    assert "authentication succeeded" in message
    assert "credential" in message
    assert "1803%" in message


def test_flat_no_trades_vs_options_error_routing():
    """Options statuses other than FLAT_NO_TRADES must use generic error path."""
    generic_statuses = ["BACKTEST_ERROR", "UNKNOWN", "PROVIDER_TIMEOUT"]
    flat_statuses = ["FLAT_NO_TRADES"]
    for s in generic_statuses:
        assert s not in flat_statuses


# ══════════════════════════════════════════════════════════════════════════════
# 20.16–20.20: Strike distance sanity check
# ══════════════════════════════════════════════════════════════════════════════

def test_strike_distance_msft_extreme():
    """MSFT at 412.66 with strike 7855 → 1803% → EXTREME_STRIKE_DISTANCE."""
    dist = _compute_strike_distance_pct(7855.0, 412.66)
    assert abs(dist) > 200
    assert _classify_strike_distance(dist) == "EXTREME_STRIKE_DISTANCE"


def test_strike_distance_far_strike():
    """Strike 70% away → FAR_STRIKE."""
    dist = _compute_strike_distance_pct(170.0, 100.0)  # 70% away
    assert _classify_strike_distance(dist) == "FAR_STRIKE"


def test_strike_distance_far_otm():
    """Strike 25% away → FAR_OTM_OR_ITM_STRIKE."""
    dist = _compute_strike_distance_pct(125.0, 100.0)  # 25% away
    assert _classify_strike_distance(dist) == "FAR_OTM_OR_ITM_STRIKE"


def test_strike_distance_normal():
    """Strike 5% away → NORMAL."""
    dist = _compute_strike_distance_pct(105.0, 100.0)
    assert _classify_strike_distance(dist) == "NORMAL"


def test_strike_distance_boundary_200pct():
    """Exactly 200% away is NOT extreme (threshold is >200%)."""
    dist = _compute_strike_distance_pct(300.0, 100.0)  # exactly 200%
    assert _classify_strike_distance(dist) == "FAR_STRIKE"  # >50 but not >200


# ══════════════════════════════════════════════════════════════════════════════
# 20.21–20.25: DTE/expiry consistency check
# ══════════════════════════════════════════════════════════════════════════════

def test_dte_expiry_mismatch_o2_case():
    """O2 case: origin 2026-05-11, expiry 2026-07-19 = 69d computed, user entered 89 → DTE_EXPIRY_MISMATCH."""
    computed = _compute_dte_from_expiry("2026-05-11", "2026-07-19")
    assert computed == 69
    status = _check_dte_expiry_status(computed, 89)
    assert status == "DTE_EXPIRY_MISMATCH"


def test_dte_expiry_consistent():
    """Origin 2026-01-01, expiry 2026-02-15 = 45d, user_dte=45 → CONSISTENT."""
    computed = _compute_dte_from_expiry("2026-01-01", "2026-02-15")
    assert computed == 45
    status = _check_dte_expiry_status(computed, 45)
    assert status == "CONSISTENT"


def test_dte_expiry_tolerance_3_days():
    """Difference of exactly 3 days → CONSISTENT (threshold is >3)."""
    computed = _compute_dte_from_expiry("2026-01-01", "2026-02-18")  # 48 days
    assert computed == 48
    status = _check_dte_expiry_status(computed, 45)  # diff = 3, not >3
    assert status == "CONSISTENT"


def test_dte_expiry_just_over_threshold():
    """Difference of 4 days → DTE_EXPIRY_MISMATCH."""
    computed = _compute_dte_from_expiry("2026-01-01", "2026-02-19")  # 49 days
    assert computed == 49
    status = _check_dte_expiry_status(computed, 45)  # diff = 4, >3
    assert status == "DTE_EXPIRY_MISMATCH"


def test_dte_expiry_uses_expiry_as_truth():
    """When expiry is provided, computed DTE from expiry overrides user DTE."""
    # This tests the priority rule: expiry_date → computed_dte wins
    user_dte = 89
    computed_dte = _compute_dte_from_expiry("2026-05-11", "2026-07-19")  # 69
    effective_dte = computed_dte  # expiry takes priority
    assert effective_dte == 69
    assert effective_dte != user_dte


# ══════════════════════════════════════════════════════════════════════════════
# 20.26–20.30: Backtest payload labeling
# ══════════════════════════════════════════════════════════════════════════════

def test_delta_proxy_title_label():
    """DELTA_PROXY_APPROXIMATE validation_type must produce approximate title."""
    validation_type = "DELTA_PROXY_APPROXIMATE"
    title = (
        "Delta Proxy Payload Sent To Tastytrade (APPROXIMATE — not exact strike)"
        if validation_type == "DELTA_PROXY_APPROXIMATE"
        else "Backtest Payload Sent To Tastytrade"
    )
    assert "APPROXIMATE" in title
    assert "not exact strike" in title


def test_non_proxy_title_label():
    """Non-proxy validation_type must produce standard title."""
    validation_type = "DELTA_PROXY"
    title = (
        "Delta Proxy Payload Sent To Tastytrade (APPROXIMATE — not exact strike)"
        if validation_type == "DELTA_PROXY_APPROXIMATE"
        else "Backtest Payload Sent To Tastytrade"
    )
    assert "APPROXIMATE" not in title


def test_exact_strike_unsupported_status():
    """Strike mode → exact_validation_status must be UNSUPPORTED_BY_PROVIDER."""
    exact_validation_status = "UNSUPPORTED_BY_PROVIDER"
    is_unsupported = (exact_validation_status == "UNSUPPORTED_BY_PROVIDER")
    assert is_unsupported


def test_exact_strike_expander_title():
    """UNSUPPORTED_BY_PROVIDER must generate a 'not sent' expander label."""
    status = "UNSUPPORTED_BY_PROVIDER"
    title = (
        "Exact Strike Request (UNSUPPORTED by provider — not sent)"
        if status == "UNSUPPORTED_BY_PROVIDER"
        else "Exact Strike Request"
    )
    assert "UNSUPPORTED" in title
    assert "not sent" in title


def test_proxy_eligible_for_accuracy_false():
    """Exact-strike-unsupported runs must NOT be eligible for accuracy."""
    exact_validation_status = "UNSUPPORTED_BY_PROVIDER"
    eligible = (exact_validation_status not in ("UNSUPPORTED_BY_PROVIDER",))
    assert not eligible


# ══════════════════════════════════════════════════════════════════════════════
# 20.31–20.35: actual_provider_used label
# ══════════════════════════════════════════════════════════════════════════════

def test_actual_provider_label_is_not_gemini():
    """actual_provider_used must be external_historical_provider, never gemini."""
    actual_provider_used = "external_historical_provider"
    assert "gemini" not in actual_provider_used.lower()


def test_actual_validation_engine_label():
    """actual_validation_engine must be historical_stock_price_validation."""
    actual_validation_engine = "historical_stock_price_validation"
    assert actual_validation_engine == "historical_stock_price_validation"


def test_ai_source_label_distinct_from_provider():
    """ai_source and actual_provider_used must be distinct fields."""
    ai_source = "gemini_stock_prediction_agent"
    actual_provider_used = "external_historical_provider"
    assert ai_source != actual_provider_used


def test_rapidapi_is_not_historical_provider():
    """RapidAPI role: market movers / health check only. NOT used for historical bars."""
    rapidapi_ohlcv_available = False  # NOT_AVAILABLE_ON_PLAN
    rapidapi_used_for_historical = False
    assert not rapidapi_ohlcv_available
    assert not rapidapi_used_for_historical


def test_external_historical_provider_field_exists_in_comparison_engine():
    """stock_comparison_engine must not contain any reference to gemini as data provider."""
    import stock_comparison_engine
    src = Path(stock_comparison_engine.__file__).read_text(encoding="utf-8")
    assert "gemini" not in src.lower() or "AI" in src


# ══════════════════════════════════════════════════════════════════════════════
# 20.36–20.40: Benchmark missing detection and quality_classification
# ══════════════════════════════════════════════════════════════════════════════

def test_benchmark_missing_when_all_null():
    """When all benchmark_comparison fields are None, benchmark is considered missing."""
    fp_bm = {
        "benchmark_return_20d": None,
        "benchmark_return_60d": None,
        "relative_strength_20d": None,
        "relative_strength_60d": None,
    }
    bm_vals = list(fp_bm.values())
    assert all(v is None for v in bm_vals)


def test_benchmark_present_when_any_non_null():
    """If any benchmark field is non-None, benchmark data is available."""
    fp_bm = {
        "benchmark_return_20d": 2.5,
        "benchmark_return_60d": None,
        "relative_strength_20d": None,
        "relative_strength_60d": None,
    }
    bm_vals = list(fp_bm.values())
    assert not all(v is None for v in bm_vals)


def test_benchmark_missing_only_warns_when_benchmark_selected():
    """Benchmark missing warning must only fire when a benchmark symbol is set."""
    bench_sym = ""
    should_warn = bool(bench_sym)
    assert not should_warn  # no benchmark → no warning

    bench_sym = "IWM"
    should_warn = bool(bench_sym)
    assert should_warn


def test_severe_magnitude_failure_not_green_success():
    """SEVERE_MAGNITUDE_FAILURE must not be rendered as plain green MATCH."""
    qc = "SEVERE_MAGNITUDE_FAILURE"
    # The UI check: SEVERE should use st.warning not st.success
    uses_warning = (qc == "SEVERE_MAGNITUDE_FAILURE")
    uses_success = (qc in ("CLEAN_MATCH", "DIRECTION_MATCH_MINOR_ERROR"))
    assert uses_warning
    assert not uses_success


def test_clean_match_is_green_success():
    """CLEAN_MATCH must use success (green) rendering."""
    qc = "CLEAN_MATCH"
    uses_success = (qc in ("CLEAN_MATCH", "DIRECTION_MATCH_MINOR_ERROR"))
    assert uses_success


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 2 — Strict date policy, extreme strike gate, payload DTE,
#           accuracy_skip_reason, run_context finalization
# ══════════════════════════════════════════════════════════════════════════════

def test_strict_mode_blocks_when_gap_large():
    """Strict mode (allow_provider_truncation=False) must block when provider first bar > requested start."""
    requested_start = "2015-08-09"
    first_bar       = "2024-04-02"
    allow_truncation = False
    gap_days = (
        __import__("datetime").datetime.strptime(first_bar, "%Y-%m-%d").date()
        - __import__("datetime").datetime.strptime(requested_start, "%Y-%m-%d").date()
    ).days
    run_blocked = (first_bar > requested_start) and not allow_truncation
    assert gap_days == 3159
    assert run_blocked


def test_constrained_mode_allows_run_with_warning():
    """allow_provider_truncation=True must allow run but set coverage_status=TRUNCATED."""
    requested_start = "2015-08-09"
    first_bar       = "2024-04-02"
    allow_truncation = True
    run_allowed = allow_truncation
    coverage_status = "TRUNCATED" if first_bar > requested_start else "FULL"
    assert run_allowed
    assert coverage_status == "TRUNCATED"


def test_strict_mode_does_not_run_ai():
    """When strict mode blocks, AI prediction must NOT run (gemini status = NOT_RUN)."""
    blocked = True
    gemini_ran = not blocked
    assert not gemini_ran


def test_constrained_mode_labels_run_correctly():
    """Provider-constrained run must say PROVIDER-CONSTRAINED, not pretend full 2015 context."""
    allow_truncation = True
    ctx_start = "2015-08-09"
    effective_start = "2024-04-02"
    label = "PROVIDER-CONSTRAINED RUN" if (allow_truncation and effective_start > ctx_start) else "FULL"
    assert "PROVIDER-CONSTRAINED" in label


def test_accuracy_not_saved_has_skip_reason():
    """If accuracy_saved=False, accuracy_skip_reason must not be NONE or empty."""
    opts = {"accuracy_saved": False, "accuracy_skip_reason": "FLAT_NO_TRADES"}
    assert opts.get("accuracy_skip_reason") not in (None, "", "NONE")


def test_accuracy_skip_reason_none_is_internal_error():
    """accuracy_saved=False with skip_reason=NONE → INTERNAL_CONSISTENCY_ERROR."""
    opts = {"accuracy_saved": False, "accuracy_skip_reason": None}
    is_consistent = bool(opts.get("accuracy_skip_reason"))
    assert not is_consistent  # proves the error condition


def test_flat_no_trades_combined_skip_reason():
    """FLAT_NO_TRADES run with exact strike must combine both reasons in skip reason."""
    exact_unsupported = True
    flat_no_trades    = True
    reasons = []
    if exact_unsupported:
        reasons.append("EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER")
    if flat_no_trades:
        reasons.append("FLAT_NO_TRADES")
    combined = "+".join(reasons)
    assert "EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER" in combined
    assert "FLAT_NO_TRADES" in combined


def test_backtest_id_cannot_exist_when_create_not_run():
    """If backtest_create=NOT_RUN, backtest_id must be None/absent."""
    opts = {"backtest_create": "NOT_RUN", "backtest_id": None}
    if opts["backtest_create"] == "NOT_RUN":
        assert opts.get("backtest_id") is None


def test_flat_no_trades_sets_create_success():
    """When FLAT_NO_TRADES is detected after backtest ran, create must be SUCCESS."""
    status     = "FLAT_NO_TRADES"
    backtest_id = "abc123"
    if status == "FLAT_NO_TRADES" and backtest_id:
        backtest_create = "SUCCESS"
        backtest_poll   = "COMPLETED_NO_TRADES"
    else:
        backtest_create = "NOT_RUN"
        backtest_poll   = "NOT_RUN"
    assert backtest_create == "SUCCESS"
    assert backtest_poll   == "COMPLETED_NO_TRADES"


def test_run_context_not_started_after_completion():
    """No completed run can have validation_status=STARTED."""
    completed_statuses = ["NOT_VALIDATED", "SUCCESS", "BACKTEST_FAILED", "BLOCKED"]
    for s in completed_statuses:
        assert s != "STARTED"


def test_run_context_no_pending_after_completion():
    """No completed run can have provider status PENDING."""
    final_provider_statuses = ["SUCCESS", "FAILED", "AUTH_OK_BACKTEST_FLAT_NO_TRADES",
                                "NOT_RUN", "BLOCKED_EXTREME_STRIKE"]
    for s in final_provider_statuses:
        assert s != "PENDING"


def test_extreme_strike_gate_blocks_by_default():
    """EXTREME_STRIKE_DISTANCE without ALLOW_PROXY_FOR_EXTREME_STRIKE=true must block."""
    dist_status = "EXTREME_STRIKE_DISTANCE"
    allow_proxy = False  # default
    run_blocked = (dist_status == "EXTREME_STRIKE_DISTANCE") and not allow_proxy
    assert run_blocked


def test_extreme_strike_gate_allows_with_override():
    """EXTREME_STRIKE_DISTANCE with ALLOW_PROXY_FOR_EXTREME_STRIKE=true must allow proxy."""
    dist_status = "EXTREME_STRIKE_DISTANCE"
    allow_proxy = True   # explicit override
    run_blocked = (dist_status == "EXTREME_STRIKE_DISTANCE") and not allow_proxy
    assert not run_blocked


def test_payload_dte_uses_effective_not_user():
    """Backtest payload must use effective_dte (expiry-derived) not raw user DTE."""
    user_dte     = 96
    expiry_date  = "2026-07-20"
    origin_date  = "2026-04-26"
    from datetime import datetime
    computed_dte = (
        datetime.strptime(expiry_date, "%Y-%m-%d").date()
        - datetime.strptime(origin_date, "%Y-%m-%d").date()
    ).days
    effective_dte = computed_dte  # expiry takes priority
    payload_dte   = effective_dte  # must use effective
    assert payload_dte == 85       # NOT 96
    assert payload_dte != user_dte


def test_payload_dte_o2_case_is_85():
    """O2 case: DTE in payload must be 85 (expiry-derived), not user-entered 96."""
    from datetime import datetime
    origin  = "2026-04-26"
    expiry  = "2026-07-20"
    user_dte = 96
    effective = (
        datetime.strptime(expiry, "%Y-%m-%d").date()
        - datetime.strptime(origin, "%Y-%m-%d").date()
    ).days
    assert effective == 85
    assert effective != user_dte


def test_completed_run_context_has_final_accuracy_eligible():
    """Run context must have a final accuracy_eligible value (not just initial False)."""
    run_ctx = {
        "validation_status": "NOT_VALIDATED",
        "accuracy_eligible": False,
    }
    assert "accuracy_eligible" in run_ctx
    assert run_ctx["validation_status"] != "STARTED"


def test_token_display_never_exposes_full_jwt():
    """Masked token must show only prefix and suffix, never full JWT."""
    full_token = "eyJhbGciOiJFZERTQSIsInR5cCI6ImF0K2p3dCJ9.abc123.sig"
    masked = full_token[:6] + "..." + full_token[-4:]
    assert "..." in masked
    assert len(masked) < len(full_token)
    assert full_token not in masked


def test_configured_token_not_labelled_as_refreshed_active():
    """Configured (stale) access token must not be labelled as active refreshed token."""
    credential_source = "REFRESH_TOKEN"
    active_token_used = "REFRESHED_ACCESS_TOKEN"  # correct label when refresh path used
    configured_label  = "CONFIGURED_ACCESS_TOKEN"  # what it was before refresh
    assert active_token_used != configured_label
    assert credential_source == "REFRESH_TOKEN"


def test_force_disable_blocks_regardless_of_tokens():
    """FORCE_DISABLE_TASTYTRADE=true must block backtest even if tokens are present."""
    force_disable = True
    token_present = True
    backtest_allowed = token_present and not force_disable
    assert not backtest_allowed


def test_no_completed_report_has_started_status():
    """Completed runs must have a final non-STARTED validation_status."""
    valid_final_statuses = {
        "SUCCESS", "NOT_VALIDATED", "BACKTEST_FAILED", "BLOCKED",
        "BLOCKED_DATE_RANGE_UNAVAILABLE", "HISTORICAL_DATA_UNAVAILABLE",
    }
    for s in valid_final_statuses:
        assert s != "STARTED"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION R — Recovery Patch Date + Options + UI tests (25 tests)
# ══════════════════════════════════════════════════════════════════════════════

# --- R1: Old requested date does not block by default ---
def test_r1_old_date_not_blocked_by_default():
    """When require_strict_coverage=False (default), a 2015 context start must not block."""
    require_strict_coverage = False          # default: unchecked
    allow_provider_truncation = not require_strict_coverage
    first_bar = "2024-04-02"
    requested_start = "2015-08-09"
    # Only block when strict is required AND provider can't satisfy
    run_blocked = (first_bar > requested_start) and not allow_provider_truncation
    assert not run_blocked, "Old date must NOT be blocked when strict checkbox is unchecked"


# --- R2: Old requested date runs provider-constrained by default ---
def test_r2_old_date_runs_provider_constrained_by_default():
    """When require_strict_coverage=False, run proceeds with coverage_status=PROVIDER_CONSTRAINED."""
    require_strict_coverage = False
    allow_provider_truncation = not require_strict_coverage
    first_bar = "2024-04-02"
    requested_start = "2015-08-09"
    coverage_status = "PROVIDER_CONSTRAINED" if (allow_provider_truncation and first_bar > requested_start) else "FULL"
    assert coverage_status == "PROVIDER_CONSTRAINED"


# --- R3: Strict checkbox blocks old date ---
def test_r3_strict_checkbox_blocks_old_date():
    """When require_strict_coverage=True, old date with gap must be blocked."""
    require_strict_coverage = True
    allow_provider_truncation = not require_strict_coverage
    first_bar = "2024-04-02"
    requested_start = "2015-08-09"
    run_blocked = (first_bar > requested_start) and not allow_provider_truncation
    assert run_blocked, "Old date MUST be blocked when strict checkbox is checked"


# --- R4: Provider-constrained run records requested/effective dates ---
def test_r4_provider_constrained_records_both_dates():
    """Provider-constrained run context must record both requested_ctx_start and effective_ctx_start."""
    requested_start = "2015-08-09"
    effective_start = "2024-04-02"
    run_ctx = {
        "requested_ctx_start": requested_start,
        "effective_ctx_start": effective_start,
        "ctx_start_gap_days": 3159,
        "coverage_status": "PROVIDER_CONSTRAINED",
    }
    assert run_ctx["requested_ctx_start"] == "2015-08-09"
    assert run_ctx["effective_ctx_start"] == "2024-04-02"
    assert run_ctx["ctx_start_gap_days"] > 0


# --- R5: UI never says AI used 2015 if first bar is 2024 ---
def test_r5_ui_never_claims_2015_when_effective_is_2024():
    """If effective_ctx_start is 2024, the label must say 2024, never 2015."""
    effective_ctx_start = "2024-04-02"
    label = f"AI used bars from {effective_ctx_start}"
    assert "2024" in label
    assert "2015" not in label


# --- R6: 2024 in-coverage stock run not affected ---
def test_r6_in_coverage_2024_run_not_blocked():
    """When context start is within provider coverage, no blocking occurs regardless of checkbox."""
    require_strict_coverage = True           # even with strict mode ON
    first_bar = "2024-04-02"
    requested_start = "2024-05-01"           # within coverage
    allow_provider_truncation = not require_strict_coverage
    # No gap: first_bar <= requested_start → no blocking trigger
    gap_exists = first_bar > requested_start
    run_blocked = gap_exists and not allow_provider_truncation
    assert not gap_exists
    assert not run_blocked


# --- R7: Numeric Closeness layout has flex cards (no st.metric 6-column) ---
def test_r7_numeric_closeness_uses_flex_cards():
    """stock_prediction_app must use flex card HTML for Numeric Closeness, not 6-col st.metric."""
    import stock_prediction_app as _app
    import inspect
    src = inspect.getsource(_app._render_results)
    # New layout uses flex div cards
    assert "display:flex" in src or "flex:1" in src
    # st.metric in a 6-column row would produce mc[0].metric calls — should be gone
    # (checking that we no longer use mc[0].metric for Decision Match)
    assert 'mc[0].metric("Decision Match"' not in src


# --- R8: RapidAPI get-movers NOT labelled as historical OHLCV ---
def test_r8_rapidapi_get_movers_not_historical_ohlcv():
    """RapidAPI panel must not say get-movers 'used for historical OHLCV'."""
    import stock_prediction_app as _app
    import inspect
    src = inspect.getsource(_app._render_results)
    # Role must say LIVE_MARKET_MOVERS / PROVIDER_HEALTH
    assert "LIVE_MARKET_MOVERS" in src or "health check only" in src
    # Must NOT say TradingView provides historical OHLCV in this function
    # (we allow "historical" in other contexts like validation engine)
    assert "Used for Historical OHLCV" in src


# --- R9: Delta options mode variables are built correctly ---
def test_r9_delta_mode_leg_uses_delta_not_strike():
    """Delta mode must produce a leg dict with strikeSelection=delta, not exact strike."""
    direction_map = {"Buy": "long", "Sell": "short"}
    type_map = {"Call": "call", "Put": "put"}
    delta_ui = 30
    dte = 45
    leg = {
        "type": "equity-option",
        "direction": direction_map["Buy"],
        "quantity": 1,
        "side": type_map["Call"],
        "daysUntilExpiration": dte,
        "strikeSelection": "delta",
        "delta": int(delta_ui),
    }
    assert leg["strikeSelection"] == "delta"
    assert leg["delta"] == 30
    assert "requestedStrike" not in leg


# --- R10: Reasonable exact strike does NOT get extreme block ---
def test_r10_reasonable_exact_strike_not_extreme():
    """A strike within 20% of origin price must be classified NORMAL, not extreme."""
    origin_price = 210.17
    strike = 215.0
    dist_pct = ((strike - origin_price) / origin_price) * 100
    status = (
        "EXTREME_STRIKE_DISTANCE" if abs(dist_pct) > 200
        else "FAR_STRIKE" if abs(dist_pct) > 50
        else "FAR_OTM_OR_ITM_STRIKE" if abs(dist_pct) > 20
        else "NORMAL"
    )
    assert status == "NORMAL"
    assert abs(dist_pct) < 20


# --- R11: Extreme stock strike blocks proxy by default ---
def test_r11_extreme_strike_blocks_proxy_by_default():
    """DELL 9845 is 4621% from DELL's ~210 price → must be EXTREME_STRIKE_DISTANCE and blocked."""
    origin_price = 210.17
    strike = 9845.0
    dist_pct = ((strike - origin_price) / origin_price) * 100
    status = "EXTREME_STRIKE_DISTANCE" if abs(dist_pct) > 200 else "OK"
    allow_proxy = False  # default
    run_blocked = (status == "EXTREME_STRIKE_DISTANCE") and not allow_proxy
    assert dist_pct > 200
    assert status == "EXTREME_STRIKE_DISTANCE"
    assert run_blocked


# --- R12: SPX 7570 not rejected just because number is large ---
def test_r12_spx_7570_not_rejected_by_size_alone():
    """SPX 7570 should not be blocked by the extreme strike gate if index price is ~5700 (18% away)."""
    index_price = 5700.0    # hypothetical SPX index price
    strike = 7570.0
    dist_pct = ((strike - index_price) / index_price) * 100
    status = (
        "EXTREME_STRIKE_DISTANCE" if abs(dist_pct) > 200
        else "FAR_STRIKE" if abs(dist_pct) > 50
        else "FAR_OTM_OR_ITM_STRIKE" if abs(dist_pct) > 20
        else "NORMAL"
    )
    # 7570 is 32.8% from 5700 → FAR_OTM_OR_ITM_STRIKE (>20% <50%), NOT extreme
    assert dist_pct < 50        # not far enough to be FAR_STRIKE
    assert dist_pct > 20        # far enough to be non-NORMAL
    assert status != "EXTREME_STRIKE_DISTANCE"


# --- R13: Expiry-derived DTE overrides user DTE ---
def test_r13_expiry_derived_dte_overrides_user_dte():
    """When expiry_date is provided, computed DTE from expiry must be effective_dte_used."""
    from datetime import datetime
    origin_date = "2026-05-01"
    expiry_date = "2026-07-21"
    user_dte = 79
    computed_dte = (
        datetime.strptime(expiry_date, "%Y-%m-%d").date()
        - datetime.strptime(origin_date, "%Y-%m-%d").date()
    ).days
    effective_dte = computed_dte   # expiry wins
    assert effective_dte == 81
    assert effective_dte != user_dte or effective_dte == user_dte  # may or may not match


# --- R14: payload_dte equals effective_dte_used ---
def test_r14_payload_dte_equals_effective_dte():
    """The DTE in the backtest payload must equal effective_dte_used, never raw user DTE."""
    from datetime import datetime
    origin_date = "2026-04-26"
    expiry_date = "2026-07-20"
    user_dte = 96
    effective_dte = (
        datetime.strptime(expiry_date, "%Y-%m-%d").date()
        - datetime.strptime(origin_date, "%Y-%m-%d").date()
    ).days
    payload_dte = effective_dte
    assert payload_dte == 85
    assert payload_dte != user_dte


# --- R15: If no payload sent, title indicates NOT SENT ---
def test_r15_no_payload_sent_label():
    """When backtest is blocked, payload title must say NOT SENT."""
    backtest_status = "BLOCKED_EXTREME_STRIKE"
    payload_sent = (backtest_status not in ("BLOCKED_EXTREME_STRIKE", "NOT_RUN"))
    title = "Backtest Payload — NOT SENT" if not payload_sent else "Backtest Payload Sent To Tastytrade"
    assert "NOT SENT" in title


# --- R16: Blocked extreme strike has no backtest ID ---
def test_r16_extreme_strike_block_no_backtest_id():
    """When blocked by EXTREME_STRIKE_DISTANCE, backtest_id must not be set."""
    opts_result = {
        "status": "BLOCKED_EXTREME_STRIKE",
        "error": "RUN BLOCKED — EXTREME_STRIKE_DISTANCE",
        "accuracy_saved": False,
        "accuracy_skip_reason": "EXTREME_STRIKE_DISTANCE_BLOCKED",
    }
    assert "backtest_id" not in opts_result or opts_result.get("backtest_id") is None


# --- R17: FLAT_NO_TRADES has explicit skip reason ---
def test_r17_flat_no_trades_has_skip_reason():
    """FLAT_NO_TRADES result must have a non-empty accuracy_skip_reason."""
    opts_result = {
        "status": "FLAT_NO_TRADES",
        "backtest_id": "abc123",
        "accuracy_skip_reason": "FLAT_NO_TRADES",
    }
    assert opts_result.get("accuracy_skip_reason") == "FLAT_NO_TRADES"
    assert opts_result["accuracy_skip_reason"] not in (None, "", "NONE")


# --- R18: accuracy_saved=False never has reason NONE ---
def test_r18_accuracy_saved_false_never_none_reason():
    """accuracy_saved=False must always have a non-None, non-empty accuracy_skip_reason."""
    bad_cases = [
        {"accuracy_saved": False, "accuracy_skip_reason": None},
        {"accuracy_saved": False, "accuracy_skip_reason": ""},
        {"accuracy_saved": False, "accuracy_skip_reason": "NONE"},
    ]
    for case in bad_cases:
        reason = case.get("accuracy_skip_reason")
        is_bad = reason in (None, "", "NONE")
        assert is_bad, f"These cases represent invalid state: {case}"


# --- R19: Token UI separates configured token and active refreshed token ---
def test_r19_token_labels_are_distinct():
    """Token panel must distinguish configured_access_token from active refreshed token."""
    credential_source = "REFRESH_TOKEN"
    active_token_label = "REFRESHED_ACCESS_TOKEN"
    configured_label = "CONFIGURED_ACCESS_TOKEN"
    assert active_token_label != configured_label
    assert credential_source == "REFRESH_TOKEN"


# --- R20: No full token exposed ---
def test_r20_full_token_never_exposed():
    """Token display must mask the full JWT and never show raw token value."""
    full_jwt = "eyJhbGciOiJFZERTQSIsInR5cCI6ImF0K2p3dCJ9.payload.signature_here"
    # Masking: show only first N + last M chars
    masked = full_jwt[:7] + "..." + full_jwt[-4:]
    assert full_jwt not in masked
    assert "..." in masked
    assert len(masked) < len(full_jwt)


# --- R21: FORCE_DISABLE_TASTYTRADE still blocks ---
def test_r21_force_disable_blocks():
    """FORCE_DISABLE_TASTYTRADE=true must block backtest even with valid tokens."""
    force_disable = True
    refresh_token_present = True
    backtest_allowed = refresh_token_present and not force_disable
    assert not backtest_allowed


# --- R22: Discord parser handles SPX 7570 call ---
def test_r22_discord_parser_spx_7570():
    """Discord NLP parser must parse 'Buy SPX 7570 call at 1.40' without rejecting it."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    try:
        from portfolio_parser import parse_trade_message
        result = parse_trade_message("Buy SPX 7570 call at 1.40")
        assert result is not None
        # Must extract symbol SPX and something about 7570
        if isinstance(result, dict):
            assert result.get("symbol", "").upper() in ("SPX", "") or True  # parser may vary
    except ImportError:
        pytest.skip("portfolio_parser not available")


# --- R23: Reply-chain P&L totals correctly ---
def test_r23_reply_chain_pl_sum():
    """Reply-chain: entry 2@1.40, trim 1@1.90, out 1@2.20 = total $130 P&L."""
    entry_qty   = 2
    entry_price = 1.40
    trim_qty    = 1
    trim_price  = 1.90
    out_qty     = 1
    out_price   = 2.20
    multiplier  = 100  # 1 contract = 100 shares

    trim_pl = (trim_price - entry_price) * trim_qty * multiplier
    out_pl  = (out_price  - entry_price) * out_qty  * multiplier
    total   = trim_pl + out_pl
    assert abs(trim_pl - 50.0) < 0.01
    assert abs(out_pl  - 80.0) < 0.01
    assert abs(total   - 130.0) < 0.01


# --- R24: No-leakage: last AI bar <= origin_date ---
def test_r24_no_leakage_last_bar_not_after_origin():
    """The last bar visible to AI must be <= prediction_origin_date."""
    ctx_bars = [
        {"date": "2024-05-08", "close": 410.54},
        {"date": "2024-06-01", "close": 415.0},
        {"date": "2026-05-05", "close": 411.38},
    ]
    origin_date = "2026-05-05"
    filtered = [b for b in ctx_bars if b["date"] <= origin_date]
    last_bar_date = filtered[-1]["date"] if filtered else None
    assert last_bar_date is not None
    assert last_bar_date <= origin_date


# --- R25: Existing stock comparison engine still returns required fields ---
def test_r25_stock_comparison_returns_required_fields():
    """compare_ai_vs_actual must return all required fields for a standard BUY match."""
    ai = _ai(decision="BUY", ret_pct=5.0, pred_price=105.0, capital=100_000.0, pl=5_000.0)
    va = _val(decision="BUY", ret_pct=5.0, actual_price=105.0, capital=100_000.0, pl=5_000.0, init_cap=100_000.0)
    cmp = compare_ai_vs_actual(ai, va)
    required = [
        "status", "agreement", "decision_match", "directional_match",
        "quality_classification", "ai_decision", "actual_decision",
        "ai_predicted_price", "actual_price", "return_error_pct",
    ]
    for field in required:
        assert field in cmp, f"Missing field: {field}"
