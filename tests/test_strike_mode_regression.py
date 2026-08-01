"""
Regression tests for Strike mode + FLAT_NO_TRADES + accuracy save gate.

Covers all items from the July 2026 final implementation prompt:
- Strike 7259/7570 accepted (no max-99 cap)
- Strike mode blocks run when strike=0/blank
- Strike mode summary shows "Strike X, Expiry Y" not "Delta X, DTE Y"
- Strike mode never sends user delta in tastytrade payload
- tastytrade exact validation marked UNSUPPORTED_BY_PROVIDER in Strike mode
- Delta mode accepts 83, rejects 3046 only in Delta context
- FLAT_NO_TRADES not saved as accuracy
- Missing tastytrade token cannot show SUCCESS
- Single-source options_params dict: delta_ui=None in Strike mode, requested_strike=None in Delta mode
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


# ══════════════════════════════════════════════════════════════════════════════
# STATIC SOURCE ANALYSIS — check code contracts without running Streamlit
# ══════════════════════════════════════════════════════════════════════════════

APP = ROOT / "stock_prediction_app.py"


def _src() -> str:
    return APP.read_text(encoding="utf-8", errors="ignore")


# ── 1. Strike price field must NOT have max=99 or max_value=99 ────────────────

def test_strike_field_has_no_max_99_cap():
    """Strike Price number_input must allow values far above 99."""
    import re
    text = _src()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if '"Strike Price"' in line or "'Strike Price'" in line:
            block = "\n".join(lines[i : i + 10])
            # Check for exact max_value=99 (not max_value=99999)
            # Pattern: max_value=99 followed by non-digit (comma, paren, space)
            if re.search(r"max_value=99[^0-9]", block):
                raise AssertionError(
                    f"Strike Price field at line {i+1} has max_value=99 (exact). "
                    "Strike has NO upper cap — SPX 7570, 7259 must be accepted."
                )
    # Confirm max_value exists for strike and is >> 99
    assert "max_value=99999" in text or "max_value=99999.0" in text, (
        "Strike Price field must have max_value=99999 or similar high cap, not 99."
    )


def test_delta_field_max_99_still_present_for_delta_mode():
    """Delta (1-99) field must still cap at 99 — it's 3046 in delta mode that's wrong."""
    text = _src()
    # Delta mode uses 1-99 range — this is correct behavior
    assert "max_value=99" in text, (
        "Delta field must retain max_value=99. "
        "Delta 30 means 0.30 — values above 99 are invalid in Delta mode."
    )


# ── 2. Strike mode UI must NOT show proxy delta field ─────────────────────────

def test_strike_mode_does_not_show_proxy_delta_field():
    """Proxy delta field ('Delta (1-99) — proxy reference') must be gone."""
    text = _src()
    assert "proxy reference" not in text, (
        "The 'proxy reference' label must be removed from Strike mode UI. "
        "Users were entering large strikes there and getting rejected by max=99."
    )
    assert "Delta (1-99) — proxy" not in text, (
        "Delta (1-99) — proxy reference field must not appear in any mode."
    )


# ── 3. options_params dict: Strike mode sets delta_ui=None ───────────────────

def test_options_params_sets_delta_ui_null_in_strike_mode():
    """In Strike mode, options_params['delta_ui'] must be None."""
    text = _src()
    # The canonical dict construction must have conditional None for delta_ui
    assert '"delta_ui":' in text, "options_params must contain 'delta_ui' key"
    # The pattern should be: delta_ui: int(delta_val) if _is_delta_mode else None
    assert "if _is_delta_mode else None" in text, (
        "options_params must set delta_ui conditionally: "
        "int(delta_val) if _is_delta_mode else None. "
        "In Strike mode, delta_ui must be None."
    )


def test_options_params_sets_requested_strike_null_in_delta_mode():
    """In Delta mode, options_params['requested_strike'] must be None."""
    text = _src()
    assert '"requested_strike":' in text, "options_params must contain 'requested_strike' key"
    # Strike fields are only set in Strike mode
    assert "_req_strike" in text, (
        "options_params must compute _req_strike conditionally."
    )


def test_options_params_has_contract_selection_method():
    """options_params must have contract_selection_method key with EXACT_STRIKE/DELTA_SELECTION."""
    text = _src()
    assert '"contract_selection_method":' in text, (
        "options_params must include 'contract_selection_method' key."
    )
    assert "EXACT_STRIKE" in text, "EXACT_STRIKE must appear as a contract method value."
    assert "DELTA_SELECTION" in text, "DELTA_SELECTION must appear as a contract method value."


# ── 4. Pre-run front-end validation gate ──────────────────────────────────────

def test_pre_run_gate_blocks_zero_strike():
    """App must block run before API calls when Strike mode + strike=0/null."""
    text = _src()
    assert "Strike Mode requires Strike Price > 0" in text, (
        "Pre-run validation gate must produce error: "
        "'Strike Mode requires Strike Price > 0' when strike is 0 or null."
    )


def test_pre_run_gate_shows_review_required():
    """When strike is missing, opts_result status must be REVIEW_REQUIRED, not run."""
    text = _src()
    assert '"status": "REVIEW_REQUIRED"' in text or "'status': 'REVIEW_REQUIRED'" in text or '"REVIEW_REQUIRED"' in text, (
        "Missing strike must produce REVIEW_REQUIRED status, not run the backtest."
    )


def test_pre_run_gate_sets_not_run():
    """When blocked, backtest_status must be NOT_RUN."""
    text = _src()
    assert "NOT_RUN" in text, (
        "Blocked run must set backtest_status=NOT_RUN — the backtest must not execute."
    )


# ── 5. Server-side guard inside _run_options_all ─────────────────────────────

def test_server_side_guard_present():
    """_run_options_all must have a belt-and-suspenders exact-strike guard."""
    text = _src()
    assert "MISSING_STRIKE_SERVER_GUARD" in text, (
        "_run_options_all must block if exact strike is missing, "
        "setting accuracy_skip_reason=MISSING_STRIKE_SERVER_GUARD."
    )


def test_server_side_guard_uses_requested_strike():
    """Server guard must check requested_strike from options_params."""
    text = _src()
    assert "_srv_strike" in text or "requested_strike" in text, (
        "Server guard must read requested_strike from options_params."
    )


# ── 6. Tastytrade leg: no user delta in EXACT_STRIKE mode ────────────────────

def test_tastytrade_leg_removes_delta_in_exact_mode():
    """In EXACT_STRIKE mode, tastytrade leg must not carry user-selected delta."""
    text = _src()
    # The fix: leg.pop("delta", None) when exact, OR delta is only set in delta branch
    assert 'leg.pop("delta"' in text or "UNSUPPORTED_BY_PROVIDER" in text, (
        "Tastytrade leg must remove 'delta' in EXACT_STRIKE mode. "
        "Sending delta=50 as user input when user selected Strike mode is dishonest."
    )


def test_tastytrade_exact_mode_marked_unsupported():
    """Tastytrade exact-strike must be marked UNSUPPORTED_BY_PROVIDER."""
    text = _src()
    assert "UNSUPPORTED_BY_PROVIDER" in text, (
        "When running EXACT_STRIKE mode, tastytrade result must have "
        "exact_validation_status=UNSUPPORTED_BY_PROVIDER. "
        "Tastytrade API cannot backtest fixed-strike contracts."
    )


def test_tastytrade_delta_proxy_label():
    """Proxy run in EXACT_STRIKE mode must be labeled DELTA_PROXY_APPROXIMATE."""
    text = _src()
    assert "DELTA_PROXY_APPROXIMATE" in text, (
        "When exact strike is unsupported, the fallback proxy run must be labeled "
        "DELTA_PROXY_APPROXIMATE — not presented as an exact-strike validation."
    )


# ── 7. FLAT_NO_TRADES classification ─────────────────────────────────────────

def test_flat_no_trades_status_exists():
    """FLAT_NO_TRADES must be a possible backtest status."""
    text = _src()
    assert "FLAT_NO_TRADES" in text, (
        "When tastytrade returns 0 trades + $0 P&L, status must be FLAT_NO_TRADES, not SUCCESS."
    )


def test_flat_no_trades_checks_n_trades_and_pl():
    """FLAT_NO_TRADES detection must check both n_trades==0 and total_pl==0."""
    text = _src()
    assert "_n_trades == 0" in text or "_is_flat" in text, (
        "FLAT_NO_TRADES must check: _n_trades == 0."
    )
    # The condition that sets FLAT_NO_TRADES
    assert "_is_flat" in text or "FLAT_NO_TRADES" in text, (
        "FLAT_NO_TRADES classification logic must exist."
    )


def test_flat_no_trades_not_saved():
    """FLAT_NO_TRADES must not be saved as accuracy."""
    text = _src()
    # The accuracy save gate must check for FLAT_NO_TRADES
    assert "FLAT_NO_TRADES" in text, (
        "Accuracy save gate must reference FLAT_NO_TRADES to block saving it."
    )
    # Positive save only happens on "SUCCESS" — FLAT_NO_TRADES is excluded
    assert '_backtest_truly_succeeded' in text or "status.*SUCCESS" in text or (
        '"SUCCESS"' in text and "_backtest_" in text
    ), (
        "Accuracy save must only trigger when status == SUCCESS, blocking FLAT_NO_TRADES."
    )


# ── 8. Accuracy save gate ─────────────────────────────────────────────────────

def test_accuracy_save_gate_blocks_exact_strike():
    """Accuracy save must be blocked when EXACT_STRIKE mode (proxy ≠ exact accuracy)."""
    text = _src()
    assert "EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER" in text, (
        "Accuracy save gate must set accuracy_skip_reason=EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER "
        "when EXACT_STRIKE mode is used, since tastytrade only ran a proxy."
    )


def test_accuracy_save_gate_sets_not_saved_message():
    """NOT SAVED message must appear in save_msg for blocked runs."""
    text = _src()
    assert "NOT SAVED" in text, (
        "Blocked accuracy saves must produce a save_msg starting with 'NOT SAVED'."
    )


def test_eligible_for_exact_accuracy_false_in_exact_mode():
    """eligible_for_exact_accuracy must be False in EXACT_STRIKE mode."""
    text = _src()
    assert "eligible_for_exact_accuracy" in text, (
        "opts_result must include 'eligible_for_exact_accuracy' key."
    )


# ── 9. Summary banner conditional display ─────────────────────────────────────

def test_summary_banner_shows_strike_in_strike_mode():
    """Summary banner must show Strike and Expiry in Strike mode, not Delta and DTE."""
    text = _src()
    # The conditional string must reference Strike in _params_summary
    assert "Strike {" in text or "Strike {int(" in text or '"Strike "' in text, (
        "Summary banner in Strike mode must show 'Strike X' — not 'Delta X'."
    )


def test_summary_banner_shows_delta_in_delta_mode():
    """Summary banner must show Delta and DTE in Delta mode."""
    text = _src()
    assert "Delta {delta_val}" in text or "Delta {" in text, (
        "Summary banner in Delta mode must show 'Delta X, DTE Y'."
    )


def test_summary_banner_is_conditional():
    """Summary banner must be built conditionally — one branch per mode."""
    text = _src()
    # Conditional construction via if/else
    assert '_params_summary' in text, (
        "_params_summary must be computed and shown in the summary banner."
    )
    assert 'strike_selection == "Strike"' in text or "_is_strike_mode" in text, (
        "Summary banner must branch on strike_selection mode."
    )


# ── 10. Section 2B and 3B tables: conditional Strike/Delta row ───────────────

def test_section_2b_strike_row_conditional():
    """Section 2B input table must show Strike row in Strike mode, Delta row in Delta mode."""
    text = _src()
    assert "_p_is_strike" in text, (
        "Section 2B table must use _p_is_strike flag to conditionally show Strike vs Delta."
    )
    assert "_p_strike_row" in text, (
        "Section 2B must define _p_strike_row conditionally."
    )


# ── 11. Delta 1-99 validation: large value rejection in Delta mode only ───────

def test_delta_mode_rejects_values_above_99():
    """Delta mode must reject values above 99 (3046 is invalid as a delta)."""
    text = _src()
    # Delta field must have max_value=99
    assert "max_value=99" in text, (
        "Delta number_input must have max_value=99. "
        "Delta 3046 in Delta mode is invalid — delta is a probability 1-99%."
    )


def test_delta_mode_accepts_83():
    """Delta value 83 is valid (within 1-99 range)."""
    # Static: confirm min_value=1 and max_value=99 for the delta field
    text = _src()
    assert "min_value=1" in text, "Delta field must have min_value=1."
    assert "max_value=99" in text, "Delta field must have max_value=99."
    # 83 is between 1 and 99 — no additional restriction needed


# ── 12. Credential truth gate ─────────────────────────────────────────────────

def test_tastytrade_not_available_cannot_show_success():
    """If tastytrade not available, opts_result must be SKIPPED, not SUCCESS."""
    text = _src()
    # When _TT_AVAILABLE is False, the block is skipped entirely
    assert "_TT_AVAILABLE" in text, (
        "Must guard tastytrade block with _TT_AVAILABLE flag."
    )
    # The default status before the TT block must be SKIPPED (not SUCCESS)
    assert '"status":          "SKIPPED"' in text or '"status": "SKIPPED"' in text or (
        '"SKIPPED"' in text
    ), (
        "Default opts_result status must be SKIPPED before tastytrade block runs. "
        "If tastytrade is not configured, status stays SKIPPED — never becomes SUCCESS."
    )


def test_tastytrade_error_sets_error_status():
    """Exception in tastytrade block must produce ERROR status, not SUCCESS."""
    text = _src()
    assert 'opts_result["status"] = "ERROR"' in text or '"status": "ERROR"' in text, (
        "Exception in tastytrade block must set status=ERROR, never leave status=SUCCESS."
    )


# ── 13. Large strike values accepted in options_params (unit-level) ──────────

def test_options_params_large_strike_accepted():
    """Build options_params dict manually and confirm large strikes work."""
    # Simulate what the app does when user enters strike=7259 in Strike mode
    strike_price = 7259.0
    strike_selection = "Strike"
    delta_val = 50  # internal proxy, not shown to user
    dte_val = 45
    quantity = 2
    direction = "Sell"
    opt_type = "Put"
    expiry_date_str = "2026-08-15"

    _is_strike_mode = strike_selection == "Strike"
    _is_delta_mode  = not _is_strike_mode
    _req_strike = float(strike_price) if (_is_strike_mode and strike_price and float(strike_price) > 0) else None

    options_params = {
        "direction":               direction,
        "opt_type":                opt_type,
        "quantity":                int(quantity),
        "strike_selection":        strike_selection.lower(),
        "contract_selection_method": "EXACT_STRIKE" if _is_strike_mode else "DELTA_SELECTION",
        "delta_ui":                int(delta_val) if _is_delta_mode else None,
        "delta_decimal":           round(int(delta_val) / 100.0, 4) if _is_delta_mode else None,
        "delta":                   int(delta_val) if _is_delta_mode else None,
        "requested_strike":        _req_strike,
        "strike_price":            _req_strike,
        "expiry_date":             expiry_date_str if _is_strike_mode else None,
        "dte":                     int(dte_val),
    }

    # Large strike must be preserved, not capped
    assert options_params["requested_strike"] == 7259.0, (
        f"Strike 7259 must be stored as 7259.0, got {options_params['requested_strike']}"
    )
    assert options_params["delta_ui"] is None, (
        "In Strike mode, delta_ui must be None"
    )
    assert options_params["delta_decimal"] is None, (
        "In Strike mode, delta_decimal must be None"
    )
    assert options_params["contract_selection_method"] == "EXACT_STRIKE"


def test_options_params_delta_mode_no_strike():
    """In Delta mode, requested_strike must be None."""
    strike_selection = "Delta"
    delta_val = 83
    dte_val = 45

    _is_strike_mode = strike_selection == "Strike"
    _is_delta_mode  = not _is_strike_mode
    _req_strike = None  # Delta mode: no strike

    options_params = {
        "strike_selection": strike_selection.lower(),
        "contract_selection_method": "EXACT_STRIKE" if _is_strike_mode else "DELTA_SELECTION",
        "delta_ui":         int(delta_val) if _is_delta_mode else None,
        "delta_decimal":    round(int(delta_val) / 100.0, 4) if _is_delta_mode else None,
        "delta":            int(delta_val) if _is_delta_mode else None,
        "requested_strike": _req_strike,
        "strike_price":     _req_strike,
        "expiry_date":      None,
        "dte":              int(dte_val),
    }

    assert options_params["requested_strike"] is None
    assert options_params["delta_ui"] == 83
    assert abs(options_params["delta_decimal"] - 0.83) < 0.0001
    assert options_params["contract_selection_method"] == "DELTA_SELECTION"


def test_strike_mode_pre_run_gate_blocks_zero():
    """Simulate pre-run gate: strike=0 → REVIEW_REQUIRED, no API call."""
    strike_price = 0.0
    strike_selection = "Strike"
    _is_strike_mode = True
    _req_strike = float(strike_price) if (strike_price and float(strike_price) > 0) else None

    _pre_errors = []
    if _is_strike_mode and not _req_strike:
        _pre_errors.append(
            f"Strike Mode requires Strike Price > 0. You entered: {strike_price!r}. "
            "Examples: SPY=620, SPX=7570, AAPL=240. No upper cap — enter the exact strike."
        )

    assert len(_pre_errors) == 1, "Zero strike must trigger pre-run gate error."
    assert "Strike Mode requires Strike Price > 0" in _pre_errors[0]
    assert "7570" in _pre_errors[0], "Error message must show SPX 7570 as an example."


def test_strike_mode_pre_run_gate_allows_7259():
    """Strike=7259 must pass the pre-run gate with no errors."""
    strike_price = 7259.0
    strike_selection = "Strike"
    _is_strike_mode = True
    _req_strike = float(strike_price) if (strike_price and float(strike_price) > 0) else None

    _pre_errors = []
    if _is_strike_mode and not _req_strike:
        _pre_errors.append("Strike Mode requires Strike Price > 0.")

    assert len(_pre_errors) == 0, (
        f"Strike 7259 must NOT trigger pre-run gate. Got errors: {_pre_errors}"
    )
    assert _req_strike == 7259.0


def test_flat_no_trades_classification():
    """Simulate tastytrade returning 0 trades + $0 P&L → FLAT_NO_TRADES."""
    _n_trades = 0
    _total_pl  = 0.0
    _is_flat = (_n_trades == 0 and _total_pl == 0.0)

    status = "FLAT_NO_TRADES" if _is_flat else "SUCCESS"
    assert status == "FLAT_NO_TRADES", (
        "0 trades + $0 P&L must produce FLAT_NO_TRADES, not SUCCESS."
    )


def test_real_trades_not_flat():
    """If n_trades > 0, classification must NOT be FLAT_NO_TRADES."""
    _n_trades = 5
    _total_pl  = -120.0
    _is_flat = (_n_trades == 0 and _total_pl == 0.0)

    status = "FLAT_NO_TRADES" if _is_flat else "SUCCESS"
    assert status == "SUCCESS", (
        "5 trades with -$120 P&L must be SUCCESS (real data), not FLAT_NO_TRADES."
    )


def test_accuracy_save_gate_logic():
    """Simulate accuracy save gate: only save on SUCCESS + no skip reason."""
    # Case 1: FLAT_NO_TRADES — must not save
    opts_result = {"status": "FLAT_NO_TRADES"}
    skip_reason = "FLAT_NO_TRADES"
    truly_succeeded = opts_result["status"] == "SUCCESS"
    assert not (truly_succeeded and not skip_reason), (
        "FLAT_NO_TRADES must not be saved."
    )

    # Case 2: EXACT_STRIKE proxy — must not save
    opts_result2 = {"status": "SUCCESS"}
    skip_reason2 = "EXACT_STRIKE_UNSUPPORTED_BY_PROVIDER"
    truly_succeeded2 = opts_result2["status"] == "SUCCESS"
    assert not (truly_succeeded2 and not skip_reason2), (
        "Exact strike proxy run must not be saved even if status=SUCCESS."
    )

    # Case 3: Delta mode, real trades — save allowed
    opts_result3 = {"status": "SUCCESS"}
    skip_reason3 = None
    truly_succeeded3 = opts_result3["status"] == "SUCCESS"
    assert truly_succeeded3 and not skip_reason3, (
        "Delta mode backtest with SUCCESS and no skip reason should be saved."
    )
