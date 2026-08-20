"""Tests for options contract selection: exact strike, delta, OCC symbol, strike range."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import (
    ContractMethod, OptionsContractRequest, OptionSide, VS,
)
from agents.options_decision_gate import compute_strike_distance, validate_options_inputs
from models.options_models import OptionsValidationState


# ── Test 1: Exact strike OCC symbol generation ────────────────────────────────

def test_occ_symbol_spy_620_call():
    req = OptionsContractRequest(
        underlying="SPY", option_type=OptionSide.CALL,
        strike=620.0, expiry_date="2026-07-17",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
    )
    occ = req.build_occ_symbol()
    assert occ == "SPY260717C00620000", f"Got: {occ}"


def test_occ_symbol_spx_7570_call():
    """Large strike (7570) must be accepted without restriction."""
    req = OptionsContractRequest(
        underlying="SPX", option_type=OptionSide.CALL,
        strike=7570.0, expiry_date="2026-07-17",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
    )
    occ = req.build_occ_symbol()
    # SPX260717C07570000
    assert occ is not None
    assert "7570000" in occ, f"Got: {occ}"


def test_occ_symbol_aapl_240_put():
    req = OptionsContractRequest(
        underlying="AAPL", option_type=OptionSide.PUT,
        strike=240.0, expiry_date="2026-08-21",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
    )
    occ = req.build_occ_symbol()
    assert occ is not None
    assert "P" in occ
    assert "AAPL" in occ


def test_occ_symbol_missing_fields():
    req = OptionsContractRequest(underlying="SPY", option_type=OptionSide.CALL)
    occ = req.build_occ_symbol()
    assert occ is None  # Missing strike and expiry


# ── Test 2: Strike distance calculation ──────────────────────────────────────

def test_strike_distance_above():
    dist = compute_strike_distance(requested_strike=630.0, origin_underlying_price=600.0)
    assert abs(dist - 5.0) < 0.01, f"Got {dist}"


def test_strike_distance_below():
    dist = compute_strike_distance(requested_strike=570.0, origin_underlying_price=600.0)
    assert abs(dist - (-5.0)) < 0.01, f"Got {dist}"


def test_large_strike_spx_distance():
    """SPX 7570 with underlying at 7500 should give ~0.93% distance."""
    dist = compute_strike_distance(7570.0, 7500.0)
    assert abs(dist - (70 / 7500 * 100)) < 0.01, f"Got {dist}"


# ── Test 3: Input validation ──────────────────────────────────────────────────

def test_exact_mode_missing_strike():
    state = OptionsValidationState(
        requested_underlying="SPY",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        requested_strike=None,
        requested_expiry="2026-07-17",
    )
    ok, reason = validate_options_inputs(state)
    assert not ok
    assert "strike" in reason.lower()


def test_exact_mode_missing_expiry_and_dte():
    state = OptionsValidationState(
        requested_underlying="SPY",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        requested_strike=620.0,
        requested_expiry=None,
        requested_dte=None,
    )
    ok, reason = validate_options_inputs(state)
    assert not ok
    assert "expiry" in reason.lower() or "dte" in reason.lower()


def test_exact_mode_valid():
    state = OptionsValidationState(
        requested_underlying="SPY",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        requested_strike=620.0,
        requested_expiry="2026-07-17",
    )
    ok, reason = validate_options_inputs(state)
    assert ok, f"Should be valid, got: {reason}"


def test_delta_mode_missing_delta():
    state = OptionsValidationState(
        requested_underlying="SPY",
        contract_selection_method=ContractMethod.DELTA_SELECTION,
        delta_ui=None,
        delta_decimal=None,
        requested_dte=45,
    )
    ok, reason = validate_options_inputs(state)
    assert not ok
    assert "delta" in reason.lower()


def test_delta_mode_valid():
    state = OptionsValidationState(
        requested_underlying="SPY",
        contract_selection_method=ContractMethod.DELTA_SELECTION,
        delta_ui=30,
        delta_decimal=0.30,
        requested_dte=45,
    )
    ok, reason = validate_options_inputs(state)
    assert ok, f"Should be valid, got: {reason}"
