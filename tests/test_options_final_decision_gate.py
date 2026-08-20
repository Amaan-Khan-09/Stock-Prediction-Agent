"""Tests for options final decision gate — no BUY/SELL contradiction allowed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import ContractMethod, OptionsValidationState, VS
from agents.options_decision_gate import apply_final_decision_gate


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(**kwargs) -> OptionsValidationState:
    defaults = dict(
        requested_underlying="SPY",
        requested_option_type="CALL",
        requested_strike=620.0,
        requested_expiry="2026-07-17",
        requested_dte=1,
        requested_contracts=2,
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        exact_validation_status=VS.SUCCESS,
        tastytrade_status=VS.SUCCESS,
        options_backtest_decision="BUY",
        options_backtest_pnl=500.0,
        ai_stock_decision="BUY",
    )
    defaults.update(kwargs)
    return OptionsValidationState(**defaults)


# ── Test 1: Options validation SELL → NO_TRADE ────────────────────────────────

def test_options_validation_sell_blocks_buy():
    """If options validation says SELL, final output must be NO_TRADE — not BUY."""
    state = _make_state(
        options_backtest_decision="SELL",
        options_backtest_pnl=-1000.0,
        ai_stock_decision="BUY",  # AI says BUY but options says SELL
    )
    result = apply_final_decision_gate(state)
    assert result.final_agent_output in (VS.NO_TRADE, "NO_TRADE"), (
        f"Expected NO_TRADE, got {result.final_agent_output}"
    )
    assert result.order_status == VS.NOT_SUBMITTED


def test_negative_pnl_blocks_order():
    """Negative options P&L must result in NO_TRADE."""
    state = _make_state(
        options_backtest_decision="BUY",
        options_backtest_pnl=-500.0,
    )
    result = apply_final_decision_gate(state)
    assert result.final_agent_output in (VS.NO_TRADE, "NO_TRADE")
    assert result.order_status == VS.NOT_SUBMITTED


# ── Test 2: Exact validation failed + delta proxy success ─────────────────────

def test_exact_failed_proxy_success_no_order_by_default():
    """Exact strike failed + proxy succeeded → REVIEW_REQUIRED by default (no env override)."""
    import os
    os.environ.pop("ALLOW_APPROXIMATE_OPTIONS_ORDER", None)
    # Reload module to get fresh value
    import importlib
    import agents.options_decision_gate as gate_mod
    importlib.reload(gate_mod)
    from agents.options_decision_gate import apply_final_decision_gate as adg

    state = OptionsValidationState(
        requested_underlying="SPY",
        requested_option_type="CALL",
        requested_strike=620.0,
        requested_expiry="2026-07-17",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        exact_validation_status=VS.FAILED,
        delta_proxy_status=VS.SUCCESS,
        tastytrade_status="SUCCESS_DELTA_PROXY",
        options_backtest_decision="BUY",
        options_backtest_pnl=500.0,
        eligible_for_exact_accuracy=False,
        fallback_used=True,
        validation_type=ContractMethod.DELTA_PROXY,
        ai_stock_decision="BUY",
    )
    result = adg(state)
    # Should be REVIEW_REQUIRED (not BUY) because exact failed & ALLOW_APPROXIMATE is false
    assert result.final_agent_output in (VS.REVIEW_REQUIRED, VS.NO_TRADE)
    assert result.order_status == VS.NOT_SUBMITTED
    assert result.eligible_for_exact_accuracy is False


def test_delta_proxy_success_not_counted_as_exact():
    """Delta proxy must never set eligible_for_exact_accuracy=True."""
    state = OptionsValidationState(
        requested_underlying="SPY",
        requested_option_type="CALL",
        requested_strike=620.0,
        requested_expiry="2026-07-17",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        exact_validation_status=VS.FAILED,
        delta_proxy_status=VS.SUCCESS,
        tastytrade_status="SUCCESS_DELTA_PROXY",
        options_backtest_decision="BUY",
        options_backtest_pnl=500.0,
        fallback_used=True,
        validation_type=ContractMethod.DELTA_PROXY,
    )
    assert state.eligible_for_exact_accuracy is False


# ── Test 3: Missing expiry/DTE → REVIEW_REQUIRED ─────────────────────────────

def test_missing_expiry_blocks():
    state = OptionsValidationState(
        requested_underlying="SPY",
        requested_option_type="CALL",
        requested_strike=620.0,
        requested_expiry=None,
        requested_dte=None,
        contract_selection_method=ContractMethod.EXACT_STRIKE,
    )
    result = apply_final_decision_gate(state)
    assert result.final_agent_output == VS.REVIEW_REQUIRED
    assert result.order_status == VS.NOT_SUBMITTED
    assert "expiry" in result.reason.lower() or "dte" in result.reason.lower()


# ── Test 4: Missing strike in exact mode → REVIEW_REQUIRED ───────────────────

def test_missing_strike_exact_mode():
    state = OptionsValidationState(
        requested_underlying="SPY",
        requested_option_type="CALL",
        requested_strike=None,
        requested_expiry="2026-07-17",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
    )
    result = apply_final_decision_gate(state)
    assert result.final_agent_output == VS.REVIEW_REQUIRED
    assert result.order_status == VS.NOT_SUBMITTED
    assert "strike" in result.reason.lower()


# ── Test 5: AI BUY + options SELL → NO_TRADE ─────────────────────────────────

def test_ai_buy_options_sell_no_contradiction():
    """AI says BUY but options validation says SELL → NO_TRADE always."""
    state = _make_state(
        options_backtest_decision="SELL",
        options_backtest_pnl=-1500.0,
        ai_stock_decision="BUY",
    )
    result = apply_final_decision_gate(state)
    # Must be NO_TRADE — never BUY when options says SELL
    assert result.final_agent_output not in ("BUY", "ENTER"), (
        "AI BUY must not override options SELL"
    )
    assert result.final_agent_output in (VS.NO_TRADE, VS.REVIEW_REQUIRED)
    assert result.order_status == VS.NOT_SUBMITTED


# ── Test 6: Tastytrade unavailable → REVIEW_REQUIRED ─────────────────────────

def test_tastytrade_unavailable():
    state = OptionsValidationState(
        requested_underlying="SPY",
        requested_option_type="CALL",
        requested_strike=620.0,
        requested_expiry="2026-07-17",
        contract_selection_method=ContractMethod.EXACT_STRIKE,
        exact_validation_status=VS.SUCCESS,
        tastytrade_status=VS.NOT_CONFIGURED,
    )
    result = apply_final_decision_gate(state)
    assert result.final_agent_output == VS.REVIEW_REQUIRED
    assert result.order_status == VS.NOT_SUBMITTED


# ── Test 7: Delta selection happy path ───────────────────────────────────────

def test_delta_selection_happy_path():
    state = OptionsValidationState(
        requested_underlying="SPY",
        requested_option_type="CALL",
        contract_selection_method=ContractMethod.DELTA_SELECTION,
        delta_ui=30,
        delta_decimal=0.30,
        requested_dte=45,
        tastytrade_status=VS.SUCCESS,
        options_backtest_decision="BUY",
        options_backtest_pnl=300.0,
        ai_stock_decision="BUY",
    )
    result = apply_final_decision_gate(state)
    assert result.final_agent_output in ("BUY", "ENTER")
    # Delta mode never generates exact_accuracy
    assert result.eligible_for_exact_accuracy is False
