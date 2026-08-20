"""Options final decision gate — eliminates BUY/SELL contradictions.

Rules (from implementation spec Part G):

1. Parse signal → validate required contract fields.
2. Check exact contract existence (Alpaca / chain check).
3. Run exact validation / delta validation.
4. Run options strategy validation (tastytrade backtest).
5. Use options strategy validation as FINAL GATE.
6. AI stock prediction is supporting context ONLY — cannot override options validation.

Contradiction fix:
  If options_validation.decision = SELL  →  final_agent_output = NO_TRADE
  If options_validation.pnl < 0         →  final_agent_output = NO_TRADE
  If AI says BUY but options says SELL   →  final_agent_output = NO_TRADE

Safety:
  - Exact strike validation failed & exact mode requested → REVIEW_REQUIRED, NOT_SUBMITTED
  - Missing expiry/DTE                                    → REVIEW_REQUIRED, NOT_SUBMITTED
  - Missing strike in exact mode                          → REVIEW_REQUIRED, NOT_SUBMITTED
  - ALLOW_APPROXIMATE_OPTIONS_ORDER env var controls paper order fallback
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import (
    ContractMethod, OptionsValidationState, VS,
)

# Config: set ALLOW_APPROXIMATE_OPTIONS_ORDER=true to allow paper orders when
# exact strike validation fails (proxy only). Default = false (blocked).
ALLOW_APPROXIMATE = os.environ.get("ALLOW_APPROXIMATE_OPTIONS_ORDER", "false").lower() == "true"


# ──────────────────────────────────────────────────────────────────────────────
# Public: apply_final_decision_gate
# ──────────────────────────────────────────────────────────────────────────────

def apply_final_decision_gate(state: OptionsValidationState) -> OptionsValidationState:
    """Apply the full decision hierarchy to a populated OptionsValidationState.

    Modifies state.final_agent_output, state.order_status, state.reason in place.
    Returns the same state object.
    """
    # ── Gate 1: Missing required fields ───────────────────────────────────
    if state.contract_selection_method == ContractMethod.EXACT_STRIKE:
        if state.requested_strike is None:
            return _block(state, VS.REVIEW_REQUIRED, VS.NOT_SUBMITTED,
                          "Missing strike price. Exact strike mode requires a specific strike.")

        if state.requested_expiry is None and state.requested_dte is None:
            return _block(state, VS.REVIEW_REQUIRED, VS.NOT_SUBMITTED,
                          "Missing expiry date and DTE. Cannot validate exact strike without expiration.")

    if state.requested_underlying is None:
        return _block(state, VS.REVIEW_REQUIRED, VS.NOT_SUBMITTED,
                      "Missing underlying symbol.")

    # ── Gate 2: Exact strike validation outcome ────────────────────────────
    exact_failed = (
        state.contract_selection_method == ContractMethod.EXACT_STRIKE
        and state.exact_validation_status == VS.FAILED
    )
    exact_success = (
        state.contract_selection_method == ContractMethod.EXACT_STRIKE
        and state.exact_validation_status == VS.SUCCESS
    )

    if exact_failed:
        # Proxy may have run — check if it succeeded
        proxy_ok = state.delta_proxy_status == VS.SUCCESS

        if proxy_ok and ALLOW_APPROXIMATE:
            # Allowed only as paper order with explicit warning
            state.manual_override_used = True
            state.fallback_used        = True
            state.validation_type      = ContractMethod.DELTA_PROXY
            state.eligible_for_exact_accuracy = False
            # Don't auto-approve yet — still need options validation gate below
        else:
            # Block: exact failed, proxy not approved
            reason = (
                "Exact strike validation failed. Delta proxy is approximate reference only "
                "and cannot approve an exact strike paper order. "
                "Set ALLOW_APPROXIMATE_OPTIONS_ORDER=true in .env to override (paper only)."
            )
            if proxy_ok:
                reason += " Delta proxy succeeded but ALLOW_APPROXIMATE_OPTIONS_ORDER is false."
            return _block(state, VS.REVIEW_REQUIRED, VS.NOT_SUBMITTED, reason)

    if exact_success:
        state.validation_type = ContractMethod.EXACT_STRIKE
        state.eligible_for_exact_accuracy = True
        state.fallback_used = False

    # ── Gate 3: Delta selection mode ──────────────────────────────────────
    if state.contract_selection_method == ContractMethod.DELTA_SELECTION:
        state.exact_validation_status = VS.NOT_REQUESTED
        state.validation_type         = ContractMethod.DELTA_SELECTION
        state.eligible_for_exact_accuracy = False

    # ── Gate 4: Options strategy (tastytrade) validation result ───────────
    bt_decision = (state.options_backtest_decision or "").upper()
    bt_pnl      = state.options_backtest_pnl

    if state.tastytrade_status not in (VS.SUCCESS, "SUCCESS_DELTA_PROXY"):
        # Tastytrade not available / failed → cannot approve
        return _block(state, VS.REVIEW_REQUIRED, VS.NOT_SUBMITTED,
                      f"Options strategy validation not available: {state.tastytrade_status}. "
                      "Cannot determine final action without backtest result.")

    if bt_decision == "SELL":
        return _block(state, VS.NO_TRADE, VS.NOT_SUBMITTED,
                      "Options strategy validation returned SELL. Agent will not enter.")

    if bt_pnl is not None and float(bt_pnl) < 0:
        return _block(state, VS.NO_TRADE, VS.NOT_SUBMITTED,
                      f"Options strategy validation has negative P&L ({bt_pnl:.2f}). "
                      "Agent will not enter.")

    # ── Gate 5: AI stock prediction vs options validation contradiction ────
    ai_dec = (state.ai_stock_decision or "").upper()
    if ai_dec == "SELL" and bt_decision in ("BUY", "ENTER"):
        # AI says SELL but options says BUY — options wins but log conflict
        state.raw_extras_note = ("AI stock prediction=SELL conflicts with options "
                                 "validation=BUY. Options validation takes precedence.")

    if ai_dec in ("BUY", "ENTER") and bt_decision == "SELL":
        return _block(state, VS.NO_TRADE, VS.NOT_SUBMITTED,
                      "AI stock prediction=BUY but options validation=SELL. "
                      "Underlying stock prediction cannot override options strategy validation.")

    # ── Gate 6: Exact contract existence (Alpaca) ─────────────────────────
    if (state.contract_selection_method == ContractMethod.EXACT_STRIKE
            and state.exact_contract_found is False):
        return _block(state, VS.REVIEW_REQUIRED, VS.NOT_SUBMITTED,
                      "Exact Alpaca contract not found for the requested strike and expiry.")

    # ── Gate 7: Approve ───────────────────────────────────────────────────
    approval_decision = bt_decision if bt_decision else "BUY"
    state.final_agent_output = approval_decision
    state.order_status       = VS.SUBMITTED  # paper only

    if state.manual_override_used:
        state.reason = (
            f"APPROXIMATE VALIDATION USED — NOT EXACT STRIKE VALIDATED. "
            f"Options backtest: {bt_decision}, P&L: {bt_pnl}. "
            "Paper order submitted with manual override."
        )
    elif state.fallback_used:
        state.reason = (
            f"Delta proxy validation used (fallback). "
            f"Options backtest: {bt_decision}, P&L: {bt_pnl}. "
            "Eligible for delta-based accuracy only (not exact strike accuracy)."
        )
    else:
        state.reason = (
            f"Exact validation: {state.exact_validation_status}. "
            f"Options backtest: {bt_decision}, P&L: {bt_pnl}. "
            "All gates passed. Paper order submitted."
        )

    return state


# ──────────────────────────────────────────────────────────────────────────────
# Public: quick helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_options_inputs(state: OptionsValidationState) -> tuple[bool, str]:
    """Pre-flight check. Returns (ok, reason)."""
    if not state.requested_underlying:
        return False, "Missing underlying symbol."
    if state.contract_selection_method == ContractMethod.EXACT_STRIKE:
        if state.requested_strike is None:
            return False, "Exact strike mode requires a strike price."
        if state.requested_expiry is None and state.requested_dte is None:
            return False, "Exact strike mode requires expiry date or DTE."
    if state.contract_selection_method == ContractMethod.DELTA_SELECTION:
        if state.delta_ui is None and state.delta_decimal is None:
            return False, "Delta selection mode requires delta value."
        if state.requested_dte is None and state.requested_expiry is None:
            return False, "Delta selection mode requires DTE or expiry."
    return True, ""


def compute_strike_distance(
    requested_strike: float,
    origin_underlying_price: float,
) -> float:
    """pct distance between requested strike and current underlying price."""
    if origin_underlying_price == 0:
        return 0.0
    return round(
        (requested_strike - origin_underlying_price) / origin_underlying_price * 100, 2
    )


# ──────────────────────────────────────────────────────────────────────────────
# Private
# ──────────────────────────────────────────────────────────────────────────────

def _block(
    state: OptionsValidationState,
    final_output: str,
    order_status: str,
    reason: str,
) -> OptionsValidationState:
    state.final_agent_output = final_output
    state.order_status       = order_status
    state.reason             = reason
    return state


# Attach raw_extras_note dynamically if needed
OptionsValidationState.raw_extras_note = None  # type: ignore[attr-defined]
