"""Signal Review Service — Part J of the implementation spec.

Orchestrates: parse → validate inputs → run AI stock prediction →
              run options backtest → apply decision gate → return full output.

Input:  review_signal(signal_payload: dict) -> dict
Output: full result dict matching Part J schema.

Can be imported by Discord bot, CLI, or Streamlit.
Zero Streamlit dependency.
"""
from __future__ import annotations

import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import (
    AssetType, ContractMethod, OptionsContractRequest,
    OptionsValidationState, VS,
    SignalAction,
)
from agents.discord_parser import parse_signal
from agents.options_decision_gate import (
    apply_final_decision_gate,
    compute_strike_distance,
    validate_options_inputs,
)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def review_signal(signal_payload: dict) -> dict:
    """Review a trading signal from any source.

    Input schema (all fields optional except raw_message or pre-parsed asset_type):

    {
      "source":         "discord" | "whatsapp" | "manual",
      "channel":        "stock-signals",
      "message_id":     "...",
      "reply_to":       "...",
      "timestamp":      "...",
      "raw_message":    "Bought SPY 620C @ 1.43, 2 contracts, exp 2026-07-17",

      # Optional pre-parsed fields (skip parser if provided)
      "asset_type":     "option" | "stock",
      "action":         "BUY" | "SELL" | ...,
      "underlying":     "SPY",
      "option_type":    "CALL" | "PUT",
      "strike":         620,
      "premium":        1.43,
      "contracts":      2,
      "expiry":         "2026-07-17",
      "dte":            1,

      # Contract selection
      "contract_selection_method": "EXACT_STRIKE" | "DELTA_SELECTION",
      "delta_ui":       30,

      # Stock prediction context
      "historical_context_start_date": "2025-01-01",
      "prediction_origin_date":        "2026-07-17",
      "decision_horizon_days":         30,
      "initial_capital":               50000,
      "benchmark":                     "SPY",

      # Misc
      "context_window": "same_day",
      "parser_confidence": 0.95,
    }

    Returns full result dict with parsed_signal, ai_prediction, options_validation,
    final_agent_output, order_status, reason.
    """
    try:
        return _review_impl(signal_payload)
    except Exception as exc:
        return {
            "parsed_successfully": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "final_agent_output": VS.REVIEW_REQUIRED,
            "order_status":       VS.NOT_SUBMITTED,
            "reason":             f"Internal error during signal review: {exc}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Implementation
# ──────────────────────────────────────────────────────────────────────────────

def _review_impl(payload: dict) -> dict:
    raw_message = payload.get("raw_message", "")
    message_id  = payload.get("message_id")
    reply_to    = payload.get("reply_to")

    # ── Step 1: Parse signal ──────────────────────────────────────────────
    parsed_sig = parse_signal(
        raw_message    = raw_message,
        reply_to_message_id = reply_to,
        message_id     = message_id,
    )

    # Allow override with pre-parsed fields from payload
    if payload.get("underlying"):
        parsed_sig.underlying = str(payload["underlying"]).upper()
    if payload.get("option_type"):
        parsed_sig.option_type = payload["option_type"].upper()
    if payload.get("strike") is not None:
        parsed_sig.strike = float(payload["strike"])
    if payload.get("premium") is not None:
        parsed_sig.premium = float(payload["premium"])
    if payload.get("contracts") is not None:
        parsed_sig.contracts = int(payload["contracts"])
    if payload.get("expiry") or payload.get("expiry_date"):
        parsed_sig.expiry_date = payload.get("expiry") or payload.get("expiry_date")
    if payload.get("dte") is not None:
        parsed_sig.dte = int(payload["dte"])
    if payload.get("asset_type"):
        at = payload["asset_type"].upper()
        parsed_sig.asset_type = AssetType.OPTION if "OPT" in at else AssetType.STOCK
    if payload.get("action"):
        parsed_sig.action = payload["action"].upper()
    if payload.get("contract_selection_method"):
        parsed_sig.contract_selection_method = payload["contract_selection_method"].upper()
    if payload.get("delta_ui") is not None:
        parsed_sig.delta_ui      = int(payload["delta_ui"])
        parsed_sig.delta_decimal = round(parsed_sig.delta_ui / 100.0, 4)

    # ── Step 2: Build contract request ────────────────────────────────────
    method = parsed_sig.contract_selection_method
    if parsed_sig.strike is not None:
        method = ContractMethod.EXACT_STRIKE

    contract_req = OptionsContractRequest(
        underlying   = parsed_sig.underlying,
        option_type  = parsed_sig.option_type,
        strike       = parsed_sig.strike,
        expiry_date  = parsed_sig.expiry_date,
        dte          = parsed_sig.dte,
        premium      = parsed_sig.premium,
        contracts    = parsed_sig.contracts or 1,
        side         = parsed_sig.action,
        contract_selection_method = method,
        delta_ui     = parsed_sig.delta_ui,
        delta_decimal = parsed_sig.delta_decimal,
    )
    contract_req.occ_symbol = contract_req.build_occ_symbol()

    # ── Step 3: Build validation state ────────────────────────────────────
    state = OptionsValidationState(
        requested_underlying   = contract_req.underlying,
        requested_option_type  = contract_req.option_type,
        requested_strike       = contract_req.strike,
        requested_expiry       = contract_req.expiry_date,
        requested_dte          = contract_req.dte,
        requested_premium      = contract_req.premium,
        requested_contracts    = contract_req.contracts,
        contract_selection_method = method,
        delta_ui               = contract_req.delta_ui,
        delta_decimal          = contract_req.delta_decimal,
        occ_symbol             = contract_req.occ_symbol,
    )

    # ── Step 4: Pre-flight validation ─────────────────────────────────────
    inputs_ok, inputs_err = validate_options_inputs(state)
    if not inputs_ok:
        state.final_agent_output = VS.REVIEW_REQUIRED
        state.order_status       = VS.NOT_SUBMITTED
        state.reason             = inputs_err
        return _build_result(parsed_sig, contract_req, state, payload)

    # ── Step 5: Strike distance check ─────────────────────────────────────
    if state.requested_strike is not None:
        origin_price = _get_origin_price(payload)
        if origin_price is not None:
            state.strike_distance_pct = compute_strike_distance(
                state.requested_strike, origin_price
            )

    # ── Step 6: Try exact or delta options backtest ────────────────────────
    if parsed_sig.asset_type == AssetType.OPTION:
        _run_backtest(state, payload)

    # ── Step 7: AI stock prediction (supporting context) ──────────────────
    if parsed_sig.underlying:
        _run_ai_prediction(state, payload)

    # ── Step 8: Apply final decision gate ─────────────────────────────────
    if parsed_sig.asset_type == AssetType.OPTION:
        apply_final_decision_gate(state)
    else:
        # Stock signal — use AI decision directly
        state.final_agent_output = state.ai_stock_decision or VS.REVIEW_REQUIRED
        state.order_status       = VS.NOT_SUBMITTED  # never auto-order
        state.reason             = "Stock signal — no auto-order. AI prediction is advisory."

    return _build_result(parsed_sig, contract_req, state, payload)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-steps
# ──────────────────────────────────────────────────────────────────────────────

def _run_backtest(state: OptionsValidationState, payload: dict) -> None:
    """Run tastytrade backtest and populate state."""
    try:
        from prediction_validation_service import _run_options_backtest

        symbol     = state.requested_underlying or ""
        origin     = payload.get("prediction_origin_date") or date.today().isoformat()
        target     = _derive_target(payload)
        direction  = payload.get("action", state.requested_option_type or "Buy")
        opt_type   = state.requested_option_type or "Call"
        quantity   = state.requested_contracts or 1
        dte        = state.requested_dte or 45
        delta      = state.delta_ui or 30

        # Map action → tastytrade direction
        dir_clean = "Buy" if (direction or "").upper() in ("BUY", "CALL") else "Sell"
        type_clean = "Call" if (opt_type or "").upper() == "CALL" else "Put"

        if state.contract_selection_method == ContractMethod.EXACT_STRIKE:
            # Attempt exact strike run (tastytrade doesn't support fixed strike directly,
            # so this is modeled as delta=50 which approximates ATM — mark as proxy)
            # Real exact strike validation would require historical options data provider.
            state.exact_validation_status = VS.FAILED
            state.eligible_for_exact_accuracy = False

            # Try delta proxy as reference
            proxy_result = _run_options_backtest(
                symbol=symbol, start_date=origin, end_date=target,
                direction=dir_clean, opt_type=type_clean,
                quantity=quantity, delta=delta, dte=dte,
                strike_selection="delta",
            )
            if proxy_result.get("status") == "SUCCESS":
                state.delta_proxy_status = VS.SUCCESS
                state.tastytrade_status  = "SUCCESS_DELTA_PROXY"
                state.validation_type    = ContractMethod.DELTA_PROXY
                state.fallback_used      = True
                _populate_backtest_result(state, proxy_result)
            else:
                state.delta_proxy_status = VS.FAILED
                state.tastytrade_status  = VS.FAILED
        else:
            # Delta selection — run directly
            result = _run_options_backtest(
                symbol=symbol, start_date=origin, end_date=target,
                direction=dir_clean, opt_type=type_clean,
                quantity=quantity, delta=delta, dte=dte,
                strike_selection="delta",
            )
            if result.get("status") == "SUCCESS":
                state.delta_proxy_status = VS.NOT_REQUESTED
                state.tastytrade_status  = VS.SUCCESS
                state.validation_type    = ContractMethod.DELTA_SELECTION
                _populate_backtest_result(state, result)
            else:
                state.tastytrade_status = result.get("status", VS.FAILED)

    except Exception as exc:
        state.tastytrade_status = f"ERROR: {exc}"


def _populate_backtest_result(state: OptionsValidationState, result: dict) -> None:
    pnl = result.get("profit_loss")
    wr  = result.get("win_rate")
    state.options_backtest_pnl      = float(pnl) if pnl is not None else None
    state.options_backtest_win_rate = float(wr)  if wr  is not None else None
    state.options_backtest_raw      = result

    # Derive decision from P&L
    if state.options_backtest_pnl is not None:
        state.options_backtest_decision = "BUY" if state.options_backtest_pnl > 0 else "SELL"
    else:
        state.options_backtest_decision = VS.NOT_AVAILABLE


def _run_ai_prediction(state: OptionsValidationState, payload: dict) -> None:
    """Run AI stock prediction and attach as supporting context."""
    try:
        from prediction_validation_service import run_stock_price_validation
        ctx_start = payload.get("historical_context_start_date",
                                _months_ago(payload.get("prediction_origin_date"), 12))
        origin    = payload.get("prediction_origin_date", date.today().isoformat())
        horizon   = int(payload.get("decision_horizon_days", 30))
        capital   = float(payload.get("initial_capital", 50_000))
        benchmark = str(payload.get("benchmark", "SPY")).upper()

        spi = {
            "symbol":                        state.requested_underlying,
            "historical_context_start_date": ctx_start,
            "prediction_origin_date":        origin,
            "decision_horizon_days":         horizon,
            "initial_capital":               capital,
            "benchmark":                     benchmark,
            "validation_mode":               "horizon_days",
            "price_basis":                   "close",
        }
        result = run_stock_price_validation(spi)
        ai = result.get("ai_prediction", {})
        state.ai_stock_decision   = ai.get("decision")
        state.ai_stock_return_pct = ai.get("predicted_return_pct")
        state.ai_confidence       = ai.get("confidence_score")
        state.ai_risk             = ai.get("risk_score")
    except Exception:
        state.ai_stock_decision = VS.NOT_AVAILABLE


def _get_origin_price(payload: dict) -> Optional[float]:
    """Try to get origin price for strike distance calc."""
    try:
        from prediction_validation_service import run_stock_price_validation
        symbol = payload.get("underlying", "")
        origin = payload.get("prediction_origin_date", date.today().isoformat())
        if not symbol or not origin:
            return None
        from historical_price_service import fetch_price_history, filter_history_up_to
        hist, _ = fetch_price_history(symbol, min_days=10)
        bars    = filter_history_up_to(hist, origin) if hist else []
        if bars:
            return float(bars[-1].get("close", 0))
        return None
    except Exception:
        return None


def _derive_target(payload: dict) -> str:
    origin  = payload.get("prediction_origin_date", date.today().isoformat())
    horizon = int(payload.get("decision_horizon_days", 30))
    try:
        dt = datetime.strptime(origin, "%Y-%m-%d")
        from datetime import timedelta
        return (dt + timedelta(days=horizon)).strftime("%Y-%m-%d")
    except Exception:
        return date.today().isoformat()


def _months_ago(origin: Optional[str], months: int = 12) -> str:
    try:
        dt = datetime.strptime(origin or date.today().isoformat(), "%Y-%m-%d")
        year  = dt.year - (months // 12)
        month = dt.month - (months % 12)
        if month <= 0:
            month += 12
            year  -= 1
        return f"{year:04d}-{month:02d}-01"
    except Exception:
        return "2024-01-01"


def _build_result(parsed_sig, contract_req, state, payload) -> dict:
    return {
        "parsed_successfully": True,
        "source":     payload.get("source", "unknown"),
        "channel":    payload.get("channel", ""),
        "message_id": payload.get("message_id"),
        "timestamp":  payload.get("timestamp", datetime.utcnow().isoformat()),

        "parsed_signal": {
            "raw_message":    parsed_sig.raw_message,
            "asset_type":     parsed_sig.asset_type,
            "action":         parsed_sig.action,
            "underlying":     parsed_sig.underlying,
            "option_type":    parsed_sig.option_type,
            "strike":         parsed_sig.strike,
            "expiry_date":    parsed_sig.expiry_date,
            "dte":            parsed_sig.dte,
            "premium":        parsed_sig.premium,
            "contracts":      parsed_sig.contracts,
            "delta_ui":       parsed_sig.delta_ui,
            "delta_decimal":  parsed_sig.delta_decimal,
            "parser_confidence": parsed_sig.parser_confidence,
            "missing_fields": parsed_sig.missing_fields,
            "high_risk_tag":  parsed_sig.high_risk_tag,
            "contract_selection_method": parsed_sig.contract_selection_method,
        },

        "requested_contract": {
            "underlying":   contract_req.underlying,
            "option_type":  contract_req.option_type,
            "strike":       contract_req.strike,
            "expiry":       contract_req.expiry_date,
            "dte":          contract_req.dte,
            "contracts":    contract_req.contracts,
            "premium":      contract_req.premium,
            "occ_symbol":   contract_req.occ_symbol,
            "selection_method": contract_req.contract_selection_method,
        },

        "ai_stock_prediction": {
            "decision":             state.ai_stock_decision,
            "predicted_return_pct": state.ai_stock_return_pct,
            "confidence":           state.ai_confidence,
            "risk":                 state.ai_risk,
        },

        "options_validation": {
            "status":                 state.tastytrade_status,
            "decision":               state.options_backtest_decision,
            "pnl":                    state.options_backtest_pnl,
            "win_rate":               state.options_backtest_win_rate,
            "validation_type":        state.validation_type,
            "exact_strike_status":    state.exact_validation_status,
            "delta_proxy_status":     state.delta_proxy_status,
            "fallback_used":          state.fallback_used,
            "eligible_for_exact_accuracy": state.eligible_for_exact_accuracy,
            "strike_distance_pct":    state.strike_distance_pct,
            "occ_symbol":             state.occ_symbol,
        },

        "final_agent_output": state.final_agent_output,
        "order_status":       state.order_status,
        "manual_override_used": state.manual_override_used,
        "reason":             state.reason,

        "debug": state.to_display_dict(),
    }
