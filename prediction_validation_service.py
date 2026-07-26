"""
Prediction Validation Service — Clean backend for Discord/Amaan.

Zero Streamlit dependency. Importable from any script or bot.

Architecture guarantee (same as UI):
  Historical Context Window   : historical_context_start_date -> prediction_origin_date  (AI only)
  Prediction/Validation Window: prediction_origin_date -> target_date                    (AI + validation)

Options backtest ONLY runs for prediction window -- NEVER historical context window.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from historical_price_service import (
    fetch_price_history,
    filter_history_up_to,
    get_context_summary,
)
from stock_prediction_agent import (
    run_stock_prediction,
    build_stock_prediction_hash,
    validate_stock_prediction_input,
)
from stock_walkforward_validator import run_stock_validation
from stock_accuracy_engine import save_stock_prediction_record


# ══════════════════════════════════════════════════════════════════════════════
# STOCK PRICE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


def run_stock_price_validation(input_dict: dict) -> dict:
    """
    Run a single walk-forward stock price validation.

    Required input_dict keys:
      symbol                        : str  e.g. "TSLA"
      historical_context_start_date : str  YYYY-MM-DD
      prediction_origin_date        : str  YYYY-MM-DD
      target_date                   : str  YYYY-MM-DD (or omit + set decision_horizon_days)
      decision_horizon_days         : int  (used if target_date not provided)
      initial_capital               : float
      benchmark                     : str  e.g. "SPY"

    Optional:
      validation_mode : "horizon_days" | "calendar_month"  (default: "horizon_days")
      price_basis     : "close" | "open" | "high" | "low"  (default: "close")

    Returns dict with keys:
      status, stock_prediction_input_hash, ai_prediction, actual_validation, comparison,
      error (if failed)
    """
    spi = _build_spi(input_dict)

    valid, err = validate_stock_prediction_input(spi)
    if not valid:
        return {"status": "FAILED", "error": f"Input validation failed: {err}", "spi": spi}

    input_hash = build_stock_prediction_hash(spi)

    today    = date.today()
    ctx_start = spi.get("historical_context_start_date", "")
    origin   = spi.get("prediction_origin_date", "")

    try:
        ctx_dt   = datetime.strptime(ctx_start, "%Y-%m-%d").date()
        days_req = max(400, (today - ctx_dt).days + 90)
    except Exception:
        days_req = 500

    hist, hist_err = fetch_price_history(spi["symbol"], min_days=days_req)
    if not hist:
        return {
            "status":              "FAILED",
            "error":               f"Price data unavailable: {hist_err}",
            "stock_prediction_input_hash": input_hash,
            "spi":                 spi,
        }

    ctx_bars = filter_history_up_to(hist, origin)
    if not ctx_bars:
        return {
            "status": "FAILED",
            "error":  f"No price bars on or before {origin}",
            "stock_prediction_input_hash": input_hash,
            "spi":    spi,
        }

    ai_result = run_stock_prediction(spi, ctx_bars)
    if ai_result.get("status") != "SUCCESS":
        return {
            "status":      "FAILED",
            "error":       f"AI prediction failed: {ai_result.get('error')}",
            "ai_prediction": ai_result,
            "stock_prediction_input_hash": input_hash,
            "spi":         spi,
        }

    val_result = run_stock_validation(spi, ai_result, hist)

    if val_result.get("status") == "SUCCESS":
        save_stock_prediction_record(spi, ai_result, val_result)

    return {
        "status":                      val_result.get("status", "FAILED"),
        "stock_prediction_input_hash": input_hash,
        "spi":                         spi,
        "ai_prediction":               ai_result,
        "actual_validation":           val_result,
        "comparison":                  val_result.get("comparison", {}),
        "ctx_summary":                 get_context_summary(hist, ctx_start, origin),
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


def run_options_strategy_validation(input_dict: dict) -> dict:
    """
    Run AI prediction + options backtest for the prediction window.

    Same base inputs as run_stock_price_validation, PLUS:
      direction        : "Buy" | "Sell"
      opt_type         : "Call" | "Put"
      quantity         : int (contracts)
      delta            : int (0-99, e.g. 30)
      strike_price     : float | None, required when strike_selection="strike"
      dte              : int (days to expiration, e.g. 45)
      strike_selection : "delta" | "strike"
      entry_schedule   : "Once at prediction origin date" | "Daily" | "Weekly" | "Monthly"
      exit_rule        : "At expiration" | "Target profit 50%" | ...

    CRITICAL: options backtest uses prediction_origin_date -> target_date
    NEVER historical_context_start_date -> target_date.
    """
    spi = _build_spi(input_dict)

    valid, err = validate_stock_prediction_input(spi)
    if not valid:
        return {"status": "FAILED", "error": f"Input validation: {err}", "spi": spi}

    input_hash = build_stock_prediction_hash(spi)
    origin     = spi["prediction_origin_date"]
    target     = spi["target_date"]
    ctx_start  = spi["historical_context_start_date"]

    today = date.today()
    try:
        ctx_dt   = datetime.strptime(ctx_start, "%Y-%m-%d").date()
        days_req = max(400, (today - ctx_dt).days + 90)
    except Exception:
        days_req = 500

    hist, hist_err = fetch_price_history(spi["symbol"], min_days=days_req)
    if not hist:
        return {
            "status": "FAILED",
            "error":  f"Price data unavailable: {hist_err}",
            "stock_prediction_input_hash": input_hash,
        }

    ctx_bars  = filter_history_up_to(hist, origin)
    ai_result = run_stock_prediction(spi, ctx_bars)

    # Options backtest: origin -> target (prediction window ONLY)
    opts_result = _run_options_backtest(
        symbol    = spi["symbol"],
        start_date= origin,    # PREDICTION ORIGIN DATE -- not ctx_start
        end_date  = target,    # TARGET DATE
        direction = input_dict.get("direction", "Sell"),
        opt_type  = input_dict.get("opt_type", "Put"),
        quantity  = int(input_dict.get("quantity", 1)),
        delta     = int(input_dict.get("delta", 30)),
        strike_price = input_dict.get("strike_price"),
        dte       = int(input_dict.get("dte", 45)),
        strike_selection = input_dict.get("strike_selection", "delta"),
    )

    val_result = run_stock_validation(spi, ai_result, hist)
    if val_result.get("status") == "SUCCESS":
        save_stock_prediction_record(spi, ai_result, val_result)

    return {
        "status":                      "SUCCESS" if ai_result.get("status") == "SUCCESS" else "FAILED",
        "stock_prediction_input_hash": input_hash,
        "spi":                         spi,
        "ai_prediction":               ai_result,
        "actual_validation":           val_result,
        "comparison":                  val_result.get("comparison", {}),
        "options_backtest":            opts_result,
        "options_backtest_window":     f"{origin} to {target} (prediction window only)",
        "ctx_summary":                 get_context_summary(hist, ctx_start, origin),
    }


def _run_options_backtest(
    symbol: str, start_date: str, end_date: str,
    direction: str = "Sell", opt_type: str = "Put",
    quantity: int = 1, delta: int = 30, dte: int = 45,
    strike_selection: str = "delta",
    strike_price: float | None = None,
) -> dict:
    """
    Run tastytrade options backtest for the given window.
    Returns result dict; does NOT raise on failure.
    """
    try:
        from src.services.tastytrade_backtester_service import (
            create_backtest as _tt_create,
            poll_backtest as _tt_poll,
            build_custom_legs_payload as _tt_legs,
        )
    except ImportError:
        return {"status": "UNAVAILABLE", "error": "tastytrade backtester not importable"}

    direction_map = {"Buy": "long", "Sell": "short"}
    type_map      = {"Call": "call", "Put": "put"}

    strike_mode = str(strike_selection).lower() == "strike"
    leg = {
        "type":              "equity-option",
        "direction":         direction_map.get(direction, "short"),
        "quantity":          quantity,
        "side":              type_map.get(opt_type, "put"),
        "daysUntilExpiration": dte,
        "strikeSelection":   "strike" if strike_mode else "delta",
    }
    if strike_mode and strike_price not in (None, ""):
        leg["strikePrice"] = float(strike_price)
    else:
        leg["delta"] = int(delta or 30)
    try:
        payload    = _tt_legs(symbol=symbol, start_date=start_date, end_date=end_date, legs=[leg])
        bt_id, err = _tt_create(payload)
        if not bt_id:
            return {"status": "CREATE_FAILED", "error": err}
        bt_data, poll_err = _tt_poll(bt_id)
        if not bt_data:
            return {"status": "POLL_FAILED", "error": poll_err}
        stats = bt_data.get("statistics") or bt_data.get("data", {}).get("statistics") or {}
        return {
            "status":         "SUCCESS",
            "backtest_id":    bt_id,
            "backtest_range": f"{start_date} to {end_date}",
            "win_rate":       stats.get("winRate") or stats.get("win_rate"),
            "profit_loss":    stats.get("profitLoss") or stats.get("profit_loss"),
            "avg_pnl":        stats.get("avgProfitLoss") or stats.get("avg_pnl"),
            "total_trades":   stats.get("totalTrades") or stats.get("total_trades"),
            "raw_stats":      stats,
        }
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════


def run_single_validation(input_dict: dict) -> dict:
    """
    Unified dispatcher: routes to stock or options validation based on
    presence of 'direction' / 'opt_type' keys in input_dict.
    """
    if input_dict.get("direction") or input_dict.get("opt_type"):
        return run_options_strategy_validation(input_dict)
    return run_stock_price_validation(input_dict)


def run_rolling_monthly_validation(config: dict) -> dict:
    """
    Run walk-forward validation for multiple past months.

    config keys:
      symbol, historical_context_start_date, n_months (default 6),
      initial_capital, benchmark, price_basis

    Returns dict with months list and summary stats.
    """
    symbol     = str(config.get("symbol", "TSLA")).upper()
    ctx_start  = str(config.get("historical_context_start_date", "2024-01-01"))
    n_months   = int(config.get("n_months", 6))
    capital    = float(config.get("initial_capital", 50_000))
    benchmark  = str(config.get("benchmark", "SPY")).upper()
    price_basis= str(config.get("price_basis", "close"))

    hist, err = fetch_price_history(symbol, min_days=750)
    if not hist:
        return {"status": "FAILED", "error": f"Price data unavailable: {err}"}

    today   = date.today()
    current = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    months  = []

    for _ in range(n_months):
        mo_end   = (current + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        origin_s = current.strftime("%Y-%m-%d")
        target_s = mo_end.strftime("%Y-%m-%d")

        try:
            ctx_dt = datetime.strptime(ctx_start, "%Y-%m-%d").date()
            if current <= ctx_dt:
                break
        except Exception:
            break

        spi = {
            "symbol":                        symbol,
            "historical_context_start_date": ctx_start,
            "prediction_origin_date":        origin_s,
            "decision_horizon_days":         (mo_end - current).days,
            "target_date":                   target_s,
            "initial_capital":               capital,
            "benchmark":                     benchmark,
            "validation_mode":               "calendar_month",
            "price_basis":                   price_basis,
        }
        ctx_bars   = filter_history_up_to(hist, origin_s)
        ai_result  = run_stock_prediction(spi, ctx_bars)
        val_result = run_stock_validation(spi, ai_result, hist)

        if ai_result.get("status") == "SUCCESS" and val_result.get("status") == "SUCCESS":
            save_stock_prediction_record(spi, ai_result, val_result)

        months.append({
            "month":          origin_s[:7],
            "origin_date":    origin_s,
            "target_date":    target_s,
            "ai_decision":    ai_result.get("decision"),
            "actual_decision": val_result.get("actual_decision"),
            "ai_return_pct":  ai_result.get("predicted_return_pct"),
            "actual_return_pct": val_result.get("actual_return_pct"),
            "decision_match": val_result.get("comparison", {}).get("decision_match"),
            "capital_error":  val_result.get("comparison", {}).get("capital_error"),
            "ai_status":      ai_result.get("status"),
            "val_status":     val_result.get("status"),
        })
        current = (current - timedelta(days=1)).replace(day=1)

    if not months:
        return {"status": "FAILED", "error": "No months could be validated"}

    n_ok      = sum(1 for m in months if m["decision_match"] is not None)
    n_match   = sum(1 for m in months if m.get("decision_match") is True)
    match_rate = (n_match / n_ok * 100) if n_ok > 0 else None

    return {
        "status":         "SUCCESS",
        "symbol":         symbol,
        "months_run":     len(months),
        "decision_match_rate_pct": round(match_rate, 1) if match_rate is not None else None,
        "months":         months,
    }


def get_validation_summary(result_dict: dict) -> str:
    """
    Return a human-readable one-paragraph summary of a validation result.
    Safe to use in Discord messages / CLI output.
    """
    status = result_dict.get("status", "UNKNOWN")
    if status == "FAILED":
        return f"Validation FAILED: {result_dict.get('error', 'Unknown error')}"

    spi  = result_dict.get("spi", {})
    ai   = result_dict.get("ai_prediction", {})
    val  = result_dict.get("actual_validation", {})
    cmp  = result_dict.get("comparison", {})
    opts = result_dict.get("options_backtest")

    sym    = spi.get("symbol", "?")
    origin = spi.get("prediction_origin_date", "?")
    target = spi.get("target_date", "?")
    cap    = float(spi.get("initial_capital", 0) or 0)

    ai_dec     = ai.get("decision", "?")
    ai_ret     = ai.get("predicted_return_pct")
    ai_cap     = ai.get("predicted_final_capital")
    act_dec    = val.get("actual_decision", "?")
    act_ret    = val.get("actual_return_pct")
    act_cap    = val.get("actual_final_capital")
    match      = cmp.get("decision_match")
    dir_match  = cmp.get("directional_match")
    ret_err    = cmp.get("return_error_pct")
    agreement  = cmp.get("agreement", "?")

    lines = [
        f"Symbol: {sym} | Period: {origin} to {target}",
        f"AI Decision: {ai_dec} (predicted return: {ai_ret:+.2f}%" if ai_ret is not None else f"AI Decision: {ai_dec}",
        f", predicted capital: ${ai_cap:,.2f})" if ai_cap is not None else ")",
        f"Actual: {act_dec} (actual return: {act_ret:+.2f}%" if act_ret is not None else f"Actual: {act_dec}",
        f", actual capital: ${act_cap:,.2f})" if act_cap is not None else ")",
        f"Agreement: {agreement} | Decision match: {'YES' if match else 'NO'} | Direction match: {'YES' if dir_match else 'NO'}",
    ]
    if ret_err is not None:
        lines.append(f"Return error: {ret_err:+.2f}pp")
    if opts:
        opts_status = opts.get("status", "?")
        pnl = opts.get("profit_loss")
        lines.append(
            f"Options backtest ({origin} to {target}): status={opts_status}"
            + (f", P&L=${float(pnl):+,.2f}" if pnl is not None else "")
        )

    return " | ".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY MAPPER
# ══════════════════════════════════════════════════════════════════════════════


def map_legacy_strategy_input(strategy_input: dict) -> dict:
    """
    Map a legacy strategy/options input to the new StockPredictionInput format.

    Legacy fields that are IGNORED (options-specific, no equivalent in stock prediction):
      direction, side, dte, delta, legs, entry_frequency, type, strategy_name

    Legacy fields that ARE MAPPED:
      symbol           -> symbol
      start_date       -> historical_context_start_date (context window start)
      end_date         -> prediction_origin_date (where AI prediction begins)
      capital / initial_capital -> initial_capital
      benchmark        -> benchmark
    """
    ignored = []
    mapped  = {}

    for key in ("direction", "side", "dte", "delta", "legs", "entry_frequency",
                "type", "strategy_name", "num_legs"):
        if key in strategy_input:
            ignored.append(key)

    mapped["symbol"]                        = str(strategy_input.get("symbol", "")).upper()
    mapped["historical_context_start_date"] = str(strategy_input.get("start_date", "2024-01-01"))
    mapped["prediction_origin_date"]        = str(strategy_input.get("end_date", ""))
    mapped["initial_capital"]               = float(
        strategy_input.get("capital") or strategy_input.get("initial_capital", 50_000)
    )
    mapped["benchmark"]                     = str(strategy_input.get("benchmark", "SPY")).upper()
    mapped["validation_mode"]               = "horizon_days"
    mapped["decision_horizon_days"]         = 30
    mapped["price_basis"]                   = "close"

    # Derive target_date
    try:
        origin_dt = datetime.strptime(mapped["prediction_origin_date"], "%Y-%m-%d")
        mapped["target_date"] = (origin_dt + timedelta(days=30)).strftime("%Y-%m-%d")
    except Exception:
        mapped["target_date"] = ""

    mapped["_mapped_from_legacy"]    = True
    mapped["_ignored_legacy_fields"] = ignored

    return mapped


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _build_spi(input_dict: dict) -> dict:
    """Normalize input dict to StockPredictionInput format."""
    spi = dict(input_dict)

    # Remove options-specific keys
    for key in ("direction", "opt_type", "quantity", "delta", "dte",
                "strike_selection", "strike_price", "entry_schedule", "exit_rule"):
        spi.pop(key, None)

    # Derive target_date if not provided
    if not spi.get("target_date"):
        horizon = int(spi.get("decision_horizon_days", 30))
        try:
            origin_dt = datetime.strptime(str(spi["prediction_origin_date"]), "%Y-%m-%d")
            spi["target_date"] = (origin_dt + timedelta(days=horizon)).strftime("%Y-%m-%d")
        except Exception:
            pass

    # Defaults
    spi.setdefault("validation_mode",   "horizon_days")
    spi.setdefault("price_basis",       "close")
    spi.setdefault("decision_horizon_days", 30)
    spi.setdefault("benchmark",         "SPY")
    spi.setdefault("initial_capital",   50_000.0)

    return spi
