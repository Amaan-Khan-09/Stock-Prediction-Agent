"""Discord output formatter — Part K of the implementation spec.

Formats a signal_review_service result dict into a clean Discord message block.
No contradiction allowed: Final Agent Output always matches the bottom line.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import VS


def format_review_result(result: dict) -> str:
    """Return a formatted multi-line string suitable for posting to Discord.

    The format strictly follows Part K of the implementation spec.
    Final Agent Output and bottom line ALWAYS agree.
    """
    if not result.get("parsed_successfully"):
        return (
            "**Signal Review Error**\n"
            f"Error: {result.get('error', 'Unknown error')}\n"
            f"Reason: {result.get('reason', 'Unknown')}"
        )

    sig   = result.get("parsed_signal", {})
    req   = result.get("requested_contract", {})
    ai    = result.get("ai_stock_prediction", {})
    opts  = result.get("options_validation", {})
    final = result.get("final_agent_output", VS.REVIEW_REQUIRED)
    order = result.get("order_status", VS.NOT_SUBMITTED)
    reason= result.get("reason", VS.NOT_PROVIDED)

    underlying  = req.get("underlying") or sig.get("underlying") or "UNKNOWN"
    opt_type    = req.get("option_type") or sig.get("option_type") or "?"
    strike      = req.get("strike")
    expiry      = req.get("expiry") or sig.get("expiry_date") or VS.NOT_PROVIDED
    occ         = req.get("occ_symbol") or opts.get("occ_symbol") or VS.NOT_PROVIDED
    method      = req.get("selection_method") or sig.get("contract_selection_method") or "?"
    contracts   = req.get("contracts") or sig.get("contracts") or "?"
    premium     = req.get("premium") or sig.get("premium")
    action      = sig.get("action", "?")

    # Title line
    strike_str = f"{strike}" if strike is not None else "??"
    title = f"**{underlying} {strike_str}{_cp_abbrev(opt_type)} {expiry} Options Signal Review**"

    # AI section
    ai_dec    = ai.get("decision") or VS.NOT_PROVIDED
    ai_ret    = _fmt_pct(ai.get("predicted_return_pct"))
    ai_conf   = ai.get("confidence") or VS.NOT_PROVIDED
    ai_risk   = ai.get("risk") or VS.NOT_PROVIDED

    # Options validation section
    bt_status = opts.get("status") or VS.NOT_PROVIDED
    bt_dec    = opts.get("decision") or VS.NOT_PROVIDED
    bt_pnl    = _fmt_dollar(opts.get("pnl"))
    bt_wr     = _fmt_pct(opts.get("win_rate"), scale=100)
    val_type  = opts.get("validation_type") or VS.NOT_PROVIDED
    exact_st  = opts.get("exact_strike_status") or VS.NOT_REQUESTED
    proxy_st  = opts.get("delta_proxy_status") or VS.NOT_REQUESTED
    fallback  = opts.get("fallback_used", False)
    eligible  = opts.get("eligible_for_exact_accuracy", False)
    dist_pct  = opts.get("strike_distance_pct")

    # Final decision bottom line
    bottom_line = _bottom_line_text(final, underlying, opt_type, strike_str)

    lines = [
        title,
        "",
        "**Parsed Signal:**",
        f"  Underlying:    {underlying}",
        f"  Option:        {strike_str} {_to_upper(opt_type)}",
        f"  Action:        {action}",
        f"  Qty:           {contracts} contract(s)",
        f"  Entry Premium: {_fmt_dollar(premium)}",
        f"  Expiry:        {expiry}",
        "",
        "**AI Stock Prediction:**",
        f"  Decision:       {ai_dec}",
        f"  Predicted Return: {ai_ret}",
        f"  Confidence / Risk: {ai_conf} / {ai_risk}",
        "",
        "**Options Strategy Validation:**",
        f"  Status:         {bt_status}",
        f"  Decision:       {bt_dec}",
        f"  P&L:            {bt_pnl}",
        f"  Win Rate:       {bt_wr}",
        f"  Validation Type:{val_type}",
        "",
        "**Exact Contract Check:**",
        f"  OCC Symbol:     {occ}",
        f"  Requested Strike: {strike_str}",
        f"  Requested Expiry: {expiry}",
        f"  Strike Distance:  {f'{dist_pct:+.2f}%' if dist_pct is not None else VS.NOT_PROVIDED}",
        "",
        "**Validation Path:**",
        f"  Exact Strike:   {exact_st}",
        f"  Delta Proxy:    {proxy_st}",
        f"  Fallback Used:  {str(fallback).upper()}",
        f"  Eligible For Exact Accuracy: {str(eligible).upper()}",
        f"  Contract Selection Method:   {method}",
        "",
        f"**Final Agent Output:**",
        f"  {final}",
        "",
        f"**Order Status:**",
        f"  {order}",
        "",
        f"**Reason:**",
        f"  {reason}",
        "",
        f"**Bottom Line:** _{bottom_line}_",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cp_abbrev(option_type: Optional[str]) -> str:
    if option_type is None:
        return ""
    return "C" if "CALL" in option_type.upper() else "P"


def _to_upper(s: Optional[str]) -> str:
    return (s or "").upper() or "?"


def _fmt_pct(val, scale: float = 1.0) -> str:
    if val is None:
        return VS.NOT_PROVIDED
    try:
        return f"{float(val) * scale:+.2f}%"
    except Exception:
        return str(val)


def _fmt_dollar(val) -> str:
    if val is None:
        return VS.NOT_PROVIDED
    try:
        return f"${float(val):,.2f}"
    except Exception:
        return str(val)


def _bottom_line_text(final: str, underlying: str, opt_type: Optional[str], strike: str) -> str:
    contract_desc = f"{underlying} {strike}{_cp_abbrev(opt_type)}"
    if final in ("BUY", "ENTER"):
        return f"Agent will enter {contract_desc} (paper only)."
    if final in ("NO_TRADE", "SELL"):
        return f"Agent will NOT enter {contract_desc}."
    return f"Agent requires human review for {contract_desc}. No automatic order placed."
