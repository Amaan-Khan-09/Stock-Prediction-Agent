"""Local preflight checks for atomic Alpaca multi-leg option strategies."""
from __future__ import annotations

from datetime import date, datetime
from math import gcd
from typing import Any

from .options_parser import ParsedOptionLeg, ParsedOptionSignal


_INFERRED_PRICE_EFFECTS = {
    "bull_call_spread": "debit",
    "bear_put_spread": "debit",
    "bear_call_spread": "credit",
    "bull_put_spread": "credit",
    "long_straddle": "debit",
    "long_strangle": "debit",
    "iron_condor": "credit",
    "iron_butterfly": "credit",
    "reverse_iron_condor": "debit",
    "butterfly_spread": "debit",
    "ratio_backspread": "debit",
}


def infer_multi_leg_price_effect(option: ParsedOptionSignal) -> str | None:
    """Return the net price direction without guessing ambiguous strategies."""
    explicit = str(option.price_effect or "").lower()
    if explicit in {"debit", "credit"}:
        return explicit

    inferred = _INFERRED_PRICE_EFFECTS.get(str(option.structure or "").lower())
    if inferred:
        return inferred

    actions = {str(leg.order_action or "").lower() for leg in option.legs}
    if actions == {"open_long"} or actions == {"close_short"}:
        return "debit"
    if actions == {"open_short"} or actions == {"close_long"}:
        return "credit"
    return None


def _mapped_action(option: ParsedOptionSignal) -> str:
    actions = {str(leg.order_action or "").lower() for leg in option.legs}
    open_actions = {"open_long", "open_short"}
    close_actions = {"close_long", "close_short"}
    if actions and actions <= open_actions:
        return "BUY"
    if actions and actions <= close_actions:
        return "SELL"
    if str(option.structure or "").lower() == "roll" and actions & open_actions and actions & close_actions:
        return "SELL" if infer_multi_leg_price_effect(option) == "credit" else "BUY"
    return "REVIEW"


def _leg_expiry(option: ParsedOptionSignal, leg: ParsedOptionLeg) -> str:
    return str(leg.expiry_date or option.expiry_date or "")[:10]


def _coverage_issues(option: ParsedOptionSignal) -> list[str]:
    """Find opening short exposure not covered by a long leg in the order."""
    longs = [leg for leg in option.legs if leg.order_action == "open_long"]
    shorts = [leg for leg in option.legs if leg.order_action == "open_short"]
    issues: list[str] = []
    for side in ("CALL", "PUT"):
        side_shorts = [leg for leg in shorts if leg.side == side]
        if not side_shorts:
            continue
        short_qty = sum(max(1, int(leg.ratio_qty)) for leg in side_shorts)
        covered_qty = 0
        for long_leg in (leg for leg in longs if leg.side == side):
            long_expiry = _leg_expiry(option, long_leg)
            eligible = any(
                not long_expiry
                or not _leg_expiry(option, short_leg)
                or long_expiry >= _leg_expiry(option, short_leg)
                for short_leg in side_shorts
            )
            if eligible:
                covered_qty += max(1, int(long_leg.ratio_qty))
        if covered_qty < short_qty:
            issues.append(
                f"{side.lower()} opening shorts are not fully covered inside the atomic strategy "
                f"({short_qty} short vs {covered_qty} covering long ratio)"
            )
    return issues


def validate_multi_leg_strategy(option: ParsedOptionSignal) -> dict[str, Any]:
    """Validate structure and broker invariants before historical or broker calls.

    This is a structural/risk preflight, not a claim of historical profitability.
    """
    issues: list[str] = []
    warnings: list[str] = []
    legs = list(option.legs or [])

    if not option.is_multi_leg:
        issues.append("signal is not a multi-leg option strategy")
    if not 2 <= len(legs) <= 4:
        issues.append("Alpaca MLeg orders require 2 to 4 option legs")

    roots = {str(leg.root or "").upper() for leg in legs}
    if not roots or "" in roots or len(roots) != 1:
        issues.append("all option legs must use one underlying")

    contract_keys: set[tuple[str, str, float, str]] = set()
    ratios: list[int] = []
    for index, leg in enumerate(legs, start=1):
        if float(leg.strike or 0) <= 0:
            issues.append(f"leg {index} has no valid strike")
        if str(leg.side or "").upper() not in {"CALL", "PUT"}:
            issues.append(f"leg {index} has no CALL/PUT side")
        if str(leg.order_action or "").lower() not in {
            "open_long", "open_short", "close_long", "close_short"
        }:
            issues.append(f"leg {index} has an unsupported position action")
        ratio = max(0, int(leg.ratio_qty or 0))
        ratios.append(ratio)
        if ratio <= 0:
            issues.append(f"leg {index} has no positive ratio")

        expiry = _leg_expiry(option, leg)
        if not expiry:
            issues.append(f"leg {index} has no expiry")
        else:
            try:
                if datetime.strptime(expiry, "%Y-%m-%d").date() < date.today():
                    issues.append(f"leg {index} expiry is already past")
            except ValueError:
                issues.append(f"leg {index} expiry is invalid")

        key = (str(leg.root or "").upper(), expiry, float(leg.strike or 0), str(leg.side or "").upper())
        if key in contract_keys:
            issues.append(f"leg {index} duplicates another exact option contract")
        contract_keys.add(key)

    positive_ratios = [ratio for ratio in ratios if ratio > 0]
    if positive_ratios:
        common = positive_ratios[0]
        for ratio in positive_ratios[1:]:
            common = gcd(common, ratio)
        if common != 1:
            issues.append("leg ratios must be reduced to their simplest form")

    coverage_issues = _coverage_issues(option)
    issues.extend(coverage_issues)
    mapped_action = _mapped_action(option)
    if mapped_action == "REVIEW":
        issues.append("leg actions do not form one atomic entry, exit, or roll")

    price_effect = infer_multi_leg_price_effect(option)
    if option.fill_price is not None and option.fill_price <= 0:
        issues.append("net strategy limit must be greater than zero")
    if option.fill_price is not None and price_effect is None:
        issues.append("net strategy price is ambiguous; include DEBIT or CREDIT")
    if option.fill_price is None:
        warnings.append("no net limit supplied; strategy will use a market order when eligible")

    return {
        "passed": not issues,
        "decision": mapped_action if not issues else "REVIEW",
        "price_effect": price_effect,
        "risk_profile": "defined" if not coverage_issues else "uncovered",
        "issues": issues,
        "warnings": warnings,
        "leg_count": len(legs),
    }
