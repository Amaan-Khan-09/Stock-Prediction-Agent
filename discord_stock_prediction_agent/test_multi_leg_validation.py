"""Offline multi-leg strategy preflight tests."""
from __future__ import annotations

from dataclasses import replace

from .multi_leg_validation import infer_multi_leg_price_effect, validate_multi_leg_strategy
from .options_parser import ParsedOptionLeg, classify_and_parse


APPROVED = (
    ("BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5", "debit"),
    ("BTO TSM 250P / STO TSM 230P 10/17 @5.25 Debit Qty 3", "debit"),
    ("BUY SPY 640C + 640P 09/19 @8.20 Debit Qty 2", "debit"),
    ("BUY QQQ 600C + 570P 10/17 @7.10 Debit Qty 4", "debit"),
    ("STO SPY 620P / BTO SPY 610P / STO SPY 670C / BTO SPY 680C 09/19 @2.15 Credit Qty 10", "credit"),
    ("SELL SPY 640C 640P BUY 650C 630P 09/19 @4.80 Credit Qty 5", "credit"),
    ("BTO MSFT 550C 12/19 / STO MSFT 550C 09/19 @5.80 Debit Qty 4", "debit"),
    ("STO 1 NVDA 210C / BTO 2 NVDA 220C 10/17 @1.50 Debit", "debit"),
)


def test_supported_strategies_pass_preflight() -> None:
    for signal, effect in APPROVED:
        option = classify_and_parse(signal).option
        assert option and option.valid, signal
        result = validate_multi_leg_strategy(option)
        assert result["passed"], (signal, result)
        assert result["risk_profile"] == "defined", (signal, result)
        assert infer_multi_leg_price_effect(option) == effect


def test_uncovered_ratio_spread_is_blocked() -> None:
    option = classify_and_parse("BTO 2 TSLA 300C / STO 4 TSLA 320C 10/17 @2.30 Credit").option
    assert option and option.valid
    result = validate_multi_leg_strategy(option)
    assert not result["passed"]
    assert any("not fully covered" in issue for issue in result["issues"])


def test_duplicate_contract_and_non_reduced_ratios_are_blocked() -> None:
    base = classify_and_parse("BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit").option
    assert base
    duplicate = replace(base, legs=(base.legs[0], replace(base.legs[0], order_action="open_short")))
    result = validate_multi_leg_strategy(duplicate)
    assert not result["passed"]
    assert any("duplicates" in issue for issue in result["issues"])

    non_reduced = replace(
        base,
        legs=tuple(replace(leg, ratio_qty=2) for leg in base.legs),
    )
    result = validate_multi_leg_strategy(non_reduced)
    assert not result["passed"]
    assert any("simplest form" in issue for issue in result["issues"])


def test_ambiguous_net_price_requires_debit_or_credit() -> None:
    base = classify_and_parse("BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit").option
    assert base
    ambiguous = replace(
        base,
        structure="multi_leg",
        price_effect=None,
        legs=(
            ParsedOptionLeg("AAPL", 240, "CALL", "close_long", 1, base.expiry_date),
            ParsedOptionLeg("AAPL", 250, "CALL", "open_long", 1, base.expiry_date),
        ),
    )
    result = validate_multi_leg_strategy(ambiguous)
    assert not result["passed"]
    assert any("DEBIT or CREDIT" in issue for issue in result["issues"])
