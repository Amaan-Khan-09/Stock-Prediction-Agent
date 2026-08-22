"""Regression tests for signal_parser.py's order-detail extraction.

Most real Discord equity messages are handled by stock_order_intent.py's
richer parser (parse_signal tries that first) -- see
test_stock_order_intent.py's own trailing-stop regression cases for that
layer. _extract_order_details is the fallback used for anything the rich
parser doesn't recognize, so it needs the same protection independently.
"""
from __future__ import annotations

from .signal_parser import _extract_order_details


def test_trailing_stop_percent_is_not_misread_as_a_stop_order() -> None:
    # Regression: "trailing stop 5%" previously matched the bare STOP regex
    # as "STOP 5", producing a stop order at $5 instead of falling through
    # to a plain market order.
    assert _extract_order_details("trailing stop 5%") == ("market", None, None, "DAY")
    assert _extract_order_details("trail stop 3%") == ("market", None, None, "DAY")


def test_plain_stop_and_stop_limit_orders_still_parse() -> None:
    assert _extract_order_details("STOP 540.00") == ("stop", None, 540.0, "DAY")
    assert _extract_order_details("STOP 245.00 LIMIT 244.50") == ("stop_limit", 244.5, 245.0, "DAY")


def test_limit_and_market_orders_still_parse() -> None:
    assert _extract_order_details("LIMIT 100.50") == ("limit", 100.5, None, "DAY")
    assert _extract_order_details("MARKET") == ("market", None, None, "DAY")
