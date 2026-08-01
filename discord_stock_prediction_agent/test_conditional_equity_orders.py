"""Regression tests for watched conditional equity orders.

_condition_is_triggered had no test coverage at all, which let a real bug
ship: "below"/"close_below" conditions required current_price to stay inside
a narrow band above the trigger, so a real gap-down past the trigger (the
exact scenario a "buy if below $X" order exists for) silently never fired --
forever, since these watch orders have no expiry.
"""
from __future__ import annotations

from . import discord_agent


def _order(condition_type: str, condition_price: float, action: str = "BUY") -> dict:
    return {"condition_type": condition_type, "condition_price": condition_price, "action": action}


def test_below_condition_triggers_on_a_gap_far_past_the_trigger() -> None:
    order = _order("below", 150.0, action="BUY")
    assert discord_agent._condition_is_triggered(order, 149.0)
    assert discord_agent._condition_is_triggered(order, 100.0)  # a real gap-down, not just a small dip
    assert not discord_agent._condition_is_triggered(order, 150.01)


def test_above_condition_triggers_on_a_gap_far_past_the_trigger() -> None:
    order = _order("close_above", 200.0, action="SELL")
    assert discord_agent._condition_is_triggered(order, 201.0)
    assert discord_agent._condition_is_triggered(order, 300.0)  # a real gap-up
    assert not discord_agent._condition_is_triggered(order, 199.99)


def test_limit_price_condition_uses_standard_limit_order_semantics() -> None:
    buy_limit = _order("limit_price", 150.0, action="BUY")
    assert discord_agent._condition_is_triggered(buy_limit, 150.0)
    assert discord_agent._condition_is_triggered(buy_limit, 90.0)  # cheaper than the limit is still fillable
    assert not discord_agent._condition_is_triggered(buy_limit, 150.01)

    sell_limit = _order("limit_price", 150.0, action="SELL")
    assert discord_agent._condition_is_triggered(sell_limit, 150.0)
    assert discord_agent._condition_is_triggered(sell_limit, 250.0)  # richer than the limit is still fillable
    assert not discord_agent._condition_is_triggered(sell_limit, 149.99)


def test_invalid_or_missing_prices_never_trigger() -> None:
    assert not discord_agent._condition_is_triggered(_order("below", 150.0), 0.0)
    assert not discord_agent._condition_is_triggered(_order("below", 0.0), 100.0)
    assert not discord_agent._condition_is_triggered(_order("unknown_condition", 150.0), 100.0)


def run_all() -> None:
    test_below_condition_triggers_on_a_gap_far_past_the_trigger()
    test_above_condition_triggers_on_a_gap_far_past_the_trigger()
    test_limit_price_condition_uses_standard_limit_order_semantics()
    test_invalid_or_missing_prices_never_trigger()
    print("CONDITIONAL EQUITY ORDER TESTS PASSED")


if __name__ == "__main__":
    run_all()
