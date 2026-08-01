from __future__ import annotations

from .discord_agent import DecisionResult, _direct_equity_embed, _parsed_equity_summary
from .options_parser import classify_and_parse
from .signal_parser import parse_signal


def test_supplied_conditional_option_contracts() -> None:
    cases = [
        (
            "BTO CRM 320C 12/19 Qty 3 ONLY IF CRM TRADES ABOVE 325, MAX PREMIUM 5.50",
            {"root": "CRM", "quantity": 3.0, "fill_price": 5.50,
             "underlying_trigger_direction": "above", "underlying_trigger_price": 325.0},
        ),
        (
            "STC 40% OF AVGO 420C 11/21 AT MARKET AND KEEP THE REST OPEN",
            {"root": "AVGO", "order_action": "close_long", "close_percent": 40.0,
             "remaining_instruction": "KEEP_OPEN"},
        ),
        (
            "STO XOM 130C 10/17 @2.10 Qty 5 GTC, BTC AT 0.65 OR EXIT IF XOM BREAKS ABOVE 128",
            {"root": "XOM", "order_action": "open_short", "quantity": 5.0,
             "fill_price": 2.10, "target_price": 0.65, "time_in_force": "GTC",
             "exit_underlying_direction": "above", "exit_underlying_price": 128.0},
        ),
        (
            "BUY 2 COST 1000P 12/19 MARKET, MAX LOSS $1,200, TAKE PROFIT AT 18.50",
            {"root": "COST", "quantity": 2.0, "maximum_loss_amount": 1200.0,
             "target_price": 18.50},
        ),
        (
            "Starter BTO 1 LLY 900C 12/19 @7.80; add 2 contracts at 6.90 and stop all at 5.80",
            {"root": "LLY", "quantity": 1.0, "fill_price": 7.80,
             "add_quantity": 2.0, "add_trigger_premium": 6.90,
             "stop_loss": 5.80, "stop_scope": "ALL_CONTRACTS"},
        ),
        (
            "Taking a starter in DIS 110P expiring 11/21 around 2.30. Add below 108, target 3.80, cut it if premium loses 1.70.",
            {"root": "DIS", "fill_price": 2.30, "entry_price_type": "APPROXIMATE",
             "add_trigger_underlying_direction": "below", "add_trigger_underlying_price": 108.0,
             "target_price": 3.80, "stop_loss": 1.70},
        ),
        (
            "STO QQQ 560P 10/17 @2.80 Qty 8. Buy back at 1.00 or stop if premium reaches 4.20.",
            {"root": "QQQ", "order_action": "open_short", "quantity": 8.0,
             "fill_price": 2.80, "target_price": 1.00, "stop_loss": 4.20},
        ),
    ]
    for raw, expected in cases:
        option = classify_and_parse(raw).option
        assert option is not None and option.valid, raw
        for field, value in expected.items():
            assert getattr(option, field) == value, f"{raw}: {field}"


def test_thirty_supplied_equity_order_contracts() -> None:
    cases = [
        ("BUY AAPL Qty 100 MARKET", "BUY", "AAPL", 100, "market", None, None, "DAY"),
        ("BUY NVDA Qty 25 LIMIT 185.50", "BUY", "NVDA", 25, "limit", 185.50, None, "DAY"),
        ("SELL TSLA Qty 40 MARKET", "SELL", "TSLA", 40, "market", None, None, "DAY"),
        ("SELL META Qty 15 LIMIT 735.00", "SELL", "META", 15, "limit", 735.00, None, "DAY"),
        ("SHORT AMD Qty 60 MARKET", "SELL_SHORT", "AMD", 60, "market", None, None, "DAY"),
        ("BUY TO COVER AMD Qty 60 MARKET", "BUY_TO_COVER", "AMD", 60, "market", None, None, "DAY"),
        ("BUY MSFT Qty 30 STOP 540.00", "BUY", "MSFT", 30, "stop", None, 540.00, "DAY"),
        ("SELL AMZN Qty 20 STOP 245.00 LIMIT 244.50", "SELL", "AMZN", 20, "stop_limit", 244.50, 245.00, "DAY"),
        ("BUY GOOGL Qty 50 LIMIT 218.75 GTC", "BUY", "GOOGL", 50, "limit", 218.75, None, "GTC"),
        ("SELL JPM Qty 75 LIMIT 305.00 DAY", "SELL", "JPM", 75, "limit", 305.00, None, "DAY"),
        ("BUY ORCL Qty 50 MARKET", "BUY", "ORCL", 50, "market", None, None, "DAY"),
        ("SELL ORCL Qty 50 MARKET", "SELL", "ORCL", 50, "market", None, None, "DAY"),
        ("BUY CRM Qty 20 LIMIT 315.75", "BUY", "CRM", 20, "limit", 315.75, None, "DAY"),
        ("SELL CRM Qty 20 LIMIT 329.50", "SELL", "CRM", 20, "limit", 329.50, None, "DAY"),
        ("BUY AVGO Qty 10 STOP 385.00", "BUY", "AVGO", 10, "stop", None, 385.00, "DAY"),
        ("SELL AVGO Qty 10 STOP 360.00", "SELL", "AVGO", 10, "stop", None, 360.00, "DAY"),
        ("BUY AMD Qty 40 STOP 180.00 LIMIT 180.50", "BUY", "AMD", 40, "stop_limit", 180.50, 180.00, "DAY"),
        ("SELL AMD Qty 40 STOP 165.00 LIMIT 164.50", "SELL", "AMD", 40, "stop_limit", 164.50, 165.00, "DAY"),
        ("SHORT NFLX Qty 15 MARKET", "SELL_SHORT", "NFLX", 15, "market", None, None, "DAY"),
        ("BUY TO COVER NFLX Qty 15 MARKET", "BUY_TO_COVER", "NFLX", 15, "market", None, None, "DAY"),
        ("BUY IBM Qty 35 LIMIT 295.00 GTC", "BUY", "IBM", 35, "limit", 295.00, None, "GTC"),
        ("SELL IBM Qty 35 LIMIT 310.00 DAY", "SELL", "IBM", 35, "limit", 310.00, None, "DAY"),
        ("BUY TSM Qty 60 MARKET IOC", "BUY", "TSM", 60, "market", None, None, "IOC"),
        ("SELL TSM Qty 60 LIMIT 245.50 FOK", "SELL", "TSM", 60, "limit", 245.50, None, "FOK"),
        ("BUY COST Qty 12 MARKET", "BUY", "COST", 12, "market", None, None, "DAY"),
        ("SELL COST Qty 12 MARKET", "SELL", "COST", 12, "market", None, None, "DAY"),
        ("BUY DIS Qty 80 LIMIT 118.25", "BUY", "DIS", 80, "limit", 118.25, None, "DAY"),
        ("SELL DIS Qty 80 LIMIT 126.00", "SELL", "DIS", 80, "limit", 126.00, None, "DAY"),
        ("SHORT INTC Qty 100 LIMIT 29.75", "SELL_SHORT", "INTC", 100, "limit", 29.75, None, "DAY"),
        ("BUY TO COVER INTC Qty 100 LIMIT 27.40", "BUY_TO_COVER", "INTC", 100, "limit", 27.40, None, "DAY"),
    ]
    for raw, action, symbol, qty, order_type, limit_price, stop_price, tif in cases:
        parsed = parse_signal(raw)
        assert parsed.valid, raw
        assert (parsed.action, parsed.symbol, parsed.quantity) == (action, symbol, float(qty)), raw
        assert parsed.order_type == order_type, raw
        assert parsed.limit_price == limit_price, raw
        assert parsed.stop_price == stop_price, raw
        assert parsed.time_in_force == tif, raw


def test_conversational_condition_is_not_confused_with_broker_order() -> None:
    parsed = parse_signal("BUY GOOGL if price closes above 205, otherwise HOLD")
    assert parsed.valid
    assert parsed.order_type == "market"
    assert parsed.condition_type == "close_above"
    assert parsed.condition_price == 205.0


def test_equity_review_displays_the_complete_parsed_order_once() -> None:
    parsed = parse_signal("SELL AMZN Qty 20 STOP 245.00 LIMIT 244.50 GTC")
    summary = _parsed_equity_summary(parsed)
    assert "Asset Type: STOCK | Status: VALID" in summary
    assert "Action: SELL | Symbol: AMZN" in summary
    assert "Order Type: STOP_LIMIT | Time In Force: GTC" in summary
    assert "Limit Price: $244.50 | Stop Price: $245.00" in summary

    embed = _direct_equity_embed(
        parsed,
        DecisionResult("SELL", "direct", 0.0, 0.0, 0.0, 100.0),
    )
    fields = {field.name: field.value for field in embed.fields}
    assert list(field.name for field in embed.fields).count("Parsed Signal") == 1
    assert fields["Parsed Signal"] == summary
