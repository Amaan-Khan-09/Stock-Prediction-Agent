"""Regression tests for the rich stock-order parser and Alpaca planner."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .stock_order_intent import (
    build_alpaca_order_plan,
    execute_alpaca_order_plan,
    flatten_order_fields,
    gate_order_plan,
    parse_stock_order,
    stock_review_chunks,
)
from .options_parser import classify_and_parse


REPRESENTATIVE_CASES = [
    (
        "BUY AAPL 10 SHARES MARKET",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "AAPL", "quantity": 10, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "SELL MSFT Qty 25 MARKET",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "MSFT", "quantity": 25, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "BUY TO COVER INTC 100 SHARES LIMIT 28.25",
        {"asset_type": "STOCK", "action": "BUY_TO_COVER", "symbol": "INTC", "quantity": 100, "order_type": "LIMIT", "limit_price": 28.25, "status": "VALID"},
    ),
    (
        "BUY MSFT Qty 15 STOP 545 LIMIT 546",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "MSFT", "quantity": 15, "order_type": "STOP_LIMIT", "stop_price": 545, "limit_price": 546, "status": "VALID"},
    ),
    (
        "SELL HALF OF MY IBM SHARES AT 312 LIMIT",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "IBM", "close_percentage": 50, "order_type": "LIMIT", "limit_price": 312, "status": "VALID_IF_POSITION_EXISTS"},
    ),
    (
        "BUY $2,500 WORTH OF AAPL AT MARKET",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "AAPL", "notional_amount": 2500, "currency": "USD", "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "BUY 100 KO AT 69.50 OR BETTER",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "KO", "quantity": 100, "order_type": "LIMIT", "limit_price": 69.5, "price_instruction": "OR_BETTER", "status": "VALID"},
    ),
    (
        "SELL 60 XOM AT 130 OR HIGHER",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "XOM", "quantity": 60, "order_type": "LIMIT", "limit_price": 130, "price_instruction": "OR_HIGHER", "status": "VALID"},
    ),
    (
        "BUY 25 QQQ MARKET AT THE OPEN",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "QQQ", "quantity": 25, "order_type": "MARKET_ON_OPEN", "execution_session": "MARKET_OPEN", "status": "VALID"},
    ),
    (
        "SELL 25 QQQ MARKET ON CLOSE",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "QQQ", "quantity": 25, "order_type": "MARKET_ON_CLOSE", "execution_session": "MARKET_CLOSE", "status": "VALID"},
    ),
    (
        # The natural phrasing for opening a short -- "SELL SHORT X" -- must
        # not be read as a plain SELL. It previously was, because the parser
        # only recognized "SHORT" as literally the first word of the message.
        "SELL SHORT TSLA QTY 10",
        {"asset_type": "STOCK", "action": "SELL_SHORT", "symbol": "TSLA", "quantity": 10, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "SHORT SELL TSLA QTY 10",
        {"asset_type": "STOCK", "action": "SELL_SHORT", "symbol": "TSLA", "quantity": 10, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "GO SHORT AMD QTY 20",
        {"asset_type": "STOCK", "action": "SELL_SHORT", "symbol": "AMD", "quantity": 20, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        # Regression: the action phrase itself ("ENTER"/"GO") used to win the
        # ticker slot for any symbol outside the small known-symbol allowlist,
        # silently building an order for the wrong company. RIVN isn't in the
        # allowlist, so this previously resolved to symbol "ENTER"/"GO".
        "ENTER LONG RIVN 100 SHARES MARKET",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "RIVN", "quantity": 100, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "GO SHORT RIVN 100 SHARES MARKET",
        {"asset_type": "STOCK", "action": "SELL_SHORT", "symbol": "RIVN", "quantity": 100, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        # "GO" is itself a real ticker (Grocery Outlet) -- confirm it still
        # resolves correctly when it isn't part of a "GO SHORT" action phrase.
        "BUY GO 10 SHARES MARKET",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "GO", "quantity": 10, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        # Any of the ~14k actively tradable symbols the live Alpaca directory
        # knows about must resolve correctly, not just the ~30 hardcoded names.
        "SELL LULU 5 SHARES MARKET",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "LULU", "quantity": 5, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        # A company name must resolve to its actual ticker, not be returned
        # as the literal (non-tradable) name text.
        "Buy 10 shares of Apple",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "AAPL", "quantity": 10, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "BUY COIN QTY 3",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "COIN", "quantity": 3, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        # Regression: several blocklisted syntax words (ALL, NOW, ON, OPEN, ...)
        # are themselves real tickers in the Alpaca directory. "ALL" here means
        # ordinary English ("sell everything"), not the Allstate ticker -- the
        # blocklist must still win over an incidental symbol-directory hit so
        # the real target (RIVN) is found instead.
        "SELL ALL MY SHARES OF RIVN",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "RIVN", "order_type": "MARKET", "status": "VALID"},
    ),
    (
        # Regression: "TRAILING STOP 5%" was previously misread as a bare
        # STOP order at a $5 stop price (from "STOP 5"), which would have
        # submitted a nonsensical stop order instead of a plain market buy.
        "BUY AAPL 10 SHARES TRAILING STOP 5%",
        {"asset_type": "STOCK", "action": "BUY", "symbol": "AAPL", "quantity": 10, "order_type": "MARKET", "status": "VALID"},
    ),
    (
        "SELL MSFT QTY 5 TRAIL STOP 3%",
        {"asset_type": "STOCK", "action": "SELL", "symbol": "MSFT", "quantity": 5, "order_type": "MARKET", "status": "VALID"},
    ),
]


def test_representative_cases() -> None:
    for signal, expected in REPRESENTATIVE_CASES:
        assert parse_stock_order(signal) == expected, signal


def _corpus_path() -> Path | None:
    configured = os.getenv("TRADING_AGENT_TEST_CASES_JSON", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parent / "test_data" / "trading_agent_300_test_cases.json",
        Path.home() / "Downloads" / "trading_agent_300_test_cases.json",
    ]
    return next((path for path in candidates if path and path.is_file()), None)


def test_uploaded_120_stock_case_corpus() -> None:
    path = _corpus_path()
    if path is None:
        return
    cases = json.loads(path.read_text(encoding="utf-8"))
    stock_cases = [case for case in cases if 1 <= int(case["id"]) <= 120]
    assert len(stock_cases) == 120
    for case in stock_cases:
        assert parse_stock_order(case["signal"]) == case["expected_output"], f"case {case['id']}"
    invalid_stock_ids = {271, 272, 273, 278, 279, 280, 286, 287, 288, 290, 291, 293, 294, 295, 296, 298, 299}
    for case in cases:
        if int(case["id"]) in invalid_stock_ids:
            assert parse_stock_order(case["signal"]) == case["expected_output"], f"case {case['id']}"


def test_alpaca_payloads_and_agent_gate() -> None:
    stop_limit = parse_stock_order("BUY MSFT Qty 15 STOP 545 LIMIT 546")
    plan = build_alpaca_order_plan(stop_limit)
    assert plan.executable
    assert plan.operations[0]["payload"] == {
        "symbol": "MSFT", "side": "buy", "type": "stop_limit", "time_in_force": "day",
        "qty": "15", "limit_price": "546", "stop_price": "545",
    }

    bracket = parse_stock_order("BUY ORCL 25 SHARES AT MARKET TAKE PROFIT 310 STOP LOSS 285")
    payload = build_alpaca_order_plan(bracket).operations[0]["payload"]
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"] == {"limit_price": "310"}
    assert payload["stop_loss"] == {"stop_price": "285"}

    oto = parse_stock_order("BUY AMD 40 SHARES LIMIT 172.50 WITH STOP LOSS 165")
    oto_payload = build_alpaca_order_plan(oto).operations[0]["payload"]
    assert oto_payload["order_class"] == "oto"
    assert oto_payload["stop_loss"] == {"stop_price": "165"}

    percentage = parse_stock_order("SELL HALF OF MY IBM SHARES AT 312 LIMIT")
    percentage_plan = build_alpaca_order_plan(percentage, position_quantity=18)
    assert percentage_plan.operations[0]["payload"]["qty"] == "9"

    held = gate_order_plan(bracket, agent_mode="ON", agent_decision="HOLD")
    assert held.blocked_reasons and "HOLD" in held.blocked_reasons[-1]
    direct = gate_order_plan(bracket, agent_mode="OFF", agent_decision="HOLD")
    assert direct.executable


def test_invalid_stock_orders_route_to_invalid() -> None:
    expected = {
        "status": "INVALID_OR_NON_EXECUTABLE",
        "issues": ["missing quantity/order details"],
        "should_execute": False,
    }
    assert parse_stock_order("BUY AAPL") == expected
    routed = classify_and_parse("BUY AAPL")
    assert routed.kind == "INVALID"
    assert routed.equity and routed.equity.order_intent == expected


def test_review_contains_every_leaf_field() -> None:
    intent = parse_stock_order(
        "BUY AMZN 30 SHARES LIMIT 140.5 GTC; IF FILLED PLACE OCO TAKE PROFIT 151.74 AND STOP LOSS 134.88"
    )
    flattened = flatten_order_fields(intent)
    review = "\n".join(stock_review_chunks(intent, max_chars=120))
    assert len(flattened) == len({name for name, _ in flattened})
    for name, value in flattened:
        assert f"`{name}`: {value}" in review


class _FakeAlpaca:
    base_url = "https://paper-api.alpaca.markets"

    def __init__(self) -> None:
        self.posts: list[dict] = []

    def ready(self) -> bool:
        return True

    def _post(self, path: str, payload: dict):
        self.posts.append({"path": path, "payload": payload})
        return {"id": f"order-{len(self.posts)}", **payload}, ""


def test_executor_submits_only_unblocked_immediate_operations() -> None:
    intent = parse_stock_order("BUY NVDA 30 SHARES LIMIT 182.50 GTC")
    plan = gate_order_plan(intent, agent_mode="OFF")
    client = _FakeAlpaca()
    result = execute_alpaca_order_plan(client, plan, client_order_id_factory=lambda index: f"discord-{index}")
    assert not result["errors"]
    assert len(result["submitted"]) == 1
    assert client.posts[0]["payload"]["time_in_force"] == "gtc"
    assert client.posts[0]["payload"]["client_order_id"] == "discord-0"

    complex_intent = parse_stock_order(
        "STARTER: BUY 22 MSFT AT MARKET, ADD 22 MORE ABOVE 116.75, "
        "MOVE STOP TO BREAKEVEN AFTER PRICE REACHES 120.25"
    )
    complex_plan = gate_order_plan(complex_intent, agent_mode="OFF")
    complex_client = _FakeAlpaca()
    complex_result = execute_alpaca_order_plan(complex_client, complex_plan)
    assert complex_result["errors"]
    assert not complex_client.posts, "complex order must not be partially submitted"

    timed = parse_stock_order(
        "BUY AAPL 20 SHARES LIMIT 100; TAKE PROFIT 108; STOP LOSS 96; "
        "CANCEL IF NOT FILLED BY 2:30 PM ET"
    )
    timed_client = _FakeAlpaca()
    timed_result = execute_alpaca_order_plan(timed_client, gate_order_plan(timed, agent_mode="OFF"))
    assert timed_result["errors"]
    assert not timed_client.posts, "time-based cancellation must not be ignored"


def run_all() -> None:
    test_representative_cases()
    test_uploaded_120_stock_case_corpus()
    test_alpaca_payloads_and_agent_gate()
    test_invalid_stock_orders_route_to_invalid()
    test_review_contains_every_leaf_field()
    test_executor_submits_only_unblocked_immediate_operations()
    print("RICH STOCK ORDER TESTS PASSED")


if __name__ == "__main__":
    run_all()
