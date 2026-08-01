"""Offline known-answer checks for Alpaca multi-leg order payloads."""
from __future__ import annotations

from .alpaca_paper import AlpacaPaperClient


def run_all() -> None:
    client = AlpacaPaperClient()
    captured = {}
    client.ready = lambda: True

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"id": "offline-test-order"}, ""

    client._post = fake_post
    legs = [
        {
            "symbol": "AAPL260918C00240000",
            "ratio_qty": "1",
            "side": "buy",
            "position_intent": "buy_to_open",
        },
        {
            "symbol": "AAPL260918C00250000",
            "ratio_qty": "1",
            "side": "sell",
            "position_intent": "sell_to_open",
        },
    ]
    order, error = client.submit_multi_leg_option_order(
        legs, 5, "limit", 4.60, "dsa-mleg-known-answer"
    )
    assert order and not error
    payload = captured["payload"]
    assert captured["path"] == "/v2/orders"
    assert payload["order_class"] == "mleg"
    assert payload["qty"] == "5"
    assert payload["type"] == "limit"
    assert payload["limit_price"] == "4.6"
    assert payload["client_order_id"] == "dsa-mleg-known-answer"
    assert payload["legs"] == legs

    ratio_legs = [
        {
            "symbol": "TSLA261016C00300000",
            "ratio_qty": "1",
            "side": "buy",
            "position_intent": "buy_to_open",
        },
        {
            "symbol": "TSLA261016C00320000",
            "ratio_qty": "2",
            "side": "sell",
            "position_intent": "sell_to_open",
        },
    ]
    order, error = client.submit_multi_leg_option_order(ratio_legs, 2, "limit", -2.30)
    assert order and not error
    payload = captured["payload"]
    assert payload["qty"] == "2"
    assert payload["limit_price"] == "-2.3"
    assert payload["legs"][1]["ratio_qty"] == "2"

    rejected, error = client.submit_multi_leg_option_order(
        [{**legs[0], "ratio_qty": "2"}, {**legs[1], "ratio_qty": "2"}],
        1,
        "limit",
        4.60,
    )
    assert not rejected and "simplest form" in error
    print("ALPACA MULTI-LEG PAYLOAD TESTS PASSED: debit, credit/ratio, and GCD safeguards")


if __name__ == "__main__":
    run_all()
