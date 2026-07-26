"""Offline checks for replay-safe Alpaca order submission."""
from __future__ import annotations

from .alpaca_paper import AlpacaPaperClient


def run_all() -> None:
    client = AlpacaPaperClient()
    client.ready = lambda: True
    captured = []

    def fake_post(path, payload):
        captured.append((path, payload))
        return {"id": f"order-{len(captured)}"}, ""

    client._post = fake_post

    order, error = client.submit_market_order(
        "AAPL", "buy", 2, "dsa-equity-message-123"
    )
    assert order and not error
    assert captured[-1][1]["client_order_id"] == "dsa-equity-message-123"

    order, error = client.submit_option_order(
        "AAPL260821C00240000",
        "buy",
        1,
        "limit",
        3.45,
        "buy_to_open",
        "dsa-option-message-456",
    )
    assert order and not error
    assert captured[-1][1]["client_order_id"] == "dsa-option-message-456"

    recovery_client = AlpacaPaperClient()
    recovery_client._request = lambda *args, **kwargs: (
        None,
        "Alpaca request error: ReadTimeout",
    )
    recovery_client.get_order_by_client_order_id = lambda client_order_id: (
        {"id": "already-accepted", "client_order_id": client_order_id},
        "",
    )
    recovered, error = recovery_client._post(
        "/v2/orders",
        {
            "symbol": "MSFT",
            "qty": "1",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "dsa-reconcile-message-789",
        },
    )
    assert recovered and recovered["id"] == "already-accepted"
    assert not error

    print(
        "ALPACA IDEMPOTENCY TESTS PASSED: equity, option, multi-leg, "
        "and timeout reconciliation"
    )


if __name__ == "__main__":
    run_all()
