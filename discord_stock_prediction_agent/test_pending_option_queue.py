"""Offline checks that duplicate pending option signals remain independent."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import discord_stock_prediction_agent.state_store as state_store


def run_all() -> None:
    original = state_store.STATE_PATH
    with TemporaryDirectory() as tmp:
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            base = {"occ_symbol": "AAPL260918C00240000", "qty": 1, "order_side": "buy"}
            state_store.add_pending_option_order({**base, "pending_key": "option:first"})
            state_store.add_pending_option_order({**base, "pending_key": "option:second"})
            pending = state_store.list_pending_option_orders()
            assert len(pending) == 2
            assert {item["pending_key"] for item in pending} == {"option:first", "option:second"}
            assert all(item["occ_symbol"] == "AAPL260918C00240000" for item in pending)
            state_store.remove_pending_option_order("option:first")
            remaining = state_store.list_pending_option_orders()
            assert len(remaining) == 1 and remaining[0]["pending_key"] == "option:second"
        finally:
            state_store.STATE_PATH = original
    print("PENDING OPTION QUEUE TEST PASSED: duplicate contract signals remain independent")


if __name__ == "__main__":
    run_all()
