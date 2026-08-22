"""Closed-market queue to market-open submission and protection tests."""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from . import discord_agent, pending_market_orders, state_store
from .config import config


class FakeAlpaca:
    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.by_client: dict[str, dict] = {}
        self.submit_calls = 0
        self.fail_error = ""
        self.position_qty = 10.0
        self.market_open = True

    def ready(self):
        return True

    def is_market_open(self):
        return self.market_open, ""

    def get_order_by_client_order_id(self, client_order_id: str):
        order = self.by_client.get(client_order_id)
        return (dict(order), "") if order else (None, "Alpaca HTTP 404")

    def submit_market_order(self, symbol: str, side: str, qty: float, client_order_id: str):
        self.submit_calls += 1
        if self.fail_error:
            return None, self.fail_error
        order_id = f"order-{self.submit_calls}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "status": "filled" if side == "buy" else "accepted",
            "filled_qty": str(qty if side == "buy" else 0),
            "filled_avg_price": "100" if side == "buy" else None,
            "client_order_id": client_order_id,
        }
        self.orders[order_id] = order
        self.by_client[client_order_id] = order
        return dict(order), ""

    def get_order(self, order_id: str):
        order = self.orders.get(order_id)
        return (dict(order), "") if order else (None, "not found")

    def has_sellable_quantity(self, symbol: str, qty: float):
        if self.position_qty >= qty:
            return True, self.position_qty, ""
        return False, self.position_qty, "not enough shares"

    def get_latest_price(self, symbol: str):
        return 100.0, ""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _run_tests() -> None:
    with TemporaryDirectory() as tmp:
        original_queue_path = pending_market_orders.QUEUE_PATH
        original_state_path = state_store.STATE_PATH
        original_alpaca = discord_agent.alpaca
        original_send = discord_agent._send_channel
        pending_market_orders.QUEUE_PATH = Path(tmp) / "pending_market_orders.sqlite3"
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        fake = FakeAlpaca()
        messages: list[str] = []

        async def capture(channel_id: int, content: str) -> None:
            messages.append(content)

        discord_agent.alpaca = fake
        discord_agent._send_channel = capture
        try:
            queued = pending_market_orders.enqueue_market_order(
                "AAPL", "buy", 2, "market_closed", 1.0
            )
            _check(len(pending_market_orders.list_queued_market_orders()) == 1, "BUY persisted")
            await discord_agent._process_pending_market_buys()
            _check(not pending_market_orders.list_queued_market_orders(), "accepted BUY removed")
            _check(len(state_store.list_pending_buys()) == 1, "accepted BUY awaits fill activation")
            await discord_agent._activate_filled_pending_buys()
            position = state_store.list_positions()[0]
            _check(position["symbol"] == "AAPL" and position["qty"] == 2, "filled BUY tracked")
            _check(abs(position["stop_price"] - 99.0) < 1e-9, "1 percent loss protection stored")
            _check(abs(position["target_price"] - 110.0) < 1e-9, "10 percent profit protection stored")
            _check(not state_store.list_pending_buys(), "filled tracker removed")
            _check(any("Protection active" in item for item in messages), "protection announced")

            # If Alpaca accepted an order before a process interruption, the
            # stable client ID reconciles it without submitting a duplicate.
            recovered = pending_market_orders.enqueue_market_order(
                "MSFT", "buy", 1, "market_closed", 0.5
            )
            existing = {
                "id": "recovered-order",
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
            fake.by_client[recovered["client_order_id"]] = existing
            fake.orders["recovered-order"] = existing
            calls_before = fake.submit_calls
            await discord_agent._process_pending_market_buys()
            _check(fake.submit_calls == calls_before, "reconciled order was not duplicated")
            _check(not pending_market_orders.list_queued_market_orders(), "reconciled queue removed")

            # Transient failures remain queued and retain diagnostic state.
            pending_market_orders.enqueue_market_order(
                "NVDA", "buy", 1, "market_closed", 0.5
            )
            fake.fail_error = "connection temporarily unavailable"
            await discord_agent._process_pending_market_buys()
            retained = pending_market_orders.list_queued_market_orders()
            _check(len(retained) == 1, "transient failure retained")
            _check(retained[0]["attempts"] == 1, "retry attempt recorded")
            fake.fail_error = ""

            # Partial fills add only newly filled shares to protection state.
            pending_market_orders.remove_queued_market_order(retained[0]["queue_id"])
            partial = pending_market_orders.enqueue_market_order(
                "META", "buy", 3, "market_closed", 0.5
            )
            partial_order = {
                "id": "partial-order",
                "status": "partially_filled",
                "filled_qty": "1",
                "filled_avg_price": "200",
                "client_order_id": partial["client_order_id"],
            }
            fake.by_client[partial["client_order_id"]] = partial_order
            fake.orders["partial-order"] = partial_order
            await discord_agent._process_pending_market_buys()
            await discord_agent._activate_filled_pending_buys()
            meta = next(item for item in state_store.list_positions() if item["symbol"] == "META")
            _check(meta["qty"] == 1, "first partial fill protected")
            fake.orders["partial-order"].update(status="filled", filled_qty="3")
            await discord_agent._activate_filled_pending_buys()
            meta = next(item for item in state_store.list_positions() if item["symbol"] == "META")
            _check(meta["qty"] == 3, "remaining partial fills protected once")

            # A queued SELL is removed only after broker acceptance.
            state_store.upsert_position("TSLA", 2, 110, "buy-tsla", 0.5)
            pending_market_orders.enqueue_market_order(
                "TSLA", "sell", 2, "market_closed", 0.5
            )
            await discord_agent._process_pending_sells()
            _check(not pending_market_orders.list_queued_market_orders(), "accepted SELL removed")
            _check(
                any(item["symbol"] == "TSLA" for item in state_store.list_positions()),
                "accepted but unfilled SELL keeps local position",
            )
            sell_order_id = next(
                order_id for order_id, item in fake.orders.items()
                if item.get("symbol") == "TSLA" and item.get("side") == "sell"
            )
            fake.orders[sell_order_id].update(
                status="filled", filled_qty="2", filled_avg_price="100"
            )
            await discord_agent._reconcile_pending_exit_orders()
            _check(
                not any(item["symbol"] == "TSLA" for item in state_store.list_positions()),
                "filled SELL closes local position",
            )
            _check(not state_store.list_pending_exit_orders(), "filled SELL tracker removed")

            # Submitted option entries become protected positions only after a fill.
            fake.orders["option-entry"] = {
                "id": "option-entry",
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
            }
            state_store.add_pending_option_entry_order(
                {
                    "order_id": "option-entry",
                    "occ_symbol": "AAPL260821C00240000",
                    "requested_qty": 1,
                    "root": "AAPL",
                    "side": "CALL",
                    "strike": 240,
                    "expiry_date": "2026-08-21",
                    "stop_loss": 2.2,
                    "target_price": 5.8,
                    "position_intent": "buy_to_open",
                }
            )
            await discord_agent._reconcile_pending_option_entry_orders()
            _check(not state_store.list_option_positions(), "unfilled option entry is not tracked")
            fake.orders["option-entry"].update(
                status="filled", filled_qty="1", filled_avg_price="3.45"
            )
            await discord_agent._reconcile_pending_option_entry_orders()
            option_position = state_store.list_option_positions()[0]
            _check(option_position["qty"] == 1, "filled option entry becomes tracked")
            _check(not state_store.list_pending_option_entry_orders(), "filled option tracker removed")

            # Old agent_state.json queue records migrate once into the new file.
            legacy = state_store.add_pending_market_buy("ORCL", 1, "market_closed")
            discord_agent._migrate_legacy_market_order_queue()
            migrated = pending_market_orders.list_queued_market_orders()
            _check(len(migrated) == 1 and migrated[0]["symbol"] == "ORCL", "legacy BUY migrated")
            _check(
                not any(item.get("queued") for item in state_store.list_pending_buys()),
                "legacy queue record removed from agent state",
            )
            pending_market_orders.remove_queued_market_order(migrated[0]["queue_id"])

            # An offline agent leaves the durable row untouched. Starting while
            # closed keeps it; starting while open submits it immediately.
            pending_market_orders.enqueue_market_order(
                "GOOGL", "buy", 1, "terminal_was_off", 1.0
            )
            fake.market_open = False
            await discord_agent._recover_persisted_orders_on_startup()
            _check(
                len(pending_market_orders.list_queued_market_orders()) == 1,
                "closed startup retained queued order",
            )
            fake.market_open = True
            await discord_agent._recover_persisted_orders_on_startup()
            _check(
                not pending_market_orders.list_queued_market_orders(),
                "open startup submitted queued order",
            )
            googl = next(
                item for item in state_store.list_positions() if item["symbol"] == "GOOGL"
            )
            _check(abs(googl["stop_price"] - 99.0) < 1e-9, "startup loss protection active")
            _check(abs(googl["target_price"] - 110.0) < 1e-9, "startup profit protection active")

            # Re-submitting the same manual signal while the market stays
            # closed reuses the same deterministic queue_id instead of
            # piling up a duplicate row.
            first = pending_market_orders.enqueue_market_order(
                "AAPL", "buy", 1, "market_closed", 1.0, "BUY:AAPL"
            )
            second = pending_market_orders.enqueue_market_order(
                "AAPL", "buy", 1, "market_closed", 1.0, "BUY:AAPL"
            )
            _check(first["_inserted"] is True, "first AAPL enqueue inserted a new row")
            _check(second["_inserted"] is False, "duplicate AAPL enqueue reused the same row")
            _check(
                len([o for o in pending_market_orders.list_queued_market_orders() if o["symbol"] == "AAPL"]) == 1,
                "duplicate enqueue did not create a second AAPL row",
            )
            pending_market_orders.remove_queued_market_order(first["queue_id"])

            # A queued order that can never legitimately succeed (position no
            # longer matches what was requested) is dropped after too many
            # attempts instead of retrying forever.
            original_max_attempts = config.pending_order_max_attempts
            try:
                object.__setattr__(config, "pending_order_max_attempts", 1)
                fake.position_qty = 0  # nothing sellable -- every attempt fails the same way
                pending_market_orders.enqueue_market_order("IBM", "sell", 5, "market_closed", 0.5)
                await discord_agent._process_pending_sells()
                retained = [o for o in pending_market_orders.list_queued_market_orders() if o["symbol"] == "IBM"]
                _check(len(retained) == 1 and retained[0]["attempts"] == 1, "first failed attempt recorded, not yet expired")
                await discord_agent._process_pending_sells()
                _check(
                    not [o for o in pending_market_orders.list_queued_market_orders() if o["symbol"] == "IBM"],
                    "queued SELL expired and was removed after exceeding max attempts",
                )
                _check(any("expired after" in item for item in messages), "expiry announced")
            finally:
                object.__setattr__(config, "pending_order_max_attempts", original_max_attempts)
                fake.position_qty = 10.0

            # A queued order that has simply sat too long is dropped on age
            # alone, even with zero failed attempts yet.
            original_max_age = config.pending_order_max_age_hours
            try:
                object.__setattr__(config, "pending_order_max_age_hours", 0.0)
                pending_market_orders.enqueue_market_order("ORCL", "buy", 1, "market_closed", 1.0)
                await discord_agent._process_pending_market_buys()
                _check(
                    not [o for o in pending_market_orders.list_queued_market_orders() if o["symbol"] == "ORCL"],
                    "queued BUY expired on age alone",
                )
            finally:
                object.__setattr__(config, "pending_order_max_age_hours", original_max_age)

            # The same age-based safety net applies to the option queue for
            # plain waiting reasons (market_closed, no contract listed yet, etc).
            original_max_age = config.pending_order_max_age_hours
            try:
                object.__setattr__(config, "pending_order_max_age_hours", 0.0)
                state_store.add_pending_option_order(
                    {
                        "pending_key": "option:stale-msft",
                        "occ_symbol": "MSFT260918C00420000",
                        "root": "MSFT",
                        "side": "CALL",
                        "qty": 1,
                        "order_side": "buy",
                        "reason": "market_closed",
                    }
                )
                await discord_agent._process_pending_option_orders()
                _check(
                    not any(item["pending_key"] == "option:stale-msft" for item in state_store.list_pending_option_orders()),
                    "queued option order expired on age alone",
                )
                _check(any("expired after" in item for item in messages), "option expiry announced")

                # A price-conditional entry ("buy once SPX crosses 6000") is
                # meant to wait indefinitely for its own trigger, so it must
                # survive the same age cutoff untouched.
                state_store.add_pending_option_order(
                    {
                        "pending_key": "option:waiting-spx",
                        "occ_symbol": "SPX261016C06000000",
                        "root": "SPX",
                        "side": "CALL",
                        "qty": 1,
                        "order_side": "buy",
                        "reason": "waiting_for_underlying_price_condition",
                        "underlying_trigger_direction": "above",
                        "underlying_trigger_price": 999999.0,
                    }
                )
                await discord_agent._process_pending_option_orders()
                _check(
                    any(item["pending_key"] == "option:waiting-spx" for item in state_store.list_pending_option_orders()),
                    "price-conditional option entry survives the age cutoff",
                )
                state_store.remove_pending_option_order("option:waiting-spx")

                # The same safety net applies to the multi-leg queue -- e.g. a
                # protective stop-loss/take-profit exit whose required legs
                # never match a held position again.
                state_store.add_pending_option_order(
                    {
                        "pending_key": "mleg-exit:stale",
                        "order_class": "mleg",
                        "root": "AAPL",
                        "qty": 1,
                        "reason": "multi_leg_stop_loss",
                        "legs": [
                            {"symbol": "AAPL261016P00130000", "requires_position": True, "ratio_qty": 1},
                            {"symbol": "AAPL261016P00140000", "requires_position": True, "ratio_qty": 1},
                        ],
                    }
                )
                await discord_agent._process_pending_option_orders()
                _check(
                    not any(item["pending_key"] == "mleg-exit:stale" for item in state_store.list_pending_option_orders()),
                    "queued multi-leg order expired on age alone",
                )
                _check(any("multi-leg" in item and "expired after" in item for item in messages), "multi-leg expiry announced")
            finally:
                object.__setattr__(config, "pending_order_max_age_hours", original_max_age)
        finally:
            discord_agent.alpaca = original_alpaca
            discord_agent._send_channel = original_send
            pending_market_orders.QUEUE_PATH = original_queue_path
            state_store.STATE_PATH = original_state_path


def run_all() -> None:
    asyncio.run(_run_tests())
    print("PENDING MARKET ORDER TESTS PASSED")


if __name__ == "__main__":
    run_all()
