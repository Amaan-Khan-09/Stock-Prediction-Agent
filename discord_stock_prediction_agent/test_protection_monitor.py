"""Integration checks for broker-facing protection monitor behavior."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from . import discord_agent, pending_market_orders, state_store
from .automate_agent import AUTOMATE_AGENT_TAG
from .options_parser import classify_and_parse


class ProtectionAlpaca:
    def __init__(self) -> None:
        self.market_open = True
        self.prices: dict[str, float] = {}
        self.quantities: dict[str, float] = {}
        self.orders: dict[str, dict] = {}
        self.submissions: list[dict] = []
        self.option_contracts: list[dict] = []

    def get_option_contracts(self, root: str, expiry, strike: float, side: str):
        return list(self.option_contracts), ""

    def ready(self):
        return True

    def is_market_open(self):
        return self.market_open, ""

    def get_position(self, symbol: str):
        qty = self.quantities.get(symbol, 0.0)
        if qty == 0:
            # Matches AlpacaPaperClient.get_position's real 404 wording --
            # _reconcile_pending_exit_orders's "fully closed" branch checks
            # for "no position" in the error text specifically.
            return None, "No position found."
        # Real Alpaca reports a short position's qty as negative -- keep the
        # sign here so tests can exercise that instead of always long qty.
        return {
            "symbol": symbol,
            "qty": str(qty),
            "current_price": str(self.prices[symbol]),
        }, ""

    def has_open_order(self, symbol: str):
        match = next(
            (
                order for order in self.orders.values()
                if order.get("symbol") == symbol
                and order.get("status") in {"accepted", "new", "pending_new"}
            ),
            None,
        )
        return match, ""

    def has_sellable_quantity(self, symbol: str, qty: float):
        held = self.quantities.get(symbol, 0.0)
        return held >= qty, held, "" if held >= qty else "not enough quantity"

    def submit_market_order(self, symbol: str, side: str, qty: float, client_order_id: str):
        return self._submit(symbol, side, qty, client_order_id)

    def submit_option_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        limit_price,
        position_intent: str,
        client_order_id: str,
    ):
        return self._submit(symbol, side, qty, client_order_id, position_intent)

    def has_multi_leg_options_trading(self):
        return True, ""

    def submit_multi_leg_option_order(
        self,
        legs: list[dict],
        qty: float,
        order_type: str,
        limit_price,
        client_order_id: str,
    ):
        order, error = self._submit(
            "MLEG", "close", qty, client_order_id, "multi_leg"
        )
        order["legs"] = legs
        self.orders[order["id"]]["legs"] = legs
        return order, error

    def _submit(self, symbol: str, side: str, qty: float, client_id: str, intent: str = ""):
        order = {
            "id": f"protection-{len(self.submissions) + 1}",
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "status": "accepted",
            "filled_qty": "0",
            "client_order_id": client_id,
            "position_intent": intent,
        }
        self.orders[order["id"]] = order
        self.submissions.append(order)
        return dict(order), ""

    def get_order(self, order_id: str):
        order = self.orders.get(order_id)
        return (dict(order), "") if order else (None, "not found")

    def wait_for_order(self, order_id: str, seconds: int = 10):
        order = self.orders.get(order_id)
        if not order:
            return None, "not found"
        if order.get("status") == "accepted":
            # Simulate a market order filling immediately at the current quote.
            fill_price = self.prices.get(order["symbol"], 0.0)
            order.update(status="filled", filled_qty=order["qty"], filled_avg_price=str(fill_price))
        return dict(order), ""

    def get_order_by_client_order_id(self, client_order_id: str):
        order = next(
            (x for x in self.orders.values() if x.get("client_order_id") == client_order_id),
            None,
        )
        return (dict(order), "") if order else (None, "not found")

    def get_latest_option_price(self, symbol: str):
        return self.prices.get(symbol, 0.0), ""

    def get_latest_price(self, symbol: str):
        return self.prices.get(symbol, 0.0), ""

    def get_clock(self):
        return {
            "is_open": self.market_open,
            "next_close": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        }, ""


async def _with_runtime(test_body) -> None:
    with TemporaryDirectory() as tmp:
        original_state = state_store.STATE_PATH
        original_queue = pending_market_orders.QUEUE_PATH
        original_alpaca = discord_agent.alpaca
        original_send = discord_agent._send_channel
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        pending_market_orders.QUEUE_PATH = Path(tmp) / "pending_market_orders.sqlite3"
        fake = ProtectionAlpaca()

        async def ignore_message(channel_id: int, content: str) -> None:
            return None

        discord_agent.alpaca = fake
        discord_agent._send_channel = ignore_message
        try:
            await test_body(fake)
        finally:
            state_store.STATE_PATH = original_state
            pending_market_orders.QUEUE_PATH = original_queue
            discord_agent.alpaca = original_alpaca
            discord_agent._send_channel = original_send


def test_equity_profit_monitor_submits_at_ten_percent() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("AAPL", 2, 100.0, "buy", 1.0, 10.0)
        fake.quantities["AAPL"] = 2
        fake.prices["AAPL"] = 110.0
        await discord_agent.stop_loss_monitor.coro()
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["symbol"] == "AAPL"
        assert state_store.list_pending_exit_orders()[0]["reason"] == "protection_take_profit"

    asyncio.run(_with_runtime(scenario))


def test_equity_stop_falls_back_to_latest_price_when_position_price_is_stale() -> None:
    """Alpaca's /v2/positions current_price can read back as 0/missing even
    while a position is genuinely held (e.g. a momentary data gap). The
    monitor must retry via get_latest_price instead of silently skipping the
    check for that cycle, mirroring the fallback the option monitor already
    has."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("AAPL", 2, 100.0, "buy", 1.0, 10.0)
        fake.quantities["AAPL"] = 2
        fake.prices["AAPL"] = 95.0  # past the 1% stop

        # Position endpoint reports no usable current_price this cycle.
        real_get_position = fake.get_position

        def stale_position(symbol: str):
            data, err = real_get_position(symbol)
            if data is not None:
                data["current_price"] = "0"
            return data, err

        fake.get_position = stale_position
        await discord_agent.stop_loss_monitor.coro()
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["symbol"] == "AAPL"
        assert state_store.list_pending_exit_orders()[0]["reason"] == "protection_stop_loss"

    asyncio.run(_with_runtime(scenario))


def test_one_positions_error_does_not_block_the_rest_from_being_checked() -> None:
    """Regression: the equity protection loop had no per-position error
    isolation, unlike every other per-symbol loop in this file -- an
    exception on a single position (a malformed record, an unexpected API
    response) would abort the whole for-loop, skipping every position after
    it that cycle. If the same condition recurred every cycle, that one
    position could permanently block all the others from ever being
    force-closed. A stuck/erroring symbol must never be able to prevent
    the rest of the book from being protected or force-closed."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("AAPL", 2, 100.0, "buy", 1.0, 10.0)
        state_store.upsert_position("MSFT", 3, 100.0, "buy", 1.0, 10.0)
        fake.quantities["AAPL"] = 2
        fake.quantities["MSFT"] = 3
        fake.prices["AAPL"] = 95.0  # past the 1% stop
        fake.prices["MSFT"] = 95.0  # past the 1% stop

        real_get_position = fake.get_position

        def raise_for_aapl(symbol: str):
            if symbol == "AAPL":
                raise RuntimeError("simulated malformed Alpaca response")
            return real_get_position(symbol)

        fake.get_position = raise_for_aapl
        await discord_agent.stop_loss_monitor.coro()

        assert len(fake.submissions) == 1, "AAPL's failure must not suppress MSFT's own check"
        assert fake.submissions[0]["symbol"] == "MSFT"
        assert state_store.list_pending_exit_orders()[0]["reason"] == "protection_stop_loss"
        # AAPL must still be tracked (not silently dropped) so it gets a
        # real chance again on the next cycle instead of vanishing.
        assert any(p["symbol"] == "AAPL" for p in state_store.list_positions())

    asyncio.run(_with_runtime(scenario))


def test_equity_closed_market_stop_is_durable_and_not_duplicated() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("MSFT", 3, 100.0, "buy", 1.0, 10.0)
        fake.quantities["MSFT"] = 3
        fake.prices["MSFT"] = 99.0
        fake.market_open = False
        await discord_agent.stop_loss_monitor.coro()
        await discord_agent.stop_loss_monitor.coro()
        queued = pending_market_orders.list_queued_market_orders()
        assert len(queued) == 1
        assert queued[0]["reason"] == "protection_stop_loss"

        # Simulate a later restart/monitor cycle after the market opens.
        fake.market_open = True
        await discord_agent.stop_loss_monitor.coro()
        assert not pending_market_orders.list_queued_market_orders()
        assert len(fake.submissions) == 1

    asyncio.run(_with_runtime(scenario))


async def _with_now_et(now_et: datetime, coro) -> None:
    original = discord_agent._now_et
    discord_agent._now_et = lambda: now_et
    try:
        await coro
    finally:
        discord_agent._now_et = original


def test_equity_exit_before_close_forces_a_flat_position_out() -> None:
    """A position tagged exit_before_market_close, priced flat (neither
    stop nor target hit), still gets force-closed once automate_agent's
    fixed daily exit-time cutoff (ET wall-clock, e.g. 12:30) has been
    reached -- covers the automate_agent fixed-window-flatten requirement.
    """
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position(
            "TSLA", 4, 100.0, "buy", 1.0, 10.0, "long",
            "automate_agent", True,
        )
        fake.quantities["TSLA"] = 4
        fake.prices["TSLA"] = 100.5  # flat -- neither the 1% stop nor 10% target
        await _with_now_et(
            datetime(2026, 8, 24, 12, 30, tzinfo=ZoneInfo("America/New_York")),
            discord_agent.stop_loss_monitor.coro(),
        )
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["symbol"] == "TSLA"
        assert fake.submissions[0]["side"] == "sell"

    asyncio.run(_with_runtime(scenario))


def test_equity_exit_before_close_does_not_fire_outside_the_window() -> None:
    """The same flagged, flat position is left alone before automate_agent's
    fixed daily exit-time cutoff has been reached."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position(
            "TSLA", 4, 100.0, "buy", 1.0, 10.0, "long",
            "automate_agent", True,
        )
        fake.quantities["TSLA"] = 4
        fake.prices["TSLA"] = 100.5
        await _with_now_et(
            datetime(2026, 8, 24, 11, 0, tzinfo=ZoneInfo("America/New_York")),
            discord_agent.stop_loss_monitor.coro(),
        )
        assert len(fake.submissions) == 0

    asyncio.run(_with_runtime(scenario))


def test_agent_positions_text_shows_side_so_shorts_are_not_ambiguous() -> None:
    with TemporaryDirectory() as tmp:
        original_state = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            state_store.upsert_position("NVDA", 10, 100.0, "long-entry", 1.0, 10.0)
            state_store.upsert_position("TSLA", 5, 100.0, "short-entry", 1.0, 10.0, "short")
            text = asyncio.run(discord_agent._build_agent_positions_text())
            assert "NVDA (LONG)" in text
            assert "TSLA (SHORT)" in text
            # A short's stop sits above entry -- without the (SHORT) label this
            # reads as a computation error rather than correct short protection.
            assert "stop $101.00" in text
        finally:
            state_store.STATE_PATH = original_state


def test_close_position_with_outcome_pnl_sign_matches_position_side() -> None:
    with TemporaryDirectory() as tmp:
        original_state = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            state_store.upsert_position("NVDA", 10, 100.0, "long-entry", 1.0, 10.0)
            long_outcome = state_store.close_position_with_outcome("NVDA", 10, 90.0, "protection_stop_loss")
            assert long_outcome["side"] == "sell_to_close_long"
            assert long_outcome["pnl_pct"] == -10.0  # bought at 100, sold at 90 -> a loss
            assert long_outcome["pnl_value"] == -100.0

            state_store.upsert_position("TSLA", 5, 100.0, "short-entry", 1.0, 10.0, "short")
            short_outcome = state_store.close_position_with_outcome("TSLA", 5, 90.0, "protection_take_profit")
            assert short_outcome["side"] == "buy_to_close_short"
            assert short_outcome["pnl_pct"] == 10.0  # shorted at 100, covered at 90 -> a gain
            assert short_outcome["pnl_value"] == 50.0
        finally:
            state_store.STATE_PATH = original_state


def test_queued_short_sell_tracks_a_new_short_entry_not_an_exit() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        pending_market_orders.enqueue_market_order(
            "AMC", "sell", 10, "manual_sell_market_closed", 1.0, "", "SELL_SHORT"
        )
        fake.prices["AMC"] = 5.0
        await discord_agent._process_pending_sells()

        assert not pending_market_orders.list_queued_market_orders()
        assert not state_store.list_pending_exit_orders(), (
            "a brand-new short entry must not be tracked as if it were closing a position"
        )
        positions = state_store.list_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AMC"
        assert positions[0]["side"] == "short"
        assert positions[0]["entry_price"] == 5.0
        assert positions[0]["qty"] == 10.0

    asyncio.run(_with_runtime(scenario))


def test_short_equity_protection_uses_buy_to_cover() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("TSLA", 5, 100.0, "short-entry", 1.0, 10.0, "short")
        position = state_store.list_positions()[0]
        assert position["side"] == "short"
        assert position["stop_price"] == 101.0  # short stop sits above entry
        assert position["target_price"] == 90.0  # short target sits below entry

        fake.quantities["TSLA"] = -5  # Alpaca reports a short position as negative qty
        fake.prices["TSLA"] = 106.0  # price rose past the short's stop level
        await discord_agent.stop_loss_monitor.coro()
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["side"] == "buy"
        assert fake.submissions[0]["symbol"] == "TSLA"
        assert state_store.list_pending_exit_orders()[0]["reason"] == "protection_stop_loss"

    asyncio.run(_with_runtime(scenario))


def test_short_equity_closed_market_stop_queues_a_cover_buy() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("GME", 4, 50.0, "short-entry", 1.0, 10.0, "short")
        fake.quantities["GME"] = -4
        fake.prices["GME"] = 51.0  # past the 1% short stop
        fake.market_open = False
        await discord_agent.stop_loss_monitor.coro()
        queued = pending_market_orders.list_queued_market_orders()
        assert len(queued) == 1
        assert queued[0]["side"] == "buy"
        assert queued[0]["action"] == "BUY_TO_COVER"

        fake.market_open = True
        await discord_agent.stop_loss_monitor.coro()
        assert not pending_market_orders.list_queued_market_orders()
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["side"] == "buy"

    asyncio.run(_with_runtime(scenario))


def test_option_loss_and_profit_monitors_submit_at_boundaries() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        loss_symbol = "AAPL260821C00240000"
        profit_symbol = "MSFT260821C00500000"
        state_store.upsert_option_position(
            loss_symbol, "AAPL", "CALL", 240, "2026-08-21", 1, 10.0, "buy-a"
        )
        state_store.upsert_option_position(
            profit_symbol, "MSFT", "CALL", 500, "2026-08-21", 1, 10.0, "buy-m"
        )
        fake.quantities.update({loss_symbol: 1, profit_symbol: 1})
        fake.prices.update({loss_symbol: 9.5, profit_symbol: 11.0})
        await discord_agent._process_option_exit_monitor(True)
        assert {x["symbol"] for x in fake.submissions} == {loss_symbol, profit_symbol}
        reasons = {x["reason"] for x in state_store.list_pending_exit_orders()}
        assert reasons == {"option_stop_loss", "option_target_price"}

    asyncio.run(_with_runtime(scenario))


def test_automate_agent_option_close_records_outcome_for_circuit_breaker() -> None:
    """automate_agent-tagged option positions must record a trade_outcomes
    entry on close (regular manual option positions never have) so
    today_realized_pnl(AUTOMATE_AGENT_TAG) -- the daily-loss circuit breaker
    -- can see option P&L, not just equity P&L."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        occ_symbol = "AAPL260821C00100000"
        state_store.upsert_option_position(
            occ_symbol, "AAPL", "CALL", 100, "2026-08-21", 1, 2.0, "buy-1",
            default_stop_loss_pct=1.0, opened_by=AUTOMATE_AGENT_TAG,
        )
        fake.quantities[occ_symbol] = 1
        fake.prices[occ_symbol] = 1.9  # past the 1% stop (entry 2.0)
        await discord_agent._process_option_exit_monitor(True)
        pending = state_store.list_pending_exit_orders()
        assert pending, "stop should have triggered an exit"

        exit_order_id = pending[0]["order_id"]
        fake.orders[exit_order_id].update(status="filled", filled_qty="1", filled_avg_price="1.9")
        fake.quantities[occ_symbol] = 0  # broker now reports the position fully closed
        await discord_agent._reconcile_pending_exit_orders()

        assert not state_store.list_option_positions()
        assert state_store.today_realized_pnl(AUTOMATE_AGENT_TAG) < 0, (
            "the option loss must count toward automate_agent's own daily-loss circuit breaker"
        )

    asyncio.run(_with_runtime(scenario))


def test_one_pending_exits_error_does_not_block_reconciling_the_rest() -> None:
    """Regression: _reconcile_pending_exit_orders had no per-order error
    isolation -- an exception reconciling a single pending exit (a
    malformed broker response, an unexpected field) would abort the whole
    loop, leaving every other pending exit in that batch un-reconciled that
    cycle. This is the function that actually finalizes a submitted sell
    into a closed position, so one bad order must never be able to prevent
    the rest of a force-close (e.g. the 12:30 cutoff) from actually
    completing."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        state_store.upsert_position("AAPL", 2, 100.0, "buy", 1.0, 10.0)
        state_store.upsert_position("MSFT", 3, 100.0, "buy", 1.0, 10.0)
        fake.quantities["AAPL"] = 2
        fake.quantities["MSFT"] = 3
        fake.prices["AAPL"] = 95.0  # past the 1% stop
        fake.prices["MSFT"] = 95.0  # past the 1% stop
        await discord_agent.stop_loss_monitor.coro()
        pending = state_store.list_pending_exit_orders()
        assert len(pending) == 2

        for item in pending:
            order_id = item["order_id"]
            fake.orders[order_id].update(status="filled", filled_qty=item["requested_qty"], filled_avg_price="95.0")
        fake.quantities["AAPL"] = 0
        fake.quantities["MSFT"] = 0

        aapl_order_id = next(p["order_id"] for p in pending if p["symbol"] == "AAPL")
        real_get_order = fake.get_order

        def raise_for_aapl_order(order_id: str):
            if order_id == aapl_order_id:
                raise RuntimeError("simulated malformed Alpaca response")
            return real_get_order(order_id)

        fake.get_order = raise_for_aapl_order
        await discord_agent._reconcile_pending_exit_orders()

        assert "MSFT" not in {p["symbol"] for p in state_store.list_positions()}, (
            "AAPL's reconciliation failure must not block MSFT's from completing"
        )
        assert "AAPL" in {p["symbol"] for p in state_store.list_positions()}, (
            "AAPL must still be tracked (not silently dropped) so it can be retried"
        )

    asyncio.run(_with_runtime(scenario))


def test_manual_option_close_does_not_record_an_outcome() -> None:
    """Regression guard: only automate_agent-tagged option positions get the
    new outcome-recording behavior -- a manual (untagged) option position
    must close exactly as it always has, with no trade_outcomes entry."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        occ_symbol = "MSFT260821C00500000"
        state_store.upsert_option_position(
            occ_symbol, "MSFT", "CALL", 500, "2026-08-21", 1, 10.0, "buy-m",
        )
        fake.quantities[occ_symbol] = 1
        fake.prices[occ_symbol] = 9.5  # past the default 5% stop
        await discord_agent._process_option_exit_monitor(True)
        pending = state_store.list_pending_exit_orders()
        assert pending

        exit_order_id = pending[0]["order_id"]
        fake.orders[exit_order_id].update(status="filled", filled_qty="1", filled_avg_price="9.5")
        fake.quantities[occ_symbol] = 0  # broker now reports the position fully closed
        before = state_store.today_realized_pnl("")
        await discord_agent._reconcile_pending_exit_orders()

        assert not state_store.list_option_positions()
        assert state_store.today_realized_pnl("") == before, (
            "manual option closes must not start recording outcomes as a side effect"
        )

    asyncio.run(_with_runtime(scenario))


def test_option_signal_risk_percent_overrides_default_stop() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        symbol = "ARM261016C00190000"
        state_store.upsert_option_position(
            symbol,
            "ARM",
            "CALL",
            190,
            "2026-10-16",
            1,
            10.0,
            "arm-entry",
            risk_stop_pct=1.0,
            position_type="starter",
        )
        position = state_store.list_option_positions()[0]
        assert position["stop_loss"] == 9.9
        assert position["stop_loss_pct"] == 1.0
        assert position["position_type"] == "starter"
        fake.quantities[symbol] = 1
        fake.prices[symbol] = 9.9
        await discord_agent._process_option_exit_monitor(True)
        assert len(fake.submissions) == 1
        assert state_store.list_pending_exit_orders()[0]["reason"] == "option_stop_loss"

    asyncio.run(_with_runtime(scenario))


def test_short_option_protection_uses_buy_to_close() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        symbol = "QQQ261016P00560000"
        state_store.upsert_option_position(
            symbol, "QQQ", "PUT", 560, "2026-10-16", 8, 2.80, "qqq-short",
            stop_loss=4.20, target_price=1.00, position_intent="sell_to_open",
        )
        fake.quantities[symbol] = 8
        fake.prices[symbol] = 1.00
        await discord_agent._process_option_exit_monitor(True)
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["side"] == "buy"
        assert fake.submissions[0]["position_intent"] == "buy_to_close"

    asyncio.run(_with_runtime(scenario))


def test_option_dollar_max_loss_and_underlying_stop_are_enforced() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        cost = "COST261218P01000000"
        xom = "XOM261016C00130000"
        state_store.upsert_option_position(
            cost, "COST", "PUT", 1000, "2026-12-18", 2, 20.0, "cost-entry",
            maximum_loss_amount=1200.0,
        )
        state_store.upsert_option_position(
            xom, "XOM", "CALL", 130, "2026-10-16", 5, 2.10, "xom-entry",
            position_intent="sell_to_open", exit_underlying_direction="above",
            exit_underlying_price=128.0,
        )
        fake.quantities.update({cost: 2, xom: 5})
        fake.prices.update({cost: 14.0, xom: 2.10, "XOM": 129.0})
        await discord_agent._process_option_exit_monitor(True)
        reasons = {item["reason"] for item in state_store.list_pending_exit_orders()}
        assert reasons == {"option_maximum_loss", "option_underlying_stop"}
        xom_order = next(item for item in fake.submissions if item["symbol"] == xom)
        assert xom_order["side"] == "buy"
        assert xom_order["position_intent"] == "buy_to_close"

    asyncio.run(_with_runtime(scenario))


def test_option_time_exit_uses_signal_minutes_before_close() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        symbol = "NFLX261218P01450000"
        state_store.upsert_option_position(
            symbol,
            "NFLX",
            "PUT",
            1450,
            "2026-12-18",
            2,
            18.60,
            "nflx-entry",
            exit_before_market_close=True,
            exit_minutes_before_close=30,
            exit_if_target_not_hit=True,
        )
        fake.quantities[symbol] = 2
        fake.prices[symbol] = 18.75
        fake.get_clock = lambda: ({
            "is_open": True,
            "next_close": (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat(),
        }, "")
        await discord_agent._process_option_exit_monitor(True)
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["qty"] == "2.0"
        assert state_store.list_pending_exit_orders()[0]["reason"] == "option_time_exit"

    asyncio.run(_with_runtime(scenario))


def test_contract_pending_option_is_submitted_once_alpaca_lists_it() -> None:
    """A signal accepted while its exact contract isn't listed yet (e.g. a
    far-dated option Alpaca doesn't carry today) must be retried by the
    monitor and submitted -- with its full SL/TP field set intact -- the
    moment alpaca.get_option_contracts starts returning it."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        option = classify_and_parse(
            "BTO NFLX 1450P 12/19/2026 @18.60 Qty 2 TP 25.00 SL 12.00"
        ).option
        assert option and option.valid
        discord_agent._queue_option_order(
            "", option, 2, "limit", option.expiry_date, {"score": 60, "risk_reward": 0.97},
            "waiting_for_tradable_contract", "buy", "buy_to_open", False, True,
        )
        assert len(state_store.list_pending_option_orders()) == 1

        # First check: Alpaca still doesn't list this contract -- must stay queued.
        fake.option_contracts = []
        await discord_agent._process_pending_option_orders()
        assert not fake.submissions
        assert len(state_store.list_pending_option_orders()) == 1

        # Second check: Alpaca now lists it -- must submit with the stored fields.
        occ_symbol = "NFLX261219P01450000"
        fake.option_contracts = [{"symbol": occ_symbol, "expiration_date": "2026-12-19"}]
        await discord_agent._process_pending_option_orders()
        assert not state_store.list_pending_option_orders()
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["symbol"] == occ_symbol
        assert fake.submissions[0]["side"] == "buy"
        assert fake.submissions[0]["position_intent"] == "buy_to_open"

        tracked = state_store.list_pending_option_entry_orders()
        assert len(tracked) == 1
        assert tracked[0]["occ_symbol"] == occ_symbol
        assert tracked[0]["stop_loss"] == 12.0
        assert tracked[0]["target_price"] == 25.0

    asyncio.run(_with_runtime(scenario))


def test_underlying_scale_in_waits_for_base_position_and_breakout() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        symbol = "AVGO261120C00420000"
        queued = discord_agent._queue_conditional_option_scale_in(
            {
                "occ_symbol": symbol,
                "root": "AVGO",
                "side": "CALL",
                "strike": 420,
                "expiry_date": "2026-11-20",
                "qty": 3,
                "add_quantity": 3,
                "add_trigger_underlying_direction": "above",
                "add_trigger_underlying_price": 425,
                "base_order_id": "avgo-base",
                "raw_input": "test",
            }
        )
        assert queued
        fake.orders["avgo-base"] = {
            "id": "avgo-base",
            "symbol": symbol,
            "status": "accepted",
            "filled_qty": "0",
        }
        fake.prices["AVGO"] = 426
        await discord_agent._process_pending_option_orders()
        assert not fake.submissions

        fake.quantities[symbol] = 2
        fake.prices[symbol] = 6.80
        fake.prices["AVGO"] = 424.99
        await discord_agent._process_pending_option_orders()
        assert not fake.submissions

        fake.orders["avgo-base"]["status"] = "filled"
        fake.orders["avgo-base"]["filled_qty"] = "2"
        fake.prices["AVGO"] = 425.01
        await discord_agent._process_pending_option_orders()
        assert len(fake.submissions) == 1
        assert fake.submissions[0]["symbol"] == symbol
        assert fake.submissions[0]["qty"] == "3.0"
        assert fake.submissions[0]["position_intent"] == "buy_to_open"
        assert not state_store.list_pending_option_orders()

    asyncio.run(_with_runtime(scenario))


def test_multi_leg_fill_activates_and_profit_exit_reconciles() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        long_leg = "AAPL260821C00240000"
        short_leg = "AAPL260821C00250000"
        legs = [
            {
                "symbol": long_leg,
                "side_order": "buy",
                "position_intent": "buy_to_open",
                "ratio_qty": 1,
            },
            {
                "symbol": short_leg,
                "side_order": "sell",
                "position_intent": "sell_to_open",
                "ratio_qty": 1,
            },
        ]
        fake.orders["mleg-entry"] = {
            "id": "mleg-entry",
            "status": "filled",
            "filled_qty": "1",
            "filled_avg_price": "4.60",
        }
        state_store.add_pending_multi_leg_entry_order(
            {
                "order_id": "mleg-entry",
                "strategy_id": "mleg-entry",
                "requested_qty": 1,
                "root": "AAPL",
                "structure": "bull_call_spread",
                "legs": legs,
                "price_effect": "debit",
                "protectable": True,
            }
        )
        await discord_agent._reconcile_pending_multi_leg_entry_orders()
        strategy = state_store.list_multi_leg_positions()[0]
        assert strategy["entry_net_price"] == 4.6
        assert strategy["stop_loss"] == 4.37
        assert strategy["target_price"] == 5.06

        fake.prices.update({long_leg: 7.0, short_leg: 1.94})
        await discord_agent._process_multi_leg_exit_monitor(True)
        exit_tracker = state_store.list_pending_exit_orders()[0]
        assert exit_tracker["asset_type"] == "option_mleg"
        exit_order = fake.orders[exit_tracker["order_id"]]
        exit_order.update(status="filled", filled_qty="1", filled_avg_price="5.06")
        await discord_agent._reconcile_pending_exit_orders()
        assert not state_store.list_multi_leg_positions()
        assert not state_store.list_pending_exit_orders()

    asyncio.run(_with_runtime(scenario))


def test_multi_leg_stalled_leg_quote_skips_silently_but_logs_a_warning(caplog) -> None:
    """A thinly-traded wing with no fresh quote must not crash or falsely
    trigger/hide an exit -- but it also must not go completely silent, or a
    spread can sit unprotected indefinitely with no way to diagnose why."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        long_leg = "AAPL260821C00240000"
        short_leg = "AAPL260821C00250000"
        legs = [
            {"symbol": long_leg, "side_order": "buy", "position_intent": "buy_to_open", "ratio_qty": 1},
            {"symbol": short_leg, "side_order": "sell", "position_intent": "sell_to_open", "ratio_qty": 1},
        ]
        fake.orders["mleg-entry-2"] = {
            "id": "mleg-entry-2", "status": "filled", "filled_qty": "1", "filled_avg_price": "4.60",
        }
        state_store.add_pending_multi_leg_entry_order(
            {
                "order_id": "mleg-entry-2",
                "strategy_id": "mleg-entry-2",
                "requested_qty": 1,
                "root": "AAPL",
                "structure": "bull_call_spread",
                "legs": legs,
                "price_effect": "debit",
                "protectable": True,
            }
        )
        await discord_agent._reconcile_pending_multi_leg_entry_orders()

        # Only the long leg has a quote this cycle; the short leg (thin wing)
        # has none, so fake.get_latest_option_price falls back to its 0.0 default.
        fake.prices.update({long_leg: 7.0})
        with caplog.at_level("WARNING", logger="discord_stock_prediction_agent"):
            await discord_agent._process_multi_leg_exit_monitor(True)

        assert not fake.submissions
        assert not state_store.list_pending_exit_orders()
        assert any(
            "multi-leg protection skipped" in record.message and short_leg in record.message
            for record in caplog.records
        )

    asyncio.run(_with_runtime(scenario))


def test_agent_health_reports_a_recent_monitor_tick() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        original = discord_agent._stop_loss_monitor_last_tick
        try:
            await discord_agent.stop_loss_monitor.coro()
            text = await discord_agent._build_agent_health_text()
            assert "Position monitor: last ran" in text
            assert "STALLED" not in text
            assert "Agent Health: READY" in text
        finally:
            discord_agent._stop_loss_monitor_last_tick = original

    asyncio.run(_with_runtime(scenario))


def test_agent_health_flags_a_stalled_monitor() -> None:
    """Regression: the position monitor stalling with no exception, no
    restart, and no log activity at all is a real incident this is meant to
    catch -- the only way anyone noticed live was manually checking
    positions well after the 12:30 force-close should have already
    happened. !agent_health must surface this proactively."""
    async def scenario(fake: ProtectionAlpaca) -> None:
        original = discord_agent._stop_loss_monitor_last_tick
        try:
            discord_agent._stop_loss_monitor_last_tick = (
                asyncio.get_event_loop().time() - 10_000
            )
            text = await discord_agent._build_agent_health_text()
            assert "Position monitor: last ran" in text
            assert "STALLED" in text
            assert "Agent Health: DEGRADED" in text
        finally:
            discord_agent._stop_loss_monitor_last_tick = original

    asyncio.run(_with_runtime(scenario))


def test_agent_health_before_any_tick_this_process() -> None:
    async def scenario(fake: ProtectionAlpaca) -> None:
        original = discord_agent._stop_loss_monitor_last_tick
        try:
            discord_agent._stop_loss_monitor_last_tick = 0.0
            text = await discord_agent._build_agent_health_text()
            assert "has not run yet this process" in text
        finally:
            discord_agent._stop_loss_monitor_last_tick = original

    asyncio.run(_with_runtime(scenario))
