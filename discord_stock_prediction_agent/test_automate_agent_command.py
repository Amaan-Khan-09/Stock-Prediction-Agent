"""Integration checks for the actual `!automate_agent` command -- the async
orchestration in discord_agent.py, not just the pure decision logic in
automate_agent.py (already covered by test_automate_agent.py).

Uses a fake Alpaca client and a mocked prediction engine so nothing here
touches a real network, Gemini, or broker call.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from . import discord_agent, state_store
from .automate_agent import AUTOMATE_AGENT_TAG


class FakeAutomateAlpaca:
    def __init__(self) -> None:
        self.market_open = True
        self.prices: dict[str, float] = {}
        self.quantities: dict[str, float] = {}
        self.submissions: list[dict] = []

    def ready(self):
        return True

    def is_market_open(self):
        return self.market_open, ""

    def get_position(self, symbol: str):
        qty = self.quantities.get(symbol, 0.0)
        if qty == 0:
            return None, "not found"
        return {"symbol": symbol, "qty": str(qty), "current_price": str(self.prices.get(symbol, 0.0))}, ""

    def get_latest_price(self, symbol: str):
        price = self.prices.get(symbol, 0.0)
        return (price, "") if price else (None, "no quote")

    def submit_market_order(self, symbol: str, side: str, qty: float, client_order_id: str = ""):
        order = {"id": f"auto-{len(self.submissions) + 1}", "symbol": symbol, "side": side, "qty": str(qty)}
        self.submissions.append(order)
        return dict(order), ""


async def _with_runtime(test_body, predictions: dict[str, str]) -> None:
    """predictions maps symbol -> decision ("BUY"/"HOLD"/"SELL"); any
    watchlist symbol not present in the map is treated as a failed lookup
    (status != SUCCESS), matching a real provider error for that symbol.
    """
    def fake_predict(symbol: str, horizon_days=None):
        if symbol not in predictions:
            return {"status": "FAILED", "symbol": symbol, "decision": "REVIEW"}
        return {"status": "SUCCESS", "symbol": symbol, "decision": predictions[symbol]}

    with TemporaryDirectory() as tmp:
        original_state = state_store.STATE_PATH
        original_alpaca = discord_agent.alpaca
        original_predict = discord_agent.run_project_prediction
        original_send = discord_agent._send_channel
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        fake = FakeAutomateAlpaca()
        sent: list[tuple[int, str]] = []

        async def capture_send(channel_id: int, content: str = "", embed=None) -> None:
            sent.append((channel_id, content))

        discord_agent.alpaca = fake
        discord_agent.run_project_prediction = fake_predict
        discord_agent._send_channel = capture_send
        # The cooldown/lock are module-level globals so they persist across
        # sequential test runs in the same process -- reset both so each
        # test starts from a clean slate regardless of run order.
        discord_agent._automate_agent_last_run = 0.0
        if discord_agent._automate_agent_lock.locked():
            discord_agent._automate_agent_lock.release()
        try:
            await test_body(fake, sent)
        finally:
            state_store.STATE_PATH = original_state
            discord_agent.alpaca = original_alpaca
            discord_agent.run_project_prediction = original_predict
            discord_agent._send_channel = original_send


def test_agent_off_takes_no_action() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        state_store.set_agent_mode("OFF")
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        text = await discord_agent._build_automate_agent_text()
        assert "agent mode is off" in text.lower()
        assert not state_store.list_positions()
        assert not fake.submissions

    predictions = {discord_agent.config.automate_agent_watchlist[0]: "BUY"}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_cooldown_blocks_immediate_re_run() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        first = await discord_agent._build_automate_agent_text()
        assert len(fake.submissions) == 1, "first run buys normally"

        second = await discord_agent._build_automate_agent_text()
        assert "cooling down" in second.lower()
        assert len(fake.submissions) == 1, "no second order placed during cooldown"

    predictions = {discord_agent.config.automate_agent_watchlist[0]: "BUY"}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_concurrent_invocations_do_not_double_fill_slots() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        for sym in discord_agent.config.automate_agent_watchlist:
            fake.prices[sym] = 50.0
        results = await asyncio.gather(
            discord_agent._build_automate_agent_text(),
            discord_agent._build_automate_agent_text(),
        )
        assert any("already running" in r.lower() for r in results), (
            "the second overlapping call should see the lock, not run a second scan"
        )
        positions = state_store.list_positions()
        assert len(positions) <= discord_agent.config.automate_agent_max_positions

    predictions = {sym: "BUY" for sym in discord_agent.config.automate_agent_watchlist}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_one_symbol_erroring_during_buy_does_not_abort_the_cycle() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        watchlist = discord_agent.config.automate_agent_watchlist
        bad_symbol, good_symbol = watchlist[0], watchlist[1]
        fake.prices[good_symbol] = 80.0
        fake.prices[bad_symbol] = 80.0

        original_submit = fake.submit_market_order
        def flaky_submit(symbol, side, qty, client_order_id=""):
            if symbol == bad_symbol and side == "buy":
                raise RuntimeError("simulated transient broker error")
            return original_submit(symbol, side, qty, client_order_id)
        fake.submit_market_order = flaky_submit

        text = await discord_agent._build_automate_agent_text()
        symbols_bought = {p["symbol"] for p in state_store.list_positions()}
        assert good_symbol in symbols_bought, "the good symbol still gets bought"
        assert bad_symbol not in symbols_bought, "the erroring symbol is skipped, not crashing the cycle"
        assert "unexpected error" in text.lower()

    predictions = {discord_agent.config.automate_agent_watchlist[0]: "BUY", discord_agent.config.automate_agent_watchlist[1]: "BUY"}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_market_closed_takes_no_action() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        fake.market_open = False
        text = await discord_agent._build_automate_agent_text()
        assert "market is closed" in text.lower()
        assert not state_store.list_positions()
        assert not fake.submissions
        assert not sent  # nothing posted to the review channel either

    asyncio.run(_with_runtime(scenario, predictions={}))


def test_no_buy_candidates_places_no_trades() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        text = await discord_agent._build_automate_agent_text()
        assert "no trades placed" in text.lower()
        assert not state_store.list_positions()
        assert not fake.submissions

    watchlist_all_hold = {sym: "HOLD" for sym in discord_agent.config.automate_agent_watchlist}
    asyncio.run(_with_runtime(scenario, predictions=watchlist_all_hold))


def test_open_market_buys_a_boom_candidate_and_tags_it() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        text = await discord_agent._build_automate_agent_text()

        assert len(fake.submissions) == 1
        assert fake.submissions[0]["symbol"] == symbol
        assert fake.submissions[0]["side"] == "buy"

        positions = state_store.list_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == symbol
        assert positions[0]["opened_by"] == AUTOMATE_AGENT_TAG
        assert positions[0]["exit_before_market_close"] is True
        assert positions[0]["stop_loss_pct"] == discord_agent.config.equity_stop_loss_pct
        assert positions[0]["take_profit_pct"] == discord_agent.config.equity_take_profit_pct

        # Summary must reach the stock-review channel, not stay silent.
        assert any(symbol in content for _, content in sent)

    predictions = {discord_agent.config.automate_agent_watchlist[0]: "BUY"}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_at_cap_evicts_oldest_automate_position_before_buying() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        watchlist = discord_agent.config.automate_agent_watchlist
        max_positions = discord_agent.config.automate_agent_max_positions
        # Fill the cap with automate_agent-tagged positions, oldest first.
        for i in range(max_positions):
            sym = watchlist[i]
            state_store.upsert_position(sym, 1, 50.0, "", 1.0, 10.0, "long", AUTOMATE_AGENT_TAG, True)
            fake.quantities[sym] = 1
            fake.prices[sym] = 50.0
        new_symbol = watchlist[max_positions]  # not yet held
        fake.prices[new_symbol] = 75.0

        text = await discord_agent._build_automate_agent_text()

        symbols_after = {p["symbol"] for p in state_store.list_positions()}
        assert new_symbol in symbols_after, "the new BUY candidate was bought"
        assert watchlist[0] not in symbols_after, "the oldest automate position was evicted"
        assert len(symbols_after) == max_positions, "still at, not above, the cap"

    predictions = {discord_agent.config.automate_agent_watchlist[discord_agent.config.automate_agent_max_positions]: "BUY"}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_a_failed_symbol_lookup_does_not_abort_the_whole_scan() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        watchlist = discord_agent.config.automate_agent_watchlist
        good_symbol = watchlist[1]
        fake.prices[good_symbol] = 60.0
        # watchlist[0] is deliberately absent from `predictions`, simulating
        # a provider failure for that one symbol.
        text = await discord_agent._build_automate_agent_text()
        assert any(p["symbol"] == good_symbol for p in state_store.list_positions())

    watchlist = None
    async def _run():
        wl = discord_agent.config.automate_agent_watchlist
        await _with_runtime(scenario, predictions={wl[1]: "BUY"})
    asyncio.run(_run())


if __name__ == "__main__":
    test_agent_off_takes_no_action()
    test_cooldown_blocks_immediate_re_run()
    test_concurrent_invocations_do_not_double_fill_slots()
    test_one_symbol_erroring_during_buy_does_not_abort_the_cycle()
    test_market_closed_takes_no_action()
    test_no_buy_candidates_places_no_trades()
    test_open_market_buys_a_boom_candidate_and_tags_it()
    test_at_cap_evicts_oldest_automate_position_before_buying()
    test_a_failed_symbol_lookup_does_not_abort_the_whole_scan()
    print("ALL AUTOMATE_AGENT COMMAND INTEGRATION TESTS PASSED")
