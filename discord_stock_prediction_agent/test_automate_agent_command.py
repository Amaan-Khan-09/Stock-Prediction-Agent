"""Integration checks for the actual `!automate_agent` command -- the async
orchestration in discord_agent.py, not just the pure decision logic in
automate_agent.py (already covered by test_automate_agent.py).

Uses a fake Alpaca client and a mocked prediction engine so nothing here
touches a real network, Gemini, or broker call.
"""
from __future__ import annotations

import asyncio
from datetime import date
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
        self.orders: dict[str, dict] = {}
        self.equity = 50_000.0
        self.account_ok = True
        # options-mode fakes: option_contracts_by_expiry maps an ISO expiry
        # date string -> list of {"symbol", "strike_price"} dicts, matching
        # Alpaca's real /v2/options/contracts shape closely enough for
        # get_option_contracts's ATM-nearest-strike selection to work.
        self.option_contracts_by_expiry: dict[str, list[dict]] = {}
        self.option_premiums: dict[str, float] = {}
        self.options_trading_enabled = True

    def ready(self):
        return True

    def is_market_open(self):
        return self.market_open, ""

    def get_account(self):
        if not self.account_ok:
            return None, "account temporarily unavailable"
        return {"equity": str(self.equity)}, ""

    def get_position(self, symbol: str):
        qty = self.quantities.get(symbol, 0.0)
        if qty == 0:
            return None, "not found"
        return {"symbol": symbol, "qty": str(qty), "current_price": str(self.prices.get(symbol, 0.0))}, ""

    def get_latest_price(self, symbol: str):
        price = self.prices.get(symbol, 0.0)
        return (price, "") if price else (None, "no quote")

    def submit_market_order(self, symbol: str, side: str, qty: float, client_order_id: str = ""):
        # status starts "accepted", not "filled" -- reconciliation tests
        # simulate the broker confirming the fill in a later call to
        # get_order, matching how a real market order isn't instantaneous.
        order = {
            "id": f"auto-{len(self.submissions) + 1}", "symbol": symbol, "side": side, "qty": str(qty),
            "status": "accepted", "filled_qty": "0", "filled_avg_price": "0",
        }
        self.submissions.append(order)
        self.orders[order["id"]] = order
        return dict(order), ""

    def get_order(self, order_id: str):
        order = self.orders.get(order_id)
        return (dict(order), "") if order else (None, "not found")

    def mark_order_filled(self, order_id: str, fill_price: float | None = None) -> None:
        """Test helper: simulate the broker confirming a fill."""
        order = self.orders.get(order_id)
        if not order:
            return
        price = fill_price if fill_price is not None else self.prices.get(order["symbol"], 0.0)
        order.update(status="filled", filled_qty=order["qty"], filled_avg_price=str(price))

    def get_option_contracts(self, underlying: str, expiration_date=None, strike=None, option_type=None):
        contracts = self.option_contracts_by_expiry.get(str(expiration_date or ""), [])
        if not contracts:
            return None, "No tradable option contract found."
        return list(contracts), ""

    def has_options_trading(self):
        return self.options_trading_enabled, "" if self.options_trading_enabled else "options trading not enabled"

    def submit_option_order(self, occ_symbol, side, qty, order_type="market", limit_price=None, position_intent=None, client_order_id=""):
        order = {
            "id": f"autoopt-{len(self.submissions) + 1}", "symbol": occ_symbol, "side": side, "qty": str(qty),
            "status": "accepted", "filled_qty": "0", "filled_avg_price": "0", "position_intent": position_intent,
        }
        self.submissions.append(order)
        self.orders[order["id"]] = order
        return dict(order), ""

    def get_latest_option_price(self, occ_symbol: str):
        price = self.option_premiums.get(occ_symbol, 0.0)
        return (price, "") if price else (None, "no quote")


async def _with_runtime(test_body, predictions: dict[str, object]) -> None:
    """predictions maps symbol -> either a bare decision string
    ("BUY"/"HOLD"/"SELL"), or a dict with a "decision" key plus optional
    "confidence_score"/"predicted_return_pct"/"needs_human_review" entries
    to exercise the ai_prediction extraction path in
    _scan_automate_agent_watchlist. Any watchlist symbol not present in the
    map is treated as a failed lookup (status != SUCCESS), matching a real
    provider error for that symbol.
    """
    def fake_predict(symbol: str, horizon_days=None):
        if symbol not in predictions:
            return {"status": "FAILED", "symbol": symbol, "decision": "REVIEW"}
        spec = predictions[symbol]
        if isinstance(spec, dict):
            ai_prediction = {k: v for k, v in spec.items() if k != "decision"}
            return {
                "status": "SUCCESS", "symbol": symbol,
                "decision": spec.get("decision", "BUY"),
                "ai_prediction": ai_prediction,
            }
        return {"status": "SUCCESS", "symbol": symbol, "decision": spec}

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


def test_one_stuck_symbol_prediction_does_not_freeze_the_whole_scan() -> None:
    """Regression: nothing previously bounded a single watchlist symbol's
    prediction call. A stalled data-fetch/AI call on one symbol would block
    asyncio.gather forever -- and since the scan runs inside
    _automate_agent_lock, that would permanently freeze !automate_agent and
    the autoscan loop, not just delay one cycle.

    Uses loop.run_until_complete directly rather than asyncio.run: asyncio.run
    cancels and awaits every remaining task (including the deliberately
    orphaned, still-sleeping stuck-symbol task) as part of its own teardown,
    which would make this test measure that unrelated cleanup wait instead of
    what actually matters -- how quickly _build_automate_agent_text itself
    returns control, which is what frees the lock for the next cycle in the
    real long-running bot process."""
    import time

    watchlist = discord_agent.config.automate_agent_watchlist
    stuck_symbol, good_symbol = watchlist[0], watchlist[1]

    def slow_predict(symbol: str, horizon_days=None):
        if symbol == stuck_symbol:
            time.sleep(5)  # far longer than the test's scan timeout below
            return {"status": "SUCCESS", "symbol": symbol, "decision": "BUY"}
        if symbol == good_symbol:
            return {
                "status": "SUCCESS", "symbol": symbol, "decision": "BUY",
                "ai_prediction": {"confidence_score": 75},
            }
        return {"status": "FAILED", "symbol": symbol, "decision": "REVIEW"}

    with TemporaryDirectory() as tmp:
        original_state = state_store.STATE_PATH
        original_alpaca = discord_agent.alpaca
        original_predict = discord_agent.run_project_prediction
        original_send = discord_agent._send_channel
        original_timeout = discord_agent.config.automate_agent_scan_timeout_seconds
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        fake = FakeAutomateAlpaca()
        fake.prices[good_symbol] = 60.0

        async def capture_send(channel_id: int, content: str = "", embed=None) -> None:
            pass

        discord_agent.alpaca = fake
        discord_agent.run_project_prediction = slow_predict
        discord_agent._send_channel = capture_send
        discord_agent._automate_agent_last_run = 0.0
        if discord_agent._automate_agent_lock.locked():
            discord_agent._automate_agent_lock.release()
        object.__setattr__(discord_agent.config, "automate_agent_scan_timeout_seconds", 1)
        loop = asyncio.new_event_loop()
        try:
            started = time.monotonic()
            loop.run_until_complete(discord_agent._build_automate_agent_text())
            elapsed = time.monotonic() - started
            assert elapsed < 4, f"scan should time out the stuck symbol quickly, took {elapsed:.1f}s"
            assert good_symbol in {p["symbol"] for p in state_store.list_positions()}, (
                "a stuck symbol must not prevent other candidates from being traded"
            )
        finally:
            # The stuck-symbol task is deliberately left running/orphaned --
            # don't wait for it, just drop the loop (matches how the real,
            # long-running bot process would behave: nothing else waits on it).
            loop.close()
            state_store.STATE_PATH = original_state
            discord_agent.alpaca = original_alpaca
            discord_agent.run_project_prediction = original_predict
            discord_agent._send_channel = original_send
            object.__setattr__(discord_agent.config, "automate_agent_scan_timeout_seconds", original_timeout)


def _inject_today_outcome(pnl_value: float, opened_by: str = AUTOMATE_AGENT_TAG) -> None:
    """Directly writes a closed-trade outcome record for today (UTC),
    bypassing the full open/close position lifecycle -- lets the circuit
    breaker test control the exact realized P&L figure it's checking
    against, rather than reverse-engineering entry/exit prices to hit one.
    """
    state = state_store.load_state()
    state.setdefault("trade_outcomes", []).append({
        "symbol": "TESTSYM",
        "pnl_value": pnl_value,
        "opened_by": opened_by,
        "closed_at": state_store._utcstamp(),
    })
    state_store.save_state(state)


def test_daily_loss_circuit_breaker_blocks_new_positions() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        fake.equity = 50_000.0
        # -3% of 50,000 is the default trip point (-1,500); simulate a
        # worse loss already realized today from automate_agent's own trades.
        _inject_today_outcome(-2000.0, opened_by=AUTOMATE_AGENT_TAG)
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0

        text = await discord_agent._build_automate_agent_text()

        assert "circuit breaker" in text.lower()
        assert not fake.submissions, "no new order should be placed once tripped"
        assert not state_store.list_positions()

    predictions = {discord_agent.config.automate_agent_watchlist[0]: "BUY"}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_daily_loss_circuit_breaker_ignores_a_real_users_losses() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        fake.equity = 50_000.0
        # A large loss on a REAL user's own trade today should never trip
        # automate_agent's breaker -- it's scoped to automate_agent's own
        # realized P&L only.
        _inject_today_outcome(-10_000.0, opened_by="")
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0

        text = await discord_agent._build_automate_agent_text()

        assert "circuit breaker" not in text.lower()
        assert len(fake.submissions) == 1, "automate_agent should still trade normally"

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_unverifiable_equity_blocks_the_cycle_instead_of_trading_blind() -> None:
    """Regression: when alpaca.get_account() fails, equity resolved to 0 and
    `if equity > 0` silently skipped the daily-loss circuit-breaker check
    entirely -- then fell back to fixed-notional sizing and traded anyway.
    That's fail-open exactly when account health can't be verified. It must
    fail-safe: skip the cycle instead."""
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        fake.account_ok = False
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0

        text = await discord_agent._build_automate_agent_text()

        assert "skipping this cycle" in text.lower()
        assert not fake.submissions, "must not trade blind when equity can't be verified"
        assert not state_store.list_positions()

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_position_sizing_scales_with_account_equity() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        fake.equity = 100_000.0  # double the 50k default

        await discord_agent._build_automate_agent_text()

        # risk_pct default 2% of 100,000 = 2,000 budget @ $100/share = 20 sh,
        # vs. 10 sh at the 50k default used elsewhere in this file.
        assert fake.submissions[0]["qty"] == "20"

    # confidence_score=75 is the exact midpoint of confidence_scaled_risk_multiplier
    # (see test_automate_agent.py), which resolves to a neutral 1.0x multiplier --
    # keeps this test isolated to equity-based scaling only, not conflated with
    # the separate confidence-based scaling covered by
    # test_position_sizing_scales_with_confidence.
    predictions = {
        discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75},
    }
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_general_agent_off_does_not_affect_automate_agent() -> None:
    """automate_agent is deliberately independent of the general agent mode
    -- agent_on/agent_off only governs whether manual/human-typed signals go
    through the prediction decision gate or straight to paper-order
    handling. Turning the general agent mode OFF must not stop
    automate_agent; only !automate_agent_off (its own switch) does that."""
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        state_store.set_agent_mode("OFF")
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        text = await discord_agent._build_automate_agent_text()
        assert "agent mode is off" not in text.lower()
        assert len(fake.submissions) == 1, "automate_agent must still trade with the general agent mode OFF"

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_automate_agent_off_blocks_it_even_when_general_agent_mode_is_on() -> None:
    """automate_agent's own switch (!automate_agent_on/!automate_agent_off)
    is independent of the general agent mode (!agent_on/!agent_off) that
    gates manual signals -- turning automate_agent off specifically must
    stop it even while the general agent mode stays ON."""
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        state_store.set_agent_mode("ON")
        state_store.set_automate_agent_mode("OFF")
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0

        text = await discord_agent._build_automate_agent_text()

        assert "automate_agent mode is off" in text.lower()
        assert not state_store.list_positions()
        assert not fake.submissions

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_automate_agent_on_by_default_runs_normally() -> None:
    """automate_agent's own switch defaults to ON, so a fresh state with no
    explicit toggle at all must still let it trade normally."""
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        assert state_store.get_automate_agent_mode() == "ON"
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0

        await discord_agent._build_automate_agent_text()

        assert len(fake.submissions) == 1

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
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

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
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

    predictions = {
        discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75},
        discord_agent.config.automate_agent_watchlist[1]: {"decision": "BUY", "confidence_score": 75},
    }
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

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def _with_extra_watchlist_symbol(test_fn) -> None:
    """max_positions defaults to the same size as the real watchlist, so
    "at the cap" tests need one extra candidate symbol beyond the cap to
    prove eviction-for-a-better-pick still works. Temporarily extends the
    watchlist by one symbol not otherwise in it, then restores it."""
    original_watchlist = discord_agent.config.automate_agent_watchlist
    extra_symbol = "ZZZTEST"
    object.__setattr__(
        discord_agent.config, "automate_agent_watchlist", original_watchlist + (extra_symbol,)
    )
    try:
        test_fn(extra_symbol)
    finally:
        object.__setattr__(discord_agent.config, "automate_agent_watchlist", original_watchlist)


def test_at_cap_evicts_oldest_automate_position_before_buying() -> None:
    def run(extra_symbol: str) -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            watchlist = discord_agent.config.automate_agent_watchlist
            max_positions = discord_agent.config.automate_agent_max_positions
            # Fill the cap with automate_agent-tagged positions, oldest first.
            for i in range(max_positions):
                sym = watchlist[i]
                state_store.upsert_position(sym, 1, 50.0, "", 1.0, 10.0, "long", AUTOMATE_AGENT_TAG, True)
                fake.quantities[sym] = 1
                fake.prices[sym] = 50.0
            fake.prices[extra_symbol] = 75.0  # not yet held

            text = await discord_agent._build_automate_agent_text()

            symbols_after = {p["symbol"] for p in state_store.list_positions()}
            assert extra_symbol in symbols_after, "the new BUY candidate was bought"
            # Regression: the eviction sell is only *submitted* here -- the
            # position must stay tracked (and protected by stop_loss_monitor)
            # until Alpaca actually confirms the fill, matching every other exit
            # path in this codebase. Deleting it immediately on mere order
            # acceptance would silently drop protection if the sell never fills.
            assert watchlist[0] in symbols_after, (
                "the evicted position must remain tracked until the exit fill is reconciled"
            )
            evict_order_id = next(
                o["id"] for o in fake.submissions if o["symbol"] == watchlist[0] and o["side"] == "sell"
            )
            fake.mark_order_filled(evict_order_id)
            await discord_agent._reconcile_pending_exit_orders()

            symbols_after_reconcile = {p["symbol"] for p in state_store.list_positions()}
            assert watchlist[0] not in symbols_after_reconcile, (
                "the oldest automate position was evicted once the sell was confirmed filled"
            )
            assert len(symbols_after_reconcile) == max_positions, "still at, not above, the cap"

        predictions = {extra_symbol: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_extra_watchlist_symbol(run)


def test_evicted_position_loss_counts_toward_the_daily_loss_circuit_breaker() -> None:
    """Regression: eviction previously called the bare remove_position,
    which never records a trade_outcomes entry -- so a real loss realized by
    evicting a losing position was invisible to
    today_realized_pnl(AUTOMATE_AGENT_TAG), the exact figure the daily-loss
    circuit breaker checks. Routing eviction through the same
    track-then-reconcile path as every other exit fixes this."""
    def run(extra_symbol: str) -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            watchlist = discord_agent.config.automate_agent_watchlist
            max_positions = discord_agent.config.automate_agent_max_positions
            for i in range(max_positions):
                sym = watchlist[i]
                state_store.upsert_position(sym, 1, 100.0, "", 1.0, 10.0, "long", AUTOMATE_AGENT_TAG, True)
                fake.quantities[sym] = 1
                fake.prices[sym] = 100.0
            # The oldest position (watchlist[0]) will be evicted at a loss.
            fake.prices[watchlist[0]] = 90.0
            fake.prices[extra_symbol] = 75.0

            await discord_agent._build_automate_agent_text()
            assert state_store.today_realized_pnl(AUTOMATE_AGENT_TAG) == 0.0, (
                "must not count the loss before the eviction sell is confirmed filled"
            )

            evict_order_id = next(
                o["id"] for o in fake.submissions if o["symbol"] == watchlist[0] and o["side"] == "sell"
            )
            fake.mark_order_filled(evict_order_id, fill_price=90.0)
            await discord_agent._reconcile_pending_exit_orders()

            assert state_store.today_realized_pnl(AUTOMATE_AGENT_TAG) == -10.0, (
                "the -$10 eviction loss (1 sh, $100 -> $90) must now count toward "
                "automate_agent's own daily-loss circuit breaker"
            )

        predictions = {extra_symbol: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_extra_watchlist_symbol(run)


def test_buy_skipped_when_a_single_share_would_exceed_the_risk_budget() -> None:
    """Regression: max(1, int(risk_budget // price)) used to force a 1-share
    buy even when that single share cost far more than the risk-based
    budget -- silently defeating fixed-fractional sizing for any stock
    pricier than the budget. It must skip instead of oversizing."""
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.equity = 1_000.0  # 2% risk_pct * confidence-neutral 1.0x = $20 budget
        fake.prices[symbol] = 50.0  # a single share already costs 2.5x the budget

        text = await discord_agent._build_automate_agent_text()

        assert not fake.submissions, "must not buy a share that costs more than the entire risk budget"
        assert not state_store.list_positions()
        assert "exceeds the" in text and "risk budget" in text

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_position_sizing_scales_with_confidence() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        watchlist = discord_agent.config.automate_agent_watchlist
        high_conf_symbol, low_conf_symbol = watchlist[0], watchlist[1]
        fake.prices[high_conf_symbol] = 100.0
        fake.prices[low_conf_symbol] = 100.0
        fake.equity = 50_000.0

        await discord_agent._build_automate_agent_text()

        submitted = {o["symbol"]: int(o["qty"]) for o in fake.submissions}
        # risk_pct=2% of 50,000 = 1,000 base budget @ $100/sh = 10 sh baseline.
        # confidence 95 -> x1.2 multiplier -> 12 sh; confidence 60 (the
        # min-confidence floor itself, so it's still eligible) -> x0.85 -> 8 sh.
        assert submitted[high_conf_symbol] == 12, submitted
        assert submitted[low_conf_symbol] == 8, submitted
        assert submitted[high_conf_symbol] > submitted[low_conf_symbol], (
            "higher-conviction pick must get a larger position, not an identical one"
        )

    predictions = {
        discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 95},
        discord_agent.config.automate_agent_watchlist[1]: {"decision": "BUY", "confidence_score": 60},
    }
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_needs_human_review_candidate_is_not_bought() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        watchlist = discord_agent.config.automate_agent_watchlist
        risky, safe = watchlist[0], watchlist[1]
        fake.prices[risky] = 50.0
        fake.prices[safe] = 60.0

        await discord_agent._build_automate_agent_text()

        symbols_bought = {p["symbol"] for p in state_store.list_positions()}
        assert risky not in symbols_bought, "needs_human_review candidate must never be auto-bought"
        assert safe in symbols_bought, "the non-flagged candidate is still bought normally"

    predictions = {
        discord_agent.config.automate_agent_watchlist[0]: {
            "decision": "BUY", "confidence_score": 95, "needs_human_review": True,
        },
        discord_agent.config.automate_agent_watchlist[1]: {
            "decision": "BUY", "confidence_score": 65, "needs_human_review": False,
        },
    }
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_buy_summary_shows_confidence_and_predicted_return() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0

        text = await discord_agent._build_automate_agent_text()

        assert "confidence 82" in text, text
        assert "predicted return +3.50%" in text, text

    predictions = {
        discord_agent.config.automate_agent_watchlist[0]: {
            "decision": "BUY", "confidence_score": 82, "predicted_return_pct": 3.5,
        },
    }
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
        await _with_runtime(scenario, predictions={wl[1]: {"decision": "BUY", "confidence_score": 75}})
    asyncio.run(_run())


def test_autoscan_task_is_a_noop_when_disabled() -> None:
    """The recurring task must not run the scan-and-trade cycle at all when
    AUTOMATE_AGENT_AUTOSCAN_ENABLED is off, even if the loop object itself
    gets ticked -- regardless of what the shipped default currently is."""
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        original = discord_agent.config.automate_agent_autoscan_enabled
        object.__setattr__(discord_agent.config, "automate_agent_autoscan_enabled", False)
        try:
            await discord_agent.automate_agent_autoscan.coro()
            assert not fake.submissions, "disabled autoscan must not place any trade"
            assert not state_store.list_positions()
        finally:
            object.__setattr__(discord_agent.config, "automate_agent_autoscan_enabled", original)

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_autoscan_task_runs_the_cycle_when_enabled() -> None:
    async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
        symbol = discord_agent.config.automate_agent_watchlist[0]
        fake.prices[symbol] = 100.0
        original = discord_agent.config.automate_agent_autoscan_enabled
        object.__setattr__(discord_agent.config, "automate_agent_autoscan_enabled", True)
        try:
            await discord_agent.automate_agent_autoscan.coro()
            assert len(fake.submissions) == 1, "enabled autoscan runs the same cycle as !automate_agent"
        finally:
            object.__setattr__(discord_agent.config, "automate_agent_autoscan_enabled", original)

    predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
    asyncio.run(_with_runtime(scenario, predictions=predictions))


def test_default_asset_mode_is_equity_only() -> None:
    # Pins the current product decision: automate_agent places equity
    # trades only unless explicitly opted into "options"/"both" -- options
    # and multi-leg support already exist (tested below) but must stay
    # dormant by default until that's revisited. If this ever flips
    # silently, automate_agent would start autonomously trading options on
    # every deployment that never asked for it.
    assert discord_agent.config.automate_agent_asset_mode == "equity"


def _with_asset_mode(mode: str, test_fn) -> None:
    original = discord_agent.config.automate_agent_asset_mode
    object.__setattr__(discord_agent.config, "automate_agent_asset_mode", mode)
    try:
        test_fn()
    finally:
        object.__setattr__(discord_agent.config, "automate_agent_asset_mode", original)


def test_options_mode_buys_an_atm_call_when_backtest_confirms_buy() -> None:
    def run() -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            symbol = discord_agent.config.automate_agent_watchlist[0]
            fake.prices[symbol] = 100.0
            today = date.today().isoformat()
            occ_symbol = f"{symbol}260101C00100000"
            fake.option_contracts_by_expiry[today] = [
                {"symbol": f"{symbol}260101C00095000", "strike_price": "95"},
                {"symbol": occ_symbol, "strike_price": "100"},  # nearest to the $100 underlying -> ATM
                {"symbol": f"{symbol}260101C00110000", "strike_price": "110"},
            ]
            fake.option_premiums[occ_symbol] = 2.0

            original_validate = discord_agent.run_options_strategy_validation
            discord_agent.run_options_strategy_validation = lambda option: {
                "status": "SUCCESS", "decision": "BUY",
            }
            try:
                await discord_agent._build_automate_agent_text()

                # The buy is only *submitted* here -- matching every other
                # option entry in this codebase, the position is created by
                # reconciliation once Alpaca confirms the fill.
                assert not state_store.list_option_positions(), (
                    "option position must not appear before the entry fill is reconciled"
                )
                entry_order_id = next(
                    o["id"] for o in fake.submissions if o["symbol"] == occ_symbol and o["side"] == "buy"
                )
                fake.mark_order_filled(entry_order_id, fill_price=2.0)
                await discord_agent._reconcile_pending_option_entry_orders()
            finally:
                discord_agent.run_options_strategy_validation = original_validate

            assert not fake.quantities.get(symbol), "options mode must not also buy shares"
            option_positions = state_store.list_option_positions()
            assert len(option_positions) == 1, option_positions
            position = option_positions[0]
            assert position["occ_symbol"] == occ_symbol
            assert position["strike"] == 100.0, "must pick the ATM strike, not the nearby OTM ones"
            assert position["opened_by"] == AUTOMATE_AGENT_TAG
            # Explicit user choice: automate_agent's options use the same
            # 1%/10% as its equity positions, not the OPTION_STOP_LOSS_PCT
            # default every other option position uses.
            assert abs(position["stop_loss"] - 2.0 * (1 - discord_agent.config.equity_stop_loss_pct / 100)) < 1e-6

        predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_asset_mode("options", run)


def test_options_mode_skips_when_backtest_does_not_confirm_buy() -> None:
    def run() -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            symbol = discord_agent.config.automate_agent_watchlist[0]
            fake.prices[symbol] = 100.0
            today = date.today().isoformat()
            fake.option_contracts_by_expiry[today] = [{"symbol": f"{symbol}260101C00100000", "strike_price": "100"}]
            fake.option_premiums[f"{symbol}260101C00100000"] = 2.0

            original_validate = discord_agent.run_options_strategy_validation
            discord_agent.run_options_strategy_validation = lambda option: {
                "status": "REVIEW", "decision": "HOLD",
            }
            try:
                await discord_agent._build_automate_agent_text()
            finally:
                discord_agent.run_options_strategy_validation = original_validate

            assert not state_store.list_option_positions(), "must never place a trade the backtest didn't confirm"
            assert not fake.submissions

        predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_asset_mode("options", run)


def test_options_mode_skips_symbol_with_no_listed_contract_within_fallback_window() -> None:
    def run() -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            symbol = discord_agent.config.automate_agent_watchlist[0]
            fake.prices[symbol] = 100.0
            # fake.option_contracts_by_expiry stays empty for every date --
            # nothing listed today or within the fallback window.
            text = await discord_agent._build_automate_agent_text()
            assert not state_store.list_option_positions()
            assert "no listed" in text.lower()

        predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_asset_mode("options", run)


def test_both_mode_shares_the_position_cap_between_equity_and_options() -> None:
    """Regression: the shared 1-10 cap must actually be enforced across both
    asset classes -- naively reusing plan.to_buy for both loops would let
    "both" mode open 2x max_positions (one equity + one option per
    candidate) instead of sharing the same ceiling."""
    def run() -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            max_positions = discord_agent.config.automate_agent_max_positions
            watchlist = discord_agent.config.automate_agent_watchlist
            # Fill the cap entirely with equity positions first.
            for i in range(max_positions):
                sym = watchlist[i]
                state_store.upsert_position(sym, 1, 50.0, "", 1.0, 10.0, "long", AUTOMATE_AGENT_TAG, True)
                fake.quantities[sym] = 1
                fake.prices[sym] = 50.0

            original_validate = discord_agent.run_options_strategy_validation
            discord_agent.run_options_strategy_validation = lambda option: {
                "status": "SUCCESS", "decision": "BUY",
            }
            try:
                await discord_agent._build_automate_agent_text()
            finally:
                discord_agent.run_options_strategy_validation = original_validate

            # At the cap with only equity's own (capped) eviction running,
            # no option positions should have been opened on top.
            assert not state_store.list_option_positions(), (
                "must not exceed the shared cap by adding options on top of an already-full equity book"
            )

        predictions = {sym: {"decision": "BUY", "confidence_score": 75} for sym in discord_agent.config.automate_agent_watchlist}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_asset_mode("both", run)


def test_options_mode_skips_duplicate_buy_while_entry_is_still_pending() -> None:
    """Regression: option entries are tracked then reconciled on confirmed
    fill (like every other option entry in this codebase), so a slow-to-fill
    order held over from a prior cycle wouldn't show up in
    list_option_positions() yet. Without an explicit pending-entry check, a
    later cycle (after the cooldown elapses) could submit a second buy for
    the same root before the first one even confirms."""
    def run() -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            symbol = discord_agent.config.automate_agent_watchlist[0]
            fake.prices[symbol] = 100.0
            today = date.today().isoformat()
            occ_symbol = f"{symbol}260101C00100000"
            fake.option_contracts_by_expiry[today] = [{"symbol": occ_symbol, "strike_price": "100"}]
            fake.option_premiums[occ_symbol] = 2.0

            original_validate = discord_agent.run_options_strategy_validation
            discord_agent.run_options_strategy_validation = lambda option: {
                "status": "SUCCESS", "decision": "BUY",
            }
            try:
                await discord_agent._build_automate_agent_text()
                assert len(fake.submissions) == 1, "first cycle should submit the option buy"

                # Simulate the next scan cycle -- same candidate, but the
                # first entry has deliberately not been reconciled yet. The
                # pending entry makes plan_automate_trades' own held_symbols
                # filter treat the root as already-held before the cycle even
                # reaches _automate_agent_buy_option's own pending check, so
                # the resulting text is "no candidates," not a per-symbol
                # skip message -- either way, the key guarantee is no second
                # submission.
                discord_agent._automate_agent_last_run = 0.0
                text = await discord_agent._build_automate_agent_text()
                assert len(fake.submissions) == 1, (
                    "must not submit a second buy for the same root while its first entry is still pending"
                )
                assert "no buy-decision candidates" in text.lower()
            finally:
                discord_agent.run_options_strategy_validation = original_validate

        predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_asset_mode("options", run)


def test_pending_option_entries_count_toward_the_shared_cap() -> None:
    """Regression: a submitted-but-not-yet-reconciled option entry from a
    prior cycle must still count against automate_agent_max_positions, or a
    slow-to-fill order could let a later cycle buy in believing there was
    more room than will actually exist once that entry confirms."""
    def run() -> None:
        async def scenario(fake: FakeAutomateAlpaca, sent: list) -> None:
            original_max = discord_agent.config.automate_agent_max_positions
            object.__setattr__(discord_agent.config, "automate_agent_max_positions", 1)
            try:
                # A pending automate_agent option entry for a symbol NOT in
                # play this cycle -- it alone should already fill the cap.
                state_store.add_pending_option_entry_order({
                    "order_id": "already-pending-1",
                    "occ_symbol": "MSFT260101C00100000",
                    "root": "MSFT",
                    "requested_qty": 1,
                    "opened_by": AUTOMATE_AGENT_TAG,
                })

                symbol = discord_agent.config.automate_agent_watchlist[0]
                fake.prices[symbol] = 100.0
                today = date.today().isoformat()
                occ_symbol = f"{symbol}260101C00100000"
                fake.option_contracts_by_expiry[today] = [{"symbol": occ_symbol, "strike_price": "100"}]
                fake.option_premiums[occ_symbol] = 2.0

                original_validate = discord_agent.run_options_strategy_validation
                discord_agent.run_options_strategy_validation = lambda option: {
                    "status": "SUCCESS", "decision": "BUY",
                }
                try:
                    await discord_agent._build_automate_agent_text()
                finally:
                    discord_agent.run_options_strategy_validation = original_validate

                assert not fake.submissions, (
                    "the cap (1) is already consumed by the pending MSFT entry alone"
                )
            finally:
                object.__setattr__(discord_agent.config, "automate_agent_max_positions", original_max)

        predictions = {discord_agent.config.automate_agent_watchlist[0]: {"decision": "BUY", "confidence_score": 75}}
        asyncio.run(_with_runtime(scenario, predictions=predictions))

    _with_asset_mode("options", run)


if __name__ == "__main__":
    test_daily_loss_circuit_breaker_blocks_new_positions()
    test_daily_loss_circuit_breaker_ignores_a_real_users_losses()
    test_position_sizing_scales_with_account_equity()
    test_agent_off_takes_no_action()
    test_cooldown_blocks_immediate_re_run()
    test_concurrent_invocations_do_not_double_fill_slots()
    test_one_symbol_erroring_during_buy_does_not_abort_the_cycle()
    test_market_closed_takes_no_action()
    test_no_buy_candidates_places_no_trades()
    test_open_market_buys_a_boom_candidate_and_tags_it()
    test_at_cap_evicts_oldest_automate_position_before_buying()
    test_position_sizing_scales_with_confidence()
    test_needs_human_review_candidate_is_not_bought()
    test_buy_summary_shows_confidence_and_predicted_return()
    test_a_failed_symbol_lookup_does_not_abort_the_whole_scan()
    print("ALL AUTOMATE_AGENT COMMAND INTEGRATION TESTS PASSED")
