"""Configuration for the Discord stock prediction agent."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent

# Load project .env first, then optional agent-local overrides.
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(AGENT_DIR / ".env", override=True)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _first_int_env(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = _int_env(name, 0)
        if value:
            return value
    return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AgentConfig:
    discord_bot_token: str = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    discord_signal_channel_id: int = _int_env("DISCORD_SIGNAL_CHANNEL_ID", 0)
    discord_review_channel_id: int = _first_int_env(
        (
            "AGENT_REVIEW_CHANNEL_ID",
            "DISCORD_AGENT_REVIEW_CHANNEL_ID",
            "SIGNAL_REVIEW_CHANNEL_ID",
            "DISCORD_SIGNAL_REVIEW_CHANNEL_ID",
            "DISCORD_REVIEW_CHANNEL_ID",
        ),
        0,
    )
    discord_paper_log_channel_id: int = _int_env("DISCORD_PAPER_LOG_CHANNEL_ID", 0)

    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "").strip()
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "").strip()
    alpaca_base_url: str = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    ).rstrip("/")
    alpaca_data_base_url: str = os.getenv(
        "ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"
    ).rstrip("/")
    alpaca_request_timeout_seconds: int = _int_env("ALPACA_REQUEST_TIMEOUT_SECONDS", 20)
    alpaca_max_concurrent_requests: int = _int_env("ALPACA_MAX_CONCURRENT_REQUESTS", 8)
    alpaca_max_request_attempts: int = _int_env("ALPACA_MAX_REQUEST_ATTEMPTS", 3)
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "").strip()
    polygon_base_url: str = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/")
    polygon_timeout_seconds: int = _int_env("POLYGON_TIMEOUT_SECONDS", 20)

    paper_trading_enabled: bool = _bool_env("PAPER_TRADING_ENABLED", True)
    # Dedicated protection settings avoid inheriting the old 0.5% stop from
    # AGENT_STOP_LOSS_PCT in existing deployments.
    equity_stop_loss_pct: float = _float_env("EQUITY_STOP_LOSS_PCT", 1.0)
    equity_take_profit_pct: float = _float_env("EQUITY_TAKE_PROFIT_PCT", 10.0)
    option_stop_loss_pct: float = _float_env("OPTION_STOP_LOSS_PCT", 5.0)
    option_take_profit_pct: float = _float_env("OPTION_TAKE_PROFIT_PCT", 10.0)
    stop_loss_pct: float = equity_stop_loss_pct  # compatibility alias
    stop_monitor_seconds: int = _int_env("PROTECTION_MONITOR_SECONDS", 15)
    # refresh_symbol_cache_from_alpaca only actually re-fetches once the
    # cache turns 24h stale (its own internal gate) -- this just controls how
    # often we check that gate. Previously the only check was at process
    # startup, so a long-lived run (the recommended deployment pattern) would
    # drift stale for days between restarts.
    symbol_cache_refresh_interval_seconds: int = _int_env(
        "SYMBOL_CACHE_REFRESH_INTERVAL_SECONDS", 3600
    )
    max_equity_qty: float = _float_env("MAX_EQUITY_QTY", 1_000_000.0)
    max_option_qty: float = _float_env("MAX_OPTION_QTY", 1_000.0)
    # 0 (the default) means unlimited -- paper trading has no real capital at
    # risk, so this throttle only exists for whoever explicitly opts into it
    # via MAX_DAILY_PAPER_TRADES in .env.
    max_daily_paper_trades: int = _int_env("MAX_DAILY_PAPER_TRADES", 0)
    allow_duplicate_paper_orders: bool = _bool_env("ALLOW_DUPLICATE_PAPER_ORDERS", True)
    per_symbol_cooldown_minutes: int = _int_env("PER_SYMBOL_COOLDOWN_MINUTES", 10)
    debug_output_enabled: bool = _bool_env("DEBUG_OUTPUT_ENABLED", False)
    signal_worker_concurrency: int = _int_env("SIGNAL_WORKER_CONCURRENCY", 4)
    signal_queue_limit: int = _int_env("SIGNAL_QUEUE_LIMIT", 20_000)
    signal_queue_poll_seconds: float = _float_env("SIGNAL_QUEUE_POLL_SECONDS", 0.5)
    signal_max_attempts: int = _int_env("SIGNAL_MAX_ATTEMPTS", 3)
    signal_retry_base_seconds: int = _int_env("SIGNAL_RETRY_BASE_SECONDS", 5)
    signal_claim_timeout_seconds: int = _int_env("SIGNAL_CLAIM_TIMEOUT_SECONDS", 600)
    pending_order_batch_size: int = _int_env("PENDING_ORDER_BATCH_SIZE", 100)
    # A queued order (waiting for the market to open) is dropped instead of
    # retried again once either threshold is crossed -- without this, an
    # order that can never legitimately succeed (e.g. the position it was
    # meant to sell no longer exists) retries silently forever.
    pending_order_max_attempts: int = _int_env("PENDING_ORDER_MAX_ATTEMPTS", 20)
    pending_order_max_age_hours: int = _int_env("PENDING_ORDER_MAX_AGE_HOURS", 48)
    runtime_log_level: str = os.getenv("RUNTIME_LOG_LEVEL", "INFO").strip().upper()
    runtime_log_max_bytes: int = _int_env("RUNTIME_LOG_MAX_BYTES", 5_000_000)
    runtime_log_backup_count: int = _int_env("RUNTIME_LOG_BACKUP_COUNT", 5)

    # !automate_agent -- autonomous intraday scan-and-trade. Reuses the
    # existing equity stop-loss/take-profit config above rather than a
    # separate knob, since both default to the same 1%/10% already.
    automate_agent_min_positions: int = _int_env("AUTOMATE_AGENT_MIN_POSITIONS", 1)
    automate_agent_max_positions: int = _int_env("AUTOMATE_AGENT_MAX_POSITIONS", 20)
    # Compulsory floor on single-leg TSLA *options* trades specifically
    # (not just "ambition" like automate_agent_min_positions above, and
    # not equity -- an equity buy never counts toward this): if fewer than
    # this many real options BUY trades have been placed by the time only
    # automate_agent_min_trades_relax_minutes remain before the daily exit
    # cutoff, the confidence bar is dropped to 0 for that cycle's
    # candidate ranking so the quota can still be met -- the model's own
    # BUY/SELL/HOLD call and needs_human_review flag are still respected
    # even then; only how *confident* it needed to be is relaxed. The
    # daily-loss circuit breaker is never overridden by this -- capital
    # protection always wins over a trade-count quota.
    automate_agent_min_options_trades_per_window: int = _int_env(
        "AUTOMATE_AGENT_MIN_OPTIONS_TRADES_PER_WINDOW", 5
    )
    # Compulsory floor on *total* orders (equity + option buys combined) --
    # same relax mechanism as the options-specific floor above (either one
    # being unmet triggers the confidence-bar relax), just not scoped to
    # one asset type. An equity buy counts toward this one; it doesn't
    # count toward automate_agent_min_options_trades_per_window.
    automate_agent_min_total_trades_per_window: int = _int_env(
        "AUTOMATE_AGENT_MIN_TOTAL_TRADES_PER_WINDOW", 10
    )
    # Hard ceiling on the flip side -- total orders (equity + option buys
    # combined) placed during the whole trading window, regardless of how
    # much position-cap headroom eviction churn might otherwise free up.
    automate_agent_max_trades_per_window: int = _int_env(
        "AUTOMATE_AGENT_MAX_TRADES_PER_WINDOW", 40
    )
    automate_agent_min_trades_relax_minutes: int = _int_env(
        "AUTOMATE_AGENT_MIN_TRADES_RELAX_MINUTES", 60
    )
    # A narrower window inside the relax window above, for the options
    # path only: once this close to the cutoff and the compulsory minimum
    # is still unmet, automate_agent buys the best available option
    # candidate even if the real historical backtest
    # (run_options_strategy_validation) didn't confirm BUY for it.
    # Relaxing the confidence bar alone (see above) was found to be
    # insufficient on 2026-08-31 -- a real candidate cleared the relaxed
    # bar but the backtest gate still vetoed it, and nothing forced a
    # trade past that veto, so the window closed at 0 trades despite the
    # quota. Per explicit direction: the compulsory minimum should win
    # over the backtest gate once genuinely out of time, at the cost of
    # sometimes taking a trade the backtest itself disagreed with. Equity
    # has no equivalent backtest gate in automate_agent, so this only
    # changes the options path.
    automate_agent_force_trade_minutes: int = _int_env(
        "AUTOMATE_AGENT_FORCE_TRADE_MINUTES", 15
    )
    # Caps how many existing positions can be swapped out in a single scan
    # cycle, even if enough higher-ranked fresh candidates exist to justify
    # more -- keeps portfolio churn gradual instead of flipping the whole
    # book at once.
    automate_agent_max_evictions_per_cycle: int = _int_env(
        "AUTOMATE_AGENT_MAX_EVICTIONS_PER_CYCLE", 1
    )
    automate_agent_exit_minutes_before_close: int = _int_env(
        "AUTOMATE_AGENT_EXIT_MINUTES_BEFORE_CLOSE", 15
    )
    # automate_agent's own trading window is intentionally shorter than the
    # full session: positions are force-closed at this fixed US Eastern
    # wall-clock time (HH:MM, 24h) rather than relative to market close, so
    # a manual "activate around the open" workflow has a predictable, fixed
    # cutoff every day regardless of early-close days. Equity positions not
    # opened by automate_agent are unaffected -- they still use the
    # relative automate_agent_exit_minutes_before_close fallback above.
    automate_agent_exit_time_et: str = os.getenv("AUTOMATE_AGENT_EXIT_TIME_ET", "12:00").strip()
    # On-demand afternoon window: if a human runs !automate_agent (not the
    # background autoscan loop -- see manual_trigger in
    # _build_automate_agent_text) after the fixed morning cutoff above has
    # already passed but the market is still open, that invocation starts
    # a brand-new bounded trading window right then instead of doing
    # nothing for the rest of the day. The window's cutoff is whichever
    # comes first: automate_agent_on_demand_window_minutes after the
    # invocation, or automate_agent_on_demand_close_buffer_minutes before
    # the real market close (from Alpaca's own clock, so early-close days
    # are handled correctly without a hardcoded "16:00"). Once started,
    # the autoscan loop picks the window up automatically for its
    # duration -- only *starting* a new window requires a manual trigger.
    automate_agent_on_demand_window_minutes: int = _int_env(
        "AUTOMATE_AGENT_ON_DEMAND_WINDOW_MINUTES", 150
    )
    automate_agent_on_demand_close_buffer_minutes: int = _int_env(
        "AUTOMATE_AGENT_ON_DEMAND_CLOSE_BUFFER_MINUTES", 5
    )
    # Fixed-fractional position sizing: risk a small, constant % of current
    # account equity per trade rather than a fixed dollar amount, so sizing
    # naturally scales with account growth and shrinks during drawdowns.
    # 0.5-2% is the widely-cited professional ceiling for this.
    automate_agent_risk_pct_per_trade: float = _float_env("AUTOMATE_AGENT_RISK_PCT_PER_TRADE", 2.0)
    # Account-level circuit breaker, independent of any single position's
    # stop-loss: if automate_agent's own cumulative realized P&L for the
    # current day drops below this % of equity, it stops opening new
    # positions for the rest of the day. A per-trade stop limits one
    # position; this limits the whole autonomous strategy in a single bad
    # session. 2-5% is the commonly cited range for this kind of breaker.
    automate_agent_max_daily_loss_pct: float = _float_env("AUTOMATE_AGENT_MAX_DAILY_LOSS_PCT", 3.0)
    # Each cycle runs the prediction engine once per watchlist symbol
    # (real historical-data fetch + AI call each time) -- a cooldown keeps
    # rapid re-triggering from burning API calls/rate limits for no benefit,
    # since the market doesn't meaningfully change signal in a few seconds.
    automate_agent_cooldown_seconds: int = _int_env("AUTOMATE_AGENT_COOLDOWN_SECONDS", 60)
    # A per-symbol ceiling on the prediction call during a scan. Without
    # this, a single watchlist symbol whose data-fetch or AI call stalls
    # (e.g. a hung network read) would block asyncio.gather forever --
    # and since the scan runs inside _automate_agent_lock, that would
    # silently freeze !automate_agent and the autoscan loop permanently,
    # not just delay one cycle.
    # Sized for the current narrow, deliberately-chosen watchlist (TSLA
    # only by default) -- each symbol is a real historical-data fetch plus
    # a real AI prediction call, run concurrently. A wider watchlist (e.g.
    # the S&P 500, still available via AUTOMATE_AGENT_WATCHLIST) needs a
    # proportionally larger value here, since every symbol shares the same
    # bounded thread pool.
    #
    # The original, measured bottleneck was historical_price_service.py's
    # fetch_price_history(): it cascaded through ~25 sequential RapidAPI
    # endpoint/exchange/path combinations (10 and 6 second timeouts each)
    # before falling back to an external (NASDAQ) provider -- and that
    # fallback was the only one that ever actually worked on the current
    # RapidAPI subscription ("historical endpoints not available on your
    # current plan"). Directly timed on 2026-08-31: fetch_price_history
    # ('TSLA') took 263s on its own, before Gemini is even called; live
    # runs on 2026-09-01 hit this again at 400s+, twice.
    #
    # Root-caused and fixed on 2026-09-03: HISTORICAL_PRICE_PROVIDER=
    # external_historical is now set in the project .env, so
    # fetch_price_history skips the doomed RapidAPI cascade entirely and
    # goes straight to the NASDAQ provider that always ends up serving the
    # data anyway. Re-measured after the fix: fetch_price_history('TSLA')
    # now takes 2-17s (was 263-400s+), and the full per-symbol pipeline
    # (price fetch + Gemini prediction call) takes 91-177s across three
    # back-to-back runs -- the AI call itself, not price data, is now the
    # dominant and only remaining source of latency here, and that's real
    # work rather than a doomed network cascade.
    #
    # A previous attempt to tighten this to 120s (reasoning "one symbol
    # should be fast", without measuring) meant every cycle timed out
    # before the old price-fetch cascade could ever finish, so
    # automate_agent placed zero trades for an entire trading day -- kept
    # here as a cautionary note against guessing at this number again.
    # This value is set with ~1.7x margin over the observed 177s worst
    # case for the AI call.
    automate_agent_scan_timeout_seconds: int = _int_env(
        "AUTOMATE_AGENT_SCAN_TIMEOUT_SECONDS", 300
    )
    # A ceiling on the whole per-symbol *buy* attempt (equity or options),
    # separate from automate_agent_scan_timeout_seconds above, which only
    # bounds the watchlist scan. Every individual call inside a buy
    # attempt already has its own timeout (Alpaca requests, the backtest
    # validation gate), but nothing previously bounded the buy attempt as
    # a whole -- the options path in particular chains several such calls
    # (a live quote, up to automate_agent_option_expiry_fallback_days+1
    # contract-listing calls, several concurrent premium fetches, the
    # backtest gate), and a run of individually-bounded-but-slow calls
    # could still add up to several minutes with nothing stopping it,
    # holding _automate_agent_lock the whole time. Same class of problem
    # as the scan timeout that caused 2026-08-31's zero-trade day, just
    # one step later in the pipeline -- closing it before it's ever
    # actually been hit live, rather than after.
    automate_agent_buy_timeout_seconds: int = _int_env(
        "AUTOMATE_AGENT_BUY_TIMEOUT_SECONDS", 120
    )
    # "equity" | "options" | "both" (default). Controls whether automate_agent's
    # autonomous buys are shares, single-leg options, or both asset classes
    # competing for the same position cap. Per direction from the user's
    # senior: normal TSLA share trades AND single-leg options (calls on a
    # BUY signal, puts on a SELL signal) side by side, on a narrow,
    # deliberately-chosen symbol list rather than scanning the whole S&P
    # 500 -- equity-only/options-only remain available via this same
    # setting for anyone who wants a narrower mode.
    automate_agent_asset_mode: str = os.getenv("AUTOMATE_AGENT_ASSET_MODE", "both").strip().lower()
    # How many calendar days forward to search for a listed option expiry
    # when a watchlist symbol has no same-day (0DTE) contracts listed.
    automate_agent_option_expiry_fallback_days: int = _int_env(
        "AUTOMATE_AGENT_OPTION_EXPIRY_FALLBACK_DAYS", 5
    )
    # automate_agent's own option positions no longer mirror the equity 1%/
    # 10% -- premium moves far more than the underlying, so the senior's
    # direction is 20% stop-loss / 25% take-profit measured on the premium
    # itself (e.g. bought at $2.00: stop at $1.60, target at $2.50).
    automate_agent_option_stop_loss_pct: float = _float_env("AUTOMATE_AGENT_OPTION_STOP_LOSS_PCT", 20.0)
    automate_agent_option_take_profit_pct: float = _float_env("AUTOMATE_AGENT_OPTION_TAKE_PROFIT_PCT", 25.0)
    # How many listed strikes nearest the current price to fetch premiums
    # for and score (automate_agent.select_best_strike) when choosing which
    # strike to buy, instead of only ever taking the nearest-the-money one.
    # Deliberately small: each candidate costs one extra live-quote call,
    # which is fine for a narrow, single-symbol watchlist but would not
    # have been for the prior full S&P 500 watchlist.
    automate_agent_strike_candidates: int = _int_env("AUTOMATE_AGENT_STRIKE_CANDIDATES", 5)
    # A strike that scores well on the expected-payoff heuristic but is
    # illiquid (a wide bid/ask spread relative to its mid-price) is likely
    # to fill far worse than the quoted mid -- exactly what the ranking is
    # trying to estimate accurately. 15% is a commonly-cited retail
    # threshold for "still reasonably tradable"; a contract with no bid/
    # ask data available at all is never dropped by this (unknown isn't
    # the same as bad -- see automate_agent.rank_strikes).
    automate_agent_max_spread_pct: float = _float_env("AUTOMATE_AGENT_MAX_SPREAD_PCT", 15.0)
    # Of those ranked strikes, how many of the top ones (best expected
    # payoff first) actually get run through the real backtest validation
    # gate (run_options_strategy_validation) before giving up. The payoff
    # ranking is still just a heuristic -- the real backtest can
    # legitimately disagree with its #1 pick; offering it #2 as well
    # means that disagreement doesn't have to end the cycle with no trade.
    # Run concurrently, not sequentially, so this doesn't multiply cycle
    # latency by the count. Kept small (default 2): each one is a real
    # network call to a real backtest/historical-data service, and this
    # project just learned the hard way (2026-08-31) how expensive an
    # unbounded external-API fallback chain can get.
    automate_agent_backtest_candidates: int = _int_env("AUTOMATE_AGENT_BACKTEST_CANDIDATES", 2)
    # A BUY the model itself didn't flag as needing review can still be a
    # thin, barely-cleared-the-bar call. This drops anything below the
    # threshold from consideration entirely, on top of the existing
    # needs_human_review exclusion.
    automate_agent_min_confidence: float = _float_env("AUTOMATE_AGENT_MIN_CONFIDENCE", 60.0)
    # On by default: the whole point of automate_agent is continuous,
    # no-human-intervention monitoring during market hours, not a one-shot
    # scan that only runs when someone happens to retype the command. Still
    # gated by every existing safety check (agent mode ON, market open,
    # per-cycle cooldown, daily-loss circuit breaker, min-confidence filter).
    automate_agent_autoscan_enabled: bool = _bool_env("AUTOMATE_AGENT_AUTOSCAN_ENABLED", True)
    automate_agent_autoscan_interval_seconds: int = _int_env(
        "AUTOMATE_AGENT_AUTOSCAN_INTERVAL_SECONDS", 900
    )
    # Narrowed to TSLA per direction from the user's senior (SPX was
    # considered but ruled out: Alpaca has no historical price data or live
    # underlying quote for the raw index, which the prediction engine and
    # ATM strike selection both depend on). The previous full S&P 500 list
    # (503 tickers, sourced from https://github.com/datasets/s-and-p-500-companies
    # and cross-checked against Alpaca's own symbol cache) is still available
    # by setting the env var back to that comma-separated list.
    automate_agent_watchlist: tuple[str, ...] = tuple(
        s.strip().upper()
        for s in os.getenv("AUTOMATE_AGENT_WATCHLIST", "TSLA").split(",")
        if s.strip()
    )

    whatsapp_webhook_enabled: bool = _bool_env("WHATSAPP_WEBHOOK_ENABLED", False)
    whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
    whatsapp_access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    whatsapp_phone_number_id: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    whatsapp_app_secret: str = os.getenv("WHATSAPP_APP_SECRET", "").strip()
    whatsapp_graph_api_version: str = os.getenv(
        "WHATSAPP_GRAPH_API_VERSION", "v26.0"
    ).strip()
    whatsapp_host: str = os.getenv("WHATSAPP_HOST", "0.0.0.0").strip()
    whatsapp_port: int = _int_env("WHATSAPP_PORT", 5000)
    whatsapp_max_body_bytes: int = _int_env("WHATSAPP_MAX_BODY_BYTES", 1_000_000)
    whatsapp_allowed_sender_ids: str = os.getenv(
        "WHATSAPP_ALLOWED_SENDER_IDS", ""
    ).strip()
    whatsapp_allowed_group_ids: str = os.getenv(
        "WHATSAPP_ALLOWED_GROUP_IDS", ""
    ).strip()
    # Separate, narrower allowlist for mode-changing/admin commands (!agent_on,
    # !agent_off, !agent_retry_dead) over WhatsApp -- mirrors the Discord
    # Administrator/Manage Server gate, since WhatsApp has no guild-permission
    # concept. Being in whatsapp_allowed_sender_ids only grants signal access.
    whatsapp_admin_sender_ids: str = os.getenv(
        "WHATSAPP_ADMIN_SENDER_IDS", ""
    ).strip()
    # Where proactive/background alerts go (protection triggers, a queued
    # order finally filling, a contract becoming tradable, ...). These fire
    # from the periodic monitor loop, not in reply to an incoming message, so
    # there's no message object to derive a WhatsApp destination from the way
    # a normal reply does -- this is the WhatsApp equivalent of
    # DISCORD_PAPER_LOG_CHANNEL_ID / AGENT_REVIEW_CHANNEL_ID. Phone number for
    # an individual, or the group ID for the signals group.
    whatsapp_alert_target: str = os.getenv("WHATSAPP_ALERT_TARGET", "").strip()
    whatsapp_alert_is_group: bool = _bool_env("WHATSAPP_ALERT_IS_GROUP", True)

    default_horizon_days: int = _int_env("DEFAULT_PREDICTION_HORIZON_DAYS", 1)
    historical_context_days: int = _int_env("HISTORICAL_CONTEXT_DAYS", 365)
    initial_capital: float = _float_env("DEFAULT_INITIAL_CAPITAL", 50000.0)
    benchmark: str = os.getenv("DEFAULT_BENCHMARK", "SPY").strip().upper()
    price_basis: str = os.getenv("DEFAULT_PRICE_BASIS", "close").strip().lower()

    buy_min_return_pct: float = _float_env("BUY_MIN_RETURN_PCT", 0.01)
    buy_strong_return_pct: float = _float_env("BUY_STRONG_RETURN_PCT", 1.0)
    buy_excellent_confidence: float = _float_env("BUY_EXCELLENT_CONFIDENCE", 80.0)
    buy_low_risk: float = _float_env("BUY_LOW_RISK", 40.0)
    buy_decision_score: float = _float_env("BUY_DECISION_SCORE", 45.0)

    sell_min_return_pct: float = _float_env("SELL_MIN_RETURN_PCT", -0.01)
    sell_strong_return_pct: float = _float_env("SELL_STRONG_RETURN_PCT", -1.0)
    sell_low_confidence: float = _float_env("SELL_LOW_CONFIDENCE", 50.0)
    sell_high_risk: float = _float_env("SELL_HIGH_RISK", 60.0)
    sell_decision_score: float = _float_env("SELL_DECISION_SCORE", 45.0)

    agent_name: str = os.getenv("AGENT_NAME", "AI Stock Prediction Agent").strip()

    default_option_qty: float = _float_env("DEFAULT_OPTION_QTY", 1.0)
    default_option_order_type: str = os.getenv("DEFAULT_OPTION_ORDER_TYPE", "auto").strip().lower()
    options_trading_enabled_override: bool = _bool_env("OPTIONS_TRADING_ENABLED", True)
    default_option_strategy_horizon_days: int = _int_env("DEFAULT_OPTION_STRATEGY_HORIZON_DAYS", 1)
    option_backtest_lookback_days: int = _int_env("OPTION_BACKTEST_LOOKBACK_DAYS", 365)
    option_exact_strike_recent_retry_days: int = _int_env("OPTION_EXACT_STRIKE_RECENT_RETRY_DAYS", 45)
    default_option_strategy_delta: int = _int_env("DEFAULT_OPTION_STRATEGY_DELTA", 30)
    default_option_strategy_dte: int = _int_env("DEFAULT_OPTION_STRATEGY_DTE", 1)
    default_option_entry_frequency: str = os.getenv("DEFAULT_OPTION_ENTRY_FREQUENCY", "every day").strip()
    default_option_exit_rule: str = os.getenv("DEFAULT_OPTION_EXIT_RULE", "Exit at target date").strip()
    option_allow_unvalidated_fallback: bool = _bool_env("OPTION_ALLOW_UNVALIDATED_FALLBACK", False)
    option_exact_strike_backtest_mode: str = os.getenv(
        "OPTION_EXACT_STRIKE_BACKTEST_MODE", "exact_first"
    ).strip().lower()
    option_strike_validation_provider: str = os.getenv(
        "OPTION_STRIKE_VALIDATION_PROVIDER", "polygon_first"
    ).strip().lower()
    option_validation_cache_ttl_hours: int = _int_env("OPTION_VALIDATION_CACHE_TTL_HOURS", 12)
    option_validation_timeout_seconds: int = _int_env("OPTION_VALIDATION_TIMEOUT_SECONDS", 35)
    suppress_discord_reconnect_tracebacks: bool = _bool_env("SUPPRESS_DISCORD_RECONNECT_TRACEBACKS", True)
    learning_enabled: bool = _bool_env("LEARNING_ENABLED", True)
    learning_min_samples: int = _int_env("LEARNING_MIN_SAMPLES", 5)
    learning_max_score_adjustment: float = _float_env("LEARNING_MAX_SCORE_ADJUSTMENT", 8.0)
    min_option_signal_quality: float = _float_env("MIN_OPTION_SIGNAL_QUALITY", 60.0)
    min_option_risk_reward: float = _float_env("MIN_OPTION_RISK_REWARD", 1.2)
    max_option_dte: int = _int_env("MAX_OPTION_DTE", 120)

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_bot_token)

    @property
    def uses_paper_alpaca_endpoint(self) -> bool:
        return "paper-api.alpaca.markets" in self.alpaca_base_url.lower()

    @property
    def has_whatsapp(self) -> bool:
        return bool(
            self.whatsapp_verify_token
            and self.whatsapp_access_token
            and self.whatsapp_phone_number_id
            and self.whatsapp_app_secret
        )

    def __post_init__(self) -> None:
        # A misconfigured pair here (e.g. AUTOMATE_AGENT_MAX_TRADES_PER_
        # WINDOW set below either minimum) would make that compulsory
        # floor permanently unreachable: the max-trades gate in
        # discord_agent.py stops ALL new orders (equity and options alike)
        # for the rest of the window as soon as it's hit, before the
        # relax logic ever gets a chance to reach its own target -- both
        # minimums are a subset of (or equal to) total trades, so the
        # ceiling must be at least as high as each floor, and the options
        # floor can never exceed the total floor either. Fail loudly at
        # startup rather than silently running a window that can never
        # place its required minimum.
        if self.automate_agent_max_trades_per_window < self.automate_agent_min_options_trades_per_window:
            raise ValueError(
                "AUTOMATE_AGENT_MAX_TRADES_PER_WINDOW "
                f"({self.automate_agent_max_trades_per_window}) must be >= "
                "AUTOMATE_AGENT_MIN_OPTIONS_TRADES_PER_WINDOW "
                f"({self.automate_agent_min_options_trades_per_window})."
            )
        if self.automate_agent_max_trades_per_window < self.automate_agent_min_total_trades_per_window:
            raise ValueError(
                "AUTOMATE_AGENT_MAX_TRADES_PER_WINDOW "
                f"({self.automate_agent_max_trades_per_window}) must be >= "
                "AUTOMATE_AGENT_MIN_TOTAL_TRADES_PER_WINDOW "
                f"({self.automate_agent_min_total_trades_per_window})."
            )
        if self.automate_agent_min_options_trades_per_window > self.automate_agent_min_total_trades_per_window:
            raise ValueError(
                "AUTOMATE_AGENT_MIN_OPTIONS_TRADES_PER_WINDOW "
                f"({self.automate_agent_min_options_trades_per_window}) must be <= "
                "AUTOMATE_AGENT_MIN_TOTAL_TRADES_PER_WINDOW "
                f"({self.automate_agent_min_total_trades_per_window}) -- options trades are "
                "a subset of total trades."
            )
        # The force-past-backtest window only makes sense as the tail end
        # of the confidence-relax window, not wider than it -- forcing a
        # trade before the confidence bar has even relaxed would be a more
        # aggressive override than intended.
        if self.automate_agent_force_trade_minutes > self.automate_agent_min_trades_relax_minutes:
            raise ValueError(
                "AUTOMATE_AGENT_FORCE_TRADE_MINUTES "
                f"({self.automate_agent_force_trade_minutes}) must be <= "
                "AUTOMATE_AGENT_MIN_TRADES_RELAX_MINUTES "
                f"({self.automate_agent_min_trades_relax_minutes})."
            )


config = AgentConfig()


def production_config_errors() -> list[str]:
    errors: list[str] = []
    if not config.has_discord:
        errors.append("DISCORD_BOT_TOKEN is missing")
    if config.discord_signal_channel_id <= 0:
        errors.append("DISCORD_SIGNAL_CHANNEL_ID is missing or invalid")
    if config.discord_review_channel_id <= 0:
        errors.append("AGENT_REVIEW_CHANNEL_ID is missing or invalid")
    if config.paper_trading_enabled:
        if not config.has_alpaca:
            errors.append("Alpaca paper API key/secret are missing")
        if not config.uses_paper_alpaca_endpoint:
            errors.append(
                "ALPACA_BASE_URL must use https://paper-api.alpaca.markets"
            )
    if not 1 <= config.signal_worker_concurrency <= 16:
        errors.append("SIGNAL_WORKER_CONCURRENCY must be between 1 and 16")
    if config.signal_queue_limit < 100:
        errors.append("SIGNAL_QUEUE_LIMIT must be at least 100")
    if config.stop_monitor_seconds < 15:
        errors.append("PROTECTION_MONITOR_SECONDS must be at least 15")
    if not 1 <= config.pending_order_batch_size <= 1_000:
        errors.append("PENDING_ORDER_BATCH_SIZE must be between 1 and 1000")
    if config.pending_order_max_attempts < 1:
        errors.append("PENDING_ORDER_MAX_ATTEMPTS must be at least 1")
    if config.pending_order_max_age_hours < 1:
        errors.append("PENDING_ORDER_MAX_AGE_HOURS must be at least 1")
    if config.whatsapp_webhook_enabled:
        if not config.has_whatsapp:
            errors.append(
                "WhatsApp webhook is enabled but verify token, access token, "
                "phone number ID, or app secret is missing"
            )
        if not 1 <= config.whatsapp_port <= 65535:
            errors.append("WHATSAPP_PORT must be between 1 and 65535")
    return errors
