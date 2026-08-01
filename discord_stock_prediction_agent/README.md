# Discord Stock Prediction Agent

Production-oriented Discord signal ingestion and Alpaca paper-trading layer for the
Stock-Prediction-Agent project. The bot accepts equity, single-leg option, and
supported multi-leg option signals; validates them with the correct project engine;
queues work durably; and reports every result in a separate Discord review channel.

> Paper trading only. The checked configuration refuses a non-paper Alpaca trading
> endpoint. This project is an experimental decision-support system, not a promise of
> profit or financial advice.

## System Overview

```text
Discord #stock-signals
        |
        v
Durable SQLite signal queue
        |
        v
Parser and symbol resolver
        |
        +-- Equity ------> Stock Price Validation
        |
        +-- Options -----> Options Strategy Validation
        |                    + exact strike / Polygon
        |                    + delta / Tastytrade
        |                    + Alpaca contract verification
        |
        v
Agent decision gate (ON) or direct signal routing (OFF)
        |
        v
Alpaca paper-order checks and submission
        |
        v
Position, pending-order, SL/TP, and 0.5% equity-stop monitors
        |
        v
Discord #agent-review
```

## What The Agent Handles

- Equity signals: BUY, SELL, HOLD, quantity, market, limit, and conditional price rules.
- Company names and ticker symbols, such as `apple`/`AAPL` and `microsoft`/`MSFT`.
- Single-leg options: BTO, STC, STO, BTC, CALL, PUT, CE, PE, strike, expiry, quantity,
  market/limit premium, stop loss, targets, risk/reward, and trailing stops.
- Multi-leg strategies: spreads, straddles, strangles, butterflies, iron condors,
  calendars, diagonals, ratios, collars, covered calls, protective puts, and rolls when
  the parsed legs and Alpaca multi-leg requirements are complete.
- Market-closed orders, unavailable contracts, unmet price conditions, and missing
  positions through persistent monitoring queues.
- Bulk traffic through concurrent workers, retry/backoff, stale-claim recovery, and a
  dead-letter state for signals that exhaust all attempts.
- Learning records for parser reliability, decision patterns, option-validation paths,
  and closed paper-trade outcomes.

## Validation Routing

The Discord agent always uses the Stock-Prediction-Agent project:

| Signal | Project mode | Main result |
|---|---|---|
| Normal stock/equity | Stock Price Validation | BUY, SELL, or HOLD |
| Options | Options Strategy Validation | Approved BUY/SELL or review/reject |

Exact-strike option signals use the requested strike and expiry. With
`OPTION_STRIKE_VALIDATION_PROVIDER=polygon_first`, Polygon exact-contract history is
tried first and the existing strategy service remains available where useful. Explicit
delta signals use the delta/DTE strategy path.

The order contract is always resolved separately through Alpaca. Historical validation
does not silently replace the requested strike in the actual paper order.

## Agent ON And OFF

The decision-making layer can be changed while the bot is running:

```text
!agent_on
!agent_off
!agent_mode
```

- **Agent ON:** current behavior. Equity signals use Stock Price Validation and options
  use Options Strategy Validation before paper-order handling.
- **Agent OFF:** every valid BUY/SELL signal proceeds directly to paper-order handling.
  Parsing, contract lookup, market-hours, position, quantity, price-condition, buying
  power, paper-endpoint, and broker safeguards remain active.
- HOLD signals never create orders in either mode.
- Only the bot owner or a member with Discord **Administrator** or **Manage Server**
  permission can change the mode.
- The selected mode is persisted in
  `discord_stock_prediction_agent/agent_state.json` and survives restarts.

Useful status commands:

```text
!agent_status
!agent_positions
!agent_option_positions
!agent_summary
!agent_learning
!agent_option_validation
```

## Discord Channels And IDs

Create two text channels:

| Suggested channel | Purpose | Environment variable |
|---|---|---|
| `#stock-signals` | Raw user input only | `DISCORD_SIGNAL_CHANNEL_ID` |
| `#agent-review` | Reviews, invalid input, decisions, queue/order updates | `AGENT_REVIEW_CHANNEL_ID` |

Optional:

| Channel | Purpose | Environment variable |
|---|---|---|
| `#paper-trade-log` | Separate broker/order and protection messages | `DISCORD_PAPER_LOG_CHANNEL_ID` |

Channel IDs are deployment-specific and must not be hard-coded into source control.
To obtain an ID:

1. Discord **User Settings > Advanced > Developer Mode**: ON.
2. Right-click the channel.
3. Select **Copy Channel ID**.
4. Paste only the numeric ID into the local `.env` file.

The review channel also accepts these legacy variable names, in this priority order:

```text
AGENT_REVIEW_CHANNEL_ID
DISCORD_AGENT_REVIEW_CHANNEL_ID
SIGNAL_REVIEW_CHANNEL_ID
DISCORD_SIGNAL_REVIEW_CHANNEL_ID
DISCORD_REVIEW_CHANNEL_ID
```

## Discord Bot Setup

1. Open the Discord Developer Portal and create an application.
2. Add a bot user.
3. Enable **Message Content Intent**.
4. Invite the bot to the server with View Channels, Send Messages, Embed Links, Read
   Message History, Add Reactions, and Use Application Commands permissions.
5. Give the bot access to `#stock-signals` and `#agent-review`.
6. Put the bot token only in the local `.env`; never commit or post it.

## Environment Configuration

The loader reads the project-root `.env`, followed by
`discord_stock_prediction_agent/.env` as an optional local override. Both are ignored by
Git.

Minimum Discord and paper-trading configuration:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_SIGNAL_CHANNEL_ID=your_stock_signals_channel_id
AGENT_REVIEW_CHANNEL_ID=your_agent_review_channel_id
DISCORD_PAPER_LOG_CHANNEL_ID=

PAPER_TRADING_ENABLED=true
ALPACA_API_KEY=your_alpaca_paper_key
ALPACA_SECRET_KEY=your_alpaca_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets

POLYGON_API_KEY=your_polygon_key
OPTION_STRIKE_VALIDATION_PROVIDER=polygon_first
```

The Stock-Prediction-Agent also uses the provider variables documented in the root
README, including Google/Gemini, RapidAPI/TradingView, and Tastytrade when those paths
are enabled.

Recommended runtime settings:

```env
AGENT_NAME=AI Stock Prediction Agent
SIGNAL_WORKER_CONCURRENCY=4
SIGNAL_QUEUE_LIMIT=20000
SIGNAL_MAX_ATTEMPTS=3
SIGNAL_RETRY_BASE_SECONDS=5
SIGNAL_CLAIM_TIMEOUT_SECONDS=600
PENDING_ORDER_BATCH_SIZE=100

PROTECTION_MONITOR_SECONDS=15
EQUITY_STOP_LOSS_PCT=1.0
EQUITY_TAKE_PROFIT_PCT=10.0
OPTION_STOP_LOSS_PCT=5.0
OPTION_TAKE_PROFIT_PCT=10.0
MAX_EQUITY_QTY=1000000
MAX_OPTION_QTY=1000
MAX_DAILY_PAPER_TRADES=20
ALLOW_DUPLICATE_PAPER_ORDERS=true

OPTIONS_TRADING_ENABLED=true
DEFAULT_OPTION_QTY=1
DEFAULT_OPTION_ORDER_TYPE=auto
OPTION_BACKTEST_LOOKBACK_DAYS=365
DEFAULT_OPTION_STRATEGY_DELTA=30
OPTION_VALIDATION_TIMEOUT_SECONDS=35
OPTION_ALLOW_UNVALIDATED_FALLBACK=false

LEARNING_ENABLED=true
LEARNING_MIN_SAMPLES=5
RUNTIME_LOG_LEVEL=INFO
DEBUG_OUTPUT_ENABLED=false
```

Decision thresholds can be tuned without changing Python code:

```env
BUY_MIN_RETURN_PCT=0.01
BUY_STRONG_RETURN_PCT=1.0
BUY_EXCELLENT_CONFIDENCE=80
BUY_LOW_RISK=40
BUY_DECISION_SCORE=45

SELL_MIN_RETURN_PCT=-0.01
SELL_STRONG_RETURN_PCT=-1.0
SELL_LOW_CONFIDENCE=50
SELL_HIGH_RISK=60
SELL_DECISION_SCORE=45
```

## Install And Run On Windows

From the project root:

```powershell
cd C:\Users\mdama\Stock-Prediction-Agent
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r discord_stock_prediction_agent\requirements.txt
```

Command Prompt activation:

```bat
venv\Scripts\activate
```

Run the Streamlit Stock-Prediction-Agent dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

Run the Discord agent in a separate terminal, from the project root:

```powershell
python -m discord_stock_prediction_agent.discord_agent
```

The same process can serve Discord and WhatsApp. Configure the `WHATSAPP_*`
variables in `.env`, set `WHATSAPP_WEBHOOK_ENABLED=true`, and restart the agent.
The signed webhook endpoints are:

```text
GET/POST http://localhost:5000/webhook
GET      http://localhost:5000/healthz
```

Meta must reach `/webhook` through a public HTTPS URL. POST requests are rejected
unless `X-Hub-Signature-256` matches `WHATSAPP_APP_SECRET`. Text, media captions,
button replies, and list replies enter the same durable queue, parser, Agent ON/OFF
decision layer, Alpaca paper-order lifecycle, and protection monitors as Discord.
Use `WHATSAPP_ALLOWED_SENDER_IDS` and `WHATSAPP_ALLOWED_GROUP_IDS` as optional
comma-separated allowlists.

Signal replies and command output already reach WhatsApp because they're sent
in response to an incoming message. Proactive/background alerts (protection
stop-loss/take-profit triggers, a queued order finally filling once the market
opens, a contract becoming tradable) are different: they're raised by the
periodic monitor loop with no incoming message to reply to, so there's nothing
to infer a WhatsApp destination from. Set `WHATSAPP_ALERT_TARGET` (a phone
number, or the signals group ID with `WHATSAPP_ALERT_IS_GROUP=true`, the
default) to also receive this whole class of alert on WhatsApp — the same
role `AGENT_REVIEW_CHANNEL_ID` / `DISCORD_PAPER_LOG_CHANNEL_ID` play on Discord.
Leave it blank to keep these alerts Discord-only.

All 12 `!agent_*` commands listed under **Useful status commands** above also work
from WhatsApp (same `!` prefix, same reply text, built from the same underlying
functions as Discord) — WhatsApp messages skip Discord's own command dispatcher, so
a dedicated router (`_try_dispatch_whatsapp_command` in `discord_agent.py`) recognizes
and answers them before normal signal parsing runs. `!agent_on`, `!agent_off`, and
`!agent_retry_dead` additionally require the sender's WhatsApp ID to be listed in
`WHATSAPP_ADMIN_SENDER_IDS` — being in `WHATSAPP_ALLOWED_SENDER_IDS` only grants
signal access, not mode-changing control, mirroring the Discord Administrator/Manage
Server gate (which has no WhatsApp equivalent).

The standalone ingress-only server is available for split deployments:

```powershell
python -m discord_stock_prediction_agent.whatsapp_webhook
```

It persists incoming events, while the Discord agent process remains the single
decision and broker worker. This avoids two processes attempting the same order.

Meta's current official Groups API requires an Official Business Account, uses
invite-only groups, and allows at most 8 participants per group. For large communities,
keep Discord as the primary signal room and use WhatsApp for small approved groups or
direct conversations. See Meta's [platform overview](https://developers.facebook.com/documentation/business-messaging/whatsapp/about-the-platform/)
and [Groups API documentation](https://developers.facebook.com/documentation/business-messaging/whatsapp/groups/).

Do not run that module while the terminal is inside
`discord_stock_prediction_agent`; the parent project directory must be the working
directory. A process lock prevents two copies from using the same queues and state.

## Example Signals

Equities:

```text
buy TSLA qty 2
SELL AAPL qty 1
HOLD MSFT. Mixed indicators and earnings tomorrow.
BUY AMZN LIMIT 235.50
SELL TSLA if price falls below 295
BUY GOOGL if price closes above 205, otherwise HOLD
```

Single-leg options:

```text
BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80
STC AAPL 240C 08/21 @6.80
STO SPY 620P 08/21 @2.40
BTC SPY 620P 08/21 @1.05
Buy 30 delta AAPL call qty 1
```

Multi-leg options:

```text
BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5
BUY SPY 640C + 640P 09/19 @8.20 Debit Qty 2
STO SPY 620P / BTO SPY 610P / STO SPY 670C / BTO SPY 680C 09/19 @2.15 Credit Qty 10
```

## Order And Monitoring Behavior

- A missing equity or option quantity defaults to 1 for an entry.
- Equity SELL checks the available Alpaca position; without quantity it closes the
  available position.
- Close-option actions require a matching Alpaca option position.
- Market-closed approved orders are queued and retried after Alpaca reports open.
- If the terminal was off when the market opened, the next agent startup checks
  Alpaca immediately. Open-market orders are submitted during startup recovery;
  closed-market orders stay persisted for the normal monitor.
- Closed-market equity BUY/SELL orders use a dedicated durable SQLite queue.
- A stable Alpaca `client_order_id` reconciles an accepted submission after a
  timeout or agent restart, preventing accidental duplicate retries.
- A queued BUY is removed from the market-order queue only after Alpaca returns
  an accepted order ID. It then remains in fill tracking until every filled share
  has the configured protection level.
- Unavailable exact option contracts are stored and checked again.
- Limit orders are submitted to Alpaca at the signal limit; Alpaca owns fill behavior.
- Conditional equity and option entries wait for the specified price condition.
- Agent-bought equities use the confirmed Alpaca fill price for default `-1%`
  loss protection and `+10%` profit protection.
- Long option positions use the confirmed premium fill for default `-5%` loss
  protection and `+10%` profit protection. Explicit signal SL/TP values override
  those defaults. Short-option protection uses the economically inverse direction.
- Filled atomic multi-leg debit/credit strategies are tracked at their net fill;
  protection closes all legs together with one Alpaca MLeg order.
- Successfully submitted queued items are removed from pending state.
- Permanent broker rejections are removed; transient failures remain eligible for retry.
- Duplicate signals may create independent paper orders when
  `ALLOW_DUPLICATE_PAPER_ORDERS=true`.

## Runtime State

These local files are created automatically and are intentionally ignored by Git:

| File | Contents |
|---|---|
| `discord_stock_prediction_agent/agent_state.json` | mode, decisions, learning, tracked positions, submitted-entry/exit reconciliation, and pending options |
| `discord_stock_prediction_agent/agent_state.backup.json` | atomic recovery copy used when the primary state file is unreadable |
| `discord_stock_prediction_agent/signal_queue.sqlite3` | durable incoming queue, retries, and dead-letter state |
| `discord_stock_prediction_agent/pending_market_orders.sqlite3` | approved equity orders waiting for Alpaca market-open submission; accepted rows are removed transactionally |
| `discord_stock_prediction_agent/options_validation_cache.json` | recent option validation cache |
| `discord_stock_prediction_agent/symbol_cache.json` | refreshed tradable-symbol directory |
| `discord_stock_prediction_agent/logs/discord_agent.log` | rotating runtime log |
| `discord_stock_prediction_agent/discord_agent.lock` | single-process runtime lock |

Back up `agent_state.json`, `agent_state.backup.json`, `signal_queue.sqlite3`, and
`pending_market_orders.sqlite3` before moving an active deployment.
Do not commit them because they can contain Discord messages and broker metadata.

## Main Files

| File | Responsibility |
|---|---|
| `discord_agent.py` | Discord events, workers, decisions, monitors, commands, and output |
| `whatsapp_webhook.py` | signed WhatsApp webhook verification and durable ingestion |
| `whatsapp_client.py` | WhatsApp Cloud API output and embed-to-text conversion |
| `signal_normalizer.py` | Discord text cleanup and format normalization |
| `signal_parser.py` | equity parsing and conditional rules |
| `options_parser.py` | single/multi-leg option parsing |
| `prediction_bridge.py` | Stock Price Validation integration |
| `options_strategy_bridge.py` | Options Strategy Validation integration |
| `polygon_options_data.py` | exact-strike historical contract data |
| `alpaca_paper.py` | paper account, contract, position, and order API |
| `durable_signal_queue.py` | SQLite signal queue and retry lifecycle |
| `pending_market_orders.py` | durable closed-market equity queue and broker-submission lifecycle |
| `state_store.py` | atomic JSON state, pending orders, learning, and positions |
| `runtime_lock.py` | prevents duplicate agent processes |
| `config.py` | environment loading and production checks |

## Verification

Compile the project:

```powershell
python -m compileall -q discord_stock_prediction_agent src tests
```

Run every Discord-agent regression module from the project root:

```powershell
Get-ChildItem discord_stock_prediction_agent\test_*.py | Sort-Object Name | ForEach-Object {
    python -m ("discord_stock_prediction_agent." + $_.BaseName)
}
```

Important coverage includes:

- 2,172 deterministic real-world parser cases.
- 2,400 mixed signals parsed and queued without loss.
- 1,000 exact-strike option payload cases in the bulk report.
- Agent mode, Alpaca idempotency, broker errors, multi-leg payloads, learning, queue
  restart recovery, and process-lock behavior.

Tests use mocks/local state unless a script explicitly states that it performs API or
paper-order operations. Never run a bulk paper-order script against an account without
reviewing its flags first.

## Production Notes

- Run one bot process per state directory.
- Use a service manager and automatic restart policy for long-running deployments.
- Keep `.env`, broker keys, tokens, state databases, and logs outside Git.
- Restrict Discord channel permissions and mode-changing permissions.
- Use `!agent_health` for broker/queue readiness and review rotating logs.
- Use `!agent_dead_letters` to inspect failures. Administrators can run
  `!agent_retry_dead [limit]` to return a bounded set to the durable queue.
- Paper-test every new parser or execution rule before considering real-money support.
- For a 2,000-member server, measure API rate limits and processing latency under the
  expected burst pattern; worker concurrency alone does not remove provider limits.

See [PRODUCTION_HARDENING.md](PRODUCTION_HARDENING.md) for the production-readiness
checklist and remaining scaling work.
