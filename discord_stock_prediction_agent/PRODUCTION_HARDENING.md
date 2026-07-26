# Discord Agent Production Hardening

This agent is designed to process noisy Discord trading signals while keeping
paper trading controlled and auditable.

## Reliability
- Discord reconnect tracebacks from temporary DNS/gateway issues are suppressed.
- Tastytrade option validation uses cache and rate-limit-safe review states.
- Agent exceptions are recorded internally and shown as safe review messages.
- Incoming Discord signals use a durable SQLite/WAL queue.
- A bounded worker pool processes multiple signals without unbounded task creation.
- Failed signal jobs retry with exponential backoff and then move to dead-letter state.
- Interrupted jobs are recovered after a process restart.
- `agent_state.json` writes are atomic and protected by a process-wide lock.

## Signal Understanding
- Routes signals as equity, option, no-trade commentary, or invalid input.
- Normalizes Discord text and webhook JSON before deterministic parsing.
- Handles code fences, Markdown, zero-width characters, Unicode punctuation,
  common signal emojis, field-name aliases, and OCC option symbols.
- Structured webhooks can provide action, symbol, quantity, order type,
  strike/right/expiry, premium, stop loss, target, and multi-leg arrays.
- Parsing remains deterministic: normalization never invents a missing symbol,
  strike, expiry, quantity, price, or position intent.
- Tracks raw signal events with normalized hashes for duplicate detection.
- Tracks bounded signal-template reliability in `agent_state.json` under
  `parser_learning`; `!agent_learning` reports learned format counts and parser
  success rate.

## Decision Quality
- Uses the Stock-Prediction-Agent decision output with flexible BUY/SELL/HOLD gates.
- Exact-strike option signals stay on exact-strike validation and order lookup.
- Explicit delta signals use delta/DTE validation.
- Decision-pattern outcomes are stored and can adjust bounded scoring support;
  parser-format learning is telemetry and does not bypass broker or safety gates.

## Agent Operating Mode
- The persistent mode defaults to `ON`.
- `!agent_on` keeps the normal prediction and options-strategy decisions active.
- `!agent_off` bypasses those decision verdicts and routes every valid incoming
  BUY/SELL signal directly to paper-order handling.
- `!agent_mode` reports the active mode, and `!agent_status` includes it.
- Only the bot owner, server administrators, or members with Manage Server
  permission can change the mode.
- Agent OFF does not bypass parsing, paper-only endpoint enforcement, exact
  contract lookup, market-hours queues, requested price/condition handling,
  position availability, buying power, Alpaca permissions, or SL/TP monitoring.
- HOLD and commentary remain non-trading inputs in both modes.

## Discord Experience
- Raw signal channel stays clean.
- Incoming signals are not decorated with acknowledgement tick reactions.
- Review/output goes to the configured review channel.
- Backend/API details are hidden unless `DEBUG_OUTPUT_ENABLED=true`.

## Paper Trading Safety
- Quantity limits:
  - `MAX_EQUITY_QTY`
  - `MAX_OPTION_QTY`
- Daily cap:
  - `MAX_DAILY_PAPER_TRADES`
- Duplicate/cooldown controls:
  - `DUPLICATE_SIGNAL_TTL_MINUTES`
  - `PER_SYMBOL_COOLDOWN_MINUTES`
- Unvalidated option fallback is disabled by default:
  - `OPTION_ALLOW_UNVALIDATED_FALLBACK=false`

## Monitoring
- Signals are stored in `signal_events`.
- Orders are stored in `order_events`.
- Blocks are stored in `safety_blocks`.
- Incoming queue rows are stored in `signal_queue.sqlite3`.
- Pending paper trades are stored in `agent_state.json` under:
  - `pending_buy_orders`
  - `pending_sell_orders`
  - `pending_option_orders`
  - `conditional_equity_orders`
- Successful queued submissions are removed from their pending queue. Submitted
  equity buys remain as fill trackers until Alpaca reports filled, rejected,
  canceled, or expired.
- Use `!agent_status` and `!agent_summary` for queue and daily counts.

## Capacity Configuration
- `SIGNAL_WORKER_CONCURRENCY=4`
- `SIGNAL_QUEUE_LIMIT=20000`
- `SIGNAL_QUEUE_POLL_SECONDS=0.5`
- `SIGNAL_MAX_ATTEMPTS=3`
- `SIGNAL_RETRY_BASE_SECONDS=5`
- `SIGNAL_CLAIM_TIMEOUT_SECONDS=600`
- `ALPACA_MAX_CONCURRENT_REQUESTS=8`
- `ALPACA_MAX_REQUEST_ATTEMPTS=3`
- `ALPACA_REQUEST_TIMEOUT_SECONDS=20`
- `PENDING_ORDER_BATCH_SIZE=100`
- `RUNTIME_LOG_LEVEL=INFO`
- `RUNTIME_LOG_MAX_BYTES=5000000`
- `RUNTIME_LOG_BACKUP_COUNT=5`

Keep worker concurrency conservative because prediction, market-data, Discord,
and broker APIs have independent rate limits. Increase it only after load tests.

## Production Readiness
Before adding a large group, test:
- valid equity buy/sell/hold
- valid options calls/puts
- invalid/random messages
- duplicate signals
- rapid-fire signals
- market-closed sell handling
- Alpaca rejection handling
- Tastytrade rate-limit handling
- webhook JSON and OCC contract symbols
- Markdown, emoji, and natural-language signal variants
- multi-leg debit/credit structures

The deterministic corpus currently covers more than 2,100 distinct normal,
single-leg option, OCC, JSON, multi-leg, noisy-format, and commentary cases.
The separate load harness processes 2,400 mixed signals through the durable
queue. These tests validate supported grammar and load behavior; they do not
guarantee that every imaginable human sentence is a safe executable order.

This implementation is production-hardened for one continuously running bot
process. Horizontal multi-instance deployment requires a shared external
database/queue and distributed order idempotency; do not run two bot processes
against the same paper account and local state files.

## Alpaca Submission Safety

- Every Discord signal and pending trade receives a stable Alpaca
  `client_order_id`.
- A retry of the same signal reuses the same ID; separate duplicate messages
  receive separate IDs.
- HTTP 429 and transient 5xx/network failures use bounded backoff.
- After an ambiguous submission failure, the agent queries Alpaca by
  `client_order_id` before treating the order as unplaced.
- Concurrent Alpaca requests are bounded to protect the account API.
- Startup refuses any trading endpoint other than
  `https://paper-api.alpaca.markets`.

## Runtime Recovery

- Interrupted signal claims return to the durable queue after restart.
- The signal scheduler and trade monitor log unexpected failures and restart.
- Startup validates the Discord channels, paper credentials, worker limits,
  queue capacity, and monitor interval before connecting.
- A process lock prevents two local bot instances from sharing the same queue,
  state files, and Alpaca paper account.
- Runtime logs rotate under `discord_stock_prediction_agent/logs/`.
- Pending equity, option, multi-leg, and conditional queues are handled in fair
  rotating batches so unavailable early items cannot starve later trades.
- Pending orders survive transient Alpaca 429/5xx, timeout, connection, and DNS
  failures and retry with the same `client_order_id`.

## Deployment Boundary

This version supports a production-style single-server, single-bot-process
paper-trading pilot. For multiple bot processes, multiple Alpaca accounts, or
live-money operation, migrate queue/state to PostgreSQL or Redis, add
distributed locks, tenant ownership, centralized metrics/logging, and a
formal human-approved risk service.
