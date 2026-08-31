"""Discord runner for the stock prediction + Alpaca paper trading agent."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from logging.handlers import RotatingFileHandler
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

try:
    import discord
    from discord.ext import commands, tasks
except Exception as exc:  # pragma: no cover - import-time setup check
    raise SystemExit(
        "discord.py is not installed. Run: pip install -r "
        "discord_stock_prediction_agent/requirements.txt"
    ) from exc

from .alpaca_paper import AlpacaPaperClient
from .config import AGENT_DIR, config, production_config_errors
from .durable_signal_queue import (
    claim_next_signal,
    complete_signal,
    enqueue_signal,
    fail_signal,
    list_dead_signals,
    queue_stats,
    recover_inflight_signals,
    retry_dead_signals,
)
from .daily_signal_context import classify_and_parse_with_daily_context
from .market_context import get_market_context, refresh_market_context_async
from .multi_leg_validation import infer_multi_leg_price_effect, validate_multi_leg_strategy
from .options_parser import ParsedOptionLeg, ParsedOptionSignal, classify_and_parse
from .options_strategy_bridge import run_options_strategy_validation
from .options_symbol import likely_unsupported_by_alpaca, listed_expiry_fallbacks, resolve_underlying_for_prediction
from .pending_market_orders import (
    enqueue_market_order,
    list_queued_market_orders,
    mark_attempt as mark_market_order_attempt,
    mark_failed as mark_market_order_failed,
    queue_summary as market_order_queue_summary,
    remove_queued_market_order,
)
from .automate_agent import (
    AUTOMATE_AGENT_TAG,
    BoomCandidate,
    StrikeQuote,
    confidence_scaled_risk_multiplier,
    count_automate_positions,
    plan_automate_trades,
    select_best_strike,
)
from .prediction_bridge import run_project_prediction
from .protection_policy import build_protection_levels, evaluate_protection
from .runtime_lock import acquire_runtime_lock
from .signal_parser import ParsedSignal
from .stock_order_intent import (
    execute_alpaca_order_plan,
    gate_order_plan,
    stock_review_chunks,
)
from .symbol_directory import refresh_symbol_cache_from_alpaca
from .whatsapp_client import output_text, send_whatsapp_text
from .whatsapp_webhook import start_whatsapp_webhook_server
from .state_store import (
    add_decision_history,
    add_pending_buy,
    add_pending_exit_order,
    add_pending_multi_leg_entry_order,
    add_pending_option_entry_order,
    add_pending_option_order,
    add_conditional_equity_order,
    list_conditional_equity_orders,
    list_multi_leg_positions,
    list_option_positions,
    list_pending_buys,
    list_pending_exit_orders,
    list_pending_multi_leg_entry_orders,
    list_pending_option_entry_orders,
    list_pending_option_orders,
    list_positions,
    list_pending_sells,
    close_option_position_with_outcome,
    reduce_or_remove_position,
    reduce_or_remove_multi_leg_position,
    record_order_event,
    record_option_journal_entry,
    record_option_validation_event,
    record_parser_learning,
    record_safety_block,
    record_signal_event,
    remove_pending_buy,
    remove_pending_exit_order,
    remove_pending_multi_leg_entry_order,
    remove_pending_option_entry_order,
    remove_pending_option_order,
    remove_option_position,
    remove_multi_leg_position,
    remove_position,
    today_realized_pnl,
    remove_pending_sell,
    remove_conditional_equity_order,
    close_position_with_outcome,
    count_today_order_events,
    get_agent_mode,
    get_automate_agent_mode,
    set_automate_agent_mode,
    get_automate_agent_report_date,
    set_automate_agent_report_date,
    list_trade_outcomes,
    get_daily_summary,
    get_learning_profile,
    get_option_validation_summary,
    get_parser_learning_summary,
    get_pattern_learning,
    get_signal_learning_summary,
    last_order_for_symbol,
    record_learning_event,
    set_agent_mode,
    upsert_option_position,
    update_option_position,
    update_pending_exit_order,
    update_pending_multi_leg_entry_order,
    update_pending_option_entry_order,
    update_pending_buy_protected_qty,
    upsert_position,
    upsert_multi_leg_position,
    count_today_automate_agent_buys,
)


alpaca = AlpacaPaperClient()
_SIGNAL_TASKS: set[asyncio.Task] = set()
_QUEUE_RECOVERED = False
_STARTUP_ORDER_RECOVERY_DONE = False
_PENDING_CURSORS: dict[str, int] = {}


@dataclass(frozen=True)
class _QueuedIdentity:
    id: object
    bot: bool = False
    transport: str = "discord"
    reply_target: str = ""
    is_group: bool = False


class _QueuedMessage:
    """Minimal Discord message interface reconstructed from durable queue data."""

    def __init__(self, item: dict):
        self.content = str(item.get("raw_text") or "")
        self.id = str(item.get("message_id") or "")
        self.transport = str(item.get("transport") or "discord").lower()
        self.reply_target = str(item.get("reply_target") or item.get("channel_id") or "")
        self.is_group = bool(item.get("is_group"))
        self.author = _QueuedIdentity(
            str(item.get("user_id") or ""), transport=self.transport
        )
        self.channel = _QueuedIdentity(
            str(item.get("channel_id") or ""),
            transport=self.transport,
            reply_target=self.reply_target,
            is_group=self.is_group,
        )

    async def add_reaction(self, _reaction: str) -> None:
        return None


class _DiscordReconnectNoiseFilter(logging.Filter):
    """Hide discord.py reconnect tracebacks caused by transient DNS/network blips."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "Attempting a reconnect" in message and record.exc_info:
            exc_text = logging.Formatter().formatException(record.exc_info)
            transient_markers = (
                "getaddrinfo failed",
                "ClientConnectorDNSError",
                "Cannot connect to host gateway",
                "Temporary failure in name resolution",
            )
            if any(marker in exc_text for marker in transient_markers):
                if not getattr(self, "_reported_once", False):
                    logging.getLogger("discord_stock_prediction_agent").warning(
                        "Discord gateway had a temporary network/DNS reconnect. "
                        "Traceback suppressed; discord.py will reconnect automatically."
                    )
                    self._reported_once = True
                return False
        return True


def _configure_runtime_logging() -> None:
    logger = logging.getLogger("discord_stock_prediction_agent")
    level = getattr(logging, config.runtime_log_level, logging.INFO)
    logger.setLevel(level)
    log_dir = AGENT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "discord_agent.log"
    if not any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", "") == str(log_path)
        for handler in logger.handlers
    ):
        try:
            handler = RotatingFileHandler(
                log_path,
                maxBytes=max(100_000, config.runtime_log_max_bytes),
                backupCount=max(1, min(20, config.runtime_log_backup_count)),
                encoding="utf-8",
            )
        except OSError:
            # A running Windows process may hold the rotating log file open.
            # Imports and auxiliary workers must remain available in that case.
            handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
        logger.addHandler(handler)
    if config.suppress_discord_reconnect_tracebacks:
        logging.getLogger("discord.client").addFilter(_DiscordReconnectNoiseFilter())


_configure_runtime_logging()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@dataclass(frozen=True)
class DecisionResult:
    action: str
    reason: str
    predicted_return: float
    confidence: float
    risk: float
    score: float
    market_regime: str = "unknown"


def _agent_is_enabled() -> bool:
    return get_agent_mode() == "ON"


def _automate_agent_is_enabled() -> bool:
    """automate_agent's own independent switch, controlled by
    !automate_agent_on / !automate_agent_off -- separate from the general
    agent mode that !agent_on / !agent_off control for manual signals.
    """
    return get_automate_agent_mode() == "ON"


def _direct_signal_decision(action: str, score: float = 100.0) -> DecisionResult:
    normalized = str(action or "").upper()
    final_action = normalized if normalized in {"BUY", "SELL"} else "HOLD"
    return DecisionResult(
        final_action,
        "Agent OFF: following the valid incoming signal directly.",
        0.0,
        0.0,
        0.0,
        max(0.0, min(100.0, float(score))),
        "signal_direct",
    )


def _direct_option_validation(action: str) -> dict:
    normalized = str(action or "").upper()
    return {
        "status": "AGENT_OFF_DIRECT",
        "decision": normalized if normalized in {"BUY", "SELL"} else "HOLD",
        "strategy_input": {"validation_method": "agent_off_direct"},
        "error": "",
    }


def _decision_color(decision: str) -> int:
    return {
        "BUY": 0x16A34A,
        "SELL": 0xDC2626,
        "HOLD": 0xF59E0B,
        "REJECT": 0xDC2626,
        "REVIEW": 0x64748B,
    }.get(str(decision).upper(), 0x64748B)


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_utc(value: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", ""))
    except Exception:
        return datetime.min


def _public_error(err: str) -> str:
    text = str(err or "")
    lowered = text.lower()
    if not text:
        return "The external service did not return a usable response."
    if "HTTP 429" in text or "rate limit" in lowered or "too many requests" in lowered:
        return "Validation service is temporarily busy. The agent safely skipped this trade."
    if "getaddrinfo" in text or "ClientConnector" in text or "Connection" in text:
        return "A network service was temporarily unavailable. The agent safely skipped this trade."
    if config.debug_output_enabled:
        return text[:500]
    if "Alpaca HTTP" in text:
        if "insufficient options buying power" in lowered:
            return "Alpaca rejected the paper order because the account has insufficient options buying power."
        if "insufficient buying power" in lowered:
            return "Alpaca rejected the paper order because the account has insufficient equity buying power."
        if "not eligible to trade uncovered option contracts" in lowered:
            return "Alpaca rejected the paper order because this account cannot trade uncovered option contracts."
        if "potential wash trade" in lowered:
            return "Alpaca rejected the paper order because its wash-trade protection detected an opposing open order."
        if "options market orders are only allowed during market hours" in lowered:
            return "Alpaca rejected the option market order because the options market is closed."
        if "no position" in lowered or "insufficient qty" in lowered:
            return "Alpaca rejected the closing order because the required position quantity is unavailable."
        return "Alpaca rejected or could not accept the paper order."
    return text[:220]


async def _block_trade(
    message: discord.Message,
    symbol: str,
    reason: str,
    category: str = "safety",
    raw_input: str = "",
) -> None:
    await asyncio.to_thread(
        record_safety_block,
        {
            "symbol": symbol,
            "category": category,
            "reason": reason,
            "raw_input": raw_input or getattr(message, "content", ""),
        },
    )
    await _send_review_or_reply(message, f"{symbol or 'Signal'}: {reason}")


def _cooldown_active(symbol: str, side: str) -> tuple[bool, str]:
    last = last_order_for_symbol(symbol, side)
    if not last:
        return False, ""
    last_time = _parse_utc(last.get("created_at", ""))
    minutes = max(0, config.per_symbol_cooldown_minutes)
    if minutes and last_time >= datetime.utcnow() - timedelta(minutes=minutes):
        return True, f"cooldown active after recent {side.upper()} order"
    return False, ""


async def _trade_guard(
    message: discord.Message,
    symbol: str,
    side: str,
    qty: float,
    asset_type: str,
) -> bool:
    max_qty = config.max_option_qty if asset_type == "option" else config.max_equity_qty
    if qty <= 0:
        await _block_trade(message, symbol, "Quantity is invalid, so no paper trade will be placed.", "quantity")
        return False
    if qty > max_qty:
        await _block_trade(
            message,
            symbol,
            f"Quantity {qty:g} is above the configured {asset_type} limit of {max_qty:g}. No paper trade placed.",
            "quantity_limit",
        )
        return False
    if (
        config.max_daily_paper_trades > 0
        and count_today_order_events(exclude_automate_agent=True) >= config.max_daily_paper_trades
    ):
        await _block_trade(
            message,
            symbol,
            f"Daily paper-trade limit reached ({config.max_daily_paper_trades}). No more trades today.",
            "daily_limit",
        )
        return False
    if not config.allow_duplicate_paper_orders:
        active, reason = _cooldown_active(symbol, side)
        if active:
            await _block_trade(
                message,
                symbol,
                f"{reason}. No duplicate paper trade placed.",
                "cooldown",
            )
            return False
    return True


def _record_order(symbol: str, side: str, qty: float, status: str, order_id: str = "", asset_type: str = "equity", detail: str = "") -> None:
    record_order_event(
        {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "qty": float(qty),
            "status": status,
            "order_id": order_id,
            "asset_type": asset_type,
            "detail": detail,
        }
    )


def _client_order_id(category: str, stable_key: object) -> str:
    digest = hashlib.sha256(str(stable_key).encode("utf-8")).hexdigest()[:28]
    safe_category = "".join(ch for ch in str(category).lower() if ch.isalnum())[:12] or "order"
    return f"dsa-{safe_category}-{digest}"[:48]


def _is_market_closed_order_error(err: str) -> bool:
    text = str(err or "").lower()
    return (
        "market hours" in text
        or "market is closed" in text
        or "outside market" in text
        or "not open" in text
    )


def _is_transient_broker_error(err: str) -> bool:
    text = str(err or "").lower()
    return any(
        marker in text
        for marker in (
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "temporary failure",
            "name resolution",
            "request error",
        )
    )


def _pending_order_age_hours(created_at: object) -> float:
    try:
        created = datetime.fromisoformat(str(created_at or "").replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=ZoneInfo("UTC"))
    return (datetime.now(ZoneInfo("UTC")) - created).total_seconds() / 3600.0


def _pending_order_expired(attempts: object, created_at: object) -> tuple[bool, str]:
    """A queued order (waiting for the market to open) that has retried too
    many times or sat too long can never legitimately succeed -- e.g. the
    position it was meant to sell no longer exists. Without this cutoff it
    silently retries forever instead of ever being surfaced or resolved.
    """
    attempt_count = int(_as_float(attempts))
    if attempt_count >= config.pending_order_max_attempts:
        return True, f"{attempt_count} attempts"
    age_hours = _pending_order_age_hours(created_at)
    if age_hours >= config.pending_order_max_age_hours:
        return True, f"{age_hours:.0f}h old"
    return False, ""


def _rotating_batch(items: list, queue_name: str) -> list:
    if not items:
        return []
    limit = max(1, min(config.pending_order_batch_size, len(items)))
    start = _PENDING_CURSORS.get(queue_name, 0) % len(items)
    ordered = items[start:] + items[:start]
    _PENDING_CURSORS[queue_name] = (start + limit) % len(items)
    return ordered[:limit]


def _queue_option_order(
    occ_symbol: str,
    option: ParsedOptionSignal,
    qty: float,
    order_type: str,
    contract_expiration: str,
    quality: dict,
    reason: str,
    order_side: str = "buy",
    position_intent: str = "buy_to_open",
    requires_position: bool = False,
    contract_pending: bool = False,
    remaining_qty: Optional[float] = None,
    move_stop_to_breakeven: bool = False,
) -> None:
    add_pending_option_order(
        {
            "pending_key": f"option:{uuid.uuid4().hex}",
            "occ_symbol": occ_symbol,
            "root": option.root,
            "side": option.side,
            "strike": option.strike,
            "expiry_date": contract_expiration or option.expiry_date or "",
            "qty": float(qty),
            "order_type": order_type,
            "limit_price": option.fill_price if str(order_type or "").lower() == "limit" else None,
            "stop_loss": option.stop_loss,
            "target_price": option.target_price,
            "target_prices": list(option.target_prices),
            "trailing_stop_pct": option.trailing_stop_pct,
            "exit_before_market_close": option.exit_before_market_close,
            "exit_minutes_before_close": option.exit_minutes_before_close,
            "exit_if_target_not_hit": option.exit_if_target_not_hit,
            "risk_stop_pct": option.risk_stop_pct,
            "maximum_loss_amount": option.maximum_loss_amount,
            "position_type": option.position_type,
            "add_quantity": option.add_quantity,
            "add_trigger_premium": option.add_trigger_premium,
            "add_trigger_underlying_direction": option.add_trigger_underlying_direction,
            "add_trigger_underlying_price": option.add_trigger_underlying_price,
            "underlying_trigger_direction": option.underlying_trigger_direction,
            "underlying_trigger_price": option.underlying_trigger_price,
            "entry_premium_direction": option.entry_premium_direction,
            "entry_premium_price": option.entry_premium_price,
            "exit_underlying_direction": option.exit_underlying_direction,
            "exit_underlying_price": option.exit_underlying_price,
            "time_in_force": option.time_in_force,
            "remaining_instruction": option.remaining_instruction,
            "stop_scope": option.stop_scope,
            "entry_price_type": option.entry_price_type,
            "semantic_contract": option.semantic_contract,
            "signal_quality": quality.get("score"),
            "risk_reward": quality.get("risk_reward"),
            "raw_input": option.raw_text,
            "reason": reason,
            "order_side": order_side,
            "position_intent": position_intent,
            "requires_position": bool(requires_position),
            "contract_pending": bool(contract_pending),
            "remaining_qty": remaining_qty,
            "move_stop_to_breakeven": bool(move_stop_to_breakeven),
        }
    )


def _queue_conditional_option_scale_in(metadata: dict) -> bool:
    """Persist a second entry that waits for the base fill and its trigger."""
    premium_trigger = _as_float(metadata.get("add_trigger_premium"))
    underlying_trigger = _as_float(metadata.get("add_trigger_underlying_price"))
    trigger_direction = str(metadata.get("add_trigger_underlying_direction") or "").lower()
    if premium_trigger <= 0 and not (
        underlying_trigger > 0 and trigger_direction in {"above", "below"}
    ):
        return False
    qty = _as_float(metadata.get("add_quantity")) or _as_float(metadata.get("qty")) or 1.0
    add_pending_option_order(
        {
            "pending_key": f"option:{uuid.uuid4().hex}",
            "occ_symbol": str(metadata.get("occ_symbol") or "").upper(),
            "root": str(metadata.get("root") or "").upper(),
            "side": metadata.get("side"),
            "strike": metadata.get("strike"),
            "expiry_date": metadata.get("expiry_date") or "",
            "qty": float(qty),
            "order_type": "limit" if premium_trigger > 0 else "market",
            "limit_price": premium_trigger or None,
            "stop_loss": metadata.get("stop_loss"),
            "target_price": metadata.get("target_price"),
            "target_prices": list(metadata.get("target_prices") or []),
            "trailing_stop_pct": metadata.get("trailing_stop_pct"),
            "exit_before_market_close": bool(metadata.get("exit_before_market_close")),
            "exit_minutes_before_close": metadata.get("exit_minutes_before_close"),
            "exit_if_target_not_hit": bool(metadata.get("exit_if_target_not_hit")),
            "risk_stop_pct": metadata.get("risk_stop_pct"),
            "maximum_loss_amount": metadata.get("maximum_loss_amount"),
            "position_type": metadata.get("position_type"),
            "exit_underlying_direction": metadata.get("exit_underlying_direction"),
            "exit_underlying_price": metadata.get("exit_underlying_price"),
            "time_in_force": metadata.get("time_in_force"),
            "stop_scope": metadata.get("stop_scope"),
            "semantic_contract": metadata.get("semantic_contract") or {},
            "underlying_trigger_direction": trigger_direction or None,
            "underlying_trigger_price": underlying_trigger or None,
            "signal_quality": metadata.get("signal_quality"),
            "raw_input": metadata.get("raw_input"),
            "reason": "conditional_scale_in",
            "order_side": "buy",
            "position_intent": "buy_to_open",
            "requires_position": True,
            "scale_in_order": True,
            "base_order_id": str(metadata.get("base_order_id") or ""),
            "contract_pending": not bool(metadata.get("occ_symbol")),
        }
    )
    return True


def _option_leg_route(leg: ParsedOptionLeg | dict) -> tuple[str, str, bool]:
    action = str(
        leg.order_action if isinstance(leg, ParsedOptionLeg) else leg.get("order_action") or "open_long"
    ).lower()
    if action == "close_long":
        return "sell", "sell_to_close", True
    if action == "open_short":
        return "sell", "sell_to_open", False
    if action == "close_short":
        return "buy", "buy_to_close", True
    return "buy", "buy_to_open", False


def _multi_leg_specs(option: ParsedOptionSignal) -> list[dict]:
    return [
        {
            "root": leg.root,
            "strike": float(leg.strike),
            "side": leg.side,
            "order_action": leg.order_action,
            "ratio_qty": max(1, int(leg.ratio_qty)),
            "expiry_date": leg.expiry_date or option.expiry_date or "",
        }
        for leg in option.legs
    ]


def _multi_leg_limit_price(option: ParsedOptionSignal) -> Optional[float]:
    if option.fill_price is None:
        return None
    price = abs(float(option.fill_price))
    return -price if infer_multi_leg_price_effect(option) == "credit" else price


def _queue_multi_leg_option_order(
    option: ParsedOptionSignal,
    qty: float,
    quality: dict,
    reason: str,
    resolved_legs: Optional[list[dict]] = None,
    contract_expiration: str = "",
) -> None:
    add_pending_option_order(
        {
            "pending_key": f"mleg:{uuid.uuid4().hex}",
            "occ_symbol": "",
            "order_class": "mleg",
            "root": option.root,
            "structure": option.structure,
            "expiry_date": contract_expiration or option.expiry_date or "",
            "expiry_mode": option.expiry_mode,
            "qty": float(qty),
            "order_type": _option_requested_order_type(option),
            "limit_price": _multi_leg_limit_price(option),
            "price_effect": infer_multi_leg_price_effect(option),
            "stop_loss": option.stop_loss,
            "target_price": option.target_price,
            "target_prices": list(option.target_prices),
            "maximum_loss_amount": option.maximum_loss_amount,
            "underlying_trigger_direction": option.underlying_trigger_direction,
            "underlying_trigger_price": option.underlying_trigger_price,
            "legs": resolved_legs or _multi_leg_specs(option),
            "contract_pending": not bool(resolved_legs),
            "signal_quality": quality.get("score"),
            "raw_input": option.raw_text,
            "reason": reason,
        }
    )


def _option_requested_order_type(option: ParsedOptionSignal) -> str:
    raw_upper = str(option.raw_text or "").upper()
    configured = str(config.default_option_order_type or "auto").lower()
    if "MARKET" in raw_upper:
        return "market"
    if "LIMIT" in raw_upper:
        return "limit"
    if configured != "auto":
        return configured
    return "limit" if option.fill_price else "market"


def _queue_option_exit(position: dict, qty: float, reason: str, trigger_price: float, current_price: float) -> None:
    occ_symbol = str(position.get("occ_symbol") or "").upper()
    if not occ_symbol:
        return
    short_position = str(position.get("position_intent") or "").lower() == "sell_to_open"
    add_pending_option_order(
        {
            "pending_key": f"option-exit:{uuid.uuid4().hex}",
            "occ_symbol": occ_symbol,
            "root": str(position.get("root") or "").upper(),
            "side": str(position.get("side") or "").upper(),
            "strike": position.get("strike"),
            "expiry_date": position.get("expiry_date") or "",
            "qty": float(qty),
            "order_type": "market",
            "limit_price": None,
            "stop_loss": position.get("stop_loss"),
            "target_price": position.get("target_price"),
            "signal_quality": position.get("signal_quality"),
            "raw_input": f"Protective option exit for {occ_symbol}",
            "reason": reason,
            "order_side": "buy" if short_position else "sell",
            "position_intent": "buy_to_close" if short_position else "sell_to_close",
            "requires_position": True,
            "trigger_price": float(trigger_price),
            "observed_price": float(current_price),
            "remaining_qty": max(0.0, _as_float(position.get("qty")) - float(qty)),
            "next_target_index": (
                int(_as_float(position.get("target_index"))) + 1
                if reason == "option_target_price" else int(_as_float(position.get("target_index")))
            ),
        }
    )


def _option_limit_entry_ready(current_price: float, limit_price: float, side: str = "buy") -> bool:
    current = _as_float(current_price)
    limit = _as_float(limit_price)
    if limit <= 0:
        return True
    if current <= 0:
        return False
    # For a buy limit, only submit when the option premium is at or below
    # the signal price. For a sell limit, require at or above the signal price.
    return current <= limit if str(side or "buy").lower() == "buy" else current >= limit


def _option_order_route(option: ParsedOptionSignal) -> tuple[str, str, bool]:
    action = str(option.order_action or "open_long").lower()
    if action == "close_long":
        return "sell", "sell_to_close", True
    if action == "open_short":
        return "sell", "sell_to_open", False
    if action == "close_short":
        return "buy", "buy_to_close", True
    return "buy", "buy_to_open", False


def _multi_leg_mapped_action(option: ParsedOptionSignal) -> str:
    return str(validate_multi_leg_strategy(option).get("decision") or "REVIEW")


def _multi_leg_contract_lookup(
    root: str,
    leg_specs: list[dict],
    expiry_date: str,
    allow_nearest_expiry: bool = False,
) -> dict:
    """Resolve each leg to its exact tradable contract, including calendar expiries."""
    if not alpaca.ready():
        return {"status": "SKIPPED", "message": "Alpaca paper trading is not configured.", "legs": []}

    errors: list[str] = []
    explicit_leg_expiries = [str(spec.get("expiry_date") or "") for spec in leg_specs]
    if any(explicit_leg_expiries):
        resolved: list[dict] = []
        used_fallback = False
        for spec in leg_specs:
            requested_expiry = str(spec.get("expiry_date") or expiry_date or "")
            candidates = listed_expiry_fallbacks(requested_expiry) or ([requested_expiry] if requested_expiry else [])
            contract = None
            selected_expiry = ""
            for candidate in candidates:
                contracts, err = alpaca.get_option_contracts(
                    root,
                    candidate or None,
                    float(spec.get("strike") or 0),
                    str(spec.get("side") or "").lower(),
                )
                if not contracts:
                    if err:
                        errors.append(err)
                    continue
                contract = next(
                    (
                        item for item in contracts
                        if str(item.get("expiration_date") or "") == candidate
                        and item.get("tradable", True) is not False
                    ),
                    None,
                )
                if contract:
                    selected_expiry = candidate
                    break
            if not contract:
                resolved = []
                break
            side, intent, requires_position = _option_leg_route(spec)
            resolved.append(
                {
                    **spec,
                    "symbol": str(contract.get("symbol") or "").upper(),
                    "side_order": side,
                    "position_intent": intent,
                    "requires_position": requires_position,
                    "expiration_date": str(contract.get("expiration_date") or selected_expiry),
                }
            )
            used_fallback = used_fallback or bool(requested_expiry and selected_expiry != requested_expiry)
        if len(resolved) == len(leg_specs) and all(item.get("symbol") for item in resolved):
            expiries = sorted({str(item.get("expiration_date") or "") for item in resolved})
            return {
                "status": "VERIFIED",
                "message": "Every option leg was found as an exact tradable Alpaca contract.",
                "expiration_date": ",".join(expiries),
                "used_fallback_expiry": used_fallback,
                "legs": resolved,
            }
        return {
            "status": "PENDING",
            "message": next((item for item in errors if item), "One or more exact option legs are not tradable yet."),
            "expiration_date": expiry_date,
            "legs": [],
        }

    candidates = listed_expiry_fallbacks(expiry_date) or ([expiry_date] if expiry_date else [])
    for candidate in candidates:
        resolved: list[dict] = []
        for spec in leg_specs:
            contracts, err = alpaca.get_option_contracts(
                root,
                candidate or None,
                float(spec.get("strike") or 0),
                str(spec.get("side") or "").lower(),
            )
            if not contracts:
                errors.append(err)
                resolved = []
                break
            contract = next(
                (
                    item for item in contracts
                    if str(item.get("expiration_date") or "") == candidate
                    and item.get("tradable", True) is not False
                ),
                None,
            )
            if not contract:
                resolved = []
                break
            side, intent, requires_position = _option_leg_route(spec)
            resolved.append(
                {
                    **spec,
                    "symbol": str(contract.get("symbol") or "").upper(),
                    "side_order": side,
                    "position_intent": intent,
                    "requires_position": requires_position,
                    "expiration_date": str(contract.get("expiration_date") or candidate),
                }
            )
        if len(resolved) == len(leg_specs) and all(item.get("symbol") for item in resolved):
            return {
                "status": "VERIFIED",
                "message": "All option legs were found as one tradable Alpaca strategy.",
                "expiration_date": candidate,
                "used_fallback_expiry": bool(expiry_date and candidate != expiry_date),
                "legs": resolved,
            }

    if allow_nearest_expiry:
        contract_lists: list[list[dict]] = []
        for spec in leg_specs:
            contracts, err = alpaca.get_option_contracts(
                root,
                None,
                float(spec.get("strike") or 0),
                str(spec.get("side") or "").lower(),
            )
            if not contracts:
                errors.append(err)
                contract_lists = []
                break
            contract_lists.append([c for c in contracts if c.get("tradable", True) is not False])
        if contract_lists:
            common_expiries = set(str(c.get("expiration_date") or "") for c in contract_lists[0])
            for contracts in contract_lists[1:]:
                common_expiries &= {str(c.get("expiration_date") or "") for c in contracts}
            for candidate in sorted(x for x in common_expiries if x):
                resolved = []
                for spec, contracts in zip(leg_specs, contract_lists):
                    contract = next(
                        (c for c in contracts if str(c.get("expiration_date") or "") == candidate),
                        None,
                    )
                    if not contract:
                        resolved = []
                        break
                    side, intent, requires_position = _option_leg_route(spec)
                    resolved.append(
                        {
                            **spec,
                            "symbol": str(contract.get("symbol") or "").upper(),
                            "side_order": side,
                            "position_intent": intent,
                            "requires_position": requires_position,
                            "expiration_date": candidate,
                        }
                    )
                if len(resolved) == len(leg_specs):
                    return {
                        "status": "VERIFIED",
                        "message": "All option legs were found using the nearest common listed expiry.",
                        "expiration_date": candidate,
                        "used_fallback_expiry": True,
                        "legs": resolved,
                    }

    return {
        "status": "PENDING",
        "message": next((item for item in errors if item), "One or more exact option legs are not tradable yet."),
        "expiration_date": expiry_date,
        "legs": [],
    }


def _alpaca_multi_leg_payload(resolved_legs: list[dict]) -> list[dict]:
    return [
        {
            "symbol": leg["symbol"],
            "ratio_qty": str(max(1, int(leg.get("ratio_qty") or 1))),
            "side": leg["side_order"],
            "position_intent": leg["position_intent"],
        }
        for leg in resolved_legs
    ]


def _option_mapped_action(option: ParsedOptionSignal) -> str:
    if option.is_multi_leg:
        return _multi_leg_mapped_action(option)
    order_side, _, _ = _option_order_route(option)
    return "SELL" if order_side == "sell" else "BUY"


def _option_order_label(option: ParsedOptionSignal) -> str:
    return {
        "open_long": "buy-to-open",
        "close_long": "sell-to-close",
        "open_short": "sell-to-open",
        "close_short": "buy-to-close",
    }.get(str(option.order_action or "open_long").lower(), "buy-to-open")


def _option_signal_action_label(option: ParsedOptionSignal) -> str:
    if option.is_multi_leg:
        verb = "EXIT" if _multi_leg_mapped_action(option) == "SELL" else "ENTER"
        return f"{verb} {str(option.structure or 'multi-leg').replace('_', ' ').upper()}"
    action = str(option.order_action or "open_long").lower()
    side = str(option.side or "OPTION").upper()
    return {
        "open_long": f"BUY {side}",
        "close_long": f"SELL TO CLOSE {side}",
        "open_short": f"SELL TO OPEN {side}",
        "close_short": f"BUY TO CLOSE {side}",
        "manage": "MANAGE POSITION",
    }.get(action, _option_order_label(option).upper())


def _record_option_validation_metrics(option: ParsedOptionSignal, validation: dict) -> None:
    exact = validation.get("exact_strike_attempt") or {}
    recent_exact = validation.get("recent_exact_strike_attempt") or {}
    polygon_exact = validation.get("polygon_strike_attempt") or {}
    proxy = validation.get("delta_proxy_attempt") or {}
    si = validation.get("strategy_input") or {}
    record_option_validation_event(
        {
            "root": option.root,
            "side": option.side,
            "strike": option.strike,
            "expiry_date": option.expiry_date,
            "status": validation.get("status"),
            "decision": validation.get("decision"),
            "method": si.get("validation_method") or si.get("strike_selection"),
            "exact_status": exact.get("status"),
            "exact_message": str(exact.get("message") or "")[:220],
            "recent_exact_status": recent_exact.get("status"),
            "recent_exact_message": str(recent_exact.get("message") or "")[:220],
            "polygon_exact_status": polygon_exact.get("status"),
            "polygon_exact_message": str(polygon_exact.get("message") or "")[:220],
            "proxy_status": proxy.get("status"),
            "proxy_message": str(proxy.get("message") or "")[:220],
        }
    )


def _ai_decision_inputs(ai_prediction: dict) -> tuple[float, float, float]:
    predicted_return = _as_float(ai_prediction.get("predicted_return_pct"))
    confidence = _as_float(ai_prediction.get("confidence_score"))
    risk = _as_float(ai_prediction.get("risk_score"))
    return predicted_return, confidence, risk


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ai_rule_explanation(user_action: str, ai_prediction: dict) -> str:
    return _evaluate_final_action(user_action, ai_prediction).reason


def _decision_rule_text() -> str:
    return (
        "Flexible score + override gate\n"
        f"BUY: tiny +{config.buy_min_return_pct:.2f}% with excellent quality, "
        f"or strong +{config.buy_strong_return_pct:.2f}%, or score >= {config.buy_decision_score:.0f}\n"
        f"SELL: negative {config.sell_min_return_pct:.2f}% with high danger, "
        f"or strong {config.sell_strong_return_pct:.2f}%, or score >= {config.sell_decision_score:.0f}"
    )


def _adaptive_settings() -> tuple[dict, dict]:
    profile = get_learning_profile()
    market = get_market_context()
    return profile, market


def _signal_keywords(raw_text: str) -> list[str]:
    text = str(raw_text or "").lower()
    keywords = []
    for word in (
        "breakout", "breakdown", "scalp", "swing", "lotto", "eod", "volume",
        "earnings", "gap", "resistance", "support", "trend", "reversal",
        "strangle", "straddle", "spread", "intraday",
    ):
        if word in text:
            keywords.append(word)
    return keywords[:6]


def _dte_bucket(expiry_date: str) -> str:
    try:
        expiry = datetime.fromisoformat(str(expiry_date)).date()
        days = (expiry - datetime.utcnow().date()).days
    except Exception:
        return "unknown"
    if days <= 0:
        return "0dte"
    if days <= 7:
        return "weekly"
    if days <= 45:
        return "monthly"
    return "long_dated"


def _learning_features(
    raw_text: str,
    asset_type: str,
    action: str,
    symbol: str,
    option: Optional[ParsedOptionSignal] = None,
    equity: Optional[ParsedSignal] = None,
) -> dict:
    direction = _equity_decision_action(action) if equity else ""
    dte = "na"
    if option:
        direction = option.side or ""
        dte = _dte_bucket(option.expiry_date or "")
    features = {
        "asset_type": asset_type,
        "action": str(action or "").upper(),
        "symbol": str(symbol or "").upper(),
        "direction": direction,
        "keywords": _signal_keywords(raw_text),
        "dte_bucket": dte,
    }
    contract: dict = {}
    if option and option.semantic_contract:
        contract = option.semantic_contract
    elif equity:
        contract = _equity_contract(equity)
    if contract:
        status = str(contract.get("status") or "VALID").upper()
        features.update({
            "semantic_fields": sorted(contract.keys()),
            "contract_status": status,
            "conditional": "CONDITIONAL" in status,
            "position_management": "POSITION" in status or status.startswith("VALID_IF_"),
            "order_type": contract.get("order_type", "MANAGEMENT"),
            "strategy": (
                contract.get("strategy")
                or (option.structure if option else None)
                or ("stock_order" if equity else "single_leg")
            ),
            "leg_count": len(option.legs) if option and option.legs else 1,
        })
    return features


def _apply_learning_overlay(decision: DecisionResult, features: dict) -> DecisionResult:
    if not config.learning_enabled or decision.action not in {"BUY", "SELL"}:
        return decision
    pattern = get_pattern_learning(features)
    seen = int(pattern.get("seen") or 0)
    if seen < max(1, config.learning_min_samples):
        return decision
    approved = int(pattern.get("approved") or 0)
    blocked = int(pattern.get("blocked") or 0)
    approval_rate = approved / max(1, seen)
    max_adjust = max(0.0, config.learning_max_score_adjustment)

    if approval_rate < 0.30 and blocked >= approved:
        return DecisionResult(
            "HOLD",
            decision.reason
            + f" Learned pattern caution: only {approval_rate:.0%} approval after {seen} similar signals. "
            "Holding until this pattern improves.",
            decision.predicted_return,
            decision.confidence,
            decision.risk,
            max(0.0, decision.score - max_adjust),
            decision.market_regime,
        )

    if approval_rate >= 0.70:
        return DecisionResult(
            decision.action,
            decision.reason + f" Learned pattern support: {approval_rate:.0%} approval over {seen} similar signals.",
            decision.predicted_return,
            decision.confidence,
            decision.risk,
            min(100.0, decision.score + max_adjust),
            decision.market_regime,
        )
    return decision


def _learn_from_decision(features: dict, decision: DecisionResult, reason: str = "") -> None:
    if not config.learning_enabled:
        return
    record_learning_event(
        features,
        {
            "final_action": decision.action,
            "predicted_return_pct": decision.predicted_return,
            "confidence": decision.confidence,
            "risk": decision.risk,
            "score": decision.score,
            "reason": reason or decision.reason,
        },
    )


def _learn_from_input(raw_text: str, kind: str, action: str, symbol: str, reason: str = "") -> None:
    if not config.learning_enabled:
        return
    features = _learning_features(raw_text, str(kind or "input").lower(), action or kind, symbol)
    record_learning_event(
        features,
        {
            "final_action": str(kind or "INPUT").upper(),
            "predicted_return_pct": 0.0,
            "confidence": 0.0,
            "risk": 0.0,
            "score": 0.0,
            "reason": reason,
        },
    )


def _option_signal_quality(option: ParsedOptionSignal) -> dict:
    score = 100.0
    issues: list[str] = []
    critical: list[str] = []
    risk_reward = None

    if not option.expiry_date:
        score -= 15
        issues.append("expiry missing; default expiry will be used")
    else:
        try:
            dte = (datetime.fromisoformat(option.expiry_date).date() - datetime.utcnow().date()).days
            if dte < 0:
                score -= 100
                critical.append("expiry is already past")
            elif dte > config.max_option_dte:
                score -= 15
                issues.append(f"DTE {dte} is above configured max {config.max_option_dte}")
        except Exception:
            score -= 20
            issues.append("expiry could not be parsed")

    if not option.fill_price or option.fill_price <= 0:
        score -= 20
        issues.append("entry premium missing; market order may be less precise")

    price_effect = infer_multi_leg_price_effect(option) if option.is_multi_leg else "debit"
    short_credit = option.is_multi_leg and price_effect == "credit"
    if option.fill_price and option.stop_loss is not None:
        invalid_stop = option.stop_loss <= option.fill_price if short_credit else option.stop_loss >= option.fill_price
        if invalid_stop:
            score -= 100
            critical.append(
                "credit strategy SL must be above entry credit"
                if short_credit else "option SL must be below entry premium"
            )

    if option.fill_price and option.target_price is not None:
        invalid_target = option.target_price >= option.fill_price if short_credit else option.target_price <= option.fill_price
        if invalid_target:
            score -= 100
            critical.append(
                "credit strategy target must be below entry credit"
                if short_credit else "option target must be above entry premium"
            )

    if option.fill_price and option.stop_loss is not None and option.target_price is not None:
        risk = (option.stop_loss - option.fill_price) if short_credit else (option.fill_price - option.stop_loss)
        reward = (option.fill_price - option.target_price) if short_credit else (option.target_price - option.fill_price)
        if risk > 0:
            risk_reward = reward / risk
            if risk_reward < config.min_option_risk_reward:
                score -= 25
                critical.append(
                    f"risk/reward {risk_reward:.2f} is below minimum {config.min_option_risk_reward:.2f}"
                )

    if option.is_multi_leg:
        structural = validate_multi_leg_strategy(option)
        if not structural["passed"]:
            critical.extend(structural["issues"])
            score = 0.0
        issues.extend(structural["warnings"])

    score = max(0.0, min(100.0, score))
    return {
        "score": score,
        "issues": issues,
        "critical": critical,
        "risk_reward": risk_reward,
        "passed": score >= config.min_option_signal_quality and not critical,
    }


def _allowed_option_strategy_statuses() -> set[str]:
    return {
        "SUCCESS",
        "SUCCESS_POLYGON_STRIKE",
        "SUCCESS_EXACT_STRIKE_RECENT",
        "SUCCESS_DELTA_PROXY",
        "FALLBACK_APPROVED",
        "MULTI_LEG_STRUCTURAL_APPROVED",
        "AGENT_OFF_DIRECT",
    }


def _option_final_decision(strategy_validation: Optional[dict], quality: Optional[dict]) -> DecisionResult:
    validation = strategy_validation or {}
    q = quality or {}
    strategy_status = str(validation.get("status") or "FAILED").upper()
    strategy_decision = str(validation.get("decision") or "REVIEW").upper()
    bt = validation.get("backtest") or {}
    pnl = bt.get("profit_loss", bt.get("total_profit_loss", "-"))
    win_rate = bt.get("win_rate", "-")
    quality_score = float(q.get("score") or 0.0)

    if not q.get("passed", True):
        issues = "; ".join(list(q.get("critical") or []) + list(q.get("issues") or []))
        return DecisionResult(
            "REJECT",
            "Option signal needs review before entry.",
            0.0,
            0.0,
            0.0,
            quality_score,
            "options_validation",
        )

    if strategy_status not in _allowed_option_strategy_statuses():
        return DecisionResult(
            "REJECT",
            "Option signal needs review before entry.",
            0.0,
            0.0,
            0.0,
            quality_score,
            "options_validation",
        )

    if strategy_decision == "BUY":
        return DecisionResult(
            "BUY",
            "Option signal approved for paper-order preparation.",
            0.0,
            0.0,
            0.0,
            quality_score,
            "options_validation",
        )

    return DecisionResult(
        "REJECT",
        "Option signal reviewed. No paper entry selected.",
        0.0,
        0.0,
        0.0,
        quality_score,
        "options_validation",
    )


async def _send_channel(channel_id: int, content: str = "", embed: Optional[discord.Embed] = None) -> None:
    """Send a proactive/background alert (not a reply to any incoming message).

    Called from the periodic monitor loop -- protection triggers, a queued
    order finally filling, a contract becoming tradable, etc. There is no
    message object here to derive a WhatsApp destination from the way a reply
    does, so WHATSAPP_ALERT_TARGET is the explicit, always-on destination for
    this whole class of alert, mirroring the Discord channel id.
    """
    if channel_id:
        channel = bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(channel_id)
            except Exception:
                channel = None
        if channel is not None:
            try:
                await channel.send(content=content, embed=embed)
            except Exception as exc:
                logging.getLogger("discord_stock_prediction_agent").warning(
                    "Discord send failed; queue will continue. %s: %s",
                    type(exc).__name__,
                    exc,
                )
    if config.whatsapp_alert_target:
        text = output_text(content, embed)
        if text:
            await asyncio.to_thread(
                send_whatsapp_text,
                config.whatsapp_alert_target,
                text,
                is_group=config.whatsapp_alert_is_group,
            )


async def _send_optional_channel(
    channel_id: int,
    fallback_channel: discord.abc.Messageable,
    content: str = "",
    embed: Optional[discord.Embed] = None,
) -> None:
    """Send only to a configured channel. Never fall back into the input channel."""
    if str(getattr(fallback_channel, "transport", "discord")) == "whatsapp":
        text = output_text(content, embed)
        if text:
            await asyncio.to_thread(
                send_whatsapp_text,
                str(getattr(fallback_channel, "reply_target", "")),
                text,
                is_group=bool(getattr(fallback_channel, "is_group", False)),
            )
        return
    if channel_id:
        await _send_channel(channel_id, content=content, embed=embed)


async def _send_review_or_reply(
    message: discord.Message,
    content: str = "",
    embed: Optional[discord.Embed] = None,
) -> None:
    """Send bot output to signal-review. Never reply in the raw input channel."""
    if str(getattr(message, "transport", "discord")) == "whatsapp":
        text = output_text(content, embed)
        if text:
            ok, error = await asyncio.to_thread(
                send_whatsapp_text,
                str(getattr(message, "reply_target", "")),
                text,
                is_group=bool(getattr(message, "is_group", False)),
            )
            if not ok:
                logging.getLogger("discord_stock_prediction_agent").warning(
                    "WhatsApp response failed: %s", error
                )
        return
    if config.discord_review_channel_id:
        await _send_channel(config.discord_review_channel_id, content=content, embed=embed)


async def _send_context_output(ctx: commands.Context, content: str) -> None:
    """Keep command output out of the input channel when review is configured."""
    if config.discord_review_channel_id:
        await _send_channel(config.discord_review_channel_id, content=content)
    elif config.discord_signal_channel_id and ctx.channel.id == config.discord_signal_channel_id:
        return
    else:
        await ctx.reply(content)


def _display_decision_score(parsed: ParsedSignal, ai_prediction: dict, decision: DecisionResult) -> float:
    if decision.score > 0 or str(parsed.action or "").upper() != "HOLD":
        return decision.score
    predicted_return, _, _ = _ai_decision_inputs(ai_prediction or {})
    if predicted_return < 0:
        return _evaluate_final_action("SELL", ai_prediction).score
    return _evaluate_final_action("BUY", ai_prediction).score


def _stock_validation_summary(prediction: dict) -> str:
    actual = prediction.get("actual_validation") or {}
    comparison = prediction.get("comparison") or actual.get("comparison") or {}
    status = str(actual.get("status") or comparison.get("status") or "PENDING").upper()
    actual_decision = str(actual.get("actual_decision") or comparison.get("actual_decision") or "PENDING").upper()
    actual_return = actual.get("actual_return_pct", comparison.get("actual_return_pct"))
    agreement = str(comparison.get("agreement") or "UNVERIFIED").upper()
    match = comparison.get("decision_match")

    if actual_return is None:
        return f"Status: {status}\nActual Decision: {actual_decision}\nAgreement: {agreement}"
    match_text = "YES" if match is True else "NO" if match is False else "PENDING"
    try:
        return (
            f"Status: {status}\n"
            f"Actual Decision: {actual_decision} ({float(actual_return):+.2f}%)\n"
            f"Agreement: {agreement} | Match: {match_text}"
        )
    except (TypeError, ValueError):
        return (
            f"Status: {status}\n"
            f"Actual Decision: {actual_decision} ({actual_return})\n"
            f"Agreement: {agreement} | Match: {match_text}"
        )


def _prediction_embed(parsed: ParsedSignal, prediction: dict, decision: Optional[DecisionResult] = None) -> discord.Embed:
    ai_prediction = prediction.get("ai_prediction") or {}
    ai_decision = str(ai_prediction.get("decision") or "REVIEW").upper()
    predicted_return, confidence, risk = _ai_decision_inputs(ai_prediction)
    decision = decision or _evaluate_final_action(_equity_decision_action(parsed.action), ai_prediction)
    final_action = decision.action
    embed = discord.Embed(
        title=f"{parsed.symbol} Stock Prediction Review",
        description="Stock Price Validation checked before any Alpaca paper equity order.",
        color=_decision_color(final_action),
    )
    spi = prediction.get("stock_prediction_input") or {}
    embed.add_field(name="User Action", value=parsed.action, inline=True)
    embed.add_field(name="Final Agent Output", value=final_action, inline=True)
    embed.add_field(name="Qty", value=str(parsed.quantity or "Default"), inline=True)
    embed.add_field(name="Parsed Signal", value=_parsed_equity_summary(parsed), inline=False)
    embed.add_field(
        name="AI Stock Prediction",
        value=(
            f"{ai_decision} "
            f"({predicted_return:+.2f}%)"
        ),
        inline=True,
    )
    embed.add_field(
        name="Stock Price Validation",
        value=_stock_validation_summary(prediction),
        inline=True,
    )
    embed.add_field(
        name="Stock Decision Gate",
        value=_decision_rule_text(),
        inline=True,
    )
    display_score = _display_decision_score(parsed, ai_prediction, decision)
    embed.add_field(
        name="Confidence / Risk / Score",
        value=f"{confidence:.0f}/{risk:.0f}/{display_score:.0f}",
        inline=True,
    )
    embed.add_field(
        name="Prediction Window",
        value=f"{spi.get('prediction_origin_date', '-')} to {spi.get('target_date', '-')}",
        inline=True,
    )
    if parsed.reason:
        embed.add_field(name="Signal Note", value=parsed.reason[:500], inline=False)
    provider_warning = str(prediction.get("provider_warning") or ai_prediction.get("provider_warning") or "")
    if provider_warning:
        embed.add_field(
            name="Provider Note",
            value=f"Gemini unavailable; baseline project engine used. {provider_warning[:500]}",
            inline=False,
        )
    reason = str(ai_prediction.get("decision_reason") or ai_prediction.get("reasoning") or "")[:900]
    if reason:
        embed.add_field(name="Reason", value=reason, inline=False)
    embed.add_field(name="Trade Gate", value=_ai_rule_explanation(parsed.action, ai_prediction), inline=False)
    embed.set_footer(text="Paper trading only. No live-money orders are placed by this agent.")
    return embed


def _equity_contract(parsed: ParsedSignal) -> dict:
    if isinstance(parsed.order_intent, dict):
        return dict(parsed.order_intent)
    contract = {
        "asset_type": "STOCK",
        "action": parsed.action,
        "symbol": parsed.symbol,
        "quantity": parsed.quantity,
        "order_type": parsed.order_type.upper(),
        "time_in_force": parsed.time_in_force.upper(),
        "limit_price": parsed.limit_price,
        "stop_price": parsed.stop_price,
        "status": "VALID",
    }
    if parsed.condition_type:
        contract["entry_condition"] = {
            "type": parsed.condition_type,
            "price": parsed.condition_price,
        }
        contract["status"] = "CONDITIONAL"
    return {key: value for key, value in contract.items() if value is not None}


def _parsed_equity_summary(parsed: ParsedSignal) -> str:
    limit_price = f"${parsed.limit_price:.2f}" if parsed.limit_price is not None else "-"
    stop_price = f"${parsed.stop_price:.2f}" if parsed.stop_price is not None else "-"
    lines = [
        "Asset Type: STOCK | Status: VALID",
        f"Action: {parsed.action} | Symbol: {parsed.symbol}",
        f"Quantity: {parsed.quantity if parsed.quantity is not None else 'Default'}",
        f"Order Type: {parsed.order_type.upper()} | Time In Force: {parsed.time_in_force.upper()}",
        f"Limit Price: {limit_price} | Stop Price: {stop_price}",
    ]
    if parsed.condition_type:
        condition_price = (
            f"${parsed.condition_price:.2f}" if parsed.condition_price is not None else "-"
        )
        lines.append(f"Condition: {parsed.condition_type} {condition_price}")
    return "\n".join(lines)


def _direct_equity_embed(parsed: ParsedSignal, decision: DecisionResult) -> discord.Embed:
    embed = discord.Embed(
        title=f"{parsed.symbol} Direct Signal Review",
        description="Agent is OFF. This valid signal is being followed without an AI decision gate.",
        color=_decision_color(decision.action),
    )
    embed.add_field(name="Signal Action", value=parsed.action, inline=True)
    embed.add_field(name="Execution Output", value=decision.action, inline=True)
    embed.add_field(name="Qty", value=str(parsed.quantity or "Default"), inline=True)
    embed.add_field(name="Parsed Signal", value=_parsed_equity_summary(parsed), inline=False)
    embed.add_field(name="Agent Mode", value="OFF", inline=True)
    embed.add_field(name="Decision Source", value="Incoming signal", inline=True)
    embed.add_field(
        name="Broker Safety",
        value="Paper account, market/price conditions, positions, permissions, and Alpaca checks remain active.",
        inline=False,
    )
    if parsed.reason:
        embed.add_field(name="Signal Note", value=parsed.reason[:500], inline=False)
    embed.set_footer(text="Paper trading only. No live-money orders are placed by this agent.")
    return embed


def _equity_decision_action(action: str) -> str:
    return {
        "SELL_SHORT": "SELL",
        "BUY_TO_COVER": "BUY",
    }.get(str(action or "").upper(), str(action or "").upper())


async def _enqueue_parsed_equity(parsed: ParsedSignal, side: str, qty: float, reason: str) -> bool:
    """Queue a manual signal for submission once the market opens. Returns
    True if this created a new queue entry, False if an equity order for the
    same symbol/action was already queued (and this call was a no-op) --
    callers use this to tell the user "already queued" instead of silently
    doing nothing, so repeated signals while the market is closed don't pile
    up duplicate orders.
    """
    queued = await asyncio.to_thread(
        enqueue_market_order,
        parsed.symbol,
        side,
        qty,
        reason,
        config.stop_loss_pct,
        f"{parsed.action}:{parsed.symbol}",
        parsed.action,
        parsed.order_type,
        parsed.limit_price,
        parsed.stop_price,
        parsed.time_in_force,
    )
    return bool(queued.get("_inserted"))


async def _submit_parsed_equity(
    parsed: ParsedSignal, side: str, qty: float, client_order_id: str
) -> tuple[Optional[dict], str]:
    submitter = getattr(alpaca, "submit_equity_order", None)
    if submitter is None and parsed.order_type == "market":
        return await asyncio.to_thread(
            alpaca.submit_market_order, parsed.symbol, side, qty, client_order_id
        )
    if submitter is None:
        return None, f"Broker adapter does not support {parsed.order_type} equity orders."
    return await asyncio.to_thread(
        submitter,
        parsed.symbol,
        side,
        qty,
        parsed.order_type,
        parsed.limit_price,
        parsed.stop_price,
        parsed.time_in_force,
        client_order_id,
    )


def _evaluate_final_action(user_action: str, ai_prediction: dict) -> DecisionResult:
    """Apply a configurable AI-only score with explicit override paths."""
    user = str(user_action or "").upper()
    predicted_return, confidence, risk = _ai_decision_inputs(ai_prediction or {})

    base = (
        f"AI return {predicted_return:+.3f}%, confidence {confidence:.0f}, "
        f"risk {risk:.0f}. "
    )

    if user == "BUY":
        profile, market = _adaptive_settings()
        buy_min_return = max(0.001, config.buy_min_return_pct + _as_float(profile.get("buy_min_return_adjustment")))
        buy_score_needed = max(30.0, min(80.0, config.buy_decision_score + _as_float(profile.get("buy_score_adjustment"))))
        return_part = _clamp(predicted_return / max(config.buy_strong_return_pct, 0.01)) * 50
        confidence_part = _clamp(confidence / 100) * 30
        risk_part = _clamp((100 - risk) / 100) * 20
        score = return_part + confidence_part + risk_part + _as_float(market.get("score_adjustment_buy"))

        strong_return = predicted_return >= config.buy_strong_return_pct
        excellent_quality = (
            predicted_return >= buy_min_return
            and confidence >= config.buy_excellent_confidence
            and risk <= config.buy_low_risk
        )
        displayed_score = round(score)
        displayed_score_needed = round(buy_score_needed)
        score_pass = predicted_return >= buy_min_return and displayed_score >= displayed_score_needed

        if strong_return or excellent_quality or score_pass:
            if strong_return:
                why = "strong upside override"
            elif excellent_quality:
                why = "small positive return with excellent confidence/risk profile"
            else:
                why = "weighted decision score passed"
            return DecisionResult(
                "BUY",
                base + (
                    f"BUY allowed by {why}. Score {score:.0f}. "
                    f"Market regime {market.get('regime', 'unknown')}; learned score gate {buy_score_needed:.0f}."
                ),
                predicted_return,
                confidence,
                risk,
                score,
                str(market.get("regime", "unknown")),
            )
        return DecisionResult(
            "HOLD",
            base + (
                f"BUY not allowed. Score {score:.0f}; weighted score path needs return >= "
                f"+{buy_min_return:.3f}% and score >= {buy_score_needed:.0f}, "
                f"or return >= +{config.buy_strong_return_pct:.2f}%, "
                f"or return >= +{buy_min_return:.3f}% with confidence >= "
                f"{config.buy_excellent_confidence:.0f} and risk <= {config.buy_low_risk:.0f}. "
                f"Market regime {market.get('regime', 'unknown')}."
            ),
            predicted_return,
            confidence,
            risk,
            score,
            str(market.get("regime", "unknown")),
        )

    if user == "SELL":
        profile, market = _adaptive_settings()
        sell_min_return = min(-0.001, config.sell_min_return_pct + _as_float(profile.get("sell_min_return_adjustment")))
        sell_score_needed = max(30.0, min(80.0, config.sell_decision_score + _as_float(profile.get("sell_score_adjustment"))))
        downside_part = _clamp(abs(predicted_return) / max(abs(config.sell_strong_return_pct), 0.01)) * 50
        low_confidence_part = _clamp((100 - confidence) / 100) * 25
        high_risk_part = _clamp(risk / 100) * 25
        score = downside_part + low_confidence_part + high_risk_part + _as_float(market.get("score_adjustment_sell"))

        strong_downside = predicted_return <= config.sell_strong_return_pct
        danger_profile = (
            predicted_return < 0
            and confidence <= config.sell_low_confidence
            and risk >= config.sell_high_risk
        )
        displayed_score = round(score)
        displayed_score_needed = round(sell_score_needed)
        score_pass = predicted_return <= sell_min_return and displayed_score >= displayed_score_needed

        if strong_downside or danger_profile or score_pass:
            if strong_downside:
                why = "strong downside override"
            elif danger_profile:
                why = "negative return with low confidence and high risk"
            else:
                why = "weighted danger score passed"
            return DecisionResult(
                "SELL",
                base + (
                    f"SELL allowed by {why}. Score {score:.0f}. "
                    f"Market regime {market.get('regime', 'unknown')}; learned score gate {sell_score_needed:.0f}."
                ),
                predicted_return,
                confidence,
                risk,
                score,
                str(market.get("regime", "unknown")),
            )
        return DecisionResult(
            "HOLD",
            base + (
                f"SELL not allowed. Score {score:.0f}; weighted score path needs return <= "
                f"{sell_min_return:.3f}% and score >= {sell_score_needed:.0f}, "
                f"or return <= {config.sell_strong_return_pct:.2f}%, "
                f"or any negative return with confidence <= "
                f"{config.sell_low_confidence:.0f} and risk >= {config.sell_high_risk:.0f}. "
                f"Market regime {market.get('regime', 'unknown')}."
            ),
            predicted_return,
            confidence,
            risk,
            score,
            str(market.get("regime", "unknown")),
        )

    if user == "HOLD":
        return DecisionResult(
            "HOLD",
            base + "HOLD signal received, so no Alpaca paper trade will be placed.",
            predicted_return,
            confidence,
            risk,
            0.0,
            "not_used",
        )

    return DecisionResult(
        "HOLD",
        base + "Unsupported action, so no Alpaca paper trade will be placed.",
        predicted_return,
        confidence,
        risk,
        0.0,
        "not_used",
    )


def _final_action(user_action: str, ai_prediction: dict) -> str:
    return _evaluate_final_action(user_action, ai_prediction).action


async def _handle_buy(message: discord.Message, parsed: ParsedSignal, prediction: dict) -> None:
    is_cover = parsed.action == "BUY_TO_COVER"
    quantity = parsed.quantity
    if is_cover:
        position, _ = await asyncio.to_thread(alpaca.get_position, parsed.symbol)
        held = _as_float((position or {}).get("qty"))
        if held >= 0:
            await _send_review_or_reply(
                message, f"{parsed.symbol}: no short position exists to buy to cover."
            )
            return
        quantity = min(quantity or abs(held), abs(held))
    quantity = quantity or 1.0
    if not await _trade_guard(message, parsed.symbol, "buy", quantity, "equity"):
        return

    market_open, market_err = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        newly_queued = await _enqueue_parsed_equity(parsed, "buy", quantity, market_err or "market_closed")
        await asyncio.to_thread(_record_order, parsed.symbol, "buy", quantity, "queued", "", "equity", "market_closed")
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: a paper BUY is already queued and waiting for market open; "
            "not adding a duplicate."
            if not newly_queued
            else f"{parsed.symbol}: paper BUY approved, but the market is closed. "
            "It has been queued and will be submitted when Alpaca reports the market is open."
        )
        return

    order, err = await _submit_parsed_equity(
        parsed, "buy", quantity, _client_order_id("equitybuy", message.id)
    )
    if not order:
        if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
            newly_queued = await _enqueue_parsed_equity(parsed, "buy", quantity, err)
            await asyncio.to_thread(_record_order, parsed.symbol, "buy", quantity, "queued", "", "equity", "broker_retry")
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: a paper BUY is already queued and waiting for market open; "
                "not adding a duplicate."
                if not newly_queued
                else f"{parsed.symbol}: paper BUY approved and queued because Alpaca is temporarily "
                "unavailable or the market is closed. The agent will retry safely."
            )
            return
        await _block_trade(message, parsed.symbol, f"Paper BUY was not placed. {_public_error(err)}", "alpaca_reject")
        return

    order_id = str(order.get("id") or "")
    await asyncio.to_thread(_record_order, parsed.symbol, "buy", quantity, "submitted", order_id, "equity")
    if is_cover:
        await _track_submitted_exit(order, parsed.symbol, quantity, "equity", "buy_to_cover")
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: BUY TO COVER submitted for {quantity:g} share(s). "
            f"Order type {parsed.order_type.upper()}, Alpaca order ID `{order_id or '-'}`.",
        )
        return
    checked = order
    if order_id:
        checked, _ = await asyncio.to_thread(alpaca.wait_for_order, order_id, 8)
        checked = checked or order

    order_status = str(checked.get("status") or "").lower()
    entry_price = _as_float(checked.get("filled_avg_price"))
    filled_qty = _as_float(checked.get("filled_qty"))
    if order_status not in {"filled", "partially_filled"}:
        entry_price = 0.0
        filled_qty = 0.0
    if entry_price <= 0 and order_status in {"filled", "partially_filled"}:
        position, _ = await asyncio.to_thread(alpaca.get_position, parsed.symbol)
        if position:
            entry_price = _as_float(position.get("avg_entry_price"))
            filled_qty = min(_as_float(position.get("qty"), quantity), quantity)

    if entry_price > 0 and filled_qty > 0:
        await asyncio.to_thread(
            upsert_position,
            parsed.symbol,
            filled_qty,
            entry_price,
            order_id,
            config.equity_stop_loss_pct,
            config.equity_take_profit_pct,
        )
        levels = build_protection_levels(
            entry_price,
            stop_loss_pct=config.equity_stop_loss_pct,
            take_profit_pct=config.equity_take_profit_pct,
        )
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: paper BUY placed for {filled_qty:g} share(s). "
            f"Entry approx ${entry_price:.2f}. Loss protection ${levels.stop_price:.2f} "
            f"(-{config.equity_stop_loss_pct:.1f}%); profit protection "
            f"${levels.target_price:.2f} (+{config.equity_take_profit_pct:.1f}%)."
        )
        if order_status == "partially_filled" and order_id:
            await asyncio.to_thread(
                add_pending_buy,
                parsed.symbol,
                quantity,
                order_id,
                config.equity_stop_loss_pct,
                filled_qty,
                config.equity_take_profit_pct,
            )
    else:
        if order_id:
            await asyncio.to_thread(
                add_pending_buy,
                parsed.symbol,
                quantity,
                order_id,
                config.equity_stop_loss_pct,
                0.0,
                config.equity_take_profit_pct,
            )
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: paper BUY order submitted. Status: "
            f"`{checked.get('status', 'submitted')}`. Protection will activate after Alpaca "
            "confirms the order is filled."
        )

    await _send_optional_channel(
        config.discord_paper_log_channel_id,
        message.channel,
        f"{parsed.symbol}: BUY order submitted. Qty {quantity:g}. Order ID `{order_id or '-'}`.",
    )


async def _handle_sell(message: discord.Message, parsed: ParsedSignal, prediction: dict) -> None:
    is_short = parsed.action == "SELL_SHORT"
    quantity = parsed.quantity
    if is_short and quantity is None:
        quantity = 1.0
    if quantity is None:
        position, pos_err = await asyncio.to_thread(alpaca.get_position, parsed.symbol)
        if not position:
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: no Alpaca position found, so nothing will be sold."
            )
            return
        quantity = _as_float(position.get("qty"))
        if quantity <= 0:
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: Alpaca position has no sellable quantity, so nothing will be sold."
            )
            return

    if quantity <= 0:
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: sell quantity is invalid, so nothing will be sold."
        )
        return
    if not await _trade_guard(message, parsed.symbol, "sell", quantity, "equity"):
        return

    market_open, clock_err = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        newly_queued = await _enqueue_parsed_equity(parsed, "sell", quantity, "manual_sell_market_closed")
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: a paper SELL is already queued and waiting for market open; "
            "not adding a duplicate."
            if not newly_queued
            else f"{parsed.symbol}: market is closed, so the SELL is queued. "
            "The agent will attempt it when Alpaca market opens."
        )
        return

    open_order, open_err = await asyncio.to_thread(alpaca.has_open_order, parsed.symbol)
    if open_order:
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: paper SELL skipped because Alpaca already has an open order for this symbol."
        )
        return

    held = 0.0
    if not is_short:
        ok, held, reason = await asyncio.to_thread(
            alpaca.has_sellable_quantity, parsed.symbol, quantity
        )
        if not ok:
            await _send_review_or_reply(message, f"{parsed.symbol}: paper SELL skipped. {reason}")
            return

    order, err = await _submit_parsed_equity(
        parsed, "sell", quantity, _client_order_id("equitysell", message.id)
    )
    if not order:
        if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
            newly_queued = await _enqueue_parsed_equity(parsed, "sell", quantity, err or "broker_retry")
            await asyncio.to_thread(
                _record_order,
                parsed.symbol,
                "sell",
                quantity,
                "queued",
                "",
                "equity",
                "broker_retry",
            )
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: a paper SELL is already queued and waiting for market open; "
                "not adding a duplicate."
                if not newly_queued
                else f"{parsed.symbol}: paper SELL approved and queued for a safe Alpaca retry.",
            )
            return
        await _block_trade(message, parsed.symbol, f"Paper SELL was not placed. {_public_error(err)}", "alpaca_reject")
        return

    order_id = str(order.get("id") or "")
    await asyncio.to_thread(_record_order, parsed.symbol, "sell", quantity, "submitted", order_id, "equity")
    if not is_short:
        await _track_submitted_exit(
            order, parsed.symbol, quantity, "equity", "manual_sell"
        )
    checked = order
    if order_id:
        checked, _ = await asyncio.to_thread(alpaca.wait_for_order, order_id, 8)
        checked = checked or order
        await _reconcile_pending_exit_orders()

    short_protection_note = ""
    if is_short:
        entry_price = _as_float(checked.get("filled_avg_price"))
        filled_qty = _as_float(checked.get("filled_qty"))
        if entry_price > 0 and filled_qty > 0:
            await asyncio.to_thread(
                upsert_position,
                parsed.symbol,
                filled_qty,
                entry_price,
                order_id,
                config.equity_stop_loss_pct,
                config.equity_take_profit_pct,
                "short",
            )
            levels = build_protection_levels(
                entry_price,
                stop_loss_pct=config.equity_stop_loss_pct,
                take_profit_pct=config.equity_take_profit_pct,
                short_position=True,
            )
            short_protection_note = (
                f" Entry approx ${entry_price:.2f}. Loss protection ${levels.stop_price:.2f} "
                f"(+{config.equity_stop_loss_pct:.1f}%); profit protection "
                f"${levels.target_price:.2f} (-{config.equity_take_profit_pct:.1f}%)."
            )
        else:
            short_protection_note = " Short position state will be confirmed by Alpaca."

    await _send_review_or_reply(
        message,
        f"{parsed.symbol}: paper {'SHORT SELL' if is_short else 'SELL'} submitted for {quantity:g} share(s). "
        + (short_protection_note.strip()
           if is_short else f"Alpaca position before order: {held:g} share(s). Position state changes only after Alpaca confirms fills.")
    )
    await _send_optional_channel(
        config.discord_paper_log_channel_id,
        message.channel,
        f"{parsed.symbol}: SELL order submitted. Qty {quantity:g}. Order ID `{order_id or '-'}`.",
    )


def _rich_intent_direction(intent: dict) -> str:
    actions = [str(intent.get("action") or "").upper()]
    actions.extend(
        str(item.get("action") or "").upper()
        for item in intent.get("actions", [])
        if isinstance(item, dict)
    )
    if any(action in {"SELL", "SELL_SHORT", "SCALE_OUT"} for action in actions):
        return "SELL"
    if any(action in {"BUY", "BUY_TO_COVER", "LADDER_BUY", "MIXED_ENTRY"} for action in actions):
        return "BUY"
    return "HOLD"


async def _process_rich_stock_order(message: discord.Message, parsed: ParsedSignal) -> bool:
    """Review, gate, and submit a normalized rich stock order when present."""
    intent = parsed.order_intent
    if not isinstance(intent, dict):
        return False

    mode = "ON" if _agent_is_enabled() else "OFF"
    await _send_review_or_reply(
        message,
        f"{parsed.symbol} normalized stock-order fields (Agent {mode}):",
    )
    for chunk in stock_review_chunks(intent):
        await _send_review_or_reply(message, chunk)

    legacy_market_fields = {
        "asset_type", "action", "symbol", "quantity", "quantity_type", "order_type", "status",
    }
    if (
        str(intent.get("action") or "").upper() in {"BUY", "SELL"}
        and str(intent.get("order_type") or "").upper() == "MARKET"
        and set(intent).issubset(legacy_market_fields)
    ):
        # Preserve the mature market-order queue, fill tracking, position state,
        # and default protection path while still showing every parsed field.
        return False

    direction = _rich_intent_direction(intent)
    features = _learning_features(
        message.content, "equity", parsed.action, parsed.symbol, equity=parsed
    )
    prediction: dict = {}
    if mode == "ON":
        prediction = await asyncio.to_thread(run_project_prediction, parsed.symbol)
        if prediction.get("status") != "SUCCESS":
            await _block_trade(
                message,
                parsed.symbol,
                "Agent ON could not produce a valid stock decision. The parsed order fields were retained, but no Alpaca order was placed.",
                "rich_stock_prediction",
            )
            return True
        decision = _apply_learning_overlay(
            _evaluate_final_action(direction, prediction.get("ai_prediction") or {}),
            features,
        )
        agent_decision = decision.action
        await _send_optional_channel(
            config.discord_review_channel_id,
            message.channel,
            embed=_prediction_embed(parsed, prediction, decision),
        )
    else:
        agent_decision = direction
        decision = _direct_signal_decision(direction)
        await _send_optional_channel(
            config.discord_review_channel_id,
            message.channel,
            embed=_direct_equity_embed(parsed, decision),
        )

    await asyncio.to_thread(_learn_from_decision, features, decision)
    await asyncio.to_thread(
        add_decision_history,
        {
            "symbol": parsed.symbol,
            "kind": "EQUITY",
            "user_action": parsed.action,
            "ai_decision": str((prediction.get("ai_prediction") or {}).get("decision") or "BYPASSED").upper(),
            "final_action": decision.action,
            "predicted_return_pct": round(decision.predicted_return, 4),
            "confidence": round(decision.confidence, 4),
            "risk": round(decision.risk, 4),
            "score": round(decision.score, 4),
            "market_regime": decision.market_regime,
            "reason": decision.reason,
            "raw_input": parsed.raw_text or message.content,
            "parsed_contract": _equity_contract(parsed),
            "agent_mode": mode,
        },
    )

    needs_position = (
        str(intent.get("status") or "").startswith("VALID_IF_")
        or intent.get("close_percentage") is not None
        or intent.get("quantity_scope") is not None
        or str(intent.get("action") or "").upper() in {"SELL", "BUY_TO_COVER", "SCALE_OUT"}
        or any(
            str(item.get("action") or "").upper() in {"SELL", "BUY_TO_COVER", "SCALE_OUT"}
            for item in intent.get("actions", [])
            if isinstance(item, dict)
        )
    )
    position_quantity = None
    if needs_position:
        position, position_error = await asyncio.to_thread(alpaca.get_position, parsed.symbol)
        if not position:
            await _block_trade(
                message,
                parsed.symbol,
                f"This order requires an existing Alpaca position. {_public_error(position_error)}",
                "rich_stock_position",
            )
            return True
        position_quantity = abs(_as_float(position.get("qty")))
        if position_quantity <= 0:
            await _block_trade(
                message,
                parsed.symbol,
                "The Alpaca position has no usable quantity, so no order was placed.",
                "rich_stock_position",
            )
            return True

    plan = gate_order_plan(
        intent,
        agent_mode=mode,
        agent_decision=agent_decision,
        position_quantity=position_quantity,
    )
    if plan.blocked_reasons:
        await _block_trade(
            message,
            parsed.symbol,
            " ".join(plan.blocked_reasons),
            "rich_stock_gate",
        )
        return True

    if position_quantity is not None:
        requested_close_qty = sum(
            _as_float((operation.get("payload") or {}).get("qty"))
            for operation in plan.operations
            if operation.get("operation") == "submit_order"
            and str((operation.get("payload") or {}).get("side") or "").lower()
            == ("buy" if str(intent.get("action") or "").upper() == "BUY_TO_COVER" else "sell")
        )
        if requested_close_qty > position_quantity + 1e-9:
            await _block_trade(
                message,
                parsed.symbol,
                f"Requested close quantity {requested_close_qty:g} exceeds the Alpaca position quantity {position_quantity:g}.",
                "rich_stock_position",
            )
            return True

    for operation in plan.operations:
        if operation.get("operation") != "submit_order":
            continue
        payload = operation.get("payload") or {}
        side = str(payload.get("side") or "").lower()
        qty = _as_float(payload.get("qty"), 1.0)
        if not await _trade_guard(message, parsed.symbol, side, qty or 1.0, "equity"):
            return True

    result = await asyncio.to_thread(
        execute_alpaca_order_plan,
        alpaca,
        plan,
        client_order_id_factory=lambda index: _client_order_id(
            f"rich{index}", message.id
        ),
    )
    if result.get("errors"):
        await _block_trade(
            message,
            parsed.symbol,
            " ".join(str(error) for error in result["errors"]),
            "rich_stock_execution",
        )
        return True

    for order in result.get("submitted", []):
        quantity = _as_float(order.get("qty") or order.get("notional"))
        await asyncio.to_thread(
            _record_order,
            parsed.symbol,
            str(order.get("side") or direction).lower(),
            quantity,
            str(order.get("status") or "submitted"),
            str(order.get("id") or ""),
            "equity",
            "rich_stock_order",
        )
    await _send_review_or_reply(
        message,
        f"{parsed.symbol}: submitted {len(result.get('submitted', []))} Alpaca paper order(s); "
        f"cancelled {len(result.get('cancelled', []))} prior order(s). "
        f"All {len(result.get('considered_fields', []))} normalized leaf fields were reviewed.",
    )
    return True


async def _process_signal(message: discord.Message, parsed: ParsedSignal) -> None:
    if await _process_rich_stock_order(message, parsed):
        return
    decision_action = _equity_decision_action(parsed.action)
    if parsed.action in {"BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"} and parsed.condition_type and parsed.condition_price:
        await asyncio.to_thread(
            add_conditional_equity_order,
            {
                "symbol": parsed.symbol,
                "action": parsed.action,
                "quantity": parsed.quantity,
                "condition_type": parsed.condition_type,
                "condition_price": parsed.condition_price,
                "order_type": parsed.order_type,
                "limit_price": parsed.limit_price,
                "stop_price": parsed.stop_price,
                "time_in_force": parsed.time_in_force,
                "raw_input": parsed.raw_text or getattr(message, "content", ""),
                "user_id": str(getattr(message.author, "id", "")),
                "channel_id": str(getattr(message.channel, "id", "")),
                "message_id": str(getattr(message, "id", "")),
            },
        )
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: conditional {parsed.action} signal accepted. "
            f"Watching for {parsed.condition_type.replace('_', ' ')} ${parsed.condition_price:g} "
            f"inside the configured near-price band. "
            "The signal will use the active Agent ON/OFF mode after the price condition is met.",
        )
        return

    if not _agent_is_enabled():
        decision = _direct_signal_decision(decision_action)
        features = _learning_features(
            message.content, "equity", parsed.action, parsed.symbol, equity=parsed
        )
        await _send_optional_channel(
            config.discord_review_channel_id,
            message.channel,
            embed=_direct_equity_embed(parsed, decision),
        )
        await asyncio.to_thread(_learn_from_decision, features, decision)
        await asyncio.to_thread(
            add_decision_history,
            {
                "symbol": parsed.symbol,
                "user_action": parsed.action,
                "ai_decision": "BYPASSED",
                "final_action": decision.action,
                "score": decision.score,
                "market_regime": decision.market_regime,
                "reason": decision.reason,
                "raw_input": parsed.raw_text or message.content,
                "parsed_contract": _equity_contract(parsed),
                "agent_mode": "OFF",
            },
        )
        if parsed.action == "HOLD":
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: HOLD signal received in Agent OFF mode. No paper order is required.",
            )
            return
        if parsed.action in {"BUY", "BUY_TO_COVER"} and decision.action == "BUY":
            await _handle_buy(message, parsed, {})
        elif parsed.action in {"SELL", "SELL_SHORT"} and decision.action == "SELL":
            await _handle_sell(message, parsed, {})
        return

    prediction = await asyncio.to_thread(run_project_prediction, parsed.symbol)
    if prediction.get("status") != "SUCCESS":
        await _send_optional_channel(
            config.discord_review_channel_id,
            message.channel,
            embed=_prediction_embed(parsed, prediction),
        )
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: prediction failed, so no paper trade will be placed. "
            f"Reason: {prediction.get('error', 'Unknown error')}"
        )
        return

    ai_prediction = prediction.get("ai_prediction") or {}
    ai_decision = str(ai_prediction.get("decision") or "REVIEW").upper()
    features = _learning_features(
        message.content, "equity", parsed.action, parsed.symbol, equity=parsed
    )
    decision = _apply_learning_overlay(_evaluate_final_action(decision_action, ai_prediction), features)
    final_action = decision.action
    await _send_optional_channel(
        config.discord_review_channel_id,
        message.channel,
        embed=_prediction_embed(parsed, prediction, decision),
    )
    await asyncio.to_thread(_learn_from_decision, features, decision)
    await asyncio.to_thread(
        add_decision_history,
        {
            "symbol": parsed.symbol,
            "user_action": parsed.action,
            "ai_decision": ai_decision,
            "final_action": final_action,
            "predicted_return_pct": round(decision.predicted_return, 4),
            "confidence": round(decision.confidence, 4),
            "risk": round(decision.risk, 4),
            "score": round(decision.score, 4),
            "market_regime": decision.market_regime,
            "reason": decision.reason,
            "raw_input": parsed.raw_text or message.content,
            "parsed_contract": _equity_contract(parsed),
            "agent_mode": "ON",
        },
    )

    if final_action == "REVIEW":
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: needs REVIEW. AI={ai_decision}. "
            "No paper trade will be placed."
        )
        return

    if parsed.action == "HOLD":
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: HOLD signal processed. AI={ai_decision}. "
            "No paper trade will be placed from a HOLD signal."
        )
        return

    if final_action == "HOLD":
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: final output is HOLD. {decision.reason} "
            "No paper trade will be placed."
        )
        return

    if parsed.action in {"BUY", "BUY_TO_COVER"} and final_action == "BUY":
        await _handle_buy(message, parsed, prediction)
    elif parsed.action in {"SELL", "SELL_SHORT"} and final_action == "SELL":
        await _handle_sell(message, parsed, prediction)


def _resolved_option_qty(option: ParsedOptionSignal) -> float:
    # Explicit quantities in the message always win; otherwise fall back to
    # the configured default (which itself defaults to 1 contract).
    return option.quantity if option.quantity != 1.0 else config.default_option_qty


def _option_label(option: ParsedOptionSignal) -> str:
    if option.is_multi_leg and option.legs:
        return f"{option.root} {str(option.structure or 'multi-leg').replace('_', ' ').title()} {option.expiry_date or '?'}"
    side_letter = "C" if option.side == "CALL" else "P"
    if option.strike is None and option.delta_target is not None:
        return f"{option.root} {option.delta_target:g}D {side_letter} {option.expiry_date or '?'}"
    if option.strike is None:
        return f"{option.root} {side_letter} {option.expiry_date or '?'}"
    return f"{option.root} {option.strike:g}{side_letter} {option.expiry_date or '?'}"


def _verify_exact_option_contract(option: ParsedOptionSignal) -> dict:
    if option.strike is None and option.delta_target is not None:
        return {
            "status": "SKIPPED_DELTA_SIGNAL",
            "message": "Delta signal: no exact strike was provided, so exact Alpaca contract lookup is skipped.",
            "contracts": None,
        }
    if not alpaca.ready():
        return {
            "status": "SKIPPED",
            "message": "Alpaca paper trading is not configured, so exact contract lookup was skipped.",
            "contracts": None,
        }
    if option.strike is None:
        return {
            "status": "FAILED",
            "message": "No strike was parsed from the signal.",
            "contracts": None,
        }

    contracts = None
    lookup_err = ""
    used_fallback_expiry = False
    for expiry_candidate in listed_expiry_fallbacks(option.expiry_date or "") or [option.expiry_date]:
        contracts, lookup_err = alpaca.get_option_contracts(
            option.root, expiry_candidate, option.strike, option.side.lower()
        )
        if contracts:
            used_fallback_expiry = expiry_candidate != option.expiry_date
            break
    if not contracts and option.expiry_mode == "0dte":
        contracts, lookup_err = alpaca.get_option_contracts(
            option.root, None, option.strike, option.side.lower()
        )
        if contracts:
            contracts = sorted(contracts, key=lambda c: str(c.get("expiration_date") or ""))
            used_fallback_expiry = True

    if not contracts:
        hint = (
            " This root is a cash-settled index that Alpaca likely does not offer options on directly."
            if likely_unsupported_by_alpaca(option.root) else ""
        )
        return {
            "status": "FAILED",
            "message": f"{lookup_err}{hint}",
            "contracts": None,
        }

    contract = contracts[0]
    occ_symbol = str(contract.get("symbol") or "")
    expiration = str(contract.get("expiration_date") or option.expiry_date or "")
    if not occ_symbol:
        return {
            "status": "FAILED",
            "message": "Alpaca contract lookup succeeded but returned no OCC symbol.",
            "contracts": contracts,
        }
    return {
        "status": "VERIFIED",
        "message": (
            f"Exact Alpaca contract found: {occ_symbol}"
            + (" using nearest listed expiry." if used_fallback_expiry else ".")
        ),
        "occ_symbol": occ_symbol,
        "expiration_date": expiration,
        "used_fallback_expiry": used_fallback_expiry,
        "contracts": contracts,
    }

def _option_public_basis(strategy_validation: Optional[dict], option: ParsedOptionSignal) -> str:
    if not strategy_validation:
        return "Signal review"
    status = str(strategy_validation.get("status") or "").upper()
    si = strategy_validation.get("strategy_input") or {}
    if status == "AGENT_OFF_DIRECT":
        return "Incoming signal (Agent OFF)"
    if status == "SUCCESS_POLYGON_STRIKE":
        return "Exact strike historical check"
    if status in {"SUCCESS", "SUCCESS_EXACT_STRIKE_RECENT"}:
        return "Exact strike strategy check" if si.get("strike_selection") == "strike" else "Delta strategy check"
    if status == "FALLBACK_APPROVED":
        return "Exact contract + signal quality check"
    if status == "MULTI_LEG_STRUCTURAL_APPROVED":
        return "Complete strategy + exact-contract gate (historical trials unavailable)"
    if status == "SUCCESS_DELTA_PROXY":
        return "Strategy check"
    return "Signal quality review"


def _option_public_result(strategy_validation: Optional[dict], decision: "DecisionResult") -> str:
    if decision.action == "BUY":
        return "APPROVED"
    if decision.action == "SELL":
        return "EXIT / SELL"
    return "REVIEW"


def _option_public_trade_gate(decision: "DecisionResult") -> str:
    if decision.action == "BUY":
        return "Trade approved. The agent will continue with paper-order preparation."
    if decision.action == "SELL":
        return "Sell-side option action approved. The agent will continue with paper-order preparation."
    return "No paper order will be placed for this signal."


def _agent_response_label(decision: "DecisionResult") -> str:
    action = str(decision.action or "").upper()
    if action in {"BUY", "SELL"}:
        return action
    return "HOLD"


def _option_action_code(option: ParsedOptionSignal) -> str:
    if option.is_multi_leg:
        return _option_signal_action_label(option).replace(" ", "_")
    return {
        "open_long": "BUY_TO_OPEN",
        "close_long": "SELL_TO_CLOSE",
        "open_short": "SELL_TO_OPEN",
        "close_short": "BUY_TO_CLOSE",
        "manage": "MANAGE_POSITION",
    }.get(str(option.order_action or "").lower(), "REVIEW")


def _option_targets(option: ParsedOptionSignal) -> list[float]:
    targets: list[float] = []
    for value in list(option.target_prices or ()) + [option.target_price]:
        price = _as_float(value)
        if price > 0 and price not in targets:
            targets.append(price)
    return targets


def _parsed_option_summary(option: ParsedOptionSignal) -> str:
    strike = f"{option.strike:g}" if option.strike is not None else "-"
    entry = f"${option.fill_price:.2f}" if option.fill_price is not None else "Market"
    stop = f"${option.stop_loss:.2f}" if option.stop_loss is not None else "Not provided"
    targets = _option_targets(option)
    target_text = (
        " | ".join(f"TP{index} ${price:.2f}" for index, price in enumerate(targets, start=1))
        if targets else "Not provided"
    )
    details = [
        f"Asset Type: OPTION | Status: {'VALID' if option.valid else 'INVALID'}\n"
        f"Action: {_option_action_code(option)} | Symbol: {option.root or '-'}\n"
        f"Option Type: {option.side or '-'} | Strike: {strike}\n"
        f"Expiration: {option.expiry_date or '-'} | Qty: {_resolved_option_qty(option):g}\n"
        f"Order Type: {_option_requested_order_type(option).upper()} | Entry: {entry}\n"
        f"Take Profit: {target_text}\n"
        f"Stop Loss: {stop}"
    ]
    if option.position_type:
        details.append(f"Position Type: {option.position_type.title()}")
    if option.entry_price_type:
        details.append(f"Entry Price Type: {option.entry_price_type.title()}")
    if option.time_in_force:
        details.append(f"Requested Time In Force: {option.time_in_force}")
    if option.underlying_trigger_price is not None:
        operator = ">" if option.underlying_trigger_direction == "above" else "<"
        details.append(
            f"Entry Condition: {option.root} underlying price {operator} "
            f"${option.underlying_trigger_price:g}"
        )
    if option.entry_premium_price is not None:
        operator = ">" if option.entry_premium_direction == "above" else "<"
        details.append(f"Entry Condition: option premium {operator} ${option.entry_premium_price:.2f}")
    if option.close_percent is not None:
        details.append(
            f"Position Close: {option.close_percent:g}%"
            + ("; keep remaining contracts open" if option.remaining_instruction == "KEEP_OPEN" else "")
        )
    if option.risk_stop_pct is not None:
        details.append(f"Risk Stop: {option.risk_stop_pct:g}% below the filled option premium")
    if option.add_trigger_premium is not None:
        add_qty = option.add_quantity or _resolved_option_qty(option)
        details.append(
            f"Scale In: add {add_qty:g} contract(s) if option premium falls to "
            f"${option.add_trigger_premium:.2f}"
        )
    if option.add_trigger_underlying_price is not None:
        add_qty = option.add_quantity or _resolved_option_qty(option)
        symbol = ">" if option.add_trigger_underlying_direction == "above" else "<"
        details.append(
            f"Scale In: add {add_qty:g} contract(s) when {option.root} {symbol} "
            f"${option.add_trigger_underlying_price:g}"
        )
    if option.exit_before_market_close:
        minutes = option.exit_minutes_before_close or 15
        qualifier = " if target has not been hit" if option.exit_if_target_not_hit else ""
        details.append(f"Time Exit: close remaining contracts {minutes} minutes before market close{qualifier}")
    if option.exit_underlying_price is not None:
        operator = ">" if option.exit_underlying_direction == "above" else "<"
        details.append(
            f"Planned Exit: close position when {option.root} underlying price {operator} "
            f"${option.exit_underlying_price:g}"
        )
    if option.maximum_loss_amount is not None:
        details.append(f"Maximum Position Loss: ${option.maximum_loss_amount:,.2f}")
    if option.stop_scope:
        details.append(f"Stop Scope: {option.stop_scope.replace('_', ' ').title()}")
    return "\n".join(details)


def _flatten_contract(value: object, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            label = str(key).replace("_", " ").title()
            path = f"{prefix} / {label}" if prefix else label
            if isinstance(nested, (dict, list)):
                lines.extend(_flatten_contract(nested, path))
            else:
                lines.append(f"{path}: {nested}")
    elif isinstance(value, list):
        for index, nested in enumerate(value, start=1):
            path = f"{prefix} {index}".strip()
            lines.extend(_flatten_contract(nested, path))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def _contract_chunks(option: ParsedOptionSignal, max_chars: int = 950) -> list[str]:
    if not option.semantic_contract:
        return []
    chunks: list[str] = []
    current = ""
    for line in _flatten_contract(option.semantic_contract):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _now_et() -> datetime:
    """Single indirection point for "current time in US market time" --
    lets tests simulate a specific wall-clock moment (e.g. "12:30 has just
    passed") by monkeypatching this one function instead of the whole
    datetime module.
    """
    return datetime.now(ZoneInfo("America/New_York"))


def _automate_agent_eod_cutoff_reached(now_et: datetime, cutoff: str) -> bool:
    """True once now_et's wall-clock time has reached the configured
    HH:MM ET cutoff (automate_agent_exit_time_et). A malformed/empty
    cutoff never forces an exit, matching the "no config, no forced
    behavior" pattern used elsewhere in this file.
    """
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(cutoff or "").strip())
    if not match:
        return False
    hour, minute = (int(value) for value in match.groups())
    return (now_et.hour, now_et.minute) >= (hour, minute)


def _automate_agent_in_min_trades_relax_window(now_et: datetime, cutoff: str, relax_minutes: int) -> bool:
    """True once now_et is within relax_minutes of the fixed daily cutoff
    but hasn't reached it yet -- the window where a still-unmet compulsory
    minimum-trades-per-window requirement should relax the confidence bar
    (see automate_agent_min_trades_per_window) rather than risk missing
    the quota entirely by waiting for a higher-conviction pick that may
    never come. A malformed/empty cutoff never relaxes anything, matching
    the "no config, no forced behavior" pattern used elsewhere here.
    """
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(cutoff or "").strip())
    if not match:
        return False
    hour, minute = (int(value) for value in match.groups())
    cutoff_minutes = hour * 60 + minute
    now_minutes = now_et.hour * 60 + now_et.minute
    return cutoff_minutes - max(0, int(relax_minutes)) <= now_minutes < cutoff_minutes


def _option_cancel_deadline_passed(contract: object, created_at: object = None) -> bool:
    if not isinstance(contract, dict):
        return False
    deadline = contract.get("cancel_if_not_triggered_by")
    if not isinstance(deadline, dict):
        return False
    match = re.fullmatch(r"(\d{2}):(\d{2})", str(deadline.get("time") or ""))
    if not match:
        return False
    now_et = _now_et()
    created_text = str(created_at or "")
    if created_text:
        try:
            created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=ZoneInfo("UTC"))
            if created.astimezone(ZoneInfo("America/New_York")).date() < now_et.date():
                return True
        except ValueError:
            pass
    hour, minute = (int(value) for value in match.groups())
    return (now_et.hour, now_et.minute) >= (hour, minute)


def _option_embed(
    option: ParsedOptionSignal,
    ai_prediction: dict,
    mapped_action: str,
    decision: "DecisionResult",
    strategy_validation: Optional[dict] = None,
    quality: Optional[dict] = None,
) -> "discord.Embed":
    validation_status = str((strategy_validation or {}).get("status") or "").upper()
    direct_mode = validation_status == "AGENT_OFF_DIRECT"
    embed = discord.Embed(
        title=f"{_option_label(option)} Options Signal Review",
        description=(
            "Agent OFF: following the valid incoming option signal directly."
            if direct_mode
            else "Options strategy validation checked before any Alpaca paper option order."
        ),
        color=_decision_color(decision.action),
    )
    option_side = (
        str(option.structure or "MULTI-LEG").replace("_", " ").upper()
        if option.is_multi_leg else option.side or "-"
    )
    embed.add_field(name="Option Side", value=option_side, inline=True)
    embed.add_field(name="Signal Action", value=_option_signal_action_label(option), inline=True)
    embed.add_field(name="Final Option Output", value=decision.action, inline=True)
    embed.add_field(name="Agent Response", value=_agent_response_label(decision), inline=True)
    embed.add_field(name="Qty (contracts)", value=str(_resolved_option_qty(option)), inline=True)
    embed.add_field(
        name="Decision Source",
        value="Incoming option signal" if direct_mode else "Options strategy validation",
        inline=True,
    )
    embed.add_field(name="Quality Score", value=f"{decision.score:.0f}/100", inline=True)
    embed.add_field(name="Parsed Signal", value=_parsed_option_summary(option), inline=False)
    for index, chunk in enumerate(_contract_chunks(option), start=1):
        label = "Parsed Field Contract" if index == 1 else f"Parsed Field Contract ({index})"
        embed.add_field(name=label, value=chunk, inline=False)
    if strategy_validation:
        bt = strategy_validation.get("backtest") or {}
        si = strategy_validation.get("strategy_input") or {}
        execution_contract = strategy_validation.get("execution_contract") or {}
        execution_contracts = strategy_validation.get("execution_contracts") or []
        status = str(strategy_validation.get("status") or "-")
        strat_decision = str(strategy_validation.get("decision") or "REVIEW")
        pnl = bt.get("profit_loss", bt.get("total_profit_loss", "-"))
        win_rate = bt.get("win_rate", "-")
        exact_attempt = strategy_validation.get("exact_strike_attempt") or {}
        recent_exact_attempt = strategy_validation.get("recent_exact_strike_attempt") or {}
        polygon_exact_attempt = strategy_validation.get("polygon_strike_attempt") or {}
        proxy_attempt = strategy_validation.get("delta_proxy_attempt") or {}
        validation_method = si.get("validation_method") or (
            "exact_strike" if si.get("strike_selection") == "strike" else "delta"
        )
        requested_strike = (
            si.get("requested_strike")
            if si.get("requested_strike") is not None
            else option.strike
        )
        validation_strike = (
            si.get("strike_price")
            if si.get("strike_price") is not None
            else si.get("strike_selection", "-")
        )
        public_basis = _option_public_basis(strategy_validation, option)
        public_result = _option_public_result(strategy_validation, decision)
        performance_lines = []
        if pnl not in (None, "", "-"):
            performance_lines.append(f"P&L: {pnl}")
        if win_rate not in (None, "", "-"):
            performance_lines.append(f"Win Rate: {win_rate}")
        embed.add_field(
            name="Options Decision",
            value=(
                f"Result: {public_result}\n"
                f"Decision: {strat_decision}\n"
                f"Basis: {public_basis}"
                + ("\n" + "\n".join(performance_lines[:2]) if performance_lines else "")
            ),
            inline=True,
        )
        if execution_contract:
            contract_status = "Verified" if str(execution_contract.get("status") or "").upper() == "VERIFIED" else "Review"
            embed.add_field(
                name="Exact Contract",
                value=(
                    f"Status: {contract_status}\n"
                    f"OCC: {execution_contract.get('occ_symbol', '-')}\n"
                    f"Expiry: {execution_contract.get('expiration_date', option.expiry_date or '-')}"
                ),
                inline=True,
            )
        if execution_contracts:
            contract_lines = [
                f"{item.get('position_intent', '-')} | {item.get('symbol', '-')}"
                for item in execution_contracts
            ]
            embed.add_field(
                name="Exact Strategy Contracts",
                value="\n".join(contract_lines)[:1000],
                inline=False,
            )
        if decision.action == "BUY" and status in {
            "AGENT_OFF_DIRECT",
            "FALLBACK_APPROVED",
            "SUCCESS_POLYGON_STRIKE",
            "SUCCESS_EXACT_STRIKE_RECENT",
            "SUCCESS",
        }:
            embed.add_field(
                name="Completion Note",
                value=(
                    "Signal accepted directly because Agent mode is OFF. Broker safeguards still apply."
                    if direct_mode
                    else "Signal accepted with the available strategy, contract, and risk checks."
                ),
                inline=False,
            )
        if option.is_multi_leg:
            leg_lines = []
            for leg in option.legs:
                side_order, intent, _ = _option_leg_route(leg)
                leg_lines.append(
                    f"{intent} | {leg.root} {leg.strike:g}{'C' if leg.side == 'CALL' else 'P'} "
                    f"| ratio {leg.ratio_qty} | {side_order}"
                )
            trade_setup = (
                "\n".join(leg_lines)
                + f"\nStrategy Qty: {_resolved_option_qty(option):g} | "
                + f"Net {str(infer_multi_leg_price_effect(option) or 'price').title()}: "
                + (f"${option.fill_price:.2f}" if option.fill_price else "market")
                + f"\nWindow: {si.get('options_backtest_start_date', '-')} to {si.get('options_backtest_end_date', '-')}"
            )
        else:
            resolved_validation_strike = validation_strike
            if resolved_validation_strike in (None, "", "-"):
                resolved_validation_strike = requested_strike if requested_strike is not None else "-"
            trade_setup = (
                f"{si.get('symbol') or option.root} | "
                f"{si.get('direction_label') or _option_action_code(option)} | "
                f"{si.get('opt_type') or si.get('side') or option.side or '-'} | "
                f"Qty {_as_float(si.get('quantity'), _resolved_option_qty(option)):g}\n"
                f"Requested Strike: {requested_strike if requested_strike is not None else '-'} | "
                f"Validation Strike: {resolved_validation_strike} | "
                f"Delta {si.get('delta', option.delta_target if option.delta_target is not None else '-')} | "
                f"DTE {si.get('dte', '-')}\n"
                f"Order: {_option_requested_order_type(option).upper()} | "
                f"Requested Expiry: {option.expiry_date or '-'}"
            )
        if option.is_multi_leg or not direct_mode:
            embed.add_field(name="Trade Setup", value=trade_setup[:1000], inline=not option.is_multi_leg)
    if option.fill_price:
        embed.add_field(name="Signal Fill Price", value=f"${option.fill_price:.2f}", inline=True)
    if option.stop_loss is not None or option.target_price is not None or option.target_prices or quality:
        q = quality or {}
        rr = q.get("risk_reward")
        rr_text = f"{rr:.2f}" if isinstance(rr, (int, float)) else "-"
        issues = list(q.get("critical") or []) + list(q.get("issues") or [])
        embed.add_field(
            name="Option Signal Quality",
            value=(
                f"Score: {float(q.get('score', 0)):.0f}/100\n"
                f"SL: {option.stop_loss if option.stop_loss is not None else '-'} | "
                f"TPs: {' / '.join(f'{price:g}' for price in _option_targets(option)) or '-'} | "
                f"R/R: {rr_text}\n"
                + (f"Notes: {'; '.join(issues)[:450]}" if issues else "Notes: clean")
            ),
            inline=False,
        )
    embed.add_field(name="Trade Gate", value=_option_public_trade_gate(decision), inline=False)
    embed.set_footer(text="Paper options use exact-contract checks. Supported multi-leg strategies are submitted atomically through Alpaca MLeg orders.")
    return embed


async def _process_multi_leg_option_signal(
    message: discord.Message, option: ParsedOptionSignal
) -> None:
    if not option.legs:
        await _send_review_or_reply(message, f"{option.root}: multi-leg signal contains no usable legs.")
        return
    if option.tense == "unknown":
        await asyncio.to_thread(
            record_option_journal_entry,
            {
                "root": option.root,
                "structure": option.structure,
                "legs": _multi_leg_specs(option),
                "raw_input": option.raw_text,
                "status": "multi_leg_journal_only",
            },
        )
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: action was unclear. Use BTO/STO/BTC/STC or BUY/SELL on each leg.",
        )
        return

    mapped_action = _multi_leg_mapped_action(option)
    quality = _option_signal_quality(option)
    features = _learning_features(message.content, "option_mleg", mapped_action, option.root, option)
    agent_enabled = _agent_is_enabled()
    if not agent_enabled:
        strategy_validation = _direct_option_validation(mapped_action)
        strategy_decision = mapped_action
        quality_score = quality.get("score")
        decision = _direct_signal_decision(
            mapped_action,
            float(quality_score) if quality_score is not None else 100.0,
        )
    else:
        if not quality.get("passed"):
            strategy_validation = {
                "status": "SKIPPED_OPTION_QUALITY",
                "decision": "REVIEW",
                "error": "Multi-leg signal quality did not pass local risk checks.",
            }
        else:
            strategy_validation = await asyncio.to_thread(run_options_strategy_validation, option)
        strategy_decision = str(strategy_validation.get("decision") or "REVIEW").upper()
        decision = _option_final_decision(strategy_validation, quality)
    strategy_status = str(strategy_validation.get("status") or "FAILED").upper()
    if agent_enabled and (
        mapped_action == "SELL"
        and strategy_decision == "SELL"
        and strategy_status in _allowed_option_strategy_statuses()
        and quality.get("passed")
    ):
        decision = DecisionResult(
            "SELL",
            "Multi-leg exit strategy approved for paper-order preparation.",
            0.0,
            0.0,
            0.0,
            float(quality.get("score") or 0),
            "options_validation",
        )
    if agent_enabled:
        decision = _apply_learning_overlay(decision, features)

    contract_check = await asyncio.to_thread(
        _multi_leg_contract_lookup,
        option.root,
        _multi_leg_specs(option),
        option.expiry_date or "",
        option.expiry_mode == "0dte",
    )
    strategy_validation = {
        **strategy_validation,
        "execution_contracts": contract_check.get("legs") or [],
        "execution_contract_status": contract_check.get("status"),
        "execution_contract_message": contract_check.get("message"),
    }
    await asyncio.to_thread(_record_option_validation_metrics, option, strategy_validation)
    await asyncio.to_thread(_learn_from_decision, features, decision)
    await asyncio.to_thread(
        add_decision_history,
        {
            "symbol": option.root,
            "kind": "OPTION_MLEG",
            "user_action": mapped_action,
            "final_action": decision.action,
            "score": round(decision.score, 4),
            "reason": decision.reason,
            "raw_input": option.raw_text,
            "structure": option.structure,
            "legs": _multi_leg_specs(option),
            "parsed_contract": option.semantic_contract,
            "options_strategy_status": strategy_status,
            "options_strategy_decision": strategy_decision,
            "agent_mode": "ON" if agent_enabled else "OFF",
        },
    )
    await _send_optional_channel(
        config.discord_review_channel_id,
        message.channel,
        embed=_option_embed(option, {}, mapped_action, decision, strategy_validation, quality),
    )

    if option.contains_equity_leg:
        await asyncio.to_thread(
            record_option_journal_entry,
            {
                "root": option.root,
                "structure": option.structure,
                "legs": _multi_leg_specs(option),
                "raw_input": option.raw_text,
                "status": "coordinated_equity_option_review",
            },
        )
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: the signal combines shares and options. It was validated and stored, "
            "but no partial paper order was submitted because Alpaca cannot execute the stock leg and "
            "option strategy as one atomic mleg order.",
        )
        return

    if strategy_status not in _allowed_option_strategy_statuses():
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: complete strategy validation did not approve this trade. No paper order was submitted.",
        )
        return
    if mapped_action not in {"BUY", "SELL"} or decision.action != mapped_action or strategy_decision != mapped_action:
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: strategy decision is {strategy_decision}, signal action is {mapped_action}; no paper order was submitted.",
        )
        return

    options_enabled, opt_err = await asyncio.to_thread(alpaca.has_multi_leg_options_trading)
    if not options_enabled:
        await _send_review_or_reply(message, f"{_option_label(option)}: {opt_err}")
        return

    qty = _resolved_option_qty(option)
    if not await _trade_guard(message, option.root, mapped_action.lower(), qty, "option"):
        return

    resolved_legs = contract_check.get("legs") or []
    if len(resolved_legs) != len(option.legs):
        await asyncio.to_thread(
            _queue_multi_leg_option_order,
            option,
            qty,
            quality,
            "waiting_for_all_strategy_contracts",
            None,
            option.expiry_date or "",
        )
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: strategy approved and queued. The agent will submit it when every exact Alpaca leg is tradable.",
        )
        return

    for leg in resolved_legs:
        if not leg.get("requires_position"):
            continue
        position, _ = await asyncio.to_thread(alpaca.get_position, str(leg.get("symbol") or ""))
        held_qty = abs(_as_float((position or {}).get("qty")))
        required_qty = qty * max(1, int(leg.get("ratio_qty") or 1))
        if held_qty < required_qty:
            await asyncio.to_thread(
                _queue_multi_leg_option_order,
                option,
                qty,
                quality,
                "waiting_for_matching_leg_positions",
                resolved_legs,
                str(contract_check.get("expiration_date") or option.expiry_date or ""),
            )
            await _send_review_or_reply(
                message,
                f"{_option_label(option)}: strategy approved and queued until all required option positions are available.",
            )
            return

    if option.underlying_trigger_direction and option.underlying_trigger_price:
        underlying_price, _ = await asyncio.to_thread(alpaca.get_latest_price, option.root)
        observed = _as_float(underlying_price)
        trigger = float(option.underlying_trigger_price)
        condition_met = (
            observed > trigger
            if option.underlying_trigger_direction == "above"
            else observed < trigger
        )
        if observed <= 0 or not condition_met:
            await asyncio.to_thread(
                _queue_multi_leg_option_order,
                option,
                qty,
                quality,
                "waiting_for_underlying_price_condition",
                resolved_legs,
                str(contract_check.get("expiration_date") or option.expiry_date or ""),
            )
            await _send_review_or_reply(
                message,
                f"{_option_label(option)}: strategy approved. Watching for {option.root} "
                f"to trade {option.underlying_trigger_direction} ${trigger:.2f} before submitting the paper order.",
            )
            return

    order_type = _option_requested_order_type(option)
    limit_price = _multi_leg_limit_price(option) if order_type == "limit" else None
    market_open, market_err = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        await asyncio.to_thread(
            _queue_multi_leg_option_order,
            option,
            qty,
            quality,
            market_err or "market_closed",
            resolved_legs,
            str(contract_check.get("expiration_date") or option.expiry_date or ""),
        )
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: complete strategy approved and queued for the next Alpaca options market session.",
        )
        return

    order, order_err = await asyncio.to_thread(
        alpaca.submit_multi_leg_option_order,
        _alpaca_multi_leg_payload(resolved_legs),
        qty,
        order_type,
        limit_price,
        _client_order_id("mleg", message.id),
    )
    if not order:
        if _is_market_closed_order_error(order_err) or _is_transient_broker_error(order_err):
            await asyncio.to_thread(
                _queue_multi_leg_option_order,
                option,
                qty,
                quality,
                order_err,
                resolved_legs,
                str(contract_check.get("expiration_date") or option.expiry_date or ""),
            )
            await _send_review_or_reply(
                message,
                f"{_option_label(option)}: Alpaca is temporarily unavailable or the options "
                "market is closed, so the strategy was queued for a safe retry.",
            )
            return
        await _block_trade(
            message,
            option.root,
            f"Multi-leg paper order was not placed. {_public_error(order_err)}",
            "alpaca_mleg_reject",
            option.raw_text,
        )
        return

    order_id = str(order.get("id") or "")
    await _track_submitted_multi_leg_entry(
        order,
        qty,
        {
            "root": option.root,
            "structure": option.structure,
            "legs": resolved_legs,
            "price_effect": infer_multi_leg_price_effect(option) or "debit",
            "stop_loss": option.stop_loss,
            "target_price": option.target_price,
            "target_prices": list(option.target_prices),
            "maximum_loss_amount": option.maximum_loss_amount,
            "protectable": all(
                str(leg.get("position_intent") or "")
                in {"buy_to_open", "sell_to_open"}
                for leg in resolved_legs
            ),
        },
    )
    await asyncio.to_thread(
        _record_order,
        option.root,
        mapped_action.lower(),
        qty,
        "submitted",
        order_id,
        "option_mleg",
        str(option.structure or "multi_leg"),
    )
    await asyncio.to_thread(
        record_option_journal_entry,
        {
            "root": option.root,
            "structure": option.structure,
            "legs": resolved_legs,
            "quantity": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "raw_input": option.raw_text,
            "status": "multi_leg_order_placed",
            "order_id": order_id,
        },
    )
    expiry_note = (
        f" Listed expiry {contract_check.get('expiration_date')} was used."
        if contract_check.get("used_fallback_expiry") else ""
    )
    await _send_review_or_reply(
        message,
        f"{_option_label(option)}: multi-leg paper order placed successfully as one atomic strategy. "
        f"Qty {qty:g}, type={order_type}"
        f"{' @ net ' + str(limit_price) if limit_price is not None else ''}. "
        f"Order ID `{order_id or '-'}`.{expiry_note}",
    )


async def _process_option_signal(message: discord.Message, option: ParsedOptionSignal) -> None:
    if not option.valid:
        await _send_review_or_reply(message, f"Options signal not actionable: {option.reason}")
        return

    if option.is_multi_leg:
        await _process_multi_leg_option_signal(message, option)
        return

    if option.order_action == "manage":
        await asyncio.to_thread(
            record_option_journal_entry,
            {
                "root": option.root, "strike": option.strike, "side": option.side,
                "expiry_date": option.expiry_date, "fill_price": option.fill_price,
                "quantity": _resolved_option_qty(option), "tense": option.tense,
                "order_action": option.order_action,
                "raw_input": option.raw_text, "status": "management_or_short_review",
            },
        )
        label = _option_label(option) if option.strike and option.side else option.root or "Option"
        await _send_review_or_reply(
            message,
            f"{label}: processed as {option.order_action}. It is logged for review because it is a management-only signal.",
        )
        return

    if option.tense == "unknown":
        await asyncio.to_thread(
            record_option_journal_entry,
            {
                "root": option.root, "strike": option.strike, "side": option.side,
                "expiry_date": option.expiry_date, "fill_price": option.fill_price,
                "stop_loss": option.stop_loss, "target_price": option.target_price,
                "quantity": _resolved_option_qty(option), "tense": option.tense,
                "raw_input": option.raw_text, "status": "journal_only",
            },
        )
        if option.fill_price:
            await asyncio.to_thread(
                upsert_option_position,
                f"{option.root}_{option.expiry_date}_{option.side}_{option.strike}",
                option.root, option.side, option.strike, option.expiry_date or "",
                _resolved_option_qty(option), option.fill_price,
                "", option.stop_loss, option.target_price, None, "buy_to_open",
            )
        note = "tense was unclear, so this was logged only. Use BTO/BUY/STO/STC/BTC to act on it."
        await _send_review_or_reply(message, f"{_option_label(option)}: {note}")
        return

    # Explicit new-order signals and complete past-tense trade alerts both proceed
    # through validation; duplicate/same-contract orders are allowed by configuration.
    ai_prediction = {}
    mapped_action = _option_mapped_action(option)
    features = _learning_features(message.content, "option", mapped_action, option.root, option)
    quality = _option_signal_quality(option)
    agent_enabled = _agent_is_enabled()
    if not agent_enabled:
        strategy_validation = _direct_option_validation(mapped_action)
        strategy_decision = mapped_action
        quality_score = quality.get("score")
        decision = _direct_signal_decision(
            mapped_action,
            float(quality_score) if quality_score is not None else 100.0,
        )
    else:
        if not quality.get("passed"):
            strategy_validation = {
                "status": "SKIPPED_OPTION_QUALITY",
                "decision": "REVIEW",
                "error": "Option signal quality did not pass local risk checks.",
            }
        else:
            strategy_validation = await asyncio.to_thread(run_options_strategy_validation, option)
        strategy_decision = str(strategy_validation.get("decision") or "REVIEW").upper()
        decision = _option_final_decision(strategy_validation, quality)
    if agent_enabled and strategy_decision == "SELL" and mapped_action == "SELL" and str(strategy_validation.get("status") or "").upper() in _allowed_option_strategy_statuses() and quality.get("passed"):
        decision = DecisionResult("SELL", "Option sell-side signal approved for paper-order preparation.", 0.0, 0.0, 0.0, float(quality.get("score") or 0), "options_validation")
    if agent_enabled:
        decision = _apply_learning_overlay(decision, features)
    contract_check = await asyncio.to_thread(_verify_exact_option_contract, option)
    strategy_validation = {
        **strategy_validation,
        "execution_contract": {k: v for k, v in contract_check.items() if k != "contracts"},
    }
    await asyncio.to_thread(_record_option_validation_metrics, option, strategy_validation)
    await asyncio.to_thread(_learn_from_decision, features, decision)

    await asyncio.to_thread(
        add_decision_history,
        {
            "symbol": option.root, "kind": "OPTION", "user_action": mapped_action,
            "ai_decision": str(ai_prediction.get("decision") or "REVIEW").upper(),
            "final_action": decision.action,
            "predicted_return_pct": round(decision.predicted_return, 4),
            "confidence": round(decision.confidence, 4), "risk": round(decision.risk, 4),
            "score": round(decision.score, 4), "market_regime": decision.market_regime,
            "reason": decision.reason, "raw_input": option.raw_text,
            "parsed_contract": option.semantic_contract,
            "option_signal_quality": round(float(quality.get("score") or 0), 2),
            "option_risk_reward": quality.get("risk_reward"),
            "options_strategy_status": strategy_validation.get("status"),
            "options_strategy_decision": strategy_decision,
            "options_strategy_error": strategy_validation.get("error", ""),
            "agent_mode": "ON" if agent_enabled else "OFF",
        },
    )

    await _send_optional_channel(
        config.discord_review_channel_id, message.channel,
        embed=_option_embed(option, ai_prediction, mapped_action, decision, strategy_validation, quality),
    )

    if option.contains_equity_leg:
        await asyncio.to_thread(
            record_option_journal_entry,
            {
                "root": option.root,
                "structure": option.structure,
                "raw_input": option.raw_text,
                "status": "coordinated_equity_option_review",
            },
        )
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: this signal also requires an equity position. It was validated and "
            "stored, but the option leg was not submitted separately because that could leave an incomplete hedge.",
        )
        return

    strategy_status = str(strategy_validation.get("status") or "FAILED").upper()
    if strategy_status not in _allowed_option_strategy_statuses():
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: review completed. Manual review selected for this signal.",
        )
        return

    if decision.action != mapped_action or strategy_decision != mapped_action:
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: review completed. Strategy decision is {strategy_decision}, signal action is {mapped_action}; no paper order will be placed."
        )
        return

    options_enabled, opt_err = await asyncio.to_thread(alpaca.has_options_trading)
    if not options_enabled:
        await _send_review_or_reply(message, f"{_option_label(option)}: {opt_err}")
        return

    order_side, position_intent, requires_position = _option_order_route(option)
    qty = _resolved_option_qty(option)
    if not await _trade_guard(message, option.root, order_side, qty, "option"):
        return

    contracts = contract_check.get("contracts")
    lookup_err = contract_check.get("message", "")
    used_fallback_expiry = False
    if not contracts and option.expiry_mode == "0dte":
        # Same-day expiry may not exist for this underlying — retry without an expiry
        # filter and use the nearest available date instead, as agreed.
        contracts, lookup_err = await asyncio.to_thread(
            alpaca.get_option_contracts, option.root, None, option.strike, option.side.lower()
        )
        if contracts:
            contracts = sorted(contracts, key=lambda c: str(c.get("expiration_date") or ""))
            used_fallback_expiry = True

    if not contracts:
        order_type = _option_requested_order_type(option)
        await asyncio.to_thread(
            _queue_option_order,
            "",
            option,
            qty,
            order_type,
            option.expiry_date or "",
            quality,
            "waiting_for_tradable_contract",
            order_side,
            position_intent,
            requires_position,
            True,
        )
        await asyncio.to_thread(_record_order, option.root, order_side, qty, "watching", "", "option", "waiting_for_tradable_contract")
        await _send_review_or_reply(
            message,
            f"{_option_label(option)}: trade accepted by the agent. The exact contract is not tradable in Alpaca yet, "
            "so it has been stored and will be monitored until it can be placed.",
        )
        return

    contract = contracts[0]
    occ_symbol = str(contract.get("symbol") or "")
    if not occ_symbol:
        await _send_review_or_reply(message, f"{_option_label(option)}: contract lookup succeeded but returned no symbol.")
        return

    contract_expiration = str(contract.get("expiration_date") or option.expiry_date or "")
    await _send_review_or_reply(
        message,
        f"{occ_symbol}: exact Alpaca option contract found. "
        f"Expiry {contract_expiration or '-'}, strike {option.strike:g}, side {option.side}. "
        f"Preparing {_option_order_label(option)} paper-order submission.",
    )

    held_qty = 0.0
    if requires_position:
        alpaca_position, _ = await asyncio.to_thread(alpaca.get_position, occ_symbol)
        held_qty = abs(_as_float((alpaca_position or {}).get("qty")))
        if held_qty <= 0:
            await asyncio.to_thread(
                _queue_option_order,
                occ_symbol,
                option,
                qty,
                "market",
                contract_expiration,
                quality,
                "waiting_for_matching_position",
                order_side,
                position_intent,
                True,
                False,
            )
            await asyncio.to_thread(_record_order, occ_symbol, order_side, qty, "watching", "", "option", "waiting_for_matching_position")
            await _send_review_or_reply(message, f"{occ_symbol}: {_option_order_label(option)} accepted, but no matching Alpaca option position is available yet. The agent will keep monitoring and place it when possible.")
            return
        if option.close_percent is not None:
            requested_percent = min(100.0, max(0.0, option.close_percent))
            percentage_qty = int(held_qty * requested_percent / 100.0)
            if percentage_qty < 1:
                await _send_review_or_reply(
                    message,
                    f"{occ_symbol}: cannot close {requested_percent:g}% of {held_qty:g} contract(s) "
                    "without closing the entire position. No close order was submitted.",
                )
                return
            qty = min(float(percentage_qty), held_qty)
        else:
            qty = min(qty, held_qty)

    if option.underlying_trigger_direction and option.underlying_trigger_price:
        if _option_cancel_deadline_passed(option.semantic_contract):
            await asyncio.to_thread(
                record_option_journal_entry,
                {
                    "root": option.root,
                    "side": option.side,
                    "strike": option.strike,
                    "expiry_date": contract_expiration,
                    "raw_input": option.raw_text,
                    "semantic_contract": option.semantic_contract,
                    "status": "conditional_entry_expired",
                },
            )
            await _send_review_or_reply(
                message,
                f"{occ_symbol}: conditional entry deadline has passed. The signal was canceled and no paper order was submitted.",
            )
            return
        underlying_price, _ = await asyncio.to_thread(alpaca.get_latest_price, option.root)
        observed = _as_float(underlying_price)
        trigger = float(option.underlying_trigger_price)
        condition_met = (
            observed > trigger
            if option.underlying_trigger_direction == "above"
            else observed < trigger
        )
        if observed <= 0 or not condition_met:
            await asyncio.to_thread(
                _queue_option_order,
                occ_symbol,
                option,
                qty,
                _option_requested_order_type(option),
                contract_expiration,
                quality,
                "waiting_for_underlying_price_condition",
                order_side,
                position_intent,
                requires_position,
                False,
            )
            await _send_review_or_reply(
                message,
                f"{occ_symbol}: exact contract found and signal approved. Watching for {option.root} "
                f"to trade {option.underlying_trigger_direction} ${trigger:.2f} before submitting the paper order.",
            )
            return

    order_type = _option_requested_order_type(option)

    market_open, market_err = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        await asyncio.to_thread(
            _queue_option_order,
            occ_symbol,
            option,
            qty,
            order_type,
            contract_expiration,
            quality,
            market_err or "market_closed",
            order_side,
            position_intent,
            requires_position,
            False,
            max(0.0, held_qty - qty) if requires_position else None,
            "BREAKEVEN" in str(option.raw_text or "").upper(),
        )
        await asyncio.to_thread(_record_order, occ_symbol, order_side, qty, "queued", "", "option", "market_closed")
        await _send_review_or_reply(
            message,
            f"{occ_symbol}: option {_option_order_label(option)} approved, but the market is closed. "
            "It has been queued and will be submitted when Alpaca reports the market is open.",
        )
        await _send_optional_channel(
            config.discord_paper_log_channel_id,
            message.channel,
            f"{occ_symbol}: option {order_side.upper()} queued until market open. Qty {qty:g}, type={order_type}, intent={position_intent}.",
        )
        return

    order, order_err = await asyncio.to_thread(
        alpaca.submit_option_order,
        occ_symbol,
        order_side,
        qty,
        order_type,
        option.fill_price if order_type == "limit" else None,
        position_intent,
        _client_order_id("option", message.id),
    )
    if not order:
        if _is_market_closed_order_error(order_err) or _is_transient_broker_error(order_err):
            await asyncio.to_thread(
                _queue_option_order,
                occ_symbol,
                option,
                qty,
                order_type,
                contract_expiration,
                quality,
                order_err,
                order_side,
                position_intent,
                requires_position,
                False,
                max(0.0, held_qty - qty) if requires_position else None,
                "BREAKEVEN" in str(option.raw_text or "").upper(),
            )
            await asyncio.to_thread(_record_order, occ_symbol, order_side, qty, "queued", "", "option", "market_closed_reject")
            await _send_review_or_reply(
                message,
                f"{occ_symbol}: option trade approved and queued because Alpaca is temporarily "
                "unavailable or the market is closed. The agent will retry safely.",
            )
            return
        await _block_trade(
            message,
            occ_symbol,
            f"Paper option order was not placed. {_public_error(order_err)}",
            "alpaca_reject",
            option.raw_text,
        )
        return

    entry_price = option.fill_price or 0.0
    await asyncio.to_thread(_record_order, occ_symbol, order_side, qty, "submitted", str(order.get("id") or ""), "option")
    if position_intent in {"buy_to_open", "sell_to_open"}:
        await _track_submitted_option_entry(
            order,
            occ_symbol,
            qty,
            {
                "root": option.root,
                "side": option.side,
                "strike": option.strike,
                "expiry_date": contract_expiration,
                "stop_loss": option.stop_loss,
                "target_price": option.target_price,
                "target_prices": list(option.target_prices),
                "trailing_stop_pct": option.trailing_stop_pct,
                "exit_before_market_close": option.exit_before_market_close,
                "exit_minutes_before_close": option.exit_minutes_before_close,
                "exit_if_target_not_hit": option.exit_if_target_not_hit,
                "risk_stop_pct": option.risk_stop_pct,
                "maximum_loss_amount": option.maximum_loss_amount,
                "position_type": option.position_type,
                "exit_underlying_direction": option.exit_underlying_direction,
                "exit_underlying_price": option.exit_underlying_price,
                "time_in_force": option.time_in_force,
                "stop_scope": option.stop_scope,
                "semantic_contract": option.semantic_contract,
                "signal_quality": float(quality.get("score") or 0),
                "position_intent": position_intent,
            },
        )
    elif position_intent in {"sell_to_close", "buy_to_close"}:
        await _track_submitted_exit(
            order,
            occ_symbol,
            qty,
            "option",
            "manual_option_exit",
            move_stop_to_breakeven=("BREAKEVEN" in str(option.raw_text or "").upper()),
        )
    await asyncio.to_thread(
        record_option_journal_entry,
        {
            "occ_symbol": occ_symbol, "root": option.root, "side": option.side,
            "strike": option.strike, "quantity": qty, "order_type": order_type,
            "stop_loss": option.stop_loss, "target_price": option.target_price,
            "signal_quality": quality.get("score"), "risk_reward": quality.get("risk_reward"),
            "raw_input": option.raw_text, "status": "order_submitted", "position_intent": position_intent,
        },
    )
    fallback_note = " (nearest available expiry used — 0DTE not listed)" if used_fallback_expiry else ""
    await _send_review_or_reply(
        message,
        f"{occ_symbol}: paper option {order_side.upper()} order submitted successfully. "
        f"Qty {qty:g} contract(s), type={order_type}"
        f"{' @ $' + f'{option.fill_price:.2f}' if option.fill_price else ''}. "
        f"Order ID `{order.get('id') or '-'}`.{fallback_note}",
    )
    await _send_optional_channel(
        config.discord_paper_log_channel_id, message.channel,
        f"{occ_symbol}: paper option {order_side.upper()} submitted successfully. Qty {qty:g}. Intent {position_intent}. Order ID `{order.get('id') or '-'}`. Position state updates after Alpaca confirms fills.",
    )

    if position_intent == "buy_to_open" and (
        option.add_trigger_premium or option.add_trigger_underlying_price
    ):
        add_qty = option.add_quantity or _resolved_option_qty(option)
        queued_scale_in = await asyncio.to_thread(
            _queue_conditional_option_scale_in,
            {
                "occ_symbol": occ_symbol,
                "root": option.root,
                "side": option.side,
                "strike": option.strike,
                "expiry_date": contract_expiration,
                "qty": add_qty,
                "add_quantity": add_qty,
                "add_trigger_premium": option.add_trigger_premium,
                "add_trigger_underlying_direction": option.add_trigger_underlying_direction,
                "add_trigger_underlying_price": option.add_trigger_underlying_price,
                "stop_loss": option.stop_loss,
                "target_price": option.target_price,
                "target_prices": list(option.target_prices),
                "trailing_stop_pct": option.trailing_stop_pct,
                "exit_before_market_close": option.exit_before_market_close,
                "exit_minutes_before_close": option.exit_minutes_before_close,
                "exit_if_target_not_hit": option.exit_if_target_not_hit,
                "risk_stop_pct": option.risk_stop_pct,
                "maximum_loss_amount": option.maximum_loss_amount,
                "position_type": option.position_type,
                "exit_underlying_direction": option.exit_underlying_direction,
                "exit_underlying_price": option.exit_underlying_price,
                "time_in_force": option.time_in_force,
                "stop_scope": option.stop_scope,
                "signal_quality": quality.get("score"),
                "raw_input": option.raw_text,
                "base_order_id": str(order.get("id") or ""),
            },
        )
        if not queued_scale_in:
            return
        trigger_text = (
            f"option premium reaches ${option.add_trigger_premium:.2f}"
            if option.add_trigger_premium
            else f"{option.root} moves {option.add_trigger_underlying_direction} "
                 f"${option.add_trigger_underlying_price:g}"
        )
        await _send_review_or_reply(
            message,
            f"{occ_symbol}: scale-in instruction stored for {add_qty:g} additional contract(s) "
            f"after the base position fills and {trigger_text}.",
        )


async def _fetch_queued_message(item: dict) -> Optional[discord.Message]:
    if str(item.get("transport") or "discord").lower() != "discord":
        return None
    channel_id = int(str(item.get("channel_id") or "0") or 0)
    message_id = int(str(item.get("message_id") or "0") or 0)
    if not channel_id or not message_id:
        return None
    channel = bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(channel_id)
        except Exception:
            return None
    try:
        return await channel.fetch_message(message_id)
    except Exception:
        return None


def _condition_is_triggered(order: dict, current_price: float) -> bool:
    """True once the watched price condition is unambiguously satisfied.

    A further move past the trigger (e.g. a gap well below a "buy if below
    $X" level) still satisfies the stated condition and must fire -- it must
    never require price to stay within a band near the trigger, or a real
    gap leaves the watch order silently stuck forever (it has no expiry).
    """
    condition = str(order.get("condition_type") or "").lower()
    trigger_price = _as_float(order.get("condition_price"))
    action = str(order.get("action") or "").upper()
    if current_price <= 0 or trigger_price <= 0:
        return False

    if condition == "limit_price":
        return current_price <= trigger_price if action == "BUY" else current_price >= trigger_price
    if condition in {"above", "close_above"}:
        return current_price >= trigger_price
    if condition in {"below", "close_below"}:
        return current_price <= trigger_price
    return False


async def _process_conditional_equity_orders() -> None:
    for order in _rotating_batch(
        list_conditional_equity_orders(), "conditional_equity"
    ):
        symbol = str(order.get("symbol") or "").upper()
        if not symbol:
            await asyncio.to_thread(remove_conditional_equity_order, str(order.get("id") or ""))
            continue
        current_price, err = await asyncio.to_thread(alpaca.get_latest_price, symbol)
        current_price = _as_float(current_price)
        if not _condition_is_triggered(order, current_price):
            continue
        message = await _fetch_queued_message(order) or _QueuedMessage(order)
        parsed = ParsedSignal(
            valid=True,
            action=str(order.get("action") or "").upper(),
            symbol=symbol,
            quantity=order.get("quantity"),
            raw_text=str(order.get("raw_input") or ""),
            reason=f"Watched price condition triggered at ${current_price:.2f}.",
            order_type=str(order.get("order_type") or "market"),
            limit_price=order.get("limit_price"),
            stop_price=order.get("stop_price"),
            time_in_force=str(order.get("time_in_force") or "DAY"),
        )
        await _send_review_or_reply(
            message,
            f"{symbol}: watched condition triggered near ${current_price:.2f}. "
            f"Processing with Agent {get_agent_mode()} mode.",
        )
        await _process_signal(message, parsed)
        await asyncio.to_thread(remove_conditional_equity_order, str(order.get("id") or ""))


async def _process_claimed_signal(item: dict) -> None:
    try:
        message = await _fetch_queued_message(item) or _QueuedMessage(item)
        await _process_queued_signal_message(message)
        await asyncio.to_thread(complete_signal, str(item.get("id") or ""))
    except Exception as exc:
        await asyncio.to_thread(
            record_safety_block,
            {
                "symbol": "",
                "category": "signal_queue_worker_exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "raw_input": str(item.get("raw_text") or ""),
            },
        )
        retry = await asyncio.to_thread(
            fail_signal,
            str(item.get("id") or ""),
            f"{type(exc).__name__}: {exc}",
            config.signal_max_attempts,
            config.signal_retry_base_seconds,
        )
        if retry.get("status") == "dead":
            await _send_channel(
                config.discord_review_channel_id,
                "A queued signal failed after all retry attempts and was moved to the dead-letter queue. "
                f"{_public_error(f'{type(exc).__name__}: {exc}')}",
            )


def _discard_signal_task(task: asyncio.Task) -> None:
    _SIGNAL_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logging.getLogger("discord_stock_prediction_agent").exception(
            "Unexpected uncaught signal worker task failure"
        )


@tasks.loop(seconds=max(0.1, config.signal_queue_poll_seconds))
async def signal_queue_worker() -> None:
    concurrency = max(1, min(16, config.signal_worker_concurrency))
    while len(_SIGNAL_TASKS) < concurrency:
        item = await asyncio.to_thread(
            claim_next_signal,
            config.signal_max_attempts,
            config.signal_claim_timeout_seconds,
        )
        if not item:
            break
        task = asyncio.create_task(
            _process_claimed_signal(item),
            name=f"signal-{str(item.get('id') or '')[:12]}",
        )
        _SIGNAL_TASKS.add(task)
        task.add_done_callback(_discard_signal_task)


@signal_queue_worker.before_loop
async def before_signal_queue_worker() -> None:
    await bot.wait_until_ready()


@signal_queue_worker.error
async def signal_queue_worker_error(error: BaseException) -> None:
    logging.getLogger("discord_stock_prediction_agent").error(
        "Signal queue scheduler failed and will be restarted.",
        exc_info=(type(error), error, error.__traceback__),
    )
    await asyncio.sleep(5)
    signal_queue_worker.restart()


@bot.event
async def on_ready() -> None:
    global _QUEUE_RECOVERED, _STARTUP_ORDER_RECOVERY_DONE
    if not _QUEUE_RECOVERED:
        recovered = await asyncio.to_thread(recover_inflight_signals)
        _QUEUE_RECOVERED = True
        if recovered:
            logging.getLogger("discord_stock_prediction_agent").warning(
                "Recovered %s interrupted signal(s) after restart.", recovered
            )
    await asyncio.to_thread(_migrate_legacy_market_order_queue)
    if not _STARTUP_ORDER_RECOVERY_DONE:
        try:
            await _recover_persisted_orders_on_startup()
        except Exception:
            logging.getLogger("discord_stock_prediction_agent").exception(
                "Startup order recovery failed; the normal monitor will retry persisted orders."
            )
        finally:
            _STARTUP_ORDER_RECOVERY_DONE = True
    if not signal_queue_worker.is_running():
        signal_queue_worker.start()
    if not stop_loss_monitor.is_running():
        stop_loss_monitor.start()
    if config.automate_agent_autoscan_enabled and not automate_agent_autoscan.is_running():
        automate_agent_autoscan.start()
    if not symbol_cache_refresh_monitor.is_running():
        symbol_cache_refresh_monitor.start()
    if config.whatsapp_webhook_enabled:
        try:
            start_whatsapp_webhook_server()
        except OSError as exc:
            logging.getLogger("discord_stock_prediction_agent").error(
                "WhatsApp webhook could not start: %s", exc
            )
    refresh_market_context_async()
    symbol_cache_status = await asyncio.to_thread(refresh_symbol_cache_from_alpaca)
    logging.getLogger("discord_stock_prediction_agent").info(
        "Symbol cache refresh: %s",
        {k: v for k, v in symbol_cache_status.items() if k not in {"error", "message"}},
    )
    print(f"{config.agent_name} ready as {bot.user}. Stop monitor active.")


async def _process_queued_signal_message(message: discord.Message) -> None:
    if str(getattr(message, "transport", "discord")) == "whatsapp":
        if await _try_dispatch_whatsapp_command(message):
            return

    routed = classify_and_parse_with_daily_context(
        message.content, str(getattr(message.channel, "id", "") or "")
    )
    symbol_for_event = ""
    action_for_event = ""
    if routed.equity:
        symbol_for_event = routed.equity.symbol
        action_for_event = routed.equity.action
    elif routed.option:
        symbol_for_event = routed.option.root
        action_for_event = "BUY" if routed.option.tense == "new_order" else routed.option.tense
    parser_valid = routed.kind != "INVALID" and not (
        routed.kind == "OPTION" and (not routed.option or not routed.option.valid)
    )
    await asyncio.to_thread(
        record_parser_learning,
        message.content,
        routed.kind,
        parser_valid,
        routed.reason,
    )
    await asyncio.to_thread(
        record_signal_event,
        message.content,
        routed.kind,
        symbol_for_event,
        action_for_event,
        str(message.author.id),
        str(message.channel.id),
    )

    if routed.kind == "INVALID":
        await asyncio.to_thread(
            _learn_from_input,
            message.content,
            routed.kind,
            "INVALID",
            symbol_for_event,
            routed.reason or "Invalid input",
        )
        if routed.equity and isinstance(routed.equity.order_intent, dict):
            await _send_review_or_reply(message, "Invalid/non-executable normalized stock-order fields:")
            for chunk in stock_review_chunks(routed.equity.order_intent):
                await _send_review_or_reply(message, chunk)
        await _send_review_or_reply(message, routed.reason or "Invalid input")
        return

    if routed.kind == "NO_TRADE":
        await asyncio.to_thread(
            add_decision_history,
            {
                "symbol": "", "user_action": "NO_TRADE", "ai_decision": "", "final_action": "NO_TRADE",
                "reason": routed.reason, "raw_input": message.content,
            },
        )
        await asyncio.to_thread(
            _learn_from_input,
            message.content,
            routed.kind,
            "NO_TRADE",
            symbol_for_event,
            routed.reason,
        )
        await _send_review_or_reply(message, routed.reason)
        return

    try:
        if routed.kind == "EQUITY":
            await _process_signal(message, routed.equity)
        elif routed.kind == "OPTION":
            await _process_option_signal(message, routed.option)
    except Exception as exc:
        await asyncio.to_thread(
            record_safety_block,
            {
                "symbol": symbol_for_event,
                "category": "agent_exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "raw_input": message.content,
            },
        )
        await _send_review_or_reply(
            message,
            f"Agent could not safely process this signal. {_public_error(f'{type(exc).__name__}: {exc}')}",
        )



@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    context = await bot.get_context(message)
    if context.valid:
        await bot.invoke(context)
        return

    if config.discord_signal_channel_id and message.channel.id != config.discord_signal_channel_id:
        return

    queued = await asyncio.to_thread(
        enqueue_signal,
        message.content,
        str(message.author.id),
        str(message.channel.id),
        str(message.id),
        config.signal_queue_limit,
    )
    if queued.get("queue_full"):
        await _send_channel(
            config.discord_review_channel_id,
            "The signal queue is temporarily full. This signal was not accepted; please retry shortly.",
        )
        return
    if queued.get("duplicate_delivery"):
        return
_stop_loss_monitor_last_tick: float = 0.0


@tasks.loop(seconds=max(15, config.stop_monitor_seconds))
async def stop_loss_monitor() -> None:
    # Recorded before anything else runs, specifically so !agent_health can
    # surface a stalled monitor loop (this incident happened live: the loop
    # stopped ticking with no exception, no restart, and no log activity at
    # all -- the only way anyone noticed was manually checking positions
    # well after the 12:30 force-close should have already happened).
    global _stop_loss_monitor_last_tick
    _stop_loss_monitor_last_tick = asyncio.get_event_loop().time()
    if not alpaca.ready():
        return
    await _activate_filled_pending_buys()
    market_open, _ = await asyncio.to_thread(alpaca.is_market_open)
    await _reconcile_pending_option_entry_orders()
    await _reconcile_pending_multi_leg_entry_orders()
    await _reconcile_pending_exit_orders()
    await _process_conditional_equity_orders()
    if market_open:
        await _process_pending_market_buys()
        # Market orders normally fill immediately. Re-check now so protection
        # can activate in the same monitor cycle instead of waiting a minute.
        await _activate_filled_pending_buys()
        await _process_pending_sells()
        await _process_pending_option_orders()
        await _reconcile_pending_option_entry_orders()
        await _reconcile_pending_multi_leg_entry_orders()
    equity_clock = None
    for position in list_positions():
        # Regression: nothing isolated one position's failure here, unlike
        # every other per-symbol loop in this file (e.g. the automate_agent
        # buy/evict loops). An exception on any single position -- a
        # malformed record, an unexpected API response shape -- would abort
        # this whole for-loop, skipping the protection/force-close check for
        # every position after it that cycle. Worse, if the same condition
        # recurs every cycle, that one position could permanently block all
        # the others from ever being checked again. This must never be able
        # to prevent "sell everything at 12:30" from actually covering
        # everything.
        try:
            symbol = str(position.get("symbol") or "").upper()
            qty = _as_float(position.get("qty"))
            entry = _as_float(position.get("entry_price"))
            short_position = str(position.get("side") or "long").lower() == "short"
            if not symbol or qty <= 0 or entry <= 0:
                continue
            levels = build_protection_levels(
                entry,
                stop_loss_pct=config.equity_stop_loss_pct,
                take_profit_pct=config.equity_take_profit_pct,
                short_position=short_position,
            )

            alpaca_position, _ = await asyncio.to_thread(alpaca.get_position, symbol)
            if not alpaca_position:
                await asyncio.to_thread(remove_position, symbol)
                continue

            # Alpaca reports a short position's qty as negative; abs() it or a
            # genuinely-held short position looks like "no position" and gets its
            # local protection tracking deleted right after opening.
            held_qty = abs(_as_float(alpaca_position.get("qty")))
            current_price = _as_float(alpaca_position.get("current_price"))
            if held_qty <= 0:
                await asyncio.to_thread(remove_position, symbol)
                continue
            if current_price <= 0:
                latest_price, _ = await asyncio.to_thread(alpaca.get_latest_price, symbol)
                current_price = _as_float(latest_price)
            trigger = evaluate_protection(current_price, levels, short_position=short_position)
            eod_forced = False
            if not trigger.triggered:
                if bool(position.get("exit_before_market_close")) and market_open:
                    if str(position.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG:
                        # automate_agent trades a fixed, shorter window (e.g.
                        # activated ~9:30 ET, force-closed 12:30 ET) rather than
                        # "close near end of day" -- a predictable daily cutoff
                        # instead of one relative to market close.
                        eod_forced = _automate_agent_eod_cutoff_reached(
                            _now_et(), config.automate_agent_exit_time_et
                        )
                    else:
                        if equity_clock is None:
                            equity_clock, _ = await asyncio.to_thread(alpaca.get_clock)
                        try:
                            next_close = datetime.fromisoformat(
                                str((equity_clock or {}).get("next_close") or "").replace("Z", "+00:00")
                            )
                            seconds_to_close = (next_close - datetime.now(next_close.tzinfo)).total_seconds()
                            eod_forced = 0 <= seconds_to_close <= config.automate_agent_exit_minutes_before_close * 60
                        except (TypeError, ValueError):
                            eod_forced = False
                if not eod_forced:
                    continue

            exit_reason = (
                "protection_stop_loss" if trigger.triggered and trigger.reason == "stop_loss"
                else "protection_take_profit" if trigger.triggered
                else "eod_forced_exit"
            )
            boundary_label = (
                "stop" if trigger.triggered and trigger.reason == "stop_loss"
                else "target" if trigger.triggered
                else "end-of-day"
            )
            exit_side = "buy" if short_position else "sell"
            exit_qty = min(qty, held_qty)
            reference_price = trigger.trigger_price if trigger.triggered else current_price

            if not market_open:
                await asyncio.to_thread(
                    enqueue_market_order,
                    symbol,
                    exit_side,
                    exit_qty,
                    exit_reason,
                    config.equity_stop_loss_pct,
                    f"protection:{symbol}:{entry:.6f}:{exit_reason}",
                    "BUY_TO_COVER" if short_position else "",
                )
                continue

            open_order, _ = await asyncio.to_thread(alpaca.has_open_order, symbol)
            if open_order:
                continue

            order, err = await asyncio.to_thread(
                alpaca.submit_market_order,
                symbol,
                exit_side,
                exit_qty,
                _client_order_id(
                    "equityprotect",
                    f"{symbol}:{entry}:{exit_reason}:{reference_price}",
                ),
            )
            if order:
                await asyncio.to_thread(_record_order, symbol, exit_side, exit_qty, "submitted", str(order.get("id") or ""), "equity", exit_reason)
                await _track_submitted_exit(
                    order, symbol, exit_qty, "equity", exit_reason
                )
                reason_text = (
                    f"{boundary_label} protection triggered at ${current_price:.2f} "
                    f"(boundary ${reference_price:.2f})"
                    if trigger.triggered
                    else f"forced end-of-day close at ${current_price:.2f}"
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: {reason_text}. Submitted {exit_qty:g} share(s) "
                    f"to {'cover the short' if short_position else 'close the position'}; "
                    "awaiting Alpaca fill confirmation.",
                )
            else:
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": symbol, "category": "protection_exit_failed", "reason": err},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: protection {'cover' if short_position else 'sell'} attempted but was not placed. {_public_error(err)}",
                )
        except Exception as exc:  # noqa: BLE001 -- one position's failure must not block the rest of the cycle
            logging.getLogger("discord_stock_prediction_agent").error(
                "stop_loss_monitor: failed to process position %s: %s",
                position.get("symbol"), exc,
            )
            continue

    await _process_option_exit_monitor(market_open)
    await _process_multi_leg_exit_monitor(market_open)
    if market_open:
        await _maybe_post_automate_agent_daily_report()


@stop_loss_monitor.before_loop
async def before_stop_loss_monitor() -> None:
    await bot.wait_until_ready()


@stop_loss_monitor.error
async def stop_loss_monitor_error(error: BaseException) -> None:
    logging.getLogger("discord_stock_prediction_agent").error(
        "Trade monitor failed and will be restarted.",
        exc_info=(type(error), error, error.__traceback__),
    )
    await asyncio.sleep(5)
    stop_loss_monitor.restart()


def _has_pending_option_exit(occ_symbol: str) -> bool:
    symbol = str(occ_symbol or "").upper()
    for pending in list_pending_option_orders():
        if str(pending.get("occ_symbol") or "").upper() != symbol:
            continue
        intent = str(pending.get("position_intent") or "").lower()
        if intent in {"sell_to_close", "buy_to_close"} or bool(pending.get("requires_position")):
            return True
    return False


async def _process_option_exit_monitor(market_open: bool) -> None:
    clock = None
    for position in list_option_positions():
        # Regression: nothing isolated one position's failure here -- an
        # exception on a single malformed/unexpected position would abort
        # this whole loop, skipping every position after it that cycle, and
        # if the same condition recurs every cycle, that one position could
        # permanently block all the others from ever being checked again.
        try:
            occ_symbol = str(position.get("occ_symbol") or "").upper()
            tracked_qty = _as_float(position.get("qty"))
            entry_price = _as_float(position.get("entry_price"))
            position_intent = str(position.get("position_intent") or "buy_to_open")
            short_position = position_intent == "sell_to_open"
            levels = build_protection_levels(
                entry_price,
                stop_loss_pct=(
                    _as_float(position.get("stop_loss_pct"))
                    or config.option_stop_loss_pct
                ),
                take_profit_pct=config.option_take_profit_pct,
                short_position=short_position,
                explicit_stop=_as_float(position.get("stop_loss")) or None,
                explicit_target=_as_float(position.get("target_price")) or None,
            )
            stop_loss = levels.stop_price
            target_price = levels.target_price
            target_prices = [
                _as_float(value) for value in (position.get("target_prices") or []) if _as_float(value) > 0
            ]
            target_index = max(0, int(_as_float(position.get("target_index"))))
            trailing_stop_pct = _as_float(position.get("trailing_stop_pct"))
            timed_exit = bool(position.get("exit_before_market_close"))
            maximum_loss_amount = _as_float(position.get("maximum_loss_amount"))
            underlying_exit_direction = str(position.get("exit_underlying_direction") or "").lower()
            underlying_exit_price = _as_float(position.get("exit_underlying_price"))
            exit_minutes_before_close = max(
                1, int(_as_float(position.get("exit_minutes_before_close"), 15))
            )
            if not occ_symbol or tracked_qty <= 0 or entry_price <= 0:
                continue

            alpaca_position, _ = await asyncio.to_thread(alpaca.get_position, occ_symbol)
            if not alpaca_position:
                last_order_id = str(position.get("last_order_id") or "")
                if last_order_id:
                    order, _ = await asyncio.to_thread(alpaca.get_order, last_order_id)
                    status = str((order or {}).get("status") or "").lower()
                    if status in {"new", "accepted", "pending_new", "partially_filled"}:
                        continue
                    if status == "filled":
                        continue
                await asyncio.to_thread(remove_option_position, occ_symbol)
                continue

            # Alpaca reports a short (sell_to_open) position's qty as negative;
            # abs() it or a genuinely-held short position looks like "no position"
            # and gets its local protection tracking deleted right after opening.
            held_qty = abs(_as_float(alpaca_position.get("qty")))
            if held_qty <= 0:
                await asyncio.to_thread(remove_option_position, occ_symbol)
                continue

            current_price = _as_float(alpaca_position.get("current_price"))
            if current_price <= 0:
                latest_price, _ = await asyncio.to_thread(alpaca.get_latest_option_price, occ_symbol)
                current_price = _as_float(latest_price)
            if current_price <= 0:
                continue

            if trailing_stop_pct > 0 and position_intent != "sell_to_open":
                peak_price = max(_as_float(position.get("peak_price")), current_price)
                if peak_price != _as_float(position.get("peak_price")):
                    await asyncio.to_thread(update_option_position, occ_symbol, peak_price=round(peak_price, 6))
                trailing_level = peak_price * (1 - trailing_stop_pct / 100.0)
                stop_loss = max(stop_loss, trailing_level)

            active_target = target_price
            if target_prices and target_index < len(target_prices):
                active_target = target_prices[target_index]

            exit_before_close_now = False
            if timed_exit and market_open:
                if clock is None:
                    clock, _ = await asyncio.to_thread(alpaca.get_clock)
                try:
                    next_close = datetime.fromisoformat(str((clock or {}).get("next_close") or "").replace("Z", "+00:00"))
                    seconds_to_close = (
                        next_close - datetime.now(next_close.tzinfo)
                    ).total_seconds()
                    exit_before_close_now = 0 <= seconds_to_close <= exit_minutes_before_close * 60
                except (TypeError, ValueError):
                    exit_before_close_now = False

            active_levels = type(levels)(stop_loss, active_target)
            protection = evaluate_protection(
                current_price,
                active_levels,
                short_position=short_position,
            )
            hit_stop = protection.triggered and protection.reason == "stop_loss"
            hit_target = protection.triggered and protection.reason == "take_profit"
            if maximum_loss_amount > 0 and str(position.get("stop_loss_source") or "").startswith("default"):
                # A signal-level dollar risk cap is the explicit stop instruction;
                # do not let the generic percentage fallback close it first.
                hit_stop = False
            contract_pnl = (
                (entry_price - current_price) if short_position else (current_price - entry_price)
            ) * held_qty * 100.0
            hit_max_loss = maximum_loss_amount > 0 and contract_pnl <= -maximum_loss_amount
            hit_underlying_exit = False
            underlying_observed = 0.0
            if underlying_exit_direction in {"above", "below"} and underlying_exit_price > 0:
                underlying_price, _ = await asyncio.to_thread(
                    alpaca.get_latest_price, str(position.get("root") or "")
                )
                underlying_observed = _as_float(underlying_price)
                if underlying_observed > 0:
                    hit_underlying_exit = (
                        underlying_observed > underlying_exit_price
                        if underlying_exit_direction == "above"
                        else underlying_observed < underlying_exit_price
                    )
            if not hit_stop and not hit_target and not hit_max_loss and not hit_underlying_exit and not exit_before_close_now:
                continue

            exit_reason = (
                "option_stop_loss" if hit_stop
                else "option_target_price" if hit_target
                else "option_maximum_loss" if hit_max_loss
                else "option_underlying_stop" if hit_underlying_exit
                else "option_time_exit"
            )
            trigger_price = (
                stop_loss if hit_stop else active_target if hit_target else
                maximum_loss_amount if hit_max_loss else underlying_exit_price if hit_underlying_exit else current_price
            )
            sell_qty = min(tracked_qty, held_qty)
            partial_target = bool(hit_target and target_prices and target_index < len(target_prices) - 1)
            if partial_target:
                remaining_targets = max(1, len(target_prices) - target_index)
                sell_qty = max(1.0, min(sell_qty, float(int(held_qty // remaining_targets) or 1)))

            if not market_open:
                if _has_pending_option_exit(occ_symbol):
                    continue
                await asyncio.to_thread(_queue_option_exit, position, sell_qty, exit_reason, trigger_price, current_price)
                await asyncio.to_thread(_record_order, occ_symbol, "sell", sell_qty, "queued", "", "option", exit_reason)
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{occ_symbol}: option exit condition reached at ${current_price:.2f}; sell-to-close queued for market open.",
                )
                continue

            open_order, _ = await asyncio.to_thread(alpaca.has_open_order, occ_symbol)
            if open_order:
                continue

            exit_side = "buy" if short_position else "sell"
            exit_intent = "buy_to_close" if short_position else "sell_to_close"
            order, err = await asyncio.to_thread(
                alpaca.submit_option_order,
                occ_symbol,
                exit_side,
                sell_qty,
                "market",
                None,
                exit_intent,
                _client_order_id(
                    "optionexit",
                    f"{occ_symbol}:{exit_reason}:{trigger_price}:{sell_qty}",
                ),
            )
            if not order:
                if _is_market_closed_order_error(err):
                    if _has_pending_option_exit(occ_symbol):
                        continue
                    await asyncio.to_thread(_queue_option_exit, position, sell_qty, exit_reason, trigger_price, current_price)
                    await asyncio.to_thread(_record_order, occ_symbol, "sell", sell_qty, "queued", "", "option", exit_reason)
                    await _send_channel(
                        config.discord_paper_log_channel_id or config.discord_review_channel_id,
                        f"{occ_symbol}: option exit condition reached at ${current_price:.2f}; sell-to-close queued for market open.",
                    )
                    continue
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": occ_symbol, "category": "option_exit_failed", "reason": err},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{occ_symbol}: option exit condition reached, but the sell-to-close needs manual review.",
                )
                continue

            await asyncio.to_thread(_record_order, occ_symbol, exit_side, sell_qty, "submitted", str(order.get("id") or ""), "option", exit_reason)
            await _track_submitted_exit(
                order,
                occ_symbol,
                sell_qty,
                "option",
                exit_reason,
                target_index_after_fill=(target_index + 1 if partial_target else None),
            )
            await asyncio.to_thread(
                record_option_journal_entry,
                {
                    "occ_symbol": occ_symbol,
                    "root": position.get("root"),
                    "side": position.get("side"),
                    "strike": position.get("strike"),
                    "expiry_date": position.get("expiry_date"),
                    "quantity": sell_qty,
                    "status": "exit_submitted",
                    "exit_reason": exit_reason,
                    "observed_price": current_price,
                    "trigger_price": trigger_price,
                },
            )
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{occ_symbol}: option exit triggered at ${current_price:.2f}. Submitted {sell_qty:g} contract(s); awaiting Alpaca fill confirmation.",
            )
        except Exception as exc:  # noqa: BLE001 -- one position's failure must not block the rest of the cycle
            logging.getLogger("discord_stock_prediction_agent").error(
                "_process_option_exit_monitor: failed to process position %s: %s",
                position.get("occ_symbol"), exc,
            )
            continue


def _multi_leg_close_specs(position: dict) -> list[dict]:
    close_specs: list[dict] = []
    for leg in position.get("legs") or []:
        intent = str(leg.get("position_intent") or "").lower()
        if intent == "buy_to_open":
            close_side, close_intent = "sell", "sell_to_close"
        elif intent == "sell_to_open":
            close_side, close_intent = "buy", "buy_to_close"
        else:
            continue
        close_specs.append(
            {
                **leg,
                "side_order": close_side,
                "position_intent": close_intent,
                "requires_position": True,
            }
        )
    return close_specs


def _has_pending_multi_leg_exit(strategy_id: str) -> bool:
    key = str(strategy_id or "")
    if any(
        str(item.get("strategy_id") or "") == key
        and bool(item.get("is_multi_leg_exit"))
        for item in list_pending_option_orders()
    ):
        return True
    return any(
        str(item.get("symbol") or "") == key
        and str(item.get("asset_type") or "").lower() == "option_mleg"
        for item in list_pending_exit_orders()
    )


def _queue_multi_leg_exit(position: dict, reason: str, current_net: float) -> None:
    strategy_id = str(position.get("strategy_id") or "")
    close_legs = _multi_leg_close_specs(position)
    if not strategy_id or len(close_legs) < 2:
        return
    add_pending_option_order(
        {
            "pending_key": f"mleg-exit:{strategy_id}",
            "order_class": "mleg",
            "is_multi_leg_exit": True,
            "strategy_id": strategy_id,
            "root": position.get("root"),
            "structure": f"exit_{position.get('structure') or 'multi_leg'}",
            "qty": position.get("qty"),
            "order_type": "market",
            "limit_price": None,
            "price_effect": position.get("price_effect"),
            "legs": close_legs,
            "contract_pending": False,
            "reason": reason,
            "observed_net_price": current_net,
        }
    )


async def _process_multi_leg_exit_monitor(market_open: bool) -> None:
    for position in list_multi_leg_positions():
        # Regression: nothing isolated one position's failure here -- an
        # exception on a single malformed/unexpected position would abort
        # this whole loop, skipping every position after it that cycle, and
        # if the same condition recurs every cycle, that one position could
        # permanently block all the others from ever being checked again.
        try:
            strategy_id = str(position.get("strategy_id") or "")
            qty = _as_float(position.get("qty"))
            entry_net = _as_float(position.get("entry_net_price"))
            close_legs = _multi_leg_close_specs(position)
            if not strategy_id or qty <= 0 or entry_net <= 0 or len(close_legs) < 2:
                continue
            current_net_signed = 0.0
            complete_quote = True
            stalled_leg = ""
            for leg in position.get("legs") or []:
                symbol = str(leg.get("symbol") or "").upper()
                price, _ = await asyncio.to_thread(alpaca.get_latest_option_price, symbol)
                observed = _as_float(price)
                if observed <= 0:
                    complete_quote = False
                    stalled_leg = symbol
                    break
                ratio = max(1, int(_as_float(leg.get("ratio_qty"), 1)))
                sign = 1.0 if str(leg.get("side_order") or "").lower() == "buy" else -1.0
                current_net_signed += sign * observed * ratio
            if not complete_quote or abs(current_net_signed) <= 0:
                if not complete_quote:
                    logging.getLogger("discord_stock_prediction_agent").warning(
                        "multi-leg protection skipped for %s: no usable quote for leg %s "
                        "this cycle (position/market-data snapshot both unavailable).",
                        strategy_id, stalled_leg,
                    )
                continue
            current_net = abs(current_net_signed)
            short_strategy = str(position.get("price_effect") or "debit").lower() == "credit"
            levels = build_protection_levels(
                entry_net,
                stop_loss_pct=config.option_stop_loss_pct,
                take_profit_pct=config.option_take_profit_pct,
                short_position=short_strategy,
                explicit_stop=_as_float(position.get("stop_loss")) or None,
                explicit_target=_as_float(position.get("target_price")) or None,
            )
            trigger = evaluate_protection(
                current_net, levels, short_position=short_strategy
            )
            maximum_loss_amount = _as_float(position.get("maximum_loss_amount"))
            strategy_pnl = (
                (entry_net - current_net) if short_strategy else (current_net - entry_net)
            ) * qty * 100.0
            hit_max_loss = maximum_loss_amount > 0 and strategy_pnl <= -maximum_loss_amount
            if (not trigger.triggered and not hit_max_loss) or _has_pending_multi_leg_exit(strategy_id):
                continue
            reason = (
                "multi_leg_maximum_loss" if hit_max_loss and not trigger.triggered
                else "multi_leg_stop_loss" if trigger.reason == "stop_loss"
                else "multi_leg_take_profit"
            )
            if not market_open:
                await asyncio.to_thread(_queue_multi_leg_exit, position, reason, current_net)
                continue
            enabled, _ = await asyncio.to_thread(alpaca.has_multi_leg_options_trading)
            if not enabled:
                continue
            order, err = await asyncio.to_thread(
                alpaca.submit_multi_leg_option_order,
                _alpaca_multi_leg_payload(close_legs),
                qty,
                "market",
                None,
                _client_order_id("mlegexit", f"{strategy_id}:{reason}"),
            )
            if not order:
                if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
                    await asyncio.to_thread(_queue_multi_leg_exit, position, reason, current_net)
                else:
                    await asyncio.to_thread(
                        record_safety_block,
                        {"symbol": position.get("root"), "category": reason, "reason": err},
                    )
                continue
            await _track_submitted_exit(
                order,
                strategy_id,
                qty,
                "option_mleg",
                reason,
            )
            await asyncio.to_thread(
                _record_order,
                str(position.get("root") or ""),
                "sell",
                qty,
                "submitted",
                str(order.get("id") or ""),
                "option_mleg",
                reason,
            )
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{position.get('root')}: {reason.replace('multi_leg_', '').replace('_', ' ')} triggered "
                f"at net ${current_net:.2f}. Atomic close submitted; awaiting Alpaca fill confirmation.",
            )
        except Exception as exc:  # noqa: BLE001 -- one position's failure must not block the rest of the cycle
            logging.getLogger("discord_stock_prediction_agent").error(
                "_process_multi_leg_exit_monitor: failed to process position %s: %s",
                position.get("strategy_id"), exc,
            )
            continue


async def _submit_or_reconcile_queued_market_order(pending: dict) -> tuple[Optional[dict], str]:
    """Submit once, or recover an order accepted before a process interruption."""
    def usable(order: Optional[dict]) -> tuple[Optional[dict], str]:
        if not order:
            return None, ""
        status = str(order.get("status") or "").lower()
        if status in {"canceled", "expired", "rejected"}:
            detail = str(order.get("reject_reason") or order.get("message") or status)
            return None, f"Alpaca order is {status}: {detail}"
        return order, ""

    client_order_id = str(pending.get("client_order_id") or "")
    if client_order_id:
        existing, _ = await asyncio.to_thread(
            alpaca.get_order_by_client_order_id, client_order_id
        )
        if existing:
            return usable(existing)

    submitter = getattr(alpaca, "submit_equity_order", None)
    order_type = str(pending.get("order_type") or "market")
    if submitter is None and order_type == "market":
        order, err = await asyncio.to_thread(
            alpaca.submit_market_order,
            str(pending.get("symbol") or ""),
            str(pending.get("side") or ""),
            _as_float(pending.get("qty")),
            client_order_id,
        )
    elif submitter is None:
        return None, f"Broker adapter does not support {order_type} equity orders."
    else:
        order, err = await asyncio.to_thread(
            submitter,
            str(pending.get("symbol") or ""),
            str(pending.get("side") or ""),
            _as_float(pending.get("qty")),
            order_type,
            pending.get("limit_price"),
            pending.get("stop_price"),
            str(pending.get("time_in_force") or "DAY"),
            client_order_id,
        )
    if order:
        return usable(order)

    # A timeout can happen after Alpaca accepted the request. Reconcile by the
    # stable client_order_id before deciding that the submission failed.
    if client_order_id:
        existing, _ = await asyncio.to_thread(
            alpaca.get_order_by_client_order_id, client_order_id
        )
        if existing:
            return usable(existing)
    return None, err


async def _track_submitted_exit(
    order: dict,
    symbol: str,
    qty: float,
    asset_type: str,
    reason: str,
    **metadata: object,
) -> None:
    order_id = str(order.get("id") or "")
    if not order_id:
        return
    await asyncio.to_thread(
        add_pending_exit_order,
        {
            "order_id": order_id,
            "symbol": symbol,
            "requested_qty": qty,
            "asset_type": asset_type,
            "reason": reason,
            **metadata,
        },
    )


async def _track_submitted_option_entry(
    order: dict,
    occ_symbol: str,
    qty: float,
    metadata: dict,
) -> None:
    order_id = str(order.get("id") or "")
    if not order_id:
        return
    await asyncio.to_thread(
        add_pending_option_entry_order,
        {
            "order_id": order_id,
            "occ_symbol": occ_symbol,
            "requested_qty": qty,
            **metadata,
        },
    )


async def _track_submitted_multi_leg_entry(
    order: dict,
    qty: float,
    metadata: dict,
) -> None:
    order_id = str(order.get("id") or "")
    if not order_id:
        return
    await asyncio.to_thread(
        add_pending_multi_leg_entry_order,
        {
            "order_id": order_id,
            "strategy_id": order_id,
            "requested_qty": qty,
            **metadata,
        },
    )


def _multi_leg_filled_net_price(order: dict, pending: dict) -> float:
    top_level = abs(_as_float(order.get("filled_avg_price")))
    if top_level > 0:
        return top_level
    broker_legs = {
        str(item.get("symbol") or "").upper(): item
        for item in (order.get("legs") or [])
    }
    net = 0.0
    found = False
    for leg in pending.get("legs") or []:
        symbol = str(leg.get("symbol") or "").upper()
        broker_leg = broker_legs.get(symbol) or {}
        price = _as_float(
            broker_leg.get("filled_avg_price") or broker_leg.get("avg_fill_price")
        )
        if price <= 0:
            continue
        ratio = max(1, int(_as_float(leg.get("ratio_qty"), 1)))
        sign = 1.0 if str(leg.get("side_order") or "").lower() == "buy" else -1.0
        net += sign * price * ratio
        found = True
    return abs(net) if found and abs(net) > 0 else 0.0


async def _reconcile_pending_multi_leg_entry_orders() -> None:
    """Activate strategy-level protection only after Alpaca confirms MLeg fills."""
    for pending in _rotating_batch(
        list_pending_multi_leg_entry_orders(), "pending_multi_leg_entries"
    ):
        order_id = str(pending.get("order_id") or "")
        if not order_id:
            await asyncio.to_thread(remove_pending_multi_leg_entry_order, order_id)
            continue
        order, err = await asyncio.to_thread(alpaca.get_order, order_id)
        if not order:
            if err:
                logging.getLogger("discord_stock_prediction_agent").warning(
                    "Could not reconcile multi-leg entry %s yet: %s",
                    order_id,
                    _public_error(err),
                )
            continue
        status = str(order.get("status") or "").lower()
        filled_qty = max(0.0, _as_float(order.get("filled_qty")))
        reconciled_qty = max(0.0, _as_float(pending.get("reconciled_qty")))
        newly_filled = max(0.0, filled_qty - reconciled_qty)
        net_fill = _multi_leg_filled_net_price(order, pending)
        if newly_filled > 0 and net_fill <= 0:
            continue
        if newly_filled > 0:
            if bool(pending.get("protectable", True)):
                await asyncio.to_thread(
                    upsert_multi_leg_position,
                    str(pending.get("strategy_id") or order_id),
                    str(pending.get("root") or ""),
                    str(pending.get("structure") or "multi_leg"),
                    list(pending.get("legs") or []),
                    newly_filled,
                    net_fill,
                    str(pending.get("price_effect") or "debit"),
                    order_id,
                    pending.get("stop_loss"),
                    pending.get("target_price"),
                    config.option_stop_loss_pct,
                    config.option_take_profit_pct,
                    pending.get("maximum_loss_amount"),
                )
            await asyncio.to_thread(
                update_pending_multi_leg_entry_order,
                order_id,
                reconciled_qty=filled_qty,
                last_status=status,
                last_fill_price=net_fill,
            )
            reconciled_qty = filled_qty
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{pending.get('root')}: Alpaca confirmed {newly_filled:g} multi-leg fill(s) "
                f"at net ${net_fill:.2f}."
                + (
                    " Strategy protection is active."
                    if bool(pending.get("protectable", True))
                    else " The roll/closing strategy was reconciled without creating a new combined protection position."
                ),
            )
        if status in {"filled", "canceled", "expired", "rejected"}:
            if status == "filled" and filled_qty > reconciled_qty:
                continue
            if status != "filled":
                await asyncio.to_thread(
                    record_safety_block,
                    {
                        "symbol": str(pending.get("root") or ""),
                        "category": "multi_leg_entry_not_filled",
                        "reason": str(order.get("reject_reason") or status),
                    },
                )
            await asyncio.to_thread(remove_pending_multi_leg_entry_order, order_id)


async def _reconcile_pending_option_entry_orders() -> None:
    """Create protected option positions only for quantities Alpaca actually filled."""
    for pending in _rotating_batch(
        list_pending_option_entry_orders(), "pending_option_entries"
    ):
        order_id = str(pending.get("order_id") or "")
        occ_symbol = str(pending.get("occ_symbol") or "").upper()
        if not order_id or not occ_symbol:
            await asyncio.to_thread(remove_pending_option_entry_order, order_id)
            continue
        order, err = await asyncio.to_thread(alpaca.get_order, order_id)
        if not order:
            if err:
                logging.getLogger("discord_stock_prediction_agent").warning(
                    "Could not reconcile option entry %s yet: %s", order_id, _public_error(err)
                )
            continue
        status = str(order.get("status") or "").lower()
        filled_qty = max(0.0, _as_float(order.get("filled_qty")))
        reconciled_qty = max(0.0, _as_float(pending.get("reconciled_qty")))
        newly_filled = max(0.0, filled_qty - reconciled_qty)
        fill_price = _as_float(order.get("filled_avg_price"))
        if newly_filled > 0 and fill_price <= 0:
            # Alpaca can expose the terminal status before fill pricing is
            # populated. Keep the durable tracker and retry next cycle.
            continue
        if newly_filled > 0 and fill_price > 0:
            opened_by = str(pending.get("opened_by") or "")
            is_automate_agent = opened_by == AUTOMATE_AGENT_TAG
            await asyncio.to_thread(
                upsert_option_position,
                occ_symbol,
                str(pending.get("root") or ""),
                str(pending.get("side") or ""),
                pending.get("strike"),
                str(pending.get("expiry_date") or ""),
                newly_filled,
                fill_price,
                order_id,
                pending.get("stop_loss"),
                pending.get("target_price"),
                pending.get("signal_quality"),
                str(pending.get("position_intent") or "buy_to_open"),
                pending.get("target_prices"),
                pending.get("trailing_stop_pct"),
                bool(pending.get("exit_before_market_close")),
                pending.get("exit_minutes_before_close"),
                bool(pending.get("exit_if_target_not_hit")),
                pending.get("risk_stop_pct"),
                pending.get("position_type"),
                pending.get("maximum_loss_amount"),
                pending.get("exit_underlying_direction"),
                pending.get("exit_underlying_price"),
                pending.get("time_in_force"),
                pending.get("stop_scope"),
                # automate_agent's own option positions use their own
                # premium-based percentages (20%/25% by default -- premium
                # swings far more than the underlying, so these are wider
                # than automate_agent's equity SL/TP on purpose); every
                # other option position keeps the normal option defaults.
                config.automate_agent_option_stop_loss_pct if is_automate_agent else config.option_stop_loss_pct,
                config.automate_agent_option_take_profit_pct if is_automate_agent else config.option_take_profit_pct,
                opened_by,
            )
            await asyncio.to_thread(
                update_pending_option_entry_order,
                order_id,
                reconciled_qty=filled_qty,
                last_status=status,
                last_fill_price=fill_price,
            )
            reconciled_qty = filled_qty
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{occ_symbol}: Alpaca confirmed {newly_filled:g} option entry fill(s) at ${fill_price:.2f}. Protection monitoring is active.",
            )

        if status in {"filled", "canceled", "expired", "rejected"}:
            if status == "filled" and filled_qty > reconciled_qty:
                continue
            if status != "filled":
                await asyncio.to_thread(
                    record_safety_block,
                    {
                        "symbol": occ_symbol,
                        "category": "option_entry_not_filled",
                        "reason": str(order.get("reject_reason") or status),
                    },
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{occ_symbol}: submitted option entry ended as {status}; no local position was created for unfilled quantity.",
                )
            await asyncio.to_thread(remove_pending_option_entry_order, order_id)


async def _reconcile_pending_exit_orders() -> None:
    """Apply position changes only after Alpaca reports actual exit fills."""
    for pending in _rotating_batch(list_pending_exit_orders(), "pending_exits"):
        # Regression: nothing isolated one pending exit's failure here --
        # an exception reconciling a single order (a malformed broker
        # response, an unexpected field) would abort this whole loop,
        # leaving every other pending exit in this batch un-reconciled that
        # cycle. This is the function that actually finalizes a sell into a
        # closed position -- it must never let one bad order prevent the
        # rest of a force-close (e.g. the 12:30 cutoff) from being recorded.
        try:
            order_id = str(pending.get("order_id") or "")
            asset_type = str(pending.get("asset_type") or "equity").lower()
            raw_symbol = str(pending.get("symbol") or "")
            symbol = raw_symbol if asset_type == "option_mleg" else raw_symbol.upper()
            if not order_id or not symbol:
                await asyncio.to_thread(remove_pending_exit_order, order_id)
                continue

            order, err = await asyncio.to_thread(alpaca.get_order, order_id)
            if not order:
                if err:
                    logging.getLogger("discord_stock_prediction_agent").warning(
                        "Could not reconcile exit order %s yet: %s", order_id, _public_error(err)
                    )
                continue

            status = str(order.get("status") or "").lower()
            filled_qty = max(0.0, _as_float(order.get("filled_qty")))
            reconciled_qty = max(0.0, _as_float(pending.get("reconciled_qty")))
            newly_filled = max(0.0, filled_qty - reconciled_qty)
            fill_price = _as_float(order.get("filled_avg_price"))
            outcome = {}

            if newly_filled > 0:
                if asset_type not in {"option", "option_mleg"} and fill_price <= 0:
                    # Never learn or close local equity state using a quote as a
                    # substitute for the broker's real fill price.
                    continue
                if asset_type == "option_mleg":
                    await asyncio.to_thread(
                        reduce_or_remove_multi_leg_position, symbol, newly_filled
                    )
                elif asset_type == "option":
                    broker_position, position_err = await asyncio.to_thread(
                        alpaca.get_position, symbol
                    )
                    broker_qty = abs(_as_float((broker_position or {}).get("qty")))
                    if broker_position and broker_qty > 0:
                        updates: dict[str, object] = {"qty": round(broker_qty, 6)}
                        if status == "filled" and pending.get("target_index_after_fill") is not None:
                            updates["target_index"] = int(pending["target_index_after_fill"])
                        if status == "filled" and pending.get("move_stop_to_breakeven"):
                            tracked = next(
                                (
                                    item for item in list_option_positions()
                                    if str(item.get("occ_symbol") or "").upper() == symbol
                                ),
                                {},
                            )
                            entry_price = _as_float(tracked.get("entry_price"))
                            if entry_price > 0:
                                updates["stop_loss"] = entry_price
                        await asyncio.to_thread(update_option_position, symbol, **updates)
                    elif status == "filled" and (
                        not position_err or "no position" in position_err.lower()
                    ):
                        tracked = next(
                            (
                                item for item in list_option_positions()
                                if str(item.get("occ_symbol") or "").upper() == symbol
                            ),
                            {},
                        )
                        if str(tracked.get("opened_by") or "") == AUTOMATE_AGENT_TAG and fill_price > 0:
                            # Only automate_agent's own option positions get a
                            # trade_outcomes entry recorded on close -- manual
                            # option positions never have, and this exists
                            # specifically so today_realized_pnl(AUTOMATE_AGENT_TAG)
                            # (the daily-loss circuit breaker) sees option P&L too.
                            await asyncio.to_thread(
                                close_option_position_with_outcome,
                                symbol,
                                newly_filled,
                                fill_price,
                                str(pending.get("reason") or "broker_exit_fill"),
                            )
                        else:
                            await asyncio.to_thread(remove_option_position, symbol)
                elif fill_price > 0:
                    outcome = await asyncio.to_thread(
                        close_position_with_outcome,
                        symbol,
                        newly_filled,
                        fill_price,
                        str(pending.get("reason") or "broker_exit_fill"),
                    )
                    if not outcome:
                        await asyncio.to_thread(reduce_or_remove_position, symbol, newly_filled)

                await asyncio.to_thread(
                    update_pending_exit_order,
                    order_id,
                    reconciled_qty=filled_qty,
                    last_status=status,
                    last_fill_price=fill_price,
                )
                reconciled_qty = filled_qty
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: Alpaca confirmed {newly_filled:g} exit fill(s)"
                    + (f" at ${fill_price:.2f}" if fill_price > 0 else "")
                    + ". Local protection state was reconciled."
                    + (f" Learned outcome: {outcome.get('pnl_pct'):+.2f}%." if outcome else ""),
                )

            if status in {"filled", "canceled", "expired", "rejected"}:
                if status == "filled" and filled_qty > reconciled_qty:
                    continue
                if status != "filled":
                    await asyncio.to_thread(
                        record_safety_block,
                        {
                            "symbol": symbol,
                            "category": "submitted_exit_not_filled",
                            "reason": str(order.get("reject_reason") or status),
                        },
                    )
                    await _send_channel(
                        config.discord_paper_log_channel_id or config.discord_review_channel_id,
                        f"{symbol}: submitted exit ended as {status}. The tracked position was retained for safety.",
                    )
                await asyncio.to_thread(remove_pending_exit_order, order_id)
        except Exception as exc:  # noqa: BLE001 -- one order's failure must not block reconciling the rest
            logging.getLogger("discord_stock_prediction_agent").error(
                "_reconcile_pending_exit_orders: failed to process order %s: %s",
                pending.get("order_id"), exc,
            )
            continue


async def _process_pending_sells() -> None:
    queued_sells = [
        item for item in list_queued_market_orders()
        if str(item.get("side") or "").lower() == "sell"
    ]
    for pending in _rotating_batch(queued_sells, "pending_sells"):
        pending_key = str(pending.get("queue_id") or "")
        # Regression: nothing isolated one item's failure here -- an
        # exception on a single item (a malformed record, an unexpected
        # API response) would abort this whole loop, skipping every item
        # after it that cycle, and if the same condition recurs every
        # cycle, that one item could permanently block all the others.
        try:
            symbol = str(pending.get("symbol") or "").upper()
            qty = _as_float(pending.get("qty"))
            if not symbol or qty <= 0:
                await asyncio.to_thread(
                    mark_market_order_failed, pending_key, "Invalid queued SELL data."
                )
                continue
            expired, expiry_detail = _pending_order_expired(pending.get("attempts"), pending.get("created_at"))
            if expired:
                await asyncio.to_thread(remove_queued_market_order, pending_key)
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": symbol, "category": "queued_sell_expired", "reason": str(pending.get("last_error") or "")},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: queued SELL expired after {expiry_detail} "
                    f"(last error: {str(pending.get('last_error') or 'none')}). "
                    "Removed -- please check manually.",
                )
                continue
            is_short = str(pending.get("action") or "").upper() == "SELL_SHORT"
            held = 0.0
            if not is_short:
                ok, held, reason = await asyncio.to_thread(alpaca.has_sellable_quantity, symbol, qty)
                if not ok:
                    await asyncio.to_thread(mark_market_order_attempt, pending_key, reason)
                    continue
            sell_qty = qty if is_short else min(qty, held)
            pending["qty"] = sell_qty
            order, err = await _submit_or_reconcile_queued_market_order(pending)
            if not order:
                if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
                    await asyncio.to_thread(mark_market_order_attempt, pending_key, err)
                    logging.getLogger("discord_stock_prediction_agent").warning(
                        "Retaining queued SELL %s after transient Alpaca failure: %s",
                        pending_key,
                        _public_error(err),
                    )
                    continue
                await asyncio.to_thread(mark_market_order_failed, pending_key, err)
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": symbol, "category": "queued_sell_failed", "reason": err},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: queued SELL attempted but was not placed. {_public_error(err)}",
                )
                continue
            await asyncio.to_thread(_record_order, symbol, "sell", sell_qty, "submitted", str(order.get("id") or ""), "equity", "queued_sell")
            if is_short:
                # A queued SELL_SHORT is a brand-new short entry, not an exit of an
                # existing position -- track it the same way _handle_sell does for
                # an immediate fill, instead of treating it as closing something.
                checked, _ = await asyncio.to_thread(alpaca.wait_for_order, str(order.get("id") or ""), 8)
                checked = checked or order
                entry_price = _as_float(checked.get("filled_avg_price"))
                filled_qty = _as_float(checked.get("filled_qty")) or sell_qty
                if entry_price > 0:
                    await asyncio.to_thread(
                        upsert_position,
                        symbol,
                        filled_qty,
                        entry_price,
                        str(order.get("id") or ""),
                        config.equity_stop_loss_pct,
                        config.equity_take_profit_pct,
                        "short",
                    )
            else:
                await _track_submitted_exit(
                    order,
                    symbol,
                    sell_qty,
                    "equity",
                    str(pending.get("reason") or "queued_sell"),
                )
            await asyncio.to_thread(remove_queued_market_order, pending_key)
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{symbol}: queued {'SHORT SELL' if is_short else 'SELL'} submitted after market opened. "
                f"Qty {sell_qty:g}; awaiting Alpaca fill confirmation.",
            )
        except Exception as exc:  # noqa: BLE001 -- one item's failure must not block the rest
            logging.getLogger("discord_stock_prediction_agent").error(
                "_process_pending_sells: failed to process %s: %s",
                pending.get("queue_id"), exc,
            )
            continue


async def _process_pending_market_buys() -> None:
    queued_buys = [
        item for item in list_queued_market_orders()
        if str(item.get("side") or "").lower() == "buy"
    ]
    for pending in _rotating_batch(queued_buys, "pending_buys"):
        # Regression: nothing isolated one item's failure here -- an
        # exception on a single item (a malformed record, an unexpected
        # API response) would abort this whole loop, skipping every item
        # after it that cycle, and if the same condition recurs every
        # cycle, that one item could permanently block all the others.
        try:
            pending_key = str(pending.get("queue_id") or "")
            symbol = str(pending.get("symbol") or "").upper()
            qty = _as_float(pending.get("qty"))
            if not symbol or qty <= 0:
                await asyncio.to_thread(
                    mark_market_order_failed, pending_key, "Invalid queued BUY data."
                )
                continue
            expired, expiry_detail = _pending_order_expired(pending.get("attempts"), pending.get("created_at"))
            if expired:
                await asyncio.to_thread(remove_queued_market_order, pending_key)
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": symbol, "category": "queued_buy_expired", "reason": str(pending.get("last_error") or "")},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: queued BUY expired after {expiry_detail} "
                    f"(last error: {str(pending.get('last_error') or 'none')}). "
                    "Removed -- please check manually.",
                )
                continue
            if str(pending.get("action") or "").upper() == "BUY_TO_COVER":
                position, reason = await asyncio.to_thread(alpaca.get_position, symbol)
                held = _as_float((position or {}).get("qty"))
                if held >= 0:
                    await asyncio.to_thread(mark_market_order_attempt, pending_key, reason or "No short position to cover.")
                    continue
                pending["qty"] = min(qty, abs(held))
            order, err = await _submit_or_reconcile_queued_market_order(pending)
            if not order:
                if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
                    await asyncio.to_thread(mark_market_order_attempt, pending_key, err)
                    continue
                await asyncio.to_thread(mark_market_order_failed, pending_key, err)
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": symbol, "category": "queued_buy_failed", "reason": err},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: queued BUY attempted but was not placed. {_public_error(err)}",
                )
                continue
            order_id = str(order.get("id") or "")
            if not order_id:
                await asyncio.to_thread(
                    mark_market_order_attempt,
                    pending_key,
                    "Alpaca accepted the request without returning an order ID.",
                )
                continue
            if str(pending.get("action") or "").upper() == "BUY_TO_COVER":
                await _track_submitted_exit(
                    order, symbol, _as_float(pending.get("qty")), "equity", "queued_buy_to_cover"
                )
                await asyncio.to_thread(remove_queued_market_order, pending_key)
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{symbol}: queued BUY TO COVER submitted after market opened. "
                    f"Qty {_as_float(pending.get('qty')):g}. Order ID `{order_id}`.",
                )
                continue
            # Track the broker order before deleting the durable queue record. This
            # ordering prevents a crash from losing the fill/protection lifecycle.
            await asyncio.to_thread(
                add_pending_buy,
                symbol,
                qty,
                order_id,
                config.equity_stop_loss_pct,
                0.0,
                config.equity_take_profit_pct,
            )
            await asyncio.to_thread(remove_queued_market_order, pending_key)
            await asyncio.to_thread(_record_order, symbol, "buy", qty, "submitted", order_id, "equity", "queued_buy")
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{symbol}: queued BUY submitted after market opened. Qty {qty:g}. Order ID `{order_id or '-'}`.",
            )
        except Exception as exc:  # noqa: BLE001 -- one item's failure must not block the rest
            logging.getLogger("discord_stock_prediction_agent").error(
                "_process_pending_market_buys: failed to process %s: %s",
                pending.get("queue_id"), exc,
            )
            continue


async def _recover_persisted_orders_on_startup() -> None:
    """Immediately resume durable orders when a restarted agent finds an open market."""
    logger = logging.getLogger("discord_stock_prediction_agent")
    market_queue = await asyncio.to_thread(market_order_queue_summary)
    option_queue_count = len(list_pending_option_orders())
    if not market_queue["queued"] and not option_queue_count:
        return
    if not alpaca.ready():
        logger.warning(
            "Persisted orders are waiting, but Alpaca paper trading is not configured or enabled."
        )
        return

    market_open, market_err = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        logger.info(
            "Startup recovery retained %s equity and %s option order(s); market is closed%s.",
            market_queue["queued"],
            option_queue_count,
            f" ({_public_error(market_err)})" if market_err else "",
        )
        return

    logger.info(
        "Market is open. Startup recovery is processing %s equity and %s option order(s).",
        market_queue["queued"],
        option_queue_count,
    )
    await _process_pending_market_buys()
    await _activate_filled_pending_buys()
    await _process_pending_sells()
    await _process_pending_option_orders()
    await _reconcile_pending_option_entry_orders()
    await _reconcile_pending_multi_leg_entry_orders()

    remaining = await asyncio.to_thread(market_order_queue_summary)
    logger.info(
        "Startup order recovery completed. Equity queue remaining: %s; option queue remaining: %s.",
        remaining["queued"],
        len(list_pending_option_orders()),
    )


def _migrate_legacy_market_order_queue() -> None:
    """Move pre-upgrade queued equity orders out of agent_state.json once."""
    for pending in list_pending_buys():
        if not pending.get("queued"):
            continue
        pending_key = str(pending.get("pending_key") or pending.get("order_id") or "")
        enqueue_market_order(
            str(pending.get("symbol") or ""),
            "buy",
            _as_float(pending.get("qty")),
            str(pending.get("reason") or "legacy_market_queue"),
            config.stop_loss_pct,
            pending_key,
        )
        remove_pending_buy(pending_key)

    for pending in list_pending_sells():
        pending_key = str(pending.get("pending_key") or pending.get("symbol") or "")
        enqueue_market_order(
            str(pending.get("symbol") or ""),
            "sell",
            _as_float(pending.get("qty")),
            str(pending.get("reason") or "legacy_market_queue"),
            config.stop_loss_pct,
            pending_key,
        )
        remove_pending_sell(pending_key)


async def _process_pending_multi_leg_order(pending: dict, pending_key: str) -> None:
    root = str(pending.get("root") or "").upper()
    qty = _as_float(pending.get("qty"))
    leg_specs = list(pending.get("legs") or [])
    if not root or qty <= 0 or not 2 <= len(leg_specs) <= 4:
        await asyncio.to_thread(remove_pending_option_order, pending_key)
        return

    # Same bounded-retry safety net as the single-leg option queue: every
    # failure branch below is a bare `return` (retry next cycle) with no
    # attempt tracking, so a leg that never gets its required position (or a
    # contract that never gets listed) would otherwise retry forever. A
    # price-conditional entry is exempt -- it is meant to wait indefinitely
    # for its own trigger, checked further down.
    is_price_conditional = bool(
        str(pending.get("underlying_trigger_direction") or "") and _as_float(pending.get("underlying_trigger_price")) > 0
    )
    if not is_price_conditional:
        expired, expiry_detail = _pending_order_expired(pending.get("attempts"), pending.get("created_at"))
        if expired:
            reason = str(pending.get("reason") or "queued_multi_leg")
            await asyncio.to_thread(remove_pending_option_order, pending_key)
            await asyncio.to_thread(
                record_safety_block,
                {"symbol": root, "category": f"queued_mleg_{reason}_expired", "reason": ""},
            )
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{root}: queued multi-leg {reason} expired after {expiry_detail}. "
                "Removed -- please check manually.",
            )
            return

    if pending.get("contract_pending") or not all(item.get("symbol") for item in leg_specs):
        contract_check = await asyncio.to_thread(
            _multi_leg_contract_lookup,
            root,
            leg_specs,
            str(pending.get("expiry_date") or ""),
            str(pending.get("expiry_mode") or "") == "0dte",
        )
        resolved_legs = contract_check.get("legs") or []
        if len(resolved_legs) != len(leg_specs):
            return
        pending["legs"] = resolved_legs
        pending["expiry_date"] = str(contract_check.get("expiration_date") or pending.get("expiry_date") or "")
        pending["contract_pending"] = False
        await asyncio.to_thread(remove_pending_option_order, pending_key)
        await asyncio.to_thread(add_pending_option_order, pending)
        leg_specs = resolved_legs

    for leg in leg_specs:
        if not leg.get("requires_position"):
            continue
        position, _ = await asyncio.to_thread(alpaca.get_position, str(leg.get("symbol") or ""))
        held_qty = abs(_as_float((position or {}).get("qty")))
        required_qty = qty * max(1, int(leg.get("ratio_qty") or 1))
        if held_qty < required_qty:
            return

    trigger_direction = str(pending.get("underlying_trigger_direction") or "").lower()
    trigger_price = _as_float(pending.get("underlying_trigger_price"))
    if trigger_direction in {"above", "below"} and trigger_price > 0:
        underlying_price, _ = await asyncio.to_thread(alpaca.get_latest_price, root)
        observed = _as_float(underlying_price)
        condition_met = observed > trigger_price if trigger_direction == "above" else observed < trigger_price
        if observed <= 0 or not condition_met:
            return

    enabled, permission_err = await asyncio.to_thread(alpaca.has_multi_leg_options_trading)
    if not enabled:
        await asyncio.to_thread(
            record_safety_block,
            {"symbol": root, "category": "queued_mleg_permission", "reason": permission_err},
        )
        return

    order_type = str(pending.get("order_type") or "market").lower()
    limit_price = pending.get("limit_price")
    order, err = await asyncio.to_thread(
        alpaca.submit_multi_leg_option_order,
        _alpaca_multi_leg_payload(leg_specs),
        qty,
        order_type,
        _as_float(limit_price) if limit_price is not None else None,
        _client_order_id("queuedmleg", pending_key),
    )
    if not order:
        if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
            return
        public_reason = _public_error(err)
        logging.getLogger("discord_stock_prediction_agent").warning(
            "Queued multi-leg order rejected for %s: %s", root, public_reason
        )
        await asyncio.to_thread(
            record_safety_block,
            {"symbol": root, "category": "queued_mleg_failed", "reason": err},
        )
        await asyncio.to_thread(remove_pending_option_order, pending_key)
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{root}: queued multi-leg option strategy was not submitted. {public_reason}",
        )
        return

    order_id = str(order.get("id") or "")
    if bool(pending.get("is_multi_leg_exit")):
        await _track_submitted_exit(
            order,
            str(pending.get("strategy_id") or ""),
            qty,
            "option_mleg",
            str(pending.get("reason") or "multi_leg_exit"),
        )
    else:
        await _track_submitted_multi_leg_entry(
            order,
            qty,
            {
                "root": root,
                "structure": pending.get("structure"),
                "legs": leg_specs,
                "price_effect": pending.get("price_effect") or "debit",
                "stop_loss": pending.get("stop_loss"),
                "target_price": pending.get("target_price"),
                "target_prices": pending.get("target_prices") or [],
                "maximum_loss_amount": pending.get("maximum_loss_amount"),
                "protectable": all(
                    str(leg.get("position_intent") or "")
                    in {"buy_to_open", "sell_to_open"}
                    for leg in leg_specs
                ),
            },
        )
    await asyncio.to_thread(remove_pending_option_order, pending_key)
    await asyncio.to_thread(
        _record_order,
        root,
        "buy",
        qty,
        "submitted",
        order_id,
        "option_mleg",
        str(pending.get("structure") or "multi_leg"),
    )
    await asyncio.to_thread(
        record_option_journal_entry,
        {
            "root": root,
            "structure": pending.get("structure"),
            "legs": leg_specs,
            "quantity": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "raw_input": pending.get("raw_input"),
            "status": "queued_multi_leg_order_placed",
            "order_id": order_id,
        },
    )
    await _send_channel(
        config.discord_paper_log_channel_id or config.discord_review_channel_id,
        f"{root}: queued multi-leg paper strategy submitted successfully. "
        f"Qty {qty:g}, order ID `{order_id or '-'}`.",
    )


async def _process_pending_option_orders() -> None:
    for pending in _rotating_batch(
        list_pending_option_orders(), "pending_options"
    ):
        pending_key = str(pending.get("pending_key") or pending.get("occ_symbol") or "")
        if str(pending.get("order_class") or "").lower() == "mleg":
            await _process_pending_multi_leg_order(pending, pending_key)
            continue
        occ_symbol = str(pending.get("occ_symbol") or "").upper()
        root = str(pending.get("root") or "").upper()
        side = str(pending.get("side") or "").upper()
        qty = _as_float(pending.get("qty"))
        order_type = str(pending.get("order_type") or "market").lower()
        limit_price = pending.get("limit_price")
        order_side = str(pending.get("order_side") or "buy").lower()
        reason = str(pending.get("reason") or "queued_option")
        if order_side not in {"buy", "sell"}:
            order_side = "buy"
        if qty <= 0 or (not occ_symbol and not pending.get("contract_pending")):
            await asyncio.to_thread(remove_pending_option_order, pending_key)
            continue

        if _option_cancel_deadline_passed(
            pending.get("semantic_contract"), pending.get("created_at")
        ):
            await asyncio.to_thread(remove_pending_option_order, pending_key)
            await asyncio.to_thread(
                record_option_journal_entry,
                {
                    "occ_symbol": occ_symbol,
                    "root": root,
                    "side": side,
                    "strike": pending.get("strike"),
                    "expiry_date": pending.get("expiry_date"),
                    "raw_input": pending.get("raw_input"),
                    "semantic_contract": pending.get("semantic_contract") or {},
                    "status": "conditional_entry_expired",
                },
            )
            await _send_channel(
                config.discord_paper_log_channel_id
                or config.discord_review_channel_id,
                f"{occ_symbol or root}: conditional entry deadline passed. The queued signal was removed without submitting an order.",
            )
            continue

        # A price-conditional entry ("buy once SPX crosses 6000") is meant to
        # wait indefinitely for its own trigger, not a fixed age cutoff --
        # _option_cancel_deadline_passed above is the intended way to bound
        # those, via the user's own stated deadline. Everything else here
        # (plain market-closed signals, contract/position not found yet, and
        # protective exits re-derived fresh every cycle from a live position)
        # has no legitimate reason to sit for weeks, so it gets the same
        # bounded-retry safety net as the equity queue.
        is_price_conditional = bool(
            str(pending.get("underlying_trigger_direction") or "") and _as_float(pending.get("underlying_trigger_price")) > 0
        )
        if not is_price_conditional:
            expired, expiry_detail = _pending_order_expired(pending.get("attempts"), pending.get("created_at"))
            if expired:
                await asyncio.to_thread(remove_pending_option_order, pending_key)
                await asyncio.to_thread(
                    record_safety_block,
                    {"symbol": occ_symbol or root, "category": f"queued_option_{reason}_expired", "reason": ""},
                )
                await _send_channel(
                    config.discord_paper_log_channel_id or config.discord_review_channel_id,
                    f"{occ_symbol or root}: queued option {order_side} ({reason}) expired after "
                    f"{expiry_detail}. Removed -- please check manually.",
                )
                continue

        if pending.get("contract_pending"):
            contracts = None
            original_expiry = str(pending.get("expiry_date") or "")
            for expiry_candidate in listed_expiry_fallbacks(original_expiry) or [original_expiry]:
                contracts, _ = await asyncio.to_thread(
                    alpaca.get_option_contracts,
                    root,
                    expiry_candidate or None,
                    _as_float(pending.get("strike")),
                    side.lower(),
                )
                if contracts:
                    break
            if not contracts and original_expiry:
                contracts, _ = await asyncio.to_thread(
                    alpaca.get_option_contracts,
                    root,
                    None,
                    _as_float(pending.get("strike")),
                    side.lower(),
                )
            if not contracts:
                continue
            contract = sorted(contracts, key=lambda c: str(c.get("expiration_date") or ""))[0]
            occ_symbol = str(contract.get("symbol") or "").upper()
            if not occ_symbol:
                continue
            pending["occ_symbol"] = occ_symbol
            pending["expiry_date"] = str(contract.get("expiration_date") or pending.get("expiry_date") or "")
            pending["contract_pending"] = False
            await asyncio.to_thread(remove_pending_option_order, pending_key)
            await asyncio.to_thread(add_pending_option_order, pending)

        trigger_direction = str(pending.get("underlying_trigger_direction") or "").lower()
        trigger_price = _as_float(pending.get("underlying_trigger_price"))
        if trigger_direction in {"above", "below"} and trigger_price > 0:
            underlying_price, _ = await asyncio.to_thread(alpaca.get_latest_price, root)
            observed = _as_float(underlying_price)
            condition_met = observed > trigger_price if trigger_direction == "above" else observed < trigger_price
            if observed <= 0 or not condition_met:
                continue

        open_order, _ = await asyncio.to_thread(alpaca.has_open_order, occ_symbol)
        if open_order:
            continue

        position_intent = str(pending.get("position_intent") or ("sell_to_close" if order_side == "sell" else "buy_to_open"))
        requires_position = bool(pending.get("requires_position"))
        if requires_position:
            base_order_id = str(pending.get("base_order_id") or "")
            if bool(pending.get("scale_in_order")) and base_order_id:
                base_order, _ = await asyncio.to_thread(alpaca.get_order, base_order_id)
                if str((base_order or {}).get("status") or "").lower() != "filled":
                    continue
            alpaca_position, _ = await asyncio.to_thread(alpaca.get_position, occ_symbol)
            held_qty = abs(_as_float((alpaca_position or {}).get("qty")))
            if held_qty <= 0:
                continue
            if not bool(pending.get("scale_in_order")):
                qty = min(qty, held_qty)

        order, err = await asyncio.to_thread(
            alpaca.submit_option_order,
            occ_symbol,
            order_side,
            qty,
            order_type,
            _as_float(limit_price) if limit_price is not None else None,
            position_intent,
            _client_order_id("queuedoption", pending_key),
        )
        if not order:
            if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
                continue
            public_reason = _public_error(err)
            logging.getLogger("discord_stock_prediction_agent").warning(
                "Queued option order rejected for %s: %s", occ_symbol, public_reason
            )
            await asyncio.to_thread(
                record_safety_block,
                {"symbol": occ_symbol, "category": f"queued_option_{order_side}_failed", "reason": err},
            )
            await asyncio.to_thread(remove_pending_option_order, pending_key)
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{occ_symbol}: queued option {order_side.upper()} was not submitted. {public_reason}",
            )
            continue

        await asyncio.to_thread(_record_order, occ_symbol, order_side, qty, "submitted", str(order.get("id") or ""), "option", reason)
        if position_intent in {"sell_to_close", "buy_to_close"}:
            await _track_submitted_exit(
                order,
                occ_symbol,
                qty,
                "option",
                reason,
                target_index_after_fill=(
                    int(_as_float(pending.get("next_target_index")))
                    if pending.get("next_target_index") is not None
                    else None
                ),
                move_stop_to_breakeven=bool(pending.get("move_stop_to_breakeven")),
            )
            await asyncio.to_thread(remove_pending_option_order, pending_key)
            await asyncio.to_thread(
                record_option_journal_entry,
                {
                    "occ_symbol": occ_symbol,
                    "root": root,
                    "side": side,
                    "quantity": qty,
                    "order_type": order_type,
                    "stop_loss": pending.get("stop_loss"),
                    "target_price": pending.get("target_price"),
                    "raw_input": pending.get("raw_input"),
                    "status": "queued_exit_submitted",
                    "exit_reason": reason,
                    "observed_price": pending.get("observed_price"),
                    "trigger_price": pending.get("trigger_price"),
                },
            )
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{occ_symbol}: queued option {position_intent} submitted after monitor check. Qty {qty:g} contract(s).",
            )
            continue

        await _track_submitted_option_entry(
            order,
            occ_symbol,
            qty,
            {
                "root": root,
                "side": side,
                "strike": pending.get("strike"),
                "expiry_date": str(pending.get("expiry_date") or ""),
                "stop_loss": pending.get("stop_loss"),
                "target_price": pending.get("target_price"),
                "target_prices": pending.get("target_prices"),
                "trailing_stop_pct": pending.get("trailing_stop_pct"),
                "exit_before_market_close": bool(pending.get("exit_before_market_close")),
                "exit_minutes_before_close": pending.get("exit_minutes_before_close"),
                "exit_if_target_not_hit": bool(pending.get("exit_if_target_not_hit")),
                "risk_stop_pct": pending.get("risk_stop_pct"),
                "maximum_loss_amount": pending.get("maximum_loss_amount"),
                "position_type": pending.get("position_type"),
                "exit_underlying_direction": pending.get("exit_underlying_direction"),
                "exit_underlying_price": pending.get("exit_underlying_price"),
                "time_in_force": pending.get("time_in_force"),
                "stop_scope": pending.get("stop_scope"),
                "semantic_contract": pending.get("semantic_contract") or {},
                "signal_quality": pending.get("signal_quality"),
                "position_intent": position_intent,
            },
        )
        if reason != "conditional_scale_in" and (
            pending.get("add_trigger_premium")
            or pending.get("add_trigger_underlying_price")
        ):
            scale_metadata = dict(pending)
            scale_metadata["occ_symbol"] = occ_symbol
            scale_metadata["base_order_id"] = str(order.get("id") or "")
            await asyncio.to_thread(_queue_conditional_option_scale_in, scale_metadata)
        await asyncio.to_thread(remove_pending_option_order, pending_key)
        await asyncio.to_thread(
            record_option_journal_entry,
            {
                "occ_symbol": occ_symbol,
                "root": root,
                "side": side,
                "strike": pending.get("strike"),
                "expiry_date": pending.get("expiry_date"),
                "quantity": qty,
                "order_type": order_type,
                "stop_loss": pending.get("stop_loss"),
                "target_price": pending.get("target_price"),
                "signal_quality": pending.get("signal_quality"),
                "risk_reward": pending.get("risk_reward"),
                "raw_input": pending.get("raw_input"),
                "status": "queued_order_submitted",
            },
        )
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{occ_symbol}: queued option {order_side.upper()} submitted successfully after monitor check. "
            f"Qty {qty:g}, type={order_type}"
            f"{' @ $' + f'{_as_float(limit_price):.2f}' if _as_float(limit_price) > 0 else ''}. "
            f"Order ID `{order.get('id') or '-'}`. Position protection activates after Alpaca confirms fills.",
        )


async def _activate_filled_pending_buys() -> None:
    for pending in list_pending_buys():
        order_id = str(pending.get("order_id") or "")
        symbol = str(pending.get("symbol") or "").upper()
        if not order_id or not symbol:
            continue
        order, _ = await asyncio.to_thread(alpaca.get_order, order_id)
        if not order:
            continue
        status = str(order.get("status") or "").lower()
        if status in {"canceled", "expired", "rejected"}:
            await asyncio.to_thread(remove_pending_buy, order_id)
            continue
        if status not in {"filled", "partially_filled"}:
            continue
        entry_price = _as_float(order.get("filled_avg_price"))
        filled_qty = _as_float(order.get("filled_qty"))
        if entry_price <= 0 or filled_qty <= 0:
            continue
        protected_qty = _as_float(pending.get("protected_qty"))
        newly_filled_qty = max(0.0, filled_qty - protected_qty)
        stop_loss_pct = config.equity_stop_loss_pct
        take_profit_pct = config.equity_take_profit_pct
        if newly_filled_qty <= 0:
            if status == "filled":
                await asyncio.to_thread(remove_pending_buy, order_id)
            continue
        await asyncio.to_thread(
            upsert_position,
            symbol,
            newly_filled_qty,
            entry_price,
            order_id,
            stop_loss_pct,
            take_profit_pct,
        )
        if status == "filled":
            await asyncio.to_thread(remove_pending_buy, order_id)
        else:
            await asyncio.to_thread(
                update_pending_buy_protected_qty, order_id, filled_qty
            )
        levels = build_protection_levels(
            entry_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{symbol}: {newly_filled_qty:g} newly filled share(s). Protection active: "
            f"loss ${levels.stop_price:.2f} (-{stop_loss_pct:.1f}%) and profit "
            f"${levels.target_price:.2f} (+{take_profit_pct:.1f}%) from fill ${entry_price:.2f}.",
        )


async def _build_agent_status_text() -> str:
    alpaca_status = "configured" if alpaca.ready() else "not configured/disabled"
    tracked = list_positions()
    tracked_options = list_option_positions()
    summary = get_daily_summary()
    queue = await asyncio.to_thread(queue_stats)
    market_queue = await asyncio.to_thread(market_order_queue_summary)
    return (
        f"{config.agent_name} is online. Agent mode: {get_agent_mode()}. "
        f"automate_agent mode: {get_automate_agent_mode()}. "
        f"Alpaca paper trading: {alpaca_status}. "
        f"Tracked equity positions: {len(tracked)}. Tracked option positions: {len(tracked_options)}. "
        f"Signal queue: {queue['queued']} waiting, {queue['processing']} processing, "
        f"{queue['dead']} dead-letter. Workers: {max(1, min(16, config.signal_worker_concurrency))}. "
        f"Market orders waiting for Alpaca: {market_queue['queued']} queued, "
        f"{market_queue['failed']} failed. "
        f"Today: {summary['signals']} signal(s), {summary['orders']} order event(s), "
        f"{summary['blocks']} safety block(s)."
    )


@bot.command(name="agent_status")
async def agent_status(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_status_text())


async def _can_manage_agent_mode(ctx: commands.Context) -> bool:
    permissions = getattr(ctx.author, "guild_permissions", None)
    if permissions and (
        getattr(permissions, "administrator", False)
        or getattr(permissions, "manage_guild", False)
    ):
        return True
    try:
        return bool(await bot.is_owner(ctx.author))
    except Exception:
        return False


def _is_whatsapp_admin(sender_id: str) -> bool:
    """WhatsApp equivalent of _can_manage_agent_mode -- there is no guild-
    permission concept on WhatsApp, so a separate, explicit allowlist gates
    mode-changing/admin commands there.
    """
    allowed = {item.strip() for item in str(config.whatsapp_admin_sender_ids or "").split(",") if item.strip()}
    return bool(allowed) and str(sender_id or "").strip() in allowed


async def _build_agent_mode_change_text(mode: str, changed_by: str) -> str:
    control = await asyncio.to_thread(set_agent_mode, mode, changed_by)
    if control["mode"] == "ON":
        detail = "Prediction and options-strategy decision checks are active."
    else:
        detail = (
            "Every valid BUY/SELL signal will proceed directly to paper-order handling. "
            "Market, contract, position, price-condition, and broker safeguards remain active."
        )
    return f"Agent mode is now {control['mode']}. {detail}"


async def _set_agent_mode_from_command(ctx: commands.Context, mode: str) -> None:
    if not await _can_manage_agent_mode(ctx):
        await _send_context_output(
            ctx,
            "You need Administrator or Manage Server permission to change Agent mode.",
        )
        return
    await _send_context_output(ctx, await _build_agent_mode_change_text(mode, str(ctx.author.id)))


@bot.command(name="agent_on")
async def agent_on(ctx: commands.Context) -> None:
    await _set_agent_mode_from_command(ctx, "ON")


@bot.command(name="agent_off")
async def agent_off(ctx: commands.Context) -> None:
    await _set_agent_mode_from_command(ctx, "OFF")


async def _build_agent_mode_text() -> str:
    mode = get_agent_mode()
    detail = (
        "normal prediction and strategy decisions are active"
        if mode == "ON"
        else "valid incoming BUY/SELL signals go directly to paper-order handling"
    )
    return f"Agent mode: {mode}. {detail}."


async def _build_automate_agent_mode_change_text(mode: str, changed_by: str) -> str:
    control = await asyncio.to_thread(set_automate_agent_mode, mode, changed_by)
    detail = (
        "automate_agent will scan and trade on its own during market hours "
        "(still gated by agent_on/agent_off, market hours, the cooldown, min-confidence "
        "filter, and the daily-loss circuit breaker)."
        if control["mode"] == "ON"
        else "automate_agent will not scan or trade until turned back on with "
        "!automate_agent_on. Manual BUY/SELL signals are unaffected."
    )
    return f"automate_agent mode is now {control['mode']}. {detail}"


async def _set_automate_agent_mode_from_command(ctx: commands.Context, mode: str) -> None:
    if not await _can_manage_agent_mode(ctx):
        await _send_context_output(
            ctx,
            "You need Administrator or Manage Server permission to change automate_agent mode.",
        )
        return
    await _send_context_output(
        ctx, await _build_automate_agent_mode_change_text(mode, str(ctx.author.id))
    )


@bot.command(name="automate_agent_on")
async def automate_agent_on(ctx: commands.Context) -> None:
    await _set_automate_agent_mode_from_command(ctx, "ON")


@bot.command(name="automate_agent_off")
async def automate_agent_off(ctx: commands.Context) -> None:
    await _set_automate_agent_mode_from_command(ctx, "OFF")


async def _build_automate_agent_mode_text() -> str:
    mode = get_automate_agent_mode()
    detail = (
        "it will scan and trade on its own during market hours"
        if mode == "ON"
        else "it will not scan or trade until turned back on with !automate_agent_on"
    )
    return f"automate_agent mode: {mode}. {detail}."


@bot.command(name="automate_agent_mode")
async def automate_agent_mode(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_automate_agent_mode_text())


@bot.command(name="automate_agent_report")
async def automate_agent_report(ctx: commands.Context) -> None:
    """On-demand version of the automatic post-cutoff report -- shows
    today's automate_agent trades and P&L so far, usable any time (not
    only after the exit-time cutoff, and it doesn't affect whether the
    automatic once-per-day report still fires later)."""
    today_label = _now_et().date().isoformat()
    await _send_context_output(ctx, _build_automate_agent_daily_report_text(today_label))


@bot.command(name="agent_mode")
async def agent_mode(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_mode_text())


async def _build_agent_positions_text() -> str:
    tracked = list_positions()
    if not tracked:
        return "No positions are currently tracked by this agent."
    lines = [
        f"{p['symbol']} ({str(p.get('side') or 'long').upper()}): qty {p['qty']}, "
        f"entry ${float(p['entry_price']):.2f}, stop ${float(p['stop_price']):.2f}, "
        f"target ${float(p.get('target_price') or 0):.2f}"
        for p in tracked
    ]
    return "\n".join(lines)


@bot.command(name="agent_positions")
async def agent_positions(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_positions_text())


async def _build_agent_option_positions_text() -> str:
    tracked = list_option_positions()
    if not tracked:
        return "No option positions are currently tracked by this agent."
    lines = [
        f"{p['occ_symbol']} ({str(p.get('position_intent') or 'buy_to_open').upper()}): "
        f"qty {p['qty']}, entry ${float(p['entry_price']):.2f}"
        for p in tracked
    ]
    return "\n".join(lines)


@bot.command(name="agent_option_positions")
async def agent_option_positions(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_option_positions_text())


async def _build_agent_summary_text() -> str:
    summary = get_daily_summary()
    queue = await asyncio.to_thread(queue_stats)
    return (
        "Daily Agent Summary\n"
        f"Date UTC: {summary['date_utc']}\n"
        f"Signals processed: {summary['signals']}\n"
        f"Decisions logged: {summary['decisions']}\n"
        f"Paper order events: {summary['orders']}\n"
        f"Safety blocks: {summary['blocks']}\n"
        f"Invalid inputs: {summary['invalid']}\n"
        f"No-trade/commentary: {summary['no_trade']}\n"
        f"Signal queue: {queue['queued']} waiting / {queue['processing']} processing / "
        f"{queue['dead']} dead-letter\n"
        f"Tracked equity positions: {summary['equity_positions']}\n"
        f"Tracked option positions: {summary['option_positions']}"
    )


@bot.command(name="agent_summary")
async def agent_summary(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_summary_text())


async def _build_agent_health_text() -> str:
    """Operational readiness without exposing credentials or account data."""
    queue = await asyncio.to_thread(queue_stats)
    market_queue = await asyncio.to_thread(market_order_queue_summary)
    market_open, market_error = await asyncio.to_thread(alpaca.is_market_open)
    # The position-protection monitor stalling with no exception, no restart,
    # and no log activity at all is a real incident this caught live -- the
    # only way anyone noticed was manually checking positions well after the
    # 12:30 force-close should have already happened. 5x its own interval
    # (floor 120s) is generous enough that one slow cycle never false-alarms.
    monitor_stale_after = max(120, config.stop_monitor_seconds * 5)
    monitor_seconds_ago = (
        asyncio.get_event_loop().time() - _stop_loss_monitor_last_tick
        if _stop_loss_monitor_last_tick > 0 else None
    )
    monitor_stalled = monitor_seconds_ago is not None and monitor_seconds_ago > monitor_stale_after
    status = "READY" if not production_config_errors() and not monitor_stalled else "DEGRADED"
    if monitor_seconds_ago is None:
        monitor_line = "Position monitor: has not run yet this process"
    else:
        monitor_line = f"Position monitor: last ran {monitor_seconds_ago:.0f}s ago"
        if monitor_stalled:
            monitor_line += " -- STALLED, positions may be unprotected; restart the process"
    lines = [
        f"Agent Health: {status}",
        f"Agent mode: {get_agent_mode()}",
        f"Alpaca paper configured: {config.has_alpaca and config.uses_paper_alpaca_endpoint}",
        f"Alpaca market open: {market_open}",
        monitor_line,
        f"Signal workers: {max(1, min(16, config.signal_worker_concurrency))}",
        f"Signal queue: {queue['queued']} queued / {queue['processing']} processing / {queue['dead']} dead",
        f"Market-order queue: {market_queue['queued']} queued / {market_queue['failed']} failed",
        f"WhatsApp webhook: {'enabled' if config.whatsapp_webhook_enabled else 'disabled'}",
    ]
    if market_error:
        lines.append("Alpaca clock check is temporarily unavailable.")
    return "\n".join(lines)


@bot.command(name="agent_health")
async def agent_health(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_health_text())


async def _build_agent_dead_letters_text() -> str:
    dead = await asyncio.to_thread(list_dead_signals, 10)
    if not dead:
        return "No dead-letter signals are waiting."
    lines = [f"Dead-letter signals: {len(dead)} shown"]
    for item in dead:
        raw = re.sub(r"\s+", " ", str(item.get("raw_text") or "")).strip()
        lines.append(
            f"- {str(item.get('id') or '')[:10]} [{item.get('transport', 'discord')}] "
            f"attempts={item.get('attempts', 0)}: {raw[:120]}"
        )
    return "\n".join(lines)


@bot.command(name="agent_dead_letters")
async def agent_dead_letters(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_dead_letters_text())


async def _build_agent_retry_dead_text(limit: int) -> str:
    retried = await asyncio.to_thread(retry_dead_signals, max(1, min(1_000, limit)))
    return f"Returned {retried} dead-letter signal(s) to the queue."


@bot.command(name="agent_retry_dead")
async def agent_retry_dead(ctx: commands.Context, limit: int = 100) -> None:
    if not await _can_manage_agent_mode(ctx):
        await _send_context_output(
            ctx, "You need Administrator or Manage Server permission to retry dead letters."
        )
        return
    await _send_context_output(ctx, await _build_agent_retry_dead_text(limit))


async def _build_agent_learning_text() -> str:
    profile = get_learning_profile()
    summary = get_signal_learning_summary()
    parser_summary = get_parser_learning_summary()
    lines = [
        "Agent Learning Summary",
        f"Learning enabled: {config.learning_enabled}",
        f"Closed equity trades learned from: {profile.get('closed_trades', 0)} "
        f"({profile.get('winning_trades', 0)} win / {profile.get('losing_trades', 0)} loss)",
        f"Pattern groups learned: {summary['total_patterns']}",
        f"Signal formats learned: {parser_summary['total_patterns']} from "
        f"{parser_summary['total_seen']} input(s)",
        f"Parser success rate: {parser_summary['success_rate']}%",
    ]
    for item in summary.get("patterns", [])[:8]:
        lines.append(
            f"- {item.get('key')}: seen {item.get('seen')}, approval {item.get('approval_rate')}%, "
            f"avg return {item.get('avg_return')}%, avg score {item.get('avg_score')}"
        )
    return "\n".join(lines)


@bot.command(name="agent_learning")
async def agent_learning(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_learning_text())


async def _build_agent_option_validation_text() -> str:
    summary = get_option_validation_summary()
    lines = [
        "Option Validation Summary",
        f"Recent validations tracked: {summary['total']}",
        f"Exact-strike attempts: {summary['exact_attempted']}",
        f"Exact-strike success rate: {summary['exact_success_rate']}%",
        f"Delta-proxy fallbacks used: {summary['delta_proxy_used']}",
    ]
    for item in summary.get("recent", [])[:5]:
        lines.append(
            f"- {item.get('root')} {item.get('strike')}{'C' if item.get('side') == 'CALL' else 'P'}: "
            f"exact={item.get('exact_status') or '-'}, proxy={item.get('proxy_status') or '-'}, "
            f"final={item.get('status')}"
        )
    return "\n".join(lines)


@bot.command(name="agent_option_validation")
async def agent_option_validation(ctx: commands.Context) -> None:
    await _send_context_output(ctx, await _build_agent_option_validation_text())


async def _automate_agent_pick_option_contract(
    root: str, side: str, predicted_target_price: Optional[float], risk_budget: float
) -> Optional[dict]:
    """Pick a same-day (0DTE) CALL or PUT for `root`, falling back to the
    nearest later listed expiry within
    config.automate_agent_option_expiry_fallback_days if nothing is listed
    today.

    Scores the config.automate_agent_strike_candidates listed strikes
    nearest the current price by their expected payoff at
    predicted_target_price (automate_agent.select_best_strike), rather than
    blindly taking the nearest-the-money strike regardless of what the
    model itself expects. There is still no options-greeks/delta data
    source anywhere in this codebase (Alpaca's contract-listing endpoint
    returns no greeks, and nothing wraps a snapshot-based delta estimate)
    -- reusing the same predicted_target_price that already produced this
    symbol's BUY/SELL decision is the closest buildable substitute for
    "which strike is actually the best trade," rather than inventing an
    unvalidated greeks estimator for a no-human-review path. Falls back to
    nearest-the-money when there's no predicted target to score against
    (see select_best_strike), matching the previous behavior.
    """
    price, price_err = await asyncio.to_thread(alpaca.get_latest_price, root)
    if not price or price <= 0:
        return None

    normalized_side = "put" if str(side or "").lower().startswith("p") else "call"
    for offset in range(0, max(0, config.automate_agent_option_expiry_fallback_days) + 1):
        expiry = (date.today() + timedelta(days=offset)).isoformat()
        contracts, _ = await asyncio.to_thread(
            alpaca.get_option_contracts, root, expiry, None, normalized_side
        )
        if not contracts:
            continue
        nearest_first = sorted(
            contracts, key=lambda c: abs(_as_float(c.get("strike_price")) - price)
        )[: max(1, config.automate_agent_strike_candidates)]
        quotes: list[StrikeQuote] = []
        for candidate in nearest_first:
            strike = _as_float(candidate.get("strike_price"))
            occ_symbol = str(candidate.get("symbol") or "")
            if strike <= 0 or not occ_symbol:
                continue
            premium, _ = await asyncio.to_thread(alpaca.get_latest_option_price, occ_symbol)
            premium = _as_float(premium)
            if premium > 0:
                quotes.append(StrikeQuote(occ_symbol=occ_symbol, strike=strike, premium=premium))
        if not quotes:
            continue
        chosen = select_best_strike(quotes, normalized_side, predicted_target_price, price, risk_budget)
        if chosen is None:
            continue
        return {
            "occ_symbol": chosen.occ_symbol,
            "strike": chosen.strike,
            "expiry_date": expiry,
            "root": root,
            "premium": chosen.premium,
        }
    return None


async def _automate_agent_buy_option(
    root: str, picked: Optional[BoomCandidate], equity: float, relaxing: bool = False
) -> str:
    """Autonomously validate and place a single-leg CALL (on a BUY signal)
    or PUT (on a SELL signal) for `root`, reusing the same tastytrade
    backtest gate (run_options_strategy_validation) every manually-typed
    option signal already goes through -- never placed without that same
    validation passing. Defaults to CALL if there's no candidate signal to
    read a direction from (matches the previous CALL-only behavior).
    """
    side = "PUT" if picked is not None and picked.decision.upper() == "SELL" else "CALL"
    side_letter = "P" if side == "PUT" else "C"
    already_held = any(
        str(p.get("root") or "").upper() == root
        and str(p.get("opened_by") or "") == AUTOMATE_AGENT_TAG
        for p in list_option_positions()
    )
    if already_held:
        return f"- {root} (option): already holding an automate_agent option position, skipped."
    # Also check for an entry already submitted-but-not-yet-reconciled this
    # cooldown period -- option entries are tracked then reconciled on
    # confirmed fill (like every other option entry in this codebase), so a
    # slow-to-fill order from a recent cycle wouldn't show up in
    # list_option_positions() yet, and without this check a second cycle
    # could submit a duplicate buy for the same root before the first
    # even confirms.
    already_pending = any(
        str(p.get("root") or "").upper() == root
        and str(p.get("opened_by") or "") == AUTOMATE_AGENT_TAG
        for p in list_pending_option_entry_orders()
    )
    if already_pending:
        return f"- {root} (option): an automate_agent option entry is still pending confirmation, skipped."

    size_multiplier = confidence_scaled_risk_multiplier(picked.confidence) if picked is not None else 1.0
    risk_budget = equity * config.automate_agent_risk_pct_per_trade / 100.0 * size_multiplier
    predicted_target_price = picked.predicted_target_price if picked is not None else None
    contract = await _automate_agent_pick_option_contract(root, side, predicted_target_price, risk_budget)
    if not contract:
        return (
            f"- {root} (option): no listed {side} contract found within "
            f"{config.automate_agent_option_expiry_fallback_days} day(s) affordable within the "
            f"${risk_budget:.2f} risk budget, skipped."
        )
    occ_symbol = contract["occ_symbol"]
    premium = _as_float(contract["premium"])

    synthetic = ParsedOptionSignal(
        valid=True,
        root=root,
        strike=contract["strike"],
        side=side,
        expiry_date=contract["expiry_date"],
        expiry_mode="explicit",
        fill_price=premium if premium > 0 else None,
        quantity=1.0,
        order_action="open_long",
        tense="new_order",
        raw_text=f"automate_agent synthetic BUY {root} {contract['strike']}{side_letter} {contract['expiry_date']}",
    )
    strategy_validation = await asyncio.to_thread(run_options_strategy_validation, synthetic)
    if str(strategy_validation.get("decision") or "").upper() != "BUY":
        return (
            f"- {occ_symbol}: strategy validation did not confirm BUY "
            f"({strategy_validation.get('status')}), skipped."
        )

    enabled, options_err = await asyncio.to_thread(alpaca.has_options_trading)
    if not enabled:
        return f"- {occ_symbol}: {_public_error(options_err)}"

    # premium/affordability were already checked by select_best_strike inside
    # _automate_agent_pick_option_contract (it never returns a strike whose
    # premium is <= 0 or exceeds risk_budget) -- contract["premium"] is
    # guaranteed sane here.
    per_contract_cost = premium * 100.0
    contracts = max(1, int(risk_budget // per_contract_cost))

    order, err = await asyncio.to_thread(
        alpaca.submit_option_order,
        occ_symbol, "buy", contracts, "market", None, "buy_to_open",
        _client_order_id("automate_option_buy", f"{occ_symbol}:{contracts}:{premium}"),
    )
    if not order:
        return f"- {occ_symbol}: option order was not placed: {_public_error(err)}"

    order_detail = "automate_agent_buy_relaxed" if relaxing else "automate_agent_buy"
    await asyncio.to_thread(_record_order, occ_symbol, "buy", contracts, "submitted", str(order.get("id") or ""), "option", order_detail)
    await _track_submitted_option_entry(
        order, occ_symbol, contracts,
        {
            "root": root,
            "side": side,
            "strike": contract["strike"],
            "expiry_date": contract["expiry_date"],
            "position_intent": "buy_to_open",
            "opened_by": AUTOMATE_AGENT_TAG,
            "exit_before_market_close": True,
            "exit_minutes_before_close": config.automate_agent_exit_minutes_before_close,
        },
    )
    conviction = f" [confidence {picked.confidence:.0f}]" if picked is not None else ""
    return (
        f"- Bought {occ_symbol}: {contracts:g} {side.lower()} contract(s) @ ~${premium:.2f} "
        f"(stop {config.automate_agent_option_stop_loss_pct:g}% of premium, "
        f"target {config.automate_agent_option_take_profit_pct:g}% of premium, "
        f"auto-closes by {config.automate_agent_exit_time_et} ET if neither is hit first)."
        f"{conviction}"
        f"{' [compulsory minimum-trades fill]' if relaxing else ''}"
    )


async def _scan_automate_agent_watchlist() -> list[BoomCandidate]:
    """Runs the existing prediction engine across the configured watchlist.

    Deliberately reuses run_project_prediction (the same engine every other
    part of this project already relies on) rather than a new, unvalidated
    heuristic. One symbol failing to fetch/predict never aborts the scan for
    the rest of the watchlist.

    Symbols are scanned concurrently, not one at a time -- each is an
    independent, real historical-data-fetch-plus-AI-call that can take real
    time, and there's no reason to make a user wait N times as long as a
    single lookup just because the watchlist has N symbols.

    The whole scan is bounded by automate_agent_scan_timeout_seconds. This
    runs inside _automate_agent_lock, so without a bound, one symbol whose
    data-fetch/AI call stalls would hold that lock (and the autoscan loop)
    hostage indefinitely. asyncio.wait_for around an individual
    asyncio.to_thread call can't actually help here -- once a thread-pool
    call is running, Python cannot preempt it, so wait_for on it just waits
    out the real duration before raising. asyncio.wait(..., timeout=...)
    is used instead: it genuinely returns control after the deadline with
    whatever results are already in, leaving any stragglers to finish
    (or hang) unobserved in the background rather than blocking this cycle.
    """
    async def _predict_one(symbol: str) -> Optional[BoomCandidate]:
        try:
            result = await asyncio.to_thread(run_project_prediction, symbol)
        except Exception as exc:  # noqa: BLE001 -- one bad symbol shouldn't kill the scan
            logging.getLogger("discord_stock_prediction_agent").warning(
                "automate_agent: prediction failed for %s: %s", symbol, exc
            )
            return None
        if result.get("status") != "SUCCESS":
            return None
        ai_prediction = result.get("ai_prediction") or {}
        return BoomCandidate(
            symbol=symbol,
            decision=str(result.get("decision") or ""),
            confidence=_as_float(ai_prediction.get("confidence_score")),
            predicted_return_pct=_as_float(ai_prediction.get("predicted_return_pct")),
            needs_human_review=bool(ai_prediction.get("needs_human_review")),
            predicted_target_price=_as_float(ai_prediction.get("predicted_target_price")) or None,
        )

    tasks = [
        asyncio.ensure_future(_predict_one(symbol))
        for symbol in config.automate_agent_watchlist
    ]
    done, pending = await asyncio.wait(
        tasks, timeout=config.automate_agent_scan_timeout_seconds
    )
    if pending:
        logging.getLogger("discord_stock_prediction_agent").warning(
            "automate_agent: scan timed out after %ss waiting on %d watchlist "
            "symbol(s); proceeding with %d completed result(s) this cycle.",
            config.automate_agent_scan_timeout_seconds, len(pending), len(done),
        )
    results = [task.result() for task in done if not task.cancelled()]
    return [c for c in results if c is not None]


_automate_agent_lock = asyncio.Lock()
_automate_agent_last_run: float = 0.0


def _build_automate_agent_daily_report_text(date_label: str = "") -> str:
    """Every field this needs (symbol, qty, entry/exit price, pnl) is
    already recorded by close_position_with_outcome at the moment a
    position actually closes -- this just formats today's automate_agent-
    tagged records into a report, it doesn't compute anything new.
    """
    outcomes = list_trade_outcomes(opened_by=AUTOMATE_AGENT_TAG, today_only=True)
    header = f"automate_agent daily report ({date_label})" if date_label else "automate_agent daily report"
    if not outcomes:
        return f"{header}: no trades were closed today."
    lines = [header]
    total_pnl = 0.0
    wins = 0
    for outcome in outcomes:
        symbol = str(outcome.get("symbol") or "")
        qty = _as_float(outcome.get("qty"))
        entry_price = _as_float(outcome.get("entry_price"))
        exit_price = _as_float(outcome.get("exit_price"))
        pnl_value = _as_float(outcome.get("pnl_value"))
        pnl_pct = _as_float(outcome.get("pnl_pct"))
        total_pnl += pnl_value
        if pnl_value > 0:
            wins += 1
        # asset_type distinguishes a plain TSLA share trade (entry/exit are
        # a share price) from a single-leg option trade (entry/exit are the
        # contract's premium, and symbol is the full OCC contract -- already
        # self-describing, e.g. TSLA260101C00350000) -- both asset classes
        # can close on the same day now that automate_agent trades both.
        unit = "contract(s)" if str(outcome.get("asset_type") or "") == "option" else "share(s)"
        lines.append(
            f"- {symbol}: {qty:g} {unit}, entered ${entry_price:.2f} -> exited ${exit_price:.2f} "
            f"({pnl_value:+.2f} USD, {pnl_pct:+.2f}%)"
        )
    lines.append(
        f"Total trades: {len(outcomes)} ({wins} win / {len(outcomes) - wins} loss) | "
        f"Total P&L for the day: {total_pnl:+.2f} USD"
    )
    return "\n".join(lines)


async def _maybe_post_automate_agent_daily_report() -> None:
    """Fires once per ET calendar day, only after every automate_agent
    position from today -- equity AND option, now that automate_agent
    trades both -- has actually settled (confirmed exit fill, not just
    "the cutoff time passed") -- posting before that would report an
    incomplete/still-changing total. If a position is somehow still open
    past the cutoff, this just keeps checking next cycle instead of posting
    a premature or duplicate report.
    """
    now_et = _now_et()
    if not _automate_agent_eod_cutoff_reached(now_et, config.automate_agent_exit_time_et):
        return
    today_label = now_et.date().isoformat()
    already_reported = await asyncio.to_thread(get_automate_agent_report_date)
    if already_reported == today_label:
        return
    still_open = any(
        str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
        for p in await asyncio.to_thread(list_positions)
    ) or any(
        str(p.get("opened_by") or "").lower() == AUTOMATE_AGENT_TAG
        for p in await asyncio.to_thread(list_option_positions)
    )
    if still_open:
        return
    report = _build_automate_agent_daily_report_text(today_label)
    await _send_channel(config.discord_review_channel_id, report)
    await asyncio.to_thread(set_automate_agent_report_date, today_label)


async def _build_automate_agent_text() -> str:
    if not alpaca.ready():
        return "automate_agent: Alpaca paper trading is not configured."

    # automate_agent is deliberately independent of the general agent mode:
    # agent_on/agent_off governs whether manual/human-typed signals go
    # through the prediction decision gate (ON) or straight to paper-order
    # handling (OFF) -- that's a single either/or switch for that one path.
    # automate_agent is a separate subsystem with its own switch
    # (!automate_agent_on / !automate_agent_off) and is not gated by
    # agent_on/agent_off at all.
    if not _automate_agent_is_enabled():
        return (
            "automate_agent: automate_agent mode is OFF. Run !automate_agent_on first "
            "if this is intentional."
        )

    market_open, _ = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        return "automate_agent: market is closed. No action taken."

    if _automate_agent_lock.locked():
        return "automate_agent: a scan-and-trade cycle is already running. Try again once it finishes."

    async with _automate_agent_lock:
        global _automate_agent_last_run
        now = asyncio.get_event_loop().time()
        elapsed = now - _automate_agent_last_run
        if elapsed < config.automate_agent_cooldown_seconds:
            wait_left = int(config.automate_agent_cooldown_seconds - elapsed)
            return f"automate_agent: cooling down, try again in {wait_left}s."
        _automate_agent_last_run = now

        account, account_err = await asyncio.to_thread(alpaca.get_account)
        equity = _as_float((account or {}).get("equity"))
        if equity <= 0:
            # Fail safe, not open: if account equity can't be verified, the
            # daily-loss circuit breaker below can't be evaluated either --
            # proceeding anyway (as this used to, falling back to a fixed
            # notional size) would mean the one guard against a runaway
            # autonomous trading day is silently skipped exactly when
            # account state is most in question.
            return (
                "automate_agent: could not verify account equity "
                f"({_public_error(account_err) or 'no account data returned'}); "
                "skipping this cycle for safety."
            )

        # Account-level circuit breaker: independent of any single
        # position's stop-loss, checked before spending time/API calls on
        # a scan we won't act on anyway. A per-trade stop limits one
        # position; this limits the whole autonomous strategy for the rest
        # of a bad day. Existing open positions are still protected and
        # closed normally by stop_loss_monitor -- this only blocks *new*
        # entries.
        realized_today = await asyncio.to_thread(today_realized_pnl, AUTOMATE_AGENT_TAG)
        loss_limit = -abs(equity * config.automate_agent_max_daily_loss_pct / 100.0)
        if realized_today <= loss_limit:
            return (
                f"automate_agent: daily loss circuit breaker tripped "
                f"(realized P&L today: ${realized_today:,.2f}, limit: ${loss_limit:,.2f}). "
                "No new positions will be opened for the rest of the day; existing positions "
                "are still protected by the normal stop-loss/take-profit monitor."
            )

        # Hard ceiling on total orders placed today (equity + option buys
        # combined, same counter the compulsory-minimum check below uses) --
        # distinct from automate_agent_max_positions, which only bounds how
        # many are held *at once*. Without this, a busy day of eviction
        # churn could place far more than automate_agent_max_trades_per_window
        # orders even though the concurrent-position cap was never exceeded.
        trades_today = await asyncio.to_thread(count_today_automate_agent_buys)
        if trades_today >= config.automate_agent_max_trades_per_window:
            return (
                f"automate_agent: today's order quota "
                f"({config.automate_agent_max_trades_per_window}) already reached "
                f"({trades_today} placed); no new positions will be opened for the rest of the "
                "window. Existing positions are still protected and will still be closed by "
                "the exit cutoff."
            )

        candidates = await _scan_automate_agent_watchlist()
        open_positions = await asyncio.to_thread(list_positions)
        # A position mid-eviction (sell submitted, not yet reconciled) is
        # still physically tracked in list_positions() -- upsert_position for
        # its replacement buy happens immediately, but remove_position for the
        # evicted symbol only happens once _reconcile_pending_exit_orders
        # later confirms the fill. Left as-is, that gap let the cap-planning
        # math see both the outgoing and incoming position at once and
        # transiently exceed automate_agent_max_positions by one per pending
        # eviction (observed live: 21 open against a cap of 20). The eviction
        # sell is economically as good as done the moment it's submitted, so
        # exclude those symbols here rather than waiting for reconciliation.
        evicting_symbols = {
            str(order.get("symbol") or "").upper()
            for order in await asyncio.to_thread(list_pending_exit_orders)
            if str(order.get("reason") or "") == "automate_agent_evict"
        }
        if evicting_symbols:
            open_positions = [
                p for p in open_positions
                if str(p.get("symbol") or "").upper() not in evicting_symbols
            ]
        # automate_agent option positions share the same 1-10 slot cap as its
        # equity positions (both asset classes compete for the same
        # AUTOMATE_AGENT_MAX_POSITIONS ceiling) -- normalize them to the same
        # {symbol, opened_by, updated_at, qty} shape plan_automate_trades
        # already expects so count_automate_positions/oldest_automate_position
        # see the combined book. Eviction itself still only ever targets
        # equity positions (see the eviction loop below) -- teaching it to
        # also sell-to-close an option position is deferred.
        automate_option_positions = [
            {
                "symbol": str(p.get("root") or "").upper(),
                "opened_by": p.get("opened_by"),
                "updated_at": p.get("updated_at"),
                "qty": p.get("qty"),
                # oldest_automate_position must never pick one of these as an
                # eviction target -- the eviction loop below can only
                # sell-to-close equity, not options.
                "asset_type": "option",
            }
            for p in await asyncio.to_thread(list_option_positions)
            if str(p.get("opened_by") or "") == AUTOMATE_AGENT_TAG
        ] + [
            # Submitted-but-not-yet-reconciled option entries also count
            # against the cap. Without this, a slow-to-fill order held over
            # from a prior cycle wouldn't be counted at all here, and the
            # next cycle could buy in believing there was more room than
            # will actually exist once that pending entry confirms.
            {
                "symbol": str(p.get("root") or "").upper(),
                "opened_by": p.get("opened_by"),
                "updated_at": p.get("created_at"),
                "qty": p.get("requested_qty"),
                "asset_type": "option",
            }
            for p in await asyncio.to_thread(list_pending_option_entry_orders)
            if str(p.get("opened_by") or "") == AUTOMATE_AGENT_TAG
        ]
        # Compulsory minimum: if too few real trades have been placed today
        # and only a short window remains before the daily exit cutoff,
        # drop the confidence floor for this cycle's ranking so the quota
        # can still be met -- BUY/SELL/HOLD and needs_human_review are
        # still the model's own call either way; only how confident it
        # needed to be is relaxed. Never applies past the cutoff itself
        # (nothing new should be bought once positions are being flattened)
        # and never bypasses the daily-loss circuit breaker above.
        # trades_today was already fetched above for the max-trades-per-
        # window gate -- same counter, reused here rather than re-queried.
        now_et = _now_et()
        min_trades_unmet = trades_today < config.automate_agent_min_trades_per_window
        relaxing = (
            min_trades_unmet
            and not _automate_agent_eod_cutoff_reached(now_et, config.automate_agent_exit_time_et)
            and _automate_agent_in_min_trades_relax_window(
                now_et, config.automate_agent_exit_time_et, config.automate_agent_min_trades_relax_minutes
            )
        )
        effective_min_confidence = 0.0 if relaxing else config.automate_agent_min_confidence
        # A SELL signal is only actionable when automate_agent can actually
        # act on it -- buying a put. Equity has no short-selling path, so
        # this stays False (unchanged) whenever options aren't in play.
        include_sell = config.automate_agent_asset_mode in {"options", "both"}

        plan = plan_automate_trades(
            candidates,
            open_positions + automate_option_positions,
            config.automate_agent_min_positions,
            config.automate_agent_max_positions,
            effective_min_confidence,
            config.automate_agent_max_evictions_per_cycle,
            include_sell,
        )

        if not plan.to_buy:
            unmet_note = (
                f" {trades_today}/{config.automate_agent_min_trades_per_window} of today's compulsory "
                "minimum trades placed so far; no candidate cleared even the relaxed bar this cycle."
                if relaxing else ""
            )
            decision_word = "BUY/SELL-decision" if include_sell else "BUY-decision"
            return (
                f"automate_agent: scanned {len(config.automate_agent_watchlist)} watchlist symbol(s), "
                f"no {decision_word} candidates found this cycle. No trades placed.{unmet_note}"
            )

        lines = ["automate_agent cycle summary"]
        if relaxing:
            lines.append(
                f"- Compulsory minimum trades not yet met ({trades_today}/"
                f"{config.automate_agent_min_trades_per_window}) with "
                f"{config.automate_agent_min_trades_relax_minutes} min left before the "
                f"{config.automate_agent_exit_time_et} ET cutoff -- confidence bar relaxed to 0 for this cycle."
            )

        # "options"-only mode skips equity entirely -- eviction's whole
        # purpose is freeing a slot for a *new equity* pick, so it's gated
        # the same way rather than evicting equity positions for no reason
        # when the agent isn't buying equity this cycle.
        equity_enabled = config.automate_agent_asset_mode in {"equity", "both"}
        equity_to_evict = plan.to_evict if equity_enabled else []
        equity_to_buy = plan.to_buy if equity_enabled else []

        for symbol in equity_to_evict:
            try:
                position = next((p for p in open_positions if p.get("symbol") == symbol), None)
                held_qty = _as_float((position or {}).get("qty"))
                if held_qty <= 0:
                    continue
                order, err = await asyncio.to_thread(
                    alpaca.submit_market_order,
                    symbol, "sell", held_qty,
                    _client_order_id("automate_evict", f"{symbol}:{held_qty}"),
                )
                if order:
                    await asyncio.to_thread(_record_order, symbol, "sell", held_qty, "submitted", str(order.get("id") or ""), "equity", "automate_agent_evict")
                    # Do NOT remove_position here -- that would delete local
                    # protection tracking (and skip recording a trade_outcomes
                    # P&L entry, which today_realized_pnl needs for the daily-
                    # loss circuit breaker) before Alpaca confirms the sell
                    # actually filled. Every other exit path in this file
                    # tracks-then-reconciles for exactly this reason; a
                    # rejected/partial eviction sell must leave the position
                    # tracked and protected, not silently orphaned.
                    await _track_submitted_exit(
                        order, symbol, held_qty, "equity", "automate_agent_evict"
                    )
                    lines.append(f"- Evicted {symbol} ({held_qty:g} sh) to free a slot at the {config.automate_agent_max_positions}-position cap; awaiting Alpaca fill confirmation.")
                else:
                    lines.append(f"- Tried to evict {symbol} but the sell was not placed: {_public_error(err)}")
            except Exception as exc:  # noqa: BLE001 -- one symbol's failure must not abort the cycle
                logging.getLogger("discord_stock_prediction_agent").error(
                    "automate_agent: eviction failed for %s: %s", symbol, exc
                )
                lines.append(f"- Tried to evict {symbol} but hit an unexpected error; left untouched.")

        candidates_by_symbol = {c.symbol.upper(): c for c in candidates}
        bought: list[str] = []
        for symbol in equity_to_buy:
            try:
                price, price_err = await asyncio.to_thread(alpaca.get_latest_price, symbol)
                if not price or price <= 0:
                    lines.append(f"- Skipped {symbol}: could not get a live price ({_public_error(price_err)}).")
                    continue
                # Fixed-fractional sizing: a constant % of *current* equity,
                # not a hardcoded dollar figure, so sizing scales with the
                # account and automatically shrinks after a drawdown. equity
                # is guaranteed > 0 here -- the whole cycle already bailed
                # out above if it couldn't be verified. On top of that, tilt
                # the size (+/-25%) by the candidate's own confidence score,
                # so a 95-confidence pick and a barely-cleared-the-bar pick
                # don't get identical risk.
                picked = candidates_by_symbol.get(symbol.upper())
                size_multiplier = confidence_scaled_risk_multiplier(picked.confidence) if picked is not None else 1.0
                risk_budget = equity * config.automate_agent_risk_pct_per_trade / 100.0 * size_multiplier
                if price > risk_budget:
                    # Whole shares only below -- max(1, ...) would otherwise
                    # force a 1-share buy that blows straight past the
                    # intended risk-based budget for any stock pricier than
                    # the budget itself, silently defeating fixed-fractional
                    # sizing for that symbol instead of just sizing down.
                    lines.append(
                        f"- Skipped {symbol}: share price ${price:.2f} exceeds the "
                        f"${risk_budget:.2f} risk budget for this trade; buying even 1 "
                        "share would oversize the position."
                    )
                    continue
                qty = max(1, int(risk_budget // price))
                order, err = await asyncio.to_thread(
                    alpaca.submit_market_order,
                    symbol, "buy", qty,
                    _client_order_id("automate_buy", f"{symbol}:{qty}:{price}"),
                )
                if order:
                    order_detail = "automate_agent_buy_relaxed" if relaxing else "automate_agent_buy"
                    await asyncio.to_thread(_record_order, symbol, "buy", qty, "submitted", str(order.get("id") or ""), "equity", order_detail)
                    await asyncio.to_thread(
                        upsert_position,
                        symbol, qty, price, str(order.get("id") or ""),
                        config.equity_stop_loss_pct, config.equity_take_profit_pct, "long",
                        AUTOMATE_AGENT_TAG, True,
                    )
                    bought.append(symbol)
                    conviction = (
                        f" [confidence {picked.confidence:.0f}, predicted return {picked.predicted_return_pct:+.2f}%, "
                        f"size x{size_multiplier:.2f}]"
                        if picked is not None
                        else ""
                    )
                    lines.append(
                        f"- Bought {symbol}: {qty:g} sh @ ~${price:.2f} "
                        f"(stop {config.equity_stop_loss_pct:g}%, target {config.equity_take_profit_pct:g}%, "
                        f"auto-closes by {config.automate_agent_exit_time_et} ET if neither is hit first)."
                        f"{conviction}"
                        f"{' [compulsory minimum-trades fill]' if relaxing else ''}"
                    )
                else:
                    lines.append(f"- Tried to buy {symbol} but the order was not placed: {_public_error(err)}")
            except Exception as exc:  # noqa: BLE001 -- one symbol's failure must not abort the cycle
                logging.getLogger("discord_stock_prediction_agent").error(
                    "automate_agent: buy failed for %s: %s", symbol, exc
                )
                lines.append(f"- Tried to buy {symbol} but hit an unexpected error; skipped.")

        if config.automate_agent_asset_mode in {"options", "both"}:
            # The equity loop above already consumed equity_to_buy/
            # equity_to_evict against the shared cap (empty in "options"-only
            # mode, since no equity buys/evictions happened at all there) --
            # re-derive how many slots are actually left before attempting
            # options for the same ranked candidates, so "both" mode can
            # never push the combined equity+option count past
            # automate_agent_max_positions.
            projected_combined_count = (
                count_automate_positions(open_positions + automate_option_positions)
                + len(equity_to_buy) - len(equity_to_evict)
            )
            remaining_slots = max(0, config.automate_agent_max_positions - projected_combined_count)
            for symbol in plan.to_buy[:remaining_slots]:
                try:
                    lines.append(await _automate_agent_buy_option(symbol, candidates_by_symbol.get(symbol.upper()), equity, relaxing))
                except Exception as exc:  # noqa: BLE001 -- one symbol's failure must not abort the cycle
                    logging.getLogger("discord_stock_prediction_agent").error(
                        "automate_agent: option buy failed for %s: %s", symbol, exc
                    )
                    lines.append(f"- Tried to buy an option on {symbol} but hit an unexpected error; skipped.")

        lines.append(
            f"Scanned {len(config.automate_agent_watchlist)} watchlist symbol(s); "
            f"{len(bought)} new position(s) opened this cycle."
        )
        summary = "\n".join(lines)
        await _send_channel(config.discord_review_channel_id, summary)
        return summary


@bot.command(name="automate_agent")
async def automate_agent(ctx: commands.Context) -> None:
    if not await _can_manage_agent_mode(ctx):
        await _send_context_output(
            ctx,
            "You need Administrator or Manage Server permission to run automate_agent "
            "-- it places real trades autonomously.",
        )
        return
    await _send_context_output(ctx, await _build_automate_agent_text())


@tasks.loop(seconds=max(60, config.automate_agent_autoscan_interval_seconds))
async def automate_agent_autoscan() -> None:
    """Optional recurring trigger for the same scan-and-trade cycle as
    !automate_agent, so the feature is actually automated instead of only
    running when a human re-types the command. Every existing safety check
    inside _build_automate_agent_text (agent mode ON, market open, the
    per-cycle cooldown, the daily-loss circuit breaker) still applies
    unchanged -- this loop is just an alternate trigger, not a bypass.
    """
    if not config.automate_agent_autoscan_enabled:
        return
    try:
        await _build_automate_agent_text()
    except Exception:
        logging.getLogger("discord_stock_prediction_agent").exception(
            "automate_agent autoscan cycle failed; will retry next interval."
        )


@automate_agent_autoscan.before_loop
async def before_automate_agent_autoscan() -> None:
    await bot.wait_until_ready()


@automate_agent_autoscan.error
async def automate_agent_autoscan_error(error: BaseException) -> None:
    logging.getLogger("discord_stock_prediction_agent").error(
        "automate_agent autoscan task failed and will be restarted.",
        exc_info=(type(error), error, error.__traceback__),
    )
    await asyncio.sleep(5)
    automate_agent_autoscan.restart()


@tasks.loop(seconds=max(300, config.symbol_cache_refresh_interval_seconds))
async def symbol_cache_refresh_monitor() -> None:
    """Periodically re-checks the Alpaca tradable-symbol cache.

    refresh_symbol_cache_from_alpaca only actually re-fetches once the cache
    is 24h+ stale -- on_ready already checks it once at startup, but that
    left a long-lived process (the recommended deployment pattern) drifting
    stale for days between restarts, since nothing ever checked again.
    """
    status = await asyncio.to_thread(refresh_symbol_cache_from_alpaca)
    if status.get("status") == "refreshed":
        logging.getLogger("discord_stock_prediction_agent").info(
            "Symbol cache refreshed: %s",
            {k: v for k, v in status.items() if k not in {"error", "message"}},
        )


@symbol_cache_refresh_monitor.before_loop
async def before_symbol_cache_refresh_monitor() -> None:
    await bot.wait_until_ready()


@symbol_cache_refresh_monitor.error
async def symbol_cache_refresh_monitor_error(error: BaseException) -> None:
    logging.getLogger("discord_stock_prediction_agent").error(
        "Symbol cache refresh task failed and will be restarted.",
        exc_info=(type(error), error, error.__traceback__),
    )
    await asyncio.sleep(5)
    symbol_cache_refresh_monitor.restart()


# ══════════════════════════════════════════════════════════════════════════════
# WhatsApp command parity -- same builder functions as the Discord @bot.command
# handlers above, since WhatsApp messages never reach Discord's command
# dispatch (they arrive via the webhook straight into the durable queue, not
# discord.py's gateway). See _try_dispatch_whatsapp_command's call site in
# _process_queued_signal_message.
# ══════════════════════════════════════════════════════════════════════════════

_WHATSAPP_ADMIN_COMMANDS = {
    "agent_on", "agent_off", "agent_retry_dead", "automate_agent_on", "automate_agent_off",
}
_WHATSAPP_NOT_AUTHORIZED_TEXT = (
    "You are not authorized to run this command from WhatsApp. "
    "Ask an operator to add your sender ID to WHATSAPP_ADMIN_SENDER_IDS."
)


async def _dispatch_whatsapp_command_text(name: str, args: list[str], sender_id: str) -> Optional[str]:
    """Return the response text for a recognized command name, or None if
    `name` isn't one of the 16 known commands (caller should then treat the
    message as a normal signal instead).
    """
    if name in _WHATSAPP_ADMIN_COMMANDS and not _is_whatsapp_admin(sender_id):
        return _WHATSAPP_NOT_AUTHORIZED_TEXT

    if name == "agent_status":
        return await _build_agent_status_text()
    if name == "agent_on":
        return await _build_agent_mode_change_text("ON", sender_id)
    if name == "agent_off":
        return await _build_agent_mode_change_text("OFF", sender_id)
    if name == "agent_mode":
        return await _build_agent_mode_text()
    if name == "agent_positions":
        return await _build_agent_positions_text()
    if name == "agent_option_positions":
        return await _build_agent_option_positions_text()
    if name == "agent_summary":
        return await _build_agent_summary_text()
    if name == "agent_health":
        return await _build_agent_health_text()
    if name == "agent_dead_letters":
        return await _build_agent_dead_letters_text()
    if name == "agent_retry_dead":
        limit = int(args[0]) if args and args[0].isdigit() else 100
        return await _build_agent_retry_dead_text(limit)
    if name == "agent_learning":
        return await _build_agent_learning_text()
    if name == "agent_option_validation":
        return await _build_agent_option_validation_text()
    if name == "automate_agent_on":
        return await _build_automate_agent_mode_change_text("ON", sender_id)
    if name == "automate_agent_off":
        return await _build_automate_agent_mode_change_text("OFF", sender_id)
    if name == "automate_agent_mode":
        return await _build_automate_agent_mode_text()
    if name == "automate_agent_report":
        return _build_automate_agent_daily_report_text(_now_et().date().isoformat())
    return None


async def _try_dispatch_whatsapp_command(message) -> bool:
    """If this WhatsApp message is one of the 16 !agent_*/!automate_agent_*
    commands, handle it and reply; return True so the caller skips normal
    signal processing.
    Returns False for anything else (including unrecognized !words), leaving
    it to flow through classify_and_parse as a normal signal.
    """
    text = str(getattr(message, "content", "") or "").strip()
    if not text.startswith("!"):
        return False
    parts = text[1:].split()
    if not parts:
        return False
    name = parts[0].strip().lower()
    args = parts[1:]
    sender_id = str(getattr(getattr(message, "author", None), "id", "") or "")

    response = await _dispatch_whatsapp_command_text(name, args, sender_id)
    if response is None:
        return False
    await _send_review_or_reply(message, response)
    return True


def main() -> None:
    errors = production_config_errors()
    if errors:
        raise SystemExit(
            "Production configuration check failed:\n- " + "\n- ".join(errors)
        )
    locked, lock_error = acquire_runtime_lock()
    if not locked:
        raise SystemExit(lock_error)
    bot.run(config.discord_bot_token)


if __name__ == "__main__":
    main()






