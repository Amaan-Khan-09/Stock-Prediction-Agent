"""Discord runner for the stock prediction + Alpaca paper trading agent."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

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
    queue_stats,
    recover_inflight_signals,
)
from .market_context import get_market_context, refresh_market_context_async
from .options_parser import ParsedOptionLeg, ParsedOptionSignal, classify_and_parse
from .options_strategy_bridge import run_options_strategy_validation
from .options_symbol import likely_unsupported_by_alpaca, listed_expiry_fallbacks, resolve_underlying_for_prediction
from .prediction_bridge import run_project_prediction
from .runtime_lock import acquire_runtime_lock
from .signal_parser import ParsedSignal
from .symbol_directory import refresh_symbol_cache_from_alpaca
from .state_store import (
    add_decision_history,
    add_pending_buy,
    add_pending_market_buy,
    add_pending_option_order,
    add_pending_sell,
    add_conditional_equity_order,
    list_conditional_equity_orders,
    list_option_positions,
    list_pending_buys,
    list_pending_option_orders,
    list_positions,
    list_pending_sells,
    reduce_or_remove_position,
    record_order_event,
    record_option_journal_entry,
    record_option_validation_event,
    record_parser_learning,
    record_safety_block,
    record_signal_event,
    remove_pending_buy,
    remove_pending_option_order,
    remove_option_position,
    remove_position,
    remove_pending_sell,
    remove_conditional_equity_order,
    close_position_with_outcome,
    count_today_order_events,
    get_agent_mode,
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
    upsert_position,
)


alpaca = AlpacaPaperClient()
_SIGNAL_TASKS: set[asyncio.Task] = set()
_QUEUE_RECOVERED = False
_PENDING_CURSORS: dict[str, int] = {}


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
        handler = RotatingFileHandler(
            log_path,
            maxBytes=max(100_000, config.runtime_log_max_bytes),
            backupCount=max(1, min(20, config.runtime_log_backup_count)),
            encoding="utf-8",
        )
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
    if count_today_order_events() >= config.max_daily_paper_trades:
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
            "underlying_trigger_direction": option.underlying_trigger_direction,
            "underlying_trigger_price": option.underlying_trigger_price,
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
    return -price if option.price_effect == "credit" else price


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
            "price_effect": option.price_effect,
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
            "order_side": "sell",
            "position_intent": "sell_to_close",
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
    actions = {leg.order_action for leg in option.legs}
    if actions and actions <= {"open_long", "open_short"}:
        return "BUY"
    if actions and actions <= {"close_long", "close_short"}:
        return "SELL"
    if option.structure == "roll" and actions & {"close_long", "close_short"} and actions & {"open_long", "open_short"}:
        return "SELL" if option.price_effect == "credit" else "BUY"
    return "REVIEW"


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
) -> dict:
    direction = ""
    dte = "na"
    if option:
        direction = option.side or ""
        dte = _dte_bucket(option.expiry_date or "")
    return {
        "asset_type": asset_type,
        "action": str(action or "").upper(),
        "symbol": str(symbol or "").upper(),
        "direction": direction,
        "keywords": _signal_keywords(raw_text),
        "dte_bucket": dte,
    }


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

    if option.fill_price and option.stop_loss is not None:
        if option.stop_loss >= option.fill_price:
            score -= 100
            critical.append("option SL must be below entry premium for a long option")

    if option.fill_price and option.target_price is not None:
        if option.target_price <= option.fill_price:
            score -= 100
            critical.append("option target must be above entry premium for a long option")

    if option.fill_price and option.stop_loss is not None and option.target_price is not None:
        risk = option.fill_price - option.stop_loss
        reward = option.target_price - option.fill_price
        if risk > 0:
            risk_reward = reward / risk
            if risk_reward < config.min_option_risk_reward:
                score -= 25
                critical.append(
                    f"risk/reward {risk_reward:.2f} is below minimum {config.min_option_risk_reward:.2f}"
                )

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
    if not channel_id:
        return
    channel = bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(channel_id)
        except Exception:
            return
    try:
        await channel.send(content=content, embed=embed)
    except Exception as exc:
        logging.getLogger("discord_stock_prediction_agent").warning(
            "Discord send failed; queue will continue. %s: %s",
            type(exc).__name__,
            exc,
        )


async def _send_optional_channel(
    channel_id: int,
    fallback_channel: discord.abc.Messageable,
    content: str = "",
    embed: Optional[discord.Embed] = None,
) -> None:
    """Send only to a configured channel. Never fall back into the input channel."""
    if channel_id:
        await _send_channel(channel_id, content=content, embed=embed)


async def _send_review_or_reply(
    message: discord.Message,
    content: str = "",
    embed: Optional[discord.Embed] = None,
) -> None:
    """Send bot output to signal-review. Never reply in the raw input channel."""
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
    decision = decision or _evaluate_final_action(parsed.action, ai_prediction)
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


def _direct_equity_embed(parsed: ParsedSignal, decision: DecisionResult) -> discord.Embed:
    embed = discord.Embed(
        title=f"{parsed.symbol} Direct Signal Review",
        description="Agent is OFF. This valid signal is being followed without an AI decision gate.",
        color=_decision_color(decision.action),
    )
    embed.add_field(name="Signal Action", value=parsed.action, inline=True)
    embed.add_field(name="Execution Output", value=decision.action, inline=True)
    embed.add_field(name="Qty", value=str(parsed.quantity or "Default"), inline=True)
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
    quantity = parsed.quantity or 1.0
    if not await _trade_guard(message, parsed.symbol, "buy", quantity, "equity"):
        return

    market_open, market_err = await asyncio.to_thread(alpaca.is_market_open)
    if not market_open:
        await asyncio.to_thread(add_pending_market_buy, parsed.symbol, quantity, market_err or "market_closed")
        await asyncio.to_thread(_record_order, parsed.symbol, "buy", quantity, "queued", "", "equity", "market_closed")
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: paper BUY approved, but the market is closed. "
            "It has been queued and will be submitted when Alpaca reports the market is open."
        )
        return

    order, err = await asyncio.to_thread(
        alpaca.submit_market_order,
        parsed.symbol,
        "buy",
        quantity,
        _client_order_id("equitybuy", message.id),
    )
    if not order:
        if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
            await asyncio.to_thread(add_pending_market_buy, parsed.symbol, quantity, err)
            await asyncio.to_thread(_record_order, parsed.symbol, "buy", quantity, "queued", "", "equity", "broker_retry")
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: paper BUY approved and queued because Alpaca is temporarily "
                "unavailable or the market is closed. The agent will retry safely."
            )
            return
        await _block_trade(message, parsed.symbol, f"Paper BUY was not placed. {_public_error(err)}", "alpaca_reject")
        return

    order_id = str(order.get("id") or "")
    await asyncio.to_thread(_record_order, parsed.symbol, "buy", quantity, "submitted", order_id, "equity")
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
            config.stop_loss_pct,
        )
        stop_price = entry_price * (1 - config.stop_loss_pct / 100)
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: paper BUY placed for {filled_qty:g} share(s). "
            f"Entry approx ${entry_price:.2f}. Protection sell triggers near ${stop_price:.2f} "
            f"({config.stop_loss_pct:.1f}% below entry)."
        )
    else:
        if order_id:
            await asyncio.to_thread(add_pending_buy, parsed.symbol, quantity, order_id)
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
    quantity = parsed.quantity
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
        await asyncio.to_thread(add_pending_sell, parsed.symbol, quantity, "manual_sell_market_closed")
        await _send_review_or_reply(
            message,
            f"{parsed.symbol}: market is closed, so the SELL is queued. "
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

    ok, held, reason = await asyncio.to_thread(
        alpaca.has_sellable_quantity, parsed.symbol, quantity
    )
    if not ok:
        await _send_review_or_reply(message, f"{parsed.symbol}: paper SELL skipped. {reason}")
        return

    order, err = await asyncio.to_thread(
        alpaca.submit_market_order,
        parsed.symbol,
        "sell",
        quantity,
        _client_order_id("equitysell", message.id),
    )
    if not order:
        if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
            await asyncio.to_thread(
                add_pending_sell,
                parsed.symbol,
                quantity,
                err or "broker_retry",
            )
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
                f"{parsed.symbol}: paper SELL approved and queued for a safe Alpaca retry.",
            )
            return
        await _block_trade(message, parsed.symbol, f"Paper SELL was not placed. {_public_error(err)}", "alpaca_reject")
        return

    order_id = str(order.get("id") or "")
    await asyncio.to_thread(_record_order, parsed.symbol, "sell", quantity, "submitted", order_id, "equity")
    checked, _ = await asyncio.to_thread(alpaca.wait_for_order, order_id, 8)
    checked = checked or order
    exit_price = _as_float(checked.get("filled_avg_price"))
    if exit_price <= 0:
        exit_price, _ = await asyncio.to_thread(alpaca.get_latest_price, parsed.symbol)
        exit_price = _as_float(exit_price)
    outcome = {}
    if exit_price > 0:
        outcome = await asyncio.to_thread(
            close_position_with_outcome,
            parsed.symbol,
            quantity,
            exit_price,
            "manual_sell",
        )
    if not outcome:
        await asyncio.to_thread(reduce_or_remove_position, parsed.symbol, quantity)
    await _send_review_or_reply(
        message,
        f"{parsed.symbol}: paper SELL placed for {quantity:g} share(s). "
        f"Alpaca position before order: {held:g} share(s)."
        + (
            f" Learned outcome: {outcome.get('pnl_pct'):+.2f}%."
            if outcome else ""
        )
    )
    await _send_optional_channel(
        config.discord_paper_log_channel_id,
        message.channel,
        f"{parsed.symbol}: SELL order submitted. Qty {quantity:g}. Order ID `{order_id or '-'}`.",
    )


async def _process_signal(message: discord.Message, parsed: ParsedSignal) -> None:
    if parsed.action in {"BUY", "SELL"} and parsed.condition_type and parsed.condition_price:
        await asyncio.to_thread(
            add_conditional_equity_order,
            {
                "symbol": parsed.symbol,
                "action": parsed.action,
                "quantity": parsed.quantity,
                "condition_type": parsed.condition_type,
                "condition_price": parsed.condition_price,
                "order_type": parsed.order_type,
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
        decision = _direct_signal_decision(parsed.action)
        features = _learning_features(message.content, "equity", parsed.action, parsed.symbol)
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
                "agent_mode": "OFF",
            },
        )
        if parsed.action == "HOLD":
            await _send_review_or_reply(
                message,
                f"{parsed.symbol}: HOLD signal received in Agent OFF mode. No paper order is required.",
            )
            return
        if decision.action == "BUY":
            await _handle_buy(message, parsed, {})
        elif decision.action == "SELL":
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
    features = _learning_features(message.content, "equity", parsed.action, parsed.symbol)
    decision = _apply_learning_overlay(_evaluate_final_action(parsed.action, ai_prediction), features)
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

    if final_action == "BUY":
        await _handle_buy(message, parsed, prediction)
    elif final_action == "SELL":
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
                + f"Net {str(option.price_effect or 'price').title()}: "
                + (f"${option.fill_price:.2f}" if option.fill_price else "market")
                + f"\nWindow: {si.get('options_backtest_start_date', '-')} to {si.get('options_backtest_end_date', '-')}"
            )
        else:
            trade_setup = (
                f"{si.get('symbol', option.root)} | {si.get('direction_label', si.get('direction', '-'))} "
                f"{si.get('opt_type', si.get('side', '-'))} | Qty {si.get('quantity', '-')}\n"
                f"Requested Strike: {requested_strike if requested_strike is not None else '-'} | "
                f"Validation Strike: {validation_strike} | "
                f"Delta {si.get('delta', '-')} | DTE {si.get('dte', '-')}\n"
                f"Entry: {si.get('entry_frequency', '-')} | Exit: {si.get('exit_rule', '-')}\n"
                f"Window: {si.get('options_backtest_start_date', si.get('prediction_origin_date', '-'))} "
                f"to {si.get('options_backtest_end_date', si.get('target_date', '-'))}"
            )
        embed.add_field(name="Trade Setup", value=trade_setup[:1000], inline=not option.is_multi_leg)
    if option.fill_price:
        embed.add_field(name="Signal Fill Price", value=f"${option.fill_price:.2f}", inline=True)
    if option.stop_loss is not None or option.target_price is not None or quality:
        q = quality or {}
        rr = q.get("risk_reward")
        rr_text = f"{rr:.2f}" if isinstance(rr, (int, float)) else "-"
        issues = list(q.get("critical") or []) + list(q.get("issues") or [])
        embed.add_field(
            name="Option Signal Quality",
            value=(
                f"Score: {float(q.get('score', 0)):.0f}/100\n"
                f"SL: {option.stop_loss if option.stop_loss is not None else '-'} | "
                f"TP: {option.target_price if option.target_price is not None else '-'} | "
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
            percentage_qty = max(1, int(held_qty * min(100.0, max(0.0, option.close_percent)) / 100.0))
            qty = min(float(percentage_qty), held_qty)
        else:
            qty = min(qty, held_qty)

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
        await asyncio.to_thread(
            upsert_option_position,
            occ_symbol, option.root, option.side, option.strike,
            contract_expiration,
            qty, entry_price, str(order.get("id") or ""),
            option.stop_loss, option.target_price, float(quality.get("score") or 0), position_intent,
            option.target_prices, option.trailing_stop_pct, option.exit_before_market_close,
        )
    elif position_intent in {"sell_to_close", "buy_to_close"}:
        remaining_qty = max(0.0, held_qty - qty)
        if remaining_qty > 0:
            updates = {"qty": round(remaining_qty, 6)}
            if "BREAKEVEN" in str(option.raw_text or "").upper():
                tracked = next(
                    (item for item in list_option_positions() if str(item.get("occ_symbol") or "").upper() == occ_symbol.upper()),
                    {},
                )
                entry = _as_float(tracked.get("entry_price"))
                if entry > 0:
                    updates["stop_loss"] = entry
            await asyncio.to_thread(update_option_position, occ_symbol, **updates)
        else:
            await asyncio.to_thread(remove_option_position, occ_symbol)
    await asyncio.to_thread(
        record_option_journal_entry,
        {
            "occ_symbol": occ_symbol, "root": option.root, "side": option.side,
            "strike": option.strike, "quantity": qty, "order_type": order_type,
            "stop_loss": option.stop_loss, "target_price": option.target_price,
            "signal_quality": quality.get("score"), "risk_reward": quality.get("risk_reward"),
            "raw_input": option.raw_text, "status": "order_placed", "position_intent": position_intent,
        },
    )
    fallback_note = " (nearest available expiry used — 0DTE not listed)" if used_fallback_expiry else ""
    await _send_review_or_reply(
        message,
        f"{occ_symbol}: paper option {order_side.upper()} order placed successfully. "
        f"Qty {qty:g} contract(s), type={order_type}"
        f"{' @ $' + f'{option.fill_price:.2f}' if option.fill_price else ''}. "
        f"Order ID `{order.get('id') or '-'}`.{fallback_note}",
    )
    await _send_optional_channel(
        config.discord_paper_log_channel_id, message.channel,
        f"{occ_symbol}: paper option {order_side.upper()} placed successfully. Qty {qty:g}. Intent {position_intent}. Order ID `{order.get('id') or '-'}`.",
    )

    if option.add_quantity and option.add_trigger_premium and position_intent == "buy_to_open":
        await asyncio.to_thread(
            add_pending_option_order,
            {
                "pending_key": f"option:{uuid.uuid4().hex}",
                "occ_symbol": occ_symbol,
                "root": option.root,
                "side": option.side,
                "strike": option.strike,
                "expiry_date": contract_expiration,
                "qty": float(option.add_quantity),
                "order_type": "limit",
                "limit_price": float(option.add_trigger_premium),
                "stop_loss": option.stop_loss,
                "target_price": option.target_price,
                "target_prices": list(option.target_prices),
                "trailing_stop_pct": option.trailing_stop_pct,
                "exit_before_market_close": option.exit_before_market_close,
                "signal_quality": quality.get("score"),
                "raw_input": option.raw_text,
                "reason": "conditional_scale_in",
                "order_side": "buy",
                "position_intent": "buy_to_open",
                "requires_position": False,
                "contract_pending": False,
            },
        )
        await _send_review_or_reply(
            message,
            f"{occ_symbol}: scale-in order queued for {option.add_quantity:g} additional contract(s) "
            f"with a ${option.add_trigger_premium:.2f} buy limit.",
        )


async def _fetch_queued_message(item: dict) -> Optional[discord.Message]:
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
    condition = str(order.get("condition_type") or "").lower()
    trigger_price = _as_float(order.get("condition_price"))
    action = str(order.get("action") or "").upper()
    if current_price <= 0 or trigger_price <= 0:
        return False

    lower_band_pct = max(0.0, _as_float(config.conditional_trigger_band_pct, 5.0))
    upper_band_pct = max(0.0, _as_float(config.conditional_trigger_upper_band_pct, 10.0))
    upper_band = trigger_price * (1 + upper_band_pct / 100)
    lower_band = trigger_price * (1 - lower_band_pct / 100)

    if condition == "limit_price":
        if action == "BUY":
            return lower_band <= current_price <= trigger_price
        return trigger_price <= current_price <= upper_band
    if condition in {"above", "close_above"}:
        return trigger_price <= current_price <= upper_band
    if condition in {"below", "close_below"}:
        return lower_band <= current_price <= trigger_price
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
        await asyncio.to_thread(remove_conditional_equity_order, str(order.get("id") or ""))
        message = await _fetch_queued_message(order)
        if not message:
            await _send_channel(
                config.discord_review_channel_id,
                f"{symbol}: watched price condition triggered near ${current_price:.2f}, but the original Discord message could not be loaded. No paper trade placed.",
            )
            continue
        parsed = ParsedSignal(
            valid=True,
            action=str(order.get("action") or "").upper(),
            symbol=symbol,
            quantity=order.get("quantity"),
            raw_text=str(order.get("raw_input") or ""),
            reason=f"Watched price condition triggered at ${current_price:.2f}.",
        )
        await _send_review_or_reply(
            message,
            f"{symbol}: watched condition triggered near ${current_price:.2f}. "
            f"Processing with Agent {get_agent_mode()} mode.",
        )
        await _process_signal(message, parsed)


async def _process_claimed_signal(item: dict) -> None:
    try:
        message = await _fetch_queued_message(item)
        if message:
            await _process_queued_signal_message(message)
        else:
            await _send_channel(
                config.discord_review_channel_id,
                f"Queued signal could not be loaded, so it was skipped safely: {str(item.get('raw_text') or '')[:180]}",
            )
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
    global _QUEUE_RECOVERED
    if not _QUEUE_RECOVERED:
        recovered = await asyncio.to_thread(recover_inflight_signals)
        _QUEUE_RECOVERED = True
        if recovered:
            logging.getLogger("discord_stock_prediction_agent").warning(
                "Recovered %s interrupted signal(s) after restart.", recovered
            )
    if not signal_queue_worker.is_running():
        signal_queue_worker.start()
    if not stop_loss_monitor.is_running():
        stop_loss_monitor.start()
    refresh_market_context_async()
    symbol_cache_status = await asyncio.to_thread(refresh_symbol_cache_from_alpaca)
    logging.getLogger("discord_stock_prediction_agent").info(
        "Symbol cache refresh: %s",
        {k: v for k, v in symbol_cache_status.items() if k not in {"error", "message"}},
    )
    print(f"{config.agent_name} ready as {bot.user}. Stop monitor active.")


async def _process_queued_signal_message(message: discord.Message) -> None:
    routed = classify_and_parse(message.content)
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
        await _send_review_or_reply(message, "Invalid input")
        return

    if routed.kind == "NO_TRADE":
        try:
            await message.add_reaction("💬")
        except Exception:
            pass
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
@tasks.loop(seconds=max(15, config.stop_monitor_seconds))
async def stop_loss_monitor() -> None:
    if not alpaca.ready():
        return
    await _activate_filled_pending_buys()
    market_open, _ = await asyncio.to_thread(alpaca.is_market_open)
    await _process_conditional_equity_orders()
    if market_open:
        await _process_pending_market_buys()
        await _process_pending_sells()
        await _process_pending_option_orders()
    for position in list_positions():
        symbol = str(position.get("symbol") or "").upper()
        qty = _as_float(position.get("qty"))
        entry = _as_float(position.get("entry_price"))
        stop_price = _as_float(position.get("stop_price")) or entry * (1 - config.stop_loss_pct / 100)
        if not symbol or qty <= 0 or entry <= 0:
            continue

        alpaca_position, _ = await asyncio.to_thread(alpaca.get_position, symbol)
        if not alpaca_position:
            await asyncio.to_thread(remove_position, symbol)
            continue

        held_qty = _as_float(alpaca_position.get("qty"))
        current_price = _as_float(alpaca_position.get("current_price"))
        if held_qty <= 0:
            await asyncio.to_thread(remove_position, symbol)
            continue
        if current_price <= 0 or current_price > stop_price:
            continue

        if not market_open:
            await asyncio.to_thread(add_pending_sell, symbol, min(qty, held_qty), "protection_market_closed")
            continue

        open_order, _ = await asyncio.to_thread(alpaca.has_open_order, symbol)
        if open_order:
            continue

        sell_qty = min(qty, held_qty)
        order, err = await asyncio.to_thread(
            alpaca.submit_market_order,
            symbol,
            "sell",
            sell_qty,
            _client_order_id(
                "equitystop",
                f"{symbol}:{position.get('entry_price')}:{position.get('stop_price')}",
            ),
        )
        if order:
            await asyncio.to_thread(_record_order, symbol, "sell", sell_qty, "submitted", str(order.get("id") or ""), "equity", "protection_stop_loss")
            outcome = await asyncio.to_thread(
                close_position_with_outcome,
                symbol,
                sell_qty,
                current_price,
                "protection_stop_loss",
            )
            if not outcome:
                await asyncio.to_thread(remove_position, symbol)
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{symbol}: protection sell triggered. Current ${current_price:.2f} <= "
                f"stop ${stop_price:.2f}. Sold {sell_qty:g} share(s)."
                + (
                    f" Learned outcome: {outcome.get('pnl_pct'):+.2f}%."
                    if outcome else ""
                ),
            )
        else:
            await asyncio.to_thread(
                record_safety_block,
                {"symbol": symbol, "category": "protection_sell_failed", "reason": err},
            )
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{symbol}: protection sell attempted but was not placed. {_public_error(err)}",
            )

    await _process_option_exit_monitor(market_open)


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
        if str(pending.get("order_side") or "buy").lower() == "sell":
            return True
    return False


async def _process_option_exit_monitor(market_open: bool) -> None:
    clock = None
    for position in list_option_positions():
        occ_symbol = str(position.get("occ_symbol") or "").upper()
        tracked_qty = _as_float(position.get("qty"))
        stop_loss = _as_float(position.get("stop_loss"))
        target_price = _as_float(position.get("target_price"))
        target_prices = [
            _as_float(value) for value in (position.get("target_prices") or []) if _as_float(value) > 0
        ]
        target_index = max(0, int(_as_float(position.get("target_index"))))
        trailing_stop_pct = _as_float(position.get("trailing_stop_pct"))
        timed_exit = bool(position.get("exit_before_market_close"))
        if not occ_symbol or tracked_qty <= 0:
            continue
        if stop_loss <= 0 and target_price <= 0 and not target_prices and trailing_stop_pct <= 0 and not timed_exit:
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

        held_qty = _as_float(alpaca_position.get("qty"))
        if held_qty <= 0:
            await asyncio.to_thread(remove_option_position, occ_symbol)
            continue

        current_price = _as_float(alpaca_position.get("current_price"))
        if current_price <= 0:
            latest_price, _ = await asyncio.to_thread(alpaca.get_latest_option_price, occ_symbol)
            current_price = _as_float(latest_price)
        if current_price <= 0:
            continue

        position_intent = str(position.get("position_intent") or "buy_to_open")
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
                exit_before_close_now = (next_close - datetime.now(next_close.tzinfo)).total_seconds() <= 15 * 60
            except (TypeError, ValueError):
                exit_before_close_now = False

        if position_intent == "sell_to_open":
            hit_stop = stop_loss > 0 and current_price >= stop_loss
            hit_target = active_target > 0 and current_price <= active_target
        else:
            hit_stop = stop_loss > 0 and current_price <= stop_loss
            hit_target = active_target > 0 and current_price >= active_target
        if not hit_stop and not hit_target and not exit_before_close_now:
            continue

        exit_reason = (
            "option_stop_loss" if hit_stop
            else "option_target_price" if hit_target
            else "option_time_exit"
        )
        trigger_price = stop_loss if hit_stop else active_target if hit_target else current_price
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

        order, err = await asyncio.to_thread(
            alpaca.submit_option_order,
            occ_symbol,
            "sell",
            sell_qty,
            "market",
            None,
            "sell_to_close",
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

        await asyncio.to_thread(_record_order, occ_symbol, "sell", sell_qty, "submitted", str(order.get("id") or ""), "option", exit_reason)
        remaining_qty = max(0.0, held_qty - sell_qty)
        if partial_target and remaining_qty > 0:
            await asyncio.to_thread(
                update_option_position,
                occ_symbol,
                qty=round(remaining_qty, 6),
                target_index=target_index + 1,
            )
        else:
            await asyncio.to_thread(remove_option_position, occ_symbol)
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
            f"{occ_symbol}: option exit triggered at ${current_price:.2f}. Sold {sell_qty:g} contract(s).",
        )


async def _process_pending_sells() -> None:
    for pending in _rotating_batch(list_pending_sells(), "pending_sells"):
        pending_key = str(pending.get("pending_key") or pending.get("symbol") or "")
        symbol = str(pending.get("symbol") or "").upper()
        qty = _as_float(pending.get("qty"))
        if not symbol or qty <= 0:
            await asyncio.to_thread(remove_pending_sell, pending_key)
            continue
        open_order, _ = await asyncio.to_thread(alpaca.has_open_order, symbol)
        if open_order:
            continue
        ok, held, reason = await asyncio.to_thread(alpaca.has_sellable_quantity, symbol, qty)
        if not ok:
            await asyncio.to_thread(remove_pending_sell, pending_key)
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{symbol}: queued SELL removed. {reason}",
            )
            continue
        sell_qty = min(qty, held)
        order, err = await asyncio.to_thread(
            alpaca.submit_market_order,
            symbol,
            "sell",
            sell_qty,
            _client_order_id("queuedsell", pending_key),
        )
        if not order:
            if _is_transient_broker_error(err):
                logging.getLogger("discord_stock_prediction_agent").warning(
                    "Retaining queued SELL %s after transient Alpaca failure: %s",
                    pending_key,
                    _public_error(err),
                )
                continue
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
        await asyncio.to_thread(remove_pending_sell, pending_key)
        exit_price, _ = await asyncio.to_thread(alpaca.get_latest_price, symbol)
        outcome = {}
        if _as_float(exit_price) > 0:
            outcome = await asyncio.to_thread(
                close_position_with_outcome,
                symbol,
                sell_qty,
                _as_float(exit_price),
                str(pending.get("reason") or "queued_sell"),
            )
        if not outcome:
            await asyncio.to_thread(reduce_or_remove_position, symbol, sell_qty)
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{symbol}: queued SELL placed after market opened. Qty {sell_qty:g}."
            + (
                f" Learned outcome: {outcome.get('pnl_pct'):+.2f}%."
                if outcome else ""
            ),
        )


async def _process_pending_market_buys() -> None:
    for pending in _rotating_batch(list_pending_buys(), "pending_buys"):
        if not pending.get("queued"):
            continue
        pending_key = str(
            pending.get("pending_key")
            or pending.get("order_id")
            or f"queued:{pending.get('symbol') or ''}"
        )
        symbol = str(pending.get("symbol") or "").upper()
        qty = _as_float(pending.get("qty"))
        if not symbol or qty <= 0:
            await asyncio.to_thread(remove_pending_buy, pending_key)
            continue
        order, err = await asyncio.to_thread(
            alpaca.submit_market_order,
            symbol,
            "buy",
            qty,
            _client_order_id("queuedbuy", pending_key),
        )
        if not order:
            if _is_market_closed_order_error(err) or _is_transient_broker_error(err):
                continue
            await asyncio.to_thread(
                record_safety_block,
                {"symbol": symbol, "category": "queued_buy_failed", "reason": err},
            )
            await asyncio.to_thread(remove_pending_buy, pending_key)
            await _send_channel(
                config.discord_paper_log_channel_id or config.discord_review_channel_id,
                f"{symbol}: queued BUY attempted but was not placed. {_public_error(err)}",
            )
            continue
        order_id = str(order.get("id") or "")
        await asyncio.to_thread(remove_pending_buy, pending_key)
        await asyncio.to_thread(_record_order, symbol, "buy", qty, "submitted", order_id, "equity", "queued_buy")
        if order_id:
            await asyncio.to_thread(add_pending_buy, symbol, qty, order_id)
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{symbol}: queued BUY submitted after market opened. Qty {qty:g}. Order ID `{order_id or '-'}`.",
        )


async def _process_pending_multi_leg_order(pending: dict, pending_key: str) -> None:
    root = str(pending.get("root") or "").upper()
    qty = _as_float(pending.get("qty"))
    leg_specs = list(pending.get("legs") or [])
    if not root or qty <= 0 or not 2 <= len(leg_specs) <= 4:
        await asyncio.to_thread(remove_pending_option_order, pending_key)
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
            alpaca_position, _ = await asyncio.to_thread(alpaca.get_position, occ_symbol)
            held_qty = abs(_as_float((alpaca_position or {}).get("qty")))
            if held_qty <= 0:
                continue
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

        await asyncio.to_thread(remove_pending_option_order, pending_key)
        await asyncio.to_thread(_record_order, occ_symbol, order_side, qty, "submitted", str(order.get("id") or ""), "option", reason)
        if position_intent in {"sell_to_close", "buy_to_close"}:
            remaining_qty = _as_float(pending.get("remaining_qty"))
            if remaining_qty > 0:
                updates = {
                    "qty": round(remaining_qty, 6),
                    "target_index": int(_as_float(pending.get("next_target_index"))),
                }
                if pending.get("move_stop_to_breakeven"):
                    tracked = next(
                        (item for item in list_option_positions() if str(item.get("occ_symbol") or "").upper() == occ_symbol.upper()),
                        {},
                    )
                    entry = _as_float(tracked.get("entry_price"))
                    if entry > 0:
                        updates["stop_loss"] = entry
                await asyncio.to_thread(
                    update_option_position,
                    occ_symbol,
                    **updates,
                )
            else:
                await asyncio.to_thread(remove_option_position, occ_symbol)
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

        await asyncio.to_thread(
            upsert_option_position,
            occ_symbol,
            root,
            side,
            _as_float(pending.get("strike")),
            str(pending.get("expiry_date") or ""),
            qty,
            _as_float(limit_price),
            str(order.get("id") or ""),
            pending.get("stop_loss"),
            pending.get("target_price"),
            pending.get("signal_quality"),
            position_intent,
            pending.get("target_prices"),
            pending.get("trailing_stop_pct"),
            bool(pending.get("exit_before_market_close")),
        )
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
                "status": "queued_order_placed",
            },
        )
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{occ_symbol}: queued option {order_side.upper()} placed successfully after monitor check. "
            f"Qty {qty:g}, type={order_type}"
            f"{' @ $' + f'{_as_float(limit_price):.2f}' if _as_float(limit_price) > 0 else ''}. "
            f"Order ID `{order.get('id') or '-'}`.",
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
        await asyncio.to_thread(
            upsert_position,
            symbol,
            filled_qty,
            entry_price,
            order_id,
            config.stop_loss_pct,
        )
        await asyncio.to_thread(remove_pending_buy, order_id)
        stop_price = entry_price * (1 - config.stop_loss_pct / 100)
        await _send_channel(
            config.discord_paper_log_channel_id or config.discord_review_channel_id,
            f"{symbol}: buy order filled. Protection active near ${stop_price:.2f} "
            f"({config.stop_loss_pct:.1f}% below ${entry_price:.2f}).",
        )


@bot.command(name="agent_status")
async def agent_status(ctx: commands.Context) -> None:
    alpaca_status = "configured" if alpaca.ready() else "not configured/disabled"
    tracked = list_positions()
    tracked_options = list_option_positions()
    summary = get_daily_summary()
    queue = await asyncio.to_thread(queue_stats)
    await _send_context_output(
        ctx,
        f"{config.agent_name} is online. Agent mode: {get_agent_mode()}. "
        f"Alpaca paper trading: {alpaca_status}. "
        f"Tracked equity positions: {len(tracked)}. Tracked option positions: {len(tracked_options)}. "
        f"Signal queue: {queue['queued']} waiting, {queue['processing']} processing, "
        f"{queue['dead']} dead-letter. Workers: {max(1, min(16, config.signal_worker_concurrency))}. "
        f"Today: {summary['signals']} signal(s), {summary['orders']} order event(s), "
        f"{summary['blocks']} safety block(s)."
    )


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


async def _set_agent_mode_from_command(ctx: commands.Context, mode: str) -> None:
    if not await _can_manage_agent_mode(ctx):
        await _send_context_output(
            ctx,
            "You need Administrator or Manage Server permission to change Agent mode.",
        )
        return
    control = await asyncio.to_thread(set_agent_mode, mode, str(ctx.author.id))
    if control["mode"] == "ON":
        detail = "Prediction and options-strategy decision checks are active."
    else:
        detail = (
            "Every valid BUY/SELL signal will proceed directly to paper-order handling. "
            "Market, contract, position, price-condition, and broker safeguards remain active."
        )
    await _send_context_output(ctx, f"Agent mode is now {control['mode']}. {detail}")


@bot.command(name="agent_on")
async def agent_on(ctx: commands.Context) -> None:
    await _set_agent_mode_from_command(ctx, "ON")


@bot.command(name="agent_off")
async def agent_off(ctx: commands.Context) -> None:
    await _set_agent_mode_from_command(ctx, "OFF")


@bot.command(name="agent_mode")
async def agent_mode(ctx: commands.Context) -> None:
    mode = get_agent_mode()
    detail = (
        "normal prediction and strategy decisions are active"
        if mode == "ON"
        else "valid incoming BUY/SELL signals go directly to paper-order handling"
    )
    await _send_context_output(ctx, f"Agent mode: {mode}. {detail}.")


@bot.command(name="agent_positions")
async def agent_positions(ctx: commands.Context) -> None:
    tracked = list_positions()
    if not tracked:
        await _send_context_output(ctx, "No positions are currently tracked by this agent.")
        return
    lines = [
        f"{p['symbol']}: qty {p['qty']}, entry ${float(p['entry_price']):.2f}, "
        f"stop ${float(p['stop_price']):.2f}"
        for p in tracked
    ]
    await _send_context_output(ctx, "\n".join(lines))


@bot.command(name="agent_option_positions")
async def agent_option_positions(ctx: commands.Context) -> None:
    tracked = list_option_positions()
    if not tracked:
        await _send_context_output(ctx, "No option positions are currently tracked by this agent.")
        return
    lines = [
        f"{p['occ_symbol']}: qty {p['qty']}, entry ${float(p['entry_price']):.2f}"
        for p in tracked
    ]
    await _send_context_output(ctx, "\n".join(lines))


@bot.command(name="agent_summary")
async def agent_summary(ctx: commands.Context) -> None:
    summary = get_daily_summary()
    queue = await asyncio.to_thread(queue_stats)
    await _send_context_output(
        ctx,
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


@bot.command(name="agent_learning")
async def agent_learning(ctx: commands.Context) -> None:
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
    await _send_context_output(ctx, "\n".join(lines))


@bot.command(name="agent_option_validation")
async def agent_option_validation(ctx: commands.Context) -> None:
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
    await _send_context_output(ctx, "\n".join(lines))


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






