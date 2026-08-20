"""Same-day group context memory for Discord/WhatsApp trading signals.

Stores an ordered list of trades for the current trading day.
Handles reply chains: "trimmed half at 1.90" linked to the entry message
collapses into a single ContextTrade record.

Storage: data/context_memory_YYYY-MM-DD.json (one file per day).
Scope: group-level (all users in the channel), not per-user.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import (
    AssetType, ContextTrade, ParsedSignal, SignalAction, SignalStatus,
)

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────────

def _day_file(day: Optional[str] = None) -> Path:
    if day is None:
        day = date.today().isoformat()
    return DATA_DIR / f"context_memory_{day}.json"


def _load_day(day: Optional[str] = None) -> List[dict]:
    f = _day_file(day)
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_day(records: List[dict], day: Optional[str] = None) -> None:
    f = _day_file(day)
    f.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _trade_to_dict(t: ContextTrade) -> dict:
    return {
        "trade_id":           t.trade_id,
        "date":               t.date,
        "message_id":         t.message_id,
        "reply_to":           t.reply_to,
        "user_label":         t.user_label,
        "underlying":         t.underlying,
        "asset_type":         t.asset_type,
        "option_type":        t.option_type,
        "strike":             t.strike,
        "expiry":             t.expiry,
        "entry_price":        t.entry_price,
        "contracts_opened":   t.contracts_opened,
        "contracts_closed":   t.contracts_closed,
        "remaining_contracts":t.remaining_contracts,
        "exit_prices":        t.exit_prices,
        "exit_contracts":     t.exit_contracts,
        "realized_pnl":       t.realized_pnl,
        "status":             t.status,
        "raw_messages":       t.raw_messages,
    }


def _dict_to_trade(d: dict) -> ContextTrade:
    t = ContextTrade(
        trade_id            = d.get("trade_id", str(uuid.uuid4())[:8]),
        date                = d.get("date", date.today().isoformat()),
        message_id          = d.get("message_id"),
        reply_to            = d.get("reply_to"),
        user_label          = d.get("user_label"),
        underlying          = d.get("underlying"),
        asset_type          = d.get("asset_type", AssetType.UNKNOWN),
        option_type         = d.get("option_type"),
        strike              = d.get("strike"),
        expiry              = d.get("expiry"),
        entry_price         = d.get("entry_price"),
        contracts_opened    = int(d.get("contracts_opened", 0)),
        contracts_closed    = int(d.get("contracts_closed", 0)),
        remaining_contracts = int(d.get("remaining_contracts", 0)),
        exit_prices         = list(d.get("exit_prices", [])),
        exit_contracts      = list(d.get("exit_contracts", [])),
        realized_pnl        = d.get("realized_pnl"),
        status              = d.get("status", SignalStatus.OPEN),
        raw_messages        = list(d.get("raw_messages", [])),
    )
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def record_signal(
    signal: ParsedSignal,
    message_id: Optional[str] = None,
    user_label: Optional[str] = None,
    day: Optional[str] = None,
) -> ContextTrade:
    """Record a new parsed signal. Returns the resulting ContextTrade.

    If the signal is a reply to an existing open trade (via reply_to_message_id
    or by matching underlying/option_type/strike), the existing trade is updated.
    Otherwise a new trade is created.
    """
    today = day or date.today().isoformat()
    records = _load_day(today)
    trades  = [_dict_to_trade(r) for r in records]

    # Try to link to an existing open trade
    linked = _find_linked_trade(signal, trades)

    if linked is not None:
        _apply_update(linked, signal, message_id, user_label)
        updated_records = [_trade_to_dict(t) for t in trades]
        _save_day(updated_records, today)
        return linked

    # Create new trade
    trade = _create_trade(signal, message_id, user_label, today)
    trades.append(trade)
    _save_day([_trade_to_dict(t) for t in trades], today)
    return trade


def get_open_trades(day: Optional[str] = None) -> List[ContextTrade]:
    """Return all OPEN or PARTIAL trades for the day."""
    records = _load_day(day)
    trades  = [_dict_to_trade(r) for r in records]
    return [t for t in trades if t.status in (SignalStatus.OPEN, SignalStatus.PARTIAL)]


def get_all_trades(day: Optional[str] = None) -> List[ContextTrade]:
    """Return all trades for the day."""
    records = _load_day(day)
    return [_dict_to_trade(r) for r in records]


def get_trade_by_message_id(message_id: str, day: Optional[str] = None) -> Optional[ContextTrade]:
    for t in get_all_trades(day):
        if t.message_id == message_id:
            return t
    return None


def summarize_day(day: Optional[str] = None) -> dict:
    """Return summary stats for the trading day."""
    trades = get_all_trades(day)
    closed = [t for t in trades if t.status == SignalStatus.CLOSED]
    open_  = [t for t in trades if t.status == SignalStatus.OPEN]
    partial = [t for t in trades if t.status == SignalStatus.PARTIAL]
    total_pnl = sum(
        t.realized_pnl for t in closed if t.realized_pnl is not None
    )
    return {
        "date":          day or date.today().isoformat(),
        "total_trades":  len(trades),
        "open":          len(open_),
        "partial":       len(partial),
        "closed":        len(closed),
        "total_realized_pnl": round(total_pnl, 2),
        "trades":        [_trade_to_dict(t) for t in trades],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_linked_trade(
    signal: ParsedSignal,
    trades: List[ContextTrade],
) -> Optional[ContextTrade]:
    """Find an open trade that this signal likely updates."""
    # Priority 1: explicit reply_to_message_id
    if signal.reply_to_message_id:
        for t in trades:
            if t.message_id == signal.reply_to_message_id:
                return t

    # Priority 2: signal is a partial exit (trim/out/sell with no new entry)
    is_exit = signal.action in (SignalAction.TRIM, SignalAction.SELL, SignalAction.EXIT)
    is_hold = signal.action == SignalAction.HOLD
    has_underlying = signal.underlying is not None

    if not has_underlying and is_exit:
        # "out rest at 2.20" with no underlying — link to most recent open
        open_trades = [t for t in trades if t.status in (SignalStatus.OPEN, SignalStatus.PARTIAL)]
        if open_trades:
            return open_trades[-1]

    # Priority 3: same underlying + option_type + strike with open status
    if has_underlying:
        for t in reversed(trades):
            if t.status not in (SignalStatus.OPEN, SignalStatus.PARTIAL):
                continue
            if t.underlying != signal.underlying:
                continue
            if signal.asset_type == AssetType.OPTION:
                if t.option_type != signal.option_type:
                    continue
                if signal.strike is not None and t.strike is not None:
                    if abs(t.strike - signal.strike) > 0.01:
                        continue
            return t

    return None


def _create_trade(
    signal: ParsedSignal,
    message_id: Optional[str],
    user_label: Optional[str],
    today: str,
) -> ContextTrade:
    trade = ContextTrade(
        trade_id  = str(uuid.uuid4())[:8].upper(),
        date      = today,
        message_id = message_id,
        reply_to  = signal.reply_to_message_id,
        user_label = user_label,
        underlying = signal.underlying,
        asset_type = signal.asset_type,
        option_type = signal.option_type,
        strike      = signal.strike,
        expiry      = signal.expiry_date,
        entry_price = signal.premium,
        contracts_opened = signal.contracts or 1,
        remaining_contracts = signal.contracts or 1,
        status      = SignalStatus.OPEN,
        raw_messages = [signal.raw_message],
    )
    return trade


def _apply_update(
    trade: ContextTrade,
    signal: ParsedSignal,
    message_id: Optional[str],
    user_label: Optional[str],
) -> None:
    """Apply a partial exit or full exit update to an existing trade."""
    trade.raw_messages.append(signal.raw_message)

    exit_price = signal.premium
    if exit_price is None:
        return

    # Determine how many contracts this exit covers
    exit_qty = signal.contracts or signal.quantity

    is_trim = signal.action == SignalAction.TRIM
    is_half_word = "half" in (signal.raw_message or "").lower()

    if is_trim or is_half_word:
        # Exit half remaining contracts
        exit_qty = exit_qty or max(1, trade.remaining_contracts // 2)
    elif signal.action in (SignalAction.SELL, SignalAction.EXIT):
        # Exit all remaining
        exit_qty = exit_qty or trade.remaining_contracts

    exit_qty = min(exit_qty or 1, trade.remaining_contracts)

    trade.exit_prices.append(exit_price)
    trade.exit_contracts.append(exit_qty)
    trade.contracts_closed += exit_qty
    trade.remaining_contracts = max(0, trade.remaining_contracts - exit_qty)

    trade.compute_pnl()

    if trade.remaining_contracts == 0:
        trade.status = SignalStatus.CLOSED
    elif trade.contracts_closed > 0:
        trade.status = SignalStatus.PARTIAL
