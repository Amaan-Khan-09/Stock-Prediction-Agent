"""Small JSON state store for paper positions opened by this agent."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

from .config import AGENT_DIR
from .signal_normalizer import signal_template_signature


STATE_PATH = AGENT_DIR / "agent_state.json"
_STATE_LOCK = threading.RLock()
_T = TypeVar("_T")


def _state_backup_path() -> Path:
    return STATE_PATH.with_name(f"{STATE_PATH.stem}.backup{STATE_PATH.suffix}")


def _state_mutation(function: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(function)
    def locked(*args: Any, **kwargs: Any) -> _T:
        with _STATE_LOCK:
            return function(*args, **kwargs)

    return locked


def _empty_state() -> Dict[str, Any]:
    return {
        "agent_positions": {},
        "pending_buy_orders": {},
        "pending_sell_orders": {},
        "pending_option_orders": {},
        "pending_option_entry_orders": {},
        "pending_multi_leg_entry_orders": {},
        "pending_exit_orders": {},
        "decision_history": [],
        "learning_profile": {
            "buy_score_adjustment": 0.0,
            "buy_min_return_adjustment": 0.0,
            "sell_score_adjustment": 0.0,
            "sell_min_return_adjustment": 0.0,
            "closed_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        },
        "trade_outcomes": [],
        "option_positions": {},
        "multi_leg_positions": {},
        "option_journal": [],
        "signal_events": [],
        "order_events": [],
        "safety_blocks": [],
        "signal_learning": {},
        "parser_learning": {},
        "option_validation_events": [],
        "conditional_equity_orders": {},
        "agent_control": {
            "mode": "ON",
            "updated_at": "",
            "updated_by": "",
        },
    }


def load_state() -> Dict[str, Any]:
    with _STATE_LOCK:
        if not STATE_PATH.exists() and not _state_backup_path().exists():
            return _empty_state()
        errors: list[str] = []
        for candidate in (STATE_PATH, _state_backup_path()):
            if not candidate.exists():
                continue
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("state root must be a JSON object")
                state = _empty_state()
                state.update(loaded)
                if candidate == _state_backup_path():
                    logging.getLogger(__name__).error(
                        "Recovered agent state from backup after the primary state file failed: %s",
                        "; ".join(errors) or "primary file unavailable",
                    )
                return state
            except Exception as exc:
                errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Agent state is unreadable; refusing to continue with empty protection state. "
            + "; ".join(errors)
        )


def save_state(state: Dict[str, Any]) -> None:
    payload = json.dumps(state, indent=2, sort_keys=True)
    with _STATE_LOCK:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        for destination in (STATE_PATH, _state_backup_path()):
            temporary = destination.with_name(
                f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Never promote a temp file that isn't actually intact. Whatever
                # transient interference (AV scan, disk hiccup) occasionally
                # corrupts a write, this read-back check keeps it from ever
                # reaching the real destination -- the recurring
                # "Recovered agent state from backup" errors in production
                # logs show the destination file itself sometimes ends up
                # truncated, even though this write path was already atomic.
                written = temporary.read_text(encoding="utf-8")
                if written != payload:
                    raise RuntimeError(
                        f"State write verification failed for {destination.name}: "
                        f"wrote {len(payload)} chars, read back {len(written)} chars."
                    )
                for attempt in range(5):
                    try:
                        os.replace(temporary, destination)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        # Windows scanners/indexers can briefly hold a JSON
                        # destination after it is replaced. Keep the atomic
                        # write and retry the replace instead of losing state.
                        time.sleep(0.02 * (2 ** attempt))
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)


def _learning_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    profile = state.setdefault("learning_profile", {})
    defaults = _empty_state()["learning_profile"]
    for key, value in defaults.items():
        profile.setdefault(key, value)
    return profile


def get_learning_profile() -> Dict[str, Any]:
    state = load_state()
    return dict(_learning_profile(state))


def get_agent_mode() -> str:
    control = load_state().get("agent_control") or {}
    mode = str(control.get("mode") or "ON").upper()
    return mode if mode in {"ON", "OFF"} else "ON"


@_state_mutation
def set_agent_mode(mode: str, updated_by: str = "") -> Dict[str, Any]:
    normalized = str(mode or "").upper()
    if normalized not in {"ON", "OFF"}:
        raise ValueError("Agent mode must be ON or OFF.")
    state = load_state()
    control = state.setdefault("agent_control", {})
    control.update(
        {
            "mode": normalized,
            "updated_at": _utcstamp(),
            "updated_by": str(updated_by or ""),
        }
    )
    save_state(state)
    return dict(control)


def get_automate_agent_mode() -> str:
    """automate_agent's own independent on/off switch -- separate from the
    general agent_control mode (which gates manual/Discord-typed signal
    processing) and not gated by it at all. Defaults to ON so the feature
    runs out of the box.
    """
    control = load_state().get("automate_agent_control") or {}
    mode = str(control.get("mode") or "ON").upper()
    return mode if mode in {"ON", "OFF"} else "ON"


@_state_mutation
def set_automate_agent_mode(mode: str, updated_by: str = "") -> Dict[str, Any]:
    normalized = str(mode or "").upper()
    if normalized not in {"ON", "OFF"}:
        raise ValueError("automate_agent mode must be ON or OFF.")
    state = load_state()
    control = state.setdefault("automate_agent_control", {})
    control.update(
        {
            "mode": normalized,
            "updated_at": _utcstamp(),
            "updated_by": str(updated_by or ""),
        }
    )
    save_state(state)
    return dict(control)


def _learning_key(features: Dict[str, Any]) -> str:
    asset = str(features.get("asset_type") or "unknown").lower()
    action = str(features.get("action") or "unknown").upper()
    direction = str(features.get("direction") or "").upper()
    keywords = ",".join(sorted(str(x).lower() for x in (features.get("keywords") or []))) or "plain"
    dte_bucket = str(features.get("dte_bucket") or "na")
    strategy = str(features.get("strategy") or "plain").lower()
    order_type = str(features.get("order_type") or "na").upper()
    contract_status = str(features.get("contract_status") or "na").upper()
    semantic_fields = ",".join(
        sorted(str(value).lower() for value in (features.get("semantic_fields") or []))
    ) or "none"
    leg_count = str(features.get("leg_count") or 1)
    return "|".join(
        [
            asset,
            action,
            direction or "na",
            strategy,
            order_type,
            contract_status,
            f"legs:{leg_count}",
            semantic_fields,
            keywords,
            dte_bucket,
        ]
    )


@_state_mutation
def record_learning_event(features: Dict[str, Any], outcome: Dict[str, Any], limit: int = 2000) -> Dict[str, Any]:
    """Store bounded pattern statistics used to tune future decisions.

    This is deliberately conservative: the agent learns reliability from
    repeated patterns, but it does not rewrite rules or execute trades without
    the normal prediction, validation, and Alpaca safety gates.
    """
    state = load_state()
    patterns = state.setdefault("signal_learning", {})
    key = _learning_key(features)
    item = patterns.setdefault(
        key,
        {
            "key": key,
            "asset_type": str(features.get("asset_type") or "unknown"),
            "action": str(features.get("action") or "").upper(),
            "direction": str(features.get("direction") or "").upper(),
            "keywords": list(features.get("keywords") or []),
            "dte_bucket": str(features.get("dte_bucket") or "na"),
            "strategy": str(features.get("strategy") or "plain"),
            "order_type": str(features.get("order_type") or "na").upper(),
            "contract_status": str(features.get("contract_status") or "na").upper(),
            "semantic_fields": list(features.get("semantic_fields") or []),
            "leg_count": int(features.get("leg_count") or 1),
            "seen": 0,
            "approved": 0,
            "blocked": 0,
            "reviewed": 0,
            "score_total": 0.0,
            "return_total": 0.0,
            "last_reason": "",
            "updated_at": "",
        },
    )
    item["seen"] = int(item.get("seen") or 0) + 1
    final_action = str(outcome.get("final_action") or "").upper()
    intended = str(features.get("action") or "").upper()
    if final_action and final_action == intended and final_action in {"BUY", "SELL"}:
        item["approved"] = int(item.get("approved") or 0) + 1
    elif final_action in {"REVIEW", "NO_TRADE", "INVALID"}:
        item["reviewed"] = int(item.get("reviewed") or 0) + 1
    else:
        item["blocked"] = int(item.get("blocked") or 0) + 1
    item["score_total"] = float(item.get("score_total") or 0.0) + float(outcome.get("score") or 0.0)
    item["return_total"] = float(item.get("return_total") or 0.0) + float(outcome.get("predicted_return_pct") or 0.0)
    item["last_reason"] = str(outcome.get("reason") or "")[:300]
    item["updated_at"] = _utcstamp()

    # Keep the map from growing forever.
    if len(patterns) > limit:
        ordered = sorted(patterns.values(), key=lambda x: str(x.get("updated_at") or ""))
        keep = {x["key"]: x for x in ordered[-limit:] if x.get("key")}
        state["signal_learning"] = keep
        item = keep.get(key, item)
    save_state(state)
    return dict(item)


def get_pattern_learning(features: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    return dict((state.get("signal_learning") or {}).get(_learning_key(features), {}))


def get_signal_learning_summary(limit: int = 8) -> Dict[str, Any]:
    patterns = list((load_state().get("signal_learning") or {}).values())
    patterns.sort(key=lambda x: (int(x.get("seen") or 0), str(x.get("updated_at") or "")), reverse=True)
    trimmed = []
    for item in patterns[:limit]:
        seen = max(1, int(item.get("seen") or 0))
        approved = int(item.get("approved") or 0)
        trimmed.append(
            {
                **item,
                "approval_rate": round(approved / seen * 100, 2),
                "avg_score": round(float(item.get("score_total") or 0.0) / seen, 2),
                "avg_return": round(float(item.get("return_total") or 0.0) / seen, 4),
            }
        )
    return {"patterns": trimmed, "total_patterns": len(patterns)}


@_state_mutation
def record_parser_learning(
    raw_text: str,
    kind: str,
    valid: bool,
    reason: str = "",
    limit: int = 2000,
) -> Dict[str, Any]:
    """Learn bounded format statistics from every incoming signal."""
    state = load_state()
    patterns = state.setdefault("parser_learning", {})
    signature = signal_template_signature(raw_text) or "<EMPTY>"
    item = patterns.setdefault(
        signature,
        {
            "signature": signature,
            "seen": 0,
            "valid": 0,
            "invalid": 0,
            "kind_counts": {},
            "last_reason": "",
            "examples": [],
            "updated_at": "",
        },
    )
    item["seen"] = int(item.get("seen") or 0) + 1
    bucket = "valid" if valid else "invalid"
    item[bucket] = int(item.get(bucket) or 0) + 1
    normalized_kind = str(kind or "UNKNOWN").upper()
    kind_counts = item.setdefault("kind_counts", {})
    kind_counts[normalized_kind] = int(kind_counts.get(normalized_kind) or 0) + 1
    item["last_reason"] = str(reason or "")[:300]
    examples = list(item.get("examples") or [])
    example = " ".join(str(raw_text or "").split())[:300]
    if example and example not in examples:
        examples.append(example)
    item["examples"] = examples[-3:]
    item["updated_at"] = _utcstamp()

    if len(patterns) > limit:
        ordered = sorted(patterns.values(), key=lambda x: str(x.get("updated_at") or ""))
        state["parser_learning"] = {
            entry["signature"]: entry
            for entry in ordered[-limit:]
            if entry.get("signature")
        }
        item = state["parser_learning"].get(signature, item)
    save_state(state)
    return dict(item)


def get_parser_learning_summary(limit: int = 8) -> Dict[str, Any]:
    patterns = list((load_state().get("parser_learning") or {}).values())
    patterns.sort(
        key=lambda item: (int(item.get("seen") or 0), str(item.get("updated_at") or "")),
        reverse=True,
    )
    total_seen = sum(int(item.get("seen") or 0) for item in patterns)
    valid = sum(int(item.get("valid") or 0) for item in patterns)
    return {
        "patterns": patterns[:limit],
        "total_patterns": len(patterns),
        "total_seen": total_seen,
        "valid": valid,
        "invalid": max(0, total_seen - valid),
        "success_rate": round(valid / total_seen * 100, 2) if total_seen else 0.0,
    }


@_state_mutation
def record_option_validation_event(record: Dict[str, Any], limit: int = 1000) -> None:
    state = load_state()
    events = state.setdefault("option_validation_events", [])
    item = dict(record)
    item["created_at"] = _utcstamp()
    events.append(item)
    if len(events) > limit:
        state["option_validation_events"] = events[-limit:]
    save_state(state)


def get_option_validation_summary(limit: int = 250) -> Dict[str, Any]:
    events = list((load_state().get("option_validation_events") or [])[-limit:])
    total = len(events)
    exact_attempted = [
        x for x in events
        if str(x.get("exact_status") or "").upper() not in {"", "SKIPPED"}
    ]
    exact_success = [
        x for x in exact_attempted
        if str(x.get("exact_status") or "").upper() == "SUCCESS"
    ]
    proxy_used = [
        x for x in events
        if str(x.get("proxy_status") or "").strip()
    ]
    return {
        "total": total,
        "exact_attempted": len(exact_attempted),
        "exact_success": len(exact_success),
        "exact_success_rate": round(len(exact_success) / max(1, len(exact_attempted)) * 100, 2),
        "delta_proxy_used": len(proxy_used),
        "recent": list(reversed(events[-8:])),
    }


@_state_mutation
def add_pending_buy(
    symbol: str,
    qty: float,
    order_id: str,
    stop_loss_pct: float = 1.0,
    protected_qty: float = 0.0,
    take_profit_pct: float = 10.0,
) -> None:
    if not order_id:
        return
    state = load_state()
    pending = state.setdefault("pending_buy_orders", {})
    previous = pending.get(order_id) or {}
    pending[order_id] = {
        "symbol": symbol.upper(),
        "qty": float(qty),
        "order_id": order_id,
        "stop_loss_pct": max(0.0, float(stop_loss_pct)),
        "take_profit_pct": max(0.0, float(take_profit_pct)),
        "protected_qty": max(
            float(previous.get("protected_qty") or 0), float(protected_qty)
        ),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_state(state)


@_state_mutation
def update_pending_buy_protected_qty(order_id: str, protected_qty: float) -> None:
    state = load_state()
    item = state.setdefault("pending_buy_orders", {}).get(str(order_id or ""))
    if not item:
        return
    item["protected_qty"] = max(0.0, float(protected_qty))
    save_state(state)


@_state_mutation
def add_pending_market_buy(symbol: str, qty: float, reason: str = "") -> Dict[str, Any]:
    state = load_state()
    pending = state.setdefault("pending_buy_orders", {})
    sym = symbol.upper()
    key = f"queued-buy:{uuid.uuid4().hex}"
    item = {
        "pending_key": key,
        "symbol": sym,
        "qty": float(qty),
        "order_id": key,
        "queued": True,
        "reason": reason,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    pending[key] = item
    save_state(state)
    return item


@_state_mutation
def remove_pending_buy(order_id: str) -> None:
    if not order_id:
        return
    state = load_state()
    state.setdefault("pending_buy_orders", {}).pop(order_id, None)
    save_state(state)


def list_pending_buys() -> List[Dict[str, Any]]:
    return list((load_state().get("pending_buy_orders") or {}).values())


@_state_mutation
def add_pending_sell(symbol: str, qty: float, reason: str = "") -> Dict[str, Any]:
    state = load_state()
    pending = state.setdefault("pending_sell_orders", {})
    sym = symbol.upper()
    key = f"queued-sell:{uuid.uuid4().hex}"
    item = {
        "pending_key": key,
        "symbol": sym,
        "qty": float(qty),
        "reason": reason,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    pending[key] = item
    save_state(state)
    return item


@_state_mutation
def remove_pending_sell(pending_key: str) -> None:
    state = load_state()
    pending = state.setdefault("pending_sell_orders", {})
    key = str(pending_key or "")
    if key in pending:
        pending.pop(key, None)
    else:
        legacy_symbol = key.upper()
        for item_key, item in list(pending.items()):
            if str(item.get("symbol") or "").upper() == legacy_symbol:
                pending.pop(item_key, None)
    save_state(state)


def list_pending_sells() -> List[Dict[str, Any]]:
    return list((load_state().get("pending_sell_orders") or {}).values())


@_state_mutation
def add_pending_option_order(order: Dict[str, Any]) -> None:
    state = load_state()
    pending = state.setdefault("pending_option_orders", {})
    key = str(order.get("pending_key") or order.get("occ_symbol") or order.get("symbol") or "")
    if not key:
        key = hashlib.sha256(json.dumps(order, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    item = dict(order)
    item["pending_key"] = key
    item["occ_symbol"] = str(item.get("occ_symbol") or "")
    item["created_at"] = _utcstamp()
    pending[key] = item
    save_state(state)


@_state_mutation
def remove_pending_option_order(occ_symbol: str) -> None:
    state = load_state()
    state.setdefault("pending_option_orders", {}).pop(str(occ_symbol or ""), None)
    save_state(state)


def list_pending_option_orders() -> List[Dict[str, Any]]:
    return list((load_state().get("pending_option_orders") or {}).values())


@_state_mutation
def add_pending_exit_order(order: Dict[str, Any]) -> None:
    """Persist a submitted broker exit until its actual fills are reconciled."""
    order_id = str(order.get("order_id") or "")
    if not order_id:
        return
    state = load_state()
    pending = state.setdefault("pending_exit_orders", {})
    previous = pending.get(order_id) or {}
    item = dict(previous)
    item.update(order)
    item["order_id"] = order_id
    item["asset_type"] = str(item.get("asset_type") or "equity").lower()
    raw_symbol = str(item.get("symbol") or "")
    item["symbol"] = (
        raw_symbol if item["asset_type"] == "option_mleg" else raw_symbol.upper()
    )
    item["requested_qty"] = max(0.0, float(item.get("requested_qty") or 0))
    item["reconciled_qty"] = max(0.0, float(previous.get("reconciled_qty") or 0))
    item.setdefault("created_at", _utcstamp())
    pending[order_id] = item
    save_state(state)


@_state_mutation
def update_pending_exit_order(order_id: str, **updates: Any) -> None:
    state = load_state()
    item = state.setdefault("pending_exit_orders", {}).get(str(order_id or ""))
    if not item:
        return
    item.update(updates)
    item["updated_at"] = _utcstamp()
    save_state(state)


@_state_mutation
def remove_pending_exit_order(order_id: str) -> None:
    state = load_state()
    state.setdefault("pending_exit_orders", {}).pop(str(order_id or ""), None)
    save_state(state)


def list_pending_exit_orders() -> List[Dict[str, Any]]:
    return list((load_state().get("pending_exit_orders") or {}).values())


@_state_mutation
def add_pending_option_entry_order(order: Dict[str, Any]) -> None:
    order_id = str(order.get("order_id") or "")
    if not order_id:
        return
    state = load_state()
    pending = state.setdefault("pending_option_entry_orders", {})
    previous = pending.get(order_id) or {}
    item = dict(previous)
    item.update(order)
    item["order_id"] = order_id
    item["occ_symbol"] = str(item.get("occ_symbol") or "").upper()
    item["requested_qty"] = max(0.0, float(item.get("requested_qty") or 0))
    item["reconciled_qty"] = max(0.0, float(previous.get("reconciled_qty") or 0))
    item.setdefault("created_at", _utcstamp())
    pending[order_id] = item
    save_state(state)


@_state_mutation
def update_pending_option_entry_order(order_id: str, **updates: Any) -> None:
    state = load_state()
    item = state.setdefault("pending_option_entry_orders", {}).get(str(order_id or ""))
    if not item:
        return
    item.update(updates)
    item["updated_at"] = _utcstamp()
    save_state(state)


@_state_mutation
def remove_pending_option_entry_order(order_id: str) -> None:
    state = load_state()
    state.setdefault("pending_option_entry_orders", {}).pop(str(order_id or ""), None)
    save_state(state)


def list_pending_option_entry_orders() -> List[Dict[str, Any]]:
    return list((load_state().get("pending_option_entry_orders") or {}).values())


@_state_mutation
def add_pending_multi_leg_entry_order(order: Dict[str, Any]) -> None:
    order_id = str(order.get("order_id") or "")
    if not order_id:
        return
    state = load_state()
    pending = state.setdefault("pending_multi_leg_entry_orders", {})
    previous = pending.get(order_id) or {}
    item = dict(previous)
    item.update(order)
    item["order_id"] = order_id
    item["requested_qty"] = max(0.0, float(item.get("requested_qty") or 0))
    item["reconciled_qty"] = max(0.0, float(previous.get("reconciled_qty") or 0))
    item.setdefault("created_at", _utcstamp())
    pending[order_id] = item
    save_state(state)


@_state_mutation
def update_pending_multi_leg_entry_order(order_id: str, **updates: Any) -> None:
    state = load_state()
    item = state.setdefault("pending_multi_leg_entry_orders", {}).get(str(order_id or ""))
    if not item:
        return
    item.update(updates)
    item["updated_at"] = _utcstamp()
    save_state(state)


@_state_mutation
def remove_pending_multi_leg_entry_order(order_id: str) -> None:
    state = load_state()
    state.setdefault("pending_multi_leg_entry_orders", {}).pop(str(order_id or ""), None)
    save_state(state)


def list_pending_multi_leg_entry_orders() -> List[Dict[str, Any]]:
    return list((load_state().get("pending_multi_leg_entry_orders") or {}).values())


@_state_mutation
def upsert_position(
    symbol: str,
    qty: float,
    entry_price: float,
    order_id: str = "",
    stop_loss_pct: float = 1.0,
    take_profit_pct: float = 10.0,
    side: str = "long",
    opened_by: str = "",
    exit_before_market_close: bool = False,
) -> None:
    state = load_state()
    positions = state.setdefault("agent_positions", {})
    sym = symbol.upper()
    previous = positions.get(sym) or {}
    previous_qty = float(previous.get("qty") or 0)
    previous_entry = float(previous.get("entry_price") or 0)
    # A symbol can only be long or short at once at the broker, so a repeat
    # upsert always shares the previous entry's side; the side argument only
    # matters for the very first fill.
    normalized_side = str(previous.get("side") or side or "long").lower()
    short_position = normalized_side == "short"
    combined_qty = previous_qty + float(qty)
    if previous_qty > 0 and previous_entry > 0:
        combined_entry = ((previous_entry * previous_qty) + (entry_price * float(qty))) / combined_qty
    else:
        combined_entry = float(entry_price)
    stop_multiplier = 1 + max(0.0, float(stop_loss_pct)) / 100.0 if short_position else 1 - max(0.0, float(stop_loss_pct)) / 100.0
    target_multiplier = 1 - max(0.0, float(take_profit_pct)) / 100.0 if short_position else 1 + max(0.0, float(take_profit_pct)) / 100.0
    positions[sym] = {
        "symbol": sym,
        "side": normalized_side,
        "qty": round(combined_qty, 6),
        "entry_price": round(combined_entry, 6),
        "stop_price": round(combined_entry * stop_multiplier, 6),
        "target_price": round(combined_entry * target_multiplier, 6),
        "stop_loss_pct": max(0.0, float(stop_loss_pct)),
        "take_profit_pct": max(0.0, float(take_profit_pct)),
        "last_order_id": order_id,
        # Sticky once set, same pattern as options' exit_before_market_close:
        # a repeat upsert (e.g. adding shares) shouldn't silently clear either
        # flag from the position's first fill.
        "opened_by": str(opened_by or previous.get("opened_by") or ""),
        "exit_before_market_close": bool(
            exit_before_market_close or previous.get("exit_before_market_close")
        ),
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_state(state)


@_state_mutation
def reduce_or_remove_position(symbol: str, qty: float) -> None:
    state = load_state()
    positions = state.setdefault("agent_positions", {})
    sym = symbol.upper()
    current = positions.get(sym)
    if not current:
        return
    remaining = float(current.get("qty") or 0) - float(qty)
    if remaining <= 0:
        positions.pop(sym, None)
    else:
        current["qty"] = round(remaining, 6)
        current["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_state(state)


@_state_mutation
def close_position_with_outcome(
    symbol: str,
    qty: float,
    exit_price: float,
    reason: str = "",
    limit: int = 500,
) -> Dict[str, Any]:
    state = load_state()
    positions = state.setdefault("agent_positions", {})
    sym = symbol.upper()
    current = positions.get(sym)
    if not current:
        save_state(state)
        return {}

    close_qty = min(float(qty), float(current.get("qty") or 0))
    entry_price = float(current.get("entry_price") or 0)
    exit_price = float(exit_price or 0)
    if close_qty <= 0 or entry_price <= 0 or exit_price <= 0:
        save_state(state)
        return {}

    short_position = str(current.get("side") or "long").lower() == "short"
    pnl_pct = (
        (entry_price - exit_price) if short_position else (exit_price - entry_price)
    ) / entry_price * 100
    pnl_value = (
        (entry_price - exit_price) if short_position else (exit_price - entry_price)
    ) * close_qty
    outcome = {
        "symbol": sym,
        "side": "buy_to_close_short" if short_position else "sell_to_close_long",
        "qty": round(close_qty, 6),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "pnl_pct": round(pnl_pct, 6),
        "pnl_value": round(pnl_value, 6),
        "profitable": pnl_pct > 0,
        "reason": reason,
        "opened_by": str(current.get("opened_by") or ""),
        "closed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    remaining = float(current.get("qty") or 0) - close_qty
    if remaining <= 0:
        positions.pop(sym, None)
    else:
        current["qty"] = round(remaining, 6)
        current["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    outcomes = state.setdefault("trade_outcomes", [])
    outcomes.append(outcome)
    if len(outcomes) > limit:
        state["trade_outcomes"] = outcomes[-limit:]

    profile = _learning_profile(state)
    profile["closed_trades"] = int(profile.get("closed_trades") or 0) + 1
    if pnl_pct > 0:
        profile["winning_trades"] = int(profile.get("winning_trades") or 0) + 1
        profile["buy_score_adjustment"] = max(-10.0, float(profile.get("buy_score_adjustment") or 0) - 1.0)
        profile["buy_min_return_adjustment"] = max(-0.03, float(profile.get("buy_min_return_adjustment") or 0) - 0.002)
        profile["sell_score_adjustment"] = max(-8.0, float(profile.get("sell_score_adjustment") or 0) - 0.5)
    else:
        profile["losing_trades"] = int(profile.get("losing_trades") or 0) + 1
        profile["buy_score_adjustment"] = min(15.0, float(profile.get("buy_score_adjustment") or 0) + 2.0)
        profile["buy_min_return_adjustment"] = min(0.08, float(profile.get("buy_min_return_adjustment") or 0) + 0.005)
        profile["sell_score_adjustment"] = max(-8.0, float(profile.get("sell_score_adjustment") or 0) - 1.0)

    save_state(state)
    return outcome


def today_realized_pnl(opened_by: str = "") -> float:
    """Sums pnl_value from today's (UTC) closed-trade outcomes.

    Pass opened_by="automate_agent" to scope this to autonomous trades only
    -- a real user's own trading shouldn't count against automate_agent's
    daily-loss circuit breaker, or vice versa.
    """
    today = _today_prefix()
    total = 0.0
    for outcome in load_state().get("trade_outcomes", []):
        closed_at = str(outcome.get("closed_at") or "")
        if not closed_at.startswith(today):
            continue
        if opened_by and str(outcome.get("opened_by") or "") != opened_by:
            continue
        total += float(outcome.get("pnl_value") or 0)
    return round(total, 6)


def list_trade_outcomes(opened_by: str = "", today_only: bool = False) -> List[Dict[str, Any]]:
    """Returns closed-trade outcome records, optionally scoped to one
    opened_by tag and/or today (UTC). Backs the automate_agent daily
    report -- every field a report row needs (symbol, qty, entry/exit
    price, pnl) is already recorded by close_position_with_outcome.
    """
    today = _today_prefix()
    results = []
    for outcome in load_state().get("trade_outcomes", []):
        if opened_by and str(outcome.get("opened_by") or "") != opened_by:
            continue
        if today_only and not str(outcome.get("closed_at") or "").startswith(today):
            continue
        results.append(outcome)
    return results


def get_automate_agent_report_date() -> str:
    """The ET calendar date (YYYY-MM-DD) automate_agent's daily report was
    last posted for -- prevents posting the same day's report twice
    (including across a bot restart, since this is persisted).
    """
    return str(load_state().get("automate_agent_report_date") or "")


@_state_mutation
def set_automate_agent_report_date(date_label: str) -> None:
    state = load_state()
    state["automate_agent_report_date"] = str(date_label or "")
    save_state(state)


@_state_mutation
def remove_position(symbol: str) -> None:
    state = load_state()
    state.setdefault("agent_positions", {}).pop(symbol.upper(), None)
    save_state(state)


def list_positions() -> List[Dict[str, Any]]:
    return list((load_state().get("agent_positions") or {}).values())


@_state_mutation
def add_decision_history(record: Dict[str, Any], limit: int = 500) -> None:
    state = load_state()
    history = state.setdefault("decision_history", [])
    entry = dict(record)
    entry["created_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    history.append(entry)
    if len(history) > limit:
        state["decision_history"] = history[-limit:]
    save_state(state)


def _utcnow() -> datetime:
    return datetime.utcnow()


def _utcstamp() -> str:
    return _utcnow().isoformat(timespec="seconds") + "Z"


def _today_prefix() -> str:
    return _utcnow().date().isoformat()


def _signal_hash(raw_text: str) -> str:
    normalized = " ".join(str(raw_text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@_state_mutation
def record_signal_event(
    raw_text: str,
    kind: str,
    symbol: str = "",
    action: str = "",
    user_id: str = "",
    channel_id: str = "",
    limit: int = 1000,
) -> Dict[str, Any]:
    state = load_state()
    events = state.setdefault("signal_events", [])
    event = {
        "hash": _signal_hash(raw_text),
        "kind": kind,
        "symbol": str(symbol or "").upper(),
        "action": str(action or "").upper(),
        "raw_text": raw_text,
        "user_id": str(user_id or ""),
        "channel_id": str(channel_id or ""),
        "created_at": _utcstamp(),
    }
    events.append(event)
    if len(events) > limit:
        state["signal_events"] = events[-limit:]
    save_state(state)
    return event


@_state_mutation
def record_order_event(record: Dict[str, Any], limit: int = 1000) -> None:
    state = load_state()
    events = state.setdefault("order_events", [])
    item = dict(record)
    item["created_at"] = _utcstamp()
    events.append(item)
    if len(events) > limit:
        state["order_events"] = events[-limit:]
    save_state(state)


@_state_mutation
def record_safety_block(record: Dict[str, Any], limit: int = 1000) -> None:
    state = load_state()
    blocks = state.setdefault("safety_blocks", [])
    item = dict(record)
    item["created_at"] = _utcstamp()
    blocks.append(item)
    if len(blocks) > limit:
        state["safety_blocks"] = blocks[-limit:]
    save_state(state)


def count_today_order_events() -> int:
    today = _today_prefix()
    return sum(
        1 for event in (load_state().get("order_events") or [])
        if str(event.get("created_at") or "").startswith(today)
        and str(event.get("status") or "").lower() in {"submitted", "placed", "accepted", "filled"}
    )


def last_order_for_symbol(symbol: str, side: str = "") -> Dict[str, Any]:
    sym = str(symbol or "").upper()
    side_norm = str(side or "").lower()
    for event in reversed(load_state().get("order_events") or []):
        if str(event.get("symbol") or "").upper() != sym:
            continue
        if side_norm and str(event.get("side") or "").lower() != side_norm:
            continue
        return dict(event)
    return {}


def get_daily_summary() -> Dict[str, Any]:
    state = load_state()
    today = _today_prefix()

    def is_today(item: Dict[str, Any]) -> bool:
        return str(item.get("created_at") or item.get("recorded_at") or "").startswith(today)

    signals = [x for x in state.get("signal_events", []) if is_today(x)]
    orders = [x for x in state.get("order_events", []) if is_today(x)]
    blocks = [x for x in state.get("safety_blocks", []) if is_today(x)]
    decisions = [x for x in state.get("decision_history", []) if is_today(x)]
    return {
        "date_utc": today,
        "signals": len(signals),
        "decisions": len(decisions),
        "orders": len(orders),
        "blocks": len(blocks),
        "equity_positions": len(state.get("agent_positions") or {}),
        "option_positions": len(state.get("option_positions") or {}),
        "no_trade": sum(1 for x in signals if x.get("kind") == "NO_TRADE"),
        "invalid": sum(1 for x in signals if x.get("kind") == "INVALID"),
    }


# ── Options positions and journal ─────────────────────────────────────────────
# Mirrors the equity position/decision-history functions above, kept separate
# so option contracts (OCC symbols, strike/side/expiry) never collide with
# equity ticker tracking.


@_state_mutation
def record_option_journal_entry(entry: Dict[str, Any], limit: int = 500) -> None:
    """Record a parsed options signal for tracking/learning, whether or not
    an order was actually placed for it (e.g. past-tense 'Bought X' journal
    entries never place an order but still get recorded here).
    """
    state = load_state()
    journal = state.setdefault("option_journal", [])
    item = dict(entry)
    item["recorded_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    journal.append(item)
    if len(journal) > limit:
        state["option_journal"] = journal[-limit:]
    save_state(state)


@_state_mutation
def upsert_option_position(
    occ_symbol: str,
    root: str,
    side: str,
    strike: float,
    expiry_date: str,
    qty: float,
    entry_price: float,
    order_id: str = "",
    stop_loss: float | None = None,
    target_price: float | None = None,
    signal_quality: float | None = None,
    position_intent: str = "buy_to_open",
    target_prices: list[float] | tuple[float, ...] | None = None,
    trailing_stop_pct: float | None = None,
    exit_before_market_close: bool = False,
    exit_minutes_before_close: int | None = None,
    exit_if_target_not_hit: bool = False,
    risk_stop_pct: float | None = None,
    position_type: str | None = None,
    maximum_loss_amount: float | None = None,
    exit_underlying_direction: str | None = None,
    exit_underlying_price: float | None = None,
    time_in_force: str | None = None,
    stop_scope: str | None = None,
    default_stop_loss_pct: float = 5.0,
    default_take_profit_pct: float = 10.0,
    opened_by: str = "",
) -> None:
    state = load_state()
    positions = state.setdefault("option_positions", {})
    previous = positions.get(occ_symbol) or {}
    previous_qty = float(previous.get("qty") or 0)
    previous_entry = float(previous.get("entry_price") or 0)
    combined_qty = previous_qty + float(qty)
    if previous_qty > 0 and previous_entry > 0:
        combined_entry = ((previous_entry * previous_qty) + (entry_price * float(qty))) / combined_qty
    else:
        combined_entry = float(entry_price)
    intent = str(position_intent or previous.get("position_intent") or "buy_to_open")
    is_short = intent == "sell_to_open"
    stop_is_explicit = stop_loss is not None or previous.get("stop_loss_source") == "signal"
    target_is_explicit = target_price is not None or previous.get("target_price_source") == "signal"
    effective_stop_loss_pct = (
        float(risk_stop_pct)
        if risk_stop_pct is not None and float(risk_stop_pct) > 0
        else float(default_stop_loss_pct)
    )
    if stop_loss is not None:
        resolved_stop = float(stop_loss)
    elif stop_is_explicit and previous.get("stop_loss") is not None:
        resolved_stop = float(previous["stop_loss"])
    else:
        resolved_stop = combined_entry * (
            1 + max(0.0, effective_stop_loss_pct) / 100.0
            if is_short else
            1 - max(0.0, effective_stop_loss_pct) / 100.0
        )
    provided_targets = list(target_prices or [])
    has_tiered_targets = bool(provided_targets or previous.get("target_prices"))
    if target_price is not None:
        resolved_target = float(target_price)
    elif target_is_explicit and previous.get("target_price") is not None:
        resolved_target = float(previous["target_price"])
    elif has_tiered_targets:
        resolved_target = None
    else:
        resolved_target = combined_entry * (
            1 - max(0.0, float(default_take_profit_pct)) / 100.0
            if is_short else
            1 + max(0.0, float(default_take_profit_pct)) / 100.0
        )
    positions[occ_symbol] = {
        "occ_symbol": occ_symbol,
        "root": root.upper(),
        "side": side.upper(),
        "strike": round(float(strike), 6) if strike is not None else None,
        "expiry_date": expiry_date,
        "qty": round(combined_qty, 6),
        "entry_price": round(combined_entry, 6),
        "stop_loss": round(resolved_stop, 6),
        "stop_loss_source": (
            "signal" if stop_is_explicit else "signal_percent" if risk_stop_pct else "default_5pct"
        ),
        "stop_loss_pct": round(effective_stop_loss_pct, 6),
        "target_price": round(resolved_target, 6) if resolved_target is not None else None,
        "target_price_source": "signal" if target_is_explicit else ("tiered" if has_tiered_targets else "default_10pct"),
        "target_prices": [round(float(value), 6) for value in (target_prices or previous.get("target_prices") or [])],
        "target_index": int(previous.get("target_index") or 0),
        "trailing_stop_pct": (
            round(float(trailing_stop_pct), 6)
            if trailing_stop_pct is not None else previous.get("trailing_stop_pct")
        ),
        "peak_price": max(float(previous.get("peak_price") or 0), float(entry_price or 0)),
        "exit_before_market_close": bool(exit_before_market_close or previous.get("exit_before_market_close")),
        "exit_minutes_before_close": int(
            exit_minutes_before_close
            or previous.get("exit_minutes_before_close")
            or 15
        ),
        "exit_if_target_not_hit": bool(
            exit_if_target_not_hit or previous.get("exit_if_target_not_hit")
        ),
        "position_type": position_type or previous.get("position_type"),
        "maximum_loss_amount": (
            round(float(maximum_loss_amount), 2)
            if maximum_loss_amount is not None else previous.get("maximum_loss_amount")
        ),
        "exit_underlying_direction": exit_underlying_direction or previous.get("exit_underlying_direction"),
        "exit_underlying_price": (
            round(float(exit_underlying_price), 6)
            if exit_underlying_price is not None else previous.get("exit_underlying_price")
        ),
        "time_in_force": time_in_force or previous.get("time_in_force") or "DAY",
        "stop_scope": stop_scope or previous.get("stop_scope"),
        "signal_quality": round(float(signal_quality), 2) if signal_quality is not None else previous.get("signal_quality"),
        "position_intent": intent,
        "last_order_id": order_id,
        "opened_by": opened_by or previous.get("opened_by") or "",
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_state(state)


@_state_mutation
def update_option_position(occ_symbol: str, **updates: Any) -> None:
    state = load_state()
    positions = state.setdefault("option_positions", {})
    symbol = str(occ_symbol or "").upper()
    if symbol not in positions:
        return
    positions[symbol].update(updates)
    positions[symbol]["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_state(state)


@_state_mutation
def close_option_position_with_outcome(
    occ_symbol: str,
    qty: float,
    exit_price: float,
    reason: str = "",
    limit: int = 500,
) -> Dict[str, Any]:
    """Option equivalent of close_position_with_outcome -- records a real
    P&L outcome (contract multiplier 100) before a fully-closed option
    position is dropped, so today_realized_pnl() can see it. Regular manual
    option positions have never recorded outcomes; this exists specifically
    so automate_agent's own option trades count toward its daily-loss
    circuit breaker the same way its equity trades already do -- callers
    should only invoke this for opened_by-tagged automate_agent positions,
    to avoid changing learning-profile behavior for manual signals.
    """
    state = load_state()
    positions = state.setdefault("option_positions", {})
    current = positions.get(occ_symbol)
    if not current:
        save_state(state)
        return {}

    close_qty = min(float(qty), float(current.get("qty") or 0))
    entry_price = float(current.get("entry_price") or 0)
    exit_price = float(exit_price or 0)
    if close_qty <= 0 or entry_price <= 0 or exit_price <= 0:
        save_state(state)
        return {}

    short_position = str(current.get("position_intent") or "buy_to_open") == "sell_to_open"
    pnl_pct = (
        (entry_price - exit_price) if short_position else (exit_price - entry_price)
    ) / entry_price * 100
    pnl_value = (
        (entry_price - exit_price) if short_position else (exit_price - entry_price)
    ) * close_qty * 100.0
    outcome = {
        "symbol": occ_symbol,
        "side": "buy_to_close_short" if short_position else "sell_to_close_long",
        "qty": round(close_qty, 6),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "pnl_pct": round(pnl_pct, 6),
        "pnl_value": round(pnl_value, 6),
        "profitable": pnl_pct > 0,
        "reason": reason,
        "opened_by": str(current.get("opened_by") or ""),
        "closed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    remaining = float(current.get("qty") or 0) - close_qty
    if remaining <= 0:
        positions.pop(occ_symbol, None)
    else:
        current["qty"] = round(remaining, 6)
        current["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    outcomes = state.setdefault("trade_outcomes", [])
    outcomes.append(outcome)
    if len(outcomes) > limit:
        state["trade_outcomes"] = outcomes[-limit:]

    save_state(state)
    return outcome


@_state_mutation
def remove_option_position(occ_symbol: str) -> None:
    state = load_state()
    state.setdefault("option_positions", {}).pop(occ_symbol, None)
    save_state(state)


def list_option_positions() -> List[Dict[str, Any]]:
    return list((load_state().get("option_positions") or {}).values())


def list_option_journal(limit: int = 200) -> List[Dict[str, Any]]:
    journal = load_state().get("option_journal") or []
    return list(reversed(journal))[:limit]


@_state_mutation
def upsert_multi_leg_position(
    strategy_id: str,
    root: str,
    structure: str,
    legs: list[Dict[str, Any]],
    qty: float,
    entry_net_price: float,
    price_effect: str,
    order_id: str,
    stop_loss: float | None = None,
    target_price: float | None = None,
    default_stop_loss_pct: float = 5.0,
    default_take_profit_pct: float = 10.0,
    maximum_loss_amount: float | None = None,
) -> None:
    state = load_state()
    positions = state.setdefault("multi_leg_positions", {})
    key = str(strategy_id or order_id or "")
    if not key:
        return
    previous = positions.get(key) or {}
    previous_qty = float(previous.get("qty") or 0)
    previous_entry = float(previous.get("entry_net_price") or 0)
    added_qty = max(0.0, float(qty))
    combined_qty = previous_qty + added_qty
    if combined_qty <= 0:
        return
    if previous_qty > 0 and previous_entry > 0:
        combined_entry = (
            (previous_entry * previous_qty) + (float(entry_net_price) * added_qty)
        ) / combined_qty
    else:
        combined_entry = abs(float(entry_net_price))
    effect = str(price_effect or previous.get("price_effect") or "debit").lower()
    short_strategy = effect == "credit"
    stop = (
        float(stop_loss)
        if stop_loss is not None and float(stop_loss) > 0
        else combined_entry * (
            1 + max(0.0, float(default_stop_loss_pct)) / 100.0
            if short_strategy else
            1 - max(0.0, float(default_stop_loss_pct)) / 100.0
        )
    )
    target = (
        float(target_price)
        if target_price is not None and float(target_price) > 0
        else combined_entry * (
            1 - max(0.0, float(default_take_profit_pct)) / 100.0
            if short_strategy else
            1 + max(0.0, float(default_take_profit_pct)) / 100.0
        )
    )
    positions[key] = {
        "strategy_id": key,
        "root": str(root or "").upper(),
        "structure": str(structure or "multi_leg"),
        "legs": list(legs or previous.get("legs") or []),
        "qty": round(combined_qty, 6),
        "entry_net_price": round(combined_entry, 6),
        "price_effect": effect,
        "stop_loss": round(stop, 6),
        "target_price": round(target, 6),
        "maximum_loss_amount": (
            float(maximum_loss_amount)
            if maximum_loss_amount is not None and float(maximum_loss_amount) > 0
            else previous.get("maximum_loss_amount")
        ),
        "last_order_id": str(order_id or ""),
        "updated_at": _utcstamp(),
    }
    save_state(state)


@_state_mutation
def reduce_or_remove_multi_leg_position(strategy_id: str, qty: float) -> None:
    state = load_state()
    positions = state.setdefault("multi_leg_positions", {})
    key = str(strategy_id or "")
    current = positions.get(key)
    if not current:
        return
    remaining = float(current.get("qty") or 0) - max(0.0, float(qty))
    if remaining <= 0:
        positions.pop(key, None)
    else:
        current["qty"] = round(remaining, 6)
        current["updated_at"] = _utcstamp()
    save_state(state)


@_state_mutation
def remove_multi_leg_position(strategy_id: str) -> None:
    state = load_state()
    state.setdefault("multi_leg_positions", {}).pop(str(strategy_id or ""), None)
    save_state(state)


def list_multi_leg_positions() -> List[Dict[str, Any]]:
    return list((load_state().get("multi_leg_positions") or {}).values())


# NOTE: the incoming-signal queue itself lives in durable_signal_queue.py (SQLite,
# WAL mode, retry/backoff/dead-letter). This module intentionally has no
# enqueue/claim/complete-signal functions -- an earlier JSON-based version of
# that queue was removed from here since discord_agent.py and whatsapp_webhook.py
# only ever imported the SQLite one.


@_state_mutation
def add_conditional_equity_order(order: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    orders = state.setdefault("conditional_equity_orders", {})
    item = dict(order)
    item.setdefault("created_at", _utcstamp())
    item.setdefault("status", "watching")
    key = str(item.get("id") or "")
    if not key:
        key = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
        item["id"] = key
    orders[key] = item
    save_state(state)
    return item


def list_conditional_equity_orders() -> List[Dict[str, Any]]:
    return list((load_state().get("conditional_equity_orders") or {}).values())


@_state_mutation
def remove_conditional_equity_order(order_id: str) -> None:
    state = load_state()
    state.setdefault("conditional_equity_orders", {}).pop(str(order_id or ""), None)
    save_state(state)

