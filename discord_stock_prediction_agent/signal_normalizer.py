"""Normalize noisy Discord and webhook trade alerts before deterministic parsing."""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Mapping, Optional


_OCC_RE = re.compile(
    r"(?<![A-Z0-9])(?:O:|\.)?([A-Z.]{1,6})(\d{6})([CP])(\d{8})(?!\d)",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|text|txt)?\s*|\s*```\s*$", re.IGNORECASE)
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_LEADING_MARKUP_RE = re.compile(r"(?m)^\s*(?:[-*•▪◦]+\s*|>\s*)")

_KEY_ALIASES = {
    "action": ("action", "signal", "side", "direction", "instruction", "order_action"),
    "symbol": ("symbol", "ticker", "stock", "underlying", "asset", "instrument"),
    "quantity": ("quantity", "qty", "shares", "contracts", "size"),
    "strike": ("strike", "strike_price"),
    "option_type": ("option_type", "right", "call_put", "put_call", "type"),
    "expiry": ("expiry", "expiration", "expiration_date", "expiry_date", "exp"),
    "price": ("limit_price", "entry_price", "entry", "premium", "price"),
    "stop_loss": ("stop_loss", "stop", "sl"),
    "target": ("take_profit", "target_price", "target", "tp", "profit_target"),
    "order_type": ("order_type",),
    "price_effect": ("price_effect", "debit_credit"),
}

_ACTION_MAP = {
    "BUY": "BUY",
    "SELL": "SELL",
    "HOLD": "HOLD",
    "LONG": "BUY",
    "SHORT": "SELL",
    "BUY_TO_OPEN": "BTO",
    "SELL_TO_OPEN": "STO",
    "BUY_TO_CLOSE": "BTC",
    "SELL_TO_CLOSE": "STC",
    "OPEN_LONG": "BTO",
    "OPEN_SHORT": "STO",
    "CLOSE_LONG": "STC",
    "CLOSE_SHORT": "BTC",
    "BTO": "BTO",
    "STO": "STO",
    "BTC": "BTC",
    "STC": "STC",
}

_EMOJI_REPLACEMENTS = {
    "🟢": " BUY ",
    "🔵": " BUY ",
    "🔴": " SELL ",
    "🟡": " HOLD ",
    "🛑": " SL ",
    "⛔": " SL ",
    "🎯": " TP ",
    "💰": " TARGET ",
    "📈": " ",
    "📉": " ",
    "🚨": " ",
    "⚡": " ",
    "✅": " ",
}


def _first(payload: Mapping[str, Any], group: str) -> Any:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in _KEY_ALIASES[group]:
        value = lowered.get(key)
        if value not in (None, "", []):
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group(0)) if match else None


def _action(value: Any) -> str:
    normalized = re.sub(r"[^A-Z]+", "_", str(value or "").upper()).strip("_")
    return _ACTION_MAP.get(normalized, normalized.replace("_", " "))


def _option_side(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"CALL", "CALLS", "C", "CE"}:
        return "C"
    if normalized in {"PUT", "PUTS", "P", "PE"}:
        return "P"
    return ""


def _format_number(value: Any) -> str:
    number = _number(value)
    if number is None:
        return ""
    return f"{number:g}"


def _leg_text(leg: Mapping[str, Any], inherited_symbol: str, inherited_expiry: str) -> str:
    action = _action(_first(leg, "action") or "BUY")
    symbol = str(_first(leg, "symbol") or inherited_symbol).upper()
    strike = _format_number(_first(leg, "strike"))
    side = _option_side(_first(leg, "option_type"))
    quantity = _format_number(_first(leg, "quantity"))
    expiry = str(_first(leg, "expiry") or inherited_expiry)
    if not (symbol and strike and side):
        return ""
    tokens = [action]
    if quantity and quantity != "1":
        tokens.append(quantity)
    tokens.extend([symbol, f"{strike}{side}"])
    if expiry:
        tokens.append(expiry)
    return " ".join(tokens)


def _json_to_signal(payload: Any) -> str:
    if isinstance(payload, list):
        parts = [_json_to_signal(item) for item in payload]
        return " / ".join(part for part in parts if part)
    if not isinstance(payload, Mapping):
        return str(payload or "")

    for key in ("message", "text", "alert_message", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    action = _action(_first(payload, "action"))
    symbol = str(_first(payload, "symbol") or "").upper()
    quantity = _format_number(_first(payload, "quantity"))
    strike = _format_number(_first(payload, "strike"))
    option_side = _option_side(_first(payload, "option_type"))
    expiry = str(_first(payload, "expiry") or "")
    price = _format_number(_first(payload, "price"))
    stop_loss = _format_number(_first(payload, "stop_loss"))
    target = _format_number(_first(payload, "target"))
    order_type = str(_first(payload, "order_type") or "").upper()
    price_effect = str(_first(payload, "price_effect") or "").upper()

    legs = payload.get("legs") or payload.get("option_legs")
    if isinstance(legs, list):
        leg_parts = [
            _leg_text(leg, symbol, expiry)
            for leg in legs
            if isinstance(leg, Mapping)
        ]
        text = " / ".join(part for part in leg_parts if part)
        suffix = []
        if price:
            suffix.append(f"@{price}")
        if price_effect in {"DEBIT", "CREDIT"}:
            suffix.append(price_effect)
        if quantity:
            suffix.append(f"QTY {quantity}")
        return " ".join([text, *suffix]).strip()

    tokens = [action, symbol]
    if strike and option_side:
        tokens.append(f"{strike}{option_side}")
        if expiry:
            tokens.append(expiry)
        if price:
            tokens.append(f"@{price}")
    else:
        if quantity:
            tokens.append(f"QTY {quantity}")
        if order_type == "LIMIT" and price:
            tokens.append(f"LIMIT {price}")
        elif price and order_type not in {"", "MARKET"}:
            tokens.extend([order_type, price])
        elif order_type == "MARKET":
            tokens.append("MARKET")
    if strike and option_side and quantity:
        tokens.append(f"QTY {quantity}")
    if stop_loss:
        tokens.append(f"SL {stop_loss}")
    if target:
        tokens.append(f"TP {target}")
    return " ".join(token for token in tokens if token).strip()


def _expand_occ(match: re.Match[str]) -> str:
    root, yymmdd, side, strike_code = match.groups()
    expiry = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
    strike = int(strike_code) / 1000
    return f"{root.upper()} {strike:g}{side.upper()} {expiry}"


def normalize_signal_input(raw: str) -> str:
    """Return canonical text while preserving all explicit trading fields."""
    text = str(raw or "")
    if len(text) > 100_000:
        text = text[:100_000]
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    if stripped.startswith(("{", "[")):
        try:
            converted = _json_to_signal(json.loads(stripped))
            if converted:
                stripped = converted
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    normalized = unicodedata.normalize("NFKC", stripped)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = _LEADING_MARKUP_RE.sub("", normalized)
    for token, replacement in _EMOJI_REPLACEMENTS.items():
        normalized = normalized.replace(token, replacement)
    normalized = (
        normalized.replace("→", " ")
        .replace("➡", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("／", "/")
    )
    normalized = _OCC_RE.sub(_expand_occ, normalized)
    normalized = re.sub(r"[*_`]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def signal_template_signature(raw: str) -> str:
    """Generalize a signal for bounded parser-learning statistics."""
    text = normalize_signal_input(raw).upper()
    text = re.sub(r"\b20\d{2}-\d{1,2}-\d{1,2}\b", "<DATE>", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", "<DATE>", text)
    text = re.sub(r"(?<![A-Z])\$?\d+(?:\.\d+)?%?", "<N>", text)
    text = re.sub(r"\b[A-Z][A-Z.]{0,5}\b", "<W>", text)
    return re.sub(r"\s+", " ", text).strip()[:300]
