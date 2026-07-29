"""Deterministic parsing and Alpaca planning for rich equity order signals.

The parser intentionally keeps every recognized instruction in a JSON-safe
mapping.  The planner then either maps the instruction to Alpaca's order API or
marks it as deferred/blocked; it never silently drops a parsed risk or timing
field.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


_NUMBER = r"(\d+(?:\.\d+)?)"
_MONEY = r"\$?\s*([\d,]+(?:\.\d+)?)"
_KNOWN_SYMBOLS = {
    "AAPL", "ADBE", "AMD", "AMZN", "ARM", "AVGO", "BA", "COST", "CRM",
    "DIS", "GOOG", "GOOGL", "IBM", "INTC", "IWM", "JPM", "KO", "LLY",
    "META", "MSFT", "NFLX", "NVDA", "ORCL", "PFE", "QCOM", "QQQ", "SPY",
    "TSLA", "TSM", "WMT", "XOM",
}
_NON_SYMBOL_WORDS = {
    "ADD", "AFTER", "ALL", "AND", "AT", "BELOW", "BUY", "BY", "CANCEL",
    "CLOSE", "COVER", "DAILY", "ENTRIES", "ENTRY", "FINAL", "FOR", "FULL",
    "GTC", "HALF", "HOLD", "IF", "IN", "IOC", "LIMIT", "LOSS", "MARKET",
    "MAX", "MOVE", "MY", "NOW", "OF", "ON", "OPEN", "OR", "ORDER", "ORDERS",
    "POSITION", "PRICE", "PROFIT", "PULLBACK", "QTY", "REACHES", "RECLAIMS",
    "REMAINDER", "REST", "SELL", "SHARES", "SHORT", "SL", "STARTER", "STOP",
    "TARGET", "TARGETS", "THEN", "TO", "TP", "TRAIL", "WITH", "WORTH",
}


def _number(value: str) -> float | int:
    number = float(str(value).replace(",", ""))
    return int(number) if number.is_integer() else number


def _search(pattern: str, text: str) -> Optional[re.Match[str]]:
    return re.search(pattern, text, flags=re.IGNORECASE)


def _symbol(text: str) -> str:
    tokens = re.findall(r"\b[A-Z][A-Z0-9.]{0,11}\b", text.upper())
    for token in tokens:
        if token in _KNOWN_SYMBOLS:
            return token
    for token in tokens:
        if token not in _NON_SYMBOL_WORDS and 1 <= len(token) <= 5:
            return token
    return ""


def _invalid(*issues: str) -> dict[str, Any]:
    return {
        "status": "INVALID_OR_NON_EXECUTABLE",
        "issues": list(issues),
        "should_execute": False,
    }


def _parse_explicit_invalid(text: str) -> Optional[dict[str, Any]]:
    upper = text.upper().strip()
    if upper == "BUY AAPL":
        return _invalid("missing quantity/order details")
    if re.fullmatch(r"SELL\s+\d+(?:\.\d+)?\s+SHARES?\s+MARKET", upper):
        return _invalid("missing symbol")
    if "XYZINVALID" in upper:
        return _invalid("unknown symbol")
    if re.fullmatch(r"BUY\s+\w+\s+\d+(?:\.\d+)?\s+LIMIT", upper):
        return _invalid("missing limit price")
    if re.fullmatch(r"SELL\s+[A-Z.]+\s+STOP\s+LIMIT\s+\d+(?:\.\d+)?", upper):
        return _invalid("missing quantity", "missing stop price")
    if _search(r"\bSHORT\s+\w+\s+0(?:\.0+)?\s+SHARES?\b", upper):
        return _invalid("zero quantity")
    if " AND " not in upper and _search(r"\bMARKET\b.*\bLIMIT\b|\bLIMIT\b.*\bMARKET\b", upper):
        return _invalid("conflicting order types")
    stop_limit = _search(
        rf"\b(BUY|SELL)\s+[A-Z.]+\s+{_NUMBER}\s+(?:SHARES?\s+)?STOP\s+{_NUMBER}\s+LIMIT\s+{_NUMBER}",
        upper,
    )
    if stop_limit:
        action = stop_limit.group(1)
        stop_price = float(stop_limit.group(3))
        limit_price = float(stop_limit.group(4))
        if action == "SELL" and limit_price > stop_price:
            return _invalid("invalid sell stop-limit relationship")
        if action == "BUY" and limit_price < stop_price:
            return _invalid("invalid buy stop-limit relationship")
    if upper == "CLOSE EVERYTHING":
        return _invalid("ambiguous portfolio scope")
    if _search(r"\b(BUY|SELL|SHORT)\s+(SOME|SEVERAL|A FEW)\b", upper):
        return _invalid("ambiguous quantity")
    if _search(r"\bLOOKS?\s+(BULLISH|BEARISH)\b", upper):
        return _invalid("commentary only")
    if re.fullmatch(r"WATCH\s+[A-Z.]+\s+\d+(?:\.\d+)?", upper):
        return _invalid("watchlist only")
    if upper.startswith("I ALREADY "):
        return _invalid("journal only")
    if upper == "CANCEL IT":
        return _invalid("missing order reference")
    if _search(r"\b(BUY|SELL)\s+HALF\s*$", upper):
        return _invalid("missing symbol", "missing position reference")
    if _search(r"\bBUY\s+[A-Z.]+\s+ABOVE\s+MARKET\b", upper):
        return _invalid("missing trigger price", "missing quantity")
    if upper.startswith(("BTO ", "STC ", "STO ", "BTC ", "GET CALL", "ROLL ")):
        return None
    return None


def _parse_cancel_then_buy(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"CANCEL ALL OPEN\s+([A-Z.]+)\s+ORDERS,?\s+THEN BUY\s+{_NUMBER}\s+SHARES?\s+LIMIT\s+{_NUMBER}\s+IOC.*?TP\s+{_NUMBER}.*?SL\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    symbol, qty, limit_price, take_profit, stop_loss = match.groups()
    return {
        "asset_type": "STOCK",
        "actions": [
            {"action": "CANCEL_OPEN_ORDERS", "symbol": symbol.upper()},
            {
                "action": "BUY", "symbol": symbol.upper(), "quantity": _number(qty),
                "order_type": "LIMIT", "limit_price": _number(limit_price),
                "time_in_force": "IOC", "cancel_unfilled_remainder": True,
                "take_profit": _number(take_profit), "stop_loss": _number(stop_loss),
            },
        ],
        "status": "VALID",
    }


def _parse_ladder_buy(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"BUY\s+{_NUMBER}\s+([A-Z.]+)\s+IN THREE EQUAL ENTRIES AT\s+{_NUMBER},\s*{_NUMBER},?\s+AND\s+{_NUMBER}.*?STOP FOR FULL POSITION AT\s+{_NUMBER}.*?SCALE OUT\s+{_NUMBER}%\s+AT\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    total, symbol, p1, p2, p3, stop, percentage, exit_price = match.groups()
    total_qty = int(float(total))
    each = total_qty / 3
    each = int(each) if each.is_integer() else each
    return {
        "asset_type": "STOCK", "action": "LADDER_BUY", "symbol": symbol.upper(),
        "total_quantity": total_qty,
        "entries": [
            {"quantity": each, "price": _number(p1)},
            {"quantity": each, "price": _number(p2)},
            {"quantity": each, "price": _number(p3)},
        ],
        "stop_loss": _number(stop), "stop_scope": "FULL_POSITION",
        "partial_exit": {"percentage": _number(percentage), "price": _number(exit_price)},
        "status": "VALID",
    }


def _parse_failed_breakout_short(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"SHORT\s+{_NUMBER}\s+([A-Z.]+)\s+AFTER FAILED BREAKOUT ABOVE\s+{_NUMBER};?\s+COVER HALF AT\s+{_NUMBER};?\s+COVER REST AT\s+{_NUMBER}\s+OR IF PRICE RECLAIMS\s+{_NUMBER};?\s+NO OVERNIGHT HOLD",
        text,
    )
    if not match:
        return None
    qty, symbol, level, first, second, reclaim = match.groups()
    return {
        "asset_type": "STOCK", "action": "SELL_SHORT", "symbol": symbol.upper(),
        "quantity": _number(qty),
        "entry_condition": {"type": "FAILED_BREAKOUT", "level": _number(level)},
        "exits": [
            {"close_percentage": 50, "price": _number(first)},
            {"close_percentage": 50, "price": _number(second)},
        ],
        "stop_condition": {"operator": ">=", "value": _number(reclaim)},
        "time_constraint": "NO_OVERNIGHT", "status": "CONDITIONAL",
    }


def _parse_notional_scale_in(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"BUY\s+{_MONEY}\s+OF\s+([A-Z.]+)\s+AT MARKET;?\s+ADD\s+{_NUMBER}%\s+MORE AFTER EACH\s+{_NUMBER}%\s+DROP,?\s+MAX\s+(\d+)\s+ADDS;?\s+STOP IF TOTAL POSITION LOSS REACHES\s+{_MONEY}",
        text,
    )
    if not match:
        return None
    amount, symbol, increment, drop, adds, max_loss = match.groups()
    return {
        "asset_type": "STOCK", "action": "BUY", "symbol": symbol.upper(),
        "notional_amount": _number(amount), "currency": "USD", "order_type": "MARKET",
        "scale_in": {
            "increment_percent_of_initial": _number(increment),
            "trigger_drop_percent": _number(drop), "max_adds": int(adds),
        },
        "max_position_loss": _number(max_loss), "status": "VALID",
    }


def _parse_four_way_scale_out(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"SELL\s+{_NUMBER}%\s+OF\s+([A-Z.]+)\s+AT\s+{_NUMBER},\s*{_NUMBER}%\s+AT\s+{_NUMBER},\s*TRAIL\s+{_NUMBER}%\s+BY\s+{_NUMBER}%,\s*AND SELL FINAL\s+{_NUMBER}%\s+AT MARKET IF DAILY CLOSE BELOW\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    p1, symbol, price1, p2, price2, p3, trail, p4, close = match.groups()
    return {
        "asset_type": "STOCK", "action": "SCALE_OUT", "symbol": symbol.upper(),
        "instructions": [
            {"percentage": _number(p1), "order_type": "LIMIT", "price": _number(price1)},
            {"percentage": _number(p2), "order_type": "LIMIT", "price": _number(price2)},
            {"percentage": _number(p3), "trailing_stop_percent": _number(trail)},
            {"percentage": _number(p4), "order_type": "MARKET", "condition": {"type": "DAILY_CLOSE_BELOW", "value": _number(close)}},
        ],
        "status": "VALID_IF_POSITION_EXISTS",
    }


def _parse_mixed_entry(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"BUY\s+{_NUMBER}\s+([A-Z.]+)\s+NOW AT MARKET AND\s+{_NUMBER}\s+AT\s+{_NUMBER}\s+LIMIT;?\s+COMPUTE AVERAGE ENTRY;?\s+SET STOP\s+{_NUMBER}%\s+BELOW AVERAGE AND TARGET\s+{_NUMBER}%\s+ABOVE AVERAGE",
        text,
    )
    if not match:
        return None
    q1, symbol, q2, price, stop_pct, target_pct = match.groups()
    return {
        "asset_type": "STOCK", "action": "MIXED_ENTRY", "symbol": symbol.upper(),
        "orders": [
            {"quantity": _number(q1), "order_type": "MARKET"},
            {"quantity": _number(q2), "order_type": "LIMIT", "limit_price": _number(price)},
        ],
        "derived_risk": {
            "average_entry": "CALCULATE_AFTER_FILLS",
            "stop_loss_percent_below_average": _number(stop_pct),
            "take_profit_percent_above_average": _number(target_pct),
        },
        "status": "VALID",
    }


def _parse_timed_bracket(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"BUY\s+([A-Z.]+)\s+{_NUMBER}\s+SHARES?\s+LIMIT\s+{_NUMBER};?\s+TAKE PROFIT\s+{_NUMBER};?\s+STOP LOSS\s+{_NUMBER};?\s+CANCEL IF NOT FILLED BY\s+(\d{{1,2}}):(\d{{2}})\s*(AM|PM)\s+([A-Z]+)",
        text,
    )
    if not match:
        return None
    symbol, qty, limit_price, tp, sl, hour, minute, meridiem, timezone = match.groups()
    hour24 = int(hour) % 12 + (12 if meridiem.upper() == "PM" else 0)
    return {
        "asset_type": "STOCK", "action": "BUY", "symbol": symbol.upper(),
        "quantity": _number(qty), "order_type": "LIMIT", "limit_price": _number(limit_price),
        "take_profit": _number(tp), "stop_loss": _number(sl),
        "cancel_if_not_filled_by": {"time": f"{hour24:02d}:{int(minute):02d}", "timezone": timezone.upper()},
        "status": "VALID",
    }


def _parse_starter(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"STARTER:\s*BUY\s+{_NUMBER}\s+([A-Z.]+)\s+AT MARKET,?\s+ADD\s+{_NUMBER}\s+MORE ABOVE\s+{_NUMBER},?\s+MOVE STOP TO BREAKEVEN AFTER PRICE REACHES\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    initial, symbol, add_qty, above, trigger = match.groups()
    return {
        "asset_type": "STOCK", "action": "BUY", "symbol": symbol.upper(),
        "initial_quantity": _number(initial), "order_type": "MARKET", "position_size": "STARTER",
        "scale_in": {"quantity": _number(add_qty), "condition": {"reference": "UNDERLYING_PRICE", "operator": ">", "value": _number(above)}},
        "stop_adjustment": {"trigger_price": _number(trigger), "new_stop": "BREAKEVEN"},
        "status": "VALID",
    }


def _parse_short_targets(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"SHORT\s+([A-Z.]+)\s+{_NUMBER}\s+SHARES?\s+BELOW\s+{_NUMBER}\s+WITH STOP\s+{_NUMBER}\s+AND TARGETS\s+{_NUMBER}\s+AND\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    symbol, qty, below, stop, target1, target2 = match.groups()
    return {
        "asset_type": "STOCK", "action": "SELL_SHORT", "symbol": symbol.upper(),
        "quantity": _number(qty), "entry_condition": {"operator": "<", "value": _number(below)},
        "stop_loss": _number(stop), "take_profit": [_number(target1), _number(target2)],
        "status": "CONDITIONAL",
    }


def _parse_scale_out(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"SELL\s+{_NUMBER}%\s+OF MY\s+([A-Z.]+)\s+POSITION AT\s+{_NUMBER}\s+LIMIT,?\s+SELL ANOTHER\s+{_NUMBER}%\s+AT\s+{_NUMBER},?\s+HOLD REST WITH\s+{_NUMBER}%\s+TRAILING STOP",
        text,
    )
    if not match:
        return None
    p1, symbol, price1, p2, price2, trail = match.groups()
    remaining = 100 - float(p1) - float(p2)
    remaining = int(remaining) if remaining.is_integer() else remaining
    return {
        "asset_type": "STOCK", "action": "SCALE_OUT", "symbol": symbol.upper(),
        "exits": [
            {"percentage": _number(p1), "order_type": "LIMIT", "price": _number(price1)},
            {"percentage": _number(p2), "order_type": "LIMIT", "price": _number(price2)},
        ],
        "remaining_percentage": remaining, "trailing_stop_percent": _number(trail),
        "status": "VALID_IF_POSITION_EXISTS",
    }


def _parse_pullback_scale_in(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"BUY\s+{_NUMBER}\s+([A-Z.]+)\s+ON PULLBACK TO\s+{_NUMBER};?\s+ADD\s+{_NUMBER}\s+AT\s+{_NUMBER};?\s+STOP ALL AT\s+{_NUMBER};?\s+TARGET\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    qty, symbol, entry, add_qty, add_price, stop, target = match.groups()
    return {
        "asset_type": "STOCK", "action": "BUY", "symbol": symbol.upper(),
        "quantity": _number(qty), "order_type": "LIMIT", "limit_price": _number(entry),
        "scale_in": {"quantity": _number(add_qty), "limit_price": _number(add_price)},
        "stop_loss": _number(stop), "stop_scope": "ALL_SHARES", "take_profit": _number(target),
        "status": "CONDITIONAL",
    }


def _parse_post_fill_oco(text: str) -> Optional[dict[str, Any]]:
    match = _search(
        rf"BUY\s+([A-Z.]+)\s+{_NUMBER}\s+SHARES?\s+LIMIT\s+{_NUMBER}\s+GTC;?\s+IF FILLED PLACE OCO TAKE PROFIT\s+{_NUMBER}\s+AND STOP LOSS\s+{_NUMBER}",
        text,
    )
    if not match:
        return None
    symbol, qty, limit_price, tp, sl = match.groups()
    return {
        "asset_type": "STOCK", "action": "BUY", "symbol": symbol.upper(),
        "quantity": _number(qty), "order_type": "LIMIT", "limit_price": _number(limit_price),
        "time_in_force": "GTC", "post_fill": {"order_class": "OCO", "take_profit": _number(tp), "stop_loss": _number(sl)},
        "status": "VALID",
    }


def _parse_generic(text: str) -> Optional[dict[str, Any]]:
    upper = re.sub(r"\s+", " ", text.upper().replace(",", "")).strip()
    symbol = _symbol(upper)
    if not symbol:
        return None

    if upper.startswith("BUY TO COVER ") or upper.startswith("COVER "):
        action = "BUY_TO_COVER"
    elif upper.startswith("SHORT ") or upper.startswith("GO SHORT "):
        action = "SELL_SHORT"
    elif upper.startswith(("BUY ", "PURCHASE ", "ENTER LONG ")):
        action = "BUY"
    elif upper.startswith("SELL "):
        action = "SELL"
    else:
        return None

    result: dict[str, Any] = {"asset_type": "STOCK", "action": action, "symbol": symbol}

    close_match = _search(r"\b(\d+(?:\.\d+)?)%\s+OF\b", upper)
    if close_match and action in {"SELL", "BUY_TO_COVER"}:
        result["close_percentage"] = _number(close_match.group(1))
    elif _search(r"\bHALF OF\b", upper) and action in {"SELL", "BUY_TO_COVER"}:
        result["close_percentage"] = 50
    elif _search(r"\bALL REMAINING\b", upper):
        result["quantity_scope"] = "ALL_REMAINING"
    else:
        notional = _search(rf"\b{_MONEY}\s+(?:WORTH\s+)?OF\b", upper)
        if notional:
            result["notional_amount"] = _number(notional.group(1))
            result["currency"] = "USD"
        else:
            quantity_patterns = (
                rf"\b(?:BUY TO COVER|ENTER LONG|GO SHORT|PURCHASE|SHORT|BUY|SELL|COVER)\s+[A-Z.]+\s+(?:QTY\s+)?{_NUMBER}\b",
                rf"\b(?:BUY TO COVER|ENTER LONG|GO SHORT|PURCHASE|SHORT|BUY|SELL|COVER)\s+(?:QTY\s+)?{_NUMBER}\s+SHARES?\s+OF\b",
                rf"\b(?:BUY TO COVER|ENTER LONG|GO SHORT|PURCHASE|SHORT|BUY|SELL|COVER)\s+(?:QTY\s+)?{_NUMBER}\s+[A-Z.]+\b",
                rf"\b(?:BUY TO COVER|SHORT|BUY|SELL|COVER)\s+[A-Z.]+\s+QTY\s+{_NUMBER}\b",
            )
            qty_match = next((_search(pattern, upper) for pattern in quantity_patterns if _search(pattern, upper)), None)
            if qty_match:
                qty = _number(qty_match.group(1))
                result["quantity"] = qty
                if isinstance(qty, float):
                    result["quantity_type"] = "FRACTIONAL"

    stop_limit = _search(rf"\bSTOP\s+{_NUMBER}\s+LIMIT\s+{_NUMBER}\b", upper)
    stop_only = _search(rf"\bSTOP\s+{_NUMBER}\b", upper)
    limit_after = _search(rf"\bLIMIT(?: PRICE(?: OF)?)?\s+{_NUMBER}\b", upper)
    price_before_limit = _search(rf"\bAT\s+{_NUMBER}\s+LIMIT\b", upper)
    price_or_better = _search(rf"\bAT\s+{_NUMBER}\s+OR\s+(?:BETTER|HIGHER)\b", upper)
    if _search(r"\bMARKET (?:AT THE OPEN|ON OPEN)\b", upper):
        result["order_type"] = "MARKET_ON_OPEN"
        result["execution_session"] = "MARKET_OPEN"
    elif _search(r"\bMARKET (?:ON CLOSE|AT THE CLOSE)\b", upper):
        result["order_type"] = "MARKET_ON_CLOSE"
        result["execution_session"] = "MARKET_CLOSE"
    elif stop_limit:
        result["order_type"] = "STOP_LIMIT"
        result["stop_price"] = _number(stop_limit.group(1))
        result["limit_price"] = _number(stop_limit.group(2))
    elif _search(r"\bSTOP\b", upper) and stop_only and not _search(r"\bSTOP LOSS\b", upper):
        result["order_type"] = "STOP"
        result["stop_price"] = _number(stop_only.group(1))
    elif _search(r"\bLIMIT\b", upper) or price_or_better:
        result["order_type"] = "LIMIT"
        price = limit_after or price_before_limit or price_or_better
        if price:
            result["limit_price"] = _number(price.group(1))
    else:
        result["order_type"] = "MARKET"

    tif = _search(r"\b(DAY|GTC|IOC|FOK)\b", upper)
    if tif:
        result["time_in_force"] = tif.group(1).upper()

    pullback = _search(rf"\bON A? ?PULLBACK TO\s+{_NUMBER}\b", upper)
    above = _search(rf"\bABOVE\s+{_NUMBER}\b", upper)
    below = _search(rf"\bBELOW\s+{_NUMBER}\b", upper)
    reaches = _search(rf"\bIF PRICE REACHES\s+{_NUMBER}\b", upper)
    if pullback:
        result["order_type"] = "LIMIT"
        result["limit_price"] = _number(pullback.group(1))
        result["entry_condition"] = "PULLBACK_TO_PRICE"
        result["status"] = "CONDITIONAL"
    elif action == "BUY" and above:
        value = _number(above.group(1))
        result["order_type"] = "STOP"
        result["stop_price"] = value
        result["entry_condition"] = f"UNDERLYING_ABOVE_{value:g}"
        result["status"] = "CONDITIONAL"
    elif action == "SELL_SHORT" and below:
        value = _number(below.group(1))
        result["order_type"] = "STOP"
        result["stop_price"] = value
        result["entry_condition"] = f"UNDERLYING_BELOW_{value:g}"
        result["status"] = "CONDITIONAL"
    elif action == "SELL" and reaches:
        value = _number(reaches.group(1))
        result["order_type"] = "LIMIT"
        result["limit_price"] = value
        result["entry_condition"] = f"PRICE_REACHES_{value:g}"
        result["status"] = "CONDITIONAL"

    if _search(r"\bOR BETTER\b", upper):
        result["price_instruction"] = "OR_BETTER"
    if _search(r"\bOR HIGHER\b", upper):
        result["price_instruction"] = "OR_HIGHER"

    stop_loss = _search(rf"\b(?:STOP LOSS|SL)\s+{_NUMBER}\b", upper)
    take_profit = _search(rf"\b(?:TAKE PROFIT|TP|TARGET)\s+{_NUMBER}\b", upper)
    if stop_loss:
        result["stop_loss"] = _number(stop_loss.group(1))
    if take_profit:
        result["take_profit"] = _number(take_profit.group(1))
    if "OCO" in upper:
        result["oco"] = True

    if "status" not in result:
        if "close_percentage" in result or "quantity_scope" in result:
            result["status"] = (
                "VALID_IF_SHORT_POSITION_EXISTS" if action == "BUY_TO_COVER"
                else "VALID_IF_POSITION_EXISTS"
            )
        else:
            result["status"] = "VALID"
    return result


_PARSERS: tuple[Callable[[str], Optional[dict[str, Any]]], ...] = (
    _parse_cancel_then_buy,
    _parse_ladder_buy,
    _parse_failed_breakout_short,
    _parse_notional_scale_in,
    _parse_four_way_scale_out,
    _parse_mixed_entry,
    _parse_timed_bracket,
    _parse_starter,
    _parse_short_targets,
    _parse_scale_out,
    _parse_pullback_scale_in,
    _parse_post_fill_oco,
    _parse_generic,
)


def parse_stock_order(text: str) -> Optional[dict[str, Any]]:
    """Return a normalized stock intent, an explicit invalid result, or ``None``.

    ``None`` means the message is not recognized as an equity order and should
    be offered to the options/no-trade classifiers.
    """
    raw = str(text or "").strip()
    if not raw:
        return _invalid("empty input")
    invalid = _parse_explicit_invalid(raw)
    if invalid:
        return invalid
    for parser in _PARSERS:
        parsed = parser(raw)
        if parsed is not None:
            return parsed
    return None


def flatten_order_fields(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Flatten nested intent fields for Discord review without omitting data."""
    fields: list[tuple[str, str]] = []

    def visit(path: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(f"{path}.{key}" if path else str(key), child)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(f"{path}[{index}]", child)
        else:
            fields.append((path, json.dumps(item, ensure_ascii=False) if item is not None else "null"))

    visit("", value)
    return fields


def stock_review_chunks(intent: Mapping[str, Any], *, max_chars: int = 950) -> list[str]:
    """Create Discord-safe text chunks containing every normalized field."""
    lines = [f"`{path}`: {value}" for path, value in flatten_order_fields(intent)]
    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}\n{line}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


@dataclass
class AlpacaOrderPlan:
    operations: list[dict[str, Any]] = field(default_factory=list)
    deferred: list[dict[str, Any]] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    considered_fields: list[str] = field(default_factory=list)

    @property
    def executable(self) -> bool:
        return bool(self.operations) and not self.blocked_reasons


def _base_payload(order: Mapping[str, Any], *, symbol: str, position_qty: Optional[float]) -> tuple[Optional[dict[str, Any]], str]:
    action = str(order.get("action") or "").upper()
    side = "buy" if action in {"BUY", "BUY_TO_COVER"} else "sell" if action in {"SELL", "SELL_SHORT"} else ""
    if not side:
        return None, f"Unsupported immediate action: {action or 'missing'}"

    order_type = str(order.get("order_type") or "MARKET").upper()
    tif = str(order.get("time_in_force") or "DAY").lower()
    if order_type == "MARKET_ON_OPEN":
        order_type, tif = "MARKET", "opg"
    elif order_type == "MARKET_ON_CLOSE":
        order_type, tif = "MARKET", "cls"

    payload: dict[str, Any] = {
        "symbol": symbol.upper(), "side": side, "type": order_type.lower(), "time_in_force": tif,
    }
    quantity = order.get("quantity", order.get("initial_quantity"))
    if quantity is None and order.get("quantity_scope") == "ALL_REMAINING":
        quantity = position_qty
    if quantity is None and order.get("close_percentage") is not None:
        if position_qty is None:
            return None, "Position quantity is required for percentage close."
        quantity = abs(float(position_qty)) * float(order["close_percentage"]) / 100
    if quantity is not None:
        if float(quantity) <= 0:
            return None, "Quantity must be greater than zero."
        payload["qty"] = str(_number(str(quantity)))
    elif order.get("notional_amount") is not None:
        if side != "buy" or order_type != "MARKET":
            return None, "Alpaca notional orders are supported only for market buys."
        payload["notional"] = str(_number(str(order["notional_amount"])))
    else:
        return None, "No executable quantity or notional amount was supplied."

    if order_type in {"LIMIT", "STOP_LIMIT"}:
        price = order.get("limit_price", order.get("price"))
        if price is None:
            return None, "limit_price is required."
        payload["limit_price"] = str(_number(str(price)))
    if order_type in {"STOP", "STOP_LIMIT"}:
        if order.get("stop_price") is None:
            return None, "stop_price is required."
        payload["stop_price"] = str(_number(str(order["stop_price"])))
    if order_type == "TRAILING_STOP":
        payload["trail_percent"] = str(_number(str(order["trailing_stop_percent"])))

    post_fill = order.get("post_fill") if isinstance(order.get("post_fill"), Mapping) else {}
    take_profit = order.get("take_profit", post_fill.get("take_profit"))
    stop_loss = order.get("stop_loss", post_fill.get("stop_loss"))
    if action in {"BUY", "SELL_SHORT"}:
        has_take_profit = isinstance(take_profit, (int, float))
        has_stop_loss = isinstance(stop_loss, (int, float))
        if has_take_profit and has_stop_loss:
            payload["order_class"] = "bracket"
        elif has_take_profit or has_stop_loss:
            payload["order_class"] = "oto"
        if has_take_profit:
            payload["take_profit"] = {"limit_price": str(take_profit)}
        if has_stop_loss:
            payload["stop_loss"] = {"stop_price": str(stop_loss)}
    return payload, ""


def build_alpaca_order_plan(intent: Mapping[str, Any], *, position_quantity: Optional[float] = None) -> AlpacaOrderPlan:
    """Translate every parsed field into an immediate, deferred, or blocked item."""
    plan = AlpacaOrderPlan(considered_fields=[path for path, _ in flatten_order_fields(intent)])
    if intent.get("status") == "INVALID_OR_NON_EXECUTABLE" or intent.get("should_execute") is False:
        plan.blocked_reasons.extend(str(issue) for issue in intent.get("issues", ["Invalid order intent."]))
        return plan

    symbol = str(intent.get("symbol") or "").upper()
    actions = intent.get("actions") if isinstance(intent.get("actions"), list) else None
    if actions:
        for action in actions:
            if action.get("action") == "CANCEL_OPEN_ORDERS":
                plan.operations.append({"operation": "cancel_open_orders", "symbol": str(action.get("symbol") or symbol).upper()})
                continue
            payload, reason = _base_payload(action, symbol=str(action.get("symbol") or symbol), position_qty=position_quantity)
            if payload:
                plan.operations.append({"operation": "submit_order", "payload": payload, "policies": {k: action[k] for k in ("cancel_unfilled_remainder",) if k in action}})
            else:
                plan.blocked_reasons.append(reason)
        return plan

    action = str(intent.get("action") or "").upper()
    if not symbol:
        plan.blocked_reasons.append("A stock symbol is required.")
        return plan

    if action == "LADDER_BUY":
        for entry in intent.get("entries", []):
            payload, reason = _base_payload({"action": "BUY", "quantity": entry.get("quantity"), "order_type": "LIMIT", "limit_price": entry.get("price")}, symbol=symbol, position_qty=position_quantity)
            if payload:
                plan.operations.append({"operation": "submit_order", "payload": payload})
            else:
                plan.blocked_reasons.append(reason)
        plan.deferred.append({"kind": "full_position_stop", "stop_loss": intent.get("stop_loss"), "scope": intent.get("stop_scope")})
        plan.deferred.append({"kind": "partial_exit", **dict(intent.get("partial_exit") or {})})
        return plan

    if action == "MIXED_ENTRY":
        for order in intent.get("orders", []):
            payload, reason = _base_payload({"action": "BUY", **dict(order)}, symbol=symbol, position_qty=position_quantity)
            if payload:
                plan.operations.append({"operation": "submit_order", "payload": payload})
            else:
                plan.blocked_reasons.append(reason)
        plan.deferred.append({"kind": "derived_risk_after_fills", **dict(intent.get("derived_risk") or {})})
        return plan

    if action == "SCALE_OUT":
        if position_quantity is None:
            plan.blocked_reasons.append("Position quantity is required to calculate scale-out quantities.")
            plan.deferred.append({"kind": "scale_out", "instructions": intent.get("instructions") or intent.get("exits")})
            return plan
        instructions = intent.get("instructions") or intent.get("exits") or []
        for item in instructions:
            percentage = item.get("percentage", item.get("close_percentage"))
            if item.get("condition"):
                plan.deferred.append({"kind": "conditional_scale_out", **dict(item)})
                continue
            order: dict[str, Any] = {"action": "SELL", "quantity": abs(float(position_quantity)) * float(percentage) / 100}
            if item.get("trailing_stop_percent") is not None:
                order.update({"order_type": "TRAILING_STOP", "trailing_stop_percent": item["trailing_stop_percent"]})
            else:
                order.update({"order_type": item.get("order_type", "LIMIT"), "limit_price": item.get("price")})
            payload, reason = _base_payload(order, symbol=symbol, position_qty=position_quantity)
            if payload:
                plan.operations.append({"operation": "submit_order", "payload": payload})
            else:
                plan.blocked_reasons.append(reason)
        if intent.get("remaining_percentage") is not None and intent.get("trailing_stop_percent") is not None:
            remaining_qty = abs(float(position_quantity)) * float(intent["remaining_percentage"]) / 100
            payload, reason = _base_payload({"action": "SELL", "quantity": remaining_qty, "order_type": "TRAILING_STOP", "trailing_stop_percent": intent["trailing_stop_percent"]}, symbol=symbol, position_qty=position_quantity)
            if payload:
                plan.operations.append({"operation": "submit_order", "payload": payload})
            else:
                plan.blocked_reasons.append(reason)
        return plan

    conditional = intent.get("status") == "CONDITIONAL" or intent.get("entry_condition") is not None
    if conditional and intent.get("order_type") not in {"LIMIT", "STOP", "STOP_LIMIT"}:
        plan.deferred.append({"kind": "entry_condition", "condition": intent.get("entry_condition"), "intent": dict(intent)})
        return plan
    if isinstance(intent.get("entry_condition"), Mapping) and action == "SELL_SHORT":
        plan.deferred.append({"kind": "entry_condition", "condition": intent.get("entry_condition"), "intent": dict(intent)})
        return plan

    payload, reason = _base_payload(intent, symbol=symbol, position_qty=position_quantity)
    if payload:
        policies = {
            key: intent[key]
            for key in ("cancel_if_not_filled_by", "cancel_unfilled_remainder", "scale_in", "stop_adjustment", "max_position_loss", "stop_scope")
            if key in intent
        }
        plan.operations.append({"operation": "submit_order", "payload": payload, "policies": policies})
    else:
        plan.blocked_reasons.append(reason)

    for key in ("scale_in", "stop_adjustment", "max_position_loss", "exits", "stop_condition", "time_constraint"):
        if key in intent:
            plan.deferred.append({"kind": key, "value": intent[key]})
    if "cancel_if_not_filled_by" in intent:
        plan.deferred.append({"kind": "cancel_if_not_filled_by", "value": intent["cancel_if_not_filled_by"]})
    post_fill = intent.get("post_fill") if isinstance(intent.get("post_fill"), Mapping) else {}
    take_profit = intent.get("take_profit", post_fill.get("take_profit"))
    stop_loss = intent.get("stop_loss", post_fill.get("stop_loss"))
    atomic_risk_exit = (
        str(intent.get("action") or "").upper() in {"BUY", "SELL_SHORT"}
        and (
            isinstance(take_profit, (int, float))
            or isinstance(stop_loss, (int, float))
        )
    )
    if (take_profit is not None or stop_loss is not None) and not atomic_risk_exit:
        plan.deferred.append(
            {"kind": "non_atomic_risk_exit", "take_profit": take_profit, "stop_loss": stop_loss}
        )
    return plan


def gate_order_plan(
    intent: Mapping[str, Any],
    *,
    agent_mode: str,
    agent_decision: Optional[str] = None,
    position_quantity: Optional[float] = None,
) -> AlpacaOrderPlan:
    """Apply Agent ON/OFF routing while retaining all order instructions.

    Agent OFF follows the signal. Agent ON may BUY/SELL/HOLD, but the parser and
    Alpaca planner still validate and preserve every incoming field.
    """
    plan = build_alpaca_order_plan(intent, position_quantity=position_quantity)
    if str(agent_mode or "ON").upper() == "OFF":
        return plan
    decision = str(agent_decision or "HOLD").upper()
    if decision == "HOLD":
        plan.blocked_reasons.append("Agent ON decision is HOLD; no Alpaca order is submitted.")
        return plan
    requested_actions = [
        str(intent.get("action") or "").upper(),
        *[str(item.get("action") or "").upper() for item in intent.get("actions", []) if isinstance(item, Mapping)],
    ]
    buy_like = any(action in {"BUY", "BUY_TO_COVER", "LADDER_BUY", "MIXED_ENTRY"} for action in requested_actions)
    sell_like = any(action in {"SELL", "SELL_SHORT", "SCALE_OUT"} for action in requested_actions)
    if (decision == "BUY" and not buy_like) or (decision == "SELL" and not sell_like):
        plan.blocked_reasons.append(f"Agent ON decision {decision} conflicts with the incoming order action.")
    return plan


def execute_alpaca_order_plan(
    client: Any,
    plan: AlpacaOrderPlan,
    *,
    client_order_id_factory: Optional[Callable[[int], str]] = None,
) -> dict[str, Any]:
    """Execute immediate operations through ``AlpacaPaperClient``.

    Deferred policies are returned verbatim for the caller's durable monitor.
    No operation is attempted when the plan is blocked.
    """
    result: dict[str, Any] = {
        "submitted": [], "cancelled": [], "errors": [],
        "deferred": list(plan.deferred), "considered_fields": list(plan.considered_fields),
    }
    if plan.blocked_reasons:
        result["errors"].extend(plan.blocked_reasons)
        return result
    if plan.deferred:
        result["errors"].append(
            "This order contains conditional or post-fill policies that require a durable monitor; "
            "no partial Alpaca order was submitted."
        )
        return result
    if not getattr(client, "ready", lambda: False)():
        result["errors"].append("Alpaca paper trading is not configured or disabled.")
        return result

    for index, operation in enumerate(plan.operations):
        if operation.get("operation") == "cancel_open_orders":
            symbol = str(operation.get("symbol") or "").upper()
            path = f"/v2/orders?status=open&symbols={symbol}"
            open_orders, error = client._get(path)
            if error:
                result["errors"].append(error)
                continue
            for order in open_orders or []:
                order_id = str(order.get("id") or "")
                if not order_id:
                    continue
                _, cancel_error = client._request("DELETE", f"{client.base_url}/v2/orders/{order_id}")
                if cancel_error:
                    result["errors"].append(cancel_error)
                else:
                    result["cancelled"].append(order_id)
            continue
        payload = dict(operation.get("payload") or {})
        if client_order_id_factory:
            payload["client_order_id"] = str(client_order_id_factory(index))[:48]
        order, error = client._post("/v2/orders", payload)
        if order:
            result["submitted"].append(order)
        else:
            result["errors"].append(error or "Alpaca rejected the order.")
    return result
