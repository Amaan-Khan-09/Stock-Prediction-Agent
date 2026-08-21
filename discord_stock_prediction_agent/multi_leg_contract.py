"""Canonical semantic contract for multi-leg option signals.

Discord rendering, decision history, and broker preparation consume this
single representation. The parser does not invent a strategy quantity when a
message omits it; the deterministic default is one strategy unit.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .options_symbol import default_expiry_date, reduce_ratios_by_gcd


ACTION_MAP = {
    "BTO": "BUY_TO_OPEN", "BUY TO OPEN": "BUY_TO_OPEN", "BUY": "BUY_TO_OPEN",
    "LONG": "BUY_TO_OPEN", "STO": "SELL_TO_OPEN",
    "SELL TO OPEN": "SELL_TO_OPEN", "SHORT": "SELL_TO_OPEN",
    "STC": "SELL_TO_CLOSE", "SELL TO CLOSE": "SELL_TO_CLOSE",
    "BTC": "BUY_TO_CLOSE", "BUY TO CLOSE": "BUY_TO_CLOSE",
}
_ACTION = (
    r"BUY\s+TO\s+OPEN|SELL\s+TO\s+OPEN|BUY\s+TO\s+CLOSE|"
    r"SELL\s+TO\s+CLOSE|BTO|STO|BTC|STC|BUY|SELL|LONG|SHORT"
)
_LEG_RE = re.compile(
    rf"(?:(?P<action>{_ACTION})\s+)?(?:(?P<ratio>\d+)\s+)?"
    r"(?:(?P<root>[A-Z][A-Z.]{0,5})\s+)?(?P<strike>\d+(?:\.\d+)?)\s*"
    r"(?P<side>CALLS?|PUTS?|CE|PE|[CP])\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b")
_SYMBOL_IGNORE = {
    "CALL", "CALLS", "PUT", "PUTS", "C", "P", "CE", "PE", "FOR", "MAX",
    "OPEN", "CLOSE", "SHORT", "LONG", "SAME", "EXPIRY", "SPREAD", "IRON",
    "CONDOR", "CONDORS", "BUTTERFLY", "FLY", "FLIES", "ROLL", "FROM", "TO", "ONLY", "THE",
}
_FLY_SHORTHAND_RE = re.compile(
    r"\b(\d{2,6})(C|P)?\s*/\s*(\d{1,6})(C|P)?\s*/\s*(\d{1,6})(C|P)?\b",
    re.IGNORECASE,
)
_FLY_KEYWORD_RE = re.compile(r"\bFLYS?\b|\bFLIES\b|\bBUTTERFLY\b", re.IGNORECASE)


def _number(value: str) -> int | float:
    result = float(value)
    return int(result) if result.is_integer() else result


def _action(value: Optional[str], inherited: str = "") -> str:
    key = re.sub(r"\s+", " ", str(value or "").upper()).strip()
    return ACTION_MAP.get(key, inherited)


def _side(value: str) -> str:
    return "PUT" if str(value).upper() in {"P", "PUT", "PUTS", "PE"} else "CALL"


def _dates(text: str) -> list[str]:
    return [match.group(1).replace("-", "/") for match in _DATE_RE.finditer(text)]


def _root_from_text(text: str) -> str:
    for match in _LEG_RE.finditer(text):
        root = str(match.group("root") or "").upper()
        if root and root not in _SYMBOL_IGNORE:
            return root
    patterns = (
        r"\bBUY\s+\d+\s+([A-Z][A-Z.]{0,5})\s+SHARES?\b",
        r"\b(?:OPEN|ROLL|SHORT|LONG)\s+(?:\d+\s+)?([A-Z][A-Z.]{0,5})\b",
        r"\b(?:OF|ON)\s+([A-Z][A-Z.]{0,5})\b",
        r"\b([A-Z][A-Z.]{0,5})\s+(?:REVERSE\s+)?IRON\s+CONDOR\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).upper() not in _SYMBOL_IGNORE:
            return match.group(1).upper()
    return ""


def _explicit_quantity(text: str) -> int:
    match = re.search(r"\bQTY\s*(\d+)\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bOPEN\s+(\d+)\s+[A-Z][A-Z.]{0,5}\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def _stock_leg(text: str) -> Optional[dict[str, Any]]:
    match = re.search(r"\bBUY\s+(\d+)\s+[A-Z][A-Z.]{0,5}\s+SHARES?\b", text, re.IGNORECASE)
    if not match:
        return None
    return {"action": "BUY", "asset_type": "STOCK", "quantity": int(match.group(1))}


def _regular_legs(text: str, root: str) -> list[dict[str, Any]]:
    dates = _dates(text)
    default_expiry = dates[-1] if dates else ""
    matches = list(_LEG_RE.finditer(text))
    legs: list[dict[str, Any]] = []
    inherited_action = ""
    inherited_root = root
    for index, match in enumerate(matches):
        action = _action(match.group("action"), inherited_action)
        if action:
            inherited_action = action
        leg_root = str(match.group("root") or "").upper()
        if leg_root and leg_root not in _SYMBOL_IGNORE:
            inherited_root = leg_root
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        local_dates = _dates(text[match.end():end])
        expiry = local_dates[0] if local_dates else default_expiry
        if not action or not expiry:
            continue
        legs.append({
            "action": action,
            "option_type": _side(match.group("side")),
            "strike": _number(match.group("strike")),
            "expiration": expiry,
            "ratio": int(match.group("ratio") or 1),
            "_root": inherited_root,
        })
    return legs


def _compact_iron_butterfly(text: str) -> dict[str, Any]:
    """Parse compact alerts where one action applies to each call/put pair."""
    match = re.search(
        r"\bSELL\s+([A-Z][A-Z.]{0,5})\s+"
        r"(\d+(?:\.\d+)?)C\s+(\d+(?:\.\d+)?)P\s+BUY\s+"
        r"(\d+(?:\.\d+)?)C\s+(\d+(?:\.\d+)?)P\b",
        text,
        re.IGNORECASE,
    )
    dates = _dates(text)
    if not match or not dates:
        return {}
    root, short_call, short_put, long_call, long_put = match.groups()
    expiry = dates[-1]
    legs = [
        {"action": "SELL_TO_OPEN", "option_type": "CALL", "strike": _number(short_call), "expiration": expiry, "ratio": 1},
        {"action": "SELL_TO_OPEN", "option_type": "PUT", "strike": _number(short_put), "expiration": expiry, "ratio": 1},
        {"action": "BUY_TO_OPEN", "option_type": "CALL", "strike": _number(long_call), "expiration": expiry, "ratio": 1},
        {"action": "BUY_TO_OPEN", "option_type": "PUT", "strike": _number(long_put), "expiration": expiry, "ratio": 1},
    ]
    contract: dict[str, Any] = {
        "asset_type": "OPTION",
        "trade_structure": "MULTI_LEG",
        "strategy": "IRON_BUTTERFLY",
        "symbol": root.upper(),
        "quantity": _explicit_quantity(text),
        "legs": legs,
        "status": "VALID",
        **_price_fields(text),
    }
    return contract


def _expand_shorthand_strike(full: str, abbreviated: str) -> str:
    """Trading-room shorthand drops the shared leading digits on the later
    legs of a compact strike list (e.g. "7725/20/15" means 7725/7720/7715).
    Only expands when the token is actually shorter -- an already-full
    strike (e.g. "7550" in "7600/7550/7500") is used as-is.
    """
    if len(abbreviated) >= len(full):
        return abbreviated
    return full[: len(full) - len(abbreviated)] + abbreviated


def _compact_fly_legs(text: str, root: str) -> list[dict[str, Any]]:
    """Parse the common retail-room "FLY" shorthand for a plain (non-iron)
    butterfly: three slash-separated strikes, same option type throughout,
    where only one token typically carries the C/P side letter and later
    strikes are often abbreviated to their trailing digits (see
    _expand_shorthand_strike). _regular_legs() can't handle this -- it
    requires each leg to independently look like "<strike><side>", but
    only the last (or outer) tokens in "7725/20/15P" or "7780C/7790/7800C"
    do.

    Standard long-butterfly construction: the outer two strikes are the
    wings (ratio 1), the middle strike is the body (ratio 2), all bought
    to open together unless the message explicitly says this was sold to
    open. Returns [] rather than guessing wrong when the side letter is
    missing/ambiguous or the FLY/BUTTERFLY keyword isn't actually present
    -- a bare N/N/N (e.g. part of a date) must never be mistaken for this.
    """
    if not _FLY_KEYWORD_RE.search(text):
        return []
    match = _FLY_SHORTHAND_RE.search(text)
    if not match:
        return []
    raw_strikes = [match.group(1), match.group(3), match.group(5)]
    side_tokens = [match.group(2), match.group(4), match.group(6)]
    sides = {token.upper() for token in side_tokens if token}
    if len(sides) != 1:
        return []
    side = "PUT" if next(iter(sides)).startswith("P") else "CALL"
    full = raw_strikes[0]
    try:
        strikes = [float(full)] + [
            float(_expand_shorthand_strike(full, token)) for token in raw_strikes[1:]
        ]
    except ValueError:
        return []
    if len(set(strikes)) != 3:
        return []
    sold_to_open = bool(re.search(r"\bSOLD\b|\bSELL\b|\bSTO\b", text, re.IGNORECASE)) and not bool(
        re.search(r"\bBOUGHT\b|\bBUY\b|\bBTO\b|\bADD(?:ING|ED)?\b", text, re.IGNORECASE)
    )
    wing_action = "SELL_TO_OPEN" if sold_to_open else "BUY_TO_OPEN"
    body_action = "BUY_TO_OPEN" if sold_to_open else "SELL_TO_OPEN"
    expiry = default_expiry_date("0dte").strftime("%m/%d/%Y")
    root_upper = (root or "").upper()
    return [
        {"action": wing_action, "option_type": side, "strike": _number(str(strikes[0])), "expiration": expiry, "ratio": 1, "_root": root_upper},
        {"action": body_action, "option_type": side, "strike": _number(str(strikes[1])), "expiration": expiry, "ratio": 2, "_root": root_upper},
        {"action": wing_action, "option_type": side, "strike": _number(str(strikes[2])), "expiration": expiry, "ratio": 1, "_root": root_upper},
    ]


def _price_fields(text: str) -> dict[str, Any]:
    effect = re.search(r"\b(DEBIT|CREDIT)\b", text, re.IGNORECASE)
    if not effect:
        return {}
    matches = list(re.finditer(
        r"(?:@|\bFOR\b|\bMAX\b)\s*\$?(\d+(?:\.\d+)?)",
        text[:effect.start()], re.IGNORECASE,
    ))
    if not matches:
        return {}
    return {
        "order_type": "NET_LIMIT",
        "net_price_type": effect.group(1).upper(),
        "net_price": _number(matches[-1].group(1)),
    }


def _risk_management(text: str) -> dict[str, Any]:
    tp = re.search(r"\bTP\s*\$?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    sl = re.search(r"\bSL\s*\$?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    maximum = re.search(r"MAX\s+LOSS\s*(?:₹|INR\s*)?\s*([\d,]+)", text, re.IGNORECASE)
    return {
        "take_profit": _number(tp.group(1)) if tp else None,
        "stop_loss": _number(sl.group(1)) if sl else None,
        "close_before_expiration": bool(re.search(r"CLOSE\s+BEFORE\s+EXPIR", text, re.IGNORECASE)),
        "maximum_loss_inr": int(maximum.group(1).replace(",", "")) if maximum else None,
    }


def _explicit_strategy(text: str) -> Optional[str]:
    upper = text.upper()
    labels = (
        ("REVERSE IRON CONDOR", "REVERSE_IRON_CONDOR"),
        ("IRON CONDOR", "IRON_CONDOR"), ("IRON BUTTERFLY", "IRON_BUTTERFLY"),
        ("BULL CALL SPREAD", "BULL_CALL_SPREAD"), ("BEAR CALL SPREAD", "BEAR_CALL_SPREAD"),
        ("BEAR PUT SPREAD", "BEAR_PUT_SPREAD"), ("BULL PUT SPREAD", "BULL_PUT_SPREAD"),
        ("BROKEN WING BUTTERFLY", "BROKEN_WING_BUTTERFLY"),
        ("DOUBLE DIAGONAL", "DOUBLE_DIAGONAL"), ("JADE LIZARD", "JADE_LIZARD"),
        ("BOX SPREAD", "BOX_SPREAD"), ("RISK REVERSAL", "RISK_REVERSAL"),
        ("COVERED CALL", "COVERED_CALL_COMBO"), ("COLLAR", "COLLAR"),
        ("CALL BUTTERFLY", "CALL_BUTTERFLY"), ("PUT BUTTERFLY", "PUT_BUTTERFLY"),
        ("CALL BACKSPREAD", "CALL_BACKSPREAD"), ("PUT BACKSPREAD", "PUT_BACKSPREAD"),
        ("CALL RATIO SPREAD", "CALL_RATIO_SPREAD"), ("PUT RATIO SPREAD", "PUT_RATIO_SPREAD"),
        ("CALENDAR", "CALENDAR_SPREAD"), ("DIAGONAL", "DIAGONAL_SPREAD"),
        ("LONG STRADDLE", "LONG_STRADDLE"), ("LONG STRANGLE", "LONG_STRANGLE"),
        ("SHORT STRADDLE", "SHORT_STRADDLE"), ("SHORT STRANGLE", "SHORT_STRANGLE"),
    )
    for phrase, label in labels:
        if phrase in upper:
            return label
    return None


def _infer_strategy(text: str, legs: list[dict[str, Any]], stock_leg: Optional[dict[str, Any]]) -> str:
    explicit = _explicit_strategy(text)
    if explicit and explicit != "DIAGONAL_SPREAD":
        return explicit
    if stock_leg:
        if len(legs) == 2:
            return "COLLAR"
        if len(legs) == 1 and legs[0]["option_type"] == "PUT" and legs[0]["action"] == "BUY_TO_OPEN":
            return "PROTECTIVE_PUT"
        return "COVERED_CALL_COMBO"
    if len(legs) == 4:
        expiries = {leg["expiration"] for leg in legs}
        strikes = [leg["strike"] for leg in legs]
        actions = [leg["action"] for leg in legs]
        sides = [leg["option_type"] for leg in legs]
        if len(expiries) > 1:
            return "DOUBLE_DIAGONAL"
        if len(set(strikes)) == 3:
            return "IRON_BUTTERFLY"
        if sides == ["CALL", "CALL", "PUT", "PUT"] and actions == ["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_OPEN", "SELL_TO_OPEN"]:
            return "BOX_SPREAD"
        if actions == ["BUY_TO_OPEN", "SELL_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_OPEN"]:
            return "IRON_CONDOR"
        if actions == ["SELL_TO_OPEN", "BUY_TO_OPEN", "BUY_TO_OPEN", "SELL_TO_OPEN"]:
            return "REVERSE_IRON_CONDOR"
        return "CUSTOM_FOUR_LEG"
    if len(legs) == 3:
        sides = {leg["option_type"] for leg in legs}
        ratios = [leg["ratio"] for leg in legs]
        if len(sides) == 1 and ratios == [1, 2, 1]:
            strikes = [float(leg["strike"]) for leg in legs]
            equal_width = abs((strikes[1] - strikes[0]) - (strikes[2] - strikes[1])) < 1e-9
            if not equal_width:
                return "BROKEN_WING_BUTTERFLY"
            return "CALL_BUTTERFLY" if legs[0]["option_type"] == "CALL" else "PUT_BUTTERFLY"
        return "JADE_LIZARD"
    if len(legs) != 2:
        return "MULTI_LEG"
    first, second = legs
    if first["expiration"] != second["expiration"]:
        if first["strike"] == second["strike"]:
            return "CALENDAR_SPREAD"
        return "DIAGONAL_CALL_SPREAD" if first["option_type"] == "CALL" else "DIAGONAL_PUT_SPREAD"
    if first["ratio"] != second["ratio"]:
        if first["action"] == "BUY_TO_OPEN":
            return "CALL_RATIO_SPREAD" if first["option_type"] == "CALL" else "PUT_RATIO_SPREAD"
        return "CALL_BACKSPREAD" if first["option_type"] == "CALL" else "PUT_BACKSPREAD"
    if first["option_type"] != second["option_type"]:
        both_long = first["action"] == second["action"] == "BUY_TO_OPEN"
        both_short = first["action"] == second["action"] == "SELL_TO_OPEN"
        if both_long:
            return "LONG_STRADDLE" if first["strike"] == second["strike"] else "LONG_STRANGLE"
        if both_short:
            return "SHORT_STRADDLE" if first["strike"] == second["strike"] else "SHORT_STRANGLE"
        if first["strike"] == second["strike"]:
            return "SYNTHETIC_LONG" if first["option_type"] == "CALL" and first["action"] == "BUY_TO_OPEN" else "SYNTHETIC_SHORT"
        return "RISK_REVERSAL"
    if first["option_type"] == "CALL":
        return "BULL_CALL_SPREAD" if first["action"] == "BUY_TO_OPEN" else "BEAR_CALL_SPREAD"
    return "BEAR_PUT_SPREAD" if first["action"] == "BUY_TO_OPEN" else "BULL_PUT_SPREAD"


def _complex_contract(text: str) -> Optional[dict[str, Any]]:
    upper = text.upper()
    dates = _dates(text)
    root = _root_from_text(text)
    quantity = _explicit_quantity(text)
    strategy = ""
    legs: list[dict[str, Any]] = []
    advanced: dict[str, Any] = {}

    def leg(action: str, side: str, strike: str | float, expiry: str, ratio: int = 1) -> dict[str, Any]:
        return {"action": action, "option_type": side, "strike": _number(str(strike)), "expiration": expiry, "ratio": ratio}

    if "IRON CONDORS" in upper and "ADJUST IF SHORT DELTA" in upper:
        match = re.search(r"([A-Z.]+)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+IRON", upper)
        if not match or not dates:
            return None
        root = match.group(1); a, b, c, d = match.groups()[1:]; strategy = "IRON_CONDOR_MANAGEMENT"
        legs = [leg("BUY_TO_OPEN", "PUT", a, dates[0]), leg("SELL_TO_OPEN", "PUT", b, dates[0]), leg("SELL_TO_OPEN", "CALL", c, dates[0]), leg("BUY_TO_OPEN", "CALL", d, dates[0])]
        pct = re.search(r"TAKE\s+(\d+(?:\.\d+)?)%\s+PROFIT", upper)
        delta = re.search(r"SHORT\s+DELTA\s+EXCEEDS\s+(\d+(?:\.\d+)?)", upper)
        advanced = {"take_profit_percent": _number(pct.group(1)) if pct else None, "adjustment_trigger": {"metric": "SHORT_LEG_DELTA", "operator": ">", "value": _number(delta.group(1)) if delta else None}}
    elif upper.startswith("ROLL ") and "CALL SPREAD" in upper and len(dates) >= 2:
        match = re.search(r"ROLL\s+([A-Z.]+)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+CALL\s+SPREAD.*?TO\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", upper)
        if not match:
            return None
        root = match.group(1); a, b, c, d = match.groups()[1:]; strategy = "ROLLING_VERTICAL"
        legs = [leg("BUY_TO_CLOSE", "CALL", a, dates[0]), leg("SELL_TO_CLOSE", "CALL", b, dates[0]), leg("BUY_TO_OPEN", "CALL", c, dates[1]), leg("SELL_TO_OPEN", "CALL", d, dates[1])]
        price = re.search(r"MAX\s+(\d+(?:\.\d+)?)\s+DEBIT", upper)
        advanced = {"maximum_roll_debit": _number(price.group(1)) if price else None}
    elif "ROLL PUT SIDE" in upper and len(dates) >= 2:
        match = re.search(r"OF\s+([A-Z.]+).*?FROM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?).*?TO\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", upper)
        if not match:
            return None
        root = match.group(1); a, b, c, d = match.groups()[1:]; strategy = "ROLLING_IRON_CONDOR"
        legs = [leg("BUY_TO_CLOSE", "PUT", b, dates[0]), leg("SELL_TO_CLOSE", "PUT", a, dates[0]), leg("SELL_TO_OPEN", "PUT", d, dates[1]), leg("BUY_TO_OPEN", "PUT", c, dates[1])]
        price = re.search(r"AT\s+LEAST\s+(\d+(?:\.\d+)?)\s+CREDIT", upper)
        advanced = {"minimum_roll_credit": _number(price.group(1)) if price else None}
    elif "BTC ONLY THE SHORT" in upper:
        legs = _regular_legs(text, root)[:1]
        if not legs:
            return None
        legs[0]["action"] = "BUY_TO_CLOSE"
        strategy = "PARTIAL_LEG_CLOSE"
        advanced = {"scope": "SINGLE_LEG_ONLY", "preserve_other_legs": True}
    elif "CALL BUTTERFLY" in upper and "ONLY IF" in upper and dates:
        match = re.search(r"OPEN\s+([A-Z.]+)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+CALL", upper)
        if not match:
            return None
        root = match.group(1); a, b, c = match.groups()[1:]; strategy = "CONDITIONAL_BUTTERFLY"
        legs = [leg("BUY_TO_OPEN", "CALL", a, dates[0]), leg("SELL_TO_OPEN", "CALL", b, dates[0], 2), leg("BUY_TO_OPEN", "CALL", c, dates[0])]
        condition = re.search(r"CLOSES\s+ABOVE\s+(\d+(?:\.\d+)?)", upper)
        price = re.search(r"MAX\s+(\d+(?:\.\d+)?)\s+DEBIT", upper)
        advanced = {"entry_condition": {"type": "DAILY_CLOSE_ABOVE", "value": _number(condition.group(1)) if condition else None}, "maximum_debit": _number(price.group(1)) if price else None}
    elif "NEXT OPEN AFTER EARNINGS" in upper:
        strategy = "EARNINGS_STRANGLE"; legs = _regular_legs(text, root)
        advanced = {"event_exit": "NEXT_MARKET_OPEN_AFTER_EARNINGS"}
    elif "HEDGE SHARES TO KEEP POSITION DELTA NEAR ZERO" in upper and dates:
        match = re.search(r"OPEN\s+([A-Z.]+)\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+CALL", upper)
        if not match:
            return None
        root = match.group(1); a, b = match.groups()[1:]; strategy = "DELTA_HEDGED_SPREAD"
        legs = [leg("BUY_TO_OPEN", "CALL", a, dates[0]), leg("SELL_TO_OPEN", "CALL", b, dates[0]), {"action": "HEDGE", "asset_type": "STOCK", "delta_target": 0}]
        advanced = {"delta_hedge_target": 0}
    elif upper.startswith("CLOSE SHORT") and "KEEP LONG" in upper and len(dates) >= 2:
        match = re.search(r"CLOSE\s+SHORT\s+([A-Z.]+)\s+(\d+(?:\.\d+)?)C.*?SELL\s+(\d+(?:\.\d+)?)C.*?KEEP\s+LONG\s+(\d+(?:\.\d+)?)C", upper)
        if not match:
            return None
        root = match.group(1); a, b, c = match.groups()[1:]; strategy = "ADJUSTED_CALENDAR"
        legs = [leg("BUY_TO_CLOSE", "CALL", a, dates[0]), leg("SELL_TO_OPEN", "CALL", b, dates[0]), leg("BUY_TO_OPEN", "CALL", c, dates[1])]
        price = re.search(r"MAX\s+(\d+(?:\.\d+)?)\s+DEBIT", upper)
        advanced = {"maximum_adjustment_debit": _number(price.group(1)) if price else None}
    elif "REVERSE IRON CONDOR" in upper and dates:
        match = re.search(r"OPEN\s+([A-Z.]+)\s+REVERSE\s+IRON\s+CONDOR\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", upper)
        if not match:
            return None
        root = match.group(1); a, b, c, d = match.groups()[1:]; strategy = "REVERSE_IRON_CONDOR"
        legs = [leg("SELL_TO_OPEN", "PUT", a, dates[0]), leg("BUY_TO_OPEN", "PUT", b, dates[0]), leg("BUY_TO_OPEN", "CALL", c, dates[0]), leg("SELL_TO_OPEN", "CALL", d, dates[0])]
        price = re.search(r"MAX\s+(\d+(?:\.\d+)?)\s+DEBIT", upper); pct = re.search(r"TP\s+(\d+(?:\.\d+)?)%", upper)
        advanced = {"maximum_debit": _number(price.group(1)) if price else None, "take_profit_percent": _number(pct.group(1)) if pct else None}
    elif "CLOSE ALL IF NET LOSS REACHES" in upper:
        strategy = "CUSTOM_FOUR_LEG"; legs = _regular_legs(text, root)
        loss = re.search(r"(?:₹|INR\s*)\s*([\d,]+)", text, re.IGNORECASE)
        advanced = {"maximum_loss_inr": int(loss.group(1).replace(",", "")) if loss else None, "emergency_action": "CLOSE_ALL_LEGS"}
    else:
        return None

    for item in legs:
        item.pop("_root", None)
    return {
        "asset_type": "OPTION", "trade_structure": "MULTI_LEG",
        "strategy": strategy, "symbol": root, "quantity": quantity,
        "legs": legs, "status": "CONDITIONAL_OR_POSITION_DEPENDENT",
        "advanced_instructions": advanced,
    }


def build_multi_leg_contract(text: str, option: object = None) -> dict[str, Any]:
    """Return the canonical contract, or an empty dict for non-mleg text."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return {}
    compact_iron_butterfly = _compact_iron_butterfly(raw)
    if compact_iron_butterfly:
        return compact_iron_butterfly
    complex_contract = _complex_contract(raw)
    if complex_contract:
        return complex_contract
    # _root_from_text() needs a root word directly adjacent to a strike+side
    # match; the compact FLY shorthand's root (if any) usually isn't
    # adjacent to the one token that carries a side letter, so fall back to
    # the root options_parser.py's own (more thorough) extractor already
    # resolved on the caller's ParsedOptionSignal.
    root = _root_from_text(raw) or str(getattr(option, "root", "") or "").upper()
    stock_leg = _stock_leg(raw)
    legs = _regular_legs(raw, root)
    if len(legs) < 2 and not stock_leg:
        legs = _compact_fly_legs(raw, root) or legs
    if len(legs) < 2 and not stock_leg:
        return {}
    if not root or any(not leg.get("_root") for leg in legs):
        return {}
    quantity = _explicit_quantity(raw)
    has_strategy_quantity = bool(re.search(r"\bQTY\s*\d+\b|\bOPEN\s+\d+\s+[A-Z]", raw, re.IGNORECASE))
    if len(legs) >= 2 and not has_strategy_quantity:
        reduced_ratios, common_ratio = reduce_ratios_by_gcd(
            [parsed_leg.get("ratio") or 1 for parsed_leg in legs]
        )
        if common_ratio > 1:
            quantity = common_ratio
            for parsed_leg, reduced_ratio in zip(legs, reduced_ratios):
                parsed_leg["ratio"] = reduced_ratio
    clean_legs: list[dict[str, Any]] = []
    if stock_leg:
        clean_legs.append(stock_leg)
    clean_legs.extend({key: value for key, value in leg.items() if key != "_root"} for leg in legs)
    contract: dict[str, Any] = {
        "asset_type": "OPTION", "trade_structure": "MULTI_LEG",
        "strategy": _infer_strategy(raw, legs, stock_leg), "symbol": root,
        "quantity": quantity, "legs": clean_legs,
        **_price_fields(raw),
    }
    advanced_risk_strategies = {
        "BROKEN_WING_BUTTERFLY", "CALL_RATIO_SPREAD", "PUT_RATIO_SPREAD",
        "CALL_BACKSPREAD", "PUT_BACKSPREAD", "JADE_LIZARD", "DOUBLE_DIAGONAL",
        "BOX_SPREAD", "SYNTHETIC_LONG", "SYNTHETIC_SHORT",
    }
    if contract["strategy"] in advanced_risk_strategies or re.search(
        r"\b(?:TP|SL|CLOSE\s+BEFORE\s+EXPIR|MAX\s+LOSS)\b", raw, re.IGNORECASE
    ):
        contract["risk_management"] = _risk_management(raw)
    contract["status"] = "VALID"
    return contract
