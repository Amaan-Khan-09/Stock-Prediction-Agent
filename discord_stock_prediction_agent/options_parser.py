"""Options-signal detection and parsing, plus the message-kind router.

This module sits IN FRONT of signal_parser.py -- it never modifies it.
classify_and_parse() decides whether a Discord message is an options signal,
a plain equity signal (delegated unchanged to signal_parser.parse_signal),
non-actionable market commentary (NO_TRADE), or empty/unparseable (INVALID).
"""
from __future__ import annotations

import re
from math import gcd
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Optional

from .signal_parser import PHRASE_ALIASES, SYMBOL_ALIASES, parse_signal, ParsedSignal
from .options_symbol import default_expiry_date
from .signal_normalizer import normalize_signal_input
from .symbol_directory import is_symbol_like, resolve_cached_symbol

# Cash-settled index roots that signal_parser.py's alias tables don't cover
# (it is an equity/ETF signal parser and was intentionally left untouched).
_INDEX_ROOTS = frozenset({"SPX", "NDX", "RUT", "DJX", "VIX", "XSP"})

_STRIKE_SIDE_RE = re.compile(
    r"\$?(\d{1,6}(?:\.\d{1,2})?)\s*(CALLS?|PUTS?|CE|PE|[CP])\b", re.IGNORECASE
)
_STRIKE_SHORTHAND_RE = re.compile(
    r"\b(\d{1,6}(?:\.\d{1,2})?)\s*S\b",
    re.IGNORECASE,
)
_SIDE_STRIKE_RE = re.compile(
    r"\b(CALLS?|PUTS?|CE|PE)\b(?:\s+OPTION)?(?:\s+(?:OF|ON|FOR)\s+[A-Z.]{1,6})?"
    r".{0,60}?\b(?:STRIKE(?:\s+RATE|\s+PRICE)?|STRIKE|AT|@)?\s*\$?(\d{1,6}(?:\.\d{1,2})?)\b",
    re.IGNORECASE,
)
_PREMIUM_RE = re.compile(r"(?:\b(?:AT|FOR|PREMIUM|ENTRY|PRICE|LIMIT)\b|@)\s*\$?(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_STOP_LOSS_RE = re.compile(
    r"\b(?:SL|BY|STOP(?:\s+LOSS)?|STOPLOSS)\b\s*(?:BELOW|UNDER|AT|@|[:=\-])?\s*\$?(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(
    r"\b(?:TP|PT|TARGETS?|TGT|PROFIT\s+TARGET|TAKE\s+PROFIT)\b\s*(?:AT|@|[:=\-])?\s*\$?(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_RISK_REWARD_RE = re.compile(
    r"\b(?:R\s*/\s*R|RR|RISK\s*[/:-]?\s*REWARD)\b\s*(?:RATIO)?\s*[:=]?\s*(?:1\s*[:/]\s*)?(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_DELTA_RE = re.compile(
    r"\b(?:DELTA\s*[:=@-]?\s*(\d{1,2}(?:\.\d+)?)|(\d{1,2}(?:\.\d+)?)\s*DELTA)\b",
    re.IGNORECASE,
)
_SIDE_ONLY_RE = re.compile(r"\b(CALLS?|PUTS?|CE|PE)\b", re.IGNORECASE)
_MULTI_LEG_RE = re.compile(
    r"(?:(BUY\s+TO\s+OPEN|SELL\s+TO\s+OPEN|BUY\s+TO\s+CLOSE|SELL\s+TO\s+CLOSE|"
    r"BTO|STO|BTC|STC|BUY|SELL|LONG|SHORT)\s+)?"
    r"(?:(\d+)\s*(?:-?LOTS?|CONTRACTS?)?\s+)?"
    r"(?:([A-Z][A-Z.]{0,5})\s+)?"
    r"\$?(\d{1,6}(?:\.\d{1,2})?)\s*(CALLS?|PUTS?|CE|PE|[CP])\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DAY_MONTH_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\b",
    re.IGNORECASE,
)
_SAME_DAY_EXPIRY_RE = re.compile(r"\b(?:SAME\s+DAY|0DTE|ZERO\s+DTE|EOD|TODAY)\b", re.IGNORECASE)
_QTY_RE = [
    re.compile(r"\bqty\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bquantity\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bcontracts?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*-?\s*(?:contracts?|lots?)\b", re.IGNORECASE),
    re.compile(r"\b(?:BTO|STO|BTC|STC|BUY|SELL|LONG|SHORT)\s+(\d+(?:\.\d+)?)\s+(?:OF\s+\d+\s+)?(?:[A-Z.]+\s+)?\d", re.IGNORECASE),
    re.compile(r"\bx\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE),
]

# Longest-key-first lookup so "iron condor" isn't shadowed by "spread".
_STRUCTURE_KEYWORDS = [
    ("iron condor", "iron_condor"),
    ("iron condors", "iron_condor"),
    ("iron butterfly", "iron_butterfly"),
    ("reverse iron condor", "reverse_iron_condor"),
    ("diagonal spread", "diagonal_spread"),
    ("calendar spread", "calendar_spread"),
    ("ratio backspread", "ratio_backspread"),
    ("ratio spread", "ratio_spread"),
    ("butterfly spread", "butterfly_spread"),
    ("protective put", "protective_put"),
    ("downside protection", "protective_put"),
    ("cash secured put", "cash_secured_put"),
    ("cash secured", "cash_secured_put"),
    ("cash-secured put", "cash_secured_put"),
    ("covered call", "covered_call"),
    ("bull call spread", "spread"),
    ("bear put spread", "spread"),
    ("credit spread", "spread"),
    ("debit spread", "spread"),
    ("vertical spread", "spread"),
    ("call spread", "spread"),
    ("put spread", "spread"),
    ("straddle", "straddle"),
    ("strangle", "strangle"),
    ("collar", "collar"),
    ("rolling", "roll"),
    ("roll", "roll"),
    ("vertical", "spread"),
    ("spread", "spread"),
    ("covered", "covered_call"),
]

_PAST_TENSE_VERBS = {"BOUGHT", "SOLD", "CLOSED", "FILLED", "EXITED"}
_NEW_ORDER_VERBS = {
    "BUY": "CALL_SIDE_HINT", "PURCHASE": "CALL_SIDE_HINT", "LONG": "CALL_SIDE_HINT",
    "GRABBED": "CALL_SIDE_HINT", "GRAB": "CALL_SIDE_HINT", "STARTER": "CALL_SIDE_HINT",
    "OPENING": "CALL_SIDE_HINT", "OPENED": "CALL_SIDE_HINT", "ENTERED": "CALL_SIDE_HINT",
    "ADD": "CALL_SIDE_HINT", "ADDING": "CALL_SIDE_HINT", "SCALE": "CALL_SIDE_HINT",
    "SCALING": "CALL_SIDE_HINT",
    "SELL": "PUT_SIDE_HINT", "SHORT": "PUT_SIDE_HINT",
    "BTO": None, "STO": None, "BTC": None, "STC": None,
}
_OPEN_LONG_TOKENS = {"BUY", "BTO", "LONG", "PURCHASE", "GRABBED", "GRAB", "STARTER", "OPENING", "OPENED", "ENTERED", "ADD", "ADDING", "SCALE", "SCALING"}
_OPEN_SHORT_TOKENS = {"STO"}
_CLOSE_LONG_TOKENS = {"STC", "TRIM"}
_CLOSE_SHORT_TOKENS = {"BTC"}
_MANAGEMENT_RE = re.compile(
    r"\b(?:MOVE\s+STOP|BREAKEVEN|TAKE\s+\d+%|CLOSE\s+REMAINING|HOLD\s+OVERNIGHT|"
    r"EXIT\s+BEFORE|TRAIL\s+STOP|TRIM\s+\d+%|SELL\s+TO\s+CLOSE|SELL\s+TO\s+OPEN|"
    r"BUY\s+TO\s+CLOSE|TAKE\s+\d+%\s+PROFITS)",
    re.IGNORECASE,
)

_ROOT_IGNORE_WORDS = {
    "AT", "EOD", "LOTTO", "STRANGLE", "STRADDLE", "COMPLETE", "SPREAD", "VERTICAL",
    "CALL", "CALLS", "PUT", "PUTS", "BTO", "STO", "BTC", "STC", "BOUGHT", "SOLD",
    "BUY", "SELL", "SHORT", "LONG", "PURCHASE", "OPENED", "CLOSED", "FILLED",
    "EXITED", "ENTERED", "IRON", "CONDOR", "CONDORS", "QTY", "QUANTITY",
    "CONTRACT", "CONTRACTS", "LOT", "LOTS", "FOR", "IS", "THE", "A", "AN",
    "WE", "NEED", "OPTION", "OPTIONS", "STRIKE", "RATE", "EXPIRY", "EXPIRES",
    "EXPIRATION", "DATE", "SAME", "DAY", "PREMIUM", "PRICE", "LIMIT", "MARKET",
    "GRABBED", "GRAB", "STARTER", "OPENING", "ADD", "ADDING", "SCALE", "SCALING",
    "INTO", "TO", "CLOSE", "OPEN", "DEBIT", "CREDIT", "CASH", "SECURED", "COVERED",
    "COLLAR", "ROLL", "ROLLING", "TRIM", "POSITION", "MANAGEMENT", "SIGNALS",
    "EARNINGS", "PROFITS", "RUNNERS", "VWAP", "DELTA", "ON", "OF", "ABOVE", "BELOW",
    "TP", "PT", "TGT", "SL", "BY", "RR",
}


@dataclass(frozen=True)
class ParsedOptionLeg:
    root: str
    strike: float
    side: str
    order_action: str
    ratio_qty: int = 1
    expiry_date: Optional[str] = None


@dataclass(frozen=True)
class ParsedOptionSignal:
    valid: bool
    root: str = ""
    strike: Optional[float] = None
    side: Optional[str] = None  # "CALL" | "PUT"
    delta_target: Optional[float] = None
    fill_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    target_prices: tuple[float, ...] = field(default_factory=tuple)
    risk_reward: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    close_percent: Optional[float] = None
    add_quantity: Optional[float] = None
    add_trigger_premium: Optional[float] = None
    underlying_trigger_direction: Optional[str] = None
    underlying_trigger_price: Optional[float] = None
    exit_before_market_close: bool = False
    contains_equity_leg: bool = False
    expiry_date: Optional[str] = None  # YYYY-MM-DD, resolved
    expiry_mode: str = "0dte"  # "explicit" | "0dte"
    structure: Optional[str] = None
    is_multi_leg: bool = False
    legs: tuple[ParsedOptionLeg, ...] = field(default_factory=tuple)
    price_effect: Optional[str] = None  # "debit" | "credit"
    tense: str = "unknown"  # "past" | "new_order" | "unknown"
    order_action: str = "unknown"  # open_long | open_short | close_long | close_short | manage
    quantity: float = 1.0
    raw_text: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ParsedMessage:
    kind: str  # "EQUITY" | "OPTION" | "NO_TRADE" | "INVALID"
    equity: Optional[ParsedSignal] = None
    option: Optional[ParsedOptionSignal] = None
    raw_text: str = ""
    reason: str = ""


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9.$@/ ]+", " ", text or "")).strip()


def _extract_root(text: str) -> str:
    upper = text.upper()
    normalized = _normalized(text).upper()
    structured_leg = _MULTI_LEG_RE.search(text)
    if structured_leg and structured_leg.group(3):
        structured_root_raw = str(structured_leg.group(3))
        if structured_root_raw == structured_root_raw.upper():
            structured_root = structured_root_raw.upper()
            return SYMBOL_ALIASES.get(structured_root, structured_root)
    for phrase, symbol in sorted(PHRASE_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return symbol
    tokens = re.findall(r"[A-Za-z][A-Za-z]{0,5}", upper)
    for token in tokens:
        if token in _INDEX_ROOTS:
            return token
    candidates = []
    for token in tokens:
        if token in _ROOT_IGNORE_WORDS:
            continue
        if token.lower()[:3] in _MONTH_NAMES:
            continue
        if token in SYMBOL_ALIASES:
            return SYMBOL_ALIASES[token]
        candidates.append(token)
    for token in candidates:
        cached = resolve_cached_symbol(token)
        if cached:
            return cached
        if is_symbol_like(token, max_len=5):
            return token
    return ""

def _side_from_action_hint(text: str) -> Optional[str]:
    action = _extract_order_action(text)
    upper = text.upper()
    tokens = set(re.findall(r"[A-Z]+", upper))
    if action in {"open_long", "close_short"} or tokens & {"BUY", "BTO", "LONG"}:
        return "CALL"
    if action == "open_short" or tokens & {"SELL", "SHORT"}:
        return "PUT"
    return None


def _extract_strike_side(text: str) -> tuple[Optional[float], Optional[str]]:
    structured_leg = _MULTI_LEG_RE.search(text)
    if structured_leg:
        strike_raw = structured_leg.group(4)
        side_token = structured_leg.group(5).upper()
        side = "CALL" if side_token.startswith("C") else "PUT"
    else:
        match = _STRIKE_SIDE_RE.search(text)
        if match:
            strike_raw = match.group(1)
            side_token = match.group(2).upper()
            side = "CALL" if side_token.startswith("C") else "PUT"
        else:
            match = _SIDE_STRIKE_RE.search(text)
            if match:
                side_token = match.group(1).upper()
                strike_raw = match.group(2)
                side = "CALL" if side_token.startswith("C") else "PUT"
            else:
                match = _STRIKE_SHORTHAND_RE.search(text)
                if not match:
                    return None, None
                strike_raw = match.group(1)
                side = _side_from_action_hint(text)
                if side is None:
                    return None, None
    try:
        strike = float(strike_raw)
    except ValueError:
        return None, None
    return strike, side


def _extract_side_only(text: str) -> Optional[str]:
    match = _SIDE_ONLY_RE.search(text)
    if not match:
        return None
    token = match.group(1).upper()
    return "CALL" if token.startswith("C") else "PUT"


def _extract_delta_target(text: str) -> Optional[float]:
    match = _DELTA_RE.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _extract_fill_price(text: str) -> Optional[float]:
    for match in _PREMIUM_RE.finditer(text):
        prefix = text[max(0, match.start() - 18):match.start()].upper()
        if re.search(r"\b(?:SL|STOP|STOPLOSS|TARGET|TGT|TP|PT|PROFIT|TRIM)\b", prefix):
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None

def _extract_optional_price(pattern: re.Pattern, text: str) -> Optional[float]:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_risk_reward(text: str) -> Optional[float]:
    match = _RISK_REWARD_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
        return value if value > 0 else None
    except ValueError:
        return None


def _derive_missing_exit_from_rr(fill_price: Optional[float], stop_loss: Optional[float], target_price: Optional[float], risk_reward: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if not fill_price or not risk_reward:
        return stop_loss, target_price
    if stop_loss is not None and target_price is None and stop_loss < fill_price:
        risk = fill_price - stop_loss
        target_price = round(fill_price + (risk * risk_reward), 6)
    elif target_price is not None and stop_loss is None and target_price > fill_price:
        reward = target_price - fill_price
        risk = reward / risk_reward if risk_reward > 0 else 0
        if risk > 0:
            stop_loss = round(fill_price - risk, 6)
    return stop_loss, target_price


def _resolve_year(month: int, day: int, today: date) -> int:
    year = today.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        return year
    return year if candidate >= today else year + 1


def _extract_explicit_expiry(text: str, today: Optional[date] = None) -> Optional[str]:
    today = today or date.today()

    iso = _ISO_DATE_RE.search(text)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    m = _NUMERIC_DATE_RE.search(text)
    if m:
        month, day, year_raw = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= month <= 12 and 1 <= day <= 31:
            if year_raw:
                year = int(year_raw)
                if year < 100:
                    year += 2000
            else:
                year = _resolve_year(month, day, today)
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None

    m2 = _MONTH_DATE_RE.search(text)
    if m2:
        month = _MONTH_NAMES.get(m2.group(1).lower())
        day = int(m2.group(2))
        if month and 1 <= day <= 31:
            year = _resolve_year(month, day, today)
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None

    m3 = _DAY_MONTH_DATE_RE.search(text)
    if m3:
        day = int(m3.group(1))
        month = _MONTH_NAMES.get(m3.group(2).lower())
        if month and 1 <= day <= 31:
            year = _resolve_year(month, day, today)
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                return None

    return None

def _extract_structure(text: str) -> tuple[Optional[str], bool]:
    normalized = _normalized(text).lower()
    for phrase, structure in _STRUCTURE_KEYWORDS:
        if phrase in normalized:
            option_leg_count = len(_MULTI_LEG_RE.findall(text))
            inherently_multi_leg = structure in {
                "iron_condor", "iron_butterfly", "reverse_iron_condor",
                "diagonal_spread", "calendar_spread", "ratio_spread",
                "ratio_backspread", "butterfly_spread", "roll", "spread",
            }
            option_only_multi_leg = inherently_multi_leg or (
                option_leg_count >= 2
                and structure not in {"covered_call", "protective_put"}
            )
            return structure, option_only_multi_leg
    upper = text.upper()
    option_leg_count = len(_MULTI_LEG_RE.findall(text))
    if option_leg_count >= 2:
        return "multi_leg", True
    if re.search(r"\b\d+(?:\.\d+)?/\d+(?:\.\d+)?/\d+(?:\.\d+)?/\d+(?:\.\d+)?\s+IC\b", upper):
        return "iron_condor", True
    if "+" in text or "/" in text:
        if option_leg_count >= 2 or re.search(r"\bIC\b", upper):
            return "multi_leg", True
    return None, False


def _extract_order_action(text: str) -> str:
    upper = text.upper()
    if re.search(r"\bSELL\s+\d+(?:\.\d+)?\s+OF\s+\d+", upper):
        return "close_long"
    if re.search(r"\bSELL\s+TO\s+CLOSE\b", upper) or re.search(r"\bSTC\b", upper):
        return "close_long"
    if re.search(r"\bBUY\s+TO\s+CLOSE\b", upper) or re.search(r"\bBTC\b", upper):
        return "close_short"
    if re.search(r"\bSELL\s+TO\s+OPEN\b", upper) or re.search(r"\bSTO\b", upper):
        return "open_short"
    if _MANAGEMENT_RE.search(text):
        return "manage"
    tokens = set(re.findall(r"[A-Z]+", upper))
    if tokens & _CLOSE_LONG_TOKENS:
        return "close_long"
    if tokens & _CLOSE_SHORT_TOKENS:
        return "close_short"
    if tokens & _OPEN_SHORT_TOKENS:
        return "open_short"
    if tokens & _OPEN_LONG_TOKENS:
        return "open_long"
    return "unknown"


def _leg_order_action(token: str, inherited: str = "unknown") -> str:
    normalized = re.sub(r"\s+", " ", str(token or "").strip().upper())
    return {
        "BUY TO OPEN": "open_long",
        "BTO": "open_long",
        "BUY": "open_long",
        "LONG": "open_long",
        "SELL TO OPEN": "open_short",
        "STO": "open_short",
        "SELL": "open_short",
        "SHORT": "open_short",
        "SELL TO CLOSE": "close_long",
        "STC": "close_long",
        "BUY TO CLOSE": "close_short",
        "BTC": "close_short",
    }.get(normalized, inherited)


def _extract_option_legs(text: str, default_root: str) -> tuple[ParsedOptionLeg, ...]:
    legs: list[ParsedOptionLeg] = []
    inherited_action = "unknown"
    inherited_root = default_root.upper()
    matches = list(_MULTI_LEG_RE.finditer(text or ""))
    for index, match in enumerate(matches):
        action_token, ratio_raw, root_token, strike_raw, side_token = match.groups()
        action = _leg_order_action(action_token or "", inherited_action)
        if (
            len(matches) == 1
            and action_token
            and str(action_token).upper() == "SELL"
            and re.search(r"\bSELL\s+\d+(?:\.\d+)?\s+OF\s+\d+", text, re.IGNORECASE)
        ):
            action = "close_long"
        if action == "unknown":
            action = _extract_order_action(text)
        if action != "unknown":
            inherited_action = action
        root = str(root_token or inherited_root).upper()
        root = SYMBOL_ALIASES.get(root, root)
        if root:
            inherited_root = root
        side = "CALL" if str(side_token).upper().startswith("C") else "PUT"
        try:
            strike = float(strike_raw)
        except (TypeError, ValueError):
            continue
        ratio = max(1, int(ratio_raw or 1))
        if ratio_raw and not action_token:
            prefix = text[max(0, match.start() - 12):match.start()]
            if re.search(r"(?:\bOF\s*|/\s*|\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*)$", prefix, re.IGNORECASE):
                ratio = 1
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        leg_expiry = _extract_explicit_expiry(text[match.end():segment_end])
        legs.append(ParsedOptionLeg(root, strike, side, action, ratio, leg_expiry))

    explicit_leg_expiries = {leg.expiry_date for leg in legs if leg.expiry_date}
    if len(explicit_leg_expiries) == 1:
        shared_expiry = next(iter(explicit_leg_expiries))
        legs = [
            ParsedOptionLeg(
                leg.root, leg.strike, leg.side, leg.order_action, leg.ratio_qty,
                leg.expiry_date or shared_expiry,
            )
            for leg in legs
        ]

    normalized = _normalized(text).upper()
    if "ROLL" in normalized and len(legs) == 2 and all(leg.order_action == "unknown" for leg in legs):
        legs = [
            ParsedOptionLeg(
                legs[0].root, legs[0].strike, legs[0].side, "close_long",
                legs[0].ratio_qty, legs[0].expiry_date,
            ),
            ParsedOptionLeg(
                legs[1].root, legs[1].strike, legs[1].side, "open_long",
                legs[1].ratio_qty, legs[1].expiry_date,
            ),
        ]
    return tuple(legs)


def _extract_compact_iron_condor(text: str, default_root: str) -> tuple[ParsedOptionLeg, ...]:
    match = re.search(
        r"\b(?:SELL|STO)\s+([A-Z.]{1,6})\s+"
        r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+IC\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return tuple()
    root = SYMBOL_ALIASES.get(match.group(1).upper(), match.group(1).upper()) or default_root.upper()
    expiry = _extract_explicit_expiry(text)
    put_short, put_long, call_short, call_long = (float(match.group(i)) for i in range(2, 6))
    return (
        ParsedOptionLeg(root, put_short, "PUT", "open_short", 1, expiry),
        ParsedOptionLeg(root, put_long, "PUT", "open_long", 1, expiry),
        ParsedOptionLeg(root, call_short, "CALL", "open_short", 1, expiry),
        ParsedOptionLeg(root, call_long, "CALL", "open_long", 1, expiry),
    )


def _normalize_leg_ratios(
    legs: tuple[ParsedOptionLeg, ...], quantity: float
) -> tuple[tuple[ParsedOptionLeg, ...], float]:
    if not legs:
        return legs, quantity
    ratios = [max(1, int(leg.ratio_qty)) for leg in legs]
    common = ratios[0]
    for ratio in ratios[1:]:
        common = gcd(common, ratio)
    if len(legs) == 1:
        common = ratios[0]
    if common <= 1:
        return legs, quantity
    normalized = tuple(
        ParsedOptionLeg(
            leg.root, leg.strike, leg.side, leg.order_action,
            max(1, int(leg.ratio_qty) // common), leg.expiry_date,
        )
        for leg in legs
    )
    return normalized, max(float(quantity), float(common))


def _extract_management_fields(text: str) -> dict:
    targets = tuple(
        float(value) for value in re.findall(
            r"\b(?:TP|TARGET|TAKE\s+PROFIT)\s*\d*\s*(?:AT|@|[:=\-])?\s*\$?(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
    )
    trail = re.search(r"\bTRAIL(?:ING)?\s+STOP\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    close_pct = re.search(r"\b(?:STC|TRIM|SELL|TAKE)\s+(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    add = re.search(
        r"\bADD\s+(\d+(?:\.\d+)?)\s+MORE\s+IF\s+PREMIUM\s+(?:FALLS|DROPS)\s+TO\s+\$?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    trigger = re.search(
        r"\b(?:STOCK\s+)?(?:BREAKS?|CLOSES?)\s+(ABOVE|BELOW)\s+\$?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    return {
        "target_prices": targets,
        "trailing_stop_pct": float(trail.group(1)) if trail else None,
        "close_percent": float(close_pct.group(1)) if close_pct else None,
        "add_quantity": float(add.group(1)) if add else None,
        "add_trigger_premium": float(add.group(2)) if add else None,
        "underlying_trigger_direction": trigger.group(1).lower() if trigger else None,
        "underlying_trigger_price": float(trigger.group(2)) if trigger else None,
        "exit_before_market_close": bool(re.search(r"\b(?:EXIT|CLOSE).{0,30}BEFORE\s+MARKET\s+CLOSE\b", text, re.IGNORECASE)),
        "contains_equity_leg": bool(re.search(r"\b\d+(?:\.\d+)?\s+[A-Z.]+\s+SHARES?\b", text, re.IGNORECASE)),
    }


def _classify_multi_leg_structure(
    legs: tuple[ParsedOptionLeg, ...], current: Optional[str]
) -> Optional[str]:
    if current and current not in {"multi_leg", "spread", "straddle", "strangle"}:
        return current
    if len(legs) == 4:
        strikes = {leg.strike for leg in legs}
        actions = [leg.order_action for leg in legs]
        if len(strikes) == 3:
            return "iron_butterfly"
        if actions == ["open_short", "open_long", "open_short", "open_long"]:
            return "iron_condor"
        if actions == ["open_long", "open_short", "open_long", "open_short"]:
            return "reverse_iron_condor"
        return current or "multi_leg"
    if len(legs) == 3:
        if len({leg.side for leg in legs}) == 1:
            return "butterfly_spread"
        return current or "multi_leg"
    if len(legs) != 2:
        return current or "multi_leg"
    first, second = legs
    if first.expiry_date and second.expiry_date and first.expiry_date != second.expiry_date:
        return "calendar_spread" if first.strike == second.strike else "diagonal_spread"
    if first.ratio_qty != second.ratio_qty:
        long_ratio = sum(leg.ratio_qty for leg in legs if leg.order_action == "open_long")
        short_ratio = sum(leg.ratio_qty for leg in legs if leg.order_action == "open_short")
        return "ratio_backspread" if long_ratio > short_ratio else "ratio_spread"
    both_open_long = first.order_action == second.order_action == "open_long"
    if both_open_long and first.side != second.side:
        return "long_straddle" if first.strike == second.strike else "long_strangle"
    if first.side == second.side == "CALL":
        if first.order_action == "open_long" and second.order_action == "open_short":
            return "bull_call_spread" if first.strike < second.strike else "bear_call_spread"
    if first.side == second.side == "PUT":
        if first.order_action == "open_long" and second.order_action == "open_short":
            return "bear_put_spread" if first.strike > second.strike else "bull_put_spread"
    return current or "multi_leg"


def _extract_tense(text: str) -> str:
    upper = text.upper()
    tokens = set(re.findall(r"[A-Z]+", upper))
    action = _extract_order_action(text)
    if action in {"close_long", "close_short", "manage"}:
        return "management"
    if action in {"open_long", "open_short"}:
        return "new_order"
    if tokens & _PAST_TENSE_VERBS:
        return "past"
    if tokens & set(_NEW_ORDER_VERBS.keys()):
        return "new_order"
    return "unknown"


def _extract_quantity(text: str) -> float:
    for pattern in _QTY_RE:
        match = pattern.search(text)
        if match:
            try:
                qty = float(match.group(1))
                if qty > 0:
                    return qty
            except ValueError:
                continue
    return 1.0


def looks_like_option_signal(text: str) -> bool:
    """Detector used by classify_and_parse() to route to the options path."""
    if _STRIKE_SIDE_RE.search(text):
        return True
    if _SIDE_STRIKE_RE.search(text):
        return True
    if _STRIKE_SHORTHAND_RE.search(text):
        tokens = set(re.findall(r"[A-Z]+", text.upper()))
        if _extract_order_action(text) in {"open_long", "open_short", "close_long", "close_short"} or tokens & {"BUY", "SELL", "BTO", "STO", "LONG", "SHORT"}:
            return True
    if _DELTA_RE.search(text) and _SIDE_ONLY_RE.search(text):
        return True
    structure, is_multi_leg = _extract_structure(text)
    if structure:
        return True
    upper = text.upper()
    tokens = set(re.findall(r"[A-Z]+", upper))
    if tokens & {"BTO", "STO", "BTC", "STC"}:
        return True
    if _MANAGEMENT_RE.search(text):
        return True
    if tokens & (_OPEN_LONG_TOKENS | _OPEN_SHORT_TOKENS | _CLOSE_LONG_TOKENS | _CLOSE_SHORT_TOKENS):
        if _STRIKE_SIDE_RE.search(text) or _SIDE_STRIKE_RE.search(text):
            return True
        if tokens & {"CALL", "CALLS", "PUT", "PUTS", "CE", "PE"}:
            return True
    if "OPTION" in tokens and tokens & {"CALL", "CALLS", "PUT", "PUTS", "CE", "PE"}:
        return True
    return False


def parse_option_signal(text: str) -> ParsedOptionSignal:
    """Parse a message already identified as options-related by looks_like_option_signal()."""
    raw = normalize_signal_input(text or "")
    root = _extract_root(raw)
    delta_target = _extract_delta_target(raw)
    if delta_target is not None and not _STRIKE_SIDE_RE.search(raw):
        strike, side = None, _extract_side_only(raw)
    else:
        strike, side = _extract_strike_side(raw)
    if side is None and delta_target is not None:
        side = _extract_side_only(raw)
    fill_price = _extract_fill_price(raw)
    stop_loss = _extract_optional_price(_STOP_LOSS_RE, raw)
    target_price = _extract_optional_price(_TARGET_RE, raw)
    risk_reward = _extract_risk_reward(raw)
    stop_loss, target_price = _derive_missing_exit_from_rr(fill_price, stop_loss, target_price, risk_reward)
    structure, is_multi_leg = _extract_structure(raw)
    parsed_legs = _extract_option_legs(raw, root)
    if is_multi_leg and len(parsed_legs) < 2 and structure == "iron_condor":
        parsed_legs = _extract_compact_iron_condor(raw, root)
    if is_multi_leg:
        structure = _classify_multi_leg_structure(parsed_legs, structure)
    order_action = parsed_legs[0].order_action if parsed_legs else _extract_order_action(raw)
    if order_action == "manage" and not (strike and side):
        root = ""
    tense = _extract_tense(raw)
    quantity = _extract_quantity(raw)
    parsed_legs, quantity = _normalize_leg_ratios(parsed_legs, quantity)
    management = _extract_management_fields(raw)
    upper_raw = raw.upper()
    price_effect = "credit" if "CREDIT" in upper_raw else "debit" if "DEBIT" in upper_raw else None

    explicit_expiry = next((leg.expiry_date for leg in parsed_legs if leg.expiry_date), None)
    if not explicit_expiry:
        explicit_expiry = _extract_explicit_expiry(raw)
    if explicit_expiry:
        expiry_mode = "explicit"
        expiry_date = explicit_expiry
    else:
        expiry_mode = "0dte"
        expiry_date = default_expiry_date("0dte").isoformat()

    if not root:
        return ParsedOptionSignal(
            valid=False,
            raw_text=raw,
            structure=structure,
            is_multi_leg=is_multi_leg,
            tense=tense,
            order_action=order_action,
            quantity=quantity,
            reason="Recognized as an options signal but no underlying ticker was found.",
        )

    if is_multi_leg:
        roots = {leg.root for leg in parsed_legs if leg.root}
        if not 2 <= len(parsed_legs) <= 4:
            return ParsedOptionSignal(
                valid=False,
                root=root,
                structure=structure,
                is_multi_leg=True,
                legs=parsed_legs,
                price_effect=price_effect,
                tense=tense,
                order_action=order_action,
                quantity=quantity,
                raw_text=raw,
                reason="Multi-leg option signals require 2 to 4 fully specified option legs.",
            )
        if len(roots) != 1:
            return ParsedOptionSignal(
                valid=False,
                root=root,
                structure=structure,
                is_multi_leg=True,
                legs=parsed_legs,
                price_effect=price_effect,
                tense=tense,
                order_action=order_action,
                quantity=quantity,
                raw_text=raw,
                reason="All legs in an Alpaca multi-leg order must use the same underlying symbol.",
            )
        return ParsedOptionSignal(
            valid=True,
            root=root,
            strike=strike,
            side=side,
            delta_target=delta_target,
            fill_price=fill_price,
            stop_loss=stop_loss,
            target_price=target_price,
            risk_reward=risk_reward,
            **management,
            expiry_date=expiry_date,
            expiry_mode=expiry_mode,
            structure=structure,
            is_multi_leg=True,
            legs=parsed_legs,
            price_effect=price_effect,
            tense=tense,
            order_action=order_action,
            quantity=quantity,
            raw_text=raw,
            reason="",
        )

    if not ((strike or delta_target) and side):
        return ParsedOptionSignal(
            valid=False,
            root=root,
            delta_target=delta_target,
            structure=structure,
            is_multi_leg=is_multi_leg,
            tense=tense,
            order_action=order_action,
            quantity=quantity,
            raw_text=raw,
            reason=f"Recognized {root} as an options signal but no strike or delta with CALL/PUT side was found.",
        )

    return ParsedOptionSignal(
        valid=True,
        root=root,
        strike=strike,
        side=side,
        delta_target=delta_target,
        fill_price=fill_price,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_reward=risk_reward,
        **management,
        expiry_date=expiry_date,
        expiry_mode=expiry_mode,
        structure=structure,
        is_multi_leg=is_multi_leg,
        legs=parsed_legs,
        price_effect=price_effect,
        tense=tense,
        order_action=order_action,
        quantity=quantity,
        raw_text=raw,
        reason="",
    )

def classify_and_parse(text: str) -> ParsedMessage:
    """Route a raw Discord message to EQUITY / OPTION / NO_TRADE / INVALID."""
    raw = text or ""
    compact = normalize_signal_input(raw)
    if not compact:
        return ParsedMessage(kind="INVALID", raw_text=raw, reason="Empty message.")

    if looks_like_option_signal(compact):
        option = parse_option_signal(compact)
        option = replace(option, raw_text=raw)
        return ParsedMessage(kind="OPTION", option=option, raw_text=raw, reason=option.reason)

    equity = parse_signal(compact)
    if equity.valid:
        equity = replace(equity, raw_text=raw)
        return ParsedMessage(kind="EQUITY", equity=equity, raw_text=raw)
    if equity.order_intent and equity.order_intent.get("status") == "INVALID_OR_NON_EXECUTABLE":
        return ParsedMessage(
            kind="INVALID",
            equity=replace(equity, raw_text=raw),
            raw_text=raw,
            reason=equity.reason or "Invalid or non-executable stock order.",
        )

    # Readable market commentary that isn't a tradeable equity or options signal.
    return ParsedMessage(
        kind="NO_TRADE",
        raw_text=raw,
        reason="No tradeable BUY/SELL/HOLD or options signal detected — logged as market commentary.",
    )






