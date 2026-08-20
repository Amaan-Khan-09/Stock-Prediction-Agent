"""Discord / WhatsApp natural-language trading signal parser.

Handles stock AND options messages including:
  Stock:   "buy AAPL", "sell TSLA qty 2", "Apple looks strong here, taking small buy",
           "exit MSFT", "trim half of AMD", "holding NVDA"

  Options: "Bought SPY 620C @ 1.43, 2 contracts, exp 2026-07-17"
           "Entered SPY 620C 0DTE @ 1.40, 2 contracts"
           "Buy SPX 7570 call option at 1.40"
           "SPY 450P lotto"
           "AAPL 240C exp 2026-08-21"
           "TSLA 300 puts weekly"
           "Buy GOOGL call delta 83 DTE 45 qty 1"
           "trimmed half at 1.90"
           "out rest at 2.20"
           "sold off this one"
           "Holding runners"

Rules:
  - Strike has NO upper cap (7570 for SPX is valid).
  - Delta UI range 0-100; internally normalized to 0.00-1.00.
  - If expiry/DTE missing for options → missing_fields includes them.
  - If exact strike missing & delta missing → missing_fields includes both.
  - parser_confidence 0.0-1.0.
"""
from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.options_models import (
    AssetType, ContractMethod, OptionSide, ParsedSignal, SignalAction, VS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Known tickers — extended S&P 500 + index set
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "BRK", "BRKB", "BRK.B", "JPM", "UNH", "V", "XOM", "JNJ", "WMT",
    "MA", "LLY", "PG", "HD", "MRK", "ABBV", "AVGO", "CVX", "PEP",
    "COST", "KO", "TMO", "ADBE", "NFLX", "CRM", "ACN", "AMD", "QCOM",
    "MCD", "TXN", "NEE", "DIS", "INTU", "RTX", "BA", "GS", "MS",
    "UBER", "LYFT", "PLTR", "SOFI", "COIN", "HOOD", "GME", "AMC",
    # Indices
    "SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "SLV", "TLT", "HYG",
    "SPX", "NDX", "RUT", "VIX",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    # Common names mapping handled in normalize step
}

COMPANY_NAME_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "meta": "META", "google": "GOOGL", "alphabet": "GOOGL", "tesla": "TSLA",
    "netflix": "NFLX", "amazon": "AMZN", "amd": "AMD", "intel": "INTC",
    "boeing": "BA", "disney": "DIS", "uber": "UBER", "coinbase": "COIN",
    "palantir": "PLTR", "sofi": "SOFI", "berkshire": "BRK.B",
}

# Action keyword maps
ACTION_BUY_WORDS = {
    "buy", "bought", "long", "entering", "entered", "add", "adding",
    "took", "taking", "grab", "grabbed", "scalp", "lotto", "call",
    "opening", "opened",
}
ACTION_SELL_WORDS = {
    "sell", "sold", "short", "shorting", "exit", "exiting", "out",
    "close", "closing", "dump", "dumped",
}
ACTION_TRIM_WORDS = {"trim", "trimmed", "trimming", "half", "partial", "partialed"}
ACTION_HOLD_WORDS = {"hold", "holding", "holding runners", "runners", "watching"}
ACTION_WATCH_WORDS = {"watching", "watch", "eyeing", "looking at"}

# Patterns for expiry date
_DATE_PATTERNS = [
    r"(?:exp(?:iry|iration)?\.?\s*)(\d{4}-\d{2}-\d{2})",   # exp 2026-07-17
    r"(?:exp(?:iry|iration)?\.?\s*)(\d{1,2}/\d{1,2}/\d{4})",# exp 7/17/2026
    r"(?:exp(?:iry|iration)?\.?\s*)(\d{1,2}/\d{1,2}/\d{2})",# exp 7/17/26
    r"(\d{4}-\d{2}-\d{2})",                                  # bare ISO date
    r"(\d{1,2}/\d{1,2}/\d{4})",                             # M/D/YYYY
    r"(\d{1,2}/\d{1,2}/\d{2})",                             # M/D/YY
]

# Compiled option symbol pattern: AAPL240C, SPY620C, SPX7570C, SPY620P
_OPT_INLINE = re.compile(
    r"\b([A-Z]{1,5})\s{0,2}(\d{3,6}(?:\.\d{1,2})?)\s*([CP])\b",
    re.IGNORECASE,
)
# Separate: underlying + strike + "call"/"put"
_OPT_WORDS = re.compile(
    r"\b([A-Z]{1,5})\s+(\d{3,6}(?:\.\d{1,2})?)\s+(call|put|calls|puts|c|p)\b",
    re.IGNORECASE,
)
# Delta pattern: "delta 83" or "d83"
_DELTA_PAT = re.compile(r"\bdelta\s*(\d{1,3})\b|\bd(\d{1,3})\b", re.IGNORECASE)
# DTE pattern: "DTE 45" or "45DTE" or "0DTE"
# NOTE: first alternative requires NO space between digit and DTE (e.g. "0DTE", "1DTE")
#       to avoid matching "delta 83 DTE 45" as dte=83.
_DTE_PAT = re.compile(r"(\d{1,3})dte\b|\bdte\s*(\d{1,3})", re.IGNORECASE)
# Premium / price
_PREMIUM_PAT = re.compile(r"[@$]\s*(\d+(?:\.\d{1,4})?)|at\s+(\d+(?:\.\d{1,4})?)", re.IGNORECASE)
# Contracts / quantity
_CONTRACTS_PAT = re.compile(r"(\d+)\s+contract(?:s?)\b|(\d+)\s+lot(?:s?)\b|qty\s*(\d+)", re.IGNORECASE)
_QTY_PAT       = re.compile(r"\bqty\s*(\d+)\b|\b(\d+)\s+shares?\b", re.IGNORECASE)
# Weekly / lotto flags
_WEEKLY_PAT    = re.compile(r"\bweekly\b|\bweeklies\b", re.IGNORECASE)
_LOTTO_PAT     = re.compile(r"\blotto\b", re.IGNORECASE)
# Partial exit patterns
_TRIM_PRICE_PAT = re.compile(r"(?:trim|out|sold|closed?)\s+(?:half|rest|all|runners?)?\s+(?:at|@)\s*(\d+(?:\.\d{1,4})?)", re.IGNORECASE)
_OUT_PRICE_PAT  = re.compile(r"(?:out|exit|close)\s+(?:rest|all|runners?)?\s+(?:at|@)\s*(\d+(?:\.\d{1,4})?)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_signal(
    raw_message: str,
    reply_to_message_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> ParsedSignal:
    """Parse a raw trading Discord/WhatsApp message into a ParsedSignal.

    Works for stocks and options. Returns ParsedSignal with missing_fields
    populated whenever required data is absent.
    """
    sig = ParsedSignal(raw_message=raw_message.strip())
    sig.reply_to_message_id = reply_to_message_id
    if message_id:
        sig.linked_signal_id = message_id

    text = raw_message.strip()
    norm = text.lower()

    sig.normalized_message = text

    # ── Step 1: Detect partial-exit / reply-context patterns first ─────────
    if _is_partial_exit(norm):
        sig.asset_type = AssetType.UNKNOWN  # linked to prior trade
        sig.action = _detect_partial_action(norm)
        price = _extract_exit_price(text)
        if price is not None:
            sig.premium = price
        # Try to extract underlying even for partial exits (e.g. "trim half of AMD")
        underlying = _extract_ticker(text)
        if underlying:
            sig.underlying = underlying
        sig.parser_confidence = 0.65
        sig.raw_extras["is_partial_exit"] = True
        return sig

    # ── Step 2: Detect option signal ──────────────────────────────────────
    opt_match = _match_option_inline(text) or _match_option_words(text)
    if opt_match:
        sig.asset_type  = AssetType.OPTION
        sig.underlying  = opt_match["underlying"]
        sig.option_type = opt_match["option_type"]
        sig.strike      = opt_match["strike"]
        sig.contract_selection_method = ContractMethod.EXACT_STRIKE
        sig.action      = _detect_action(norm)
        sig.high_risk_tag = bool(_LOTTO_PAT.search(text))

        # Extract extras
        sig.expiry_date = _extract_expiry(text)
        sig.dte         = _extract_dte(text)
        sig.premium     = _extract_premium(text)
        sig.contracts   = _extract_contracts(text)

        # Check for weekly → mark expiry_type
        if _WEEKLY_PAT.search(text):
            sig.raw_extras["expiry_type"] = "weekly"
            if not sig.expiry_date and not sig.dte:
                # Infer next Friday
                next_fri = _next_friday()
                sig.expiry_date = next_fri.strftime("%Y-%m-%d")
                sig.raw_extras["expiry_inferred"] = True

        # Missing field detection
        if not sig.expiry_date and sig.dte is None:
            sig.missing_fields.append("expiry_or_dte")
        sig.parser_confidence = _option_confidence(sig)
        return sig

    # ── Step 3: Detect delta-based option ────────────────────────────────
    delta_m = _DELTA_PAT.search(text)
    if delta_m and _has_option_keyword(norm):
        sig.asset_type = AssetType.OPTION
        sig.contract_selection_method = ContractMethod.DELTA_SELECTION
        delta_raw = int(delta_m.group(1) or delta_m.group(2))
        sig.delta_ui = delta_raw
        sig.delta_decimal = _normalize_delta(delta_raw)
        sig.underlying = _extract_ticker(text)
        sig.option_type = _detect_option_type(norm)
        sig.dte = _extract_dte(text)
        sig.contracts = _extract_contracts(text)
        sig.premium = _extract_premium(text)
        sig.action = _detect_action(norm)
        if not sig.underlying:
            sig.missing_fields.append("underlying")
        if not sig.option_type:
            sig.missing_fields.append("option_type")
        if sig.dte is None:
            sig.missing_fields.append("dte")
        sig.parser_confidence = 0.75 if not sig.missing_fields else 0.50
        return sig

    # ── Step 4: Stock signal ──────────────────────────────────────────────
    underlying = _extract_ticker(text)
    if underlying:
        sig.asset_type  = AssetType.STOCK
        sig.underlying  = underlying
        sig.action      = _detect_action(norm)
        sig.quantity    = _extract_qty(text)
        sig.parser_confidence = 0.85 if sig.action != SignalAction.UNKNOWN else 0.55
        return sig

    # ── Step 5: Fallback — unclear message ────────────────────────────────
    sig.parser_confidence = 0.10
    sig.missing_fields = ["underlying", "action", "asset_type"]
    return sig


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _match_option_inline(text: str) -> Optional[Dict]:
    """Match patterns like SPY620C, SPX7570P, AAPL240C."""
    m = _OPT_INLINE.search(text)
    if not m:
        return None
    underlying = m.group(1).upper()
    strike     = float(m.group(2))
    cp         = m.group(3).upper()
    option_type = OptionSide.CALL if cp == "C" else OptionSide.PUT
    return {"underlying": underlying, "strike": strike, "option_type": option_type}


def _match_option_words(text: str) -> Optional[Dict]:
    """Match patterns like 'SPY 620 call', 'AAPL 240 put', 'SPX 7570 call option'."""
    m = _OPT_WORDS.search(text)
    if not m:
        return None
    underlying = m.group(1).upper()
    strike     = float(m.group(2))
    raw_type   = m.group(3).lower()
    option_type = OptionSide.CALL if raw_type.startswith("c") else OptionSide.PUT
    return {"underlying": underlying, "strike": strike, "option_type": option_type}


def _has_option_keyword(norm: str) -> bool:
    return bool(re.search(r"\b(call|put|calls|puts|option|options|straddle|strangle)\b", norm))


def _detect_option_type(norm: str) -> Optional[str]:
    if re.search(r"\b(call|calls)\b", norm):
        return OptionSide.CALL
    if re.search(r"\b(put|puts)\b", norm):
        return OptionSide.PUT
    return None


def _extract_ticker(text: str) -> Optional[str]:
    """Extract stock ticker from text. Tries known tickers first, then regex."""
    upper = text.upper()
    # Check known tickers
    for tk in sorted(KNOWN_TICKERS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(tk) + r"\b", upper):
            return tk
    # Check company names
    lower = text.lower()
    for name, tk in COMPANY_NAME_MAP.items():
        if name in lower:
            return tk
    # Fallback: 1-5 uppercase letters as standalone word
    m = re.search(r"\b([A-Z]{2,5})\b", upper)
    if m and not m.group(1) in {"DTE", "BUY", "SELL", "ATM", "OTM", "ITM", "RSI", "ATR"}:
        return m.group(1)
    return None


def _extract_expiry(text: str) -> Optional[str]:
    """Extract expiry date from text, normalize to YYYY-MM-DD."""
    for pat in _DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            dt = _parse_date_flexible(raw)
            if dt:
                return dt.strftime("%Y-%m-%d")
    return None


def _parse_date_flexible(s: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except Exception:
            continue
    return None


def _extract_dte(text: str) -> Optional[int]:
    m = _DTE_PAT.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    return int(val) if val is not None else None


def _extract_premium(text: str) -> Optional[float]:
    m = _PREMIUM_PAT.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    return float(val) if val else None


def _extract_contracts(text: str) -> Optional[int]:
    m = _CONTRACTS_PAT.search(text)
    if not m:
        return None
    val = m.group(1) or m.group(2) or m.group(3)
    return int(val) if val else None


def _extract_qty(text: str) -> Optional[int]:
    m = _QTY_PAT.search(text) or _CONTRACTS_PAT.search(text)
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


def _extract_exit_price(text: str) -> Optional[float]:
    m = _TRIM_PRICE_PAT.search(text) or _OUT_PRICE_PAT.search(text)
    if m:
        return float(m.group(1))
    # bare "at X.XX"
    m2 = re.search(r"\bat\s+(\d+(?:\.\d{1,4})?)\b", text, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None


def _normalize_delta(delta_ui: int) -> float:
    """Map UI delta 0-100 to decimal 0.00-1.00."""
    return round(max(0.0, min(1.0, delta_ui / 100.0)), 4)


def _detect_action(norm: str) -> str:
    if any(w in norm for w in ACTION_TRIM_WORDS):
        return SignalAction.TRIM
    if any(w in norm for w in ACTION_HOLD_WORDS):
        return SignalAction.HOLD
    if any(w in norm for w in ACTION_SELL_WORDS):
        return SignalAction.SELL
    if any(w in norm for w in ACTION_BUY_WORDS):
        return SignalAction.BUY
    if any(w in norm for w in ACTION_WATCH_WORDS):
        return SignalAction.WATCH
    return SignalAction.UNKNOWN


def _is_partial_exit(norm: str) -> bool:
    partial_keywords = [
        "trimmed half", "trim half", "out rest", "out all",
        "sold off this", "sold off", "holding runners", "runners",
        "trimmed", "out at", "closed at", "exit at",
    ]
    return any(kw in norm for kw in partial_keywords)


def _detect_partial_action(norm: str) -> str:
    if any(w in norm for w in ["trim", "half", "partial"]):
        return SignalAction.TRIM
    if any(w in norm for w in ["out", "exit", "close", "sold"]):
        return SignalAction.SELL
    if any(w in norm for w in ["hold", "runner"]):
        return SignalAction.HOLD
    return SignalAction.SELL


def _option_confidence(sig: ParsedSignal) -> float:
    score = 0.6
    if sig.underlying:         score += 0.10
    if sig.option_type:        score += 0.05
    if sig.strike is not None: score += 0.05
    if sig.expiry_date:        score += 0.10
    if sig.dte is not None:    score += 0.05
    if sig.premium is not None:score += 0.05
    # Penalize missing fields
    score -= len(sig.missing_fields) * 0.08
    return max(0.0, min(1.0, round(score, 2)))


def _next_friday() -> date:
    today = date.today()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    return today + timedelta(days=days_until_friday)


# ──────────────────────────────────────────────────────────────────────────────
# Reply-chain P&L calculator
# ──────────────────────────────────────────────────────────────────────────────

def calculate_reply_chain_pnl(signals: List[ParsedSignal]) -> dict:
    """Resolve a list of linked signals (entry + exits) into realized P&L.

    Example:
        Entry:   SPY 620C @ 1.40, qty 2
        Trim:    trimmed half at 1.90  (exit 1 contract)
        Exit:    out rest at 2.20      (exit 1 contract)
        → realized_pnl = (1.90 - 1.40)*1*100 + (2.20 - 1.40)*1*100 = 50 + 80 = $130

    Returns dict with:
        status: CLOSED | PARTIAL | OPEN | REVIEW_REQUIRED
        entry_premium
        entry_qty
        exits: [{price, qty, pnl}]
        total_qty_exited
        remaining_qty
        realized_pnl
        avg_exit_price
        errors
    """
    if not signals:
        return {"status": "REVIEW_REQUIRED", "errors": ["No signals provided."], "realized_pnl": None}

    # Find entry signal
    entry = next((s for s in signals if s.action in (SignalAction.BUY, SignalAction.SELL)
                  and s.premium is not None), None)
    if entry is None:
        return {
            "status": "REVIEW_REQUIRED",
            "errors": ["No entry signal with premium found — cannot compute P&L."],
            "realized_pnl": None,
        }

    entry_premium = entry.premium
    entry_qty     = entry.quantity or entry.contracts or 1
    entry_side    = entry.action  # BUY = long call/put; SELL = short

    # Collect exit events (TRIM, SELL, HOLD with price context)
    exits = []
    for s in signals:
        if s is entry:
            continue
        if s.action in (SignalAction.TRIM, SignalAction.SELL) and s.premium is not None:
            exit_qty = s.quantity or s.contracts or None
            exits.append({"price": s.premium, "qty": exit_qty, "signal": s})

    if not exits:
        return {
            "status": "OPEN",
            "entry_premium": entry_premium,
            "entry_qty": entry_qty,
            "exits": [],
            "total_qty_exited": 0,
            "remaining_qty": entry_qty,
            "realized_pnl": None,
            "errors": ["No exit signals found — position still open."],
        }

    # Distribute qty across exits
    total_qty = entry_qty
    qty_left  = total_qty
    exit_records = []
    total_pnl = 0.0
    errors    = []

    for ex in exits:
        if qty_left <= 0:
            break
        exit_price = ex["price"]
        exit_qty   = ex["qty"]

        # "trimmed half" → half of remaining qty
        if exit_qty is None:
            sig_action = ex["signal"].action
            if sig_action == SignalAction.TRIM:
                exit_qty = max(1, qty_left // 2)
            else:
                exit_qty = qty_left

        exit_qty = min(exit_qty, qty_left)

        # P&L per contract * 100 shares
        if entry_side == SignalAction.BUY:
            pnl = (exit_price - entry_premium) * exit_qty * 100
        else:
            pnl = (entry_premium - exit_price) * exit_qty * 100

        exit_records.append({"price": exit_price, "qty": exit_qty, "pnl": round(pnl, 2)})
        total_pnl += pnl
        qty_left  -= exit_qty

    total_exited = total_qty - qty_left
    avg_exit = (
        sum(e["price"] * e["qty"] for e in exit_records) / sum(e["qty"] for e in exit_records)
        if exit_records else None
    )

    status = "CLOSED" if qty_left == 0 else "PARTIAL"
    if qty_left < 0:
        errors.append("Exit qty exceeds entry qty — data inconsistency.")
        status = "REVIEW_REQUIRED"

    return {
        "status":           status,
        "entry_premium":    entry_premium,
        "entry_qty":        total_qty,
        "exits":            exit_records,
        "total_qty_exited": total_exited,
        "remaining_qty":    max(0, qty_left),
        "realized_pnl":     round(total_pnl, 2),
        "avg_exit_price":   round(avg_exit, 4) if avg_exit else None,
        "errors":           errors,
    }
