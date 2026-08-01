"""OCC option symbol construction and underlying-ticker helpers.

Pure functions only -- no network calls, no Alpaca API access. Used by
options_parser.py (to know what to build) and alpaca_paper.py (to know what
to look up / submit).
"""
from __future__ import annotations

from datetime import date, timedelta
from math import gcd
from typing import List, Optional, Sequence, Tuple


# Cash-settled index roots have no OCC-cleared option contract of their own
# on most retail brokers (including Alpaca). This maps them to the nearest
# liquid ETF proxy for PREDICTION PURPOSES ONLY -- the option contract that
# actually gets looked up / traded still uses the literal root the user typed.
_INDEX_TO_ETF_PROXY = {
    "SPX": "SPY",
    "NDX": "QQQ",
    "RUT": "IWM",
    "DJX": "DIA",
    "VIX": "VIXY",
}

# Roots Alpaca is known to support for options (standard equity/ETF options).
# Not exhaustive -- the real source of truth is the contracts-lookup API call
# in alpaca_paper.get_option_contracts(). This set only drives an early,
# friendly warning before that call is made.
_LIKELY_UNSUPPORTED_ROOTS = frozenset({"SPX", "NDX", "RUT", "DJX", "VIX", "XSP"})


def resolve_underlying_for_prediction(root: str) -> str:
    """Map a cash-settled index root to its ETF proxy for the AI prediction step.

    The prediction engine fetches historical bars via historical_price_service,
    which has no concept of index tickers -- only exchange-listed equities/ETFs.
    Pass-through unchanged for anything not in the index map.
    """
    return _INDEX_TO_ETF_PROXY.get(root.upper().strip(), root.upper().strip())


def likely_unsupported_by_alpaca(root: str) -> bool:
    """True if this root is a cash-settled index unlikely to have an Alpaca-tradable
    option contract. Not authoritative -- always confirm with get_option_contracts()
    before relying on this; used only to give an early, clear heads-up.
    """
    return root.upper().strip() in _LIKELY_UNSUPPORTED_ROOTS


def default_expiry_date(expiry_mode: str, today: Optional[date] = None) -> date:
    """Resolve a default expiry date when the signal did not name one explicitly.

    expiry_mode == "0dte" -> today (same trading day), matching "EOD"/"lotto" slang.
    Any other value -> nearest upcoming Friday (weekly expiry convention).
    Weekend "today" values roll forward to the next trading day equivalent
    (Mon for 0dte requested on a Sat/Sun) since 0DTE only exists on trading days.
    """
    day = today or date.today()
    if expiry_mode == "0dte":
        if day.weekday() == 5:  # Saturday
            return day + timedelta(days=2)
        if day.weekday() == 6:  # Sunday
            return day + timedelta(days=1)
        return day

    # weekly fallback: next Friday (today if today already is Friday)
    days_until_friday = (4 - day.weekday()) % 7
    return day + timedelta(days=days_until_friday)


def build_occ_symbol(root: str, expiry: date, side: str, strike: float) -> str:
    """Build a standard OCC-format option symbol: ROOT + YYMMDD + C/P + 8-digit strike.

    Example: build_occ_symbol("SPX", date(2026, 7, 16), "CALL", 7570.0)
             -> "SPX260716C07570000"

    strike is multiplied by 1000 and zero-padded to 8 digits (OCC convention
    for whole-dollar and fractional strikes alike, e.g. 150.5 -> 00150500).
    """
    root_clean = root.upper().strip()
    side_letter = "C" if side.upper().startswith("C") else "P"
    date_part = expiry.strftime("%y%m%d")
    strike_thousandths = round(strike * 1000)
    strike_part = f"{strike_thousandths:08d}"
    return f"{root_clean}{date_part}{side_letter}{strike_part}"


def parse_occ_symbol(occ_symbol: str) -> Optional[dict]:
    """Best-effort inverse of build_occ_symbol(), for logging/display only."""
    sym = occ_symbol.strip().upper()
    for i, ch in enumerate(sym):
        if ch.isdigit():
            root = sym[:i]
            rest = sym[i:]
            break
    else:
        return None
    if len(rest) < 15:
        return None
    date_part, side_letter, strike_part = rest[:6], rest[6], rest[7:15]
    try:
        expiry = date(2000 + int(date_part[:2]), int(date_part[2:4]), int(date_part[4:6]))
        strike = int(strike_part) / 1000.0
    except (ValueError, IndexError):
        return None
    return {
        "root": root,
        "expiry": expiry.isoformat(),
        "side": "CALL" if side_letter == "C" else "PUT",
        "strike": strike,
    }


def listed_expiry_fallbacks(expiry_date: str) -> list[str]:
    """Return exchange-listed expiry candidates for a user-provided date.

    Discord rooms often write the Saturday OCC/monthly expiration date. Equity
    options are normally listed for the prior Friday, so contract lookup should
    try that date after the exact user date.
    """
    if not expiry_date:
        return []
    try:
        day = date.fromisoformat(str(expiry_date)[:10])
    except ValueError:
        return [str(expiry_date)]
    candidates = [day]
    if day.weekday() == 5:  # Saturday monthly-style shorthand.
        candidates.append(day - timedelta(days=1))
    elif day.weekday() == 6:  # Sunday typo/weekend shorthand.
        candidates.append(day - timedelta(days=2))
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        text = item.isoformat()
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def reduce_ratios_by_gcd(ratios: Sequence[int]) -> Tuple[List[int], int]:
    """Reduce a list of multi-leg ratios to lowest terms via their GCD.

    Shared by options_parser.py and multi_leg_contract.py, which each parse
    legs into different intermediate shapes (dataclasses vs. dicts) but were
    previously re-implementing this exact same reduction independently.

    Returns (reduced_ratios, common_gcd). common_gcd is 1 (ratios unchanged,
    clamped to >= 1) when there is no shared factor greater than 1 -- including
    the single-ratio case, where the "common factor" is the ratio itself, so a
    lone leg with ratio 3 reduces to ratio 1 with common_gcd 3 (i.e. that ratio
    is folded into quantity by the caller instead of staying a per-leg multiplier).
    """
    clean = [max(1, int(r)) for r in ratios]
    if not clean:
        return clean, 1
    common = clean[0]
    for r in clean[1:]:
        common = gcd(common, r)
    if common <= 1:
        return clean, 1
    return [r // common for r in clean], common
