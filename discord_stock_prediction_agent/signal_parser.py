"""Parse Discord stock signals into normalized actions and tickers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .signal_normalizer import normalize_signal_input
from .stock_order_intent import parse_stock_order
from .symbol_directory import is_symbol_like, resolve_cached_symbol


SYMBOL_ALIASES = {
    # Most common user-friendly names.
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "FACEBOOK": "META",
    "META": "META",
    "TESLA": "TSLA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "NETFLIX": "NFLX",
    "PALANTIR": "PLTR",
    "AMD": "AMD",
    "ORACLE": "ORCL",
    "LULU": "LULU",
    "LULULEMON": "LULU",
    "LULULEMON ATHLETICA": "LULU",
    "LULU": "LULU",
    "LULULEMON": "LULU",
    "LULULEMON ATHLETICA": "LULU",
    "DELL": "DELL",
    "DELL TECHNOLOGIES": "DELL",
    "FORD": "F",
    "GENERAL MOTORS": "GM",
    "GM": "GM",
    "INTEL": "INTC",
    "QUALCOMM": "QCOM",
    "CISCO": "CSCO",
    "ADOBE": "ADBE",
    "PAYPAL": "PYPL",
    # BLOCK/SQUARE/TARGET intentionally omitted -- see PHRASE_ALIASES below for why.
    "UBER": "UBER",
    "LYFT": "LYFT",
    "WALMART": "WMT",
    "HOME DEPOT": "HD",
    "LOWES": "LOW",
    "MCDONALDS": "MCD",
    "STARBUCKS": "SBUX",
    "COCA COLA": "KO",
    "COKE": "KO",
    "PEPSI": "PEP",
    "DISNEY": "DIS",
    "BOEING": "BA",
    "LOCKHEED": "LMT",
    "RAYTHEON": "RTX",
    "VISA": "V",
    "MASTERCARD": "MA",
    "AMERICAN EXPRESS": "AXP",
    "BANK OF AMERICA": "BAC",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "MORGAN STANLEY": "MS",
    "GOLDMAN SACHS": "GS",
    "WELLS FARGO": "WFC",
    "CITI": "C",
    "CITIGROUP": "C",
    "EXXON": "XOM",
    "CHEVRON": "CVX",
    "OCCIDENTAL": "OXY",
    "PFIZER": "PFE",
    "MODERNA": "MRNA",
    "MERCK": "MRK",
    "ELI LILLY": "LLY",
    "UNITEDHEALTH": "UNH",
    "JOHNSON AND JOHNSON": "JNJ",
    "ABBVIE": "ABBV",
    "AT&T": "T",
    "ATT": "T",
    "VERIZON": "VZ",
    "T MOBILE": "TMUS",
    "COMCAST": "CMCSA",
    "ROKU": "ROKU",
    "ZOOM": "ZM",
    "DOCUSIGN": "DOCU",
    "SNOWFLAKE": "SNOW",
    "CROWDSTRIKE": "CRWD",
    "PALO ALTO": "PANW",
    "DATADOG": "DDOG",
    "CLOUDFLARE": "NET",
    "AIRBNB": "ABNB",
    "DOORDASH": "DASH",
    "RIVIAN": "RIVN",
    "LUCID": "LCID",
    "NIO": "NIO",
    "XPENG": "XPEV",
    "LI AUTO": "LI",
    "IBM": "IBM",
    "BROADCOM": "AVGO",
    "COSTCO": "COST",
    "COINBASE": "COIN",
    "SHOPIFY": "SHOP",
    "SALESFORCE": "CRM",
    "SNAPCHAT": "SNAP",
    "SNAP": "SNAP",
    "SMCI": "SMCI",
    "SUPERMICRO": "SMCI",
    "SUPER": "SMCI",
    "BERKSHIRE": "BRK.B",
    "BRK.B": "BRK.B",
    "BRKB": "BRK.B",
    "BRK.A": "BRK.A",
    "BRKA": "BRK.A",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "NASDAQ": "QQQ",
    "DOW": "DIA",
    "DIA": "DIA",
    "GOLD": "GLD",
    "BTC": "BTCUSD",
    "BITCOIN": "BTCUSD",
    "BTCUSDT": "BTCUSD",
    "ETH": "ETHUSD",
    "ETHEREUM": "ETHUSD",
    "EURUSD": "EURUSD",
}

PHRASE_ALIASES = {
    "GOOGLE": "GOOGL",
    "GOOGLE STOCK": "GOOGL",
    "ALPHABET": "GOOGL",
    "ALPHABET INC": "GOOGL",
    "APPLE": "AAPL",
    "APPLE STOCK": "AAPL",
    "APPLE INC": "AAPL",
    "MICROSOFT": "MSFT",
    "MICROSOFT STOCK": "MSFT",
    "MICROSOFT CORPORATION": "MSFT",
    "AMAZON": "AMZN",
    "AMAZON STOCK": "AMZN",
    "AMAZON COM": "AMZN",
    "NVIDIA": "NVDA",
    "NVIDIA STOCK": "NVDA",
    "DELL": "DELL",
    "DELL STOCK": "DELL",
    "DELL TECHNOLOGIES": "DELL",
    "FORD": "F",
    "FORD STOCK": "F",
    "FORD MOTOR": "F",
    "GENERAL MOTORS": "GM",
    "GM STOCK": "GM",
    "INTEL": "INTC",
    "INTEL STOCK": "INTC",
    "QUALCOMM": "QCOM",
    "CISCO": "CSCO",
    "ADOBE": "ADBE",
    "PAYPAL": "PYPL",
    # BLOCK/SQUARE/TARGET are deliberately omitted here even though they're
    # real company names (Block Inc, Square, Target Corp) -- they're also
    # extremely common trading-signal vocabulary ("square off my position",
    # "block this trade", "hit my target"), so treating the bare word as a
    # company reference misreads the intended symbol far more often than it
    # helps. Users can still reference these by their literal ticker (SQ/TGT).
    "WALMART": "WMT",
    "HOME DEPOT": "HD",
    "LOWES": "LOW",
    "MCDONALDS": "MCD",
    "STARBUCKS": "SBUX",
    "COCA COLA": "KO",
    "COKE": "KO",
    "PEPSI": "PEP",
    "DISNEY": "DIS",
    "BOEING": "BA",
    "LOCKHEED": "LMT",
    "RAYTHEON": "RTX",
    "VISA": "V",
    "MASTERCARD": "MA",
    "AMERICAN EXPRESS": "AXP",
    "MORGAN STANLEY": "MS",
    "GOLDMAN SACHS": "GS",
    "WELLS FARGO": "WFC",
    "CITI": "C",
    "CITIGROUP": "C",
    "EXXON": "XOM",
    "CHEVRON": "CVX",
    "OCCIDENTAL": "OXY",
    "PFIZER": "PFE",
    "MODERNA": "MRNA",
    "MERCK": "MRK",
    "ELI LILLY": "LLY",
    "UNITEDHEALTH": "UNH",
    "ABBVIE": "ABBV",
    "AT&T": "T",
    "ATT": "T",
    "VERIZON": "VZ",
    "T MOBILE": "TMUS",
    "COMCAST": "CMCSA",
    "ROKU": "ROKU",
    "ZOOM": "ZM",
    "DOCUSIGN": "DOCU",
    "SNOWFLAKE": "SNOW",
    "CROWDSTRIKE": "CRWD",
    "PALO ALTO": "PANW",
    "DATADOG": "DDOG",
    "CLOUDFLARE": "NET",
    "AIRBNB": "ABNB",
    "DOORDASH": "DASH",
    "RIVIAN": "RIVN",
    "LUCID": "LCID",
    "NIO": "NIO",
    "XPENG": "XPEV",
    "LI AUTO": "LI",
    "TESLA": "TSLA",
    "TESLA STOCK": "TSLA",
    "META PLATFORMS": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "PALANTIR": "PLTR",
    "ADVANCED MICRO DEVICES": "AMD",
    "BANK OF AMERICA": "BAC",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "JOHNSON AND JOHNSON": "JNJ",
    "BERKSHIRE HATHAWAY": "BRK.B",
    "SUPER MICRO": "SMCI",
    "SUPER MICRO COMPUTER": "SMCI",
}

COMMON_TICKERS = """
AAPL MSFT NVDA AMZN GOOGL GOOG META TSLA AVGO BRK.B BRK.A JPM LLY V XOM UNH MA COST HD PG
NFLX JNJ BAC ABBV KO CRM ORCL MRK CVX WFC AMD ADBE CSCO ACN MCD IBM QCOM GE INTC DIS VZ T
CMCSA PEP AMAT TXN CAT DHR RTX SPGI PFE LOW HON AMGN NEE BKNG PM UPS TMO BA GS MS BLK NOW
INTU ISRG SBUX DE LMT MDT PLD ADP GILD MDLZ TJX ADI PANW MU COP C AMT CB SYK EL REGN MMC
VRTX SCHW SO BMY KLAC LRCX ZTS FI PGR ETN BSX CI CME MO EQIX PYPL SNPS CDNS AON SLB APD
DUK WM EOG HCA ITW NOC USB CL CSX MCK EMR GD TGT FCX NSC GM F DELL HPQ HPE WMT TGT COST
ORCL CRM SHOP SQ UBER LYFT ABNB DASH ROKU ZM DOCU SNOW CRWD DDOG NET PLTR COIN HOOD RIVN
LCID NIO XPEV LI SMCI ARM MRVL ON ENPH FSLR RUN SEDG TSN DAL UAL AAL LUV CCL RCL NCLH
BA LMT RTX NOC GD XOM CVX OXY COP SLB HAL BP SHEL TTE PBR VALE BHP RIO FCX GOLD NEM
JPM BAC WFC C GS MS USB SCHW AXP V MA PYPL SOFI UPST AFRM
PFE MRNA MRK LLY UNH JNJ ABBV BMY GILD AMGN CVS HUM TMO ISRG SYK MDT BSX
SPY QQQ DIA IWM GLD SLV TLT HYG BTCUSD ETHUSD
""".split()

SYMBOL_ALIASES.update({ticker: ticker for ticker in COMMON_TICKERS})

NON_TRADABLE_SYMBOL_WORDS = {
    # WWE and UFC are brands/events under TKO Group Holdings. The tradable
    # public ticker is TKO, so these should not be accepted as symbols.
    "WWE",
    "UFC",
}

PHRASE_ALIAS_BLOCKLIST = {
    # Common trade-instruction words that also happen to be company aliases.
    # Users can still trade these with the ticker itself, e.g. TGT.
    "TARGET",
}

ACTION_WORDS = {
    "BUY": "BUY",
    "LONG": "BUY",
    "ENTER": "BUY",
    "INVEST": "BUY",
    "ACCUMULATE": "BUY",
    "SELL": "SELL",
    "SHORT": "SELL",
    "EXIT": "SELL",
    "CLOSE": "SELL",
    "BOOK": "SELL",
    "REDUCE": "SELL",
    "TRIM": "SELL",
    "HOLD": "HOLD",
    "WATCH": "HOLD",
    "WAIT": "HOLD",
}

BUY_HINTS = (
    "STRONG BULLISH", "BULLISH", "BREAKOUT", "BREAKS RESISTANCE",
    "GOLDEN CROSS", "GOLDEN CROSSOVER", "CUP AND HANDLE", "HIGHER HIGHS",
    "UPTREND", "ASCENDING TRIANGLE", "MACD BULLISH", "BULLISH CROSSOVER",
    "BULLISH DIVERGENCE", "BULLISH ENGULFING", "MORNING STAR",
    "HAMMER", "DOUBLE BOTTOM", "INVERSE HEAD AND SHOULDERS",
    "ABOVE 200 EMA", "ABOVE KEY MOVING AVERAGES", "BOUNCING",
    "BOUNCE FROM", "RECOVERING", "BUYERS DEFENDING", "SUPPORT HOLDING",
    "SUPPORT RETEST SUCCESSFUL", "RETEST SUCCESSFUL", "ACCUMULATION",
    "INSTITUTIONAL BUYING", "CALL BUYING", "POSITIVE ANALYST",
    "ANALYST UPGRADE", "POSITIVE FORWARD GUIDANCE", "BEATS EARNINGS",
    "NEW AI PRODUCT", "PATENT APPROVAL", "RISK-ON", "TECHNOLOGY SECTOR STRONG",
    "RETAIL BUYING", "TREND ACCELERATING", "MOMENTUM INCREASING",
    "REMAINS POSITIVE", "BUYING AFTER DIP", "RETRACING TO", "OVERSOLD",
    "RSI CROSSING ABOVE", "MACD HISTOGRAM INCREASING", "ABOVE 20 50 200 EMA",
)

SELL_HINTS = (
    "STRONG BEARISH", "BEARISH", "BREAKDOWN", "BREAKS MAJOR SUPPORT",
    "DEATH CROSS", "HEAVY SELLING", "PANIC VOLUME", "SELLING PRESSURE",
    "LOWER LOWS", "REJECTED FROM RESISTANCE", "REJECTED", "LOSING MOMENTUM",
    "WEAK CLOSE", "DAILY LOWS", "RSI FALLING", "BEARISH DIVERGENCE",
    "DECREASING BUYING VOLUME", "FAILING TO MAKE NEW HIGHS", "FAILED IMMEDIATELY",
    "FALSE BREAKOUT", "TRAP BREAKOUT", "LACKS FOLLOW THROUGH", "SUPPORT COLLAPSE",
    "NEW 3-MONTH LOW", "OVERBOUGHT", "MACD BEARISH", "BEARISH CROSSOVER",
    "BEARISH ENGULFING", "SHOOTING STAR", "EVENING STAR", "THREE BLACK CROWS",
    "BELOW ALL MOVING AVERAGES", "FALLING DESPITE", "VOLUME DRYING UP",
    "MISSES EARNINGS", "WEAK QUARTERLY", "NEGATIVE ANALYST", "ANALYST DOWNGRADE",
    "CEO RESIGNS", "GOVERNMENT INVESTIGATION", "PUT BUYING", "RISK-OFF",
    "SEMICONDUCTOR WEAKNESS", "INSTITUTIONAL DISTRIBUTION", "GAP DOWN",
    "FLASH CRASH", "CIRCUIT BREAKER", "TRADING HALT", "MARKET CRASHING",
    "INSTITUTIONAL SELLING", "MOMENTUM SLOWING", "CLOSES BELOW TRENDLINE",
    "NEW 3 MONTH LOW", "TREND WEAKENING", "STOCK FALLING",
)

HOLD_HINTS = (
    "HOLD", "WATCH", "WAIT", "CONSOLIDATING", "SIDEWAYS", "NO CLEAR TREND",
    "MIXED", "WAITING", "LOW VOLATILITY", "RANGE BOUND", "INDECISIVE",
    "DOJI", "SPINNING TOP", "NEAR ZERO LINE", "FLATTENING", "EARNINGS TOMORROW",
    "HIGH VOLATILITY EXPECTED", "NOT SURE", "MAYBE", "COULD REVERSE",
    "POSSIBLE BREAKOUT", "RISKY", "WEAK VOLUME", "CONTRADICTORY",
    "TREND REVERSAL CONFIRMED",
)

HIGH_RISK_HINTS = (
    "RSI 88", "RSI 82", "OVERBOUGHT", "MARKET CRASHING", "STRONG EARNINGS BUT STOCK FALLING",
    "OVERSOLD RSI BUT HEAVY", "GAP UP", "GAP DOWN", "TRADING HALT", "CIRCUIT BREAKER",
    "FLASH CRASH", "HIGH IMPLIED VOLATILITY", "VERY HIGH VOLATILITY",
)

UNCERTAIN_HINTS = (
    "NOT SURE", "MAYBE", "COULD REVERSE", "POSSIBLE BREAKOUT", "RISKY",
    "WEAK VOLUME", "CONTRADICTORY",
)


@dataclass(frozen=True)
class ParsedSignal:
    valid: bool
    action: str = ""
    symbol: str = ""
    quantity: Optional[float] = None
    raw_text: str = ""
    reason: str = ""
    condition_type: str = ""
    condition_price: Optional[float] = None
    order_type: str = "market"
    order_intent: Optional[dict] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"


def _normalize_symbol(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", value or "").upper().replace("-", ".").strip(".")
    return SYMBOL_ALIASES.get(cleaned) or resolve_cached_symbol(cleaned) or cleaned


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9.]+", " ", (text or "").upper())).strip()


def _infer_action(text: str) -> str:
    normalized = _normalized_text(text)
    if re.search(r"\bBUY\s+TO\s+COVER\b|\bCOVER(?:ING)?\b", normalized):
        return "BUY_TO_COVER"
    if re.search(r"\b(?:SHORT\s+SELL|SELL\s+SHORT|SHORT(?:ING)?)\b", normalized):
        return "SELL_SHORT"
    explicit = ""
    for word, mapped in ACTION_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            explicit = mapped
            break
    if explicit:
        return explicit

    buy_score = sum(1 for hint in BUY_HINTS if hint in normalized)
    sell_score = sum(1 for hint in SELL_HINTS if hint in normalized)
    hold_score = sum(1 for hint in HOLD_HINTS if hint in normalized)
    high_risk = any(hint in normalized for hint in HIGH_RISK_HINTS)
    uncertain = any(hint in normalized for hint in UNCERTAIN_HINTS)

    if uncertain and (buy_score or sell_score or hold_score):
        return "HOLD"
    if high_risk and (buy_score or sell_score):
        return "HOLD"
    if buy_score and sell_score:
        return "HOLD"
    if hold_score and not buy_score and not sell_score:
        return "HOLD"
    if buy_score > sell_score:
        return "BUY"
    if sell_score > buy_score:
        return "SELL"
    return ""


def _extract_quantity(text: str) -> Optional[float]:
    patterns = [
        r"\bqty\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
        r"\bquantity\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
        r"\bshares?\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
        r"\b(\d+(?:\.\d+)?)\s*(?:shares?|qty)\b",
        r"\b(?:BUY|SELL|SHORT|COVER)\s+(\d+(?:\.\d+)?)\s+[A-Z.]{1,12}\b",
        r"\bBUY\s+TO\s+COVER\s+(\d+(?:\.\d+)?)\s+[A-Z.]{1,12}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            qty = float(match.group(1))
            return qty if qty > 0 else None
    return None


def _is_conditional_setup(text: str) -> bool:
    normalized = _normalized_text(text)
    has_condition_word = any(re.search(rf"\b{word}\b", normalized) for word in ("IF", "UNLESS", "ONLY", "OTHERWISE"))
    has_trigger = any(
        phrase in normalized
        for phrase in (
            "CLOSES ABOVE", "CLOSE ABOVE", "ABOVE", "FALLS BELOW", "FALL BELOW",
            "BREAKS ABOVE", "BREAK ABOVE", "BREAKS BELOW", "BREAK BELOW",
            "OPENS ABOVE", "OPENS BELOW", "ONLY ABOVE", "ONLY BELOW",
        )
    )
    return has_condition_word and has_trigger



def _extract_price_condition(text: str, action: str) -> tuple[str, Optional[float]]:
    """Extract executable price instructions from equity signals."""
    raw = text or ""
    normalized = _normalized_text(raw)

    patterns = [
        ("close_above", r"\b(?:PRICE\s+)?CLOSES?\s+ABOVE\s*\$?\s*(\d+(?:\.\d+)?)\b"),
        ("close_below", r"\b(?:PRICE\s+)?CLOSES?\s+BELOW\s*\$?\s*(\d+(?:\.\d+)?)\b"),
        ("below", r"\b(?:PRICE\s+)?(?:FALLS?|DROPS?|BREAKS?|GOES?|MOVES?)\s+BELOW\s*\$?\s*(\d+(?:\.\d+)?)\b"),
        ("above", r"\b(?:PRICE\s+)?(?:BREAKS?|GOES?|MOVES?)\s+ABOVE\s*\$?\s*(\d+(?:\.\d+)?)\b"),
        ("below", r"\bBELOW\s*\$?\s*(\d+(?:\.\d+)?)\b"),
        ("above", r"\bABOVE\s*\$?\s*(\d+(?:\.\d+)?)\b"),
    ]
    for kind, pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return kind, float(match.group(1))

    return "", None


def _extract_order_details(text: str) -> tuple[str, Optional[float], Optional[float], str]:
    raw = text or ""
    upper = _normalized_text(raw)
    tif_match = re.search(r"\b(GTC|DAY|IOC|FOK)\b", upper)
    tif = tif_match.group(1) if tif_match else "DAY"
    stop_match = re.search(r"\bSTOP\s*[:@]?\s*\$?\s*(\d+(?:\.\d+)?)\b", raw, re.IGNORECASE)
    limit_match = re.search(r"\bLIMIT\s*[:@]?\s*\$?\s*(\d+(?:\.\d+)?)\b", raw, re.IGNORECASE)
    stop_price = float(stop_match.group(1)) if stop_match else None
    limit_price = float(limit_match.group(1)) if limit_match else None
    if stop_price is not None and limit_price is not None:
        return "stop_limit", limit_price, stop_price, tif
    if stop_price is not None:
        return "stop", None, stop_price, tif
    if limit_price is not None:
        return "limit", limit_price, None, tif
    at_match = re.search(r"(?:@|\bAT\b)\s*\$?\s*(\d+(?:\.\d+)?)\b", raw, re.IGNORECASE)
    if at_match and "MARKET" not in upper:
        return "limit", float(at_match.group(1)), None, tif
    return "market", None, None, tif

def _extract_symbol(text: str) -> str:
    normalized = _normalized_text(text)
    has_explicit_action = any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in ACTION_WORDS)
    for phrase, symbol in sorted(PHRASE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase in PHRASE_ALIAS_BLOCKLIST:
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return symbol

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.\-]{0,12}", text or "")
    known_symbols = set(SYMBOL_ALIASES.values()) | set(SYMBOL_ALIASES.keys())
    ignored = set(ACTION_WORDS.keys()) | {
        "NOW", "ABOVE", "BELOW", "ENTRY", "TARGET", "TARGETS", "SL", "STOP",
        "LOSS", "PRICE", "MARKET", "LIMIT", "SHARE", "SHARES", "QTY", "QUANTITY",
        "CLOSE", "BOOK", "PROFIT", "REDUCE", "TRIM", "POSITION", "PARTIAL",
        "PLEASE", "CAN", "YOU", "THE", "A", "AN", "AT", "TO", "FOR", "OF", "IF", "WITH",
        "BUT", "AND", "OR", "AS", "IN", "ON", "BY", "IS", "ARE", "THIS", "THAT",
        "WHILE", "UNLESS", "UNTIL", "YET", "WHEN", "THEN", "ELSE", "IT", "ITS",
        "LOOKS", "LOOK", "POSSIBLE", "MOMENTUM", "COULD", "REVERSE", "SOON",
        "NOT", "SURE", "MAY", "MIGHT", "MAYBE",
        "COVER", "GTC", "DAY", "IOC", "FOK",
        "WE", "NEED", "OPTION", "OPTIONS", "STRIKE", "RATE", "EXPIRY", "EXPIRES",
        "EXPIRATION", "SAME", "DAY", "PREMIUM",
        "HIGH", "LOW", "VOLUME", "RSI", "MACD", "EMA", "SMA", "ATR", "CONFIDENCE",
        "TREND", "BULLISH", "BEARISH", "BREAKOUT", "BREAKDOWN", "SUPPORT",
        "RESISTANCE", "DAILY", "WEEKLY", "MONTHLY", "STRONG", "MILD", "MODERATE",
        "TARGET", "STOP", "CONFIRMED", "CROSSOVER", "CROSS", "CANDLE", "FROM",
        "AFTER", "BEFORE", "TOMORROW", "TODAY", "MAYBE", "WAIT", "RISKY",
        "OPEN", "NEAR", "DECISION", "MAKING", "WATCHING", "CONSOLIDATING",
        "CPI", "VIX", "OBV", "VWAP", "ADX", "CCI", "ROC", "AWS", "EMA",
        "FED", "FEDERAL", "RESERVE", "TREASURY", "YIELDS", "DOLLAR", "INDEX",
        "OIL", "AIRLINE", "AIRLINES", "GLOBAL", "MARKETS", "MARKET", "BREADTH",
        "R", "MIXED", "TECHNICAL", "TECHNICALS", "INDICATOR", "INDICATORS", "EARNINGS",
        # Ordinary trading-signal syntax words that are also real tickers or
        # company names (GO=Grocery Outlet, MY, OUT=Outfront Media, HALF,
        # ENTIRE, THINK, PICK) -- without excluding them, casual phrasing like
        # "close out amzn" or "go long on sofi" resolves to the wrong symbol.
        "GO", "MY", "OUT", "HALF", "ENTIRE", "THINK", "PICK", "SOME", "ALL", "UP",
    }
    for token in tokens:
        token_key = re.sub(r"[^A-Za-z0-9.\-]", "", token or "").upper().replace("-", ".").strip(".")
        if token_key in ignored:
            continue
        candidate = _normalize_symbol(token)
        if len(candidate) == 1 and token != token.upper():
            continue
        if candidate in NON_TRADABLE_SYMBOL_WORDS:
            continue
        if candidate in ignored:
            continue
        cached = resolve_cached_symbol(candidate)
        if cached:
            return cached
        if candidate in known_symbols:
            return SYMBOL_ALIASES.get(candidate, candidate)
        if token == token.upper() and is_symbol_like(candidate):
            return candidate
        if has_explicit_action and is_symbol_like(candidate, max_len=5):
            return candidate
    return ""


def parse_signal(message: str) -> ParsedSignal:
    raw = message or ""
    compact = normalize_signal_input(raw)
    if not compact:
        return ParsedSignal(False, raw_text=raw, reason="Empty message.")

    rich_intent = parse_stock_order(compact)
    if rich_intent and rich_intent.get("status") == "INVALID_OR_NON_EXECUTABLE":
        return ParsedSignal(
            valid=False,
            raw_text=raw,
            reason="; ".join(str(issue) for issue in rich_intent.get("issues", [])),
            order_intent=rich_intent,
        )
    if rich_intent and rich_intent.get("asset_type") == "STOCK":
        action = str(rich_intent.get("action") or "").upper()
        if not action and isinstance(rich_intent.get("actions"), list):
            action = next(
                (
                    str(item.get("action") or "").upper()
                    for item in rich_intent["actions"]
                    if str(item.get("action") or "").upper() != "CANCEL_OPEN_ORDERS"
                ),
                "",
            )
        symbol = str(rich_intent.get("symbol") or "").upper()
        if not symbol and isinstance(rich_intent.get("actions"), list):
            symbol = next(
                (str(item.get("symbol") or "").upper() for item in rich_intent["actions"] if item.get("symbol")),
                "",
            )
        quantity = rich_intent.get("quantity")
        if quantity is None:
            quantity = rich_intent.get("initial_quantity", rich_intent.get("total_quantity"))
        condition_type, condition_price = _extract_price_condition(compact, action)
        parsed_order_type = str(rich_intent.get("order_type") or "MARKET").lower()
        parsed_limit_price = rich_intent.get("limit_price")
        parsed_stop_price = rich_intent.get("stop_price")
        # A close-confirmation condition is observed by the monitor. It is not an
        # Alpaca stop order because the intraday price may cross before the close.
        if condition_type in {"close_above", "close_below"}:
            parsed_order_type = "market"
            parsed_limit_price = None
            parsed_stop_price = None
        return ParsedSignal(
            valid=bool(action and symbol),
            action=action,
            symbol=symbol,
            quantity=float(quantity) if quantity is not None else None,
            raw_text=raw,
            reason="Rich stock order parsed; every normalized field is attached as order_intent.",
            condition_type=condition_type,
            condition_price=condition_price,
            order_type=parsed_order_type,
            order_intent=rich_intent,
            limit_price=(
                float(parsed_limit_price)
                if parsed_limit_price is not None
                else None
            ),
            stop_price=(
                float(parsed_stop_price)
                if parsed_stop_price is not None
                else None
            ),
            time_in_force=str(rich_intent.get("time_in_force") or "DAY").upper(),
        )

    action = _infer_action(compact)
    if not action:
        return ParsedSignal(False, raw_text=raw, reason="No BUY, SELL, or HOLD action found.")

    symbol = _extract_symbol(compact)
    if not symbol:
        return ParsedSignal(False, action=action, raw_text=raw, reason="No stock symbol found.")

    quantity = _extract_quantity(compact)
    condition_type, condition_price = _extract_price_condition(compact, action)
    order_type, limit_price, stop_price, time_in_force = _extract_order_details(compact)
    reason = ""
    if action in {"BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"} and condition_type and condition_price:
        reason = (
            f"Price condition detected: {condition_type.replace('_', ' ')} "
            f"${condition_price:g}. The agent will watch this condition before paper trading."
        )

    return ParsedSignal(
        valid=True,
        action=action,
        symbol=symbol,
        quantity=quantity,
        raw_text=raw,
        reason=reason,
        condition_type=condition_type,
        condition_price=condition_price,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
    )
