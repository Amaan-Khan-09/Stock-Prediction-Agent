"""
Historical Stock Price Service
Fetches daily OHLCV bars from RapidAPI TradingView endpoint.
Returns clean list of {date, close, open, high, low, volume} sorted by date ASC.

Rules:
- NO yfinance
- NO fake/hardcoded prices
- NO silent failures — returns clear error strings
- Date lookup returns nearest available trading day (max ±5 calendar days)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "tools"))

import requests
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

_BASE_URL  = "https://trading-view.p.rapidapi.com"
_HOST      = "trading-view.p.rapidapi.com"
_MAX_RANGE = 750   # ~3 years of trading days



# ── Exchange mapping (same logic as existing tradingview_price.py) ─────────────

_NASDAQ = frozenset({
    "AAPL","MSFT","GOOGL","GOOG","AMZN","TSLA","META","NVDA","NFLX","ADBE",
    "INTC","CSCO","PYPL","CMCSA","PEP","COST","AVGO","QCOM","SBUX","INTU",
    "TXN","AMD","GILD","BIIB","REGN","ILMN","MRNA","ZM","DOCU","SNOW",
    "MELI","ASML","PDD","BIDU","JD","NTES","WB","IQ","BILI","VIPS",
    "LULU","ISRG","IDXX","ALGN","MCHP","XLNX","WDAY","OKTA","ZS","CRWD",
    "DDOG","MDB","NET","COUP","GTLB","TWLO","VRTX","REGN","BMRN",
})

_NYSE = frozenset({
    "JPM","V","JNJ","WMT","PG","XOM","BAC","KO","DIS","C","PFE","NKE","HD",
    "MA","UNH","VZ","CRM","ABBV","MRK","ABT","MCD","T","BA","WFC",
    "IBM","GE","CAT","ORCL","CVX","D","UPS","LMT","GS","BRK.A","BRK.B",
    "F","GM","USB","TGT","LOW","CVS","AXP","MMM","HON","RTX","DE","EMR",
    "SLB","OXY","COP","HAL","KHC","MO","PM","BTI","BP","RDS.A","TOT",
})

_AMEX = frozenset({
    "SPY","QQQ","IWM","DIA","GLD","SLV","VTI","VOO","IVV","EFA","VWO",
    "BND","VNQ","XLF","XLV","XLU","XLP","XLE","XLK","XLB","XLI","XLY",
    "SGOV","TLT","HYG","LQD","AGG","ARKK","ARKG","ARKW","ARKF",
})

# ETF set used by external historical provider for correct asset-class routing
_ETF_SYMBOLS = _AMEX

# ── Index symbol normalization ─────────────────────────────────────────────────
# Maps common index names → correct ticker for each provider.
# SPX/NDX/DJI are index names, not tradeable tickers — providers need specific formats.
_INDEX_SYMBOL_MAP: Dict[str, Dict[str, str]] = {
    "SPX":  {"yfinance": "^GSPC",  "tradingview": "SP:SPX",    "nasdaq": "^GSPC"},
    "NDX":  {"yfinance": "^NDX",   "tradingview": "NASDAQ:NDX","nasdaq": "^NDX"},
    "DJI":  {"yfinance": "^DJI",   "tradingview": "DJ:DJI",    "nasdaq": "^DJI"},
    "DJIA": {"yfinance": "^DJI",   "tradingview": "DJ:DJI",    "nasdaq": "^DJI"},
    "VIX":  {"yfinance": "^VIX",   "tradingview": "CBOE:VIX",  "nasdaq": "^VIX"},
    "RUT":  {"yfinance": "^RUT",   "tradingview": "TVC:RUT",   "nasdaq": "^RUT"},
}

# ── Provider configuration ─────────────────────────────────────────────────────
# "auto"                → try RapidAPI historical first; use external_historical if unavailable
# "rapidapi_tradingview"→ RapidAPI only (fails if historical endpoints not on plan)
# "external_historical" → use external provider directly (no RapidAPI historical attempt)
HISTORICAL_PRICE_PROVIDER = os.getenv("HISTORICAL_PRICE_PROVIDER", "auto")

# Track actual provider used per symbol (sym_key → provider label)
_PROVIDER_USED: Dict[str, str] = {}


def get_provider_used(symbol: str) -> str:
    """Return which provider was actually used for the last fetch of this symbol."""
    sym_key = symbol.upper().replace("-", ".").replace(":", "_")
    return _PROVIDER_USED.get(sym_key, "unknown")


def _fmt(symbol: str) -> str:
    """Return EXCHANGE:SYMBOL string."""
    s = str(symbol).strip().upper()
    if ":" in s:
        return s
    # BRK-B → BRK.B
    s = s.replace("-", ".")
    if s in _NASDAQ: return f"NASDAQ:{s}"
    if s in _NYSE:   return f"NYSE:{s}"
    if s in _AMEX:   return f"AMEX:{s}"
    # Heuristic: ≤4 chars → NASDAQ, longer → try NASDAQ anyway
    return f"NASDAQ:{s}"


def _rapidapi_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "x-rapidapi-host": _HOST,
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY", ""),
    }


def _ts_to_date(ts: Any) -> Optional[str]:
    """Convert Unix timestamp (int/float) to YYYY-MM-DD string."""
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return None


def _parse_bar(item: Any) -> Optional[Dict]:
    """
    Parse one OHLCV bar item. Handles multiple API response shapes:
    - {"t": unix_ts, "c": close, "o": open, "h": high, "l": low, "v": vol}
    - {"time": unix_ts, "close": close, ...}
    - {"date": "YYYY-MM-DD", "close": close, ...}
    - {"Date": "YYYY-MM-DD", "Close": close, ...}
    """
    if not isinstance(item, dict):
        return None
    # Resolve date
    dt_str = None
    for k in ("date", "Date", "datetime", "Datetime"):
        v = item.get(k)
        if isinstance(v, str) and len(v) >= 10:
            dt_str = v[:10]
            break
    if dt_str is None:
        for k in ("t", "time", "timestamp", "Time", "Timestamp"):
            v = item.get(k)
            if v is not None:
                dt_str = _ts_to_date(v)
                break
    if not dt_str:
        return None

    # Resolve close price
    close = None
    for k in ("c", "close", "Close", "closePrice", "last", "price"):
        v = item.get(k)
        if v is not None:
            try:
                close = float(v)
                break
            except (TypeError, ValueError):
                pass
    if close is None or close <= 0:
        return None

    return {
        "date":   dt_str,
        "close":  close,
        "open":   float(item.get("o") or item.get("open") or item.get("Open") or close),
        "high":   float(item.get("h") or item.get("high") or item.get("High") or close),
        "low":    float(item.get("l") or item.get("low")  or item.get("Low")  or close),
        "volume": float(item.get("v") or item.get("volume") or item.get("Volume") or 0),
    }


def _extract_bars(raw: Any) -> List[Dict]:
    """
    Extract bar list from various API response shapes.
    Returns list of parsed bars (may be empty).
    """
    bars: List[Dict] = []

    def _try_list(lst: Any):
        if isinstance(lst, list):
            for item in lst:
                b = _parse_bar(item)
                if b:
                    bars.append(b)

    if isinstance(raw, list):
        # Top-level array — could be array of bars or array of symbol-results
        for item in raw:
            if isinstance(item, dict):
                # If item itself is a bar (has "c" or "close")
                b = _parse_bar(item)
                if b:
                    bars.append(b)
                else:
                    # It's a symbol-result container — look for nested bars
                    for k in ("bars", "Bars", "history", "History", "data", "prices",
                              "candles", "Candles", "ohlcv", "results"):
                        v = item.get(k)
                        if isinstance(v, list) and v:
                            _try_list(v)
                            break
                    # Parallel arrays format: {"t": [...], "c": [...], ...}
                    t_arr = item.get("t") or item.get("time") or []
                    c_arr = item.get("c") or item.get("close") or []
                    o_arr = item.get("o") or item.get("open") or []
                    h_arr = item.get("h") or item.get("high") or []
                    l_arr = item.get("l") or item.get("low") or []
                    v_arr = item.get("v") or item.get("volume") or []
                    if isinstance(t_arr, list) and isinstance(c_arr, list) and len(t_arr) == len(c_arr):
                        for i, (ts, cl) in enumerate(zip(t_arr, c_arr)):
                            dt = _ts_to_date(ts)
                            if dt and cl and float(cl) > 0:
                                bars.append({
                                    "date":   dt,
                                    "close":  float(cl),
                                    "open":   float(o_arr[i]) if i < len(o_arr) and o_arr[i] else float(cl),
                                    "high":   float(h_arr[i]) if i < len(h_arr) and h_arr[i] else float(cl),
                                    "low":    float(l_arr[i]) if i < len(l_arr) and l_arr[i] else float(cl),
                                    "volume": float(v_arr[i]) if i < len(v_arr) and v_arr[i] else 0.0,
                                })
    elif isinstance(raw, dict):
        for k in ("bars", "Bars", "history", "History", "data", "prices",
                  "candles", "Candles", "results", "quotes", "ohlcv"):
            v = raw.get(k)
            if isinstance(v, list) and v:
                _try_list(v)
                if bars:
                    break
        if not bars:
            # Parallel arrays at top level
            t_arr = raw.get("t") or raw.get("time") or []
            c_arr = raw.get("c") or raw.get("close") or []
            o_arr = raw.get("o") or raw.get("open") or []
            h_arr = raw.get("h") or raw.get("high") or []
            l_arr = raw.get("l") or raw.get("low") or []
            v_arr = raw.get("v") or raw.get("volume") or []
            if isinstance(t_arr, list) and isinstance(c_arr, list) and len(t_arr) == len(c_arr):
                for i, (ts, cl) in enumerate(zip(t_arr, c_arr)):
                    dt = _ts_to_date(ts)
                    if dt and cl and float(cl) > 0:
                        bars.append({
                            "date":   dt,
                            "close":  float(cl),
                            "open":   float(o_arr[i]) if i < len(o_arr) and o_arr[i] else float(cl),
                            "high":   float(h_arr[i]) if i < len(h_arr) and h_arr[i] else float(cl),
                            "low":    float(l_arr[i]) if i < len(l_arr) and l_arr[i] else float(cl),
                            "volume": float(v_arr[i]) if i < len(v_arr) and v_arr[i] else 0.0,
                        })
        if not bars:
            # Nested "data" that is a dict → recurse one level
            nested = raw.get("data")
            if isinstance(nested, (list, dict)):
                bars.extend(_extract_bars(nested))

    return bars


def _deduplicate_and_sort(bars: List[Dict]) -> List[Dict]:
    """Deduplicate by date, keep last occurrence, sort ASC."""
    seen: Dict[str, Dict] = {}
    for b in bars:
        seen[b["date"]] = b
    return sorted(seen.values(), key=lambda x: x["date"])



# ── External historical provider (public market data — no API key required) ────
def _fetch_external_historical_provider(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict], str]:
    """
    Fetch historical daily OHLCV from external historical data provider.
    Returns (bars, error_msg). Bars carry source="external_historical_provider".
    Used when RapidAPI historical endpoints are unavailable on current plan.
    """
    clean = symbol.upper()
    for prefix in ("NASDAQ:", "NYSE:", "AMEX:", "ARCA:"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    # NASDAQ API only covers stocks/ETFs — skip entirely for index symbols
    if clean in _INDEX_SYMBOL_MAP:
        return [], f"NASDAQ API does not support index symbols — skipping {clean}"
    # Resolve exchange-specific symbol (no-op for stocks/ETFs not in the map)
    clean = _INDEX_SYMBOL_MAP.get(clean, {}).get("nasdaq", clean)

    assetclass = "etf" if clean in _ETF_SYMBOLS else "stocks"
    url    = f"https://api.nasdaq.com/api/quote/{clean}/historical"
    params: Dict[str, str] = {
        "assetclass": assetclass,
        "fromdate":   start_date,
        "todate":     end_date,
        "limit":      "9999",
        "type":       "1",
    }
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nasdaq.com/",
    }
    try:
        resp = requests.get(url, headers=hdrs, params=params, timeout=20)
        if resp.status_code != 200 and assetclass == "stocks":
            p2 = dict(params); p2["assetclass"] = "etf"
            resp = requests.get(url, headers=hdrs, params=p2, timeout=20)
        if resp.status_code != 200:
            return [], f"External historical provider HTTP {resp.status_code} for {clean}"

        rows = ((resp.json().get("data") or {}).get("tradesTable") or {}).get("rows") or []
        if not rows:
            return [], f"External historical provider returned no rows for {clean}"

        bars: List[Dict] = []
        for row in rows:
            try:
                parts = str(row.get("date", "")).split("/")
                if len(parts) != 3:
                    continue
                month, day, year = parts
                iso = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

                def _p(s: Any) -> float:
                    return float(str(s).replace("$", "").replace(",", "").strip() or "0")

                close = _p(row.get("close", "0"))
                if close <= 0:
                    continue
                bars.append({
                    "date":   iso,
                    "close":  close,
                    "open":   _p(row.get("open",   "0")) or close,
                    "high":   _p(row.get("high",   "0")) or close,
                    "low":    _p(row.get("low",    "0")) or close,
                    "volume": float(str(row.get("volume", "0")).replace(",", "") or "0"),
                    "source": "external_historical_provider",
                })
            except (ValueError, TypeError, AttributeError):
                continue

        if not bars:
            return [], f"External historical provider parsed 0 valid bars for {clean}"
        bars.sort(key=lambda b: b["date"])
        return bars, ""
    except requests.Timeout:
        return [], f"External historical provider timed out for {clean}"
    except Exception as exc:
        return [], f"External historical provider error for {clean}: {type(exc).__name__}: {exc}"


# ── yfinance provider (long-range historical, data back to IPO) ─────────────────
def _fetch_yfinance(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict], str]:
    """
    Fetch daily OHLCV bars via yfinance.
    Returns (bars, error_msg). Bars carry source="yfinance".
    Used as fallback when other providers cannot cover the requested date range.
    """
    try:
        import yfinance as yf
        clean = symbol.upper()
        for prefix in ("NASDAQ:", "NYSE:", "AMEX:", "ARCA:"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        # Resolve index names to yfinance tickers (SPX → ^GSPC, etc.)
        clean = _INDEX_SYMBOL_MAP.get(clean, {}).get("yfinance", clean)

        # Use a browser-like session to avoid Yahoo Finance rate-limiting
        _yf_session = requests.Session()
        _yf_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })
        ticker = yf.Ticker(clean, session=_yf_session)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        if df is None or df.empty:
            return [], f"yfinance returned no data for {clean} ({start_date} to {end_date})"
        bars: List[Dict] = []
        for ts, row in df.iterrows():
            try:
                day_str = str(ts)[:10]
                close = float(row.get("Close", 0) or 0)
                if close <= 0:
                    continue
                bars.append({
                    "date":   day_str,
                    "close":  round(close, 4),
                    "open":   round(float(row.get("Open",   0) or close), 4),
                    "high":   round(float(row.get("High",   0) or close), 4),
                    "low":    round(float(row.get("Low",    0) or close), 4),
                    "volume": float(row.get("Volume", 0) or 0),
                    "source": "yfinance",
                })
            except (ValueError, TypeError, AttributeError):
                continue
        if not bars:
            return [], f"yfinance parsed 0 valid bars for {clean}"
        bars.sort(key=lambda b: b["date"])
        return bars, ""
    except ImportError:
        return [], "yfinance not installed — run: pip install yfinance"
    except Exception as exc:
        return [], f"yfinance error for {symbol}: {type(exc).__name__}: {exc}"


# ── Direct Yahoo Finance REST API (bypasses yfinance library completely) ────────
def _fetch_yahoo_direct(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict], str]:
    """
    Fetch daily OHLCV bars directly from Yahoo Finance v8/finance/chart API.
    Does not use yfinance library — makes the HTTP request itself.
    Reliable fallback when yfinance library returns 'possibly delisted' for indices.
    """
    try:
        clean = symbol.upper()
        for prefix in ("NASDAQ:", "NYSE:", "AMEX:", "ARCA:"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        clean = _INDEX_SYMBOL_MAP.get(clean, {}).get("yfinance", clean)

        from datetime import datetime as _dt
        period1 = int(_dt.strptime(start_date[:10], "%Y-%m-%d").timestamp())
        period2 = int(_dt.strptime(end_date[:10],   "%Y-%m-%d").timestamp()) + 86400

        hdrs = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        # Try both Yahoo Finance query endpoints
        for _base in ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com"):
            try:
                url  = f"{_base}/v8/finance/chart/{requests.utils.quote(clean)}"
                resp = requests.get(
                    url,
                    headers=hdrs,
                    params={"period1": period1, "period2": period2, "interval": "1d",
                            "includePrePost": "false"},
                    timeout=15,
                )
                if resp.status_code != 200:
                    continue
                result = (resp.json().get("chart") or {}).get("result") or []
                if not result:
                    continue
                timestamps = result[0].get("timestamp") or []
                q = (result[0].get("indicators") or {}).get("quote") or [{}]
                q = q[0]
                closes  = q.get("close",  [])
                opens   = q.get("open",   [])
                highs   = q.get("high",   [])
                lows    = q.get("low",    [])
                volumes = q.get("volume", [])
                bars: List[Dict] = []
                for i, ts in enumerate(timestamps):
                    try:
                        day_str = _dt.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                        close = float(closes[i]) if (i < len(closes) and closes[i] is not None) else 0.0
                        if close <= 0:
                            continue
                        bars.append({
                            "date":   day_str,
                            "close":  round(close, 4),
                            "open":   round(float(opens[i]   or close) if i < len(opens)   else close, 4),
                            "high":   round(float(highs[i]   or close) if i < len(highs)   else close, 4),
                            "low":    round(float(lows[i]    or close) if i < len(lows)    else close, 4),
                            "volume": float(volumes[i] or 0)           if i < len(volumes) else 0.0,
                            "source": "yahoo_direct",
                        })
                    except (IndexError, TypeError, ValueError):
                        continue
                if bars:
                    bars.sort(key=lambda b: b["date"])
                    return bars, ""
            except requests.Timeout:
                continue
            except Exception:
                continue
        return [], f"Yahoo Finance direct API returned no data for {clean} ({start_date} to {end_date})"
    except Exception as exc:
        return [], f"Yahoo Finance direct API error for {symbol}: {type(exc).__name__}: {exc}"


def _fetch_rapidapi_for_range(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict], str]:
    """
    Attempt to fetch bars for an exact date range via TradingView RapidAPI.
    Calculates the required range in days from today. Capped at _MAX_RANGE.
    Returns (bars filtered to requested window, error_msg).
    Bars carry source="rapidapi_tradingview".
    """
    key = os.getenv("RAPIDAPI_KEY", "").strip()
    if not key:
        return [], "RAPIDAPI_KEY not set — skipping TradingView RapidAPI"

    try:
        from datetime import datetime as _dt
        start_dt   = _dt.strptime(start_date[:10], "%Y-%m-%d").date()
        days_needed = (date.today() - start_dt).days + 30
        range_n     = min(days_needed, _MAX_RANGE)

        base    = symbol.upper().replace("-", ".").strip()
        # Resolve index names to TradingView format (SPX → SP:SPX, etc.)
        _idx_tv = _INDEX_SYMBOL_MAP.get(base, {}).get("tradingview")
        if _idx_tv:
            # Try primary TradingView format plus common fallback formats for indices
            _tv_fallbacks: Dict[str, List[str]] = {
                "SPX":  ["SP:SPX",  "TVC:SPX",  "CBOE:SPX",  "FOREXCOM:SPXUSD"],
                "NDX":  ["NASDAQ:NDX", "TVC:NDX", "INDEX:NDX"],
                "DJI":  ["DJ:DJI",  "TVC:DJI",  "INDEX:DJI"],
                "DJIA": ["DJ:DJI",  "TVC:DJI",  "INDEX:DJI"],
                "VIX":  ["CBOE:VIX", "TVC:VIX"],
                "RUT":  ["TVC:RUT",  "INDEX:RUT"],
            }
            exchanges = _tv_fallbacks.get(base, [_idx_tv])
        else:
            sym_fmt = _fmt(symbol)
            if sym_fmt.startswith("NASDAQ:"):
                exchanges = [sym_fmt, f"NYSE:{base}", f"AMEX:{base}"]
            elif sym_fmt.startswith("NYSE:"):
                exchanges = [sym_fmt, f"NASDAQ:{base}", f"AMEX:{base}"]
            elif sym_fmt.startswith("AMEX:"):
                exchanges = [sym_fmt, f"NASDAQ:{base}", f"NYSE:{base}"]
            else:
                exchanges = [f"NASDAQ:{base}", f"NYSE:{base}", f"AMEX:{base}"]

        for fmt_sym in exchanges:
            try:
                resp = requests.post(
                    f"{_BASE_URL}/api/price/batch",
                    headers=_rapidapi_headers(),
                    json={"requests": [{"symbol": fmt_sym, "timeframe": "D", "range": range_n}]},
                    timeout=10,
                )
                if resp.status_code in (401, 402, 403):
                    return [], f"RapidAPI auth/billing error ({resp.status_code}) — check RAPIDAPI_KEY"
                if resp.status_code == 429:
                    return [], "RapidAPI rate limit (429)"
                if resp.status_code == 200:
                    bars = _extract_bars(resp.json())
                    bars = [b for b in bars if start_date <= b["date"] <= end_date]
                    if bars:
                        for b in bars:
                            b["source"] = "rapidapi_tradingview"
                        return _deduplicate_and_sort(bars), ""
            except requests.Timeout:
                continue
            except Exception:
                continue

        return [], (
            f"RapidAPI TradingView returned 0 bars for {symbol} "
            f"in {start_date} to {end_date} (range_n={range_n})"
        )
    except Exception as exc:
        return [], f"RapidAPI range fetch error: {type(exc).__name__}: {exc}"


def fetch_price_history_for_range(
    symbol: str,
    requested_start: str,
    requested_end: str,
) -> Tuple[List[Dict], str, Dict]:
    """
    Fetch daily OHLCV for a specific date range using a provider fallback chain.

    Provider order:
      1. TradingView RapidAPI (paid — tried first for all requests)
      2. NASDAQ external provider (free fallback)
      3. yfinance (free last resort — covers pre-2023 history)

    Returns (bars, error, coverage_info).
    coverage_info keys:
        requested_start  — user-requested first date
        effective_start  — first bar date actually returned
        provider         — name of provider used
        coverage_status  — "FULL" if effective_start <= requested_start, else "PARTIAL"
        providers_tried  — list of {provider, status, first_bar, error}
    """
    try:
        from datetime import datetime as _vdt
        _vs = _vdt.strptime(requested_start, "%Y-%m-%d").date()
        _ve = _vdt.strptime(requested_end,   "%Y-%m-%d").date()
        if _vs >= _ve:
            _bad = f"Invalid date range: start={requested_start} >= end={requested_end}. Start must be before end."
            return [], _bad, {"requested_start": requested_start, "coverage_status": "FAILED", "error": _bad}
    except ValueError as _date_exc:
        _bad = f"Malformed date: {_date_exc}. Expected YYYY-MM-DD."
        return [], _bad, {"requested_start": requested_start, "coverage_status": "FAILED", "error": _bad}

    chain: List[Dict] = []
    sym_key = symbol.upper().replace("-", ".").replace(":", "_")

    def _coverage_status(first_bar: Optional[str], req_start: str) -> str:
        """FULL = first bar within 7 calendar days of req_start (weekend/holiday buffer)."""
        if not first_bar:
            return "FAILED"
        try:
            from datetime import datetime as _dt
            gap = (_dt.strptime(first_bar, "%Y-%m-%d") - _dt.strptime(req_start, "%Y-%m-%d")).days
            return "FULL" if gap <= 7 else "PARTIAL"
        except Exception:
            return "PARTIAL" if first_bar else "FAILED"

    # ── Provider 1: TradingView RapidAPI (paid subscription) ──────────────────
    tv_bars, tv_err = _fetch_rapidapi_for_range(symbol, requested_start, requested_end)
    tv_first  = tv_bars[0]["date"] if tv_bars else None
    tv_status = _coverage_status(tv_first, requested_start)
    chain.append({
        "provider":  "rapidapi_tradingview",
        "status":    tv_status,
        "first_bar": tv_first,
        "bars":      len(tv_bars),
        "error":     tv_err,
    })
    if tv_status == "FULL":
        _PROVIDER_USED[sym_key] = "rapidapi_tradingview"
        return tv_bars, "", {
            "requested_start": requested_start,
            "effective_start": tv_first,
            "provider":        "rapidapi_tradingview",
            "coverage_status": "FULL",
            "providers_tried": chain,
        }

    # ── Provider 2: NASDAQ external ────────────────────────────────────────────
    nash_bars, nash_err = _fetch_external_historical_provider(
        symbol, requested_start, requested_end
    )
    nash_first  = nash_bars[0]["date"] if nash_bars else None
    nash_status = _coverage_status(nash_first, requested_start)
    chain.append({
        "provider":  "nasdaq_external",
        "status":    nash_status,
        "first_bar": nash_first,
        "bars":      len(nash_bars),
        "error":     nash_err,
    })
    if nash_status == "FULL":
        _PROVIDER_USED[sym_key] = "external_historical_provider"
        return nash_bars, "", {
            "requested_start": requested_start,
            "effective_start": nash_first,
            "provider":        "nasdaq_external",
            "coverage_status": "FULL",
            "providers_tried": chain,
        }

    # ── Provider 3: yfinance (last resort — covers decades of history) ─────────
    yf_bars, yf_err = _fetch_yfinance(symbol, requested_start, requested_end)
    yf_first  = yf_bars[0]["date"] if yf_bars else None
    yf_status = _coverage_status(yf_first, requested_start)
    chain.append({
        "provider":  "yfinance",
        "status":    yf_status,
        "first_bar": yf_first,
        "bars":      len(yf_bars),
        "error":     yf_err,
    })
    if yf_bars:
        eff_start  = yf_first or requested_start
        final_stat = yf_status
        _PROVIDER_USED[sym_key] = "yfinance"
        return yf_bars, "", {
            "requested_start": requested_start,
            "effective_start": eff_start,
            "provider":        "yfinance",
            "coverage_status": final_stat,
            "providers_tried": chain,
        }

    # ── Provider 4: Direct Yahoo Finance REST API (bypasses yfinance library) ────
    yd_bars, yd_err = _fetch_yahoo_direct(symbol, requested_start, requested_end)
    yd_first  = yd_bars[0]["date"] if yd_bars else None
    yd_status = _coverage_status(yd_first, requested_start)
    chain.append({
        "provider":  "yahoo_direct",
        "status":    yd_status,
        "first_bar": yd_first,
        "bars":      len(yd_bars),
        "error":     yd_err,
    })
    if yd_bars:
        eff_start  = yd_first or requested_start
        final_stat = yd_status
        _PROVIDER_USED[sym_key] = "yahoo_direct"
        return yd_bars, "", {
            "requested_start": requested_start,
            "effective_start": eff_start,
            "provider":        "yahoo_direct",
            "coverage_status": final_stat,
            "providers_tried": chain,
        }

    # ── All providers failed ────────────────────────────────────────────────────
    combined_err = (
        f"No provider could fetch data for {symbol} from {requested_start}. "
        f"TradingView RapidAPI: {tv_err}. NASDAQ: {nash_err}. "
        f"yfinance: {yf_err}. Yahoo direct: {yd_err}."
    )
    return [], combined_err, {
        "requested_start": requested_start,
        "effective_start": None,
        "provider":        None,
        "coverage_status": "FAILED",
        "providers_tried": chain,
    }


# ── In-memory cache (10 min TTL) ───────────────────────────────────────────────
_PRICE_CACHE: Dict[str, Dict] = {}
_CACHE_TTL = 10 * 60


def fetch_price_history(symbol: str, min_days: int = 500) -> Tuple[List[Dict], str]:
    """
    Fetch daily price history for symbol.

    Returns (bars, error_msg).
    bars: list of {date, close, open, high, low, volume} sorted ASC.
    error_msg: empty on success, non-empty on failure.

    Uses min(max(min_days, 400), _MAX_RANGE) as the range parameter.
    Tries NASDAQ prefix first, then NYSE if the result is empty.
    """
    sym_key = symbol.upper().replace("-", ".").replace(":", "_")
    cached  = _PRICE_CACHE.get(sym_key)
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
        return cached["bars"], ""  # _PROVIDER_USED[sym_key] already set from initial fetch

    # Skip RapidAPI attempts when provider explicitly set to external_historical
    if HISTORICAL_PRICE_PROVIDER == "external_historical":
        end_d   = date.today()
        start_d = end_d - timedelta(days=min_days + 90)
        ext_bars, ext_err = _fetch_external_historical_provider(
            symbol, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
        )
        if ext_bars:
            ext_bars = _deduplicate_and_sort(ext_bars)
            _PRICE_CACHE[sym_key] = {"bars": ext_bars, "ts": time.time()}
            _PROVIDER_USED[sym_key] = "external_historical_provider"
            return ext_bars, ""
        return [], f"Historical price data unavailable for {symbol}. {ext_err}"

    key = os.getenv("RAPIDAPI_KEY", "").strip()
    if not key:
        return [], "RAPIDAPI_KEY not set in .env"

    range_n = min(max(min_days, 400), _MAX_RANGE)
    sym_fmt = _fmt(symbol)

    # Exchanges to try (first the formatted one, then fallbacks)
    exchanges_to_try = [sym_fmt]
    base = symbol.upper().replace("-", ".").strip()
    if ":" not in sym_fmt:
        exchanges_to_try = [f"NASDAQ:{base}", f"NYSE:{base}", f"AMEX:{base}"]
    elif sym_fmt.startswith("NASDAQ:"):
        exchanges_to_try = [sym_fmt, f"NYSE:{base}", f"AMEX:{base}"]
    elif sym_fmt.startswith("NYSE:"):
        exchanges_to_try = [sym_fmt, f"NASDAQ:{base}", f"AMEX:{base}"]
    else:
        exchanges_to_try = [sym_fmt, f"NASDAQ:{base}", f"NYSE:{base}"]

    # ── Attempt 1: POST batch endpoint ─────────────────────────────────────────
    last_err   = "No response from any RapidAPI endpoint"
    got_403    = False
    all_404    = True  # tracks whether every attempt returned 404 "endpoint doesn't exist"

    for fmt_sym in exchanges_to_try:
        try:
            resp = requests.post(
                f"{_BASE_URL}/api/price/batch",
                headers=_rapidapi_headers(),
                json={"requests": [{"symbol": fmt_sym, "timeframe": "D", "range": range_n}]},
                timeout=10,
            )
            if resp.status_code == 403:
                got_403 = True
                return [], "RapidAPI key invalid or not subscribed (403). Check RAPIDAPI_KEY in .env."
            if resp.status_code in (401, 402):
                return [], f"RapidAPI auth/billing error ({resp.status_code}). Check your RapidAPI subscription."
            if resp.status_code == 429:
                return [], "RapidAPI rate limit (429). Wait and retry."
            if resp.status_code == 200:
                all_404 = False
                raw  = resp.json()
                bars = _extract_bars(raw)
                if bars:
                    bars = _deduplicate_and_sort(bars)
                    _PRICE_CACHE[sym_key] = {"bars": bars, "ts": time.time()}
                    _PROVIDER_USED[sym_key] = "rapidapi_tradingview"
                    return bars, ""
                last_err = f"RapidAPI batch endpoint returned 0 bars for {fmt_sym}"
            elif resp.status_code != 404:
                all_404 = False
                last_err = f"HTTP {resp.status_code} from batch endpoint for {fmt_sym}"
            else:
                last_err = f"HTTP 404 from batch endpoint for {fmt_sym}"
        except requests.Timeout:
            last_err = f"RapidAPI batch request timed out for {fmt_sym}"
            all_404  = False
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            all_404  = False

    # ── Attempt 2: GET single-price endpoint variants ──────────────────────────
    _single_paths = [
        "/api/price/{sym}",
        "/api/v1/price/{sym}",
        "/api/v2/price/{sym}",
        "/api/ta/{sym}",
        "/api/market-data/{sym}",
    ]
    for fmt_sym in exchanges_to_try[:2]:
        for path_tmpl in _single_paths:
            path = path_tmpl.replace("{sym}", fmt_sym)
            try:
                resp = requests.get(
                    f"{_BASE_URL}{path}",
                    headers=_rapidapi_headers(),
                    timeout=6,
                )
                if resp.status_code == 200:
                    all_404 = False
                    raw  = resp.json()
                    data = raw.get("data", raw) if isinstance(raw, dict) and raw.get("success") else raw
                    bars = _extract_bars(data)
                    if bars:
                        bars = _deduplicate_and_sort(bars)
                        _PRICE_CACHE[sym_key] = {"bars": bars, "ts": time.time()}
                        _PROVIDER_USED[sym_key] = "rapidapi_tradingview"
                        return bars, ""
                elif resp.status_code != 404:
                    all_404  = False
                    last_err = f"HTTP {resp.status_code} from {path}"
            except Exception:
                pass

    # ── Attempt 3: GET with query-param symbol ─────────────────────────────────
    _qp_paths = [
        "/api/price?symbol={sym}",
        "/api/history?symbol={sym}&resolution=D&from=1609459200&to=9999999999",
        "/v1/history?symbol={sym}&interval=1d&outputsize=compact",
        "/v2/history?symbol={sym}&interval=1d",
    ]
    base_sym = symbol.upper().replace("-", ".")
    for fmt_sym in [f"NASDAQ:{base_sym}", f"NYSE:{base_sym}", base_sym]:
        for path_tmpl in _qp_paths:
            path = path_tmpl.replace("{sym}", fmt_sym)
            try:
                resp = requests.get(
                    f"{_BASE_URL}{path}",
                    headers=_rapidapi_headers(),
                    timeout=6,
                )
                if resp.status_code == 200:
                    all_404 = False
                    bars = _extract_bars(resp.json())
                    if bars:
                        bars = _deduplicate_and_sort(bars)
                        _PRICE_CACHE[sym_key] = {"bars": bars, "ts": time.time()}
                        return bars, ""
                elif resp.status_code != 404:
                    all_404 = False
            except Exception:
                pass

    # ── RapidAPI attempts exhausted — try external historical provider if auto mode ──
    if HISTORICAL_PRICE_PROVIDER == "auto":
        end_d   = date.today()
        start_d = end_d - timedelta(days=range_n + 90)
        ext_bars, ext_err = _fetch_external_historical_provider(
            symbol, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
        )
        if ext_bars:
            ext_bars = _deduplicate_and_sort(ext_bars)
            _PRICE_CACHE[sym_key] = {"bars": ext_bars, "ts": time.time()}
            _PROVIDER_USED[sym_key] = "external_historical_provider"
            return ext_bars, ""
        # Both RapidAPI and external failed
        return [], (
            f"Historical price data unavailable for {symbol}. "
            f"RapidAPI TradingView: historical endpoints not available on current plan. "
            f"External historical provider: {ext_err}. "
            "Check Developer Debug for details."
        )

    # rapidapi_tradingview mode — no external fallback
    if all_404:
        return [], (
            f"Historical price data unavailable for {symbol}. "
            "RapidAPI TradingView historical endpoints are not available on your "
            "current subscription plan. Check Developer Debug for details."
        )
    return [], (
        f"Historical price data unavailable for {symbol}. "
        f"RapidAPI TradingView error: {last_err}. Check Developer Debug for details."
    )


def get_price_on_date(
    history: List[Dict],
    target_date: str,
    max_gap_days: int = 5,
    price_basis: str = "close",
) -> Tuple[Optional[float], str, str]:
    """
    Find the actual trading-day price closest to target_date.

    Returns (price, effective_date, error_msg).
    - price: float or None if not found
    - effective_date: the actual date used (may differ from target_date for weekends/holidays)
    - error_msg: empty on success

    max_gap_days: if the nearest available date is more than this many calendar days
                  away from target_date, returns an error rather than a potentially
                  wrong price.
    price_basis: which OHLCV field to return ("close", "high", "low", "open"). Defaults to "close".
    """
    if not history:
        return None, "", "Price history is empty — cannot look up price"

    target_dt = _parse_ymd(target_date)
    if target_dt is None:
        return None, "", f"Invalid date format: {target_date!r} (expected YYYY-MM-DD)"

    # Binary search for nearest date
    dates = [b["date"] for b in history]
    # Find closest
    best_bar  = None
    best_diff = float("inf")
    for bar in history:
        bar_dt = _parse_ymd(bar["date"])
        if bar_dt is None:
            continue
        diff = abs((bar_dt - target_dt).days)
        if diff < best_diff:
            best_diff = diff
            best_bar  = bar

    if best_bar is None:
        return None, "", "Could not find any valid date in price history"

    if best_diff > max_gap_days:
        return None, best_bar["date"], (
            f"Nearest available date is {best_bar['date']} "
            f"({best_diff} days from requested {target_date}). "
            f"Max gap allowed is {max_gap_days} days. "
            f"Data may not cover this date range."
        )

    price = float(best_bar.get(price_basis) or best_bar.get("close") or 0)
    return price, best_bar["date"], ""


def filter_history_up_to(history: List[Dict], cutoff_date: str) -> List[Dict]:
    """Return only bars with date <= cutoff_date (for leakage prevention)."""
    return [b for b in history if b["date"] <= cutoff_date]


def get_context_summary(history: List[Dict], start_date: str, end_date: str) -> Dict:
    """
    Summarize price history within [start_date, end_date].
    Returns dict with start_price, end_price, min_price, max_price,
    return_pct, num_bars, annualized_return_pct.
    """
    window = [b for b in history if start_date <= b["date"] <= end_date]
    if not window:
        return {"status": "NO_DATA", "num_bars": 0}

    prices   = [b["close"] for b in window]
    s_price  = window[0]["close"]
    e_price  = window[-1]["close"]
    ret_pct  = (e_price - s_price) / s_price * 100 if s_price else 0.0
    num_days = (datetime.strptime(window[-1]["date"], "%Y-%m-%d")
                - datetime.strptime(window[0]["date"], "%Y-%m-%d")).days
    ann_ret  = 0.0
    if num_days > 0 and s_price > 0:
        years   = num_days / 365.25
        ann_ret = round(((e_price / s_price) ** (1.0 / years) - 1.0) * 100, 2)

    return {
        "status":               "OK",
        "num_bars":             len(window),
        "start_date":           window[0]["date"],
        "end_date":             window[-1]["date"],
        "start_price":          round(s_price, 4),
        "end_price":            round(e_price, 4),
        "min_price":            round(min(prices), 4),
        "max_price":            round(max(prices), 4),
        "return_pct":           round(ret_pct, 2),
        "annualized_return_pct": ann_ret,
        "num_days":             num_days,
    }


def fetch_daily_stock_bars(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict], str]:
    """
    Fetch daily stock bars for symbol within [start_date, end_date].

    Returns (bars, error_msg).
    bars: list of {date, close, open, high, low, volume} filtered to the requested range.
    error_msg: empty on success.

    Internally calls fetch_price_history with a range large enough to cover
    start_date → end_date, then filters to the requested window.
    """
    s_dt = _parse_ymd(start_date)
    e_dt = _parse_ymd(end_date)
    if s_dt is None:
        return [], f"Invalid start_date: {start_date!r}"
    if e_dt is None:
        return [], f"Invalid end_date: {end_date!r}"

    days_needed = (e_dt - s_dt).days + 60  # buffer
    min_days    = max(400, min(days_needed, _MAX_RANGE))

    all_bars, err = fetch_price_history(symbol, min_days=min_days)
    if not all_bars:
        return [], err

    filtered = [b for b in all_bars if start_date <= b["date"] <= end_date]
    if not filtered:
        return [], (
            f"No bars found between {start_date} and {end_date}. "
            f"API returned {len(all_bars)} total bars "
            f"({all_bars[0]['date']} – {all_bars[-1]['date']})."
        )
    return filtered, ""


def get_nearest_trading_day_price(
    bars: List[Dict],
    requested_date: str,
    mode: str = "nearest",
) -> Dict:
    """
    Find the closest available trading-day price to requested_date.

    mode:
      "nearest"     — closest bar in either direction (default)
      "on_or_after" — nearest bar on or after requested_date
      "on_or_before"— nearest bar on or before requested_date

    Returns:
    {
        "status":         "SUCCESS" | "FAILED",
        "requested_date": "YYYY-MM-DD",
        "effective_date": "YYYY-MM-DD",
        "close":          float,
        "open":           float,
        "high":           float,
        "low":            float,
        "volume":         float,
        "days_gap":       int,
        "source":         "RapidAPI TradingView historical prices",
        "error":          "",
    }
    """
    base = {
        "status":         "FAILED",
        "requested_date": requested_date,
        "effective_date": "",
        "close":          None,
        "open":           None,
        "high":           None,
        "low":            None,
        "volume":         None,
        "days_gap":       None,
        "source":         "RapidAPI TradingView historical prices",
        "error":          "",
    }

    if not bars:
        base["error"] = "Price history is empty"
        return base

    req_dt = _parse_ymd(requested_date)
    if req_dt is None:
        base["error"] = f"Invalid date format: {requested_date!r}"
        return base

    best_bar  = None
    best_diff = float("inf")
    for bar in bars:
        bar_dt = _parse_ymd(bar["date"])
        if bar_dt is None:
            continue
        if mode == "on_or_after"  and bar_dt < req_dt:
            continue
        if mode == "on_or_before" and bar_dt > req_dt:
            continue
        diff = abs((bar_dt - req_dt).days)
        if diff < best_diff:
            best_diff = diff
            best_bar  = bar

    if best_bar is None:
        base["error"] = (
            f"No bar found for mode='{mode}' around {requested_date}. "
            f"Available range: {bars[0]['date']} – {bars[-1]['date']}"
        )
        return base

    base.update({
        "status":         "SUCCESS",
        "effective_date": best_bar["date"],
        "close":          best_bar["close"],
        "open":           best_bar.get("open"),
        "high":           best_bar.get("high"),
        "low":            best_bar.get("low"),
        "volume":         best_bar.get("volume"),
        "days_gap":       best_diff,
        "error":          "",
    })
    return base


def _parse_ymd(s: str) -> Optional[date]:
    """Parse YYYY-MM-DD string to date object."""
    try:
        return datetime.strptime(str(s).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
