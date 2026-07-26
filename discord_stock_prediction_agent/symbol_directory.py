"""Shared stock/ETF symbol directory for equity and options signal parsing.

The hardcoded alias dictionaries are useful for common company names, but they
can never cover every listed symbol. This module adds a local cache of active
Alpaca assets and safe ticker-shaped fallback helpers so both equity and option
parsers can recognize less-common symbols such as LULU without treating random
English words as tickers.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

from .config import AGENT_DIR, config

CACHE_PATH = AGENT_DIR / "symbol_cache.json"
_CACHE: Dict[str, Any] = {"symbols": {}, "names": {}}
_CACHE_LOADED = False

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|PLC|LTD|LIMITED|HOLDINGS?|"
    r"GROUP|TECHNOLOGIES|TECHNOLOGY|CLASS\s+[A-Z]|COMMON\s+STOCK|ORDINARY\s+SHARES?)\b",
    re.IGNORECASE,
)


def normalize_symbol(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "", value or "").upper().replace("-", ".").strip(".")


def normalize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9& ]+", " ", value or "").upper()
    text = _COMPANY_SUFFIX_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_symbol_like(value: str, max_len: int = 7) -> bool:
    symbol = normalize_symbol(value)
    return bool(1 <= len(symbol) <= max_len and re.match(r"^[A-Z][A-Z0-9]*(?:\.[A-Z])?$", symbol))


def _load_cache() -> Dict[str, Any]:
    global _CACHE_LOADED, _CACHE
    if _CACHE_LOADED:
        return _CACHE
    _CACHE_LOADED = True
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _CACHE = {
                    "symbols": dict(data.get("symbols") or {}),
                    "names": dict(data.get("names") or {}),
                    "updated_at": data.get("updated_at"),
                }
    except Exception:
        _CACHE = {"symbols": {}, "names": {}}
    return _CACHE


def _save_cache(symbols: Dict[str, Dict[str, Any]], names: Dict[str, str]) -> None:
    global _CACHE, _CACHE_LOADED
    _CACHE = {
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "symbols": symbols,
        "names": names,
    }
    _CACHE_LOADED = True
    CACHE_PATH.write_text(json.dumps(_CACHE, indent=2, sort_keys=True), encoding="utf-8")


def _name_keys(name: str) -> Iterable[str]:
    normalized = normalize_name(name)
    if normalized:
        yield normalized
    compact = normalized.replace(" ", "")
    if compact and compact != normalized:
        yield compact


def resolve_cached_symbol(value: str) -> str:
    cache = _load_cache()
    symbol = normalize_symbol(value)
    if symbol in (cache.get("symbols") or {}):
        return symbol
    name_key = normalize_name(value)
    if name_key in (cache.get("names") or {}):
        return str(cache["names"][name_key])
    compact = name_key.replace(" ", "")
    if compact in (cache.get("names") or {}):
        return str(cache["names"][compact])
    return ""


def is_known_symbol(value: str) -> bool:
    return bool(resolve_cached_symbol(value))


def cache_stats() -> Dict[str, int]:
    cache = _load_cache()
    return {
        "symbols": len(cache.get("symbols") or {}),
        "names": len(cache.get("names") or {}),
    }


def refresh_symbol_cache_from_alpaca(max_age_hours: int = 24) -> Dict[str, Any]:
    """Refresh active US equity/ETF symbols from Alpaca when configured.

    Returns a small status dictionary. It never includes API keys/secrets.
    """
    _load_cache()
    updated_at = _CACHE.get("updated_at")
    if updated_at:
        try:
            age = datetime.utcnow() - datetime.fromisoformat(str(updated_at).replace("Z", ""))
            if age < timedelta(hours=max_age_hours):
                return {"status": "fresh", **cache_stats()}
        except Exception:
            pass

    if not config.has_alpaca:
        return {"status": "not_configured", **cache_stats()}

    headers = {
        "APCA-API-KEY-ID": config.alpaca_api_key,
        "APCA-API-SECRET-KEY": config.alpaca_secret_key,
    }
    url = f"{config.alpaca_base_url}/v2/assets?status=active&asset_class=us_equity"
    try:
        resp = requests.get(url, headers=headers, timeout=25)
        if not (200 <= resp.status_code < 300):
            return {"status": "error", "http_status": resp.status_code, **cache_stats()}
        assets = resp.json()
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, **cache_stats()}

    symbols: Dict[str, Dict[str, Any]] = {}
    names: Dict[str, str] = {}
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            symbol = normalize_symbol(str(asset.get("symbol") or ""))
            if not symbol or not is_symbol_like(symbol, max_len=12):
                continue
            name = str(asset.get("name") or "")
            symbols[symbol] = {
                "name": name,
                "exchange": asset.get("exchange") or "",
                "tradable": bool(asset.get("tradable", False)),
                "class": asset.get("class") or "",
            }
            for key in _name_keys(name):
                names.setdefault(key, symbol)

    if symbols:
        _save_cache(symbols, names)
        return {"status": "refreshed", **cache_stats()}
    return {"status": "empty", **cache_stats()}
