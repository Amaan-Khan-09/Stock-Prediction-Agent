"""Unit tests for symbol_directory.py -- previously had zero test coverage.

Covers the local Alpaca symbol/name cache and, most importantly, the
freshness gate inside refresh_symbol_cache_from_alpaca that the new periodic
symbol_cache_refresh_monitor task in discord_agent.py relies on: it must
skip re-fetching when the cache is still fresh, and actually refresh once
it's stale or missing.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from . import symbol_directory as sd


class _FakeResponse:
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _with_config_overrides(**overrides):
    """config is a frozen dataclass -- plain setattr (what monkeypatch.setattr
    would do) raises FrozenInstanceError, so overrides go through
    object.__setattr__ with manual restore, matching this codebase's
    established pattern (see test_whatsapp_alert_routing.py)."""
    originals = {name: getattr(sd.config, name) for name in overrides}
    for name, value in overrides.items():
        object.__setattr__(sd.config, name, value)
    return originals


def _restore_config_overrides(originals: dict) -> None:
    for name, value in originals.items():
        object.__setattr__(sd.config, name, value)


def _with_temp_cache(test_body) -> None:
    with TemporaryDirectory() as tmp:
        original_path = sd.CACHE_PATH
        original_cache = sd._CACHE
        original_loaded = sd._CACHE_LOADED
        sd.CACHE_PATH = Path(tmp) / "symbol_cache.json"
        sd._CACHE = {"symbols": {}, "names": {}}
        sd._CACHE_LOADED = False
        try:
            test_body()
        finally:
            sd.CACHE_PATH = original_path
            sd._CACHE = original_cache
            sd._CACHE_LOADED = original_loaded


def test_is_symbol_like_accepts_ticker_shapes_rejects_malformed() -> None:
    # is_symbol_like is a pure shape check (1-7 chars, ticker-like pattern) --
    # it deliberately doesn't do semantic filtering of real English words
    # that happen to be ticker-shaped (e.g. "HERE"); that's handled upstream
    # by options_parser's own root-ignore list, not here.
    assert sd.is_symbol_like("LULU")
    assert sd.is_symbol_like("BRK.B")
    assert not sd.is_symbol_like("")
    assert not sd.is_symbol_like("TOOLONGTICKER")
    assert not sd.is_symbol_like("123ABC")  # must start with a letter


def test_resolve_cached_symbol_matches_by_symbol_and_company_name() -> None:
    def body() -> None:
        sd._save_cache(
            {"LULU": {"name": "Lululemon Athletica Inc", "exchange": "NASDAQ", "tradable": True, "class": "us_equity"}},
            {"LULULEMON ATHLETICA": "LULU", "LULULEMONATHLETICA": "LULU"},
        )
        assert sd.resolve_cached_symbol("lulu") == "LULU"
        assert sd.resolve_cached_symbol("Lululemon Athletica") == "LULU"
        assert sd.resolve_cached_symbol("Some Random Word") == ""
        assert sd.is_known_symbol("LULU")
        assert not sd.is_known_symbol("NOTREAL")

    _with_temp_cache(body)


def test_refresh_skips_network_call_when_cache_is_fresh(monkeypatch) -> None:
    def body() -> None:
        sd._save_cache({"AAPL": {"name": "Apple Inc", "exchange": "NASDAQ", "tradable": True, "class": "us_equity"}}, {})

        called = {"count": 0}

        def fake_get(*args, **kwargs):
            called["count"] += 1
            raise AssertionError("should not hit the network when the cache is fresh")

        monkeypatch.setattr(sd.requests, "get", fake_get)
        status = sd.refresh_symbol_cache_from_alpaca()
        assert status["status"] == "fresh"
        assert called["count"] == 0

    _with_temp_cache(body)


def test_refresh_fetches_when_cache_is_stale(monkeypatch) -> None:
    def body() -> None:
        # A cache older than max_age_hours must trigger a real refresh.
        sd._save_cache({"OLD": {"name": "Old Corp", "exchange": "NYSE", "tradable": True, "class": "us_equity"}}, {})
        stale_time = (datetime.utcnow() - timedelta(hours=48)).isoformat(timespec="seconds") + "Z"
        sd._CACHE["updated_at"] = stale_time
        sd.CACHE_PATH.write_text(json.dumps(sd._CACHE), encoding="utf-8")

        # has_alpaca is a read-only property derived from these two fields --
        # setting them makes it True naturally.
        originals = _with_config_overrides(
            alpaca_api_key="key",
            alpaca_secret_key="secret",
            alpaca_base_url="https://paper-api.alpaca.markets",
        )
        try:
            payload = [
                {"symbol": "NVDA", "name": "NVIDIA Corp", "exchange": "NASDAQ", "tradable": True, "class": "us_equity"},
            ]

            def fake_get(url, headers=None, timeout=None):
                assert "assets" in url
                return _FakeResponse(200, payload)

            monkeypatch.setattr(sd.requests, "get", fake_get)
            status = sd.refresh_symbol_cache_from_alpaca()
            assert status["status"] == "refreshed"
            assert sd.is_known_symbol("NVDA")
            assert not sd.is_known_symbol("OLD"), "a real refresh replaces the old symbol set, not merges it"
        finally:
            _restore_config_overrides(originals)

    _with_temp_cache(body)


def test_refresh_reports_not_configured_without_alpaca_creds() -> None:
    def body() -> None:
        # has_alpaca is a read-only property derived from these two fields --
        # clearing them makes it False naturally.
        originals = _with_config_overrides(alpaca_api_key="", alpaca_secret_key="")
        try:
            status = sd.refresh_symbol_cache_from_alpaca()
            assert status["status"] == "not_configured"
        finally:
            _restore_config_overrides(originals)

    _with_temp_cache(body)


def test_symbol_cache_refresh_monitor_task_runs_without_error(monkeypatch) -> None:
    """Covers the new periodic task in discord_agent.py -- it must call
    refresh_symbol_cache_from_alpaca and not raise, whether or not a real
    refresh happened this cycle."""
    from . import discord_agent

    calls = {"count": 0}

    def fake_refresh(*args, **kwargs):
        calls["count"] += 1
        return {"status": "refreshed", "symbols": 1, "names": 1}

    monkeypatch.setattr(discord_agent, "refresh_symbol_cache_from_alpaca", fake_refresh)
    asyncio.run(discord_agent.symbol_cache_refresh_monitor.coro())
    assert calls["count"] == 1
