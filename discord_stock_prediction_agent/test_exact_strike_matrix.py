"""Bulk exact-strike routing tests for varied Discord option signals.

These tests do not call Tastytrade. They verify that parsed option signals
produce exact-strike validation payloads first, across many symbols/formats.

Run from project root:
    venv\\Scripts\\python.exe -m discord_stock_prediction_agent.test_exact_strike_matrix
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from .options_parser import classify_and_parse
import discord_stock_prediction_agent.options_strategy_bridge as bridge


PASS = 0
FAIL = 0


SIGNALS = [
    "BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80",
    "Buy AAPL 230 CE qty 1",
    "Buy SPY 600 CE qty 1",
    "BTO MSFT 520C 09/18 @7.10",
    "BUY NVDA 180C premium 4.20",
    "BUY TSLA 300 PE @ 5.50 qty 2",
    "BTO META 780C 08/21 @8.40",
    "Buy AMZN 235 CE premium 3.10",
    "Buy GOOGL 210C @2.35",
    "BTO AMD 185C 08/21 @4.60",
    "BUY NFLX 1350 CE @12.25",
    "Buy QQQ 570C @6.75",
    "BTO IWM 240P 08/21 @2.85",
    "BUY PLTR 185 CE premium 5.20",
    "Buy COIN 430C @9.10",
    "BTO AVGO 300C 09/18 @11.50",
    "BUY ORCL 250 CE @3.25",
    "BTO CRM 260P 08/21 @4.80",
    "Buy IBM 320C premium 2.70",
    "BUY COST 1000 CE @6.40",
]


def _assert(condition: bool, name: str, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def test_many_signals_use_exact_strike_first() -> None:
    print("\nTest: many option signal formats use exact strike first")
    original = bridge.run_options_backtest
    original_polygon = bridge.validate_exact_strike_with_polygon
    original_cache = bridge.CACHE_PATH
    calls = []

    def fake_backtest(**kwargs):
        calls.append(kwargs)
        return {
            "status": "SUCCESS",
            "passed_validation": True,
            "profit_loss": 100.0,
            "win_rate": 0.60,
            "num_trials": 100,
        }

    bridge.run_options_backtest = fake_backtest
    bridge.validate_exact_strike_with_polygon = lambda *_args, **_kwargs: {
        "status": "UNAVAILABLE",
        "message": "Disabled in deterministic payload-routing test.",
    }
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            for signal in SIGNALS:
                routed = classify_and_parse(signal)
                option = routed.option
                _assert(routed.kind == "OPTION" and option and option.valid, f"parse option: {signal}")
                before = len(calls)
                result = bridge.run_options_strategy_validation(option)
                after_calls = calls[before:]
                leg = after_calls[0]["custom_legs"][0] if after_calls else {}
                _assert(result["status"] == "SUCCESS", f"validation success: {signal}", str(result))
                _assert(len(after_calls) == 1, f"one validation call: {signal}", str(len(after_calls)))
                _assert(leg.get("strikeSelection") == "strike", f"exact strike selected: {signal}", str(leg))
                _assert(float(leg.get("strikePrice")) == float(option.strike), f"strike preserved: {signal}", str(leg))
        finally:
            bridge.run_options_backtest = original
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.CACHE_PATH = original_cache


def run_all() -> None:
    print("=" * 60)
    print("EXACT STRIKE MATRIX TEST HARNESS")
    print("No API calls -- payload routing tests")
    print("=" * 60)
    test_many_signals_use_exact_strike_first()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL:
        raise SystemExit(1)
    print("\nALL EXACT STRIKE MATRIX TESTS PASSED")


if __name__ == "__main__":
    run_all()
