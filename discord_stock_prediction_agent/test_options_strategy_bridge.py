"""Known-answer tests for Discord options strategy validation fallback logic.

These tests do not call Tastytrade. They monkeypatch the bridge-level
run_options_backtest function to verify exact-strike and delta routing.

Run from project root:
    venv\\Scripts\\python.exe -m discord_stock_prediction_agent.test_options_strategy_bridge
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from .options_parser import classify_and_parse
import discord_stock_prediction_agent.options_strategy_bridge as bridge

PASS = 0
FAIL = 0


def _polygon_disabled(*_args, **_kwargs):
    return {"status": "SKIPPED", "message": "POLYGON_API_KEY is not configured for this test."}


def _assert(condition: bool, name: str, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def test_fixed_strike_uses_exact_strike_first() -> None:
    print("\nTest: fixed-strike signal uses exact strike first when validation succeeds")
    calls = []

    def fake_backtest(**kwargs):
        calls.append(kwargs)
        return {
            "status": "SUCCESS",
            "passed_validation": True,
            "profit_loss": 100.0,
            "win_rate": 0.55,
            "num_trials": 200,
        }

    original = bridge.run_options_backtest
    original_polygon = bridge.validate_exact_strike_with_polygon
    original_cache = bridge.CACHE_PATH
    bridge.run_options_backtest = fake_backtest
    bridge.validate_exact_strike_with_polygon = _polygon_disabled
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.run_options_backtest = original
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "SUCCESS", "status == SUCCESS", str(result))
    _assert(result["decision"] == "BUY", "decision == BUY", str(result))
    _assert(len(calls) == 1, "called exact strike once", str(len(calls)))
    _assert(calls[0]["custom_legs"][0]["strikeSelection"] == "strike", "call uses exact strike")
    _assert(calls[0]["custom_legs"][0]["strikePrice"] == 240.0, "exact strike price preserved")
    _assert(result["exact_strike_attempt"]["status"] == "SUCCESS", "exact strike attempt succeeded")


def test_polygon_exact_strike_wins_before_tastytrade() -> None:
    print("\nTest: Polygon exact-strike validation wins before Tastytrade")
    tastytrade_calls = []
    polygon_calls = []

    def fake_polygon(option, start_date, end_date):
        polygon_calls.append((option, start_date, end_date))
        return {
            "status": "SUCCESS",
            "decision": "BUY",
            "polygon_ticker": "O:AAPL260821C00240000",
            "bars": 12,
            "entry_price": 3.45,
            "exit_price": 4.10,
            "return_pct": 18.84,
            "profit_loss": 65.0,
            "win_rate": 1.0,
            "message": "Polygon exact-strike historical aggregate validation succeeded.",
        }

    def fake_backtest(**kwargs):
        tastytrade_calls.append(kwargs)
        return {"status": "SUCCESS", "profit_loss": -100.0, "win_rate": 0.10}

    original_polygon = bridge.validate_exact_strike_with_polygon
    original_backtest = bridge.run_options_backtest
    original_provider = bridge.config.option_strike_validation_provider
    original_cache = bridge.CACHE_PATH
    bridge.validate_exact_strike_with_polygon = fake_polygon
    bridge.run_options_backtest = fake_backtest
    object.__setattr__(bridge.config, "option_strike_validation_provider", "polygon_first")
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.run_options_backtest = original_backtest
            object.__setattr__(bridge.config, "option_strike_validation_provider", original_provider)
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "SUCCESS_POLYGON_STRIKE", "status == SUCCESS_POLYGON_STRIKE", str(result))
    _assert(result["decision"] == "BUY", "decision == BUY", str(result))
    _assert(len(polygon_calls) == 1, "Polygon called once", str(len(polygon_calls)))
    _assert(len(tastytrade_calls) == 0, "Tastytrade not called after Polygon success", str(len(tastytrade_calls)))
    _assert(result["polygon_strike_attempt"]["polygon_ticker"] == "O:AAPL260821C00240000", "Polygon ticker recorded")


def test_delta_signal_uses_previous_strategy_not_polygon() -> None:
    print("\nTest: delta signal uses previous strategy path, not Polygon")
    tastytrade_calls = []
    polygon_calls = []

    def fake_polygon(option, start_date, end_date):
        polygon_calls.append((option, start_date, end_date))
        return {"status": "SUCCESS", "decision": "BUY"}

    def fake_backtest(**kwargs):
        tastytrade_calls.append(kwargs)
        return {
            "status": "SUCCESS",
            "passed_validation": True,
            "profit_loss": 42.0,
            "win_rate": 0.58,
            "num_trials": 99,
        }

    original_polygon = bridge.validate_exact_strike_with_polygon
    original_backtest = bridge.run_options_backtest
    original_cache = bridge.CACHE_PATH
    bridge.validate_exact_strike_with_polygon = fake_polygon
    bridge.run_options_backtest = fake_backtest
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("Buy 30 delta AAPL call qty 1").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.run_options_backtest = original_backtest
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "SUCCESS", "status == SUCCESS", str(result))
    _assert(result["decision"] == "BUY", "decision == BUY", str(result))
    _assert(len(polygon_calls) == 0, "Polygon not called for delta signal", str(len(polygon_calls)))
    _assert(len(tastytrade_calls) == 1, "Tastytrade called once for delta signal", str(len(tastytrade_calls)))
    leg = tastytrade_calls[0]["custom_legs"][0]
    _assert(leg["strikeSelection"] == "delta", "delta strikeSelection used", str(leg))
    _assert(leg["delta"] == 30, "requested delta preserved", str(leg))
    _assert("strikePrice" not in leg, "no strikePrice for delta signal", str(leg))


def test_polygon_failure_preserved_and_exact_strike_stops_at_review() -> None:
    print("\nTest: Polygon failure is preserved and exact strike does not use delta proxy")
    tastytrade_calls = []
    polygon_calls = []

    def fake_polygon(option, start_date, end_date):
        polygon_calls.append((option, start_date, end_date))
        return {
            "status": "SKIPPED",
            "decision": "REVIEW",
            "message": "POLYGON_API_KEY is not configured.",
        }

    def fake_backtest(**kwargs):
        tastytrade_calls.append(kwargs)
        return {
            "status": "VALIDATION_FAILED",
            "message": "statistics=null; trials=null/empty",
            "passed_validation": False,
        }

    original_polygon = bridge.validate_exact_strike_with_polygon
    original_backtest = bridge.run_options_backtest
    original_provider = bridge.config.option_strike_validation_provider
    original_cache = bridge.CACHE_PATH
    bridge.validate_exact_strike_with_polygon = fake_polygon
    bridge.run_options_backtest = fake_backtest
    object.__setattr__(bridge.config, "option_strike_validation_provider", "polygon_first")
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.run_options_backtest = original_backtest
            object.__setattr__(bridge.config, "option_strike_validation_provider", original_provider)
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "FALLBACK_APPROVED", "complete exact strike fallback approved", str(result))
    _assert(len(polygon_calls) == 1, "Polygon was attempted once", str(len(polygon_calls)))
    _assert(result["polygon_strike_attempt"]["status"] == "SKIPPED", "Polygon attempt status preserved")
    _assert("POLYGON_API_KEY" in result.get("polygon_error", ""), "Polygon error is visible", str(result))
    _assert(len(tastytrade_calls) == 2, "full exact and recent exact only", str(len(tastytrade_calls)))
    _assert(all(c["custom_legs"][0]["strikeSelection"] == "strike" for c in tastytrade_calls), "all calls kept exact strike")
    _assert(not result.get("delta_proxy_attempt"), "no delta proxy attempt recorded")
    _assert(result.get("fallback_approval") == "complete_exact_contract_signal", "complete exact fallback reason recorded")


def test_incomplete_exact_strike_validation_failure_blocks_by_default() -> None:
    print("\nTest: incomplete exact strike validation failure blocks by default")

    def fake_backtest(**kwargs):
        return {
            "status": "VALIDATION_FAILED",
            "message": "statistics=null; trials=null/empty",
            "passed_validation": False,
        }

    original = bridge.run_options_backtest
    original_polygon = bridge.validate_exact_strike_with_polygon
    original_cache = bridge.CACHE_PATH
    bridge.run_options_backtest = fake_backtest
    bridge.validate_exact_strike_with_polygon = _polygon_disabled
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("Buy AAPL 230 CE qty 1").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.run_options_backtest = original
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "FALLBACK_REVIEW", "status == FALLBACK_REVIEW", str(result))
    _assert(result["decision"] == "REVIEW", "decision == REVIEW", str(result))
    _assert("fallback checks failed" in result["error"], "error explains fallback failure")
    _assert(result["exact_strike_attempt"]["status"] == "VALIDATION_FAILED", "exact strike attempted first")
    _assert(not result.get("delta_proxy_attempt"), "delta proxy not attempted for exact strike")


def test_recent_exact_retry_runs_without_delta_proxy() -> None:
    print("\nTest: recent exact-strike retry runs without delta proxy")
    calls = []

    def fake_backtest(**kwargs):
        calls.append(kwargs)
        start_date = str(kwargs.get("start_date"))
        if len(calls) == 1:
            return {
                "status": "VALIDATION_FAILED",
                "message": "statistics=null; trials=null/empty",
                "passed_validation": False,
            }
        return {
            "status": "SUCCESS",
            "passed_validation": True,
            "profit_loss": 88.0,
            "win_rate": 0.61,
            "num_trials": 18,
            "start_date_seen": start_date,
        }

    original = bridge.run_options_backtest
    original_polygon = bridge.validate_exact_strike_with_polygon
    original_cache = bridge.CACHE_PATH
    bridge.run_options_backtest = fake_backtest
    bridge.validate_exact_strike_with_polygon = _polygon_disabled
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.run_options_backtest = original
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "SUCCESS_EXACT_STRIKE_RECENT", "status == SUCCESS_EXACT_STRIKE_RECENT", str(result))
    _assert(result["decision"] == "BUY", "decision == BUY", str(result))
    _assert(len(calls) == 2, "called full exact then recent exact only", str(len(calls)))
    _assert(calls[0]["custom_legs"][0]["strikeSelection"] == "strike", "first call exact strike")
    _assert(calls[1]["custom_legs"][0]["strikeSelection"] == "strike", "second call still exact strike")
    _assert(calls[1]["custom_legs"][0]["strikePrice"] == 240.0, "recent retry preserves exact strike")
    _assert("recent_exact_strike_attempt" in result, "recent exact attempt recorded")
    _assert(not result.get("delta_proxy_attempt"), "delta proxy not used after recent exact success")


def test_rate_limit_returns_review_not_error() -> None:
    print("\nTest: Tastytrade HTTP 429 becomes controlled validation review")

    def fake_backtest(**kwargs):
        return {
            "status": "ERROR",
            "message": "Failed to create backtest: HTTP 429",
            "passed_validation": False,
        }

    original = bridge.run_options_backtest
    original_polygon = bridge.validate_exact_strike_with_polygon
    original_cache = bridge.CACHE_PATH
    bridge.run_options_backtest = fake_backtest
    bridge.validate_exact_strike_with_polygon = _polygon_disabled
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse("BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80").option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.run_options_backtest = original
            bridge.validate_exact_strike_with_polygon = original_polygon
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "VALIDATION_RATE_LIMITED", "status == VALIDATION_RATE_LIMITED", str(result))
    _assert(result["decision"] == "REVIEW", "decision == REVIEW", str(result))
    _assert("temporarily busy" in result["error"], "user-safe rate-limit message")


def test_multi_leg_payload_preserves_complete_strategy() -> None:
    print("\nTest: multi-leg strategy payload preserves every exact leg")
    option = classify_and_parse(
        "BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5"
    ).option
    strategy = bridge.build_options_strategy_input(option)
    legs = strategy["custom_legs"]
    _assert(strategy["legs"] == 2, "strategy has two legs", str(strategy))
    _assert(strategy["structure"] == "bull_call_spread", "bull call spread classified")
    _assert(strategy["price_effect"] == "debit", "net debit preserved")
    _assert(legs[0]["direction"] == "long", "first leg is long")
    _assert(legs[0]["strikePrice"] == 240.0, "first exact strike preserved")
    _assert(legs[1]["direction"] == "short", "second leg is short")
    _assert(legs[1]["strikePrice"] == 250.0, "second exact strike preserved")
    _assert(legs[0]["quantity"] == 5 and legs[1]["quantity"] == 5, "strategy qty applied to both legs")

    calendar = classify_and_parse(
        "BTO MSFT 550C 12/19 / STO MSFT 550C 09/19 @5.80 Debit Qty 4"
    ).option
    calendar_strategy = bridge.build_options_strategy_input(calendar)
    calendar_legs = calendar_strategy["custom_legs"]
    _assert(calendar.structure == "calendar_spread", "calendar spread classified")
    _assert(
        calendar_legs[0]["daysUntilExpiration"] != calendar_legs[1]["daysUntilExpiration"],
        "per-leg calendar expiries produce different DTE values",
    )


def test_empty_multi_leg_backtest_uses_structural_paper_gate() -> None:
    print("\nTest: empty multi-leg history uses transparent structural paper gate")
    calls = []

    def fake_backtest(**kwargs):
        calls.append(kwargs)
        return {
            "status": "VALIDATION_FAILED",
            "message": "statistics=null; trials=null/empty - no trial data returned",
        }

    original = bridge.run_options_backtest
    original_cache = bridge.CACHE_PATH
    bridge.run_options_backtest = fake_backtest
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            results = [
                bridge.run_options_strategy_validation(classify_and_parse(signal).option)
                for signal in (
                    "BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5",
                    "BTO TSLA 290P / STO TSLA 270P 09/19 @5.10 Debit Qty 1",
                    "BUY SPY 640C + 640P 09/19 @8.20 Debit Qty 2",
                    "BUY QQQ 600C + 570P 10/17 @7.10 Debit Qty 4",
                    "STO SPY 620P / BTO SPY 610P / STO SPY 670C / BTO SPY 680C 09/19 @2.15 Credit Qty 10",
                )
            ]
        finally:
            bridge.run_options_backtest = original
            bridge.CACHE_PATH = original_cache

    _assert(len(calls) == 10, "full and recent exact-strike attempts were made for all five strategies")
    _assert({call["num_legs"] for call in calls} == {2, 4}, "Tastytrade receives the real leg count")
    _assert(
        all(result["status"] == "MULTI_LEG_STRUCTURAL_APPROVED" for result in results),
        "all clean screenshot-style strategies use explicit structural approval status",
        str(results),
    )
    _assert(
        all(result["decision"] == "BUY" for result in results),
        "all entry strategies map to BUY",
        str(results),
    )
    _assert(
        all(
            result.get("fallback_approval") == "complete_multi_leg_structure"
            for result in results
        ),
        "every result records structural fallback basis",
    )


def test_valid_negative_multi_leg_backtest_is_not_overridden() -> None:
    print("\nTest: valid negative multi-leg history remains SELL")

    def fake_backtest(**kwargs):
        return {
            "status": "SUCCESS",
            "profit_loss": -125.0,
            "win_rate": 0.30,
            "trials": [{"profitLoss": -125.0}],
            "statistics": {"profitLoss": -125.0},
        }

    original = bridge.run_options_backtest
    original_cache = bridge.CACHE_PATH
    bridge.run_options_backtest = fake_backtest
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            option = classify_and_parse(
                "BUY SPY 640C + 640P 09/19 @8.20 Debit Qty 2"
            ).option
            result = bridge.run_options_strategy_validation(option)
        finally:
            bridge.run_options_backtest = original
            bridge.CACHE_PATH = original_cache

    _assert(result["status"] == "SUCCESS", "valid historical result is preserved", str(result))
    _assert(result["decision"] == "SELL", "negative historical result remains SELL", str(result))


def test_concurrent_cache_writes_preserve_every_result() -> None:
    print("\nTest: concurrent option validations preserve every cache entry")
    original_cache = bridge.CACHE_PATH
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        try:
            with ThreadPoolExecutor(max_workers=12) as pool:
                list(
                    pool.map(
                        lambda index: bridge._store_cache_entry(
                            f"key-{index}", {"status": "SUCCESS", "index": index}
                        ),
                        range(100),
                    )
                )
            loaded = json.loads(bridge.CACHE_PATH.read_text(encoding="utf-8"))
        finally:
            bridge.CACHE_PATH = original_cache
    _assert(len(loaded) == 100, "all concurrent cache writes survived", str(len(loaded)))


def test_expired_cache_entries_are_pruned_on_write() -> None:
    print("\nTest: expired options-validation cache entries are pruned, not kept forever")
    original_cache = bridge.CACHE_PATH
    original_ttl = bridge.config.option_validation_cache_ttl_hours
    with TemporaryDirectory() as tmp:
        bridge.CACHE_PATH = Path(tmp) / "cache.json"
        object.__setattr__(bridge.config, "option_validation_cache_ttl_hours", 1)
        try:
            stale_cache = {
                "stale-key": {"created_at": 0.0, "result": {"status": "SUCCESS"}},
            }
            bridge.CACHE_PATH.write_text(json.dumps(stale_cache), encoding="utf-8")
            bridge._store_cache_entry("fresh-key", {"status": "SUCCESS"})
            loaded = json.loads(bridge.CACHE_PATH.read_text(encoding="utf-8"))
        finally:
            bridge.CACHE_PATH = original_cache
            object.__setattr__(bridge.config, "option_validation_cache_ttl_hours", original_ttl)
    _assert("stale-key" not in loaded, "expired entry was pruned")
    _assert("fresh-key" in loaded, "newly written entry survives pruning")
    _assert(len(loaded) == 1, "cache does not grow unbounded across TTL-expired entries", str(len(loaded)))


def run_all() -> None:
    print("=" * 60)
    print("OPTIONS STRATEGY BRIDGE TEST HARNESS")
    print("No API calls -- monkeypatched validation flow tests")
    print("=" * 60)

    test_fixed_strike_uses_exact_strike_first()
    test_polygon_exact_strike_wins_before_tastytrade()
    test_delta_signal_uses_previous_strategy_not_polygon()
    test_polygon_failure_preserved_and_exact_strike_stops_at_review()
    test_incomplete_exact_strike_validation_failure_blocks_by_default()
    test_recent_exact_retry_runs_without_delta_proxy()
    test_rate_limit_returns_review_not_error()
    test_multi_leg_payload_preserves_complete_strategy()
    test_empty_multi_leg_backtest_uses_structural_paper_gate()
    test_valid_negative_multi_leg_backtest_is_not_overridden()
    test_concurrent_cache_writes_preserve_every_result()
    test_expired_cache_entries_are_pruned_on_write()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL:
        raise SystemExit(1)
    print("\nALL OPTIONS STRATEGY BRIDGE TESTS PASSED")


if __name__ == "__main__":
    run_all()


