"""Pure tests for bounded signal-learning state.

Run from project root:
    venv\\Scripts\\python.exe -m discord_stock_prediction_agent.test_learning_store
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from . import state_store


PASS = 0
FAIL = 0


def _assert(condition: bool, name: str, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def test_learning_pattern_counts() -> None:
    print("\nTest: learning pattern counts and summaries")
    with tempfile.TemporaryDirectory() as tmp:
        original = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            features = {
                "asset_type": "option",
                "action": "BUY",
                "direction": "CALL",
                "keywords": ["breakout", "volume"],
                "dte_bucket": "monthly",
            }
            for _ in range(3):
                state_store.record_learning_event(
                    features,
                    {"final_action": "BUY", "score": 72, "predicted_return_pct": 1.4},
                )
            state_store.record_learning_event(
                features,
                {"final_action": "HOLD", "score": 40, "predicted_return_pct": 0.1},
            )
            pattern = state_store.get_pattern_learning(features)
            _assert(pattern.get("seen") == 4, "seen count == 4", f"got {pattern.get('seen')}")
            _assert(pattern.get("approved") == 3, "approved count == 3", f"got {pattern.get('approved')}")
            _assert(pattern.get("blocked") == 1, "blocked count == 1", f"got {pattern.get('blocked')}")
            summary = state_store.get_signal_learning_summary()
            _assert(summary["total_patterns"] == 1, "one pattern stored", f"got {summary['total_patterns']}")
            _assert(summary["patterns"][0]["approval_rate"] == 75.0, "approval rate == 75%")
        finally:
            state_store.STATE_PATH = original


def test_pending_option_orders() -> None:
    print("\nTest: pending option orders can be queued and removed")
    with tempfile.TemporaryDirectory() as tmp:
        original = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            state_store.add_pending_option_order(
                {
                    "occ_symbol": "AAPL260821C00240000",
                    "root": "AAPL",
                    "side": "CALL",
                    "strike": 240.0,
                    "expiry_date": "2026-08-21",
                    "qty": 1,
                    "order_type": "market",
                }
            )
            pending = state_store.list_pending_option_orders()
            _assert(len(pending) == 1, "one pending option order", f"got {len(pending)}")
            _assert(pending[0]["occ_symbol"] == "AAPL260821C00240000", "pending OCC symbol preserved")
            state_store.remove_pending_option_order("AAPL260821C00240000")
            _assert(len(state_store.list_pending_option_orders()) == 0, "pending option order removed")
        finally:
            state_store.STATE_PATH = original


def test_pending_market_buy_orders() -> None:
    print("\nTest: pending equity market buys can be queued and removed")
    with tempfile.TemporaryDirectory() as tmp:
        original = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            queued = state_store.add_pending_market_buy("AAPL", 2, "market_closed")
            pending = state_store.list_pending_buys()
            _assert(len(pending) == 1, "one pending market buy", f"got {len(pending)}")
            _assert(pending[0]["symbol"] == "AAPL", "pending buy symbol preserved")
            _assert(bool(pending[0].get("queued")), "pending buy marked queued")
            state_store.remove_pending_buy(str(queued.get("pending_key") or ""))
            _assert(len(state_store.list_pending_buys()) == 0, "pending market buy removed")
        finally:
            state_store.STATE_PATH = original


def test_option_validation_summary() -> None:
    print("\nTest: option validation summary tracks exact-strike success rate")
    with tempfile.TemporaryDirectory() as tmp:
        original = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            state_store.record_option_validation_event(
                {"root": "AAPL", "exact_status": "SUCCESS", "proxy_status": "", "status": "SUCCESS"}
            )
            state_store.record_option_validation_event(
                {"root": "MSFT", "exact_status": "VALIDATION_FAILED", "proxy_status": "SUCCESS", "status": "SUCCESS_DELTA_PROXY"}
            )
            summary = state_store.get_option_validation_summary()
            _assert(summary["total"] == 2, "two validation events", f"got {summary['total']}")
            _assert(summary["exact_attempted"] == 2, "two exact attempts", f"got {summary['exact_attempted']}")
            _assert(summary["exact_success"] == 1, "one exact success", f"got {summary['exact_success']}")
            _assert(summary["exact_success_rate"] == 50.0, "exact success rate == 50%")
            _assert(summary["delta_proxy_used"] == 1, "one delta proxy fallback", f"got {summary['delta_proxy_used']}")
        finally:
            state_store.STATE_PATH = original


def test_parser_learning_summary() -> None:
    print("\nTest: parser learning records normalized format reliability")
    with tempfile.TemporaryDirectory() as tmp:
        original = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            signal = (
                '{"action":"buy_to_open","underlying":"AAPL","strike":240,'
                '"right":"call","expiration":"2026-09-18","contracts":2}'
            )
            state_store.record_parser_learning(signal, "OPTION", True)
            state_store.record_parser_learning(signal, "OPTION", True)
            state_store.record_parser_learning("watching markets", "NO_TRADE", False, "No trade action")
            summary = state_store.get_parser_learning_summary()
            _assert(summary["total_patterns"] == 2, "two normalized parser patterns")
            _assert(summary["total_seen"] == 3, "three parser observations")
            _assert(summary["valid"] == 2, "two valid parser observations")
            _assert(summary["invalid"] == 1, "one invalid parser observation")
            _assert(summary["success_rate"] == 66.67, "parser success rate == 66.67%")
        finally:
            state_store.STATE_PATH = original


def run_all() -> None:
    print("=" * 60)
    print("LEARNING STORE TEST HARNESS")
    print("=" * 60)
    test_learning_pattern_counts()
    test_pending_option_orders()
    test_pending_market_buy_orders()
    test_option_validation_summary()
    test_parser_learning_summary()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
