"""Production queue and state-concurrency regression tests."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from . import durable_signal_queue as durable
from . import state_store


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_durable_queue_concurrency() -> None:
    with TemporaryDirectory() as tmp:
        original_path = durable.QUEUE_PATH
        original_initialized_paths = set(durable._INITIALIZED_PATHS)
        durable.QUEUE_PATH = Path(tmp) / "signal_queue.sqlite3"
        durable._INITIALIZED_PATHS.clear()
        try:
            def enqueue(index: int):
                return durable.enqueue_signal(
                    f"BUY TEST{index}",
                    f"user-{index % 25}",
                    "channel-1",
                    f"message-{index}",
                    5_000,
                )

            with ThreadPoolExecutor(max_workers=16) as pool:
                accepted = list(pool.map(enqueue, range(500)))
            _check(all(item.get("accepted") for item in accepted), "all 500 signals accepted")
            _check(durable.queue_stats()["queued"] == 500, "500 signals persisted")

            duplicate = durable.enqueue_signal(
                "BUY TEST0", "user-0", "channel-1", "message-0", 5_000
            )
            _check(duplicate.get("duplicate_delivery"), "gateway redelivery is idempotent")

            with ThreadPoolExecutor(max_workers=16) as pool:
                claimed = list(pool.map(lambda _: durable.claim_next_signal(), range(500)))
            ids = [item.get("id") for item in claimed if item]
            _check(len(ids) == 500, "all 500 signals claimed")
            _check(len(set(ids)) == 500, "no signal was claimed twice")

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(durable.complete_signal, ids))
            _check(durable.queue_stats()["depth"] == 0, "completed signals were deleted")
        finally:
            durable.QUEUE_PATH = original_path
            durable._INITIALIZED_PATHS.clear()
            durable._INITIALIZED_PATHS.update(original_initialized_paths)


def test_retry_dead_letter_and_restart_recovery() -> None:
    with TemporaryDirectory() as tmp:
        original_path = durable.QUEUE_PATH
        original_initialized_paths = set(durable._INITIALIZED_PATHS)
        durable.QUEUE_PATH = Path(tmp) / "signal_queue.sqlite3"
        durable._INITIALIZED_PATHS.clear()
        try:
            first = durable.enqueue_signal("BUY AAPL", "u", "c", "retry-1")
            claimed = durable.claim_next_signal(max_attempts=2)
            _check(claimed.get("id") == first.get("id"), "retry signal claimed")
            result = durable.fail_signal(claimed["id"], "temporary", 2, 1)
            _check(result["status"] == "queued", "first failure was requeued")
            time.sleep(1.05)
            claimed_again = durable.claim_next_signal(max_attempts=2)
            result = durable.fail_signal(claimed_again["id"], "still failing", 2, 1)
            _check(result["status"] == "dead", "maximum attempts moved signal to dead-letter")
            _check(durable.queue_stats()["dead"] == 1, "dead-letter count recorded")

            second = durable.enqueue_signal("SELL MSFT", "u", "c", "recover-1")
            processing = durable.claim_next_signal()
            _check(processing.get("id") == second.get("id"), "recovery signal claimed")
            _check(durable.recover_inflight_signals() == 1, "one interrupted signal recovered")
            recovered = durable.claim_next_signal()
            _check(recovered.get("id") == second.get("id"), "recovered signal became claimable")
            durable.complete_signal(recovered["id"])
        finally:
            durable.QUEUE_PATH = original_path
            durable._INITIALIZED_PATHS.clear()
            durable._INITIALIZED_PATHS.update(original_initialized_paths)


def test_atomic_state_and_duplicate_trade_queues() -> None:
    with TemporaryDirectory() as tmp:
        original_path = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            def record(index: int) -> None:
                state_store.record_order_event(
                    {"symbol": f"T{index % 10}", "side": "buy", "status": "test"},
                    limit=1_000,
                )

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(record, range(500)))
            parsed = json.loads(state_store.STATE_PATH.read_text(encoding="utf-8"))
            _check(len(parsed.get("order_events") or []) == 500, "no concurrent state events lost")

            state_store.add_pending_market_buy("AAPL", 1, "market closed")
            state_store.add_pending_market_buy("AAPL", 1, "market closed")
            state_store.add_pending_sell("MSFT", 1, "market closed")
            state_store.add_pending_sell("MSFT", 1, "market closed")
            _check(len(state_store.list_pending_buys()) == 2, "duplicate queued buys stay independent")
            _check(len(state_store.list_pending_sells()) == 2, "duplicate queued sells stay independent")
        finally:
            state_store.STATE_PATH = original_path


def run_all() -> None:
    test_durable_queue_concurrency()
    print("PASS durable queue: 500 concurrent signals, unique claims, cleanup")
    test_retry_dead_letter_and_restart_recovery()
    print("PASS retry/dead-letter and restart recovery")
    test_atomic_state_and_duplicate_trade_queues()
    print("PASS atomic JSON state and duplicate queued equity trades")
    print("PRODUCTION QUEUE TESTS PASSED")


if __name__ == "__main__":
    run_all()
