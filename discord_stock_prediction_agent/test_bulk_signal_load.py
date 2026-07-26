"""Offline burst-load test for mixed Discord signal intake and parsing."""
from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import durable_signal_queue as durable
from .options_parser import classify_and_parse


SIGNALS = (
    "BUY AAPL QTY 2",
    "SELL TSLA QTY 1",
    "HOLD MSFT. Mixed indicators.",
    "BUY GOOGL IF PRICE CLOSES ABOVE 205 OTHERWISE HOLD",
    "BTO AAPL 240C 08/21 @3.45 SL 2.20 TP 5.80",
    "STC NVDA 190C 09/19 @8.50",
    "BTO AAPL 240C / STO AAPL 250C 09/19 @4.60 Debit Qty 5",
    "STO SPY 620P / BTO SPY 610P / STO SPY 670C / BTO SPY 680C 09/19 @2.15 Credit Qty 2",
    "No clear edge today.",
    "BUY",
)


def run_all() -> None:
    total = 2_400
    started = time.perf_counter()
    messages = [SIGNALS[index % len(SIGNALS)] for index in range(total)]
    with ThreadPoolExecutor(max_workers=32) as pool:
        parsed = list(pool.map(classify_and_parse, messages))
    assert len(parsed) == total
    assert all(item.kind in {"EQUITY", "OPTION", "NO_TRADE", "INVALID"} for item in parsed)
    assert any(item.kind == "EQUITY" for item in parsed)
    assert any(item.kind == "OPTION" for item in parsed)
    parse_seconds = time.perf_counter() - started

    with tempfile.TemporaryDirectory() as tmp:
        original = durable.QUEUE_PATH
        durable.QUEUE_PATH = Path(tmp) / "load_queue.sqlite3"
        try:
            durable.initialize_signal_queue()

            def enqueue(index: int):
                return durable.enqueue_signal(
                    messages[index],
                    f"user-{index % 2000}",
                    "channel-load-test",
                    f"message-{index}",
                    10_000,
                )

            with ThreadPoolExecutor(max_workers=32) as pool:
                results = list(pool.map(enqueue, range(total)))
            assert all(
                item.get("accepted") and item.get("status") == "queued"
                for item in results
            )
            assert durable.queue_stats()["queued"] == total

            sample_to_process = 200
            claimed = 0
            while claimed < sample_to_process:
                item = durable.claim_next_signal()
                if not item:
                    break
                durable.complete_signal(str(item["id"]))
                claimed += 1
            assert claimed == sample_to_process
            stats = durable.queue_stats()
            assert stats["queued"] == total - sample_to_process
            assert stats["processing"] == 0
        finally:
            durable.QUEUE_PATH = original

    print(
        f"BULK SIGNAL LOAD TEST PASSED: {total} mixed signals parsed and queued "
        f"without loss; {sample_to_process} atomically claimed/cleaned "
        f"(parse {parse_seconds:.2f}s, total {time.perf_counter() - started:.2f}s)"
    )


if __name__ == "__main__":
    run_all()
