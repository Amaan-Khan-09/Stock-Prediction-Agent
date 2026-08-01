from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from jsonl_store import append_jsonl


def test_concurrent_jsonl_appends_are_complete(tmp_path):
    destination = tmp_path / "accuracy.jsonl"

    def write(index: int) -> None:
        append_jsonl(destination, {"index": index, "payload": "x" * 256})

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(250)))

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 250
    assert {row["index"] for row in rows} == set(range(250))
