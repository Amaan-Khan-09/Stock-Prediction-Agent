"""Offline process-lock checks for the production Discord agent."""
from __future__ import annotations

import subprocess
import sys

from .runtime_lock import acquire_runtime_lock, release_runtime_lock


def run_all() -> None:
    acquired, error = acquire_runtime_lock()
    assert acquired and not error
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from discord_stock_prediction_agent.runtime_lock import "
                "acquire_runtime_lock; ok, _ = acquire_runtime_lock(); "
                "raise SystemExit(1 if ok else 0)"
            ),
        ],
        check=False,
        timeout=20,
    )
    assert child.returncode == 0
    release_runtime_lock()
    reacquired, error = acquire_runtime_lock()
    assert reacquired and not error
    release_runtime_lock()
    print("RUNTIME LOCK TESTS PASSED: duplicate process blocked and clean restart allowed")


if __name__ == "__main__":
    run_all()
