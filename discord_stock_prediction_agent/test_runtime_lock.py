"""Offline process-lock checks for the production Discord agent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from . import runtime_lock
from .runtime_lock import acquire_runtime_lock, release_runtime_lock


def run_all() -> None:
    # Use an isolated lock file rather than the real LOCK_PATH -- otherwise
    # this test collides with (and spuriously fails against) an actual
    # Discord agent process that happens to be running live on the same
    # machine, since both would be locking the same real file.
    original_lock_path = runtime_lock.LOCK_PATH
    with TemporaryDirectory() as tmp:
        runtime_lock.LOCK_PATH = Path(tmp) / "test.lock"
        try:
            acquired, error = acquire_runtime_lock()
            assert acquired and not error
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from discord_stock_prediction_agent import runtime_lock; "
                        f"runtime_lock.LOCK_PATH = Path({str(runtime_lock.LOCK_PATH)!r}); "
                        "ok, _ = runtime_lock.acquire_runtime_lock(); "
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
        finally:
            runtime_lock.LOCK_PATH = original_lock_path
    print("RUNTIME LOCK TESTS PASSED: duplicate process blocked and clean restart allowed")


if __name__ == "__main__":
    run_all()
