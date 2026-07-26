"""Cross-platform single-process lock for the local Discord agent runtime."""
from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import BinaryIO, Optional

from .config import AGENT_DIR


LOCK_PATH = AGENT_DIR / "discord_agent.lock"
_LOCK_HANDLE: Optional[BinaryIO] = None


def acquire_runtime_lock() -> tuple[bool, str]:
    global _LOCK_HANDLE
    if _LOCK_HANDLE is not None:
        return True, ""

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+b")
    if LOCK_PATH.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        handle.close()
        return False, (
            "Another Discord agent process is already using this project state. "
            "Stop the existing process before starting another."
        )

    handle.seek(0)
    handle.write(str(os.getpid()).encode("ascii"))
    handle.flush()
    _LOCK_HANDLE = handle
    atexit.register(release_runtime_lock)
    return True, ""


def release_runtime_lock() -> None:
    global _LOCK_HANDLE
    handle = _LOCK_HANDLE
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()
        _LOCK_HANDLE = None
