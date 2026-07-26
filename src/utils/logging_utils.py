"""Structured logging helpers — never logs tokens or credentials."""
from __future__ import annotations

import logging
import sys
from typing import Any

_REDACTED = "[REDACTED]"
_SENSITIVE = {"authorization", "access_token", "refresh_token", "client_secret", "password", "token", "api_key", "x-rapidapi-key"}


def _scrub(msg: str) -> str:
    lower = msg.lower()
    for key in _SENSITIVE:
        if key in lower:
            return f"[LOG REDACTED — message contained sensitive key: {key}]"
    return msg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_api_call(logger: logging.Logger, service: str, endpoint: str, status: Any) -> None:
    logger.info(f"[{service}] {endpoint} -> {status}")


def log_error(logger: logging.Logger, service: str, error: Exception) -> None:
    msg = _scrub(str(error))
    logger.error(f"[{service}] ERROR: {msg}")
