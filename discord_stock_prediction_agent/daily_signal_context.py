"""Per-channel, same-trading-day default root/underlying memory.

options_parser.classify_and_parse() is intentionally stateless -- each
message is parsed with no memory of earlier messages (see its module
docstring). Real trade-alert rooms lean on shared context constantly: the
underlying is announced once ("SPX : Bought SPX 7665C...") and every
follow-up in that day's thread ("Sold 50% at 4.80", "Sold 70% 7770C...")
omits it entirely.

This module adds exactly one piece of memory on top of the stateless
parser: the last root/underlying seen per channel, valid only for the
current UTC calendar day. It never remembers strike, side, or quantity, so
it can only ever fill in a root a message was completely missing -- it can
never override a root the message itself stated, and it can't fabricate a
strike that was never there. The moment the UTC date rolls over, a
channel's remembered root is discarded on first access -- nothing carries
into a new trading day.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .options_parser import ParsedMessage, classify_and_parse

_DEFAULT_CHANNEL = "_default"

_context: dict[str, dict[str, str]] = {}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def remember_root(channel_key: str, root: str) -> None:
    root = (root or "").strip().upper()
    if not root:
        return
    _context[channel_key or _DEFAULT_CHANNEL] = {"date": _today(), "root": root}


def get_default_root(channel_key: str) -> str:
    key = channel_key or _DEFAULT_CHANNEL
    entry = _context.get(key)
    if not entry:
        return ""
    if entry.get("date") != _today():
        del _context[key]
        return ""
    return entry.get("root", "")


def clear_channel(channel_key: str) -> None:
    _context.pop(channel_key or _DEFAULT_CHANNEL, None)


def clear_all() -> None:
    _context.clear()


def classify_and_parse_with_daily_context(text: str, channel_key: str = "") -> ParsedMessage:
    """classify_and_parse(), aided by same-day root memory for this channel.

    Only ever fills in a root that was otherwise completely missing from the
    message -- never overrides a root the message itself stated -- so this
    cannot misroute a signal to a different underlying than the one written.
    """
    parsed = classify_and_parse(text)

    already_resolved_option = (
        parsed.kind == "OPTION" and parsed.option is not None and parsed.option.valid
    )
    if already_resolved_option:
        remember_root(channel_key, parsed.option.root)
        return parsed

    if parsed.kind not in {"OPTION", "NO_TRADE", "INVALID"}:
        return parsed

    default_root = get_default_root(channel_key)
    if not default_root:
        return parsed

    retried = classify_and_parse(f"{default_root} {text}")
    if retried.kind == "OPTION" and retried.option is not None and retried.option.valid:
        remember_root(channel_key, retried.option.root)
        return retried
    return parsed
