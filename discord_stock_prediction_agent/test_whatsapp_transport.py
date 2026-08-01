"""Deterministic tests for WhatsApp parsing and transport-aware durable queueing."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import tempfile
from pathlib import Path

from . import durable_signal_queue as queue
from .whatsapp_client import output_text
from .whatsapp_webhook import _extract_messages, verify_webhook_signature


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    payload = {
        "entry": [{"changes": [{"value": {"messages": [
            {
                "id": "wamid.direct-1", "from": "15550001111", "type": "text",
                "text": {"body": "BTO AAPL 240C 08/21 @3.45"},
            },
            {
                "id": "wamid.group-1", "from": "15550002222",
                "group_id": "120000000000001", "type": "image",
                "image": {"caption": "BUY MSFT Qty 2 MARKET"},
            },
        ]}}]}]
    }
    messages = _extract_messages(payload)
    _check(len(messages) == 2, "expected direct and group messages")
    _check(messages[0]["target"] == "15550001111", "direct reply target mismatch")
    _check(messages[1]["target"] == "120000000000001", "group target mismatch")
    _check(messages[1]["text"] == "BUY MSFT Qty 2 MARKET", "caption not extracted")
    _check(output_text("Done", None) == "Done", "plain output conversion failed")

    # Signature verification is fail-closed and uses Meta's exact raw request bytes.
    from .whatsapp_webhook import config as webhook_config

    original_secret = webhook_config.whatsapp_app_secret
    object.__setattr__(webhook_config, "whatsapp_app_secret", "test-app-secret")
    try:
        raw_body = b'{"object":"whatsapp_business_account"}'
        signature = "sha256=" + hmac.new(
            b"test-app-secret", raw_body, hashlib.sha256
        ).hexdigest()
        _check(verify_webhook_signature(raw_body, signature), "valid signature rejected")
        _check(not verify_webhook_signature(raw_body + b"x", signature), "tampering accepted")
    finally:
        object.__setattr__(webhook_config, "whatsapp_app_secret", original_secret)

    # The Discord worker must route a reconstructed WhatsApp message back through
    # Cloud API output, never into a Discord review channel.
    from . import discord_agent as agent

    sent: list[tuple[str, str, bool]] = []
    original_sender = agent.send_whatsapp_text

    def fake_sender(target: str, text: str, *, is_group: bool = False):
        sent.append((target, text, is_group))
        return True, ""

    agent.send_whatsapp_text = fake_sender
    try:
        queued_message = agent._QueuedMessage(
            {
                "raw_text": messages[1]["text"],
                "message_id": messages[1]["id"],
                "user_id": messages[1]["sender"],
                "channel_id": messages[1]["target"],
                "transport": "whatsapp",
                "reply_target": messages[1]["target"],
                "is_group": True,
            }
        )
        asyncio.run(agent._send_review_or_reply(queued_message, "Signal accepted"))
    finally:
        agent.send_whatsapp_text = original_sender
    _check(sent == [(messages[1]["target"], "Signal accepted", True)], "reply routing failed")

    original_path = queue.QUEUE_PATH
    original_initialized = set(queue._INITIALIZED_PATHS)
    try:
        with tempfile.TemporaryDirectory() as directory:
            queue.QUEUE_PATH = Path(directory) / "signals.sqlite3"
            queue._INITIALIZED_PATHS.clear()
            accepted = queue.enqueue_signal(
                messages[1]["text"], messages[1]["sender"], messages[1]["target"],
                messages[1]["id"], transport="whatsapp",
                reply_target=messages[1]["target"], is_group=True,
            )
            _check(accepted["accepted"], "WhatsApp signal was not queued")
            duplicate = queue.enqueue_signal(
                messages[1]["text"], messages[1]["sender"], messages[1]["target"],
                messages[1]["id"], transport="whatsapp",
                reply_target=messages[1]["target"], is_group=True,
            )
            _check(duplicate.get("duplicate_delivery"), "delivery idempotency failed")
            claimed = queue.claim_next_signal()
            _check(claimed.get("transport") == "whatsapp", "transport not persisted")
            _check(bool(claimed.get("is_group")), "group flag not persisted")
            queue.fail_signal(claimed["id"], "test", max_attempts=1)
            _check(queue.dead_letter_count() == 1, "dead letter not recorded")
            _check(queue.retry_dead_signals() == 1, "dead letter not retried")
            retried = queue.claim_next_signal()
            queue.complete_signal(retried["id"])
            _check(queue.queue_stats()["depth"] == 0, "completed signal not removed")
    finally:
        queue.QUEUE_PATH = original_path
        queue._INITIALIZED_PATHS.clear()
        queue._INITIALIZED_PATHS.update(original_initialized)

    print("WhatsApp transport tests passed.")


if __name__ == "__main__":
    main()
