"""Regression tests: WhatsApp parity for the 12 !agent_* Discord commands.

Discord messages reach @bot.command handlers via bot.get_context()/bot.invoke()
inside on_message, before anything is enqueued. WhatsApp messages never pass
through that -- they arrive via whatsapp_webhook.py straight into the durable
queue and are reconstructed as a _QueuedMessage shim, which has no command
concept at all. _try_dispatch_whatsapp_command() is the WhatsApp-side router
that reuses the exact same _build_*_text() helpers the Discord commands call,
so both transports share one implementation per command.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from . import discord_agent as agent
from . import state_store


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _queued(text: str, sender: str = "15550001111", is_group: bool = False) -> agent._QueuedMessage:
    return agent._QueuedMessage(
        {
            "raw_text": text,
            "message_id": "wamid.test-1",
            "user_id": sender,
            "channel_id": sender,
            "transport": "whatsapp",
            "reply_target": sender,
            "is_group": is_group,
        }
    )


def _run_dispatch(text: str, sender: str = "15550001111") -> tuple[bool, list[str]]:
    """Run _try_dispatch_whatsapp_command and capture what would have been sent."""
    sent: list[str] = []
    original_sender = agent.send_whatsapp_text

    def fake_sender(target: str, body: str, *, is_group: bool = False):
        sent.append(body)
        return True, ""

    agent.send_whatsapp_text = fake_sender
    try:
        handled = asyncio.run(agent._try_dispatch_whatsapp_command(_queued(text, sender)))
    finally:
        agent.send_whatsapp_text = original_sender
    return handled, sent


def _isolated_state():
    """Context manager-like helper mirroring test_agent_mode.py's isolation pattern."""
    tmp = TemporaryDirectory()
    original_path = state_store.STATE_PATH
    state_store.STATE_PATH = Path(tmp.name) / "agent_state.json"
    return tmp, original_path


def test_read_only_commands_return_nonempty_text():
    for command in (
        "agent_status", "agent_mode", "agent_positions", "agent_option_positions",
        "agent_summary", "agent_health", "agent_dead_letters", "agent_learning",
        "agent_option_validation", "automate_agent_mode",
    ):
        handled, sent = _run_dispatch(f"!{command}")
        _check(handled, f"!{command} was not recognized as a command")
        _check(len(sent) == 1 and bool(sent[0].strip()), f"!{command} produced no reply text")


def test_case_insensitive_command_name():
    handled, sent = _run_dispatch("!Agent_Status")
    _check(handled, "mixed-case command name was not recognized")
    _check(bool(sent[0].strip()), "mixed-case command produced no reply text")


def test_unknown_bang_word_falls_through_to_signal_processing():
    handled, sent = _run_dispatch("!notarealcommand")
    _check(not handled, "unrecognized !word must not be treated as a command")
    _check(sent == [], "no reply should be sent for a non-command")


def test_plain_signal_is_not_intercepted():
    handled, sent = _run_dispatch("buy AAPL qty 2")
    _check(not handled, "a normal trade signal must not be captured by the command router")
    _check(sent == [], "no reply should be sent by the command router for a normal signal")


def test_non_admin_cannot_change_agent_mode():
    tmp, original_path = _isolated_state()
    try:
        original_admins = getattr(agent.config, "whatsapp_admin_sender_ids")
        object.__setattr__(agent.config, "whatsapp_admin_sender_ids", "")
        try:
            handled, sent = _run_dispatch("!agent_on", sender="15550009999")
            _check(handled, "!agent_on must still be recognized as a command")
            _check(agent._WHATSAPP_NOT_AUTHORIZED_TEXT in sent[0], "non-admin must be rejected")
            _check(state_store.get_agent_mode() == "ON", "agent mode must be unchanged (was already ON by default)")
        finally:
            object.__setattr__(agent.config, "whatsapp_admin_sender_ids", original_admins)
    finally:
        state_store.STATE_PATH = original_path
        tmp.cleanup()


def test_admin_can_change_agent_mode():
    tmp, original_path = _isolated_state()
    try:
        admin_id = "15550003333"
        original_admins = getattr(agent.config, "whatsapp_admin_sender_ids")
        object.__setattr__(agent.config, "whatsapp_admin_sender_ids", admin_id)
        try:
            handled, sent = _run_dispatch("!agent_off", sender=admin_id)
            _check(handled, "!agent_off must be recognized")
            _check("now OFF" in sent[0], f"expected OFF confirmation, got: {sent[0]!r}")
            _check(state_store.get_agent_mode() == "OFF", "agent mode must actually change to OFF")

            handled2, sent2 = _run_dispatch("!agent_mode", sender=admin_id)
            _check(handled2 and "OFF" in sent2[0], "agent_mode must reflect the change")

            handled3, sent3 = _run_dispatch("!agent_on", sender=admin_id)
            _check(handled3 and "now ON" in sent3[0], "admin must be able to turn it back ON")
        finally:
            object.__setattr__(agent.config, "whatsapp_admin_sender_ids", original_admins)
    finally:
        state_store.STATE_PATH = original_path
        tmp.cleanup()


def test_non_admin_cannot_change_automate_agent_mode():
    tmp, original_path = _isolated_state()
    try:
        original_admins = getattr(agent.config, "whatsapp_admin_sender_ids")
        object.__setattr__(agent.config, "whatsapp_admin_sender_ids", "")
        try:
            handled, sent = _run_dispatch("!automate_agent_on", sender="15550009999")
            _check(handled, "!automate_agent_on must still be recognized as a command")
            _check(agent._WHATSAPP_NOT_AUTHORIZED_TEXT in sent[0], "non-admin must be rejected")
            _check(
                state_store.get_automate_agent_mode() == "ON",
                "automate_agent mode must be unchanged (was already ON by default)",
            )
        finally:
            object.__setattr__(agent.config, "whatsapp_admin_sender_ids", original_admins)
    finally:
        state_store.STATE_PATH = original_path
        tmp.cleanup()


def test_admin_can_change_automate_agent_mode_independently_of_agent_mode():
    tmp, original_path = _isolated_state()
    try:
        admin_id = "15550003333"
        original_admins = getattr(agent.config, "whatsapp_admin_sender_ids")
        object.__setattr__(agent.config, "whatsapp_admin_sender_ids", admin_id)
        try:
            handled, sent = _run_dispatch("!automate_agent_off", sender=admin_id)
            _check(handled, "!automate_agent_off must be recognized")
            _check("now OFF" in sent[0], f"expected OFF confirmation, got: {sent[0]!r}")
            _check(state_store.get_automate_agent_mode() == "OFF", "automate_agent mode must actually change to OFF")
            _check(
                state_store.get_agent_mode() == "ON",
                "the general agent mode (manual signals) must be untouched by this switch",
            )

            handled2, sent2 = _run_dispatch("!automate_agent_mode", sender=admin_id)
            _check(handled2 and "OFF" in sent2[0], "automate_agent_mode must reflect the change")

            handled3, sent3 = _run_dispatch("!automate_agent_on", sender=admin_id)
            _check(handled3 and "now ON" in sent3[0], "admin must be able to turn it back ON")
        finally:
            object.__setattr__(agent.config, "whatsapp_admin_sender_ids", original_admins)
    finally:
        state_store.STATE_PATH = original_path
        tmp.cleanup()


def test_retry_dead_requires_admin_and_accepts_optional_limit():
    original_admins = getattr(agent.config, "whatsapp_admin_sender_ids")
    object.__setattr__(agent.config, "whatsapp_admin_sender_ids", "")
    try:
        handled, sent = _run_dispatch("!agent_retry_dead", sender="15550009999")
        _check(handled, "!agent_retry_dead must be recognized")
        _check(agent._WHATSAPP_NOT_AUTHORIZED_TEXT in sent[0], "non-admin must be rejected")
    finally:
        object.__setattr__(agent.config, "whatsapp_admin_sender_ids", original_admins)

    admin_id = "15550003333"
    original_admins = getattr(agent.config, "whatsapp_admin_sender_ids")
    object.__setattr__(agent.config, "whatsapp_admin_sender_ids", admin_id)
    try:
        handled, sent = _run_dispatch("!agent_retry_dead 5", sender=admin_id)
        _check(handled and "dead-letter signal" in sent[0], "numeric-arg retry must succeed")
        handled2, sent2 = _run_dispatch("!agent_retry_dead notanumber", sender=admin_id)
        _check(handled2 and "dead-letter signal" in sent2[0], "garbage arg must fall back to default limit, not crash")
    finally:
        object.__setattr__(agent.config, "whatsapp_admin_sender_ids", original_admins)


def run_all() -> None:
    test_read_only_commands_return_nonempty_text()
    test_case_insensitive_command_name()
    test_unknown_bang_word_falls_through_to_signal_processing()
    test_plain_signal_is_not_intercepted()
    test_non_admin_cannot_change_agent_mode()
    test_admin_can_change_agent_mode()
    test_non_admin_cannot_change_automate_agent_mode()
    test_admin_can_change_automate_agent_mode_independently_of_agent_mode()
    test_retry_dead_requires_admin_and_accepts_optional_limit()
    print("WHATSAPP COMMAND PARITY TESTS PASSED")


if __name__ == "__main__":
    run_all()
