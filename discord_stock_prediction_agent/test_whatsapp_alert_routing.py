"""Regression test: proactive/background alerts also reach WhatsApp.

_send_channel is the function the periodic monitor loop uses for alerts that
are NOT a reply to any incoming message (protection triggers, a queued order
finally filling, a contract becoming tradable, ...). Unlike _send_optional_channel
(used for direct replies), it had no WhatsApp path at all -- it only ever
called Discord's bot.get_channel/channel.send. Any Discord-only deployment
that turns on WHATSAPP_ALERT_TARGET should now also receive these ~30
call sites' worth of alerts, without touching each call site individually.
"""
from __future__ import annotations

import asyncio

from . import discord_agent as agent
from .config import config


def test_send_channel_dual_dispatches_to_whatsapp_when_configured() -> None:
    sent: list[tuple[str, str, bool]] = []

    def fake_sender(target: str, text: str, *, is_group: bool = False):
        sent.append((target, text, is_group))
        return True, ""

    original_sender = agent.send_whatsapp_text
    original_target = config.whatsapp_alert_target
    original_is_group = config.whatsapp_alert_is_group
    agent.send_whatsapp_text = fake_sender
    object.__setattr__(config, "whatsapp_alert_target", "120000000000099")
    object.__setattr__(config, "whatsapp_alert_is_group", True)
    try:
        # channel_id=0 (no Discord channel configured) must not prevent the
        # WhatsApp alert from still going out.
        asyncio.run(agent._send_channel(0, "Protection triggered for AAPL."))
    finally:
        agent.send_whatsapp_text = original_sender
        object.__setattr__(config, "whatsapp_alert_target", original_target)
        object.__setattr__(config, "whatsapp_alert_is_group", original_is_group)

    assert sent == [("120000000000099", "Protection triggered for AAPL.", True)]


def test_send_channel_skips_whatsapp_when_alert_target_not_configured() -> None:
    sent: list[tuple[str, str, bool]] = []

    def fake_sender(target: str, text: str, *, is_group: bool = False):
        sent.append((target, text, is_group))
        return True, ""

    original_sender = agent.send_whatsapp_text
    original_target = config.whatsapp_alert_target
    agent.send_whatsapp_text = fake_sender
    object.__setattr__(config, "whatsapp_alert_target", "")
    try:
        asyncio.run(agent._send_channel(0, "Protection triggered for AAPL."))
    finally:
        agent.send_whatsapp_text = original_sender
        object.__setattr__(config, "whatsapp_alert_target", original_target)

    assert sent == []


def run_all() -> None:
    test_send_channel_dual_dispatches_to_whatsapp_when_configured()
    test_send_channel_skips_whatsapp_when_alert_target_not_configured()
    print("WHATSAPP ALERT ROUTING TESTS PASSED")


if __name__ == "__main__":
    run_all()
