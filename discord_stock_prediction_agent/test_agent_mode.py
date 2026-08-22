"""Regression tests for the persistent Agent ON/OFF operating mode."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from . import discord_agent
from . import state_store


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_mode_persistence() -> None:
    with TemporaryDirectory() as tmp:
        original_path = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            _check(state_store.get_agent_mode() == "ON", "new state must default to Agent ON")
            control = state_store.set_agent_mode("off", "admin-1")
            _check(control["mode"] == "OFF", "Agent OFF must be stored")
            _check(control["updated_by"] == "admin-1", "mode author must be stored")
            _check(state_store.get_agent_mode() == "OFF", "Agent OFF must persist")
            state_store.set_agent_mode("ON", "admin-2")
            _check(state_store.get_agent_mode() == "ON", "Agent ON must persist")
        finally:
            state_store.STATE_PATH = original_path


def test_invalid_mode_rejected() -> None:
    with TemporaryDirectory() as tmp:
        original_path = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            try:
                state_store.set_agent_mode("MAYBE")
            except ValueError:
                return
            raise AssertionError("invalid Agent mode was accepted")
        finally:
            state_store.STATE_PATH = original_path


def test_direct_signal_mapping() -> None:
    buy = discord_agent._direct_signal_decision("BUY", 72)
    sell = discord_agent._direct_signal_decision("SELL", 68)
    hold = discord_agent._direct_signal_decision("HOLD", 40)
    _check(buy.action == "BUY" and buy.score == 72, "direct BUY mapping failed")
    _check(sell.action == "SELL" and sell.score == 68, "direct SELL mapping failed")
    _check(hold.action == "HOLD", "HOLD must remain non-trading")


def test_direct_option_mapping() -> None:
    validation = discord_agent._direct_option_validation("BUY")
    _check(validation["status"] == "AGENT_OFF_DIRECT", "direct option status missing")
    _check(validation["decision"] == "BUY", "direct option BUY mapping failed")
    _check(
        "AGENT_OFF_DIRECT" in discord_agent._allowed_option_strategy_statuses(),
        "direct option status must pass the routing gate",
    )


def test_tick_reaction_removed() -> None:
    source = Path(discord_agent.__file__).read_text(encoding="utf-8")
    _check('add_reaction("✅")' not in source, "tick reaction is still present")


def test_mode_commands_registered() -> None:
    commands = {command.name for command in discord_agent.bot.commands}
    _check(
        {"agent_on", "agent_off", "agent_mode"}.issubset(commands),
        "Agent mode commands are not all registered",
    )


def test_automate_agent_mode_persistence() -> None:
    """automate_agent's own on/off switch is independent state, separate
    from agent_control -- toggling it must not touch the general agent
    mode, and vice versa."""
    with TemporaryDirectory() as tmp:
        original_path = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            _check(
                state_store.get_automate_agent_mode() == "ON",
                "new state must default to automate_agent ON",
            )
            control = state_store.set_automate_agent_mode("off", "admin-1")
            _check(control["mode"] == "OFF", "automate_agent OFF must be stored")
            _check(control["updated_by"] == "admin-1", "mode author must be stored")
            _check(state_store.get_automate_agent_mode() == "OFF", "automate_agent OFF must persist")
            _check(
                state_store.get_agent_mode() == "ON",
                "toggling automate_agent's switch must not touch the general agent mode",
            )
            state_store.set_automate_agent_mode("ON", "admin-2")
            _check(state_store.get_automate_agent_mode() == "ON", "automate_agent ON must persist")
        finally:
            state_store.STATE_PATH = original_path


def test_automate_agent_invalid_mode_rejected() -> None:
    with TemporaryDirectory() as tmp:
        original_path = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            try:
                state_store.set_automate_agent_mode("MAYBE")
            except ValueError:
                return
            raise AssertionError("invalid automate_agent mode was accepted")
        finally:
            state_store.STATE_PATH = original_path


def test_automate_agent_mode_commands_registered() -> None:
    commands = {command.name for command in discord_agent.bot.commands}
    _check(
        {"automate_agent_on", "automate_agent_off", "automate_agent_mode"}.issubset(commands),
        "automate_agent mode commands are not all registered",
    )


def run_all() -> None:
    test_mode_persistence()
    test_invalid_mode_rejected()
    test_direct_signal_mapping()
    test_direct_option_mapping()
    test_tick_reaction_removed()
    test_mode_commands_registered()
    test_automate_agent_mode_persistence()
    test_automate_agent_invalid_mode_rejected()
    test_automate_agent_mode_commands_registered()
    print("AGENT MODE TESTS PASSED")


if __name__ == "__main__":
    run_all()
