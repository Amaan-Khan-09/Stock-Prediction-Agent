"""Review-card checks for complete option signal details in both agent modes."""
from __future__ import annotations

from . import discord_agent
from .options_parser import classify_and_parse


SIGNAL = "BTO NVDA 210C 10/17 @5.40 Qty 5 LIMIT TP1 7.20 TP2 9.00 TP3 11.50 SL 4.10"


def _fields(embed) -> dict[str, str]:
    return {field.name: field.value for field in embed.fields}


def _field_contract(fields: dict[str, str]) -> str:
    return "\n".join(
        value
        for name, value in fields.items()
        if name.startswith("Parsed Field Contract")
    )


def _assert_complete_summary(fields: dict[str, str]) -> None:
    summary = fields["Parsed Signal"]
    assert "Asset Type: OPTION" in summary
    assert "Status: VALID" in summary
    assert "Action: BUY_TO_OPEN" in summary
    assert "Symbol: NVDA" in summary
    assert "Option Type: CALL" in summary
    assert "Strike: 210" in summary
    assert "Expiration: 2026-10-17" in summary
    assert "Qty: 5" in summary
    assert "Order Type: LIMIT" in summary
    assert "Entry: $5.40" in summary
    assert "TP1 $7.20 | TP2 $9.00 | TP3 $11.50" in summary
    assert "Stop Loss: $4.10" in summary
    assert "TPs: 7.2 / 9 / 11.5" in fields["Option Signal Quality"]


def test_agent_off_review_contains_complete_parsed_signal() -> None:
    option = classify_and_parse(SIGNAL).option
    assert option and option.valid
    decision = discord_agent._direct_signal_decision("BUY", 100)
    validation = discord_agent._direct_option_validation("BUY")
    quality = discord_agent._option_signal_quality(option)
    embed = discord_agent._option_embed(option, {}, "BUY", decision, validation, quality)
    fields = _fields(embed)
    _assert_complete_summary(fields)
    assert "Trade Setup" not in fields
    assert fields["Decision Source"] == "Incoming option signal"


def test_agent_on_review_contains_same_complete_parsed_signal() -> None:
    option = classify_and_parse(SIGNAL).option
    assert option and option.valid
    decision = discord_agent.DecisionResult("BUY", "approved", 0, 0, 0, 100, "test")
    validation = {
        "status": "SUCCESS",
        "decision": "BUY",
        "strategy_input": {
            "symbol": "NVDA",
            "direction_label": "Buy",
            "opt_type": "Call",
            "quantity": 5,
            "requested_strike": 210,
            "strike_price": 210,
            "delta": 30,
            "dte": 80,
        },
        "backtest": {"profit_loss": 100, "win_rate": 0.6},
    }
    quality = discord_agent._option_signal_quality(option)
    embed = discord_agent._option_embed(option, {}, "BUY", decision, validation, quality)
    fields = _fields(embed)
    _assert_complete_summary(fields)
    assert "Trade Setup" in fields
    assert fields["Decision Source"] == "Options strategy validation"


def test_lifecycle_conditions_are_visible_in_both_agent_modes() -> None:
    signals = {
        "BTO AVGO 420C 11/21 @6.80 Qty 2. Add 3 more contracts above 425 breakout.": (
            "Scale In: add 3 contract(s) when AVGO > $425",
        ),
        "BTO NFLX 1450P 12/19 @18.60 Qty 2 EXIT ALL POSITIONS 30 MINUTES BEFORE MARKET CLOSE IF TARGET NOT HIT": (
            "Time Exit: close remaining contracts 30 minutes before market close if target has not been hit",
        ),
        "Starter on ARM 190C 10/17 @4.35. Looking to add over 195. Risking only 1% on this trade.": (
            "Position Type: Starter",
            "Risk Stop: 1% below the filled option premium",
            "Scale In: add 1 contract(s) when ARM > $195",
        ),
    }
    for signal, expected_lines in signals.items():
        option = classify_and_parse(signal).option
        assert option and option.valid
        quality = discord_agent._option_signal_quality(option)
        direct = discord_agent._option_embed(
            option,
            {},
            "BUY",
            discord_agent._direct_signal_decision("BUY", 100),
            discord_agent._direct_option_validation("BUY"),
            quality,
        )
        agent_on = discord_agent._option_embed(
            option,
            {},
            "BUY",
            discord_agent.DecisionResult("BUY", "approved", 0, 0, 0, 100, "test"),
            {"status": "SUCCESS", "decision": "BUY"},
            quality,
        )
        for embed in (direct, agent_on):
            summary = _fields(embed)["Parsed Signal"]
            for expected in expected_lines:
                assert expected in summary


def test_agent_modes_render_the_same_complete_semantic_contract() -> None:
    signal = (
        "BTO CRM 320C 12/19/2026 QTY 3 MAX PREMIUM 5.50 ONLY IF CRM > 325 "
        "CANCEL 3:45 PM ET MAX RISK $900 EXIT 20 MIN BEFORE CLOSE"
    )
    option = classify_and_parse(signal).option
    assert option and option.valid
    quality = discord_agent._option_signal_quality(option)
    direct = discord_agent._option_embed(
        option,
        {},
        "BUY",
        discord_agent._direct_signal_decision("BUY", 100),
        discord_agent._direct_option_validation("BUY"),
        quality,
    )
    agent_on = discord_agent._option_embed(
        option,
        {},
        "BUY",
        discord_agent.DecisionResult("BUY", "approved", 0, 0, 0, 100, "test"),
        {"status": "SUCCESS", "decision": "BUY"},
        quality,
    )
    direct_contract = _field_contract(_fields(direct))
    agent_contract = _field_contract(_fields(agent_on))
    assert direct_contract == agent_contract
    for expected in (
        "Asset Type: OPTION",
        "Action: BUY_TO_OPEN",
        "Quantity: 3",
        "Maximum Entry Price: 5.5",
        "Entry Condition / Reference: UNDERLYING_PRICE",
        "Entry Condition / Operator: >",
        "Entry Condition / Value: 325",
        "Cancel If Not Triggered By / Time: 15:45",
        "Maximum Loss: 900",
        "Time Exit / Offset Minutes Before Market Close: 20",
        "Status: CONDITIONAL",
    ):
        assert expected in direct_contract


def test_conditional_deadline_expires_after_restart_day() -> None:
    contract = {
        "cancel_if_not_triggered_by": {"time": "23:59", "timezone": "ET"}
    }
    assert discord_agent._option_cancel_deadline_passed(
        contract, "2000-01-01T12:00:00Z"
    )
    assert not discord_agent._option_cancel_deadline_passed({}, "2000-01-01T12:00:00Z")
