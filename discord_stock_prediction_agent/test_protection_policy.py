"""Boundary and persisted-position tests for equity and option protection."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from . import state_store
from .protection_policy import build_protection_levels, evaluate_protection


def test_equity_protection_boundaries() -> None:
    levels = build_protection_levels(
        100.0, stop_loss_pct=1.0, take_profit_pct=10.0
    )
    assert levels.stop_price == 99.0
    assert levels.target_price == 110.0
    assert evaluate_protection(99.01, levels).triggered is False
    assert evaluate_protection(99.0, levels).reason == "stop_loss"
    assert evaluate_protection(109.99, levels).triggered is False
    assert evaluate_protection(110.0, levels).reason == "take_profit"


def test_long_option_protection_boundaries() -> None:
    levels = build_protection_levels(
        10.0, stop_loss_pct=5.0, take_profit_pct=10.0
    )
    assert levels.stop_price == 9.5
    assert levels.target_price == 11.0
    assert evaluate_protection(9.51, levels).triggered is False
    assert evaluate_protection(9.5, levels).reason == "stop_loss"
    assert evaluate_protection(10.99, levels).triggered is False
    assert evaluate_protection(11.0, levels).reason == "take_profit"


def test_short_option_protection_directions() -> None:
    levels = build_protection_levels(
        10.0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
        short_position=True,
    )
    assert levels.stop_price == 10.5
    assert levels.target_price == 9.0
    assert evaluate_protection(10.5, levels, short_position=True).reason == "stop_loss"
    assert evaluate_protection(9.0, levels, short_position=True).reason == "take_profit"


def test_explicit_option_levels_override_defaults() -> None:
    levels = build_protection_levels(
        10.0,
        stop_loss_pct=5.0,
        take_profit_pct=10.0,
        explicit_stop=8.0,
        explicit_target=15.0,
    )
    assert levels.stop_price == 8.0
    assert levels.target_price == 15.0


def test_fill_based_levels_are_persisted_and_recomputed() -> None:
    with TemporaryDirectory() as tmp:
        original = state_store.STATE_PATH
        state_store.STATE_PATH = Path(tmp) / "agent_state.json"
        try:
            state_store.upsert_position("AAPL", 2, 100.0, "fill-1", 1.0, 10.0)
            equity = state_store.list_positions()[0]
            assert equity["stop_price"] == 99.0
            assert equity["target_price"] == 110.0

            state_store.upsert_option_position(
                "AAPL260821C00240000",
                "AAPL",
                "CALL",
                240,
                "2026-08-21",
                1,
                10.0,
                "option-fill-1",
            )
            option = state_store.list_option_positions()[0]
            assert option["stop_loss"] == 9.5
            assert option["target_price"] == 11.0
            assert option["stop_loss_source"] == "default_5pct"
            assert option["target_price_source"] == "default_10pct"

            # A later partial fill changes the weighted entry and therefore
            # recomputes default levels instead of keeping stale boundaries.
            state_store.upsert_option_position(
                "AAPL260821C00240000",
                "AAPL",
                "CALL",
                240,
                "2026-08-21",
                1,
                12.0,
                "option-fill-2",
            )
            option = state_store.list_option_positions()[0]
            assert option["entry_price"] == 11.0
            assert option["stop_loss"] == 10.45
            assert option["target_price"] == 12.1
        finally:
            state_store.STATE_PATH = original
