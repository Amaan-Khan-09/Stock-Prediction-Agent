"""Shared, deterministic protection levels for filled paper positions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtectionLevels:
    stop_price: float
    target_price: float


@dataclass(frozen=True)
class ProtectionTrigger:
    triggered: bool
    reason: str = ""
    trigger_price: float = 0.0


def build_protection_levels(
    entry_price: float,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    short_position: bool = False,
    explicit_stop: float | None = None,
    explicit_target: float | None = None,
) -> ProtectionLevels:
    """Build absolute protection levels from the broker's fill price.

    Long positions lose value when price falls. Short-option positions lose
    value when the premium rises, so their default stop/target directions are
    reversed. Explicit signal SL/TP values take precedence over defaults.
    """
    entry = float(entry_price or 0)
    if entry <= 0:
        return ProtectionLevels(0.0, 0.0)
    stop_pct = max(0.0, float(stop_loss_pct or 0)) / 100.0
    target_pct = max(0.0, float(take_profit_pct or 0)) / 100.0
    default_stop = entry * (1 + stop_pct if short_position else 1 - stop_pct)
    default_target = entry * (1 - target_pct if short_position else 1 + target_pct)
    stop = float(explicit_stop) if explicit_stop is not None and float(explicit_stop) > 0 else default_stop
    target = (
        float(explicit_target)
        if explicit_target is not None and float(explicit_target) > 0
        else default_target
    )
    return ProtectionLevels(round(stop, 6), round(target, 6))


def evaluate_protection(
    current_price: float,
    levels: ProtectionLevels,
    *,
    short_position: bool = False,
) -> ProtectionTrigger:
    """Return the first protection boundary reached at the observed price."""
    current = float(current_price or 0)
    if current <= 0:
        return ProtectionTrigger(False)
    if short_position:
        if levels.stop_price > 0 and current >= levels.stop_price:
            return ProtectionTrigger(True, "stop_loss", levels.stop_price)
        if levels.target_price > 0 and current <= levels.target_price:
            return ProtectionTrigger(True, "take_profit", levels.target_price)
    else:
        if levels.stop_price > 0 and current <= levels.stop_price:
            return ProtectionTrigger(True, "stop_loss", levels.stop_price)
        if levels.target_price > 0 and current >= levels.target_price:
            return ProtectionTrigger(True, "take_profit", levels.target_price)
    return ProtectionTrigger(False)
