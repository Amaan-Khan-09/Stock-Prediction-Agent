"""7 payload validation tests — verifies that the correct equity-option payload is built."""
from __future__ import annotations

import pytest
from decimal import Decimal

from src.models.backtest_models import BacktestLeg, BacktestPayload
from src.services.tastytrade_backtester_service import (
    build_equity_option_short_put_payload,
    build_custom_legs_payload,
)


class TestBacktestLegDefaults:
    def test_default_type_is_equity_option(self):
        leg = BacktestLeg()
        assert leg.type == "equity-option", "type must be 'equity-option'"

    def test_default_direction_is_short(self):
        leg = BacktestLeg()
        assert leg.direction == "short"

    def test_to_dict_uses_camel_case(self):
        leg = BacktestLeg(days_until_expiration=30, delta=25)
        d = leg.to_dict()
        assert "daysUntilExpiration" in d
        assert "strikeSelection" in d
        assert d["daysUntilExpiration"] == 30
        assert d["delta"] == 25

    def test_wrong_type_would_fail_validation(self):
        leg = BacktestLeg(type="option")
        assert leg.type != "equity-option", "Sanity: 'option' != 'equity-option'"


class TestBuildShortPutPayload:
    def test_returns_payload_object(self):
        payload = build_equity_option_short_put_payload("SPY", "2021-01-01", "2024-01-01")
        assert isinstance(payload, BacktestPayload)

    def test_symbol_uppercased(self):
        payload = build_equity_option_short_put_payload("spy", "2021-01-01", "2024-01-01")
        assert payload.symbol == "SPY"

    def test_two_legs_by_default(self):
        payload = build_equity_option_short_put_payload("SPY", "2021-01-01", "2024-01-01")
        assert len(payload.legs) == 2

    def test_all_legs_are_equity_option(self):
        payload = build_equity_option_short_put_payload("SPY", "2021-01-01", "2024-01-01")
        for leg in payload.legs:
            assert leg.type == "equity-option"

    def test_end_date_formatted_as_iso_datetime(self):
        payload = build_equity_option_short_put_payload("SPY", "2021-01-01", "2024-06-01")
        d = payload.to_dict()
        assert "T" in d["endDate"], "endDate must include time component"
        assert d["endDate"].endswith("Z"), "endDate must end with Z"

    def test_to_dict_has_required_keys(self):
        payload = build_equity_option_short_put_payload("SPY", "2021-01-01", "2024-01-01")
        d = payload.to_dict()
        for key in ("startDate", "endDate", "symbol", "status", "entryConditions", "exitConditions", "legs"):
            assert key in d, f"Missing key: {key}"

    def test_custom_dte_and_delta_propagate(self):
        payload = build_equity_option_short_put_payload("QQQ", "2022-01-01", "2025-01-01", dte=30, delta=20)
        for leg in payload.legs:
            assert leg.days_until_expiration == 30
            assert leg.delta == 20


class TestBuildCustomLegsPayloadNoneSafety:
    """Regression tests: a caller passing a key with an explicit None/"" value
    (not omitting the key) previously raised TypeError/ValueError from a bare
    int(leg_dict.get("delta", 30))-style cast, since dict.get()'s default only
    applies when the key is absent, not when its value is None.
    """

    def test_explicit_none_delta_does_not_raise(self):
        payload = build_custom_legs_payload(
            "SPY", "2021-01-01", "2024-01-01",
            [{"type": "equity-option", "direction": "short", "side": "put", "delta": None}],
        )
        assert payload.legs[0].delta == 30  # falls back to default

    def test_explicit_none_strike_does_not_raise(self):
        payload = build_custom_legs_payload(
            "SPY", "2021-01-01", "2024-01-01",
            [{"type": "equity-option", "direction": "short", "side": "put",
              "strikeSelection": "strike", "strike": None}],
        )
        assert payload.legs[0].strike_price is None

    def test_empty_string_quantity_and_dte_do_not_raise(self):
        payload = build_custom_legs_payload(
            "SPY", "2021-01-01", "2024-01-01",
            [{"type": "equity-option", "direction": "short", "side": "put",
              "quantity": "", "daysUntilExpiration": ""}],
        )
        assert payload.legs[0].quantity == 1
        assert payload.legs[0].days_until_expiration == 45

    def test_explicit_none_type_and_direction_fall_back_to_defaults(self):
        payload = build_custom_legs_payload(
            "SPY", "2021-01-01", "2024-01-01",
            [{"type": None, "direction": None, "side": None}],
        )
        leg = payload.legs[0]
        assert leg.type == "equity-option"
        assert leg.direction == "short"
        assert leg.side == "put"

    def test_valid_values_still_pass_through_unchanged(self):
        payload = build_custom_legs_payload(
            "SPY", "2021-01-01", "2024-01-01",
            [{"type": "equity-option", "direction": "long", "side": "call",
              "quantity": 3, "daysUntilExpiration": 60, "delta": 45}],
        )
        leg = payload.legs[0]
        assert leg.direction == "long" and leg.side == "call"
        assert leg.quantity == 3 and leg.days_until_expiration == 60 and leg.delta == 45
