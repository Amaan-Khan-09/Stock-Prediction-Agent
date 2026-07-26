"""Strike-price contract tests for options strategy validation.

These tests ensure fixed-strike option inputs keep their strike all the way
into the tastytrade payload, while delta-selected inputs do not carry a stale
strike value.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prediction_validation_service import _build_spi
from src.services.tastytrade_backtester_service import build_custom_legs_payload


def test_build_spi_strips_strike_price_from_stock_prediction_input():
    raw = {
        "symbol": "AAPL",
        "historical_context_start_date": "2025-01-01",
        "prediction_origin_date": "2026-03-01",
        "target_date": "2026-04-01",
        "initial_capital": 50000.0,
        "benchmark": "SPY",
        "direction": "Buy",
        "opt_type": "Call",
        "quantity": 1,
        "delta": 30,
        "dte": 45,
        "strike_selection": "strike",
        "strike_price": 240.0,
    }

    spi = _build_spi(raw)

    assert "strike_price" not in spi
    assert "strike_selection" not in spi
    assert spi["symbol"] == "AAPL"


def test_tastytrade_payload_preserves_fixed_strike_price():
    payload = build_custom_legs_payload(
        symbol="AAPL",
        start_date="2026-03-01",
        end_date="2026-04-01",
        legs=[
            {
                "type": "equity-option",
                "direction": "long",
                "quantity": 1,
                "side": "call",
                "daysUntilExpiration": 45,
                "strikeSelection": "strike",
                "delta": 30,
                "strikePrice": 240.0,
            }
        ],
    ).to_dict()

    leg = payload["legs"][0]

    assert leg["strikeSelection"] == "strike"
    assert leg["strikePrice"] == 240.0
    assert "delta" not in leg


def test_tastytrade_payload_accepts_legacy_strike_alias():
    payload = build_custom_legs_payload(
        symbol="AAPL",
        start_date="2026-03-01",
        end_date="2026-04-01",
        legs=[
            {
                "type": "equity-option",
                "direction": "long",
                "quantity": 1,
                "side": "call",
                "daysUntilExpiration": 45,
                "strikeSelection": "strike",
                "delta": 30,
                "strike": 240.0,
            }
        ],
    ).to_dict()

    leg = payload["legs"][0]

    assert leg["strikeSelection"] == "strike"
    assert leg["strikePrice"] == 240.0
    assert "delta" not in leg


def test_tastytrade_payload_omits_strike_price_for_delta_selection():
    payload = build_custom_legs_payload(
        symbol="AAPL",
        start_date="2026-03-01",
        end_date="2026-04-01",
        legs=[
            {
                "type": "equity-option",
                "direction": "long",
                "quantity": 1,
                "side": "call",
                "daysUntilExpiration": 45,
                "strikeSelection": "delta",
                "delta": 30,
            }
        ],
    ).to_dict()

    leg = payload["legs"][0]

    assert leg["strikeSelection"] == "delta"
    assert "strikePrice" not in leg
