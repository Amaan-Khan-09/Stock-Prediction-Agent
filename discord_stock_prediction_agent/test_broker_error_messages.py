"""Regression tests for safe, useful Alpaca rejection messages."""
from __future__ import annotations

from dataclasses import replace

from . import discord_agent


def run_all() -> None:
    original_config = discord_agent.config
    discord_agent.config = replace(original_config, debug_output_enabled=False)
    try:
        cases = [
            (
                'Alpaca HTTP 403: {"message":"insufficient buying power","buying_power":"0"}',
                "insufficient equity buying power",
            ),
            (
                'Alpaca HTTP 403: {"message":"insufficient options buying power","options_buying_power":"0"}',
                "insufficient options buying power",
            ),
            (
                'Alpaca HTTP 403: {"message":"account not eligible to trade uncovered option contracts"}',
                "cannot trade uncovered option contracts",
            ),
            (
                'Alpaca HTTP 403: {"message":"potential wash trade detected"}',
                "wash-trade protection",
            ),
            (
                'Alpaca HTTP 422: {"message":"options market orders are only allowed during market hours"}',
                "options market is closed",
            ),
        ]
        for raw, expected in cases:
            rendered = discord_agent._public_error(raw)
            assert expected in rendered, (expected, rendered)
            assert "buying_power" not in rendered
        print(f"BROKER ERROR MESSAGE TESTS PASSED: {len(cases)} safe rejection categories")
    finally:
        discord_agent.config = original_config


if __name__ == "__main__":
    run_all()
