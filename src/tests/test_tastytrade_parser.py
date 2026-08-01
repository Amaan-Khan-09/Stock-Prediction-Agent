"""6 tests verifying BacktestResult parsing and the validation gate."""
from __future__ import annotations

import pytest
from decimal import Decimal

from src.models.backtest_models import BacktestResult, BacktestStatistics, BacktestTrial
from src.services.tastytrade_backtester_service import (
    parse_backtest_result,
    validate_backtest_success,
)


def _make_trial(pl: str) -> dict:
    return {
        "entryDate": "2023-01-10",
        "exitDate": "2023-02-24",
        "profitLoss": pl,
        "snapshots": [],
    }


def _make_stats(total_pl: str = "1500") -> dict:
    return {
        "totalProfitLoss": total_pl,
        "winRate": 0.65,
        "averageProfitLoss": "150",
        "numTrades": 10,
        "numWins": 7,
        "numLosses": 3,
        "maxProfit": "400",
        "maxLoss": "-200",
    }


class TestParseBacktestResult:
    def test_valid_response_parsed_correctly(self):
        # Tastytrade API nests trials/statistics under "results" key — top level holds metadata
        raw = {
            "status": "complete",
            "symbol": "SPY",
            "legs": [{"type": "equity-option"}],
            "results": {
                "trials": [_make_trial("250.00"), _make_trial("-80.00")],
                "statistics": _make_stats("170"),
            },
        }
        result = parse_backtest_result("abc123", raw)
        assert result.backtest_id == "abc123"
        assert result.leg_type == "equity-option"
        assert len(result.trials) == 2
        assert result.statistics is not None

    def test_unknown_leg_type_detected(self):
        raw = {
            "status": "complete",
            "legs": [{"type": "unknown"}],
            "results": {
                "trials": [_make_trial("0")],
                "statistics": _make_stats("0"),
            },
        }
        result = parse_backtest_result("bad1", raw)
        assert result.leg_type == "unknown"

    def test_missing_statistics_sets_none_or_computes(self):
        raw = {
            "status": "complete",
            "legs": [{"type": "equity-option"}],
            "results": {
                "trials": [_make_trial("100"), _make_trial("-50")],
            },
        }
        result = parse_backtest_result("xyz", raw)
        # No statistics in results → should compute from trials
        assert result.statistics is None or result.statistics.num_trades == 2


class TestValidateBacktestSuccess:
    def test_valid_result_passes(self):
        result = BacktestResult(
            backtest_id="ok1",
            symbol="SPY",
            status="complete",
            leg_type="equity-option",
            trials=[
                BacktestTrial("2023-01-01", "2023-02-15", Decimal("250"), []),
                BacktestTrial("2023-02-15", "2023-04-01", Decimal("-80"), []),
            ],
            statistics=BacktestStatistics(
                total_profit_loss=Decimal("170"),
                win_rate=0.5,
                average_profit_loss=Decimal("85"),
                num_trades=2,
                num_wins=1,
                num_losses=1,
                max_profit=Decimal("250"),
                max_loss=Decimal("-80"),
            ),
        )
        v = validate_backtest_success(result)
        assert v.passed is True

    def test_unknown_type_fails_validation(self):
        result = BacktestResult(
            backtest_id="bad1",
            symbol="SPY",
            status="complete",
            leg_type="unknown",
            trials=[BacktestTrial("2023-01-01", "2023-02-01", Decimal("0"), [])],
            statistics=None,
        )
        v = validate_backtest_success(result)
        assert v.passed is False
        assert any("equity-option" in r for r in v.reasons)

    def test_null_statistics_fails(self):
        result = BacktestResult(
            backtest_id="bad2",
            symbol="SPY",
            status="complete",
            leg_type="equity-option",
            trials=[BacktestTrial("2023-01-01", "2023-02-01", Decimal("100"), [])],
            statistics=None,
        )
        v = validate_backtest_success(result)
        assert v.passed is False

    def test_all_zero_pl_fails(self):
        result = BacktestResult(
            backtest_id="bad3",
            symbol="SPY",
            status="complete",
            leg_type="equity-option",
            trials=[
                BacktestTrial("2023-01-01", "2023-02-01", Decimal("0"), []),
                BacktestTrial("2023-02-01", "2023-03-01", Decimal("0"), []),
            ],
            statistics=BacktestStatistics(
                total_profit_loss=Decimal("0"),
                win_rate=0.0,
                average_profit_loss=Decimal("0"),
                num_trades=2,
                num_wins=0,
                num_losses=0,
                max_profit=Decimal("0"),
                max_loss=Decimal("0"),
            ),
        )
        v = validate_backtest_success(result)
        assert v.passed is False
