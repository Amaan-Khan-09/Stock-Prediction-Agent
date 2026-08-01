"""Regression test: run_options_backtest() must not silently discard a
degraded-payload signal from create_backtest().

create_backtest() can return (backtest_id, non_empty_message) when it had to
fall back to a mutated payload (e.g. "EXIT_COND_DROPPED:..." after Tastytrade
rejected the requested exit conditions). The only prior check was
`if not backtest_id:`, which is truthy here, so the degraded run was silently
reported as a clean SUCCESS with no indication the requested stop-loss/
take-profit was never actually applied.
"""
from __future__ import annotations

from unittest.mock import patch

from src.models.backtest_models import (
    BacktestResult,
    BacktestStatistics,
    BacktestTrial,
    BacktestValidationResult,
)
from src.services.tastytrade_backtester_service import run_options_backtest


def _fake_result() -> BacktestResult:
    trial = BacktestTrial.from_dict({"profitLoss": "100", "entryDate": "2024-01-01", "exitDate": "2024-01-02"})
    stats = BacktestStatistics.from_dict({"Total profit/loss": "100", "Win percentage": "100"})
    return BacktestResult(
        backtest_id="bt-1", symbol="SPY", status="complete", leg_type="equity-option",
        trials=[trial], statistics=stats, raw={},
    )


@patch("src.services.tastytrade_backtester_service.extract_backtest_summary")
@patch("src.services.tastytrade_backtester_service.validate_backtest_success")
@patch("src.services.tastytrade_backtester_service.parse_backtest_result")
@patch("src.services.tastytrade_backtester_service.poll_backtest")
@patch("src.services.tastytrade_backtester_service.create_backtest")
def test_degraded_exit_conditions_are_surfaced_not_discarded(
    mock_create, mock_poll, mock_parse, mock_validate, mock_extract
):
    result = _fake_result()
    mock_create.return_value = ("bt-1", "EXIT_COND_DROPPED:BACKTEST_HTTP_400:...")
    mock_poll.return_value = ({"status": "complete"}, "")
    mock_parse.return_value = result
    mock_validate.return_value = BacktestValidationResult.ok(result)
    mock_extract.return_value = {"backtest_id": "bt-1", "total_profit_loss": 100.0}

    summary = run_options_backtest("SPY", "2021-01-01", "2024-01-01", custom_legs=[
        {"type": "equity-option", "direction": "short", "side": "put"}
    ])

    assert summary["status"] == "SUCCESS"
    assert summary["exit_conditions_degraded"] is True
    assert "EXIT_COND_DROPPED" in summary["exit_conditions_degraded_reason"]


@patch("src.services.tastytrade_backtester_service.extract_backtest_summary")
@patch("src.services.tastytrade_backtester_service.validate_backtest_success")
@patch("src.services.tastytrade_backtester_service.parse_backtest_result")
@patch("src.services.tastytrade_backtester_service.poll_backtest")
@patch("src.services.tastytrade_backtester_service.create_backtest")
def test_clean_success_has_no_degraded_flag(
    mock_create, mock_poll, mock_parse, mock_validate, mock_extract
):
    result = _fake_result()
    mock_create.return_value = ("bt-1", "")  # empty message == full clean success
    mock_poll.return_value = ({"status": "complete"}, "")
    mock_parse.return_value = result
    mock_validate.return_value = BacktestValidationResult.ok(result)
    mock_extract.return_value = {"backtest_id": "bt-1", "total_profit_loss": 100.0}

    summary = run_options_backtest("SPY", "2021-01-01", "2024-01-01", custom_legs=[
        {"type": "equity-option", "direction": "short", "side": "put"}
    ])

    assert summary["exit_conditions_degraded"] is False
    assert "exit_conditions_degraded_reason" not in summary


@patch("src.services.tastytrade_backtester_service.extract_backtest_summary")
@patch("src.services.tastytrade_backtester_service.validate_backtest_success")
@patch("src.services.tastytrade_backtester_service.parse_backtest_result")
@patch("src.services.tastytrade_backtester_service.poll_backtest")
@patch("src.services.tastytrade_backtester_service.create_backtest")
def test_combined_tp_sl_gets_unverified_caveat(
    mock_create, mock_poll, mock_parse, mock_validate, mock_extract
):
    """When both stop_loss_pct and take_profit_pct are requested and the full
    (non-degraded) payload succeeds, the summary must still flag that the
    stop-loss side of Tastytrade's single-discriminator payload is unverified.
    """
    result = _fake_result()
    mock_create.return_value = ("bt-1", "")
    mock_poll.return_value = ({"status": "complete"}, "")
    mock_parse.return_value = result
    mock_validate.return_value = BacktestValidationResult.ok(result)
    mock_extract.return_value = {"backtest_id": "bt-1", "total_profit_loss": 100.0}

    with patch(
        "src.services.tastytrade_backtester_service.build_custom_legs_payload"
    ) as mock_build:
        from src.models.backtest_models import BacktestPayload
        mock_build.return_value = BacktestPayload(
            symbol="SPY", start_date="2021-01-01", end_date="2024-01-01",
            stop_loss_pct=50.0, take_profit_pct=50.0,
        )
        summary = run_options_backtest("SPY", "2021-01-01", "2024-01-01", custom_legs=[
            {"type": "equity-option", "direction": "short", "side": "put"}
        ])

    assert "exit_conditions_caveat" in summary
    assert "unverified" in summary["exit_conditions_caveat"].lower()
