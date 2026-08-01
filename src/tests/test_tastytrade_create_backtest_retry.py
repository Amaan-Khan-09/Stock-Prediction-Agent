"""Regression tests for create_backtest()'s retry dispatch.

HTTP 429 (rate limited) and HTTP 400 (payload/format issue) must use two
distinct strategies: a 429 backs off and resubmits the SAME payload, while a
400 mutates the payload (drop combined exit conditions, then drop exit
conditions entirely). They were previously conflated -- a 429 fell into the
same payload-mutation ladder as a 400 instead of backing off.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.backtest_models import BacktestPayload
from src.services.tastytrade_backtester_service import (
    _parse_retry_after_seconds,
    create_backtest,
)


def _payload() -> BacktestPayload:
    return BacktestPayload(symbol="SPY", start_date="2021-01-01", end_date="2024-01-01T00:00:00Z")


def _payload_with_combined_exit_conditions() -> BacktestPayload:
    return BacktestPayload(
        symbol="SPY", start_date="2021-01-01", end_date="2024-01-01T00:00:00Z",
        stop_loss_pct=50.0, take_profit_pct=50.0,
    )


class TestParseRetryAfter:
    def test_numeric_value_used(self):
        assert _parse_retry_after_seconds("RATE_LIMITED:429:retry_after=5") == 5.0

    def test_unknown_falls_back_to_default(self):
        assert _parse_retry_after_seconds("RATE_LIMITED:429:retry_after=unknown") == 2.0

    def test_value_clamped_to_max(self):
        assert _parse_retry_after_seconds("RATE_LIMITED:429:retry_after=999") == 30.0

    def test_missing_marker_falls_back_to_default(self):
        assert _parse_retry_after_seconds("BACKTEST_HTTP_400:some other error") == 2.0


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None, headers=None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers = headers or {}
        self.text = str(json_body or "")

    def json(self):
        return self._json_body


@patch("src.services.tastytrade_backtester_service.time.sleep")
@patch("src.services.tastytrade_backtester_service.get_auth_headers", return_value={})
@patch("src.services.tastytrade_backtester_service.get_access_token", return_value=("tok", ""))
@patch("src.services.tastytrade_backtester_service.requests.post")
def test_429_retries_same_payload_without_mutation(mock_post, _mock_token, _mock_headers, mock_sleep):
    """A 429 must resubmit the identical body -- not the take-profit-only or
    no-exit-conditions mutated payloads -- and must sleep using Retry-After."""
    responses = [
        _FakeResponse(429, headers={"Retry-After": "3"}),
        _FakeResponse(429, headers={"Retry-After": "1"}),
        _FakeResponse(201, {"id": "bt-123"}),
    ]
    mock_post.side_effect = responses

    backtest_id, err = create_backtest(_payload())

    assert backtest_id == "bt-123"
    assert err == ""
    assert mock_post.call_count == 3
    bodies = [call.kwargs["json"] for call in mock_post.call_args_list]
    assert bodies[0] == bodies[1] == bodies[2], "429 retries must resubmit the exact same payload"
    mock_sleep.assert_any_call(3.0)
    mock_sleep.assert_any_call(1.0)


@patch("src.services.tastytrade_backtester_service.time.sleep")
@patch("src.services.tastytrade_backtester_service.get_auth_headers", return_value={})
@patch("src.services.tastytrade_backtester_service.get_access_token", return_value=("tok", ""))
@patch("src.services.tastytrade_backtester_service.requests.post")
def test_429_gives_up_after_max_attempts(mock_post, _mock_token, _mock_headers, mock_sleep):
    mock_post.return_value = _FakeResponse(429, headers={"Retry-After": "1"})

    backtest_id, err = create_backtest(_payload())

    assert backtest_id is None
    assert "RATE_LIMITED" in err
    assert mock_post.call_count == 3  # _RATE_LIMIT_MAX_ATTEMPTS


@patch("src.services.tastytrade_backtester_service.time.sleep")
@patch("src.services.tastytrade_backtester_service.get_auth_headers", return_value={})
@patch("src.services.tastytrade_backtester_service.get_access_token", return_value=("tok", ""))
@patch("src.services.tastytrade_backtester_service.requests.post")
def test_400_still_mutates_payload_not_backoff(mock_post, _mock_token, _mock_headers, mock_sleep):
    """A plain 400 (not rate limited) must go straight into the existing
    payload-mutation ladder, with no sleep/backoff involved."""
    responses = [
        _FakeResponse(400, {"message": "invalid exitConditions"}),
        _FakeResponse(201, {"id": "bt-456"}),
    ]
    mock_post.side_effect = responses

    backtest_id, err = create_backtest(_payload_with_combined_exit_conditions())

    assert backtest_id == "bt-456"
    mock_sleep.assert_not_called()
    # Retry 2 (take-profit only) must have been the payload actually sent.
    second_body = mock_post.call_args_list[1].kwargs["json"]
    assert second_body["exitConditions"] == {"type": "profit_percentage", "profitPercentage": 50.0}
