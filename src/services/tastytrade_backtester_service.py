"""Tastytrade Backtester Service.

Key discovery (breakthrough after many days of testing):
- type MUST be "equity-option" — using "option" or "equity" returns type="unknown",
  statistics=null, trials=null, profitLoss="0" in every snapshot.
- direction MUST be "short" or "long" (NOT "sell" or "buy").
- This API is OPTIONS-only. It cannot backtest plain equities/stocks.

Security rules:
- Never log Authorization header or token values.
- Do not expose tokens in exception messages.
- Do not fake/mock statistics and display them as real results.
- Do not show success unless validation passes.
- Use production API only: https://backtester.vast.tastyworks.com
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config.settings import settings
from src.models.backtest_models import (
    BacktestLeg,
    BacktestPayload,
    BacktestResult,
    BacktestStatistics,
    BacktestTrial,
    BacktestValidationResult,
)
from src.services.tastytrade_auth_service import get_access_token, get_auth_headers
from src.utils.date_utils import default_backtest_range, to_iso_datetime
from src.utils.logging_utils import get_logger, log_api_call, log_error
from src.utils.math_utils import win_rate as calc_win_rate

logger = get_logger(__name__)

_BASE = settings.tastytrade_backtester_base_url
_MAX_POLL = 60
_POLL_INTERVAL = 3


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def build_equity_option_short_put_payload(
    symbol: str,
    start_date: str,
    end_date: str,
    dte: int = 45,
    delta: int = 30,
    quantity: int = 1,
    num_legs: int = 2,
) -> BacktestPayload:
    """Build the verified working short-put payload.

    Two identical legs is the baseline that was validated to return real results.
    type="equity-option" is the only value that works — do not change it.
    """
    legs = [
        BacktestLeg(
            type="equity-option",
            direction="short",
            quantity=quantity,
            side="put",
            days_until_expiration=dte,
            strike_selection="delta",
            delta=delta,
        )
        for _ in range(num_legs)
    ]
    return BacktestPayload(
        symbol=symbol.upper(),
        start_date=start_date,
        end_date=to_iso_datetime(end_date),
        legs=legs,
    )


def build_custom_legs_payload(
    symbol: str,
    start_date: str,
    end_date: str,
    legs: List[Dict[str, Any]],
) -> BacktestPayload:
    """Build a payload from user-supplied leg definitions."""
    parsed_legs = []
    for leg_dict in legs:
        strike_value = leg_dict.get("strikePrice", leg_dict.get("strike"))
        delta_value = leg_dict.get("delta", 30)
        if delta_value in (None, ""):
            delta_value = 30
        parsed_legs.append(BacktestLeg(
            type=leg_dict.get("type", "equity-option"),
            direction=leg_dict.get("direction", "short"),
            quantity=int(leg_dict.get("quantity", 1)),
            side=leg_dict.get("side", "put"),
            days_until_expiration=int(leg_dict.get("daysUntilExpiration", 45)),
            strike_selection=leg_dict.get("strikeSelection", "delta"),
            delta=int(delta_value),
            strike_price=(
                float(strike_value)
                if strike_value not in (None, "")
                else None
            ),
        ))
    return BacktestPayload(
        symbol=symbol.upper(),
        start_date=start_date,
        end_date=to_iso_datetime(end_date),
        legs=parsed_legs,
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def create_backtest(payload: BacktestPayload) -> Tuple[Optional[str], str]:
    """POST /backtests — returns (backtest_id, error_message)."""
    token, err = get_access_token()
    if not token:
        return None, err or "No access token available."

    headers = get_auth_headers(token)
    url = f"{_BASE}/backtests"

    try:
        body = payload.to_dict()
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        log_api_call(logger, "TastytradeBacktester", "POST /backtests", resp.status_code)

        if resp.status_code in (200, 201):
            data = resp.json()
            backtest_id = (
                data.get("id")
                or data.get("backtestId")
                or data.get("data", {}).get("id")
            )
            if backtest_id:
                logger.info(f"[TastytradeBacktester] Created backtest {backtest_id}")
                return str(backtest_id), ""
            return None, "Backtest created but no ID returned."

        return None, f"Failed to create backtest: HTTP {resp.status_code}"

    except Exception as exc:
        log_error(logger, "TastytradeBacktester", exc)
        return None, f"Create backtest error: {type(exc).__name__}"


def get_backtest_status(backtest_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """GET /backtests/{id} — returns (data, error_message)."""
    token, err = get_access_token()
    if not token:
        return None, err or "No access token."

    headers = get_auth_headers(token)
    url = f"{_BASE}/backtests/{backtest_id}"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        log_api_call(logger, "TastytradeBacktester", f"GET /backtests/{backtest_id}", resp.status_code)

        if resp.status_code == 200:
            return resp.json(), ""
        return None, f"Get backtest failed: HTTP {resp.status_code}"

    except Exception as exc:
        log_error(logger, "TastytradeBacktester", exc)
        return None, f"Get backtest error: {type(exc).__name__}"


def poll_backtest(backtest_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Poll until backtest completes or times out."""
    for attempt in range(_MAX_POLL):
        data, err = get_backtest_status(backtest_id)
        if err:
            return None, err

        status = (data or {}).get("status", "")
        logger.info(f"[TastytradeBacktester] Poll #{attempt+1}: status={status}")

        if status in ("complete", "completed", "success"):
            return data, ""
        if status in ("error", "failed", "failure"):
            return None, f"Backtest failed with status: {status}"

        time.sleep(_POLL_INTERVAL)

    return None, f"Backtest timed out after {_MAX_POLL * _POLL_INTERVAL}s."


def get_backtest_logs(backtest_id: str) -> List[str]:
    """GET /backtests/{id}/logs — returns log lines or empty list."""
    token, _ = get_access_token()
    if not token:
        return []

    headers = get_auth_headers(token)
    url = f"{_BASE}/backtests/{backtest_id}/logs"

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            raw = resp.json()
            if isinstance(raw, list):
                return [str(line) for line in raw]
            if isinstance(raw, dict):
                return raw.get("logs") or raw.get("messages") or []
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

def parse_backtest_result(backtest_id: str, data: Dict[str, Any]) -> BacktestResult:
    """Parse raw API response into a structured BacktestResult.

    The backtester API nests trials/statistics/snapshots under data["results"].
    Top-level keys hold metadata (symbol, status, legs, dates).
    """
    # Leg type lives at top level
    legs = data.get("legs") or []
    leg_type = "unknown"
    if legs:
        leg_type = legs[0].get("type", "unknown")

    # Results are nested under "results" key
    results_obj: Dict[str, Any] = data.get("results") or {}

    # Trials — API returns {profitLoss, openDateTime, closeDateTime}
    raw_trials = (
        results_obj.get("trials")
        or results_obj.get("snapshots")
        or data.get("trials")
        or data.get("snapshots")
        or []
    )
    trials = [BacktestTrial.from_dict(t) for t in raw_trials]

    # Statistics — API returns human-readable keys like "Win percentage", "Total profit/loss"
    raw_stats = (
        results_obj.get("statistics")
        or results_obj.get("stats")
        or data.get("statistics")
        or data.get("stats")
    )
    statistics = BacktestStatistics.from_dict(raw_stats) if raw_stats else _compute_statistics(trials)

    symbol = data.get("symbol") or ""

    return BacktestResult(
        backtest_id=backtest_id,
        symbol=symbol,
        status=data.get("status", ""),
        leg_type=leg_type,
        trials=trials,
        statistics=statistics,
        raw=data,
    )


def _compute_statistics(trials: List[BacktestTrial]) -> Optional[BacktestStatistics]:
    """Derive statistics from trials when the API doesn't return them."""
    if not trials:
        return None

    profits = [t.profit_loss for t in trials]
    total_pl = sum(profits, Decimal("0"))
    avg_pl = total_pl / len(profits) if profits else Decimal("0")
    wins = sum(1 for p in profits if p > Decimal("0"))
    losses = sum(1 for p in profits if p < Decimal("0"))
    max_p = max(profits) if profits else Decimal("0")
    min_p = min(profits) if profits else Decimal("0")

    return BacktestStatistics(
        total_profit_loss=total_pl,
        win_rate=wins / len(profits) if profits else 0.0,
        average_profit_loss=avg_pl,
        num_trades=len(profits),
        num_wins=wins,
        num_losses=losses,
        max_profit=max_p,
        max_loss=min_p,
        raw={},
    )


def validate_backtest_success(result: BacktestResult) -> BacktestValidationResult:
    """Strict validation gate — never pass if data is fake/incomplete."""
    reasons = []

    if result.leg_type == "unknown":
        reasons.append(
            "leg type='unknown' — payload used wrong type value. "
            "Must be 'equity-option', not 'option' or 'equity'."
        )

    if result.statistics is None:
        reasons.append("statistics=null — backtester returned no statistics.")

    if not result.trials:
        reasons.append("trials=null/empty — no trial data returned.")
    else:
        all_zero = all(t.profit_loss == Decimal("0") for t in result.trials)
        if all_zero:
            reasons.append(
                "All profitLoss values are 0 — this indicates a misconfigured payload "
                "or an API error. Not showing as a successful result."
            )

    if reasons:
        return BacktestValidationResult.fail(*reasons)

    return BacktestValidationResult.ok(result)


def extract_backtest_summary(result: BacktestResult) -> Dict[str, Any]:
    """Return a clean dict summary for UI rendering."""
    stats = result.statistics
    trials = result.trials

    trial_rows = [
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "profit_loss": float(t.profit_loss),
        }
        for t in trials[:500]
    ]

    return {
        "backtest_id": result.backtest_id,
        "symbol": result.symbol,
        "status": result.status,
        "leg_type": result.leg_type,
        "num_trials": len(trials),
        "total_profit_loss": float(stats.total_profit_loss) if stats else None,
        "win_rate": stats.win_rate if stats else None,
        "average_profit_loss": float(stats.average_profit_loss) if stats else None,
        "num_trades": stats.num_trades if stats else None,
        "num_wins": stats.num_wins if stats else None,
        "num_losses": stats.num_losses if stats else None,
        "max_profit": float(stats.max_profit) if stats else None,
        "max_loss": float(stats.max_loss) if stats else None,
        "sharpe_ratio": stats.sharpe_ratio if stats else None,
        "trial_rows": trial_rows,
    }


# ---------------------------------------------------------------------------
# High-level convenience entry point
# ---------------------------------------------------------------------------

def run_options_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    dte: int = 45,
    delta: int = 30,
    quantity: int = 1,
    num_legs: int = 2,
    custom_legs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create, poll, validate, and return a complete options backtest."""
    # Build payload
    if custom_legs:
        payload = build_custom_legs_payload(symbol, start_date, end_date, custom_legs)
    else:
        payload = build_equity_option_short_put_payload(
            symbol, start_date, end_date, dte=dte, delta=delta,
            quantity=quantity, num_legs=num_legs,
        )

    # Create
    backtest_id, err = create_backtest(payload)
    if not backtest_id:
        return {"status": "ERROR", "message": err, "passed_validation": False}

    # Poll
    data, err = poll_backtest(backtest_id)
    if not data:
        return {"status": "ERROR", "message": err, "backtest_id": backtest_id, "passed_validation": False}

    # Parse
    result = parse_backtest_result(backtest_id, data)

    # Validate — never show success on bad data
    validation = validate_backtest_success(result)
    if not validation.passed:
        return {
            "status": "VALIDATION_FAILED",
            "message": "; ".join(validation.reasons),
            "backtest_id": backtest_id,
            "passed_validation": False,
            "leg_type": result.leg_type,
            "raw": data,
        }

    summary = extract_backtest_summary(result)
    summary["status"] = "SUCCESS"
    summary["passed_validation"] = True
    return summary
