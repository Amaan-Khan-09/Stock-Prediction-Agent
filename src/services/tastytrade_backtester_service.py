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
            quantity=min(quantity, 10),  # Tastytrade limit: 1–10
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


def _coerce_int(value: Any, default: int) -> int:
    """int(value) with a safe default -- unlike dict.get(key, default), this also
    covers the case where the key is PRESENT but explicitly None or "" (e.g. a
    caller passing {"delta": None} instead of omitting the key), which used to
    raise TypeError/ValueError from a bare int(leg_dict.get("delta", 30)).
    """
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> Optional[float]:
    """float(value) or None -- same None/"" safety as _coerce_int, for fields
    that are legitimately optional (strike_price, percentage_otm, etc.) so a
    bad/empty value becomes None here rather than blowing up later inside
    BacktestLeg.to_dict()'s float(self.strike_price) cast.
    """
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_custom_legs_payload(
    symbol: str,
    start_date: str,
    end_date: str,
    legs: List[Dict[str, Any]],
    entry_frequency: str = "every day",
    exit_rule: str = "Exit at target date",
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    exit_after_days: Optional[int] = None,
) -> BacktestPayload:
    """Build a payload from user-supplied leg definitions.

    entry_frequency maps to Tastytrade entryConditions.frequency.
    Confirmed working: "every day", "on_exact_dte_match".
    Unverified: "weekly", "monthly" — may default to daily at API level.

    exit_rule maps to Tastytrade exitConditions.
    Confirmed working: "Exit at target date" (empty dict = API default).
    Best-effort: "Target profit 50%", "Stop loss 200%", "Exit after N days".
    """
    parsed_legs = []
    for leg_dict in legs:
        parsed_legs.append(BacktestLeg(
            type=leg_dict.get("type") or "equity-option",
            direction=leg_dict.get("direction") or "short",
            quantity=min(_coerce_int(leg_dict.get("quantity"), 1), 10),
            side=leg_dict.get("side") or "put",
            days_until_expiration=_coerce_int(leg_dict.get("daysUntilExpiration"), 45),
            strike_selection=leg_dict.get("strikeSelection") or "delta",
            delta=_coerce_int(leg_dict.get("delta"), 30),
            strike_price=_coerce_optional_float(leg_dict.get("strikePrice", leg_dict.get("strike"))),
            percentage_otm=_coerce_optional_float(leg_dict.get("percentageOtm")),
            price_offset=_coerce_optional_float(leg_dict.get("priceOffset")),
            premium_amount=_coerce_optional_float(leg_dict.get("premium")),
        ))
    return BacktestPayload(
        symbol=symbol.upper(),
        start_date=start_date,
        end_date=to_iso_datetime(end_date),
        legs=parsed_legs,
        entry_frequency=entry_frequency,
        exit_rule=exit_rule,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        exit_after_days=exit_after_days,
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def _post_backtest(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[Optional[str], str]:
    """Internal: POST one backtest body, return (backtest_id_or_None, error_str)."""
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

    if resp.status_code == 429:
        _ra = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset-After") or "unknown"
        return None, f"RATE_LIMITED:429:retry_after={_ra}"

    _err_body = ""
    try:
        _err_body = (resp.text or "")[:2000]
    except Exception:
        pass
    _field_hint = ""
    try:
        _err_json = resp.json()
        _err_msg = str(_err_json.get("message") or _err_json.get("error") or _err_json.get("errors") or "")
        if _err_msg:
            _field_hint = f" | API_MSG: {_err_msg[:400]}"
    except Exception:
        pass

    return None, f"BACKTEST_HTTP_{resp.status_code}{_field_hint}:{_err_body}"


_RATE_LIMIT_MAX_ATTEMPTS = 3
_RATE_LIMIT_DEFAULT_BACKOFF_SECONDS = 2.0
_RATE_LIMIT_MAX_BACKOFF_SECONDS = 30.0


def _parse_retry_after_seconds(err_msg: str) -> float:
    """Extract the retry_after value from a 'RATE_LIMITED:429:retry_after=<val>'
    error string. Falls back to the default backoff when the header was absent
    ('unknown') or non-numeric.
    """
    marker = "retry_after="
    idx = err_msg.find(marker)
    if idx == -1:
        return _RATE_LIMIT_DEFAULT_BACKOFF_SECONDS
    raw = err_msg[idx + len(marker):].strip()
    try:
        return max(0.0, min(float(raw), _RATE_LIMIT_MAX_BACKOFF_SECONDS))
    except ValueError:
        return _RATE_LIMIT_DEFAULT_BACKOFF_SECONDS


def create_backtest(payload: BacktestPayload) -> Tuple[Optional[str], str]:
    """POST /backtests — returns (backtest_id, error_message).

    Two distinct retry strategies, never conflated:
      HTTP 429 (rate limited): back off for the server-specified Retry-After
        (or a bounded default) and resubmit the SAME unmodified payload, up to
        _RATE_LIMIT_MAX_ATTEMPTS times. Mutating the payload does nothing for a
        rate limit and previously risked compounding it with extra requests.
      HTTP 400 (payload/format issues): mutate and retry, since the account
        isn't being throttled:
          Attempt 1: full payload (combined exit conditions as flat struct)
          Attempt 2: take-profit only (proven single-condition format)
          Attempt 3: no exit conditions (guaranteed safe fallback)
    """
    token, err = get_access_token()
    if not token:
        return None, err or "No access token available."

    headers = get_auth_headers(token)
    url = f"{_BASE}/backtests"

    try:
        body = payload.to_dict()
        bid, err_msg = _post_backtest(url, body, headers)
        if bid:
            return bid, ""

        # Rate limit: back off and resubmit the same payload -- never mutate it.
        if "RATE_LIMITED" in err_msg:
            for attempt in range(1, _RATE_LIMIT_MAX_ATTEMPTS):
                wait_s = _parse_retry_after_seconds(err_msg)
                logger.warning(
                    f"[TastytradeBacktester] Rate limited (attempt {attempt}/"
                    f"{_RATE_LIMIT_MAX_ATTEMPTS}). Backing off {wait_s:.1f}s before retry."
                )
                time.sleep(wait_s)
                bid, err_msg = _post_backtest(url, body, headers)
                if bid:
                    return bid, ""
                if "RATE_LIMITED" not in err_msg:
                    break  # a different error now — fall through to normal handling below
            if "RATE_LIMITED" in err_msg:
                logger.error(
                    f"[TastytradeBacktester] Still rate limited after "
                    f"{_RATE_LIMIT_MAX_ATTEMPTS} attempts. Giving up."
                )
                return None, err_msg

        # Retry only on HTTP 400 (payload/format issues)
        if "BACKTEST_HTTP_400" not in err_msg:
            logger.error(
                f"[TastytradeBacktester] HTTP non-400 error from POST /backtests. "
                f"Payload sent: {str(body)[:800]}."
            )
            return None, err_msg

        # ── Retry 2: take-profit only ────────────────────────────────────────
        ec = body.get("exitConditions", {})
        has_combined = bool(ec.get("profitPercentage") and ec.get("lossPercentage"))
        if has_combined:
            body2 = {
                **body,
                "exitConditions": {
                    "type": "profit_percentage",
                    "profitPercentage": ec["profitPercentage"],
                },
            }
            logger.warning(
                "[TastytradeBacktester] Combined exitConditions rejected (HTTP 400). "
                "Retrying with take-profit only."
            )
            bid2, err2 = _post_backtest(url, body2, headers)
            if bid2:
                return bid2, ""
            err_msg = err2  # carry forward latest error

        # ── Retry 3: no exit conditions (guaranteed fallback) ────────────────
        if body.get("exitConditions"):
            body3 = {**body, "exitConditions": {}}
            logger.warning(
                "[TastytradeBacktester] Exit conditions rejected. "
                "Retrying with empty exitConditions (API default expiry)."
            )
            bid3, err3 = _post_backtest(url, body3, headers)
            if bid3:
                return bid3, "EXIT_COND_DROPPED:" + err_msg
            err_msg = err3

        logger.error(
            f"[TastytradeBacktester] All retry attempts failed. "
            f"Last error: {err_msg[:400]}. Payload: {str(body)[:600]}"
        )
        return None, err_msg

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
    raw_trials = results_obj.get("trials") or results_obj.get("snapshots") or []
    trials = [BacktestTrial.from_dict(t) for t in raw_trials]

    # Statistics — API returns human-readable keys like "Win percentage", "Total profit/loss"
    raw_stats = results_obj.get("statistics") or results_obj.get("stats")
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
    entry_frequency: str = "every day",
) -> Dict[str, Any]:
    """Create, poll, validate, and return a complete options backtest."""
    # Build payload
    if custom_legs:
        payload = build_custom_legs_payload(symbol, start_date, end_date, custom_legs, entry_frequency=entry_frequency)
    else:
        payload = build_equity_option_short_put_payload(
            symbol, start_date, end_date, dte=dte, delta=delta,
            quantity=quantity, num_legs=num_legs,
        )

    # Create
    backtest_id, create_note = create_backtest(payload)
    if not backtest_id:
        return {"status": "ERROR", "message": create_note, "passed_validation": False}

    # A non-empty create_note alongside a real backtest_id means create_backtest
    # had to fall back to a degraded payload (e.g. "EXIT_COND_DROPPED:...") --
    # the exit conditions Tastytrade actually ran against are NOT what was
    # requested. Silently discarding this (checking only `if backtest_id:`) was
    # a real bug: the result would look like a clean success even though, say,
    # the stop-loss/take-profit the user asked for was never applied.
    exit_conditions_degraded = bool(create_note)
    exit_conditions_degraded_reason = create_note if exit_conditions_degraded else ""

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
    summary["exit_conditions_degraded"] = exit_conditions_degraded
    if exit_conditions_degraded:
        summary["exit_conditions_degraded_reason"] = exit_conditions_degraded_reason
        logger.warning(
            f"[TastytradeBacktester] Backtest {backtest_id} succeeded with degraded exit "
            f"conditions: {exit_conditions_degraded_reason}"
        )
    # Requesting BOTH a stop-loss and take-profit is sent as a single flat dict with
    # only one 'type' discriminator field ("proven working format" per BacktestPayload
    # .to_dict()) -- whether Tastytrade actually enforces the non-discriminator side
    # (lossPercentage here) is unverified against the live API. Surface that honestly
    # rather than silently trusting both conditions were applied.
    has_tp = bool(getattr(payload, "take_profit_pct", None))
    has_sl = bool(getattr(payload, "stop_loss_pct", None))
    if has_tp and has_sl and not exit_conditions_degraded:
        summary["exit_conditions_caveat"] = (
            "Both take-profit and stop-loss were requested. Tastytrade's exitConditions "
            "payload only carries one 'type' discriminator (set to profit_percentage); "
            "whether the API enforces the stop-loss side under that discriminator is "
            "unverified. Treat max-loss figures from this run with caution."
        )
    return summary
